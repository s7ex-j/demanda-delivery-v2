#!/usr/bin/env python3
"""
Diagnósticos completos del IV 2SLS principal (con controles) y del Control Function — V3.2.5.

Genera reports/certified/iv_cf_diagnosticos_v325.json con TODOS los valores numéricos
que el informe atribuye a la especificación IV principal y a la CF puntual:
  - 2SLS con controles: coef, EE robusto (HC) y EE agrupado por center_id
  - Primera etapa: R2, F overall, F conjunto (homocedástico, HC1, cluster), R2 parcial de
    Shea (1 endógena => R2 parcial conjunto de los instrumentos), contribución única por instrumento
  - Hansen J (IVGMM robusto), Wu-Hausman, Wooldridge score
  - IV justamente identificado (log_base_price) con controles
  - Especificación de referencia sin controles (solo promociones) para trazabilidad
  - Coeficientes de promociones en el 2SLS principal
  - CF puntual: coeficientes, F exogeneidad (t^2), Wald rcs_2=rcs_3=0, F anidado de no linealidad,
    elasticidades en media y mediana (diferencia central), RCS-OLS misma muestra
  - Interacción precio x promoción (src/modeling.py), OLS principal
Uso: python scripts/analysis/iv_controles_diagnosticos.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS, IVGMM
from patsy import build_design_matrices
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from modeling import construir_instrumentos, modelo_interaccion_precio_promo  # noqa: E402
from rcs_utils import build_canonical_rcs_design, apply_rcs_design  # noqa: E402

CTRL = "emailer_for_promotion + homepage_featured + hubo_aumento + C(category) + C(center_type)"
ENDO = "[log_checkout_price ~ log_base_price + log_precio_prom_otros_centros]"


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "demanda_limpia.csv")
    d = construir_instrumentos(df).dropna(subset=["log_precio_prom_otros_centros"]).copy()
    out = {"version": "3.2.5", "n_total": int(len(df)), "n_iv": int(len(d))}

    # ---- 2SLS principal
    f_main = f"log_num_orders ~ 1 + {CTRL} + {ENDO}"
    r_rob = IV2SLS.from_formula(f_main, d).fit(cov_type="robust")
    r_cl = IV2SLS.from_formula(f_main, d).fit(cov_type="clustered", clusters=d["center_id"])
    b = float(r_rob.params["log_checkout_price"])
    se_cl = float(r_cl.std_errors["log_checkout_price"])
    z = stats.norm.ppf(0.975)
    out["iv_principal"] = {
        "coef": b, "ee_robusto": float(r_rob.std_errors["log_checkout_price"]),
        "ee_cluster_center": se_cl, "ic95_cluster": [b - z * se_cl, b + z * se_cl],
        "emailer": float(r_rob.params["emailer_for_promotion"]),
        "homepage": float(r_rob.params["homepage_featured"]),
        "hubo_aumento": float(r_rob.params["hubo_aumento"]),
        "ee_emailer": float(r_rob.std_errors["emailer_for_promotion"]),
        "ee_homepage": float(r_rob.std_errors["homepage_featured"]),
        "wu_hausman_F": float(r_rob.wu_hausman().stat), "wu_hausman_p": float(r_rob.wu_hausman().pval),
        "wooldridge_score": float(r_rob.wooldridge_score.stat), "wooldridge_p": float(r_rob.wooldridge_score.pval),
    }
    fs_diag = r_rob.first_stage.diagnostics
    out["iv_principal"]["linearmodels_first_stage"] = {str(k): float(v) for k, v in fs_diag.iloc[0].items()
                                                       if isinstance(v, (int, float, np.floating))}
    j = IVGMM.from_formula(f_main, d).fit(cov_type="robust").j_stat
    out["iv_principal"]["hansen_J"] = float(j.stat); out["iv_principal"]["hansen_J_p"] = float(j.pval)
    rj = IV2SLS.from_formula(f"log_num_orders ~ 1 + {CTRL} + [log_checkout_price ~ log_base_price]", d).fit(cov_type="robust")
    rj_cl = IV2SLS.from_formula(f"log_num_orders ~ 1 + {CTRL} + [log_checkout_price ~ log_base_price]", d).fit(
        cov_type="clustered", clusters=d["center_id"])
    out["iv_just_id_base"] = {"coef": float(rj.params["log_checkout_price"]),
                              "ee_robusto": float(rj.std_errors["log_checkout_price"]),
                              "ee_cluster_center": float(rj_cl.std_errors["log_checkout_price"])}

    # ---- Especificación de referencia sin controles (versiones previas)
    f_old = f"log_num_orders ~ 1 + emailer_for_promotion + homepage_featured + {ENDO}"
    r_old = IV2SLS.from_formula(f_old, d).fit(cov_type="robust")
    j_old = IVGMM.from_formula(f_old, d).fit(cov_type="robust").j_stat
    out["iv_sin_controles_referencia"] = {"coef": float(r_old.params["log_checkout_price"]),
                                          "ee_robusto": float(r_old.std_errors["log_checkout_price"]),
                                          "hansen_J": float(j_old.stat)}

    # ---- Primera etapa: F y R2 parciales
    fs = smf.ols(f"log_checkout_price ~ log_base_price + log_precio_prom_otros_centros + {CTRL}", d).fit()
    fs0 = smf.ols(f"log_checkout_price ~ {CTRL}", d).fit()
    fsB = smf.ols(f"log_checkout_price ~ log_precio_prom_otros_centros + {CTRL}", d).fit()
    fsP = smf.ols(f"log_checkout_price ~ log_base_price + {CTRL}", d).fit()
    dfr = fs.df_resid
    names = list(fs.params.index)
    R = np.zeros((2, len(names)))
    R[0, names.index("log_base_price")] = 1; R[1, names.index("log_precio_prom_otros_centros")] = 1
    f_hc1 = smf.ols(fs.model.formula, d).fit(cov_type="HC1").wald_test(R, use_f=True, scalar=True)
    f_cl = smf.ols(fs.model.formula, d).fit(cov_type="cluster", cov_kwds={"groups": d["center_id"]}).wald_test(
        R, use_f=True, scalar=True)
    out["primera_etapa"] = {
        "R2": float(fs.rsquared), "F_overall": float(fs.fvalue),
        "F_conjunto_homoc": float(((fs0.ssr - fs.ssr) / 2) / (fs.ssr / dfr)),
        "F_conjunto_HC1": float(np.squeeze(f_hc1.statistic)),
        "F_conjunto_cluster_center": float(np.squeeze(f_cl.statistic)),
        "R2_parcial_Shea": float(1 - fs.ssr / fs0.ssr),
        "R2_parcial_unico_otros_centros": float(1 - fs.ssr / fsP.ssr),
        "R2_parcial_unico_base_price": float(1 - fs.ssr / fsB.ssr),
        "F_unico_otros_centros": float((fsP.ssr - fs.ssr) / (fs.ssr / dfr)),
        "F_unico_base_price": float((fsB.ssr - fs.ssr) / (fs.ssr / dfr)),
        "coef_base_price": float(fs.params["log_base_price"]), "ee_base_price": float(fs.bse["log_base_price"]),
        "coef_otros_centros": float(fs.params["log_precio_prom_otros_centros"]),
        "ee_otros_centros": float(fs.bse["log_precio_prom_otros_centros"]),
        "intercepto": float(fs.params["Intercept"]), "ee_intercepto": float(fs.bse["Intercept"]),
        "stock_yogo_tamano_10pct_K2_2": 19.93,
    }

    # ---- CF puntual
    d["v_hat"] = fs.resid
    di, knots = build_canonical_rcs_design(d["log_checkout_price"].to_numpy())
    S = apply_rcs_design(di, d["log_checkout_price"].to_numpy())
    for c in S.columns:
        d[c] = S[c].to_numpy()
    cf = smf.ols(f"log_num_orders ~ 1 + rcs_1 + rcs_2 + rcs_3 + v_hat + {CTRL}", d).fit()
    cf_hc3 = smf.ols(f"log_num_orders ~ 1 + rcs_1 + rcs_2 + rcs_3 + v_hat + {CTRL}", d).fit(cov_type="HC3")
    cf_lin = smf.ols(f"log_num_orders ~ 1 + log_checkout_price + v_hat + {CTRL}", d).fit()
    rcs_only = smf.ols(f"log_num_orders ~ 1 + rcs_1 + rcs_2 + rcs_3 + {CTRL}", d).fit()
    nm = list(cf.params.index)
    R2 = np.zeros((2, len(nm))); R2[0, nm.index("rcs_2")] = 1; R2[1, nm.index("rcs_3")] = 1

    def el(model, price):
        h = 1e-5
        lp = np.log(np.atleast_1d(price))
        dB = (np.asarray(apply_rcs_design(di, lp + h)) - np.asarray(apply_rcs_design(di, lp - h))) / (2 * h)
        return dB @ np.array([model.params[c] for c in ["rcs_1", "rcs_2", "rcs_3"]])

    pm, pmed = float(d["checkout_price"].mean()), float(d["checkout_price"].median())
    out["cf_puntual"] = {
        "coef": {k: float(cf.params[k]) for k in ["Intercept", "rcs_1", "rcs_2", "rcs_3", "v_hat"]},
        "ee_convencional": {k: float(cf.bse[k]) for k in ["Intercept", "rcs_1", "rcs_2", "rcs_3", "v_hat"]},
        "ee_hc3": {k: float(cf_hc3.bse[k]) for k in ["Intercept", "rcs_1", "rcs_2", "rcs_3", "v_hat"]},
        "p_convencional": {k: float(cf.pvalues[k]) for k in ["rcs_1", "rcs_2", "rcs_3", "v_hat"]},
        "F_exogeneidad": float(cf.tvalues["v_hat"] ** 2),
        "F_wald_rcs2_rcs3": float(np.squeeze(cf.wald_test(R2, use_f=True, scalar=True).statistic)),
        "F_anidado_no_linealidad": float(((cf_lin.ssr - cf.ssr) / 2) / (cf.ssr / cf.df_resid)),
        "gl_anidado": [2, float(cf.df_resid)],
        "coef_lineal_equivalente_2SLS": float(cf_lin.params["log_checkout_price"]),
        "R2_adj": float(cf.rsquared_adj),
        "precio_medio": pm, "precio_mediana": pmed,
        "elasticidad_media": float(el(cf, pm)[0]), "elasticidad_mediana": float(el(cf, pmed)[0]),
        "rcs_ols_misma_muestra_media": float(el(rcs_only, pm)[0]),
        "rcs_ols_misma_muestra_mediana": float(el(rcs_only, pmed)[0]),
        "nudos_precio": np.exp(knots).tolist(),
    }
    grid = np.exp(np.linspace(knots[0], knots[3], 400))
    out["rcs_ols_rango_P5_P95"] = {"min": float(el(rcs_only, grid).min()),
                                   "precio_min": float(grid[np.argmin(el(rcs_only, grid))])}

    # ---- OLS principal e interacción
    ols = smf.ols(f"log_num_orders ~ log_checkout_price + {CTRL}", df).fit()
    out["ols_principal"] = {k: float(v) for k, v in ols.params.items()}
    m, w, _ = modelo_interaccion_precio_promo(df)
    out["interaccion"] = {k: float(m.params[k]) for k in [
        "log_checkout_price_c", "emailer_for_promotion", "homepage_featured",
        "log_checkout_price_c:emailer_for_promotion", "log_checkout_price_c:homepage_featured"]}
    out["interaccion"].update({"F": w["estadistico_F"], "gl": list(w["grados_libertad"]), "p": w["p_valor"]})
    out["wald_categoria_p_log10"] = float(stats.f.logsf(519.56, 13, 456515) / np.log(10))

    path = ROOT / "reports" / "certified" / "iv_cf_diagnosticos_v325.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
