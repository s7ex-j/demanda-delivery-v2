#!/usr/bin/env python3
"""
Genera los CSV de dashboard derivados de modelos (V3.3.0):
  - dashboard/data/elasticidad_por_categoria_detallada.csv  (interacción precio × categoría)
  - dashboard/data/elasticidad_rcs_por_precio.csv           (curva RCS canónica, coeficientes puntuales)
Son elasticidades descriptivas (asociaciones observacionales). No se generan
"recomendaciones de precio": combinar elasticidades de modelos distintos o leerlas
como efectos causales no está respaldado por el análisis.
Uso: python scripts/analysis/generar_dashboard_modelos.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from modeling import test_igualdad_elasticidad_por_categoria  # noqa: E402
from rcs_utils import apply_rcs_design, build_canonical_rcs_design  # noqa: E402

OUT = ROOT / "dashboard" / "data"
CTRL = "emailer_for_promotion + homepage_featured + hubo_aumento + C(category) + C(center_type)"


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "demanda_limpia.csv")
    cat = test_igualdad_elasticidad_por_categoria(df)
    t = (pd.Series(cat["elasticidad_por_categoria"], name="elasticidad").rename_axis("category").reset_index()
         .assign(elasticidad=lambda x: x.elasticidad.round(4),
                 especificacion="OLS log-log con interacción precio x categoría y controles",
                 nota="Asociación observacional; no es una recomendación de precio")
         .sort_values("elasticidad"))
    t.to_csv(OUT / "elasticidad_por_categoria_detallada.csv", index=False)

    di, knots = build_canonical_rcs_design(df["log_checkout_price"].to_numpy())
    S = apply_rcs_design(di, df["log_checkout_price"].to_numpy())
    d = df.copy()
    for c in S.columns:
        d[c] = S[c].to_numpy()
    m = smf.ols(f"log_num_orders ~ rcs_1 + rcs_2 + rcs_3 + {CTRL}", d).fit()
    precios = np.arange(125, 651, 25, dtype=float)
    h = 1e-5
    lp = np.log(precios)
    dB = (np.asarray(apply_rcs_design(di, lp + h)) - np.asarray(apply_rcs_design(di, lp - h))) / (2 * h)
    el = dB @ m.params[["rcs_1", "rcs_2", "rcs_3"]].to_numpy()
    pd.DataFrame({"precio": precios, "log_precio": lp, "elasticidad_rcs": el.round(4),
                  "especificacion": "RCS canónica P5/P35/P65/P95 con controles (coeficientes puntuales)",
                  "nota": "Elasticidad puntual descriptiva; banda bootstrap en el informe (Figura 12)"}
                 ).to_csv(OUT / "elasticidad_rcs_por_precio.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
