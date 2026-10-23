#!/usr/bin/env python3
"""
Bootstrap clusterizado canónico de la Función de Control (CF-RCS) — V3.2.5.

Reimplementación vectorizada (numpy) del algoritmo de
`vm_bootstrap_vhat_canonical.py`, con el mismo generador y la misma secuencia de
remuestreo (numpy default_rng(seed), rng.choice(clusters ordenados, 77, replace=True)).
A diferencia de la versión anterior:
  * GUARDA LAS RÉPLICAS BRUTAS (coeficientes, estadísticos y curva de elasticidad)
    en un .npz sin pickle, para que cualquier percentil sea recomputable.
  * Calcula la elasticidad puntual del CF en cada réplica sobre una grilla fija,
    de modo que la banda percentil corresponde a la elasticidad (no a v_hat).
  * Reporta además el contraste F ANIDADO correcto de no linealidad
    (CF lineal en log-precio vs. CF-RCS, 2 g.l.).

Especificación (idéntica a src/rcs_utils.py):
  1ª etapa: log_checkout_price ~ log_base_price + log_precio_prom_otros_centros
            + emailer + homepage + hubo_aumento + C(category) + C(center_type)
  2ª etapa: log_num_orders ~ 1 + rcs_1 + rcs_2 + rcs_3 + v_hat + mismos controles
  Nudos RCS fijos en muestra completa (P5/P35/P65/P95), design_info fijo.

Uso:
  python scripts/vm/bootstrap_cf_canonical_replicas.py [--iterations 5000] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import build_design_matrices

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rcs_utils import build_canonical_rcs_design  # noqa: E402

GRID = np.linspace(50.0, 800.0, 200)


def control_matrix(df: pd.DataFrame, cats, ctypes) -> np.ndarray:
    cols = [np.ones(len(df)), df["emailer_for_promotion"].to_numpy(float),
            df["homepage_featured"].to_numpy(float), df["hubo_aumento"].to_numpy(float)]
    cols += [(df["category"].to_numpy() == c).astype(float) for c in cats[1:]]
    cols += [(df["center_type"].to_numpy() == c).astype(float) for c in ctypes[1:]]
    return np.column_stack(cols)


def ols(X, y):
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    rank = np.linalg.matrix_rank(XtX)
    s2 = (e @ e) / (len(y) - rank)
    return b, e, XtX_inv, s2, rank


def basis(design_info, logp):
    return np.asarray(build_design_matrices([design_info], {"log_checkout_price": logp})[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "demanda_limpia.csv"))
    ap.add_argument("--output-dir", default=str(ROOT / "reports" / "certified"))
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    # Instrumentos canónicos en muestra completa (para fijar nudos con la muestra IV)
    g = df.groupby(["meal_id", "week"])["checkout_price"]
    other = (g.transform("sum") - df["checkout_price"]) / (g.transform("count") - 1).replace(0, np.nan)
    full = df.loc[other.notna()].copy().reset_index(drop=True)
    # Igual que el script original: se remuestrea desde la muestra IV (N=456,544)
    df = full
    design_info, knots = build_canonical_rcs_design(full["log_checkout_price"].to_numpy())

    cats = sorted(df["category"].unique())
    ctypes = sorted(df["center_type"].unique())
    clusters = np.asarray(sorted(df["center_id"].dropna().unique()))
    idx_by_c = {c: np.flatnonzero(df["center_id"].to_numpy() == c) for c in clusters}

    # Derivada de la base en la grilla y en precio medio / mediana (diferencia central)
    h = 1e-5
    lg = np.log(GRID)
    dB = (basis(design_info, lg + h) - basis(design_info, lg - h)) / (2 * h)
    p_mean, p_med = full["checkout_price"].mean(), full["checkout_price"].median()
    lpts = np.log([p_mean, p_med])
    dBp = (basis(design_info, lpts + h) - basis(design_info, lpts - h)) / (2 * h)

    meal = df["meal_id"].to_numpy(); week = df["week"].to_numpy()
    price = df["checkout_price"].to_numpy(float)
    lbase = np.log(df["base_price"].to_numpy(float))
    lp = df["log_checkout_price"].to_numpy(float)
    ly = df["log_num_orders"].to_numpy(float)
    W_all = control_matrix(df, cats, ctypes)
    B_all = basis(design_info, lp)
    mw_codes = pd.factorize(pd.Series(meal).astype(str) + "_" + pd.Series(week).astype(str))[0]

    rng = np.random.default_rng(a.seed)
    n = a.iterations
    names = ["v_hat", "rcs_1", "rcs_2", "rcs_3", "f_exog", "f_wald_rcs23", "f_nested_nonlin",
             "elast_mean", "elast_median"]
    R = {k: np.full(n, np.nan) for k in names}
    curves = np.full((n, len(GRID)), np.nan)
    t0 = time.time()
    for i in range(n):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        ix = np.concatenate([idx_by_c[c] for c in sampled])
        try:
            codes = mw_codes[ix]
            s = np.bincount(codes, weights=price[ix]); c = np.bincount(codes)
            cnt = c[codes] - 1
            keep = cnt > 0
            loo = np.log((s[codes][keep] - price[ix][keep]) / cnt[keep])
            jx = ix[keep]
            W = W_all[jx]
            X1 = np.column_stack([lbase[jx], loo, W])
            _, vhat, *_ = ols(X1, lp[jx])
            X2 = np.column_stack([B_all[jx], vhat, W])
            b, e, XtXi, s2, _ = ols(X2, ly[jx])
            f_exog = b[3] ** 2 / (s2 * XtXi[3, 3])
            Vb = s2 * XtXi[1:3, 1:3]
            f_w23 = float(b[1:3] @ np.linalg.solve(Vb, b[1:3]) / 2)
            Xlin = np.column_stack([lp[jx], vhat, W])
            _, elin, *_ = ols(Xlin, ly[jx])
            df_res = len(jx) - X2.shape[1]
            f_nest = ((elin @ elin - e @ e) / 2) / ((e @ e) / df_res)
            coef = b[:3]
            R["v_hat"][i] = b[3]; R["rcs_1"][i], R["rcs_2"][i], R["rcs_3"][i] = coef
            R["f_exog"][i] = f_exog; R["f_wald_rcs23"][i] = f_w23; R["f_nested_nonlin"][i] = f_nest
            R["elast_mean"][i], R["elast_median"][i] = dBp @ coef
            curves[i] = dB @ coef
        except Exception as exc:  # pragma: no cover
            print(f"iter {i+1} descartada: {exc}", flush=True)
        if (i + 1) % 250 == 0:
            print(f"{i+1}/{n} ({(time.time()-t0)/60:.1f} min)", flush=True)

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "bootstrap_cf_canonical_replicas.npz", price_grid=GRID,
                        elasticity_curves=curves, knots_log_price=knots, **R)

    def st(x):
        x = x[np.isfinite(x)]
        return {"media": float(x.mean()), "std": float(x.std(ddof=1)),
                "ic95_inf": float(np.percentile(x, 2.5)), "ic95_sup": float(np.percentile(x, 97.5)), "n": int(x.size)}
    summary = {
        "version": "3.2.5",
        "script": "scripts/vm/bootstrap_cf_canonical_replicas.py",
        "seed": a.seed, "n_iteraciones_total": n,
        "n_iteraciones_validas": int(np.isfinite(R["v_hat"]).sum()),
        "cluster_variable": "center_id", "n_clusters": int(len(clusters)),
        "nudos_precio": np.exp(knots).tolist(),
        "precio_medio": float(p_mean), "precio_mediana": float(p_med),
        **{k: st(v) for k, v in R.items()},
        "banda_elasticidad": {
            "price_grid": GRID.tolist(),
            "ic95_inf": np.nanpercentile(curves, 2.5, axis=0).tolist(),
            "ic95_sup": np.nanpercentile(curves, 97.5, axis=0).tolist(),
        },
        "nota": ("Réplicas brutas en bootstrap_cf_canonical_replicas.npz (sin pickle). "
                 "f_wald_rcs23 = Wald H0: rcs_2=rcs_3=0 (NO es un test de linealidad con base cr centrada); "
                 "f_nested_nonlin = F anidado CF lineal vs CF-RCS (test correcto de no linealidad)."),
    }
    (out / "bootstrap_cf_canonical_replicas.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "banda_elasticidad"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
