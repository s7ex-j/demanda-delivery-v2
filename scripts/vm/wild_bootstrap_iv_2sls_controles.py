#!/usr/bin/env python3
"""
Wild cluster bootstrap (Rademacher por center_id) del IV 2SLS — ESPECIFICACIÓN PRINCIPAL
CON CONTROLES (V3.2.5).

Misma lógica que `vm_wild_bootstrap_iv_2sls_full.py` (ambas etapas perturbadas con el
mismo peso por cluster y reestimadas), pero con los controles armonizados del modelo
principal: emailer_for_promotion, homepage_featured, hubo_aumento, C(category), C(center_type).

Guarda las réplicas brutas en reports/certified/wild_bootstrap_iv_2sls_controles.npz
y un resumen JSON.  Uso: python scripts/vm/wild_bootstrap_iv_2sls_controles.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vm"))
from bootstrap_cf_canonical_replicas import control_matrix  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "demanda_limpia.csv"))
    ap.add_argument("--output-dir", default=str(ROOT / "reports" / "certified"))
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    g = df.groupby(["meal_id", "week"])["checkout_price"]
    other = (g.transform("sum") - df["checkout_price"]) / (g.transform("count") - 1).replace(0, np.nan)
    df = df.loc[other.notna()].copy()
    df["log_precio_prom_otros_centros"] = np.log(other[other.notna()])
    df["log_base_price"] = np.log(df["base_price"])

    W = control_matrix(df, sorted(df["category"].unique()), sorted(df["center_type"].unique()))
    Z = np.column_stack([df["log_base_price"], df["log_precio_prom_otros_centros"], W])
    p = df["log_checkout_price"].to_numpy(float)
    y = df["log_num_orders"].to_numpy(float)
    ZtZi = np.linalg.pinv(Z.T @ Z)

    def fit(p_, y_):
        b1 = ZtZi @ (Z.T @ p_)
        phat = Z @ b1
        X = np.column_stack([phat, W])
        b2 = np.linalg.pinv(X.T @ X) @ (X.T @ y_)
        return b1, phat, b2

    b1, phat, b2 = fit(p, y)
    fsres = p - phat
    Xs = np.column_stack([phat, W])
    ssfit = Xs @ b2
    ssres = y - ssfit
    cl = df["center_id"].to_numpy()
    clusters = np.unique(cl)
    pos = np.searchsorted(clusters, cl)

    rng = np.random.default_rng(a.seed)
    coefs = np.full(a.iterations, np.nan)
    t0 = time.time()
    for i in range(a.iterations):
        w = rng.choice([-1.0, 1.0], size=len(clusters))[pos]
        _, _, b2b = fit(phat + w * fsres, ssfit + w * ssres)
        coefs[i] = b2b[0]
        if (i + 1) % 250 == 0:
            print(f"{i+1}/{a.iterations} ({(time.time()-t0)/60:.1f} min)", flush=True)

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "wild_bootstrap_iv_2sls_controles.npz", coefs=coefs)
    v = coefs[np.isfinite(coefs)]
    res = {
        "version": "3.2.5",
        "script": "scripts/vm/wild_bootstrap_iv_2sls_controles.py",
        "especificacion": "log_num_orders ~ 1 + emailer + homepage + hubo_aumento + C(category) + C(center_type) "
                          "+ [log_checkout_price ~ log_base_price + log_precio_prom_otros_centros]",
        "n_obs": int(len(df)), "clusters": int(len(clusters)), "semilla": a.seed,
        "coef_base": float(b2[0]),
        "coef_media": float(v.mean()), "coef_std": float(v.std(ddof=1)),
        "ic_95_inferior": float(np.percentile(v, 2.5)), "ic_95_superior": float(np.percentile(v, 97.5)),
        "n_iteraciones_validas": int(v.size), "n_iteraciones_total": a.iterations,
        "nota": "Wild cluster bootstrap (Rademacher, center_id), ambas etapas perturbadas y reestimadas; "
                "réplicas brutas en wild_bootstrap_iv_2sls_controles.npz. Sensibilidad; no corrige Hansen J.",
    }
    (out / "wild_bootstrap_iv_2sls_controles.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
