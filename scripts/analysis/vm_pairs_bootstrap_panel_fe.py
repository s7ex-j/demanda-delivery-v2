"""Pairs cluster bootstrap del panel con efectos fijos de dos vías (entidad y semana).

Remuestrea los 77 centros con reemplazo (semilla 42) y re-etiqueta cada copia de un centro
para que las entidades centro x comida duplicadas no colisionen. En cada réplica estima la
elasticidad within con un estimador de dos vías por demeaning alternado, que da el mismo
coeficiente que linearmodels.PanelOLS (se comprueba en la muestra completa antes de empezar)
pero unas 60 veces más rápido.

Salidas en reports/diagnostics/:
  pairs_bootstrap_panel_fe_results.npy      resumen (media, DE, IC percentil)
  pairs_bootstrap_panel_fe_replicas.npz     las réplicas brutas (sin pickle)

Uso: python scripts/analysis/vm_pairs_bootstrap_panel_fe.py --iterations 1000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vm"))
from vm_common import OUT_DIR, load_data, save_result  # noqa: E402

COLS = ["log_num_orders", "log_checkout_price", "emailer_for_promotion", "homepage_featured"]


def coef_within(df: pd.DataFrame, entidad: np.ndarray, tol: float = 1e-12) -> float:
    """Coeficiente de log_checkout_price con efectos fijos de entidad y semana."""
    e = pd.factorize(entidad)[0]
    t = pd.factorize(df["week"].to_numpy())[0]
    ne, nt = e.max() + 1, t.max() + 1
    ce, ct = np.bincount(e, minlength=ne), np.bincount(t, minlength=nt)
    M = df[COLS].to_numpy(float).copy()
    for _ in range(1000):
        old = M.copy()
        for j in range(M.shape[1]):
            M[:, j] -= (np.bincount(e, M[:, j], ne) / ce)[e]
            M[:, j] -= (np.bincount(t, M[:, j], nt) / ct)[t]
        if np.max(np.abs(M - old)) < tol:
            break
    return float(np.linalg.lstsq(M[:, 1:], M[:, 0], rcond=None)[0][0])


def comprobar_contra_panelols(df: pd.DataFrame, base: float) -> None:
    from linearmodels.panel import PanelOLS
    d = df.assign(entity=df.center_id.astype(str) + "_" + df.meal_id.astype(str)).set_index(["entity", "week"])
    r = PanelOLS.from_formula("log_num_orders ~ log_checkout_price + emailer_for_promotion + "
                              "homepage_featured + EntityEffects + TimeEffects", d).fit()
    ref = float(r.params["log_checkout_price"])
    assert abs(ref - base) < 1e-9, (ref, base)
    print(f"PanelOLS {ref:.10f} = within {base:.10f}", flush=True)


def run(n_iter: int = 1000, seed: int = 42) -> dict:
    df = load_data()
    meal = df["meal_id"].astype(str).to_numpy()
    base = coef_within(df, df["center_id"].astype(str).to_numpy() + "_" + meal)
    comprobar_contra_panelols(df, base)

    centros = np.unique(df.center_id)
    idx = {c: np.flatnonzero(df.center_id.to_numpy() == c) for c in centros}
    rng = np.random.default_rng(seed)
    coefs = np.full(n_iter, np.nan)
    t0 = time.time()
    for i in range(n_iter):
        draws = rng.choice(centros, size=len(centros), replace=True)
        filas = np.concatenate([idx[c] for c in draws])
        etiqueta = np.concatenate([np.full(len(idx[c]), k) for k, c in enumerate(draws)])
        sub = df.iloc[filas]
        try:
            coefs[i] = coef_within(sub, etiqueta.astype(str) + "_" + meal[filas])
        except Exception:  # noqa: BLE001
            pass
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{n_iter} {(time.time() - t0) / 60:.1f} min", flush=True)

    v = coefs[np.isfinite(coefs)]
    np.savez_compressed(OUT_DIR / "pairs_bootstrap_panel_fe_replicas.npz", coefs=coefs, seed=seed, coef_base=base)
    r = {"coef_media": float(v.mean()), "coef_std": float(v.std(ddof=1)),
         "ic_95_inferior": float(np.percentile(v, 2.5)), "ic_95_superior": float(np.percentile(v, 97.5)),
         "n_iteraciones_validas": int(v.size), "n_iteraciones_total": n_iter, "coef_base": base,
         "clusters": int(len(centros)), "seed": seed,
         "nota": "Pairs bootstrap por center_id con re-etiquetado de cada copia; FE de entidad y semana. "
                 "Réplicas brutas en pairs_bootstrap_panel_fe_replicas.npz."}
    save_result(r, "pairs_bootstrap_panel_fe_results.npy")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(run(a.iterations, a.seed))
