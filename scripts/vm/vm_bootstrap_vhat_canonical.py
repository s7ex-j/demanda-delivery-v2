#!/usr/bin/env python3
"""
Bootstrap canónico del Control Function con RCS corregido.

Especificación:
- Cluster bootstrap por center_id (77 clusters)
- RCS canónica: nudos P5/P35/P65/P95, 3 bases (rcs_1, rcs_2, rcs_3)
- Primera etapa: OLS con instrumentos Hausman-Nevo
- Segunda etapa: OLS con RCS + v_hat + controles
- Ambas etapas reestimadas en cada réplica
- Seed=42, 5000 iteraciones
- Checkpointing cada 100 iteraciones
- Percentil bootstrap para IC 95%

V3.2.5: SUPERADO por scripts/vm/bootstrap_cf_canonical_replicas.py, que reproduce exactamente
la misma secuencia de remuestreo, guarda las réplicas brutas (.npz) y calcula la banda de
elasticidad y el F anidado de no linealidad. Este script se conserva por trazabilidad
(se corrigieron la ruta ROOT y el import de build_design_matrices).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import build_design_matrices  # V3.2.5: import faltante

ROOT = Path(__file__).resolve().parents[2]  # V3.2.5: raíz del proyecto (antes parents[1] = scripts/)
sys.path.insert(0, str(ROOT / "src"))
from rcs_utils import (
    build_canonical_rcs_design,
    first_stage_formula,
    second_stage_formula,
    construir_instrumentos,
)


def configure_vm():
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "4")


def checkpoint_path(out: Path) -> Path:
    return out / "bootstrap_vhat_canonical_CHECKPOINT.pkl"


def stats(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"media": np.nan, "std": np.nan, "ic95_inf": np.nan, "ic95_sup": np.nan, "n": 0}
    return {
        "media": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "ic95_inf": float(np.percentile(arr, 2.5)),
        "ic95_sup": float(np.percentile(arr, 97.5)),
        "n": int(arr.size)
    }


def save_ckpt(path: Path, state: dict):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def main():
    configure_vm()
    ap = argparse.ArgumentParser(description="Bootstrap canónico Control Function con RCS P5/P35/P65/P95.")
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "demanda_limpia.csv"))
    ap.add_argument("--output-dir", default=str(ROOT / "reports" / "certified"))
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint_path(out)

    if args.restart and ckpt.exists():
        ckpt.unlink()

    # Cargar datos
    df = pd.read_csv(args.data)
    required = {"center_id", "meal_id", "week", "base_price", "checkout_price",
                "log_checkout_price", "log_num_orders", "emailer_for_promotion",
                "homepage_featured", "hubo_aumento", "category", "center_type"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    # Construir instrumentos en muestra completa (para RCS design)
    df = construir_instrumentos(df).dropna(subset=['log_precio_prom_otros_centros']).copy()

    # RCS canónica: pre-compilar en muestra completa
    log_price_full = df["log_checkout_price"].to_numpy()
    design_info, knots = build_canonical_rcs_design(log_price_full)

    # Clusters
    clusters = np.asarray(sorted(df["center_id"].dropna().unique()))
    grouped = {c: g for c, g in df.groupby("center_id", sort=False)}

    # Arrays de estadísticas bootstrap
    names = ["vhat", "rcs_1", "rcs_2", "rcs_3", "f_exog", "f_nonlinearity"]
    state = {
        "version": 1,
        "completed": 0,
        "valid": 0,
        "invalid": 0,
        "n_iter": args.iterations,
        "rng_state": None
    }
    arrays = {n: np.full(args.iterations, np.nan, dtype=float) for n in names}

    if ckpt.exists():
        with ckpt.open("rb") as f:
            stored = pickle.load(f)
        if stored.get("n_iter") != args.iterations:
            raise RuntimeError("Checkpoint incompatible; usa --restart.")
        state.update({k: stored[k] for k in state if k in stored})
        for n in names:
            if n in stored["arrays"]:
                arrays[n][:] = stored["arrays"][n]

    rng = np.random.default_rng(args.seed)
    if state["rng_state"] is not None:
        rng.bit_generator.state = state["rng_state"]

    t0 = time.time()

    for i in range(state["completed"], args.iterations):
        # Remuestrear clusters con reemplazo
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([grouped[c] for c in sampled], ignore_index=True)

        try:
            work = sample.copy()

            # Reconstruir instrumentos en la réplica (leave-one-out mean)
            group = work.groupby(["meal_id", "week"])["checkout_price"]
            total = group.transform("sum")
            count = group.transform("count")
            other = (total - work["checkout_price"]) / (count - 1).replace(0, np.nan)
            work["log_base_price"] = np.log(work["base_price"])
            work["log_precio_prom_otros_centros"] = np.log(other)
            work = work.dropna(subset=["log_precio_prom_otros_centros"]).copy()

            # PRIMERA ETAPA: OLS -> v_hat
            fs = smf.ols(first_stage_formula(), data=work).fit()
            work["v_hat"] = fs.resid

            # RCS: usar design_info FIJO (nudos de muestra completa)
            S = build_design_matrices([design_info],
                                       {'log_checkout_price': work["log_checkout_price"].to_numpy()},
                                       return_type='dataframe')[0]
            S.columns = ['rcs_1', 'rcs_2', 'rcs_3']
            for c in S.columns:
                work[c] = S[c].to_numpy()

            # SEGUNDA ETAPA: OLS con RCS + v_hat + controles
            ss = smf.ols(second_stage_formula(), data=work).fit()
            params = ss.params

            # Extraer coeficientes
            vals = [
                params.get("v_hat", np.nan),
                params.get("rcs_1", np.nan),
                params.get("rcs_2", np.nan),
                params.get("rcs_3", np.nan),
                np.nan,  # f_exog
                np.nan,  # f_nonlinearity
            ]

            # Test exogeneidad: H0: coef(v_hat) = 0 (1 g.l.)
            if "v_hat" in params.index:
                R = np.zeros((1, len(params)))
                R[0, list(params.index).index("v_hat")] = 1
                vals[4] = float(np.asarray(ss.wald_test(R).fvalue).squeeze())

            # Test no-linealidad condicional: H0: coef(rcs_2)=coef(rcs_3)=0 (2 g.l.)
            R = np.zeros((2, len(params)))
            for j, nm in enumerate(["rcs_2", "rcs_3"]):
                if nm in params.index:
                    R[j, list(params.index).index(nm)] = 1
            vals[5] = float(np.asarray(ss.wald_test(R).fvalue).squeeze())

            if not all(np.isfinite(vals[:5])):
                raise ValueError("parámetros no finitos")

            for n, v in zip(names, vals):
                arrays[n][i] = v
            state["valid"] += 1

        except Exception as exc:
            state["invalid"] += 1
            print(f"Iteración {i+1}: descartada ({exc})", flush=True)

        state["completed"] = i + 1
        state["rng_state"] = rng.bit_generator.state

        if (i + 1) % args.checkpoint_every == 0 or i + 1 == args.iterations:
            save_ckpt(ckpt, {**state, "arrays": arrays})
            print(f"checkpoint {i+1}/{args.iterations} | valid={state['valid']} invalid={state['invalid']} | {(time.time()-t0)/60:.1f} min", flush=True)

    # Resultados finales
    payload = {
        "version": 1,
        "n_iteraciones_total": args.iterations,
        "n_iteraciones_validas": state["valid"],
        "n_descartadas": state["invalid"],
        "seed": args.seed,
        "cluster_variable": "center_id",
        "n_clusters": len(clusters),
        "nudos_percentiles": [5, 35, 65, 95],
        "nudos_log_precio": knots.tolist(),
        "nudos_precio": np.exp(knots).tolist(),
        "v_hat": stats(arrays["vhat"]),
        "rcs_1": stats(arrays["rcs_1"]),
        "rcs_2": stats(arrays["rcs_2"]),
        "rcs_3": stats(arrays["rcs_3"]),
        "test_exogeneidad": stats(arrays["f_exog"]),
        "test_no_linealidad_condicional": stats(arrays["f_nonlinearity"]),
        "nota": ("Bootstrap clusterizado canónico por center_id (77 clusters). "
                 "RCS con nudos explícitos P5/P35/P65/P95, 3 bases cr centradas (rcs_1..rcs_3). "
                 "Ambas etapas reestimadas en cada réplica. IC percentil bootstrap."),
        "provenance": {
            "has_raw_replicates": False,
            "stored_content": "summary_statistics_only",
            "visualization_method": "percentile_bootstrap_IC_from_empirical_replicates",
            "bootstrap_type": "cluster_bootstrap",
            "replicates_available": "summary_only_mean_std_IC",
            "theoretical_approximation": "none_empirical_percentile_IC",
            "description": "5000 raw replicates computed but only summary statistics (mean, std, percentile IC) stored. "
                          "For visualization with raw replicates, NPY would need shape (n_iter, n_params) array."
        }
    }

    np.save(out / "bootstrap_vhat_canonical.npy", payload, allow_pickle=True)

    # Guardar JSON legible
    def conv(x):
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, np.ndarray):
            return conv(x.tolist())
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x

    json_path = out / "bootstrap_vhat_canonical.json"
    tmp = json_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(conv(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(json_path)

    ckpt.unlink(missing_ok=True)
    print(json.dumps(conv(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()