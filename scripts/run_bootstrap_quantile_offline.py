#!/usr/bin/env pyhon
"""
Bootstrap Quantile Regression — Ejecución offline (batch)
Genera IC 95% percentil para τ ∈ {0.1, 0.25, 0.5, 0.75, 0.9}

Uso:
    nohup python run_bootstrap_quantile_offline.py > bootstrap_quantile.log 2>&1 &
    tail -f bootstrap_quantile.log
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from modeling import bootstrap_quantile_regression

# ─── Configuración ──────────────────────────────────────────────
N_ITER = 200          # Subir a 500-1000 para publicación final
TAUS = [0.1, 0.25, 0.5, 0.75, 0.9]
SEED = 42
OUT_DIR = Path('reports/notebook_outputs')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / 'bootstrap_quantile_ci.json'
LOG_FILE = OUT_DIR / 'bootstrap_quantile_detailed.log'
# ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BOOTSTRAP QUANTILE REGRESSION — OFFLINE")
    print(f"Dataset: 456,548 obs | Iteraciones: {N_ITER} | Cuantiles: {TAUS}")
    print(f"Salida: {OUT_FILE}")
    print("=" * 60)

    # Cargar datos
    t0 = time.time()
    df = pd.read_csv('data/processed/demanda_limpia.csv')
    print(f"[{time.time()-t0:.1f}s] Datos cargados: {len(df):,} filas")

    # Fórmula idéntica al modelo principal (Fase 3)
    formula = (
        'log_num_orders ~ log_checkout_price '
        '+ emailer_for_promotion + homepage_featured + hubo_aumento '
        '+ C(category) + C(center_type)'
    )

    resultados = {}

    # Loop secuencial por cuantil
    for i, tau in enumerate(TAUS, 1):
        print(f"\n--- Cuantil {i}/{len(TAUS)}: τ={tau} ---")
        t1 = time.time()

        res = bootstrap_quantile_regression(
            df, tau=tau, n_iteraciones=N_ITER, semilla=SEED, formula=formula
        )

        elapsed = time.time() - t1
        resultados[str(tau)] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in res.items()
        }

        print(f"  τ={tau}: media={res['elasticidad_media']:.4f}  "
              f"IC95%=[{res['ic_95_inferior']:.4f}, {res['ic_95_superior']:.4f}]  "
              f"n_valid={res['n_iteraciones_validas']}/{N_ITER}  "
              f"t={elapsed:.1f}s")

       # Guardado incremental (por si se interrumpe)
        with open(OUT_FILE, 'w') as f:
            json.dump(resultados, f, indent=2)

    # Resumen final
    total_time = time.time() - t0
    print("\n" + "=" * 60)
    print("COMPLETADO")
    print(f"Tiempo total: {total_time/3600:.2f} horas")
    print(f"Resultados guardados en: {OUT_FILE}")
    print("=" * 60)

    # Tabla resumen legible
    print("\nRESUMEN IC 95%:")
    print(f"{'τ':>6} | {'Media':>8} | {'IC Inf':>8} | {'IC Sup':>8} | {'Std':>7} | {'Validas':>7}")
    print("-" * 60)
    for tau in TAUS:
        r = resultados[str(tau)]
        print(f"{tau:>6} | {r['elasticidad_media']:>8.4f} | "
              f"{r['ic_95_inferior']:>8.4f} | {r['ic_95_superior']:>8.4f} | "
              f"{r['elasticidad_std']:>7.4f} | {r['n_iteraciones_validas']:>7}")

if __name__ == '__main__':
    main()
