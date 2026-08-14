"""
ETAPA 3 — Computaciones pesadas para VM (separado del pipeline principal de notebooks)

Este script contiene ÚNICAMENTE las computaciones que tardan horas y requieren
recursos dedicados (GPU + 32GB RAM). El usuario lo ejecutará en una VM aparte.

PESADOS (tiempos estimados en VM con GPU 100% dedicado + 32GB RAM):
1. bootstrap_quantile_regression: 5 cuantiles (τ=0.1, 0.25, 0.5, 0.75, 0.9) × 1000 iteraciones
   → ~13.5 horas (reportado por el usuario en ejecución previa)
2. bootstrap_elasticidad_clusterizado (panel FE): 200 iteraciones, cluster por center_id
   → ~2-3 horas

OUTPUTS:
- reports/notebook_outputs/bootstrap_quantile_results.npy
- reports/notebook_outputs/bootstrap_panel_fe_results.npy

El script principal (reproducir_resultados_etapa3.py) DEJARÁ estos resultados
como placeholders ("PENDIENTE_VM") y se completarán al recibir los .npy del usuario.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from multiprocessing import Process

# Rutas
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from modeling import (
    bootstrap_quantile_regression,
    bootstrap_elasticidad_clusterizado,
)

# Archivos de entrada/salida
DATA_CSV = PROJECT_ROOT / "data" / "processed" / "demanda_limpia_v2.csv"
OUT_DIR = PROJECT_ROOT / "reports" / "notebook_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUANTILE_OUT = OUT_DIR / "bootstrap_quantile_results.npy"
PANEL_FE_OUT = OUT_DIR / "bootstrap_panel_fe_results.npy"


def cargar_datos():
    print("[1/3] Cargando demanda_limpia_v2.csv ...")
    df = pd.read_csv(DATA_CSV)
    print(f"     N = {len(df):,} filas, {len(df.columns)} columnas")
    print(f"     Hash esperado: 90d41670226e961b3eb3ceb6b5b312575082280198a6898b3b1f1bccf5a4707e")
    return df


def run_bootstrap_quantile(df):
    """Bootstrap Quantile Regression: 1000 iteraciones × 5 cuantiles"""
    print("\n" + "=" * 70)
    print("[2/3] BOOTSTRAP QUANTILE REGRESSION (1000 iter × 5 cuantiles)")
    print("=" * 70)
    print("Iniciando... esto tardará ~13.5 horas en VM con GPU + 32GB RAM")

    taus = [0.1, 0.25, 0.5, 0.75, 0.9]
    results = {}

    for tau in taus:
        print(f"\n  → Cuantil τ = {tau} (1000 iteraciones)...")
        res = bootstrap_quantile_regression(df, tau=tau, n_iteraciones=1000, semilla=42)
        results[f'tau_{tau}'] = {
            'coef': res['elasticidad_media'],
            'ee': res['elasticidad_std'],
            'ic95': (res['ic_95_inferior'], res['ic_95_superior']),
            'n_iter_validas': res['n_iteraciones_validas'],
        }
        print(f"     coef = {res['elasticidad_media']:.4f}, IC95 = [{res['ic_95_inferior']:.4f}, {res['ic_95_superior']:.4f}], válidas = {res['n_iteraciones_validas']}")

    # Guardar resultados
    np.save(QUANTILE_OUT, results)
    print(f"\n✓ Resultados guardados en: {QUANTILE_OUT}")

    return results


def run_bootstrap_panel_fe(df):
    """Bootstrap clusterizado Panel FE: 200 iteraciones, cluster por center_id"""
    print("\n" + "=" * 70)
    print("[3/3] BOOTSTRAP PANEL FE CLUSTERIZADO (200 iteraciones)")
    print("=" * 70)
    print("Iniciando... esto tardará ~2-3 horas en VM")

    res = bootstrap_elasticidad_clusterizado(df, cluster_col='center_id', n_iteraciones=200, semilla=42)

    results = {
        'elasticidad_media': res['elasticidad_media'],
        'elasticidad_std': res['elasticidad_std'],
        'ic_95_inferior': res['ic_95_inferior'],
        'ic_95_superior': res['ic_95_superior'],
        'n_iteraciones_validas': res['n_iteraciones_validas'],
    }

    np.save(PANEL_FE_OUT, results)
    print(f"\n✓ Resultados guardados en: {PANEL_FE_OUT}")
    print(f"  Elasticidad media: {results['elasticidad_media']:.4f}")
    print(f"  IC 95%: [{results['ic_95_inferior']:.4f}, {results['ic_95_superior']:.4f}]")
    print(f"  Iteraciones válidas: {results['n_iteraciones_validas']}/200")

    return results


def main():
    print("=" * 70)
    print("ETAPA 3 — COMPUTACIONES PESADAS PARA VM")
    print("=" * 70)
    print("""
Este script ejecuta SOLO las computaciones que requieren horas:
  1. bootstrap_quantile_regression (1000 iter × 5 cuantiles) → ~13.5h
  2. bootstrap_elasticidad_clusterizado panel FE (200 iter)    → ~2-3h

Requisitos VM: GPU 100% dedicado, 32GB RAM
Dataset: demanda_limpia_v2.csv (hash: 90d41670226e961b3eb3ceb6b5b312575082280198a6898b3b1f1bccf5a4707e)
""")

    df = cargar_datos()

    # Ejecutar en paralelo (procesos separados para aprovechar multi-core)
    processes = []
    p1 = Process(target=run_bootstrap_quantile, args=(df,))
    p2 = Process(target=run_bootstrap_panel_fe, args=(df,))
    processes.extend([p1, p2])

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print("\n" + "=" * 70)
    print("✅ TODAS LAS COMPUTACIONES PESADAS COMPLETADAS")
    print("=" * 70)
    print(f"Resultados guardados en:")
    print(f"  - {QUANTILE_OUT}")
    print(f"  - {PANEL_FE_OUT}")
    print("\nEntregar estos archivos .npy para completar el pipeline principal.")


if __name__ == "__main__":
    main()