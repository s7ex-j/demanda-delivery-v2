"""
ETAPA 4 / P0-1 — Wild Cluster Bootstrap (Cameron, Gelbach & Miller 2008)
===========================================================================

Script para VM offline: wild cluster bootstrap con pesos Rademacher (±1)
para el modelo OLS log-log con controles, clusterizado por center_id (77 clusters).

Especificación:
- Cluster: center_id (77 centros)
- Iteraciones: 500 (recomendado para wild bootstrap con pocos clusters)
- Pesos: Rademacher (±1 con probabilidad 0.5 cada uno)
- Modelo: log_num_orders ~ log_checkout_price + controles (categoría, center_type, promociones, hubo_aumento)

Output:
- reports/notebook_outputs/wild_cluster_bootstrap_results.npy
  Dict con: elasticidad_media, elasticidad_std, ic_95_inferior, ic_95_superior, n_iteraciones_validas
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Rutas
def _find_project_root(start: Path) -> Path:
    """Busca hacia arriba desde start la carpeta que contiene src/modeling.py
    y data/processed/, sin importar en qué subcarpeta se coloque este script."""
    for candidate in [start] + list(start.parents):
        if (candidate / "src" / "modeling.py").exists() and (candidate / "data" / "processed").is_dir():
            return candidate
    raise RuntimeError(
        "No se pudo encontrar la raíz del proyecto (se buscó src/modeling.py y "
        "data/processed/ subiendo desde " + str(start) + "). "
        "Verifica que el proyecto tenga esa estructura."
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _find_project_root(SCRIPT_DIR)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_CSV = PROJECT_ROOT / "data" / "processed" / "demanda_limpia_v2.csv"
OUT_DIR = PROJECT_ROOT / "reports" / "notebook_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WILD_BOOT_OUT = OUT_DIR / "wild_cluster_bootstrap_results.npy"


def cargar_datos():
    print("[1/2] Cargando demanda_limpia_v2.csv ...")
    df = pd.read_csv(DATA_CSV)
    print(f"     N = {len(df):,} filas, {len(df.columns)} columnas")
    print(f"     Centros únicos: {df['center_id'].nunique()}")
    return df


def run_wild_cluster_bootstrap(df, n_iteraciones=500, semilla=42):
    """
    Wild cluster bootstrap para OLS con clustering por center_id.

    Usa pesos Rademacher (±1) por cluster. Recomendado cuando número de clusters es pequeño
    (aquí 77 centros). Cameron, Gelbach, Miller (2008) y Webb (2014).

    Returns:
        dict con elasticidad_media, elasticidad_std, ic_95_inferior, ic_95_superior, n_iteraciones_validas
    """
    rng = np.random.default_rng(semilla)
    clusters = df['center_id'].unique()
    n_clusters = len(clusters)

    formula = (
        'log_num_orders ~ log_checkout_price '
        '+ emailer_for_promotion + homepage_featured + hubo_aumento '
        '+ C(category) + C(center_type)'
    )

    # Modelo original
    modelo_base = smf.ols(formula, data=df).fit()
    coef_base = modelo_base.params.get('log_checkout_price', np.nan)
    residuos = modelo_base.resid
    X = modelo_base.model.exog

    coefs_boot = np.empty(n_iteraciones)

    print(f"\n{'='*70}")
    print(f"WILD CLUSTER BOOTSTRAP (Rademacher) - Cameron et al. 2008")
    print(f"{'='*70}")
    print(f"Clusters: {n_clusters} centros | Iteraciones: {n_iteraciones}")
    print(f"Semilla: {semilla}")

    import time
    t_inicio = time.time()
    CHECKPOINT_CADA = 100
    CHECKPOINT_OUT = OUT_DIR / "wild_cluster_bootstrap_results_CHECKPOINT.npy"

    for i in range(n_iteraciones):
        if i % 50 == 0 and i > 0:
            transcurrido = time.time() - t_inicio
            ritmo = transcurrido / i
            eta_min = ritmo * (n_iteraciones - i) / 60
            n_validas_hasta_ahora = int(np.sum(~np.isnan(coefs_boot[:i])))
            print(f"  Progreso: {i}/{n_iteraciones}  "
                  f"({transcurrido/60:.1f} min transcurridos, ~{eta_min:.1f} min restantes, "
                  f"{n_validas_hasta_ahora} válidas hasta ahora)")

        if i > 0 and i % CHECKPOINT_CADA == 0:
            parcial = coefs_boot[:i]
            parcial_v = parcial[~np.isnan(parcial)]
            checkpoint = {
                'elasticidad_media': float(np.mean(parcial_v)) if len(parcial_v) > 0 else np.nan,
                'elasticidad_std': float(np.std(parcial_v)) if len(parcial_v) > 0 else np.nan,
                'ic_95_inferior': float(np.percentile(parcial_v, 2.5)) if len(parcial_v) > 0 else np.nan,
                'ic_95_superior': float(np.percentile(parcial_v, 97.5)) if len(parcial_v) > 0 else np.nan,
                'n_iteraciones_validas': int(len(parcial_v)),
                'CHECKPOINT_PARCIAL': True,
                'iteraciones_completadas_hasta_checkpoint': i,
            }
            np.save(CHECKPOINT_OUT, checkpoint)
            print(f"  [checkpoint guardado en {CHECKPOINT_OUT.name}: {i}/{n_iteraciones} iteraciones]")

        # Pesos Rademacher por cluster
        weights = rng.choice([-1, 1], size=n_clusters)
        weight_map = dict(zip(clusters, weights))
        w = df['center_id'].map(weight_map).to_numpy()

        # Y* = X*beta + w * residuos
        y_star = X @ modelo_base.params + w * residuos

        # Re-estimar en el dataset original con Y* como dependiente
        # Para esto, creamos una copia del df con la nueva variable dependiente
        df_star = df.copy()
        df_star['log_num_orders_star'] = y_star

        formula_star = (
            'log_num_orders_star ~ log_checkout_price '
            '+ emailer_for_promotion + homepage_featured + hubo_aumento '
            '+ C(category) + C(center_type)'
        )
        try:
            modelo_b = smf.ols(formula_star, data=df_star).fit()
            coefs_boot[i] = modelo_b.params.get('log_checkout_price', np.nan)
        except Exception:
            coefs_boot[i] = np.nan
            continue

    coefs_boot = coefs_boot[~np.isnan(coefs_boot)]

    results = {
        'elasticidad_media': float(np.mean(coefs_boot)) if len(coefs_boot) > 0 else np.nan,
        'elasticidad_std': float(np.std(coefs_boot)) if len(coefs_boot) > 0 else np.nan,
        'ic_95_inferior': float(np.percentile(coefs_boot, 2.5)) if len(coefs_boot) > 0 else np.nan,
        'ic_95_superior': float(np.percentile(coefs_boot, 97.5)) if len(coefs_boot) > 0 else np.nan,
        'n_iteraciones_validas': len(coefs_boot),
        'nota': 'Wild cluster bootstrap (Rademacher) por center_id. Cameron, Gelbach & Miller (2008).'
    }

    # Guardar
    np.save(WILD_BOOT_OUT, results)
    if CHECKPOINT_OUT.exists():
        CHECKPOINT_OUT.unlink()

    print(f"\nWild cluster bootstrap completado: {len(coefs_boot)}/{n_iteraciones} iteraciones válidas")
    print(f"  Elasticidad media: {results['elasticidad_media']:.4f}")
    print(f"  EE bootstrap: {results['elasticidad_std']:.4f}")
    print(f"  IC 95%: [{results['ic_95_inferior']:.4f}, {results['ic_95_superior']:.4f}]")
    print(f"  Resultados guardados en: {WILD_BOOT_OUT}")

    return results


def main():
    print("=" * 70)
    print("ETAPA 4/P0-1 — WILD CLUSTER BOOTSTRAP (VM OFFLINE)")
    print("=" * 70)
    print(f"""
Este script implementa el wild cluster bootstrap (Rademacher weights)
para validar la inferencia clusterizada con pocos clusters (77 centros).

Referencia metodológica:
  Cameron, Gelbach & Miller (2008) - "Bootstrap-Based Improvements for Inference with Clustered Errors"
  Webb (2014) - "Reworking Wild Bootstrap Based Inference for Clustered Errors"

Dataset: demanda_limpia_v2.csv (hash: 90d41670226e961b3eb3ceb6b5b312575082280198a6898b3b1f1bccf5a4707e)
Output: reports/notebook_outputs/wild_cluster_bootstrap_results.npy
""")

    df = cargar_datos()
    results = run_wild_cluster_bootstrap(df, n_iteraciones=500, semilla=42)

    print("\n" + "=" * 70)
    print("WILD CLUSTER BOOTSTRAP COMPLETADO")
    print("=" * 70)
    print(f"Entregar: {WILD_BOOT_OUT}")


if __name__ == "__main__":
    main()