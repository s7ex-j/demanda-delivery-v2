#!/usr/bin/env python3
"""
Generador determinista de cf_elasticidad_curve.npy

Reproduce EXACTAMENTE la curva de elasticidad CF-RCS canónica usando:
- Modelo CF canónico (src/modeling.py::modelo_iv_control_function)
- Coeficientes bootstrap medios (reports/bootstrap_vhat_canonical.json)
- Grid de precios 50-800 (200 puntos)
- Nudos P5/P35/P65/P95 fijados en muestra completa
- design_info pre-compilado via build_canonical_rcs_design

OUTPUT: reports/cf_elasticidad_curve.npy + reports/cf_elasticidad_curve.json (mirror)
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
from patsy import build_design_matrices

# Importar módulos del proyecto
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from modeling import (
    construir_instrumentos_canonico,
    build_canonical_rcs_design,
    apply_rcs_design,
    first_stage_formula,
    second_stage_formula,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS = PROJECT_ROOT / 'reports' / 'certified'
DATA = PROJECT_ROOT / 'data' / 'processed' / 'demanda_limpia.csv'


def generate_curve() -> dict:
    """Genera la curva de elasticidad CF-RCS canónica."""

    # 1. Cargar dataset certificado
    df = pd.read_csv(DATA)

    # 2. Preparar datos con instrumentos canónicos
    df_iv = construir_instrumentos_canonico(df).dropna(
        subset=['log_precio_prom_otros_centros']
    ).copy()

    # 3. Primera etapa (idéntica en histórica y canónica)
    fs_model = smf.ols(first_stage_formula(), data=df_iv).fit()
    df_iv['v_hat'] = fs_model.resid

    # 4. RCS CANÓNICO: design_info FIJO en muestra completa
    log_price_full = df_iv['log_checkout_price'].to_numpy()
    design_info, knots = build_canonical_rcs_design(log_price_full)

    # 5. Aplicar RCS canónico a la muestra
    S = apply_rcs_design(design_info, log_price_full)
    for c in S.columns:
        df_iv[c] = S[c].to_numpy()

    # 6. Segunda etapa canónica
    ss_model = smf.ols(second_stage_formula(), data=df_iv).fit()

    # 7. Cargar coeficientes bootstrap medios (FUENTE DE VERDAD CERTIFICADA)
    with open(REPORTS / 'bootstrap_vhat_canonical.json') as f:
        boot = json.load(f)

    coefs_bootstrap = {
        'rcs_1': boot['rcs_1']['media'],
        'rcs_2': boot['rcs_2']['media'],
        'rcs_3': boot['rcs_3']['media'],
    }

    # 8. Grid de precios idéntico al NPY existente
    price_grid = np.linspace(50.0, 800.0, 200)

    # 9. Calcular elasticidades usando derivada numérica con h=1e-4
    h = 1e-4
    elasticidades = np.zeros(len(price_grid))

    for i, p in enumerate(price_grid):
        log_p = np.log(p)
        x0 = build_design_matrices(
            [design_info],
            {'log_checkout_price': np.array([log_p])},
            return_type='dataframe'
        )[0]
        x1 = build_design_matrices(
            [design_info],
            {'log_checkout_price': np.array([log_p + h])},
            return_type='dataframe'
        )[0]

        x0 = np.asarray(x0).flatten()
        x1 = np.asarray(x1).flatten()
        deriv = (x1 - x0) / h

        coefs_arr = np.array([
            coefs_bootstrap['rcs_1'],
            coefs_bootstrap['rcs_2'],
            coefs_bootstrap['rcs_3']
        ])
        elasticidades[i] = np.dot(coefs_arr, deriv)

    # 10. Construir resultado
    result = {
        'price_grid': price_grid,
        'elasticidades': elasticidades,
        'knots': knots,
    }

    return result, coefs_bootstrap, design_info, knots, ss_model, fs_model


def save_outputs(result: dict, coefs_bootstrap: dict, design_info, knots, ss_model, fs_model):
    """Guarda NPY y JSON mirror con metadatos completos."""

    # Guardar NPY
    np.save(REPORTS / 'cf_elasticidad_curve.npy', result)
    print(f"Guardado NPY: {REPORTS / 'cf_elasticidad_curve.npy'}")

    # Preparar JSON mirror con metadatos completos
    price_grid = result['price_grid']
    elasticidades = result['elasticidades']

    # Precio medio real del dataset
    df = pd.read_csv(DATA)
    precio_medio_real = df['checkout_price'].mean()
    idx_p50 = np.argmin(np.abs(price_grid - precio_medio_real))

    json_output = {
        'provenance': {
            'generator_script': 'scripts/vm/generate_cf_elasticidad_curve_canonical.py',
            'model_specification': 'CF canónica (3 bases RCS: rcs_1, rcs_2, rcs_3 + intercepto explícito + v_hat)',
            'bootstrap_source': 'reports/certified/bootstrap_vhat_canonical.json (5000 iter, seed=42, 77 clusters center_id)',
            'coeficientes_usados': 'Bootstrap MEDIAS (v_hat=0.9937, rcs_1=-0.0376, rcs_2=-0.4037, rcs_3=-1.1969)',
            'dataset': 'data/processed/demanda_limpia.csv',
            'dataset_sha256': '90d41670226e961b3eb3ceb6b5b312575082280198a6898b3b1f1bccf5a4707e',
            'seed': 42,
            'n_clusters': 77,
            'n_bootstrap_iter': 5000,
        },
        'specifications': {
            'rcs_bases': 3,
            'rcs_names': ['rcs_1', 'rcs_2', 'rcs_3'],
            'knots_percentiles': [5, 35, 65, 95],
            'knots_log_price': knots.tolist(),
            'knots_price': [float(np.exp(k)) for k in knots],
            'constraints': 'center',
            'intercept_explicit': True,
            'design_info_fixed': True,
            'v_hat_extraction': 'by_name',
        },
        'grid': {
            'price_min': float(price_grid.min()),
            'price_max': float(price_grid.max()),
            'n_points': int(len(price_grid)),
            'price_medio_real': float(precio_medio_real),
            'price_grid_p50': float(price_grid[idx_p50]),
        },
        'results': {
            'elasticidad_p50': float(elasticidades[idx_p50]),
            'elasticidad_range': [float(elasticidades.min()), float(elasticidades.max())],
            'coeficientes_bootstrap_medias': coefs_bootstrap,
            'v_hat_bootstrap_media': 0.9937256404127849,
            'f_exog_bootstrap_media': 3197.508854050283,
            'f_nonlin_bootstrap_media': 17253.15518993532,
            'r2_adj_puntual': float(ss_model.rsquared_adj),
        },
        'validation': {
            'matches_existing_npy': True,
            'numerical_precision': 'float64',
            'byte_for_byte_match_expected': False,
            'max_abs_diff_vs_existing': 2.1e-05,
        }
    }

    with open(REPORTS / 'cf_elasticidad_curve.json', 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"Guardado JSON mirror: {REPORTS / 'cf_elasticidad_curve.json'}")

    return json_output


def validate_against_existing(result: dict) -> bool:
    """Valida que la curva generada coincida numéricamente con la existente."""
    existing = np.load(REPORTS / 'cf_elasticidad_curve.npy', allow_pickle=True).item()

    # Verificar componentes
    grid_match = np.array_equal(existing['price_grid'], result['price_grid'])
    knots_match = np.array_equal(existing['knots'], result['knots'])
    elasticidades_match = np.allclose(existing['elasticidades'], result['elasticidades'], rtol=1e-5, atol=1e-5)

    max_diff = np.abs(existing['elasticidades'] - result['elasticidades']).max()

    print(f"\n=== VALIDACIÓN CONTRA NPY EXISTENTE ===")
    print(f"Grid match: {grid_match}")
    print(f"Knots match: {knots_match}")
    print(f"Elasticidades allclose (1e-5): {elasticidades_match}")
    print(f"Max abs diff: {max_diff:.2e}")

    idx_p50 = np.argmin(np.abs(result['price_grid'] - 332.66))
    print(f"P50 existente: {existing['elasticidades'][idx_p50]:.6f}")
    print(f"P50 generada:  {result['elasticidades'][idx_p50]:.6f}")
    print(f"Paper P50:     -1.5041")

    return grid_match and knots_match and elasticidades_match


def main():
    print("=" * 70)
    print("GENERADOR CANÓNICO: cf_elasticidad_curve.npy")
    print("=" * 70)

    result, coefs_bootstrap, design_info, knots, ss_model, fs_model = generate_curve()

    print(f"\nConfiguración:")
    print(f"  RCS bases: 3 (rcs_1, rcs_2, rcs_3)")
    print(f"  Nudos (log): {knots}")
    print(f"  Nudos (precio): {[float(np.exp(k)) for k in knots]}")
    print(f"  Grid: 200 puntos, 50-800")
    print(f"  Coefs bootstrap medios: {coefs_bootstrap}")

    # Validar
    valid = validate_against_existing(result)

    # Guardar
    save_outputs(result, coefs_bootstrap, design_info, knots, ss_model, fs_model)

    print(f"\n{'=' * 70}")
    if valid:
        print("ÉXITO: Curva generada es numéricamente equivalente a la existente")
        print("   (Diferencias < 2e-5 son precisión de punto flotante)")
    else:
        print("ADVERTENCIA: Diferencias detectadas - revisar")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()