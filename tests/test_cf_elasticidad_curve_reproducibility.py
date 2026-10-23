#!/usr/bin/env python3
"""
Test de reproducibilidad de cf_elasticidad_curve.npy

Verifica que la curva puede regenerarse determinísticamente desde:
- Modelo CF canónico (src/modeling.py::modelo_iv_control_function)
- Coeficientes bootstrap medios (reports/bootstrap_vhat_canonical.json)
- Grid de precios 50-800 (200 puntos)
- Nudos P5/P35/P65/P95 fijados en muestra completa
"""

import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
from patsy import build_design_matrices
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from modeling import (
    construir_instrumentos_canonico,
    build_canonical_rcs_design,
    apply_rcs_design,
    first_stage_formula,
    second_stage_formula,
)

REPORTS = Path(__file__).resolve().parents[1] / 'reports' / 'certified'
DATA = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'demanda_limpia.csv'


def test_cf_elasticidad_curve_reproducible():
    """Test principal: la curva se regenera idénticamente (salvo precisión FP)."""

    # 1. Cargar curva existente
    existing = np.load(REPORTS / 'cf_elasticidad_curve.npy', allow_pickle=True).item()

    # 2. Regenerar usando el mismo procedimiento
    df = pd.read_csv(DATA)
    df_iv = construir_instrumentos_canonico(df).dropna(
        subset=['log_precio_prom_otros_centros']
    ).copy()

    fs_model = smf.ols(first_stage_formula(), data=df_iv).fit()
    df_iv['v_hat'] = fs_model.resid

    log_price_full = df_iv['log_checkout_price'].to_numpy()
    design_info, knots = build_canonical_rcs_design(log_price_full)

    S = apply_rcs_design(design_info, log_price_full)
    for c in S.columns:
        df_iv[c] = S[c].to_numpy()

    ss_model = smf.ols(second_stage_formula(), data=df_iv).fit()

    with open(REPORTS / 'bootstrap_vhat_canonical.json') as f:
        boot = json.load(f)

    coefs_bootstrap = {
        'rcs_1': boot['rcs_1']['media'],
        'rcs_2': boot['rcs_2']['media'],
        'rcs_3': boot['rcs_3']['media'],
    }

    price_grid = np.linspace(50.0, 800.0, 200)
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

    # 3. Verificar componentes estructurales
    assert np.array_equal(existing['price_grid'], price_grid), "Grid de precios difiere"
    assert np.array_equal(existing['knots'], knots), "Nudos difieren"

    # 4. Verificar elasticidades (tolerancia de precisión de punto flotante)
    max_abs_diff = np.abs(existing['elasticidades'] - elasticidades).max()
    assert max_abs_diff < 1e-4, f"Diferencia máxima {max_abs_diff} excede tolerancia 1e-4"

    # 5. Verificar P50 match paper
    precio_medio_real = df['checkout_price'].mean()
    idx_p50 = np.argmin(np.abs(price_grid - precio_medio_real))
    p50 = elasticidades[idx_p50]
    assert abs(p50 - (-1.5041)) < 0.01, f"P50={p50} no match paper (-1.5041 ± 0.01)"

    print(f"TEST PASSED: max_diff={max_abs_diff:.2e}, P50={p50:.6f}")


def test_cf_elasticidad_curve_json_mirror_exists():
    """Verifica que existe JSON mirror con metadatos completos."""
    json_path = REPORTS / 'cf_elasticidad_curve.json'
    assert json_path.exists(), "JSON mirror no existe"

    with open(json_path) as f:
        data = json.load(f)

    required_keys = [
        'provenance', 'specifications', 'grid', 'results', 'validation'
    ]
    for key in required_keys:
        assert key in data, f"Falta clave requerida: {key}"

    # Verificar proveniencia
    assert data['provenance']['generator_script'] == 'scripts/vm/generate_cf_elasticidad_curve_canonical.py'
    assert data['provenance']['n_bootstrap_iter'] == 5000
    assert data['provenance']['seed'] == 42
    assert data['provenance']['n_clusters'] == 77

    print("JSON MIRROR VALIDADO")


def test_cf_curve_matches_paper_values():
    """Verifica que la curva es consistente con valores reportados en paper."""
    existing = np.load(REPORTS / 'cf_elasticidad_curve.npy', allow_pickle=True).item()

    price_grid = existing['price_grid']
    elasticidades = existing['elasticidades']
    knots = existing['knots']

    # Nudos match paper (P5=121.28, P35=260.02, P65=349.26, P95=640.23)
    knots_price = np.exp(knots)
    expected_knots = [121.28, 260.02, 349.26, 640.23]
    for i, (actual, expected) in enumerate(zip(knots_price, expected_knots)):
        assert abs(actual - expected) < 0.01, f"Nudo {i}: {actual} != {expected}"

    # Elasticidad P50 match paper (-1.5041 ± 0.01)
    precio_medio_real = pd.read_csv(DATA)['checkout_price'].mean()
    idx_p50 = np.argmin(np.abs(price_grid - precio_medio_real))
    p50 = elasticidades[idx_p50]
    assert abs(p50 - (-1.5041)) < 0.01, f"P50={p50} no match paper"

    # Rango monótona decreciente
    assert elasticidades[0] > elasticidades[-1], "Curva no es monótona decreciente"
    assert elasticidades.min() < -1.5, "Elasticidad mínima no llega a -1.5"
    assert elasticidades.max() > -0.2, "Elasticidad máxima no cerca de 0"

    print(f"VALORES PAPER: P50={p50:.4f}, nudos={knots_price}, rango=[{elasticidades.min():.4f}, {elasticidades.max():.4f}]")


if __name__ == '__main__':
    test_cf_elasticidad_curve_reproducible()
    test_cf_elasticidad_curve_json_mirror_exists()
    test_cf_curve_matches_paper_values()
    print("\nTODOS LOS TESTS DE REPRODUCIBILIDAD PASARON")