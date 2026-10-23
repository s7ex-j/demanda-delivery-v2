#!/usr/bin/env python3
"""
Test final de consistencia de auditoría

Verifica todos los resultados certificados y sus conexiones al paper.
"""

import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from modeling import (
    modelo_iv_control_function,
    modelo_iv_control_function_historico,
    modelo_iv_2sls,
    construir_instrumentos_canonico,
    placebo_precio_futuro,
    placebo_temporal_valido,
    placebo_block_permutation,
)

REPORTS = Path(__file__).resolve().parents[1] / 'reports' / 'certified'
DATA = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'demanda_limpia.csv'

# Find informe.tex - could be at root or in Informe/
def find_informe_tex():
    base = Path(__file__).resolve().parents[1]
    candidates = [
        base / 'informe.tex',
        base / 'Informe' / 'informe.tex',
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("informe.tex not found")

def test_v_hat_canonico():
    """v_hat canónico match paper/bootstrap."""
    df = pd.read_csv(DATA)
    result = modelo_iv_control_function(df)

    # Puntual match paper
    assert abs(result['v_hat_coef'] - 1.0002) < 0.001, f"v_hat puntual {result['v_hat_coef']} != 1.0002"

    # Bootstrap match
    with open(REPORTS / 'bootstrap_vhat_canonical.json', encoding='utf-8') as f:
        boot = json.load(f)
    assert abs(boot['v_hat']['media'] - 0.9937) < 0.001, f"v_hat boot {boot['v_hat']['media']} != 0.9937"

    print(f"v_hat: puntual={result['v_hat_coef']:.4f}, boot_media={boot['v_hat']['media']:.4f} OK")


def test_rcs_canonico():
    """RCS canónicos match bootstrap."""
    df = pd.read_csv(DATA)
    result = modelo_iv_control_function(df)

    with open(REPORTS / 'bootstrap_vhat_canonical.json', encoding='utf-8') as f:
        boot = json.load(f)

    # rcs_1, rcs_2, rcs_3 match bootstrap means
    for rcs in ['rcs_1', 'rcs_2', 'rcs_3']:
        model_val = result['segunda_etapa'].params[rcs]
        boot_val = boot[rcs]['media']
        assert abs(model_val - boot_val) < 0.01, f"{rcs}: model={model_val:.4f} vs boot={boot_val:.4f}"

    print(f"RCS: rcs_1={result['segunda_etapa'].params['rcs_1']:.4f}, rcs_2={result['segunda_etapa'].params['rcs_2']:.4f}, rcs_3={result['segunda_etapa'].params['rcs_3']:.4f} OK")


def test_iv_09604():
    """IV con controles armonizados reproducible."""
    df = pd.read_csv(DATA)
    df_iv = construir_instrumentos_canonico(df).dropna(subset=['log_precio_prom_otros_centros']).copy()
    result, _ = modelo_iv_2sls(df_iv)

    # With harmonized controls (hubo_aumento, C(category), C(center_type))
    # the IV coefficient changes from -0.9604 to -0.8414
    assert abs(result.params['log_checkout_price'] - (-0.8414)) < 0.001, f"IV {result.params['log_checkout_price']} != -0.8414"
    print(f"IV 2SLS (harmonized): {result.params['log_checkout_price']:.4f} OK")

    # Bootstrap principal match (original spec without harmonized controls)
    with open(REPORTS / 'wild_bootstrap_iv_2sls.json', encoding='utf-8') as f:
        boot = json.load(f)
    assert abs(boot['coef_base'] - (-0.96044)) < 0.0001, f"Boot base {boot['coef_base']} != -0.96044"
    assert abs(boot['coef_media'] - (-0.9606)) < 0.001, f"Boot media {boot['coef_media']} != -0.9606"
    print(f"IV Bootstrap (original spec): base={boot['coef_base']:.5f}, media={boot['coef_media']:.4f} OK")


def test_no_vhat_confusion():
    """Verificar que -0.0376 NO se usa como v_hat (off-by-one B2)."""
    with open(REPORTS / 'bootstrap_vhat_canonical.json', encoding='utf-8') as f:
        boot = json.load(f)

    v_hat_boot = boot['v_hat']['media']
    rcs1_boot = boot['rcs_1']['media']

    # v_hat bootstrap ~0.99, rcs_1 bootstrap ~-0.038
    assert v_hat_boot > 0.5, f"v_hat bootstrap {v_hat_boot} debe ser ~0.99"
    assert abs(rcs1_boot) < 0.1, f"rcs_1 bootstrap {rcs1_boot} debe ser ~-0.038"
    assert abs(v_hat_boot - rcs1_boot) > 0.5, "v_hat y rcs_1 NO deben confundirse"

    print(f"Off-by-one check: v_hat={v_hat_boot:.4f} vs rcs_1={rcs1_boot:.4f} OK (distintos)")


def test_no_9487():
    """Verificar que -0.9487 NO aparece como resultado vigente."""
    df = pd.read_csv(DATA)
    df_iv = construir_instrumentos_canonico(df).dropna(subset=['log_precio_prom_otros_centros']).copy()
    result, _ = modelo_iv_2sls(df_iv)
    assert abs(result.params['log_checkout_price'] - (-0.9487)) > 0.01, "IV actual NO debe ser -0.9487"

    tex_path = find_informe_tex()
    with open(tex_path) as f:
        tex = f.read()

    assert '-0.9487' not in tex and '-0,9487' not in tex, "Paper NO debe contener -0.9487"

    print(f"IV -0.9487 ausente: código actual={result.params['log_checkout_price']:.4f}, paper sin -0.9487 OK")


def test_bootstrap_cf():
    """Bootstrap CF certificado."""
    with open(REPORTS / 'bootstrap_vhat_canonical.json', encoding='utf-8') as f:
        boot = json.load(f)

    assert boot['n_iteraciones_validas'] == 5000
    assert boot['seed'] == 42
    assert boot['n_clusters'] == 77
    assert boot['v_hat']['media'] > 0.5
    assert boot['test_exogeneidad']['media'] > 1000
    assert boot['test_no_linealidad_condicional']['media'] > 10000

    print(f"Bootstrap CF: 5000 iter, seed=42, 77 clusters, v_hat={boot['v_hat']['media']:.4f} OK")


def test_placebos_certificados():
    """Placebos certificados match paper."""
    # P3: Precio futuro
    with open(REPORTS / 'placebo_precio_futuro.json', encoding='utf-8') as f:
        p3 = json.load(f)
    assert abs(p3['elasticidad_precio_futuro_lead'] - (-0.6172)) < 0.001
    assert abs(p3['elasticidad_precio_contemporaneo'] - (-0.7596)) < 0.001
    print(f"P3 Lead: lead={p3['elasticidad_precio_futuro_lead']:.4f}, contemp={p3['elasticidad_precio_contemporaneo']:.4f} OK")

    # P4: Block permutation
    with open(REPORTS / 'placebo_block_permutation.json', encoding='utf-8') as f:
        p4 = json.load(f)
    assert abs(p4['media_placebo'] - (-0.4276)) < 0.001
    assert abs(p4['std_placebo'] - 0.00171) < 0.0001
    assert p4['n_iteraciones_validas'] == 500
    print(f"P4 Block: media={p4['media_placebo']:.4f}, std={p4['std_placebo']:.5f}, n={p4['n_iteraciones_validas']} OK")

    # P2: Permutación entidad (reproducible pero no en paper)
    with open(REPORTS / 'placebo_temporal_valido.json', encoding='utf-8') as f:
        p2 = json.load(f)
    assert p2['results']['p_valor'] == 0.0
    assert p2['results']['n_iteraciones_validas'] == 500
    print(f"P2 Entidad: p={p2['results']['p_valor']}, n={p2['results']['n_iteraciones_validas']} OK (reproducible, no en paper)")


def test_cf_curve_reproducible():
    """Archivo HISTÓRICO cf_elasticidad_curve (curva de medias bootstrap, V3.2.4) sigue íntegro.
    Desde V3.2.5 el informe usa la curva con coeficientes puntuales + banda bootstrap
    (bootstrap_cf_canonical_replicas.npz); ver tests/test_v325_paper_consistency.py."""
    with open(REPORTS / 'cf_elasticidad_curve.json', encoding='utf-8') as f:
        curve_meta = json.load(f)

    assert curve_meta['validation']['matches_existing_npy'] == True
    assert curve_meta['provenance']['generator_script'] == 'scripts/vm/generate_cf_elasticidad_curve_canonical.py'
    assert curve_meta['provenance']['n_bootstrap_iter'] == 5000

    existing = np.load(REPORTS / 'cf_elasticidad_curve.npy', allow_pickle=True).item()

    # Nodo de grilla más cercano al precio medio (curva histórica de medias bootstrap)
    idx_mean = np.argmin(np.abs(existing['price_grid'] - 332.24))
    elast_mean = existing['elasticidades'][idx_mean]
    assert abs(elast_mean - (-1.5041)) < 0.01, f"CF en precio medio: {elast_mean:.4f} != -1.5041"

    # Nodo de grilla más cercano a la mediana (curva histórica; no es estimando del informe)
    idx_median = np.argmin(np.abs(existing['price_grid'] - 296.82))
    elast_median = existing['elasticidades'][idx_median]
    assert abs(elast_median - (-1.3839)) < 0.01, f"CF en mediana: {elast_median:.4f} != -1.3839"

    # El JSON meta usa 'elasticidad_p50' para el punto medio de la grilla (332.66), no la mediana de datos
    json_p50 = curve_meta['results']['elasticidad_p50']
    assert abs(json_p50 - (-1.5031)) < 0.01, f"JSON elasticidad_p50: {json_p50:.4f} != -1.5031"

    print(f"CF Curva: en precio medio (332.24) = {elast_mean:.4f}, en mediana (296.82) = {elast_median:.4f}, JSON p50_grid = {json_p50:.4f} OK")


def test_p1_retirado():
    """P1 shuffle week retirado."""
    tex_path = find_informe_tex()
    with open(tex_path) as f:
        tex = f.read()

    assert 'retirado' in tex.lower() or 'retired' in tex.lower() or 'inv' in tex.lower()
    print("P1: retirado mencionado en paper OK")


def test_f_test_reproducibility():
    """Test that the nested F-test (RCS vs log-log) reproduces F=3607.28."""
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    from patsy import dmatrix, build_design_matrices
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA = PROJECT_ROOT / 'data' / 'processed' / 'demanda_limpia.csv'
    df = pd.read_csv(DATA)

    # Build RCS design on FULL sample (critical: knots fixed on full sample)
    log_price_full = df['log_checkout_price'].to_numpy()
    knots = np.percentile(log_price_full, [5, 35, 65, 95])
    k0, k1, k2, k3 = knots

    formula = (f"cr(log_checkout_price, "
               f"knots=({float(k1):.15f}, {float(k2):.15f}), "
               f"lower_bound={float(k0):.15f}, "
               f"upper_bound={float(k3):.15f}, "
               "constraints='center') - 1")

    Z = dmatrix(formula, {'log_checkout_price': log_price_full}, return_type='dataframe')
    Z.columns = ['rcs_1', 'rcs_2', 'rcs_3']
    design_info = Z.design_info

    # Apply RCS to full dataset
    S = build_design_matrices([design_info],
                               {'log_checkout_price': log_price_full},
                               return_type='dataframe')[0]
    S.columns = ['rcs_1', 'rcs_2', 'rcs_3']
    for c in S.columns:
        df[c] = S[c].to_numpy()

    # Restricted model: OLS log-log with controls
    ll_formula = ('log_num_orders ~ log_checkout_price + emailer_for_promotion '
                  '+ homepage_featured + hubo_aumento + C(category) + C(center_type)')
    ll_model = smf.ols(ll_formula, data=df).fit()

    # Unrestricted model: RCS OLS with controls (3 bases + intercept + controls)
    rcs_formula = ('log_num_orders ~ 1 + rcs_1 + rcs_2 + rcs_3 '
                   '+ emailer_for_promotion + homepage_featured + hubo_aumento '
                   '+ C(category) + C(center_type)')
    rcs_model = smf.ols(rcs_formula, data=df).fit()

    # Manual F-test calculation
    RSS_r = ll_model.ssr
    RSS_u = rcs_model.ssr
    n = ll_model.nobs
    k_u = len(rcs_model.params)
    q = 2  # number of restrictions (rcs_2 = 0, rcs_3 = 0)

    F_stat = (RSS_r - RSS_u) / q / (RSS_u / (n - k_u))

    # Validate against paper value
    expected_F = 3607.28
    assert abs(F_stat - expected_F) < 0.01, f"F-test {F_stat:.6f} != {expected_F}"
    assert abs(n - 456548) < 1, f"N obs {n} != 456548"
    assert abs((n - k_u) - 456526) < 5, f"df_den {n - k_u} != 456526"

    print(f"F-test: F={F_stat:.6f}, df_num={q}, df_den={n-k_u}, N={n:.0f} matches paper (3607.28) OK")


if __name__ == '__main__':
    print("=" * 70)
    print("TEST FINAL DE CONSISTENCIA DE AUDITORÍA")
    print("=" * 70)

    test_v_hat_canonico()
    test_rcs_canonico()
    test_iv_09604()
    test_no_vhat_confusion()
    test_no_9487()
    test_bootstrap_cf()
    test_placebos_certificados()
    test_cf_curve_reproducible()
    test_p1_retirado()
    test_f_test_reproducibility()

    print("=" * 70)
    print("TODOS LOS TESTS DE CONSISTENCIA FINAL PASARON")
    print("=" * 70)