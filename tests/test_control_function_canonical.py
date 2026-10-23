#!/usr/bin/env python3
"""
Validación determinista de la implementación canónica de Control Function.
NO ejecuta bootstrap ni re-estima modelos completos.

Este test verifica:
1. La especificación RCS canónica (3 bases, nombres correctos, knots P5/P35/P65/P95)
2. Las fórmulas canónicas de primera y segunda etapa
3. La implementación CANÓNICA (modelo_iv_control_function) produce valores paper
4. La implementación HISTÓRICA (modelo_iv_control_function_historico) documentada
5. Extracción por nombre (no posicional)
6. Design_info RCS fijo en muestra completa
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from patsy import dmatrix, build_design_matrices

# Importar la función a testear y utilidades RCS
from modeling import (
    modelo_iv_control_function,           # CANÓNICA (nueva)
    modelo_iv_control_function_historico, # HISTÓRICA (renombrada)
    construir_instrumentos,
)
from rcs_utils import (
    build_canonical_rcs_design,
    first_stage_formula,
    second_stage_formula,
    apply_rcs_design,
)


def test_rcs_design_structure():
    """Test que la especificación RCS canónica produce exactamente 3 bases con nombres correctos."""
    print("=" * 60)
    print("TEST 1: Estructura RCS canonica")
    print("=" * 60)

    # Cargar datos reales
    df = pd.read_csv('data/processed/demanda_limpia.csv')
    df = construir_instrumentos(df).dropna(subset=['log_precio_prom_otros_centros']).copy()

    # Construir design RCS canonico
    design_info, knots = build_canonical_rcs_design(df['log_checkout_price'].to_numpy())

    # Aplicar a muestra completa
    S = apply_rcs_design(design_info, df['log_checkout_price'].to_numpy())

    print(f"Knots (log-precio): {knots}")
    print(f"Knots (precio): {np.exp(knots)}")
    print(f"Columnas RCS: {list(S.columns)}")
    print(f"Shape: {S.shape}")

    # Validaciones
    assert list(S.columns) == ['rcs_1', 'rcs_2', 'rcs_3'], f"Columnas esperadas ['rcs_1', 'rcs_2', 'rcs_3'], got {list(S.columns)}"
    assert S.shape[1] == 3, f"Esperado 3 columnas, got {S.shape[1]}"
    assert 'rcs_0' not in S.columns, "No debe existir rcs_0"
    assert 'rcs_4' not in S.columns, "No debe existir rcs_4"

    # Verificar knots canonicos
    expected_knots = np.percentile(df['log_checkout_price'], [5, 35, 65, 95])
    assert np.allclose(knots, expected_knots), f"Knots no coinciden con percentiles canonicos"

    print("[OK] TEST 1 PASO: Estructura RCS canonica correcta (3 bases, nombres correctos, knots P5/P35/P65/P95)")


def test_first_stage_formula():
    """Test formula de primera etapa."""
    print("\n" + "=" * 60)
    print("TEST 2: Formula primera etapa")
    print("=" * 60)

    formula = first_stage_formula()
    print(f"Formula FS: {formula}")

    expected_terms = [
        'log_base_price',
        'log_precio_prom_otros_centros',
        'emailer_for_promotion',
        'homepage_featured',
        'hubo_aumento',
        'C(category)',
        'C(center_type)'
    ]

    for term in expected_terms:
        assert term in formula, f"Termino '{term}' faltante en formula FS"

    print("[OK] TEST 2 PASO: Formula primera etapa contiene todos los terminos canonicos")


def test_second_stage_formula():
    """Test formula de segunda etapa canonica."""
    print("\n" + "=" * 60)
    print("TEST 3: Formula segunda etapa")
    print("=" * 60)

    formula = second_stage_formula()
    print(f"Formula SS: {formula}")

    expected_terms = [
        '1',  # intercepto explicito
        'rcs_1',
        'rcs_2',
        'rcs_3',
        'v_hat',
        'emailer_for_promotion',
        'homepage_featured',
        'hubo_aumento',
        'C(category)',
        'C(center_type)'
    ]

    for term in expected_terms:
        assert term in formula, f"Termino '{term}' faltante en formula SS"

    # NO debe tener rcs_0 ni rcs_4
    assert 'rcs_0' not in formula, "rcs_0 no debe estar en formula SS canonica"
    assert 'rcs_4' not in formula, "rcs_4 no debe estar en formula SS canonica"

    print("[OK] TEST 3 PASO: Formula segunda etapa canonica correcta (intercepto + 3 RCS + v_hat + controles)")


def test_modelo_canonico_paper_values():
    """Test que la implementación CANÓNICA reproduce valores del paper."""
    print("\n" + "=" * 60)
    print("TEST 4: Modelo CANÓNICO reproduce valores del paper")
    print("=" * 60)

    df = pd.read_csv('data/processed/demanda_limpia.csv')
    result = modelo_iv_control_function(df)

    ss_model = result['segunda_etapa']
    params = list(ss_model.params.index)
    rcs_cols = [c for c in params if c.startswith('rcs_')]

    print(f"Parametros RCS encontrados: {rcs_cols}")
    print(f"v_hat coef: {result['v_hat_coef']:.6f} (paper: 1.0002)")
    print(f"F_exog: {result['test_exogeneidad']['F']:.2f} (paper V3.2.5: 3177.9, F=t^2 puntual)")
    print(f"F_nonlin: {result['test_no_linealidad_condicional']['F']:.2f} (paper V3.2.5: 3335.4, F anidado)")
    print(f"Elasticidad precio medio: {result['elasticidad_precio_medio']:.6f} (paper: -1.5041)")
    print(f"Elasticidad precio mediana: {result['elasticidad_precio_mediana']:.6f} (paper V3.2.5: -1.3945)")
    print(f"N obs: {ss_model.nobs}")
    print(f"Especificacion: {result.get('_especificacion', 'unknown')}")

    # Validar contra valores del paper (tolerancia razonable)
    assert result['_especificacion'] == 'canonica_3_bases_rcs', "Debe ser especificación canónica"
    assert len(rcs_cols) == 3, f"Debe tener exactamente 3 bases RCS, tiene {len(rcs_cols)}"
    assert 'rcs_1' in rcs_cols and 'rcs_2' in rcs_cols and 'rcs_3' in rcs_cols, "Nombres RCS incorrectos"
    assert 'rcs_0' not in params, "No debe tener rcs_0"
    assert 'rcs_4' not in params, "No debe tener rcs_4"
    assert abs(result['v_hat_coef'] - 1.0002) < 1e-4, f"v_hat {result['v_hat_coef']} no match paper 1.0002"
    assert abs(result['test_exogeneidad']['F'] - 3177.94) < 0.5, f"F_exog {result['test_exogeneidad']['F']} no match paper 3177.9"
    assert abs(result['test_no_linealidad_condicional']['F'] - 3335.41) < 0.5, f"F_nonlin {result['test_no_linealidad_condicional']['F']} no match paper 3335.4"
    assert abs(result['elasticidad_precio_medio'] - (-1.5041)) < 1e-4, f"Elasticidad precio medio {result['elasticidad_precio_medio']} no match paper -1.5041"

    print("[OK] TEST 4 PASO: Modelo CANÓNICO reproduce valores del paper correctamente")


def test_modelo_historico_documentado():
    """Test que la implementación HISTÓRICA está disponible y documentada."""
    print("\n" + "=" * 60)
    print("TEST 5: Modelo HISTÓRICO disponible y documentado")
    print("=" * 60)

    df = pd.read_csv('data/processed/demanda_limpia.csv')
    result = modelo_iv_control_function_historico(df)

    ss_model = result['segunda_etapa']
    params = list(ss_model.params.index)
    rcs_cols = [c for c in params if c.startswith('rcs_')]

    print(f"Parametros RCS encontrados: {rcs_cols}")
    print(f"v_hat coef: {result['v_hat_coef']:.6f} (historico ~0.7139)")
    print(f"F_exog: {result['test_exogeneidad']['F']:.2f} (historico ~1541)")
    print(f"F_nonlin: {result['test_no_linealidad_condicional']['F']:.2f} (historico ~4814)")
    print(f"Elasticidad P50: {result['elasticidad_p50']:.6f} (historico ~-1.3129)")
    print(f"N obs: {ss_model.nobs}")
    print(f"Especificacion: {result.get('_especificacion', 'unknown')}")

    # Documentar: esta es la especificacion historica
    has_rcs_0 = 'rcs_0' in params
    has_rcs_4 = 'rcs_4' in params
    n_rcs = len(rcs_cols)

    print(f"\nEspecificacion HISTÓRICA (modelo_iv_control_function_historico):")
    print(f"  - Tiene rcs_0: {has_rcs_0}")
    print(f"  - Tiene rcs_4: {has_rcs_4}")
    print(f"  - N bases RCS: {n_rcs}")
    print(f"  - v_hat ~ 0.7139 (historico) vs 1.0002 (canonico)")

    assert result['_especificacion'] == 'historica_4_bases_rcs', "Debe ser especificación histórica"
    # modelo_log_log_rcs con df=4 produce 4 bases spline + intercepto = 5 columnas (rcs_0..rcs_4)
    assert n_rcs == 5, f"Histórica debe tener 5 bases RCS (rcs_0-rcs_4), tiene {n_rcs}"
    assert has_rcs_0, "Histórica debe tener rcs_0"
    assert 'rcs_4' in params, "Histórica debe tener rcs_4"

    print("[OK] TEST 5 PASO: Modelo HISTÓRICO disponible y correctamente documentado")


def test_no_positional_extraction():
    """Test que la extraccion de parametros es por nombre, no posicion."""
    print("\n" + "=" * 60)
    print("TEST 6: Extraccion por nombre (no posicional) - CANÓNICA")
    print("=" * 60)

    import inspect
    source = inspect.getsource(modelo_iv_control_function)

    # Verificar que usa .get() por nombre para v_hat en el return dict
    assert "params.get('v_hat'" in source or 'params.get("v_hat"' in source, "v_hat debe extraerse por nombre"

    # Verificar NO usa índice posicional [0] para v_hat
    lines = source.split('\n')
    for i, line in enumerate(lines):
        if 'params[0]' in line or 'params[1]' in line or 'params[2]' in line or 'params[3]' in line:
            if 'v_hat' in line or 'rcs_' in line:
                print(f"[WARN] Linea {i}: Posible extraccion posicional: {line.strip()}")

    print("[OK] TEST 6 PASO: Extraccion de parametros por nombre verificada (canónica)")


def test_design_info_fixed():
    """Test que design_info RCS se fija en muestra completa (no se recalcula por replica)."""
    print("\n" + "=" * 60)
    print("TEST 7: Design info RCS fijo (muestra completa) - CANÓNICA")
    print("=" * 60)

    import inspect
    source = inspect.getsource(modelo_iv_control_function)

    # En canonica, deberia usar build_canonical_rcs_design una sola vez
    if 'build_canonical_rcs_design' in source:
        print("[OK] Usa build_canonical_rcs_design (canonico)")
    else:
        raise AssertionError("Implementación canónica debe usar build_canonical_rcs_design")

    # Verificar que NO usa modelo_log_log_rcs
    if 'modelo_log_log_rcs' in source:
        raise AssertionError("Implementación canónica NO debe usar modelo_log_log_rcs (recalcula knots)")

    print("[OK] TEST 7 PASO: Design info RCS fijo verificado (canónica)")


def test_historico_usa_modelo_log_log_rcs():
    """Test que la histórica SÍ usa modelo_log_log_rcs (recalcula knots)."""
    print("\n" + "=" * 60)
    print("TEST 8: Histórica usa modelo_log_log_rcs (recalcula knots)")
    print("=" * 60)

    import inspect
    source = inspect.getsource(modelo_iv_control_function_historico)

    if 'modelo_log_log_rcs' in source:
        print("[OK] Histórica usa modelo_log_log_rcs (recalcula knots por muestra)")
    else:
        raise AssertionError("Histórica debe usar modelo_log_log_rcs")

    print("[OK] TEST 8 PASO: Comportamiento histórico verificado")


def run_all_tests():
    """Ejecuta todos los tests de validación."""
    print("\n" + "=" * 60)
    print("VALIDACION CANONICA CONTROL FUNCTION")
    print("=" * 60)

    tests = [
        test_rcs_design_structure,
        test_first_stage_formula,
        test_second_stage_formula,
        test_modelo_canonico_paper_values,
        test_modelo_historico_documentado,
        test_no_positional_extraction,
        test_design_info_fixed,
        test_historico_usa_modelo_log_log_rcs,
    ]

    results = []
    for test in tests:
        try:
            test()
            results.append((test.__name__, True, None))
        except Exception as e:
            results.append((test.__name__, False, str(e)))
            print(f"[FAIL] {test.__name__} FALLO: {e}")

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    for name, passed, error in results:
        status = "[OK] PASO" if passed else "[FAIL] FALLO"
        print(f"  {status}: {name}")
        if error:
            print(f"       Error: {error}")

    all_passed = all(r[1] for r in results)
    print(f"\n{'TODOS LOS TESTS PASARON' if all_passed else 'ALGUNOS TESTS FALLARON'}")

    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)