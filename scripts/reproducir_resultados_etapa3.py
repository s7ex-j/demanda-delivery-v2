"""
ETAPA 3 — Reproducibilidad econométrica completa (versión ligera para gate).
Ejecuta todos los modelos RÁPIDOS y deja los PESADOS como TODOs para VM.

Uso:
    python scripts/reproducir_resultados_etapa3.py

Resultados:
    - Imprime tabla de concordancia en consola
    - Guarda CSV: reports/notebook_outputs/etapa3_concordancia.csv
    - Guarda JSON detallado: reports/notebook_outputs/etapa3_resultados_completos.json

PESADOS (ejecutar aparte en VM, ver scripts/vm_computaciones_pesadas.py):
    1. bootstrap_quantile_regression (tau=0.1,0.25,0.5,0.75,0.9, n_iter=1000) ~ 13.5h
    2. bootstrap_elasticidad_clusterizado (panel FE, n_iter=200) ~ 2-3h
    3. Wild cluster bootstrap (n_iter=500) ya incluido en comparar_errores_estandar (rápido, 1-2 min)

Los resultados pesados se cargan desde .npy generados por vm_computaciones_pesadas.py:
    - reports/notebook_outputs/bootstrap_quantile_results.npy
    - reports/notebook_outputs/bootstrap_panel_fe_results.npy
"""
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Rutas ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_prep import pipeline_completo
import modeling as md

DATA_CSV = PROJECT_ROOT / "data" / "processed" / "demanda_limpia_v2.csv"
OUT_DIR = PROJECT_ROOT / "reports" / "notebook_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONCORDANCIA_CSV = OUT_DIR / "etapa3_concordancia.csv"
RESULTADOS_JSON = OUT_DIR / "etapa3_resultados_completos.json"

# Archivos de resultados VM (se generan con scripts/vm_computaciones_pesadas.py)
QUANTILE_NPY = OUT_DIR / "bootstrap_quantile_results.npy"
PANEL_FE_NPY = OUT_DIR / "bootstrap_panel_fe_results.npy"


def cargar_datos():
    print("[1/1] Cargando demanda_limpia_v2.csv ...")
    df = pd.read_csv(DATA_CSV)
    print(f"     N = {len(df):,}, columnas = {len(df.columns)}")
    return df


def redondear4(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), 4)


def main():
    print("=" * 70)
    print("ETAPA 3 — REPRODUCIBILIDAD ECONOMÉTRICA (versión gate)")
    print("=" * 70)

    df = cargar_datos()

    resultados = {}
    concordancia = []

    # ============================================================
    # 1. OLS log-log base (sin controles) - RÁPIDO
    # ============================================================
    print("\n[2/12] OLS log-log base...")
    m_base = md.modelo_log_log_base(df)
    coef_base = redondear4(m_base.params.get('log_checkout_price'))
    ee_base = redondear4(m_base.bse.get('log_checkout_price'))
    ic_base = (redondear4(m_base.conf_int().loc['log_checkout_price', 0]),
               redondear4(m_base.conf_int().loc['log_checkout_price', 1]))
    resultados['ols_base'] = {'coef': coef_base, 'ee': ee_base, 'ic95': ic_base}
    concordancia.append({
        'modelo': 'OLS log-log base',
        'resultado_pdf': -0.7513,  # referencia: modelo con controles
        'resultado_reproducido': coef_base,
        'ee_pdf': None,
        'ee_reproducido': ee_base,
        'nota': 'sin controles'
    })

    # ============================================================
    # 2. OLS log-log CON controles (modelo principal OLS) - RÁPIDO
    # ============================================================
    print("[3/12] OLS log-log con controles...")
    m_ols = md.modelo_log_log_controles(df)
    coef_ols = redondear4(m_ols.params.get('log_checkout_price'))
    ee_ols = redondear4(m_ols.bse.get('log_checkout_price'))
    ic_ols = (redondear4(m_ols.conf_int().loc['log_checkout_price', 0]),
              redondear4(m_ols.conf_int().loc['log_checkout_price', 1]))
    resultados['ols_controles'] = {'coef': coef_ols, 'ee': ee_ols, 'ic95': ic_ols}
    concordancia.append({
        'modelo': 'OLS log-log (controles)',
        'resultado_pdf': -0.7513,
        'resultado_reproducido': coef_ols,
        'ee_pdf': 0.0421,  # cluster center reportado en PDF
        'ee_reproducido': ee_ols,
        'nota': 'comparar con EE cluster center'
    })

    # ============================================================
    # 3. Semi-log - RÁPIDO
    # ============================================================
    print("[4/12] Semi-log...")
    m_semi = md.modelo_semilog(df)
    coef_semi = redondear4(m_semi.params.get('checkout_price'))
    ee_semi = redondear4(m_semi.bse.get('checkout_price'))
    resultados['semilog'] = {'coef': coef_semi, 'ee': ee_semi}
    concordancia.append({
        'modelo': 'Semi-log',
        'resultado_pdf': None,
        'resultado_reproducido': coef_semi,
        'ee_pdf': None,
        'ee_reproducido': ee_semi,
        'nota': 'coef es dlogQ/dP, no elasticidad directa'
    })

    # ============================================================
    # 4. Panel FE two-way (entidad + tiempo) - RÁPIDO (~30-60s)
    # ============================================================
    print("[5/12] Panel FE two-way (entidad + tiempo)...")
    try:
        panel_res = md.modelo_panel_efectos_fijos(df)
        coef_fe = redondear4(panel_res.params.get('log_checkout_price'))
        ee_fe = redondear4(panel_res.std_errors.get('log_checkout_price'))

        # Intentar cargar bootstrap clusterizado desde VM
        if PANEL_FE_NPY.exists():
            print(f"     Cargando bootstrap panel FE VM desde {PANEL_FE_NPY}...")
            panel_fe_vm = np.load(PANEL_FE_NPY, allow_pickle=True).item()
            ic95_vm = (redondear4(panel_fe_vm.get('ic_95_inferior')),
                       redondear4(panel_fe_vm.get('ic_95_superior')))
            ee_fe_display = f"EE={ee_fe}, IC95(VM)=[{ic95_vm[0]}, {ic95_vm[1]}] (VM bootstrap 200 iter)"
        else:
            ee_fe_display = ee_fe

        resultados['panel_fe_twoway'] = {'coef': coef_fe, 'ee': ee_fe}
        concordancia.append({
            'modelo': 'Panel FE two-way',
            'resultado_pdf': -1.5755,
            'resultado_reproducido': coef_fe,
            'ee_pdf': 0.0236,  # cluster entidad
            'ee_reproducido': ee_fe_display,
            'nota': 'EE clusterizado por entidad; bootstrap clusterizado en VM (scripts/vm_computaciones_pesadas.py)'
        })
    except Exception as e:
        print(f"     ERROR Panel FE: {e}")
        resultados['panel_fe_twoway'] = {'error': str(e)}
        concordancia.append({'modelo': 'Panel FE two-way', 'error': str(e)})

    # ============================================================
    # 5. IV 2SLS (2 instrumentos) - RÁPIDO
    # ============================================================
    print("[6/12] IV 2SLS (Hausman-Nevo, 2 instrumentos)...")
    try:
        df_iv = md.construir_instrumentos(df)
        iv_res, df_iv_clean = md.modelo_iv_2sls(df_iv)
        coef_iv = redondear4(iv_res.params.get('log_checkout_price'))
        ee_iv = redondear4(iv_res.std_errors.get('log_checkout_price'))
        # Hansen J
        hansen = md.test_sobreidentificacion_hansen_j(df_iv_clean)
        hansen_stat = redondear4(hansen.stat)
        hansen_p = redondear4(hansen.pval)
        hansen_df = redondear4(hansen.df)
        resultados['iv_2sls_2inst'] = {
            'coef': coef_iv, 'ee': ee_iv,
            'hansen_j': {'stat': hansen_stat, 'p': hansen_p, 'df': hansen_df}
        }
        concordancia.append({
            'modelo': 'IV 2SLS (2 inst)',
            'resultado_pdf': -0.9487,
            'resultado_reproducido': coef_iv,
            'ee_pdf': 0.0032,  # robusto
            'ee_reproducido': ee_iv,
            'nota': f'Hansen J: stat={hansen_stat}, p={hansen_p}, df={hansen_df}'
        })
    except Exception as e:
        print(f"     ERROR IV 2SLS: {e}")
        resultados['iv_2sls_2inst'] = {'error': str(e)}
        concordancia.append({'modelo': 'IV 2SLS (2 inst)', 'error': str(e)})

    # ============================================================
    # 6. IV just-identified (1 instrumento: log_base_price) - RÁPIDO
    # ============================================================
    print("[7/12] IV just-identified (log_base_price)...")
    try:
        df_iv = md.construir_instrumentos(df)
        iv_ji = md.modelo_iv_justamente_identificado(df_iv, instrumento='log_base_price')
        coef_ji = redondear4(iv_ji.params.get('log_checkout_price'))
        ee_ji = redondear4(iv_ji.std_errors.get('log_checkout_price'))
        resultados['iv_just_identified'] = {'coef': coef_ji, 'ee': ee_ji}
        concordancia.append({
            'modelo': 'IV just-identified (base_price)',
            'resultado_pdf': -0.9627,
            'resultado_reproducido': coef_ji,
            'ee_pdf': None,
            'ee_reproducido': ee_ji,
            'nota': 'sin Hansen J (just-identified)'
        })
    except Exception as e:
        print(f"     ERROR IV just-identified: {e}")
        resultados['iv_just_identified'] = {'error': str(e)}
        concordancia.append({'modelo': 'IV just-identified', 'error': str(e)})

    # ============================================================
    # 7. Control Function (Wooldridge 2015) con RCS + v_hat - RÁPIDO
    # ============================================================
    print("[8/12] Control Function (RCS + v_hat)...")
    try:
        df_iv = md.construir_instrumentos(df)
        cf_res = md.modelo_iv_control_function(df_iv)
        coef_cf = redondear4(cf_res['segunda_etapa'].params.get('log_checkout_price'))
        ee_cf = redondear4(cf_res['segunda_etapa'].bse.get('log_checkout_price'))
        test_exog = cf_res['test_exogeneidad']
        test_nl = cf_res['test_no_linealidad_condicional']
        resultados['control_function'] = {
            'coef': coef_cf, 'ee': ee_cf,
            'test_exogeneidad': {'F': redondear4(test_exog['F']), 'p': redondear4(test_exog['p']), 'rechaza': test_exog['rechaza_H0_05']},
            'test_no_linealidad': {'F': redondear4(test_nl['F']), 'p': redondear4(test_nl['p']), 'rechaza': test_nl['rechaza_H0_05']},
        }
        concordancia.append({
            'modelo': 'Control Function (RCS + v_hat)',
            'resultado_pdf': None,
            'resultado_reproducido': coef_cf,
            'ee_pdf': None,
            'ee_reproducido': ee_cf,
            'nota': f"exog p={test_exog['p']}, NL p={test_nl['p']}"
        })
    except Exception as e:
        print(f"     ERROR Control Function: {e}")
        resultados['control_function'] = {'error': str(e)}
        concordancia.append({'modelo': 'Control Function', 'error': str(e)})

    # ============================================================
    # 8. RCS (4 nudos P5,35,65,95) - RÁPIDO
    # ============================================================
    print("[9/12] RCS (df=4, nudos P5,35,65,95)...")
    try:
        m_rcs, spline_design, nudos = md.modelo_log_log_rcs(df)
        # Test Wald no-linealidad (3 gl)
        spline_cols = list(spline_design.columns)
        wald_nl = md.test_wald_no_linealidad_rcs(m_rcs, spline_cols)
        wald_f = redondear4(wald_nl['estadistico_F'])
        wald_p = redondear4(wald_nl['p_valor'])
        wald_df = wald_nl['grados_libertad']
        # Elasticidades puntuales en grid de precios
        precios_grid = np.array([50, 100, 150, 200, 250, 300, 350])
        elast_puntual = md.elasticidad_puntual_rcs(m_rcs, spline_design, nudos, precios_grid)
        elast_puntual = [redondear4(e) for e in elast_puntual]
        resultados['rcs'] = {
            'nudos': [redondear4(n) for n in nudos],
            'test_wald_nl': {'F': wald_f, 'p': wald_p, 'gl': wald_df, 'rechaza': wald_nl['rechaza_H0_05']},
            'elasticidades_puntuales': dict(zip(precios_grid, elast_puntual)),
        }
        concordancia.append({
            'modelo': 'RCS (variable)',
            'resultado_pdf': 'variable',
            'resultado_reproducido': f"elast_puntual: {elast_puntual}",
            'ee_pdf': None,
            'ee_reproducido': None,
            'nota': f"Wald NL: F={wald_f}, p={wald_p}, gl={wald_df}"
        })
    except Exception as e:
        print(f"     ERROR RCS: {e}")
        resultados['rcs'] = {'error': str(e)}
        concordancia.append({'modelo': 'RCS', 'error': str(e)})

    # ============================================================
    # 9. Quantile Regression (PUNTUALES + cargar bootstrap de VM) - RÁPIDO
    # ============================================================
    print("[10/12] Quantile Regression (tau=0.1,0.25,0.5,0.75,0.9) — PUNTUALES + bootstrap VM...")
    taus = [0.1, 0.25, 0.5, 0.75, 0.9]
    qr_resultados = {}

    # Intentar cargar resultados de bootstrap desde VM
    bootstrap_vm_results = None
    if QUANTILE_NPY.exists():
        print(f"     Cargando bootstrap VM desde {QUANTILE_NPY}...")
        bootstrap_vm_results = np.load(QUANTILE_NPY, allow_pickle=True).item()

    for tau in taus:
        try:
            qr = md.modelo_quantile_regression(df, tau=tau)
            coef = redondear4(qr.params.get('log_checkout_price'))
            qr_resultados[f'tau_{tau}'] = coef

            # Si hay resultados de VM, usar sus IC
            if bootstrap_vm_results and f'tau_{tau}' in bootstrap_vm_results:
                vm = bootstrap_vm_results[f'tau_{tau}']
                ee_vm = redondear4(vm.get('ee'))
                ic95_vm = (redondear4(vm.get('ic95', [None, None])[0]),
                           redondear4(vm.get('ic95', [None, None])[1]))
                ee_reproducido = f"EE={ee_vm}, IC95=[{ic95_vm[0]}, {ic95_vm[1]}] (VM bootstrap 1000 iter)"
            else:
                ee_reproducido = 'PENDIENTE_VM_BOOTSTRAP_1000'

            concordancia.append({
                'modelo': f'Quantile tau={tau}',
                'resultado_pdf': {-0.1: -0.66, 0.25: -0.75, 0.5: -0.81, 0.75: -0.92, 0.9: -1.00}.get(tau),
                'resultado_reproducido': coef,
                'ee_pdf': None,
                'ee_reproducido': ee_reproducido,
                'nota': 'bootstrap 1000 iter en VM (scripts/vm_computaciones_pesadas.py)'
            })
        except Exception as e:
            print(f"     ERROR QR tau={tau}: {e}")
            qr_resultados[f'tau_{tau}'] = {'error': str(e)}
    resultados['quantile_regression'] = qr_resultados

    # ============================================================
    # 10. RESET test - RÁPIDO
    # ============================================================
    print("[11/12] RESET test...")
    try:
        from statsmodels.stats.diagnostic import linear_reset
        m_ols = md.modelo_log_log_controles(df)
        reset_res = linear_reset(m_ols, power=3, use_f=True)
        reset_f = redondear4(reset_res.fvalue)
        reset_p = redondear4(reset_res.pvalue)
        resultados['reset_test'] = {'F': reset_f, 'p': reset_p}
        concordancia.append({
            'modelo': 'RESET test',
            'resultado_pdf': 'F=119.57, p≈0',
            'resultado_reproducido': f'F={reset_f}, p={reset_p}',
            'ee_pdf': None,
            'ee_reproducido': None,
            'nota': 'rechaza linealidad log-log'
        })
    except Exception as e:
        print(f"     ERROR RESET: {e}")
        resultados['reset_test'] = {'error': str(e)}
        concordancia.append({'modelo': 'RESET test', 'error': str(e)})

    # ============================================================
    # 11. Comparar errores estándar (HC3, cluster, multiway, panel, bootstrap, wild) - RÁPIDO
    # ============================================================
    print("[12/12] Comparar errores estándar (incl. wild bootstrap)...")
    try:
        comp = md.comparar_errores_estandar(df)
        resultados['comparacion_ee'] = {k: {kk: redondear4(vv) if isinstance(vv, (int, float)) else vv
                                             for kk, vv in v.items()} if isinstance(v, dict) else redondear4(v)
                                        for k, v in comp.items()}
        # Extraer ratios clave
        ratio_ch = resultados['comparacion_ee'].get('ratio_cluster_hc3')
        ratio_mw = resultados['comparacion_ee'].get('ratio_multiway_hc3')
        concordancia.append({
            'modelo': 'Comparación EE (ratios vs HC3)',
            'resultado_pdf': 'cluster/HC3 ≈ 7.92x',
            'resultado_reproducido': f'cluster/HC3={ratio_ch}, multiway/HC3={ratio_mw}',
            'ee_pdf': None,
            'ee_reproducido': None,
            'nota': 'ver dict completo en JSON'
        })
    except Exception as e:
        print(f"     ERROR comparar EE: {e}")
        resultados['comparacion_ee'] = {'error': str(e)}
        concordancia.append({'modelo': 'Comparación EE', 'error': str(e)})

    # ============================================================
    # 12. Placebos válidos - RÁPIDO
    # ============================================================
    print("\n[+] Placebos válidos...")
    # 12a. Permutación precio dentro de entidad
    print("     Placebo: permutación precio dentro de entidad...")
    try:
        p1 = md.placebo_temporal_valido(df, n_iteraciones=200)
        resultados['placebo_perm_entidad'] = {k: redondear4(v) if isinstance(v, (int, float)) else v
                                               for k, v in p1.items() if k != 'elasticidades_placebo'}
        concordancia.append({
            'modelo': 'Placebo permutación entidad',
            'resultado_pdf': None,
            'resultado_reproducido': f"p={p1['p_valor']}",
            'nota': 'debe NO rechazar (p > 0.05) si estructura temporal no importa'
        })
    except Exception as e:
        print(f"     ERROR placebo entidad: {e}")
        resultados['placebo_perm_entidad'] = {'error': str(e)}

    # 12b. Precio futuro (lead)
    print("     Placebo: precio futuro (lead)...")
    try:
        p2 = md.placebo_precio_futuro(df)
        resultados['placebo_precio_futuro'] = {k: redondear4(v) if isinstance(v, (int, float)) else v
                                                for k, v in p2.items()}
        concordancia.append({
            'modelo': 'Placebo precio futuro',
            'resultado_pdf': None,
            'resultado_reproducido': f"lead_coef={p2['elasticidad_precio_futuro_lead']}",
            'nota': 'lead ≈ 0 esperado; si lead ≈ contemp → confusión temporal'
        })
    except Exception as e:
        print(f"     ERROR placebo lead: {e}")
        resultados['placebo_precio_futuro'] = {'error': str(e)}

    # 12c. Block permutation
    print("     Placebo: block permutation...")
    try:
        p3 = md.placebo_block_permutation(df, block_size=4, n_iteraciones=200)
        resultados['placebo_block'] = {k: redondear4(v) if isinstance(v, (int, float)) else v
                                        for k, v in p3.items() if k != 'elasticidades_placebo'}
        concordancia.append({
            'modelo': 'Placebo block permutation',
            'resultado_pdf': None,
            'resultado_reproducido': f"p={p3.get('p_valor')}",
            'nota': 'preserva dependencia local'
        })
    except Exception as e:
        print(f"     ERROR placebo block: {e}")
        resultados['placebo_block'] = {'error': str(e)}

    # ============================================================
    # GUARDAR RESULTADOS
    # ============================================================
    print("\n" + "=" * 70)
    print("GUARDANDO RESULTADOS...")
    print("=" * 70)

    # JSON completo
    with open(RESULTADOS_JSON, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=str)
    print(f"JSON: {RESULTADOS_JSON}")

    # CSV concordancia
    df_conc = pd.DataFrame(concordancia)
    df_conc.to_csv(CONCORDANCIA_CSV, index=False)
    print(f"CSV concordancia: {CONCORDANCIA_CSV}")

    # ============================================================
    # IMPRIMIR TABLA DE CONCORDANCIA (gate)
    # ============================================================
    print("\n" + "=" * 70)
    print("TABLA DE CONCORDANCIA — GATE ETAPA 3")
    print("=" * 70)
    for row in concordancia:
        if 'error' in row:
            print(f"  ❌ {row['modelo']}: ERROR - {row['error']}")
        else:
            pdf = row.get('resultado_pdf')
            rep = row.get('resultado_reproducido')
            ee_pdf = row.get('ee_pdf')
            ee_rep = row.get('ee_reproducido')
            print(f"  {row['modelo']:35s} | PDF: {str(pdf):>12s} | REP: {str(rep):>12s} | EE_pdf: {str(ee_pdf):>8s} | EE_rep: {str(ee_rep):>8s}")

    print("\n" + "=" * 70)
    print("✅ ETAPA 3 COMPLETADA (modelos rápidos)")
    print("=" * 70)

    # ============================================================
    # TODOs para VM (pesados) - EJECUTAR scripts/vm_computaciones_pesadas.py
    # ============================================================
    print("\n" + "=" * 70)
    print("📋 TODOs PARA VM (ejecutar aparte: python scripts/vm_computaciones_pesadas.py)")
    print("=" * 70)
    print(f"""
1. BOOTSTRAP QUANTILE REGRESSION (n_iter=1000, tau=0.1,0.25,0.5,0.75,0.9)
   Script: scripts/vm_computaciones_pesadas.py
   Tiempo estimado: ~13.5h (GPU + 32GB RAM)
   Output: {QUANTILE_NPY}
   → Se cargará automáticamente en próxima ejecución de este script

2. BOOTSTRAP CLUSTERIZADO PANEL FE (n_iter=200, cluster center_id)
   Script: scripts/vm_computaciones_pesadas.py
   Tiempo estimado: ~2-3h
   Output: {PANEL_FE_NPY}
   → Se cargará automáticamente en próxima ejecución de este script

3. WILD CLUSTER BOOTSTRAP (n_iter=500, center_id) — YA INCLUIDO en comparar_errores_estandar()
   (tarda ~1-2 min, no necesita VM separada)

⚠️  IMPORTANTE: Usar demanda_limpia_v2.csv (hash: 90d41670226e961b3eb3ceb6b5b312575082280198a6898b3b1f1bccf5a4707e)

ARCHIVOS VM ESPERADOS:
  - {QUANTILE_NPY} ({'EXISTE' if QUANTILE_NPY.exists() else 'PENDIENTE'})
  - {PANEL_FE_NPY} ({'EXISTE' if PANEL_FE_NPY.exists() else 'PENDIENTE'})
""")

    return concordancia, resultados


if __name__ == '__main__':
    main()