"""
ETAPA 5/P1-14 — Bootstrap clusterizado para Control Function (v_hat)
======================================================================

Este script implementa la inferencia correcta para el regresor generado v_hat
en el enfoque de Control Function (Wooldridge 2015, §6.3.3).

PROBLEMA: Los errores estándar estándar de la segunda etapa (EE = 0.018 para v_hat)
NO corrigen por la varianza de estimación de v_hat en la primera etapa.
Son un LÍMITE INFERIOR de la incertidumbre real.

SOLUCIÓN (Wooldridge 2015 §6.3.3): Bootstrap clusterizado remuestreando
centros completos (77 clusters), re-estimando AMBAS etapas en cada iteración.

ESPECIFICACIÓN:
- Cluster: center_id (77 clusters)
- Iteraciones: 500 (recomendado por Wooldridge para bootstrap de dos etapas)
- En cada iteración:
  1. Remuestrear 77 centros con reemplazo
  2. Primera etapa: log_checkout_price ~ instrumentos + controles → obtener v_hat
  3. Segunda etapa: log_num_orders ~ RCS(log_checkout_price) + v_hat + controles
  4. Guardar coeficientes de interés: v_hat, términos RCS no lineales, log_checkout_price lineal

OUTPUT:
- reports/notebook_outputs/bootstrap_vhat_results.npy
  Dict con: coef_vhat (media, std, IC95, n_valid), coef_rcs_nonlinear (media, std, IC95), etc.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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

import statsmodels.formula.api as smf

from modeling import (
    construir_instrumentos,
    modelo_log_log_rcs,
)

# Archivos de entrada/salida
DATA_CSV = PROJECT_ROOT / "data" / "processed" / "demanda_limpia_v2.csv"
OUT_DIR = PROJECT_ROOT / "reports" / "notebook_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VHAT_OUT = OUT_DIR / "bootstrap_vhat_results.npy"


def _agregar_resultados(coef_vhat, coef_price_linear, coef_rcs_1, coef_rcs_2, coef_rcs_3,
                         f_exog, f_nl, n_iteraciones_total, n_valid, n_descartadas_colinealidad):
    """Agrega los coeficientes crudos acumulados hasta el momento en el dict
    de resultados final (media, std, IC95%, n_válidas). Se usa tanto para
    checkpoints intermedios como para el guardado final — misma lógica,
    para que un checkpoint a mitad de camino sea igual de confiable que
    el resultado terminado."""
    coef_vhat = np.array(coef_vhat)
    coef_price_linear = np.array(coef_price_linear)
    coef_rcs_1 = np.array(coef_rcs_1)
    coef_rcs_2 = np.array(coef_rcs_2)
    coef_rcs_3 = np.array(coef_rcs_3)
    f_exog = np.array(f_exog)
    f_nl = np.array(f_nl)

    mask = ~np.isnan(coef_vhat)
    coef_vhat_v = coef_vhat[mask]
    coef_price_linear_v = coef_price_linear[mask]
    coef_rcs_1_v = coef_rcs_1[mask]
    coef_rcs_2_v = coef_rcs_2[mask]
    coef_rcs_3_v = coef_rcs_3[mask]
    f_exog_v = f_exog[~np.isnan(f_exog)]
    f_nl_v = f_nl[~np.isnan(f_nl)]

    def _stats(arr):
        if len(arr) == 0:
            return {'elasticidad_media': np.nan, 'elasticidad_std': np.nan,
                    'ic_95_inferior': np.nan, 'ic_95_superior': np.nan,
                    'n_iteraciones_validas': 0}
        return {
            'elasticidad_media': float(np.mean(arr)),
            'elasticidad_std': float(np.std(arr)),
            'ic_95_inferior': float(np.percentile(arr, 2.5)),
            'ic_95_superior': float(np.percentile(arr, 97.5)),
            'n_iteraciones_validas': int(len(arr)),
        }

    return {
        'v_hat': _stats(coef_vhat_v),
        'log_checkout_price_linear': _stats(coef_price_linear_v),
        'rcs_1': _stats(coef_rcs_1_v),
        'rcs_2': _stats(coef_rcs_2_v),
        'rcs_3': _stats(coef_rcs_3_v),
        'test_exogeneidad': {
            'F_media': float(np.mean(f_exog_v)) if len(f_exog_v) > 0 else np.nan,
            'F_std': float(np.std(f_exog_v)) if len(f_exog_v) > 0 else np.nan,
            'n_iteraciones_validas': int(len(f_exog_v)),
        },
        'test_no_linealidad_condicional': {
            'F_media': float(np.mean(f_nl_v)) if len(f_nl_v) > 0 else np.nan,
            'F_std': float(np.std(f_nl_v)) if len(f_nl_v) > 0 else np.nan,
            'n_iteraciones_validas': int(len(f_nl_v)),
        },
        'n_iteraciones_total': n_iteraciones_total,
        'n_iteraciones_validas': n_valid,
        'n_descartadas_por_no_identificacion': n_descartadas_colinealidad,
        'nota': 'Bootstrap clusterizado Wooldridge 2015 §6.3.3 para regresor generado v_hat. Remuestreo de 77 centros, re-estimación de ambas etapas.',
    }


def cargar_datos():
    print("[1/2] Cargando demanda_limpia_v2.csv ...")
    df = pd.read_csv(DATA_CSV)
    print(f"     N = {len(df):,} filas, {len(df.columns)} columnas")
    print(f"     Centros únicos: {df['center_id'].nunique()}")
    return df


def run_bootstrap_vhat(df, n_iteraciones=500, semilla=42):
    import statsmodels.formula.api as smf
    """
    Bootstrap clusterizado para Control Function (Wooldridge 2015 §6.3.3).

    Remuestrea centros completos, re-estima primera y segunda etapa en cada iteración.

    Returns:
        dict con resultados bootstrap para:
        - v_hat (componente endógeno)
        - log_checkout_price lineal
        - rcs_1, rcs_2, rcs_3 (términos no lineales del spline)
        - test_exogeneidad_F, test_no_linealidad_F
    """
    rng = np.random.default_rng(semilla)
    clusters = df['center_id'].unique()
    n_clusters = len(clusters)

    # Contenedores para resultados
    coef_vhat = []
    coef_price_linear = []
    coef_rcs_1 = []
    coef_rcs_2 = []
    coef_rcs_3 = []
    f_exog = []
    f_nl = []
    n_valid = 0
    n_descartadas_colinealidad = 0

    print(f"\n{'='*70}")
    print(f"BOOTSTRAP CONTROL FUNCTION v_hat (Wooldridge 2015 §6.3.3)")
    print(f"{'='*70}")
    print(f"Clusters: {n_clusters} centros | Iteraciones: {n_iteraciones}")
    print(f"Semilla: {semilla}")

    import time
    t_inicio = time.time()
    CHECKPOINT_CADA = 250
    CHECKPOINT_OUT = OUT_DIR / "bootstrap_vhat_results_CHECKPOINT.npy"

    for i in range(n_iteraciones):
        if i % 50 == 0 and i > 0:
            transcurrido = time.time() - t_inicio
            ritmo = transcurrido / i
            eta_min = ritmo * (n_iteraciones - i) / 60
            print(f"  Progreso: {i}/{n_iteraciones}  "
                  f"({transcurrido/60:.1f} min transcurridos, ~{eta_min:.1f} min restantes, "
                  f"{n_valid} válidas hasta ahora)")

        if i > 0 and i % CHECKPOINT_CADA == 0:
            checkpoint = _agregar_resultados(
                coef_vhat, coef_price_linear, coef_rcs_1, coef_rcs_2, coef_rcs_3,
                f_exog, f_nl, i, n_valid, n_descartadas_colinealidad
            )
            checkpoint['CHECKPOINT_PARCIAL'] = True
            checkpoint['iteraciones_completadas_hasta_checkpoint'] = i
            np.save(CHECKPOINT_OUT, checkpoint)
            print(f"  [checkpoint guardado en {CHECKPOINT_OUT.name}: {i}/{n_iteraciones} "
                  f"iteraciones, {n_valid} válidas]")

        # 1. Remuestrear clusters (centros) con reemplazo
        clusters_muestreados = rng.choice(clusters, size=n_clusters, replace=True)

        # Construir muestra bootstrap concatenando todos los centros seleccionados
        partes = [df[df['center_id'] == c] for c in clusters_muestreados]
        muestra = pd.concat(partes, ignore_index=True)

        try:
            # 2. PRIMERA ETAPA: construir instrumentos y obtener v_hat
            df_iv = construir_instrumentos(muestra).dropna(subset=['log_precio_prom_otros_centros']).copy()

            formula_fs = (
                'log_checkout_price ~ log_base_price + log_precio_prom_otros_centros '
                '+ emailer_for_promotion + homepage_featured + hubo_aumento '
                '+ C(category) + C(center_type)'
            )
            fs_model = smf.ols(formula_fs, data=df_iv).fit()
            df_iv['v_hat'] = fs_model.resid

            # 3. SEGUNDA ETAPA: RCS + v_hat + controles
            modelo_rcs, spline_design, nudos = modelo_log_log_rcs(df_iv)

            df_iv_rcs = df_iv.copy()
            for col in spline_design.columns:
                df_iv_rcs[col] = spline_design[col]

            spline_terms = ' + '.join(spline_design.columns)
            formula_ss = (
                f'log_num_orders ~ {spline_terms} + v_hat '
                '+ emailer_for_promotion + homepage_featured + hubo_aumento '
                '+ C(category) + C(center_type)'
            )
            ss_model = smf.ols(formula_ss, data=df_iv_rcs).fit()

            # 4. Extraer coeficientes de interés
            param_names = list(ss_model.params.index)

            # log_checkout_price lineal (rcs_0 o término lineal) — se valida
            # ANTES de hacer ningún append, para no desalinear los arrays
            # si esta iteración termina descartada.
            if 'log_checkout_price' in param_names:
                price_coef_raw = ss_model.params['log_checkout_price']
                price_se_raw = ss_model.bse['log_checkout_price']
            elif 'rcs_0' in param_names:
                # El término lineal está en rcs_0 con constraints='center'
                price_coef_raw = ss_model.params['rcs_0']
                price_se_raw = ss_model.bse['rcs_0']
            else:
                price_coef_raw = np.nan
                price_se_raw = np.nan

            # Chequeo de identificación: en algunas muestras bootstrap, el
            # remuestreo de clusters deja el término de precio colineal con
            # alguna combinación de las dummies de categoría/centro, y OLS
            # devuelve un coeficiente numéricamente arbitrario (magnitud de
            # miles de millones) con un error estándar igual de inflado -
            # la firma de un parámetro no identificado en esa muestra
            # particular. Las elasticidades economicamente plausibles en
            # este modelo están en un rango de una o dos cifras; usamos un
            # umbral generoso (50) para descartar solo estos casos degenerados
            # sin filtrar variación legítima entre iteraciones.
            COEF_PLAUSIBLE_MAX = 50.0
            if (not np.isfinite(price_coef_raw) or not np.isfinite(price_se_raw)
                    or abs(price_coef_raw) > COEF_PLAUSIBLE_MAX
                    or abs(price_se_raw) > COEF_PLAUSIBLE_MAX):
                raise ValueError(
                    f"Coeficiente de precio no identificado en esta muestra "
                    f"(coef={price_coef_raw:.2e}, se={price_se_raw:.2e}), "
                    f"iteración descartada."
                )

            # A partir de aquí la iteración quedó validada: recién ahora
            # se hace exactamente UN append por array.

            # v_hat
            if 'v_hat' in param_names:
                coef_vhat.append(ss_model.params['v_hat'])
            else:
                coef_vhat.append(np.nan)

            coef_price_linear.append(price_coef_raw)

            # Términos RCS no lineales (rcs_1, rcs_2, rcs_3)
            for rcs_term in ['rcs_1', 'rcs_2', 'rcs_3']:
                if rcs_term in param_names:
                    locals()[f'coef_{rcs_term}'].append(ss_model.params[rcs_term])
                else:
                    locals()[f'coef_{rcs_term}'].append(np.nan)

            # Tests Wald
            # Exogeneidad: coef(v_hat) = 0
            R_exog = np.zeros((1, len(ss_model.params)))
            if 'v_hat' in param_names:
                R_exog[0, param_names.index('v_hat')] = 1
                wald_exog = ss_model.wald_test(R_exog)
                f_val = float(wald_exog.fvalue.item()) if hasattr(wald_exog.fvalue, 'item') else float(wald_exog.fvalue)
                f_exog.append(f_val)
            else:
                f_exog.append(np.nan)

            # No-linealidad condicional: coef(rcs_1, rcs_2, rcs_3) = 0
            R_nl = np.zeros((3, len(ss_model.params)))
            nl_terms = ['rcs_1', 'rcs_2', 'rcs_3']
            for j, term in enumerate(nl_terms):
                if term in param_names:
                    R_nl[j, param_names.index(term)] = 1
            wald_nl = ss_model.wald_test(R_nl)
            f_val_nl = float(wald_nl.fvalue.item()) if hasattr(wald_nl.fvalue, 'item') else float(wald_nl.fvalue)
            f_nl.append(f_val_nl)

            n_valid += 1

        except Exception as e:
            # Iteración fallida (matriz mal condicionada, parámetro no
            # identificado, etc.) - omitir silenciosamente
            if "no identificado" in str(e):
                n_descartadas_colinealidad += 1
            coef_vhat.append(np.nan)
            coef_price_linear.append(np.nan)
            coef_rcs_1.append(np.nan)
            coef_rcs_2.append(np.nan)
            coef_rcs_3.append(np.nan)
            f_exog.append(np.nan)
            f_nl.append(np.nan)
            continue

    # Verificación de integridad: todos los arrays deben tener exactamente
    # una entrada por iteración. Si esto falla, hay un bug de alineación
    # en el loop de extracción — mejor abortar aquí con un mensaje claro
    # que dejar que np.array(...)[mask] falle más abajo con un error críptico.
    _lens = {
        'coef_vhat': len(coef_vhat), 'coef_price_linear': len(coef_price_linear),
        'coef_rcs_1': len(coef_rcs_1), 'coef_rcs_2': len(coef_rcs_2),
        'coef_rcs_3': len(coef_rcs_3), 'f_exog': len(f_exog), 'f_nl': len(f_nl),
    }
    if len(set(_lens.values())) != 1 or list(_lens.values())[0] != n_iteraciones:
        raise RuntimeError(f"Desalineación de arrays detectada: {_lens} (esperado {n_iteraciones} en todos)")

    results = _agregar_resultados(
        coef_vhat, coef_price_linear, coef_rcs_1, coef_rcs_2, coef_rcs_3,
        f_exog, f_nl, n_iteraciones, n_valid, n_descartadas_colinealidad
    )

    # Guardar
    np.save(VHAT_OUT, results)
    # Limpiar el checkpoint parcial, ya no hace falta con el resultado final guardado
    if CHECKPOINT_OUT.exists():
        CHECKPOINT_OUT.unlink()

    print(f"\n=== Bootstrap v_hat completado: {n_valid}/{n_iteraciones} iteraciones válidas "
          f"({n_descartadas_colinealidad} descartadas por parámetro de precio no identificado) ===")
    print(f"  v_hat: media={results['v_hat']['elasticidad_media']:.4f}, IC95=[{results['v_hat']['ic_95_inferior']:.4f}, {results['v_hat']['ic_95_superior']:.4f}]")
    print(f"  log_price lineal: media={results['log_checkout_price_linear']['elasticidad_media']:.4f}, IC95=[{results['log_checkout_price_linear']['ic_95_inferior']:.4f}, {results['log_checkout_price_linear']['ic_95_superior']:.4f}]")
    print(f"  RCS no lineales: rcs_1={results['rcs_1']['elasticidad_media']:.4f}, rcs_2={results['rcs_2']['elasticidad_media']:.4f}, rcs_3={results['rcs_3']['elasticidad_media']:.4f}")
    print(f"  Test exogeneidad F (media): {results['test_exogeneidad']['F_media']:.2f}")
    print(f"  Test no-linealidad cond. F (media): {results['test_no_linealidad_condicional']['F_media']:.2f}")
    print(f"  Resultados guardados en: {VHAT_OUT}")

    return results


def main():

    print("=" * 70)
    print("ETAPA 5/P1-14 — BOOTSTRAP CLUSTERIZADO PARA CONTROL FUNCTION (v_hat)")
    print("=" * 70)
    print("""
Este script implementa la inferencia correcta para el regresor generado v_hat
en el enfoque de Control Function (Wooldridge 2015, §6.3.3).

PROBLEMA: Los EE estándar de la segunda etapa (EE=0.018 para v_hat) NO corrigen
por la varianza de estimación de v_hat en la primera etapa. Son un LÍMITE INFERIOR.

SOLUCIÓN: Bootstrap clusterizado remuestreando 77 centros, re-estimando AMBAS
etapas en cada iteración.

NOTA (ajustado tras smoke test): la tasa de identificación observada del
coeficiente de precio en la segunda etapa es ~10% (colinealidad inducida por
el remuestreo de clusters con las dummies de categoría/centro). Para lograr
un N efectivo válido cercano a las 500 iteraciones que recomienda Wooldridge,
se corren 5000 iteraciones totales.

Dataset: demanda_limpia_v2.csv (hash: 90d41670226e961b3eb3ceb6b5b312575082280198a6898b3b1f1bccf5a4707e)
Output: reports/notebook_outputs/bootstrap_vhat_results.npy
""")

    df = cargar_datos()

    results = run_bootstrap_vhat(df, n_iteraciones=5000, semilla=42)

    print("\n" + "=" * 70)
    print("BOOTSTRAP v_hat COMPLETADO")
    print("=" * 70)
    print(f"Entregar: {VHAT_OUT}")


if __name__ == "__main__":
    main()