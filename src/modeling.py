"""
Funciones de modelado para la estimación de elasticidad precio-demanda.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix, build_design_matrices
from scipy.stats import f as _f_dist

# PROJECT_ROOT para cargar archivos .npy desde reports/notebook_outputs
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Importar utilidades RCS canónicas
from rcs_utils import (
    build_canonical_rcs_design,
    apply_rcs_design,
    first_stage_formula,
    second_stage_formula,
    construir_instrumentos as construir_instrumentos_canonico,
)


def modelo_log_log_base(df: pd.DataFrame):
    """Modelo base: log(num_orders) ~ log(checkout_price).

    El coeficiente de log_checkout_price se interpreta directamente
    como la elasticidad precio de la demanda.
    """
    modelo = smf.ols('log_num_orders ~ log_checkout_price', data=df).fit()
    return modelo


def modelo_log_log_controles(df: pd.DataFrame):
    """Modelo log-log con variables de control: promoción, presencia en
    homepage, si hubo aumento de precio respecto a la referencia, y
    efectos fijos por categoría de comida y tipo de centro.
    """
    formula = (
        'log_num_orders ~ log_checkout_price '
        '+ emailer_for_promotion + homepage_featured + hubo_aumento '
        '+ C(category) + C(center_type)'
    )
    modelo = smf.ols(formula, data=df).fit()
    return modelo


def modelo_semilog(df: pd.DataFrame):
    """Especificación alternativa semi-log: log(num_orders) ~ checkout_price
    (precio en nivel, no logarítmico). Sirve como punto de comparación
    frente a la especificación log-log mediante criterios de información.
    """
    formula = (
        'log_num_orders ~ checkout_price '
        '+ emailer_for_promotion + homepage_featured + hubo_aumento '
        '+ C(category) + C(center_type)'
    )
    modelo = smf.ols(formula, data=df).fit()
    return modelo


def tabla_comparacion_modelos(modelos: dict) -> pd.DataFrame:
    """Arma una tabla comparativa de AIC/BIC/R2 ajustado para un
    diccionario de modelos ya ajustados, con el nombre del modelo como clave.
    """
    filas = []
    for nombre, modelo in modelos.items():
        filas.append({
            'modelo': nombre,
            'aic': modelo.aic,
            'bic': modelo.bic,
            'r2_ajustado': modelo.rsquared_adj,
            'n_observaciones': int(modelo.nobs),
        })
    return pd.DataFrame(filas).sort_values('aic').reset_index(drop=True)


def bootstrap_elasticidad(df: pd.DataFrame, n_iteraciones: int = 1000,
                           tamano_muestra: int = None,
                           semilla: int = 42) -> dict:
    """Estima un intervalo de confianza por bootstrap para el coeficiente
    de elasticidad del modelo base (log_checkout_price).

    En cada iteración se toma una remuestra con reemplazo del mismo tamaño
    que el dataset original (o el indicado en tamano_muestra), se ajusta el
    modelo base, y se guarda el coeficiente estimado. El intervalo de
    confianza al 95% se obtiene con los percentiles 2.5 y 97.5 de la
    distribución de coeficientes resultante.
    """
    rng = np.random.default_rng(semilla)
    n = tamano_muestra or len(df)
    coeficientes = np.empty(n_iteraciones)

    for i in range(n_iteraciones):
        muestra = df.sample(n=n, replace=True, random_state=int(rng.integers(0, 1_000_000)))
        modelo = smf.ols('log_num_orders ~ log_checkout_price', data=muestra).fit()
        coeficientes[i] = modelo.params['log_checkout_price']

    return {
        'elasticidad_media': float(np.mean(coeficientes)),
        'elasticidad_std': float(np.std(coeficientes)),
        'ic_95_inferior': float(np.percentile(coeficientes, 2.5)),
        'ic_95_superior': float(np.percentile(coeficientes, 97.5)),
        'coeficientes': coeficientes,
    }


# ======================================================================
# Fase 3b — Panel con efectos fijos de dos vías
# ======================================================================

def preparar_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara el DataFrame con el índice de panel (entidad, tiempo) que
    requiere linearmodels.PanelOLS.

    La entidad es la combinación center_id-meal_id (un "producto" observado
    repetidamente a lo largo de las semanas); el tiempo es la semana.
    """
    df_panel = df.copy()
    df_panel['entidad'] = (
        df_panel['center_id'].astype(str) + '_' + df_panel['meal_id'].astype(str)
    )
    df_panel = df_panel.set_index(['entidad', 'week'])
    return df_panel


def modelo_panel_efectos_fijos(df: pd.DataFrame):
    """Modelo log-log con efectos fijos de dos vías: por entidad
    (combinación center_id-meal_id) y por tiempo (semana), con errores
    estándar clusterizados por entidad.

    Se usa PanelOLS (transformación within) en vez de dummies explícitas
    porque el número de entidades (hasta ~3,900 combinaciones posibles)
    haría inviable en memoria una matriz de diseño con esa cantidad de
    columnas dummy.
    """
    from linearmodels import PanelOLS

    df_panel = preparar_panel(df)
    formula = (
        'log_num_orders ~ log_checkout_price + emailer_for_promotion '
        '+ homepage_featured + EntityEffects + TimeEffects'
    )
    modelo = PanelOLS.from_formula(formula, data=df_panel)
    resultado = modelo.fit(cov_type='clustered', cluster_entity=True)
    return resultado


def bootstrap_elasticidad_clusterizado(df: pd.DataFrame, cluster_col: str = 'center_id',
                                        n_iteraciones: int = 200, semilla: int = 42) -> dict:
    """Bootstrap por bloques (cluster bootstrap): en cada iteración se
    remuestrean las UNIDADES de cluster (p. ej. centros completos), no
    filas individuales, preservando la estructura de dependencia dentro
    de cada cluster. Se reajusta el modelo de panel en cada iteración.

    Nota: menos iteraciones que el bootstrap simple de la Fase 3
    (200 en vez de 1,000) porque cada iteración reajusta un modelo de
    panel completo, más costoso computacionalmente.
    """
    rng = np.random.default_rng(semilla)
    clusters = df[cluster_col].unique()
    coeficientes = []

    for i in range(n_iteraciones):
        clusters_muestreados = rng.choice(clusters, size=len(clusters), replace=True)
        partes = [df[df[cluster_col] == c] for c in clusters_muestreados]
        muestra = pd.concat(partes, ignore_index=True)
        try:
            resultado = modelo_panel_efectos_fijos(muestra)
            coeficientes.append(resultado.params['log_checkout_price'])
        except Exception:
            continue  # alguna remuestra puede quedar mal condicionada; se omite

    coeficientes = np.array(coeficientes)
    return {
        'elasticidad_media': float(np.mean(coeficientes)),
        'elasticidad_std': float(np.std(coeficientes)),
        'ic_95_inferior': float(np.percentile(coeficientes, 2.5)),
        'ic_95_superior': float(np.percentile(coeficientes, 97.5)),
        'n_iteraciones_validas': len(coeficientes),
    }


# ======================================================================
# Fase 3c — Variables instrumentales reales (Hausman-Nevo)
# ======================================================================

def construir_instrumentos(df: pd.DataFrame) -> pd.DataFrame:
    """Construye los dos instrumentos reales para checkout_price:

    1. log_base_price: log del precio de referencia administrativo
       (ya existe base_price en el dataset original).
    2. log_precio_prom_otros_centros: log del precio promedio de la MISMA
       comida (meal_id) en OTROS centros esa MISMA semana (leave-one-out
       mean), técnica de Hausman-Nevo.
    """
    df = df.copy()
    df['log_base_price'] = np.log(df['base_price'])

    # Suma y conteo del precio por (meal_id, week), para poder restar la
    # observación propia y obtener el promedio de "los demás centros".
    grupo = df.groupby(['meal_id', 'week'])['checkout_price']
    suma_grupo = grupo.transform('sum')
    conteo_grupo = grupo.transform('count')

    suma_otros = suma_grupo - df['checkout_price']
    conteo_otros = conteo_grupo - 1

    # Si una comida-semana solo existe en un centro, no hay "otros centros"
    # con los que construir el instrumento para esa fila -> queda NaN.
    precio_prom_otros = suma_otros / conteo_otros.replace(0, np.nan)
    df['precio_prom_otros_centros'] = precio_prom_otros
    df['log_precio_prom_otros_centros'] = np.log(df['precio_prom_otros_centros'])

    return df


def modelo_iv_2sls(df: pd.DataFrame):
    """Modelo log-log con checkout_price instrumentado por 2SLS usando
    los dos instrumentos de construir_instrumentos().

    Requiere que el DataFrame ya tenga las columnas de instrumentos
    (correr construir_instrumentos() antes) y que se excluyan las filas
    con NaN en el instrumento (comida-semana sin "otros centros").

    ESPECIFICACIÓN ARMONIZADA: incluye los mismos controles que otros modelos
    (hubo_aumento, C(category), C(center_type)) para comparabilidad.
    """
    from linearmodels.iv import IV2SLS

    df_iv = df.dropna(subset=['log_precio_prom_otros_centros']).copy()

    formula = (
        'log_num_orders ~ 1 + emailer_for_promotion + homepage_featured + hubo_aumento '
        '+ C(category) + C(center_type) '
        '+ [log_checkout_price ~ log_base_price + log_precio_prom_otros_centros]'
    )
    modelo = IV2SLS.from_formula(formula, data=df_iv)
    resultado = modelo.fit(cov_type='robust')
    return resultado, df_iv


def test_sobreidentificacion_hansen_j(df_iv: pd.DataFrame):
    """Test de sobre-identificación de Hansen J vía IVGMM. Solo tiene
    sentido con más instrumentos que variables endógenas (aquí: 2
    instrumentos para 1 endógena, 1 grado de libertad de sobre-id).

    H0: todos los instrumentos son válidos (no correlacionados con el
    error). Un p-valor bajo rechaza H0 -- sugiere que al menos uno de
    los instrumentos viola el supuesto de exclusión.
    """
    from linearmodels.iv import IVGMM

    formula = (
        'log_num_orders ~ 1 + emailer_for_promotion + homepage_featured + hubo_aumento '
        '+ C(category) + C(center_type) '
        '+ [log_checkout_price ~ log_base_price + log_precio_prom_otros_centros]'
    )
    modelo = IVGMM.from_formula(formula, data=df_iv)
    resultado = modelo.fit(cov_type='robust')
    return resultado.j_stat  # objeto con .stat, .pval, .df


def modelo_iv_justamente_identificado(df: pd.DataFrame, instrumento: str = 'log_base_price'):
    """Versión de robustez del 2SLS usando UN SOLO instrumento (por
    defecto, log_base_price). Al quedar justamente identificado (1
    instrumento para 1 endógena), no hay grados de libertad de
    sobre-identificación -- no existe Hansen J que pueda rechazar, lo
    que evita el problema de colinealidad entre instrumentos detectado
    en el modelo con 2 instrumentos (ver bitácora, Fase 3c).

    Sirve como chequeo de robustez: si la elasticidad se mantiene
    cercana a la del 2SLS con 2 instrumentos, refuerza la confianza en
    el resultado pese al rechazo de Hansen J.
    """
    from linearmodels.iv import IV2SLS

    df_iv = df.dropna(subset=[instrumento]).copy()

    formula = (
        'log_num_orders ~ 1 + emailer_for_promotion + homepage_featured + hubo_aumento '
        '+ C(category) + C(center_type) '
        f'+ [log_checkout_price ~ {instrumento}]'
    )
    modelo = IV2SLS.from_formula(formula, data=df_iv)
    resultado = modelo.fit(cov_type='robust')
    return resultado


# ======================================================================
# Fase 3x — Especificación flexible: Splines en log-precio (RCS)
# ======================================================================

def modelo_log_log_rcs(df: pd.DataFrame, df_splines: int = 4,
                       percentiles: tuple = (5, 35, 65, 95)):
    """Modelo log-log con Restricted Cubic Splines (RCS) en log_checkout_price.

    USO HISTÓRICO — DEPRECADO PARA NUEVOS ANÁLISIS.

    Esta implementación usa `patsy.dmatrix` con `cr(..., df=4, constraints='center')`
    que genera 5 bases (rcs_0..rcs_4) con intercepto redundante. NO es la
    especificación canónica utilizada en el paper final.

    Para análisis nuevos, usar la especificación canónica en `src/rcs_utils.py`:
    - `build_canonical_rcs_design()` — compila design_info con nudos fijos P5/P35/P65/P95
    - `apply_rcs_design()` — aplica la base a nuevos datos usando design_info precompilado
    - `first_stage_formula()` / `second_stage_formula()` — fórmulas canónicas
    - Genera exactamente 3 bases: rcs_1, rcs_2, rcs_3 (base cr centrada; ninguna columna es lineal por sí sola) + intercepto explícito
    - Nudos fijados en muestra completa; bootstrap usa `build_design_matrices` con design_info fijo
    - v_hat se extrae por nombre (`params.get('v_hat')`), no por posición

    Args:
        df: DataFrame con las variables
        df_splines: Grados de libertad del spline (número de nudos = df - 1 para RCS)
        percentiles: Tupla con percentiles para la ubicación de los nudos

    Returns:
        Tupla (modelo_ajustado, matriz_diseno_splines, percentiles_usados)

    .. warning::
        Esta función se mantiene SOLO para reproducibilidad histórica.
        Los resultados del paper usan la especificación canónica (CF/RCS con bootstrap).
        Los coeficientes de esta función histórica (ej. rcs_1=1.2221, rcs_2=0.8049)
        NO aparecen en el paper final ni son reproducibles via bootstrap certificado.
    """
    import warnings
    warnings.warn(
        "modelo_log_log_rcs es HISTÓRICA y DEPRECADA. "
        "Usar build_canonical_rcs_design/apply_rcs_design de rcs_utils.py para especificación canónica.",
        DeprecationWarning,
        stacklevel=2
    )
    # Percentiles de los nudos en log_checkout_price
    log_price = df['log_checkout_price']
    nudos = np.percentile(log_price, percentiles)

    # Construir matriz de diseño con RCS (centrada para interpretabilidad)
    # cr() = cubic regression spline, constraints='center' centra en la media
    transformed_x = dmatrix(
        f"cr(log_checkout_price, df={df_splines}, constraints='center')",
        {"log_checkout_price": log_price},
        return_type='dataframe'
    )

    # Renombrar columnas para claridad
    spline_cols = [f'rcs_{i}' for i in range(transformed_x.shape[1])]
    transformed_x.columns = spline_cols

    # Preparar DataFrame para la regresión (sin intercepto redundante si usa C())
    df_rcs = df.copy()
    for col in spline_cols:
        df_rcs[col] = transformed_x[col]

    # Fórmula: splines + controles (sin log_checkout_price lineal, lo capturan los splines)
    spline_terms = ' + '.join(spline_cols)
    formula = (
        f'log_num_orders ~ {spline_terms} '
        '+ emailer_for_promotion + homepage_featured + hubo_aumento '
        '+ C(category) + C(center_type)'
    )

    modelo = smf.ols(formula, data=df_rcs).fit()
    return modelo, transformed_x, nudos


def elasticidad_puntual_rcs(modelo, spline_design_matrix, nudos_spline: np.ndarray, precio_vals: np.ndarray) -> np.ndarray:
    """Calcula la elasticidad puntual d(log y)/d(log x)
    para un modelo con RCS en log-precio, evaluada en valores específicos de precio.

    Para log-log con splines en log(x): elasticidad = suma(coef_i * d(spline_i)/d(log x))
    donde la derivada de la base de splines se evalúa en log(precio_vals).
    Usa el transformador CR de la matriz de diseño original para evaluar en nuevos puntos.
    """
    from patsy import dmatrix
    import numpy as np

    log_precios = np.log(precio_vals)
    n_splines = spline_design_matrix.shape[1]

    # Obtener el transformador CR de la design_info de la matriz original
    design_info = spline_design_matrix.design_info
    cr_transform = None
    for k, v in design_info.factor_infos.items():
        transforms = v.state['transforms']
        for name, transform in transforms.items():
            if hasattr(transform, '_all_knots'):  # Es el objeto CR
                cr_transform = transform
                break
        if cr_transform is not None:
            break

    if cr_transform is None:
        raise ValueError("No se encontró el transformador CR en la design_info")

    h = 1e-4
    elasticidades = np.zeros(len(precio_vals))

    # Coeficientes del modelo para los términos de spline (excluyendo intercepto rcs_0)
    spline_cols = [c for c in modelo.params.index if c.startswith('rcs_')]
    # rcs_0 es el intercepto/constante, no una base de spline
    # Los términos de spline reales son rcs_1, rcs_2, rcs_3, rcs_4 (4 términos)
    spline_basis_cols = [c for c in spline_cols if c != 'rcs_0']
    coefs_spline = np.array([modelo.params[c] for c in spline_basis_cols])

    for i, log_p in enumerate(log_precios):
        # Evaluar base de splines en log_p y log_p + h usando el transformador CR
        x0 = cr_transform.transform(np.array([log_p]))
        x1 = cr_transform.transform(np.array([log_p + h]))

        # x0 y x1 son arrays (1, n_splines_basis) = (1, 4)
        x0 = np.asarray(x0).flatten()
        x1 = np.asarray(x1).flatten()

        # Derivada numérica de la base de splines
        deriv_splines = (x1 - x0) / h

        # Elasticidad = suma(coef_i * d(spline_i)/d(log x))
        elasticidades[i] = np.dot(coefs_spline, deriv_splines)

    return elasticidades


def test_wald_no_linealidad_rcs(modelo, spline_cols: list) -> dict:
    """Test de Wald para H0: coeficientes de los términos no lineales del spline = 0.

    En RCS con df=k, el primer término captura la componente lineal,
    los k-1 restantes capturan la no-linealidad.
    """
    # Los términos no lineales son todos menos el primero (lineal)
    terminos_no_lineales = spline_cols[1:]

    # Construir matriz de restricciones R * beta = 0
    R = np.zeros((len(terminos_no_lineales), len(modelo.params)))
    param_names = list(modelo.params.index)
    for i, term in enumerate(terminos_no_lineales):
        if term in param_names:
            R[i, param_names.index(term)] = 1

    wald_result = modelo.wald_test(R)
    # fvalue y pvalue son arrays 1x1 en versiones recientes de statsmodels
    f_val = float(wald_result.fvalue.item()) if hasattr(wald_result.fvalue, 'item') else float(wald_result.fvalue)
    p_val = float(wald_result.pvalue.item()) if hasattr(wald_result.pvalue, 'item') else float(wald_result.pvalue)
    return {
        'estadistico_F': f_val,
        'p_valor': p_val,
        'grados_libertad': (float(wald_result.df_num), float(wald_result.df_denom)),
        'rechaza_H0_05': p_val < 0.05,
    }


# ======================================================================
# Plan técnico v2.1 — Sección 2.1: Mejoras de robustez
# ======================================================================

def modelo_quantile_regression(df: pd.DataFrame, tau: float = 0.5,
                                formula: str = None):
    """Regresión cuantil (Quantile Regression) para elasticidad precio-demanda.

    Estima el efecto del precio en diferentes cuantiles de la distribución
    condicional de la demanda (log_num_orders). Permite detectar heterogeneidad:
    ¿la elasticidad es distinta para pedidos bajos (τ=0.1) vs. medianos (τ=0.5)
    vs. altos (τ=0.9)?

    Args:
        df: DataFrame con las variables
        tau: Cuantil a estimar (0.1, 0.5, 0.9)
        formula: Fórmula opcional (por defecto usa log-log con controles)

    Returns:
        Resultado ajustado de statsmodels.regression.quantile_regression
    """
    if formula is None:
        formula = (
            'log_num_orders ~ log_checkout_price '
            '+ emailer_for_promotion + homepage_featured + hubo_aumento '
            '+ C(category) + C(center_type)'
        )

    mod = smf.quantreg(formula, data=df)
    resultado = mod.fit(q=tau, max_iter=1000)
    return resultado


def bootstrap_quantile_regression(df: pd.DataFrame, tau: float = 0.5,
                                   n_iteraciones: int = 500, semilla: int = 42,
                                   formula: str = None) -> dict:
    """Bootstrap para intervalos de confianza en regresión cuantil.

    La regresión cuantil no tiene errores estándar analíticos simples
    (requiere estimación de densidad condicional); el bootstrap por
    remuestreo con reemplazo es el método estándar.

    Args:
        df: DataFrame con las variables
        tau: Cuantil a estimar
        n_iteraciones: Número de replicaciones bootstrap (recomendado 500+)
        semilla: Semilla para reproducibilidad
        formula: Fórmula opcional

    Returns:
        Dict con media, desv. est., IC 95% del coeficiente de log_checkout_price
    """
    rng = np.random.default_rng(semilla)
    if formula is None:
        formula = (
            'log_num_orders ~ log_checkout_price '
            '+ emailer_for_promotion + homepage_featured + hubo_aumento '
            '+ C(category) + C(center_type)'
        )

    n = len(df)
    coeficientes = np.empty(n_iteraciones)

    for i in range(n_iteraciones):
        muestra = df.sample(n=n, replace=True, random_state=int(rng.integers(0, 1_000_000)))
        try:
            mod = smf.quantreg(formula, data=muestra)
            res = mod.fit(q=tau, max_iter=1000)
            coeficientes[i] = res.params['log_checkout_price']
        except Exception:
            coeficientes[i] = np.nan

    coeficientes = coeficientes[~np.isnan(coeficientes)]
    return {
        'tau': tau,
        'elasticidad_media': float(np.mean(coeficientes)),
        'elasticidad_std': float(np.std(coeficientes)),
        'ic_95_inferior': float(np.percentile(coeficientes, 2.5)),
        'ic_95_superior': float(np.percentile(coeficientes, 97.5)),
        'n_iteraciones_validas': len(coeficientes),
    }


def modelo_interaccion_precio_promo(df: pd.DataFrame):
    """Modelo log-log con interacción Precio × Promoción (precio centrado).

    Prueba si el efecto de las promociones (emailer, homepage) depende
    del nivel de precio, o equivalentemente, si la elasticidad-precio
    cambia cuando hay promoción activa. Incluye términos de interacción:
    log_checkout_price_c × emailer_for_promotion y log_checkout_price_c × homepage_featured,
    donde log_checkout_price_c = log_checkout_price - mean(log_checkout_price).

    Centrar el precio hace que el coeficiente principal de la promoción sea interpretable
    en el precio medio (no en log(P)=0, i.e., P=1).

    H0: coeficientes de interacción = 0 (efectos aditivos, sin interacción)
    H1: al menos un coeficiente de interacción ≠ 0

    Returns:
        Tupla (modelo_con_interaccion, test_wald_interaccion, media_log_precio)
    """
    df = df.copy()
    mean_log_price = df['log_checkout_price'].mean()
    df['log_checkout_price_c'] = df['log_checkout_price'] - mean_log_price

    formula = (
        'log_num_orders ~ log_checkout_price_c '
        '+ emailer_for_promotion + homepage_featured + hubo_aumento '
        '+ log_checkout_price_c:emailer_for_promotion '
        '+ log_checkout_price_c:homepage_featured '
        '+ C(category) + C(center_type)'
    )
    modelo = smf.ols(formula, data=df).fit()

    # Test de Wald conjunto para los dos términos de interacción
    terminos_interaccion = [
        'log_checkout_price_c:emailer_for_promotion',
        'log_checkout_price_c:homepage_featured'
    ]
    R = np.zeros((len(terminos_interaccion), len(modelo.params)))
    param_names = list(modelo.params.index)
    for i, term in enumerate(terminos_interaccion):
        if term in param_names:
            R[i, param_names.index(term)] = 1

    wald_result = modelo.wald_test(R)
    f_val = float(wald_result.fvalue.item()) if hasattr(wald_result.fvalue, 'item') else float(wald_result.fvalue)
    p_val = float(wald_result.pvalue.item()) if hasattr(wald_result.pvalue, 'item') else float(wald_result.pvalue)

    return modelo, {
        'estadistico_F': f_val,
        'p_valor': p_val,
        'grados_libertad': (float(wald_result.df_num), float(wald_result.df_denom)),
        'rechaza_H0_05': p_val < 0.05,
    }, float(mean_log_price)


def modelo_iv_control_function_historico(df: pd.DataFrame):
    """IV no-lineal vía Control Function approach (Wooldridge 2015) — ESPECIFICACIÓN HISTÓRICA.

    NOTA: Esta implementación usa la especificación histórica (pre-canónica) con:
    - 4 bases RCS (rcs_0, rcs_1, rcs_2, rcs_3) + intercepto implícito OLS
    - Nudos recalculados por muestra (no fijados en muestra completa)
    - Extracción posicional de v_hat (fuente del off-by-one error en B2)
    - v_hat puntual ~0.7139 (vs 0.9937 canónico bootstrap)

    MANTENIDA PARA COMPATIBILIDAD HISTÓRICA Y REPRODUCIBILIDAD.
    NO USA PARA NUEVOS ANÁLISIS — usar modelo_iv_control_function() (canónico).

    Enfoque en dos etapas:
    1. Primera etapa: log_checkout_price ~ instrumentos + controles → obtener residuos v_hat
    2. Segunda etapa: log_num_orders ~ log_checkout_price + splines(log_checkout_price) + v_hat + controles

    Tests:
    - H0: coef(v_hat) = 0 (exogeneidad de log_checkout_price)
    - H0: coef(splines) = 0 (linealidad en segunda etapa, condicional a endogeneidad)

    Returns:
        Dict con resultados de primera y segunda etapa, tests
    """
    # Primera etapa: construir instrumentos y obtener residuos
    df_iv = construir_instrumentos(df).dropna(subset=['log_precio_prom_otros_centros']).copy()

    formula_fs = (
        'log_checkout_price ~ log_base_price + log_precio_prom_otros_centros '
        '+ emailer_for_promotion + homepage_featured + hubo_aumento '
        '+ C(category) + C(center_type)'
    )
    fs_model = smf.ols(formula_fs, data=df_iv).fit()
    df_iv['v_hat'] = fs_model.resid

    # Segunda etapa: modelo con RCS histórico + v_hat
    modelo_rcs, spline_design, nudos = modelo_log_log_rcs(df_iv)
    # Extraer coeficientes de spline y v_hat del modelo RCS
    # (el modelo RCS ya incluye v_hat si lo pasamos en el df)
    # Aquí volvemos a estimar incluyendo v_hat explícitamente
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

    # Tests
    # 1. Exogeneidad: coef(v_hat) = 0
    # 2. No-linealidad condicional: coef(splines no lineales) = 0
    spline_cols = [c for c in spline_design.columns if c != 'rcs_0']
    spline_nonlinear = spline_cols[1:]  # excluir el lineal

    # Test Wald para v_hat
    R_exog = np.zeros((1, len(ss_model.params)))
    param_names_ss = list(ss_model.params.index)
    if 'v_hat' in param_names_ss:
        R_exog[0, param_names_ss.index('v_hat')] = 1
        wald_exog = ss_model.wald_test(R_exog)
        f_exog = float(wald_exog.fvalue.item()) if hasattr(wald_exog.fvalue, 'item') else float(wald_exog.fvalue)
        p_exog = float(wald_exog.pvalue.item()) if hasattr(wald_exog.pvalue, 'item') else float(wald_exog.pvalue)
        rechaza_exog = p_exog < 0.05 if p_exog is not None else None
    else:
        f_exog = p_exog = None
        rechaza_exog = None

    # Test Wald para no-linealidad condicional
    R_nl = np.zeros((len(spline_nonlinear), len(ss_model.params)))
    for i, term in enumerate(spline_nonlinear):
        if term in param_names_ss:
            R_nl[i, param_names_ss.index(term)] = 1
    wald_nl = ss_model.wald_test(R_nl)
    f_nl = float(wald_nl.fvalue.item()) if hasattr(wald_nl.fvalue, 'item') else float(wald_nl.fvalue)
    p_nl = float(wald_nl.pvalue.item()) if hasattr(wald_nl.pvalue, 'item') else float(wald_nl.pvalue)

    # Elasticidad puntual en precio medio (usando la curva RCS + v_hat)
    # La elasticidad es la derivada de los splines evaluada en el precio medio
    precio_medio = df_iv['checkout_price'].mean()
    elasticidad_p50 = elasticidad_puntual_rcs(ss_model, spline_design, nudos, np.array([precio_medio]))[0]

    return {
        'primera_etapa': fs_model,
        'segunda_etapa': ss_model,
        'test_exogeneidad': {'F': f_exog, 'p': p_exog, 'rechaza_H0_05': rechaza_exog},
        'test_no_linealidad_condicional': {'F': f_nl, 'p': p_nl, 'rechaza_H0_05': p_nl < 0.05},
        'nudos': nudos,
        'elasticidad_p50': float(elasticidad_p50),
        'v_hat_coef': float(ss_model.params.get('v_hat', np.nan)),
        'v_hat_ee': float(ss_model.bse.get('v_hat', np.nan)),
        '_especificacion': 'historica_4_bases_rcs',
    }


def modelo_iv_control_function(df: pd.DataFrame):
    """IV no-lineal vía Control Function approach (Wooldridge 2015) — ESPECIFICACIÓN CANÓNICA.

    Esta implementación corresponde a la especificación certificada en:
    - bootstrap_vhat_canonical.json/.npy (5000 iter, cluster bootstrap by center_id)
    - Paper (Informe/informe.tex, V3.2.5): v_hat=1.0002 (puntual), F_exog=3177.9 (t^2), F_nonlin anidado=3335.4,
      elasticidad=-1.5041 (precio medio), -1.3945 (mediana)
    - cf_elasticidad_curve.npy (curva elástica canónica)

    ESPECIFICACIÓN CANÓNICA:
    - RCS: 3 bases explícitas (rcs_1, rcs_2, rcs_3), constraints='center', -1 (sin rcs_0)
    - Intercepto EXPLÍCITO en segunda etapa (+ 1 en fórmula)
    - Nudos FIJOS: P5, P35, P65, P95 de log_checkout_price (muestra completa)
    - design_info RCS pre-compilado (build_canonical_rcs_design) — NO recalcula por réplica
    - v_hat extraído POR NOMBRE (params.get('v_hat')), NO posicional
    - Primera etapa idéntica a histórica (2 instrumentos Hausman-Nevo)

    Returns:
        Dict con resultados de primera y segunda etapa, tests
    """
    # Preparar datos con instrumentos
    df_iv = construir_instrumentos_canonico(df).dropna(subset=['log_precio_prom_otros_centros']).copy()

    # PRIMERA ETAPA — fórmula canónica (idéntica a histórica)
    fs_model = smf.ols(first_stage_formula(), data=df_iv).fit()
    df_iv['v_hat'] = fs_model.resid

    # RCS CANÓNICO: design_info FIJO en muestra completa
    log_price_full = df_iv['log_checkout_price'].to_numpy()
    design_info, knots = build_canonical_rcs_design(log_price_full)

    # Aplicar RCS canónico a la muestra
    S = apply_rcs_design(design_info, log_price_full)
    for c in S.columns:
        df_iv[c] = S[c].to_numpy()

    # SEGUNDA ETAPA — fórmula canónica con intercepto explícito + 3 RCS + v_hat
    ss_model = smf.ols(second_stage_formula(), data=df_iv).fit()

    # Tests
    param_names_ss = list(ss_model.params.index)

    # 1. Test exogeneidad: H0: coef(v_hat) = 0 (1 g.l.)
    R_exog = np.zeros((1, len(ss_model.params)))
    if 'v_hat' in param_names_ss:
        R_exog[0, param_names_ss.index('v_hat')] = 1
        wald_exog = ss_model.wald_test(R_exog)
        f_exog = float(wald_exog.fvalue.item()) if hasattr(wald_exog.fvalue, 'item') else float(wald_exog.fvalue)
        p_exog = float(wald_exog.pvalue.item()) if hasattr(wald_exog.pvalue, 'item') else float(wald_exog.pvalue)
        rechaza_exog = p_exog < 0.05 if p_exog is not None else None
    else:
        f_exog = p_exog = None
        rechaza_exog = None

    # 2. Test de no linealidad (V3.2.5): F ANIDADO entre la CF lineal en log-precio
    #    (log_checkout_price + v_hat + controles) y la CF-RCS (2 g.l.).
    #    NOTA: con la base cr(..., constraints='center'), rcs_1 por sí sola NO es lineal,
    #    por lo que H0: rcs_2 = rcs_3 = 0 no es una hipótesis de linealidad; ese Wald se
    #    conserva solo como referencia ('test_wald_rcs2_rcs3').
    ss_lin = smf.ols(second_stage_formula().replace('rcs_1 + rcs_2 + rcs_3', 'log_checkout_price'),
                     data=df_iv).fit()
    f_nl = float(((ss_lin.ssr - ss_model.ssr) / 2) / (ss_model.ssr / ss_model.df_resid))
    p_nl = float(_f_dist.sf(f_nl, 2, ss_model.df_resid))
    R_nl = np.zeros((2, len(ss_model.params)))
    for j, nm in enumerate(['rcs_2', 'rcs_3']):
        if nm in param_names_ss:
            R_nl[j, param_names_ss.index(nm)] = 1
    f_w23 = float(np.squeeze(ss_model.wald_test(R_nl, scalar=True).statistic))

    # Elasticidad puntual en precio medio
    precio_medio = df_iv['checkout_price'].mean()
    # Usar la design matrix del RCS canónico para derivadas
    elasticidad_precio_medio = elasticidad_puntual_rcs_canonico(ss_model, design_info, knots, np.array([precio_medio]))[0]

    # Elasticidad puntual en mediana empírica
    precio_mediana = df_iv['checkout_price'].median()
    elasticidad_precio_mediana = elasticidad_puntual_rcs_canonico(ss_model, design_info, knots, np.array([precio_mediana]))[0]

    return {
        'primera_etapa': fs_model,
        'segunda_etapa': ss_model,
        'test_exogeneidad': {'F': f_exog, 'p': p_exog, 'rechaza_H0_05': rechaza_exog},
        'test_no_linealidad_condicional': {'F': f_nl, 'p': p_nl, 'rechaza_H0_05': p_nl < 0.05,
                                           'tipo': 'F anidado CF lineal vs CF-RCS (2 g.l.)'},
        'test_wald_rcs2_rcs3': {'F': f_w23, 'nota': 'no es test de linealidad con base cr centrada'},
        'coef_lineal_equivalente_2sls': float(ss_lin.params['log_checkout_price']),
        'nudos': knots,
        'elasticidad_precio_medio': float(elasticidad_precio_medio),
        'elasticidad_precio_mediana': float(elasticidad_precio_mediana),
        'v_hat_coef': float(ss_model.params.get('v_hat', np.nan)),
        'v_hat_ee': float(ss_model.bse.get('v_hat', np.nan)),
        '_especificacion': 'canonica_3_bases_rcs',
    }


def elasticidad_puntual_rcs_canonico(modelo, design_info, knots: np.ndarray, precio_vals: np.ndarray) -> np.ndarray:
    """Calcula elasticidad puntual para especificación RCS canónica (3 bases: rcs_1, rcs_2, rcs_3).

    Args:
        modelo: Modelo OLS ajustado con rcs_1, rcs_2, rcs_3 + intercepto + v_hat + controles
        design_info: patsy DesignInfo de build_canonical_rcs_design()
        knots: Nudos canónicos [P5, P35, P65, P95]
        precio_vals: Array de precios donde evaluar la elasticidad

    Returns:
        Array de elasticidades puntuales
    """
    log_precios = np.log(precio_vals)
    h = 1e-5
    elasticidades = np.zeros(len(precio_vals))

    # Coeficientes de los 3 términos RCS canónicos
    spline_basis_cols = ['rcs_1', 'rcs_2', 'rcs_3']
    coefs_spline = np.array([modelo.params[c] for c in spline_basis_cols])

    for i, log_p in enumerate(log_precios):
        # Evaluar base de splines en log_p y log_p + h usando design_info FIJO
        x0 = build_design_matrices([design_info], {'log_checkout_price': np.array([log_p - h])}, return_type='dataframe')[0]
        x1 = build_design_matrices([design_info], {'log_checkout_price': np.array([log_p + h])}, return_type='dataframe')[0]

        x0 = np.asarray(x0).flatten()
        x1 = np.asarray(x1).flatten()

        # Derivada numérica de la base de splines (diferencia CENTRAL, error O(h^2); V3.2.5)
        deriv_splines = (x1 - x0) / (2 * h)

        # Elasticidad = suma(coef_i * d(spline_i)/d(log x))
        elasticidades[i] = np.dot(coefs_spline, deriv_splines)

    return elasticidades


# ======================================================================
# Plan técnico v2.1 — Sección 2.3: Items de robustez (bajo esfuerzo)
# ======================================================================

def test_igualdad_elasticidad_por_categoria(df: pd.DataFrame) -> dict:
    """Test de Wald conjunto: ¿difieren las elasticidades por categoría?

    Estima un modelo con interacción log_checkout_price × C(category)
    y testa H0: todos los coeficientes de interacción = 0.
    Si se rechaza, la elasticidad-precio varía significativamente entre categorías.
    """
    formula = (
        'log_num_orders ~ log_checkout_price + C(category) '
        '+ log_checkout_price:C(category) '
        '+ emailer_for_promotion + homepage_featured + hubo_aumento '
        '+ C(center_type)'
    )
    modelo = smf.ols(formula, data=df).fit()

    # Identificar coeficientes de interacción log_checkout_price:C(category)[T.XXX]
    terminos_interaccion = [
        c for c in modelo.params.index
        if c.startswith('log_checkout_price:C(category)[')
    ]

    if not terminos_interaccion:
        return {'error': 'No se encontraron términos de interacción'}

    R = np.zeros((len(terminos_interaccion), len(modelo.params)))
    param_names = list(modelo.params.index)
    for i, term in enumerate(terminos_interaccion):
        R[i, param_names.index(term)] = 1

    wald_result = modelo.wald_test(R)
    f_val = float(wald_result.fvalue.item()) if hasattr(wald_result.fvalue, 'item') else float(wald_result.fvalue)
    p_val = float(wald_result.pvalue.item()) if hasattr(wald_result.pvalue, 'item') else float(wald_result.pvalue)

    # Extraer elasticidad por categoría (efecto marginal)
    elasticidades = {}
    coef_base = modelo.params.get('log_checkout_price', 0)
    for cat in df['category'].unique():
        term = f'log_checkout_price:C(category)[T.{cat}]'
        if term in modelo.params:
            elasticidades[cat] = coef_base + modelo.params[term]
        else:
            elasticidades[cat] = coef_base  # categoría base

    return {
        'test_wald': {
            'estadistico_F': f_val,
            'p_valor': p_val,
            'grados_libertad': (float(wald_result.df_num), float(wald_result.df_denom)),
            'rechaza_H0_05': p_val < 0.05,
        },
        'elasticidad_por_categoria': elasticidades,
        'modelo': modelo,
    }


def submuestra_obs_minimas(df: pd.DataFrame, min_obs: int = 50) -> pd.DataFrame:
    """Filtra el dataset a combinaciones centro-comida con ≥ min_obs observaciones semanales.

    Reduce ruido en items esporádicos que pueden sesgar la estimación.
    """
    conteo = df.groupby(['center_id', 'meal_id']).size().reset_index(name='n_obs')
    entidades_validas = conteo[conteo['n_obs'] >= min_obs][['center_id', 'meal_id']]
    df_filtrado = df.merge(entidades_validas, on=['center_id', 'meal_id'], how='inner')
    return df_filtrado


def sensibilidad_excluir_outliers_precio(df: pd.DataFrame, pct_bajo: float = 1, pct_alto: float = 1) -> pd.DataFrame:
    """Excluye el top/bottom pct% de checkout_price para test de sensibilidad.

    Verifica que outliers de precio no dominen la estimación de elasticidad.
    """
    p_low = np.percentile(df['checkout_price'], pct_bajo)
    p_high = np.percentile(df['checkout_price'], 100 - pct_alto)
    df_filtrado = df[(df['checkout_price'] >= p_low) & (df['checkout_price'] <= p_high)].copy()
    return df_filtrado


def comparar_errores_estandar(df: pd.DataFrame) -> dict:
    """Compara HC3 vs cluster por centro vs cluster por entidad centro-comida vs Bootstrap en el modelo OLS log-log con controles.

    Retorna la elasticidad y sus errores estándar bajo enfoques:
    1. HC3 (heterocedasticidad robusto) -- SOLO para referencia, NO usar como inferencia principal
    2. Cluster por center_id
    3. Cluster por entidad centro-comida: center_id × meal_id (no es clustering bidimensional por separado)
    4. Panel FE clusterizado (entidad = center_id-meal_id)
    5. Bootstrap simple (modelo base) -- no reproduce estructura de dependencia
    6. Wild cluster bootstrap (robustez)

    NOTA: Durbin-Watson = 1.195 indica autocorrelación; HC3 no corrige esto.
    El EE cluster por center_id es ~7.92x mayor que HC3 (0.0421 vs 0.0053).
    La inferencia principal DEBE usar clustering coherente con la estructura de panel.
    """
    from linearmodels import PanelOLS

    formula = (
        'log_num_orders ~ log_checkout_price '
        '+ emailer_for_promotion + homepage_featured + hubo_aumento '
        '+ C(category) + C(center_type)'
    )

    # 1. OLS con HC3 (solo referencia, NO inferencia principal)
    modelo_ols = smf.ols(formula, data=df).fit(cov_type='HC3')
    ee_hc3 = modelo_ols.bse.get('log_checkout_price', np.nan)
    coef_hc3 = modelo_ols.params.get('log_checkout_price', np.nan)

    # 2. OLS con cluster por center_id
    modelo_cluster = smf.ols(formula, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['center_id']}
    )
    ee_cluster = modelo_cluster.bse.get('log_checkout_price', np.nan)
    coef_cluster = modelo_cluster.params.get('log_checkout_price', np.nan)

    # 3. Cluster por entidad centro-comida: center_id × meal_id (no es clustering bidimensional por separado) (Cameron, Gelbach, Miller 2011)
    # Usamos statsmodels con grupos de interacción
    df_multiway = df.copy()
    df_multiway['center_meal'] = df_multiway['center_id'].astype(str) + '_' + df_multiway['meal_id'].astype(str)
    modelo_multiway = smf.ols(formula, data=df_multiway).fit(
        cov_type='cluster', cov_kwds={'groups': df_multiway['center_meal']}
    )
    ee_multiway = modelo_multiway.bse.get('log_checkout_price', np.nan)
    coef_multiway = modelo_multiway.params.get('log_checkout_price', np.nan)

    # 4. Panel FE clusterizado (entidad = center_id-meal_id)
    df_panel = preparar_panel(df)
    formula_panel = (
        'log_num_orders ~ log_checkout_price + emailer_for_promotion + homepage_featured '
        '+ EntityEffects + TimeEffects'
    )
    panel_model = PanelOLS.from_formula(formula_panel, data=df_panel)
    panel_result = panel_model.fit(cov_type='clustered', cluster_entity=True)
    ee_panel = panel_result.std_errors.get('log_checkout_price', np.nan)
    coef_panel = panel_result.params.get('log_checkout_price', np.nan)

    # 5. Bootstrap simple (modelo base, ya implementado) -- no reproduce dependencia
    boot = bootstrap_elasticidad(df, n_iteraciones=1000)
    ee_boot = boot['elasticidad_std']
    coef_boot = boot['elasticidad_media']

    # 6. Wild cluster bootstrap (robustez para pocos clusters)
    # Cargar desde .npy si existe (generado por scripts/vm_wild_bootstrap.py)
    wild_boot_path = PROJECT_ROOT / "reports" / "notebook_outputs" / "wild_cluster_bootstrap_results.npy"
    if wild_boot_path.exists():
        wild_boot = np.load(wild_boot_path, allow_pickle=True).item()
    else:
        wild_boot = {
            'elasticidad_media': np.nan,
            'elasticidad_std': np.nan,
            'ic_95_inferior': np.nan,
            'ic_95_superior': np.nan,
            'n_iteraciones_validas': 0,
            'nota': 'PENDIENTE: ejecutar en VM con scripts/vm_wild_bootstrap.py'
        }

    # Ratios
    ratio_cluster_hc3 = ee_cluster / ee_hc3 if ee_hc3 and ee_cluster else np.nan
    ratio_multiway_hc3 = ee_multiway / ee_hc3 if ee_hc3 and ee_multiway else np.nan

    return {
        'ols_hc3': {'elasticidad': coef_hc3, 'ee': ee_hc3, 'nota': 'SOLO REFERENCIA - no corrige autocorrelación'},
        'ols_cluster_center': {'elasticidad': coef_cluster, 'ee': ee_cluster},
        'ols_multiway_center_meal': {'elasticidad': coef_multiway, 'ee': ee_multiway, 'nota': 'Agrupación por entidad centro-comida; clave histórica mantenida por compatibilidad'},
        'panel_fe_cluster_entidad': {'elasticidad': coef_panel, 'ee': ee_panel},
        'bootstrap_simple': {'elasticidad': coef_boot, 'ee': ee_boot, 'nota': 'No reproduce estructura panel'},
        'wild_cluster_bootstrap': wild_boot,
        'ratio_cluster_hc3': ratio_cluster_hc3,
        'ratio_multiway_hc3': ratio_multiway_hc3,
        'ratio_bootstrap_hc3': ee_boot / ee_hc3 if ee_hc3 else np.nan,
    }


# ======================================================================
# Plan técnico v2.1 — Sección 2.2: Alto impacto / Medio esfuerzo
# ======================================================================

def panel_fe_time_varying(df: pd.DataFrame) -> dict:
    """
    Modelo Panel FE con interacción tiempo-variante (log_price × semana).

    Estima si la elasticidad-precio varía sistemáticamente entre semanas.
    Modelo: log_orders ~ log_price + log_price × week_dummies + EntityEffects + TimeEffects

    Test de Wald conjunto para H0: coeficientes de interacción = 0.

    NOTA (fix de rendimiento/memoria): la versión anterior usaba
    PanelOLS.from_formula de linearmodels, que construye la matriz de
    diseño completa vía patsy (incluyendo ~144 columnas dummy para la
    interacción) y agota la memoria disponible en datasets de este tamaño
    (456K filas, 3597 entidades, 145 semanas) — se verificó que el proceso
    moría por falta de RAM incluso sin la interacción, solo por los
    efectos fijos de doble vía (entidad + tiempo).

    Esta versión es matemáticamente equivalente (validada contra
    linearmodels en subconjuntos: coeficientes idénticos hasta 12+
    decimales, estadístico de Wald idéntico) pero usa "demeaning"
    iterativo (teorema de Frisch-Waugh-Lovell / "within transform") para
    absorber los efectos fijos de entidad y tiempo en vez de construir
    dummies explícitas, y clustered SE calculados manualmente (sandwich
    CR1, mismo ajuste de grados de libertad que linearmodels: N - K -
    (n_entidades-1) - (n_semanas-1)). Reduce el uso de RAM de >3.9GB
    (falla) a ~1.2GB, y corre en ~2-3 minutos en vez de fallar.

    IMPORTANTE: el test de Wald de linearmodels reporta un estadístico
    chi-cuadrado (no F, pese a que el código original lo llamaba 'F') —
    esta versión reproduce exactamente ese mismo estadístico y lo
    devuelve bajo la misma clave 'F' por compatibilidad con el resto del
    pipeline, pero su distribución de referencia es chi2(gl), no F.
    """
    from scipy import stats

    df_local = df.copy()
    df_local['entidad'] = df_local['center_id'].astype(str) + '_' + df_local['meal_id'].astype(str)

    ent_codes, _ = pd.factorize(df_local['entidad'])
    week_codes, weeks_u = pd.factorize(df_local['week'])
    n_ent, n_time = int(ent_codes.max() + 1), int(week_codes.max() + 1)
    weeks_sorted = sorted(int(w) for w in weeks_u)

    # --- Matriz de diseño: [y, log_price, controles, interacciones price×week] ---
    price = df_local['log_checkout_price'].to_numpy(dtype=np.float64)
    # drop_first=True: se omite la primera semana (categoría base), igual que
    # el comportamiento por defecto de C(week_dummy) en patsy/statsmodels.
    week_dummies = pd.get_dummies(pd.Categorical(week_codes), drop_first=True).to_numpy(dtype=np.float64)
    interaccion = week_dummies * price[:, None]
    Z = np.column_stack([
        df_local['log_num_orders'].to_numpy(dtype=np.float64),
        price,
        df_local[['emailer_for_promotion', 'homepage_featured', 'hubo_aumento']].to_numpy(dtype=np.float64),
        interaccion,
    ])
    del week_dummies, interaccion

    # --- Demeaning iterativo (absorbe EntityEffects + TimeEffects, exacto por FWL) ---
    cnt_e = np.bincount(ent_codes, minlength=n_ent).astype(np.float64)
    cnt_t = np.bincount(week_codes, minlength=n_time).astype(np.float64)
    n_iter_max = 200
    for it in range(n_iter_max):
        se = np.zeros((n_ent, Z.shape[1]))
        np.add.at(se, ent_codes, Z)
        corr_e = (se / cnt_e[:, None])[ent_codes]
        Z -= corr_e
        st = np.zeros((n_time, Z.shape[1]))
        np.add.at(st, week_codes, Z)
        corr_t = (st / cnt_t[:, None])[week_codes]
        Z -= corr_t
        delta = max(np.max(np.abs(corr_e)), np.max(np.abs(corr_t)))
        if delta < 1e-8:
            break

    y_dm, X_dm = Z[:, 0], Z[:, 1:]
    K = X_dm.shape[1]
    N = X_dm.shape[0]

    # --- OLS sobre datos "demeaned" (equivalente exacto a incluir las FE como dummies) ---
    coef, *_ = np.linalg.lstsq(X_dm, y_dm, rcond=None)
    resid = y_dm - X_dm @ coef
    XtX_inv = np.linalg.pinv(X_dm.T @ X_dm)

    # --- Clustered SE por entidad (sandwich CR1), mismo ajuste de gl que linearmodels ---
    Xr = X_dm * resid[:, None]
    sums_score = np.zeros((n_ent, K))
    np.add.at(sums_score, ent_codes, Xr)
    meat = sums_score.T @ sums_score  # = sum_g outer(score_g, score_g)
    G = n_ent
    df_resid = N - K - (n_ent - 1) - (n_time - 1)
    dof_corr = (G / (G - 1)) * ((N - 1) / df_resid)
    V = dof_corr * XtX_inv @ meat @ XtX_inv

    # --- Extraer coeficientes: [0]=log_price, [1:4]=controles, [4:]=interacciones ---
    coef_base = float(coef[0])
    interaction_weeks = weeks_sorted[1:]  # se omitió la primera semana como base
    interacciones = {int(w): float(coef[4 + i]) for i, w in enumerate(interaction_weeks)}

    # --- Test de Wald conjunto: H0: todos los coef. de interacción = 0 ---
    idx_interaccion = list(range(4, K))
    R = np.zeros((len(idx_interaccion), K))
    for i, idx in enumerate(idx_interaccion):
        R[i, idx] = 1
    Rb = R @ coef
    RVR = R @ V @ R.T
    q = len(idx_interaccion)
    chi2_stat = float(Rb.T @ np.linalg.pinv(RVR) @ Rb)
    p_val = float(1 - stats.chi2.cdf(chi2_stat, q))

    elasticidad_por_semana = {int(w): coef_base for w in weeks_sorted}
    for w, v in interacciones.items():
        elasticidad_por_semana[w] = coef_base + v

    return {
        'coef_base': coef_base,
        'interacciones': interacciones,
        'elasticidad_por_semana': {int(k): float(v) for k, v in elasticidad_por_semana.items()},
        'test_wald': {
            'F': chi2_stat,  # en rigor es un estadistico chi2(gl), no F (ver docstring)
            'p_valor': p_val,
            'rechaza_H0_05': p_val < 0.05,
        },
    }


def placebo_temporal_test(df: pd.DataFrame, n_iteraciones: int = 500, seed: int = 42) -> dict:
    """
    Placebo test temporal: randomiza las asignaciones de semana y re-estima el modelo.

    Construye una distribución nula de la elasticidad bajo la hipótesis de que
    la estructura temporal verdadera no aporta información distinta de una
    asignación aleatoria de semanas (es decir, que la elasticidad no depende
    de forma espuria de en qué semana ocurre cada observación).

    NOTA (fix): la versión anterior de esta función mezclaba la columna
    'week' pero la fórmula del modelo nunca incluía 'week' como regresor,
    por lo que el shuffle no tenía ningún efecto y las 500 iteraciones
    reestimaban el mismo modelo idéntico. Aquí 'week' se incorpora como
    efecto fijo (C(week)) tanto en el modelo real como en cada iteración
    placebo, para que el test sea válido. Por eficiencia, el efecto fijo de
    semana se implementa vía "demeaning" (teorema de Frisch-Waugh-Lovell) en
    vez de generar 145 columnas dummy con patsy en cada iteración: restar la
    media por grupo de semana de (y, X) es matemáticamente equivalente a
    incluir C(week) como regresores, pero ~50-70x más rápido.

    Retorna p-valor = proporción de estimaciones placebo <= estimación real.
    """
    rng = np.random.default_rng(seed)

    # --- Matriz de diseño fija (no depende de la asignación de semana) ---
    y = df['log_num_orders'].to_numpy(dtype=float)
    x_price = df['log_checkout_price'].to_numpy(dtype=float)
    controles = df[['emailer_for_promotion', 'homepage_featured', 'hubo_aumento']].to_numpy(dtype=float)
    cat_dummies = pd.get_dummies(df['category'], drop_first=True).to_numpy(dtype=float)
    center_dummies = pd.get_dummies(df['center_type'], drop_first=True).to_numpy(dtype=float)
    X0 = np.column_stack([x_price, controles, cat_dummies, center_dummies])
    YX = np.column_stack([y, X0])  # columna 0 = y, resto = X (log_price es la columna 1)

    week_codes_real, _ = pd.factorize(df['week'])
    n_weeks = week_codes_real.max() + 1

    def elasticidad_con_semana_fe(week_codes: np.ndarray) -> float:
        """Coeficiente de log_price incluyendo C(week) vía demeaning (FWL)."""
        sums = np.zeros((n_weeks, YX.shape[1]))
        np.add.at(sums, week_codes, YX)
        counts = np.bincount(week_codes, minlength=n_weeks).astype(float)
        means = sums / counts[:, None]
        YX_dm = YX - means[week_codes]
        coef, *_ = np.linalg.lstsq(YX_dm[:, 1:], YX_dm[:, 0], rcond=None)
        return float(coef[0])

    # Elasticidad real: con la estructura de semanas verdadera
    elasticidad_real = elasticidad_con_semana_fe(week_codes_real)

    # Distribución nula: semanas asignadas al azar en cada iteración
    elasticidades_placebo = np.empty(n_iteraciones, dtype=float)
    for i in range(n_iteraciones):
        semanas_shuffled = rng.permutation(week_codes_real)
        elasticidades_placebo[i] = elasticidad_con_semana_fe(semanas_shuffled)

    elasticidades_placebo = elasticidades_placebo[~np.isnan(elasticidades_placebo)]

    # p-valor: proporción de placebo <= real (one-sided, más negativo = más elástico)
    p_valor = float(np.mean(elasticidades_placebo <= elasticidad_real))

    return {
        'elasticidad_real': float(elasticidad_real),
        'elasticidades_placebo': elasticidades_placebo.tolist(),
        'media_placebo': float(np.mean(elasticidades_placebo)),
        'std_placebo': float(np.std(elasticidades_placebo)),
        'p_valor': p_valor,
        'n_iteraciones': n_iteraciones,
        'rechaza_H0_05': p_valor < 0.05,
        'nota': 'PLACEBO INVÁLIDO - solo baraja week manteniendo (P,Q) emparejados. No destruye relación precio-demanda. Ver placebo_temporal_valido() y placebo_precio_futuro().'
    }


def placebo_temporal_valido(df: pd.DataFrame, n_iteraciones: int = 500, seed: int = 42) -> dict:
    """
    Placebo temporal VÁLIDO: permuta precios DENTRO de cada entidad (center_id × meal_id),
    preservando la distribución marginal de precios pero rompiendo la alineación temporal
    con la demanda. Esto SÍ destruye la relación precio-demanda temporal.

    H0: la estructura temporal no importa (elasticidad con precios permutados = real)
    H1: la estructura temporal importa (elasticidad real ≠ placebo)
    """
    rng = np.random.default_rng(seed)

    # Preparar matriz de diseño fija (sin semana)
    y = df['log_num_orders'].to_numpy(dtype=float)
    controles = df[['emailer_for_promotion', 'homepage_featured', 'hubo_aumento']].to_numpy(dtype=float)
    cat_dummies = pd.get_dummies(df['category'], drop_first=True).to_numpy(dtype=float)
    center_dummies = pd.get_dummies(df['center_type'], drop_first=True).to_numpy(dtype=float)
    X_controles = np.column_stack([controles, cat_dummies, center_dummies])

    # Entidades para permutación dentro de entidad
    df_local = df.copy()
    df_local['entidad'] = df_local['center_id'].astype(str) + '_' + df_local['meal_id'].astype(str)
    ent_codes, _ = pd.factorize(df_local['entidad'])
    n_ent = int(ent_codes.max() + 1)
    price = df_local['log_checkout_price'].to_numpy(dtype=float)

    def elasticidad_con_precio(price_vec: np.ndarray) -> float:
        """Coeficiente de precio con controles y FE de entidad (demeaning)."""
        X = np.column_stack([price_vec, X_controles])
        YX = np.column_stack([y, X])

        # Demeaning por entidad (absorbe EntityEffects)
        sums_e = np.zeros((n_ent, YX.shape[1]))
        np.add.at(sums_e, ent_codes, YX)
        counts_e = np.bincount(ent_codes, minlength=n_ent).astype(float)
        means_e = sums_e / counts_e[:, None]
        YX_dm = YX - means_e[ent_codes]

        coef, *_ = np.linalg.lstsq(YX_dm[:, 1:], YX_dm[:, 0], rcond=None)
        return float(coef[0])

    # Elasticidad real: con precios originales
    elasticidad_real = elasticidad_con_precio(price)

    # Distribución nula: permutar precios DENTRO de cada entidad
    elasticidades_placebo = np.empty(n_iteraciones, dtype=float)
    for i in range(n_iteraciones):
        price_perm = price.copy()
        for e in range(n_ent):
            mask = (ent_codes == e)
            if mask.sum() > 1:
                price_perm[mask] = rng.permutation(price_perm[mask])
        elasticidades_placebo[i] = elasticidad_con_precio(price_perm)

    elasticidades_placebo = elasticidades_placebo[~np.isnan(elasticidades_placebo)]
    p_valor = float(np.mean(elasticidades_placebo <= elasticidad_real))

    return {
        'elasticidad_real': float(elasticidad_real),
        'elasticidades_placebo': elasticidades_placebo.tolist(),
        'media_placebo': float(np.mean(elasticidades_placebo)),
        'std_placebo': float(np.std(elasticidades_placebo)),
        'p_valor': p_valor,
        'n_iteraciones': n_iteraciones,
        'rechaza_H0_05': p_valor < 0.05,
        'tipo': 'permutación_precio_dentro_entidad',
        'nota': 'Placebo VÁLIDO - rompe alineación temporal precio-demanda dentro de cada entidad'
    }


def placebo_precio_futuro(df: pd.DataFrame) -> dict:
    """
    Placebo con precio futuro (lead): Q_it ~ P_i,t+1

    Un precio futuro no debería explicar causalmente demanda presente
    después de controlar adecuadamente. Si lo hace, sugiere confusión
    (p. ej., tendencias comunes, expectativas).
    """
    df_local = df.copy()
    df_local['entidad'] = df_local['center_id'].astype(str) + '_' + df_local['meal_id'].astype(str)

    # Crear lead de precio por entidad
    df_local = df_local.sort_values(['entidad', 'week'])
    df_local['log_checkout_price_lead'] = df_local.groupby('entidad')['log_checkout_price'].shift(-1)

    # Eliminar última semana de cada entidad (sin lead)
    df_lead = df_local.dropna(subset=['log_checkout_price_lead']).copy()

    if len(df_lead) < 100:
        return {'error': 'Insuficientes observaciones para lead'}

    # Modelo con precio futuro
    formula_lead = (
        'log_num_orders ~ log_checkout_price_lead '
        '+ emailer_for_promotion + homepage_featured + hubo_aumento '
        '+ C(category) + C(center_type)'
    )
    modelo_lead = smf.ols(formula_lead, data=df_lead).fit()

    # Modelo con precio contemporáneo (mismo dataset)
    formula_contemp = (
        'log_num_orders ~ log_checkout_price '
        '+ emailer_for_promotion + homepage_featured + hubo_aumento '
        '+ C(category) + C(center_type)'
    )
    modelo_contemp = smf.ols(formula_contemp, data=df_lead).fit()

    coef_lead = modelo_lead.params.get('log_checkout_price_lead', np.nan)
    coef_contemp = modelo_contemp.params.get('log_checkout_price', np.nan)

    return {
        'elasticidad_precio_futuro_lead': float(coef_lead) if not np.isnan(coef_lead) else None,
        'elasticidad_precio_contemporaneo': float(coef_contemp) if not np.isnan(coef_contemp) else None,
        'n_obs': len(df_lead),
        'nota': 'Placebo VÁLIDO - precio futuro no debe explicar demanda presente. Si coef_lead ≈ coef_contemp, hay confusión temporal.'
    }


def placebo_block_permutation(df: pd.DataFrame, block_size: int = 4, n_iteraciones: int = 500, seed: int = 42) -> dict:
    """
    Placebo por permutación de bloques temporales (block permutation).

    Permuta bloques de semanas consecutivas del PRECIO (variable independiente)
    preservando dependencia local dentro del bloque, mientras se mantiene
    la DEMANDA (variable dependiente) y controles en su orden temporal original.
    Esto rompe la alineación temporal precio-demanda a nivel de bloque,
    preservando la estructura de autocorrelación local del precio.

    H0: la estructura temporal por bloques no importa (coef. precio con bloques permutados = real)
    H1: la estructura temporal por bloques importa

    Especificación: OLS log-log con controles y dummies C(category) + C(center_type)
    (MISMA especificación que el modelo principal y placebo_precio_futuro, SIN week FE).
    """
    rng = np.random.default_rng(seed)

    weeks = sorted(df['week'].unique())
    n_weeks = len(weeks)
    n_blocks = n_weeks // block_size
    remainder_weeks = n_weeks % block_size

    if n_blocks < 2:
        return {'error': 'Pocas semanas para block permutation'}

    # Preparar datos ordenados por semana
    df_sorted = df.sort_values('week').reset_index(drop=True)
    week_codes, _ = pd.factorize(df_sorted['week'])
    n_weeks_codes = week_codes.max() + 1

    y = df_sorted['log_num_orders'].to_numpy(dtype=float)
    price = df_sorted['log_checkout_price'].to_numpy(dtype=float)
    controles = df_sorted[['emailer_for_promotion', 'homepage_featured', 'hubo_aumento']].to_numpy(dtype=float)
    cat_dummies = pd.get_dummies(df_sorted['category'], drop_first=True).to_numpy(dtype=float)
    center_dummies = pd.get_dummies(df_sorted['center_type'], drop_first=True).to_numpy(dtype=float)

    def elasticidad_sin_week_fe(price_vec: np.ndarray, y_vec: np.ndarray,
                                  controles_vec: np.ndarray,
                                  cat_dummies_vec: np.ndarray,
                                  center_dummies_vec: np.ndarray) -> float:
        """Coeficiente de price con dummies category + center_type (NO week FE).
        Misma especificación que modelo_log_log_controles y placebo_precio_futuro.
        """
        X = np.column_stack([price_vec, controles_vec, cat_dummies_vec, center_dummies_vec])
        # Añadir intercepto
        X = np.column_stack([np.ones(len(y_vec)), X])
        coef, *_ = np.linalg.lstsq(X, y_vec, rcond=None)
        return float(coef[1])  # índice 1 = price (0 = intercepto)

    # Elasticidad real: con precio original
    elasticidad_real = elasticidad_sin_week_fe(price, y, controles, cat_dummies, center_dummies)

    # Distribución nula: permutar bloques de PRECIO solamente
    # Mantener y, controles, dummies en orden original; solo permutar price
    elasticidades_placebo = np.empty(n_iteraciones, dtype=float)

    # Construir slices por bloque (incluir semanas restantes en el último bloque)
    block_slices = []
    for b in range(n_blocks):
        start_week = b * block_size
        # Último bloque incluye las semanas restantes
        end_week = min((b + 1) * block_size, n_weeks) if b < n_blocks - 1 else n_weeks
        mask = (week_codes >= start_week) & (week_codes < end_week)
        block_slices.append(mask)

    for i in range(n_iteraciones):
        # Permutar índices de bloques
        block_indices = list(range(n_blocks))
        rng.shuffle(block_indices)

        # Reconstruir price concatenando bloques en orden permutado
        price_parts = []
        for new_b in block_indices:
            mask = block_slices[new_b]
            price_parts.append(price[mask])
        price_perm = np.concatenate(price_parts)

        elasticidades_placebo[i] = elasticidad_sin_week_fe(
            price_perm, y, controles, cat_dummies, center_dummies
        )

    elasticidades_placebo = elasticidades_placebo[~np.isnan(elasticidades_placebo)]
    p_valor = float(np.mean(elasticidades_placebo <= elasticidad_real))

    return {
        'elasticidad_real': float(elasticidad_real),
        'elasticidades_placebo': elasticidades_placebo.tolist(),
        'media_placebo': float(np.mean(elasticidades_placebo)),
        'std_placebo': float(np.std(elasticidades_placebo)),
        'p_valor': p_valor,
        'n_iteraciones': n_iteraciones,
        'block_size': block_size,
        'rechaza_H0_05': p_valor < 0.05,
        'tipo': 'block_permutation',
        'nota': 'Placebo VÁLIDO - permuta bloques de precio preservando dependencia local, rompe alineación precio-demanda temporal. Especificación: OLS log-log con C(category)+C(center_type) (sin week FE), igual que modelo principal.'
    }
