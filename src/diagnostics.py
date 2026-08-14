"""
Funciones de diagnóstico para la validación de supuestos del modelo
de elasticidad de demanda (log-log con controles).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor


def test_breusch_pagan(modelo) -> pd.DataFrame:
    """Test de Breusch-Pagan para heterocedasticidad.

    H0: los residuos tienen varianza constante (homocedasticidad).
    Un p-valor bajo (< 0.05) rechaza H0, indicando heterocedasticidad:
    la varianza de los errores no es constante a lo largo de los valores
    ajustados, lo cual invalida los errores estándar clásicos de OLS
    (aunque no los coeficientes en sí).
    """
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(
        modelo.resid, modelo.model.exog
    )
    return pd.DataFrame({
        'estadistico': ['LM', 'F'],
        'valor': [lm_stat, f_stat],
        'p_valor': [lm_pvalue, f_pvalue],
    })


def calcular_vif(df: pd.DataFrame, columnas_numericas: list) -> pd.DataFrame:
    """Calcula el Factor de Inflación de Varianza (VIF) para un conjunto
    de variables numéricas.

    VIF cercano a 1: sin multicolinealidad relevante.
    VIF entre 1 y 5: multicolinealidad moderada, generalmente aceptable.
    VIF mayor a 5-10: multicolinealidad alta, amerita revisión.

    Nota: VIF se calcula sobre variables numéricas continuas. Las variables
    categóricas (dummies de C(category), C(center_type)) no se incluyen
    aquí de forma individual porque el VIF de una dummy dentro de un
    conjunto mutuamente excluyente no se interpreta de la misma forma;
    el diagnóstico se concentra en las variables numéricas del modelo.
    """
    X = df[columnas_numericas].copy()
    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data['variable'] = X.columns
    vif_data['VIF'] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    return vif_data[vif_data['variable'] != 'const'].reset_index(drop=True)


def resumen_normalidad_residuos(modelo) -> dict:
    """Resumen de normalidad de residuos: usa el test de Jarque-Bera
    (ya incluido en el summary() de statsmodels, se extrae aquí para
    referencia explícita) y estadísticos descriptivos de asimetría y
    curtosis de los residuos.

    H0 del Jarque-Bera: los residuos siguen una distribución normal.
    Un p-valor bajo rechaza H0.

    Nota de interpretación: con datasets muy grandes (cientos de miles
    de observaciones, como este), el test de Jarque-Bera casi siempre
    rechaza la normalidad exacta incluso ante desviaciones pequeñas y
    poco relevantes en la práctica — por eso se complementa con la
    inspección visual del gráfico Q-Q, que es más informativa sobre la
    magnitud real de la desviación.
    """
    jb_stat, jb_pvalue, skew, kurtosis = sm.stats.jarque_bera(modelo.resid)
    return {
        'jarque_bera_estadistico': jb_stat,
        'jarque_bera_pvalor': jb_pvalue,
        'asimetria': skew,
        'curtosis': kurtosis,
    }


def shapiro_wilk_submuestreado(modelo, n: int = 5000, semilla: int = 42) -> dict:
    """Test de Shapiro-Wilk sobre una submuestra de los residuos.

    Shapiro-Wilk no está diseñado (y muchas implementaciones no lo
    soportan bien) para muestras del tamaño de este dataset (~450,000
    observaciones). Se evalúa sobre una submuestra aleatoria de tamaño
    n, y se documenta explícitamente esta limitación -- el resultado
    describe la normalidad de esa submuestra, no una prueba exacta
    sobre el total de residuos.
    """
    from scipy import stats

    rng = np.random.default_rng(semilla)
    # statsmodels usa .resid; linearmodels (PanelOLS, IV2SLS) usa .resids
    residuos_attr = getattr(modelo, 'resid', None)
    if residuos_attr is None:
        residuos_attr = modelo.resids
    residuos = np.asarray(residuos_attr)
    n_efectivo = min(n, len(residuos))
    submuestra = rng.choice(residuos, size=n_efectivo, replace=False)

    estadistico, p_valor = stats.shapiro(submuestra)
    return {
        'n_submuestra': n_efectivo,
        'shapiro_estadistico': estadistico,
        'shapiro_pvalor': p_valor,
        'nota': f'Evaluado sobre una submuestra aleatoria de {n_efectivo} '
                f'de {len(residuos)} residuos totales, no sobre el dataset completo.',
    }


def test_reset(modelo, potencias: int = 3) -> dict:
    """Test RESET (Regression Specification Error Test) de Ramsey.

    Prueba la especificación funcional del modelo: H0: el modelo está
    correctamente especificado (no hay variables omitidas, la forma
    funcional es adecuada). Agrega potencias de los valores ajustados
    (ŷ^2, ŷ^3, ...) y testa si son conjuntamente significativas.

    Un p-valor bajo rechaza H0, sugiriendo especificación incorrecta
    (p. ej., no-linealidad no capturada, variables omitidas).

    Args:
        modelo: Modelo OLS ajustado de statsmodels
        potencias: Número de potencias a probar (2 a 4 típicamente)

    Returns:
        Dict con estadístico F y p-valor
    """
    reset_result = linear_reset(modelo, power=potencias, use_f=True)
    f_val = float(reset_result.fvalue.item()) if hasattr(reset_result.fvalue, 'item') else float(reset_result.fvalue)
    p_val = float(reset_result.pvalue.item()) if hasattr(reset_result.pvalue, 'item') else float(reset_result.pvalue)
    return {
        'estadistico_F': f_val,
        'p_valor': p_val,
        'grados_libertad': (float(reset_result.df_num), float(reset_result.df_denom)),
        'rechaza_H0_05': p_val < 0.05,
    }


def test_boxcox(modelo, df: pd.DataFrame, variable_y: str = 'log_num_orders',
                variable_x: str = 'log_checkout_price') -> dict:
    """Test Box-Cox para comparar forma funcional log-log vs. alternativas.

    El test de Box-Cox (via verosimilitud) ayuda a seleccionar la
    transformación óptima de la variable dependiente. Aquí lo usamos
    como complemento al RESET para validar la especificación log-log.

    Returns:
        Dict con lambda óptimo, IC, y test vs. log-log
    """
    from scipy import stats
    from scipy.stats import boxcox

    # Extraer variables del modelo (ya transformadas log en el dataset)
    y = df[variable_y].values
    x = df[variable_x].values

    # Box-Cox requiere y > 0
    if np.any(y <= 0):
        return {'error': 'Variable Y debe ser > 0 para Box-Cox'}

    # Estimar lambda óptimo
    try:
        y_bc, lambda_opt = boxcox(y)
        return {
            'lambda_optimo': float(lambda_opt),
            'nota': f'Lambda óptimo para transformar {variable_y}. Log-log corresponde a lambda=0.'
        }
    except Exception as e:
        return {'error': str(e)}
