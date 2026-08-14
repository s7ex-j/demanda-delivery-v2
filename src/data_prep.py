"""
Funciones de limpieza y preparación de datos para el análisis de
elasticidad de demanda en delivery de comida (v2).
"""

import numpy as np
import pandas as pd


def cargar_y_unir_datos(raw_dir: str) -> pd.DataFrame:
    """Carga las tablas train, meal_info y fulfilment_center_info,
    y las une en un solo DataFrame por meal_id y center_id.
    """
    train = pd.read_csv(f'{raw_dir}/train.csv')
    meal_info = pd.read_csv(f'{raw_dir}/meal_info.csv')
    centers = pd.read_csv(f'{raw_dir}/fulfilment_center_info.csv')

    n_rows_antes = train.shape[0]
    df = train.merge(meal_info, on='meal_id', how='left') \
              .merge(centers, on='center_id', how='left')

    assert df.shape[0] == n_rows_antes, 'El merge cambió el número de filas'
    return df


def construir_variables_precio(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las variables derivadas de precio.

    'base_price' es un precio de referencia, no un mínimo garantizado:
    en aproximadamente 25% de las observaciones, checkout_price supera a
    base_price (el precio subió respecto a su referencia). Por eso la
    variable se llama 'variacion_precio' y no 'descuento' — puede tomar
    valores negativos (el precio subió) o positivos (el precio bajó).
    """
    df = df.copy()
    df['variacion_precio'] = df['base_price'] - df['checkout_price']
    df['variacion_precio_pct'] = (df['variacion_precio'] / df['base_price']) * 100
    df['hubo_aumento'] = (df['variacion_precio'] < 0).astype(int)
    return df


def aplicar_transformacion_log(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega versiones logarítmicas de precio y cantidad,
    necesarias para el modelo de elasticidad log-log.

    Nota: se usa log(num_orders) en vez de log1p(num_orders) porque
    el mínimo observado es 13 (no hay ceros en la muestra), por lo que
    log() está bien definida y la interpretación del coeficiente es
    exactamente la elasticidad de num_orders (no log(1+num_orders)).
    """
    df = df.copy()
    df['log_checkout_price'] = np.log(df['checkout_price'])
    df['log_num_orders'] = np.log(df['num_orders'])
    return df


def pipeline_completo(raw_dir: str) -> pd.DataFrame:
    """Ejecuta el pipeline completo de carga, unión y preparación de variables."""
    df = cargar_y_unir_datos(raw_dir)
    df = construir_variables_precio(df)
    df = aplicar_transformacion_log(df)
    return df


if __name__ == '__main__':
    df = pipeline_completo('../data/raw')
    df.to_csv('../data/processed/demanda_limpia.csv', index=False)
    print(f'Dataset procesado: {df.shape[0]} filas, {df.shape[1]} columnas')
    print('Guardado en data/processed/demanda_limpia.csv')
