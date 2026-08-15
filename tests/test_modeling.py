"""
Tests unitarios para src/modeling.py
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from modeling import (
    modelo_log_log_base,
    modelo_log_log_controles,
    modelo_semilog,
    bootstrap_elasticidad,
    tabla_comparacion_modelos,
)


def crear_dataset_prueba(n=1000, semilla=42):
    """Crea un dataset sintético para tests."""
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame({
        'checkout_price': rng.lognormal(mean=5.5, sigma=0.3, size=n),
        'base_price': rng.lognormal(mean=5.5, sigma=0.3, size=n),
        'emailer_for_promotion': rng.integers(0, 2, size=n),
        'homepage_featured': rng.integers(0, 2, size=n),
        'category': rng.choice(['A', 'B', 'C', 'D'], size=n),
        'center_type': rng.choice(['TYPE_A', 'TYPE_B', 'TYPE_C'], size=n),
    })
    # Generar num_orders con elasticidad conocida ~ -0.8
    log_price = np.log(df['checkout_price'])
    log_orders = 8.0 - 0.8 * log_price + rng.normal(0, 0.5, size=n)
    df['num_orders'] = np.exp(log_orders) - 1
    df['log_checkout_price'] = log_price
    df['log_num_orders'] = np.log1p(df['num_orders'])
    df['variacion_precio'] = df['base_price'] - df['checkout_price']
    df['variacion_precio_pct'] = (df['variacion_precio'] / df['base_price']) * 100
    df['hubo_aumento'] = (df['variacion_precio'] < 0).astype(int)
    return df


class TestModeloLogLogBase:
    """Tests para el modelo log-log base."""

    def test_modelo_ajusta(self):
        df = crear_dataset_prueba()
        modelo = modelo_log_log_base(df)
        assert hasattr(modelo, 'params')
        assert 'log_checkout_price' in modelo.params.index

    def test_elasticidad_negativa(self):
        df = crear_dataset_prueba()
        modelo = modelo_log_log_base(df)
        elasticidad = modelo.params['log_checkout_price']
        assert elasticidad < 0, "La elasticidad debe ser negativa"

    def test_significancia(self):
        df = crear_dataset_prueba()
        modelo = modelo_log_log_base(df)
        p_valor = modelo.pvalues['log_checkout_price']
        assert p_valor < 0.05, "El coeficiente debe ser significativo"


class TestModeloLogLogControles:
    """Tests para el modelo log-log con controles."""

    def test_modelo_ajusta(self):
        df = crear_dataset_prueba()
        modelo = modelo_log_log_controles(df)
        assert hasattr(modelo, 'params')

    def test_controles_presentes(self):
        df = crear_dataset_prueba()
        modelo = modelo_log_log_controles(df)
        param_names = list(modelo.params.index)
        # Debe tener dummies de categoría y centro
        assert any('category' in p for p in param_names)
        assert any('center_type' in p for p in param_names)
        assert 'emailer_for_promotion' in param_names
        assert 'homepage_featured' in param_names


class TestModeloSemiLog:
    """Tests para modelo semi-log."""

    def test_modelo_ajusta(self):
        df = crear_dataset_prueba()
        modelo = modelo_semilog(df)
        assert hasattr(modelo, 'params')
        assert 'checkout_price' in modelo.params.index


class TestBootstrapElasticidad:
    """Tests para bootstrap de elasticidad."""

    def test_bootstrap_ejecuta(self):
        df = crear_dataset_prueba(n=500)
        resultado = bootstrap_elasticidad(df, n_iteraciones=50, semilla=123)
        assert 'elasticidad_media' in resultado
        assert 'ic_95_inferior' in resultado
        assert 'ic_95_superior' in resultado
        assert resultado['ic_95_inferior'] < resultado['elasticidad_media'] < resultado['ic_95_superior']

    def test_bootstrap_reproducible(self):
        df = crear_dataset_prueba(n=500)
        r1 = bootstrap_elasticidad(df, n_iteraciones=50, semilla=42)
        r2 = bootstrap_elasticidad(df, n_iteraciones=50, semilla=42)
        assert np.isclose(r1['elasticidad_media'], r2['elasticidad_media'])


class TestTablaComparacionModelos:
    """Tests para tabla de comparación."""

    def test_genera_dataframe(self):
        df = crear_dataset_prueba(n=500)
        modelos = {
            'base': modelo_log_log_base(df),
            'controles': modelo_log_log_controles(df),
        }
        tabla = tabla_comparacion_modelos(modelos)
        assert isinstance(tabla, pd.DataFrame)
        assert len(tabla) == 2
        assert 'aic' in tabla.columns
        assert 'bic' in tabla.columns
        assert 'r2_ajustado' in tabla.columns
        assert 'n_observaciones' in tabla.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])