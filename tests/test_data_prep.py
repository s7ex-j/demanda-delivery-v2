"""
Tests unitarios para src/data_prep.py
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_prep import (
    cargar_y_unir_datos,
    construir_variables_precio,
    aplicar_transformacion_log,
    pipeline_completo,
)


class TestCargarYUnirDatos:
    """Tests para la carga y unión de datos."""

    def test_merge_preserva_filas(self):
        """El merge no debe duplicar ni perder filas del train original."""
        # Crear datos de prueba mínimos
        train = pd.DataFrame({
            'meal_id': [1, 2, 3],
            'center_id': [10, 20, 30],
            'num_orders': [100, 200, 300],
            'checkout_price': [10.0, 15.0, 20.0],
            'base_price': [12.0, 14.0, 18.0],
        })
        meal_info = pd.DataFrame({
            'meal_id': [1, 2, 3],
            'category': ['A', 'B', 'C'],
            'cuisine': ['X', 'Y', 'Z'],
        })
        centers = pd.DataFrame({
            'center_id': [10, 20, 30],
            'center_type': ['TYPE_A', 'TYPE_B', 'TYPE_C'],
            'city_code': [100, 200, 300],
        })

        # Guardar archivos temporales
        tmpdir = 'tests/tmp'
        os.makedirs(tmpdir, exist_ok=True)
        train.to_csv(f'{tmpdir}/train.csv', index=False)
        meal_info.to_csv(f'{tmpdir}/meal_info.csv', index=False)
        centers.to_csv(f'{tmpdir}/fulfilment_center_info.csv', index=False)

        df = cargar_y_unir_datos(tmpdir)

        assert df.shape[0] == 3, "El número de filas debe mantenerse igual"
        assert 'category' in df.columns
        assert 'center_type' in df.columns

        # Limpiar
        import shutil
        shutil.rmtree(tmpdir)


class TestConstruirVariablesPrecio:
    """Tests para variables derivadas de precio."""

    def test_variacion_precio(self):
        df = pd.DataFrame({
            'base_price': [100.0, 100.0, 100.0],
            'checkout_price': [90.0, 100.0, 110.0],
        })
        df = construir_variables_precio(df)

        # base - checkout
        assert df['variacion_precio'].tolist() == [10.0, 0.0, -10.0]
        # % variación
        assert df['variacion_precio_pct'].tolist() == [10.0, 0.0, -10.0]
        # hubo_aumento: 1 si checkout > base (precio subió)
        assert df['hubo_aumento'].tolist() == [0, 0, 1]

    def test_variacion_precio_pct_formula(self):
        df = pd.DataFrame({
            'base_price': [200.0],
            'checkout_price': [180.0],
        })
        df = construir_variables_precio(df)
        # (200 - 180) / 200 * 100 = 10%
        assert df['variacion_precio_pct'].iloc[0] == 10.0


class TestAplicarTransformacionLog:
    """Tests para transformaciones logarítmicas."""

    def test_log_transform(self):
        df = pd.DataFrame({
            'checkout_price': [100.0, 200.0],
            'num_orders': [10, 20],
        })
        df = aplicar_transformacion_log(df)

        assert 'log_checkout_price' in df.columns
        assert 'log_num_orders' in df.columns
        assert np.isclose(df['log_checkout_price'].iloc[0], np.log(100.0))
        assert np.isclose(df['log_num_orders'].iloc[0], np.log(10))

    def test_log_num_orders_usa_log(self):
        """log_num_orders debe usar log (no log1p) porque min(num_orders) >= 13."""
        df = pd.DataFrame({
            'checkout_price': [100.0],
            'num_orders': [10],
        })
        df = aplicar_transformacion_log(df)
        assert np.isclose(df['log_num_orders'].iloc[0], np.log(10))


class TestPipelineCompleto:
    """Test de integración del pipeline completo."""

    def test_pipeline_completo_output_shape(self):
        # Crear datos de prueba
        train = pd.DataFrame({
            'meal_id': [1, 2],
            'center_id': [10, 20],
            'num_orders': [100, 200],
            'checkout_price': [10.0, 15.0],
            'base_price': [12.0, 14.0],
            'emailer_for_promotion': [0, 1],
            'homepage_featured': [1, 0],
        })
        meal_info = pd.DataFrame({
            'meal_id': [1, 2],
            'category': ['A', 'B'],
            'cuisine': ['X', 'Y'],
        })
        centers = pd.DataFrame({
            'center_id': [10, 20],
            'center_type': ['TYPE_A', 'TYPE_B'],
            'city_code': [100, 200],
            'region_code': [10, 20],
        })

        tmpdir = 'tests/tmp'
        os.makedirs(tmpdir, exist_ok=True)
        train.to_csv(f'{tmpdir}/train.csv', index=False)
        meal_info.to_csv(f'{tmpdir}/meal_info.csv', index=False)
        centers.to_csv(f'{tmpdir}/fulfilment_center_info.csv', index=False)

        df = pipeline_completo(tmpdir)

        # Verificar columnas clave
        expected_cols = [
            'log_checkout_price', 'log_num_orders',
            'variacion_precio', 'variacion_precio_pct', 'hubo_aumento',
            'category', 'center_type', 'emailer_for_promotion', 'homepage_featured'
        ]
        for col in expected_cols:
            assert col in df.columns, f"Falta columna: {col}"

        assert df.shape[0] == 2

        import shutil
        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])