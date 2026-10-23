# Datos para dashboard

Ocho CSV listos para Power BI, Looker Studio o cualquier herramienta de BI. Se generan desde
`data/processed/demanda_limpia.csv` (456 548 filas; 145 semanas × 77 centros × 51 comidas).

| Archivo | Lo genera | Filas |
|---|---|---|
| `demanda_por_categoria.csv` | notebook 06 (vista `v_demanda_por_categoria`) | 14 |
| `efecto_promociones.csv` | notebook 06 (vista `v_efecto_promociones`) | 4 |
| `tendencia_semanal.csv` | notebook 06 (vista `v_tendencia_semanal`) | 145 |
| `comparacion_centros.csv` | notebook 06 (vista `v_comparacion_centros`) | 3 |
| `elasticidad_por_categoria.csv` | notebook 06 (SQL + correlación en pandas) | 14 |
| `kpis_ejecutivos.csv` | notebook 06 | 9 |
| `elasticidad_por_categoria_detallada.csv` | `scripts/analysis/generar_dashboard_modelos.py` | 14 |
| `elasticidad_rcs_por_precio.csv` | `scripts/analysis/generar_dashboard_modelos.py` | 22 |

Las vistas SQL están en [`sql/queries_negocio.sql`](../sql/queries_negocio.sql); el notebook 06
crea la base SQLite `sql/demanda_delivery.db` (no versionada) y exporta los CSV.

## Columnas

**Comunes a las tablas agregadas**

| Columna | Definición |
|---|---|
| `n_obs` | Número de filas semana × centro × comida en el grupo (no es número de pedidos) |
| `demanda_promedio` | Promedio de `num_orders` por fila |
| `demanda_total` | Suma de `num_orders` (pedidos) |
| `precio_promedio` | Promedio de `checkout_price` |
| `precio_base_promedio` | Promedio de `base_price` (solo por categoría) |

Los precios están en las **unidades monetarias anónimas** del conjunto de Kaggle; el dataset no
indica moneda ni país.

**Tablas específicas**

- `efecto_promociones.csv`: matriz 2 × 2 de `emailer_for_promotion` × `homepage_featured`.
- `comparacion_centros.csv`: por `center_type`, con `n_centros`.
- `elasticidad_por_categoria.csv`: `n`, medias de log-demanda y log-precio y `corr_log_log`, la
  correlación simple entre ambos logaritmos dentro de la categoría. Es un descriptivo, **no una
  elasticidad**.
- `elasticidad_por_categoria_detallada.csv`: elasticidad por categoría del modelo OLS log-log con
  interacción precio × categoría y controles (sección de heterogeneidad del informe).
- `elasticidad_rcs_por_precio.csv`: elasticidad puntual de la RCS canónica (nudos P5/P35/P65/P95)
  entre 125 y 650 unidades de precio, con coeficientes puntuales. Las bandas bootstrap están en el
  informe.
- `kpis_ejecutivos.csv`: pares `metrica`/`valor` (totales, conteos y las tres elasticidades
  principales del informe).

## Cómo leer las elasticidades

Son asociaciones estimadas con datos observacionales, no efectos causales de cambiar un precio.
Elasticidades de modelos distintos miden cosas distintas (promedio de la muestra, variación
*within*, elasticidad local por nivel de precio) y no deben combinarse en una sola regla de
precios. Por eso el repositorio no incluye "recomendaciones de pricing".
