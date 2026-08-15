-- queries_negocio.sql
-- Queries de negocio para el proyecto de elasticidad de demanda en delivery
-- Se ejecutan desde notebooks/06_sql_bi_extraccion.ipynb
-- Base de datos: sql/demanda_delivery.db

-- ============================================================
-- VISTAS REUTILIZABLES
-- ============================================================

-- Vista base: métricas por categoría y semana
CREATE VIEW IF NOT EXISTS vw_categoria_semana AS
SELECT
    category,
    week,
    COUNT(*) as n_observaciones,
    SUM(num_orders) as demanda_total,
    AVG(num_orders) as demanda_promedio,
    AVG(checkout_price) as precio_promedio,
    AVG(base_price) as precio_base_promedio,
    SUM(CASE WHEN emailer_for_promotion = 1 THEN 1 ELSE 0 END) as n_emailer,
    SUM(CASE WHEN homepage_featured = 1 THEN 1 ELSE 0 END) as n_homepage
FROM demanda_limpia
GROUP BY category, week;

-- Vista: métricas por centro y semana
CREATE VIEW IF NOT EXISTS vw_centro_semana AS
SELECT
    center_id,
    center_type,
    week,
    COUNT(*) as n_observaciones,
    SUM(num_orders) as demanda_total,
    AVG(num_orders) as demanda_promedio,
    AVG(checkout_price) as precio_promedio
FROM demanda_limpia
GROUP BY center_id, center_type, week;

-- Vista: efecto de promociones por categoría
CREATE VIEW IF NOT EXISTS vw_promo_categoria AS
SELECT
    category,
    emailer_for_promotion,
    homepage_featured,
    COUNT(*) as n_observaciones,
    AVG(num_orders) as demanda_promedio,
    AVG(checkout_price) as precio_promedio
FROM demanda_limpia
GROUP BY category, emailer_for_promotion, homepage_featured;

-- Vista: resumen por comida (meal_id)
CREATE VIEW IF NOT EXISTS vw_comida_resumen AS
SELECT
    meal_id,
    category,
    cuisine,
    COUNT(*) as n_semanas,
    SUM(num_orders) as demanda_total,
    AVG(num_orders) as demanda_promedio,
    AVG(checkout_price) as precio_promedio,
    AVG(base_price) as precio_base_promedio,
    MIN(week) as primera_semana,
    MAX(week) as ultima_semana
FROM demanda_limpia
GROUP BY meal_id, category, cuisine;

-- ============================================================
-- CONSULTAS DE NEGOCIO PARA DASHBOARD
-- ============================================================

-- 1. DEMANDA POR CATEGORÍA (Top 10)
SELECT
    category,
    SUM(num_orders) as demanda_total,
    AVG(num_orders) as demanda_promedio_semanal,
    AVG(checkout_price) as precio_promedio,
    COUNT(DISTINCT meal_id) as n_comidas,
    COUNT(DISTINCT center_id) as n_centros
FROM demanda_limpia
GROUP BY category
ORDER BY demanda_total DESC
LIMIT 10;

-- 2. EFECTO PROMOCIONES (matriz 2x2)
SELECT
    CASE emailer_for_promotion WHEN 1 THEN 'Con Emailer' ELSE 'Sin Emailer' END as emailer,
    CASE homepage_featured WHEN 1 THEN 'En Homepage' ELSE 'No en Homepage' END as homepage,
    COUNT(*) as n_observaciones,
    ROUND(AVG(num_orders), 2) as demanda_promedio,
    ROUND(AVG(checkout_price), 2) as precio_promedio,
    ROUND(AVG(num_orders) * 100.0 / (SELECT AVG(num_orders) FROM demanda_limpia), 1) as indice_demanda_vs_promedio
FROM demanda_limpia
GROUP BY emailer_for_promotion, homepage_featured
ORDER BY emailer_for_promotion DESC, homepage_featured DESC;

-- 3. TENDENCIA SEMANAL (serie completa)
SELECT
    week,
    SUM(num_orders) as demanda_total_semanal,
    AVG(num_orders) as demanda_promedio_semanal,
    AVG(checkout_price) as precio_promedio_semanal,
    COUNT(DISTINCT center_id) as centros_activos,
    COUNT(DISTINCT meal_id) as comidas_activas
FROM demanda_limpia
GROUP BY week
ORDER BY week;

-- 4. COMPARACIÓN POR TIPO DE CENTRO
SELECT
    center_type,
    COUNT(*) as n_observaciones,
    SUM(num_orders) as demanda_total,
    AVG(num_orders) as demanda_promedio,
    AVG(checkout_price) as precio_promedio,
    COUNT(DISTINCT center_id) as n_centros_unicos,
    ROUND(SUM(num_orders) * 100.0 / (SELECT SUM(num_orders) FROM demanda_limpia), 1) as pct_demanda_total
FROM demanda_limpia
GROUP BY center_type
ORDER BY demanda_total DESC;

-- 5. ELASTICIDAD IMPLÍCITA POR CATEGORÍA (correlación log-log)
SELECT
    category,
    COUNT(*) as n,
    ROUND(CORR(LOG(num_orders + 1), LOG(checkout_price)), 4) as corr_log_log,
    ROUND(AVG(LOG(num_orders + 1)), 4) as log_demanda_prom,
    ROUND(AVG(LOG(checkout_price)), 4) as log_precio_prom
FROM demanda_limpia
GROUP BY category
HAVING COUNT(*) > 100
ORDER BY corr_log_log;

-- 6. TOP 20 COMIDAS POR DEMANDA TOTAL
SELECT
    meal_id,
    category,
    cuisine,
    SUM(num_orders) as demanda_total,
    AVG(num_orders) as demanda_promedio,
    AVG(checkout_price) as precio_promedio,
    COUNT(DISTINCT center_id) as n_centros,
    COUNT(DISTINCT week) as n_semanas
FROM demanda_limpia
GROUP BY meal_id, category, cuisine
ORDER BY demanda_total DESC
LIMIT 20;

-- 7. VARIACIÓN DE PRECIO (checkout vs base) POR CATEGORÍA
SELECT
    category,
    AVG(checkout_price - base_price) as diff_promedio,
    AVG((checkout_price - base_price) / base_price * 100) as pct_variacion_promedio,
    SUM(CASE WHEN checkout_price > base_price THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_con_aumento,
    SUM(CASE WHEN checkout_price < base_price THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_con_descuento,
    SUM(CASE WHEN checkout_price = base_price THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_sin_cambio
FROM demanda_limpia
GROUP BY category
ORDER BY pct_con_aumento DESC;

-- 8. KPIs EJECUTIVOS (resumen único)
SELECT
    'Demanda Total' as kpi,
    CAST(SUM(num_orders) AS INTEGER) as valor,
    'unidades' as unidad
FROM demanda_limpia
UNION ALL
SELECT
    'Precio Promedio',
    ROUND(AVG(checkout_price), 2),
    'soles'
FROM demanda_limpia
UNION ALL
SELECT
    'Elasticidad Principal (OLS controles)',
    -0.7513,
    'coeficiente'
UNION ALL
SELECT
    'Elasticidad Panel FE',
    -1.5755,
    'coeficiente'
UNION ALL
SELECT
    'Elasticidad 2SLS (Causal)',
    -0.9487,
    'coeficiente'
UNION ALL
SELECT
    'N Observaciones',
    CAST(COUNT(*) AS INTEGER),
    'filas'
FROM demanda_limpia
UNION ALL
SELECT
    'N Centros Únicos',
    CAST(COUNT(DISTINCT center_id) AS INTEGER),
    'centros'
FROM demanda_limpia
UNION ALL
SELECT
    'N Comidas Únicas',
    CAST(COUNT(DISTINCT meal_id) AS INTEGER),
    'comidas'
FROM demanda_limpia
UNION ALL
SELECT
    'Semanas Cubiertas',
    CAST(COUNT(DISTINCT week) AS INTEGER),
    'semanas'
FROM demanda_limpia
UNION ALL
SELECT
    'Categorías',
    CAST(COUNT(DISTINCT category) AS INTEGER),
    'categorías'
FROM demanda_limpia;

-- 9. IMPACTO EN INGRESO: SIMULACIÓN DE CAMBIO DE PRECIO 5%
-- Usando elasticidad principal -0.7513
SELECT
    'Escenario Base' as escenario,
    SUM(num_orders) as demanda_total,
    AVG(checkout_price) as precio_promedio,
    SUM(num_orders) * AVG(checkout_price) as ingreso_total
FROM demanda_limpia
UNION ALL
SELECT
    'Precio +5% (elast. -0.75)',
    SUM(num_orders) * (1 - 0.7513 * 0.05),
    AVG(checkout_price) * 1.05,
    SUM(num_orders) * (1 - 0.7513 * 0.05) * AVG(checkout_price) * 1.05
FROM demanda_limpia
UNION ALL
SELECT
    'Precio -5% (elast. -0.75)',
    SUM(num_orders) * (1 + 0.7513 * 0.05),
    AVG(checkout_price) * 0.95,
    SUM(num_orders) * (1 + 0.7513 * 0.05) * AVG(checkout_price) * 0.95
FROM demanda_limpia;