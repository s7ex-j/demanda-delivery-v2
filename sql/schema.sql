-- schema.sql
-- Esquema de la base de datos SQLite para el proyecto de elasticidad de demanda
-- Generado desde demanda_limpia.csv

-- Tabla principal: demanda_limpia
CREATE TABLE IF NOT EXISTS demanda_limpia (
    -- Identificadores
    week INTEGER NOT NULL,                    -- Semana (1-145)
    center_id INTEGER NOT NULL,               -- ID del centro de fulfillment
    meal_id INTEGER NOT NULL,                 -- ID de la comida

    -- Variables principales
    num_orders INTEGER NOT NULL,              -- Número de órdenes (variable dependiente)
    checkout_price REAL NOT NULL,             -- Precio final pagado por el cliente
    base_price REAL NOT NULL,                 -- Precio de referencia administrativo

    -- Promociones (binarias)
    emailer_for_promotion INTEGER NOT NULL,   -- 1 = promocionado por email
    homepage_featured INTEGER NOT NULL,       -- 1 = destacado en homepage

    -- Atributos categóricos
    category TEXT NOT NULL,                   -- Categoría de la comida
    cuisine TEXT NOT NULL,                    -- Tipo de cocina
    city_code INTEGER NOT NULL,               -- Código de ciudad
    region_code INTEGER NOT NULL,             -- Código de región
    center_type TEXT NOT NULL,                -- Tipo de centro (TYPE_A, TYPE_B, TYPE_C)

    -- Variables derivadas (Fase 2)
    variacion_precio REAL,                    -- checkout_price - base_price
    variacion_precio_pct REAL,                -- (checkout_price - base_price) / base_price * 100
    hubo_aumento INTEGER,                     -- 1 si checkout_price > base_price, 0 otherwise

    -- Variables transformadas para modelado (Fase 3)
    log_num_orders REAL,                      -- LOG(1 + num_orders)
    log_checkout_price REAL,                  -- LOG(checkout_price)
    log_base_price REAL,                      -- LOG(base_price)

    -- Instrumentos (Fase 3c)
    precio_prom_otros_centros REAL,           -- Precio promedio leave-one-out
    log_precio_prom_otros_centros REAL        -- LOG(precio_prom_otros_centros)
);

-- Índices para performance en queries de negocio
CREATE INDEX IF NOT EXISTS idx_week ON demanda_limpia(week);
CREATE INDEX IF NOT EXISTS idx_center_id ON demanda_limpia(center_id);
CREATE INDEX IF NOT EXISTS idx_meal_id ON demanda_limpia(meal_id);
CREATE INDEX IF NOT EXISTS idx_category ON demanda_limpia(category);
CREATE INDEX IF NOT EXISTS idx_center_type ON demanda_limpia(center_type);
CREATE INDEX IF NOT EXISTS idx_center_meal ON demanda_limpia(center_id, meal_id);
CREATE INDEX IF NOT EXISTS idx_category_week ON demanda_limpia(category, week);
CREATE INDEX IF NOT EXISTS idx_promo ON demanda_limpia(emailer_for_promotion, homepage_featured);

-- Vistas principales (también definidas en queries_negocio.sql)

-- Vista: métricas por categoría y semana
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

-- Vista: resumen por comida
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

-- Metadata de la base de datos
CREATE TABLE IF NOT EXISTS db_metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT OR REPLACE INTO db_metadata (key, value) VALUES
    ('project', 'demanda-delivery-v2'),
    ('description', 'Análisis de Elasticidad de Demanda en Delivery de Comida'),
    ('source_data', 'Kaggle Food Demand Forecasting (Genpact / Analytics Vidhya)'),
    ('n_rows', '456548'),
    ('n_weeks', '145'),
    ('n_centers', '55'),
    ('n_meals', '116'),
    ('n_categories', '14'),
    ('elasticity_ols_controls', '-0.7513'),
    ('elasticity_panel_fe', '-1.5755'),
    ('elasticity_2sls', '-0.9487'),
    ('created_at', datetime('now'));