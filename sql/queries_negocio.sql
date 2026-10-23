-- Vistas de negocio sobre la tabla demanda_limpia (SQLite).
-- Las ejecuta notebooks/06_sql_bi_extraccion.ipynb; cada vista alimenta un CSV de dashboard/data/.

CREATE VIEW IF NOT EXISTS v_demanda_por_categoria AS
SELECT category,
       COUNT(*)            AS n_obs,
       AVG(num_orders)     AS demanda_promedio,
       SUM(num_orders)     AS demanda_total,
       AVG(checkout_price) AS precio_promedio,
       AVG(base_price)     AS precio_base_promedio
FROM demanda_limpia
GROUP BY category;

CREATE VIEW IF NOT EXISTS v_efecto_promociones AS
SELECT emailer_for_promotion,
       homepage_featured,
       COUNT(*)            AS n_obs,
       AVG(num_orders)     AS demanda_promedio,
       AVG(checkout_price) AS precio_promedio
FROM demanda_limpia
GROUP BY emailer_for_promotion, homepage_featured;

CREATE VIEW IF NOT EXISTS v_tendencia_semanal AS
SELECT week,
       COUNT(*)            AS n_obs,
       AVG(num_orders)     AS demanda_promedio,
       SUM(num_orders)     AS demanda_total,
       AVG(checkout_price) AS precio_promedio
FROM demanda_limpia
GROUP BY week;

CREATE VIEW IF NOT EXISTS v_comparacion_centros AS
SELECT center_type,
       COUNT(*)                  AS n_obs,
       AVG(num_orders)           AS demanda_promedio,
       SUM(num_orders)           AS demanda_total,
       AVG(checkout_price)       AS precio_promedio,
       COUNT(DISTINCT center_id) AS n_centros
FROM demanda_limpia
GROUP BY center_type;
