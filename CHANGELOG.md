# Changelog

## 3.3.0 — 2026-10-21 · Primera versión pública

Auditoría completa previa a la publicación. Toda cifra del informe se recalculó desde el código.

### Cifras corregidas en el informe
- Tablas de diagnóstico de supuestos (Breusch-Pagan, Jarque-Bera, Shapiro, Durbin-Watson) y
  tabla de efectos fijos (R², poolability F = 411,15) alineadas con la salida actual del código.
- Semi-log: IC de la elasticidad equivalente corregido a [−1,086; −1,062] (antes no contenía la
  estimación puntual −1,0742).
- Robustez: submuestras, recorte de colas, EE HC3 0,0054 y por centro 0,0424, ratios 7,87 / 7,99,
  Wald por categoría 519,13 y elasticidades por categoría y por semana recalculadas.
- Bootstrap i.i.d. del modelo base (Tabla 11, Figura 8): ahora usa la misma función que el resto
  del código (`bootstrap_elasticidad`, 1 000 réplicas, semilla 42): media −0,9514, DE 0,0034.
- Regresión cuantil: los IC bootstrap de versiones anteriores no eran reproducibles (el archivo
  de resultados no se conservó) y se reemplazaron por IC analíticos. Las estimaciones puntuales
  no cambian.
- Bootstrap por centro del panel (B = 200) marcado como referencia histórica no reproducible; el
  pairs bootstrap (1 000 réplicas) es el que se reporta.
- Literatura: Bumagat (URL y rango), Andreyeva (−0,81), Christensen & Osman (−0,84) y la fila de
  este estudio corregidas en la tabla comparativa y la Figura 9; se retiró una cifra de
  Christensen (2023) que no figura en la fuente.

### Repositorio
- Datos excluidos; `scripts/download_data.py` los descarga de Kaggle y verifica SHA-256.
- Tests que dependen de los datos se omiten automáticamente si no están presentes.
- Dependencias fijadas en `requirements.txt`; `pyproject.toml` a 3.3.0, Python ≥ 3.11.
- Notebooks limpiados: sin celdas de guardado de versiones antiguas ni capturas de salida,
  valores obsoletos reemplazados, re-ejecutados de principio a fin.
- Scripts de figuras: uno por figura, todos escriben en `Informe/figuras/`; duplicados eliminados.
- Nuevo `sql/queries_negocio.sql` (antes el notebook 06 lo leía pero no existía).
- Dashboard: se retiraron las "recomendaciones de pricing", que combinaban elasticidades de
  modelos distintos como si fueran causales; la columna `n_ordenes` (que contaba filas, no
  pedidos) pasa a llamarse `n_obs`.
- Material de auditorías internas retirado del repositorio público.

## 3.2.5 — 2026-10-21

Corrige los hallazgos de la auditoría externa de 3.2.4.
- Especificación IV armonizada con los controles del OLS: −0,8414 (antes −0,9604, sin controles).
- Inferencia IV agrupada por centro (EE 0,050); la diferencia IV–OLS deja de presentarse como
  significativa.
- Hansen J recalculado: 33,28 con ponderación heterocedástica y 0,92 (p = 0,34) con ponderación
  agrupada.
- R² parcial de Shea 0,9226; valor crítico de Stock-Yogo corregido a 19,93.
- Contraste de no linealidad RCS sustituido por un F anidado (3 607,28); el anterior contrastaba
  una hipótesis distinta.
- Función de control con coeficientes puntuales y banda bootstrap (5 000 réplicas guardadas):
  −1,5041 en el precio medio, IC [−1,630; −1,370].
- Nuevo wild cluster bootstrap del IV con controles (2 000 réplicas guardadas).

## Versiones anteriores (3.0 – 3.2.4)

Desarrollo de las especificaciones de panel, IV, RCS, regresión cuantil, función de control y
placebos, más varias rondas de corrección de coherencia entre informe y código. La v1 del
estudio (encuesta propia) es un trabajo distinto y sus datos no se reutilizan.
