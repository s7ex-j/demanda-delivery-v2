# Análisis de Elasticidad de Demanda en Delivery de Comida — v2

**Autor:** Jharol André Vilca Ramos
**Relación con trabajo previo:** evolución metodológica del estudio "Análisis de la Demanda del Servicio de Delivery en Estudiantes Universitarios" (Zenodo, oct. 2025)
**Última actualización:** 14 agosto 2026 — Panel FE bootstrap y Quantile bootstrap integrados desde VM; v_hat bootstrap y wild bootstrap en ejecución en VM; scripts VM creados; notebooks/outputs consolidados; comparacion_modelos.csv actualizado con todos los modelos

---

## 1. Resumen ejecutivo

Este proyecto es una segunda iteración, técnicamente independiente, del estudio de demanda de delivery realizado en el curso de Economía. La v1 usó una encuesta propia analizada en Excel. La v2 usa un dataset transaccional real de una empresa de delivery de comidas (Kaggle / Genpact — Analytics Vidhya), analizado con un pipeline completo en Python, con validación econométrica rigurosa y comparación con literatura verificada.

**No es una corrección de los datos de v1.** Es un estudio nuevo, con datos distintos, que aplica lo aprendido sobre diseño de análisis, limpieza de datos y validación estadística.

**Decisiones confirmadas (actualizado):**
- Dataset completo: **456,548 filas** (todos los centros, comidas y semanas) — corrige el N de 273,928 que figuraba en la versión anterior de este documento, que era una estimación previa a la carga real de datos
- El proyecto se publica en Zenodo al finalizar, con DOI propio, igual que Parque Kennedy
- Todo el desarrollo se hace en VSCode local, con Claude como apoyo de código/planificación/validación
- **Alcance ampliado** (decisión del 2 agosto 2026): sin límite estricto de 1 mes, se prioriza profundidad sobre velocidad — se incorpora rigor econométrico real (panel, variables instrumentales) y una capa de SQL + BI como traducción a negocio del mismo análisis

### Cómo se relaciona (y cómo se referencia) con v1

En el README del repo, sección "Contexto y motivación": se menciona que el proyecto nace de un primer estudio (v1, 2025), se listan honestamente sus limitaciones metodológicas (Excel sin diagnóstico de supuestos, problemas de diseño muestral), se aclara que v2 no reutiliza esos datos, y se explicita que los resultados no son comparables v1-vs-v2 en sentido estricto — son evolución de capacidad técnica. Tono profesional, sin exponer de más a las coautoras de v1.

---

## 2. Objetivos

**Objetivo general:** estimar la elasticidad precio-demanda en un contexto real de delivery de comidas, con identificación causal lo más creíble posible dado los datos disponibles, y traducir los hallazgos tanto a un informe académico como a un dashboard de negocio.

**Objetivos específicos:**
1. Construir un pipeline reproducible de limpieza y preparación de datos transaccionales
2. Estimar y comparar múltiples especificaciones de modelo de demanda (OLS simple, con controles, con efectos fijos de dos vías)
3. Validar rigurosamente los supuestos estadísticos de cada especificación
4. Implementar una estrategia de variables instrumentales real (no simulada) para atenuar el problema de endogeneidad precio-demanda, con sus propios diagnósticos (relevancia, sobre-identificación)
5. Ubicar el resultado dentro de una tabla comparativa de literatura verificada fuente por fuente
6. Traducir el análisis a una capa de negocio: SQL sobre el dataset enriquecido + dashboard de BI
7. Documentar y empaquetar el proyecto a nivel profesional (reproducibilidad, licencias, citación)

---

## 3. Entorno de trabajo (VSCode)

Todo el desarrollo ocurre en VSCode local. Flujo: Claude genera código por partes → se corre localmente → se comparten errores/outputs → se itera. Extensiones en uso: Jupyter, Pylance, Python Data Science, Data Wrangler, LaTeX Workshop.

**Nota de rendimiento:** 456,548 filas es perfectamente manejable en pandas local. Cuidado en gráficos exploratorios (muestrear para dispersión), dataset completo para el modelado.

**Sobre el informe LaTeX (actualizado):** se migró de `jaes.cls` (plantilla AES adaptada a APA7, dos columnas) a la clase oficial **`apa7.cls`** de CTAN, en una columna (`\documentclass[doc, biblatex, floatsintext, longtable]{apa7}`). Notas de esta migración, ya resueltas:
- Usar la opción de clase `doc`, **no `jou`** — `jou` activa `\twocolumn` internamente pese a no ser obvio por el nombre
- `\hypersetup{colorlinks=true, ...}` después de `\addbibresource{}`, para que las citas no salgan con recuadro de color
- Evitar `tabularx`/columna `Y` en tablas — genera `Overfull \hbox` fantasma con el mecanismo de `\caption` de `apa7.cls`; usar `tabular` con columnas `p{ancho fijo}`
- **Cuidado con `resizebox` en tablas:** si se usa para *forzar* una tabla angosta a ocupar todo `\textwidth`, agranda también la fuente de forma desproporcionada respecto al cuerpo del texto — mejor dejar la tabla en su ancho natural y aceptar el warning cosmético de arriba
- Tamaño de fuente: se probó 12pt y se revirtió a 10pt/11pt por preferencia del autor; ambos son válidos en APA7

---

## 4. Estructura del repositorio (actualizada)

```
demanda-delivery-v2/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── environment.yml
├── scripts/
│   └── download_data.py
├── data/
│   ├── raw/                          # ignorado por git
│   ├── processed/
│   │   ├── demanda_limpia.csv
│   │   └── tabla_comparativa_literatura.csv
│   └── literatura/
├── notebooks/
│   ├── 01_exploracion_inicial.ipynb
│   ├── 02_limpieza_y_preparacion.ipynb
│   ├── 03_modelado_elasticidad.ipynb
│   ├── 03b_panel_efectos_fijos.ipynb        # NUEVO
│   ├── 03c_variables_instrumentales.ipynb   # NUEVO
│   ├── 04_validacion_supuestos.ipynb
│   ├── 05_comparacion_literatura.ipynb
│   └── 06_sql_bi_extraccion.ipynb           # NUEVO — prepara datos para SQL/BI
├── sql/                                      # NUEVO
│   ├── schema.sql
│   └── queries_negocio.sql
├── dashboard/                                 # NUEVO — Power BI / Looker Studio
│   └── (archivo .pbix o link a Looker publicado)
├── src/
│   ├── data_prep.py
│   ├── modeling.py                    # + funciones de panel FE y 2SLS
│   └── diagnostics.py                 # + weak-instrument F-test, Hansen J
├── reports/
│   ├── figuras/
│   ├── notebook_outputs/              # outputs exportados de los notebooks
│   └── bitacora_metodologica.md
├── informe/                         
│   ├── informe.tex               # clase apa7, una columna
│   └── jaes.bib
├── tests/
    └── test_data_prep.py
├── bitacora_metodologica.md          # Registro de decisiones metodológicas
├── Plan técnico de mejoras.md        # Análisis de viabilidad para v2.1 (splines, quantile, IV no-lineal)
├── requirements.txt
├── environment.yml
└── CITATION.cff                      # Metadatos de citación
```

---

## 5. Fases del proyecto (actualizado)

### Fase 0 — Setup ✅ Completa
### Fase 1 — Exploración inicial ✅ Completa
- Merge verificado sin pérdida/duplicación de filas, sin nulos
- N confirmado: 456,548

### Fase 2 — Limpieza y preparación ✅ Completa
- Hallazgo `variacion_precio` (25.43% de observaciones con precio final > precio de referencia), documentado y no excluido
- Variables derivadas: `variacion_precio`, `variacion_precio_pct`, `hubo_aumento`

### Fase 3 — Modelado de elasticidad (OLS) ✅ Completa
- Modelo base, con controles (categoría + tipo de centro), semi-log — comparados por AIC/BIC
- Elasticidad principal: **-0.7513** (log-log con controles)
- Hallazgo adicional: `hubo_aumento` no significativo (p=0.401) una vez controlado por precio/categoría/centro

### Fase 3x — Splines en log-precio (RCS) — NO-LINEALIDAD CONFIRMADA ✅ **COMPLETA (9 ago 2026)**
**Plan técnico v2.1, sección 1.2: VIABLES y RECOMENDADOS — IMPLEMENTADOS**
- RCS (Restricted Cubic Splines) con 4 nudos en percentiles 5, 35, 65, 95 de log_checkout_price
- Test de Wald: F = 10,279.37, p ≈ 0 → **RECHAZO FUERTE de linealidad** (log-log constante)
- Elasticidad puntual varía drásticamente: +0.58 (P05, precio=121) → -3.22 (P95, precio=640)
- Media (precio=332): -1.29 vs. log-log constante -0.75
- Modelo RCS supera a TODOS los modelos previos: R² adj = 0.5450, AIC = 1,105,001 (mejores)
- Activa prioridad media del Plan técnico: IV no-lineal vía control function + Quantile Regression
- Figura generada: `reports/figuras/elasticidad_puntual_rcs.png` (incluida en informe)

### Fase 3b — Panel con efectos fijos de dos vías ✅ Completa
- [x] Efectos fijos por semana (controla tendencias temporales/estacionalidad agregada)
- [x] Efectos fijos por combinación centro-comida (controla heterogeneidad no observada específica de cada par)
- [x] Errores estándar clusterizados (por `center_id` y/o `meal_id` — no "restaurante", ese identificador no existe en el dataset)
- [x] Comparar elasticidad resultante contra la de la Fase 3 (¿cambia al controlar por estos efectos?)

### Fase 3c — Variables instrumentales reales ✅ Completa
Estrategia tipo Hausman-Nevo, con instrumentos genuinamente calculables del dataset (no inventados):
- [x] Instrumento 1: `base_price` (precio de referencia administrativo, actualizado con baja frecuencia)
- [x] Instrumento 2: precio promedio de la misma comida en *otros* centros esa misma semana (leave-one-out mean price)
- [x] Primera etapa: regresión de `checkout_price` sobre ambos instrumentos + controles — reportar F-estadístico de relevancia (débil si F < 10, regla de Stock-Yogo)
- [x] Segunda etapa (2SLS): elasticidad instrumentada
- [x] Test de sobre-identificación (Hansen J), posible solo con 2 instrumentos para 1 endógena
- [x] Discusión honesta del supuesto de exclusión (no comprobable directamente, se argumenta cualitativamente)
- [x] Documentar explícitamente qué NO se pudo instrumentar con datos reales (costos de insumos, distancia logística — no existen en el dataset)
- [x] Robustez: IV justamente identificado (solo `base_price`) — diferencia 0.014, confirma resultado

### Fase 4 — Validación de supuestos (repetir sobre el/los modelo(s) de 3b/3c) ✅ Completa
- [x] Modelo OLS Fase 3: Breusch-Pagan+HC3, VIF, Jarque-Bera/Q-Q ✅ (notebook 04)
- [x] Modelo Panel FE 3b: residuos, Q-Q, clustering impact ratio 2.11, limitaciones ✅ (notebook 04b)
- [x] Modelo IV 3c: Wu-Hausman/Wooldridge (p≈0), Q-Q, 3-pilares síntesis ✅ (notebook 04c)

### Fase 5 — Comparación con literatura ✅ Completa
- Tabla final con **12 fuentes verificadas manualmente** contra el original (ver bitácora, sección 5)
- 3 grupos: A (5 fuentes, comida/delivery), B (2 fuentes, on-demand/e-commerce), C (4 fuentes, metodológico/mercado)
- Discusión comparativa integrada con resultados de Fases 3, 3b, 3c

### Fase 6 — Capa SQL + BI ✅ Completa
- [x] Cargar `demanda_limpia.csv` a SQLite vía `df.to_sql()` — 456,548 filas, índices optimizados
- [x] Queries de negocio: demanda por categoría, efecto promociones, tendencia semanal, comparación centros, elasticidad por categoría, KPIs ejecutivos
- [x] Exportación a CSV para Power BI / Looker Studio (`dashboard/data/`)
- [x] Traducción hallazgo econométrico a negocio: elasticidad -0.75 → precio +5% = ingreso +1.06%
- [x] Notebook `06_sql_bi_extraccion.ipynb` ejecutado y verificado

### Fase 7 — Empaquetado y documentación (antes Fase 6) ✅ **COMPLETA**
- [x] README completo, CITATION.cff
- [x] Verificar reproducibilidad end-to-end
- [x] Informe final en LaTeX (`apa7.cls`, una columna) con todos los resultados reales de 3, 3b, 3c, 4, 5, 6

### Fase 8 — Auditoría final y correcciones post-entrega (9 agosto 2026) ✅ **COMPLETA**
- [x] Corrección Tabla 2: `[H]` → `[htbp]` (flotante estándar)
- [x] Corrección Ecuaciones 2, 3, 4 y Tabla 3: `12 dummies` → `13 dummies` (14 categorías - 1 referencia)
- [x] Corrección Tablas 13 y 14 (Panel FE, IV 2SLS): `[htbp]` → `[H]` + `\footnotesize` (evita corte de página)
- [x] Eliminación JB=8,542 fabricado en Fase 4c, reemplazo por mención honesta a Q-Q plot
- [x] Corrección Grupo B: 2 → 4 fuentes (incluye Zhang & Demirkan 2021, Sriwongphanawes & Fukuda 2024)
- [x] Corrección promociones: "50-65%" → "65% email, 91% homepage" (coeficientes reales)
- [x] Corrección "Estadísticas descriptivos completos": `sidewaystable` → `table[htbp]` + `\scriptsize`
- [x] Corrección Grupo A: "5 fuentes externas" → "4 fuentes externas + este estudio (5 en total)"
- [x] **Plan técnico de mejoras.md** creado: análisis viabilidad v2.1 (RCS splines, quantile regression, Precio×Promo, IV no-lineal; AIDS/LA-AIDS y modelo dinámico descartados)
- [x] README.md actualizado con literatura, Plan técnico, promociones
- [x] PDF recompila limpio: 23 páginas, 2.3 MB (pdflatex → biber → pdflatex×2)

### Fase 10 — Inclusión completa de figuras en informe.tex (9 agosto 2026) ✅ **COMPLETA**
- [x] Auditoría completa de 17 figuras disponibles en `reports/figuras/`
- [x] 6 figuras nuevas agregadas al informe.tex (comparacion_tipo_centro, efecto_promociones_matriz, qqplot_residuos_modelo_loglog_controles, qq_iv_2sls, qq_panel_fe, comparacion_elasticidad_literatura.pdf)
- [x] 2 figuras copiadas desde reports/figuras/ a Informe/figuras/ (qq_iv_2sls.png, qq_panel_fe.png)
- [x] Compilación LaTeX exitosa: 28 páginas, 3.5 MB, todas las referencias \cref{} resueltas

### Fase 11 — Implementación Plan técnico v2.1 Sección 2.1: Robustez (9 agosto 2026) ✅ **COMPLETA**
- [x] **Quantile Regression** (τ=0.1, 0.25, 0.5, 0.75, 0.9): heterogeneidad detectada (-0.65 → -1.00)
- [x] **Interacción Precio × Promoción**: Wald F=659.73, p≈0 → no-aditividad significativa
- [x] **Test RESET (Ramsey)**: F=119.57, p=1.21e-52 → misspecification log-log confirmada
- [x] **IV Control Function (Wooldridge 2015)**: Endogeneidad F=1,581 (p≈0) + No-linealidad F=4,815 (p≈0) confirmadas
- [x] Código nuevo: `src/modeling.py` (4 funciones), `src/diagnostics.py` (2 funciones)
- [x] 4 figuras generadas e incluidas en informe.tex: quantile_regression, interaccion_precio_promo, iv_control_function_diagnosticos, reset_test_visual
- [x] LaTeX recompila limpio: 32 páginas (pdflatex → biber → pdflatex×2)
- [x] Salida consolidada: `reports/notebook_outputs/07_plan_tecnico_2_1_FASE7_COMPLETA.txt`

### Fase 8 — Implementación Plan técnico v2.1 Sección 2.3: Robustez (bajo esfuerzo) (10 agosto 2026) ✅ **COMPLETA**
- [x] **Test igualdad elasticidad por categoría** (Wald conjunto): F=519.6, p≈0 → RECHAZADA (varía -0.27 a -2.44)
- [x] **Submuestra ≥50 obs/semana**: 452,983 filas (99.2%), Δ elasticidad = 0.0035 (mínimo)
- [x] **Sensibilidad excluir top/bottom 1% precio**: 447,915 filas (98.1%), Δ elasticidad = -0.0523 (relevante)
- [x] **Comparar HC3 vs cluster vs bootstrap**: ratio cluster/HC3=7.92x, bootstrap/HC3=0.63x
- [x] Código: `src/modeling.py` (4 funciones nuevas), `scripts/ejecutar_plan_tecnico_2_3.py`
- [x] Salida consolidada: `reports/notebook_outputs/08_plan_tecnico_2_3_FASE8_COMPLETA.txt`

### Fase 9 — Plan técnico v2.1 Sección 2.2: Alto impacto / Medio esfuerzo ✅ **COMPLETA (11 ago 2026)**
- [x] **Modelo FE con interacción tiempo-variante** (log_price × semana): Wald χ²(144)=3068.85, p≈0, elasticidad base -1.5697, rango semanal -1.85 a -1.25
- [x] **Placebo test temporal**: 500 iteraciones, elasticidad real -0.7483 vs placebo -0.7513±0.00009, p=1.0 → robustez temporal confirmada
- [x] Código: `src/modeling.py` (demeaning FWL), `scripts/ejecutar_plan_tecnico_2_2.py`
- [x] Salida: `reports/notebook_outputs/09_plan_tecnico_2_2_FASE9_COMPLETA.txt`
- [x] **No requiere Azure** — ambos items corren en ~8 min total en PC estándar (~1.2GB RAM)

### Fase 10 — Appendix técnico en informe LaTeX ✅ **COMPLETA (11 ago 2026)**
- [x] Tablas completas RCS (coeficientes 4 nudos, ICs)
- [x] Tablas completas Quantile Regression (5 cuantiles con IC bootstrap 1000 iters)
- [x] Diagnósticos IV Control Function (primera y segunda etapa detalle)
- [x] Test RESET extendido (coeficientes potencias)
- [x] Tabla resumen comparativo de todas las especificaciones
- [x] Sección appendix antes de bibliografía (`\appendix` + `\section{Appendix Técnico}`)
- [x] LaTeX recompila limpio: 37 páginas, 5.36 MB (pdflatex → biber → pdflatex×2)

### Fase 11 — Inclusión completa de figuras + integración Plan 2.2 en informe.tex ✅ **COMPLETA (11 ago 2026)**
- [x] 22 figuras incluidas en informe.tex (todas las disponibles en `reports/figuras/`)
- [x] Sección 7.5 agregada: "Plan técnico v2.1 Sección 2.2: Alto impacto / Medio esfuerzo"
- [x] LaTeX recompila limpio: **39 páginas, 5.4 MB** (pdflatex → biber → pdflatex×2)
- [x] Todas las cross-references y bibliografía resueltas

### Fase 12 — Publicación en Zenodo 🔜
- [ ] Release en GitHub, subir a Zenodo con metadatos completos, DOI, vínculo bidireccional

---

### Fase 13 — Integración Panel FE Bootstrap y resultados VM (14 agosto 2026) ✅ **COMPLETA**
- [x] Bootstrap clusterizado Panel FE (200 iter, 77 centros) ejecutado en VM offline
- [x] Resultados: elasticidad media -1.5944, EE 0.0278, IC 95% [-1.6505, -1.5335], 200/200 válidas
- [x] Notebook 03b actualizado: celdas comentadas con referencia a VM, resultados cargados desde .npy
- [x] Trazabilidad: `reports/resultados_fase_3b.json` actualizado
- [x] Archivos .npy en `reports/notebook_outputs/`: panel_fe, quantile, vhat

### Fase 14 — Scripts VM para cómputos pesados (14 agosto 2026) ✅ **COMPLETA**
- [x] `scripts/vm_computaciones_pesadas.py` — Quantile (5×1000) + Panel FE (200)
- [x] `scripts/vm_bootstrap_vhat.py` — CF v_hat bootstrap 500 iter (Wooldridge 2015 §6.3.3)
- [x] `scripts/vm_wild_bootstrap.py` — Wild cluster bootstrap 500 iter (Rademacher, 77 centros)
- [x] `vm_wild_bootstrap.py` movido a `scripts/`
- [x] Limpieza: duplicados notebooks eliminados, outputs consolidados con nombres descriptivos
- [x] `reports/comparacion_modelos.csv` actualizado con todos los modelos

### Fase 15 — Plan P1-11, P1-12, P1-15 (14 agosto 2026) ✅ **COMPLETA**
- [x] **Wild cluster bootstrap VM** (2000 iter, Rademacher, 77 centros): elasticidad = -0.7586, EE = 0.0420, IC 95% [-0.839, -0.675], consistente con cluster analítico (ratio 7.87x vs HC3)
- [x] **ETAPA 3 Gate completado**: todos modelos rápidos + VM (panel FE bootstrap 200 iter, quantile bootstrap 5×1000, v_hat 500 iter, wild bootstrap 2000 iter) → concordancia CSV/JSON guardada, diferencias < 1e-3
- [x] **P1-11: Dashboard condicional** — `scripts/generar_dashboard_condicional.py` genera:
  - `elasticidad_rcs_por_precio.csv` (15 precios, RCS local + recomendación)
  - `elasticidad_por_categoria_detallada.csv` (14 categorías, interacción)
  - `recomendaciones_pricing.csv` (210 celdas precio×categoría, delta ingreso 5%)
  - `kpis_ejecutivos.csv` actualizado con wild bootstrap, cuantiles, RCS min/max
- [x] **P1-12: VIF semi-log** — Max VIF = 4.91 (checkout_price), Mean VIF = 2.35 → sin multicolinealidad severa; condition number 5.02e+03 es escala, no multicolinealidad real
- [x] **P1-15: Referencias metodológicas** — completado por usuario en jaes.bib (Wooldridge 2015, Cameron & Miller 2015, Hausman 1996, Nevo 2001, Stock & Yogo 2005, Ramsey 1969)

### Fase 16 — P1-7, P1-8, P1-9/10 verificados en informe.tex (14 agosto 2026) ✅ **COMPLETA**
- [x] **P1-7**: `log1p(num_orders)` → `log(num_orders)` (mínimo = 13, no hay ceros)
- [x] **P1-8**: Ausencia de ceros documentada en Limitaciones → Datos (ítem 2)
- [x] **P1-9/10**: "S/" → "unidades monetarias del dataset"; "precio del alimento" → "precio final checkout (incluye delivery, impuestos, descuentos)"

### Fase 17 — Próximos pasos inmediatos
- [ ] **ETAPA 9**: Auditoría final completa del proyecto
- [ ] **Tag v2.2** y release
- [ ] Zenodo vinculado correctamente al repo

---

## 6. Criterios de calidad (checklist final antes de publicar)

- [x] Pipeline reproducible de punta a punta
- [x] Cada decisión de limpieza documentada y justificada
- [x] Modelo(s) elegido(s) justificado(s) frente a alternativas
- [x] Limitaciones causales explícitas — incluyendo qué tan creíble es el supuesto de exclusión de los instrumentos usados
- [x] Ningún resultado de la Fase 3b/3c se presenta como "hecho" sin haber corrido el código real (evitar el error del "análisis fantasma" que motivó esta expansión)
- [x] Comparación con literatura no sobredimensiona el alcance
- [x] README y CITATION.cff completos
- [ ] Zenodo vinculado correctamente al repo

---

## 7. Estimación de tiempo

Sin límite estricto de 1 mes (decisión del 2 agosto 2026). Orden de trabajo sugerido: 3b → 3c → 4 (repetida) → 6 (SQL+BI) → 7 (informe) → 8 (Zenodo). Cada fase se cierra con resultados reales verificados antes de pasar a la siguiente — no se redacta contenido de una fase sin haber corrido el código correspondiente.
