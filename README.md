# Análisis de Elasticidad Precio-Demanda en Delivery de Comidas (v2)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## 📋 Resumen

Este proyecto estima la **elasticidad precio de la demanda** en una plataforma de *food delivery* utilizando datos transaccionales reales (456,548 observaciones) del dataset público **Food Demand Forecasting** (Kaggle/Genpact). Es la **segunda versión metodológica** (v2) de un estudio previo que utilizó encuesta propia; esta versión reemplaza la fuente por datos transaccionales reales e implementa un pipeline reproducible completo en Python.

**Resultado principal:** Elasticidad = **−0.7513** (IC 95% HC3: [−0.762, −0.741]), inelástica, consistente con la literatura de comida fuera del hogar.

**Informe final:** 39 páginas, compilación LaTeX limpia (apa7.cls, pdflatex → biber → pdflatex×2).

---

## 🗂️ Estructura del Repositorio

```
demanda-delivery-v2/
├── data/
│   ├── raw/                          # Datos originales de Kaggle (no versionados)
│   └── processed/                    # Datos limpios listos para modelado
│       └── demanda_limpia.csv        # 456,548 filas, 23 columnas
├── notebooks/                        # Pipeline completo en 10 notebooks
│   ├── 01_exploracion_inicial.ipynb
│   ├── 02_limpieza_y_preparacion.ipynb
│   ├── 03_modelado_elasticidad.ipynb
│   ├── 03b_panel_efectos_fijos.ipynb
│   ├── 03c_variables_instrumentales.ipynb
│   ├── 04_validacion_supuestos.ipynb
│   ├── 04b_validacion_panel.ipynb
│   ├── 04c_validacion_iv.ipynb
│   ├── 05_comparacion_literatura.ipynb
│   └── 06_sql_bi_extraccion.ipynb
├── src/                              # Código modular reutilizable
│   ├── data_prep.py
│   ├── modeling.py
│   └── diagnostics.py
├── sql/                              # Capa SQL + BI
│   ├── demanda_delivery.db           # SQLite con 456,548 filas
│   ├── schema.sql
│   └── queries_negocio.sql
├── dashboard/
│   └── data/                         # 6 CSVs para Power BI / Looker Studio
├── reports/
│   ├── notebook_outputs/             # Salidas consolidadas por fase
│   │   ├── 03_modelado_elasticidad.txt
│   │   ├── 04_validacion_supuestos.txt
│   │   ├── 05_comparacion_literatura_CONSOLIDADO.txt
│   │   ├── FASE6_SQL_BI_EXTRACCION.txt
│   │   ├── PLAN_TECNICO_v2.1_ROBUSTEZ_ALTO_IMPACTO.txt
│   │   ├── PLAN_TECNICO_v2.1_ROBUSTEZ_MEDIO_ESFUERZO.txt
│   │   ├── PLAN_TECNICO_v2.1_ITEMS_ROBUSTEZ.txt
│   │   ├── bootstrap_panel_fe_results.npy
│   │   ├── bootstrap_quantile_results.npy
│   │   ├── bootstrap_vhat_results.npy
│   │   ├── etapa3_concordancia.csv
│   │   └── etapa3_resultados_completos.json
│   └── figuras/                      # Gráficos para el informe
├── Informe/
│   ├── informe.tex                   # Informe final en LaTeX (apa7.cls)
│   ├── informe.pdf                   # PDF compilado
│   └── jaes.bib                      # Bibliografía verificada (14 entradas)
├── tests/                            # Tests unitarios
├── bitacora_metodologica.md          # Registro de decisiones metodológicas
├── requirements.txt
├── environment.yml
└── CITATION.cff                      # Metadatos de citación
```

---

## 🚀 Instalación

### Opción A: Conda (recomendado)
```bash
conda env create -f environment.yml
conda activate demanda-delivery
```

### Opción B: pip
```bash
pip install -r requirements.txt
```

### Dependencias principales
- Python ≥ 3.11
- pandas, numpy, scipy, statsmodels, linearmodels
- matplotlib, seaborn
- jupyter, nbconvert
- sqlite3 (stdlib)

---

## ▶️ Ejecución del Pipeline

### Ejecutar notebooks en orden (Jupyter)
```bash
jupyter notebook
# Abrir y ejecutar: 01 → 02 → 03 → 03b → 03c → 04 → 04b → 04c → 05 → 06
```

### Ejecutar todo en batch (nbconvert)
```bash
cd notebooks
for nb in *.ipynb; do
    jupyter nbconvert --to notebook --execute "$nb" --output "executed_$nb"
done
```

### Generar salidas consolidadas
```bash
# Desde la raíz del proyecto
python -m scripts.consolidate_outputs
```

---

## 📊 Resultados Principales

### Elasticidades estimadas (seis estrategias de identificación + especificaciones flexibles)

| Especificación | Elasticidad | Qué corrige | Limitación principal |
|----------------|-------------|-------------|---------------------|
| **OLS log-log con controles (Fase 3)** | **−0.7513** | Heterogeneidad observable (categoría, centro, promos) | No corrige endogeneidad ni FE entidad/tiempo |
| **Panel FE dos vías (Fase 3b)** | **−1.5755** | Heterogeneidad no observada entidad + tendencias temporales | No corrige endogeneidad (simultaneidad) |
| **Panel FE bootstrap clusterizado (VM)** | **−1.5944** (IC: −1.65, −1.53) | Validación bootstrap del Panel FE (200 iter, 77 centros) | Cómputo pesado → VM offline |
| **IV 2SLS Hausman-Nevo (Fase 3c)** | **−0.9487** | Simultaneidad precio-demanda | Hansen J rechaza validez conjunta (mitigado por robustez 1-inst: −0.9627) |
| **Wild cluster bootstrap (VM, 2000 iter)** | **−0.7586** (IC: −0.839, −0.675) | Inferencia con pocos clusters (77 centros, Rademacher 2000 iter) | EE=0.042, consistente con cluster center_id (0.0424) |
| **RCS Splines log-precio (Fase 3x)** | **Variable** (+0.58 a −3.22) | No-linealidad elástica según nivel de precio | Asume exogeneidad; elasticidad no constante |
| **Control Function IV-RCS** | **Variable** (R² adj=0.5467) | Endogeneidad + no-linealidad simultánea (Wooldridge 2015) | EE estándar límite inferior; bootstrap v_hat en VM |
| **Panel FE time-varying (Plan 2.2)** | **−1.57 a −1.85** (base −1.57) | Heterogeneidad temporal (semana × precio) | No corrige endogeneidad; solo within |
| **Placebo temporal (Plan 2.2)** | **p=1.0** (real −0.748 vs placebo −0.751) | Confusión temporal espuria | No rechaza H0 → evidencia de robustez |

> **Conclusión:** No hay un único número "verdadero". Cada modelo corrige una amenaza distinta y todos coexisten en el informe con su discusión honesta. El test de Wald (F=10,279, p≈0) **rechaza fuertemente la linealidad log-log** — la elasticidad varía drásticamente con el nivel de precio.

### Modelos adicionales (Plan técnico v2.1)

| Modelo | Elasticidad / Resultado | Contexto |
|--------|------------------------|----------|
| Quantile Regression (τ=0.1) | −0.6523 | Demanda baja: inelástica |
| Quantile Regression (τ=0.5) | −0.7702 | Mediana: similar a OLS |
| Quantile Regression (τ=0.9) | −1.0048 | Demanda alta: elástica |
| Semi-log con controles | −1.0624 (eval. precio medio) | Especificación alternativa |
| IV just-identified (base_price) | −0.9627 | Robustez 1 instrumento |

### Validación de supuestos (Fases 4, 4b, 4c)

| Test | Modelo | Resultado | Acción |
|------|--------|-----------|--------|
| Breusch-Pagan (heterocedasticidad) | OLS Fase 3 | p = 0.0 → **rechaza H₀** | Errores HC3 (SE +15%) |
| VIF (multicolinealidad) | OLS Fase 3 | Máx 1.204 | ✅ Sin problema |
| Jarque-Bera (normalidad) | OLS Fase 3 | p = 0.0 (N grande) | TCL OK, Q-Q plot aceptable |
| Durbin-Watson | OLS Fase 3 | 1.195 | Autocorrelación positiva → abordada en Fase 3b con clustering |
| Wu-Hausman / Wooldridge (endogeneidad) | OLS vs 2SLS | p ≈ 0 → **rechaza H₀** | IV 2SLS justificado |
| Hansen J (sobreidentificación) | IV 2SLS | χ² = 320, p ≈ 0 → **rechaza** | Discutido: colinealidad 0.953 + poder excesivo N=456k; robustez 1-inst OK |
| Clustering ratio (entidad vs HC3) | Panel FE 3b | **7.92x** (0.0236/0.0030) | Clustering NECESARIO y justificado |
| Shapiro-Wilk (n=5000) | Panel FE 3b | W=0.9837, p<0.001 | Colas pesadas típicas, centro OK |
| Durbin-Wu-Hausman (endogeneidad) | IV 2SLS | F=3492, p≈0 → **rechaza H₀** | Endogeneidad CONFIRMADA (IV necesario) |
| Panel FE time-varying Wald | Panel FE 2.2 | χ²(144)=3068.85, p≈0 | Heterogeneidad temporal confirmada |
| Placebo temporal | Panel FE 2.2 | p=1.0 | Robustez temporal confirmada |

### Robustez adicional (Plan técnico v2.1 - Secciones 2.1, 2.2, 2.3)

| Mejora / Item | Resultado | Impacto |
|---------------|-----------|---------|
| **Sección 2.1 (Alto impacto / Bajo esfuerzo)** | | |
| Quantile Regression (τ=0.1→0.9) | Elast.: -0.65 → -1.00 | ALTO - Heterogeneidad real por cuantil |
| Precio × Promo Interacción | F=659.73, p≈0 | ALTO - No aditividad significativa |
| RESET Test | F=119.57, p≈0 | ALTO - Misspecification confirmada |
| IV Control Function | Exog. F=1,581, No-lin. F=4,815 | ALTO - Doble corrección (endog. + no-lin.) |
| **Sección 2.2 (Alto impacto / Medio esfuerzo)** | | |
| Panel FE time-varying | χ²(144)=3068.85, p≈0 | ALTO - Elasticidad varía por semana |
| Placebo temporal | p=1.0 | ALTO - Robustez temporal confirmada |
| **Sección 2.3 (Medio impacto / Bajo esfuerzo)** | | |
| RCS Wald (no-linealidad) | F=10,279, p≈0 | ALTO - Rechaza linealidad log-log |
| Igualdad elasticidad por categoría | F(13)=519.6, p≈0 | RECHAZADA - Varían de -0.27 a -2.44 |
| Submuestra ≥50 obs/semana | Δ=+0.0035 | MÍNIMO - Robusto a items esporádicos |
| Sensibilidad top/bottom 1% precio | Δ=-0.0523 | RELEVANTE (7%) - Outliers afectan |
| HC3 vs Cluster vs Bootstrap | Ratio 7.92x | Cluster >> HC3 (dependencia intra-entidad) |

### Traducción a negocio (Fase 6)

| Pregunta | Respuesta (elasticidad −0.75) |
|----------|-------------------------------|
| **Subir precio 5% → ingreso** | +1.06% (demanda cae ~3.75%) ✅ |
| **Bajar precio 5% → ingreso** | −1.44% (demanda sube ~3.75%) ❌ |
| **Promociones vs precio** | Emailer +65%, Homepage +91% > efecto precio |
| **Pricing general (plataforma)** | OLS −0.75 o 2SLS −0.95 → inelástica |
| **Pricing granular (ítem)** | Panel FE −1.58 → más sensible |

---

## 📚 Literatura Verificada (Fase 5)

**12 fuentes externas verificadas contra fuente primaria + el estudio propio (13 en total)**, agrupadas en 3 categorías:

| Grupo | Fuentes | Descripción |
|-------|---------|-------------|
| **A: Comparación directa (comida/delivery)** | 4 externas + este estudio (5) | Andreyeva et al. 2010 (meta-análisis), Damayanti et al. 2023 (Indonesia), Bumagat & Ibana 2025 (pico/no pico), Li & Wang 2025 (contexto), **Este estudio** |
| **B: Contextual (on-demand/e-commerce)** | 4 | Christensen & Osman 2021/2023 (Uber El Cairo), Zhang & Demirkan 2021 (CPG online), Sriwongphanawes & Fukuda 2024 (Uber Japón) |
| **C: Soporte metodológico/mercado** | 4 | Statista 2024, Kahneman et al. 1991, Davidson 1994, Long & Ervin 2000 |

*Nota: Zhang & Demirkan 2021 y Sriwongphanawes & Fukuda 2024 están incluidos en el Grupo B tras verificación (ver `jaes.bib`).*

---

## 🗺️ Plan Técnico de Mejoras (v2.1)

Se ha documentado un **Plan técnico de mejoras.md** con análisis de viabilidad para una futura versión 2.1 del estudio, incluyendo:

- **Prioridad alta (viables ya)**: Splines RCS en log-precio (test de no-linealidad), Quantile Regression (τ=0.1, 0.25, 0.5, 0.75, 0.9), Interacción Precio × Promoción
- **Prioridad media (condicional)**: IV no-lineal vía control function (si splines rechaza linealidad)
- **Descartados**: AIDS/LA-AIDS/EASI (datos incompatibles - no son datos de canasta), Modelo dinámico con lags (sesgo de Nickell severo en panel desbalanceado con Min Obs=1)

Ver `Plan técnico de mejoras.md` para el roadmap completo, código de ejemplo y justificación técnica detallada.

---

## 🖥️ Scripts VM para cómputos pesados

Los siguientes scripts ejecutan bootstrap pesados en VM offline (GPU dedicado + 32GB RAM):

| Script | Iteraciones | Descripción |
|--------|-------------|-------------|
| `scripts/vm_computaciones_pesadas.py` | 5×1000 + 200 | Quantile Regression (5 cuantiles) + Panel FE bootstrap clusterizado |
| `scripts/vm_bootstrap_vhat.py` | 500 | Control Function v_hat bootstrap (Wooldridge 2015 §6.3.3) |
| `scripts/vm_wild_bootstrap.py` | 2000 | Wild cluster bootstrap (Rademacher, 77 centros) ✅ COMPLETADO |

Resultados se guardan en `reports/notebook_outputs/*.npy` y se cargan en notebooks para trazabilidad.

---

## 📄 Informe Final

El informe completo está en `Informe/informe.tex` (clase `apa7.cls`, una columna). Para compilar:

```bash
cd Informe
latexmk -pdf informe.tex
# Requiere: texlive (apa7, biblatex, biber, etc.)
```

PDF generado: `Informe/informe.pdf`

---

## 🔬 Reproducibilidad

- **Pipeline completo**: 10 notebooks ejecutables en orden
- **Datos**: Kaggle/Genpact Food Demand Forecasting (enlace en `jaes.bib`)
- **Semillas fijas**: bootstrap (seed=42), todos los modelos
- **Salidas**: CSVs, SQLite, figuras, reportes TXT consolidados
- **Bibliografía**: 14 entradas en `jaes.bib` verificadas manualmente

---

## 📝 Citar este trabajo

```bibtex
@misc{vilca2026elasticidad,
  author       = {Vilca Ramos, Jharol André},
  title        = {Elasticidad Precio-Demanda en el Servicio de Delivery de Comidas: Evidencia desde Datos Transaccionales (v2)},
  year         = {2026},
  howpublished = {\url{https://github.com/usuario/demanda-delivery-v2}},
  note         = {Dataset: Kaggle/Genpact Food Demand Forecasting; 456,548 observaciones; Pipeline reproducible en Python}
}
```

Ver `CITATION.cff` para metadatos completos (incluye DOI de Zenodo al publicar).

---

## ⚖️ Licencia

MIT License — ver `LICENSE` (por agregar).

---

## 🙏 Agradecimientos

- Genpact / Analytics Vidhya por el dataset público en Kaggle
- Comunidad de econometría aplicada por referencias metodológicas
- Revisores que identificaron errores en la versión v1 (encuesta propia)

---

## 📧 Contacto

**Jharol André Vilca Ramos** — Estudiante de Ingeniería Industrial y Ciencia de Datos  
ORCID: [0009-0009-7897-3439](https://orcid.org/0009-0009-7897-3439)