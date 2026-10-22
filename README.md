# Elasticidad precio-demanda en delivery de comidas

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Licencia: MIT](https://img.shields.io/badge/Licencia-MIT-green.svg)](LICENSE)

Investigación independiente que estima la elasticidad precio de la demanda en una plataforma de
entrega de comidas con **456 548 observaciones semana × centro × comida** del conjunto público
*Food Demand Forecasting* (Genpact / Analytics Vidhya, en Kaggle). Incluye el informe completo en
LaTeX (`Informe/informe.pdf`, 49 páginas), el código, los notebooks y todos los resultados
certificados que cita el informe.

**Autor:** Jharol André Vilca Ramos · [ORCID 0009-0009-7897-3439](https://orcid.org/0009-0009-7897-3439) · Versión 3.3.0

---

## Resultado principal

La elasticidad **no es un parámetro constante**: depende del nivel de precio, la categoría, la
semana y la fuente de variación que se use para estimarla.

| Especificación | Elasticidad | Inferencia | Qué representa |
|---|---|---|---|
| **OLS log-log con controles (principal)** | **−0,759** | EE agrupado por entidad 0,043; IC 95 % [−0,843; −0,675] | Asociación promedio, inelástica |
| Panel con efectos fijos de dos vías | −1,5985 | EE agrupado 0,024; pairs bootstrap [−1,651; −1,546] | Variación *within* de cada centro-comida |
| IV 2SLS (Hausman-Nevo, mismos controles) | −0,841 | EE agrupado por centro 0,050; IC [−0,940; −0,743] | Sensibilidad; no distinguible del OLS |
| RCS (splines en log-precio) | +0,04 (P5) · −1,44 (P50) · ≈ −1,51 (precio medio) · −1,25 (P95) | Bandas bootstrap por centro | Elasticidad local, no monótona |
| Función de control (RCS + v̂) | −1,504 en el precio medio | IC bootstrap [−1,630; −1,370] | Casi igual a la RCS-OLS en el mismo punto |
| Regresión cuantil (τ = 0,1 → 0,9) | −0,67 → −1,01 | EE analíticos | Más elástica donde la demanda es alta |

Tests de especificación: RESET F(2; 456 526) = 153,07; contraste F anidado log-log vs. RCS
F(2; 456 526) = 3 607,28. El IV y la función de control se presentan como **análisis de
sensibilidad**: los instrumentos son fuertes (R² parcial de Shea 0,92), pero la exclusión no es
contrastable y Hansen J solo rechaza con ponderación heterocedástica (33,3), no con ponderación
agrupada por centro (0,92; p = 0,34). Ninguna estimación debe leerse como efecto causal de una
política de precios.

---

## Cómo reproducir

```bash
git clone https://github.com/s7ex-j/demanda-delivery-v2.git
cd demanda-delivery-v2
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_data.py      # descarga de Kaggle, verifica SHA-256 y genera data/processed/
pytest -q                            # 53 tests: reproducen las cifras del informe
```

Los datos **no se incluyen** en el repositorio: pertenecen a la competencia original y se
descargan con la API de Kaggle (credenciales en `~/.kaggle/kaggle.json`) o manualmente en
`data/raw/` desde [kaggle.com/datasets/kannanaikkal/food-demand-forecasting](https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting).
`download_data.py` comprueba que los archivos coinciden con los usados en el análisis.

| Paso | Comando | Salida |
|---|---|---|
| Notebooks (narrativa, en orden 01 → 06) | `jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb` | `reports/notebook_outputs/`, `dashboard/data/` |
| CSV de dashboard derivados de modelos | `python scripts/analysis/generar_dashboard_modelos.py` | `dashboard/data/` |
| Diagnósticos IV / función de control | `python scripts/analysis/iv_controles_diagnosticos.py` | `reports/certified/iv_cf_diagnosticos_v325.json` |
| Figuras del informe | `python scripts/figure_scripts/generar_todas_las_figuras.py` | `Informe/figuras/` |
| Bootstrap de la función de control (5 000 réplicas) | `python scripts/vm/bootstrap_cf_canonical_replicas.py` | `reports/certified/` |
| Wild cluster bootstrap del IV (2 000 réplicas) | `python scripts/vm/wild_bootstrap_iv_2sls_controles.py` | `reports/certified/` |
| Informe | `cd Informe && lualatex informe && biber informe && lualatex informe && lualatex informe` | `Informe/informe.pdf` |

Los notebooks tardan unos 3 minutos en total; los bootstrap de miles de réplicas pueden tardar
bastante más según la máquina.

Los bootstrap más pesados (pairs bootstrap del panel, bandas RCS, wild bootstrap del OLS,
jackknife) están en `scripts/analysis/` y sus resultados en `reports/diagnostics/`; los tests y
las figuras leen esos artefactos sin re-ejecutarlos.

---

## Estructura

```
├── Informe/                 informe.tex, jaes.bib, figuras/ e informe.pdf
├── src/                     data_prep.py, modeling.py, rcs_utils.py, diagnostics.py
├── notebooks/               01–06: exploración, limpieza, modelos, validación, literatura, SQL/BI
├── scripts/
│   ├── download_data.py     datos de Kaggle → data/processed/demanda_limpia.csv
│   ├── analysis/            diagnósticos y bootstrap (IV, CF, panel, RCS, Hansen J)
│   ├── vm/                  bootstrap pesados con réplicas guardadas
│   ├── figure_scripts/      un script por figura del informe
│   └── runners/             placebo temporal (permutación dentro de entidad)
├── reports/
│   ├── certified/           resultados que cita el informe (JSON/NPZ/NPY)
│   └── diagnostics/         bootstrap, jackknife y tabla maestra de comparación
├── sql/queries_negocio.sql  vistas SQLite usadas por el notebook 06
├── dashboard/data/          8 CSV descriptivos para Power BI / Looker Studio
├── tests/                   53 tests (pytest)
├── CHANGELOG.md · CITATION.cff · LICENSE
```

---

## Limitaciones principales

- Datos observacionales agregados, sin fecha calendario ni ceros de demanda (mínimo 13 pedidos).
- El precio no se asigna aleatoriamente; el panel FE no resuelve simultaneidad y el IV depende de
  un supuesto de exclusión no contrastable.
- La elasticidad RCS depende de la elección de nudos y las bandas bootstrap son punto a punto.
- Un solo dataset y contexto geográfico no especificado.

Detalle completo en la sección "Limitaciones y Trabajo Futuro" del informe.

## Uso de IA

Durante el proyecto se usaron asistentes de IA para revisar código, auditar la coherencia entre
el informe y los resultados, y redactar documentación. Las decisiones metodológicas, la
verificación de cada fuente bibliográfica contra el original y la interpretación de resultados
son del autor. En una etapa temprana, una IA generó citas con autores inexistentes; se
detectaron, se descartaron y la revisión de literatura se reconstruyó verificando cada fuente
contra el original. Cada resultado del informe se reproduce desde el código de este repositorio
(ver `tests/`).

## Citar

Ver [`CITATION.cff`](CITATION.cff).

```bibtex
@misc{vilcaramos2026elasticidad,
  author = {Vilca Ramos, Jharol André},
  title  = {Elasticidad precio-demanda en el servicio de delivery de comidas: evidencia desde datos transaccionales},
  year   = {2026},
  note   = {Versión 3.3.0},
  url    = {https://github.com/s7ex-j/demanda-delivery-v2}
}
```

## Licencia

Código y documentación bajo licencia MIT ([`LICENSE`](LICENSE)). Los datos de Kaggle no se
redistribuyen y se rigen por los términos de su fuente original.
