# Price elasticity of demand in food delivery

[![tests](https://github.com/s7ex-j/demanda-delivery-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/s7ex-j/demanda-delivery-v2/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1430542939.svg)](https://doi.org/10.5281/zenodo.23585772)

<sub>[Español](README.md) · **English**</sub>

Independent research that estimates the price elasticity of demand on a food delivery platform
with **456,548 week × center × meal observations** from the public *Food Demand Forecasting*
dataset (Genpact / Analytics Vidhya, on Kaggle). It includes the full LaTeX report
(`Informe/informe.pdf`, 49 pages, in Spanish), the code, the notebooks and every certified
result the report cites.

**Author:** Jharol André Vilca Ramos · [ORCID 0009-0009-7897-3439](https://orcid.org/0009-0009-7897-3439) · Version 3.3.0

**Report:** [doi.org/10.5281/zenodo.23586040](https://doi.org/10.5281/zenodo.23586040) · **Code:** [doi.org/10.5281/zenodo.23585772](https://doi.org/10.5281/zenodo.23585772) · **ResearchGate:** [publication](https://www.researchgate.net/publication/416623085_Elasticidad_Precio-Demanda_en_el_Servicio_de_Delivery_de_Comidas_Evidencia_desde_Datos_Transaccionales)

---

## Main result

Elasticity **is not a constant parameter**: it depends on the price level, the category, the
week and the source of variation used to estimate it.

| Specification | Elasticity | Inference | What it represents |
|---|---|---|---|
| **Log-log OLS with controls (main)** | **−0.759** | SE clustered by entity 0.043; 95% CI [−0.843; −0.675] | Average association, inelastic |
| Two-way fixed effects panel | −1.5985 | Clustered SE 0.024; pairs bootstrap [−1.651; −1.546] | *Within* variation of each center-meal |
| IV 2SLS (Hausman-Nevo, same controls) | −0.841 | SE clustered by center 0.050; CI [−0.940; −0.743] | Sensitivity; not distinguishable from OLS |
| RCS (splines on log-price) | +0.04 (P5) · −1.44 (P50) · ≈ −1.51 (mean price) · −1.25 (P95) | Bootstrap bands by center | Local elasticity, non-monotonic |
| Control function (RCS + v̂) | −1.504 at the mean price | Bootstrap CI [−1.630; −1.370] | Almost the same as RCS-OLS at that point |
| Quantile regression (τ = 0.1 → 0.9) | −0.67 → −1.01 | Analytical SE | More elastic where demand is high |

Specification tests: RESET F(2; 456,526) = 153.07; nested F test log-log vs. RCS
F(2; 456,526) = 3,607.28. The IV and the control function are presented as **sensitivity
analyses**: the instruments are strong (Shea partial R² 0.92), but exclusion cannot be tested,
and Hansen J only rejects under heteroskedasticity-robust weighting (33.3), not under weighting
clustered by center (0.92; p = 0.34). No estimate should be read as the causal effect of a
pricing policy.

---

## How to reproduce

```bash
git clone https://github.com/s7ex-j/demanda-delivery-v2.git
cd demanda-delivery-v2
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_data.py      # downloads from Kaggle, checks SHA-256 and builds data/processed/
pytest -q                            # 53 tests: they reproduce the report's figures
```

Without the data, `pytest` runs the code and certified-artifact tests and skips the ones that
recompute the report's figures; that is also how GitHub Actions runs on every push.

The data is **not included** in the repository: it belongs to the original competition and is
downloaded with the Kaggle API (credentials in `~/.kaggle/kaggle.json`) or manually into
`data/raw/` from [kaggle.com/datasets/kannanaikkal/food-demand-forecasting](https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting).
`download_data.py` checks that the files match the ones used in the analysis.

| Step | Command | Output |
|---|---|---|
| Notebooks (narrative, in order 01 → 06) | `jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb` | `reports/notebook_outputs/`, `dashboard/data/` |
| Dashboard CSVs derived from models | `python scripts/analysis/generar_dashboard_modelos.py` | `dashboard/data/` |
| IV / control function diagnostics | `python scripts/analysis/iv_controles_diagnosticos.py` | `reports/certified/iv_cf_diagnosticos_v325.json` |
| Report figures | `python scripts/figure_scripts/generar_todas_las_figuras.py` | `Informe/figuras/` |
| Control function bootstrap (5,000 replicas) | `python scripts/vm/bootstrap_cf_canonical_replicas.py` | `reports/certified/` |
| IV wild cluster bootstrap (2,000 replicas) | `python scripts/vm/wild_bootstrap_iv_2sls_controles.py` | `reports/certified/` |
| Report | `cd Informe && lualatex informe && biber informe && lualatex informe && lualatex informe` | `Informe/informe.pdf` |

The notebooks take about 3 minutes in total; the bootstraps with thousands of replicas can take
much longer depending on the machine.

The heaviest bootstraps (panel pairs bootstrap, RCS bands, OLS wild bootstrap, jackknife) live
in `scripts/analysis/` and their results in `reports/diagnostics/`; the tests and figures read
those artifacts without re-running them.

---

## Structure

```
├── Informe/                 informe.tex, jaes.bib, figuras/ and informe.pdf
├── src/                     data_prep.py, modeling.py, rcs_utils.py, diagnostics.py
├── notebooks/               01–06: exploration, cleaning, models, validation, literature, SQL/BI
├── scripts/
│   ├── download_data.py     Kaggle data → data/processed/demanda_limpia.csv
│   ├── analysis/            diagnostics and bootstrap (IV, CF, panel, RCS, Hansen J)
│   ├── vm/                  heavy bootstraps with saved replicas
│   ├── figure_scripts/      one script per report figure
│   └── runners/             temporal placebo (within-entity permutation)
├── reports/
│   ├── certified/           results cited by the report (JSON/NPZ/NPY)
│   └── diagnostics/         bootstrap, jackknife and master comparison table
├── sql/queries_negocio.sql  SQLite views used by notebook 06
├── dashboard/data/          8 descriptive CSVs for Power BI / Looker Studio
├── tests/                   53 tests (pytest)
├── CHANGELOG.md · CITATION.cff · LICENSE
```

---

## Main limitations

- Aggregated observational data, with no calendar date and no zero-demand rows (minimum 13 orders).
- Price is not randomly assigned; the FE panel does not solve simultaneity and the IV relies on
  an exclusion assumption that cannot be tested.
- The RCS elasticity depends on the choice of knots, and the bootstrap bands are pointwise.
- A single dataset and an unspecified geographic context.

Full detail in the "Limitaciones y Trabajo Futuro" (Limitations and future work) section of the
report.

## Use of AI

AI assistants were used during the project to review code, audit consistency between the report
and the results, and draft documentation. The methodological decisions, the check of every
bibliographic source against the original and the interpretation of results are the author's.
Early on, an AI produced citations with non-existent authors; they were caught and discarded,
and the literature review was rebuilt by checking every source against the original. Every
result in the report is reproduced from this repository's code (see `tests/`).

## Cite

See [`CITATION.cff`](CITATION.cff).

```bibtex
@techreport{vilcaramos2026elasticidad,
  author      = {Vilca Ramos, Jharol André},
  title       = {Elasticidad precio-demanda en el servicio de delivery de comidas: evidencia desde datos transaccionales},
  institution = {Zenodo},
  type        = {Working paper},
  year        = {2026},
  doi         = {10.5281/zenodo.23586040},
  url         = {https://doi.org/10.5281/zenodo.23586040}
}
```

## License

Code and documentation under the MIT license ([`LICENSE`](LICENSE)). The Kaggle data is not
redistributed and is governed by the terms of its original source.
