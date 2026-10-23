#!/usr/bin/env python3
"""
Figura 20: IV Control Function - Primera Etapa
Gráfico de dispersión: Precio observado vs Precio ajustado con instrumentos Hausman-Nevo + controles
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
INFORME_FIG_DIR = PROJECT_ROOT / "Informe" / "figuras"
REPORTS_FIG_DIR = INFORME_FIG_DIR  # una sola copia de las figuras (Informe/figuras)

INFORME_FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(fig, name):
    fig.savefig(INFORME_FIG_DIR / f"{name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(REPORTS_FIG_DIR / f"{name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] {name}.png")

# Cargar datos y construir instrumentos
df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "demanda_limpia.csv")
df['log_checkout_price'] = np.log(df['checkout_price'])
df['log_base_price'] = np.log(df['base_price'])

# Instrumentos Hausman-Nevo (canónico: promedio de OTROS centros en nivel, excluyendo propia observación)
grupo = df.groupby(['meal_id', 'week'])['checkout_price']
suma_grupo = grupo.transform('sum')
conteo_grupo = grupo.transform('count')
suma_otros = suma_grupo - df['checkout_price']
conteo_otros = conteo_grupo - 1
precio_prom_otros = suma_otros / conteo_otros.replace(0, np.nan)
df['log_precio_prom_otros_centros'] = np.log(precio_prom_otros)
df = df.dropna(subset=['log_precio_prom_otros_centros']).copy()

# Primera etapa: log_checkout_price ~ log_base_price + log_precio_prom_otros_centros + controles
exog = df[['log_base_price', 'log_precio_prom_otros_centros', 'emailer_for_promotion', 'homepage_featured', 'hubo_aumento', 'category', 'center_type']].copy()
exog = pd.get_dummies(exog, columns=['category', 'center_type'], drop_first=True)
exog = exog.astype(float)
exog = sm.add_constant(exog, has_constant='add')
endog = df['log_checkout_price'].astype(float)

mod = sm.OLS(endog, exog)
res = mod.fit()

# Shea's partial R² for each instrument
# For over-identified case (2 instruments, 1 endogenous), Shea's partial R²
# is the R² from regressing the endogenous var on all instruments minus
# the R² from regressing on all instruments EXCEPT the one of interest
# Equivalently: R²_shea_j = 1 - (RSS_full / RSS_reduced) * (df_reduced / df_full)

# Full model (both instruments)
rss_full = res.ssr
df_full = res.df_resid

# Reduced model without log_base_price
exog_no_base = exog.drop(columns=['log_base_price'])
mod_no_base = sm.OLS(endog, exog_no_base)
res_no_base = mod_no_base.fit()
rss_no_base = res_no_base.ssr
df_no_base = res_no_base.df_resid
shea_partial_r2_base = 1 - (rss_full / rss_no_base) * (df_no_base / df_full)

# Reduced model without log_precio_prom_otros_centros
exog_no_prom = exog.drop(columns=['log_precio_prom_otros_centros'])
mod_no_prom = sm.OLS(endog, exog_no_prom)
res_no_prom = mod_no_prom.fit()
rss_no_prom = res_no_prom.ssr
df_no_prom = res_no_prom.df_resid
shea_partial_r2_prom = 1 - (rss_full / rss_no_prom) * (df_no_prom / df_full)

# First-stage F-statistic (joint test of instruments)
# F = ((RSS_reduced - RSS_full) / q) / (RSS_full / df_full)
# where q = number of instruments excluded = 2
rss_reduced = sm.OLS(endog, exog.drop(columns=['log_base_price', 'log_precio_prom_otros_centros'])).fit().ssr
f_first_stage = ((rss_reduced - rss_full) / 2) / (rss_full / df_full)

print(f"R² = {res.rsquared:.4f}")
print(f"F overall = {res.fvalue:.2e}")
print(f"F first stage (joint) = {f_first_stage:.2e}")
print(f"Shea partial R² (log_base_price) = {shea_partial_r2_base:.6f}")
print(f"Shea partial R² (log_precio_prom_otros_centros) = {shea_partial_r2_prom:.6f}")

# Predicciones
fitted = res.fittedvalues.values
observed = endog.values

# Submuestreo para visualización
n_plot = min(5000, len(observed))
np.random.seed(42)
idx = np.random.choice(len(observed), n_plot, replace=False)
obs_sample = observed[idx]
fit_sample = fitted[idx]

# Correlación
corr = np.corrcoef(obs_sample, fit_sample)[0,1]

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(fit_sample, obs_sample, alpha=0.4, s=8, color='#3498DB', label=f'Observaciones (n={n_plot:,})')
# Línea 45°
min_val = min(obs_sample.min(), fit_sample.min())
max_val = max(obs_sample.max(), fit_sample.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Línea 45° (ajuste perfecto)')

ax.set_xlabel('Log precio ajustado (instrumentos + controles)', fontsize=11)
ax.set_ylabel('Log precio observado (checkout_price)', fontsize=11)
ax.set_title('IV Control Function — Primera etapa: Precio observado vs. ajustado\n'
             f'R² = {res.rsquared:.4f}, F 1ª etapa (conjunto) = {f_first_stage:.2e}, n = {len(df):,}',
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axis('equal')

plt.tight_layout()
save_fig(fig, 'iv_cf_panel_a_primera_etapa')