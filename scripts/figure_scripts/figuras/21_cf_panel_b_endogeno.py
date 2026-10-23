#!/usr/bin/env python3
"""
Figura 21: IV Control Function - Componente Endógeno (v_hat)
Distribución de los residuos de la primera etapa (v_hat) con curva normal teórica superpuesta
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
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

# Residuos de primera etapa = v_hat
v_hat = res.resid.values

print(f"v_hat: media={v_hat.mean():.6f}, std={v_hat.std():.4f}, n={len(v_hat):,}")

# Submuestreo para histograma
n_plot = min(10000, len(v_hat))
np.random.seed(42)
idx = np.random.choice(len(v_hat), n_plot, replace=False)
v_hat_sample = v_hat[idx]

fig, ax = plt.subplots(figsize=(10, 6))

# Histograma de v_hat real
counts, bins, patches = ax.hist(v_hat_sample, bins=50, density=True, alpha=0.6, color='#3498DB',
                                edgecolor='white', label=f'Residuos 1ª etapa (v̂) n={n_plot:,}')

# Curva normal teórica
x = np.linspace(v_hat_sample.min(), v_hat_sample.max(), 200)
y = norm.pdf(x, v_hat_sample.mean(), v_hat_sample.std())
ax.plot(x, y, 'r-', linewidth=2.5, label=f'Normal teórica (μ={v_hat_sample.mean():.4f}, σ={v_hat_sample.std():.4f})')

ax.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Cero')
ax.set_xlabel('Residuo de primera etapa (v̂)', fontsize=11)
ax.set_ylabel('Densidad', fontsize=11)
ax.set_title('Control Function — Componente endógeno: Distribución de v̂ (residuos 1ª etapa)\n'
             f'Media ≈ {v_hat_sample.mean():.4f}, Desv. est. ≈ {v_hat_sample.std():.4f}, R² 1ª etapa = {res.rsquared:.4f}, n = {len(df):,}',
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_fig(fig, 'iv_cf_panel_b_componente_endogeno')