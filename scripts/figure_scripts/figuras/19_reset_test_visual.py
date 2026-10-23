#!/usr/bin/env python3
"""
Figura 19: Test RESET (Ramsey) - Evidencia Visual

Valores reales del paper (verificados):
- F = 153.07, 2 gl num, 456526 gl den, p < 1e-300
- Términos: potencias de valores ajustados (ŷ², ŷ³)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import linear_reset
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

# Cargar datos y estimar modelo OLS log-log con controles (formula API para consistencia con paper)
df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "demanda_limpia.csv")
df['log_checkout_price'] = np.log(df['checkout_price'])
df['log_num_orders'] = np.log(df['num_orders'])

formula = ('log_num_orders ~ log_checkout_price + emailer_for_promotion '
           '+ homepage_featured + hubo_aumento + C(category) + C(center_type)')
model = smf.ols(formula, data=df).fit()

# RESET test real (Ramsey 1969): potencias de valores ajustados
reset_result = linear_reset(model, power=3, test_type='fitted', use_f=True)
F_stat = float(reset_result.statistic)
df_num = int(reset_result.df_num)
df_den = int(reset_result.df_denom)
p_val = reset_result.pvalue

print(f"RESET F = {F_stat:.2f}, df = ({df_num}, {df_den}), p = {p_val:.2e}")

# Obtener residuos y valores ajustados REALES
fitted = model.fittedvalues.values
residuals = model.resid.values

# Submuestreo para visualización (5000 puntos)
n_plot = min(5000, len(fitted))
np.random.seed(42)
sample_idx = np.random.choice(len(fitted), n_plot, replace=False)
fitted_sample = fitted[sample_idx]
residuals_sample = residuals[sample_idx]

# Panel B: residuos vs log-precio (datos reales)
log_price_sample = df['log_checkout_price'].values[sample_idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel A: Residuos vs Valores ajustados (datos reales)
ax1.scatter(fitted_sample, residuals_sample, alpha=0.4, s=8, color='#3498DB')
ax1.axhline(0, color='red', linestyle='--', linewidth=1)
ax1.set_xlabel('Valores ajustados (log num_orders)', fontsize=10)
ax1.set_ylabel('Residuos', fontsize=10)
ax1.set_title('Panel A: Residuos vs. Valores ajustados (datos reales)', fontsize=11)
ax1.grid(True, alpha=0.3)

# Panel B: Residuos vs log-precio (datos reales)
ax2.scatter(log_price_sample, residuals_sample, alpha=0.4, s=8, color='#E74C3C')
ax2.axhline(0, color='red', linestyle='--', linewidth=1)
ax2.set_xlabel('Log precio de checkout', fontsize=10)
ax2.set_ylabel('Residuos', fontsize=10)
ax2.set_title('Panel B: Residuos vs. Log-precio (datos reales)', fontsize=11)
ax2.grid(True, alpha=0.3)

p_str = f"{p_val:.2e}" if p_val >= 1e-10 else "< 1e-10"
fig.suptitle(f'Test RESET (Ramsey): Evidencia visual de misspecification funcional\n'
             f'$F({df_num}, {df_den:,}) = {F_stat:.2f}$, $p {p_str}$\n'
             f'Modelo OLS log-log con controles, N = {len(df):,}',
             fontsize=12)

plt.tight_layout()
save_fig(fig, 'reset_test_visual')
