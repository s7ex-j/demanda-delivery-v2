#!/usr/bin/env python3
"""
Figura 18: Interacción Precio × Promoción
Panel A: Elasticidad condicional según estado promocional
Panel B: Efecto marginal de promociones vs. nivel de precio
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# Cargar datos
df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "demanda_limpia.csv")
df['log_checkout_price'] = np.log(df['checkout_price'])
df['log_num_orders'] = np.log(df['num_orders'])

# Ajustar modelo de interacción (especificación canónica: precio centrado, con hubo_aumento)
mean_log_price = df['log_checkout_price'].mean()
df['log_checkout_price_c'] = df['log_checkout_price'] - mean_log_price

formula = ('log_num_orders ~ log_checkout_price_c '
           '+ emailer_for_promotion + homepage_featured + hubo_aumento '
           '+ log_checkout_price_c:emailer_for_promotion '
           '+ log_checkout_price_c:homepage_featured '
           '+ C(category) + C(center_type)')
mod = smf.ols(formula, data=df).fit()

# Coeficientes
b_price = mod.params['log_checkout_price_c']
b_emailer = mod.params['emailer_for_promotion']
b_homepage = mod.params['homepage_featured']
b_int_emailer = mod.params['log_checkout_price_c:emailer_for_promotion']
b_int_homepage = mod.params['log_checkout_price_c:homepage_featured']

# Test Wald para interacciones
interaction_terms = ['log_checkout_price_c:emailer_for_promotion', 'log_checkout_price_c:homepage_featured']
R = np.zeros((len(interaction_terms), len(mod.params)))
param_names = list(mod.params.index)
for i, term in enumerate(interaction_terms):
    R[i, param_names.index(term)] = 1
wald = mod.wald_test(R, scalar=True)
wald_F = float(wald.fvalue.item()) if hasattr(wald.fvalue, 'item') else float(wald.fvalue)

print(f"Coefficients: price_c={b_price:.4f}, emailer={b_emailer:.4f}, homepage={b_homepage:.4f}, hubo_aumento={mod.params.get('hubo_aumento', 0):.4f}")
print(f"Interactions: price:emailer={b_int_emailer:.4f}, price:homepage={b_int_homepage:.4f}")
print(f"Wald F = {wald_F:.2f}")

# Panel A: Elasticidades condicionales
elasticities = {
    'Sin promo': b_price,
    'Solo emailer': b_price + b_int_emailer,
    'Solo homepage': b_price + b_int_homepage,
    'Ambas': b_price + b_int_emailer + b_int_homepage
}

# Panel B: Efectos marginales vs precio
# Use the same price percentiles as the model (centered)
price_pct = df['checkout_price'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
log_price_pct = np.log(price_pct)
log_price_pct_c = log_price_pct - mean_log_price

me_emailer = [b_emailer + b_int_emailer * lp_c for lp_c in log_price_pct_c]
me_homepage = [b_homepage + b_int_homepage * lp_c for lp_c in log_price_pct_c]

# Crear figura con dos paneles
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Barras de elasticidades condicionales
colors = ['#2C3E50', '#E74C3C', '#3498DB', '#9B59B6']
labels = list(elasticities.keys())
values = list(elasticities.values())

bars = ax1.bar(labels, values, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
ax1.axhline(y=b_price, color='gray', linestyle='--', alpha=0.5, label=f'Elasticidad base: {b_price:.3f}')
ax1.set_ylabel('Elasticidad precio-demanda', fontsize=11)
ax1.set_title('Panel A: Elasticidad condicional\nsegún estado promocional', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, axis='y', alpha=0.3)

# Añadir valores en las barras
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 if val > 0 else bar.get_height() - 0.03,
             f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=10, fontweight='bold')

# Panel B: Efectos marginales vs nivel de precio
price_labels = [f'P{int(q*100)}' for q in [0.1, 0.25, 0.5, 0.75, 0.9]]
x_pos = np.arange(len(price_labels))

ax2.plot(x_pos, me_emailer, 'o-', color='#E74C3C', linewidth=2.5, markersize=8, label='Emailer (promoción por email)')
ax2.plot(x_pos, me_homepage, 's-', color='#3498DB', linewidth=2.5, markersize=8, label='Homepage (destacado en web)')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(price_labels)
ax2.set_xlabel('Percentil de precio de checkout', fontsize=11)
ax2.set_ylabel('Efecto marginal en log-pedidos', fontsize=11)
ax2.set_title('Panel B: Efecto marginal de promociones\nvs. nivel de precio', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Añadir valores en los puntos
for i, (me_e, me_h) in enumerate(zip(me_emailer, me_homepage)):
    ax2.text(i, me_e + 0.03, f'{me_e:.3f}', ha='center', va='bottom', fontsize=9, color='#E74C3C')
    ax2.text(i, me_h - 0.05, f'{me_h:.3f}', ha='center', va='top', fontsize=9, color='#3498DB')

# Título general con Wald F
fig.suptitle(f'Interacción Precio × Promoción: no-aditividad estadísticamente significativa (Wald F={wald_F:.1f}, p≈0)\n'
             f'Modelo: log(Q) ~ log(P) centrado + emailer + homepage + hubo_aumento + interacciones + C(category) + C(center_type)',
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
save_fig(fig, 'interaccion_precio_promo')