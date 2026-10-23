"""Figura 16: relevancia de instrumentos (primera etapa con los controles del modelo principal).
Panel A: precio observado vs. predicho. Panel B: R2 parciales (Shea conjunto y contribución única)."""
from common import *
import sys
sys.path.insert(0, str(ROOT / "src"))
from modeling import construir_instrumentos
import statsmodels.formula.api as smf

CTRL = "emailer_for_promotion + homepage_featured + hubo_aumento + C(category) + C(center_type)"
df = construir_instrumentos(load_df()).dropna(subset=['log_precio_prom_otros_centros']).copy()
full = smf.ols(f'log_checkout_price ~ log_base_price + log_precio_prom_otros_centros + {CTRL}', data=df).fit()
r0 = smf.ols(f'log_checkout_price ~ {CTRL}', data=df).fit().ssr
rB = smf.ols(f'log_checkout_price ~ log_precio_prom_otros_centros + {CTRL}', data=df).fit().ssr
rP = smf.ols(f'log_checkout_price ~ log_base_price + {CTRL}', data=df).fit().ssr
shea = 1 - full.ssr / r0
uP, uB = 1 - full.ssr / rP, 1 - full.ssr / rB

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1.4, 1]})
idx = df.sample(min(10000, len(df)), random_state=42).index
a.scatter(full.fittedvalues.loc[idx], df.loc[idx, 'log_checkout_price'], s=6, alpha=.3)
lo = min(full.fittedvalues.loc[idx].min(), df.loc[idx, 'log_checkout_price'].min())
hi = max(full.fittedvalues.loc[idx].max(), df.loc[idx, 'log_checkout_price'].max())
a.plot([lo, hi], [lo, hi], 'r--', lw=1.2)
a.set_xlabel('Log precio predicho (instrumentos + controles)'); a.set_ylabel('Log precio observado')
a.set_title(f'A. Primera etapa: R² = {full.rsquared:.4f}, n = {len(df):,}'); apply_grid(a)
labels = ['Conjunto (Shea)', 'Único: otros centros', 'Único: base_price']
vals = [shea, uP, uB]
bars = b.bar(labels, vals, color=['#2E86C1', '#85C1E9', '#AED6F1'])
for r_, v in zip(bars, vals):
    b.text(r_.get_x() + r_.get_width() / 2, v + 0.01, f'{v:.4f}', ha='center', fontsize=9)
b.set_ylim(0, 1); b.set_ylabel('R² parcial'); b.set_title('B. R² parcial de los instrumentos'); apply_grid(b)
b.tick_params(axis='x', labelsize=8)
fig.tight_layout(); save(fig, 'iv_primera_etapa_instrumentos.png')
print(f"Shea={shea:.4f} unico_otros={uP:.4f} unico_base={uB:.4f}")
