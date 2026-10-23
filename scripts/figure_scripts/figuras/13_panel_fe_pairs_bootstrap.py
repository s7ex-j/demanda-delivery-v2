"""Figura 13: distribución de las réplicas del pairs cluster bootstrap del panel FE.
Lee reports/diagnostics/pairs_bootstrap_panel_fe_replicas.npz
(scripts/analysis/vm_pairs_bootstrap_panel_fe.py; 1 000 réplicas, semilla 42)."""
from common import *

z = np.load(REPORTS / 'diagnostics' / 'pairs_bootstrap_panel_fe_replicas.npz')
v = z['coefs'][np.isfinite(z['coefs'])]
lo, hi = np.percentile(v, [2.5, 97.5])
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(v, bins=45, color='#4C72B0', alpha=0.8)
ax.axvline(float(z['coef_base']), color='black', ls='--', lw=1.5, label=f"Estimación puntual = {float(z['coef_base']):.4f}")
ax.axvline(v.mean(), color='red', lw=1.5, label=f'Media bootstrap = {v.mean():.4f}')
ax.axvspan(lo, hi, color='green', alpha=0.12, label=f'IC 95% percentil [{lo:.4f}, {hi:.4f}]')
ax.set_xlabel('Elasticidad within (panel FE)'); ax.set_ylabel('Frecuencia')
ax.set_title(f'Pairs cluster bootstrap por centro: {v.size} réplicas, 77 centros, semilla {int(z["seed"])}')
apply_grid(ax); ax.legend(fontsize=9); fig.tight_layout()
save(fig, 'panel_fe_pairs_bootstrap.png')
print(f'media={v.mean():.4f} de={v.std(ddof=1):.4f} IC=[{lo:.4f},{hi:.4f}] n={v.size}')
