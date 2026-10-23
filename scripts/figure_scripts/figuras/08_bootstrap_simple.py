"""Figura 8: bootstrap i.i.d. de la elasticidad del modelo base (1000 réplicas, semilla 42),
usando la misma función que reporta la Tabla 11 (src/modeling.py::bootstrap_elasticidad)."""
from common import *
import sys
sys.path.insert(0, str(ROOT / 'src'))
from modeling import bootstrap_elasticidad

r = bootstrap_elasticidad(load_df(), n_iteraciones=1000, semilla=42)
vals = r['coeficientes']
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(vals, bins=45)
ax.axvline(r['elasticidad_media'], ls='--', lw=1.5, color='red', label=f"Media = {r['elasticidad_media']:.4f}")
for q in (r['ic_95_inferior'], r['ic_95_superior']):
    ax.axvline(q, ls=':', lw=1.2, color='blue')
ax.set_xlabel('Elasticidad bootstrap'); ax.set_ylabel('Frecuencia')
ax.set_title(f"Bootstrap i.i.d. del modelo base (1000 réplicas): IC 95% [{r['ic_95_inferior']:.4f}, {r['ic_95_superior']:.4f}]")
apply_grid(ax); ax.legend(); fig.tight_layout(); save(fig, 'bootstrap_elasticidad_distribucion.png')
print({k: v for k, v in r.items() if k != 'coeficientes'})
