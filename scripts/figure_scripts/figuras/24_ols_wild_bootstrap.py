#!/usr/bin/env python3
"""
Figura 24: OLS Wild Cluster Bootstrap

USA datos precalculados de reports/diagnostics/wild_cluster_bootstrap_results.npy
El NPY SOLO tiene estadísticas resumidas (media, std, IC), NO réplicas brutas.
Visualización honesta: densidad normal teórica con etiqueta explícita.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

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

# Cargar resultados precalculados
npy_path = PROJECT_ROOT / "reports" / "diagnostics" / "wild_cluster_bootstrap_results.npy"
if not npy_path.exists():
    raise FileNotFoundError(f"No existe {npy_path}")

data = np.load(npy_path, allow_pickle=True).item()

bs_mean = data['elasticidad_media']
bs_std = data['elasticidad_std']
bs_ci_inf = data['ic_95_inferior']
bs_ci_sup = data['ic_95_superior']
coef_base = data['coef_base']
n_iter = data['n_iteraciones_validas']
clusters = data['clusters']
seed = data['semilla']
metodo = data['metodo']

print(f"  Cargado: media={bs_mean:.4f}, std={bs_std:.4f}, IC=[{bs_ci_inf:.4f}, {bs_ci_sup:.4f}]")
print(f"  {n_iter} iteraciones, {clusters} clusters, seed={seed}")
print(f"  Método: {metodo}")

# IMPORTANTE: No hay réplicas brutas en el NPY
# Visualización honesta: densidad normal teórica

fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(bs_mean - 4*bs_std, bs_mean + 4*bs_std, 200)
y = norm.pdf(x, bs_mean, bs_std)
ax.plot(x, y, color='#3498DB', linewidth=2.5,
        label=f'Densidad normal teórica (aprox. desde estadísticas resumidas)')
ax.fill_between(x, y, alpha=0.3, color='#3498DB')

ax.axvline(coef_base, color='black', linestyle='--', linewidth=2,
           label=f'Coef. analítico OLS (controles): {coef_base:.4f}')
ax.axvline(bs_mean, color='#2E86C1', linestyle='-', linewidth=2,
           label=f'Media bootstrap: {bs_mean:.4f}')
ax.axvspan(bs_ci_inf, bs_ci_sup, alpha=0.2, color='green',
           label=f'IC 95% bootstrap: [{bs_ci_inf:.3f}, {bs_ci_sup:.3f}]')

ax.set_xlabel('Elasticidad (log_checkout_price)', fontsize=11)
ax.set_ylabel('Densidad', fontsize=11)
ax.set_title(f'OLS Log-Log con controles: Wild Cluster Bootstrap\n'
             f'{n_iter} iteraciones, {clusters} clusters (center_id), seed={seed}\n'
             f'NOTA: Curva teórica aproximada (réplicas brutas no almacenadas en NPY)\n'
             f'EE bootstrap ({bs_std:.4f}) valida inferencia multiway cluster (0.0430, ratio {bs_std/0.0430:.2f}×)',
             fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
save_fig(fig, 'ols_wild_bootstrap')

print("\nNOTA: Esta figura usa aproximación normal teórica desde estadísticas resumidas.")
print("Para histograma real, el wild bootstrap debe guardar las réplicas brutas.")