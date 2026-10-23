#!/usr/bin/env python3
"""
Figura 11: Elasticidad puntual RCS vs precio

USA datos precalculados de reports/certified/cf_elasticidad_curve.json (CF canónica)
y reports/diagnostics/bootstrap_rcs_confidence_bands.npy (RCS OLS bootstrap).
"""

import json
import numpy as np
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

# Cargar curva CF canónica (con bootstrap 5000 iters, seed=42, 77 clusters)
cf_curve_path = PROJECT_ROOT / "reports" / "certified" / "cf_elasticidad_curve.json"
with open(cf_curve_path, 'r', encoding='utf-8') as f:
    cf_data = json.load(f)

cf_price_grid = np.array(cf_data['grid']['price_grid'] if 'price_grid' in cf_data['grid'] else np.linspace(50, 800, 200))
cf_elasticidad = np.array([cf_data['results']['elasticidad_p50']] * len(cf_price_grid))  # placeholder

# Cargar RCS OLS bootstrap (2000 iters, seed=42, 77 clusters)
rcs_npy_path = PROJECT_ROOT / "reports" / "diagnostics" / "bootstrap_rcs_confidence_bands.npy"
rcs_data = np.load(rcs_npy_path, allow_pickle=True).item()

rcs_price_grid = np.asarray(rcs_data['price_grid'], dtype=float)
rcs_elasticidad = np.asarray(rcs_data['elasticidad_media'], dtype=float)

# Nudos (P5, P35, P65, P95)
knots_log_price = np.asarray(rcs_data['knots_log_price'], dtype=float)
knot_prices = np.exp(knots_log_price)

# OLS log-log con controles (referencia)
ols = -0.759

# Puntos de referencia: P5, P50 (mediana 296.82), P95
points = [
    (float(knot_prices[0]), float(np.interp(float(knot_prices[0]), rcs_price_grid, rcs_elasticidad))),
    (296.82, float(np.interp(296.82, rcs_price_grid, rcs_elasticidad))),
    (float(knot_prices[-1]), float(np.interp(float(knot_prices[-1]), rcs_price_grid, rcs_elasticidad)))
]

fig, ax = plt.subplots(figsize=(10, 5.8))

# Curva RCS OLS bootstrap
ax.plot(rcs_price_grid, rcs_elasticidad, linewidth=2.2, color='#E74C3C', label="Elasticidad puntual RCS (4 nudos, bootstrap 2000 iters)")
# OLS referencia
ax.axhline(ols, linestyle=":", linewidth=1.6, color='#3498DB', label=f"OLS log-log constante ({ols:.3f})")
# Nudos
for x in knot_prices:
    ax.axvline(x, linestyle="--", linewidth=0.9, alpha=0.35, color='gray')
# Puntos de referencia
xs = [x for x, _ in points]
ys = [y for _, y in points]
ax.scatter(xs, ys, s=55, zorder=5, color='black', label="Percentiles de referencia (P5, P50, P95)")
for label, (x, y) in zip(["P5", "P50 (mediana)", "P95"], points):
    ax.annotate(f"{label}: {y:.2f}", (x, y), xytext=(6, 8), textcoords="offset points", fontsize=9)
ax.axhline(0, linewidth=0.8, alpha=0.45)
ax.set_xlabel("Precio de checkout (unidades monetarias)", fontsize=11)
ax.set_ylabel("Elasticidad precio-demanda", fontsize=11)
ax.set_title("Elasticidad puntual vs. precio — RCS OLS (4 nudos P5/P35/P65/P95)\nBootstrap 2000 iteraciones, 77 clusters (center_id), seed=42", fontsize=11)
ax.grid(True, axis="both", alpha=0.25)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
save_fig(fig, 'elasticidad_puntual_rcs')