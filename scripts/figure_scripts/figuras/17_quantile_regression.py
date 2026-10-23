#!/usr/bin/env python3
"""
Figura 17: Elasticidad por Regresión Cuantil

USA valores reales de reports/diagnostics/tabla_maestra_comparacion.csv
NO usa función fantasma get_model_values()
"""

import pandas as pd
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

# Cargar tabla maestra
comp_csv = PROJECT_ROOT / "reports" / "diagnostics" / "tabla_maestra_comparacion.csv"
df = pd.read_csv(comp_csv)

# Filtrar regresión cuantil
qr_rows = df[df['modelo'].str.contains('Quantile', na=False)].copy()
print(f"  Filas QR encontradas: {len(qr_rows)}")

# Extraer tau y elasticidad (manejar τ unicode)
qr_rows['tau'] = qr_rows['modelo'].str.extract(r'tau=([0-9.]+)')[0]
qr_rows['tau'] = qr_rows['tau'].fillna(qr_rows['modelo'].str.extract(r'τ=([0-9.]+)')[0])
qr_rows['tau'] = qr_rows['tau'].astype(float)
qr_rows = qr_rows.sort_values('tau')

taus = qr_rows['tau'].values
elasticidades = qr_rows['estimacion'].values

print(f"  Taus: {taus}")
print(f"  Elasticidades: {elasticidades}")

# OLS referencia
ols_ref = -0.759

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(taus, elasticidades, 'o-', color='#E74C3C', linewidth=2, markersize=8, label='Regresión cuantil')
ax.axhline(ols_ref, linestyle='--', linewidth=1.5, color='#3498DB', label=f'OLS log-log controles ({ols_ref:.3f})')
ax.axhline(-1, linestyle=':', linewidth=1, color='gray', label='Elasticidad unitaria (-1)')

ax.set_xlabel('Cuantil (τ)', fontsize=11)
ax.set_ylabel('Elasticidad precio-demanda', fontsize=11)
ax.set_title('Elasticidad estimada por regresión cuantil\nFuente: Tabla maestra comparativa (diagnósticos)', fontsize=11)
ax.set_xticks([0.1, 0.25, 0.5, 0.75, 0.9])
ax.set_xlim(0.05, 0.95)
ax.grid(True, axis="both", alpha=0.25)
ax.legend(loc='best', fontsize=9)

# Etiquetas de valores
for tau, eta in zip(taus, elasticidades):
    ax.annotate(f'{eta:.3f}', (tau, eta), xytext=(0, 10), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

fig.tight_layout()
save_fig(fig, 'quantile_regression_elasticidad')