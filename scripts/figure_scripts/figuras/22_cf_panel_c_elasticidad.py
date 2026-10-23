#!/usr/bin/env python3
"""
Figura 22 (V3.2.5): Elasticidad puntual CF-RCS con coeficientes PUNTUALES y banda bootstrap
percentil 95% punto a punto (réplicas brutas de bootstrap_cf_canonical_replicas.npz),
comparada con la RCS-OLS (sin v_hat) estimada en la MISMA muestra IV (N=456,544).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from modeling import construir_instrumentos  # noqa: E402
from rcs_utils import build_canonical_rcs_design, apply_rcs_design  # noqa: E402

CTRL = "emailer_for_promotion + homepage_featured + hubo_aumento + C(category) + C(center_type)"
rep = np.load(ROOT / "reports" / "certified" / "bootstrap_cf_canonical_replicas.npz")
grid = rep["price_grid"]
curves = rep["elasticity_curves"]
lo, hi = np.nanpercentile(curves, 2.5, axis=0), np.nanpercentile(curves, 97.5, axis=0)

d = construir_instrumentos(pd.read_csv(ROOT / "data" / "processed" / "demanda_limpia.csv"))
d = d.dropna(subset=["log_precio_prom_otros_centros"]).copy()
fs = smf.ols(f"log_checkout_price ~ log_base_price + log_precio_prom_otros_centros + {CTRL}", d).fit()
d["v_hat"] = fs.resid
di, knots = build_canonical_rcs_design(d["log_checkout_price"].to_numpy())
S = apply_rcs_design(di, d["log_checkout_price"].to_numpy())
for c in S.columns:
    d[c] = S[c].to_numpy()
cf = smf.ols(f"log_num_orders ~ 1 + rcs_1 + rcs_2 + rcs_3 + v_hat + {CTRL}", d).fit()
ro = smf.ols(f"log_num_orders ~ 1 + rcs_1 + rcs_2 + rcs_3 + {CTRL}", d).fit()


def elast(model, p):
    h = 1e-5
    lp = np.log(np.atleast_1d(p))
    dB = (np.asarray(apply_rcs_design(di, lp + h)) - np.asarray(apply_rcs_design(di, lp - h))) / (2 * h)
    return dB @ np.array([model.params[c] for c in ["rcs_1", "rcs_2", "rcs_3"]])


e_cf, e_ro = elast(cf, grid), elast(ro, grid)
pm, pmed = d["checkout_price"].mean(), d["checkout_price"].median()

fig, ax = plt.subplots(figsize=(10, 5.8))
ax.fill_between(grid, lo, hi, color="#1ABC9C", alpha=0.25, label="IC 95% bootstrap CF (5000 réplicas, centro)")
ax.plot(grid, e_cf, color="#117A65", lw=2.2, label="Control Function (RCS + v̂), coef. puntuales")
ax.plot(grid, e_ro, color="#E74C3C", lw=1.8, ls="--", label="RCS-only (OLS, misma muestra)")
ax.axhline(-0.759, ls=":", lw=1.4, color="#3498DB", label="OLS log-log constante (-0.759)")
for k in np.exp(knots):
    ax.axvline(k, ls="--", lw=0.8, alpha=0.3, color="gray")
for p, name, col in [(pm, "media", "#117A65"), (pmed, "mediana", "#F39C12")]:
    v = elast(cf, p)[0]
    ax.scatter([p], [v], s=60, zorder=5, color=col, label=f"CF en {name} ({p:.2f}) = {v:.4f}")
ax.set_xlabel("Precio de checkout (unidades monetarias)")
ax.set_ylabel("Elasticidad precio-demanda")
ax.set_title("Elasticidad puntual: Control Function vs. RCS-only (N = 456,544)\n"
             "RCS canónica P5/P35/P65/P95; banda percentil 95% punto a punto (no simultánea)", fontsize=11)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8.5, loc="upper right")
fig.tight_layout()
for out in [ROOT / "Informe" / "figuras"]:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "iv_cf_panel_c_elasticidad_puntual.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(json.dumps({"cf_media": float(elast(cf, pm)[0]), "cf_mediana": float(elast(cf, pmed)[0]),
                  "rcs_media": float(elast(ro, pm)[0]), "rcs_mediana": float(elast(ro, pmed)[0])}))
