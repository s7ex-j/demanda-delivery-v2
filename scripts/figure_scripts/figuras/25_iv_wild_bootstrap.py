#!/usr/bin/env python3
"""
Figura 24 (V3.2.5): Wild cluster bootstrap del IV 2SLS PRINCIPAL (con controles).
Histograma de las réplicas brutas en reports/certified/wild_bootstrap_iv_2sls_controles.npz
(generadas por scripts/vm/wild_bootstrap_iv_2sls_controles.py).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
coefs = np.load(ROOT / "reports" / "certified" / "wild_bootstrap_iv_2sls_controles.npz")["coefs"]
meta = json.loads((ROOT / "reports" / "certified" / "wild_bootstrap_iv_2sls_controles.json").read_text(encoding="utf-8"))
coefs = coefs[np.isfinite(coefs)]
lo, hi = np.percentile(coefs, [2.5, 97.5])

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(coefs, bins=50, density=True, color="#9B59B6", alpha=0.6, label=f"Réplicas bootstrap (n={coefs.size})")
ax.axvline(meta["coef_base"], color="black", ls="--", lw=2, label=f"Coef. 2SLS con controles: {meta['coef_base']:.4f}")
ax.axvline(coefs.mean(), color="#6C3483", lw=2, label=f"Media bootstrap: {coefs.mean():.4f}")
ax.axvspan(lo, hi, alpha=0.15, color="green", label=f"IC 95% percentil: [{lo:.3f}, {hi:.3f}]")
ax.set_xlabel("Coeficiente log_checkout_price")
ax.set_ylabel("Densidad")
ax.set_title(f"IV 2SLS (Hausman-Nevo, con controles): Wild Cluster Bootstrap\n"
             f"{coefs.size} réplicas, {meta['clusters']} clusters (center_id), seed={meta['semilla']}, "
             f"DE = {coefs.std(ddof=1):.4f}", fontsize=11)
ax.legend(fontsize=9)
fig.tight_layout()
for out in [ROOT / "Informe" / "figuras"]:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "iv_wild_bootstrap.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"media={coefs.mean():.4f} sd={coefs.std(ddof=1):.4f} IC=[{lo:.4f},{hi:.4f}]")
