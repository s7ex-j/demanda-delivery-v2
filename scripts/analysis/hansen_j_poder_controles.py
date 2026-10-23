#!/usr/bin/env python3
"""
Diagnóstico descriptivo de la sensibilidad del rechazo de Hansen J al tamaño muestral,
sobre la ESPECIFICACIÓN IV PRINCIPAL CON CONTROLES (V3.2.5).

Submuestreo aleatorio sin reemplazo (50 repeticiones por tamaño, seed=42), IVGMM robusto.
Salida: reports/diagnostics/hansen_j_poder_controles.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import IVGMM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from modeling import construir_instrumentos  # noqa: E402

F = ("log_num_orders ~ 1 + emailer_for_promotion + homepage_featured + hubo_aumento + C(category) + C(center_type) "
     "+ [log_checkout_price ~ log_base_price + log_precio_prom_otros_centros]")
SIZES = [2000, 5000, 10000, 20000, 50000, 100000, 200000]


def main(reps: int = 50, seed: int = 42):
    df = construir_instrumentos(pd.read_csv(ROOT / "data" / "processed" / "demanda_limpia.csv"))
    d = df.dropna(subset=["log_precio_prom_otros_centros"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    full = IVGMM.from_formula(F, d).fit(cov_type="robust").j_stat
    out = {"especificacion": F, "n_total": int(len(d)), "j_muestra_completa": float(full.stat),
           "p_muestra_completa": float(full.pval), "n_repeticiones": reps, "seed": seed, "resultados_por_n": {}}
    for n in SIZES:
        js, ps = [], []
        for _ in range(reps):
            idx = rng.choice(len(d), size=n, replace=False)
            try:
                j = IVGMM.from_formula(F, d.iloc[idx]).fit(cov_type="robust").j_stat
                js.append(float(j.stat)); ps.append(float(j.pval))
            except Exception:
                pass
        js, ps = np.array(js), np.array(ps)
        out["resultados_por_n"][str(n)] = {"j_media": float(js.mean()), "tasa_rechazo_5pct": float((ps < 0.05).mean()),
                                           "n_validas": int(js.size)}
        print(n, out["resultados_por_n"][str(n)], flush=True)
    path = ROOT / "reports" / "diagnostics" / "hansen_j_poder_controles.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
