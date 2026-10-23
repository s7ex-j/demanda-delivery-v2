"""Regenera las 24 figuras del informe en Informe/figuras/ (ver FIGURE_MAP.csv).

Uso (desde cualquier carpeta): python scripts/figure_scripts/generar_todas_las_figuras.py
Opcional: pasar números de figura para generar solo algunas, p. ej. ... 09 13
"""
import os
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

FIGS = Path(__file__).resolve().parent / "figuras"
sys.path.insert(0, str(FIGS))
os.chdir(FIGS)

solo = set(sys.argv[1:])
for p in sorted(FIGS.glob("[0-9]*.py")):
    if solo and p.name[:2] not in solo:
        continue
    print(f"\n=== {p.name} ===", flush=True)
    runpy.run_path(str(p), run_name="__main__")
