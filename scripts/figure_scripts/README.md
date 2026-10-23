# Scripts de figuras

Cada script de `figuras/` genera una figura del informe en `Informe/figuras/` (la única copia
versionada). `FIGURE_MAP.csv` relaciona número de figura, script y archivo.

```bash
python scripts/figure_scripts/generar_todas_las_figuras.py   # requiere data/processed/demanda_limpia.csv
```

Los scripts que muestran resultados bootstrap leen los artefactos de `reports/certified/` y
`reports/diagnostics/` (no re-ejecutan los bootstrap), salvo la Figura 8, que repite el bootstrap i.i.d. de 1 000 réplicas (unos minutos). Se puede pasar una lista de números para generar solo esas figuras: `... generar_todas_las_figuras.py 09 13`. `common.py` define las rutas del proyecto.
