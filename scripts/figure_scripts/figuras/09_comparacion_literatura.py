"""Figura 9: elasticidades de la literatura verificada (Grupos A y B) frente al presente estudio.
Lee data/processed/tabla_comparativa_literatura.csv (solo filas con elasticidad numérica)."""
from common import *

lit = pd.read_csv(DATA.parent / 'tabla_comparativa_literatura.csv')
d = lit[lit.grupo.isin(['A', 'B'])].dropna(subset=['elasticidad']).copy()
rows = []
for _, r in d.iterrows():
    if r.autor.startswith('Zhang'):
        rows += [(f'{r.autor} ({r.anio}) online', -0.87, None, None, 'B'),
                 (f'{r.autor} ({r.anio}) offline', -1.74, None, None, 'B')]
    elif r.autor.startswith('Bumagat'):
        rows.append((f'{r.autor} ({r.anio}) pico / no pico', None, r.ic_inf, r.ic_sup, 'A'))
    else:
        rows.append((f'{r.autor} ({r.anio})', r.elasticidad, r.ic_inf, r.ic_sup, r.grupo))
rows.sort(key=lambda x: x[1] if x[1] is not None else (x[2] + x[3]) / 2)

fig, ax = plt.subplots(figsize=(9.5, 5.5))
col = {'A': '#2c7fb8', 'B': '#8c8c8c'}
for y, (lab, v, lo, hi, g) in enumerate(rows):
    estudio = lab.startswith('Este estudio')
    c = '#d95f02' if estudio else col[g]
    if v is None:
        ax.plot([lo, hi], [y, y], color=c, lw=4, solid_capstyle='round')
    else:
        if lo == lo and hi == hi and lo is not None:  # no NaN
            ax.plot([lo, hi], [y, y], color=c, lw=1.5)
        ax.scatter(v, y, s=110 if estudio else 70, marker='D' if estudio else 'o', color=c, zorder=3)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.axvline(-1, ls='--', color='gray', lw=1)
ax.axvline(-0.759, ls=':', color='#d95f02', lw=1.3)
ax.set_xlabel('Elasticidad precio-demanda')
ax.set_title('Literatura verificada vs. este estudio (azul: Grupo A; gris: Grupo B; naranja: este estudio)', fontsize=10)
apply_grid(ax)
fig.tight_layout()
save(fig, 'comparacion_elasticidad_literatura.png')
