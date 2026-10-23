"""Figura 14: Q-Q plot de los residuos del panel con efectos fijos de dos vías
(src/modeling.py::modelo_panel_efectos_fijos), la misma submuestra de 5 000 que usa
diagnostics.shapiro_wilk_submuestreado (notebook 03b; semilla 42)."""
from common import *
import sys
import statsmodels.api as sm
from scipy import stats
sys.path.insert(0, str(ROOT / 'src'))
from modeling import modelo_panel_efectos_fijos

r = modelo_panel_efectos_fijos(load_df())
res = np.random.default_rng(42).choice(np.asarray(r.resids), size=5000, replace=False)
w, p = stats.shapiro(res)
fig = sm.qqplot(res, line='45', fit=True)
fig.set_size_inches(6, 6)
plt.title('Q-Q plot: residuos del panel con efectos fijos de dos vías (n = 5 000)')
save(fig, 'qqplot_panel_fe_residuos.png')
print(f'Shapiro-Wilk W={w:.4f} p={p:.3g}')
