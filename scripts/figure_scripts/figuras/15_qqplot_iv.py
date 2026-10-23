from common import *
import sys
sys.path.insert(0,str(ROOT/'src'))
from modeling import construir_instrumentos, modelo_iv_2sls
import statsmodels.api as sm

df=construir_instrumentos(load_df()); res,dfiv=modelo_iv_2sls(df); resid=np.asarray(res.resids).reshape(-1); resid=pd.Series(resid).sample(min(5000,len(resid)),random_state=42)
fig=sm.qqplot(resid,line='45',fit=True); fig.set_size_inches(6,6); plt.title('Q-Q plot — residuos 2SLS (con controles), n=5000'); save(fig,'qqplot_iv_2sls_residuos.png')
