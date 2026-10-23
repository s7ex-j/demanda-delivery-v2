from common import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
df=load_df(); m=smf.ols('log_num_orders ~ log_checkout_price + emailer_for_promotion + homepage_featured + hubo_aumento + C(category)+C(center_type)',data=df).fit(); fig=sm.qqplot(m.resid.sample(min(5000,len(m.resid)),random_state=42),line='45',fit=True); fig.set_size_inches(6,6); plt.title('Q-Q plot de los residuos del modelo log-log con controles'); save(fig,'qqplot_residuos_modelo_loglog_controles.png')