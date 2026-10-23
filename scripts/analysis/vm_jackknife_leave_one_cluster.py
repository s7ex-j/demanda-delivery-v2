"""Jackknife leave-one-center-out del OLS principal, con checkpoint por centro."""
from __future__ import annotations
import sys, numpy as np, argparse
from pathlib import Path
import statsmodels.formula.api as smf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vm')); from vm_common import OUT_DIR,load_data,save_result
OUT='jackknife_leave_one_cluster_results.npy'
FORM='log_num_orders ~ log_checkout_price + emailer_for_promotion + homepage_featured + hubo_aumento + C(category) + C(center_type)'
def run():
 df=load_data(); full=smf.ols(FORM,data=df).fit(); cf=float(full.params['log_checkout_price']); co={}; nobs={}
 centers=np.unique(df.center_id)
 for c in centers:
  sub=df[df.center_id!=c]; m=smf.ols(FORM,data=sub).fit(); co[str(c)]=float(m.params['log_checkout_price']); nobs[str(c)]=int(len(df)-len(sub))
 arr=np.array(list(co.values())); dif={k:abs(v-cf) for k,v in co.items()}; inf=max(dif,key=dif.get); r={'coef_completo':cf,'coefs_jackknife_por_centro':co,'n_obs_excluidas_por_centro':nobs,'coef_jk_media':float(arr.mean()),'coef_jk_std':float(arr.std()),'coef_jk_min':float(arr.min()),'coef_jk_max':float(arr.max()),'centro_mas_influyente':inf,'diferencia_maxima_absoluta':float(dif[inf]),'n_centros_evaluados':len(arr)}; save_result(r,OUT); return r
def main(): print(run())
if __name__=='__main__': main()
