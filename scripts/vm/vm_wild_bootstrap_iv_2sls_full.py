"""Wild cluster bootstrap IV 2SLS: reestima ambas etapas; checkpoints reanudables."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
import numpy as np, statsmodels.api as sm
from patsy import dmatrices
sys.path.insert(0,str(Path(__file__).resolve().parent)); from vm_common import OUT_DIR,load_data,load_checkpoint,save_npz_checkpoint,save_result
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src')); from modeling import construir_instrumentos
# Output now goes to reports/certified
REPORTS_CERTIFIED = Path(__file__).resolve().parents[2] / 'reports' / 'certified'
OUT='wild_bootstrap_iv_2sls_full_results.npy'; CHECK=OUT_DIR/'wild_bootstrap_iv_2sls_full_results.checkpoint.pkl'
FS='log_checkout_price ~ log_base_price + log_precio_prom_otros_centros + emailer_for_promotion + homepage_featured'; SS='log_num_orders ~ log_checkout_price_hat + emailer_for_promotion + homepage_featured'

def fit2(df):
 yf,Xf_df=dmatrices(FS,df,return_type='dataframe'); yf=np.asarray(yf).ravel(); Xf=np.asarray(Xf_df); b1=np.linalg.lstsq(Xf,yf,rcond=None)[0]; yhat=Xf@b1
 d=df.copy(); d['log_checkout_price_hat']=yhat; ys,Xs_df=dmatrices(SS,d,return_type='dataframe'); ys=np.asarray(ys).ravel(); Xs=np.asarray(Xs_df); b2=np.linalg.lstsq(Xs,ys,rcond=None)[0]; idx=list(Xs_df.design_info.column_names).index('log_checkout_price_hat'); return b1,Xf,yf,b2,Xs,ys,idx

def run(n_iter=2000,seed=42,checkpoint_every=100,resume=True):
 df=construir_instrumentos(load_data()).dropna(subset=['log_precio_prom_otros_centros']).copy(); clusters=np.unique(df.center_id); base=fit2(df); b1,Xf,yf,b2,Xs,ys,idx=base; fsres=yf-Xf@b1; ssfit=Xs@b2; ssres=ys-ssfit
 rng=np.random.default_rng(seed); coefs=np.full(n_iter,np.nan); fboot=np.full(n_iter,np.nan); start=0
 if resume:
  cp=load_checkpoint(CHECK)
  if cp and cp.get('signature')==(n_iter,seed): start=cp['next_iter']; coefs[:len(cp['coefs'])]=cp['coefs']; fboot[:len(cp['fboot'])]=cp['fboot']; rng.bit_generator.state=cp['rng_state']; print('Reanudando',start)
 for i in range(start,n_iter):
  wmap=dict(zip(clusters,rng.choice([-1.,1.],size=len(clusters)))); w=df.center_id.map(wmap).to_numpy(); price_star=Xf@b1+w*fsres; dstar=df.copy(); dstar['log_checkout_price']=price_star; # reestimate FS
  try:
   dstar['log_num_orders']=ssfit + w*ssres; b1b,Xfb,yfb,b2b,Xsb,ysb,idxb=fit2(dstar); coefs[i]=b2b[idxb]; fboot[i]=sm.OLS(yfb,Xfb).fit().fvalue
  except Exception: pass
  if (i+1)%checkpoint_every==0 or i==n_iter-1: save_npz_checkpoint(CHECK,{'signature':(n_iter,seed),'next_iter':i+1,'coefs':coefs.copy(),'fboot':fboot.copy(),'rng_state':rng.bit_generator.state}); print('CHECKPOINT',i+1,n_iter)
 v=coefs[np.isfinite(coefs)]; fv=fboot[np.isfinite(fboot)]; r={'coef_media':float(v.mean()),'coef_std':float(v.std()),'ic_95_inferior':float(np.percentile(v,2.5)),'ic_95_superior':float(np.percentile(v,97.5)),'n_iteraciones_validas':int(len(v)),'n_iteraciones_total':n_iter,'convergencia_tasa':float(len(v)/n_iter),'coef_base':float(b2[idx]),'f_first_stage_base':float(sm.OLS(yf,Xf).fit().fvalue),'f_first_stage_media_bootstrap':float(fv.mean()) if len(fv) else np.nan,'f_first_stage_std_bootstrap':float(fv.std()) if len(fv) else np.nan,'clusters':int(len(clusters)),'semilla':seed,'nota':'Wild cluster bootstrap IV con ambas etapas reestimadas; sensibilidad, no identificación causal.','provenance':{'has_raw_replicates':False,'stored_content':'summary_statistics_only','visualization_method':'theoretical_normal_approximation_from_summary','bootstrap_type':'wild_cluster_bootstrap','replicates_available':'summary_only_mean_std_IC','theoretical_approximation':'normal_density_from_bootstrap_mean_std','description':'2000 raw replicates computed but only summary statistics (mean, std, percentile IC) stored. Figures 24/25 use theoretical normal density from bootstrap mean/std for visualization.'}}; REPORTS_CERTIFIED.mkdir(parents=True, exist_ok=True); save_result(r,REPORTS_CERTIFIED/OUT); CHECK.unlink(missing_ok=True); return r

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--iterations',type=int,default=2000); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--checkpoint-every',type=int,default=100); ap.add_argument('--no-resume',action='store_true'); a=ap.parse_args(); print(run(a.iterations,a.seed,a.checkpoint_every,not a.no_resume))
if __name__=='__main__': main()
