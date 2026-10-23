"""Wild cluster bootstrap-t del OLS principal con SE clusterizado por center_id."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
import numpy as np
from patsy import dmatrices
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vm')); from vm_common import OUT_DIR,load_data,load_checkpoint,save_npz_checkpoint,save_result
FORMULA='log_num_orders ~ log_checkout_price + emailer_for_promotion + homepage_featured + hubo_aumento + C(category) + C(center_type)'
OUT='bootstrap_t_intervals_results.npy'; CHECK=OUT_DIR/'bootstrap_t_intervals_results.checkpoint.pkl'

def cluster_se(X,resid,groups,inv_xx,idx):
 meat=np.zeros((X.shape[1],X.shape[1]))
 for g in np.unique(groups):
  q=X[groups==g].T@resid[groups==g]; meat += np.outer(q,q)
 cov=inv_xx@meat@inv_xx; return float(np.sqrt(max(cov[idx,idx],0)))

def run(n_iter=2000,seed=42,checkpoint_every=100,resume=True):
 df=load_data(); ydf,Xdf=dmatrices(FORMULA,df,return_type='dataframe'); y=np.asarray(ydf).ravel(); X=np.asarray(Xdf); names=Xdf.design_info.column_names; idx=names.index('log_checkout_price'); beta=np.linalg.lstsq(X,y,rcond=None)[0]; resid=y-X@beta; inv=np.linalg.pinv(X.T@X); se_hat=cluster_se(X,resid,df.center_id.to_numpy(),inv,idx)
 clusters=np.unique(df.center_id); gid=df.center_id.to_numpy(); rng=np.random.default_rng(seed); coefs=np.full(n_iter,np.nan); t=np.full(n_iter,np.nan); start=0
 if resume:
  cp=load_checkpoint(CHECK)
  if cp and cp.get('signature')==(n_iter,seed): start=cp['next_iter']; coefs[:start]=cp['coefs']; t[:start]=cp['t']; rng.bit_generator.state=cp['rng_state']; print('Reanudando',start)
 for i in range(start,n_iter):
  wmap=dict(zip(clusters,rng.choice([-1.,1.],size=len(clusters)))); w=np.vectorize(wmap.get)(gid); ys=X@beta+w*resid; bb=np.linalg.lstsq(X,ys,rcond=None)[0]; rb=ys-X@bb; seb=cluster_se(X,rb,gid,inv,idx); coefs[i]=bb[idx]; t[i]=(bb[idx]-beta[idx])/seb if seb>0 else np.nan
  if (i+1)%checkpoint_every==0 or i==n_iter-1: save_npz_checkpoint(CHECK,{'signature':(n_iter,seed),'next_iter':i+1,'coefs':coefs.copy(),'t':t.copy(),'rng_state':rng.bit_generator.state}); print('CHECKPOINT',i+1,n_iter)
 v=t[np.isfinite(t)]; c=coefs[np.isfinite(t)]; q025,q975=np.percentile(v,[2.5,97.5]); lo=beta[idx]-se_hat*q975; hi=beta[idx]-se_hat*q025
 r={'coef_hat':float(beta[idx]),'se_hat_clusterizado_center':se_hat,'ic_95_percentil_simple':[float(np.percentile(c,2.5)),float(np.percentile(c,97.5))],'ic_95_bootstrap_t':[float(lo),float(hi)],'t_boot_q025':float(q025),'t_boot_q975':float(q975),'n_iteraciones_validas':len(v),'n_iteraciones_total':n_iter,'seed':seed,'nota':'Bootstrap-t wild cluster por center_id; complemento de inferencia, no two-way clustering.','provenance':{'has_raw_replicates':False,'stored_content':'summary_statistics_only','visualization_method':'bootstrap_t_percentile_from_empirical_t_distribution','bootstrap_type':'wild_cluster_bootstrap_t','replicates_available':'summary_only_t_statistics_coef_percentile','theoretical_approximation':'bootstrap_t_from_empirical_t_dist','description':'2000 raw replicates of t-statistics computed but only quantiles and coefficient percentiles stored. Bootstrap-t uses empirical t-distribution quantiles for asymmetric CI.'}}; save_result(r,OUT); CHECK.unlink(missing_ok=True); return r

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--iterations',type=int,default=2000); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--checkpoint-every',type=int,default=100); ap.add_argument('--no-resume',action='store_true'); a=ap.parse_args(); print(run(a.iterations,a.seed,a.checkpoint_every,not a.no_resume))
if __name__=='__main__': main()
