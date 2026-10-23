"""Bootstrap por clusters de la curva RCS corregida.

La especificación fija explícitamente los nudos P5/P35/P65/P95 de log-precio.
Se usa bootstrap de centros con reemplazo mediante multiplicidades, lo que
permite reanudar desde checkpoints sin duplicar físicamente 456k filas.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from patsy import dmatrix, build_design_matrices
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vm')); from vm_common import OUT_DIR, load_data, load_checkpoint, save_npz_checkpoint, save_result

OUT='bootstrap_rcs_confidence_bands.npy'; CHECK=OUT_DIR/'bootstrap_rcs_confidence_bands.checkpoint.pkl'

def make_basis(df):
 x=df['log_checkout_price'].to_numpy(); knots=np.percentile(x,[5,35,65,95]);
 formula=("cr(log_checkout_price, " + f"knots=({float(knots[1])!r}, {float(knots[2])!r}), lower_bound={float(knots[0])!r}, upper_bound={float(knots[-1])!r}, constraints='center') - 1")
 z=dmatrix(formula, {'log_checkout_price':x}, return_type='dataframe')
 z.columns=[f'rcs_{i+1}' for i in range(z.shape[1])]
 # Fixed categorical controls and intercept.
 ctrl=dmatrix('1 + emailer_for_promotion + homepage_featured + hubo_aumento + C(category) + C(center_type)',df,return_type='dataframe')
 X=np.column_stack([np.asarray(ctrl),np.asarray(z)])
 names=list(ctrl.design_info.column_names)+list(z.columns)
 p5,p95=np.percentile(df.checkout_price,[5,95]); grid=np.linspace(p5,p95,100); lp=np.log(grid); h=1e-4
 z0=build_design_matrices([z.design_info],{'log_checkout_price':lp})[0]; z1=build_design_matrices([z.design_info],{'log_checkout_price':lp+h})[0]; D=(np.asarray(z1)-np.asarray(z0))/h
 return X,names,knots,grid,D

def solve_wls_cluster(X,y,weights,clusters_unique,cluster_idx=None):
 w=weights; Xw=X*w[:,None]; A=X.T@Xw; b=X.T@(w*y); beta=np.linalg.lstsq(A,b,rcond=None)[0]; return beta

def run(n_iter=2000,seed=42,checkpoint_every=100,resume=True):
 df=load_data(); X,names,knots,grid,D=make_basis(df); y=df.log_num_orders.to_numpy(); clusters=np.unique(df.center_id); cid=df.center_id.to_numpy(); rng=np.random.default_rng(seed)
 e0=X.T@X; y0=X.T@y; beta0=np.linalg.lstsq(e0,y0,rcond=None)[0]
 coefs=np.full((n_iter,len(grid)),np.nan); start=0
 if resume:
  cp=load_checkpoint(CHECK)
  if cp and cp.get('signature')==(n_iter,seed): start=cp['next_iter']; coefs[:start]=cp['coefs']; rng.bit_generator.state=cp['rng_state']; print('Reanudando',start)
 t0=time.time()
 for i in range(start,n_iter):
  sampled=rng.choice(clusters,size=len(clusters),replace=True); mult=dict.fromkeys(clusters.tolist(),0)
  for c in sampled: mult[c]+=1
  w=df.center_id.map(mult).to_numpy(float)
  beta=solve_wls_cluster(X,y,w,clusters)
  coefs[i]=D@beta[-D.shape[1]:]
  if (i+1)%checkpoint_every==0 or i==n_iter-1: save_npz_checkpoint(CHECK,{'signature':(n_iter,seed),'next_iter':i+1,'coefs':coefs.copy(),'rng_state':rng.bit_generator.state}); print('CHECKPOINT',i+1,n_iter,'elapsed',round((time.time()-t0)/60,1),'min')
 valid=coefs[np.isfinite(coefs).all(axis=1)]; mean=valid.mean(axis=0); lo=np.percentile(valid,2.5,axis=0); hi=np.percentile(valid,97.5,axis=0)
 out={'price_grid':grid,'elasticidad_media':mean,'elasticidad_std':valid.std(axis=0),'ic_95_inferior':lo,'ic_95_superior':hi,'n_iteraciones_validas':len(valid),'n_iteraciones_total':n_iter,'knots_log_price':knots,'knots_percentiles':[5,35,65,95],'clusters':int(len(clusters)),'seed':seed,'nota':'Bootstrap de centros con reemplazo. RCS con nudos explícitos P5/P35/P65/P95 de log-precio; IC punto a punto no simultáneo.','provenance':{'has_raw_replicates':False,'stored_content':'summary_statistics_only','visualization_method':'percentile_bootstrap_IC_from_empirical_replicates','bootstrap_type':'cluster_bootstrap','replicates_available':'summary_only_mean_std_IC_pointwise','theoretical_approximation':'none_empirical_percentile_IC','description':'2000 raw replicates computed for each grid point but only pointwise summary statistics (mean, std, percentile IC) stored. Elasticity curve grid shape (100,) from summary statistics only.'}}
 save_result(out,OUT); CHECK.unlink(missing_ok=True); return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--iterations',type=int,default=2000); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--checkpoint-every',type=int,default=100); ap.add_argument('--no-resume',action='store_true'); a=ap.parse_args(); r=run(a.iterations,a.seed,a.checkpoint_every,not a.no_resume); print('validas',r['n_iteraciones_validas'])
if __name__=='__main__': main()
