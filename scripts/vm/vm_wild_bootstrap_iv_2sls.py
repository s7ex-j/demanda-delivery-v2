"""Wild cluster bootstrap IV 2SLS con primera etapa fija y checkpoints."""
from __future__ import annotations
import argparse,time,sys
from pathlib import Path
import numpy as np, statsmodels.api as sm
from patsy import dmatrices
sys.path.insert(0,str(Path(__file__).resolve().parent)); from vm_common import OUT_DIR,load_data,load_checkpoint,save_npz_checkpoint,save_result
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src')); from modeling import construir_instrumentos
# Output now goes to reports/certified
REPORTS_CERTIFIED = Path(__file__).resolve().parents[2] / 'reports' / 'certified'
OUT='wild_bootstrap_iv_2sls_results.npy'; CHECK=OUT_DIR/'wild_bootstrap_iv_2sls_results.checkpoint.pkl'
FS='log_checkout_price ~ log_base_price + log_precio_prom_otros_centros + emailer_for_promotion + homepage_featured'
SS='log_num_orders ~ log_checkout_price_hat + emailer_for_promotion + homepage_featured'

def run(n_iter=2000,seed=42,checkpoint_every=200,resume=True):
    df=construir_instrumentos(load_data()).dropna(subset=['log_precio_prom_otros_centros']).copy(); clusters=np.unique(df.center_id); cid=df.center_id.to_numpy();
    yf,Xf=dmatrices(FS,df,return_type='dataframe'); yf=np.asarray(yf).ravel(); Xf=np.asarray(Xf); b1=np.linalg.lstsq(Xf,yf,rcond=None)[0]; ph=Xf@b1
    df['log_checkout_price_hat']=ph; ys,Xs_df=dmatrices(SS,df,return_type='dataframe'); ys=np.asarray(ys).ravel(); Xs=np.asarray(Xs_df); b2=np.linalg.lstsq(Xs,ys,rcond=None)[0]; idx=list(Xs_df.design_info.column_names).index('log_checkout_price_hat'); resid=ys-Xs@b2
    rng=np.random.default_rng(seed); coefs=np.full(n_iter,np.nan); start=0
    if resume:
        cp=load_checkpoint(CHECK)
        if cp and cp.get('signature')==(n_iter,seed): start=cp['next_iter']; coefs[:len(cp['coefs'])]=cp['coefs']; rng.bit_generator.state=cp['rng_state']; print('Reanudando',start)
    t0=time.time()
    for i in range(start,n_iter):
        weights=rng.choice([-1.,1.],size=len(clusters)); wm=dict(zip(clusters,weights)); w=df.center_id.map(wm).to_numpy(); ys_star=Xs@b2+w*resid
        try: coefs[i]=np.linalg.lstsq(Xs,ys_star,rcond=None)[0][idx]
        except Exception: pass
        if (i+1)%checkpoint_every==0 or i==n_iter-1:
            save_npz_checkpoint(CHECK,{'signature':(n_iter,seed),'next_iter':i+1,'coefs':coefs.copy(),'rng_state':rng.bit_generator.state}); print('CHECKPOINT',i+1,n_iter)
    v=coefs[np.isfinite(coefs)]; r={'coef_media':float(v.mean()),'coef_std':float(v.std()),'ic_95_inferior':float(np.percentile(v,2.5)),'ic_95_superior':float(np.percentile(v,97.5)),'n_iteraciones_validas':int(len(v)),'n_iteraciones_total':n_iter,'convergencia_tasa':float(len(v)/n_iter),'coef_base':float(b2[idx]),'f_first_stage_base':float(sm.OLS(yf,Xf).fit().fvalue),'clusters':int(len(clusters)),'semilla':seed,'nota':'Wild cluster bootstrap IV con primera etapa fija; análisis de sensibilidad, no corrige Hansen J.','provenance':{'has_raw_replicates':False,'stored_content':'summary_statistics_only','visualization_method':'percentile_bootstrap_IC_from_empirical_replicates','bootstrap_type':'wild_cluster_bootstrap_fixed_first_stage','replicates_available':'summary_only_mean_std_IC','theoretical_approximation':'none_empirical_percentile_IC','description':'2000 raw replicates computed (fixed first stage) but only summary statistics (mean, std, percentile IC) stored. Sensitivity analysis, does not correct for Hansen J.'}}; REPORTS_CERTIFIED.mkdir(parents=True, exist_ok=True); save_result(r,REPORTS_CERTIFIED/OUT); CHECK.unlink(missing_ok=True); return r

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--iterations',type=int,default=2000); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--checkpoint-every',type=int,default=200); ap.add_argument('--no-resume',action='store_true'); a=ap.parse_args(); print(run(a.iterations,a.seed,a.checkpoint_every,not a.no_resume))
if __name__=='__main__': main()
