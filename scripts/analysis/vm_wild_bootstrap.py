"""Wild cluster bootstrap OLS principal con checkpoints reanudables."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import statsmodels.formula.api as smf
from patsy import dmatrices
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vm'))
from vm_common import PROJECT_ROOT, DATA_CSV, OUT_DIR, load_data, load_checkpoint, save_npz_checkpoint, save_result

FORMULA = ('log_num_orders ~ log_checkout_price + emailer_for_promotion + '
           'homepage_featured + hubo_aumento + C(category) + C(center_type)')
OUT = 'wild_cluster_bootstrap_results.npy'
CHECK = OUT_DIR / 'wild_cluster_bootstrap_results.checkpoint.pkl'

def run(n_iter=500, seed=42, checkpoint_every=100, resume=True):
    df=load_data()
    y,X_df=dmatrices(FORMULA,df,return_type='dataframe'); y=np.asarray(y).ravel(); X=np.asarray(X_df)
    base=np.linalg.lstsq(X,y,rcond=None)[0]; resid=y-X@base
    clusters=df.center_id.to_numpy(); uniq=np.unique(clusters); cidx={c:np.flatnonzero(clusters==c) for c in uniq}
    rng=np.random.default_rng(seed)
    coefs=np.full(n_iter,np.nan); start=0
    if resume:
        cp=load_checkpoint(CHECK)
        if cp and cp.get('signature')==(n_iter,seed):
            start=int(cp['next_iter']); coefs[:len(cp['coefs'])]=cp['coefs']; rng.bit_generator.state=cp['rng_state']; print(f'Reanudando en {start}/{n_iter}')
    t0=time.time()
    for i in range(start,n_iter):
        wmap=rng.choice([-1.0,1.0],size=len(uniq)); w=np.empty(len(df))
        for j,c in enumerate(uniq): w[cidx[c]]=wmap[j]
        ys=X@base+w*resid
        try: coefs[i]=np.linalg.lstsq(X,ys,rcond=None)[0][X_df.design_info.column_names.index('log_checkout_price')]
        except Exception: pass
        if (i+1)%checkpoint_every==0 or i==n_iter-1:
            payload={'signature':(n_iter,seed),'next_iter':i+1,'coefs':coefs.copy(),'rng_state':rng.bit_generator.state}
            save_npz_checkpoint(CHECK,payload); print('CHECKPOINT',i+1,n_iter,flush=True)
    valid=coefs[np.isfinite(coefs)]
    res={'elasticidad_media':float(np.mean(valid)),'elasticidad_std':float(np.std(valid,ddof=0)),
         'ic_95_inferior':float(np.percentile(valid,2.5)),'ic_95_superior':float(np.percentile(valid,97.5)),
         'n_iteraciones_validas':int(len(valid)),'n_iteraciones_total':n_iter,'coef_base':float(base[X_df.design_info.column_names.index('log_checkout_price')]),
         'clusters':int(len(uniq)),'semilla':seed,'metodo':'Wild cluster bootstrap OLS por center_id; inferencia de sensibilidad.',
         'provenance':{'has_raw_replicates':False,'stored_content':'summary_statistics_only','visualization_method':'theoretical_normal_approximation_from_summary','bootstrap_type':'wild_cluster_bootstrap_ols','replicates_available':'summary_only_mean_std_IC','theoretical_approximation':'normal_density_from_bootstrap_mean_std','description':'500 raw replicates computed but only summary statistics (mean, std, percentile IC) stored. Figure 24 uses theoretical normal density from bootstrap mean/std for visualization.'}}
    save_result(res,OUT); CHECK.unlink(missing_ok=True); return res

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--iterations',type=int,default=500); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--checkpoint-every',type=int,default=100); ap.add_argument('--no-resume',action='store_true'); a=ap.parse_args(); r=run(a.iterations,a.seed,a.checkpoint_every,not a.no_resume); print(r)
if __name__=='__main__': main()
