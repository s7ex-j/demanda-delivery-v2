"""Placebos revisados: precio futuro y permutación válida de bloques por entidad.

La antigua implementación de block permutation se retiró porque era degenerada.
Esta versión permuta bloques consecutivos de 4 semanas del PRECIO dentro de cada
center_id x meal_id, dejando demanda y controles intactos. Esto rompe la alineación
temporal precio-demanda pero conserva estructura local del precio dentro de la entidad.
"""
from __future__ import annotations
import argparse,sys,time,json
from pathlib import Path
import numpy as np,pandas as pd,statsmodels.formula.api as smf
sys.path.insert(0,str(Path(__file__).resolve().parent)); from vm_common import OUT_DIR,load_data,save_json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src')); from modeling import placebo_precio_futuro
# Output now goes to reports/certified
REPORTS_CERTIFIED = Path(__file__).resolve().parents[2] / 'reports' / 'certified'
FORM='log_num_orders ~ log_checkout_price + emailer_for_promotion + homepage_featured + hubo_aumento + C(category) + C(center_type)'
def future(df):
 r=placebo_precio_futuro(df); return r

def block_perm_once(df,rng,block=4):
 out=df.copy(); out.sort_values(['center_id','meal_id','week'],inplace=True); groups=[]
 for _,g in out.groupby(['center_id','meal_id'],sort=False):
  idx=g.index.to_numpy(); blocks=[idx[i:i+block] for i in range(0,len(idx),block)]; perm=rng.permutation(len(blocks)); src=np.concatenate([blocks[j] for j in perm]); out.loc[idx,'log_checkout_price']=out.loc[src,'log_checkout_price'].to_numpy(); groups.append(len(blocks))
 m=smf.ols(FORM,data=out).fit(); return float(m.params['log_checkout_price'])

def run(df,iterations=500,seed=42,checkpoint_every=50):
 real=float(smf.ols(FORM,data=df).fit().params['log_checkout_price']); rng=np.random.default_rng(seed); co=np.full(iterations,np.nan); t0=time.time(); cp=OUT_DIR/'placebo_block_permutation.checkpoint.npy'
 start=0
 if cp.exists():
  d=np.load(cp,allow_pickle=True).item();
  if d.get('signature')==(iterations,seed): start=d['next_iter']; co[:start]=d['coefs']; rng.bit_generator.state=d['rng_state']; print('Reanudando',start)
 for i in range(start,iterations):
  try: co[i]=block_perm_once(df,rng)
  except Exception: pass
  if (i+1)%checkpoint_every==0 or i==iterations-1: np.save(cp,{'signature':(iterations,seed),'next_iter':i+1,'coefs':co.copy(),'rng_state':rng.bit_generator.state},allow_pickle=True); print('CHECKPOINT',i+1,iterations)
 v=co[np.isfinite(co)]; p=float((np.abs(v)>=abs(real)).mean()) if len(v) else np.nan
 out={'elasticidad_real':real,'media_placebo':float(v.mean()),'std_placebo':float(v.std()),'p_valor_empirico':p,'n_iteraciones_validas':len(v),'n_iteraciones_total':iterations,'block_weeks':4,'seed':seed,'nota':'Permutación de bloques de 4 semanas dentro de cada entidad center_id x meal_id; preserva autocorrelación local del precio y rompe su alineación temporal con la demanda.'}; REPORTS_CERTIFIED.mkdir(parents=True, exist_ok=True); save_json(out,REPORTS_CERTIFIED/'placebo_block_permutation.json'); cp.unlink(missing_ok=True); return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--iterations',type=int,default=500); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--checkpoint-every',type=int,default=50); a=ap.parse_args(); df=load_data(); REPORTS_CERTIFIED.mkdir(parents=True, exist_ok=True); save_json(future(df),REPORTS_CERTIFIED/'placebo_precio_futuro.json'); print(run(df,a.iterations,a.seed,a.checkpoint_every))
if __name__=='__main__': main()
