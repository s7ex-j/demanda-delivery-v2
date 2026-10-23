"""Diagnóstico de poder de Hansen J usando la implementación real del proyecto y checkpoints."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vm')); from vm_common import OUT_DIR,load_data,load_checkpoint,save_npz_checkpoint,save_result
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src')); from modeling import construir_instrumentos,test_sobreidentificacion_hansen_j
OUT='hansen_j_power_diagnostic_results.npy'; CHECK=OUT_DIR/'hansen_j_power_diagnostic.checkpoint.pkl'
def calc(df):
 d=construir_instrumentos(df).dropna(subset=['log_precio_prom_otros_centros']); j=test_sobreidentificacion_hansen_j(d); return float(j.stat),float(j.pval)
def run(sizes=(2000,5000,10000,20000,50000,100000,200000),reps=50,seed=42,checkpoint_every=1,resume=True):
 df=load_data(); n=len(df); sizes=[int(x) for x in sizes if x<=n]; sizes=sorted(set(sizes)); rng=np.random.default_rng(seed); results={}; completed_n=[]; start_state=None
 if resume:
  cp=load_checkpoint(CHECK)
  if cp and cp.get('signature')==(tuple(sizes),reps,seed): results=cp['results']; completed_n=cp['completed_n']; rng.bit_generator.state=cp['rng_state']; print('Reanudando tamaños ya completos:',completed_n)
 t0=time.time()
 for ns in sizes:
  key=str(ns)
  if ns==n and key in results: continue
  rep_start=0; js=[]; ps=[]
  if key in results and results[key].get('partial_reps',0)>0: rep_start=results[key]['partial_reps']; js=results[key]['j_values']; ps=results[key]['p_values']
  total_rep=1 if ns==n else reps
  js=list(js)+[np.nan]*max(0,total_rep-len(js)); ps=list(ps)+[np.nan]*max(0,total_rep-len(ps))
  for r in range(rep_start,total_rep):
   sample=df if ns==n else df.iloc[rng.choice(n,size=ns,replace=False)]
   try: js[r],ps[r]=calc(sample)
   except Exception: pass
   if (r+1)%checkpoint_every==0:
    results[key]={'j_values':js,'p_values':ps,'partial_reps':r+1}; save_npz_checkpoint(CHECK,{'signature':(tuple(sizes),reps,seed),'results':results,'completed_n':completed_n,'rng_state':rng.bit_generator.state}); print('CHECKPOINT',ns,r+1,total_rep)
  v=np.isfinite(ps); jv=np.asarray(js)[v]; pv=np.asarray(ps)[v]; results[key]={'j_media':float(jv.mean()) if len(jv) else np.nan,'j_std':float(jv.std()) if len(jv) else np.nan,'p_media':float(pv.mean()) if len(pv) else np.nan,'tasa_rechazo_5pct':float((pv<.05).mean()) if len(pv) else np.nan,'n_repeticiones_validas':int(v.sum()),'n_errores':int(total_rep-v.sum())}; completed_n.append(ns); save_npz_checkpoint(CHECK,{'signature':(tuple(sizes),reps,seed),'results':results,'completed_n':completed_n,'rng_state':rng.bit_generator.state})
 jfull,pfull=calc(df); out={'j_muestra_completa':jfull,'p_muestra_completa':pfull,'n_total':n,'tamanos_evaluados':sizes,'resultados_por_n':results,'n_repeticiones':reps,'seed':seed,'tiempo_total_seg':time.time()-t0,'nota':'Diagnóstico descriptivo de sensibilidad del rechazo de Hansen J al tamaño muestral. No es una prueba de validez de instrumentos.'}; save_result(out,OUT); CHECK.unlink(missing_ok=True); return out
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=50); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--checkpoint-every',type=int,default=1); ap.add_argument('--no-resume',action='store_true'); a=ap.parse_args(); print(run(reps=a.reps,seed=a.seed,checkpoint_every=a.checkpoint_every,resume=not a.no_resume))
if __name__=='__main__': main()
