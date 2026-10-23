from common import *
import numpy as np
# Load full curve from NPY (contains price_grid, elasticidad_media, ic_95_inferior, ic_95_superior)
npy = np.load(REPORTS / 'diagnostics' / 'bootstrap_rcs_confidence_bands.npy', allow_pickle=True).item()
p = np.asarray(npy['price_grid'])
e = np.asarray(npy['elasticidad_media'])
lo = np.asarray(npy['ic_95_inferior'])
hi = np.asarray(npy['ic_95_superior'])
fig,ax=plt.subplots(figsize=(10,5.2)); ax.fill_between(p,lo,hi,alpha=.25,label='IC 95% punto a punto'); ax.plot(p,e,linewidth=2,label='Elasticidad RCS'); ax.axhline(-.759,linestyle='--',linewidth=1,label='OLS principal'); ax.set_xlabel('Precio de checkout'); ax.set_ylabel('Elasticidad precio-demanda'); ax.set_title('Elasticidad puntual RCS con bandas de confianza bootstrap'); apply_grid(ax); ax.legend(); fig.tight_layout(); save(fig,'rcs_elasticidad_bandas_confianza_bootstrap.png',dpi=300)