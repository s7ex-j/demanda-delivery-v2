from common import *
df=load_df(); g=df.groupby('center_type')['num_orders'].mean()
fig,ax=plt.subplots(figsize=(8,3.5)); g.plot.bar(ax=ax); ax.set_xlabel('Tipo de centro'); ax.set_ylabel('Demanda promedio'); ax.set_title('Comparación de demanda promedio por tipo de centro'); apply_grid(ax); fig.tight_layout(); save(fig,'comparacion_tipo_centro.png')