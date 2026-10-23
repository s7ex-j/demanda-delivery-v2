from common import *
df=load_df(); g=df.groupby('meal_id')['num_orders'].sum().nlargest(10).sort_values()
fig,ax=plt.subplots(figsize=(8,5.2)); g.plot.barh(ax=ax); ax.set_xlabel('Pedidos acumulados'); ax.set_ylabel('meal_id'); ax.set_title('Top 10 comidas por demanda total acumulada'); apply_grid(ax); fig.tight_layout(); save(fig,'top10_comidas_demanda.png')