from common import *
df=load_df(); g=df.groupby('week').agg(precio=('checkout_price','mean'), demanda=('num_orders','sum')).reset_index()
fig,ax1=plt.subplots(figsize=(10,4.2)); ax1.plot(g.week,g.precio,label='Precio promedio de checkout',linewidth=1.8); ax1.set_xlabel('Semana'); ax1.set_ylabel('Precio promedio')
ax2=ax1.twinx(); ax2.plot(g.week,g.demanda,label='Demanda agregada',linewidth=1.8,alpha=.8); ax2.set_ylabel('Pedidos agregados')
ax1.set_title('Evolución semanal del precio promedio y la demanda agregada'); apply_grid(ax1); fig.tight_layout(); save(fig,'evolucion_precio_demanda_semanal.png')