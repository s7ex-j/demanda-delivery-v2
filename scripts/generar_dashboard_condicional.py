"""
P1-11 — Dashboard condicional: recomendaciones de pricing por precio x categoria x especificacion.

Genera CSVs en dashboard/data/ con:
- elasticidad_rcs_por_precio.csv: elasticidad puntual RCS en grid de precios
- elasticidad_por_categoria_detallada.csv: elasticidades categoria (interaccion)
- recomendaciones_pricing.csv: recomendaciones condicionales (subir/mantener/bajar)
- kpis_ejecutivos.csv: actualizado con 3 elasticidades + wild bootstrap

USA SOLO resultados verificados del ETAPA 3 gate.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix

# --- Rutas ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_CSV = PROJECT_ROOT / "data" / "processed" / "demanda_limpia_v2.csv"
DASH_DIR = PROJECT_ROOT / "dashboard" / "data"
DASH_DIR.mkdir(parents=True, exist_ok=True)
JSON_RESULTADOS = PROJECT_ROOT / "reports" / "notebook_outputs" / "etapa3_resultados_completos.json"


def main():
    print("=" * 70)
    print("P1-11: Dashboard condicional precio x categoria x especificacion")
    print("=" * 70)

    df = pd.read_csv(DATA_CSV)
    print(f"  N = {len(df):,}")

    # ============================================================
    # 1. Elasticidad RCS puntual por nivel de precio
    # ============================================================
    print("\n[1/4] Elasticidad RCS puntual por nivel de precio...")
    log_price = df['log_checkout_price']
    transformed_x = dmatrix(
        "cr(log_checkout_price, df=4, constraints='center')",
        {"log_checkout_price": log_price},
        return_type='dataframe'
    )
    spline_cols = [f'rcs_{i}' for i in range(transformed_x.shape[1])]
    transformed_x.columns = spline_cols
    df_rcs = df.copy()
    df_rcs[spline_cols] = transformed_x[spline_cols]
    spline_terms = ' + '.join(spline_cols)
    formula_rcs = (
        f'log_num_orders ~ {spline_terms} + emailer_for_promotion '
        '+ homepage_featured + hubo_aumento + C(category) + C(center_type)'
    )
    modelo_rcs = smf.ols(formula_rcs, data=df_rcs).fit()

    # Extraer transformador CR
    design_info = transformed_x.design_info
    cr_transform = None
    for k, v in design_info.factor_infos.items():
        for name, transform in v.state['transforms'].items():
            if hasattr(transform, '_all_knots'):
                cr_transform = transform
                break
        if cr_transform is not None:
            break

    h = 1e-4
    spline_basis_cols = [c for c in modelo_rcs.params.index
                         if c.startswith('rcs_') and c != 'rcs_0']
    coefs_spline = np.array([modelo_rcs.params[c] for c in spline_basis_cols])

    precios_grid = np.array([50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500])
    filas_rcs = []
    for p in precios_grid:
        log_p = np.log(p)
        x0 = np.asarray(cr_transform.transform(np.array([log_p]))).flatten()
        x1 = np.asarray(cr_transform.transform(np.array([log_p + h]))).flatten()
        deriv = (x1 - x0) / h
        elast = float(np.dot(coefs_spline, deriv))
        # Recomendacion basada en signo de ingreso marginal:
        # %dR = (1+d)^(1+eta) - 1; para d pequeno, dR/d d = 1+eta
        # Si eta > -1 (inelastica): subir precio aumenta ingreso -> SUBIR
        # Si eta < -1 (elastica): bajar precio aumenta ingreso -> BAJAR
        # Si eta ~ -1: MANTENER
        if elast > -0.9:
            rec = 'SUBIR (inelastica: ingreso sube con precio)'
        elif elast < -1.1:
            rec = 'BAJAR (elastica: ingreso sube con precio alto)'
        else:
            rec = 'MANTENER (cerca de elasticidad unitaria)'
        filas_rcs.append({
            'precio': int(p),
            'log_precio': round(log_p, 4),
            'elasticidad_rcs': round(elast, 4),
            'recomendacion': rec,
            'especificacion': 'RCS local (4 nudos)',
        })

    df_rcs_out = pd.DataFrame(filas_rcs)
    df_rcs_out.to_csv(DASH_DIR / 'elasticidad_rcs_por_precio.csv', index=False)
    print(f"  -> {DASH_DIR / 'elasticidad_rcs_por_precio.csv'}")
    print(df_rcs_out[['precio', 'elasticidad_rcs', 'recomendacion']].to_string(index=False))

    # ============================================================
    # 2. Elasticidad por categoria (interaccion)
    # ============================================================
    print("\n[2/4] Elasticidad por categoria (interaccion)...")
    formula_cat = (
        'log_num_orders ~ log_checkout_price + C(category) '
        '+ log_checkout_price:C(category) '
        '+ emailer_for_promotion + homepage_featured + hubo_aumento '
        '+ C(center_type)'
    )
    modelo_cat = smf.ols(formula_cat, data=df).fit()
    coef_base = modelo_cat.params.get('log_checkout_price', 0)

    filas_cat = []
    for cat in sorted(df['category'].unique()):
        term = f'log_checkout_price:C(category)[T.{cat}]'
        if term in modelo_cat.params:
            elast = coef_base + modelo_cat.params[term]
        else:
            elast = coef_base  # categoria base
        if elast > -0.9:
            rec = 'SUBIR'
        elif elast < -1.1:
            rec = 'BAJAR'
        else:
            rec = 'MANTENER'
        filas_cat.append({
            'category': cat,
            'elasticidad': round(float(elast), 4),
            'recomendacion': rec,
            'especificacion': 'interaccion precio x categoria',
        })

    df_cat_out = pd.DataFrame(filas_cat).sort_values('elasticidad')
    df_cat_out.to_csv(DASH_DIR / 'elasticidad_por_categoria_detallada.csv', index=False)
    print(f"  -> {DASH_DIR / 'elasticidad_por_categoria_detallada.csv'}")
    print(df_cat_out.to_string(index=False))

    # ============================================================
    # 3. Recomendaciones de pricing condicionales (matriz)
    # ============================================================
    print("\n[3/4] Matriz de recomendaciones condicionales...")
    # Combinar RCS (precio) x categoria usando elasticidad promedio por categoria
    # como ajuste sobre la elasticidad RCS global
    elast_cat_map = dict(zip(df_cat_out['category'], df_cat_out['elasticidad']))
    elast_global_cat = float(np.mean(list(elast_cat_map.values())))

    filas_rec = []
    for _, row_r in df_rcs_out.iterrows():
        precio = row_r['precio']
        elast_rcs = row_r['elasticidad_rcs']
        for cat, elast_cat in elast_cat_map.items():
            # Elasticidad condicional aproximada: RCS base + desviacion categoria
            ajuste = elast_cat - elast_global_cat
            elast_cond = elast_rcs + ajuste
            if elast_cond > -0.9:
                rec = 'SUBIR'
                ingreso_5pct = (1.05)**(1 + elast_cond) - 1
            elif elast_cond < -1.1:
                rec = 'BAJAR'
                ingreso_5pct = (0.95)**(1 + elast_cond) - 1
            else:
                rec = 'MANTENER'
                ingreso_5pct = (1.05)**(1 + elast_cond) - 1
            filas_rec.append({
                'precio': precio,
                'categoria': cat,
                'elasticidad_rcs_global': elast_rcs,
                'elasticidad_categoria': elast_cat,
                'elasticidad_condicional': round(elast_cond, 4),
                'recomendacion': rec,
                'delta_ingreso_5pct': round(ingreso_5pct * 100, 2),
            })

    df_rec = pd.DataFrame(filas_rec)
    df_rec.to_csv(DASH_DIR / 'recomendaciones_pricing.csv', index=False)
    print(f"  -> {DASH_DIR / 'recomendaciones_pricing.csv'} ({len(df_rec)} filas)")

    # Resumen de recomendaciones
    print("\n  Resumen recomendaciones:")
    print(df_rec['recomendacion'].value_counts().to_string())

    # ============================================================
    # 4. KPIs ejecutivos actualizados
    # ============================================================
    print("\n[4/4] Actualizando kpis_ejecutivos.csv...")
    # Cargar resultados del gate
    with open(JSON_RESULTADOS, 'r', encoding='utf-8') as f:
        resultados = json.load(f)

    wild = resultados.get('comparacion_ee', {}).get('wild_cluster_bootstrap', {})

    kpis = [
        ('Total ordenes', df['num_orders'].sum()),
        ('Precio promedio', round(df['checkout_price'].mean(), 2)),
        ('Elasticidad OLS log-log (asociacion promedio)', resultados['ols_controles']['coef']),
        ('Elasticidad Panel FE (benchmark global)', resultados['panel_fe_twoway']['coef']),
        ('Elasticidad IV 2SLS (sensibilidad, Hansen J rechazado)', resultados['iv_2sls_2inst']['coef']),
        ('Elasticidad RCS min (precio bajo)', min(df_rcs_out['elasticidad_rcs'])),
        ('Elasticidad RCS max (precio alto)', max(df_rcs_out['elasticidad_rcs'])),
        ('Wild cluster bootstrap media', wild.get('elasticidad_media')),
        ('Wild cluster bootstrap IC inf', wild.get('ic_95_inferior')),
        ('Wild cluster bootstrap IC sup', wild.get('ic_95_superior')),
        ('EE multiway (inferencia principal)', resultados['comparacion_ee']['ols_multiway_center_meal']['ee']),
        ('Ratio cluster/HC3', resultados['comparacion_ee']['ratio_cluster_hc3']),
        ('N observaciones', len(df)),
        ('N centros unicos', df['center_id'].nunique()),
        ('N comidas unicas', df['meal_id'].nunique()),
        ('Semanas cubiertas', df['week'].nunique()),
        ('Quantile tau=0.1 (demanda baja)', resultados['quantile_regression']['tau_0.1']),
        ('Quantile tau=0.5 (mediana)', resultados['quantile_regression']['tau_0.5']),
        ('Quantile tau=0.9 (demanda alta)', resultados['quantile_regression']['tau_0.9']),
    ]
    df_kpis = pd.DataFrame(kpis, columns=['metrica', 'valor'])
    df_kpis.to_csv(DASH_DIR / 'kpis_ejecutivos.csv', index=False)
    print(f"  -> {DASH_DIR / 'kpis_ejecutivos.csv'}")

    print("\n" + "=" * 70)
    print("Dashboard condicional COMPLETADO")
    print("=" * 70)
    print(f"Archivos generados en {DASH_DIR}:")
    for f in sorted(DASH_DIR.glob('*.csv')):
        print(f"  - {f.name}")
    print("""
Logica de recomendaciones:
  - Elasticidad RCS LOCAL por nivel de precio (operativa para pricing)
  - Elasticidad Panel FE -1.5755 como BENCHMARK GLOBAL (elasticidad constante)
  - Recomendacion: SUBIR si eta > -0.9 | MANTENER si -1.1 < eta < -0.9 | BAJAR si eta < -1.1
  - Delta ingreso 5%: formula exacta (1+d)^(1+eta) - 1
""")


if __name__ == '__main__':
    main()
