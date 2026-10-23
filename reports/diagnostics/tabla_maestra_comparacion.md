# Tabla maestra de comparación (V3.2.5)

| Modelo | Estimación | EE | IC 95% | N/iter. | Interpretación |
|---|---:|---:|---|---:|---|
| OLS log-log base | -0.9516 | 0.0034 | [-0.958,-0.945] | 456548 | Estimación base; EE HC3 (bootstrap i.i.d.: DE 0.0034, IC [-0.9578,-0.9450]); no reproduce dependencia panel |
| OLS log-log controles | -0.7590 | 0.0430 | [-0.843,-0.675] | 456548 | Especificación principal; EE agrupado por entidad center_id×meal_id |
| Wild Bootstrap OLS | -0.7585 | 0.0422 | [-0.83798,-0.67297] | 500 | Sensibilidad; pesos de Rademacher por centro; 500/500 válidas |
| Bootstrap-t OLS | -0.7590 |  | [-0.84431,-0.67625] | 2000 | Sensibilidad; bootstrap-t; 2000/2000 válidas; por center_id |
| Jackknife leave-one-center-out | -0.7590 | 0.0423 | [-0.842,-0.676] | 77 | EE JK=sqrt(n-1)·DE(0.00485)=0.0423; rango coeficientes [-0.76935,-0.75064]; máximo cambio absoluto 0.01039 |
| OLS semi-log controles | -1.0742 | 0.0062 | [-1.086,-1.062] | 456548 | Elasticidad equivalente en precio medio; no constante |
| Panel FE 2 vías | -1.5985 | 0.0239 | [-1.645,-1.552] | 456548 | Benchmark within; cluster por entidad panel |
| Pairs Bootstrap Panel FE | -1.5992 | 0.0271 | [-1.65060,-1.54553] | 1000 | Sensibilidad; remuestreo por centro y relabeling; 1000/1000 válidas |
| IV 2SLS (2 instrumentos, con controles) | -0.8414 | 0.0502 | [-0.940,-0.743] | 456544 | Sensibilidad; EE agrupado center_id; Hansen J=33.28 (HC) / 0.92 p=0.34 (agrupado); exclusión no contrastable |
| IV justamente identificado (base_price, con controles) | -0.8545 | 0.0475 | [-0.948,-0.761] | 456544 | Sensibilidad; identificación exacta; sin Hansen J |
| Wild Bootstrap IV sin controles (1ª etapa fija) [histórico] | -0.9606 | 0.0268 | [-1.01173,-0.90791] | 2000 | Referencia histórica: especificación SIN controles (-0.9604); no corresponde al IV principal |
| Wild Bootstrap IV sin controles (ambas etapas) [histórico] | -0.9607 | 0.0281 | [-1.01414,-0.90430] | 2000 | Referencia histórica: especificación SIN controles (-0.9604); no corresponde al IV principal |
| RCS OLS (4 nudos P5/P35/P65/P95) at P50 | -1.4429 |  |  | 456548 | Elasticidad puntual; P5=0.0377; P50=-1.4429; P95=-1.2479; AIC=1124067.47; R²adj=0.53795 |
| Bootstrap RCS bandas punto a punto | -1.5106 |  | P5 [-0.1009,0.1601]; P50 [-1.6550,-1.3641]; P95 [-1.3650,-1.1383] | 2000 | Bootstrap por centro (2000); valor mínimo de la curva ≈ precio 331; IC 95% punto a punto no simultáneo |
| IV Control Function (RCS + vhat) at precio medio | -1.5041 | 0.0177 |  | 456544 | Sensibilidad; R²adj=0.5412; EE convencional; no causal |
| Bootstrap Control Function v_hat | 0.9937 | 0.1654 | [0.66419,1.31352] | 5000 | Bootstrap por centro, ambas etapas (5000); IC de v_hat excluye cero (corrige etiqueta previa que mostraba rcs_1) |
| Quantile τ=0.10 | -0.6685 |  |  | 456548 | Elasticidad condicional |
| Quantile τ=0.25 | -0.7330 |  |  | 456548 | Elasticidad condicional |
| Quantile τ=0.50 | -0.7748 |  |  | 456548 | Elasticidad condicional |
| Quantile τ=0.75 | -0.8760 |  |  | 456548 | Elasticidad condicional |
| Quantile τ=0.90 | -1.0068 |  |  | 456548 | Elasticidad condicional |
| Wild Bootstrap IV con controles (ambas etapas) | -0.8410 | 0.0504 | [-0.93976,-0.74236] | 2000 | V3.2.5; Rademacher por center_id; réplicas en reports/certified/wild_bootstrap_iv_2sls_controles.npz |
| IV Control Function: elasticidad en precio medio (bootstrap) | -1.5041 | 0.0663 | [-1.62994,-1.37015] | 5000 | V3.2.5; banda de la elasticidad CF (réplicas en bootstrap_cf_canonical_replicas.npz) |
| IV Control Function: elasticidad en mediana (bootstrap) | -1.3945 | 0.0727 | [-1.53390,-1.24851] | 5000 | V3.2.5; mediana 296.82 |
