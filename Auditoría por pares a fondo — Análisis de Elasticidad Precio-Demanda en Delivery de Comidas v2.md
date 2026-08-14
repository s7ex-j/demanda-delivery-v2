# Auditoría por pares a fondo
## Análisis de Elasticidad Precio-Demanda en el Servicio de Delivery de Comidas — v2

**Documento auditado:** informe.pdf, bitácora metodológica, estructura del proyecto y bibliografía `jaes.bib`  
**Tipo de revisión:** revisión por pares econométrica, metodológica, estadística y de reproducibilidad  
**Criterio:** estándar de un working paper empírico serio / artículo aplicado, no únicamente estándar de proyecto académico  
**Veredicto global:** **revisión mayor antes de presentar el trabajo como evidencia causal**

---

## 1. Veredicto ejecutivo

El proyecto es **muy superior a un análisis exploratorio convencional**. Tiene varias virtudes excepcionales para un proyecto individual: pipeline reproducible, trazabilidad de decisiones, múltiples especificaciones, panel de efectos fijos, IV, pruebas de robustez, auditoría de citas, SQL/BI y una disposición poco común a documentar resultados que contradicen las hipótesis iniciales. La documentación demuestra además que se corrigieron errores reales de versiones anteriores en vez de esconderlos. La integridad del merge y la ausencia de nulos están bien documentadas, y el dataset final tiene 456,548 observaciones.

Sin embargo, **el núcleo econométrico todavía no sostiene las afirmaciones causales fuertes que el informe llega a hacer**.

Mi evaluación sería:

| Dimensión | Evaluación |
|---|---:|
| Integridad y trazabilidad de datos | **8.5/10** |
| Ingeniería/reproducibilidad | **9/10** |
| Exploración y diagnósticos | **8/10** |
| Econometría descriptiva | **8/10** |
| Identificación causal | **4.5/10** |
| Inferencia estadística | **5.5/10** |
| Especificación funcional | **5/10** |
| Literatura y citación | **7/10** |
| Traducción de resultados a negocio | **5.5/10** |
| Calidad global actual | **6.5/10** |
| Potencial tras correcciones | **8.5–9/10** |

**Decisión de referee:** **Major Revision / Revise and Resubmit.**

No recomiendo publicar actualmente la frase equivalente a **“−0.9487 es el mejor estimado causal disponible”**. El informe demuestra endogeneidad, pero no demuestra la validez del supuesto de exclusión de los instrumentos; de hecho, el Hansen J rechaza la validez conjunta. La estabilidad del estimado con un único instrumento no resuelve ese problema.

La conclusión mucho más defendible hoy es:

> **El dataset contiene evidencia fuerte de una relación negativa entre precio de checkout y pedidos, pero no identifica todavía de manera convincente una única elasticidad causal. Las estimaciones varían materialmente según la fuente de variación utilizada, y la relación además es no lineal y heterogénea.**

Ese cambio de framing sería importante y, paradójicamente, haría el trabajo **más sólido**, no más débil.

---

# 2. El problema central: el proyecto encontró que su modelo principal está mal especificado, pero sigue tratando ese modelo como resultado principal

Este es probablemente el hallazgo más importante de toda la auditoría.

El modelo principal es:

\[
\log(1+Q_{it})=\alpha+\beta\log(P_{it})+X_{it}\gamma+\epsilon_{it}
\]

con \(\hat\beta=-0.7513\).

Pero posteriormente el propio proyecto encuentra:

- RESET: **F = 119.57, p = 1.21×10⁻⁵²**
- RCS: **F = 10,279.37, p ≈ 0**
- interacción precio × promoción: **F = 659.73, p ≈ 0**
- elasticidad cuantil: aproximadamente **−0.66 a −1.00**
- heterogeneidad temporal en FE: aproximadamente **−1.25 a −1.85**
- elasticidad RCS reportada desde **+0.58 hasta −3.22** según el nivel de precio.

Es decir: el propio proyecto ya encontró evidencia abrumadora de que **“una elasticidad constante de −0.7513” no describe adecuadamente la relación estructural**.

Eso convierte al −0.7513 en un **resumen lineal promedio**, no en una descripción completa de la demanda.

### Problema editorial

El informe dice primero que la especificación log-log es la principal y luego descubre que es funcionalmente incorrecta. Sin embargo, conclusiones y decisiones de pricing siguen girando alrededor del −0.7513.

La jerarquía debería invertirse:

**Resultado principal actualizado**
1. relación precio-demanda negativa;
2. relación no lineal;
3. elasticidad heterogénea por nivel de precio;
4. heterogeneidad por categoría;
5. heterogeneidad temporal;
6. estimaciones causales solo bajo supuestos adicionales.

El −0.7513 debería presentarse como:

> **elasticidad promedio de la proyección log-log lineal de referencia**

y no como “la elasticidad de la plataforma”.

---

# 3. Hallazgo crítico P0: la inferencia OLS principal no está corregida adecuadamente por la dependencia del panel

El informe encuentra:

**Durbin-Watson = 1.195**, lo que indica dependencia positiva en los residuos. El propio texto reconoce explícitamente que HC3 es robusto a heterocedasticidad, **pero no a autocorrelación**.

Esto sería suficiente para exigir una modificación de la inferencia.

Pero existe una evidencia todavía más fuerte.

En la propia Tabla 23, el error estándar de:

- OLS HC3 = **0.0053**
- OLS clusterizado por `center_id` = **0.0421**

Es decir, el EE clusterizado es aproximadamente **7.92 veces mayor**.

Eso es enorme.

Por lo tanto, el intervalo:

\[
[-0.762,-0.741]
\]

alrededor de −0.7513 es **muchísimo más optimista de lo que debería ser** si existe dependencia intra-centro.

El problema no es que el coeficiente probablemente cambie mucho; el problema es que **la incertidumbre estadística reportada puede estar severamente subestimada**.

Cameron y Miller explican precisamente que, cuando hay correlación arbitraria dentro de clusters, los errores estándar convencionales pueden sobreestimar artificialmente la precisión y que la elección del nivel de clustering importa; también discuten clustering multiway y número reducido de clusters.

### Qué debería hacerse

Para el OLS principal, volvería a estimar al menos:

\[
\log Q_{ict}
=
\beta \log P_{ict}
+\alpha_c+\gamma_t+X_{ict}\delta+\epsilon_{ict}
\]

con errores:

- cluster por `center_id`;
- cluster por `meal_id`;
- **multiway clustering `center_id × meal_id`**;
- y como inferencia de robustez, **wild cluster bootstrap**.

No elegiría el estimador de varianza porque produzca un p-value más conveniente.

### Punto importante

El argumento “N = 456,548 es enorme” no resuelve esto. El tamaño muestral no crea 456,548 unidades independientes. El número de clusters relevantes es mucho menor.

---

# 4. Hallazgo crítico P0: el IV es interesante, pero todavía no es creíble como identificación causal

Este es el segundo gran problema.

El proyecto hace algo correcto: **no inventa instrumentos** y construye dos instrumentos realmente calculables del dataset:

1. `base_price`
2. precio medio de la misma comida en otros centros en la misma semana.

Las correlaciones con el precio son 0.956 y 0.986; la primera etapa tiene F ≈ 1.46×10⁷. Por fuerza predictiva, los instrumentos son extraordinariamente fuertes.

Pero:

\[
Hansen\ J=320.1,\quad p\approx0
\]

rechaza la validez conjunta.

Eso no puede convertirse en:

> “Hansen J rechaza, pero como el resultado cambia solo 0.014 con un instrumento, entonces el IV es confiable.”

Esa inferencia **no está justificada**.

Un modelo justamente identificado no permite hacer Hansen J. Que el estimado con `base_price` sea −0.9627 no demuestra que `base_price` sea exógeno.

### 4.1 El problema con `base_price`

El propio proyecto interpreta `base_price` como un **precio de referencia** y construye `hubo_aumento` precisamente para capturar efectos de referencia.

Por tanto, hay una posible violación muy seria de exclusión:

\[
base\_price \rightarrow reference\ perception
\rightarrow demand
\]

y no solamente:

\[
base\_price \rightarrow checkout\_price \rightarrow demand
\]

Si el consumidor responde al precio de referencia o al descuento percibido, `base_price` tiene un canal directo hacia demanda.

Esto es especialmente incómodo porque el proyecto usa precisamente Kahneman et al. para justificar que las referencias de precio pueden afectar comportamiento.

El instrumento podría estar violando el supuesto que el propio trabajo considera económicamente relevante.

### 4.2 El problema con el precio de otros centros

El segundo instrumento también tiene una amenaza clara:

\[
P_{other,ct}
\]

puede reflejar shocks comunes de:

- demanda regional;
- campañas de plataforma;
- disponibilidad;
- estacionalidad;
- costes comunes;
- cambios corporativos de pricing.

La propia bitácora reconoce esta posibilidad.

Por eso no aceptaría la frase:

> “La estabilidad del resultado confirma que el rechazo de Hansen se debe a colinealidad/poder.”

**No lo confirma.**

Puede ser compatible con esa hipótesis, pero también con que ambos instrumentos compartan un problema de exclusión.

### 4.3 El etiquetado “Hausman-Nevo” es demasiado fuerte

Hay otra cuestión bibliográfica/metodológica.

La literatura de Hausman/Nevo sobre instrumentos de precios en mercados diferenciados es bastante más específica que simplemente “precio de la misma comida en otros centros”.

Hausman tiene trabajos sobre identificación mediante variación común de costes y demanda entre mercados, mientras Nevo trabaja dentro del marco de demanda de productos diferenciados y estructura industrial.

Por ello recomiendo:

> **No llamar a esto “estrategia Hausman-Nevo” sin explicar exactamente qué supuesto, ecuación y construcción instrumental provienen de qué referencia.**

---

# 5. Hallazgo crítico P0: el “placebo temporal” no es un placebo válido

Este es, a mi juicio, el error metodológico más claro detectado en la versión más reciente.

El placebo hace esto:

> randomiza la etiqueta `week` entre observaciones, manteniendo precio y demanda exactamente emparejados como originalmente.

El informe reporta:

- elasticidad real = −0.7483
- placebo = −0.7513
- desviación placebo = 0.0001
- p = 1.0.

Y concluye que la “alineación temporal” fue destruida.

Pero **no se destruyó la relación precio-demanda observacional**.

Cada fila sigue teniendo exactamente:

\[
(P_i,Q_i)
\]

original.

Solo cambió el número almacenado en `week`.

Por tanto, la regresión sigue viendo la misma correlación precio-demanda.

El resultado de que el placebo sea casi idéntico al real es, en realidad, lo esperado bajo esa permutación.

### Esto debe corregirse de raíz

Yo eliminaría completamente este placebo de las conclusiones.

Un placebo temporal válido tendría que romper la relación temporal relevante.

Ejemplos:

**Placebo 1 — precio futuro**

Regresar:

\[
Q_{it}
\sim P_{i,t+1}
\]

o una versión lead apropiada.

Un precio futuro no debería explicar causalmente demanda presente después de controlar adecuadamente.

**Placebo 2 — permutación de precio dentro de entidad**

Permutar los precios **dentro de cada `center_id × meal_id`** preservando la distribución, pero rompiendo su alineación temporal con pedidos.

**Placebo 3 — block permutation**

Permutar bloques temporales preservando dependencia local.

**Placebo 4 — pseudo-treatment**

Aplicar artificialmente un “shock de precio” a periodos donde no ocurrió.

Este punto debe estar en la lista de correcciones obligatorias.

---

# 6. Hallazgo crítico P0: el RCS presenta una posible inconsistencia en el test formal de no-linealidad

El informe dice que usa 4 nudos y afirma que el test de Wald de no-linealidad tiene **4 grados de libertad**.

Pero una base de spline restringida debe revisarse con mucho cuidado aquí.

Si el modelo tiene:

- 1 componente lineal;
- 3 componentes no lineales adicionales;

entonces la hipótesis:

\[
H_0:\text{“no hay no-linealidad”}
\]

debería restringir solamente los componentes no lineales.

Eso normalmente produce **3 restricciones**, no 4.

El informe, sin embargo, presenta:

\[
F=10,279.37,\quad 4\ gl
\]

y lo interpreta como prueba de “linealidad constante”.

No puedo afirmar sin inspeccionar el código que el test esté mal implementado, pero **la especificación tal como está documentada levanta una bandera roja importante**.

### Acción obligatoria

Revisar el código que genera:

- la matriz de restricciones;
- qué columnas del spline se están probando;
- cuántos grados de libertad corresponden a los términos no lineales.

No basta con que `p ≈ 0`.

El test debe probar exactamente:

\[
H_0:\beta_{NL,1}=\beta_{NL,2}=\cdots=0
\]

mientras permite que sobreviva el término lineal.

---

# 7. Hallazgo crítico P1: `log(1 + num_orders)` no debe presentarse alegremente como una elasticidad convencional

El informe define:

\[
y=\ln(1+Q)
\]

y luego interpreta directamente el coeficiente como:

> “un aumento de 1% en precio reduce 0.7513% los pedidos”.

El propio informe reconoce que el mínimo observado es 13, así que en realidad **no existen ceros en la muestra**.

Por tanto, la pregunta natural es:

**¿por qué usar `log1p`?**

Si:

\[
Q \ge 13
\]

para toda la muestra, entonces:

\[
\ln(1+Q)
\]

no es necesaria para definir el logaritmo.

Además:

\[
\frac{\partial\ln(1+Q)}{\partial\ln P}
=
\frac{Q}{Q+1}
\frac{\partial\ln Q}{\partial\ln P}
\]

Por lo tanto, el coeficiente de `log1p(Q)` **no es exactamente la elasticidad de \(Q\)**.

Probablemente la diferencia sea pequeña para \(Q\) altos, pero el problema es conceptual, no numérico.

### Recomendación

Usar:

\[
\ln(Q)
\]

para el análisis de elasticidad principal si los ceros realmente no existen.

Y, de forma secundaria:

- PPML si se incorporan observaciones con cero;
- modelos de conteo como robustez.

---

# 8. Hallazgo P1: la unidad monetaria “S/” no está demostrada por la fuente

El informe etiqueta todos los precios como soles:

> “precio final (S/)”.

Pero la fuente pública del dataset define `checkout_price` como precio final incluyendo descuento, impuestos y cargos de entrega; no establece en esa descripción que la unidad monetaria sea PEN/soles.

Esto es una señal de alerta documental.

### Debe decirse

> “unidades monetarias del dataset”

o indicar la moneda solamente si la documentación primaria de origen la confirma.

No asumiría “S/”.

---

# 9. Hallazgo P1: `checkout_price` no es necesariamente “precio del alimento”

La fuente del dataset define:

> `checkout_price`: **Final price including discount, taxes & delivery charges**.

Esto es importantísimo.

Entonces el parámetro estimado no es necesariamente:

\[
\text{elasticidad de la demanda del precio del plato}
\]

sino algo más parecido a:

\[
\text{elasticidad de pedidos respecto del precio final de checkout}
\]

que puede incluir:

- precio del alimento;
- descuento;
- impuestos;
- cargos de entrega.

Eso cambia sustancialmente la interpretación económica.

### Título más preciso

En lugar de:

> Elasticidad precio-demanda del servicio de delivery

preferiría:

> **Elasticidad de los pedidos respecto del precio final de checkout en un dataset de food delivery**

o una formulación equivalente.

---

# 10. Hallazgo P1: la procedencia del dataset está descrita con demasiada certeza

El dataset proviene de un challenge orientado originalmente a **forecasting de demanda** para las semanas 146–155.

Eso no invalida el proyecto.

Pero sí exige una distinción entre:

> “dataset transaccional real”

y:

> “dataset público histórico de una competencia basada en datos de una empresa de delivery”.

La segunda formulación es metodológicamente más prudente.

La fuente pública demuestra muy bien la estructura del problema y las variables, pero no proporciona toda la trazabilidad que uno esperaría para afirmar que se dispone de un registro empresarial auditado y directamente recogido con fines de investigación.

Yo bajaría un poco la intensidad de frases como:

> “datos transaccionales reales”

a:

> “datos históricos agregados de pedidos a nivel centro-comida-semana provenientes de un dataset público de forecasting de food delivery”.

Eso es mucho más difícil de atacar en una revisión.

---

# 11. Hallazgo P1: no hay observaciones con cero pedidos

El mínimo de `num_orders` reportado es 13.

Esto plantea una pregunta que el paper todavía no responde:

> ¿Qué ocurrió con las combinaciones centro-comida-semana con cero demanda?

El dataset original fue creado para predecir demanda de combinaciones centro-meal, pero el análisis que se entrega no demuestra si:

1. los ceros fueron eliminados en la construcción original;
2. solo se registraron combinaciones activas;
3. nunca existieron;
4. fueron filtrados.

Esto puede generar **selección sobre el margen extensivo**.

Si un precio alto hace que un producto pase de 5 pedidos a 0, y esos ceros no existen en la muestra, el estimador puede describir únicamente la respuesta **condicional entre combinaciones con demanda positiva**.

### Debe añadirse

Una sección explícita:

**“Cobertura del margen extensivo y ausencia de observaciones con cero pedidos.”**

---

# 12. Hallazgo P1: el modelo OLS principal no controla suficientemente la heterogeneidad espacial/centro

El OLS utiliza:

- categoría;
- tipo de centro;
- promociones;
- `hubo_aumento`.

Pero no usa:

- `center_id`;
- `region_code`;
- `city_code`;
- `op_area`.

Eso es una limitación importante.

El propio dataset tiene esas variables.

`center_type` es solo tres categorías.

Decir que controla “heterogeneidad geográfica” mediante `center_type` es demasiado fuerte.

### Buena noticia

El modelo FE posteriormente soluciona buena parte de esto porque absorbe la heterogeneidad invariante de cada combinación centro-comida.

Esto refuerza mi conclusión de que el OLS con controles simples **no debería seguir ocupando el papel de estimación principal**.

---

# 13. Hallazgo P1: el clustering del panel probablemente debería reconsiderarse

El panel FE usa:

\[
center\_id \times meal\_id
\]

como entidad y clusteriza por esa entidad.

Eso captura dependencia longitudinal dentro de una combinación específica.

Pero puede existir dependencia:

\[
\epsilon_{c,m,t}
\leftrightarrow
\epsilon_{c,m',t}
\]

porque varios platos del mismo centro comparten:

- personal;
- capacidad;
- shocks logísticos;
- clima local;
- demanda local;
- campañas;
- restricciones de capacidad.

El cluster por `center-meal` no captura necesariamente eso.

Dado que hay solo 77 centros, una opción razonable es:

- cluster por `center_id`;
- o multiway clustering;
- y wild cluster bootstrap como robustez.

La literatura de inferencia clusterizada recomienda explícitamente considerar estructura de clusters, pocos clusters y clustering multiway.

---

# 14. Hallazgo P1: el resultado de Quantile Regression está sobreinterpretado

El trabajo afirma:

> “los consumidores con alta demanda ... son más sensibles al precio”.

Pero una regresión cuantil de:

\[
Q_\tau(Y|X)
\]

no identifica automáticamente diferentes **tipos de consumidores**.

Un cuantil alto del número de pedidos puede estar generado por:

- ciertos productos;
- centros grandes;
- ciertas semanas;
- categorías;
- promociones;
- combinaciones centro-meal con demanda estructuralmente alta.

Por tanto:

\[
\tau=0.9
\]

no equivale necesariamente a:

> “consumidores frecuentes”.

La evidencia demuestra heterogeneidad de la función de demanda condicional; **no identifica segmentos conductuales de consumidores**.

La frase debe corregirse.

---

# 15. Hallazgo P1: la interacción precio × promoción está interpretada incorrectamente

El informe encuentra interacciones significativas.

Pero cuando el precio no está centrado:

\[
\log Q =
\beta_P\log P
+\beta_D D
+\beta_{PD}(\log P\times D)
+\cdots
\]

el coeficiente de \(D\):

\[
\beta_D
\]

es el efecto de promoción cuando:

\[
\log P=0
\]

es decir:

\[
P=1
\]

en las unidades del dataset.

Eso no es el efecto de promoción al precio medio.

La interpretación económica debe hacerse mediante **efectos marginales a precios relevantes**, por ejemplo:

- P10,
- P25,
- mediana,
- P75,
- P90.

Además, si el término de interacción es positivo, la lectura de cómo cambia el efecto de promoción con el precio debe revisarse algebraicamente.

### Solución

Centrar:

\[
\log P_c=\log P-\overline{\log P}
\]

y entonces el coeficiente de promoción será interpretable aproximadamente al precio medio.

---

# 16. Hallazgo P1: la traducción de elasticidad a ingresos está numéricamente mal planteada

El informe afirma que con elasticidad −0.7513:

> +5% precio → −3.75% cantidad → **+1.06% ingresos**

y:

> −5% precio → +3.75% cantidad → **−1.44% ingresos**.

Con un modelo log-log constante, la manera coherente de hacerlo es:

\[
Q_1/Q_0=(P_1/P_0)^\eta
\]

No:

\[
Q_1/Q_0=1+\eta \Delta P
\]

Para \(\eta=-0.7513\):

### Precio +5%

\[
R_1/R_0=1.05(1.05^{-0.7513})
\]

≈ **1.0122**

→ **+1.22% ingresos**, no +1.06%.

### Precio −5%

\[
R_1/R_0=0.95(0.95^{-0.7513})
\]

≈ **0.9873**

→ **−1.27%**, no −1.44%.

Y lo más importante:

**las recomendaciones cambian radicalmente si se cambia de estimador.**

Con \(\eta=-0.9487\):

- +5% precio → aproximadamente **+0.25% ingreso**
- −5% precio → aproximadamente **−0.26%**

Con \(\eta=-1.5755\):

- +5% precio → aproximadamente **−2.77%**
- −5% precio → aproximadamente **+3.00%**

Por tanto, el dashboard no debería decir simplemente:

> “subir precio aumenta ingresos”

porque **esa conclusión depende del estimador elegido**.

---

# 17. Hallazgo P1: el modelo de negocio está utilizando una elasticidad que el propio proyecto encontró que no es constante

Este es un problema aún mayor.

El RCS reporta elasticidades muy distintas a lo largo del precio.

Entonces una recomendación como:

> “subir precio 5%”

necesita responder:

**¿subir el precio de qué producto, desde qué precio inicial y en qué categoría?**

Una política de pricing basada en:

\[
\eta=-0.7513
\]

puede ser razonable como aproximación agregada, pero no como regla operativa granular.

Esto contradice parcialmente la recomendación del propio proyecto de traducir el resultado directamente a pricing.

---

# 18. Hallazgo P1: el modelo semi-log no debería descartarse solo porque su condition number sea alto

El proyecto hace algo correcto al investigar la advertencia numérica y descubrir que no parece ser multicolinealidad real.

Pero la selección final:

> “log-log gana porque es interpretable y el semi-log tiene condition number alto”

es demasiado débil.

La evidencia muestra:

- semi-log tiene **AIC menor**;
- BIC menor;
- R² ajustado mayor.

Pero posteriormente aparece una no-linealidad fuerte.

La solución moderna no es elegir entre:

- log-log constante;
- semi-log constante en pendiente.

La solución es estimar una especificación flexible y estable.

Por eso el RCS es mucho más relevante para la versión final.

---

# 19. Hallazgo P1: el bootstrap simple del modelo base no aporta la robustez que el lector puede creer

El informe lo reconoce honestamente:

el bootstrap se ejecutó para el **modelo base sin controles**, no para el modelo principal con controles.

Además, posteriormente se demuestra que el bootstrap iid tiene un error estándar aproximadamente 0.63 veces el HC3, precisamente porque no reproduce la estructura de dependencia.

Eso significa que el bootstrap simple:

\[
EE=0.0034
\]

no debe utilizarse como evidencia fuerte de precisión.

La documentación posterior ya reconoce esto correctamente.

Mi recomendación es simplemente **relegarlo al apéndice** y usar como inferencia principal cluster-robust / bootstrap clusterizado.

---

# 20. Hallazgo P1: falta aclarar la inferencia del control function con regresor generado

El control function implementado es metodológicamente interesante. Wooldridge documenta precisamente el uso de control functions para variables endógenas en modelos lineales y no lineales.

Pero cuando se estima:

1. primera etapa → \(\hat v\);
2. segunda etapa → \(Y\sim spline(P)+\hat v+X\),

la varianza de segunda etapa debe tratar correctamente el hecho de que \(\hat v\) es **regresor generado**.

El informe reporta tests F extremadamente precisos, pero la documentación presentada no demuestra con suficiente claridad que toda la inferencia tenga la corrección apropiada.

No digo que los coeficientes sean incorrectos.

Digo que:

> **la inferencia del control function necesita una auditoría de implementación y una descripción explícita del estimador de varianza.**

---

# 21. Hallazgo P1: la ausencia de fecha calendario limita muchísimo la interpretación temporal

El informe reconoce que solo existen semanas 1–145 y no fechas reales.

Esto significa que no pueden identificarse directamente:

- Navidad;
- vacaciones;
- COVID;
- inflación;
- cambios regulatorios;
- temporadas;
- feriados;
- eventos locales.

El panel FE por semana controla shocks comunes, pero **no permite interpretarlos**.

Por ello evitaría hablar de:

> “estacionalidad”

como si estuviera identificada económicamente.

Es más preciso decir:

> “efectos temporales comunes por semana”.

---

# 22. Hallazgo P1: el resultado “hubo_aumento no significativo” no debe interpretarse como ausencia de efecto de precio de referencia

El coeficiente de `hubo_aumento` es −0.0024, p=.401.

Eso solamente dice que **ese indicador binario adicional**, condicionado a `log_checkout_price`, no añade poder explicativo estadísticamente detectable.

No demuestra:

> “los consumidores no reaccionan a reference pricing”.

Porque `hubo_aumento` es una representación muy pobre del concepto de referencia.

Podrían importar:

- magnitud del descuento;
- porcentaje de descuento;
- distancia respecto al precio de referencia;
- saliencia;
- duración;
- cambios de referencia;
- interacciones con promoción.

Así que la interpretación debe ser:

> “no se detecta un efecto adicional de esta variable binaria específica, condicionado al precio final.”

---

# 23. Hallazgo P1: las afirmaciones de “promociones +65% / +91%” son asociacionales, no necesariamente efectos causales

El OLS encuentra:

- emailer = +0.5028
- homepage = +0.6492.

Convertirlos mediante:

\[
e^\beta-1
\]

es matemáticamente correcto para una dummy en un modelo log-lineal, pero la interpretación:

> “la promoción aumenta pedidos 65%”

es causal solo si la asignación de promoción es exógena.

En una plataforma es muy plausible que los productos destacados sean precisamente los productos:

- con alta demanda prevista;
- estratégicos;
- de mayor margen;
- que la empresa quiere impulsar.

Por tanto:

> “las observaciones con promoción presentan aproximadamente 65% más pedidos”

es defendible.

Mientras:

> “la promoción aumenta los pedidos 65%”

requiere identificación adicional.

---

# 24. Hallazgo P1: la categoría no es una simple “robustez”; hay heterogeneidad estructural real

El proyecto encuentra que la igualdad de elasticidades entre categorías se rechaza fuertemente y que las elasticidades varían ampliamente.

Esto es muy importante.

Si:

\[
\eta_k
\]

cambia considerablemente entre categorías, el parámetro global −0.7513 es una mezcla.

Esto sugiere que la siguiente versión debería estimar directamente:

\[
\log Q =
\beta\log P
+
\sum_k \theta_k(\log P\times Category_k)
+
FE
+
\epsilon.
\]

Eso sería económicamente más informativo que únicamente probar igualdad después.

---

# 25. Hallazgo P1: la heterogeneidad temporal merece más atención que el placebo defectuoso

El FE con interacción temporal sí es interesante.

El resultado:

\[
\chi^2(144)=3068.85,\quad p\approx0
\]

y rango semanal aproximadamente:

\[
-1.85\;\text{a}\;-1.25
\]

es evidencia real de heterogeneidad temporal.

Eso merece permanecer.

Lo que debe eliminarse es el placebo temporal mal diseñado.

De hecho, la heterogeneidad temporal es mucho más útil para la historia económica del trabajo que el placebo.

---

# 26. Literatura: sólida en intención, pero necesita una última limpieza

La auditoría bibliográfica es una de las mejores partes del proyecto. La bitácora documenta tres rondas de verificación y correcciones de errores previos.

Las fuentes principales revisadas son reales y pertinentes: Andreyeva et al., Damayanti et al., Christensen et al., Zhang & Demirkan, Sriwongphanawes & Fukuda, Li & Wang, entre otras. El propio PDF contiene sus referencias.

Sin embargo, detecto una **inconsistencia entre la sofisticación metodológica del texto y la bibliografía final**.

El texto invoca explícitamente:

- Hausman;
- Nevo;
- Wooldridge;
- Stock-Yogo;
- Ramsey/RESET;

pero estas referencias metodológicas no aparecen completas en la bibliografía `jaes.bib` que acompañó la entrega.

Esto debe corregirse.

Por ejemplo, Wooldridge 2015 existe en *Journal of Human Resources* 50(2), 420–445.

Cameron & Miller (2015), extremadamente relevantes para el problema de clustering encontrado, también deberían figurar.

Hausman tiene la referencia original correspondiente a la metodología citada.

---

# 27. La comparación con literatura debe ser menos “puntualista”

El cuadro es útil, pero comparar:

- FAFH;
- delivery;
- Uber;
- CPG online/offline;
- ride-hailing;

en una sola escala de elasticidades puede inducir a la falsa impresión de comparabilidad.

El informe reconoce esta limitación, lo cual es positivo.

Pero yo haría una separación gráfica mucho más estricta:

### A. Comparables directos
- delivery;
- comida preparada;
- datos de pedidos.

### B. Parcialmente comparables
- FAFH;
- e-commerce.

### C. Solo contexto metodológico
- Uber.

Y no mezclaría sus intervalos como si fueran una evidencia conjunta.

---

# 28. La fuente de Andreyeva está bien usada; cuidado con el signo

La fuente reporta elasticidades de demanda en FAFH alrededor del rango general 0.27–0.81 en magnitud y el informe usa −0.81 como punto concreto. El paper original es una revisión sistemática/meta-análisis de 160 estudios. La comparación es pertinente como benchmark, pero no como estimador sustituto del delivery.

---

# 29. La afirmación sobre Li & Wang debería suavizarse

El informe presenta Li & Wang como confirmación de que prácticamente no existen estudios empíricos previos sobre elasticidad en plataformas de delivery.

La contribución del paper es ciertamente relevante para el contexto de plataformas on-demand, pero recomendaría no utilizar una frase tan absoluta.

Mejor:

> “La literatura empírica sobre delivery on-demand sigue siendo relativamente escasa, lo que refuerza el interés de estimar elasticidades con datos de plataformas.”

Es una formulación menos vulnerable.

---

# 30. La contribución real del proyecto no es necesariamente “encontrar −0.7513”

A mi juicio, la contribución más interesante es otra:

### El proyecto descubre simultáneamente:

1. endogeneidad;
2. heterogeneidad no observada;
3. no linealidad;
4. heterogeneidad por categoría;
5. heterogeneidad temporal;
6. dependencia intra-cluster;
7. fuerte papel de promociones/visibilidad.

Eso es mucho más interesante que un único número.

En otras palabras:

> **El proyecto ha encontrado que la elasticidad de delivery no es un parámetro escalar simple.**

Esa podría ser la historia central de la versión definitiva.

---

# 31. Cómo reestructuraría el argumento económico

En vez de:

> “La elasticidad es −0.7513.”

usar:

> “La relación entre precio final de checkout y pedidos es robustamente negativa, pero su magnitud depende de la especificación, del nivel de precio, del producto y del momento. La especificación log-log lineal produce −0.7513 como resumen promedio; el panel FE produce valores cercanos a −1.58, mientras la estrategia IV produce alrededor de −0.95 bajo supuestos de exclusión controvertidos.”

Eso es mucho más científico.

---

# 32. Qué estimación consideraría principal después de corregirlo

No escogería hoy ninguna de las tres como “la elasticidad verdadera”.

Haría una arquitectura de resultados:

### Modelo A — Referencia descriptiva

OLS:

\[
\ln Q=\beta\ln P+X+\epsilon
\]

con inferencia clusterizada.

### Modelo B — Principal observacional

Two-way FE:

\[
\ln Q_{ict}
=
f(\ln P_{ict})
+\alpha_{ic}
+\lambda_t
+\epsilon_{ict}
\]

con spline o interacción flexible y clustering adecuado.

### Modelo C — Identificación causal

IV/Control Function **solo si se defiende explícitamente el supuesto de exclusión**.

Si no puede defenderse:

> se presenta como análisis de sensibilidad, no como estimador causal definitivo.

Esta sería una postura de referee muy sólida.

---

# 33. Qué haría con el RCS

El RCS tiene potencial para convertirse en el centro de la versión final.

Pero hay que:

1. verificar la matriz de restricciones;
2. corregir la prueba de linealidad si corresponde;
3. usar una base numéricamente más estable;
4. reportar intervalos de confianza para la elasticidad puntual;
5. evitar extrapolar los extremos;
6. mostrar claramente el intervalo central de precios donde hay suficiente soporte.

La propia estimación reporta un condition number enorme:

\[
9.97\times10^{15}
\]

y eigenvalue mínimo cercano a \(10^{-26}\).

Eso no significa automáticamente que las curvas sean inútiles, pero exige mucha más atención numérica.

---

# 34. Una preocupación adicional con los extremos del spline

Un spline puede producir comportamientos extremos en percentiles cercanos a los límites.

Si el estimador pasa de:

\[
+0.58
\]

a:

\[
-3.22
\]

hay que demostrar:

- densidad suficiente;
- intervalos de confianza;
- estabilidad frente a número de knots;
- estabilidad frente a exclusión del 1% superior/inferior;
- estabilidad frente a transformaciones alternativas.

El propio proyecto ya encontró que excluir 1% de los extremos modifica la elasticidad aproximadamente 0.0523. Eso indica que los extremos sí importan.

---

# 35. Un punto fuerte importante: el proyecto sí ha hecho algo que muchos trabajos no hacen

Hay que destacarlo.

La bitácora documenta explícitamente que existió una versión anterior con resultados IV ficticios y que fue descartada.

Además, se corrigió en el informe un resultado JB que había sido fabricado en una iteración anterior.

Esto es excelente práctica científica.

No es un defecto.

Es exactamente el tipo de auditoría interna que hace el proyecto más confiable.

---

# 36. Reproducibilidad

Aquí la evaluación es alta.

El proyecto documenta:

- notebooks;
- `src/`;
- SQL;
- dashboard;
- LaTeX;
- CITATION;
- salidas consolidadas;
- pruebas;
- decisiones metodológicas.

También documenta que el informe recompiló limpiamente a 39 páginas y que el appendix técnico fue integrado.

Esto es una fortaleza real.

---

# 37. Pero “reproducible” debe significar algo más concreto antes de Zenodo

Antes de publicar:

### Debe garantizarse

- `environment.yml` reproducible;
- versiones exactas de Python y paquetes;
- seed documentadas;
- scripts de ejecución de cero a final;
- outputs regenerables;
- hashes o versionado de datos;
- comprobación de que las rutas no dependen del PC local;
- instrucciones de ejecución;
- licencia del dataset.

La página pública del dataset muestra información de licencia/proveniencia que debe comprobarse contra la fuente exacta utilizada antes de redistribuir datos o empaquetarlos en Zenodo.

---

# 38. Qué no debe subirse a Zenodo

No subiría automáticamente:

- raw dataset completo;
- copias no necesarias del dataset;
- archivos con licencia ambigua.

Publicaría:

- código;
- variables derivadas;
- documentación;
- outputs;
- scripts;
- esquema;
- metadatos;
- instrucciones para descargar el original.

Eso también refuerza reproducibilidad.

---

# 39. Correcciones obligatorias antes de cualquier publicación académica

## P0 — Bloqueantes

**1. Sustituir la inferencia OLS HC3 como inferencia principal.**  
Usar clustering coherente con la estructura de panel.

**2. Retirar la afirmación de que −0.9487 es un “estimado causal creíble” sin reservas.**

**3. Replantear el IV a partir del supuesto de exclusión.**

**4. Eliminar/reemplazar el placebo temporal actual.**

**5. Auditar matemáticamente el test RCS de no-linealidad y sus grados de libertad.**

**6. Dejar de presentar −0.7513 como elasticidad estructural constante después de haber rechazado la forma funcional.**

---

# 40. Correcciones P1

**7. Sustituir `log1p(num_orders)` por `log(num_orders)` para el modelo de elasticidad, dado que el mínimo observado es 13, o justificar formalmente la decisión.**

**8. Aclarar la ausencia de ceros.**

**9. Cambiar “S/” por la unidad monetaria documentada por la fuente.**

**10. Cambiar “precio del alimento” por “precio final de checkout” cuando corresponda.**

**11. Reestimar interacciones con precio centrado.**

**12. Corregir la interpretación de quantile regression.**

**13. Revisar la estructura de clustering del panel.**

**14. Revisar inferencia del control function.**

**15. Añadir referencias metodológicas faltantes.**

**16. Corregir las fórmulas de ingresos.**

**17. Separar asociaciones de efectos causales en promociones.**

**18. Reescribir la sección de limitaciones, porque actualmente todavía presenta elasticidad constante como una limitación futura pese a que el propio estudio ya la rechazó y estimó RCS.**

---

# 41. Qué mantendría exactamente

No eliminaría:

- análisis del merge;
- tratamiento de `base_price`;
- documentación de errores previos;
- panel FE;
- investigación IV;
- RCS;
- quantile regression;
- heterogeneidad por categoría;
- heterogeneidad temporal;
- SQL/BI;
- appendix técnico;
- auditoría bibliográfica.

La infraestructura del proyecto es buena.

Lo que necesita corrección es **la jerarquía inferencial de los resultados**, no rehacer todo desde cero.

---

# 42. Mi lectura final de cada estimación

### −0.7513 — OLS con controles

**Qué es:** asociación condicionada promedio.

**Problema:** inferencia HC3 demasiado optimista ante dependencia; además forma funcional rechazada.

**Uso recomendado:** benchmark descriptivo.

---

### −1.5755 — Panel FE two-way

**Qué es:** asociación within de precio y pedidos, después de eliminar heterogeneidad fija centro-comida y shocks comunes de semana.

**Ventaja:** mucho mejor control de heterogeneidad no observada.

**Problema:** no elimina simultaneidad; clustering necesita reforzarse.

**Uso recomendado:** **resultado observacional de referencia**, posiblemente el más importante después de corregir inferencia.

---

### −0.9487 — IV 2SLS

**Qué es:** estimación instrumental bajo un supuesto de exclusión fuerte.

**Ventaja:** aborda endogeneidad.

**Problema:** Hansen J rechaza; ambos instrumentos están muy correlacionados; exclusión no demostrada.

**Uso recomendado:** análisis de sensibilidad/identificación tentativa, **no “verdad causal”**.

---

### RCS

**Qué es:** evidencia de no linealidad.

**Fortaleza:** potencialmente la parte más novedosa.

**Problema:** revisar implementación numérica y test de linealidad.

**Uso recomendado:** candidato fuerte a resultado central.

---

# 43. Recomendación para la narrativa definitiva

La nueva tesis del paper podría ser:

> **La elasticidad precio-demanda en food delivery no es constante: la sensibilidad de los pedidos depende del nivel de precio, del producto y del momento. Una especificación lineal log-log resume la relación con una elasticidad de aproximadamente −0.75, mientras que la variación within de panel produce una sensibilidad cercana a −1.6 y la estrategia IV produce aproximadamente −0.95 bajo supuestos de exclusión que no pueden verificarse completamente.**

Esto es mucho más interesante que:

> “La elasticidad es −0.7513.”

---

# 44. Lo que un referee probablemente preguntaría en una segunda ronda

1. ¿Qué observaciones tienen cero pedidos y por qué no aparecen?
2. ¿Cuál es exactamente la unidad monetaria?
3. ¿Qué incluye exactamente `checkout_price`?
4. ¿La unidad de análisis representa clientes, pedidos o agregados?
5. ¿Por qué `log1p` si no hay ceros?
6. ¿Por qué HC3 en presencia de dependencia panel?
7. ¿Por qué cluster por `center_id-meal_id` y no `center_id`?
8. ¿Qué justifica la exclusión de `base_price` como instrumento?
9. ¿Puede `base_price` afectar demanda directamente mediante reference pricing?
10. ¿Puede el precio de otros centros capturar shocks regionales comunes?
11. ¿Por qué el Hansen J rechazado no cambia la interpretación IV?
12. ¿Cuál es exactamente la matriz de restricciones del RCS?
13. ¿Por qué el test de no-linealidad tiene 4 df?
14. ¿Por qué el placebo de semanas es un placebo válido si conserva \(P_i,Q_i\)?
15. ¿Qué significa un cuantil alto: consumidores o unidades centro-comida de alta demanda?
16. ¿La promoción es exógena?
17. ¿Por qué una sola categoría/center effect no es suficiente?
18. ¿Por qué usar AIC/BIC para decidir entre formas que luego son rechazadas por RESET/RCS?
19. ¿Qué parte del resultado es causal y qué parte es asociacional?
20. ¿Qué exactamente puede recomendarse a un gerente de pricing con estos datos?

El manuscrito actual no responde convincentemente a todas.

---

# 45. Calificación final de referee

## Veredicto: **Major Revision**

### Fortalezas decisivas

El trabajo tiene:

- muy buena disciplina de documentación;
- excelente trazabilidad;
- honestidad frente a errores;
- profundidad econométrica inusual para un proyecto académico;
- buena infraestructura de reproducibilidad;
- evidencia de heterogeneidad realmente interesante;
- una capa de negocio útil.

### Debilidades decisivas

Pero actualmente:

- la inferencia principal usa errores estándar que no reflejan bien la dependencia;
- el IV no tiene una exclusión suficientemente defendible;
- el Hansen J es tratado con demasiada ligereza;
- el placebo temporal está mal diseñado;
- el modelo principal es rechazado funcionalmente por las propias pruebas del proyecto;
- el RCS necesita auditoría matemática;
- `log1p` complica la interpretación estricta como elasticidad;
- algunas interpretaciones comerciales exceden lo que los datos identifican;
- hay varios detalles de procedencia, unidad monetaria y bibliografía que deben corregirse.

---

# 46. Conclusión del referee

**No recomendaría abandonar el proyecto.**

Tampoco recomendaría “parchar” dos o tres líneas y publicarlo tal como está.

La inversión ya realizada permite algo mejor:

> **hacer una v3 metodológicamente mucho más fuerte a partir de v2 sin rehacer el pipeline completo.**

El trabajo está cerca de convertirse en un análisis econométrico realmente serio, pero necesita una segunda reorganización en la que la identificación y la inferencia tengan prioridad sobre la acumulación de tests.

El cambio conceptual más importante sería:

**dejar de preguntar “¿cuál es la elasticidad?” y pasar a preguntar “¿qué componente de la variación de precios identifica qué elasticidad, bajo qué supuestos y para qué unidad de decisión?”**

Ese cambio resolvería gran parte de las tensiones que hoy existen entre OLS, FE, IV, RCS, cuantiles y las recomendaciones de negocio.

### Mi conclusión final, en una frase

**Proyecto técnicamente ambicioso y notablemente bien documentado, con evidencia empírica interesante y una infraestructura reproducible fuerte, pero todavía no listo para sostener una afirmación causal de elasticidad precio-demanda; requiere revisión mayor centrada en inferencia clusterizada, validez de IV, corrección del placebo, especificación no lineal y reformulación de las conclusiones de negocio.**