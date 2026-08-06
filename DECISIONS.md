# DECISIONS — kaggle-rogii

Registro de decisiones y de los **negativos medidos**, que en este proyecto han
sido la mitad del valor: cada avenida cerrada con números evitó meter ruido en el
envío que cuenta.

## ROG-001 · Formular el problema como interpolación de superficie (2026-07-30)

**Medido:** `TVT = S_k(X,Y) − Z + C_well,k` con residuo std de 0,0065 ft en el
100% de los pozos y las 6 formaciones. **Decisión:** abandonar el enfoque de
serie temporal (extrapolar el dip) y tratarlo como reconstrucción de una
superficie geológica + offset por pozo. Es el cimiento de todo lo demás.

## ROG-002 · No usar datasets de artifacts ajenos (2026-07-30)

La aglomeración 6,4-6,7 del leaderboard son forks de un pipeline compartido que
carga modelos preentrenados de terceros (`ravaghi`, `fleongg`, `pilkwang`).
**Decisión:** cero dependencias externas. Motivos: son *pickles* no auditables,
el dueño puede borrarlos en plena re-ejecución, y adoptarlos nos clava en esa
aglomeración sin ventaja propia. Coste asumido: renunciar a ~1,5 ft "gratis".

## ROG-003 · Cero LB probing; selección por CV local (2026-07-30)

Existe probing documentado en notebooks públicos (sumar constantes a pozos
concretos calibrando contra el público). **Decisión:** los envíos solo evalúan
candidatos genuinos; la selección se hace con el CV local de 770 pozos. El
leaderboard público solo se usa para calibrar el factor local→LB.

## ROG-004 · Métrica anisótropa para la superficie (2026-07-31)

Los pozos son líneas casi paralelas; con métrica isótropa los *k* vecinos salen
del mismo pozo contiguo en vez de dar sección transversal. **Medido:** rotar a la
dirección principal (`theta = 2.278489`, PCA+11°) y estirar el eje paralelo
(`aniso=16`) baja la superficie de 24,2 → 12,9 ft. Es la aportación original con
más impacto y no aparece en ningún notebook público.

## ROG-005 · Media simple entre semillas del PF, no softmax (2026-07-31)

La receta pública pondera las semillas con `softmax(loglik/scale)`, scales
{3,5,8,12}. **Medido:** con pocas semillas eso colapsa a *argmax* y empeora
(14,03 vs 13,07 de la media simple). **Decisión:** media simple. Nota: con S=64
el loglik sí discrimina (corr −0,34), pero la media sigue siendo competitiva y es
más robusta.

## ROG-006 · Fusión por varianza inversa dependiente del contexto (2026-08-03)

En vez de un peso fijo entre superficie y PF, `w = σ_p²/(σ_s²+σ_p²)` calculado
**por punto** con σ ajustadas en 200 pozos disjuntos. **Medido:** 9,978 → 9,145
(k=150), y la mejora se concentra en el **tercil más aislado** (−1,67), que es el
régimen del test oculto. Confirmado en LB: factor local→LB pasó a 1,00.
Extensión 2D (`σ_s(nn, md_since)`): 9,145 → 8,949.

## ROG-007 · El prefijo pre-PS no predice el post-PS (tres negativos)

Probado por tres vías independientes, todas negativas:

- **Offset por pozo** (2026-08-03): corr(offset del backtest, offset real) =
  −0,05 con n=368. El oráculo dice que un solo offset por pozo vale 4,9 ft, pero
  no es capturable. LightGBM captura ~3% de la varianza y no transfiere.
- **Pesos por pozo** (2026-08-04): señal débil real (spearman +0,17, acierto de
  signo 0,64) pero mejora máxima −0,12 ft, bajo el umbral de ruido pareado y con
  IC95 que cruza cero.
- **Selección dura por backtest**: no consolida entre subconjuntos.

**Conclusión:** lo que distingue a un pozo después del PS no está escrito en su
prefijo. Es el resultado negativo más importante del proyecto.

## ROG-008 · El gamma-ray rastrea pero no localiza (2026-07-30)

**Medido:** el typewell está perfectamente alineado (corr +0,814 en el prefijo),
pero post-PS el pozo va horizontal dentro de una capa: sensibilidad 0,93 API/ft
frente a 7,66 API de ruido ⇒ **1σ equivale a 8,3 ft de TVT**. Por eso fracasan
todos los métodos de búsqueda global (MAP por ventanas 13,5 ft; NCC 29-35 ft) y
funciona el filtrado secuencial (particle filter, 13,8). Explica por qué el
enfoque bayesiano "correcto" que se diseñó al principio (HMM con emisión
gaussiana) rendía tan poco.

## ROG-009 · Criterio anti-ruido: consistencia entre subconjuntos (2026-08-04)

Umbrales de ruido pareado medidos: 0,56 ft (k=60), 0,35 (k=150), 0,15 (770).
**Decisión:** una mejora solo entra si es consistente en k=60 **y** k=150. Aplicado
el último día, descartó dos candidatos que parecían ganadores:

- **Covarianza tipo Markowitz** con 5 señales: −0,21 en k=150 pero **+0,28 en
  k=60**. Las 9 variantes mejoraban en k=150 (parecía señal), pero el flip de
  signo lo delata.
- **Mezcla de v7 y v8**: −0,03 en k=150, +0,06 en k=60.

## ROG-010 · Envíos finales (2026-08-04, corregida el mismo día)

**Recomendación final: v7 principal + v8 cobertura.**

La recomendación inicial era la contraria (v8 principal), basada en que v8 ganaba
en k=60 y k=150. La medición sobre los **770 pozos** la invirtió:

| | 770 pozos | proxy LB | LB real |
|---|---|---|---|
| **v7** | **9,151** | **9,794** | **9,072** |
| v8 | 9,792 | 10,628 | 9,079 |

v7 gana por 0,64 ft (umbral de ruido a esa escala: 0,15) y las tres evidencias
independientes coinciden. Se mantienen los dos como envíos finales porque son
estructuralmente distintos —uno corrige con árboles, el otro pondera por
varianza—: si el privado castiga una filosofía, la otra cubre.

## ROG-011 · Un subconjunto de validación puede mentir aunque sea determinista (2026-08-04)

El harness usaba subconjuntos deterministas (k=60, k=150) para poder comparar
variantes de forma pareada, lo cual es correcto y necesario. Pero **la elección
entre dos modelos finales se hizo mal con ellos**: v8 ganaba en ambos y perdió
por 0,64 ft en los 770. Los 150 pozos, aun estratificados por `n_pred`, no
representaban la distribución completa.

**Regla:** las decisiones intermedias pueden tomarse en subconjuntos (por coste),
pero **la elección del envío final se mide sobre el conjunto completo**, aunque
cueste 3 horas de cómputo. El coste de no hacerlo era mandar el peor de los dos
modelos al leaderboard privado.

## ROG-012 · Más datos de entrenamiento no mejoran el GBM (2026-08-04)

Último intento del día 4: reentrenar el árbitro con **574 pozos en vez de 380**
(+51%) y validarlo con OOF sobre los 770 (`research/v9e_gbm770.py`), para tener
un número directamente comparable y evitar el sesgo de subconjunto de ROG-011.

**Medido:** GBM OOF = 9,917 ft, peor que v7 (9,151). El diagnóstico está en los
folds: `iters = 1, 2, 3, 4, 20` — el early stopping corta casi al instante. El
GBM ya había saturado con 380 pozos; es un corrector ligero (mejora su base solo
−0,13 ft), no un modelo limitado por datos. Añadir muestras no aporta.

**Cierre:** agotada la última palanca identificada. v7 queda como envío principal.

## Descartes menores con número

Multi-formación (explota a k=150: 44,0), kriging ordinario y RBF (12,76 y 14,76
vs 12,68 del IDW), `C_well` con tendencia amortiguada (24 combinaciones, todas
peores), nube más densa (subsample 3: peor y 20× más cara), HMM con velocidad en
el estado (25,1 vs 24,4), polinomio de grado 2 al extrapolar (427-869 ft),
detección de fallas (no existen: 0 saltos >20 ft en 200 pozos), σ_p recalibrada
con S=64 (8,949 → 8,945, irrelevante), blend con geométrico por varianza inversa
(−0,03, marginal).

## ROG-013 · Vía pública autorizada para optar a medalla (2026-08-05)

ROG-002 descartaba usar los *datasets de artifacts* de terceros por criterio de
calidad. El último día el propietario lo revirtió explícitamente («haz también la
A») al constatar que la medalla de bronce está detrás de esa puerta: los ~900
equipos con medalla son forks del mismo pipeline público, y usar notebooks y
datasets públicos es legal y estándar en Kaggle.

Enviado un fork de `raunakdey07/rogii-ultra-sub-6-rmse` con sus 7 datasets de
artifacts, en paralelo a nuestra solución propia. **No es trabajo nuestro** y así
debe declararse en cualquier uso de portfolio: opta a la medalla del leaderboard
público, mientras el v11 cubre el escenario de shakeup —que es justo donde estos
pipelines, afinados contra el público, son más frágiles—.

## ROG-014 · Recalibrar lo heredado: el mejor hallazgo del proyecto (2026-08-05)

Los hiperparámetros del particle filter venían del notebook público de Roman,
ajustados contra *otro* leaderboard. Nadie del pelotón los ha tocado. Barridos
contra nuestro CV (`research/v9f_pfparams.py`, `v9g_pffino.py`):

| clip de σ_GR | RMSE k=150 | Δ |
|---|---|---|
| [10,60] (heredado) | 9,181 | — |
| [15,90] | 9,008 | −0,173 |
| **[20,120]** | **8,810** | **−0,370** |
| [20,120] + N=1500 | 8,731 | −0,450 |

Subir el clip hace que el PF confíe menos en el gamma-ray, coherente con ROG-008
(1σ de ruido ≈ 8,3 ft de TVT). **Transfirió al leaderboard: 9,072 → 8,893.**

Lección: cuando se adopta una pieza de un tercero, sus constantes son hipótesis
sobre *sus* datos, no verdades. Recalibrarlas contra la validación propia fue más
rentable que cualquier añadido arquitectónico de los últimos tres días.

## ROG-015 · El shakeup confirmó la tesis (2026-08-05, cierre)

Resultado final: **puesto 1.082 de 6.191 (top 17%)**, +2.299 puestos respecto al
público. Degradación público→privado por envío:

| envío | público | privado | Δ |
|---|---|---|---|
| vía pública (artifacts) | 6,470 | 9,471 | **+3,00** |
| v11 (nuestro) | 8,893 | 9,422 | +0,53 |
| **v8 (nuestro, el más simple)** | 9,079 | **9,167** | **+0,09** |
| v7 (nuestro, + GBM) | 9,072 | 9,346 | +0,27 |
| v6 | 10,032 | 9,666 | −0,37 |

Tres conclusiones que valen para la próxima competición:

1. **Lo prestado se evapora.** El pipeline de artifacts, afinado contra el
   leaderboard visible, perdió 3 puntos enteros y acabó por debajo de *todos*
   nuestros modelos. La decisión original de ROG-002 era correcta para el privado.
2. **La simplicidad predice estabilidad.** El orden de degradación siguió el orden
   de complejidad: v8 (blend puro) +0,09 < v7 (+GBM) +0,27 < v11 (+GBM+σ) +0,53.
   Cada capa de maquinaria compró score público y lo pagó en el privado.
3. **Seleccionar por score público penaliza al modelo honesto.** v8 era nuestro
   mejor privado y no entró en los dos finales porque era el peor en el escaparate.
   Para la próxima: si el CV local es fiable (aquí lo era), merece la pena forzar
   la selección manual del mejor por CV en vez de aceptar el default de Kaggle.
