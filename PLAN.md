# ROGII Wellbore — plan de solución

Objetivo: el score más bajo posible (RMSE de dTVT en pies). Deadline 2026-08-05.

## La formulación correcta del problema

Medido en 120 pozos de train, las 6 columnas de formación cumplen, dentro de cada pozo:

```
TVT_i = S_k(X_i, Y_i) − Z_i + C_well,k     residuo std = 0.0065 ft  (= el redondeo de los datos)
```

en el 100% de los pozos y para las 6 formaciones (ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA).

Consecuencias que reordenan todo:

1. **TVT no es una serie temporal a extrapolar: es la altura de una superficie
   geológica evaluada en la trayectoria.** Predecir TVT ≡ reconstruir la
   superficie `S(X,Y)` del campo y restarle la Z conocida del pozo.
2. Las 6 formaciones son **la misma geometría desplazada** dentro de un pozo
   (todas dan idéntico residuo). Entre pozos los espesores sí varían → las 6
   superficies son 6 medidas redundantes de la estructura, útiles para promediar
   error de interpolación, no 6 informaciones distintas.
3. **El error de este predictor NO deriva con la distancia a PS** — que es
   exactamente el modo de fallo de nuestro baseline geométrico (11 ft de RMSE por
   cada 1000 ft extrapolados, techo duro ~35 ft).
4. Solo hay dos incógnitas: la superficie `S` (global, aprendible de los 773
   pozos de train) y `C_well` (una constante por pozo, calibrable con el prefijo
   pre-PS que SÍ conocemos en test).
5. El GR pasa a ser el **sensor de corrección**: informa dónde `S` interpolada se
   desvía de la realidad local. No es el motor.

Esto explica por qué la aglomeración pública se estanca en 6.4: tratan el problema
como tracking de GR (particle filters sobre el nivel estratigráfico) y meten lo
espacial como una feature más de un GBDT, con **1 punto resumen por pozo** o 60
puntos submuestreados. La estructura geométrica exacta está infraexplotada.

## Los bloques, por palanca

### 1. La superficie del campo — la palanca grande
Los 773 pozos de train aportan **todos sus puntos** `(X, Y, Z_formación)`: millones
de observaciones de `S`, no 773 ni 46k (lo que usan los públicos). Reconstruir `S`
densa y precisa:
- Interpolador que capture **dip local**: plane-fit local ponderado / RBF /
  kriging con variograma real (a diferencia del IDW puro de los notebooks, que
  colapsa a media local y no modela pendiente).
- Las 6 superficies a la vez → consenso y estimación de incertidumbre local.
- **LOWO estricto**: al validar un pozo, excluir todos sus propios puntos. Sin
  esto la CV miente (y es la explicación candidata de nuestro gap 39.4 local vs
  46.6 LB).
- Salida obligatoria: mapa de incertidumbre (distancia a vecinos, dispersión
  entre las 6 superficies) → alimenta el bloque 3.

### 2. Calibración con el prefijo pre-PS
El prefijo da `TVT_input` real, así que revela el error de `S` en ese tramo:
- `C_well` = mediana robusta de `(TVT_input + Z − S_interp)` en el prefijo.
- No solo un offset: ajustar también **gradiente/tendencia** del residuo contra
  distancia horizontal — corrige el error sistemático local de la superficie.
- Variantes por segmento (early/mid/late del prefijo) para detectar si el residuo
  deriva; elegir por backtest (bloque 5).

### 3. El GR como corrector — MEDIDO 2026-07-30, corrige el diseño inicial

Resultados de la primera implementación (`model.py`, `research/diag_*.py`):

- **Superficie sola: 25,5 ft** (k=40) — confirma los 24,7 del research.
- **HMM con emisión gaussiana punto a punto: −0,8 ft solamente** (24,7), y solo
  con difusión casi rígida (σ_r=0,01); con más libertad **empeora**.
- Causa raíz medida: la emisión discrimina (`ll(r_real) > ll(0)` en el **71,3%**
  de ventanas) pero es débil — **sensibilidad 0,93 API/ft frente a 7,66 API de
  ruido ⇒ 1σ ≈ 8,3 ft de TVT**. Con esa SNR aparecen alias con verosimilitud
  mayor que el valor verdadero, y el MAP falla a cualquier span (13,5 vs 12,9 ft
  de no hacer nada).
- El typewell cubre el 100% del rango; el prefijo propio solo el 50%. Cambiar de
  mapa de referencia no arregla nada (tw 25,6 / self 27,3 / ambos 25,2).

**Conclusión:** la verosimilitud gaussiana sobre valores absolutos de GR no es la
vía. Hay que pasar a **correlación de forma** (NCC multiescala / DTW / beam),
invariante a escala y offset local — que es justamente lo que usa toda la familia
pública. El HMM sigue siendo válido como integrador, pero con emisión NCC en vez
de gaussiana.

### 3-bis. El GR como corrector bayesiano — diseño original
HMM exacto 1D sobre TVT discretizado (0.2–0.5 ft), resuelto con **forward–backward**:
- **Transición**: prior derivado de la superficie interpolada, `dS/dMD` a lo largo
  de la trayectoria. Aquí está la diferencia con todo lo público: sus particle
  filters usan random-walk con momentum ciego; nuestro modelo de proceso lleva
  dentro la geología del campo.
- **Emisión**: `N(GR_hw ; GR_typewell(TVT), σ)`, con calibración afín del GR del
  heel (`a·GR_tw + b` por mínimos cuadrados en el prefijo) y σ estimada de los
  residuos del prefijo. La señal existe: corr mediana +0,68 con 0% de pozos negativos.
- **Estimador = media posterior**, óptima para RMSE. Todos los PF públicos filtran
  hacia delante; el smoothing hacia atrás recoloca la trayectoria con evidencia
  posterior. Ventaja estructural con métrica offline.
- Es fusión probabilística en un solo modelo (superficie = prior, GR =
  verosimilitud), no un blend heurístico de dos predicciones.
- No modelar fallas: medido, no existen (0 saltos >20 ft en 200 pozos; TVT suave
  en el 100%). Se ahorra la mezcla bimodal de Winkler.

### 4. GBM residual
Sobre las señales de 1–3 (superficie por formación, media y **varianza** posterior
del HMM, incertidumbre de interpolación, geometría de trayectoria, tortuosidad
Q-3D): target `ΔTVT`, GroupKFold por pozo. Captura lo que el modelo generativo no ve.

### 5. Selección y calibración por pozo (leak-free)
Enmascarar la cola del prefijo conocido (cortes 0,5/0,65/0,75), backtestear cada
candidato (superficie sola, HMM, GBM, blends, variantes de `C_well`) en ese
holdout local y elegir/ponderar **por pozo**, con movimiento acotado. Es la
validación honesta cuando el test es oculto, y es probablemente el salto 7,3 → 6,5
de la familia pública.

### 6. Remate
- Proyección robusta IRLS del nivel estratigráfico + Savitzky-Golay por pozo.
- **Contact override**: el test oculto contiene copias de pozos de train;
  reconstruir TVT desde sus columnas de formación e imponerlo solo si reproduce el
  prefijo visible con RMSE < 1 ft, **interpolando por MD** (no por índice de fila:
  las copias ocultas no están alineadas). Guard ⇒ nunca empeora.
- Kernel endurecido: numba en lo caro, presupuesto de runtime para ~770 pozos
  ocultos, submission failsafe escrita antes de calcular, auditoría de ids.

## Qué no se hace y por qué

- **Nada de datasets de artifacts ajenos** (ravaghi/fleongg/pilkwang): pickles no
  auditables, borrables por su dueño en plena re-ejecución, y nos clavan en la
  aglomeración 6.4 sin edge propio.
- **Nada de LB probing** (hay quien suma +0,522 ft a un pozo concreto calibrando
  contra el público): overfit garantizado y quema envíos.
- DTW puro, detección de fallas, polinomios de grado ≥2 al extrapolar (medido:
  427–869 ft, catastrófico), y los 14 beam searches (retorno marginal sobre el
  resto del stack).

## Criterio de decisión

Ningún componente entra sin mejorar la **CV local LOWO sobre los 770 pozos**. El LB
público solo calibra el mapeo local→oculto. Los dos envíos finales: el mejor por CV
local (apuesta anti-shakeup, y la aglomeración está afinada al público) y el mejor
por LB público (cobertura).

## Sobre el premio

La bolsa son 50.000 USD y se reparte entre los primeros puestos; el líder actual
está en 4.679 con 5.959 equipos compitiendo. El plan maximiza score; el puesto
final no está bajo nuestro control. La ruta de valor que sí controlamos es score
propio + medalla, con método reproducible y sin dependencias ajenas.
