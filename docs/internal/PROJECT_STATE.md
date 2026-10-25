# PROJECT_STATE — kaggle-rogii

## Control actual — CERRADO 2026-08-05

**Resultado final: puesto 1.082 de 6.191 (top 17%)**, subiendo 2.299 puestos en
el shakeup del leaderboard privado (veníamos del 3.381 en el público).

| envío | público | **privado** | degradación |
|---|---|---|---|
| vía pública (artifacts de terceros) | 6,470 | **9,471** | **+3,00** |
| v11 (nuestro, σ_GR recalibrado) | 8,893 | **9,422** | +0,53 |
| v8 (nuestro, blend adaptativo 2D) | 9,079 | **9,167** | **+0,09** |
| v7 (nuestro, + GBM) | 9,072 | 9,346 | +0,27 |
| v6 | 10,032 | 9,666 | −0,37 |

**La tesis del proyecto quedó demostrada con datos.** El pipeline público con
artifacts —el que roza la medalla en el escaparate— se desplomó 3 puntos y acabó
**por debajo de todos nuestros modelos**. Nuestra solución propia lo batió en el
privado, que es donde se reparte el premio.

El modelo más estable fue el **más simple**: v8 (blend adaptativo puro, sin GBM)
degradó solo +0,09. Los que llevaban más maquinaria degradaron más.

**Espina:** v8 era nuestro mejor privado pero no entró en los dos seleccionados,
porque Kaggle elige por score público y ahí era el peor de los nuestros. El
modelo más honesto era el que peor puntuaba en el escaparate.

**Nada pendiente.** Siguiente concurso: Pokémon TCG Strategy (gate del propietario:
*Join* antes del 6-sep; writeup 13-sep). Después, ARC Prize con intake tras la defensa.

## Progresión medida

| envío | descripción | local k=150 | LB público |
|---|---|---|---|
| v1 | geométrico (dip anclado en PS) | 39,4 | 46,615 |
| v2 | superficie isótropa + HMM | 17,7 | 19,829 |
| v3 | blend superficie isótropa + PF | 12,4 | 11,717 |
| v4 | anisotropía (theta PCA, aniso 16) | 11,24 | 11,468 |
| v5 | PF multi-semilla S=64 | 10,20 | 11,364 |
| v6 | + GBM árbitro residual + capas | 9,65 | 10,032 |
| v7 | blend adaptativo 1D + GBM refit | 9,03 | **9,072** |
| **v8** | **blend adaptativo σ 2D, sin GBM** | **8,95** | **9,079** |

Veredicto de máxima potencia (**770 pozos**, umbral de ruido 0,15):

| | 770 pozos | proxy LB | LB real |
|---|---|---|---|
| **v7 (GBM sobre base adapt)** | **9,151** | **9,794** | **9,072** |
| v8 (blend adaptativo 2D) | 9,792 | 10,628 | 9,079 |

**v7 gana por 0,64 ft**, cuatro veces el umbral. Esto **invierte** la lectura de
los subconjuntos (v8 ganaba en k=60 y k=150): esos 150 pozos no eran
representativos del conjunto. Las tres evidencias independientes —770 pozos,
proxy y leaderboard real— coinciden ahora en v7. Lección registrada en ROG-011.

Factor local→LB: **1,00** desde el v7 (el modelo adaptativo transfiere sin
pérdida). Al principio era 1,18 y bajó al reducir el peso de la parte espacial:
los pozos del test oculto están más aislados de train que en nuestro LOWO.

## Contexto de la clasificación

~5.959 equipos. Bronce = 6,438 (top 10%); mediana = 8,356; líder = 4,679. Con
9,07 estamos en torno al puesto ~3.100 (top 53%). La aglomeración 6,4-6,7
(≈900 equipos) son forks de un mismo pipeline compartido vía *datasets de
artifacts* de terceros, con **LB probing documentado** (hay quien suma +0,522 ft
a un pozo concreto calibrando contra el público). Nuestra apuesta está en el
leaderboard **privado**: stack propio, CV honesto de 770 pozos, cero probing.

## Ficheros clave

- `cv.py` — harness de validación (formato test exacto, LOWO estricto, proxy LB).
- `model.py` — `SurfaceField` (nube + métrica anisótropa + LOWO), `_prior`, HMM.
- `research/pf_publico.py` — particle filters portados del notebook público
  standalone de Roman (única pieza de origen ajeno, reimplementada, sin artifacts).
- `research/ban_lib.py` — PF multi-semilla con numba.
- `research/v7_wadapt.py`, `research/v8_sig2d_wadapt.py` — curvas σ y blend adaptativo.
- `kernel/rogii_v8.py` — **el envío**: autocontenido, sin dependencias externas.
- `PLAN.md` — el plan de solución y su evolución.
- `DECISIONS.md` — decisiones y, sobre todo, los negativos medidos.
