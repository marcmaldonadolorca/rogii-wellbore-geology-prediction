# PROJECT_STATE — kaggle-rogii

## Control actual

- **Estado:** competición en su último día (cierra **2026-08-05 23:59 UTC**).
- **Mejor resultado: 9,072 público** (envío v7); v8 empata en 9,079 y gana en CV.
- **Los dos candidatos finales YA están enviados y son seleccionables.** Pase lo
  que pase con el resto, el resultado está asegurado.
- **Próxima acción (del propietario, el 5-ago):** marcar en Kaggle los 2 envíos
  que cuentan para el privado. Recomendación: **v7 principal** + **v8 cobertura**
  (ver ROG-010; la recomendación se invirtió al medir sobre los 770 pozos).

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
