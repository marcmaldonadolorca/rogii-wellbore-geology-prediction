"""SUR-01: cache de la nube (X,Y,6 formaciones) de los 773 pozos de train.

Guarda a resolucion COMPLETA (sin submuestreo) para poder barrer 1/1, 1/3, 1/10
desde el mismo fichero. X,Y en float64 (coordenadas ~3e6 ft), superficies float32.
Ademas guarda la direccion principal (PCA) de cada trayectoria para el estudio
de anisotropia.
"""
import glob
import os
import time

import numpy as np
import pandas as pd

RAW = "/home/ftpx100/work/active/kaggle-rogii/data/raw/train"
OUT = "/home/ftpx100/work/active/kaggle-rogii/research/sur01_cloud.npz"
FORM = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]

files = sorted(glob.glob(os.path.join(RAW, "*__horizontal_well.csv")))
print(f"pozos: {len(files)}", flush=True)

t0 = time.time()
XY, S, WI, ids, dirs = [], [], [], [], []
for i, f in enumerate(files):
    wid = os.path.basename(f).split("__")[0]
    d = pd.read_csv(f, usecols=["X", "Y"] + FORM)
    XY.append(d[["X", "Y"]].to_numpy(np.float64))
    S.append(d[FORM].to_numpy(np.float32))
    WI.append(np.full(len(d), i, np.int32))
    ids.append(wid)
    # direccion principal de la trayectoria (PCA 2D sobre X,Y)
    p = d[["X", "Y"]].to_numpy(np.float64)
    p = p - p.mean(0)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    dirs.append(vt[0])
    if i % 100 == 0:
        print(f"  {i} ({time.time()-t0:.0f}s)", flush=True)

xy = np.vstack(XY)
s = np.vstack(S)
wi = np.concatenate(WI)
np.savez(OUT, xy=xy, s=s, wi=wi, ids=np.array(ids), form=np.array(FORM),
         dirs=np.array(dirs))
print(f"filas={len(xy)}  nan por formacion: "
      f"{dict(zip(FORM, np.isnan(s).sum(0).tolist()))}")
print(f"pozos con NaN por formacion: "
      f"{dict(zip(FORM, [len(np.unique(wi[np.isnan(s[:,j])])) for j in range(6)]))}")
print(f"-> {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB, {time.time()-t0:.0f}s)")

# direccion principal global de las trayectorias (PCA sobre las direcciones,
# con signo canonico para que no se cancelen)
D = np.array(dirs)
D = D * np.sign(D[:, 0:1] + 1e-12)
M = D.T @ D
w, v = np.linalg.eigh(M)
print(f"direccion principal global: {v[:,-1]}  (angulo {np.degrees(np.arctan2(v[1,-1], v[0,-1])):.1f} deg)")
print(f"autovalores {w}  -> concentracion {w[-1]/w.sum():.3f}")
