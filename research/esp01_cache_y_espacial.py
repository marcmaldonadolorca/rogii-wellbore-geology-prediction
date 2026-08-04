"""ESP-01: cachea columnas clave de los 773 pozos y mide distribucion espacial.

Salidas:
  research/cache_wells.npz  (arrays float64 concatenados + well_idx int32 + lista ids)
  stdout: bounding box, NN distancias entre pozos, clusters.
"""
import glob
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

RAW = "/home/ftpx100/work/active/kaggle-rogii/data/raw/train"
OUT = "/home/ftpx100/work/active/kaggle-rogii/research/cache_wells.npz"

files = sorted(glob.glob(os.path.join(RAW, "*__horizontal_well.csv")))
print(f"pozos: {len(files)}")

cols = ["MD", "X", "Y", "Z", "ANCC", "BUDA", "TVT", "TVT_input"]
chunks = []
ids = []
for i, f in enumerate(files):
    wid = os.path.basename(f).split("__")[0]
    df = pd.read_csv(f, usecols=cols)
    arr = df[cols].to_numpy(dtype=np.float64)
    idx = np.full(len(df), i, dtype=np.int32)
    chunks.append((idx, arr))
    ids.append(wid)

well_idx = np.concatenate([c[0] for c in chunks])
data = np.vstack([c[1] for c in chunks])
np.savez_compressed(OUT, well_idx=well_idx, data=data, ids=np.array(ids), cols=np.array(cols))
print(f"cache: {data.shape[0]} filas -> {OUT}")

X, Y = data[:, 1], data[:, 2]
print(f"bbox X: [{X.min():.0f}, {X.max():.0f}]  span {X.max()-X.min():.0f} ft")
print(f"bbox Y: [{Y.min():.0f}, {Y.max():.0f}]  span {Y.max()-Y.min():.0f} ft")

# centroide y extremos por pozo
n = len(ids)
cx = np.zeros(n); cy = np.zeros(n)
lat_len = np.zeros(n)
for i in range(n):
    m = well_idx == i
    cx[i] = X[m].mean(); cy[i] = Y[m].mean()
    lat_len[i] = np.hypot(X[m].max()-X[m].min(), Y[m].max()-Y[m].min())
print(f"longitud lateral XY por pozo: mediana {np.median(lat_len):.0f} ft, p10 {np.percentile(lat_len,10):.0f}, p90 {np.percentile(lat_len,90):.0f}")

tree = cKDTree(np.c_[cx, cy])
d, _ = tree.query(np.c_[cx, cy], k=2)
nn = d[:, 1]
print(f"NN centroide-centroide: mediana {np.median(nn):.0f} ft, p10 {np.percentile(nn,10):.0f}, p90 {np.percentile(nn,90):.0f}, max {nn.max():.0f}")

# NN trayectoria-trayectoria (subsampleo cada 20 puntos)
sub = slice(None, None, 20)
pts = []
pidx = []
for i in range(n):
    m = np.where(well_idx == i)[0][sub]
    pts.append(np.c_[X[m], Y[m]])
    pidx.append(np.full(len(m), i, dtype=np.int32))
pts = np.vstack(pts); pidx = np.concatenate(pidx)
tree2 = cKDTree(pts)
traj_nn = np.full(n, np.inf)
for i in range(n):
    mine = pts[pidx == i]
    # k=40 vecinos y filtra los propios
    dd, jj = tree2.query(mine, k=40)
    other = pidx[jj] != i
    dmin = np.where(other, dd, np.inf).min()
    traj_nn[i] = dmin
finite = traj_nn[np.isfinite(traj_nn)]
print(f"NN trayectoria-trayectoria (aprox, k=40): mediana {np.median(finite):.0f} ft, p10 {np.percentile(finite,10):.0f}, p90 {np.percentile(finite,90):.0f}, n_inf {np.sum(~np.isfinite(traj_nn))}")

# clusters: DBSCAN sobre centroides con eps = 3 * mediana NN
from sklearn.cluster import DBSCAN
eps = 3 * np.median(nn)
lab = DBSCAN(eps=eps, min_samples=3).fit_predict(np.c_[cx, cy])
uniq, cnt = np.unique(lab, return_counts=True)
print(f"DBSCAN eps={eps:.0f}: {np.sum(uniq>=0)} clusters, ruido {cnt[uniq==-1].sum() if (-1 in uniq) else 0}")
order = np.argsort(-cnt)
for u, c in list(zip(uniq[order], cnt[order]))[:8]:
    print(f"  cluster {u}: {c} pozos")
