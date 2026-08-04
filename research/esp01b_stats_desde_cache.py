"""ESP-01b: estadisticas espaciales desde cache_wells.npz (sin releer CSVs)."""
import numpy as np
from scipy.spatial import cKDTree

C = np.load("/home/ftpx100/work/active/kaggle-rogii/research/cache_wells.npz", allow_pickle=True)
data = C["data"]; well_idx = C["well_idx"]; cols = list(C["cols"]); ids = C["ids"]
iX, iY = cols.index("X"), cols.index("Y")
X, Y = data[:, iX], data[:, iY]
n = well_idx.max() + 1
print(f"pozos: {n}, filas: {len(well_idx)}")
print(f"bbox X: [{X.min():.0f}, {X.max():.0f}]  span {X.max()-X.min():.0f} ft")
print(f"bbox Y: [{Y.min():.0f}, {Y.max():.0f}]  span {Y.max()-Y.min():.0f} ft")

cx = np.zeros(n); cy = np.zeros(n); lat_len = np.zeros(n)
for i in range(n):
    m = well_idx == i
    cx[i] = X[m].mean(); cy[i] = Y[m].mean()
    lat_len[i] = np.hypot(X[m].max()-X[m].min(), Y[m].max()-Y[m].min())
print(f"longitud lateral XY por pozo: mediana {np.median(lat_len):.0f} ft, p10 {np.percentile(lat_len,10):.0f}, p90 {np.percentile(lat_len,90):.0f}")

tree = cKDTree(np.c_[cx, cy])
d, _ = tree.query(np.c_[cx, cy], k=2)
nn = d[:, 1]
print(f"NN centroide-centroide: mediana {np.median(nn):.0f} ft, p10 {np.percentile(nn,10):.0f}, p90 {np.percentile(nn,90):.0f}, max {nn.max():.0f}")

sub = slice(None, None, 20)
pts = []; pidx = []
for i in range(n):
    m = np.where(well_idx == i)[0][sub]
    pts.append(np.c_[X[m], Y[m]])
    pidx.append(np.full(len(m), i, dtype=np.int32))
pts = np.vstack(pts); pidx = np.concatenate(pidx)
tree2 = cKDTree(pts)
traj_nn = np.full(n, np.inf)
for i in range(n):
    mine = pts[pidx == i]
    dd, jj = tree2.query(mine, k=40, workers=4)
    other = pidx[jj] != i
    traj_nn[i] = np.where(other, dd, np.inf).min()
finite = traj_nn[np.isfinite(traj_nn)]
print(f"NN trayectoria-trayectoria (aprox k=40, sub20): mediana {np.median(finite):.0f} ft, p10 {np.percentile(finite,10):.0f}, p90 {np.percentile(finite,90):.0f}, max {finite.max():.0f}, n_inf {np.sum(~np.isfinite(traj_nn))}")

from sklearn.cluster import DBSCAN
eps = 3 * np.median(nn)
lab = DBSCAN(eps=eps, min_samples=3).fit_predict(np.c_[cx, cy])
uniq, cnt = np.unique(lab, return_counts=True)
noise = cnt[uniq == -1].sum() if (-1 in uniq) else 0
print(f"DBSCAN eps={eps:.0f}: {np.sum(uniq>=0)} clusters, ruido {noise}")
order = np.argsort(-cnt)
for u, c in list(zip(uniq[order], cnt[order]))[:8]:
    print(f"  cluster {u}: {c} pozos")
