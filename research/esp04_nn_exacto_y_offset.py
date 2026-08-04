"""ESP-04: NN trayectoria exacto (arbol por-pozo excluyente) + suavidad espacial del offset a_pozo.

a_pozo = mediana(TVT + Z - Z_BUDA) por pozo. ¿Es a_pozo suave en XY (interpolable
de vecinos) o arbitrario por pozo? Si fuera suave, no haria falta calibrar pre-PS.
"""
import numpy as np
from scipy.spatial import cKDTree

C = np.load("/home/ftpx100/work/active/kaggle-rogii/research/cache_wells.npz", allow_pickle=True)
data = C["data"]; well_idx = C["well_idx"]; cols = list(C["cols"])
iX, iY, iZ, iBUDA, iTVT = [cols.index(c) for c in ["X", "Y", "Z", "BUDA", "TVT"]]
n = well_idx.max() + 1
X, Y = data[:, iX], data[:, iY]

# NN trayectoria exacto: subsampleo cada 20, arbol global, query con k grande adaptativo
sub = slice(None, None, 20)
pts = []; pidx = []
for i in range(n):
    m = np.where(well_idx == i)[0][sub]
    pts.append(np.c_[X[m], Y[m]])
    pidx.append(np.full(len(m), i, dtype=np.int32))
pts = np.vstack(pts); pidx = np.concatenate(pidx)
tree = cKDTree(pts)
traj_nn = np.zeros(n)
for i in range(n):
    mine = pts[pidx == i]
    k = 40
    dmin = np.inf
    while not np.isfinite(dmin) and k <= 2560:
        dd, jj = tree.query(mine, k=k, workers=4)
        other = pidx[jj] != i
        dmin = np.where(other, dd, np.inf).min()
        k *= 2
    traj_nn[i] = dmin
print(f"NN trayectoria EXACTO: mediana {np.median(traj_nn):.0f} ft, p10 {np.percentile(traj_nn,10):.0f}, "
      f"p90 {np.percentile(traj_nn,90):.0f}, p99 {np.percentile(traj_nn,99):.0f}, max {traj_nn.max():.0f}")
for thr in [100, 300, 500, 1000, 2000, 5000]:
    print(f"  pozos con NN > {thr} ft: {(traj_nn > thr).sum()}")

# offset a_pozo: ¿suave en el espacio?
cx = np.zeros(n); cy = np.zeros(n); a = np.zeros(n)
for i in range(n):
    m = well_idx == i
    cx[i] = X[m].mean(); cy[i] = Y[m].mean()
    a[i] = np.median(data[m, iTVT] + data[m, iZ] - data[m, iBUDA])
ctree = cKDTree(np.c_[cx, cy])
d, j = ctree.query(np.c_[cx, cy], k=6)
# diferencia de a con el vecino mas cercano y media de 5 vecinos
d1 = np.abs(a[j[:, 1]] - a)
dk = np.abs(a[j[:, 1:]].mean(axis=1) - a)
print(f"a_pozo (TVT+Z-Z_BUDA): std global {a.std():.1f} ft")
print(f"|a - a_NN1|: mediana {np.median(d1):.2f} ft, p90 {np.percentile(d1,90):.2f}, max {d1.max():.2f}")
print(f"|a - media(5 NN)|: mediana {np.median(dk):.2f} ft, p90 {np.percentile(dk,90):.2f}")
np.savez("/home/ftpx100/work/active/kaggle-rogii/research/esp04_nn.npz", traj_nn=traj_nn, a=a, cx=cx, cy=cy)
