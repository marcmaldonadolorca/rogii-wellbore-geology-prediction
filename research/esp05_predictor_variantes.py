"""ESP-05: variantes del predictor espacial sobre los mismos 150 held-out (seed 42).

Variantes (todas calibradas con mediana pre-PS del offset):
  V1 IDW k=8 BUDA (referencia esp02)
  V2 plano local: ajuste lineal Z=f(X,Y) con los k=32 vecinos ponderado IDW -> menos sesgo con dip
  V3 promedio de superficies ANCC y BUDA (cada una con su offset propio calibrado)
Guarda per-pozo: research/esp05_perwell.csv (well_id, nn_dist, rmse_v1, rmse_v2, rmse_v3, n_post)
"""
import numpy as np
from scipy.spatial import cKDTree

C = np.load("/home/ftpx100/work/active/kaggle-rogii/research/cache_wells.npz", allow_pickle=True)
data = C["data"]; well_idx = C["well_idx"]; cols = list(C["cols"]); ids = C["ids"]
iX, iY, iZ, iANCC, iBUDA, iTVT, iTVTin = [cols.index(c) for c in ["X","Y","Z","ANCC","BUDA","TVT","TVT_input"]]
n = well_idx.max() + 1

rng = np.random.default_rng(42)
held = np.sort(rng.choice(n, size=150, replace=False))

sub_mask = np.zeros(len(well_idx), dtype=bool)
for i in range(n):
    idx = np.where(well_idx == i)[0]
    sub_mask[idx[::10]] = True
sub_w = well_idx[sub_mask]
sub_xy = data[sub_mask][:, [iX, iY]]
sub_b = data[sub_mask][:, iBUDA]
sub_a = data[sub_mask][:, iANCC]
fin_a = np.isfinite(sub_a)  # ANCC tiene NaN en 7 pozos (0.9%)

def idw_est(tree, vals, q, k=8):
    d, j = tree.query(q, k=k, workers=4)
    w = 1.0 / (d ** 2 + 1e-6)
    return (w * vals[j]).sum(axis=1) / w.sum(axis=1), d[:, 0]

def plane_est(tree, xy, vals, q, k=32):
    d, j = tree.query(q, k=k, workers=4)
    out = np.empty(len(q))
    for r in range(len(q)):
        P = xy[j[r]]; v = vals[j[r]]
        w = 1.0 / (d[r] ** 2 + 1e-6)
        A = np.c_[P[:, 0] - q[r, 0], P[:, 1] - q[r, 1], np.ones(k)]
        Aw = A * w[:, None]
        try:
            coef, *_ = np.linalg.lstsq(Aw.T @ A, Aw.T @ v, rcond=None)
            out[r] = coef[2]
        except np.linalg.LinAlgError:
            out[r] = (w * v).sum() / w.sum()
    return out

rows = []
err1_all, err2_all, err3_all = [], [], []
for i in held:
    keep = sub_w != i
    xyk = sub_xy[keep]
    tree = cKDTree(xyk)
    bk = sub_b[keep]
    keep_a = keep & fin_a
    xyk_a = sub_xy[keep_a]
    tree_a = cKDTree(xyk_a)
    ak = sub_a[keep_a]
    m = well_idx == i
    q = data[m][:, [iX, iY]]
    z = data[m, iZ]; tvt = data[m, iTVT]; tin = data[m, iTVTin]
    pre = np.isfinite(tin); post = ~pre

    eb, dnn = idw_est(tree, bk, q, 8)
    ea, _ = idw_est(tree_a, ak, q, 8)
    pb = plane_est(tree, xyk, bk, q, 32)

    pre_idx = np.where(pre)[0]
    last = pre_idx[-500:]  # calib ult-500 (mejor en esp02: 24.8 vs 28.6)

    def pred(est):
        a0 = np.median(tin[last] + z[last] - est[last])
        return a0 + est[post] - z[post]

    p1 = pred(eb)
    p2 = pred(pb)
    p3 = 0.5 * (pred(eb) + pred(ea))
    t = tvt[post]
    e1, e2, e3 = p1 - t, p2 - t, p3 - t
    err1_all.append(e1); err2_all.append(e2); err3_all.append(e3)
    rows.append((ids[i], float(np.median(dnn)), float(np.sqrt(np.mean(e1**2))),
                 float(np.sqrt(np.mean(e2**2))), float(np.sqrt(np.mean(e3**2))), int(post.sum())))

import csv
with open("/home/ftpx100/work/active/kaggle-rogii/research/esp05_perwell.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["well_id", "nn_med", "rmse_idw_buda", "rmse_plane_buda", "rmse_mean_ab", "n_post"])
    w.writerows(rows)

for name, es in [("V1 IDW k=8 BUDA", err1_all), ("V2 plano k=32 BUDA", err2_all), ("V3 media ANCC+BUDA", err3_all)]:
    e = np.concatenate(es)
    pw = np.array([np.sqrt(np.mean(x**2)) for x in es])
    print(f"{name}: RMSE global {np.sqrt(np.mean(e**2)):.3f} ft | por-pozo mediana {np.median(pw):.3f}, p90 {np.percentile(pw,90):.3f}, max {pw.max():.3f}")
