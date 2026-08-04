"""ESP-02: interpolabilidad de superficies (BUDA/ANCC) + predictor espacial puro.

Para 150 pozos held-out (seed fija):
  A) interpola Z_form en sus (X,Y) con KNN-IDW (cKDTree) usando SOLO los demas pozos
     -> RMSE superficie interpolada vs real (todos los puntos y solo post-PS)
  B) predictor: a = mediana(TVT_input + Z - surf_interp) sobre pre-PS;
     TVT_pred = a + (surf_interp - Z) post-PS -> RMSE vs TVT real (metrica competicion)
Variantes: k=1/8/16, potencia 2; calibracion con todos los pre-PS vs ultimos 500.
"""
import numpy as np
from scipy.spatial import cKDTree

C = np.load("/home/ftpx100/work/active/kaggle-rogii/research/cache_wells.npz", allow_pickle=True)
data = C["data"]; well_idx = C["well_idx"]; cols = list(C["cols"])
iMD, iX, iY, iZ, iANCC, iBUDA, iTVT, iTVTin = [cols.index(c) for c in ["MD","X","Y","Z","ANCC","BUDA","TVT","TVT_input"]]
n = well_idx.max() + 1

rng = np.random.default_rng(42)
held = np.sort(rng.choice(n, size=150, replace=False))
held_set = set(held.tolist())

# puntos para el arbol: subsampleo cada 10
sub_mask = np.zeros(len(well_idx), dtype=bool)
for i in range(n):
    idx = np.where(well_idx == i)[0]
    sub_mask[idx[::10]] = True
sub_idx_well = well_idx[sub_mask]
sub_xy = data[sub_mask][:, [iX, iY]]
sub_buda = data[sub_mask][:, iBUDA]
sub_ancc = data[sub_mask][:, iANCC]
print(f"puntos arbol: {sub_mask.sum()}")

def idw(tree, vals, q, k):
    d, j = tree.query(q, k=k, workers=4)
    if k == 1:
        return vals[j], d
    w = 1.0 / (d ** 2 + 1e-6)
    return (w * vals[j]).sum(axis=1) / w.sum(axis=1), d[:, 0]

results = {}
for form_name, sub_vals, icol in [("BUDA", sub_buda, iBUDA), ("ANCC", sub_ancc, iANCC)]:
    for k in [1, 8, 16]:
        surf_err_all, surf_err_post = [], []
        pred_err_post_all, pred_err_post_last500 = [], []
        per_well_rmse = []
        nn_dist_all = []
        for i in held:
            keep = sub_idx_well != i
            tree = cKDTree(sub_xy[keep])
            vals = sub_vals[keep]
            m = well_idx == i
            q = data[m][:, [iX, iY]]
            z = data[m, iZ]; zf = data[m, icol]
            tvt = data[m, iTVT]; tin = data[m, iTVTin]
            est, dnn = idw(tree, vals, q, k)
            err = est - zf
            pre = np.isfinite(tin); post = ~pre
            surf_err_all.append(err)
            surf_err_post.append(err[post])
            nn_dist_all.append(dnn)
            # predictor
            a_all = np.median(tin[pre] + z[pre] - est[pre])
            pre_idx = np.where(pre)[0]
            last = pre_idx[-500:]
            a_last = np.median(tin[last] + z[last] - est[last])
            tp_all = a_all + est[post] - z[post]
            tp_last = a_last + est[post] - z[post]
            pred_err_post_all.append(tp_all - tvt[post])
            pred_err_post_last500.append(tp_last - tvt[post])
            per_well_rmse.append(np.sqrt(np.mean((tp_all - tvt[post]) ** 2)))
        sa = np.concatenate(surf_err_all); sp = np.concatenate(surf_err_post)
        pa = np.concatenate(pred_err_post_all); pl = np.concatenate(pred_err_post_last500)
        pw = np.array(per_well_rmse)
        nn = np.concatenate(nn_dist_all)
        key = (form_name, k)
        results[key] = dict(surf_all=np.sqrt(np.mean(sa**2)), surf_post=np.sqrt(np.mean(sp**2)),
                            pred_all=np.sqrt(np.mean(pa**2)), pred_last=np.sqrt(np.mean(pl**2)),
                            pw_med=np.median(pw), pw_p90=np.percentile(pw, 90), pw_max=pw.max())
        print(f"[{form_name} k={k}] RMSE superficie: all {results[key]['surf_all']:.2f} ft, post-PS {results[key]['surf_post']:.2f} ft | "
              f"RMSE predictor post-PS: calib-todos {results[key]['pred_all']:.2f} ft, calib-ult500 {results[key]['pred_last']:.2f} ft | "
              f"por-pozo mediana {results[key]['pw_med']:.2f}, p90 {results[key]['pw_p90']:.2f}, max {results[key]['pw_max']:.2f}")
        if k == 8:
            # error vs distancia al vecino
            for lo, hi in [(0, 100), (100, 300), (300, 1000), (1000, 1e9)]:
                mm = (nn >= lo) & (nn < hi)
                if mm.sum() > 0:
                    print(f"    d_nn [{lo},{hi if hi<1e9 else 'inf'}): {mm.sum()} pts, RMSE surf {np.sqrt(np.mean(sa[mm]**2)):.2f} ft")
