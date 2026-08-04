"""ESP-03: relacion TVT vs columnas de formacion (Z - Z_formacion).

Por pozo ajusta TVT = a + b*(Z - Z_form) para BUDA y ANCC.
Mide: distribucion de b (¿~ -1?), variabilidad de a intra-pozo y entre pozos,
RMSE residual del ajuste lineal por pozo.
"""
import numpy as np

C = np.load("/home/ftpx100/work/active/kaggle-rogii/research/cache_wells.npz", allow_pickle=True)
data = C["data"]; well_idx = C["well_idx"]; cols = list(C["cols"])
iMD, iX, iY, iZ, iANCC, iBUDA, iTVT, iTVTin = [cols.index(c) for c in ["MD","X","Y","Z","ANCC","BUDA","TVT","TVT_input"]]
n = well_idx.max() + 1

for form, icol in [("BUDA", iBUDA), ("ANCC", iANCC)]:
    bs, as_, res_rmse, a_std_intra = [], [], [], []
    exact = []  # std por pozo de (TVT + Z - Z_form) si b=-1 exacto
    for i in range(n):
        m = well_idx == i
        z = data[m, iZ]; zf = data[m, icol]; tvt = data[m, iTVT]
        ok = np.isfinite(z) & np.isfinite(zf) & np.isfinite(tvt)
        if ok.sum() < 100:
            continue
        x = z[ok] - zf[ok]; y = tvt[ok]
        b, a = np.polyfit(x, y, 1)
        pred = a + b * x
        bs.append(b); as_.append(a)
        res_rmse.append(np.sqrt(np.mean((y - pred) ** 2)))
        aa = y + x  # a implicito si b=-1
        exact.append(aa.std())
    bs = np.array(bs); as_ = np.array(as_); res_rmse = np.array(res_rmse); exact = np.array(exact)
    print(f"--- {form} ---")
    print(f"b: mediana {np.median(bs):.4f}, p10 {np.percentile(bs,10):.4f}, p90 {np.percentile(bs,90):.4f}")
    print(f"RMSE residual ajuste lineal por pozo: mediana {np.median(res_rmse):.3f} ft, p90 {np.percentile(res_rmse,90):.3f}")
    print(f"std intra-pozo de (TVT + Z - Z_{form}) [b=-1 forzado]: mediana {np.median(exact):.3f} ft, p90 {np.percentile(exact,90):.3f}, max {exact.max():.3f}")
    print(f"a entre pozos: media {as_.mean():.1f}, std {as_.std():.2f}, rango [{as_.min():.1f}, {as_.max():.1f}]")

# ¿PS donde cae? fraccion del pozo con TVT_input valido
fr = []
npts_post = []
for i in range(n):
    m = well_idx == i
    tin = data[m, iTVTin]
    k = np.isfinite(tin)
    fr.append(k.mean())
    npts_post.append((~k).sum())
fr = np.array(fr); npts_post = np.array(npts_post)
print("--- PS ---")
print(f"fraccion pre-PS: mediana {np.median(fr):.3f}, p10 {np.percentile(fr,10):.3f}, p90 {np.percentile(fr,90):.3f}")
print(f"puntos post-PS por pozo: mediana {np.median(npts_post):.0f}, p90 {np.percentile(npts_post,90):.0f}")
