"""SUR-09: barrido fino de anisotropia (factor, angulo, k, exponente p) y, encima
del mejor, las 6 formaciones y los interpoladores con dip.

La anisotropia resulto ser LA palanca: estirar el eje paralelo a las
trayectorias antes del KDTree hace que los k vecinos vengan de VARIOS pozos a la
misma coordenada a lo largo del pozo (una seccion transversal) en vez de 16
puntos del mismo pozo vecino.
"""
import sys

import numpy as np

sys.path.insert(0, "/home/ftpx100/work/active/kaggle-rogii")
sys.path.insert(0, "/home/ftpx100/work/active/kaggle-rogii/research")

import sur02_lib as L          # noqa: E402
from sur06_sweep import Runner, surf_cloud  # noqa: E402

block = sys.argv[1] if len(sys.argv) > 1 else "fino"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 60
R = Runner(K)

if block == "fino":
    for a in (5, 8, 12, 16, 24, 40, 64):
        c = L.Cloud(subsample=10, aniso=a)
        for k in (8, 16, 24, 32):
            R.run(f"IDW s10 k{k} a={a}",
                  surf_cloud(c, lambda cc, d, gi, X, Y, k=k: L.idw(cc, d, gi, k)))
        del c

elif block == "p":
    for a in (8, 16, 24):
        c = L.Cloud(subsample=10, aniso=a)
        for k in (16, 24):
            for p in (1.0, 1.5, 2.0, 3.0, 4.0):
                R.run(f"IDW s10 k{k} a={a} p={p}",
                      surf_cloud(c, lambda cc, d, gi, X, Y, k=k, p=p: L.idw(cc, d, gi, k, p)))
        del c

elif block == "theta":
    base = L.THETA
    for dth in (-30, -20, -10, -5, 0, 5, 10, 20, 30):
        c = L.Cloud(subsample=10, aniso=16, theta=base + np.radians(dth))
        R.run(f"IDW s10 k16 a=16 theta{dth:+d}",
              surf_cloud(c, lambda cc, d, gi, X, Y: L.idw(cc, d, gi, 16)))
        del c

elif block == "sub":
    for sub in (10, 5, 3):
        c = L.Cloud(subsample=sub, aniso=16)
        for k in (16, 24, 32):
            R.run(f"IDW s{sub} k{k} a=16",
                  surf_cloud(c, lambda cc, d, gi, X, Y, k=k: L.idw(cc, d, gi, k)))
        del c

elif block == "form":
    c = L.Cloud(subsample=10, aniso=16)
    it = lambda cc, d, gi, X, Y: L.idw(cc, d, gi, 16)   # noqa: E731
    for mode in ("buda", "mean", "median", "best", "winv", "winv_sd", "softmax"):
        R.run(f"IDW a16 k16 {mode}", surf_cloud(c, it, mode=mode))
    for tn in (2, 3, 4):
        R.run(f"IDW a16 k16 top{tn}", surf_cloud(c, it, mode="topn", topn=tn))
    for sw in (500, 1000, 4000):
        R.run(f"IDW a16 k16 winv sw{sw}", surf_cloud(c, it, mode="winv", score_win=sw))

elif block == "dip":
    c = L.Cloud(subsample=10, aniso=16)
    for k, lam in [(16, 1e-2), (24, 1e-2), (24, 1e-1), (24, 1.0), (24, 10.0),
                   (32, 1e-1), (32, 1.0), (32, 10.0)]:
        R.run(f"ridge a16 k{k} lam{lam:g}",
              surf_cloud(c, lambda cc, d, gi, X, Y, k=k, lam=lam:
                         L.linridge(cc, d, gi, X, Y, k, lam)))
    for k, rg in [(16, 600.0), (24, 600.0), (24, 1500.0), (24, 4000.0)]:
        R.run(f"krige a16 k{k} r{rg:g}",
              surf_cloud(c, lambda cc, d, gi, X, Y, k=k, rg=rg:
                         L.krige(cc, d, gi, X, Y, k, 1.0, 1000.0, rg)))

elif block == "rbf":
    c = L.Cloud(subsample=10, aniso=16)
    for kern, sm, deg in [("linear", 0.0, None), ("linear", 10.0, None),
                          ("thin_plate_spline", 1.0, 1), ("thin_plate_spline", 100.0, 1),
                          ("multiquadric", 10.0, 1)]:
        R.run(f"rbf a16 k16 {kern} sm{sm:g} deg{deg}",
              surf_cloud(c, lambda cc, d, gi, X, Y, kern=kern, sm=sm, deg=deg:
                         L.rbf_local(cc, d, gi, X, Y, 16, kern, sm, deg)))

R.save(f"aniso2_{block}_k{K}")
