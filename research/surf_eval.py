"""Motor de barrido: evalua MUCHAS configuraciones de superficie en una sola
pasada por los pozos (el coste dominante es construir el arbol local).

Reproduce EXACTAMENTE la metrica de cv.evaluate (misma seleccion de pozos, mismo
cut, mismo pooled RMSE y mismo rmse_lb_proxy). Se valida contra cv.evaluate en
surf01_calibra.py antes de usarse para decidir nada.

Config = dict:
    name    etiqueta
    sub     submuestreo de la nube (1, 3, 10, ...)
    aniso   factor de escala del eje Y tras rotar theta (1.0 = isotropo)
    theta   rotacion en radianes antes de escalar
    detrend interpola el residuo de la tendencia global cuadratica
    method  'idw' | 'plane' | 'krige' | 'rbf'
    kw      kwargs del metodo
    k       vecinos
    frms    lista de indices de formacion (0..5); 5 = BUDA
    comb    'mean' | 'median' | 'wrmse' | 'best' | 'inv2'  (irrelevante si len(frms)==1)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv as cvmod  # noqa: E402
from surf_lib import METHODS, Cloud, LocalField  # noqa: E402

RAW = Path(__file__).resolve().parent.parent / "data/raw"
CAL_TAIL = 500
LB_PROXY_NPRED = cvmod.LB_PROXY_NPRED


def geom_key(c):
    return (c.get("sub", 10), c.get("aniso", 1.0), c.get("theta", 0.0),
            bool(c.get("detrend", False)), c.get("margin", None))


def combine(preds, errs, mode):
    """preds (F, n) post-PS ; errs (F,) rmse en el prefijo -> (n,)"""
    if len(preds) == 1:
        return preds[0]
    P = np.asarray(preds)
    if mode == "mean":
        return P.mean(0)
    if mode == "median":
        return np.median(P, 0)
    if mode == "best":
        return P[int(np.argmin(errs))]
    e = np.maximum(np.asarray(errs, float), 1e-3)
    if mode == "wrmse":
        w = 1.0 / e
    elif mode == "inv2":
        w = 1.0 / e**2
    elif mode == "inv4":
        w = 1.0 / e**4
    elif mode == "soft":
        w = np.exp(-(e / max(e.min(), 1e-3)))
    else:
        raise ValueError(mode)
    w = w / w.sum()
    return (w[:, None] * P).sum(0)


def eval_configs(configs, k_wells=150, wells=None, cloud=None, verbose=True,
                 nn_bins=False, per_well=False):
    cloud = cloud or Cloud()
    if wells is None:
        wells = cvmod.well_ids()
        if k_wells is not None:
            wells = cvmod._select(wells, k_wells, cvmod.SEED)

    groups = {}
    for i, c in enumerate(configs):
        groups.setdefault(geom_key(c), []).append(i)

    sse = np.zeros(len(configs))
    npt = 0
    sse_px = np.zeros(len(configs))
    npt_px = 0
    pw = [[] for _ in configs]
    bins = np.array([0, 100, 300, 600, 1000, np.inf])
    bsse = np.zeros((len(configs), 5))
    bn = np.zeros(5)

    for wi, wid in enumerate(wells):
        df = pd.read_csv(RAW / "train" / f"{wid}__horizontal_well.csv",
                         usecols=["X", "Y", "Z", "TVT", "TVT_input"])
        m = df.TVT_input.isna()
        cut = int(m.idxmax()) if m.any() else len(df)
        if cut < 20 or cut >= len(df):
            continue
        X = df.X.values.astype(float); Y = df.Y.values.astype(float)
        Z = df.Z.values.astype(float)
        ti = df.TVT_input.values.astype(float)
        truth = df.TVT.values[cut:].astype(float)
        n_pred = len(df) - cut
        lo = max(0, cut - CAL_TAIL)
        npt += n_pred
        if n_pred >= LB_PROXY_NPRED:
            npt_px += n_pred

        nnd = None
        for gk, idxs in groups.items():
            sub, aniso, theta, dt, marg = gk
            kwlf = {} if marg is None else {"margin": marg}
            lf = LocalField(cloud, wid, X, Y, subsample=sub, aniso=aniso,
                            theta=theta, detrend=dt, **kwlf)
            kmax = max(configs[i]["k"] for i in idxs)
            frms = sorted({j for i in idxs for j in configs[i]["frms"]})
            Q, D, I = {}, {}, {}
            for j in frms:
                Q[j], D[j], I[j] = lf.query(X, Y, j, kmax)
            lf.assert_margin()
            if nnd is None and aniso == 1.0 and theta == 0.0:
                nnd = D[frms[0]][:, 0].copy()
            cache = {}
            for i in idxs:
                c = configs[i]
                fn = METHODS[c["method"]]
                k = c["k"]
                preds, errs = [], []
                for j in c["frms"]:
                    ck = (c["method"], k, j, tuple(sorted(c.get("kw", {}).items())))
                    if ck not in cache:
                        s = fn(lf, Q[j], D[j][:, :k], I[j][:, :k], j, **c.get("kw", {}))
                        s = lf.finish(s, X, Y, j)
                        cache[ck] = s
                    s = cache[ck]
                    C = np.median(ti[lo:cut] + Z[lo:cut] - s[lo:cut])
                    p = s - Z + C
                    errs.append(float(np.sqrt(np.mean((p[lo:cut] - ti[lo:cut]) ** 2))))
                    preds.append(p[cut:])
                pred = combine(preds, errs, c.get("comb", "mean"))
                d2 = (truth - pred) ** 2
                s_ = float(d2.sum())
                sse[i] += s_
                if n_pred >= LB_PROXY_NPRED:
                    sse_px[i] += s_
                if per_well:
                    pw[i].append((wid, n_pred, float(np.sqrt(s_ / n_pred))))
                if nn_bins and nnd is not None:
                    bi = np.digitize(nnd[cut:], bins) - 1
                    for b in range(5):
                        mm = bi == b
                        if mm.any():
                            bsse[i, b] += float(d2[mm].sum())
            del lf, Q, D, I, cache
        if nn_bins and nnd is not None:
            bi = np.digitize(nnd[cut:], bins) - 1
            bn += np.bincount(bi, minlength=5)
        if verbose and (wi + 1) % 25 == 0:
            print(f"  ...{wi+1}/{len(wells)} pozos", flush=True)

    out = pd.DataFrame({
        "name": [c["name"] for c in configs],
        "rmse": np.sqrt(sse / npt),
        "rmse_lb_proxy": np.sqrt(sse_px / max(npt_px, 1)),
    })
    res = {"table": out.sort_values("rmse").reset_index(drop=True),
           "n_wells": len(wells), "n_points": npt}
    if nn_bins:
        lab = ["<100", "100-300", "300-600", "600-1000", ">1000"]
        res["bins"] = pd.DataFrame(np.sqrt(bsse / np.maximum(bn, 1)),
                                   index=[c["name"] for c in configs], columns=lab)
        res["bins_n"] = pd.Series(bn.astype(int), index=lab)
    if per_well:
        res["per_well"] = {c["name"]: pd.DataFrame(pw[i], columns=["well", "n_pred", "rmse"])
                           for i, c in enumerate(configs)}
    return res


def show(res, title=""):
    if title:
        print(f"\n=== {title} ===")
    t = res["table"]
    print(f"pozos={res['n_wells']} puntos={res['n_points']}")
    for _, r in t.iterrows():
        print(f"  {r['name']:<42s} rmse={r['rmse']:7.3f}  lb_proxy={r['rmse_lb_proxy']:7.3f}")
    if "bins" in res:
        print("\nRMSE por distancia al vecino mas cercano (ft):")
        print(res["bins"].round(2).to_string())
        print("n por bin:", res["bins_n"].to_dict())
