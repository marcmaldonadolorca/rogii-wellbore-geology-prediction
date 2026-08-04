"""v6 pesos — variante POR PUNTO extendida + validacion estratificada por nn.

softmax lineal sobre scores de S y G (P = referencia):
  score_m = c0 + c . z   con z = [log1p(nn), md/1e3, pstd, log_btPS, cstd, m_dSG]
(3 primeras por punto, 3 ultimas por pozo). Ajuste Powell por SSE pooled en los
380 pozos train; evaluacion en eval60/eval150 con P16 y P64 (sstd para pstd).

Tambien: estratificacion por quintil de nn_med del ganador vs base fija, y
ajuste alternativo SOLO (nn, md) re-ponderando el train hacia pozos aislados.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v6_pesos_rig import W_GLOBAL, load_set as _load_base, pooled  # noqa: E402


def load_set(which, p64=False):
    """load_set del rig + pstd por punto (sstd S=64 si p64)."""
    z = np.load(HERE / f"v4_gbm_{which}.npz")
    X = pd.DataFrame(z["X"], columns=[str(c) for c in z["cols"]])
    w = z["well"]
    ids = [str(i) for i in z["ids"]]
    ms = None
    if p64:
        tag = {"eval60": "k60", "eval150": "k150"}[which]
        ms = np.load(HERE / f"v4_multiseed_{tag}.npz")
    wells = _load_base(which, p64=p64)
    by_wid = {}
    for j in np.unique(w):
        m = w == j
        by_wid[ids[j]] = (ms[f"{ids[j]}_sstd"].astype(np.float64) if ms is not None
                          else X.pstd.values[m].astype(np.float64))
    for wl in wells:
        wl["pstd_pt"] = by_wid[wl["wid"]]
    return wells


def zfeat(wl):
    n = len(wl["y"])
    f = wl["feats"]
    return np.column_stack([
        np.log1p(wl["nn_pt"]), wl["md_pt"] / 1e3, wl["pstd_pt"],
        np.full(n, f["log_btPS"]), np.full(n, f["cstd"]), np.full(n, f["m_dSG"]),
    ])


def make_weights(mu, sd):
    def weights(theta, Z):
        z = (Z - mu) / sd
        k = Z.shape[1]
        sS = theta[0] + z @ theta[1:1 + k]
        sG = theta[1 + k] + z @ theta[2 + k:2 + 2 * k]
        mx = np.maximum.reduce([sS, np.zeros(len(z)), sG])
        e = np.exp(np.column_stack([sS, np.zeros(len(z)), sG]) - mx[:, None])
        return e / e.sum(1, keepdims=True)
    return weights


def fit(train, cols, wtrain=None, label=""):
    Ztr = np.concatenate([zfeat(wl)[:, cols] for wl in train])
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
    weights = make_weights(mu, sd)
    wl_w = np.ones(len(train)) if wtrain is None else wtrain

    def loss(theta):
        sse = n = 0.0
        for wl, ww in zip(train, wl_w):
            W = weights(theta, zfeat(wl)[:, cols])
            p = (wl["A"] * W).sum(1)
            sse += ww * float(((wl["y"] - p) ** 2).sum())
            n += ww * len(wl["y"])
        return np.sqrt(sse / n)

    k = len(cols)
    x0 = np.zeros(2 * (k + 1))
    x0[0], x0[k + 1] = np.log(0.45 / 0.55), -4.0
    res = minimize(loss, x0, method="Powell",
                   options=dict(maxiter=8000, xtol=1e-3, ftol=1e-4))
    print(f"fit {label}: train rmse {res.fun:.3f} theta={np.round(res.x, 3)}", flush=True)
    return res.x, mu, sd, weights


def strat(wells, w_of, label):
    nn = np.array([wl["nn_med"] for wl in wells])
    qs = np.quantile(nn, [0.2, 0.4, 0.6, 0.8])
    grp = np.digitize(nn, qs)
    vals = []
    for g in range(5):
        sub = [wl for wl, gg in zip(wells, grp) if gg == g]
        vals.append(pooled(sub, w_of))
    print(f"  {label:26s} quintiles nn: " + "  ".join(f"{v:6.2f}" for v in vals))
    return vals


def main():
    train = load_set("train")
    evals = {t: load_set(w, p64=(p == "P64"))
             for w, t, p in (("eval60", "k60_P64", "P64"), ("eval150", "k150_P64", "P64"),
                             ("eval60", "k60_P16", "P16"), ("eval150", "k150_P16", "P16"))}
    rows = []

    def ev(name, w_of_by_set):
        for tag, wells in evals.items():
            r = pooled(wells, w_of_by_set(tag))
            rows.append({"variante": name, "set": tag, "rmse": r})
            print(f"  [{tag}] {name} -> {r:8.3f}", flush=True)

    variants = {
        "punto2 nn,md": [0, 1],
        "punto3 nn,md,pstd": [0, 1, 2],
        "punto6 full": [0, 1, 2, 3, 4, 5],
    }
    fits = {}
    for name, cols in variants.items():
        th, mu, sd, wfun = fit(train, cols, label=name)
        fits[name] = dict(theta=th.tolist(), mu=mu.tolist(), sd=sd.tolist(), cols=cols)
        ev(name, lambda tag, c=cols, t=th, w=wfun: (
            lambda wl: w(t, zfeat(wl)[:, c])))

    # ajuste re-ponderado hacia pozos aislados (simula test mas aislado):
    # peso del pozo = 1 + (rank(nn)/n)^2 * 3
    nn = np.array([wl["nn_med"] for wl in train])
    rk = np.argsort(np.argsort(nn)) / (len(nn) - 1)
    wtr = 1.0 + 3.0 * rk ** 2
    th_i, mu_i, sd_i, wfun_i = fit(train, [0, 1, 2], wtrain=wtr, label="punto3-isoW")
    fits["punto3-isoW"] = dict(theta=th_i.tolist(), mu=mu_i.tolist(),
                               sd=sd_i.tolist(), cols=[0, 1, 2])
    ev("punto3-isoW", lambda tag: (lambda wl: wfun_i(th_i, zfeat(wl)[:, [0, 1, 2]])))

    # estratificacion por nn (validacion del regimen aislado)
    for tag in ("k150_P64", "k60_P64"):
        wells = evals[tag]
        print(f"\nESTRATIFICACION {tag} (quintil 5 = mas aislado):")
        strat(wells, lambda wl: W_GLOBAL, "base fija")
        for name in ("punto2 nn,md", "punto3 nn,md,pstd", "punto6 full", "punto3-isoW"):
            c = fits[name]["cols"]
            th = np.array(fits[name]["theta"])
            wfun = make_weights(np.array(fits[name]["mu"]), np.array(fits[name]["sd"]))
            strat(wells, lambda wl, c=c, th=th, w=wfun: w(th, zfeat(wl)[:, c]), name)

    pd.DataFrame(rows).to_csv(HERE / "v6_pesos_punto2.csv", index=False)
    (HERE / "v6_pesos_punto2_params.json").write_text(json.dumps(fits, indent=1))
    print("\nguardado v6_pesos_punto2.csv / _params.json")


if __name__ == "__main__":
    main()
