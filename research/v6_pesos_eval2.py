"""v6-pesos etapa 3: refinamientos de la ganadora D (por punto).

  D2 = D + termino por punto log(sstd) (spread entre semillas del PF: la señal
       de confianza medida, corr(spread,|err|)=+0.45) en ambos logits.
  E2 = D/D2 con sesgo por pozo de C (ridge features -> pesos oraculo): los
       logits por punto + [log(w_C) - log(w_global)].

Reporta ademas rmse_lb_proxy (pozos n_pred>=5147 dentro de k150) y los estratos
nn de la ganadora final. Guarda v6_pesos_params_final.json.
"""
import json

import numpy as np
import pandas as pd

from v6_pesos_lib import HERE, PesosCache, W_GLOBAL, subsets
import v6_pesos_eval as E

LB_NPRED = 5147
EPS = 1e-6


class D2Data(E.D):
    def __init__(self):
        super().__init__()
        self.sstd = {w: self.c.arr(w, "sstd") for w in self.c.meta.well}


def d2_weights(params, md, nn, sstd):
    a0, a1, a2, a3, b0, b1, b2, b3 = params
    mdk = md / 1000.0
    lnn = np.log(np.maximum(nn, 1.0))
    lst = np.log(np.maximum(sstd, 0.1))
    u = np.column_stack([a0 + a1 * mdk + a2 * lnn + a3 * lst,
                         np.zeros(len(md)),
                         b0 + b1 * mdk + b2 * lnn + b3 * lst])
    return E.softmax_w3(u)


def fit_d2(d, wells, x0):
    from scipy.optimize import minimize
    Y = np.concatenate([d.sig[w][0] for w in wells])
    S = np.vstack([d.sig[w][1] for w in wells])
    MD = np.concatenate([d.sig[w][2] for w in wells])
    NN = np.concatenate([d.sig[w][3] for w in wells])
    ST = np.concatenate([d.sstd[w] for w in wells])

    def loss(p):
        W = d2_weights(p, MD, NN, ST)
        return float(np.mean((Y - (S * W).sum(1)) ** 2))

    print(f"  D2: mse inicial {loss(x0):.3f}")
    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"maxiter": 3000, "xatol": 1e-3, "fatol": 1e-4})
    print(f"  D2: mse final {res.fun:.3f} ({np.sqrt(res.fun):.3f} ft) "
          f"params={np.round(res.x, 3).tolist()} iters={res.nit}")
    return res.x


def wf_d2(d, params):
    def f(wid):
        _, _, md, nn = d.sig[wid]
        return d2_weights(params, md, nn, d.sstd[wid])
    return f


def wf_e2(d, base_wf_point, ridge_tab, lam):
    """Logits por punto + sesgo por pozo hacia los pesos de C."""
    def f(wid):
        W = np.maximum(base_wf_point(wid), EPS)
        wc = (1 - lam) * W_GLOBAL + lam * ridge_tab[wid]
        off = np.log(np.maximum(wc, EPS)) - np.log(np.maximum(W_GLOBAL, EPS))
        return E.softmax_w3(np.log(W) + off[None, :])
    return f


def lb_proxy(d, wells, weight_of):
    sel = [w for w in wells if d.c.meta_ix.loc[w].n_pred >= LB_NPRED]
    r, _, _ = E.rmse_set(d, sel, weight_of)
    return r, len(sel)


def main():
    d = D2Data()
    wf_a = lambda wid: W_GLOBAL                               # noqa: E731

    E.report(d, "A  fijo 0.45/0.55", wf_a, "control")
    with open(HERE / "v6_pesos_params.json") as f:
        p1 = json.load(f)
    params_d = np.array(p1["D_params"])
    wf_d = E.wf_d(d, params_d)
    E.report(d, "D  por punto softmax(md,nn)", wf_d)

    x0 = np.array([params_d[0], params_d[1], params_d[2], 0.0,
                   params_d[3], params_d[4], params_d[5], 0.0])
    params_d2 = fit_d2(d, d.train, x0)
    wf_dd = wf_d2(d, params_d2)
    E.report(d, "D2 por punto softmax(md,nn,sstd)", wf_dd)

    model, mu, sd, alpha = E.fit_ridge_c(d)
    Xe = E.feat_frame(d.c.meta_ix).values
    P = np.clip(model.predict((Xe - mu) / sd), 0, None)
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-9)
    tab = {w: P[i] for i, w in enumerate(d.c.meta_ix.index)}
    for lam in (0.5, 0.75, 1.0):
        E.report(d, f"E2 D2 + sesgo C lam={lam:.2f}", wf_e2(d, wf_dd, tab, lam))
        E.report(d, f"E2 D  + sesgo C lam={lam:.2f}", wf_e2(d, wf_d, tab, lam))

    df = pd.DataFrame(E.ROWS)
    df.to_csv(HERE / "v6_pesos_resultados2.csv", index=False)
    cand = df[~df.variante.str.startswith(("ORACULO", "A "))]
    win = cand.loc[cand.rmse_k150.idxmin()]
    print(f"\nGANADORA FINAL: {win.variante} k150={win.rmse_k150:.3f} "
          f"k60={win.rmse_k60:.3f}")

    wfs = {"A": wf_a, "D": wf_d, "D2": wf_dd}
    for lam in (0.5, 0.75, 1.0):
        wfs[f"E2 D2 + sesgo C lam={lam:.2f}"] = wf_e2(d, wf_dd, tab, lam)
        wfs[f"E2 D  + sesgo C lam={lam:.2f}"] = wf_e2(d, wf_d, tab, lam)
    wf_win = wfs.get(win.variante.split()[0], wfs.get(win.variante, wf_dd))
    if win.variante.startswith("E2"):
        wf_win = wfs[win.variante]

    for nombre, wf in (("A", wf_a), ("GANADORA", wf_win)):
        r, nsel = lb_proxy(d, d.k150, wf)
        print(f"  lb_proxy k150 (n_pred>={LB_NPRED}, {nsel} pozos) {nombre}: {r:.3f}")

    E.estratos_nn(d, wf_win, wf_a, "ganadora")

    out = {
        "w_global": W_GLOBAL.tolist(), "orden": ["surf", "pf", "geom"],
        "surf": {"aniso": 16.0, "k": 24, "theta": 2.278489, "cal_tail": 500},
        "pf": "media S=64 seeds 1..64 N=600 (la de v5)",
        "geom": "recta dip anclada PS, win=700",
        "ganadora": str(win.variante),
        "rmse_k150": float(win.rmse_k150), "rmse_k60": float(win.rmse_k60),
        "D_params": [float(x) for x in params_d],
        "D_forma": "u_s=a0+a1*md/1000+a2*ln(max(nn,1)); u_p=0; "
                   "u_g=b0+b1*md/1000+b2*ln(nn); w=softmax; nn en metrica aniso",
        "D2_params": [float(x) for x in params_d2],
        "D2_forma": "como D + a3*ln(max(sstd,0.1)) en u_s y b3 en u_g "
                    "(orden a0,a1,a2,a3,b0,b1,b2,b3)",
        "C_ridge": {"alpha": alpha, "feats": E.FEATS, "mu": mu.tolist(),
                    "sd": sd.tolist(), "coef": model.coef_.tolist(),
                    "intercept": model.intercept_.tolist(),
                    "post": "clip>=0, normalizar suma 1"},
        "E2": "logits por punto + log(wC_shrunk)-log(w_global)",
        "backtest_descartado": "NNLS prefijo enmascarado NO transfiere "
                               "(lam>0 empeora monotonicamente)",
    }
    with open(HERE / "v6_pesos_params_final.json", "w") as f:
        json.dump(out, f, indent=1)
    print("guardado v6_pesos_params_final.json")


if __name__ == "__main__":
    main()
