"""v6-pesos etapa 2: evaluacion de pesos por pozo / por punto del blend.

Variantes (orden de pesos siempre (surf, pf, geom), global = (0.45, 0.55, 0)):
  A  baseline fijo 0.45/0.55            (control: 10.200 en k=150)
  B  NNLS sobre el backtest enmascarado del prefijo + shrinkage lam hacia global
  B2 pesos 1/rmse_bt^2 (inverso-varianza del backtest) + shrinkage
  C  regresion ridge features_pozo -> pesos NNLS oraculo, ajustada en ~400 pozos
     disjuntos; prediccion clip>=0 y normalizada; + shrinkage
  D  pesos POR PUNTO softmax-lineal en (md_since/1000, log nn) ajustados por MSE
     pooled en los pozos de entrenamiento (6 parametros)
  E  D con sesgo por pozo: logits de D + log(pesos B(lam*)) como offset

Reporta k=150 y k=60 (señales identicas al blend de referencia: surf theta-opt
+ PF media S=64 de los npz v4), y RMSE por decil de nn_med para la variante
ganadora (regimen aislado = el que el LB castiga).

Salidas: v6_pesos_resultados.csv, v6_pesos_params.json, log por stdout.
"""
import json

import numpy as np
import pandas as pd

from v6_pesos_lib import HERE, PesosCache, W_GLOBAL, subsets

EPS = 1e-6
LAMS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


# ------------------------------------------------------------------ datos
class D:
    def __init__(self):
        self.c = PesosCache()
        _, k60, k150 = subsets()
        have = set(self.c.meta.well)
        self.k60 = sorted(k60 & have)
        self.k150 = sorted(k150 & have)
        self.train = self.c.wells("run8")
        self.sig = {}      # wid -> (y, S(n,3), md, nn)
        for w in self.c.meta.well:
            y = self.c.arr(w, "y")
            S = np.column_stack([self.c.arr(w, "surf"), self.c.arr(w, "pf"),
                                 self.c.arr(w, "geom")])
            self.sig[w] = (y, S, self.c.arr(w, "md"), self.c.arr(w, "nn"))

    def w_bt(self, wid):
        r = self.c.meta_ix.loc[wid]
        w = np.array([r.w_bt_s, r.w_bt_p, r.w_bt_g])
        return None if np.isnan(w).any() else w

    def w_or(self, wid):
        r = self.c.meta_ix.loc[wid]
        w = np.array([r.w_or_s, r.w_or_p, r.w_or_g])
        return None if np.isnan(w).any() else w


def rmse_set(d, wells, weight_of):
    """weight_of(wid) -> (3,) o (n,3). RMSE pooled + sse por pozo."""
    sse_l, n_l = [], []
    for w in wells:
        y, S, _, _ = d.sig[w]
        W = np.asarray(weight_of(w))
        p = S @ W if W.ndim == 1 else (S * W).sum(1)
        sse_l.append(float(((y - p) ** 2).sum()))
        n_l.append(len(y))
    sse, n = np.array(sse_l), np.array(n_l)
    return float(np.sqrt(sse.sum() / n.sum())), sse, n


ROWS = []
_REF = {}


def boot_delta(sse_v, sse_a, n, B=2000):
    """sd pareada de la diferencia de RMSE pooled (bootstrap por pozo)."""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(n), size=(B, len(n)))
    rv = np.sqrt(sse_v[idx].sum(1) / n[idx].sum(1))
    ra = np.sqrt(sse_a[idx].sum(1) / n[idx].sum(1))
    return float(np.std(rv - ra))


def report(d, nombre, weight_of, nota=""):
    r150, sse, n = rmse_set(d, d.k150, weight_of)
    r60, _, _ = rmse_set(d, d.k60, weight_of)
    if "A" not in _REF:
        _REF["A"] = (r150, sse, n)
    else:
        ra, sse_a, n_a = _REF["A"]
        sd = boot_delta(sse, sse_a, n_a)
        nota = f"d150={r150-ra:+.3f} (sd_par {sd:.3f}) " + nota
    ROWS.append({"variante": nombre, "rmse_k150": r150, "rmse_k60": r60, "nota": nota})
    print(f"{nombre:42s} k150={r150:8.3f}  k60={r60:8.3f}  {nota}", flush=True)
    return r150


# ------------------------------------------------------------------ B / B2
def wf_backtest(d, lam):
    def f(wid):
        wb = d.w_bt(wid)
        if wb is None:
            return W_GLOBAL
        return (1 - lam) * W_GLOBAL + lam * wb
    return f


def wf_invvar(d, lam):
    def f(wid):
        r = d.c.meta_ix.loc[wid]
        rr = np.array([r.rmse_surf_bt, r.rmse_pf_bt, r.rmse_geom_bt])
        if np.isnan(rr).any() or (rr <= 0).any():
            return W_GLOBAL
        w = 1.0 / rr ** 2
        w = w / w.sum()
        return (1 - lam) * W_GLOBAL + lam * w
    return f


# ------------------------------------------------------------------ C
FEATS = ["log_nn", "log_nn90", "log_sstd", "ll_sd", "log_dis_sp", "log_dis_sg",
         "log_dis_pg", "mdk_len", "ps_frac", "npk", "w_bt_s", "w_bt_p", "w_bt_g",
         "log_r_s_bt", "log_r_p_bt", "log_r_g_bt"]


def feat_frame(meta):
    f = pd.DataFrame(index=meta.index)
    f["log_nn"] = np.log(np.maximum(meta.nn_med, 1.0))
    f["log_nn90"] = np.log(np.maximum(meta.nn_p90, 1.0))
    f["log_sstd"] = np.log(np.maximum(meta.sstd_mean, 1e-3))
    f["ll_sd"] = meta.ll_sd
    f["log_dis_sp"] = np.log(np.maximum(meta.dis_sp, 1e-3))
    f["log_dis_sg"] = np.log(np.maximum(meta.dis_sg, 1e-3))
    f["log_dis_pg"] = np.log(np.maximum(meta.dis_pg, 1e-3))
    f["mdk_len"] = meta.md_len / 1000.0
    f["ps_frac"] = meta.ps_frac
    f["npk"] = meta.n_pred / 1000.0
    for c in ("w_bt_s", "w_bt_p", "w_bt_g"):
        f[c] = meta[c].fillna(W_GLOBAL[["w_bt_s", "w_bt_p", "w_bt_g"].index(c)])
    for c, src in (("log_r_s_bt", "rmse_surf_bt"), ("log_r_p_bt", "rmse_pf_bt"),
                   ("log_r_g_bt", "rmse_geom_bt")):
        f[c] = np.log(np.maximum(meta[src].fillna(meta[src].median()), 1e-2))
    return f[FEATS]


def fit_ridge_c(d, alphas=(0.1, 1.0, 10.0, 100.0)):
    """Ridge multisalida sobre pozos train; alpha por CV 5-fold a nivel pozo."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    tr = d.c.meta[d.c.meta.pf_source == "run8"].dropna(subset=["w_or_s"])
    X = feat_frame(tr).values
    Y = tr[["w_or_s", "w_or_p", "w_or_g"]].values
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    best, best_mse = None, np.inf
    for a in alphas:
        mse = []
        for itr, ite in KFold(5, shuffle=True, random_state=0).split(Xs):
            m = Ridge(alpha=a).fit(Xs[itr], Y[itr])
            mse.append(float(np.mean((m.predict(Xs[ite]) - Y[ite]) ** 2)))
        if np.mean(mse) < best_mse:
            best, best_mse = a, float(np.mean(mse))
    model = Ridge(alpha=best).fit(Xs, Y)
    print(f"  C ridge: alpha={best} cv_mse_pesos={best_mse:.4f} "
          f"R2={model.score(Xs, Y):.3f}")
    return model, mu, sd, best


def wf_ridge(d, model, mu, sd, lam):
    Xe = feat_frame(d.c.meta_ix).values
    P = model.predict((Xe - mu) / sd)
    P = np.clip(P, 0, None)
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-9)
    tab = {w: P[i] for i, w in enumerate(d.c.meta_ix.index)}

    def f(wid):
        return (1 - lam) * W_GLOBAL + lam * tab[wid]
    return f


# ------------------------------------------------------------------ D
def softmax_w3(U):
    U = U - U.max(1, keepdims=True)
    E = np.exp(U)
    return E / E.sum(1, keepdims=True)


def d_weights(params, md, nn):
    a0, a1, a2, b0, b1, b2 = params
    mdk = md / 1000.0
    lnn = np.log(np.maximum(nn, 1.0))
    u = np.column_stack([a0 + a1 * mdk + a2 * lnn,
                         np.zeros(len(md)),
                         b0 + b1 * mdk + b2 * lnn])
    return softmax_w3(u)


def fit_d(d, wells, x0=None):
    from scipy.optimize import minimize
    packs = [d.sig[w] for w in wells]
    Y = np.concatenate([p[0] for p in packs])
    S = np.vstack([p[1] for p in packs])
    MD = np.concatenate([p[2] for p in packs])
    NN = np.concatenate([p[3] for p in packs])

    def loss(params):
        W = d_weights(params, MD, NN)
        return float(np.mean((Y - (S * W).sum(1)) ** 2))

    if x0 is None:
        x0 = np.array([np.log(0.45 / 0.55), 0.0, 0.0, -3.0, 0.0, 0.0])
    print(f"  D: mse inicial {loss(x0):.3f} ({np.sqrt(loss(x0)):.3f} ft)")
    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-3, "fatol": 1e-4})
    print(f"  D: mse final {res.fun:.3f} ({np.sqrt(res.fun):.3f} ft) "
          f"params={np.round(res.x, 3).tolist()} iters={res.nit}")
    return res.x


def wf_d(d, params):
    def f(wid):
        _, _, md, nn = d.sig[wid]
        return d_weights(params, md, nn)
    return f


# ------------------------------------------------------------------ E
def wf_e(d, params, lam):
    base = wf_backtest(d, lam)

    def f(wid):
        _, _, md, nn = d.sig[wid]
        W = d_weights(params, md, nn)
        wb = np.maximum(base(wid), EPS)
        wg = np.maximum(W_GLOBAL, EPS)
        U = np.log(W) + (np.log(wb) - np.log(wg))[None, :]
        return softmax_w3(U)
    return f


# ------------------------------------------------------------------ estratos
def estratos_nn(d, weight_of, weight_ref, etiqueta):
    wells = sorted(set(d.k150) | set(d.k60))
    nn = d.c.meta_ix.loc[wells].nn_med.values
    q = np.quantile(nn, np.linspace(0, 1, 11))
    q[-1] += 1
    print(f"\n-- estratos nn_med ({etiqueta} vs baseline, {len(wells)} pozos eval --")
    for lo, hi in zip(q[:-1], q[1:]):
        sel = [w for w, v in zip(wells, nn) if lo <= v < hi]
        if not sel:
            continue
        r_v, _, _ = rmse_set(d, sel, weight_of)
        r_a, _, _ = rmse_set(d, sel, weight_ref)
        print(f"  nn[{lo:8.0f},{hi:8.0f}) n={len(sel):3d}  base={r_a:7.3f}  "
              f"{etiqueta}={r_v:7.3f}  d={r_v - r_a:+7.3f}", flush=True)


# ==================================================================== main
def main():
    d = D()
    print(f"pozos: eval k150={len(d.k150)} k60={len(d.k60)} train={len(d.train)}\n")

    wf_a = lambda wid: W_GLOBAL                                # noqa: E731
    report(d, "A  fijo 0.45/0.55", wf_a, "control")

    # oraculo local (cota superior con estas 3 señales)
    def wf_oracle(wid):
        wo = d.w_or(wid)
        return W_GLOBAL if wo is None else wo
    report(d, "ORACULO NNLS 3-señales por pozo", wf_oracle, "cota, usa TVT real")

    print()
    for lam in LAMS:
        report(d, f"B  backtest NNLS lam={lam:.1f}", wf_backtest(d, lam))
    print()
    for lam in LAMS:
        report(d, f"B2 backtest 1/rmse^2 lam={lam:.1f}", wf_invvar(d, lam))

    print()
    model, mu, sd, alpha = fit_ridge_c(d)
    for lam in (0.25, 0.5, 0.75, 1.0):
        report(d, f"C  ridge features lam={lam:.2f}", wf_ridge(d, model, mu, sd, lam))

    print()
    params_d = fit_d(d, d.train)
    report(d, "D  por punto softmax(md,nn)", wf_d(d, params_d))

    print()
    best_b = min((r for r in ROWS if r["variante"].startswith("B ")),
                 key=lambda r: r["rmse_k150"])
    lam_b = float(best_b["variante"].split("lam=")[1])
    report(d, f"E  D + sesgo B(lam={lam_b:.1f})", wf_e(d, params_d, lam_b))

    df = pd.DataFrame(ROWS)
    df.to_csv(HERE / "v6_pesos_resultados.csv", index=False)
    print(f"\nguardado v6_pesos_resultados.csv")

    # ganadora (sin contar el oraculo) y estratos nn
    cand = df[~df.variante.str.startswith(("ORACULO", "A "))]
    win = cand.loc[cand.rmse_k150.idxmin()]
    print(f"\nGANADORA: {win.variante} k150={win.rmse_k150:.3f} k60={win.rmse_k60:.3f}")
    tag = win.variante.split()[0]
    wf_win = {"B": lambda: wf_backtest(d, float(win.variante.split("lam=")[1])),
              "B2": lambda: wf_invvar(d, float(win.variante.split("lam=")[1])),
              "C": lambda: wf_ridge(d, model, mu, sd, float(win.variante.split("lam=")[1])),
              "D": lambda: wf_d(d, params_d),
              "E": lambda: wf_e(d, params_d, lam_b)}[tag]()
    estratos_nn(d, wf_win, wf_a, tag)
    # el regimen aislado tambien para B con la lam elegida aunque no gane
    if tag != "B":
        estratos_nn(d, wf_backtest(d, lam_b), wf_a, f"B lam={lam_b:.1f}")

    params = {
        "w_global": W_GLOBAL.tolist(),
        "orden": ["surf", "pf", "geom"],
        "backtest": {"frac": 0.65, "S": 8, "seeds": "1..8", "N": 600,
                     "min_n_bt": 30, "nnls": "normalizado a suma 1"},
        "ganadora": win.variante,
        "rmse_k150": float(win.rmse_k150), "rmse_k60": float(win.rmse_k60),
        "B_lam_mejor": lam_b,
        "C_ridge": {"alpha": alpha, "feats": FEATS, "mu": mu.tolist(),
                    "sd": sd.tolist(), "coef": model.coef_.tolist(),
                    "intercept": model.intercept_.tolist()},
        "D_params": [float(x) for x in params_d],
    }
    with open(HERE / "v6_pesos_params.json", "w") as f:
        json.dump(params, f, indent=1)
    print("guardado v6_pesos_params.json")


if __name__ == "__main__":
    main()
