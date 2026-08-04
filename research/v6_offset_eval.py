"""v6_offset: evaluacion honesta con cv.py — pred = blend16 + off_hat(pozo).

blend16 = 0.45*superficie(theta=2.278489, aniso=16, k=24, LOWO) +
          0.55*PF multiseed media S=64 (npz de v4, el del envio).
off_hat sale de research/v6_offset_estimator.json entrenado SOLO en pozos
disjuntos de k60/k150 (v6_offset_fit.py); las features de los pozos de eval
estan precalculadas leak-free en v6_offset_ds.csv (v6_offset_ds.py).

Variantes evaluadas: base (sin offset), shrink, huber, lgbm y las *_g05
(mitad de gamma, el lado seguro del shrinkage).

Uso: nice -n 10 .venv/bin/python research/v6_offset_eval.py [60|150|both]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))

from cv import evaluate                    # noqa: E402
from model import SurfaceField             # noqa: E402

THETA_OPT, ANISO, W_SURF, CAL_TAIL = 2.278489, 16.0, 0.45, 500

EST = json.load(open(R / "v6_offset_estimator.json"))
DS = pd.read_csv(R / "v6_offset_ds.csv").set_index("well")

PF64 = {}
for f in ("v4_multiseed_k60.npz", "v4_multiseed_k150.npz"):
    z = np.load(R / f)
    for kk in z.files:
        if kk.endswith("_pred"):
            PF64[kk[:-5]] = z[kk].astype(float)

_LGBM = None


def _xrow(wid):
    row = DS.loc[wid]
    return np.array([row[c] if np.isfinite(row[c]) else EST["imputation"][c]
                     for c in EST["feats"]], float)


def off_hat(wid, model, gscale=1.0):
    global _LGBM
    if model == "base":
        return 0.0
    clip = EST["clip"]
    if model == "shrink":
        m = EST["models"]["shrink"]
        v = m["lam"] * gscale * float(DS.loc[wid, m["feature"]])
    elif model == "huber":
        m = EST["models"]["huber"]
        xs = (_xrow(wid) - np.array(m["scaler_mean"])) / np.array(m["scaler_scale"])
        v = m["gamma"] * gscale * (float(np.dot(m["coef"], xs)) + m["intercept"])
    elif model == "lgbm":
        import lightgbm as lgb
        if _LGBM is None:
            _LGBM = lgb.Booster(model_file=str(R / EST["models"]["lgbm"]["file"]))
        v = EST["models"]["lgbm"]["gamma"] * gscale * float(_LGBM.predict(_xrow(wid)[None])[0])
    else:
        raise ValueError(model)
    return float(np.clip(v, -clip, clip))


def make_predict(field, model, gscale=1.0):
    def predict(df_h, tw, wid=None):
        m = df_h.TVT_input.isna()
        cut = int(m.idxmax()) if m.any() else len(df_h)
        s, _ = field.interp(df_h.X.values, df_h.Y.values, exclude=wid)
        lo = max(0, cut - CAL_TAIL)
        z, tvti = df_h.Z.values, df_h.TVT_input.values
        c = np.median(tvti[lo:cut] + z[lo:cut] - s[lo:cut])
        prior = (s - z + c)[cut:]
        blend = W_SURF * prior + (1 - W_SURF) * PF64[wid]
        return blend + off_hat(wid, model, gscale)
    return predict


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "both"
    ks = [60, 150] if arg == "both" else [int(arg)]
    t0 = time.time()
    field = SurfaceField(aniso=ANISO, theta=THETA_OPT)
    print(f"SurfaceField: {time.time()-t0:.0f}s", flush=True)

    variants = [("base", "base", 1.0), ("shrink", "shrink", 1.0),
                ("shrink_g05", "shrink", 0.5), ("huber", "huber", 1.0),
                ("huber_g05", "huber", 0.5), ("lgbm", "lgbm", 1.0),
                ("lgbm_g05", "lgbm", 0.5)]
    res = {}
    for k in ks:
        print(f"\n===== k={k} =====", flush=True)
        base_pw = None
        for name, model, gs in variants:
            r = evaluate(make_predict(field, model, gs), k=k, verbose=False)
            res[(k, name)] = r
            d = ""
            if name == "base":
                base_pw = r["per_well"].set_index("well")
            else:
                d = f"  (delta {r['rmse']-res[(k,'base')]['rmse']:+.3f})"
            print(f"  {name:12s} rmse {r['rmse']:7.3f}  lb_proxy {r['rmse_lb_proxy']:7.3f}{d}",
                  flush=True)

        # el mejor no-base: mejora por decil de aislamiento (nn_p90)
        best = min((n for n, m, g in variants if n != "base"),
                   key=lambda n: res[(k, n)]["rmse"])
        pw = res[(k, best)]["per_well"].set_index("well")
        j = base_pw.join(pw, rsuffix="_v").join(DS[["nn_p90", "off_bt_tail"]])
        j["d_sse"] = j.sse_v - j.sse
        dec = pd.qcut(j.nn_p90, 4, duplicates="drop")
        print(f"\n  mejor={best}: delta SSE por cuartil de nn_p90 (neg = mejora):")
        for iv, g in j.groupby(dec, observed=True):
            print(f"    nn_p90 {iv}: dSSE {g.d_sse.sum():+.3e}  "
                  f"({(g.d_sse<0).sum()}/{len(g)} pozos mejoran)")
        n_worse = (j.d_sse > 0).sum()
        print(f"  pozos que empeoran: {n_worse}/{len(j)}")
    print("\nreferencias: blend puro 10.200 @k150 (v4_k150_confirm)")


if __name__ == "__main__":
    main()
