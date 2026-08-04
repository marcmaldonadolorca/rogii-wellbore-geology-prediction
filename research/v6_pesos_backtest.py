"""v6 pesos — variante 1: NNLS por pozo sobre la COLA ENMASCARADA del prefijo.

Por pozo eval: c2 = round(0.65*cut); se enmascara TVT_input desde c2 y se
re-predicen [c2:cut] con S (superficie theta+11, LOWO), P (PF ancc S=16) y G
(dip anclado). NNLS contra el TVT_input real de esa cola (conocido en test,
sin leak) -> w_bt. Aplicacion: w = (1-lam)*(0.45,0.55,0) + lam*w_bt, barrido lam.

Fase compute (cara, ~2 s/pozo): python v6_pesos_backtest.py compute
Fase eval (desde el rig):        python v6_pesos_backtest.py eval
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

OUT = HERE / "v6_pesos_backtest.json"


def compute():
    from cv import TEST_COLS, load_well
    from v4_gbm import candidates, get_field, hdist

    z60 = np.load(HERE / "v4_gbm_eval60.npz"); z150 = np.load(HERE / "v4_gbm_eval150.npz")
    ids = sorted(set(str(i) for i in z60["ids"]) | set(str(i) for i in z150["ids"]))
    field = get_field()
    out, t0 = {}, time.time()
    for j, wid in enumerate(ids):
        df, tw, cut = load_well(wid)
        if cut < 20 or cut >= len(df):
            continue
        c2 = int(round(0.65 * cut))
        rec = dict(w2=[0.45, 0.55], w3=[0.45, 0.55, 0.0], n_bt=cut - c2)
        if c2 >= 60 and cut - c2 >= 30:
            df_h = df[TEST_COLS].copy()
            hd = hdist(df_h)
            d2 = df_h.iloc[:cut].copy()
            d2.loc[d2.index[c2:], "TVT_input"] = np.nan
            y = df_h.TVT_input.values[c2:cut]
            try:
                c = candidates(d2, tw, wid, c2, field, hd[:cut])
                A3 = np.column_stack([c["S"], c["P"], c["G"]])
                w3, _ = nnls(A3, y)
                s = w3.sum()
                if s > 0:
                    rec["w3"] = (w3 / s).tolist()
                w2, _ = nnls(A3[:, :2], y)
                s = w2.sum()
                if s > 0:
                    rec["w2"] = (w2 / s).tolist()
                rec["bt"] = [float(np.sqrt(np.mean((y - A3[:, i]) ** 2))) for i in range(3)]
            except Exception as e:
                rec["err"] = str(e)
        out[wid] = rec
        if (j + 1) % 25 == 0:
            print(f"  {j+1}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)
    OUT.write_text(json.dumps(out))
    print(f"guardado {OUT} ({len(out)} pozos, {(time.time()-t0)/len(out):.2f}s/pozo)")


def evaluate():
    from v6_pesos_rig import LAMS, W_GLOBAL, clip_norm, load_set, pooled
    bt = json.loads(OUT.read_text())
    rows = []
    for which, tag, p64 in (("eval60", "k60_P64", True), ("eval150", "k150_P64", True),
                            ("eval60", "k60_P16", False), ("eval150", "k150_P16", False)):
        wells = load_set(which, p64=p64)
        base = pooled(wells, lambda wl: W_GLOBAL)
        print(f"[{tag}] base {base:.3f}")
        for key in ("w2", "w3"):
            for lam in LAMS:
                def w_of(wl, key=key, lam=lam):
                    w = np.array(bt[wl["wid"]][key] + ([0.0] if key == "w2" else []))
                    return clip_norm((1 - lam) * W_GLOBAL + lam * w)
                r = pooled(wells, w_of)
                rows.append({"variante": f"btNNLS {key} lam={lam:.1f}", "set": tag, "rmse": r})
                print(f"  btNNLS {key} lam={lam:.1f} -> {r:8.3f}", flush=True)
    pd.DataFrame(rows).to_csv(HERE / "v6_pesos_backtest.csv", index=False)


if __name__ == "__main__":
    (compute if sys.argv[1:2] == ["compute"] else evaluate)()
