"""Los 4 oraculos: cotas superiores de lo que queda por ganar.

  1) seleccion  : min por pozo sobre {geom,surf,ancc,pfz,beam} elegido con el TVT real
  2) blend      : pesos optimos por pozo (LS libre y convexo) con el TVT real
  3) offset     : mejor constante por pozo sobre el mejor predictor
  4) offset+pdte: mejor r(md)=r0+s*md_since por pozo

Se reportan sobre k=60 (comparable con 12.402), k=150 y los 770 pozos.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from orclib import Cache, PRED, rmse, subset  # noqa: E402

BEST = dict(surf=0.25, ancc=0.75)


def pooled(sse, n):
    return float(np.sqrt(sse.sum() / n.sum()))


def analiza(c, ids, nombre, rows):
    m = c.mask(ids)
    y = c.d["y"]
    wsel = np.zeros(len(c.lens), bool)
    for w in (ids if ids is not None else c.wells):
        if w in c.widx:
            wsel[c.widx[w]] = True
    idxs = np.where(wsel)[0]

    def add(k, v, extra=""):
        rows.append({"set": nombre, "variante": k, "rmse": v, "nota": extra})
        print(f"  {k:34s} {v:8.3f}  {extra}")

    print(f"\n=== {nombre} ({len(idxs)} pozos, {int(m.sum())} puntos) ===")
    # --- predictores individuales
    for p in PRED:
        add(p, rmse(y, c.d[p], m))
    mix = c.blend(**BEST)
    add("MEJOR mix .25surf+.75ancc", rmse(y, mix, m))
    add("media(ancc,pfz,beam)", rmse(y, (c.d["ancc"] + c.d["pfz"] + c.d["beam"]) / 3, m))

    # --- 1) oraculo de seleccion (por pozo, min SSE) --------------------------
    cands = list(PRED) + ["mix"]
    P = {p: c.d[p] for p in PRED}
    P["mix"] = mix
    sse = {p: np.zeros(len(idxs)) for p in cands}
    npt = np.zeros(len(idxs))
    for j, i in enumerate(idxs):
        s = c.sl(i)
        npt[j] = c.lens[i]
        for p in cands:
            sse[p][j] = float(((y[s] - P[p][s]) ** 2).sum())
    S5 = np.column_stack([sse[p] for p in PRED])
    S6 = np.column_stack([sse[p] for p in cands])
    add("ORACULO seleccion (5 base)", pooled(S5.min(1), npt),
        f"reparto={dict(zip(PRED, np.bincount(S5.argmin(1), minlength=5)))}")
    add("ORACULO seleccion (+mix)", pooled(S6.min(1), npt),
        f"reparto={dict(zip(cands, np.bincount(S6.argmin(1), minlength=6)))}")
    # seleccion realizable: siempre el mejor GLOBAL (referencia)
    add("  (seleccion trivial: siempre mix)", pooled(sse["mix"], npt))

    # --- 2) oraculo de blend --------------------------------------------------
    sse_ls = np.zeros(len(idxs))
    sse_cv = np.zeros(len(idxs))
    sse_2 = np.zeros(len(idxs))
    for j, i in enumerate(idxs):
        s = c.sl(i)
        A = np.column_stack([c.d[p][s] for p in PRED])
        t = y[s]
        w = np.linalg.lstsq(A, t, rcond=None)[0]
        sse_ls[j] = float(((A @ w - t) ** 2).sum())
        # convexo (pesos >=0 suma 1) por proyeccion simple: NNLS + normalizacion
        from scipy.optimize import nnls
        wc = nnls(A, t)[0]
        ssum = wc.sum()
        wc = wc / ssum if ssum > 0 else np.ones(len(PRED)) / len(PRED)
        sse_cv[j] = float(((A @ wc - t) ** 2).sum())
        # blend de 2: surf y ancc con peso libre
        A2 = np.column_stack([c.d["surf"][s], c.d["ancc"][s]])
        w2 = np.linalg.lstsq(A2, t, rcond=None)[0]
        sse_2[j] = float(((A2 @ w2 - t) ** 2).sum())
    add("ORACULO blend LS 5 (por pozo)", pooled(sse_ls, npt))
    add("ORACULO blend convexo 5 (pozo)", pooled(sse_cv, npt))
    add("ORACULO blend 2 surf+ancc (pozo)", pooled(sse_2, npt))

    # --- 3) y 4) oraculos de offset / offset+pendiente ------------------------
    for base_name, base in (("mix", mix), ("ancc", c.d["ancc"]), ("surf", c.d["surf"])):
        s_off = np.zeros(len(idxs))
        s_lin = np.zeros(len(idxs))
        offs = np.zeros(len(idxs))
        slps = np.zeros(len(idxs))
        for j, i in enumerate(idxs):
            s = c.sl(i)
            r = y[s] - base[s]
            offs[j] = r.mean()
            s_off[j] = float(((r - r.mean()) ** 2).sum())
            x = c.d["md_since"][s]
            A = np.column_stack([np.ones(len(x)), x])
            co = np.linalg.lstsq(A, r, rcond=None)[0]
            slps[j] = co[1]
            s_lin[j] = float(((r - A @ co) ** 2).sum())
        add(f"ORACULO offset sobre {base_name}", pooled(s_off, npt),
            f"|offset| mediana={np.median(np.abs(offs)):.2f} ft")
        add(f"ORACULO offset+pdte sobre {base_name}", pooled(s_lin, npt),
            f"|pdte| mediana={np.median(np.abs(slps))*1000:.2f} ft/1000ft")

    # --- oraculo combinado: seleccion + offset --------------------------------
    s_so = np.zeros(len(idxs))
    for j, i in enumerate(idxs):
        s = c.sl(i)
        best = 1e30
        for p in cands:
            r = y[s] - P[p][s]
            best = min(best, float(((r - r.mean()) ** 2).sum()))
        s_so[j] = best
    add("ORACULO seleccion+offset", pooled(s_so, npt))
    return rows


def main():
    c = Cache()
    rows = []
    for k in (60, 150, None):
        ids = subset(k) if k else None
        analiza(c, ids, f"k={k}" if k else "TODOS(770)", rows)
    pd.DataFrame(rows).to_csv(HERE / "orc02_oraculos.csv", index=False)
    print(f"\nguardado {HERE/'orc02_oraculos.csv'}")


if __name__ == "__main__":
    main()
