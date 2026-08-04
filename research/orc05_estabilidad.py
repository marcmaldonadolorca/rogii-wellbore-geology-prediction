"""Estabilidad: cuanta diferencia entre variantes es senal y cuanta es ruido.

  a) 5 subconjuntos DISJUNTOS de ~154 pozos: media y desviacion del RMSE.
  b) bootstrap por pozo (B=2000) sobre los 770: sd del RMSE y sd de la DIFERENCIA
     pareada contra el mejor modelo (lo que de verdad importa al comparar).
  c) el mismo bootstrap restringido a k=60 y k=150 (los tamanyos que usamos).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from orclib import Cache, PRED, subset  # noqa: E402

B = 2000


def sse_n(c, pred):
    e2 = (c.d["y"] - pred) ** 2
    return np.bincount(c.wpt, weights=e2, minlength=len(c.lens)), c.lens.astype(float)


def main():
    c = Cache()
    mods = {"mix .25surf+.75ancc": c.blend(surf=0.25, ancc=0.75),
            "mix .20surf+.80ancc": c.blend(surf=0.20, ancc=0.80),
            "mix .30surf+.70ancc": c.blend(surf=0.30, ancc=0.70),
            "mix .25s+.65a+.10pfz": c.blend(surf=0.25, ancc=0.65, pfz=0.10),
            "ancc": c.d["ancc"], "surf": c.d["surf"], "pfz": c.d["pfz"],
            "beam": c.d["beam"]}
    S = {k: sse_n(c, v)[0] for k, v in mods.items()}
    n = c.lens.astype(float)
    nw = len(n)

    # --- a) 5 folds disjuntos -------------------------------------------------
    rng = np.random.default_rng(11)
    perm = rng.permutation(nw)
    folds = np.array_split(perm, 5)
    print(f"=== 5 subconjuntos disjuntos ({[len(f) for f in folds]} pozos) ===")
    print(f"{'modelo':24s} " + " ".join(f"f{i+1:<7d}" for i in range(5)) +
          f" {'media':>8s} {'sd':>7s} {'pooled770':>10s}")
    rows = []
    for k, s in S.items():
        vals = [float(np.sqrt(s[f].sum() / n[f].sum())) for f in folds]
        pool = float(np.sqrt(s.sum() / n.sum()))
        print(f"{k:24s} " + " ".join(f"{v:8.3f}" for v in vals) +
              f" {np.mean(vals):8.3f} {np.std(vals, ddof=1):7.3f} {pool:10.3f}")
        rows.append({"modelo": k, "media_folds": np.mean(vals),
                     "sd_folds": np.std(vals, ddof=1), "pooled770": pool,
                     **{f"fold{i+1}": v for i, v in enumerate(vals)}})

    # --- b) bootstrap por pozo -----------------------------------------------
    print(f"\n=== bootstrap por pozo, B={B} ===")
    for tag, ids in (("770 pozos", None), ("k=150", subset(150)), ("k=60", subset(60))):
        if ids is None:
            idx0 = np.arange(nw)
        else:
            idx0 = np.array([c.widx[w] for w in ids if w in c.widx])
        rr = np.random.default_rng(3)
        draws = rr.integers(0, len(idx0), size=(B, len(idx0)))
        pick = idx0[draws]
        nn = n[pick].sum(1)
        boot = {k: np.sqrt(S[k][pick].sum(1) / nn) for k in S}
        base = "mix .25surf+.75ancc"
        print(f"\n-- {tag} ({len(idx0)} pozos) --")
        print(f"{'modelo':24s} {'RMSE':>8s} {'sd':>7s} {'IC95':>17s} "
              f"{'dif vs mix':>11s} {'sd(dif)':>8s} {'P(mejor)':>9s}")
        for k in S:
            pt = float(np.sqrt(S[k][idx0].sum() / n[idx0].sum()))
            d = boot[k] - boot[base]
            lo, hi = np.percentile(boot[k], [2.5, 97.5])
            print(f"{k:24s} {pt:8.3f} {boot[k].std():7.3f} [{lo:7.3f},{hi:7.3f}] "
                  f"{d.mean():11.3f} {d.std():8.3f} {(d < 0).mean():9.3f}")
            if ids is None:
                for r in rows:
                    if r["modelo"] == k:
                        r["sd_boot770"] = float(boot[k].std())
                        r["sd_dif_vs_mix"] = float(d.std())
    pd.DataFrame(rows).to_csv(HERE / "orc05_estabilidad.csv", index=False)
    print(f"\nguardado {HERE/'orc05_estabilidad.csv'}")


if __name__ == "__main__":
    main()
