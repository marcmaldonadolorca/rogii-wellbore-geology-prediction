"""Cache v2: candidatos con el PF PROMEDIADO sobre R replicas (el PF de 600
particulas tiene sd=1.0 ft en el pooled RMSE; ver ad02) + senales de
incertidumbre + backtest leak-free en la cola del prefijo.

Guarda por pozo (post-PS y por tramo de backtest):
  y, S(superficie), H(sup+HMM), P(pf_ancc medio de R), Z(pf_z medio de RZ),
  B(beam), G(geometrico), Pspr (sd entre replicas del PF = incertidumbre
  Monte Carlo), Pstd (sd posterior del PF), nn, md_since, hd_since, MDv, Zv.

Uso:  python research/ad03_cache.py <k> <REP> <tag>
      tag=eval -> los k pozos de cv._select ; tag=fit -> k pozos DISJUNTOS
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from cv import SEED, TEST_COLS, _select, load_well, well_ids  # noqa: E402
import model as M  # noqa: E402
import pf_publico as PF  # noqa: E402
from ad01_cache import geom_pred, hdist  # noqa: E402

FRACS = (0.5, 0.65, 0.75)
NAMES = ("S", "H", "P", "Z", "B", "G")
EXTRA = ("Pspr", "Pstd")


def candidates(df_h, tw, wid, cut, field, hd, rep, repz):
    n = len(df_h)
    s_full, nn = M._prior(df_h, field, wid, cut)
    S = s_full[cut:]
    H = S + M.hmm_refine(df_h, tw, s_full, cut)
    t, g = PF._tw(tw)
    ps, pst = [], []
    for _ in range(rep):
        a, b = PF.run_pf_ancc(df_h, t, g)
        ps.append(np.asarray(a, float)); pst.append(np.asarray(b, float))
    P = np.mean(ps, 0)
    Pspr = np.std(ps, 0) if rep > 1 else np.zeros(n - cut)
    Pstd = np.mean(pst, 0)
    Zp = np.mean([np.asarray(PF.run_pf_z(df_h, t, g)[0], float) for _ in range(repz)], 0)
    bs, mc, es, r, _ = PF.BEAMS[0]
    B = np.asarray(PF.beam_search(df_h.GR.values[cut:], t, g,
                                  float(df_h.TVT_input.values[cut - 1]), bs, mc, es, r), float)
    G = geom_pred(df_h, cut, hd)[cut:]
    out = dict(S=S, H=H, P=P, Z=Zp, B=B, G=G, Pspr=Pspr, Pstd=Pstd)
    for k, v in out.items():
        assert len(v) == n - cut, (k, len(v), n - cut)
    return out, nn[cut:]


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rep = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    tag = sys.argv[3] if len(sys.argv) > 3 else "eval"
    repz = max(1, rep // 2)
    ev = _select(well_ids(), 60, SEED)
    ev150 = _select(well_ids(), 150, SEED)
    if tag == "eval":
        ids = _select(well_ids(), k, SEED)
    else:
        excl = set(ev) | set(ev150)
        rest = [w for w in well_ids() if w not in excl]
        rng = np.random.default_rng(7)
        ids = sorted(rng.choice(rest, size=min(k, len(rest)), replace=False))
    field = M.SurfaceField()
    print(f"tag={tag} k={k} rep={rep} pozos={len(ids)}", flush=True)

    cols = list(NAMES) + list(EXTRA)
    store = {n: [] for n in cols + ["y", "nn", "md_since", "hd_since", "MDv", "Zv"]}
    bt = {f: {n: [] for n in cols + ["y", "md_since"]} for f in FRACS}
    lens, btlens, meta = [], {f: [] for f in FRACS}, []
    t0 = time.time()
    for j, wid in enumerate(ids):
        df, tw, cut = load_well(wid)
        if cut < 20 or cut >= len(df):
            continue
        df_h = df[TEST_COLS].copy()
        hd = hdist(df_h)
        cand, nn = candidates(df_h, tw, wid, cut, field, hd, rep, repz)
        for n in cols:
            store[n].append(cand[n].astype(np.float32))
        store["y"].append(df.TVT.values[cut:].astype(np.float32))
        store["nn"].append(nn.astype(np.float32))
        store["md_since"].append((df_h.MD.values[cut:] - df_h.MD.values[cut - 1]).astype(np.float32))
        store["hd_since"].append((hd[cut:] - hd[cut - 1]).astype(np.float32))
        store["MDv"].append(df_h.MD.values[cut:].astype(np.float32))
        store["Zv"].append(df_h.Z.values[cut:].astype(np.float32))
        lens.append(len(cand["S"]))

        m = {"well": wid, "cut": cut, "n": len(df), "n_pred": len(df) - cut,
             "nn_med": float(np.median(nn)),
             "tv0": float(df_h.TVT_input.values[cut - 1]),
             "md0": float(df_h.MD.values[cut - 1]), "z0": float(df_h.Z.values[cut - 1])}
        for f in FRACS:
            c2 = int(round(f * cut))
            if c2 < 60 or cut - c2 < 30:
                btlens[f].append(0)
                for n in cols + ["y", "md_since"]:
                    bt[f][n].append(np.zeros(0, np.float32))
                continue
            d2 = df_h.iloc[:cut].copy()
            d2.loc[d2.index[c2:], "TVT_input"] = np.nan
            c2d, _ = candidates(d2, tw, wid, c2, field, hd[:cut], rep, repz)
            yb = df_h.TVT_input.values[c2:cut]
            for n in cols:
                bt[f][n].append(c2d[n].astype(np.float32))
                if n in NAMES:
                    m[f"bt{n}{f}"] = float(np.sqrt(np.mean((yb - c2d[n]) ** 2)))
            bt[f]["y"].append(yb.astype(np.float32))
            bt[f]["md_since"].append(
                (df_h.MD.values[c2:cut] - df_h.MD.values[c2 - 1]).astype(np.float32))
            btlens[f].append(cut - c2)
        meta.append(m)
        if (j + 1) % 10 == 0:
            print(f"  {j+1}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)

    out = {n: np.concatenate(v) for n, v in store.items()}
    out["lens"] = np.array(lens)
    for f in FRACS:
        ft = str(f).replace(".", "")
        out[f"btlens_{ft}"] = np.array(btlens[f])
        for n in cols + ["y", "md_since"]:
            out[f"bt{n}_{ft}"] = np.concatenate(bt[f][n])
    outf = ROOT / f"research/ad03_cache_{tag}_k{k}_r{rep}.npz"
    np.savez_compressed(outf, **out)
    pd.DataFrame(meta).to_csv(str(outf).replace(".npz", "_meta.csv"), index=False)
    y = out["y"].astype(np.float64)
    print(f"\nguardado {outf}  pozos={len(lens)} puntos={sum(lens)}  {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/len(lens):.2f} s/pozo)")
    for n in NAMES:
        print(f"  RMSE {n}: {np.sqrt(np.mean((y - out[n].astype(np.float64))**2)):7.3f}")
    for w in (0.2, 0.25, 0.3):
        mix = w * out["S"].astype(np.float64) + (1 - w) * out["P"].astype(np.float64)
        print(f"  RMSE {w:.2f}S+{1-w:.2f}P: {np.sqrt(np.mean((y-mix)**2)):7.3f}")


if __name__ == "__main__":
    main()
