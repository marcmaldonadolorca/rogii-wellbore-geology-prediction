"""V4-TECHO: oraculos, desglose de error, varianza y sesgo LB del ensemble actual.

El cache maestro orc01 (770 pozos, seed0) tiene {geom, ancc, pfz, beam} vigentes
pero su columna `surf` es la superficie VIEJA (isotropa, 24.3 a k=60). Aqui:

  1) `cache`  : recalcula SOLO la superficie con la config ganadora
                (SurfaceField aniso=16, k=24, BUDA, offset cola 500) y de paso
                aniso=5, con LOWO exacto, para los 770 pozos del cache.
                Salida: v4_techo_surf16.npz  (~0.9 s/pozo).
  2) `analiza`: oraculos (seleccion/blend NNLS/por punto/offset+pendiente),
                desglose del MSE del blend 0.25*surf16+0.75*ancc, varianza en
                5 subconjuntos disjuntos + bootstrap pareado, sesgo LB,
                y multiseed del PF con las 5 replicas de ad02_pf_reps.npz.
                Salida: v4_techo_resultados.csv + informe por stdout.

Uso:  .venv/bin/python research/v4_techo.py cache
      .venv/bin/python research/v4_techo.py analiza
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "research"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from cv import TEST_COLS, LB_PROXY_NPRED, load_well  # noqa: E402
import model as M  # noqa: E402
from orclib import Cache, rmse, subset  # noqa: E402

SURF_NPZ = HERE / "v4_techo_surf16.npz"
OUT_CSV = HERE / "v4_techo_resultados.csv"
W_SURF = 0.25                     # blend de referencia 0.25*surf + 0.75*ancc


# ─────────────────────────────── etapa 1: cache ──────────────────────────────

def build_cache():
    c = Cache()
    fields = {"surf16": M.SurfaceField(aniso=16.0), "surf5": M.SurfaceField(aniso=5.0)}
    print(f"campos listos ({len(fields['surf16'].s)} pts)", flush=True)
    store = {k: [] for k in ("surf16", "surf5", "nn16")}
    t0 = time.time()
    for j, (wid, cut_meta) in enumerate(zip(c.meta.well, c.meta.cut)):
        df, tw, cut = load_well(wid)
        assert cut == cut_meta, f"{wid}: cut {cut} != meta {cut_meta}"
        df_h = df[TEST_COLS].copy()
        p16, nn16 = M._prior(df_h, fields["surf16"], wid, cut)
        p5, _ = M._prior(df_h, fields["surf5"], wid, cut)
        store["surf16"].append(p16[cut:])
        store["surf5"].append(p5[cut:])
        store["nn16"].append(nn16[cut:])
        if (j + 1) % 100 == 0:
            print(f"  {j+1}/{len(c.meta)}  {time.time()-t0:.0f}s", flush=True)
    dt = time.time() - t0
    outd = {k: np.concatenate(v).astype(np.float32) for k, v in store.items()}
    outd["lens"] = c.lens
    np.savez_compressed(SURF_NPZ, **outd)
    print(f"guardado {SURF_NPZ}  {dt:.0f}s = {dt/len(c.meta)*1000:.0f} ms/pozo")
    y = c.d["y"]
    for k in ("surf16", "surf5"):
        print(f"  RMSE {k}: {np.sqrt(np.mean((y - outd[k].astype(float))**2)):7.3f}")


# ─────────────────────────────── etapa 2: analisis ───────────────────────────

def pooled(sse, n):
    return float(np.sqrt(np.sum(sse) / np.sum(n)))


def per_well(c, idxs, pred):
    """(sse, n) por pozo para los indices de pozo dados."""
    y = c.d["y"]
    sse = np.empty(len(idxs))
    for j, i in enumerate(idxs):
        s = c.sl(i)
        sse[j] = float(((y[s] - pred[s]) ** 2).sum())
    return sse, c.lens[idxs].astype(float)


def carga():
    c = Cache()
    z = np.load(SURF_NPZ)
    assert (z["lens"] == c.lens).all(), "surf16 desalineado con orc01"
    for k in ("surf16", "surf5", "nn16"):
        c.d[k] = z[k].astype(np.float64)
    c.d["blend16"] = W_SURF * c.d["surf16"] + (1 - W_SURF) * c.d["ancc"]
    c.d["blend5"] = W_SURF * c.d["surf5"] + (1 - W_SURF) * c.d["ancc"]
    return c


def idxs_de(c, ids):
    if ids is None:
        return np.arange(len(c.lens))
    return np.array(sorted(c.widx[w] for w in ids if w in c.widx))


VAR5 = ("geom", "surf16", "ancc", "pfz", "blend16")   # el conjunto de la mision
BASE4 = ("geom", "surf16", "ancc", "pfz")             # base linealmente independiente


def oraculos(c, ids, nombre, rows):
    from scipy.optimize import nnls
    y = c.d["y"]
    idxs = idxs_de(c, ids)
    m = c.mask(ids)
    npt = c.lens[idxs].astype(float)

    def add(k, v, extra=""):
        rows.append({"bloque": "oraculo", "set": nombre, "variante": k,
                     "rmse": v, "nota": extra})
        print(f"  {k:36s} {v:8.3f}  {extra}")

    print(f"\n=== ORACULOS {nombre} ({len(idxs)} pozos, {int(m.sum())} puntos) ===")
    for p in VAR5 + ("blend5", "beam"):
        add(p, rmse(y, c.d[p], m))

    # a) seleccion por pozo con TVT real
    S = np.column_stack([per_well(c, idxs, c.d[p])[0] for p in VAR5])
    reparto = dict(zip(VAR5, np.bincount(S.argmin(1), minlength=len(VAR5))))
    add("ORACULO a) seleccion por pozo", pooled(S.min(1), npt), f"reparto={reparto}")

    # b) blend por pozo NNLS (y LS libre como referencia)
    sse_nn, sse_ls = np.empty(len(idxs)), np.empty(len(idxs))
    for j, i in enumerate(idxs):
        s = c.sl(i)
        A = np.column_stack([c.d[p][s] for p in BASE4])
        t = y[s]
        w = nnls(A, t)[0]
        sse_nn[j] = float(((A @ w - t) ** 2).sum())
        wl = np.linalg.lstsq(A, t, rcond=None)[0]
        sse_ls[j] = float(((A @ wl - t) ** 2).sum())
    add("ORACULO b) blend NNLS por pozo", pooled(sse_nn, npt))
    add("  (referencia: LS libre por pozo)", pooled(sse_ls, npt))

    # c) blend POR PUNTO: cota inferior de cualquier pesado puntual convexo
    P4 = np.column_stack([c.d[p] for p in BASE4])
    lo4, hi4 = P4.min(1), P4.max(1)
    e_hull4 = np.clip(y, lo4, hi4) - y
    add("ORACULO c) por punto hull 4", float(np.sqrt(np.mean(e_hull4[m] ** 2))))
    P3 = np.column_stack([c.d[p] for p in ("surf16", "ancc", "pfz")])
    e_hull3 = np.clip(y, P3.min(1), P3.max(1)) - y
    add("  (hull sin geom: surf16/ancc/pfz)", float(np.sqrt(np.mean(e_hull3[m] ** 2))))
    E = np.abs(np.column_stack([c.d[p] - y for p in VAR5]))
    add("  (seleccion por punto, 5 vars)", float(np.sqrt(np.mean(E.min(1)[m] ** 2))))

    # d) offset por pozo sobre el mejor blend; y offset+pendiente
    base = c.d["blend16"]
    s_off, s_lin = np.empty(len(idxs)), np.empty(len(idxs))
    offs = np.empty(len(idxs))
    for j, i in enumerate(idxs):
        s = c.sl(i)
        r = y[s] - base[s]
        offs[j] = r.mean()
        s_off[j] = float(((r - r.mean()) ** 2).sum())
        A = np.column_stack([np.ones(c.lens[i]), c.d["md_since"][s]])
        co = np.linalg.lstsq(A, r, rcond=None)[0]
        s_lin[j] = float(((r - A @ co) ** 2).sum())
    add("ORACULO d) offset/pozo sobre blend16", pooled(s_off, npt),
        f"|offset| mediana={np.median(np.abs(offs)):.2f} ft")
    add("ORACULO d) offset+pendiente", pooled(s_lin, npt))


def desglose(c, ids, nombre, rows):
    y = c.d["y"]
    m = c.mask(ids)
    e2 = (y - c.d["blend16"]) ** 2
    tot = e2[m].sum()
    print(f"\n=== DESGLOSE blend16 {nombre} (RMSE {np.sqrt(e2[m].mean()):.3f}) ===")

    def tabla(x, edges, etiqueta, fmt="{:.0f}"):
        print(f"  por {etiqueta}:")
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = m & (x >= lo) & (x < hi)
            n = int(sel.sum())
            if n == 0:
                continue
            fr = e2[sel].sum() / tot
            r = float(np.sqrt(e2[sel].mean()))
            rows.append({"bloque": "desglose", "set": nombre,
                         "variante": f"{etiqueta} [{fmt.format(lo)},{fmt.format(hi)})",
                         "rmse": r, "nota": f"n={n} %MSE={100*fr:.1f}"})
            print(f"    [{fmt.format(lo):>6},{fmt.format(hi):>6})  n={n:7d}  "
                  f"rmse={r:7.2f}  %SSE={100*fr:5.1f}")

    tabla(c.d["md_since"], np.array([0, 1000, 2000, 3000, 4000, 6000, 8000, 1e9]),
          "md_since")
    q = np.quantile(c.d["nn16"][m], [0, .25, .5, .75, .9, 1.0])
    q[-1] += 1
    tabla(c.d["nn16"], q, "nn16(metrica aniso)")
    npred_pt = c.lens[c.wpt].astype(float)
    qq = np.quantile(c.meta.n_pred, [0, .2, .4, .6, .8, 1.0]).astype(float)
    qq[-1] += 1
    tabla(npred_pt, qq, "n_pred del pozo")

    # top-20 pozos por SSE aportado
    idxs = idxs_de(c, ids)
    sse, n = per_well(c, idxs, c.d["blend16"])
    orden = np.argsort(-sse)
    print("  top-20 pozos por SSE (blend16):")
    top_sse = 0.0
    for r_ in orden[:20]:
        i = idxs[r_]
        top_sse += sse[r_]
        print(f"    {c.wells[i]}  n={int(n[r_]):6d}  rmse={np.sqrt(sse[r_]/n[r_]):7.2f}  "
              f"%SSE={100*sse[r_]/sse.sum():5.2f}  nn_med={c.meta.nn_med.iloc[i]:6.0f}")
    fr10 = sse[orden[:10]].sum() / sse.sum()
    fr20 = top_sse / sse.sum()
    rows.append({"bloque": "desglose", "set": nombre, "variante": "concentracion",
                 "rmse": float(np.sqrt(sse.sum() / n.sum())),
                 "nota": f"top10={100*fr10:.1f}%SSE top20={100*fr20:.1f}%SSE de {len(idxs)} pozos"})
    print(f"  concentracion: top-10 pozos = {100*fr10:.1f}% del SSE, "
          f"top-20 = {100*fr20:.1f}%")


def varianza(c, rows):
    print("\n=== VARIANZA ===")
    idxs = np.arange(len(c.lens))
    # 5 subconjuntos disjuntos estratificados por n_pred (reparto round-robin)
    orden = np.argsort(c.meta.n_pred.values)
    grupos = [orden[g::5] for g in range(5)]
    for nombre_v in ("blend16", "blend5", "ancc", "surf16"):
        sse, n = per_well(c, idxs, c.d[nombre_v])
        rs = [pooled(sse[g], n[g]) for g in grupos]
        rows.append({"bloque": "varianza", "set": "5x154", "variante": nombre_v,
                     "rmse": float(np.mean(rs)),
                     "nota": f"std={np.std(rs):.3f} grupos={np.round(rs,2).tolist()}"})
        print(f"  {nombre_v:8s} 5 grupos de 154: media={np.mean(rs):6.3f}  "
              f"std={np.std(rs):5.3f}  {np.round(rs, 2)}")

    # bootstrap PAREADO: sd de la diferencia de RMSE entre variantes cercanas
    rng = np.random.default_rng(0)
    B = 2000
    pares = [("blend16", "blend5"), ("blend16", "ancc"), ("surf16", "surf5")]
    sses = {v: per_well(c, idxs, c.d[v])[0] for v in
            {p for par in pares for p in par}}
    n = c.lens.astype(float)
    for a, b in pares:
        for kk in (60, 150, 770):
            sel = rng.integers(0, len(idxs), size=(B, kk))
            ra = np.sqrt(sses[a][sel].sum(1) / n[sel].sum(1))
            rb = np.sqrt(sses[b][sel].sum(1) / n[sel].sum(1))
            d = ra - rb
            if kk == 60:
                rows.append({"bloque": "varianza", "set": f"boot k={kk}",
                             "variante": f"{a}-{b}", "rmse": float(np.mean(d)),
                             "nota": f"sd_pareada={np.std(d):.3f} => senal si |d|>{2*np.std(d):.2f}"})
            print(f"  boot k={kk:3d}  {a}-{b}: media={np.mean(d):+7.3f}  "
                  f"sd={np.std(d):.3f}  (senal 2sd: {2*np.std(d):.2f} ft)")


def multiseed(c, rows):
    f = HERE / "ad02_pf_reps.npz"
    if not f.exists():
        print("\n(sin ad02_pf_reps.npz: multiseed omitido)")
        return
    z = np.load(f)
    y2, P = z["y"].astype(float), z["P"].astype(float)
    ids = sorted(subset(60))
    idxs = idxs_de(c, ids)
    assert (z["lens"] == c.lens[idxs]).all(), "ad02 desalineado"
    m = c.mask(ids)
    assert np.allclose(y2, c.d["y"][m], atol=1e-3)
    s16 = c.d["surf16"][m]
    print("\n=== MULTISEED PF (5 replicas, k=60) ===")
    r1 = [float(np.sqrt(np.mean((y2 - p) ** 2))) for p in P]
    print(f"  pf solo por replica: {np.round(r1,3)}  media={np.mean(r1):.3f} sd={np.std(r1):.3f}")
    rb1 = [float(np.sqrt(np.mean((y2 - (W_SURF*s16 + (1-W_SURF)*p)) ** 2))) for p in P]
    pm = P.mean(0)
    rbm = float(np.sqrt(np.mean((y2 - (W_SURF*s16 + (1-W_SURF)*pm)) ** 2)))
    print(f"  blend16 con 1 seed : {np.round(rb1,3)}  media={np.mean(rb1):.3f} sd={np.std(rb1):.3f}")
    print(f"  blend16 con MEDIA de 5 seeds: {rbm:.3f}  "
          f"(gana {np.mean(rb1)-rbm:+.3f} ft sobre la media a 1 seed)")
    rows.append({"bloque": "multiseed", "set": "k=60", "variante": "blend16 media5seeds",
                 "rmse": rbm, "nota": f"1seed media={np.mean(rb1):.3f} sd={np.std(rb1):.3f}"})


def sesgo_lb(c, rows):
    print("\n=== SESGO LB (770): rmse vs rmse_lb_proxy (n_pred>=%d) ===" % LB_PROXY_NPRED)
    idxs = np.arange(len(c.lens))
    largo = c.meta.n_pred.values >= LB_PROXY_NPRED
    for v in ("blend16", "blend5", "ancc", "surf16", "pfz", "geom"):
        sse, n = per_well(c, idxs, c.d[v])
        r_all = pooled(sse, n)
        r_px = pooled(sse[largo], n[largo])
        rows.append({"bloque": "sesgo_lb", "set": "770", "variante": v, "rmse": r_all,
                     "nota": f"proxy={r_px:.3f} ratio={r_px/r_all:.3f} ({int(largo.sum())} pozos)"})
        print(f"  {v:8s} rmse={r_all:7.3f}  proxy={r_px:7.3f}  ratio={r_px/r_all:.3f}")


def analiza():
    c = carga()
    rows = []
    for k in (60, 150, None):
        oraculos(c, subset(k) if k else None, f"k={k}" if k else "770", rows)
    desglose(c, None, "770", rows)
    desglose(c, subset(60), "k=60", rows)
    varianza(c, rows)
    multiseed(c, rows)
    sesgo_lb(c, rows)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nguardado {OUT_CSV}")


def multiseed150(k=150, reps=5):
    """Confirmacion k=150 de la mejor variante: 0.25*surf16 + 0.75*media(PF reps).

    Como ad02: sin reseed explicito, cada rep avanza el RNG de numba.
    """
    import pf_publico as P
    c = carga()
    ids = sorted(subset(k))
    idxs = idxs_de(c, ids)
    m = c.mask(ids)
    y, s16 = c.d["y"][m], c.d["surf16"][m]
    t0 = time.time()
    reps_p = []
    for r in range(reps):
        chunks = []
        for wid in ids:
            df, tw, cut = load_well(wid)
            if cut < 20 or cut >= len(df):
                continue
            df_h = df[TEST_COLS].copy()
            t, g = P._tw(tw)
            p, _ = P.run_pf_ancc(df_h, t, g)
            chunks.append(np.asarray(p, float))
        reps_p.append(np.concatenate(chunks))
        print(f"  rep {r}: pf={np.sqrt(np.mean((y-reps_p[-1])**2)):.3f}  "
              f"blend={np.sqrt(np.mean((y-(W_SURF*s16+(1-W_SURF)*reps_p[-1]))**2)):.3f}  "
              f"{time.time()-t0:.0f}s", flush=True)
    pm = np.mean(reps_p, 0)
    r1 = [float(np.sqrt(np.mean((y - (W_SURF*s16 + (1-W_SURF)*p)) ** 2))) for p in reps_p]
    rbm = float(np.sqrt(np.mean((y - (W_SURF*s16 + (1-W_SURF)*pm)) ** 2)))
    dt = time.time() - t0
    print(f"k={k}: blend16 1seed media={np.mean(r1):.3f} sd={np.std(r1):.3f}  "
          f"blend16 MEDIA{reps}seeds={rbm:.3f}  "
          f"({dt/(len(ids)*reps)*1000:.0f} ms/pozo/seed)")


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "analiza"
    if modo == "cache":
        build_cache()
    elif modo == "multiseed150":
        multiseed150()
    else:
        analiza()
