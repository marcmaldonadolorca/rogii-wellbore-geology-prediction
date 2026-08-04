"""v9 pesos: calibracion de pesos POR POZO via backtest del prefijo enmascarado.

La bolsa: oraculo NNLS por pozo = 4.24 vs v8 = 8.949 (k150). La familia publica
probo que backtestear candidatos sobre la cola enmascarada del prefijo y mover
la prediccion con alpha acotado funciona ("gold calibration"). Nuestro offset
por pozo fracaso (corr -0.05); los PESOS son la pregunta distinta: no que valor
corregir sino a quien creer (superficie vs PF).

Todo pareado contra v8 "s2d multiplic." (k150=8.949, k60=9.699) reconstruido
desde los MISMOS artefactos: v6_pesos_cache.npz (surf/pf S64/geom/nn/md post-PS
+ backtest cut 0.65 con PF S=8) y v8_s2d_tablas.npz (w adaptativo 2D).
El PF de backtest S=16 se RECUPERA de research/v6_offset_cache/bt_*.npz:
    resid = y_bt - (0.45*surf_bt + 0.55*pf16)  =>  pf16 = (y_bt-resid-0.45*s)/0.55
(surf_bt identico entre v6_offset y v6_pesos: mismo campo, mismo corte 0.65,
misma cola de calibracion 500).

Fases:
  check : reproduce v8 desde el cache (tiene que dar 8.949/9.699) y valida
          la recuperacion del pf16 contra el pf8 del cache pesos.
  bt50  : backtest fresco al corte 0.5 (PF S=8, ~2-4 s/pozo, 196 pozos eval),
          restartable en research/v9_pesos_bt50/.
  eval  : diagnostico de señal + barrido de variantes (lam, T, clip) pareado
          k60/k150 + estratos nn. Salida research/v9_pesos_resultados.csv.

Uso:  nice -n 10 .venv/bin/python research/v9_pesos_backtest.py {check|bt50|eval}
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))

from cv import TEST_COLS, load_well, well_ids, _select, SEED   # noqa: E402
import v7_wadapt as v7                                          # noqa: E402
import v8_sig2d_wadapt as v8                                    # noqa: E402

THETA_OPT = 2.278489
W_CAP = (0.05, 0.9)
PESOS_NPZ = R / "v6_pesos_cache.npz"
OFFSET_DIR = R / "v6_offset_cache"
TABLAS = R / "v8_s2d_tablas.npz"
BT50_DIR = R / "v9_pesos_bt50"
OUT_CSV = R / "v9_pesos_resultados.csv"

# umbrales de ruido pareado (misma seleccion, mismo cache PF)
NOISE = {60: 0.56, 150: 0.35}


# ---------------------------------------------------------------- infra
def eval_sets():
    ids = well_ids()
    k60 = _select(ids, 60, SEED)
    k150 = _select(ids, 150, SEED)
    return sorted(set(k60) | set(k150)), set(k60), set(k150)


_Z = None


def zcache():
    global _Z
    if _Z is None:
        _Z = np.load(PESOS_NPZ)
    return _Z


def arr(wid, name):
    return zcache()[f"{wid}_{name}"].astype(np.float64)


_WFN = None


def w_adapt_fn():
    """w_s por punto del blend v8 ganador: ss = s_mult 2D, sp = sig_p_1d."""
    global _WFN
    if _WFN is None:
        z = np.load(TABLAS)
        ss2 = v8._interp2d(z["nn_bins"], z["md_bins"], z["s_mult"])
        sig_p, md_bins = z["sig_p_1d"], z["md_bins"]

        def w(nn, md):
            ss = ss2(nn, md)
            sp = v7._interp_curve(md, md_bins, sig_p)
            return np.clip(sp ** 2 / (ss ** 2 + sp ** 2), *W_CAP)

        _WFN = w
    return _WFN


def pf16_bt(wid, y_bt, surf_bt):
    """PF S=16 de backtest recuperado del residuo del blend fijo (v6_offset)."""
    f = OFFSET_DIR / f"bt_{wid}.npz"
    if not f.exists():
        return None
    resid = np.load(f)["resid"].astype(np.float64)
    if len(resid) != len(y_bt):
        return None
    return (y_bt - resid - 0.45 * surf_bt) / 0.55


def nnls_w2(S, P, y):
    """NNLS sobre 2 candidatos -> w_s escalar en [0,1] (None si degenerado)."""
    w, _ = nnls(np.column_stack([S, P]), y)
    s = w.sum()
    return None if s <= 1e-9 else float(w[0] / s)


def nnls_w3(S, P, G, y):
    w, _ = nnls(np.column_stack([S, P, G]), y)
    s = w.sum()
    return None if s <= 1e-9 else w / s


def rmse(a):
    return float(np.sqrt(np.mean(a ** 2)))


def pooled(sse_n):
    sse = sum(s for s, _ in sse_n)
    n = sum(m for _, m in sse_n)
    return float(np.sqrt(sse / n))


# ---------------------------------------------------------------- check
def check():
    ev, k60, k150 = eval_sets()
    wfn = w_adapt_fn()
    rows = []
    for wid in ev:
        y, s, p = arr(wid, "y"), arr(wid, "surf"), arr(wid, "pf")
        nn, md = arr(wid, "nn"), arr(wid, "md")
        w = wfn(nn, md)
        base = w * s + (1 - w) * p
        rows.append((wid, float(((y - base) ** 2).sum()), len(y)))
    for k, ks in ((60, k60), (150, k150)):
        r = pooled([(ss, n) for wid, ss, n in rows if wid in ks])
        print(f"v8 reconstruido k{k}: {r:.3f}  (esperado {'9.699' if k == 60 else '8.949'})")

    # validacion de la recuperacion pf16
    difs, cors = [], []
    for wid in ev[:50]:
        y_bt, s_bt, p8 = arr(wid, "y_bt"), arr(wid, "surf_bt"), arr(wid, "pf_bt")
        p16 = pf16_bt(wid, y_bt, s_bt)
        if p16 is None:
            continue
        difs.append(rmse(p16 - p8))
        if np.std(p16) > 0 and np.std(p8) > 0:
            cors.append(float(np.corrcoef(p16, p8)[0, 1]))
    print(f"pf16 recuperado vs pf8 (50 pozos): rms dif mediana {np.median(difs):.2f} ft, "
          f"corr mediana {np.median(cors):.3f}, n={len(difs)}")


# ---------------------------------------------------------------- bt50
def bt50():
    from model import SurfaceField
    from ban_lib import run_pf_ancc_multi
    from pf_publico import _tw

    BT50_DIR.mkdir(exist_ok=True)
    ev, _, _ = eval_sets()
    todo = [w for w in ev if not (BT50_DIR / f"bt50_{w}.npz").exists()]
    print(f"bt50: {len(todo)} pozos por hacer", flush=True)
    if not todo:
        return
    field = SurfaceField(aniso=16.0, theta=THETA_OPT)
    t0 = time.time()
    for i, wid in enumerate(todo):
        f = BT50_DIR / f"bt50_{wid}.npz"
        df, tw, cut = load_well(wid)
        c50 = int(round(0.5 * cut))
        if cut < 20 or cut >= len(df) or c50 < 20 or cut - c50 < 30:
            np.savez_compressed(f, ok=np.array(0))
            continue
        df_h = df[TEST_COLS].copy()
        s_all, nn_all = field.interp(df_h.X.values, df_h.Y.values, exclude=wid, k=24)
        z = df_h.Z.values
        tvti = df_h.TVT_input.values
        lo = max(0, c50 - 500)
        c = np.median(tvti[lo:c50] + z[lo:c50] - s_all[lo:c50])
        surf_bt = (s_all - z + c)[c50:cut]
        df_bt = df_h.iloc[:cut].copy()
        df_bt.loc[df_bt.index[c50:], "TVT_input"] = np.nan
        t, g = _tw(tw)
        pts, _ = run_pf_ancc_multi(df_bt, t, g,
                                   seeds=np.arange(1, 9, dtype=np.int64), N=600)
        md = df_h.MD.values
        np.savez_compressed(
            f, ok=np.array(1), y=tvti[c50:cut], s=surf_bt,
            p=pts.mean(0).astype(np.float64),
            md=md[c50:cut] - md[c50 - 1], nn=nn_all[c50:cut])
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(todo)}  {el/(i+1):.1f}s/pozo  "
                  f"quedan ~{el/(i+1)*(len(todo)-i-1)/60:.0f} min", flush=True)
    print("DONE_BT50", flush=True)


# ---------------------------------------------------------------- eval
def _well_features(wid, wfn):
    """Arrays post-PS + stats de backtest (65 y, si existe, 50) de un pozo."""
    d = dict(y=arr(wid, "y"), s=arr(wid, "surf"), p=arr(wid, "pf"),
             g=arr(wid, "geom"), nn=arr(wid, "nn"), md=arr(wid, "md"))
    d["w_pt"] = wfn(d["nn"], d["md"])
    d["base"] = d["w_pt"] * d["s"] + (1 - d["w_pt"]) * d["p"]

    cuts = []
    # ---- corte 0.65 (cache v6_pesos + pf16 recuperado de v6_offset)
    y_bt, s_bt = arr(wid, "y_bt"), arr(wid, "surf_bt")
    if len(y_bt) >= 30:
        p16 = pf16_bt(wid, y_bt, s_bt)
        p_bt = p16 if p16 is not None else arr(wid, "pf_bt")
        g_bt = arr(wid, "geom_bt")
        cuts.append(dict(tag=65, y=y_bt, s=s_bt, p=p_bt, g=g_bt,
                         nn=arr(wid, "nn_bt"), md=arr(wid, "md_bt")))
    # ---- corte 0.5 (fresco, si esta)
    f50 = BT50_DIR / f"bt50_{wid}.npz"
    if f50.exists():
        z = np.load(f50)
        if int(z["ok"]) == 1:
            cuts.append(dict(tag=50, y=z["y"].astype(float), s=z["s"].astype(float),
                             p=z["p"].astype(float), g=None,
                             nn=z["nn"].astype(float), md=z["md"].astype(float)))

    for c in cuts:
        c["rs"], c["rp"] = rmse(c["y"] - c["s"]), rmse(c["y"] - c["p"])
        w_pt_bt = wfn(c["nn"], c["md"])
        c["rb"] = rmse(c["y"] - (w_pt_bt * c["s"] + (1 - w_pt_bt) * c["p"]))
        c["w2"] = nnls_w2(c["s"], c["p"], c["y"])
        c["w3"] = None if c["g"] is None else nnls_w3(c["s"], c["p"], c["g"], c["y"])
    d["cuts"] = cuts
    return d


def _scalar_weight_pred(d, w_well):
    return w_well * d["s"] + (1 - w_well) * d["p"]


def eval_phase():
    ev, k60, k150 = eval_sets()
    wfn = w_adapt_fn()
    print("cargando features por pozo...", flush=True)
    W = {wid: _well_features(wid, wfn) for wid in ev}
    n50 = sum(any(c["tag"] == 50 for c in d["cuts"]) for d in W.values())
    print(f"pozos={len(W)}  con bt65={sum(bool(d['cuts']) for d in W.values())}  con bt50={n50}")

    # ------------- diagnostico: ¿el backtest predice a quien creer? -------------
    rows = []
    for wid, d in W.items():
        c65 = next((c for c in d["cuts"] if c["tag"] == 65), None)
        if c65 is None:
            continue
        rs_post, rp_post = rmse(d["y"] - d["s"]), rmse(d["y"] - d["p"])
        w_or = nnls_w2(d["s"], d["p"], d["y"])           # oraculo (usa y real)
        rows.append(dict(wid=wid, dr_bt=c65["rs"] - c65["rp"], dr_post=rs_post - rp_post,
                         w_bt=c65["w2"], w_or=w_or,
                         nn_med=float(np.median(d["nn"]))))
    dg = pd.DataFrame(rows).dropna(subset=["dr_bt", "dr_post"])
    pear = float(np.corrcoef(dg.dr_bt, dg.dr_post)[0, 1])
    spear = float(spearmanr(dg.dr_bt, dg.dr_post).statistic)
    sig = dg[np.abs(dg.dr_post) > 2.0]
    acc = float(np.mean(np.sign(sig.dr_bt) == np.sign(sig.dr_post)))
    dgw = dg.dropna(subset=["w_bt", "w_or"])
    pw = float(np.corrcoef(dgw.w_bt, dgw.w_or)[0, 1])
    sw = float(spearmanr(dgw.w_bt, dgw.w_or).statistic)
    print("\n== DIAGNOSTICO DE SEÑAL (corte 0.65, n=%d) ==" % len(dg))
    print(f"  corr(rmse_s-rmse_p bt, post): pearson {pear:+.3f}  spearman {spear:+.3f}")
    print(f"  acierto de signo (|dif post|>2ft, n={len(sig)}): {acc:.2f}")
    print(f"  corr(w_nnls bt, w_nnls oraculo): pearson {pw:+.3f}  spearman {sw:+.3f}")

    # tambien con el doble corte
    rows2 = []
    for wid, d in W.items():
        if len(d["cuts"]) < 2:
            continue
        dr = np.mean([c["rs"] - c["rp"] for c in d["cuts"]])
        rs_post, rp_post = rmse(d["y"] - d["s"]), rmse(d["y"] - d["p"])
        rows2.append((dr, rs_post - rp_post))
    if len(rows2) > 20:
        a = np.array(rows2)
        print(f"  doble-corte (n={len(a)}): pearson {np.corrcoef(a[:,0],a[:,1])[0,1]:+.3f}  "
              f"spearman {spearmanr(a[:,0],a[:,1]).statistic:+.3f}")

    # ------------- oraculos de referencia (cota, usan y real) -------------
    for k, ks in ((60, k60), (150, k150)):
        sse_or = []
        for wid, d in W.items():
            if wid not in ks:
                continue
            w3 = nnls_w3(d["s"], d["p"], d["g"], d["y"])
            pr = d["base"] if w3 is None else w3[0] * d["s"] + w3[1] * d["p"] + w3[2] * d["g"]
            sse_or.append((float(((d["y"] - pr) ** 2).sum()), len(d["y"])))
        print(f"  oraculo NNLS3 por pozo k{k}: {pooled(sse_or):.3f} (cota)")

    # ------------- variantes -------------
    def cand_weight(d, scheme, T=1.0, gate=0.0):
        """w_s escalar por pozo segun el backtest; None => sin señal (usa base)."""
        cuts = d["cuts"]
        if not cuts:
            return None
        if "_dc" in scheme:
            use = cuts                      # doble corte promediado
        else:
            use = [c for c in cuts if c["tag"] == 65]
        if not use:
            return None
        if gate > 0:                        # solo corregir con evidencia clara
            if abs(np.mean([c["rs"] for c in use]) -
                   np.mean([c["rp"] for c in use])) < gate:
                return None
        if scheme.startswith("nnls"):
            ws = [c["w2"] for c in use if c["w2"] is not None]
            return None if not ws else float(np.mean(ws))
        if scheme.startswith("soft"):
            rs = np.mean([c["rs"] for c in use])
            rp = np.mean([c["rp"] for c in use])
            e = np.exp(-np.array([rs, rp]) / T)
            return float(e[0] / e.sum())
        if scheme.startswith("ivw"):
            rs = np.mean([c["rs"] for c in use])
            rp = np.mean([c["rp"] for c in use])
            return float(rp ** 2 / (rs ** 2 + rp ** 2 + 1e-12))
        raise ValueError(scheme)

    LAMS = (0.15, 0.3, 0.5, 0.7, 1.0)
    CLIPS = (5.0, 10.0, 20.0, np.inf)
    TS = (0.5, 1.0, 2.0, 4.0)

    schemes = [("nnls65", {}), ("ivw65", {}), ("nnls_dc", {}), ("ivw_dc", {})]
    schemes += [(f"soft65 T={t}", dict(T=t)) for t in TS]
    schemes += [(f"soft_dc T={t}", dict(T=t)) for t in TS]
    schemes += [(f"nnls_dc g={g}", dict(gate=g)) for g in (1.0, 2.0, 4.0)]

    # precomputa predicciones candidatas por pozo y esquema
    cand_pred = {}
    for name, kw in schemes:
        sch = "soft_dc" if name.startswith("soft_dc") else \
              "soft65" if name.startswith("soft65") else name.split()[0]
        for wid, d in W.items():
            w_well = cand_weight(d, sch, **kw)
            cand_pred[(name, wid)] = None if w_well is None else _scalar_weight_pred(d, w_well)

    # nnls3 (con geometrico), solo corte 65
    schemes.append(("nnls3_65", {}))
    for wid, d in W.items():
        c65 = next((c for c in d["cuts"] if c["tag"] == 65), None)
        w3 = None if c65 is None else c65["w3"]
        cand_pred[("nnls3_65", wid)] = None if w3 is None else \
            w3[0] * d["s"] + w3[1] * d["p"] + w3[2] * d["g"]

    # factores de sigma por pozo (multiplican las sigmas 2D; shrink via lam)
    z8 = np.load(TABLAS)
    ss2 = v8._interp2d(z8["nn_bins"], z8["md_bins"], z8["s_mult"])
    for wid, d in W.items():
        c65 = next((c for c in d["cuts"] if c["tag"] == 65), None)
        if c65 is None:
            d["sigfac"] = None
            continue
        ss_exp = float(np.sqrt(np.mean(ss2(c65["nn"], c65["md"]) ** 2)))
        sp_exp = float(np.sqrt(np.mean(v7._interp_curve(c65["md"], z8["md_bins"],
                                                        z8["sig_p_1d"]) ** 2)))
        d["sigfac"] = (c65["rs"] / max(ss_exp, 1e-6), c65["rp"] / max(sp_exp, 1e-6))

    def sigfac_pred(d, lam):
        if d["sigfac"] is None:
            return d["base"]
        fs, fp = d["sigfac"]
        ss = ss2(d["nn"], d["md"]) * fs ** lam
        sp = v7._interp_curve(d["md"], z8["md_bins"], z8["sig_p_1d"]) * fp ** lam
        w = np.clip(sp ** 2 / (ss ** 2 + sp ** 2), *W_CAP)
        return w * d["s"] + (1 - w) * d["p"]

    # ------------- barrido pareado -------------
    res = []
    base_r = {}
    for k, ks in ((60, k60), (150, k150)):
        base_r[k] = pooled([(float(((d["y"] - d["base"]) ** 2).sum()), len(d["y"]))
                            for wid, d in W.items() if wid in ks])
    print(f"\ncontrol v8: k60={base_r[60]:.3f}  k150={base_r[150]:.3f}  "
          f"(umbral ruido {NOISE[60]}/{NOISE[150]})")

    def score(pred_of):
        out = {}
        for k, ks in ((60, k60), (150, k150)):
            out[k] = pooled([(float(((d["y"] - pred_of(wid, d)) ** 2).sum()), len(d["y"]))
                             for wid, d in W.items() if wid in ks])
        return out

    all_names = [n for n, _ in schemes]
    for name in all_names:
        for lam in LAMS:
            for D in CLIPS:
                def pred_of(wid, d, name=name, lam=lam, D=D):
                    c = cand_pred.get((name, wid))
                    if c is None:
                        return d["base"]
                    return d["base"] + np.clip(lam * (c - d["base"]), -D, D)
                r = score(pred_of)
                res.append(dict(variante=name, lam=lam, clip=D,
                                k60=r[60], k150=r[150],
                                d60=r[60] - base_r[60], d150=r[150] - base_r[150]))
    for lam in LAMS:
        r = score(lambda wid, d, lam=lam: sigfac_pred(d, lam))
        res.append(dict(variante="sigfac65", lam=lam, clip=np.inf,
                        k60=r[60], k150=r[150],
                        d60=r[60] - base_r[60], d150=r[150] - base_r[150]))

    df = pd.DataFrame(res).sort_values("k150")
    df.to_csv(OUT_CSV, index=False)
    print(f"\n== TOP 15 por k150 (de {len(df)} combos) ==")
    print(df.head(15).to_string(index=False,
          formatters={c: "{:.3f}".format for c in ("k60", "k150", "d60", "d150")}))
    print("\n== mejores por variante (k150) ==")
    print(df.groupby("variante").first().sort_values("k150").to_string(
          formatters={c: "{:.3f}".format for c in ("k60", "k150", "d60", "d150")}))

    # ------------- bootstrap sobre pozos de combos elegidos a priori -------------
    # elegidos ANTES de mirar la tabla: nnls con shrinkage fuerte y clip corto
    print("\n== bootstrap pareado sobre pozos (2000 resamples) ==")
    for name_b, lam_b, D_b in (("nnls_dc", 0.3, 5.0), ("nnls_dc", 0.15, 5.0)):
        for k, ks in ((60, k60), (150, k150)):
            wl = [wid for wid in W if wid in ks]
            b_sse, v_sse, ns = [], [], []
            for w in wl:
                d = W[w]
                c = cand_pred.get((name_b, w))
                pr = d["base"] if c is None else \
                    d["base"] + np.clip(lam_b * (c - d["base"]), -D_b, D_b)
                b_sse.append(float(((d["y"] - d["base"]) ** 2).sum()))
                v_sse.append(float(((d["y"] - pr) ** 2).sum()))
                ns.append(len(d["y"]))
            b_sse, v_sse, ns = map(np.array, (b_sse, v_sse, ns))
            rng = np.random.default_rng(0)
            boots = []
            for _ in range(2000):
                ix = rng.integers(0, len(wl), len(wl))
                boots.append(np.sqrt(v_sse[ix].sum() / ns[ix].sum())
                             - np.sqrt(b_sse[ix].sum() / ns[ix].sum()))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            n_up = int((v_sse < b_sse).sum())
            print(f"  {name_b} lam={lam_b} clip={D_b} k{k}: delta IC95 "
                  f"[{lo:+.3f}, {hi:+.3f}]  pozos mejor/peor {n_up}/{int((v_sse > b_sse).sum())}")

    # ------------- estratos nn del mejor combo -------------
    best = df.iloc[0]
    name, lam, D = best["variante"], float(best["lam"]), float(best["clip"])
    print(f"\n== estratos nn (terciles, k150) del mejor: {name} lam={lam} clip={D} ==")
    nn_med = {wid: float(np.median(d["nn"])) for wid, d in W.items() if wid in k150}
    qs = np.quantile(list(nn_med.values()), [1 / 3, 2 / 3])
    for t, (lo, hi) in enumerate([(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], np.inf)]):
        ws = [w for w, v in nn_med.items() if lo <= v < hi]
        sb, sv = [], []
        for wid in ws:
            d = W[wid]
            if name == "sigfac65":
                pr = sigfac_pred(d, lam)
            else:
                c = cand_pred.get((name, wid))
                pr = d["base"] if c is None else d["base"] + np.clip(lam * (c - d["base"]), -D, D)
            sb.append((float(((d["y"] - d["base"]) ** 2).sum()), len(d["y"])))
            sv.append((float(((d["y"] - pr) ** 2).sum()), len(d["y"])))
        print(f"  tercil {t} (nn {lo:.0f}-{hi:.0f}, n={len(ws)}): "
              f"base {pooled(sb):.3f} -> {pooled(sv):.3f}  d={pooled(sv)-pooled(sb):+.3f}")
    print("DONE_V9_PESOS")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    dict(check=check, bt50=bt50, eval=eval_phase)[cmd]()
