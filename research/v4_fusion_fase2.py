"""Fase 2 autonoma: lee fase1, refina el mejor sigma_s con init/kappa/blend a
k=60, y CONFIRMA el mejor global + la referencia blend25 a k=150."""
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from v4_fusion_pf import Fusion, VARIANTS  # noqa: E402
from cv import evaluate  # noqa: E402


def parse_log(p):
    rows = []
    for ln in Path(p).read_text().splitlines():
        m = re.match(r"(\S+)\s+rmse=\s*([\d.]+)\s+proxy=\s*([\d.]+)", ln)
        if m:
            rows.append((m.group(1), float(m.group(2)), float(m.group(3))))
    return rows


def main():
    fus = Fusion()

    def run(name, kw, k):
        t0 = time.time()
        r = evaluate(fus.make(**kw), k=k, verbose=False)
        dt = time.time() - t0
        print(f"[k={k}] {name:22s} rmse={r['rmse']:7.3f}  proxy={r['rmse_lb_proxy']:7.3f}"
              f"  ({dt:.0f}s, {dt/r['n_wells']:.2f}s/pozo)", flush=True)
        return r["rmse"]

    f1 = parse_log(_HERE / "v4_fase1_k60.log")
    fusion_f1 = [r for r in f1 if r[0].startswith(("sig", "ad_"))]
    best_name, best_rmse, _ = min(fusion_f1, key=lambda r: r[1])
    print(f"fase1 mejor fusion: {best_name} rmse={best_rmse:.3f}", flush=True)
    base = dict(VARIANTS[best_name])

    f2 = [(f"{best_name}+init", dict(base, init_prior=True)),
          (f"{best_name}+k001", dict(base, kappa=0.001)),
          (f"{best_name}+k01", dict(base, kappa=0.01)),
          (f"{best_name}+init+k001", dict(base, init_prior=True, kappa=0.001)),
          (f"{best_name}+blend15", dict(base, blend=0.15))]
    scores = {best_name: best_rmse}
    variants = {best_name: base}
    for name, kw in f2:
        scores[name] = run(name, kw, 60)
        variants[name] = kw

    win = min(scores, key=scores.get)
    print(f"\nmejor global k=60: {win} rmse={scores[win]:.3f}", flush=True)
    print("\n== confirmacion k=150 ==", flush=True)
    run(f"CONFIRM {win}", variants[win], 150)
    run("REF blend25", dict(blend=0.25), 150)


if __name__ == "__main__":
    main()
