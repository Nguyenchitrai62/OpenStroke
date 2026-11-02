"""E41: drop noise layers using the strokes the model caught (user 2026-11-02): the model holds recall, this step raises precision.

No labels, no model run: for each group g of a page (layer, style, or both) on saved val scores,
    cov   = length of g with p >= t_hi / length of g          (how much of the group the model believes in)
    share = length of g with p >= t_hi / same on the whole page (is it a main group of the page's network)
Keep g when cov >= c_min or share >= s_min. A shape is true when p >= t_lo in a kept group, or p >= t_safe anywhere.
Parameters are tuned per mode on TUNE and applied unchanged to TEST; ``plain`` gets its threshold from TUNE in the same way.

    py -3.12 research/probes/layerfilter_probe.py --tune e30 --test e37
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from postproc_oracle_probe import load_run  # noqa: E402
from openstroke.metrics import length_counts, prf  # noqa: E402

T_LO = (0.02, 0.05, 0.1, 0.2)
T_HI = (0.2, 0.3, 0.5)
C_MIN = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
S_MIN = (0.02, 0.05, 0.1, 0.2, 0.5, 1.01)
T_SAFE = 0.8
GROUPS = ("layer", "style", "layer+style")
PLAIN = np.round(np.arange(0.02, 0.96, 0.01), 2)


def prepare(pages):
    for pg in pages:
        pg["g"] = {"layer": pg["layer"], "style": pg["style"],
                   "layer+style": np.unique(np.c_[pg["layer"], pg["style"]], axis=0, return_inverse=True)[1].ravel()}
    return pages


def predict(pg, group, t_lo, t_hi, c_min, s_min):
    g, L, p = pg["g"][group], pg["L"], pg["p"]
    n = g.max() + 1
    conf = p >= t_hi
    tot = np.bincount(g, L, n)
    got = np.bincount(g[conf], L[conf], n)
    cov = got / np.maximum(tot, 1e-9)
    share = got / max(got.sum(), 1e-9)
    keep = (cov >= c_min) | (share >= s_min)
    return (keep[g] & (p >= t_lo)) | (p >= T_SAFE)


def score(pages, f):
    return prf(*np.sum([length_counts(f(pg), pg["y"], pg["L"]) for pg in pages], 0))


def show(name, m):
    print(f"    {name:34s} F2 {m['f2']:.3f}  R {m['recall']:.3f}  P {m['precision']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="e30")
    ap.add_argument("--test", default="e37")
    a = ap.parse_args()
    tune, test = prepare(load_run(a.tune)), prepare(load_run(a.test))
    total = {k: np.zeros(3) for k in ("plain", "filter")}
    for mode in ("fire", "hvac"):
        tu, te = [p for p in tune if p["mode"] == mode], [p for p in test if p["mode"] == mode]
        t0 = max(PLAIN, key=lambda t: score(tu, lambda pg: pg["p"] >= t)["f2"])
        grid = list(itertools.product(GROUPS, T_LO, T_HI, C_MIN, S_MIN))
        scored = [(score(tu, lambda pg, q=q: predict(pg, *q))["f2"], q) for q in grid]
        scored.sort(key=lambda x: -x[0])
        best = scored[0][1]
        print(f"{mode}: picked on {a.tune}: plain t {t0:.2f}; filter {best}; top of grid on {a.tune}: {[round(s, 3) for s, _ in scored[:3]]}")
        show(f"{a.tune} plain", score(tu, lambda pg: pg["p"] >= t0))
        show(f"{a.tune} filter", score(tu, lambda pg: predict(pg, *best)))
        show(f"{a.test} plain (threshold from {a.tune})", score(te, lambda pg: pg["p"] >= t0))
        show(f"{a.test} filter (params from {a.tune})", score(te, lambda pg: predict(pg, *best)))
        total["plain"] += np.sum([length_counts(pg["p"] >= t0, pg["y"], pg["L"]) for pg in te], 0)
        total["filter"] += np.sum([length_counts(predict(pg, *best), pg["y"], pg["L"]) for pg in te], 0)
        for pg in sorted(te, key=lambda pg: pg["stem"]):
            x, y = prf(*length_counts(pg["p"] >= t0, pg["y"], pg["L"])), prf(*length_counts(predict(pg, *best), pg["y"], pg["L"]))
            print(f"      {pg['stem'][:21]:21s} plain {x['f2']:.3f} (R {x['recall']:.3f} P {x['precision']:.3f})  filter {y['f2']:.3f} (R {y['recall']:.3f} P {y['precision']:.3f})")
    print(f"all, {a.test}, parameters from {a.tune}")
    show("plain", prf(*total["plain"]))
    show("filter", prf(*total["filter"]))


if __name__ == "__main__":
    main()
