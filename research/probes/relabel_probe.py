"""E36 / LEDGER F41: post-processing that applies the label rule to the model's own confident strokes.

GT (and the pipeline's score) label a shape ``true`` when most of its length lies within 2 pt of a pipe line, whatever layer or style it
has. Here the "lines" are the segments of the shapes the model scores >= t; every shape along them becomes true (``labels.label_shapes``,
the GT rule itself), plus the confident shapes. ``t`` is picked per mode on all scored pages, like the plain threshold.

    py -3.12 research/probes/relabel_probe.py e30 [--folds 3,4] [--modes fire,hvac]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.geometry import PageSegments  # noqa: E402
from openstroke.labels import GTGeometry, label_shapes  # noqa: E402
from openstroke.metrics import length_counts, prf  # noqa: E402

DATA = Path("datasets/e3")
PLAIN = np.round(np.arange(0.05, 0.9, 0.01), 2)
SEEDS = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)


def relabel(a, p, t):
    n = len(p)
    off = a["shape_offsets"].astype(np.int64)
    seg = a["segments"].astype(np.float64)
    owner = np.repeat(np.arange(n), np.diff(off))
    pred = p >= t
    m = pred[owner]
    geoms = list(shapely.linestrings(seg[m].reshape(-1, 2, 2)))
    g = GTGeometry("fire", geoms, ["Line"] * len(geoms), [""] * len(geoms), [], [])
    return label_shapes(PageSegments(seg=seg, owner=owner, shape_offsets=off), g).label.astype(bool) | pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--modes", default="fire,hvac")
    a = ap.parse_args()
    modes = a.modes.split(",")
    pages = []
    for k in map(int, a.folds.split(",")):
        r = json.load(open(f"runs_colab/{a.run}/fold{k}/result.json"))
        z = np.load(f"runs_colab/{a.run}/fold{k}/val_scores.npz")
        for pg in r["pages"]:
            s = pg["stem"]
            d = np.load(DATA / f"{s}.npz")
            mode = json.loads(str(d["meta"]))["mode"]
            if mode not in modes:
                continue
            p = np.zeros(len(d["label"]))
            p[z[f"{s}__index"]] = z[s]
            y, L = d["label"].astype(bool), d["length"].astype(np.float64)
            pages.append(dict(stem=s, mode=mode, y=y, L=L, p=p, rel={t: relabel(d, p, t) for t in SEEDS}))
    tot = {"plain": np.zeros(3), "relabel": np.zeros(3)}
    for mode in modes:
        pm = [pg for pg in pages if pg["mode"] == mode]
        if not pm:
            continue
        cnt = lambda f: np.sum([length_counts(f(pg), pg["y"], pg["L"]) for pg in pm], 0)  # noqa: E731
        tp = max(PLAIN, key=lambda t: prf(*cnt(lambda pg: pg["p"] >= t))["f2"])
        tr = max(SEEDS, key=lambda t: prf(*cnt(lambda pg: pg["rel"][t]))["f2"])
        c0, c1 = cnt(lambda pg: pg["p"] >= tp), cnt(lambda pg: pg["rel"][tr])
        tot["plain"] += c0
        tot["relabel"] += c1
        for name, t, c in (("plain", tp, c0), ("relabel", tr, c1)):
            m = prf(*c)
            print(f"{mode:4s} {name:7s} t {t:.2f}  F2 {m['f2']:.3f}  R {m['recall']:.3f}  P {m['precision']:.3f}")
    for name, c in tot.items():
        m = prf(*c)
        print(f"all  {name:7s}         F2 {m['f2']:.3f}  R {m['recall']:.3f}  P {m['precision']:.3f}")


if __name__ == "__main__":
    main()
