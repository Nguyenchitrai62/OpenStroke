"""Post-processing on a run's saved val scores (``fold<k>/val_scores.npz``, best.pt + TTA): does a fixed, label-free rule on
top of the model raise length-weighted F2? Rules (fixed before looking at the numbers, HYPOTHESES E9):

- ``none``: the model alone, threshold picked on the pooled val pages (what the trainer reports).
- ``layer``: p' = 0.5 p + 0.5 * length-weighted mean of p over the shape's layer on that page.
- ``style``: same with the style group (stroke colour + width + type, ``data.style_ids``) instead of the layer.
- ``graph``: p' = 0.5 p + 0.5 * mean of p over the shape's neighbours in the link graph (g3, ``openstroke/graph.py``).

Each rule's threshold is the best one on the pooled pages, as for ``none``, so the comparison is like for like (optimistic for
all of them in the same way).

    py -3.12 research/probes/postproc_probe.py runs_colab/e8/runs/e8a/fold0 datasets/e2
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.metrics import length_counts, prf  # noqa: E402

THRS = np.round(np.arange(0.02, 0.96, 0.01), 2)


def group_mean(p, L, g):
    w = L + 1e-6
    _, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=p * w)
    n = np.bincount(inv, weights=w)
    return (s / n)[inv]


def graph_mean(p, data, stem):
    f = sorted(Path(data).glob(f"{stem}.graph.*.npz"))
    if not f:
        return None
    e = np.load(f[0])["edges"]
    src, dst = e[:, 0], e[:, 1]
    s = np.bincount(src, weights=p[dst], minlength=len(p)) + np.bincount(dst, weights=p[src], minlength=len(p))
    n = np.bincount(src, minlength=len(p)) + np.bincount(dst, minlength=len(p))
    return np.where(n > 0, s / np.maximum(n, 1), p)


def style_key(z):
    rgb = np.nan_to_num(z["stroke_rgb"], nan=-1).round(2)
    return np.unique(np.c_[rgb, np.round(z["width"], 2), z["type_code"]], axis=0, return_inverse=True)[1].ravel()


def main(fold_dir, data):
    sc = np.load(Path(fold_dir) / "val_scores.npz")
    pages = {}
    for stem in sc.files:
        z = np.load(Path(data) / f"{stem}.npz")
        p, y, L = sc[stem].astype(np.float64), z["label"].astype(bool), z["length"].astype(np.float64)
        rules = {"none": p, "layer": 0.5 * p + 0.5 * group_mean(p, L, z["layer_id"]),
                 "style": 0.5 * p + 0.5 * group_mean(p, L, style_key(z))}
        gm = graph_mean(p, data, stem)
        if gm is not None:
            rules["graph"] = 0.5 * p + 0.5 * gm
        pages[stem] = (rules, y, L)
    for rule in ("none", "layer", "style", "graph"):
        if not all(rule in r for r, _, _ in pages.values()):
            print(f"{rule:6s} skipped (missing on some page)")
            continue
        best = max(THRS, key=lambda t: prf(*np.sum([length_counts(r[rule] >= t, y, L) for r, y, L in pages.values()], 0))["f2"])
        tot = np.sum([length_counts(r[rule] >= best, y, L) for r, y, L in pages.values()], 0)
        m = prf(*tot)
        per = " ".join(f"{prf(*length_counts(r[rule] >= best, y, L))['f2']:.2f}" for r, y, L in pages.values())
        print(f"{rule:6s} thr {best:.2f}  R {m['recall']:.3f} P {m['precision']:.3f} F2 {m['f2']:.3f} | pages {per}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
