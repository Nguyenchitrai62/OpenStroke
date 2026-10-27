"""E35: learned post-processing on a run's 5-fold val scores + page-relative style features (research/HYPOTHESES.md H47).

The network scores each shape; a person also compares a shape's style group with the rest of the page ("the thin gray strokes are
80 % of this sheet, they are the background"). Per shape: model score, layer-head score, mean score of its style group / layer /
link-graph neighbours, the group's share of the page's length, darkness and width as length-weighted percentiles within the page,
white fill. A HistGradientBoosting classifier (sklearn, BSD) is fit on the val pages of 4 folds and applied to the 5th (all scores are
out-of-sample for the network), then one pooled threshold, as for the baseline.

    py -3.12 research/probes/stack_probe.py e30
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.metrics import length_counts, prf  # noqa: E402

DATA = Path("datasets/e3")
THRS = np.round(np.arange(0.04, 0.9, 0.01), 2)
NAMES = ["p", "q", "p_style", "p_layer", "p_graph", "p_style_max", "style_share", "layer_share", "dark", "dark_pct", "width_pct",
         "style_dark_pct", "loglen", "type", "white_fill", "fire", "n_styles"]


def gmean(v, w, g):
    return (np.bincount(g, weights=v * w) / np.maximum(np.bincount(g, weights=w), 1e-9))[g]


def wpct(v, w):
    """Length-weighted percentile of each value within the page (0 = lightest / thinnest)."""
    o = np.argsort(v, kind="stable")
    c = np.cumsum(w[o]) / max(w.sum(), 1e-9)
    u, first = np.unique(v[o], return_index=True)
    last = np.r_[first[1:], len(o)] - 1
    return c[last][np.searchsorted(u, v)]


def features(stem, p, q):
    a = np.load(DATA / f"{stem}.npz")
    L = a["length"].astype(np.float64)
    w = L + 0.5
    srgb, frgb = np.nan_to_num(a["stroke_rgb"], nan=-1.0), np.nan_to_num(a["fill_rgb"], nan=-1.0)
    has_s = srgb[:, 0] >= 0
    rgb = np.where(has_s[:, None], srgb, frgb)
    dark = np.where(rgb[:, 0] >= 0, 1.0 - rgb @ np.array([0.299, 0.587, 0.114]), 0.0)
    width = a["width"].astype(np.float64)
    key = np.c_[srgb.round(2), frgb.round(2), width.round(2), a["type_code"]]
    style = np.unique(key, axis=0, return_inverse=True)[1].ravel()
    lid = a["layer_id"].astype(int)
    e = sorted(DATA.glob(f"{stem}.graph.*.npz"))
    if e:
        ed = np.load(e[0])["edges"].astype(np.int64)
        s = np.bincount(ed[:, 0], weights=p[ed[:, 1]], minlength=len(p)) + np.bincount(ed[:, 1], weights=p[ed[:, 0]], minlength=len(p))
        n = np.bincount(ed[:, 0], minlength=len(p)) + np.bincount(ed[:, 1], minlength=len(p))
        pg = np.where(n > 0, s / np.maximum(n, 1), p)
    else:
        pg = p
    tot = w.sum()
    smax = np.zeros(style.max() + 1)
    np.maximum.at(smax, style, p)
    sd = gmean(dark, w, style)
    X = np.c_[p, q, gmean(p, w, style), gmean(p, w, lid), pg, smax[style], (np.bincount(style, weights=w) / tot)[style],
              (np.bincount(lid, weights=w) / tot)[lid], dark, wpct(dark, w), wpct(width, w), wpct(sd, w), np.log1p(L),
              a["type_code"], (~has_s & (frgb.min(1) > 0.95)).astype(float),
              np.full(len(p), float(json.loads(str(a["meta"]))["mode"] == "fire")), np.full(len(p), float(style.max() + 1))]
    return X.astype(np.float32), a["label"].astype(bool), L, json.loads(str(a["meta"]))["mode"]


def main(run):
    from sklearn.ensemble import HistGradientBoostingClassifier

    pages = []
    for k in range(5):
        z = np.load(f"runs_colab/{run}/fold{k}/val_scores.npz")
        for s in [f for f in z.files if not f.endswith(("__index", "__layer"))]:
            n = len(np.load(DATA / f"{s}.npz")["label"])
            p, q = np.zeros(n), np.zeros(n)
            p[z[f"{s}__index"]] = z[s]
            if f"{s}__layer" in z.files:
                q[z[f"{s}__index"]] = z[f"{s}__layer"]
            X, y, L, mode = features(s, p, q)
            pages.append(dict(stem=s, fold=k, X=X, y=y, L=L, mode=mode, p=p))
    rng = np.random.default_rng(0)
    for k in range(5):
        tr = [pg for pg in pages if pg["fold"] != k]
        Xs, ys, ws = [], [], []
        for pg in tr:  # equal total weight per page, sampled by length
            m = min(40000, len(pg["y"]))
            i = rng.choice(len(pg["y"]), m, replace=False, p=pg["L"] / pg["L"].sum()) if m < len(pg["y"]) else np.arange(m)
            Xs.append(pg["X"][i]), ys.append(pg["y"][i]), ws.append(np.full(m, 1.0 / m))
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                             l2_regularization=1.0, random_state=0)
        clf.fit(np.concatenate(Xs), np.concatenate(ys), sample_weight=np.concatenate(ws) * 1e4)
        for pg in pages:
            if pg["fold"] == k:
                pg["s"] = clf.predict_proba(pg["X"])[:, 1]

    def pooled(key, t):
        acc = {}
        for pg in pages:
            for g in ("all", pg["mode"]):
                acc[g] = acc.get(g, np.zeros(3)) + length_counts(pg[key] >= t, pg["y"], pg["L"])
        return {g: prf(*v) for g, v in acc.items()}

    for key in ("p", "s"):
        t = max(THRS, key=lambda t: pooled(key, t)["all"]["f2"])
        m = pooled(key, t)
        print(f"{'model' if key == 'p' else 'stack'} thr {t:.2f}  " +
              "  ".join(f"{g}: F2 {v['f2']:.3f} (R {v['recall']:.3f} P {v['precision']:.3f})" for g, v in m.items()))
    tp, ts = (max(THRS, key=lambda t, k=k: pooled(k, t)["all"]["f2"]) for k in ("p", "s"))
    for pg in sorted(pages, key=lambda pg: (pg["mode"], pg["stem"])):
        a, b = prf(*length_counts(pg["p"] >= tp, pg["y"], pg["L"])), prf(*length_counts(pg["s"] >= ts, pg["y"], pg["L"]))
        print(f"   f{pg['fold']} {pg['mode']:4s} {pg['stem'][:26]:26s} model {a['f2']:.3f} (P {a['precision']:.2f})  "
              f"stack {b['f2']:.3f} (P {b['precision']:.2f})")


if __name__ == "__main__":
    main(sys.argv[1])
