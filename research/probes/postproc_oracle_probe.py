"""E40 diagnosis: how far could a post-processing rule of each kind go on a run's saved val scores? (no model is run)

Every ceiling uses the labels, so none of them is a result: it only says which kind of rule is worth designing. A rule of kind
``g`` keeps or drops whole groups of the shapes with p >= t; the oracle keeps the groups purer than one per-mode cut (the optimal
F2 choice over all cuts). ``page`` is the oracle threshold of each page.

    py -3.12 research/probes/postproc_oracle_probe.py e37
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.metrics import length_counts, prf  # noqa: E402

DATA = Path("datasets/e3")
THRS = (0.02, 0.05, 0.1, 0.2, 0.3)
GRID = np.round(np.arange(0.02, 0.96, 0.01), 2)


def load_run(run):
    pages = []
    for k in range(5):
        r = json.load(open(f"runs_colab/{run}/fold{k}/result.json"))
        z = np.load(f"runs_colab/{run}/fold{k}/val_scores.npz")
        for pg in r["pages"]:
            s = pg["stem"]
            a = np.load(DATA / f"{s}.npz")
            p = np.zeros(len(a["label"]))
            p[z[f"{s}__index"]] = z[s]
            style = np.unique(np.c_[np.nan_to_num(a["stroke_rgb"], nan=-1).round(2), np.round(a["width"], 2), a["type_code"]],
                              axis=0, return_inverse=True)[1].ravel()
            pages.append(dict(stem=s, fold=k, mode=json.loads(str(a["meta"]))["mode"], y=a["label"].astype(bool),
                              L=a["length"].astype(np.float64), p=p, layer=a["layer_id"].astype(np.int64), style=style,
                              edges=np.load(sorted(DATA.glob(f"{s}.graph.*.npz"))[0])["edges"].astype(np.int64)))
    return pages


def components(pg, keep):
    n = len(keep)
    e = pg["edges"]
    m = keep[e[:, 0]] & keep[e[:, 1]]
    g = coo_matrix((np.ones(m.sum(), bool), (e[m, 0], e[m, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def groups_of(pg, keep, kind):
    if kind == "layer":
        return pg["layer"]
    if kind == "style":
        return pg["style"]
    if kind == "layer+style":
        return np.unique(np.c_[pg["layer"], pg["style"]], axis=0, return_inverse=True)[1].ravel()
    return components(pg, keep)


def oracle_groups(pages, t, kind):
    """-> (F2, R, P, groups kept + dropped per page): keep groups whose purity beats the best single cut."""

    rows = []
    n_groups = []
    for pg in pages:
        keep = pg["p"] >= t
        g = groups_of(pg, keep, kind)[keep]
        _, inv = np.unique(g, return_inverse=True)
        L, y = pg["L"][keep], pg["y"][keep]
        tp, tot = np.bincount(inv, L * y), np.bincount(inv, L)
        rows.append(np.c_[tp, tot])
        n_groups.append(len(tp))
    a = np.concatenate(rows) if rows else np.zeros((0, 2))
    all_true = sum(pg["L"][pg["y"]].sum() for pg in pages)
    order = np.argsort(-(a[:, 0] / np.maximum(a[:, 1], 1e-9)))
    tp, tot = np.cumsum(a[order, 0]), np.cumsum(a[order, 1])
    best = max(((prf(tp[i], tot[i] - tp[i], all_true - tp[i])) for i in range(len(order))), key=lambda m: m["f2"], default=prf(0, 0, all_true))
    return best, float(np.median(n_groups))


def main(run):
    pages = load_run(run)
    for mode in ("fire", "hvac"):
        pm = [pg for pg in pages if pg["mode"] == mode]
        c = lambda t: np.sum([length_counts(pg["p"] >= t, pg["y"], pg["L"]) for pg in pm], 0)  # noqa: E731
        t0 = max(GRID, key=lambda t: prf(*c(t))["f2"])
        m = prf(*c(t0))
        print(f"{mode} pages {len(pm)}  one threshold t {t0:.2f}  F2 {m['f2']:.3f} R {m['recall']:.3f} P {m['precision']:.3f}")
        cp = np.sum([max((length_counts(pg["p"] >= t, pg["y"], pg["L"]) for t in GRID), key=lambda x: prf(*x)["f2"]) for pg in pm], 0)
        m = prf(*cp)
        print(f"  oracle threshold per page             F2 {m['f2']:.3f} R {m['recall']:.3f} P {m['precision']:.3f}")
        for kind in ("layer", "style", "layer+style", "component"):
            for t in THRS:
                m, ng = oracle_groups(pm, t, kind)
                print(f"  oracle {kind:12s} over p >= {t:<5} F2 {m['f2']:.3f} R {m['recall']:.3f} P {m['precision']:.3f}  groups/page {ng:.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
