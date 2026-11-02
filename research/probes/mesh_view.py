"""Close-up of what ``mesh_strokes`` takes, for the suspicious groups, and of curves on pipe layers left out of the mask.

    py -3.12 research/probes/mesh_view.py runs_colab/viz
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.geometry import PageSegments  # noqa: E402
from openstroke.postprocess import by_group, load_params, mesh_strokes, style_groups  # noqa: E402

WANT = [("doc000162_p0001_gt108", "KH VIDIFI"), ("doc000191_p0001_gt120", "INNO-ACMV-B.EA"), ("doc000194_p0001_gt121", "360D-ACMV-B.EA"), ("doc000190_p0001_gt119", "M-_ACONSUPP"), ("doc000022_p0001_gt11", "M-_MEVSEQFR")]
HALF = 45


def load(stem):
    d = np.load(f"datasets/e3/{stem}.npz")
    g = np.load(f"runs_colab/chain_cache/e37/{stem}.npz")
    off = d["shape_offsets"].astype(np.int64)
    segs = PageSegments(seg=d["segments"].astype(np.float64), owner=np.repeat(np.arange(len(off) - 1), np.diff(off)), shape_offsets=off)
    group = style_groups({x: d[x] for x in ("layer_id", "stroke_rgb", "width", "type_code")})
    return d, g, segs, group


def main(out):
    out = Path(out)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=85)
    for ax, (stem, key) in zip(axes.ravel(), WANT):
        d, g, segs, group = load(stem)
        L = g["L"]
        names = [str(x) for x in d["layer_names"]]
        mesh = mesh_strokes(segs, group, L)
        cand = [i for i in np.unique(group[mesh]) if key in names[g["lay"][np.flatnonzero(group == i)[0]]]] or list(np.unique(group[mesh]))
        i = max(cand, key=lambda q: L[group == q].sum())
        chosen = (mesh & (group == i))[segs.owner]
        mid = (segs.seg[:, :2] + segs.seg[:, 2:]) / 2
        cen = np.median(mid[chosen], axis=0)
        near = np.flatnonzero(chosen)[np.argmin(np.abs(mid[chosen] - cen).sum(1))]
        cen = mid[near]
        ins = (np.abs(mid - cen) < HALF).all(1)
        rest = ins & ~chosen
        ax.add_collection(LineCollection(segs.seg[rest].reshape(-1, 2, 2), colors="#bbbbbb", linewidths=0.6))
        ax.add_collection(LineCollection(segs.seg[ins & chosen].reshape(-1, 2, 2), colors="#d0108f", linewidths=0.8))
        ax.set_xlim(cen[0] - HALF, cen[0] + HALF)
        ax.set_ylim(cen[1] + HALF, cen[1] - HALF)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{stem[:9]} {key}: mesh {(mesh & (group == i)).sum()} of {(group == i).sum()} strokes", fontsize=10)
    # curves on a pipe layer left out, the page where most are
    params, _ = load_params("runs_colab/e37/fold0/best.pt")
    ax = axes.ravel()[5]
    d, g, segs, group = load("doc000191_p0001_gt120")
    L, p, y, lay = g["L"], g["p"], g["y"], g["lay"]
    rule = g["obj"] | (p >= params["hvac"]["sure"]) | by_group(p, L, group, params["hvac"]["group"])
    n = lay.max() + 1
    pipe = (np.bincount(lay[y], L[y], n) / np.maximum(np.bincount(lay, L, n), 1e-9) >= 0.5)[lay]
    miss = (d["op_counts"][:, 1] > 0) & pipe & ~rule
    mid = (segs.seg[:, :2] + segs.seg[:, 2:]) / 2
    print("doc191 curves on pipe layers out of the mask:", int(miss.sum()), "length", round(L[miss].sum()), "mean p", round(float(p[miss].mean()), 3), "labelled true", round(L[miss & y].sum() / max(L[miss].sum(), 1), 3))
    q = np.floor(mid[miss[segs.owner]] / 120).astype(np.int64)
    u, c = np.unique(q, axis=0, return_counts=True)
    cen = (u[c.argmax()] + 0.5) * 120
    ins = (np.abs(mid - cen) < 60).all(1)
    cls = rule[segs.owner]
    ax.add_collection(LineCollection(segs.seg[ins & ~cls].reshape(-1, 2, 2), colors="#bbbbbb", linewidths=0.5))
    ax.add_collection(LineCollection(segs.seg[ins & cls].reshape(-1, 2, 2), colors="#2a9d2a", linewidths=0.8))
    ax.add_collection(LineCollection(segs.seg[ins & miss[segs.owner]].reshape(-1, 2, 2), colors="#e01010", linewidths=1.2))
    ax.set_xlim(cen[0] - 60, cen[0] + 60)
    ax.set_ylim(cen[1] + 60, cen[1] - 60)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("doc191: green mask, red curve on pipe layer left out", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "mesh_groups.png")


if __name__ == "__main__":
    main(sys.argv[1])
