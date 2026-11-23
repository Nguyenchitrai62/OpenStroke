"""Error pictures for a run's saved val scores: every stroke of a page drawn with its outcome at a threshold (no model run).

    green TP (pipe found) | red FP (called pipe, is not) | blue FN (pipe missed) | light grey TN

    py -3.12 tools/visualize_errors.py <data dir> <val_scores.npz> <threshold> <out dir> [stem ...]
Without stems, every page of the scores file. Writes <out dir>/<stem>.png with the page's length-weighted F2 / P / R and FP / FN length.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

COLORS = {"tn": (0.74, 0.74, 0.74, 0.25), "tp": (0.0, 0.784, 0.325, 1.0), "fp": (1.0, 0.09, 0.267, 1.0), "fn": (0.161, 0.475, 1.0, 1.0)}


def draw(data: Path, scores, stem: str, thr: float, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    with np.load(data / f"{stem}.npz") as z:
        seg, off, y, L = z["segments"], z["shape_offsets"].astype(np.int64), z["label"] > 0, z["length"].astype(float)
    p = np.zeros(len(y))
    p[scores[f"{stem}__index"]] = scores[stem].astype(float)
    pick = p >= thr
    cls = np.where(pick & y, "tp", np.where(pick & ~y, "fp", np.where(~pick & y, "fn", "tn")))
    tp, fp, fn = (float(L[cls == k].sum()) for k in ("tp", "fp", "fn"))
    P, R = tp / max(tp + fp, 1e-9), tp / max(tp + fn, 1e-9)
    f2 = 5 * P * R / max(4 * P + R, 1e-9)
    owner = np.repeat(np.arange(len(off) - 1), np.diff(off))
    lines = seg.reshape(-1, 2, 2)
    w, h = float(seg[:, [0, 2]].max() - seg[:, [0, 2]].min()), float(seg[:, [1, 3]].max() - seg[:, [1, 3]].min())
    fig, ax = plt.subplots(figsize=(24, 24 * h / max(w, 1)), dpi=110)
    for k, lw in (("tn", 0.3), ("tp", 0.9), ("fn", 1.4), ("fp", 1.4)):   # errors drawn last, on top
        m = cls[owner] == k
        if m.any():
            ax.add_collection(LineCollection(lines[m], colors=[COLORS[k]], linewidths=lw))
    ax.set_xlim(seg[:, [0, 2]].min(), seg[:, [0, 2]].max())
    ax.set_ylim(seg[:, [1, 3]].max(), seg[:, [1, 3]].min())   # page frame is top-left
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{stem}  thr {thr:.2f}  F2 {f2:.3f}  P {P:.3f}  R {R:.3f}  |  FP {fp:,.0f} pt  FN {fn:,.0f} pt  "
                 "(green TP, red FP, blue FN)", fontsize=14)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    return {"stem": stem, "f2": f2, "P": P, "R": R, "fp_pt": fp, "fn_pt": fn}


def main(data, scores, thr, out, *stems):
    data, out = Path(data), Path(out)
    z = np.load(scores)
    stems = stems or [k for k in z.files if "__" not in k]
    for st in stems:
        r = draw(data, z, st, float(thr), out)
        print(f"{st}: F2 {r['f2']:.3f} P {r['P']:.3f} R {r['R']:.3f} FP {r['fp_pt']:,.0f} pt FN {r['fn_pt']:,.0f} pt -> {out / (st + '.png')}",
              flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
