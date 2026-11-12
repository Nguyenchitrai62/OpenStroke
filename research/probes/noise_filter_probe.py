"""How much background can a recall-safe pre-filter drop? 0 GPU, GT tokens only (no layers used).
Per stroke: length (px at 4 px / pt), width, luminance, primitive type, and local density (strokes within 16 px). For simple rules,
the share of strokes removed and the share of GT object stroke LENGTH and of GT OBJECTS kept (an object is kept when >= 90 % of its
length survives), on train and val pages separately (thresholds are chosen on train only).

    py -3.12 research/probes/noise_filter_probe.py <obb_hvac7 folder>
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from openstroke.obb.gt_dataset import GT_SPEC


def feats(path):
    with np.load(path) as z:
        t = np.asarray(z["tokens"], np.float64)
    pts = t[:, 1:17].reshape(-1, 8, 2)
    L = np.linalg.norm(np.diff(pts, axis=1), axis=2).sum(1)
    ext = (pts.max(1) - pts.min(1)).max(1)
    lum = np.clip(t[:, 17:20], 0, 1) @ [0.299, 0.587, 0.114]
    c = pts.mean(1)
    dens = np.array([len(x) for x in cKDTree(c).query_ball_point(c, 16.0)])
    return dict(L=L, ext=ext, lum=lum, w=t[:, 20], typ=t[:, 0], dens=dens, inst=np.rint(t[:, 22]).astype(int))


def report(name, pages, keep_fn):
    rm = n = gl = gk = ok = no = 0
    for f in pages:
        keep = keep_fn(f)
        n += len(keep); rm += (~keep).sum()
        g = f["inst"] >= 0
        gl += f["L"][g].sum(); gk += f["L"][g & keep].sum()
        for o in np.unique(f["inst"][g]):
            s = f["inst"] == o
            no += 1; ok += f["L"][s & keep].sum() >= 0.9 * f["L"][s].sum()
    return f"{name:44s} removed {rm / n:6.1%} | GT length kept {gk / gl:6.2%} | GT objects kept {ok / no:6.2%}"


def main(folder):
    folder = Path(folder)
    data = {sp: [feats(folder / "vectors" / sp / f"{r['stem']}.npz") for r in GT_SPEC if r["split"] == sp] for sp in ("train", "val")}
    allg = np.concatenate([f["inst"] >= 0 for f in data["train"]])
    print("GT strokes share of all strokes: train %.2f%%" % (100 * allg.mean()))
    rules = {
        "light strokes (lum > 0.8)": lambda f: f["lum"] <= 0.8,
        "tiny strokes (extent < 3 px)": lambda f: f["ext"] >= 3,
        "small strokes (extent < 6 px)": lambda f: f["ext"] >= 6,
        "small strokes (extent < 10 px)": lambda f: f["ext"] >= 10,
        "small strokes (extent < 16 px)": lambda f: f["ext"] >= 16,
        "tiny + dense (extent < 12 px, >= 30 nbrs)": lambda f: ~((f["ext"] < 12) & (f["dens"] >= 30)),
        "text-like (extent < 24 px, >= 15 nbrs)": lambda f: ~((f["ext"] < 24) & (f["dens"] >= 15)),
        "very dense (>= 60 nbrs, extent < 40 px)": lambda f: ~((f["dens"] >= 60) & (f["ext"] < 40)),
    }
    for sp in ("train", "val"):
        print(f"-- {sp} ({len(data[sp])} pages, {sum(len(f['L']) for f in data[sp])} strokes)")
        for k, fn in rules.items():
            print(report(k, data[sp], fn))
        print(report("tiny+dense OR light", data[sp], lambda f: rules["tiny + dense (extent < 12 px, >= 30 nbrs)"](f) & (f["lum"] <= 0.8)))


if __name__ == "__main__":
    main(*sys.argv[1:])
