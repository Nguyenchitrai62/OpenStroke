"""Would a colour pre-filter (drop strokes by colour before drawing the page and tokenising) keep the HVAC strokes? 0 GPU.

Reads the OpenOBB25 vector tokens of obb_hvac7 (rgb in columns 17-19, inst >= 0 = stroke of a GT object). For the 8 val pages, per rule:
share of strokes removed (fewer tokens, lighter image) and share of GT object stroke length kept (recall the model can still reach).
- light: drop light strokes (luminance > 0.75), whatever the page
- palette: keep only colours (quantised to 1/8 per channel) that carried a GT object on some *train* page
- page oracle: keep the colours that carry GT on this very page (needs the answer: an upper bound, not usable)

    py -3.12 research/probes/color_filter_probe.py datasets/obb_hvac7
"""

import sys
from pathlib import Path

import numpy as np


def load(folder):
    for f in sorted(Path(folder).glob("*.npz")):
        if "_rs" in f.stem:
            continue
        with np.load(f) as z:
            t = z["tokens"].astype(np.float64)
        pts = t[:, 1:17].reshape(-1, 8, 2)
        length = np.linalg.norm(np.diff(pts, axis=1), axis=2).sum(1)
        rgb = np.clip(np.nan_to_num(t[:, 17:20]), 0, 1)
        yield f.stem, rgb, length, t[:, 22] >= 0


def q(rgb):
    return [tuple(c) for c in np.round(rgb * 8).astype(int)]


def main(root):
    root = Path(root)
    palette = set()
    for _, rgb, _, gt in load(root / "vectors" / "train"):
        palette |= set(q(rgb[gt]))
    print(f"train palette: {len(palette)} quantised colours carry GT objects")
    tot = {k: np.zeros(4) for k in ("light", "palette", "page oracle")}
    print(f"{'page':28s} {'strokes':>8s} | light: removed / GT kept | palette: removed / GT kept | page oracle: removed / GT kept")
    for stem, rgb, L, gt in load(root / "vectors" / "val"):
        lum = rgb @ np.array([0.299, 0.587, 0.114])
        keys = q(rgb)
        own = set(q(rgb[gt]))
        rules = {"light": lum <= 0.75,
                 "palette": np.fromiter((k in palette for k in keys), bool, len(keys)),
                 "page oracle": np.fromiter((k in own for k in keys), bool, len(keys))}
        row = []
        for k, keep in rules.items():
            removed, kept = 1 - keep.mean(), L[gt & keep].sum() / max(L[gt].sum(), 1e-9)
            tot[k] += [len(keep), (~keep).sum(), L[gt & keep].sum(), L[gt].sum()]
            row.append(f"{removed:6.1%} / {kept:6.1%}")
        print(f"{stem[:28]:28s} {len(L):8d} | " + " | ".join(row))
    print("all val pages: " + " | ".join(f"{k}: removed {v[1] / v[0]:.1%}, GT length kept {v[2] / v[3]:.1%}" for k, v in tot.items()))


if __name__ == "__main__":
    main(*sys.argv[1:])
