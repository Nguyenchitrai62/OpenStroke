"""What stops the model from fitting its own train pages, before any learning? 0 GPU, GT only. Each train page is cut into 1024 px
tiles (200 px overlap) exactly as at inference, with the 8192-token cap (longest strokes kept); a GT object is reachable when >= 50 % of
its stroke length is a model input in at least one tile... and when its whole extent fits in one tile (otherwise no tile shows it all).
Reports the share of GT objects / stroke length the model can never see, per page and in total, with and without min_extent=3.

    py -3.12 research/probes/train_blockers_probe.py <obb_hvac7 folder>
"""
import sys
from pathlib import Path

import numpy as np

from openstroke.obb.data.vectors import cut_tile, page_columns
from openstroke.obb.gt_dataset import GT_SPEC


def windows(n, size=1024, gap=200):
    if n <= size:
        return [0]
    s = list(range(0, n - size, size - gap))
    return s + [n - size]


def main(folder, split="train"):
    folder = Path(folder)
    for min_ext in (0.0, 3.0):
        tot_o = lost_cap = lost_big = 0
        tot_l = seen_l = 0.0
        for r in [r for r in GT_SPEC if r["split"] == split]:
            with np.load(folder / "vectors" / split / f"{r['stem']}.npz") as z:
                page = {k: z[k] for k in z.files}
            t, _, inst, pid = page_columns(page)
            pts = t[:, 1:17].reshape(-1, 8, 2)
            L = np.linalg.norm(np.diff(pts, axis=1), axis=2).sum(1)
            W, H = int(pts[..., 0].max()) + 1, int(pts[..., 1].max()) + 1
            seen = np.zeros(len(t), bool)
            for y0 in windows(H):
                for x0 in windows(W):
                    tok = cut_tile(page, x0, y0, 1024, members=True, max_tokens=8192, min_extent=min_ext)
                    if len(tok):
                        at = {int(p): i for i, p in enumerate(pid)}
                        seen[[at[int(p)] for p in tok[:, 23]]] = True
            for o in np.unique(inst[inst >= 0]):
                s = inst == o
                tot_o += 1; tot_l += L[s].sum(); seen_l += L[s & seen].sum()
                p = pts[s].reshape(-1, 2)
                if max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) > 1024:
                    lost_big += 1
                elif L[s & seen].sum() < 0.5 * L[s].sum():
                    lost_cap += 1
        print(f"{split} min_extent={min_ext:g}: GT objects {tot_o}; never seen (token cap) {lost_cap} ({lost_cap / tot_o:.1%}); "
              f"larger than a tile {lost_big} ({lost_big / tot_o:.1%}); GT stroke length seen {seen_l / tot_l:.1%}")


if __name__ == "__main__":
    main(*sys.argv[1:])
