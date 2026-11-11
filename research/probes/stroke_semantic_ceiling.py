"""Ceiling of a stroke-semantic architecture (0 GPU, GT only): every stroke gets the class of its GT object (perfect per-stroke
semantics), objects = connected components of same-class strokes whose sample points are within r px; scored with the native
object F2 (stroke-length IoU >= 0.5, same class). Shows how much of the object task is solved by per-stroke classes + simple grouping.

    py -3.12 research/probes/stroke_semantic_ceiling.py <obb_hvac7 folder>
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from openstroke.objects import object_counts
from openstroke.obb.object_score import PIPELINE_PAGES
from openstroke.obb.vector_validation import DatasetVectorGT


def main(folder):
    folder = Path(folder)
    names = list(yaml.safe_load((folder / "data.yaml").read_text(encoding="utf-8"))["names"].values())
    gt = DatasetVectorGT(folder / "images" / "val", names)
    for r in (1.0, 4.0, 8.0):
        tot = 0
        for stem, (length, inst, gcls, at) in gt.pages.items():
            if stem[:9] not in PIPELINE_PAGES:
                continue
            with np.load(folder / "vectors" / "val" / f"{stem}.npz") as z:
                pts = np.asarray(z["tokens"])[:, 1:17].reshape(-1, 8, 2)
            rows = np.flatnonzero(inst >= 0)
            cls = np.array([gcls[int(inst[i])] for i in rows])
            P = pts[rows].reshape(-1, 2)
            owner = np.repeat(np.arange(len(rows)), 8)
            pairs = cKDTree(P).query_pairs(r, output_type="ndarray")
            a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
            keep = (a != b) & (cls[a] == cls[b])
            g = coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])), shape=(len(rows), len(rows)))
            n, lab = connected_components(g, directed=False)
            pred = np.full(len(inst), -1, np.int64)
            pred[rows] = lab
            pcls = np.zeros(n, np.int64)
            pcls[lab] = cls
            tot = tot + object_counts(pred, pcls, inst, gcls, length)
        m, p, g_ = tot.sum(0)
        P_, R_ = m / max(p, 1), m / max(g_, 1)
        print(f"r {r:3.0f} px: F2 {5 * P_ * R_ / max(4 * P_ + R_, 1e-9):.3f}  R {R_:.3f}  P {P_:.3f}  ({p} objects for {g_} GT)")


if __name__ == "__main__":
    main(*sys.argv[1:])
