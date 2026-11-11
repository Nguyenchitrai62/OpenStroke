"""E69's own stroke classes, regrouped by stroke proximity instead of by box (0 GPU, saved predictions, no inference): every
stroke takes the class of the E69 object that holds it (native assignment, score order), objects = connected components of
same-class strokes within r px. Compares with E69's box grouping on the same strokes.

    py -3.12 research/probes/e69_regroup_probe.py <obb_hvac7 folder> <final_json_members folder> [conf]
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from openstroke.assemble import assign_members
from openstroke.objects import object_counts
from openstroke.obb.object_score import PIPELINE_PAGES
from openstroke.obb.vector_validation import DatasetVectorGT, HVAC_CLASSES


def f2(t):
    m, p, g = t.sum(0)
    P, R = m / max(p, 1), m / max(g, 1)
    return 5 * P * R / max(4 * P + R, 1e-9), R, P


def main(folder, preds, conf="0.2"):
    folder, preds, conf = Path(folder), Path(preds), float(conf)
    names = list(yaml.safe_load((folder / "data.yaml").read_text(encoding="utf-8"))["names"].values())
    gt = DatasetVectorGT(folder / "images" / "val", names)
    canon = {n.casefold(): i for i, n in enumerate(HVAC_CLASSES)}
    box, regroup = 0, {r: 0 for r in (4.0, 8.0, 16.0)}
    for stem, (length, inst, gcls, at) in gt.pages.items():
        if stem[:9] not in PIPELINE_PAGES:
            continue
        with np.load(folder / "vectors" / "val" / f"{stem}.npz") as z:
            pts = np.asarray(z["tokens"])[:, 1:17].reshape(-1, 8, 2)
        keep = []
        for o in json.loads((preds / f"{stem}.json").read_text(encoding="utf-8"))["objects"]:
            if o["score"] >= conf:
                keep.append((canon[o["shape_name"].replace("_", " ").casefold()], np.empty((0, 2)), float(o["score"]),
                             [at[p] for p in o["shapes"] if p in at]))
        pred = assign_members(keep, length, order="score", dedupe=False)
        pcls = np.array([k[0] for k in keep], np.int64)
        box = box + object_counts(pred, pcls, inst, gcls, length)
        rows = np.flatnonzero(pred >= 0)
        cls = pcls[pred[rows]]
        P = pts[rows].reshape(-1, 2)
        owner = np.repeat(np.arange(len(rows)), 8)
        tree = cKDTree(P)
        for r in regroup:
            pairs = tree.query_pairs(r, output_type="ndarray")
            a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
            ok = (a != b) & (cls[a] == cls[b])
            n, lab = connected_components(coo_matrix((np.ones(ok.sum()), (a[ok], b[ok])), shape=(len(rows),) * 2), directed=False)
            p2 = np.full(len(inst), -1, np.int64)
            p2[rows] = lab
            c2 = np.zeros(n, np.int64)
            c2[lab] = cls
            regroup[r] = regroup[r] + object_counts(p2, c2, inst, gcls, length)
    print("E69 box grouping       F2 %.3f  R %.3f  P %.3f" % f2(box))
    for r, t in regroup.items():
        print(f"regroup same class {r:4.0f} px F2 %.3f  R %.3f  P %.3f" % f2(t))


if __name__ == "__main__":
    main(*sys.argv[1:])
