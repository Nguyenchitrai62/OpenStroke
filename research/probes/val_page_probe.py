"""Where is val F2 lost, page by page and class by class? 0 GPU: E69 boxes + label-rule strokes (saved final_json_rule), native GT.
Per page: objects, F2 / R / P, and the missed / wrong-class / merged shares; per class on all val pages: GT count, R, P.

    py -3.12 research/probes/val_page_probe.py <obb_hvac7 folder> <final_json folder>
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from openstroke.assemble import assign_members
from openstroke.objects import object_counts
from openstroke.obb.object_score import PIPELINE_PAGES
from openstroke.obb.vector_validation import DatasetVectorGT, HVAC_CLASSES


def f(m, p, g):
    P, R = m / max(p, 1), m / max(g, 1)
    return 5 * P * R / max(4 * P + R, 1e-9), R, P


def main(folder, preds, conf="0.2"):
    folder, preds, conf = Path(folder), Path(preds), float(conf)
    names = list(yaml.safe_load((folder / "data.yaml").read_text(encoding="utf-8"))["names"].values())
    gt = DatasetVectorGT(folder / "images" / "val", names)
    canon = {n.casefold(): i for i, n in enumerate(HVAC_CLASSES)}
    tot = 0
    print(f"{'page':24s} {'GT':>5s} {'pred':>5s}  F2     R      P    | GT strokes  dominant pred colours")
    for stem, (length, inst, gcls, at) in gt.pages.items():
        if stem[:9] not in PIPELINE_PAGES:
            continue
        keep = []
        for o in json.loads((preds / f"{stem}.json").read_text(encoding="utf-8"))["objects"]:
            if o["score"] >= conf:
                keep.append((canon[o["shape_name"].replace("_", " ").casefold()], np.empty((0, 2)), float(o["score"]),
                             [at[p] for p in o["shapes"] if p in at]))
        pred = assign_members(keep, length, order="score", dedupe=False)
        pcls = np.array([k[0] for k in keep] or [0], np.int64)
        c = object_counts(pred, pcls, inst, gcls, length)
        tot = tot + c
        m, p, g = c.sum(0)
        F, R, P = f(m, p, g)
        print(f"{stem[:24]:24s} {g:5d} {p:5d}  {F:.3f}  {R:.3f}  {P:.3f} | {int((inst >= 0).sum()):6d}")
    print("-- by class (all 7 pages)")
    for k in np.argsort(-tot[:, 2]):
        m, p, g = tot[k]
        if g == 0 and p == 0:
            continue
        F, R, P = f(m, p, g)
        print(f"{HVAC_CLASSES[k][:22]:22s} GT {g:5d} pred {p:5d}  F2 {F:.3f}  R {R:.3f}  P {P:.3f}")
    m, p, g = tot.sum(0)
    print("all: F2 %.3f R %.3f P %.3f" % f(m, p, g))


if __name__ == "__main__":
    main(*sys.argv[1:])
