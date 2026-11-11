"""Check of the OpenStroke19 decoder on GT (0 GPU, no model): every stroke gets its GT class with probability 1 (background
strokes: background), predicted centre = the stroke's own centre (offset head still at zero, as at the start of training);
`panoptic_objects` decodes the page and the native scorer gives F2 object on the 7 comparison pages.

    py -3.12 research/probes/panoptic_decode_ceiling.py <obb_hvac7 folder>
"""
import sys
from pathlib import Path

import numpy as np
import yaml

from openstroke.obb.models.stroke_panoptic import PanopticPage, panoptic_objects
from openstroke.obb.object_score import PIPELINE_PAGES
from openstroke.obb.vector_validation import DatasetVectorGT


def main(folder):
    folder = Path(folder)
    names = list(yaml.safe_load((folder / "data.yaml").read_text(encoding="utf-8"))["names"].values())
    gt = DatasetVectorGT(folder / "images" / "val", names, comparison_pages=PIPELINE_PAGES)
    lab = {}
    dets = {}
    for stem in gt.pages:
        with np.load(folder / "vectors" / "val" / f"{stem}.npz") as z:
            tok = np.asarray(z["tokens"])
        rows = [l.split() for l in (folder / "labels" / "val" / f"{stem}.txt").read_text().splitlines() if l.strip()]
        cls_of = [int(r[0]) for r in rows]
        inst, pid = tok[:, 22].astype(int), tok[:, 23].astype(int)
        prob = np.zeros((len(tok), len(names) + 1))
        prob[np.arange(len(tok)), [cls_of[i] if i >= 0 else len(names) for i in inst]] = 1.0
        page = PanopticPage(len(names))
        page.add(pid, prob, tok[:, 1:17].reshape(-1, 8, 2).mean(1))
        dets[stem] = panoptic_objects(page, names, dict(tokens=tok))
    r = gt.evaluate(dets, "panoptic", names)
    print(f"decoder on exact classes, zero offsets: F2 {r['F2']:.3f}  R {r['R']:.3f}  P {r['P']:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
