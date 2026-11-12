"""Does the page-level class-wise NMS (IoU 0.1, used to join the same object seen by overlapping tiles) also delete or fuse distinct GT
objects? 0 GPU: the GT objects' own 4-point boxes, score 1, through the same nms_poly per class; share of GT objects suppressed (they would be
fused into a neighbour) at several IoU thresholds, per class.

    py -3.12 research/probes/page_nms_probe.py <score_pack folder>
"""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from openstroke.obb.ops.obb import nms_poly
from openstroke.objects import HVAC_CLASSES, base6


def main(pack):
    pack = Path(pack)
    for thr in (0.1, 0.3, 0.5, 0.7):
        tot, sup = Counter(), Counter()
        for r in json.loads((pack / "pages.json").read_text(encoding="utf-8")):
            by = {}
            for o in json.loads((pack / "gt" / f"{r['stem']}.json").read_text(encoding="utf-8")):
                c = base6(o.get("shape_name"))
                if c is None or len(o.get("vertices") or []) < 3:
                    continue
                v = np.asarray(o["vertices"], np.float32)
                by.setdefault(c, []).append(cv2.boxPoints(cv2.minAreaRect(v)).reshape(-1).astype(np.float64))
            for c, polys in by.items():
                keep = nms_poly(np.stack(polys), np.ones(len(polys)), thr)
                tot[c] += len(polys); sup[c] += len(polys) - len(keep)
        all_t, all_s = sum(tot.values()), sum(sup.values())
        print(f"IoU {thr}: {all_s}/{all_t} GT objects suppressed ({all_s / all_t:.1%}) | "
              + ", ".join(f"{HVAC_CLASSES[c]} {sup[c] / tot[c]:.1%}" for c in sorted(tot, key=lambda k: -tot[k])))


if __name__ == "__main__":
    main(*sys.argv[1:])
