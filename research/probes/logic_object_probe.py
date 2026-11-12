"""AI_Takeoff's HVAC geometry logic as the object builder, scored with the product metric (object F2, layer-free label rule on the
boxes, 6 classes). 0 GPU, no logic run here: reads the logic outputs cached by research/probes/logic_hvac_probe.py
(runs_colab/logic_hvac2/<stem>/<arm>.json): gt = logic on the GT stroke mask, ctrl = logic on the pipeline's own layers,
e69_c0.3 / e37_* = logic on a model's mask. Pages without an arm are skipped and listed.

    py -3.12 research/probes/logic_object_probe.py <score_pack folder> runs_colab/logic_hvac2 [arm ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

from openstroke.obb import object_score as osc
from openstroke.objects import HVAC_CLASSES


def main(pack, cache, *arms):
    pack, cache = Path(pack), Path(cache)
    names = [n.replace("_", " ") for n in json.loads((pack / "names.json").read_text(encoding="utf-8"))]
    recs = json.loads((pack / "pages.json").read_text(encoding="utf-8"))
    for arm in arms or ("gt", "ctrl", "stored", "e69_c0.3", "e37_union0.1"):
        dets, missing = {}, []
        for r in recs:
            f = cache / r["stem"] / f"{arm}.json"
            if not f.exists():
                missing.append(r["stem"][:9]); continue
            rows = []
            for o in json.loads(f.read_text(encoding="utf-8"))["objects"]:
                n = (o.get("shape_name") or "").replace("_", " ")
                c = next((h for h in HVAC_CLASSES[1:] if n == h or n.startswith(h + " ") or n.startswith(h)), None)
                v = o.get("vertices") or []
                if c is None or c not in names or len(v) < 3:
                    continue
                v = np.asarray(v, float)
                lo, hi = v.min(0), v.max(0)
                if len(v) != 4:   # the pack scorer takes 4-point boxes: the min-area box of the polygon
                    import cv2
                    v = cv2.boxPoints(cv2.minAreaRect(v.astype(np.float32))).astype(float)
                rows.append({"class_id": names.index(c), "score": 1.0, "members": [], "poly": (v * r["scale"]).ravel().tolist()})
            dets[r["stem"]] = rows
        if not dets:
            print(f"{arm}: no cached pages"); continue
        res = osc.score(pack, dets, "rule_nl", log=lambda _: None, merge=True)
        rep = osc.report(res, "rule_nl", log=lambda _: None)
        print(f"{arm:14s} 6-class object F2 {rep['F2']:.3f}  R {rep['R']:.3f}  P {rep['P']:.3f}  pages {len(dets)}"
              + (f" (missing {','.join(missing)})" if missing else ""))


if __name__ == "__main__":
    main(*sys.argv[1:])
