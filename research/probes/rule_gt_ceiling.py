"""Ceiling of the decode alone: the GT objects' own boxes (4-point min-area box of each polygon) through the layer-free label rule,
6 classes, product metric. 0 GPU.

    py -3.12 research/probes/rule_gt_ceiling.py <score_pack folder>
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from openstroke.obb import object_score as osc
from openstroke.objects import HVAC6_MERGE


def main(pack):
    pack = Path(pack)
    names = [n.replace("_", " ") for n in json.loads((pack / "names.json").read_text(encoding="utf-8"))]
    dets = {}
    for r in json.loads((pack / "pages.json").read_text(encoding="utf-8")):
        rows = []
        for o in json.loads((pack / "gt" / f"{r['stem']}.json").read_text(encoding="utf-8")):
            from openstroke.objects import HVAC_CLASSES, class_of, merge6
            c = class_of(o.get("shape_name") or "")
            m = HVAC_CLASSES[merge6(c)] if c and merge6(c) is not None else None
            if m is None or m not in names or len(o.get("vertices") or []) < 3:
                continue
            v = np.asarray(o["vertices"], np.float32)
            b = cv2.boxPoints(cv2.minAreaRect(v)) if len(v) != 4 else v
            rows.append({"class_id": names.index(m), "score": 1.0, "members": [], "poly": (np.asarray(b, float) * r["scale"]).ravel().tolist()})
        dets[r["stem"]] = rows
    for how in ("rule_nl",):
        res = osc.score(pack, dets, how, log=lambda _: None, merge=True)
        rep = osc.report(res, how, log=lambda _: None)
        print(f"GT boxes + {how}, 6 classes: F2 %.3f R %.3f P %.3f" % (rep["F2"], rep["R"], rep["P"]))
        per_class(res)


def per_class(res):
    from openstroke.objects import HVAC_CLASSES
    tot = sum(np.asarray(v) for v in res[osc.CONFS[0]].values())
    for k in np.argsort(-tot[:, 2]):
        m, p, g = tot[k]
        if g:
            P, R = m / max(p, 1), m / max(g, 1)
            print(f"  {HVAC_CLASSES[k]:22s} GT {g:5d}  F2 {5 * P * R / max(4 * P + R, 1e-9):.3f}  R {R:.3f}  P {P:.3f}")


if __name__ == "__main__" and len(sys.argv) == 2:
    main(sys.argv[1])


def polygons(pack):
    """Same, with the GT polygons themselves (curved elbows keep their shape) instead of 4-point boxes."""
    from openstroke.objects import HVAC_CLASSES, assign_objects, class_of, merge6, object_counts
    from openstroke.relabel import STYLE_KEYS, page_segments_of
    pack = Path(pack)
    res = {c: {} for c in osc.CONFS}
    for r in json.loads((pack / "pages.json").read_text(encoding="utf-8")):
        stem = r["stem"]
        with np.load(pack / "npz" / f"{stem}.npz") as z:
            d = {k: z[k] for k in z.files}
        segs = page_segments_of(d)
        y, L = d["label"].astype(bool), d["length"].astype(np.float64)
        attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
        attrs["layer_id"] = np.zeros_like(attrs["layer_id"])
        gobj = json.loads((pack / "gt" / f"{stem}.json").read_text(encoding="utf-8"))
        gobj = [o for o in gobj if class_of(o.get("shape_name") or "") and merge6(class_of(o["shape_name"])) is not None
                and len(o.get("vertices") or []) >= 3]
        gobj = [dict(o, shape_name=HVAC_CLASSES[merge6(class_of(o["shape_name"]))]) for o in gobj]
        gcls, ginst, _ = assign_objects(segs, gobj, y)
        gcls_of = {int(i): int(c) for i, c in zip(ginst[ginst >= 0], gcls[ginst >= 0])}
        keep = [(HVAC_CLASSES.index(o["shape_name"]), np.asarray(o["vertices"], float), 1.0, []) for o in gobj]
        pi = osc.boxes_to_strokes("rule", keep, segs, attrs, L)
        pcls = np.array([k[0] for k in keep] or [0], np.int64)
        c = object_counts(pi.astype(np.int64), pcls, ginst, gcls_of, L)
        for conf in osc.CONFS:
            res[conf][stem] = c
    per_class(res)
    rep = osc.report(res, "rule_nl", log=lambda _: None)
    print("GT polygons + layer-free rule, 6 classes: F2 %.3f R %.3f P %.3f" % (rep["F2"], rep["R"], rep["P"]))


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "poly":
    polygons(sys.argv[1])
