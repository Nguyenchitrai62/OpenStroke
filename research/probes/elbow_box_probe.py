"""Why do GT elbow boxes lose their elbows under the label rule? 0 GPU. GT boxes (min-area 4-point box of each GT polygon) through
the layer-free rule; for each GT elbow, where its stroke length goes: its own box, a Duct box, another class, nothing; and how
much of its length lies inside its own box.

    py -3.12 research/probes/elbow_box_probe.py <score_pack folder>
"""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from matplotlib.path import Path as MPath

from openstroke.obb import object_score as osc
from openstroke.objects import HVAC_CLASSES, assign_objects, class_of, merge6
from openstroke.relabel import STYLE_KEYS, page_segments_of


def main(pack):
    pack = Path(pack)
    E = HVAC_CLASSES.index("Elbow")
    where, inside_share, n = Counter(), [], 0
    took = Counter()
    ious = []
    for r in json.loads((pack / "pages.json").read_text(encoding="utf-8")):
        with np.load(pack / "npz" / f"{r['stem']}.npz") as z:
            d = {k: z[k] for k in z.files}
        segs = page_segments_of(d)
        y, L = d["label"].astype(bool), d["length"].astype(np.float64)
        attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
        attrs["layer_id"] = np.zeros_like(attrs["layer_id"])
        gobj = json.loads((pack / "gt" / f"{r['stem']}.json").read_text(encoding="utf-8"))
        gobj = [dict(o, shape_name=HVAC_CLASSES[merge6(class_of(o["shape_name"]))]) for o in gobj
                if class_of(o.get("shape_name") or "") and merge6(class_of(o["shape_name"])) is not None and len(o.get("vertices") or []) >= 3]
        gcls, ginst, _ = assign_objects(segs, gobj, y)
        keep = []
        for o in gobj:
            v = np.asarray(o["vertices"], np.float32)
            keep.append((HVAC_CLASSES.index(o["shape_name"]), cv2.boxPoints(cv2.minAreaRect(v)).astype(float), 1.0, []))
        pi = osc.boxes_to_strokes("rule", keep, segs, attrs, L)
        mid = (segs.seg[:, :2] + segs.seg[:, 2:]) / 2
        sl = np.linalg.norm(segs.seg[:, 2:] - segs.seg[:, :2], axis=1)
        for g, o in enumerate(gobj):
            if HVAC_CLASSES.index(o["shape_name"]) != E:
                continue
            sel = ginst == g
            if not sel.any():
                continue
            n += 1
            for s in np.flatnonzero(sel):
                p = pi[s]
                k = "nothing" if p < 0 else "own box" if p == g else f"{HVAC_CLASSES[keep[p][0]]} box"
                where[k] += L[s]
            mine = pi == g
            for s_ in np.flatnonzero(mine):
                gi = ginst[s_]
                took["its own elbow" if gi == g else "background" if gi < 0 else f"GT {HVAC_CLASSES[gcls[s_]]}"] += L[s_]
            inter = L[mine & sel].sum(); ious.append(inter / max(L[mine].sum() + L[sel].sum() - inter, 1e-9))
            rows = np.isin(segs.owner, np.flatnonzero(sel))
            ins = MPath(keep[g][1]).contains_points(mid[rows])
            inside_share.append(sl[rows][ins].sum() / max(sl[rows].sum(), 1e-9))
    tot = sum(where.values())
    print(f"{n} GT elbows; their stroke length goes to: " + ", ".join(f"{k} {v / tot:.1%}" for k, v in where.most_common()))
    t2 = sum(took.values())
    print("strokes the rule gives an elbow box: " + ", ".join(f"{k} {v / t2:.1%}" for k, v in took.most_common()))
    print(f"elbow box vs its GT elbow, stroke IoU: median {np.median(ious):.2f}, >= 0.5 for {np.mean(np.array(ious) >= 0.5):.0%}")
    s = np.array(inside_share)
    print(f"share of an elbow's length inside its own box: median {np.median(s):.2f}, >= 0.9 for {np.mean(s >= 0.9):.0%}")


if __name__ == "__main__":
    main(*sys.argv[1:])
