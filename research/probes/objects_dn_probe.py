"""E62 analysis: how good is AI_Takeoff's final HVAC JSON beyond shapes? On the HVAC GT pages with a stored pipeline run (not ``Y1_10-20.pdf``):
objects matched by shared strokes, then size (``dn``) agreement on matched objects, and the bill of quantities (duct length and fitting count
per class and size) against GT. No model run. AI_Takeoff is only read.

    py -3.12 research/probes/objects_dn_probe.py datasets/e6
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_v3_probe import job_of  # noqa: E402
from openstroke.geometry import IDENTITY, transform_points  # noqa: E402
from openstroke.objects import HVAC_CLASSES, MATCH_IOU, assign_objects, class_of  # noqa: E402
from openstroke.relabel import STYLE_KEYS, object_strokes, page_segments_of  # noqa: E402
from openstroke.sources import load_gt_pages  # noqa: E402

DB = r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3"


def dn_of(o):
    v = o.get("dn") or []
    return "|".join(sorted(str(x).replace(" ", "").lower() for x in (v if isinstance(v, list) else [v]))) or ""


def quantity(o):
    """Duct: length of the long side of its outline (pt); fittings: 1."""

    v = np.asarray(o.get("vertices") or [], float)
    if class_of(o.get("shape_name")) != 1 or len(v) < 3:
        return 1.0
    r = shapely.minimum_rotated_rectangle(shapely.multipoints(v))
    c = shapely.get_coordinates(r)[:4]
    e = np.hypot(*np.diff(np.vstack([c, c[:1]]), axis=0).T)
    return float(e.max())


def main(data):
    man = {r["stem"]: r for r in json.loads((Path(data) / "manifest.json").read_text(encoding="utf-8"))["records"] if r.get("status") == "ok"}
    gt = {p.stem: p for p in load_gt_pages(DB)}
    tot = defaultdict(float)
    boq_gt, boq_pipe = defaultdict(float), defaultdict(float)
    dn_rows = defaultdict(lambda: [0, 0, 0])  # class -> matched, same dn, GT has dn
    for s, rec in sorted(man.items()):
        if rec["mode"] != "hvac" or rec.get("pseudo") or s not in gt or s.startswith("doc000202"):
            continue
        j = job_of(rec["document_id"], "hvac", rec["page_number"])
        if not j:
            continue
        z = np.load(Path(data) / rec["npz"])
        d = {k: z[k] for k in z.files}
        w, h = rec["page_size"]["payload"]
        t = rec.get("gt_transform_applied")
        gobj = [o for o in gt[s].objects if len(o.get("vertices") or [])]
        if t and t != IDENTITY:
            gobj = [{**o, "vertices": transform_points(t, np.asarray(o["vertices"], float), w, h).tolist()} for o in gobj]
        segs = page_segments_of(d)
        y, L = d["label"].astype(bool), d["length"].astype(np.float64)
        gcls, ginst, _ = assign_objects(segs, gobj, y)
        pobj = [o for o in j[0] if o.get("vertices")]
        attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
        pcls, pinst, _ = assign_objects(segs, pobj, object_strokes("hvac", pobj, segs, attrs, L))
        for o in gobj:
            c = class_of(o.get("shape_name"))
            if c:
                boq_gt[(c, dn_of(o))] += quantity(o)
        for o in pobj:
            c = class_of(o.get("shape_name"))
            if c:
                boq_pipe[(c, dn_of(o))] += quantity(o)
        both = (ginst >= 0) & (pinst >= 0)
        key = pinst[both].astype(np.int64) * 100000 + ginst[both]
        u, inv = np.unique(key, return_inverse=True)
        inter = np.bincount(inv, L[both])
        lp = {int(i): float(L[pinst == i].sum()) for i in np.unique(pinst[pinst >= 0])}
        lg = {int(i): float(L[ginst == i].sum()) for i in np.unique(ginst[ginst >= 0])}
        cand = sorted(((a / max(lp[int(k // 100000)] + lg[int(k % 100000)] - a, 1e-9), int(k // 100000), int(k % 100000)) for k, a in zip(u, inter)), reverse=True)
        used_p, used_g = set(), set()
        for iou, p, g in cand:
            if iou < MATCH_IOU or p in used_p or g in used_g or class_of(pobj[p].get("shape_name")) != class_of(gobj[g].get("shape_name")):
                continue
            used_p.add(p)
            used_g.add(g)
            c = class_of(gobj[g].get("shape_name"))
            r = dn_rows[c]
            r[0] += 1
            r[1] += dn_of(pobj[p]) == dn_of(gobj[g])
            r[2] += bool(dn_of(gobj[g]))
        tot["pages"] += 1
    print(f"{int(tot['pages'])} pages")
    print("matched objects: same size as GT / matched, GT objects with a size")
    for c in sorted(dn_rows):
        m, same, has = dn_rows[c]
        print(f"  {HVAC_CLASSES[c]:22s} matched {m:5d}  same size {same / max(m, 1):.3f}  GT has size {has / max(m, 1):.3f}")
    m = np.sum(list(dn_rows.values()), 0)
    print(f"  all: same size {m[1] / max(m[0], 1):.3f}")
    for c in sorted({k[0] for k in boq_gt}):
        keys = {k for k in boq_gt if k[0] == c} | {k for k in boq_pipe if k[0] == c}
        g = sum(boq_gt[k] for k in keys)
        err = sum(abs(boq_gt[k] - boq_pipe[k]) for k in keys)
        print(f"BOQ {HVAC_CLASSES[c]:22s} GT {g:10.0f} {'pt' if c == 1 else 'pcs'}, pipeline {sum(boq_pipe[k] for k in keys):10.0f}; error summed over sizes {err / max(g, 1e-9):.3f} of GT")


if __name__ == "__main__":
    main(*sys.argv[1:])
