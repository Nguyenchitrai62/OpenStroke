"""E63 / H78: OpenOBB's HVAC boxes on the val pages, turned into vector objects and scored like the pipeline.

Boxes (``val_dets.json`` from ``datasets/obb_hvac*/predict_val.py``) go from px to pt; the label rule (``relabel.object_strokes``) picks the
strokes that make up each box, ``objects.assign_objects`` gives every stroke one object. Scored two ways: objects matched by shared strokes
(the same match as ``objects_oracle_probe.py``, so the pipeline number compares), and boxes matched by polygon IoU >= 0.5. Writes the final
JSON per page (class, polygon, score, the payload shape indices) beside the detections. No model run. AI_Takeoff is only read.

    py -3.12 research/probes/obb_score_probe.py runs_colab/e63 datasets/obb_hvac2 datasets/e6
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.geometry import IDENTITY, transform_points  # noqa: E402
from openstroke.objects import HVAC_CLASSES, N_CLASSES, assign_objects, class_of, object_counts  # noqa: E402
from openstroke.relabel import STYLE_KEYS, object_strokes, page_segments_of  # noqa: E402
from openstroke.sources import load_gt_pages  # noqa: E402

DB = r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3"
CONFS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
PIPELINE_PAGES = ("doc000022", "doc000030", "doc000180", "doc000189", "doc000190", "doc000200", "doc000201")


def box_counts(pred, gt):
    """(N_CLASSES, 3) matched / predicted / GT boxes, polygon IoU >= 0.5, same class, best first."""

    out = np.zeros((N_CLASSES, 3), np.int64)
    for c, _ in pred:
        out[c, 1] += 1
    for c, _ in gt:
        out[c, 2] += 1
    if not pred or not gt:
        return out
    gp = [shapely.Polygon(v).buffer(0) for _, v in gt]
    tree = shapely.STRtree(gp)
    cand = []
    for i, (c, v) in enumerate(pred):
        pp = shapely.Polygon(v).buffer(0)
        for j in tree.query(pp):
            if gt[j][0] != c:
                continue
            inter = pp.intersection(gp[j]).area
            iou = inter / max(pp.union(gp[j]).area, 1e-9)
            if iou >= 0.5:
                cand.append((iou, i, int(j)))
    up, ug = set(), set()
    for _, i, j in sorted(cand, reverse=True):
        if i in up or j in ug:
            continue
        up.add(i)
        ug.add(j)
        out[pred[i][0], 0] += 1
    return out


def f1(c):
    tp, npred, ngt = c[1:].sum(0)
    return 2 * tp / max(npred + ngt, 1), tp / max(npred, 1), tp / max(ngt, 1)


def main(run, obb, built, how="rule", split="val", dets_file="val_dets.json"):
    """``how``: rule = strokes from the label rule on each box; members = the strokes the model itself returned per box (OpenOBB25).
    ``split`` / ``dets_file``: score another split (train pages: does the model fit what it was trained on?)."""
    run, obb, built = Path(run), Path(obb), Path(built)
    dets = json.loads((run / dets_file).read_text(encoding="utf-8"))
    pages = {r["stem"]: r for r in json.loads((obb / "pages.json").read_text(encoding="utf-8"))
             if r["split"] == split and r["stem"] in dets and (built / f"{r['stem']}.npz").exists()}
    names = [ln.split(": ", 1)[1].strip() for ln in (obb / "data.yaml").read_text(encoding="utf-8").splitlines() if ln.startswith("  ") and ": " in ln]
    gt = {p.stem: p for p in load_gt_pages(DB)}
    res = {c: {} for c in CONFS}
    out_dir = run / f"final_json_{how}" if split == "val" else run / f"final_json_{how}_{split}"
    out_dir.mkdir(exist_ok=True)
    for stem, rec in pages.items():
        z = np.load(built / f"{stem}.npz")
        d = {k: z[k] for k in z.files}
        segs = page_segments_of(d)
        y, L = d["label"].astype(bool), d["length"].astype(np.float64)
        attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
        w, h = rec["size_pt"]
        t = rec["transform"]
        gobj = [o for o in gt[stem].objects if len(o.get("vertices") or [])]
        if t != IDENTITY:
            gobj = [{**o, "vertices": transform_points(t, np.asarray(o["vertices"], float), w, h).tolist()} for o in gobj]
        gcls, ginst, _ = assign_objects(segs, gobj, y)
        gcls_of = {int(i): int(c) for i, c in zip(ginst[ginst >= 0], gcls[ginst >= 0])}
        gboxes = []
        for o in gobj:
            k = class_of(o.get("shape_name"))
            v = np.asarray(o["vertices"], float)
            q = shapely.get_coordinates(shapely.minimum_rotated_rectangle(shapely.multipoints(v)))[:4] if len(v) >= 3 else []
            if k and len(q) == 4:
                gboxes.append((k, q))
        allp = [(HVAC_CLASSES.index(names[int(q["class_id"])]), np.asarray(q["poly"], float).reshape(4, 2) / rec["scale"], float(q["score"]), q.get("members") or [])
                for q in dets.get(stem, [])]
        for conf in CONFS:
            keep = [p for p in allp if p[2] >= conf]
            objs = [{"shape_name": HVAC_CLASSES[c], "vertices": v.tolist(), "score": s} for c, v, s, _ in keep]
            if how == "members":  # the model's own strokes; a stroke claimed by two boxes goes to the higher score
                pi = np.full(len(L), -1, np.int64)
                for k in np.argsort([q[2] for q in keep]):
                    m = np.asarray(keep[k][3], np.int64)
                    pi[m[(m >= 0) & (m < len(L))]] = k
                pm = pi >= 0
            else:
                pm = object_strokes("hvac", objs, segs, attrs, L) if objs else np.zeros(len(L), bool)
                pc, pi, _ = assign_objects(segs, objs, pm)
            pcls = np.array([c for c, _, _, _ in keep] or [0], np.int64)
            res[conf][stem] = {"obj": object_counts(pi.astype(np.int64), pcls, ginst, gcls_of, L).tolist(),
                               "box": box_counts([(c, v) for c, v, _, _ in keep], gboxes).tolist(),
                               "stroke": [float(L[pm & y].sum()), float(L[pm & ~y].sum()), float(L[~pm & y].sum())]}
            if conf == 0.25:
                members = {}
                for s_idx in np.flatnonzero(pi >= 0):
                    members.setdefault(int(pi[s_idx]), []).append(int(s_idx))
                final = [{**o, "vertices": np.round(o["vertices"], 2).tolist(), "shapes": members.get(i, [])} for i, o in enumerate(objs) if members.get(i)]
                (out_dir / f"{stem}.json").write_text(json.dumps({"page": stem, "coordinate_system": "display_top_left", "objects": final}, ensure_ascii=False), encoding="utf-8")
        print("scored", stem, len(allp), "detections", flush=True)
    Path(run / f"obb_score_{how}{'' if split == 'val' else '_' + split}.json").write_text(json.dumps({str(k): v for k, v in res.items()}), encoding="utf-8")
    subsets = (("7 pages with a pipeline run", lambda s: s[:9] in PIPELINE_PAGES), ("all val pages", lambda s: True)) if split == "val" else ((f"all {split} pages", lambda s: True),)
    for subset, keyf in subsets:
        print(f"\n{subset}")
        for conf in CONFS:
            r = [v for s, v in res[conf].items() if keyf(s)]
            o, b = np.sum([x["obj"] for x in r], 0), np.sum([x["box"] for x in r], 0)
            st = np.sum([x["stroke"] for x in r], 0)
            fo, po, ro = f1(o)
            fb, _, _ = f1(b)
            low = sum((np.asarray(x["obj"])[1:, 0].sum() / max(np.asarray(x["obj"])[1:, 2].sum(), 1)) < 0.5 for x in r)
            rs, ps = st[0] / max(st[0] + st[2], 1), st[0] / max(st[0] + st[1], 1)
            print(f"  conf {conf:.2f} objects F1 {fo:.3f} (P {po:.3f} R {ro:.3f}) | boxes IoU>=0.5 F1 {fb:.3f} | strokes R {rs:.3f} P {ps:.3f} | pages with object recall < 0.5: {low}")
        best = max(CONFS, key=lambda c: f1(np.sum([v["obj"] for s, v in res[c].items() if keyf(s)], 0))[0])
        o = np.sum([v["obj"] for s, v in res[best].items() if keyf(s)], 0)
        print(f"  per class at conf {best}: " + ", ".join(f"{HVAC_CLASSES[k]} {2 * o[k, 0] / max(o[k, 1] + o[k, 2], 1):.2f} ({o[k, 2]})" for k in range(1, N_CLASSES) if o[k, 2]))


if __name__ == "__main__":
    main(*sys.argv[1:])
