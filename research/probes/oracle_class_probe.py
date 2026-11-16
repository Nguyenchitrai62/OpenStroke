"""F160: ceiling of "the model also gives each stroke its class" (0 GPU). The closed regions of the picked strokes (the app's order:
items -> end-touching regions, openstroke.hvac_faces.priority_candidates) are named by a length-weighted vote of the GT class of the
picked strokes on their outline (a region with no GT-class stroke on its outline is dropped); the logic's objects fill the rest. Scored
like faces_eval (rule_nl, 6 classes, GT4 / GT5 / viewer8). Compare with the shape classifier in the same order (faces_eval --post --items).

    py -3.12 research/probes/oracle_class_probe.py <score_pack> <val_scores.npz> <logic cache> <arm> [thr=0.3] [--no-logic]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from faces_eval import GROUPS, post_rows  # noqa: E402
from openstroke.obb import object_score as osc  # noqa: E402
from openstroke.relabel import page_segments_of  # noqa: E402


def gt_stroke_class(pack, st, segs, y):
    from openstroke.objects import HVAC_CLASSES, assign_objects, base6, class_of

    gobj = json.loads((pack / "gt" / f"{st}.json").read_text(encoding="utf-8"))
    keep = []
    for o in gobj:
        n = o.get("shape_name") or ""
        if str(n).startswith("Tee"):
            keep.append(dict(o, shape_name="Tee"))
        elif class_of(n) and base6(n) is not None:
            keep.append(dict(o, shape_name=HVAC_CLASSES[osc.merge6_any(base6(n))]))
    gcls, _, _ = assign_objects(segs, keep, y)
    return np.asarray(gcls)


def main(pack, scores, cache, arm, thr="0.3", *rest):
    import shapely
    from shapely.geometry import Polygon

    from openstroke.hvac_faces import priority_candidates
    from openstroke.objects import HVAC_CLASSES

    pack, cache, thr = Path(pack), Path(cache), float(thr)
    names = [n.replace("_", " ") for n in json.loads((pack / "names.json").read_text(encoding="utf-8"))]
    sc = np.load(scores)
    dets = {}
    for r in json.loads((pack / "pages.json").read_text(encoding="utf-8")):
        st = r["stem"]
        if f"{st}__index" not in sc.files:
            continue
        with np.load(pack / "npz" / f"{st}.npz") as z:
            d = {k: z[k] for k in z.files}
        segs = page_segments_of(d)
        y = d["label"].astype(bool)
        p = np.zeros(len(y)); p[sc[f"{st}__index"]] = sc[st]
        picked = p >= thr
        gcls = gt_stroke_class(pack, st, segs, y)
        regions, _, _ = priority_candidates(segs, picked, curved=d["op_counts"][:, 1] > 0)
        m = np.flatnonzero(picked[segs.owner])
        lines = shapely.linestrings(segs.seg[m].reshape(-1, 2, 2))
        tree = shapely.STRtree(lines)
        L = segs.length[m]
        cls_seg = gcls[segs.owner[m]]
        objs = []
        for g in regions:
            near = tree.query(g.exterior.buffer(0.8), predicate="intersects")
            if not len(near):
                continue
            on = shapely.length(shapely.intersection(lines[near], g.exterior.buffer(0.8)))
            votes = np.bincount(cls_seg[near], weights=on, minlength=len(HVAC_CLASSES))
            votes[0] = 0
            if votes.sum() <= 0.3 * on.sum():                     # mostly non-GT strokes on its outline: not an object
                continue
            objs.append({"shape_name": HVAC_CLASSES[int(np.argmax(votes))], "face": g, "score": 1.0, "refined": "face",
                         "vertices": np.asarray(g.exterior.coords)[:-1].tolist()})
        if "--no-logic" not in rest:
            fu = shapely.union_all([o["face"] for o in objs]) if objs else None
            for o in json.loads((cache / st / f"{arm}.json").read_text(encoding="utf-8"))["objects"]:
                v = o.get("vertices") or []
                if len(v) >= 3:
                    gg = Polygon(np.asarray(v, float)).buffer(0)
                    if gg.area > 0 and (fu is None or gg.intersection(fu).area <= 0.3 * gg.area):
                        objs.append(o)
        dets[st] = post_rows(objs, names, r["scale"])
        print(f"{st[:22]} regions {len(regions)} objects {len(dets[st])}", flush=True)
    res = osc.score(pack, dets, "rule_nl", log=lambda _: None, merge=True)
    parts = []
    for g, pages in GROUPS:
        rep = osc.report(res, "rule_nl", log=lambda _: None, comparison_pages=set(pages), reference=False)
        parts.append(f"{g} {rep['F2']:.3f} (R {rep['R']:.3f} P {rep['P']:.3f})")
        if g == "GT5":
            per = rep["per_class"]
    rep = osc.report(res, "rule_nl", log=lambda _: None, comparison_pages={k[:9] for k in dets if k.startswith("vw_")}, reference=False)
    parts.append(f"viewer8 {rep['F2']:.3f} (R {rep['R']:.3f} P {rep['P']:.3f})")
    print("oracle class vote" + (" (no logic)" if "--no-logic" in rest else " + logic") + ": " + " | ".join(parts))
    print("  GT5 per class: " + " | ".join(f"{c} P {v['P']:.2f} R {v['R']:.2f} ({v['tp']}/{v['pred']}/{v['gt']})" for c, v in per.items()))


if __name__ == "__main__":
    main(*sys.argv[1:])
