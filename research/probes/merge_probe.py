"""What the "merged" error bucket is, and its ceiling (F132). 0 GPU: saved E69 predictions (final_json_*) scored against a score pack.

For one decode (members / rule_nl) at one confidence:
  - buckets of the GT objects (matched / wrong class / merged / split / missed), as vector_validation._buckets;
  - for the merged ones: how many GT objects the absorbing prediction covers, and the class pairs (Duct+Duct, Duct+Elbow, ...);
  - oracle ceilings: (a) every prediction split along GT object boundaries (each part keeps the prediction's class),
    (b) same plus the part takes its GT class. (a) - baseline = what perfect instance separation alone would give.

    py -3.12 research/probes/merge_probe.py <score_pack folder> <final_json folder> <how> <conf>
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from openstroke.obb import object_score as osc
from openstroke.objects import ANY_REDUCER, HVAC6_MERGE, HVAC_CLASSES, assign_objects, base6, class_of, object_counts
from openstroke.relabel import STYLE_KEYS, page_segments_of


def f2(c):
    m, p, g = c[:, 0].sum(), c[:, 1].sum(), c[:, 2].sum()
    P, R = m / max(p, 1), m / max(g, 1)
    return 5 * P * R / max(4 * P + R, 1e-9), R, P


def main(pack, preds, how, conf):
    pack, preds, conf = Path(pack), Path(preds), float(conf)
    names = [n.replace("_", " ") for n in json.loads((pack / "names.json").read_text(encoding="utf-8"))]
    canon = {n.replace("_", " "): n for n in HVAC_CLASSES}
    tot = {k: np.zeros((len(HVAC_CLASSES), 3), np.int64) for k in ("base", "split", "split_cls")}
    buckets, sizes, pairs = Counter(), Counter(), Counter()
    miss_cls, miss_page, miss_box, all_cls, miss_len = Counter(), Counter(), Counter(), Counter(), []
    import shapely
    from shapely.geometry import Polygon
    for rec in json.loads((pack / "pages.json").read_text(encoding="utf-8")):
        stem = rec["stem"]
        f = preds / f"{stem}.json"
        if stem[:9] not in osc.PIPELINE_PAGES or not f.exists():
            continue
        with np.load(pack / "npz" / f"{stem}.npz") as z:
            d = {k: z[k] for k in z.files}
        segs = page_segments_of(d)
        y, L = d["label"].astype(bool), d["length"].astype(np.float64)
        attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
        if how == "rule_nl":
            attrs["layer_id"] = np.zeros_like(attrs["layer_id"])
        gobj = json.loads((pack / "gt" / f"{stem}.json").read_text(encoding="utf-8"))
        gobj = [o for o in gobj if not class_of(o.get("shape_name") or "") or base6(o["shape_name"]) is not None]
        gobj = [dict(o, shape_name=HVAC_CLASSES[osc.merge6_any(base6(o["shape_name"]))]) if class_of(o.get("shape_name") or "") else o for o in gobj]
        gcls, ginst, _ = assign_objects(segs, gobj, y)
        gcls_of = {int(i): int(c) for i, c in zip(ginst[ginst >= 0], gcls[ginst >= 0])}
        keep = []
        for o in json.loads(f.read_text(encoding="utf-8"))["objects"]:
            if o["score"] < conf:
                continue
            name = HVAC6_MERGE.get(o["shape_name"].replace("_", " "), o["shape_name"].replace("_", " "))
            if name is None:
                continue
            keep.append((HVAC_CLASSES.index(canon.get(name, name)), np.asarray(o["vertices"], float).reshape(4, 2), o["score"], o["shapes"]))
        pi = osc.boxes_to_strokes(how, keep, segs, attrs, L).astype(np.int64)
        pcls = np.array([c for c, _, _, _ in keep] or [0], np.int64)
        tot["base"] += object_counts(pi, pcls, ginst, gcls_of, L)
        # oracle split: (pred, gt) pairs become objects; strokes of a pred outside any GT stay with the pred
        key = np.where(pi >= 0, pi * (int(ginst.max()) + 2) + (ginst + 1), -1)
        u, inv = np.unique(key[key >= 0], return_inverse=True)
        sp = np.full_like(pi, -1)
        sp[key >= 0] = inv
        owner = u // (int(ginst.max()) + 2)
        gpart = u % (int(ginst.max()) + 2) - 1
        scls = pcls[owner]
        tot["split"] += object_counts(sp, scls, ginst, gcls_of, L)
        scls2 = np.array([gcls_of.get(int(g), int(c)) if g >= 0 else int(c) for g, c in zip(gpart, scls)] or [0], np.int64)
        tot["split_cls"] += object_counts(sp, scls2, ginst, gcls_of, L)
        # buckets + merged anatomy
        lp = np.bincount(pi[pi >= 0], L[pi >= 0], minlength=len(pcls))
        for g in np.unique(ginst[ginst >= 0]):
            sel = ginst == g
            lg = L[sel].sum()
            share = np.bincount(pi[sel & (pi >= 0)], L[sel & (pi >= 0)], minlength=len(pcls))
            iou = share / np.maximum(lp + lg - share, 1e-9)
            b = int(np.argmax(iou))
            if iou[b] >= 0.5:
                buckets["matched" if pcls[b] == gcls_of[int(g)] else "wrong_class"] += 1
            elif share.max() >= 0.5 * lg:
                buckets["merged"] += 1
                p = int(np.argmax(share))
                inside = pi == p
                covered = [int(h) for h in np.unique(ginst[inside & (ginst >= 0)])
                           if L[inside & (ginst == h)].sum() >= 0.5 * L[ginst == h].sum()]
                sizes[min(len(covered), 6)] += 1
                bg = L[inside & (ginst < 0)].sum() / max(lp[p], 1e-9)
                sizes["bg>50%" if bg > 0.5 else "bg<=50%"] += 1
                cl = sorted({HVAC_CLASSES[gcls_of[h]] for h in covered})
                pairs[" + ".join(cl)] += 1
            elif share.sum() >= 0.2 * lg:
                buckets["split_partial"] += 1
            else:
                buckets["missed"] += 1
                c = HVAC_CLASSES[gcls_of[int(g)]]
                miss_cls[c] += 1
                miss_page[stem[:9]] += 1
                miss_len.append(lg)
                v = gobj[int(g)].get("vertices") or []
                gp = Polygon(v).buffer(0) if len(v) >= 3 else None
                best = 0.0
                if gp is not None and gp.area > 0:
                    for _, box, _, _ in keep:
                        bp = Polygon(box).buffer(0)
                        if bp.intersects(gp):
                            best = max(best, bp.intersection(gp).area / gp.area)
                miss_box["box covers >=50% of its area" if best >= 0.5 else "box covers 10-50%" if best >= 0.1 else "no box"] += 1
            all_cls[HVAC_CLASSES[gcls_of[int(g)]]] += 1
    n = sum(buckets.values())
    print(f"{how} conf {conf}: GT objects {n}")
    print("  buckets: " + " | ".join(f"{k} {v / n:.1%}" for k, v in buckets.most_common()))
    print("  merged: GT objects covered by the absorbing prediction: " + ", ".join(f"{k}: {v}" for k, v in sorted(sizes.items(), key=str)))
    print("  merged class sets: " + ", ".join(f"{k} {v}" for k, v in pairs.most_common(8)))
    print("  missed by class (missed / all): " + ", ".join(f"{k} {v}/{all_cls[k]}" for k, v in miss_cls.most_common()))
    print("  missed by page: " + ", ".join(f"{k} {v}" for k, v in miss_page.most_common()))
    print("  missed vs boxes: " + ", ".join(f"{k} {v}" for k, v in miss_box.most_common()))
    if miss_len:
        print(f"  missed stroke length (pt): median {np.median(miss_len):.0f}, p25 {np.percentile(miss_len, 25):.0f}, p75 {np.percentile(miss_len, 75):.0f}")
    for k, lab in (("base", "as predicted"), ("split", "oracle split along GT"), ("split_cls", "oracle split + GT class")):
        F, R, P = f2(tot[k])
        print(f"  F2 {lab:28s} {F:.3f} (R {R:.3f} P {P:.3f})")


if __name__ == "__main__":
    main(*sys.argv[1:])
