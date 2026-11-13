"""F146: where AI_Takeoff's HVAC logic loses objects after the stroke model (0 GPU, no model). A cached logic arm (runs_colab/logic_e8/<stem>/<arm>.json)
against the score pack's GT, with the product scorer's stroke assignment (rule_nl, 6 classes):

GT side, per class: matched (same class, stroke-length IoU >= 0.5) | wrong class (IoU >= 0.5 with another class) | merged (>= 50 % of its strokes
in one prediction that is too big: IoU < 0.5) | split (covered >= 50 % by several predictions, none enough) | missed (< 50 % covered).
Prediction side: matched | wrong class | too big (covers several GT) | part (a piece of one GT) | false (no GT strokes).

    py -3.12 research/probes/logic_error_probe.py <score_pack folder> runs_colab/logic_e8 <arm> [--pages doc000180,...]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.obb.object_score import boxes_to_strokes, merge6_any  # noqa: E402
from openstroke.objects import HVAC_CLASSES, assign_objects, base6, class_of  # noqa: E402
from openstroke.relabel import STYLE_KEYS, page_segments_of  # noqa: E402

RECORDS = []
GT_PAGES = ("doc000180", "doc000187", "doc000189", "doc000200", "doc000201")


def logic_boxes(f, names, scale):
    import cv2
    rows = []
    for o in json.loads(Path(f).read_text(encoding="utf-8"))["objects"]:
        n = (o.get("shape_name") or "").replace("_", " ")
        c = HVAC_CLASSES[class_of(n)] if class_of(n) else None   # exact class rules: "Elbow vuông" is not "Elbow"
        v = o.get("vertices") or []
        if c is None or len(v) < 3:
            continue
        c = HVAC_CLASSES[merge6_any(base6(c))] if base6(c) is not None else None
        if c is None:
            continue
        v = np.asarray(v, float)
        if len(v) != 4:
            v = cv2.boxPoints(cv2.minAreaRect(v.astype(np.float32))).astype(float)
        rows.append((HVAC_CLASSES.index(c), v, 1.0, []))
    return rows


def main(pack, cache, arm, *rest):
    pack, cache = Path(pack), Path(cache)
    pages = rest[rest.index("--pages") + 1].split(",") if "--pages" in rest else GT_PAGES
    names = [n.replace("_", " ") for n in json.loads((pack / "names.json").read_text(encoding="utf-8"))]
    gt_side, pr_side = defaultdict(Counter), defaultdict(Counter)
    for rec in json.loads((pack / "pages.json").read_text(encoding="utf-8")):
        stem = rec["stem"]
        if stem[:9] not in pages or not (cache / stem / f"{arm}.json").exists():
            continue
        with np.load(pack / "npz" / f"{stem}.npz") as z:
            d = {k: z[k] for k in z.files}
        segs = page_segments_of(d)
        y, L = d["label"].astype(bool), d["length"].astype(np.float64)
        attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
        attrs["layer_id"] = np.zeros_like(attrs["layer_id"])
        gobj = json.loads((pack / "gt" / f"{stem}.json").read_text(encoding="utf-8"))
        gobj = [o for o in gobj if not class_of(o.get("shape_name") or "") or base6(o["shape_name"]) is not None]
        gobj = [dict(o, shape_name=HVAC_CLASSES[merge6_any(base6(o["shape_name"]))]) if class_of(o.get("shape_name") or "") else o for o in gobj]
        gcls, ginst, _ = assign_objects(segs, gobj, y)
        keep = logic_boxes(cache / stem / f"{arm}.json", names, rec["scale"])
        pinst = boxes_to_strokes("rule_nl", keep, segs, attrs, L).astype(np.int64)
        pcls = np.array([c for c, _, _, _ in keep] or [0])
        gids = np.unique(ginst[ginst >= 0])
        pids = np.unique(pinst[pinst >= 0])
        lg = {g: L[ginst == g].sum() for g in gids}
        lp = {p: L[pinst == p].sum() for p in pids}
        both = (ginst >= 0) & (pinst >= 0)
        inter = Counter()
        for g, p, w in zip(ginst[both], pinst[both], L[both]):
            inter[(int(g), int(p))] += w
        by_g, by_p = defaultdict(list), defaultdict(list)
        for (g, p), w in inter.items():
            by_g[g].append((p, w))
            by_p[p].append((g, w))
        gclass = {int(g): int(gcls[ginst == g][0]) for g in gids}
        curved = d["op_counts"][:, 1] > 0 if d["op_counts"].shape[1] > 1 else np.zeros(len(L), bool)
        for g in gids:
            g = int(g)
            c = HVAC_CLASSES[gclass[g]]
            ov = sorted(by_g.get(g, []), key=lambda t: -t[1])
            cov = sum(w for _, w in ov) / max(lg[g], 1e-9)
            best = ov[0] if ov else None
            iou = best[1] / max(lg[g] + lp[best[0]] - best[1], 1e-9) if best else 0.0
            if best and iou >= 0.5:
                gt_side[c]["matched" if pcls[best[0]] == gclass[g] else "wrong class"] += 1
            elif best and best[1] >= 0.5 * lg[g]:
                gt_side[c]["merged"] += 1
            elif cov >= 0.5:
                gt_side[c]["split"] += 1
            else:
                gt_side[c]["missed"] += 1
        for p in pids:
            p = int(p)
            c = HVAC_CLASSES[pcls[p]]
            ov = sorted(by_p.get(p, []), key=lambda t: -t[1])
            v = keep[p][1]
            e = np.hypot(*(np.roll(v, -1, 0) - v).T)
            m = pinst == p
            rec_p = {"page": stem[:9], "cls": c, "long": float(max(e[0], e[1])), "short": float(min(e[0], e[1])), "n": int(m.sum()),
                     "curve": float(L[m & curved].sum() / max(L[m].sum(), 1e-9))}
            if not ov:
                out = "false"
            else:
                g, w = ov[0]
                iou = w / max(lg[g] + lp[p] - w, 1e-9)
                rec_p["gt"] = HVAC_CLASSES[gclass[g]]
                out = ("matched" if pcls[p] == gclass[g] else "wrong class") if iou >= 0.5 else                       "too big" if len(ov) > 1 and sum(x for _, x in ov) >= 0.5 * lp[p] else "part"
            pr_side[c][out] += 1
            rec_p["out"] = out
            RECORDS.append(rec_p)
    print(f"{arm} on {', '.join(pages)}")
    for title, side, keys in (("GT objects", gt_side, ("matched", "wrong class", "merged", "split", "missed")),
                              ("Predicted objects", pr_side, ("matched", "wrong class", "too big", "part", "false"))):
        print(f"\n{title}:")
        print(f"  {'class':22s} {'n':>5s} " + " ".join(f"{k:>12s}" for k in keys))
        tot = Counter()
        for c in sorted(side, key=lambda c: -sum(side[c].values())):
            n = sum(side[c].values())
            tot.update(side[c])
            print(f"  {c:22s} {n:5d} " + " ".join(f"{side[c][k] / n:11.1%} " for k in keys))
        n = sum(tot.values())
        print(f"  {'ALL':22s} {n:5d} " + " ".join(f"{tot[k] / max(n, 1):11.1%} " for k in keys))
    if "--dump" in rest:
        Path(rest[rest.index("--dump") + 1]).write_text(json.dumps(RECORDS), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
