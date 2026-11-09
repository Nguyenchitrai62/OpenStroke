"""E77 (0 CU): where a detector loses objects. Every GT object of a split gets one outcome against the boxes at ``conf`` (rotated IoU of the
GT polygon's min rectangle with the boxes): found with the right class (IoU >= 0.5), found with another class, partly covered
(best IoU 0.1-0.5: split / merged / misplaced), missed (< 0.1). Every box gets one too: correct, wrong class, partial, or on nothing. Ducts are
also split by length. No model run.

    py -3.12 research/probes/obb_error_probe.py runs_colab/e69 datasets/obb_hvac7 0.3 [val|train] [val_dets.json]
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.geometry import IDENTITY, transform_points  # noqa: E402
from openstroke.objects import HVAC_CLASSES, class_of  # noqa: E402
from openstroke.sources import load_gt_pages  # noqa: E402

P = ("doc000022", "doc000030", "doc000180", "doc000189", "doc000190", "doc000200", "doc000201")
BINS = (0, 20, 40, 80, 160, 320, 1e9)


def main(run, obb, conf, split="val", dets_file="val_dets.json"):
    run, obb, conf = Path(run), Path(obb), float(conf)
    dets = json.loads((run / dets_file).read_text(encoding="utf-8"))
    names = [l.split(": ", 1)[1].strip() for l in (obb / "data.yaml").read_text(encoding="utf-8").splitlines() if l.startswith("  ")]
    keep = (lambda s: s[:9] in P) if split == "val" else (lambda s: s in dets and s[:9] not in ("doc000211", "doc000214"))
    pages = {r["stem"]: r for r in json.loads((obb / "pages.json").read_text(encoding="utf-8")) if r["split"] == split and keep(r["stem"])}
    gt = {p.stem: p for p in load_gt_pages(r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3")}
    out_gt = {k: Counter() for k in range(1, len(HVAC_CLASSES))}
    out_box = {k: Counter() for k in range(1, len(HVAC_CLASSES))}
    confusion = Counter()
    duct = [[0, 0, 0] for _ in BINS[:-1]]  # GT, right class found, any box
    for stem, rec in pages.items():
        w, h = rec["size_pt"]
        g = []
        for o in gt[stem].objects:
            k = class_of(o.get("shape_name"))
            v = np.asarray(o.get("vertices") or [], float)
            if not k or len(v) < 3:
                continue
            if rec["transform"] != IDENTITY:
                v = transform_points(rec["transform"], v, w, h)
            q = shapely.get_coordinates(shapely.minimum_rotated_rectangle(shapely.multipoints(v)))[:4]
            if len(q) == 4:
                g.append((k, shapely.Polygon(q).buffer(0), max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1]))))
        pr = [(HVAC_CLASSES.index(names[int(d["class_id"])]), shapely.Polygon(np.asarray(d["poly"], float).reshape(4, 2) / rec["scale"]).buffer(0))
              for d in dets[stem] if d["score"] >= conf]
        gtree, ptree = shapely.STRtree([x for _, x, _ in g]), shapely.STRtree([x for _, x in pr])

        def best(poly, tree, items, cls=None):
            b, bc = 0.0, None
            for j in tree.query(poly):
                if cls is not None and items[j][0] != cls:
                    continue
                o = poly.intersection(items[j][1]).area / max(poly.union(items[j][1]).area, 1e-9)
                if o > b:
                    b, bc = o, items[j][0]
            return b, bc

        for k, gp, L in g:
            same, _ = best(gp, ptree, pr, k)
            anyb, ac = best(gp, ptree, pr)
            if same >= 0.5:
                r = "found"
            elif anyb >= 0.5:
                r = "wrong class"
                confusion[(k, ac)] += 1
            elif anyb >= 0.1:
                r = "partial (split / merged / shifted)"
            else:
                r = "missed"
            out_gt[k][r] += 1
            if k == 1:
                b = int(np.searchsorted(BINS, L, side="right") - 1)
                duct[b][0] += 1
                duct[b][1] += r == "found"
                duct[b][2] += anyb >= 0.5
        for k, pp in pr:
            same, _ = best(pp, gtree, g and [(c, x) for c, x, _ in g], k)
            anyb, _ = best(pp, gtree, [(c, x) for c, x, _ in g])
            out_box[k]["correct" if same >= 0.5 else "wrong class" if anyb >= 0.5 else "partial" if anyb >= 0.1 else "on nothing in GT"] += 1
    print(f"{run.name} {split}, conf {conf}: GT objects")
    for k, c in out_gt.items():
        n = sum(c.values())
        if n:
            print(f"  {HVAC_CLASSES[k]:20s} {n:5d}  " + "  ".join(f"{r} {c[r] / n:.2f}" for r in ("found", "wrong class", "partial (split / merged / shifted)", "missed")))
    tot = sum((Counter(c) for c in out_gt.values()), Counter())
    n = sum(tot.values())
    print(f"  {'all':20s} {n:5d}  " + "  ".join(f"{r} {tot[r] / n:.2f}" for r in ("found", "wrong class", "partial (split / merged / shifted)", "missed")))
    print("boxes")
    tb = sum((Counter(c) for c in out_box.values()), Counter())
    m = sum(tb.values())
    print(f"  {'all':20s} {m:5d}  " + "  ".join(f"{r} {tb[r] / m:.2f}" for r in ("correct", "wrong class", "partial", "on nothing in GT")))
    print("top confusions (GT -> predicted):", ", ".join(f"{HVAC_CLASSES[a]} -> {HVAC_CLASSES[b]} {v}" for (a, b), v in confusion.most_common(6)))
    print("ducts by length (pt): " + ", ".join(f"{BINS[i]:g}-{BINS[i + 1]:g}: {d[0]} GT, found {d[1] / max(d[0], 1):.2f}" for i, d in enumerate(duct)))


if __name__ == "__main__":
    main(*sys.argv[1:6])
