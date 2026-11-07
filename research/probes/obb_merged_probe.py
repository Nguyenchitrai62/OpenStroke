"""E75 (0 CU): detector boxes scored with and without the round / square distinctions that AI_Takeoff's own logic takes from layer
names (Chân rẽ vuông tròn 0.91 with real layers, 0.02 with style groups; F76). Box match (IoU >= 0.5, same class) on the 7 val pages.

    py -3.12 research/probes/obb_merged_probe.py runs_colab/e69 datasets/obb_hvac7 0.3
"""

import json
import sys
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.geometry import IDENTITY, transform_points  # noqa: E402
from openstroke.objects import HVAC_CLASSES, class_of  # noqa: E402
from openstroke.sources import load_gt_pages  # noqa: E402

P = ("doc000022", "doc000030", "doc000180", "doc000189", "doc000190", "doc000200", "doc000201")
C = {n: i for i, n in enumerate(HVAC_CLASSES)}
MERGES = {
    "as labelled": {},
    "round = square (Chân rẽ, Elbow, Reducer vuông tròn)": {C["Chân rẽ vuông tròn"]: C["Chân rẽ"], C["Elbow vuông"]: C["Elbow"],
                                                             C["Reducer vuông tròn"]: C["Concentric Reducer"]},
    "+ concentric = eccentric": {C["Chân rẽ vuông tròn"]: C["Chân rẽ"], C["Elbow vuông"]: C["Elbow"], C["Reducer vuông tròn"]: C["Concentric Reducer"],
                                 C["Eccentric Reducer"]: C["Concentric Reducer"]},
}


def main(run, obb, conf):
    run, obb, conf = Path(run), Path(obb), float(conf)
    dets = json.loads((run / "val_dets.json").read_text(encoding="utf-8"))
    names = [l.split(": ", 1)[1].strip() for l in (obb / "data.yaml").read_text(encoding="utf-8").splitlines() if l.startswith("  ")]
    pages = {r["stem"]: r for r in json.loads((obb / "pages.json").read_text(encoding="utf-8")) if r["split"] == "val" and r["stem"][:9] in P}
    gt = {p.stem: p for p in load_gt_pages(r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3")}
    for label, m in MERGES.items():
        tp = npred = ngt = 0
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
                    g.append((m.get(k, k), shapely.Polygon(q).buffer(0)))
            pr = [(m.get(k, k), shapely.Polygon(np.asarray(d["poly"], float).reshape(4, 2) / rec["scale"]).buffer(0))
                  for d in dets[stem] if d["score"] >= conf for k in [HVAC_CLASSES.index(names[int(d["class_id"])])]]
            tree = shapely.STRtree([p for _, p in pr])
            used = set()
            for k, gp in g:
                best, bj = 0, None
                for j in tree.query(gp):
                    if j not in used and pr[j][0] == k:
                        iou = gp.intersection(pr[j][1]).area / max(gp.union(pr[j][1]).area, 1e-9)
                        if iou > best:
                            best, bj = iou, j
                if best >= 0.5:
                    used.add(bj)
                    tp += 1
            npred += len(pr)
            ngt += len(g)
        print(f"{label:55s} box F1 {2 * tp / (npred + ngt):.3f} (P {tp / npred:.3f} R {tp / ngt:.3f})")


if __name__ == "__main__":
    main(*sys.argv[1:4])
