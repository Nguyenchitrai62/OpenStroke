"""F148: how many GT objects a face of the strokes would give (0 GPU, no model). Faces of the GT member strokes (perfect mask) or of a run's
picks, matched to the score pack's GT objects by polygon area IoU (>= 0.5), 6 classes.

    py -3.12 research/probes/faces_ceiling.py <score_pack folder> [val_scores.npz thr]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from shapely import STRtree
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.hvac_faces import page_faces  # noqa: E402
from openstroke.objects import HVAC_CLASSES, base6, class_of  # noqa: E402
from openstroke.obb.object_score import merge6_any  # noqa: E402
from openstroke.relabel import page_segments_of  # noqa: E402


def gt_objects(pack, stem):
    out = []
    for o in json.loads((pack / "gt" / f"{stem}.json").read_text(encoding="utf-8")):
        b = base6(o.get("shape_name")) if class_of(o.get("shape_name") or "") else None
        v = o.get("vertices") or []
        if b is None or len(v) < 3:
            continue
        p = Polygon(np.asarray(v, float)).buffer(0)
        if p.area > 0:
            out.append((HVAC_CLASSES[merge6_any(b)], p))
    return out


def main(pack, scores=None, thr="0.05"):
    pack = Path(pack)
    sc = np.load(scores) if scores else None
    per, n_faces, faces_hit = defaultdict(Counter), 0, 0
    for rec in json.loads((pack / "pages.json").read_text(encoding="utf-8")):
        st = rec["stem"]
        with np.load(pack / "npz" / f"{st}.npz") as z:
            d = {k: z[k] for k in ("segments", "shape_offsets", "label")}
        segs = page_segments_of(d)
        if sc is not None:
            p = np.zeros(len(d["label"])); p[sc[f"{st}__index"]] = sc[st]
            picked = p >= float(thr)
        else:
            picked = d["label"].astype(bool)
        faces = page_faces(segs, picked)
        gts = gt_objects(pack, st)
        tree = STRtree(faces) if faces else None
        used = set()
        for c, g in gts:
            best = 0.0
            if tree is not None:
                for i in tree.query(g, predicate="intersects"):
                    f = faces[i]
                    iou = f.intersection(g).area / max(f.union(g).area, 1e-9)
                    if iou > best:
                        best, bi = iou, i
            ok = best >= 0.5
            per[st[:9]][c, ok] += 1
            if ok:
                used.add(bi)
        n_faces += len(faces); faces_hit += len(used)
        hit = sum(v for (c, ok), v in per[st[:9]].items() if ok); tot = sum(per[st[:9]].values())
        print(f"{st[:22]} GT {tot:5d} matched by a face {hit / max(tot, 1):6.1%} | faces {len(faces):5d}", flush=True)
    for grp, pages in (("GT", ("doc000180", "doc000187", "doc000189", "doc000200", "doc000201")), ("viewer", None)):
        agg = Counter()
        for k, v in per.items():
            if (pages is None and k.startswith("vw_")) or (pages is not None and k in pages):
                agg.update(v)
        cls = sorted({c for c, _ in agg})
        print(f"\n{grp}: " + " | ".join(f"{c} {agg[c, True] / max(agg[c, True] + agg[c, False], 1):.1%} of {agg[c, True] + agg[c, False]}" for c in cls))
        tot = sum(agg.values()); print(f"  all {sum(v for (c, ok), v in agg.items() if ok) / max(tot, 1):.1%} of {tot}")
    print(f"faces {n_faces}, faces matched to a GT object {faces_hit}")


if __name__ == "__main__":
    main(*sys.argv[1:])
