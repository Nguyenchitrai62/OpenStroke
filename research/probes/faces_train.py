"""F148: training set of the face classifier (0 GPU): faces of the GT member strokes on the TRAIN pages of datasets/e8 (split hvac10b), each
matched to a GT object (polygon IoU >= 0.5 -> its 6-class name) or "none". GT objects: AI_Takeoff DB (doc pages), Viewer "Labels V1"
(vw pages). Val pages never enter.

    py -3.12 research/probes/faces_train.py datasets/e8 datasets/obb_hvac10b <labels.json> <out.npz> [--viewer N]
"""
import json
import sys
from pathlib import Path

import numpy as np
from shapely import STRtree
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from openstroke.hvac_faces import FEATURES, face_features, page_faces  # noqa: E402
from openstroke.objects import HVAC_CLASSES, base6, class_of  # noqa: E402
from openstroke.obb.object_score import merge6_any  # noqa: E402
from openstroke.relabel import page_segments_of  # noqa: E402

DB = r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3"
CLASSES = ("none", "Duct", "Elbow", "Elbow vuông", "Concentric Reducer", "Eccentric Reducer", "Chân rẽ")


def name6(n):
    b = base6(n) if class_of(n or "") else None
    return HVAC_CLASSES[merge6_any(b)] if b is not None else None


def main(e8, split, labels_json, out, *rest):
    from openstroke.sources import load_gt_pages
    from viewer_obb_dataset import CLASS_MAP, SET

    n_viewer = int(rest[rest.index("--viewer") + 1]) if "--viewer" in rest else 0
    e8, split = Path(e8), Path(split)
    rows = [r for r in json.loads((split / "pages.json").read_text(encoding="utf-8")) if r["split"] == "train"]
    gt = {p.stem: p for p in load_gt_pages(DB)}
    lab = json.loads(Path(labels_json).read_text(encoding="utf-8"))
    pdf_of = {p["sha256"][:12]: p for p in lab["pdfs"]}
    vw = [r for r in rows if r["stem"].startswith("vw_")][:n_viewer]
    rows = [r for r in rows if not r["stem"].startswith("vw_")] + vw
    X, Y, P = [], [], []
    for r in rows:
        st = r["stem"]
        if st.startswith("vw_"):
            key, page = st[3:].rsplit("_p", 1)
            pg = next(x for x in pdf_of[key]["pages"] if int(x["page"]) == int(page))
            objs = [{"shape_name": CLASS_MAP[o["class"]], "vertices": o["points"]} for o in (pg["labels"].get(SET) or {}).get("objects") or []
                    if o["class"] in CLASS_MAP]
        else:
            objs = [o for o in gt[st].objects if len(o.get("vertices") or []) >= 3]
        g = []
        for o in objs:
            n = name6(o.get("shape_name"))
            if n is None or len(o.get("vertices") or []) < 3:
                continue
            p = Polygon(np.asarray(o["vertices"], float)).buffer(0)
            if p.area > 0:
                g.append((n, p))
        with np.load(e8 / f"{st}.npz") as z:
            d = {k: z[k] for k in ("segments", "shape_offsets", "label", "op_counts")}
        segs = page_segments_of(d)
        picked = d["label"].astype(bool)
        curved = d["op_counts"][:, 1] > 0
        faces = page_faces(segs, picked)
        if not faces:
            continue
        x = face_features(faces, segs, picked, curved)
        tree = STRtree([p for _, p in g]) if g else None
        y = []
        for f in faces:
            best, bn = 0.0, "none"
            if tree is not None:
                for i in tree.query(f, predicate="intersects"):
                    iou = f.intersection(g[i][1]).area / max(f.union(g[i][1]).area, 1e-9)
                    if iou > best:
                        best, bn = iou, g[i][0]
            y.append(CLASSES.index(bn) if best >= 0.5 else 0)
        X.append(x); Y += y; P += [st] * len(y)
        print(f"{st[:22]} faces {len(faces):5d} objects {sum(1 for v in y if v):5d}", flush=True)
    np.savez_compressed(out, X=np.concatenate(X), y=np.asarray(Y), page=np.asarray(P), features=np.asarray(FEATURES), classes=np.asarray(CLASSES))
    print(out, len(Y), "faces", np.bincount(Y, minlength=len(CLASSES)))


if __name__ == "__main__":
    main(*sys.argv[1:])
