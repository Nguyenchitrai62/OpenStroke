"""F152: one page through the app's post-processing, every intermediate kept, for finding where objects go wrong (0 GPU; the model on CPU
for one page). Writes <out>/state.npz|json and a full-page PNG: picked strokes (red), logic objects (outline by class), face objects
(filled), faces the classifier dropped (blue dashed).

    py -3.12 research/probes/post_debug.py <page.json.gz> <logic.json> <model.pt> <faces.joblib> <out dir> [x0 y0 x1 y1 scale]
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

COL = {"Duct": "#e0143c", "Elbow": "#2ca02c", "Elbow vuông": "#17becf", "Concentric Reducer": "#1f5fd6", "Eccentric Reducer": "#9467bd",
       "Chân rẽ": "#ff7f0e", "Tee": "#8c564b"}


def main(page, logic, model, faces_model, out, *crop):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPoly

    from openstroke.build import shape_attributes
    from openstroke.geometry import page_segments
    from openstroke.hvac_faces import PICK_THR, face_features, load_classifier, page_faces
    from openstroke.hvac_refine import refine
    from openstroke.objects import HVAC_CLASSES, class_of

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    pay = json.load(gzip.open(page, "rt", encoding="utf-8"))
    shapes = pay["shapes"]
    segs = page_segments(shapes)
    attrs = shape_attributes(shapes, segs)
    pf = out / "prob.npy"
    if pf.exists():
        prob = np.load(pf)
    else:
        from openstroke.infer import Predictor
        prob = Predictor(model, device="cpu").predict_payload(pay, "hvac")["prob"]
        np.save(pf, prob)
    objs = json.loads(Path(logic).read_text(encoding="utf-8"))["objects"]
    objs = refine(objs, segs, attrs, prob, 0.05)
    clf, classes = load_classifier(faces_model)
    curved = np.asarray(attrs["op_counts"])[:, 1] > 0
    picked = prob >= PICK_THR
    faces = page_faces(segs, picked)
    X = face_features(faces, segs, picked, curved)
    pr = clf.predict_proba(X) if len(faces) else np.zeros((0, len(classes)))
    rows = []
    for i, (f, p) in enumerate(zip(faces, pr)):
        k = int(np.argmax(p))
        rows.append({"i": i, "cls": classes[int(clf.classes_[k])], "p": round(float(p[k]), 3),
                     "probs": {classes[int(c)]: round(float(v), 3) for c, v in zip(clf.classes_, p)},
                     "xy": [round(v, 1) for v in f.centroid.coords[0]], "feat": dict(zip(__import__("openstroke.hvac_faces", fromlist=["FEATURES"]).FEATURES, np.round(X[i], 3).tolist()))})
    (out / "faces.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    W, H = float(pay["document_metadata"]["page_width"]), float(pay["document_metadata"]["page_height"])
    x0, y0, x1, y1, sc = [float(v) for v in crop] if crop else (0, 0, W, H, 1.5)
    fig = plt.figure(figsize=((x1 - x0) * sc / 100, (y1 - y0) * sc / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
    s = segs.seg
    m = (np.maximum(s[:, 0], s[:, 2]) >= x0) & (np.minimum(s[:, 0], s[:, 2]) <= x1) & (np.maximum(s[:, 1], s[:, 3]) >= y0) & (np.minimum(s[:, 1], s[:, 3]) <= y1)
    from matplotlib.collections import LineCollection
    pk = picked[segs.owner]
    ax.add_collection(LineCollection(s[m & ~pk].reshape(-1, 2, 2), colors="#cccccc", linewidths=0.3))
    ax.add_collection(LineCollection(s[m & pk].reshape(-1, 2, 2), colors="#e0143c", linewidths=0.8))
    for o in objs:
        v = np.asarray(o.get("vertices") or [], float)
        if len(v) >= 3:
            n = HVAC_CLASSES[class_of(o.get("shape_name"))] if class_of(o.get("shape_name")) else "?"
            ax.add_patch(MPoly(v, closed=True, fill=False, ec=COL.get(n, "k"), lw=1.2))
    for f, r in zip(faces, rows):
        v = np.asarray(f.exterior.coords)
        if r["cls"] == "none" or r["p"] < 0.5:
            ax.add_patch(MPoly(v, closed=True, fill=False, ec="#3355ff", lw=0.6, ls="--"))
        else:
            ax.add_patch(MPoly(v, closed=True, fc=COL.get(r["cls"], "k"), alpha=0.25, ec="none"))
        if x0 <= r["xy"][0] <= x1 and y0 <= r["xy"][1] <= y1 and crop:
            ax.text(*r["xy"], f"{r['i']}:{r['cls'][:6]} {r['p']:.2f}", fontsize=6, color="#000")
    fig.savefig(out / ("crop.png" if crop else "page.png")); plt.close(fig)
    print(len(objs), "logic objects,", len(faces), "faces", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
