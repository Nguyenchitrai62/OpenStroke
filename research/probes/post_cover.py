"""F153: does every stroke the model picked end up in an object? One page through the app's post-processing (AI_Takeoff logic on layer_1
-> shape rules -> faces), then the picked strokes (p >= thr) that no final object covers, by length, and a PNG with them in orange.
0 GPU (probabilities cached by research/probes/post_debug.py).

    py -3.12 research/probes/post_cover.py <page.json.gz> <prob.npy> <logic.json> <faces.joblib> <out.png> [thr=0.3] [x0 y0 x1 y1 scale]
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MUST = "--no-must" not in sys.argv
COL = {"Duct": "#e0143c", "Elbow": "#2ca02c", "Elbow vuông": "#17becf", "Concentric Reducer": "#1f5fd6", "Eccentric Reducer": "#9467bd",
       "Chân rẽ": "#ff7f0e", "Tee": "#8c564b"}


def covered_strokes(objects, segs, n, tol=0.8):
    """Per shape: share of its length whose segment midpoints lie inside (or within tol of) a final object's polygon."""
    import shapely
    from shapely.geometry import Polygon

    polys = [Polygon(np.asarray(o["vertices"], float)).buffer(0) for o in objects if len(o.get("vertices") or []) >= 3]
    if not polys:
        return np.zeros(n)
    u = shapely.union_all(polys).buffer(tol)
    mid = (segs.seg[:, :2] + segs.seg[:, 2:]) / 2
    inside = shapely.contains_xy(u, mid[:, 0], mid[:, 1])
    L = segs.length
    return np.bincount(segs.owner, weights=L * inside, minlength=n) / np.maximum(np.bincount(segs.owner, weights=L, minlength=n), 1e-9)


def final_objects(shapes, segs, attrs, prob, logic, faces_model, thr, faces=True):
    from openstroke.hvac_faces import PICK_THR, load_classifier, post_objects
    from openstroke.hvac_refine import refine

    objs = refine(logic, segs, attrs, prob, 0.05)
    n_logic = len(objs)
    if faces:
        clf = load_classifier(faces_model)
        objs = post_objects(objs, segs, prob, np.asarray(attrs["op_counts"])[:, 1] > 0, *clf, thr=max(thr, PICK_THR), must=MUST,
                            order="items" if "--items" in sys.argv else ("faces" if "--faces-first" in sys.argv else "logic"), graph=True)
    return objs, n_logic


def main(page, prob, logic, faces_model, out, thr="0.3", *crop):
    crop = [c for c in crop if not c.startswith("--")]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon as MPoly

    from openstroke.build import shape_attributes
    from openstroke.geometry import page_segments
    from openstroke.objects import HVAC_CLASSES, class_of

    thr = float(thr)
    pay = json.load(gzip.open(page, "rt", encoding="utf-8"))
    shapes = pay["shapes"]
    segs = page_segments(shapes)
    attrs = shape_attributes(shapes, segs)
    prob = np.load(prob)
    objs, n_logic = final_objects(shapes, segs, attrs, prob, json.loads(Path(logic).read_text(encoding="utf-8"))["objects"], faces_model, thr)
    cov = covered_strokes(objs, segs, len(shapes))
    L = np.bincount(segs.owner, weights=segs.length, minlength=len(shapes))
    pk = prob >= thr
    miss = pk & (cov < 0.5)
    print(f"objects {len(objs)} (logic {n_logic}); picked strokes {int(pk.sum())}, length {L[pk].sum():.0f} pt; "
          f"not in any object: {int(miss.sum())} strokes, {L[miss].sum():.0f} pt = {L[miss].sum() / max(L[pk].sum(), 1e-9):.1%}")
    W, H = float(pay["document_metadata"]["page_width"]), float(pay["document_metadata"]["page_height"])
    x0, y0, x1, y1, sc = [float(v) for v in crop] if crop else (0, 0, W, H, 1.5)
    fig = plt.figure(figsize=((x1 - x0) * sc / 100, (y1 - y0) * sc / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
    s = segs.seg
    own = segs.owner
    ax.add_collection(LineCollection(s[~pk[own]].reshape(-1, 2, 2), colors="#dddddd", linewidths=0.3))
    for o in objs:
        v = np.asarray(o.get("vertices") or [], float)
        if len(v) >= 3:
            n = HVAC_CLASSES[class_of(o.get("shape_name"))] if class_of(o.get("shape_name")) else "?"
            ax.add_patch(MPoly(v, closed=True, fc=COL.get(n, "k"), alpha=0.22, ec=COL.get(n, "k"), lw=0.8))
    ax.add_collection(LineCollection(s[(pk & ~miss)[own]].reshape(-1, 2, 2), colors="#e0143c", linewidths=0.7))
    ax.add_collection(LineCollection(s[miss[own]].reshape(-1, 2, 2), colors="#ff8c00", linewidths=2.2))
    fig.savefig(out); plt.close(fig)
    np.save(Path(out).with_suffix(".miss.npy"), np.flatnonzero(miss))


if __name__ == "__main__":
    main(*sys.argv[1:])
