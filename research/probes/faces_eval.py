"""F148: the face post-processing scored like the logic arms (0 GPU). A classifier fitted on the TRAIN faces (research/probes/faces_train.py)
names every face of a run's picks on the val pages; a kept face becomes one object (its min-area box, score = class probability), scored
with the pack's rule_nl decode, 6 classes, per group GT4 (no doc187) / GT5 / viewer8 - the same protocol as runs_colab/e114/refine_scores.txt.

    py -3.12 research/probes/faces_eval.py <score_pack> <faces_train.npz> <val_scores.npz> [thr=0.05] [--viewer-train]
        [--base runs_colab/logic_e8 e114_style0.05_rthin]   # logic objects kept, faces only where the logic left nothing
        [--no-faces]                                         # the base arm alone (same groups / scorer)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.hvac_faces import face_features, page_faces  # noqa: E402
from openstroke.obb import object_score as osc  # noqa: E402
from openstroke.relabel import page_segments_of  # noqa: E402

GROUPS = (("GT4 (no doc187)", ("doc000180", "doc000189", "doc000200", "doc000201")),
          ("GT5", ("doc000180", "doc000187", "doc000189", "doc000200", "doc000201")))


def fit(train, viewer=False):
    from sklearn.ensemble import HistGradientBoostingClassifier

    z = np.load(train)
    keep = np.ones(len(z["y"]), bool) if viewer else ~np.char.startswith(z["page"].astype(str), "vw_")
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    clf.fit(z["X"][keep], z["y"][keep])
    return clf, [str(c) for c in z["classes"]]


def page_objects(clf, classes, segs, picked, curved):
    import cv2

    faces = page_faces(segs, picked)
    if not faces:
        return []
    pr = clf.predict_proba(face_features(faces, segs, picked, curved))
    out = []
    for f, p in zip(faces, pr):
        k = int(np.argmax(p))
        c = classes[clf.classes_[k]]
        if c == "none":
            continue
        box = cv2.boxPoints(cv2.minAreaRect(np.asarray(f.exterior.coords, np.float32)))
        out.append((c, float(p[k]), box.astype(float), f))
    return out


COVERED = 0.3   # a face whose area is more than this share inside the logic's objects is the logic's


def logic_rows(f, names, scale):
    import cv2
    from shapely.geometry import Polygon

    from openstroke.objects import HVAC_CLASSES, class_of

    rows, polys = [], []
    for o in __import__("json").loads(Path(f).read_text(encoding="utf-8"))["objects"]:
        n = (o.get("shape_name") or "").replace("_", " ")
        c = HVAC_CLASSES[class_of(n)] if class_of(n) else None
        v = np.asarray(o.get("vertices") or [], float)
        if c is None or c not in names or len(v) < 3:
            continue
        polys.append(Polygon(v).buffer(0))
        if len(v) != 4:
            v = cv2.boxPoints(cv2.minAreaRect(v.astype(np.float32))).astype(float)
        rows.append({"class_id": names.index(c), "score": 1.0, "members": [], "poly": (v * scale).ravel().tolist()})
    return rows, polys


def main(pack, train, scores, thr="0.05", *rest):
    import shapely

    pack = Path(pack)
    base = (Path(rest[rest.index("--base") + 1]), rest[rest.index("--base") + 2]) if "--base" in rest else None
    clf, classes = fit(train, "--viewer-train" in rest)
    names = [n.replace("_", " ") for n in json.loads((pack / "names.json").read_text(encoding="utf-8"))]
    sc = np.load(scores)
    dets = {}
    for r in json.loads((pack / "pages.json").read_text(encoding="utf-8")):
        st = r["stem"]
        if f"{st}__index" not in sc.files:   # a pack wider than the scores (all pages vs val_scores): skipped
            continue
        with np.load(pack / "npz" / f"{st}.npz") as z:
            d = {k: z[k] for k in z.files}
        segs = page_segments_of(d)
        p = np.zeros(len(d["label"])); p[sc[f"{st}__index"]] = sc[st]
        objs = [] if "--no-faces" in rest else page_objects(clf, classes, segs, p >= float(thr), d["op_counts"][:, 1] > 0)
        rows, polys = logic_rows(base[0] / st / f"{base[1]}.json", names, r["scale"]) if base else ([], [])
        cover = shapely.union_all(polys) if polys else None
        rows += [{"class_id": names.index(c), "score": s, "members": [], "poly": (b * r["scale"]).ravel().tolist()}
                 for c, s, b, f in objs if c in names and (cover is None or f.intersection(cover).area <= COVERED * f.area)]
        dets[st] = rows
        print(f"{st[:22]} objects {len(dets[st])}", flush=True)
    res = osc.score(pack, dets, "rule_nl", log=lambda _: None, merge=True)
    parts = []
    for g, pages in GROUPS:
        rep = osc.report(res, "rule_nl", log=lambda _: None, comparison_pages=set(pages), reference=False)
        parts.append(f"{g} {rep['F2']:.3f} (R {rep['R']:.3f} P {rep['P']:.3f}, conf {rep['best_conf']})")
        if g == "GT5":
            per = rep["per_class"]
    vw = {k[:9] for k in dets if k.startswith("vw_")}
    rep = osc.report(res, "rule_nl", log=lambda _: None, comparison_pages=vw, reference=False)
    parts.append(f"viewer8 {rep['F2']:.3f} (R {rep['R']:.3f} P {rep['P']:.3f})")
    print(f"faces@{thr}" + (f" + {base[1]}" if base else "") + " " + " | ".join(parts))
    print("  GT5 per class: " + " | ".join(f"{c} P {v['P']:.2f} R {v['R']:.2f} ({v['tp']}/{v['pred']}/{v['gt']})" for c, v in per.items()))


if __name__ == "__main__":
    main(*sys.argv[1:])
