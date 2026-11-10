"""Score OpenOBB val detections against a self-contained pack (no AI_Takeoff DB): the same numbers as research/probes/obb_score_probe.py.

Runs on Colab after ``tools/make_score_pack.py`` built the pack on the local machine (GT objects already in page coordinates, the e6 stroke
arrays of the 8 val pages, the val page records and class names). Prints object F1 / P / R per confidence and per-class P / R at the best
confidence, for the model's own strokes (``members``) and for the label rule on each box (``rule``), on the 7 pages that have a pipeline run.

    python obb_score_pack.py PACK_DIR VAL_DETS_JSON [members,rule]
"""

import json
import sys
from pathlib import Path

import numpy as np

from openstroke.objects import HVAC_CLASSES, N_CLASSES, assign_objects, object_counts
from openstroke.relabel import STYLE_KEYS, object_strokes, page_segments_of

CONFS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
PIPELINE_PAGES = ("doc000022", "doc000030", "doc000180", "doc000189", "doc000190", "doc000200", "doc000201")
REF = {"members": "E69 members F2 0.513 (P 0.460 R 0.528)", "rule": "E69 rule F2 0.584 (P 0.571 R 0.588)"}
PIPELINE = "pipeline with layers F2 0.888 (P 0.814 R 0.908)"


def score(pack: Path, dets: dict, how: str) -> dict:
    names = json.loads((pack / "names.json").read_text(encoding="utf-8"))
    res = {c: {} for c in CONFS}
    for rec in json.loads((pack / "pages.json").read_text(encoding="utf-8")):
        stem = rec["stem"]
        if stem not in dets:
            print("no detections for", stem)
            continue
        z = np.load(pack / "npz" / f"{stem}.npz")
        d = {k: z[k] for k in z.files}
        segs = page_segments_of(d)
        y, L = d["label"].astype(bool), d["length"].astype(np.float64)
        attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
        gobj = json.loads((pack / "gt" / f"{stem}.json").read_text(encoding="utf-8"))
        gcls, ginst, _ = assign_objects(segs, gobj, y)
        gcls_of = {int(i): int(c) for i, c in zip(ginst[ginst >= 0], gcls[ginst >= 0])}
        allp = [(HVAC_CLASSES.index(names[int(q["class_id"])]), np.asarray(q["poly"], float).reshape(4, 2) / rec["scale"], float(q["score"]),
                 q.get("members") or []) for q in dets[stem]]
        for conf in CONFS:
            keep = [p for p in allp if p[2] >= conf]
            objs = [{"shape_name": HVAC_CLASSES[c], "vertices": v.tolist(), "score": s} for c, v, s, _ in keep]
            if how == "members":  # the model's own strokes; a stroke claimed by two boxes goes to the higher score
                pi = np.full(len(L), -1, np.int64)
                for k in np.argsort([q[2] for q in keep]):
                    m = np.asarray(keep[k][3], np.int64)
                    pi[m[(m >= 0) & (m < len(L))]] = k
            else:
                pm = object_strokes("hvac", objs, segs, attrs, L) if objs else np.zeros(len(L), bool)
                _, pi, _ = assign_objects(segs, objs, pm)
            pcls = np.array([c for c, _, _, _ in keep] or [0], np.int64)
            res[conf][stem] = object_counts(pi.astype(np.int64), pcls, ginst, gcls_of, L)
        print(f"  {how}: scored {stem} ({len(allp)} detections)", flush=True)
    return res


def report(res: dict, how: str) -> None:
    def f1(c):
        tp, npred, ngt = c[1:].sum(0)
        return 2 * tp / max(npred + ngt, 1), tp / max(npred, 1), tp / max(ngt, 1)

    def f2(c):  # recall weighted twice (AI_Takeoff evaluate.fbeta, beta = 2)
        _, p, r = f1(c)
        return 5 * p * r / max(4 * p + r, 1e-9)

    tot = {c: np.sum([v for s, v in res[c].items() if s[:9] in PIPELINE_PAGES], 0) for c in CONFS}
    print(f"\n=== {how}: object F1 on the 7 pipeline pages | reference {REF[how]}; {PIPELINE}")
    for c in CONFS:
        fo, po, ro = f1(tot[c])
        print(f"  conf {c:.2f}  F2 {f2(tot[c]):.3f}  F1 {fo:.3f}  P {po:.3f}  R {ro:.3f}")
    best = max(CONFS, key=lambda c: f2(tot[c]))
    o = tot[best]
    fo, po, ro = f1(o)
    print(f"  best conf {best} (by F2): F2 {f2(o):.3f}  F1 {fo:.3f} (P {po:.3f} R {ro:.3f})")
    for k in range(1, N_CLASSES):
        tp, npred, ngt = o[k]
        if ngt or npred:
            p, r = tp / max(npred, 1), tp / max(ngt, 1)
            print(f"    {HVAC_CLASSES[k]:20s} P {p:.2f} R {r:.2f} F2 {5 * p * r / max(4 * p + r, 1e-9):.2f}  (tp {tp}, pred {npred}, gt {ngt})")


if __name__ == "__main__":
    pack, dets_file = Path(sys.argv[1]), Path(sys.argv[2])
    hows = (sys.argv[3] if len(sys.argv) > 3 else "members,rule").split(",")
    dets = json.loads(dets_file.read_text(encoding="utf-8"))
    for how in hows:
        report(score(pack, dets, how), how)
