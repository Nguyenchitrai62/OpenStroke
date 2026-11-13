"""Ceiling of "model marks strokes, simple post-processing builds objects" (0 GPU, no model), 6 classes, score pack val pages.
    py -3.12 research/probes/stroke_mask_ceiling.py datasets/openstroke_score_pack_hvac10b.zip [TOL pt]
Strokes = the GT member strokes (perfect mask). Objects = connected components of strokes that touch (endpoint within TOL pt of another
stroke's endpoint), either
  bin : one mask for all classes, object class = majority GT class of its strokes (perfect class)
  cls : one mask per class (perfect per-stroke class), components inside each class
F2 / P / R with the scorer's matcher (stroke-length IoU >= 0.5, same class)."""
import json, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from openstroke.obb.object_score import open_pack, _f, CONFS
from openstroke.objects import HVAC_CLASSES, assign_objects, object_counts, base6, class_of
from openstroke.obb.object_score import merge6_any
from openstroke.relabel import page_segments_of
pack = open_pack(sys.argv[1]); TOL = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
recs = json.loads((pack / "pages.json").read_text(encoding="utf-8"))
comp = json.loads((pack / "comparison.json").read_text(encoding="utf-8")) if (pack / "comparison.json").exists() else {"groups": {"all": [r["stem"][:9] for r in recs]}}

def components(idx, segs, n):
    """union-find over strokes in idx whose segment endpoints come within TOL."""
    parent = {int(i): int(i) for i in idx}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    sel = np.isin(segs.owner, idx)
    P = np.r_[segs.seg[sel][:, :2], segs.seg[sel][:, 2:]]
    own = np.r_[segs.owner[sel], segs.owner[sel]]
    cell = np.floor(P / TOL).astype(np.int64)
    buckets = {}
    for (cx, cy), o in zip(map(tuple, cell), own):
        buckets.setdefault((cx, cy), set()).add(int(o))
    for (cx, cy), os_ in list(buckets.items()):
        group = set(os_)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                group |= buckets.get((cx + dx, cy + dy), set())
        g = list(group); r0 = find(g[0])
        for o in g[1:]:
            r = find(o)
            if r != r0:
                parent[r] = r0
    lab = np.full(n, -1, np.int64)
    roots = {}
    for i in idx:
        lab[int(i)] = roots.setdefault(find(int(i)), len(roots))
    return lab

tot = {}
for rec in recs:
    st = rec["stem"]
    with np.load(pack / "npz" / f"{st}.npz") as z:
        d = {k: z[k] for k in z.files}
    segs = page_segments_of(d); y = d["label"].astype(bool); L = d["length"].astype(np.float64); n = len(L)
    gobj = json.loads((pack / "gt" / f"{st}.json").read_text(encoding="utf-8"))
    gobj = [o for o in gobj if not class_of(o.get("shape_name") or "") or base6(o["shape_name"]) is not None]
    gobj = [dict(o, shape_name=HVAC_CLASSES[merge6_any(base6(o["shape_name"]))]) if class_of(o.get("shape_name") or "") else o for o in gobj]
    gcls, ginst, _ = assign_objects(segs, gobj, y)
    gcls_of = {int(i): int(c) for i, c in zip(ginst[ginst >= 0], gcls[ginst >= 0])}
    member = np.flatnonzero(ginst >= 0)
    res = {}
    # bin: components over all member strokes, class = length-majority GT class
    lab = components(member, segs, n)
    pcls = np.zeros(max(lab.max() + 1, 1), np.int64)
    for p in range(lab.max() + 1):
        m = lab == p
        pcls[p] = np.bincount(gcls[m], weights=L[m]).argmax()
    res["bin"] = object_counts(lab, pcls, ginst, gcls_of, L)
    # cls: components inside each class
    lab2 = np.full(n, -1, np.int64); pc2 = []
    for c in np.unique(gcls[member]):
        idx = member[gcls[member] == c]
        l = components(idx, segs, n)
        k = l >= 0
        lab2[k] = l[k] + len(pc2); pc2 += [int(c)] * (l.max() + 1)
    res["cls"] = object_counts(lab2, np.array(pc2 or [0]), ginst, gcls_of, L)
    tot[st] = res
for g, pages in comp["groups"].items():
    for arm in ("bin", "cls"):
        c = sum(v[arm] for s, v in tot.items() if s[:9] in pages)
        f2, f1, p, r = _f(c)
        print(f"TOL {TOL} {g:6s} {arm}: F2 {f2:.3f} P {p:.3f} R {r:.3f}")
for s, v in tot.items():
    print("   ", s[:22], " ".join(f"{a} {_f(v[a])[0]:.3f}" for a in ("bin", "cls")))
