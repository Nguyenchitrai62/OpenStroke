"""Sensitivity of "per-stroke class + same-class components" to stroke errors (0 GPU). GT member strokes with injected noise:
    py -3.12 research/probes/stroke_mask_noise.py datasets/openstroke_score_pack_hvac10b.zip
miss p (object stroke -> background), flip p (object stroke -> another class), fp p (p x #object strokes background strokes, among
those within 3 pt of an object stroke, get a random class). Object F2 with the scorer's matcher. Seed 0, TOL 0.5 pt."""
import json, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
sys.argv = [sys.argv[0], sys.argv[1], "0.5"] + sys.argv[2:]
import importlib.util
spec = importlib.util.spec_from_file_location("smc", str(__import__("pathlib").Path(__file__).resolve().parent / "stroke_mask_ceiling.py"))
src = open(spec.origin, encoding="utf-8").read().split("tot = {}")[0]
g = {"__file__": spec.origin}; exec(compile(src, spec.origin, "exec"), g)
pack, components = g["pack"], g["components"]
from openstroke.obb.object_score import _f
from openstroke.objects import HVAC_CLASSES, assign_objects, object_counts, base6, class_of
from openstroke.obb.object_score import merge6_any
from openstroke.relabel import page_segments_of
from shapely import STRtree
from openstroke.labels import segment_lines
recs = json.loads((pack / "pages.json").read_text(encoding="utf-8"))
comp = json.loads((pack / "comparison.json").read_text(encoding="utf-8"))
pages = []
for rec in recs:
    st = rec["stem"]
    with np.load(pack / "npz" / f"{st}.npz") as z:
        d = {k: z[k] for k in z.files}
    segs = page_segments_of(d); y = d["label"].astype(bool); L = d["length"].astype(np.float64)
    gobj = json.loads((pack / "gt" / f"{st}.json").read_text(encoding="utf-8"))
    gobj = [o for o in gobj if not class_of(o.get("shape_name") or "") or base6(o["shape_name"]) is not None]
    gobj = [dict(o, shape_name=HVAC_CLASSES[merge6_any(base6(o["shape_name"]))]) if class_of(o.get("shape_name") or "") else o for o in gobj]
    gcls, ginst, _ = assign_objects(segs, gobj, y)
    gcls_of = {int(i): int(c) for i, c in zip(ginst[ginst >= 0], gcls[ginst >= 0])}
    mem = np.flatnonzero(ginst >= 0)
    # background strokes near object strokes (candidates for false positives)
    lines = segment_lines(segs.seg)
    tree = STRtree(lines[np.isin(segs.owner, mem)])
    near_seg = np.unique(tree.query(lines, predicate="dwithin", distance=3.0)[0])
    near = np.setdiff1d(np.unique(segs.owner[near_seg]), mem)
    pages.append((st, segs, L, gcls, ginst, gcls_of, mem, near))
classes = sorted({int(c) for p in pages for c in p[3][p[6]]})
def run(kind, p):
    rng = np.random.default_rng(0)
    tot = {}
    for st, segs, L, gcls, ginst, gcls_of, mem, near in pages:
        n = len(L); cls = np.full(n, -1, np.int64); cls[mem] = gcls[mem]
        if kind == "miss":
            cls[mem[rng.random(len(mem)) < p]] = -1
        elif kind == "flip":
            f = mem[rng.random(len(mem)) < p]
            cls[f] = [rng.choice([c for c in classes if c != cls[i]]) for i in f]
        elif kind == "fp" and len(near):
            k = min(len(near), int(round(p * len(mem))))
            add = rng.choice(near, k, replace=False)
            cls[add] = rng.choice(classes, k)
        lab = np.full(n, -1, np.int64); pc = []
        for c in classes:
            idx = np.flatnonzero(cls == c)
            if not len(idx):
                continue
            l = components(idx, segs, n); k = l >= 0
            lab[k] = l[k] + len(pc); pc += [c] * (l.max() + 1)
        tot[st] = object_counts(lab, np.array(pc or [0]), ginst, gcls_of, L)
    return {gname: _f(sum(v for s, v in tot.items() if s[:9] in pg)) for gname, pg in comp["groups"].items()}
print("noise   p     GT F2 (P / R)          viewer F2 (P / R)")
for kind in ("miss", "flip", "fp"):
    for p in (0.02, 0.05, 0.10, 0.20):
        r = run(kind, p)
        print(f"{kind:5s} {p:5.2f}   " + "   ".join(f"{r[k][0]:.3f} ({r[k][2]:.2f} / {r[k][3]:.2f})" for k in ("GT", "viewer")), flush=True)
