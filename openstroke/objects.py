"""HVAC objects straight from the model (E62, docs/OPENSTROKE7.md): every pipe stroke gets the class of the object it draws and the
offset from its centre to that object's centre; strokes whose centres point at the same place are one object.

- ``assign_objects``: the targets, from a page's objects (GT, or the pipeline's for scoring): class, object index, object centre per shape.
- ``decode_objects``: the model's classes and offsets -> objects, each a list of shapes (the drawing commands) with a polygon.
- ``object_counts``: objects matched by the strokes they share (length IoU >= 0.5, same class), per class.

Class names follow AI_Takeoff's HVAC ``shape_name``; rare variants share a class (``CLASS_OF``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .geometry import IDENTITY, PageSegments, transform_points

HVAC_CLASSES = ("other", "Duct", "Elbow", "Elbow vuông", "Concentric Reducer", "Eccentric Reducer", "Reducer vuông tròn",
                "Chân rẽ", "Chân rẽ vuông tròn", "Tee")
# Product scope (user 2026-11-12): the model finds Duct, Elbow, Elbow vuông, Concentric / Eccentric Reducer and Chân rẽ; the
# square-to-round variants are decided by post-processing logic and Tee is out of scope. name -> merged name (None = dropped).
HVAC6_MERGE = {"Reducer vuông tròn": "Concentric Reducer", "Chân rẽ vuông tròn": "Chân rẽ", "Tee": None}
HVAC6_MERGE_IDX = {HVAC_CLASSES.index(a): (HVAC_CLASSES.index(b) if b else None) for a, b in HVAC6_MERGE.items()}


REDUCER_BASES = (HVAC_CLASSES.index("Concentric Reducer"), HVAC_CLASSES.index("Eccentric Reducer"))
ANY_REDUCER = HVAC_CLASSES.index("Reducer vuông tròn")   # square-round is an attribute; the GT does not record its base


def resolve_any_reducer(pred, pred_cls, gt, gt_cls_of, length, any_ids):
    """GT objects in any_ids are square-round reducers whose base (concentric / eccentric) the GT does not record: each takes
    the base of the reducer prediction that holds most of its stroke length, so either base counts as correct for it; with no
    reducer prediction on it, it stays Concentric. -> a copy of gt_cls_of."""
    out = dict(gt_cls_of)
    if not any_ids:
        return out
    for g in any_ids:
        sel = (gt == g) & (pred >= 0)
        if not sel.any():
            continue
        share = np.bincount(pred[sel], length[sel], minlength=len(pred_cls))
        for p in np.argsort(-share):
            if share[p] <= 0:
                break
            if int(pred_cls[p]) in REDUCER_BASES:
                out[int(g)] = int(pred_cls[p])
                break
    return out


def merge6(cls):
    """Canonical class index -> its 6-class index (None = out of scope)."""
    return HVAC6_MERGE_IDX.get(int(cls), int(cls))


CLASS_RULES = (
    (re.compile(r"^Duct$"), 1),
    (re.compile(r"^Elbow vuông"), 3),
    (re.compile(r"^Elbow"), 2),
    (re.compile(r"^((Concentric|Eccentric) )?Reducer vuông tròn"), 6),
    (re.compile(r"^Concentric Reducer"), 4),
    (re.compile(r"^Eccentric Reducer"), 5),
    (re.compile(r"^Chân rẽ vuông tròn"), 8),
    (re.compile(r"^Chân rẽ"), 7),
    (re.compile(r"^Tee"), 9),
)
N_CLASSES = len(HVAC_CLASSES)
OFFSET_SCALE = 100.0  # pt: the model predicts offset / this
SAMPLE_STEP, TOL = 4.0, 2.0  # pt: points along strokes, distance to an object polygon (the label rule's tolerance)
VOTE_EPS = 6.0  # pt: centres closer than this are one object
MATCH_IOU = 0.5


def class_of(name: str | None) -> int:
    for rx, k in CLASS_RULES:
        if rx.search(str(name or "")):
            return k
    return 0


def _polygons(objects: list[dict]):
    keep, polys = [], []
    for i, o in enumerate(objects):
        v = np.asarray(o.get("vertices") or [], float)
        if class_of(o.get("shape_name")) == 0 or len(v) < 3:
            continue
        g = shapely.make_valid(shapely.Polygon(v))
        if g.is_empty or g.area <= 0:
            continue
        keep.append(i)
        polys.append(g)
    return keep, polys


def assign_objects(segs: PageSegments, objects: list[dict], eligible: np.ndarray, tol: float = TOL, step: float = SAMPLE_STEP):
    """-> (cls (n,) int8, inst (n,) int32 index into ``objects`` or -1, centre (n, 2) float32 or NaN) for the ``eligible`` shapes:
    the object holding most of the shape's length (the smallest polygon where they overlap, so a fitting wins over the duct it sits on)."""

    n = len(segs.shape_offsets) - 1
    cls = np.zeros(n, np.int8)
    inst = np.full(n, -1, np.int32)
    ctr = np.full((n, 2), np.nan, np.float32)
    keep, polys = _polygons(objects)
    m = eligible[segs.owner]
    if not keep or not m.any():
        return cls, inst, ctr
    s = segs.seg[m]
    own = segs.owner[m]
    ln = np.hypot(*(s[:, 2:] - s[:, :2]).T)
    k = np.maximum(np.ceil(ln / step).astype(np.int64), 1)
    rep = np.repeat(np.arange(len(s)), k)
    t = (np.arange(len(rep)) - np.repeat(np.cumsum(k) - k, k) + 0.5) / k[rep]
    pts = s[rep, :2] + (s[rep, 2:] - s[rep, :2]) * t[:, None]
    w = (ln / k)[rep]
    hit_pt, hit_poly = shapely.STRtree(polys).query(shapely.points(pts), predicate="dwithin", distance=tol)
    if not len(hit_pt):
        return cls, inst, ctr
    area = shapely.area(np.asarray(polys, dtype=object))
    order = np.lexsort((area[hit_poly], hit_pt))
    first = np.r_[True, hit_pt[order][1:] != hit_pt[order][:-1]]
    p, g = hit_pt[order][first], hit_poly[order][first]
    pair = own[rep[p]].astype(np.int64) * len(polys) + g
    u, inv = np.unique(pair, return_inverse=True)
    tot = np.bincount(inv, w[p])
    sh, ob = u // len(polys), u % len(polys)
    best = np.lexsort((-tot, sh))
    one = np.r_[True, sh[best][1:] != sh[best][:-1]]
    sh, ob = sh[best][one], ob[best][one]
    cen = shapely.get_coordinates(shapely.centroid(np.asarray(polys, dtype=object)))
    inst[sh] = np.asarray(keep)[ob]
    cls[sh] = [class_of(objects[keep[j]].get("shape_name")) for j in ob]
    ctr[sh] = cen[ob]
    return cls, inst, ctr


def decode_objects(xy: np.ndarray, cls: np.ndarray, off: np.ndarray, mask: np.ndarray, eps: float = VOTE_EPS) -> tuple[np.ndarray, np.ndarray]:
    """-> (object id per shape, -1 outside the mask; class per object). ``off`` in pt. Votes of one class closer than ``eps`` chain into one object."""

    obj = np.full(len(xy), -1, np.int64)
    classes = []
    for c in np.unique(cls[mask]):
        if c <= 0:
            continue
        idx = np.flatnonzero(mask & (cls == c))
        v = xy[idx] + off[idx]
        pairs = cKDTree(v).query_pairs(eps, output_type="ndarray")
        lab = connected_components(coo_matrix((np.ones(len(pairs), bool), (pairs[:, 0], pairs[:, 1])), shape=(len(idx), len(idx))), directed=False)[1]
        obj[idx] = lab + len(classes)
        classes.extend([int(c)] * (int(lab.max()) + 1))
    return obj, np.asarray(classes, np.int64)


def object_counts(pred: np.ndarray, pred_cls: np.ndarray, gt: np.ndarray, gt_cls_of: dict, length: np.ndarray) -> np.ndarray:
    """(N_CLASSES, 3): matched, predicted, GT objects per class. A predicted object matches a GT object of its class when the stroke
    length they share is at least half of their union; each object matches once, largest overlap first."""

    out = np.zeros((N_CLASSES, 3), np.int64)
    pids, gids = np.unique(pred[pred >= 0]), np.unique(gt[gt >= 0])
    for p in pids:
        out[pred_cls[p], 1] += 1
    for g in gids:
        out[gt_cls_of[int(g)], 2] += 1
    both = (pred >= 0) & (gt >= 0)
    if not both.any():
        return out
    lp = {int(p): float(length[pred == p].sum()) for p in pids}
    lg = {int(g): float(length[gt == g].sum()) for g in gids}
    key = pred[both] * (int(gt.max()) + 1) + gt[both]
    u, inv = np.unique(key, return_inverse=True)
    inter = np.bincount(inv, length[both])
    cand = []
    for k, a in zip(u, inter):
        p, g = int(k // (int(gt.max()) + 1)), int(k % (int(gt.max()) + 1))
        iou = a / max(lp[p] + lg[g] - a, 1e-9)
        if iou >= MATCH_IOU and pred_cls[p] == gt_cls_of[g]:
            cand.append((iou, p, g))
    used_p, used_g = set(), set()
    for _, p, g in sorted(cand, reverse=True):
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        out[pred_cls[p], 0] += 1
    return out


def object_polygon(seg: np.ndarray, owner: np.ndarray, members: np.ndarray, cls: int) -> list[list[float]]:
    """Outline of an object from its strokes: the smallest rotated rectangle for straight parts, the convex hull for elbows."""

    pts = seg[np.isin(owner, members)].reshape(-1, 2)
    g = shapely.multipoints(pts)
    poly = shapely.convex_hull(g) if HVAC_CLASSES[cls].startswith("Elbow") else shapely.minimum_rotated_rectangle(g)
    return np.round(shapely.get_coordinates(poly)[:-1], 2).tolist() if poly.geom_type == "Polygon" else np.round(pts, 2).tolist()


def _object_page(job: tuple) -> dict:
    src, out, rec, objects = job
    from .relabel import page_segments_of

    z = np.load(src / rec["npz"])
    d = {k: z[k] for k in z.files}
    n = len(d["length"])
    if rec["mode"] == "hvac":
        w, h = rec["page_size"]["payload"]
        t = rec.get("gt_transform_applied")
        if t and t != IDENTITY:
            objects = [{**o, "vertices": transform_points(t, np.asarray(o.get("vertices") or [], float), w, h).tolist()} if len(o.get("vertices") or []) else o for o in objects]
        cls, inst, ctr = assign_objects(page_segments_of(d), objects, d["label"].astype(bool))
    else:
        cls, inst, ctr = np.zeros(n, np.int8), np.full(n, -1, np.int32), np.full((n, 2), np.nan, np.float32)
    tmp = out / f"{rec['stem']}.tmp.npz"
    np.savez_compressed(tmp, **d, ocls=cls, oinst=inst, octr=ctr)
    tmp.replace(out / rec["npz"])
    L = d["length"].astype(np.float64)
    y = d["label"].astype(bool)
    return {"stem": rec["stem"], "mode": rec["mode"], "objects": int(np.unique(inst[inst >= 0]).size), "true_with_class": float(L[y & (cls > 0)].sum() / max(L[y].sum(), 1e-9))}


def object_dataset(src: Path, out: Path, ai_takeoff_be: Path, workers: int = 3, log=print) -> None:
    """``src`` (labels v3b) -> ``out``: the same pages and files plus ``ocls``, ``oinst``, ``octr`` on HVAC pages. AI_Takeoff is only read."""

    from .relabel import _page_objects
    from .sources import load_gt_pages

    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    gt_pages = {p.stem: p for p in load_gt_pages(ai_takeoff_be / "data" / "ai_takeoff.sqlite3")}
    recs = [r for r in manifest["records"] if r.get("status") == "ok"]
    gone = [r["stem"] for r in recs if r["mode"] == "hvac" and not r.get("pseudo") and r["stem"] not in gt_pages]
    if gone:  # the GT was deleted or replaced in AI_Takeoff since ``src`` was built: no object targets for those pages
        log(f"{len(gone)} HVAC pages lost their GT, no object targets: {', '.join(gone[:5])}{' ...' if len(gone) > 5 else ''}")
    jobs = [(src, out, r, _page_objects(r, ai_takeoff_be, gt_pages) if r["mode"] == "hvac" and r["stem"] not in gone else []) for r in recs if not (out / r["npz"]).exists()]
    log(f"{len(recs)} pages, {len(jobs)} to do")
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(_object_page, jobs), 1):
            if r["mode"] == "hvac":
                log(f"[{i}/{len(jobs)}] {r['stem']} {r['objects']} objects, {r['true_with_class']:.3f} of true length has a class")
    for f in src.iterdir():
        target = out / f.name
        if target.exists() or (f.suffix == ".npz" and ".graph." not in f.name) or f.is_dir() or f.name == "manifest.json":
            continue
        try:
            os.link(f, target)
        except OSError:
            shutil.copy2(f, target)
    manifest["objects"] = {"classes": list(HVAC_CLASSES), "rule": "openstroke/objects.py assign_objects", "from": str(src)}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
