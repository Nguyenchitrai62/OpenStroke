"""Ceiling of ducts built from the stroke graph instead of boxes (OpenStroke10 design check, 0 GPU).

Ducts are the class the box detector loses most (E83: 33 % of GT ducts with no box at IoU 0.5; F81: 16-31 % with no stroke held, 12-17 %
merged with a neighbour). A duct is two parallel walls; GT cuts a run of duct into pieces at fittings, branches and size changes, which a
box cannot see. Here the duct strokes are taken as known (the GT's own, i.e. a perfect per-stroke "duct wall" head) and joined into objects
by geometry only: walls touching end to end, a wall ending on another, and the two parallel walls of one duct. Fitting strokes are not
duct strokes, so a fitting cuts the run. Scored like the product (object F2 of the Duct class, objects matched by shared stroke length).

    py -3.12 research/probes/duct_graph_oracle.py PACK [RULES: E end-end, T end-on-wall, P parallel walls] [WIDTH_PT ...]
"""

import json
import sys
from pathlib import Path

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.objects import N_CLASSES, assign_objects, object_counts  # noqa: E402
from openstroke.relabel import page_segments_of  # noqa: E402

PIPE = ("doc000022", "doc000030", "doc000180", "doc000189", "doc000190", "doc000200", "doc000201")
DUCT = 1


def duct_objects(segs, strokes: np.ndarray, width: float, tol: float = 2.0, angle: float = 5.0, overlap: float = 0.3, rules: str = "ETP") -> np.ndarray:
    """-> object id per stroke (-1 outside ``strokes``): components of the duct-stroke graph."""

    n = len(segs.shape_offsets) - 1
    obj = np.full(n, -1, np.int64)
    m = strokes[segs.owner]
    if not m.any():
        return obj
    s, own = segs.seg[m], segs.owner[m]
    k = len(s)
    a, b = s[:, :2], s[:, 2:]
    pairs = []
    ends = np.r_[a, b]
    eo = np.r_[np.arange(k), np.arange(k)]
    if "E" in rules:
        for i, j in cKDTree(ends).query_pairs(tol, output_type="ndarray"):  # end to end
            pairs.append((eo[i], eo[j]))
    lines = shapely.linestrings(np.stack([a, b], 1))
    tree = shapely.STRtree(lines)
    pi, pj = tree.query(shapely.points(ends), predicate="dwithin", distance=tol)  # an end on another wall (T, end cap)
    if "T" in rules:
        pairs += list(zip(eo[pi], pj))
    d = b - a
    ln = np.maximum(np.hypot(*d.T), 1e-9)
    u = d / ln[:, None]
    qi, qj = tree.query(lines, predicate="dwithin", distance=width)  # the two parallel walls of one duct
    keep = qi < qj
    qi, qj = qi[keep], qj[keep]
    cos = np.abs((u[qi] * u[qj]).sum(1))
    par = cos >= np.cos(np.radians(angle))
    qi, qj = qi[par], qj[par]
    t0 = ((a[qj] - a[qi]) * u[qi]).sum(1)  # projection of wall j on wall i's axis
    t1 = ((b[qj] - a[qi]) * u[qi]).sum(1)
    lo, hi = np.maximum(np.minimum(t0, t1), 0), np.minimum(np.maximum(t0, t1), ln[qi])
    ov = (hi - lo) / np.minimum(ln[qi], ln[qj])
    ok = (ov >= overlap) & (np.minimum(ln[qi], ln[qj]) >= 2 * tol)
    if "P" in rules:
        pairs += list(zip(qi[ok], qj[ok]))
    if pairs:
        p = np.asarray(pairs, np.int64)
        e = (own[p[:, 0]], own[p[:, 1]])
    else:
        e = (np.zeros(0, np.int64), np.zeros(0, np.int64))
    g = coo_matrix((np.ones(len(e[0]), bool), e), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    obj[strokes] = lab[strokes]
    return obj


def main(pack, rules="ETP", *widths):
    pack = Path(pack)
    widths = [float(w) for w in widths] or [20.0, 40.0, 60.0]
    pages = [r for r in json.loads((pack / "pages.json").read_text(encoding="utf-8")) if r["stem"][:9] in PIPE]
    data = []
    for rec in pages:
        z = np.load(pack / "npz" / f"{rec['stem']}.npz")
        d = {k: z[k] for k in z.files}
        segs = page_segments_of(d)
        y, L = d["label"].astype(bool), d["length"].astype(np.float64)
        gobj = json.loads((pack / "gt" / f"{rec['stem']}.json").read_text(encoding="utf-8"))
        gcls, ginst, _ = assign_objects(segs, gobj, y)
        data.append((segs, L, gcls, ginst))
    for w in widths:
        tot = np.zeros((N_CLASSES, 3), np.int64)
        for segs, L, gcls, ginst in data:
            duct = gcls == DUCT
            obj = duct_objects(segs, duct, w, rules=rules)
            ids, inv = np.unique(obj[obj >= 0], return_inverse=True)
            pred = np.full(len(obj), -1, np.int64)
            pred[obj >= 0] = inv
            gi = np.where(duct, ginst, -1)
            gcls_of = {int(i): DUCT for i in np.unique(gi[gi >= 0])}
            tot += object_counts(pred, np.full(len(ids) or 1, DUCT, np.int64), gi.astype(np.int64), gcls_of, L)
        tp, npred, ngt = tot[DUCT]
        p, r = tp / max(npred, 1), tp / max(ngt, 1)
        print(f"rules {rules}, wall pairing <= {w:.0f} pt: Duct objects {npred} vs GT {ngt}; P {p:.3f} R {r:.3f} F2 {5 * p * r / max(4 * p + r, 1e-9):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])


def candidate_edges(segs, strokes: np.ndarray, width: float, tol: float = 2.0, angle: float = 5.0, overlap: float = 0.3) -> np.ndarray:
    """(e, 2) stroke pairs a learnt edge head would score: end to end / end on a wall (<= tol) and parallel walls (<= width)."""

    m = strokes[segs.owner]
    s, own = segs.seg[m], segs.owner[m]
    if not len(s):
        return np.zeros((0, 2), np.int64)
    a, b = s[:, :2], s[:, 2:]
    k = len(s)
    ends = np.r_[a, b]
    eo = np.r_[np.arange(k), np.arange(k)]
    lines = shapely.linestrings(np.stack([a, b], 1))
    tree = shapely.STRtree(lines)
    pi, pj = tree.query(shapely.points(ends), predicate="dwithin", distance=tol)
    d = b - a
    ln = np.maximum(np.hypot(*d.T), 1e-9)
    u = d / ln[:, None]
    qi, qj = tree.query(lines, predicate="dwithin", distance=width)
    keep = qi < qj
    qi, qj = qi[keep], qj[keep]
    par = np.abs((u[qi] * u[qj]).sum(1)) >= np.cos(np.radians(angle))
    qi, qj = qi[par], qj[par]
    t0, t1 = ((a[qj] - a[qi]) * u[qi]).sum(1), ((b[qj] - a[qi]) * u[qi]).sum(1)
    lo, hi = np.maximum(np.minimum(t0, t1), 0), np.minimum(np.maximum(t0, t1), ln[qi])
    ok = ((hi - lo) / np.minimum(ln[qi], ln[qj]) >= overlap) & (np.minimum(ln[qi], ln[qj]) >= 2 * tol)
    e = np.r_[np.c_[own[eo[pi]], own[pj]], np.c_[own[qi[ok]], own[qj[ok]]]]
    e = e[e[:, 0] != e[:, 1]]
    return np.unique(np.sort(e, 1), axis=0)


def edge_oracle(pack, width="40", flip="0"):
    """Objects = components of the candidate edges whose two strokes share a GT object (a perfect edge head), class from the GT
    (a perfect stroke head). ``flip``: share of edge decisions made wrong at random, a rough model error."""

    pack, width, flip = Path(pack), float(width), float(flip)
    rng = np.random.default_rng(0)
    pages = [r for r in json.loads((pack / "pages.json").read_text(encoding="utf-8")) if r["stem"][:9] in PIPE]
    tot = np.zeros((N_CLASSES, 3), np.int64)
    ne = npos = 0
    for rec in pages:
        z = np.load(pack / "npz" / f"{rec['stem']}.npz")
        d = {k: z[k] for k in z.files}
        segs = page_segments_of(d)
        y, L = d["label"].astype(bool), d["length"].astype(np.float64)
        gobj = json.loads((pack / "gt" / f"{rec['stem']}.json").read_text(encoding="utf-8"))
        gcls, ginst, _ = assign_objects(segs, gobj, y)
        obj_strokes = ginst >= 0
        e = candidate_edges(segs, obj_strokes, width)
        same = ginst[e[:, 0]] == ginst[e[:, 1]]
        if flip:
            same = same ^ (rng.random(len(same)) < flip)
        ne, npos = ne + len(e), npos + int(same.sum())
        n = len(L)
        g = coo_matrix((np.ones(int(same.sum()), bool), (e[same, 0], e[same, 1])), shape=(n, n))
        _, lab = connected_components(g, directed=False)
        lab = np.where(obj_strokes, lab, -1)
        ids, inv = np.unique(lab[lab >= 0], return_inverse=True)
        pred = np.full(n, -1, np.int64)
        pred[lab >= 0] = inv
        pcls = np.zeros(len(ids) or 1, np.int64)  # class of a component: the GT class holding most of its length (perfect stroke head)
        for k in range(len(ids)):
            mk = pred == k
            pcls[k] = np.bincount(gcls[mk].astype(np.int64), L[mk], minlength=N_CLASSES).argmax()
        gcls_of = {int(i): int(c) for i, c in zip(ginst[obj_strokes], gcls[obj_strokes])}
        tot += object_counts(pred, pcls, ginst.astype(np.int64), gcls_of, L)
    tp, npred, ngt = tot[1:].sum(0)
    p, r = tp / max(npred, 1), tp / max(ngt, 1)
    print(f"edge oracle, parallel walls <= {width:.0f} pt, {flip:.0%} edges wrong: {ne} candidate edges ({npos / max(ne, 1):.0%} same object); "
          f"objects P {p:.3f} R {r:.3f} F2 {5 * p * r / max(4 * p + r, 1e-9):.3f}")
    from openstroke.objects import HVAC_CLASSES
    print("   " + "; ".join(f"{HVAC_CLASSES[k]} R {tot[k, 0] / max(tot[k, 2], 1):.2f}" for k in range(1, N_CLASSES) if tot[k, 2]))


def knn_edges(segs, strokes: np.ndarray, k: int = 8, radius: float = 40.0) -> np.ndarray:
    """(e, 2) stroke pairs the GPU head can afford: k nearest of the 3 key points (two ends, middle) of each stroke's 8 arc-length points
    (the token's own points), within ``radius`` pt. Same points as the vector tokens (``features.shape_points``)."""

    from openstroke.features import shape_points

    off = segs.shape_offsets
    n = len(off) - 1
    idx = np.flatnonzero(strokes & (np.diff(off) > 0))
    if len(idx) < 2:
        return np.zeros((0, 2), np.int64)
    rel = shape_points(segs.seg, off, 8)
    lo = np.minimum.reduceat(np.minimum(segs.seg[:, :2], segs.seg[:, 2:]), off[:-1].clip(max=len(segs.seg) - 1))
    hi = np.maximum.reduceat(np.maximum(segs.seg[:, :2], segs.seg[:, 2:]), off[:-1].clip(max=len(segs.seg) - 1))
    pts = rel + ((lo + hi) / 2)[:, None, :]
    key = pts[idx][:, [0, 4, 7]].reshape(-1, 2)
    owner = np.repeat(idx, 3)
    dist, nb = cKDTree(key).query(key, k=min(k + 1, len(key)), distance_upper_bound=radius)
    a = np.repeat(owner, nb.shape[1])
    ok = np.isfinite(dist.ravel())
    b = owner[np.minimum(nb.ravel(), len(owner) - 1)]
    e = np.c_[a[ok], b[ok & True][: ok.sum()]] if False else np.c_[a, b][ok]
    e = e[e[:, 0] != e[:, 1]]
    return np.unique(np.sort(e, 1), axis=0)


def knn_oracle(pack, k="8", radius="40"):
    """Edge oracle on the k-NN candidate set (what the model scores), every stroke of the page a candidate (pipe or not)."""

    pack, k, radius = Path(pack), int(k), float(radius)
    pages = [r for r in json.loads((pack / "pages.json").read_text(encoding="utf-8")) if r["stem"][:9] in PIPE]
    tot = np.zeros((N_CLASSES, 3), np.int64)
    ne = npos = 0
    for rec in pages:
        z = np.load(pack / "npz" / f"{rec['stem']}.npz")
        d = {kk: z[kk] for kk in z.files}
        segs = page_segments_of(d)
        y, L = d["label"].astype(bool), d["length"].astype(np.float64)
        gobj = json.loads((pack / "gt" / f"{rec['stem']}.json").read_text(encoding="utf-8"))
        gcls, ginst, _ = assign_objects(segs, gobj, y)
        n = len(L)
        e = knn_edges(segs, np.ones(n, bool), k, radius)
        same = (ginst[e[:, 0]] >= 0) & (ginst[e[:, 0]] == ginst[e[:, 1]])
        ne, npos = ne + len(e), npos + int(same.sum())
        g = coo_matrix((np.ones(int(same.sum()), bool), (e[same, 0], e[same, 1])), shape=(n, n))
        _, lab = connected_components(g, directed=False)
        objs = ginst >= 0
        lab = np.where(objs, lab, -1)
        ids, inv = np.unique(lab[lab >= 0], return_inverse=True)
        pred = np.full(n, -1, np.int64)
        pred[lab >= 0] = inv
        pcls = np.zeros(len(ids) or 1, np.int64)
        for q in range(len(ids)):
            mq = pred == q
            pcls[q] = np.bincount(gcls[mq].astype(np.int64), L[mq], minlength=N_CLASSES).argmax()
        gcls_of = {int(i): int(c) for i, c in zip(ginst[objs], gcls[objs])}
        tot += object_counts(pred, pcls, ginst.astype(np.int64), gcls_of, L)
        print(f"  {rec['stem']}: {n} strokes, {len(e)} edges", flush=True)
    tp, npred, ngt = tot[1:].sum(0)
    p, r = tp / max(npred, 1), tp / max(ngt, 1)
    print(f"k-NN oracle k={k} radius {radius:.0f} pt: {ne} edges ({npos / max(ne, 1):.1%} same object); objects P {p:.3f} R {r:.3f} "
          f"F2 {5 * p * r / max(4 * p + r, 1e-9):.3f}")
