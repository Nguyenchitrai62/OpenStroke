"""OpenStroke34 (F186 / F187): pieces of strokes, the unit an object can be built from.

One PDF item (a shape) often draws several objects in one run: a CAD polyline goes along a duct's wall, round an elbow and on along the
next duct (19 % of the true shapes, 64 % of the true length of the GT pages draw 2+ GT objects, F186). A piece is a run of a shape's
segments cut wherever an object can end (user 2026-11-18: at the corners / meeting points of the smallest drawing commands):
  - where the chain breaks (a gap between two consecutive segments) or the shape changes;
  - at a corner (turn >= CORNER_DEG between consecutive segments);
  - where a straight run meets an arc (an arc = consecutive chords of similar length turning a little at each vertex: a bezier is sampled
    into geometry.BEZIER_STEPS segments);
  - where another stroke ends on it (a vertex touched by another stroke's end, or a point of a segment's body: the joint line between two
    ducts ends on their common wall, which CAD often draws as one segment).
On the 24 GT pages pieces of the true shapes draw 2+ GT objects on 0.1 % of the true length (items: 61.9 %), for 2.8 pieces per item (F187).
Pieces carry their own geometry (resampled points, length, curved flag) and their shape; labels come from piece_objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CORNER_DEG = 30.0          # a turn this sharp between consecutive segments ends a piece
SMOOTH_DEG = (0.3, 35.0)   # turns of an arc's sampled vertices
ARC_LEN_RATIO = 3.0        # an arc's consecutive chords differ by less than this; a straight segment next to it is much longer
SNAP = 0.5                 # pt: another stroke's end this close to a vertex / a segment's body cuts the piece there
GAP = 1e-3                 # pt: consecutive segments farther apart than this are not one chain
PIECE_POINTS = 8           # points resampled along each piece
EDGE_K = 4                 # links per piece end: its nearest other ends within SNAP


@dataclass
class Pieces:
    shape: np.ndarray      # (P,) shape index of each piece
    pts: np.ndarray        # (P, PIECE_POINTS, 2) points resampled along the piece, page frame
    length: np.ndarray     # (P,) pt
    curved: np.ndarray     # (P,) bool
    seg_start: np.ndarray  # (P,) first (sub)segment row in ``sub``
    seg_stop: np.ndarray   # (P,) one past the last
    sub: np.ndarray        # (S, 4) the (sub)segments of all pieces, page frame
    edges: np.ndarray      # (E, 2) pieces that meet: ends within SNAP of each other (consecutive pieces of a shape included)

    def __len__(self) -> int:
        return len(self.shape)


def _empty() -> Pieces:
    z = np.zeros(0)
    return Pieces(z.astype(np.int64), np.zeros((0, PIECE_POINTS, 2)), z, z.astype(bool), z.astype(np.int64), z.astype(np.int64),
                  np.zeros((0, 4)), np.zeros((0, 2), np.int64))


def _chain_ends(seg: np.ndarray, owner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start / end points of every chain (rows grouped by owner, broken at gaps) -> (points (K, 2), owner (K,))."""
    if not len(seg):
        return np.zeros((0, 2)), np.zeros(0, np.int64)
    brk = np.r_[True, (owner[1:] != owner[:-1]) | (np.hypot(seg[1:, 0] - seg[:-1, 2], seg[1:, 1] - seg[:-1, 3]) > GAP)]
    starts = np.flatnonzero(brk)
    stops = np.r_[starts[1:] - 1, len(seg) - 1]
    return np.r_[seg[starts, :2], seg[stops, 2:]], np.r_[owner[starts], owner[stops]]


def _resample(poly: np.ndarray, k: int = PIECE_POINTS) -> np.ndarray:
    d = np.r_[0.0, np.cumsum(np.hypot(*(np.diff(poly, axis=0)).T))]
    if d[-1] <= 0:
        return np.repeat(poly[:1], k, 0)
    t = np.linspace(0.0, d[-1], k)
    return np.stack([np.interp(t, d, poly[:, 0]), np.interp(t, d, poly[:, 1])], 1)


def shape_pieces(segs, shapes, others=None, snap: float = SNAP) -> Pieces:
    """Pieces of ``shapes`` (shape indices). ``others``: shape indices whose chain ends may cut them (default: ``shapes``)."""
    import shapely
    from scipy.spatial import cKDTree

    shapes = np.unique(np.asarray(shapes, np.int64))
    others = shapes if others is None else np.unique(np.asarray(others, np.int64))
    if not len(shapes):
        return _empty()
    off = segs.shape_offsets
    rows = np.concatenate([np.arange(off[s], off[s + 1]) for s in shapes]) if len(shapes) else np.zeros(0, np.int64)
    if not len(rows):
        return _empty()
    seg = segs.seg[rows]
    own = segs.owner[rows]
    orow = np.concatenate([np.arange(off[s], off[s + 1]) for s in others])
    ends, end_own = _chain_ends(segs.seg[orow], segs.owner[orow])
    # 1. cut segment bodies where another stroke's end lies on them (T)
    L0 = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    cuts = [[] for _ in range(len(seg))]
    if len(ends):
        lines = shapely.linestrings(seg.reshape(-1, 2, 2))
        pi, li = shapely.STRtree(lines).query(shapely.points(ends), predicate="dwithin", distance=snap)
        for p, k in zip(pi, li):
            if end_own[p] == own[k] or L0[k] <= 2 * snap:
                continue
            a, b = seg[k, :2], seg[k, 2:]
            t = float(np.dot(ends[p] - a, b - a) / (L0[k] * L0[k]))
            if snap / L0[k] < t < 1 - snap / L0[k]:
                cuts[k].append(t)
    sub, sub_own = [], []
    for k in range(len(seg)):
        if cuts[k]:
            ts = np.r_[0.0, np.unique(np.round(cuts[k], 6)), 1.0]
            a, b = seg[k, :2], seg[k, 2:]
            q = a + ts[:, None] * (b - a)
            sub.append(np.c_[q[:-1], q[1:]])
            sub_own.append(np.full(len(ts) - 1, own[k]))
        else:
            sub.append(seg[k:k + 1])
            sub_own.append(own[k:k + 1])
    q = np.concatenate(sub)
    qo = np.concatenate(sub_own)
    n = len(q)
    d = q[:, 2:] - q[:, :2]
    L = np.hypot(d[:, 0], d[:, 1])
    # 2. vertex tests between consecutive rows
    same = qo[1:] == qo[:-1]
    connected = same & (np.hypot(q[1:, 0] - q[:-1, 2], q[1:, 1] - q[:-1, 3]) <= GAP)
    na, nb = L[:-1] + 1e-12, L[1:] + 1e-12
    turn = np.degrees(np.arccos(np.clip((d[:-1] * d[1:]).sum(1) / (na * nb), -1.0, 1.0)))
    ratio = nb / na
    smooth = connected & (turn > SMOOTH_DEG[0]) & (turn < SMOOTH_DEG[1]) & (ratio < ARC_LEN_RATIO) & (ratio > 1 / ARC_LEN_RATIO)
    # a row is curved when it lies in a run of >= 2 smooth vertices: (left & left-left) | (left & right) | (right & right-right)
    sv = np.r_[False, False, smooth, False, False]         # sv[i + 2] = smooth vertex i (between rows i and i + 1)
    left, ll = sv[1:n + 1], sv[0:n]                         # vertex i - 1, i - 2 for row i
    right, rr = sv[2:n + 2], sv[3:n + 3]                    # vertex i, i + 1
    curved = (left & ll) | (left & right) | (right & rr)
    touched = np.zeros(max(n - 1, 0), bool)
    if len(ends) and n > 1:
        tree = cKDTree(ends)
        for v, hit in enumerate(tree.query_ball_point(q[1:, :2], snap)):
            if hit and connected[v] and np.any(end_own[hit] != qo[v]):
                touched[v] = True
    cut = ~connected | (turn >= CORNER_DEG) | (curved[:-1] != curved[1:]) | touched
    bounds = np.r_[0, np.flatnonzero(cut) + 1, n]
    a_, b_ = bounds[:-1], bounds[1:]
    length = np.add.reduceat(L, a_)
    cur = np.add.reduceat(curved.astype(float), a_) / (b_ - a_) >= 0.5
    pts = np.stack([_resample(np.r_[q[a:b, :2], q[b - 1:b, 2:]]) for a, b in zip(a_, b_)])
    starts, stops = q[a_, :2], q[b_ - 1, 2:]
    e_pts = np.r_[starts, stops]
    P = len(a_)
    # each end links to its EDGE_K nearest other ends within snap (junctions of many strokes would otherwise form dense cliques)
    dist, nb = cKDTree(e_pts).query(e_pts, k=EDGE_K + 1, distance_upper_bound=snap)
    src = np.repeat(np.arange(len(e_pts)), EDGE_K + 1)
    ok = np.isfinite(dist.ravel()) & (nb.ravel() < len(e_pts)) & (nb.ravel() != src)
    pairs = np.c_[src[ok], nb.ravel()[ok]]
    if len(pairs):
        pe = np.sort(pairs % P, axis=1)
        pe = np.unique(pe[pe[:, 0] != pe[:, 1]], axis=0)
    else:
        pe = np.zeros((0, 2), np.int64)
    return Pieces(qo[a_].astype(np.int64), pts, length, cur, a_.astype(np.int64), b_.astype(np.int64), q, pe.astype(np.int64))


def piece_objects(pc: Pieces, objects: list[dict], tol: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """-> (object index (P,) or -1, share of the piece's length on that object): each (sub)segment goes to the object whose outline is
    nearest its midpoint (within tol; the smaller object on a tie, F180), the piece to the object holding most of its length."""
    import shapely
    from shapely.geometry import Polygon

    from .objects import EDGE_TIE

    P = len(pc)
    obj = np.full(P, -1, np.int64)
    share = np.zeros(P)
    polys, idx = [], []
    for i, o in enumerate(objects):
        v = o.get("vertices") or []
        if len(v) >= 3:
            g = Polygon(np.asarray(v, float)).buffer(0)
            if g.area > 0:
                polys.append(g)
                idx.append(i)
    if not P or not polys:
        return obj, share
    idx = np.asarray(idx)
    area = np.asarray([g.area for g in polys])
    mid = (pc.sub[:, :2] + pc.sub[:, 2:]) / 2
    sl = np.hypot(pc.sub[:, 2] - pc.sub[:, 0], pc.sub[:, 3] - pc.sub[:, 1])
    pi, gi = shapely.STRtree(polys).query(shapely.points(mid), predicate="dwithin", distance=tol)
    seg_obj = np.full(len(mid), -1, np.int64)
    if len(pi):
        bd = shapely.distance(shapely.points(mid[pi]), shapely.boundary(np.asarray(polys, dtype=object)[gi]))
        ok = bd <= tol
        pi, gi, bd = pi[ok], gi[ok], bd[ok]
        order = np.lexsort((area[gi], np.floor(bd / EDGE_TIE), pi))
        firsts = np.r_[True, pi[order][1:] != pi[order][:-1]]
        seg_obj[pi[order][firsts]] = idx[gi[order][firsts]]
    owner = np.repeat(np.arange(P), pc.seg_stop - pc.seg_start)
    tot_len = np.bincount(owner, weights=sl, minlength=P)
    m = seg_obj >= 0
    if m.any():
        key = owner[m] * (len(objects) + 1) + seg_obj[m]
        u, inv = np.unique(key, return_inverse=True)
        w = np.bincount(inv, weights=sl[m])
        pu, ou = u // (len(objects) + 1), u % (len(objects) + 1)
        best = np.lexsort((-w, pu))
        first = np.r_[True, pu[best][1:] != pu[best][:-1]]
        obj[pu[best][first]] = ou[best][first]
        share[pu[best][first]] = w[best][first] / np.maximum(tot_len[pu[best][first]], 1e-9)
    return obj, share


FACE_DIST = (1.0, 100.0)   # pt: perpendicular distance between two facing pieces (the two walls of a duct)
FACE_DEG = 10.0            # max angle between facing pieces
FACE_OVERLAP = 0.3         # their projections overlap by >= this share of the shorter one
FACE_K = 2                 # facing links kept per piece (nearest first)
FACE_NEAR = 24             # neighbours (by centre) examined per piece


def facing_edges(pts: np.ndarray, length: np.ndarray, curved: np.ndarray) -> np.ndarray:
    """OpenStroke34 (F187): links between straight pieces that face each other: nearly parallel, FACE_DIST apart, overlapping when projected
    on each other. Two walls of a duct do not touch; without these links a duct whose end lines belong to its neighbours is two pieces of
    nothing. -> (E, 2) piece index pairs (a < b), FACE_K nearest per piece."""
    from scipy.spatial import cKDTree

    P = len(pts)
    ok = (~curved) & (length >= 2.0)
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return np.zeros((0, 2), np.int64)
    a, b = pts[idx, 0], pts[idx, -1]
    c = (a + b) / 2
    d = b - a
    L = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
    u = d / L[:, None]
    k = min(FACE_NEAR + 1, len(idx))
    _, nb = cKDTree(c).query(c, k=k)
    nb = nb.reshape(len(idx), -1)
    out = []
    for r in range(len(idx)):
        cand = []
        for s in nb[r, 1:]:
            if s >= len(idx) or s == r:
                continue
            if abs(float(np.dot(u[r], u[s]))) < np.cos(np.radians(FACE_DEG)):
                continue
            v = c[s] - c[r]
            perp = abs(float(v[0] * u[r, 1] - v[1] * u[r, 0]))
            if not (FACE_DIST[0] <= perp <= FACE_DIST[1]):
                continue
            ta, tb = np.dot(a[s] - a[r], u[r]), np.dot(b[s] - a[r], u[r])
            lo, hi = max(0.0, min(ta, tb)), min(L[r], max(ta, tb))
            if hi - lo < FACE_OVERLAP * min(L[r], L[s]):
                continue
            cand.append((perp, s))
        for _, s in sorted(cand)[:FACE_K]:
            out.append((min(idx[r], idx[s]), max(idx[r], idx[s])))
    return np.unique(np.asarray(out, np.int64), axis=0) if out else np.zeros((0, 2), np.int64)
