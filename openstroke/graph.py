"""Endpoint graph of a page: two shapes are linked when segment endpoints of both fall in the same ``TOL`` pt cell
(two grids offset by half a cell). Pipes are drawn as chains of shapes; the graph carries that structure to the
model (OpenStroke3 relation blocks) and to post-processing. Cells touched by more than ``HUB`` shapes (hatch, dense
text) are skipped: k^2 links that say nothing about chains.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TOL = 0.5
HUB = 32
GAP = 8.0  # pt: longest dash gap linked
GAP_COS = 0.995  # |cos| between the two segments and the gap direction (about 6 degrees)
GAP_OFFSET = 0.75  # pt: largest sideways offset of the next dash
K_GAP = 8  # nearest endpoints examined per endpoint


def endpoint_edges(seg: np.ndarray, owner: np.ndarray, tol: float = TOL, hub: int = HUB) -> np.ndarray:
    """-> (E, 2) int64 sorted unique shape pairs (i < j)."""

    if len(seg) == 0:
        return np.zeros((0, 2), np.int64)
    pts = np.concatenate([seg[:, :2], seg[:, 2:]]).astype(np.float64)
    own = np.concatenate([owner, owner]).astype(np.int64)
    out = []
    for shift in (0.0, tol / 2):
        q = np.floor((pts + shift) / tol).astype(np.int64)
        cell = (q[:, 0] + 2_000_000) * 8_000_000 + (q[:, 1] + 2_000_000)
        co = np.unique(np.stack([cell, own], 1), axis=0)
        _, start, cnt = np.unique(co[:, 0], return_index=True, return_counts=True)
        for k in range(2, hub + 1):  # vectorised per cell size
            s = start[cnt == k]
            if not len(s):
                continue
            o = co[s[:, None] + np.arange(k)[None, :], 1]  # (cells, k)
            i, j = np.triu_indices(k, 1)
            out.append(np.stack([o[:, i].ravel(), o[:, j].ravel()], 1))
    if not out:
        return np.zeros((0, 2), np.int64)
    e = np.sort(np.concatenate(out), axis=1)
    e = e[e[:, 0] != e[:, 1]]
    return np.unique(e, axis=0)


def gap_edges(seg: np.ndarray, owner: np.ndarray, gap: float = GAP, cos_min: float = GAP_COS,
              offset: float = GAP_OFFSET) -> np.ndarray:
    """Dash links: an endpoint of one segment and an endpoint of a segment of another shape, ``TOL`` < distance <=
    ``gap``, both segments parallel to the gap within ``cos_min``, sideways offset <= ``offset``. -> (E, 2) i < j."""

    from scipy.spatial import cKDTree

    seg = np.asarray(seg, np.float64)
    if len(seg) < 2:
        return np.zeros((0, 2), np.int64)
    d = seg[:, 2:] - seg[:, :2]
    ln = np.hypot(d[:, 0], d[:, 1])
    ok = ln > 1e-6
    u = np.zeros_like(d)
    u[ok] = d[ok] / ln[ok, None]
    pts = np.concatenate([seg[:, :2], seg[:, 2:]])
    sid = np.concatenate([np.arange(len(seg)), np.arange(len(seg))])
    valid = np.concatenate([ok, ok])
    pts, sid = pts[valid], sid[valid]
    tree = cKDTree(pts)
    rows = []
    for c0 in range(0, len(pts), 200_000):  # k nearest endpoints within ``gap``: bounded memory on dense hatch
        dd, ii = tree.query(pts[c0:c0 + 200_000], k=K_GAP + 1, distance_upper_bound=gap)
        src = np.repeat(np.arange(c0, c0 + len(dd)), K_GAP + 1).reshape(len(dd), K_GAP + 1)
        m = np.isfinite(dd) & (ii != src) & (ii < len(pts))
        rows.append(np.stack([src[m], ii[m]], 1))
    pairs = np.concatenate(rows) if rows else np.zeros((0, 2), np.int64)
    if not len(pairs):
        return np.zeros((0, 2), np.int64)
    a, b = sid[pairs[:, 0]], sid[pairs[:, 1]]
    oa, ob = owner[a], owner[b]
    keep = oa != ob
    pairs, a, b, oa, ob = pairs[keep], a[keep], b[keep], oa[keep], ob[keep]
    v = pts[pairs[:, 1]] - pts[pairs[:, 0]]
    dist = np.hypot(v[:, 0], v[:, 1])
    far = dist > TOL
    vu = np.where(far[:, None], v / np.maximum(dist, 1e-9)[:, None], 0.0)
    ca = np.abs((u[a] * vu).sum(1))
    cb = np.abs((u[b] * vu).sum(1))
    cab = np.abs((u[a] * u[b]).sum(1))
    side = np.abs(u[a, 0] * v[:, 1] - u[a, 1] * v[:, 0])
    good = far & (ca >= cos_min) & (cb >= cos_min) & (cab >= cos_min) & (side <= offset)
    e = np.sort(np.stack([oa[good], ob[good]], 1), axis=1)
    return np.unique(e, axis=0) if len(e) else np.zeros((0, 2), np.int64)


def tee_edges(seg: np.ndarray, owner: np.ndarray, tol: float = TOL, hub: int = HUB, chunk: int = 50_000,
              tree_chunk: int = 400_000) -> np.ndarray:
    """T-junctions: an endpoint of one shape within ``tol`` of the *interior* of a segment of another shape (a branch
    teed into a main). Endpoints near more than ``hub`` segments (hatch) are skipped. -> (E, 2) i < j."""

    import shapely
    from shapely import STRtree

    seg = np.asarray(seg, np.float64)
    if len(seg) < 2:
        return np.zeros((0, 2), np.int64)
    pts = np.concatenate([seg[:, :2], seg[:, 2:]])
    sid = np.concatenate([np.arange(len(seg)), np.arange(len(seg))])
    own = owner[sid]
    # one query point per (cell, shape): a polyline's inner vertices and paths drawn there and back repeat points
    q = np.floor(pts / tol).astype(np.int64)
    cell = (q[:, 0] + 2_000_000) * 8_000_000 + (q[:, 1] + 2_000_000)
    _, cand = np.unique(np.stack([cell, own], 1), axis=0, return_index=True)
    # skip points in cells crowded with endpoints (hatch, stacked strokes): a tee there means nothing and the
    # point-in-segment query result explodes (a 854k-shape page ran out of memory)
    _, cinv, ccnt = np.unique(cell, return_inverse=True, return_counts=True)
    cand = cand[ccnt[cinv.reshape(-1)[cand]] <= 2 * hub]
    if not len(cand):
        return np.zeros((0, 2), np.int64)
    seg_ids = np.flatnonzero(np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) > 1e-9)  # no zero-length lines
    hits_p, hits_s = [], []
    for s0 in range(0, len(seg_ids), tree_chunk):  # several small trees (a 3M-segment page crashed GEOS)
        ids = seg_ids[s0:s0 + tree_chunk]
        tree = STRtree(shapely.linestrings(seg[ids].reshape(-1, 2, 2)))
        todo = [(c0, min(c0 + chunk, len(cand))) for c0 in range(0, len(cand), chunk)]
        while todo:
            a, b = todo.pop()
            try:
                pi, si = tree.query(shapely.points(pts[cand[a:b]]), predicate="dwithin", distance=tol)
            except (RuntimeError, MemoryError):  # result too large: split; a single point that still fails is skipped
                if b - a > 1:
                    m = (a + b) // 2
                    todo += [(a, m), (m, b)]
                continue
            hits_p.append(pi + a)
            hits_s.append(ids[si])
        del tree
    out = []
    if hits_p:
        pi, si = np.concatenate(hits_p), np.concatenate(hits_s)
        n_hit = np.bincount(pi, minlength=len(cand))
        ok = n_hit[pi] <= hub
        pi, si = cand[pi[ok]], si[ok]
        oa, ob = owner[sid[pi]], owner[si]
        keep = oa != ob
        out.append(np.stack([oa[keep], ob[keep]], 1))
    if not out:
        return np.zeros((0, 2), np.int64)
    e = np.sort(np.concatenate(out), axis=1)
    return np.unique(e, axis=0) if len(e) else np.zeros((0, 2), np.int64)


def all_edges(seg: np.ndarray, owner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Union of the three link kinds -> (E, 2) pairs and (E,) kind bits (1 endpoint, 2 gap, 4 tee)."""

    kinds = [(1, endpoint_edges(seg, owner)), (2, gap_edges(seg, owner)), (4, tee_edges(seg, owner))]
    e = np.concatenate([k for _, k in kinds]) if any(len(k) for _, k in kinds) else np.zeros((0, 2), np.int64)
    bits = np.concatenate([np.full(len(k), b, np.int64) for b, k in kinds])
    if not len(e):
        return e, bits
    key = e[:, 0] * (int(owner.max()) + 2) + e[:, 1]
    uk, inv = np.unique(key, return_inverse=True)
    kind = np.zeros(len(uk), np.int64)
    np.bitwise_or.at(kind, inv.reshape(-1), bits)
    first = np.zeros(len(uk), np.int64)
    first[inv.reshape(-1)[::-1]] = np.arange(len(key))[::-1]
    return e[first], kind


GRAPH_VERSION = "g3"  # g3: tee queried once per (cell, shape), crowded cells skipped (g2 crashed on a 854k-shape page)  # bump when a link rule changes (cached <stem>.graph.npz are then rebuilt)


def shape_graph(raw_npz) -> tuple[np.ndarray, np.ndarray]:
    """(E, 2) payload-shape pairs and (E,) kind bits of one page npz (``segments`` + ``shape_offsets``)."""

    off = raw_npz["shape_offsets"]
    owner = np.repeat(np.arange(len(off) - 1), np.diff(off))
    return all_edges(raw_npz["segments"], owner)


def facing_shape_edges(raw) -> np.ndarray:
    """Bounded nearest-candidate search for overlapping parallel straight pipe walls."""
    from scipy.spatial import cKDTree

    offsets = raw["shape_offsets"]
    segments = raw["segments"]
    length = raw["length"]
    ids = np.flatnonzero((np.diff(offsets) > 0) & (length >= 2))
    start, end = segments[offsets[ids], :2], segments[offsets[ids + 1] - 1, 2:]
    vec = end - start
    chord = np.linalg.norm(vec, axis=1)
    straight = (chord >= 0.98 * length[ids]) & (chord <= 1.02 * length[ids])
    ids, start, end, vec, chord = ids[straight], start[straight], end[straight], vec[straight], chord[straight]
    if len(ids) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    centres = (start + end) / 2
    unit = vec / chord[:, None]
    tree = cKDTree(centres)
    pairs = []
    for offset in range(0, len(ids), 32_768):
        rows = np.arange(offset, min(offset + 32_768, len(ids)))
        _, neighbour = tree.query(centres[rows], k=min(25, len(ids)))
        left, right = rows[:, None], neighbour
        u = unit[left]
        delta = centres[right] - centres[left]
        width = np.abs(u[..., 0] * delta[..., 1] - u[..., 1] * delta[..., 0])
        angle = np.abs((u * unit[right]).sum(axis=2))
        p = ((start[right] - start[left]) * u).sum(axis=2)
        q = ((end[right] - start[left]) * u).sum(axis=2)
        overlap = np.minimum(chord[left], np.maximum(p, q)) - np.maximum(0, np.minimum(p, q))
        keep = ((width >= 1) & (width <= 100) & (angle >= np.cos(np.deg2rad(10)))
                & (overlap >= 0.3 * np.minimum(chord[left], chord[right])))
        priority = np.argsort(np.where(keep, width, np.inf), axis=1, kind="stable")[:, :2]
        good = np.take_along_axis(keep, priority, axis=1)
        a = np.broadcast_to(left, priority.shape)[good]
        b = np.take_along_axis(right, priority, axis=1)[good]
        pairs.append(np.sort(np.column_stack((ids[a], ids[b])), axis=1))
    return np.unique(np.concatenate(pairs), axis=0)


def load_or_build(npz_path, cache_dir=None, *, facing: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """``<stem>.graph.npz`` next to the page (``openstroke build-graph``) or in ``cache_dir``; built and saved if missing."""

    npz_path = Path(npz_path)
    version = GRAPH_VERSION + ("f1" if facing else "")
    name = npz_path.name.replace(".npz", f".graph.{version}.npz")
    for d in (npz_path.parent, Path(cache_dir) if cache_dir else None):
        if d is not None and (d / name).exists():
            g = np.load(d / name)
            return g["edges"].astype(np.int64), g["kind"].astype(np.int64)
    if facing:
        e, k = load_or_build(npz_path, cache_dir)
        with np.load(npz_path) as raw:
            peer = facing_shape_edges(raw)
            n = len(raw["shape_offsets"]) - 1
        key = np.r_[e[:, 0] * n + e[:, 1], peer[:, 0] * n + peer[:, 1]]
        unique, inv = np.unique(key, return_inverse=True)
        kind = np.zeros(len(unique), dtype=np.int64)
        np.bitwise_or.at(kind, inv.ravel(), np.r_[k, np.full(len(peer), 8, dtype=np.int64)])
        e, k = np.column_stack(np.divmod(unique, max(n, 1))), kind
    else:
        with np.load(npz_path) as raw:
            e, k = shape_graph(raw)
    out = (Path(cache_dir) if cache_dir else npz_path.parent) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, edges=e.astype(np.int32), kind=k.astype(np.uint8))
    return e, k


def to_positions(edges: np.ndarray, kind: np.ndarray, index: np.ndarray, n_shapes: int):
    """Payload-shape pairs -> pairs of positions in the page's token order (``Page.index``: position -> shape)."""

    pos_of = np.full(n_shapes, -1, np.int64)
    pos_of[index] = np.arange(len(index))
    p = pos_of[edges] if len(edges) else np.zeros((0, 2), np.int64)
    ok = (p >= 0).all(1) if len(p) else np.zeros(0, bool)
    return p[ok], kind[ok]
