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


def tee_edges(seg: np.ndarray, owner: np.ndarray, tol: float = TOL, hub: int = HUB, chunk: int = 500_000) -> np.ndarray:
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
    if not len(cand):
        return np.zeros((0, 2), np.int64)
    tree = STRtree(shapely.linestrings(seg.reshape(-1, 2, 2)))
    out = []
    for c0 in range(0, len(cand), chunk):
        cc = cand[c0:c0 + chunk]
        pi, si = tree.query(shapely.points(pts[cc]), predicate="dwithin", distance=tol)
        if not len(pi):
            continue
        n_hit = np.bincount(pi, minlength=len(cc))
        ok = n_hit[pi] <= hub
        pi, si = cc[pi[ok]], si[ok]
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


GRAPH_VERSION = "g2"  # g2: tee links queried once per (cell, shape) instead of every endpoint  # bump when a link rule changes (cached <stem>.graph.npz are then rebuilt)


def shape_graph(raw_npz) -> tuple[np.ndarray, np.ndarray]:
    """(E, 2) payload-shape pairs and (E,) kind bits of one page npz (``segments`` + ``shape_offsets``)."""

    off = raw_npz["shape_offsets"]
    owner = np.repeat(np.arange(len(off) - 1), np.diff(off))
    return all_edges(raw_npz["segments"], owner)


def load_or_build(npz_path, cache_dir=None) -> tuple[np.ndarray, np.ndarray]:
    """``<stem>.graph.npz`` next to the page (``openstroke build-graph``) or in ``cache_dir``; built and saved if missing."""

    npz_path = Path(npz_path)
    name = npz_path.name.replace(".npz", f".graph.{GRAPH_VERSION}.npz")
    for d in (npz_path.parent, Path(cache_dir) if cache_dir else None):
        if d is not None and (d / name).exists():
            g = np.load(d / name)
            return g["edges"].astype(np.int64), g["kind"].astype(np.int64)
    e, k = shape_graph(np.load(npz_path))
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
