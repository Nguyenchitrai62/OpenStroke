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


def _unique_sorted_int64(a: np.ndarray) -> np.ndarray:
    """Fast sorted 1-D int64 deduplication (avoids NumPy 2.4 hash-table slowdown on high-cardinality int64)."""
    if not len(a):
        return np.zeros(0, dtype=np.int64)
    s = np.sort(a, kind="quicksort")
    mask = np.empty(len(s), dtype=bool)
    mask[0] = True
    if len(s) > 1:
        mask[1:] = s[1:] != s[:-1]
    return s[mask]


def _unique_pairs(a: np.ndarray, b: np.ndarray, n_max: int) -> np.ndarray:
    """Sorted unique undirected int64 pairs (lo < hi) from two equal-length index arrays."""
    if not len(a):
        return np.zeros((0, 2), dtype=np.int64)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    stride = int(n_max) + 1
    if stride > 0 and int(lo.max()) * stride + int(hi.max()) < (1 << 62):
        uk = _unique_sorted_int64(lo * stride + hi)
        return np.column_stack(np.divmod(uk, stride)).astype(np.int64, copy=False)
    return np.unique(np.column_stack((lo, hi)), axis=0)


def endpoint_edges(seg: np.ndarray, owner: np.ndarray, tol: float = TOL, hub: int = HUB) -> np.ndarray:
    """-> (E, 2) int64 sorted unique shape pairs (i < j)."""

    if len(seg) == 0:
        return np.zeros((0, 2), np.int64)
    pts = np.concatenate([seg[:, :2], seg[:, 2:]]).astype(np.float64)
    own = np.concatenate([owner, owner]).astype(np.int64)
    n_max = int(own.max())
    ostride = n_max + 1
    out_a, out_b = [], []
    for shift in (0.0, tol / 2):
        q = np.floor((pts + shift) / tol).astype(np.int64)
        qx = q[:, 0] - int(q[:, 0].min())
        qy = q[:, 1] - int(q[:, 1].min())
        ystride = int(qy.max()) + 1
        cell = qx * ystride + qy
        if len(cell) and int(cell.max()) * ostride + n_max < (1 << 62):
            uk = _unique_sorted_int64(cell * ostride + own)
            co_cell, co_own = np.divmod(uk, ostride)
        else:
            co = np.unique(np.stack([cell, own], 1), axis=0)
            co_cell, co_own = co[:, 0], co[:, 1]
        if not len(co_cell):
            continue
        chg = np.flatnonzero(co_cell[1:] != co_cell[:-1]) + 1
        start = np.empty(len(chg) + 1, dtype=np.int64)
        start[0] = 0
        start[1:] = chg
        cnt = np.empty(len(start), dtype=np.int64)
        cnt[:-1] = start[1:] - start[:-1]
        cnt[-1] = len(co_cell) - start[-1]
        for k in range(2, hub + 1):  # vectorised per cell size
            s = start[cnt == k]
            if not len(s):
                continue
            o = co_own[s[:, None] + np.arange(k)[None, :]]  # (cells, k), already sorted ascending per row
            i, j = np.triu_indices(k, 1)
            out_a.append(o[:, i].ravel())
            out_b.append(o[:, j].ravel())
    if not out_a:
        return np.zeros((0, 2), np.int64)
    ea = np.concatenate(out_a)
    eb = np.concatenate(out_b)
    neq = ea != eb
    if not neq.any():
        return np.zeros((0, 2), np.int64)
    return _unique_pairs(ea[neq], eb[neq], n_max)


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
    if len(pts) < 2:
        return np.zeros((0, 2), np.int64)
    tree = cKDTree(pts)
    workers = -1 if len(pts) >= 4096 else 1
    own_pt = owner[sid]
    out_a, out_b = [], []
    n_pts = len(pts)
    for c0 in range(0, n_pts, 200_000):  # k nearest endpoints within ``gap``: bounded memory on dense hatch
        dd, ii = tree.query(pts[c0:c0 + 200_000], k=K_GAP + 1, distance_upper_bound=gap, workers=workers)
        m = (dd > TOL) & np.isfinite(dd) & (ii < n_pts)
        if not m.any():
            continue
        r_idx, c_idx = np.nonzero(m)
        p0 = r_idx + c0
        p1 = ii[r_idx, c_idx]
        oa = own_pt[p0]
        ob = own_pt[p1]
        diff = oa != ob
        if not diff.any():
            continue
        p0, p1, oa, ob = p0[diff], p1[diff], oa[diff], ob[diff]
        dist = dd[r_idx[diff], c_idx[diff]]
        a, b = sid[p0], sid[p1]
        v = pts[p1] - pts[p0]
        vu = v / dist[:, None]
        ca = np.abs((u[a] * vu).sum(1))
        cb = np.abs((u[b] * vu).sum(1))
        cab = np.abs((u[a] * u[b]).sum(1))
        side = np.abs(u[a, 0] * v[:, 1] - u[a, 1] * v[:, 0])
        good = (ca >= cos_min) & (cb >= cos_min) & (cab >= cos_min) & (side <= offset)
        if good.any():
            out_a.append(oa[good])
            out_b.append(ob[good])
    if not out_a:
        return np.zeros((0, 2), np.int64)
    return _unique_pairs(np.concatenate(out_a), np.concatenate(out_b), int(owner.max()))


def _pt_seg_dwithin(pxy: np.ndarray, sg: np.ndarray, tol2: float) -> np.ndarray:
    """Exact squared point-to-segment distance <= tol2 (pure NumPy, matches GEOS dwithin)."""
    vx = sg[:, 2] - sg[:, 0]
    vy = sg[:, 3] - sg[:, 1]
    wx = pxy[:, 0] - sg[:, 0]
    wy = pxy[:, 1] - sg[:, 1]
    vv = vx * vx + vy * vy
    t = np.clip((wx * vx + wy * vy) / vv, 0.0, 1.0)
    dx = wx - t * vx
    dy = wy - t * vy
    return (dx * dx + dy * dy) <= tol2


def tee_edges(seg: np.ndarray, owner: np.ndarray, tol: float = TOL, hub: int = HUB, chunk: int = 50_000,
              tree_chunk: int = 400_000) -> np.ndarray:
    """T-junctions: an endpoint of one shape within ``tol`` of the *interior* of a segment of another shape (a branch
    teed into a main). Endpoints near more than ``hub`` segments (hatch) are skipped. -> (E, 2) i < j."""

    import os
    from concurrent.futures import ThreadPoolExecutor

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
    qx = q[:, 0] - int(q[:, 0].min())
    qy = q[:, 1] - int(q[:, 1].min())
    cell = qx * (int(qy.max()) + 1) + qy
    ostride = int(own.max()) + 1
    if len(cell) and int(cell.max()) * ostride + int(own.max()) < (1 << 62):
        ckey = cell * ostride + own
        order = np.argsort(ckey, kind="quicksort")
        sk = ckey[order]
        first = np.empty(len(sk), dtype=bool)
        first[0] = True
        if len(sk) > 1:
            first[1:] = sk[1:] != sk[:-1]
        starts = np.flatnonzero(first)
        cand = np.minimum.reduceat(order, starts)  # exact first index per (cell, own), matching np.unique
        scell = cell[order]
        c_chg = np.flatnonzero(scell[1:] != scell[:-1]) + 1
        c_cnts = np.diff(np.r_[0, c_chg, len(scell)])
        cell_id = np.empty(len(order), dtype=np.int64)
        cell_id[0] = 0
        if len(order) > 1:
            cell_id[1:] = np.cumsum(scell[1:] != scell[:-1])
        cand = cand[c_cnts[cell_id[starts]] <= 2 * hub]
    else:
        _, cand = np.unique(np.stack([cell, own], 1), axis=0, return_index=True)
        _, cinv, ccnt = np.unique(cell, return_inverse=True, return_counts=True)
        cand = cand[ccnt[cinv.reshape(-1)[cand]] <= 2 * hub]
    if not len(cand):
        return np.zeros((0, 2), np.int64)
    seg_ids = np.flatnonzero(np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) > 1e-9)  # no zero-length lines
    if not len(seg_ids):
        return np.zeros((0, 2), np.int64)
    cpts = pts[cand]
    qpts_all = shapely.points(cpts)
    slices = [(c0, min(c0 + chunk, len(cand))) for c0 in range(0, len(cand), chunk)]
    tol2 = tol * tol
    n_threads = min(8, os.cpu_count() or 1) if len(cand) >= 16_384 and len(slices) > 1 else 1

    hits_p, hits_s = [], []
    for s0 in range(0, len(seg_ids), tree_chunk):  # several small trees (a 3M-segment page crashed GEOS)
        ids = seg_ids[s0:s0 + tree_chunk]
        s_sub = seg[ids]
        # Build R-tree on segment bboxes expanded by tol: predicate=None does pure C++ R-tree bbox overlap,
        # then exact vectorized NumPy point-to-segment distance filters hits (~8-10x faster than GEOS dwithin).
        env = np.empty((len(ids), 2, 2), dtype=np.float64)
        env[:, 0, 0] = np.minimum(s_sub[:, 0], s_sub[:, 2]) - tol
        env[:, 0, 1] = np.minimum(s_sub[:, 1], s_sub[:, 3]) - tol
        env[:, 1, 0] = np.maximum(s_sub[:, 0], s_sub[:, 2]) + tol
        env[:, 1, 1] = np.maximum(s_sub[:, 1], s_sub[:, 3]) + tol
        tree = STRtree(shapely.linestrings(env))
        if n_threads > 1:
            _ = tree.query(qpts_all[:1], predicate=None)  # force GEOS STRtree build before threads enter

        def _query_slice(ab):
            a, b = ab
            res_p, res_s = [], []
            todo = [(a, b)]
            while todo:
                lo, hi = todo.pop()
                try:
                    pi, si = tree.query(qpts_all[lo:hi], predicate=None)
                except (RuntimeError, MemoryError):  # result too large: split; a single point that still fails is skipped
                    if hi - lo > 1:
                        m = (lo + hi) // 2
                        todo += [(lo, m), (m, hi)]
                    continue
                if len(pi):
                    hit = _pt_seg_dwithin(cpts[pi + lo], s_sub[si], tol2)
                    if hit.any():
                        res_p.append(pi[hit] + lo)
                        res_s.append(ids[si[hit]])
            return res_p, res_s

        if n_threads > 1:
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                for rp, rs in ex.map(_query_slice, slices):
                    hits_p.extend(rp)
                    hits_s.extend(rs)
        else:
            for ab in slices:
                rp, rs = _query_slice(ab)
                hits_p.extend(rp)
                hits_s.extend(rs)
        del tree
    if not hits_p:
        return np.zeros((0, 2), np.int64)
    pi, si = np.concatenate(hits_p), np.concatenate(hits_s)
    n_hit = np.bincount(pi, minlength=len(cand))
    ok = n_hit[pi] <= hub
    pi, si = cand[pi[ok]], si[ok]
    oa, ob = own[pi], owner[si]
    keep = oa != ob
    if not keep.any():
        return np.zeros((0, 2), np.int64)
    return _unique_pairs(oa[keep], ob[keep], int(owner.max()))


def all_edges(seg: np.ndarray, owner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Union of the three link kinds -> (E, 2) pairs and (E,) kind bits (1 endpoint, 2 gap, 4 tee)."""

    kinds = [(1, endpoint_edges(seg, owner)), (2, gap_edges(seg, owner)), (4, tee_edges(seg, owner))]
    nonempty = [(b, k) for b, k in kinds if len(k)]
    if not nonempty:
        return np.zeros((0, 2), np.int64), np.zeros(0, np.int64)
    e = np.concatenate([k for _, k in nonempty])
    bits = np.concatenate([np.full(len(k), b, np.int64) for b, k in nonempty])
    stride = int(owner.max()) + 2
    key = e[:, 0] * stride + e[:, 1]
    order = np.argsort(key, kind="quicksort")
    sk = key[order]
    chg = np.empty(len(sk), dtype=bool)
    chg[0] = True
    if len(sk) > 1:
        chg[1:] = sk[1:] != sk[:-1]
    starts = np.flatnonzero(chg)
    kind = np.bitwise_or.reduceat(bits[order], starts)
    return e[order[starts]], kind


GRAPH_VERSION = "g3"  # g3: tee queried once per (cell, shape), crowded cells skipped (g2 crashed on a 854k-shape page)  # bump when a link rule changes (cached <stem>.graph.npz are then rebuilt)


def shape_graph(raw_npz) -> tuple[np.ndarray, np.ndarray]:
    """(E, 2) payload-shape pairs and (E,) kind bits of one page npz (``segments`` + ``shape_offsets``)."""

    off = raw_npz["shape_offsets"]
    owner = np.repeat(np.arange(len(off) - 1), np.diff(off))
    return all_edges(raw_npz["segments"], owner)


def shape_graph_from_arrays(raw, *, facing: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """In-memory equivalent of ``load_or_build`` without touching disk (for inference)."""
    e, k = shape_graph(raw)
    if not facing:
        return e, k
    peer = facing_shape_edges(raw)
    if not len(peer):
        return e, k
    n = len(raw["shape_offsets"]) - 1
    stride = max(n, 1)
    key = np.r_[e[:, 0] * stride + e[:, 1], peer[:, 0] * stride + peer[:, 1]]
    bits = np.r_[k, np.full(len(peer), 8, dtype=np.int64)]
    order = np.argsort(key, kind="quicksort")
    sk = key[order]
    chg = np.empty(len(sk), dtype=bool)
    chg[0] = True
    if len(sk) > 1:
        chg[1:] = sk[1:] != sk[:-1]
    starts = np.flatnonzero(chg)
    unique = sk[starts]
    kind = np.bitwise_or.reduceat(bits[order], starts)
    return np.column_stack(np.divmod(unique, stride)), kind


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
