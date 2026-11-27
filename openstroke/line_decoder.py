"""Direct vector-to-pipe-line decoder for thin-line MEP systems (FIRE & PLUMBING, layer-free).

Unlike HVAC (2D closed duct polygons), FIRE and PLUMBING pipes in CAD PDFs are 1D straight single-line networks
(100.0% 2-vertex segments on FIRE GT, 99.9% on PLUMBING GT; dashed pipes are exploded into short straight segments
with 6-20 pt gaps). Once the stroke model picks pipe strokes (p >= thr), pure vector geometry recovers clean
2-vertex pipe ``Line`` objects and topology-derived ``Elbow`` / ``Tee`` / ``Cross`` fittings directly:

  1. Extract picked segments and split continuous CAD polylines at T-junctions / bends (non-touching X-crossings stay uncut).
  2. Merge collinear segments / exploded dashes within ``gap`` pt (stopping at degree >= 3 branch junctions).
  3. Snap endpoints into canonical graph nodes (``snap_tol = 2.0 pt``, matching AI_Takeoff ``NODE_TOLERANCE``),
     deduplicate overlapping parallel runs, and prune tiny isolated noise ticks.
  4. Derive topology fittings (``Elbow``, ``Tee``, ``Cross``) from node degrees & bend angles.

Zero PDF layer usage; < 0.2 s/page CPU. Oracle Line-Length F2 on exact GT strokes: 0.990 FIRE1 / 0.967 PLUMBING.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

#: Default tolerances aligned with AI_Takeoff FIRE (evaluate.py) and PLUMBING (plumbing_evaluate.py).
ANGLE_DEG = 2.0
PERP_TOL = 1.5
NODE_TOL = 2.0
FIRE_GAP = 12.0
PLUMBING_GAP = 20.0
MIN_STUB = 2.0
FITTING_STRAIGHT_DOT = -0.985   # ~10 deg bend threshold for degree-2 Elbow vs straight pass-through


def _seg_dir_len(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = s[:, 2:4] - s[:, 0:2]
    L = np.hypot(v[:, 0], v[:, 1])
    u = np.divide(v, np.maximum(L[:, None], 1e-9))
    # Canonical orientation: angle in [0, pi)
    flip = (u[:, 0] < -1e-9) | ((np.abs(u[:, 0]) <= 1e-9) & (u[:, 1] < 0))
    u[flip] *= -1.0
    return u, L


def _split_at_t_junctions(seg: np.ndarray, owner: np.ndarray, tol: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Split any segment whose interior (not endpoints) is touched within ``tol`` by another segment's endpoint.
    Non-touching X-crossings are intentionally NOT split (pipes crossing at different elevations)."""
    if len(seg) <= 1:
        return seg, owner
    p0 = seg[:, 0:2]
    p1 = seg[:, 2:4]
    v = p1 - p0
    L = np.hypot(v[:, 0], v[:, 1])
    ok = L > 1e-6
    seg, owner, p0, p1, v, L = seg[ok], owner[ok], p0[ok], p1[ok], v[ok], L[ok]
    if len(seg) <= 1:
        return seg, owner

    pts = np.vstack([p0, p1])
    pt_seg = np.concatenate([np.arange(len(seg)), np.arange(len(seg))])
    tree = cKDTree(pts)
    mid = 0.5 * (p0 + p1)
    rad = 0.5 * L + tol + 0.5

    out_s = []
    out_o = []
    for i in range(len(seg)):
        if L[i] < 2.0 * tol:
            out_s.append(seg[i])
            out_o.append(owner[i])
            continue
        cand = tree.query_ball_point(mid[i], float(rad[i]))
        if not cand:
            out_s.append(seg[i])
            out_o.append(owner[i])
            continue
        c = np.asarray(cand, np.int64)
        c = c[pt_seg[c] != i]
        if not len(c):
            out_s.append(seg[i])
            out_o.append(owner[i])
            continue
        d = pts[c] - p0[i]
        ui = v[i] / L[i]
        t = d[:, 0] * ui[0] + d[:, 1] * ui[1]
        perp = np.abs(d[:, 0] * (-ui[1]) + d[:, 1] * ui[0])
        hit = (perp <= tol) & (t > tol) & (t < L[i] - tol)
        if not hit.any():
            out_s.append(seg[i])
            out_o.append(owner[i])
            continue
        ts = np.sort(t[hit])
        # Cluster nearby split parameters within tol
        cuts = [0.0]
        for tv in ts:
            if tv - cuts[-1] > tol:
                cuts.append(float(tv))
        if L[i] - cuts[-1] <= tol:
            cuts[-1] = float(L[i])
        else:
            cuts.append(float(L[i]))
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a > 1e-4:
                qa = p0[i] + ui * a
                qb = p0[i] + ui * b
                out_s.append(np.array([qa[0], qa[1], qb[0], qb[1]], np.float64))
                out_o.append(owner[i])
    if not out_s:
        return np.zeros((0, 4), np.float64), np.zeros(0, np.int64)
    return np.asarray(out_s, np.float64), np.asarray(out_o, np.int64)


def _merge_collinear(
    seg: np.ndarray,
    members: list[list[int]] | list[dict[int, float]],
    *,
    seg_cls: np.ndarray | None = None,
    angle_deg: float = ANGLE_DEG,
    perp_tol: float = PERP_TOL,
    gap: float = FIRE_GAP,
    node_tol: float = NODE_TOL,
) -> tuple[np.ndarray, list[Any]]:
    """Junction-aware & optional class-barrier-aware collinear merging and parallel duplicate collapse.

    ``members`` may be either ``list[list[int]]`` (legacy) or ``list[dict[int, float]]`` (per-shape contributed
    length on each segment, preventing giant multi-segment shapes from overwhelming class votes on short lines).
    """
    if len(seg) <= 1:
        return seg, members
    u, L = _seg_dir_len(seg)
    cos_min = math.cos(math.radians(angle_deg))
    track_len = isinstance(members[0], dict)

    # Build endpoint spatial index to detect degree >= 3 non-collinear branch junctions
    pts = np.vstack([seg[:, 0:2], seg[:, 2:4]])
    pt_tree = cKDTree(pts)
    n = len(seg)
    is_junc = np.zeros(len(pts), dtype=bool)
    nbrs = pt_tree.query_ball_point(pts, node_tol)
    for k, js in enumerate(nbrs):
        si = k % n
        if L[si] < 1.0:
            continue
        other_k = k + n if k < n else k - n
        pk, po = pts[k], pts[other_k]
        # Only mark the endpoint of si that is ACTUALLY closer to the meeting branch (never the far end of a short 1.8pt dash!)
        for j in js:
            sj = j % n
            if sj != si and L[sj] >= 1.0 and abs(float(u[si, 0] * u[sj, 0] + u[si, 1] * u[sj, 1])) < cos_min:
                pj = pts[j]
                if float(np.hypot(*(pk - pj))) <= float(np.hypot(*(po - pj))) + 1e-6:
                    is_junc[k] = True
                    break
    junc_pts = pts[is_junc]
    junc_tree = cKDTree(junc_pts) if len(junc_pts) else None

    order = np.argsort(-L)   # longest segments anchor the line direction first
    used = np.zeros(n, dtype=bool)

    # Spatial index on segment midpoints for fast candidate lookup
    mids = 0.5 * (seg[:, 0:2] + seg[:, 2:4])
    mid_tree = cKDTree(mids)
    max_half = float(0.5 * L.max())

    out_s = []
    out_m: list[Any] = []

    for seed in order:
        if used[seed]:
            continue
        ui = u[seed]
        ni = np.array([-ui[1], ui[0]])
        origin = seg[seed, 0:2]
        seed_c = int(seg_cls[seed]) if seg_cls is not None else 0
        # Gather all unused segments collinear with seed's infinite line (within perp_tol & angle_deg)
        # Iterate expanding along the line so long chains of dashes get collected completely
        chain = [int(seed)]
        used[seed] = True
        p0s = float((seg[seed, 0:2] - origin) @ ui)
        p1s = float((seg[seed, 2:4] - origin) @ ui)
        lo, hi = min(p0s, p1s), max(p0s, p1s)

        changed = True
        while changed:
            changed = False
            center = origin + ui * (0.5 * (lo + hi))
            search_r = 0.5 * (hi - lo) + gap + max_half + 1.0
            cands = mid_tree.query_ball_point(center, float(search_r))
            for j in cands:
                if used[j]:
                    continue
                if seg_cls is not None:
                    cj = int(seg_cls[j])
                    # Keep clean node splits across confident opposite pipe classes (e.g., inline main -> branch reducer)
                    if seed_c > 0 and cj > 0 and cj != seed_c:
                        continue
                if abs(float(u[j] @ ui)) < cos_min:
                    continue
                d0 = seg[j, 0:2] - origin
                d1 = seg[j, 2:4] - origin
                if abs(float(d0 @ ni)) > perp_tol or abs(float(d1 @ ni)) > perp_tol:
                    continue
                t0, t1 = float(d0 @ ui), float(d1 @ ui)
                jlo, jhi = min(t0, t1), max(t0, t1)
                if jlo <= hi + gap and jhi >= lo - gap:
                    chain.append(int(j))
                    used[j] = True
                    lo = min(lo, jlo)
                    hi = max(hi, jhi)
                    changed = True

        # Project chain segments onto 1D line axis and merge intervals that don't cross a branch junction
        ivals = []
        for idx in chain:
            t0 = float((seg[idx, 0:2] - origin) @ ui)
            t1 = float((seg[idx, 2:4] - origin) @ ui)
            o_off = 0.5 * float(((seg[idx, 0:2] - origin) @ ni) + ((seg[idx, 2:4] - origin) @ ni))
            ivals.append((min(t0, t1), max(t0, t1), o_off, idx))
        ivals.sort(key=lambda x: x[0])

        # Find any branch junctions lying along this infinite line
        line_juncs: list[float] = []
        if junc_tree is not None:
            center = origin + ui * (0.5 * (lo + hi))
            jc = junc_tree.query_ball_point(center, float(0.5 * (hi - lo) + node_tol + 1.0))
            for jk in jc:
                dj = junc_pts[jk] - origin
                if abs(float(dj @ ni)) <= node_tol:
                    line_juncs.append(float(dj @ ui))

        cur_a, cur_b = ivals[0][0], ivals[0][1]
        s0len = max(cur_b - cur_a, 1e-6)
        w_off, w_len = ivals[0][2] * s0len, s0len
        cur_mem: Any = dict(members[ivals[0][3]]) if track_len else list(members[ivals[0][3]])
        junc_pad = 0.5   # only block a gap when a branch sits at/inside the gap, not 2pt away past an adjacent 1.8pt dash
        for a, b, off, idx in ivals[1:]:
            seglen = max(b - a, 1e-6)
            blocked = False
            if a > cur_b:
                # Check if a branch junction sits inside the gap or right at its boundary
                for jt in line_juncs:
                    if cur_b - junc_pad <= jt <= a + junc_pad:
                        blocked = True
                        break
            else:
                # Overlapping/touching: split if a branch junction sits right at the meeting point (strictly inside union)
                for jt in line_juncs:
                    if cur_a + junc_pad < jt < max(cur_b, b) - junc_pad and abs(jt - cur_b) <= junc_pad and abs(jt - a) <= junc_pad:
                        blocked = True
                        break
            if (a <= cur_b + gap) and not blocked:
                cur_b = max(cur_b, b)
                w_off += off * seglen
                w_len += seglen
                if track_len:
                    for km, vl in members[idx].items():
                        cur_mem[km] = cur_mem.get(km, 0.0) + vl
                else:
                    cur_mem.extend(members[idx])
            else:
                mean_off = w_off / max(w_len, 1e-6)
                pa = origin + ui * cur_a + ni * mean_off
                pb = origin + ui * cur_b + ni * mean_off
                out_s.append([pa[0], pa[1], pb[0], pb[1]])
                out_m.append(cur_mem if track_len else sorted(set(cur_mem)))
                cur_a, cur_b, w_off, w_len = a, b, off * seglen, seglen
                cur_mem = dict(members[idx]) if track_len else list(members[idx])
        mean_off = w_off / max(w_len, 1e-6)
        pa = origin + ui * cur_a + ni * mean_off
        pb = origin + ui * cur_b + ni * mean_off
        out_s.append([pa[0], pa[1], pb[0], pb[1]])
        out_m.append(cur_mem if track_len else sorted(set(cur_mem)))

    return np.asarray(out_s, np.float64), out_m


def _radius_bounded_nodes(pts: np.ndarray, n_seg: int, snap_tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Radius-bounded, self-non-collapsing endpoint clustering (no transitive chaining across adjacent short pipes)."""
    tree = cKDTree(pts)
    pairs = sorted(
        (float(np.hypot(*(pts[a] - pts[b]))), a, b)
        for a, b in tree.query_pairs(snap_tol)
        if (a % n_seg) != (b % n_seg)
    )
    parent = list(range(2 * n_seg))
    c_pts: dict[int, list[int]] = {k: [k] for k in range(2 * n_seg)}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        comb = c_pts[ra] + c_pts[rb]
        if len({k % n_seg for k in comb}) < len(comb):   # never collapse a segment's own two endpoints
            continue
        xy = pts[comb]
        ctr = xy.mean(axis=0)
        if float(np.hypot(*(xy - ctr).T).max()) > snap_tol:   # radius-bounded: blocks transitive chaining
            continue
        if len(c_pts[ra]) < len(c_pts[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        c_pts[ra].extend(c_pts.pop(rb))

    node_id = np.empty(2 * n_seg, np.int64)
    node_xy = np.empty((len(c_pts), 2), np.float64)
    for idx, ks in enumerate(c_pts.values()):
        node_xy[idx] = pts[ks].mean(axis=0)
        for k in ks:
            node_id[k] = idx
    return node_id, node_xy


def _snap_and_graph(
    seg: np.ndarray,
    members: list[list[int]] | list[dict[int, float]],
    *,
    snap_tol: float = NODE_TOL,
    min_stub: float = MIN_STUB,
    perp_tol: float = PERP_TOL,
    return_member_lengths: bool = False,
    return_graph: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    """Radius-bounded endpoint snapping (no transitive chaining and never collapsing a short dash's own two endpoints),
    zero-kink trunk coordinate preservation, centerline averaging of double-wall filled pipes, micro-stub pruning, and
    topology fittings."""
    if not len(seg):
        return ([], [], np.zeros((0, 2), np.float64)) if return_graph else ([], [])
    n = len(seg)
    track_len = isinstance(members[0], dict)
    pts = np.vstack([seg[:, 0:2], seg[:, 2:4]])
    node_id, node_xy = _radius_bounded_nodes(pts, n, snap_tol)

    # Deduplicate overlapping edges sharing (u, v) (averaging double-wall filled pipe walls to centerline),
    # while preserving exact CAD trunk alignment (never tilting straight trunks perpendicular to their axis at Tees).
    dedup_tol = max(perp_tol, snap_tol)
    buckets: dict[tuple[int, int], list[list[Any]]] = {}
    for i in range(n):
        u, v = int(node_id[i]), int(node_id[n + i])
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        p0, p1 = seg[i, 0:2], seg[i, 2:4]
        dv = p1 - p0
        sl = float(np.hypot(dv[0], dv[1]))
        if sl > 1e-6:
            ni = np.array([-dv[1] / sl, dv[0] / sl])
            da = node_xy[u] - p0
            db = node_xy[v] - p1
            pa = node_xy[u] if (float(np.hypot(da[0], da[1])) <= 0.5 and abs(float(da @ ni)) <= 0.04) else p0
            pb = node_xy[v] if (float(np.hypot(db[0], db[1])) <= 0.5 and abs(float(db @ ni)) <= 0.04) else p1
        else:
            pa, pb = p0, p1
        geom = np.array([pa[0], pa[1], pb[0], pb[1]], np.float64)
        lst = buckets.setdefault(key, [])
        mi = 0.5 * (pa + pb)
        merged = False
        for entry in lst:
            og = entry[1]
            om = 0.5 * (og[:2] + og[2:])
            if float(np.hypot(*(mi - om))) <= dedup_tol:
                if track_len:
                    for km, vl in members[i].items():
                        entry[0][km] = entry[0].get(km, 0.0) + vl
                else:
                    entry[0].update(members[i])
                li = float(np.hypot(*(pb - pa)))
                lo = float(np.hypot(*(og[2:] - og[:2])))
                if min(li, lo) >= 0.8 * max(li, lo, 1e-6):
                    g_al = np.array([pb[0], pb[1], pa[0], pa[1]], np.float64) if float((pb - pa) @ (og[2:] - og[:2])) < 0 else geom
                    entry[1] = 0.5 * (og + g_al)
                elif li > lo:
                    entry[1] = geom
                merged = True
                break
        if not merged:
            lst.append([dict(members[i]) if track_len else set(members[i]), geom])

    # Compute node degrees and drop tiny dangling stubs (< min_stub with at least one leaf endpoint)
    deg = np.zeros(len(node_xy), np.int64)
    for (u, v), lst in buckets.items():
        deg[u] += len(lst)
        deg[v] += len(lst)

    kept_edges: list[tuple[int, int, Any, float, np.ndarray]] = []
    for (u, v), lst in buckets.items():
        for mems, geom in lst:
            L = float(np.hypot(geom[2] - geom[0], geom[3] - geom[1]))
            if L < min_stub and (deg[u] <= 1 or deg[v] <= 1):
                continue
            kept_edges.append((u, v, mems, L, geom))

    # Recompute incident edges per node for exact fitting classification
    adj: dict[int, list[int]] = {}
    for u, v, _, _, _ in kept_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    lines = []
    for u, v, mems, L, geom in kept_edges:
        entry_ln: dict[str, Any] = {
            "shape_name": "Line",
            "vertices": [
                [round(float(geom[0]), 3), round(float(geom[1]), 3)],
                [round(float(geom[2]), 3), round(float(geom[3]), 3)],
            ],
            "length": round(L, 3),
            "members": sorted(mems.keys()) if track_len else sorted(mems),
        }
        if return_member_lengths and track_len:
            entry_ln["_member_lengths"] = mems
        if return_graph:
            entry_ln["_uv"] = (u, v)
        lines.append(entry_ln)

    fittings: list[dict[str, Any]] = []
    for u, nbrs in adj.items():
        d = len(nbrs)
        pt = [round(float(node_xy[u, 0]), 3), round(float(node_xy[u, 1]), 3)]
        if d == 2:
            v1 = node_xy[nbrs[0]] - node_xy[u]
            v2 = node_xy[nbrs[1]] - node_xy[u]
            n1 = float(np.hypot(v1[0], v1[1]))
            n2 = float(np.hypot(v2[0], v2[1]))
            if n1 > 1e-6 and n2 > 1e-6:
                dot = float((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2))
                if dot > FITTING_STRAIGHT_DOT:
                    fittings.append({"shape_name": "Elbow", "point": pt})
        elif d == 3:
            fittings.append({"shape_name": "Tee", "point": pt})
        elif d >= 4:
            fittings.append({"shape_name": "Cross", "point": pt})

    if return_graph:
        return lines, fittings, node_xy
    return lines, fittings


def decode_lines(
    segs,
    prob: np.ndarray,
    thr: float = 0.5,
    *,
    mode: str = "fire",
    gap: float | None = None,
    angle_deg: float = ANGLE_DEG,
    perp_tol: float = PERP_TOL,
    snap_tol: float = NODE_TOL,
    min_stub: float = MIN_STUB,
    shape_cls: np.ndarray | None = None,
    return_member_lengths: bool = False,
    return_graph: bool = False,
) -> dict[str, Any]:
    """Decode picked vector strokes (``prob >= thr``) directly into clean 2-vertex pipe Lines + topology fittings."""
    picked = np.asarray(prob, np.float64) >= thr
    if not picked.any() or not len(segs.seg):
        return {"lines": [], "fittings": []}
    m = picked[segs.owner]
    if not m.any():
        return {"lines": [], "fittings": []}
    seg = np.asarray(segs.seg[m], np.float64)
    own = np.asarray(segs.owner[m], np.int64)
    gap_tol = (PLUMBING_GAP if mode == "plumbing" else FIRE_GAP) if gap is None else float(gap)

    # Pass 1: split continuous CAD lines at T-junctions
    seg, own = _split_at_t_junctions(seg, own, tol=perp_tol)
    if not len(seg):
        return {"lines": [], "fittings": []}

    # Pass 2: junction-aware (and optional class-barrier-aware) collinear dash/fragment bridging
    seg_cls = np.asarray(shape_cls, np.int8)[own] if shape_cls is not None else None
    if return_member_lengths:
        sl_init = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
        members: list[Any] = [{int(o): float(lv)} for o, lv in zip(own.tolist(), sl_init.tolist())]
    else:
        members = [[int(o)] for o in own]
    seg, members = _merge_collinear(
        seg, members, seg_cls=seg_cls, angle_deg=angle_deg, perp_tol=perp_tol, gap=gap_tol, node_tol=snap_tol,
    )

    # Pass 3: re-split any merged segment touched mid-span by a bridged branch endpoint
    if len(seg) > 1:
        idx_arr = np.arange(len(seg), dtype=np.int64)
        seg_pre_len = np.maximum(np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]), 1e-6) if return_member_lengths else None
        seg, orig_idx = _split_at_t_junctions(seg, idx_arr, tol=snap_tol)
        if return_member_lengths and seg_pre_len is not None:
            sl_post = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
            members = [
                {k: v * min(1.0, nl / float(seg_pre_len[oi])) for k, v in members[oi].items()}
                for oi, nl in zip(orig_idx.tolist(), sl_post.tolist())
            ]
        else:
            members = [members[int(k)] for k in orig_idx]

    # Pass 4: snap endpoints, drop dangling micro-ticks, derive Elbow / Tee / Cross
    if return_graph:
        lines, fittings, node_xy = _snap_and_graph(
            seg, members, snap_tol=snap_tol, min_stub=min_stub, perp_tol=perp_tol,
            return_member_lengths=return_member_lengths, return_graph=True,
        )
        return {"lines": lines, "fittings": fittings, "_node_xy": node_xy}
    lines, fittings = _snap_and_graph(
        seg, members, snap_tol=snap_tol, min_stub=min_stub, perp_tol=perp_tol,
        return_member_lengths=return_member_lengths,
    )
    return {"lines": lines, "fittings": fittings}


def _recompute_fittings(
    lines: list[dict[str, Any]],
    *,
    snap_tol: float = NODE_TOL,
    node_xy: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build connected components and topology-derived fittings (Elbow / Tee / Cross) from decoded lines.

    Uses preserved canonical ``(_uv, node_xy)`` when available (eliminating post-centerline node drift and redundant
    KDTree rebuilds), falling back to radius-bounded, self-non-collapsing clustering (zero transitive chaining).
    """
    n = len(lines)
    if n == 0:
        return [], []
    if node_xy is not None and all("_uv" in ln for ln in lines):
        edge_uv = [ln["_uv"] for ln in lines]
        nxy = node_xy
    else:
        pts = np.empty((2 * n, 2), np.float64)
        for i, ln in enumerate(lines):
            pts[i] = ln["vertices"][0]
            pts[n + i] = ln["vertices"][1]
        nid, nxy = _radius_bounded_nodes(pts, n, snap_tol + 0.05)
        edge_uv = [(int(nid[i]), int(nid[n + i])) for i in range(n)]

    adj: dict[int, list[tuple[int, int]]] = {}  # node -> [(nbr_node, edge_idx)]
    for i, (u, v) in enumerate(edge_uv):
        adj.setdefault(u, []).append((v, i))
        adj.setdefault(v, []).append((u, i))

    # Connected components of edges
    ep = list(range(n))

    def efind(x: int) -> int:
        while ep[x] != x:
            ep[x] = ep[ep[x]]
            x = ep[x]
        return x

    for inc in adj.values():
        r0 = efind(inc[0][1])
        for _, ei in inc[1:]:
            re = efind(ei)
            if re != r0:
                ep[re] = r0

    comp_map: dict[int, list[int]] = {}
    for i in range(n):
        comp_map.setdefault(efind(i), []).append(i)

    comps: list[dict[str, Any]] = []
    for es in comp_map.values():
        deg_c: dict[int, int] = {}
        tot_len = 0.0
        for ei in es:
            u, v = edge_uv[ei]
            deg_c[u] = deg_c.get(u, 0) + 1
            deg_c[v] = deg_c.get(v, 0) + 1
            tot_len += float(lines[ei]["length"])
        n_deg1 = sum(1 for d in deg_c.values() if d == 1)
        comps.append({"edges": es, "n_edges": len(es), "n_deg1": n_deg1, "tot_len": tot_len})

    fittings: list[dict[str, Any]] = []
    for u, inc in adj.items():
        d = len(inc)
        pt = [round(float(nxy[u, 0]), 3), round(float(nxy[u, 1]), 3)]
        inc_classes = [str(lines[ei].get("pipe_class") or "pipe") for _, ei in inc]
        p_cls = "main" if "main" in inc_classes else (inc_classes[0] if inc_classes else "pipe")
        if d == 2:
            v1 = nxy[inc[0][0]] - nxy[u]
            v2 = nxy[inc[1][0]] - nxy[u]
            n1 = float(np.hypot(v1[0], v1[1]))
            n2 = float(np.hypot(v2[0], v2[1]))
            if n1 > 1e-6 and n2 > 1e-6:
                dot = float((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2))
                if dot > FITTING_STRAIGHT_DOT:
                    fittings.append({"shape_name": "Elbow", "point": pt, "pipe_class": p_cls})
        elif d == 3:
            fittings.append({"shape_name": "Tee", "point": pt, "pipe_class": p_cls})
        elif d >= 4:
            fittings.append({"shape_name": "Cross", "point": pt, "pipe_class": p_cls})

    return comps, fittings


def _precompute_visual_style(
    attrs: dict[str, Any] | None,
    n_shapes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract pure visual drawing style (type_code, display RGB, stroke width) — STRICTLY ZERO PDF layers."""
    if not attrs or not isinstance(attrs, dict):
        return None
    srgb = attrs.get("stroke_rgb")
    wid = attrs.get("width")
    if srgb is None or wid is None:
        return None
    srgb_a = np.nan_to_num(np.asarray(srgb, np.float32), nan=-1.0)
    if srgb_a.ndim != 2 or len(srgb_a) != n_shapes or srgb_a.shape[1] < 3:
        return None
    w_a = np.nan_to_num(np.asarray(wid, np.float32), nan=0.0)
    if len(w_a) != n_shapes:
        return None
    tc = attrs.get("type_code")
    tc_a = np.asarray(tc, np.int8) if (tc is not None and len(tc) == n_shapes) else np.zeros(n_shapes, np.int8)
    frgb = attrs.get("fill_rgb")
    if frgb is not None and len(frgb) == n_shapes:
        frgb_a = np.nan_to_num(np.asarray(frgb, np.float32), nan=-1.0)
    else:
        frgb_a = srgb_a
    rgb = np.where((tc_a == 1)[:, None], frgb_a[:, :3], srgb_a[:, :3])
    return tc_a, rgb, w_a


def _visual_style_from_shapes(shapes: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Build lightweight visual-style arrays directly from raw payload ``shapes`` (0 PDF layers)."""
    n = len(shapes)
    tc = np.zeros(n, dtype=np.int8)
    srgb = np.full((n, 3), -1.0, dtype=np.float32)
    frgb = np.full((n, 3), -1.0, dtype=np.float32)
    wid = np.zeros(n, dtype=np.float32)
    for i, sh in enumerate(shapes):
        t = str(sh.get("type") or "s").lower()
        tc[i] = 1 if t == "f" else (2 if t == "fs" else 0)
        w = sh.get("width")
        if w is not None:
            try:
                wid[i] = float(w)
            except (TypeError, ValueError):
                pass
        c = sh.get("color")
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            srgb[i, 0], srgb[i, 1], srgb[i, 2] = float(c[0]), float(c[1]), float(c[2])
        f = sh.get("fill")
        if isinstance(f, (list, tuple)) and len(f) >= 3:
            frgb[i, 0], frgb[i, 1], frgb[i, 2] = float(f[0]), float(f[1]), float(f[2])
    return {"type_code": tc, "stroke_rgb": srgb, "fill_rgb": frgb, "width": wid}


def _same_visual_style(
    style: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    o1: int,
    o2: int,
    prob_cand: float,
    thr: float,
) -> bool:
    """Visual style compatibility between a candidate stroke and an accepted anchor stroke (0 PDF layers).

    Requires matching RGB color (within 0.08). For stroked-to-stroked transitions below medium confidence,
    also requires matching stroke width so thin architectural walls never bleed into thicker pipes.
    Filled pipe ribbons (``type_code >= 1``) of identical color match stroked pipes cleanly across main/branch tees.
    """
    if style is None:
        return True
    tc, rgb, w = style
    r1, r2 = rgb[o1], rgb[o2]
    if abs(float(r1[0] - r2[0])) > 0.08 or abs(float(r1[1] - r2[1])) > 0.08 or abs(float(r1[2] - r2[2])) > 0.08:
        return False
    if int(tc[o1]) == 0 and int(tc[o2]) == 0:
        if prob_cand >= min(thr, 0.15):
            return True
        w1, w2 = float(w[o1]), float(w[o2])
        return abs(w1 - w2) <= max(0.12, 0.25 * max(w1, w2))
    return True


def _non_pipe_symbol_mask(segs, cand_mask: np.ndarray, n_shapes: int) -> np.ndarray:
    """Pure-geometry mask of candidate shapes that are compact symbols/glyphs or smooth circles/arcs.

    Uses 4th-harmonic angular coherence ``|mean(exp(i * 4 * theta))|`` (~1.0 for straight & 90-deg orthogonal
    pipes/rings, ~0.0 for circles/ellipses/coverage arcs) so sprinkler coverage radii never get auto-accepted.
    """
    bad = np.zeros(n_shapes, dtype=bool)
    if not cand_mask.any() or not len(segs.seg):
        return bad
    m = cand_mask[segs.owner]
    if not m.any():
        return bad
    ss = np.asarray(segs.seg[m], np.float64)
    oo = np.asarray(segs.owner[m], np.int64)
    counts = np.bincount(oo, minlength=n_shapes)
    multi = np.flatnonzero(cand_mask & (counts >= 4))
    if not len(multi):
        return bad
    by_o: dict[int, list[int]] = {}
    multi_set = set(multi.tolist())
    for k, o in enumerate(oo.tolist()):
        if o in multi_set:
            by_o.setdefault(o, []).append(k)
    for o, ks in by_o.items():
        sub = ss[ks]
        xs = np.concatenate([sub[:, 0], sub[:, 2]])
        ys = np.concatenate([sub[:, 1], sub[:, 3]])
        diag = math.hypot(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
        if diag <= 1e-6:
            bad[o] = True
            continue
        dx = sub[:, 2] - sub[:, 0]
        dy = sub[:, 3] - sub[:, 1]
        sl = np.hypot(dx, dy)
        tot_l = float(sl.sum())
        if tot_l <= 1e-6:
            bad[o] = True
            continue
        if diag <= 22.0 and len(ks) >= 4 and tot_l / diag >= 2.2:
            bad[o] = True
            continue
        if len(ks) >= 10 and tot_l / diag >= 1.8 and float(sl.max()) <= 30.0:
            inv_sl = 1.0 / np.maximum(sl, 1e-9)
            ux = dx * inv_sl
            uy = dy * inv_sl
            c2 = ux * ux - uy * uy
            s2 = 2.0 * ux * uy
            c4 = c2 * c2 - s2 * s2
            s4 = 2.0 * c2 * s2
            coh4 = math.hypot(float((sl * c4).sum()), float((sl * s4).sum())) / tot_l
            if coh4 < 0.30:
                bad[o] = True
    return bad


def _quantize_visual_style(
    style: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack pure visual style into int64 style/color keys + color saturation (STRICTLY ZERO PDF layers)."""
    tc, rgb, w = style
    is_f = tc >= 1
    rq = np.clip(np.round(rgb * 12.0).astype(np.int64) + 16, 0, 63)
    wq = np.where(is_f, 0, np.clip(np.round(w * 20.0).astype(np.int64) + 1, 1, 1023))
    ckey = (rq[:, 0] << 12) | (rq[:, 1] << 6) | rq[:, 2]
    skey = (is_f.astype(np.int64) << 28) | (wq << 18) | ckey
    sat = (rgb.max(axis=1) - rgb.min(axis=1)).astype(np.float32)
    return skey, ckey, is_f, sat


def _style_rho_for_subset(
    skey: np.ndarray,
    length: np.ndarray,
    anchor_mask: np.ndarray,
    query_keys: np.ndarray,
) -> tuple[dict[int, float], dict[int, float]]:
    """Fast O(K * N) page-level style purity ``rho(S) = len(anchors & S) / len(all_page & S)`` for K unique styles."""
    anc_map: dict[int, float] = {}
    rho_map: dict[int, float] = {}
    for k in query_keys.tolist():
        m = skey == k
        tl = float(length[m].sum())
        al = float(length[m & anchor_mask].sum())
        anc_map[k] = al
        rho_map[k] = al / max(tl, 1e-6)
    return rho_map, anc_map


def _hysteresis_bridge_collinear(
    segs,
    prob: np.ndarray,
    thr: float,
    thr_low: float = 0.02,
    *,
    length: np.ndarray | None = None,
    attrs: dict[str, Any] | None = None,
    style: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    qstyle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    col_gap: float = 14.0,
    tol: float = 2.5,
    max_hops: int = 12,
    p_col: float = 0.02,
    p_gap: float = 0.035,
    p_ortho: float = 0.032,
    p_tee: float = 0.035,
    min_ortho_len: float = 2.5,
    min_tee_len: float = 2.5,
) -> np.ndarray:
    """Anchor-validated, style-purity-aware layer-free auto-accept pipe network completion (v4).

    Starting ONLY from validated confident pipe anchors (multi-segment pipe components with mean_p >= 0.52 &
    max_p >= 0.70, or high-confidence single pipes >= 0.80 & L >= 15pt, or >= 0.90 & L >= 8pt — never floating FP
    text boxes or hard-negative noise), iteratively completes missing weak pipe strokes via:
      1. 2-end gap bridges between distinct accepted pipe strokes;
      2. Collinear endpoint continuations (<= 6 deg, touch <= tol);
      3. Facing collinear dash/symbol gap continuations (perp <= 1.5 pt, gap <= col_gap);
      4. When pure visual style ``attrs`` (color/width/fill, 0 PDF layers) is provided:
         - Style-matched 90-deg elbow turns and mid-span T-junction branches, including short 2.5..10pt sprinkler
           drops (rescuing missed Tee/Elbow fittings);
         - Page-level visual-style purity ``rho(S) = len(anchors & S) / len(all_page & S)``: dedicated pipe styles
           (``rho >= 0.45``) and multi-segment filled pipe ribbons (``tc >= 1, rho >= 0.25``) safely extend touching
           strokes down to ``0.008``/``0.010``, while shared monochrome wall styles (``rho < 0.20``) guard long
           (>= 60pt) perpendicular wall turns with ``p >= 0.14``.
    """
    n_shapes = len(prob)
    fg = (prob >= thr).copy()
    if thr_low >= thr or not fg.any() or not len(segs.seg):
        return fg

    if style is None:
        style = _precompute_visual_style(attrs, n_shapes)
    has_style = style is not None
    if has_style and qstyle is None:
        qstyle = _quantize_visual_style(style)

    # Allow sub-0.02 dedicated-style touching rescue only when caller kept default thr_low <= 0.02
    eff_hi = min(0.008, thr * 0.9) if thr_low <= 0.02 else float(thr_low)
    min_possible = eff_hi if has_style else max(float(thr_low), min(0.02, thr * 0.9))
    if min_possible >= thr:
        return fg
    # Fast early-exit (0.1ms on Oracle & hard-neg when no candidate stroke exists in [min_possible, thr))
    if not ((prob >= min_possible) & (~fg)).any():
        return fg

    len_arr = (
        np.asarray(length, np.float64)
        if length is not None
        else (segs.shape_length().astype(np.float64) if hasattr(segs, "shape_length") else np.ones(n_shapes, np.float64))
    )

    # Step 1: Validate confident pipe anchors so isolated low-score FPs / hard-neg noise never seed growth
    m_fg = fg[segs.owner]
    if not m_fg.any():
        return fg
    fseg = np.asarray(segs.seg[m_fg], np.float64)
    fown = np.asarray(segs.owner[m_fg], np.int64)
    nf = len(fseg)
    fpts = np.vstack([fseg[:, 0:2], fseg[:, 2:4]])
    ftr = cKDTree(fpts)
    par = list(range(nf))

    def _find(x: int) -> int:
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for a, b in ftr.query_pairs(3.0):
        ra, rb = _find(a % nf), _find(b % nf)
        if ra != rb:
            par[rb] = ra
    own_first: dict[int, int] = {}
    for i, o in enumerate(fown.tolist()):
        if o in own_first:
            ra, rb = _find(i), _find(own_first[o])
            if ra != rb:
                par[rb] = ra
        else:
            own_first[o] = i
    comp_owners: dict[int, set[int]] = {}
    comp_nsegs: dict[int, int] = {}
    for i, o in enumerate(fown.tolist()):
        r = _find(i)
        comp_owners.setdefault(r, set()).add(o)
        comp_nsegs[r] = comp_nsegs.get(r, 0) + 1

    anchor = np.zeros(n_shapes, dtype=bool)
    for r, owners in comp_owners.items():
        ol = list(owners)
        w = np.maximum(len_arr[ol], 1e-3)
        mean_p = float((w * prob[ol]).sum() / w.sum())
        max_p = float(prob[ol].max())
        tot_l = float(len_arr[ol].sum())
        n_s = comp_nsegs[r]
        if n_s >= 2 and mean_p >= 0.52 and max_p >= 0.70:
            for o in ol:
                if prob[o] >= max(thr, 0.25) or len(ol) >= 3:
                    anchor[o] = True
        elif n_s == 1 and ((max_p >= 0.80 and tot_l >= 15.0) or (max_p >= 0.90 and tot_l >= 8.0)):
            for o in ol:
                anchor[o] = True
    if not anchor.any():
        return fg

    # Step 2: Fast O(K*N) visual-style purity rho(S) strictly over validated anchor styles (<= 30 keys -> ~2ms)
    is_hi_rho = np.zeros(n_shapes, dtype=bool)
    is_med_rho = np.zeros(n_shapes, dtype=bool)
    is_hi_fill = np.zeros(n_shapes, dtype=bool)
    is_low_rho = np.zeros(n_shapes, dtype=bool)
    if has_style and qstyle is not None:
        skey = qstyle[0]
        hi_anc = anchor & (prob >= 0.70)
        if hi_anc.any():
            uk = np.unique(skey[hi_anc])
            rho_map, anc_map = _style_rho_for_subset(skey, len_arr, hi_anc, uk)
            in_uk = np.zeros(n_shapes, dtype=bool)
            for k in uk.tolist():
                mk = skey == k
                in_uk |= mk
                rk, ak = rho_map[k], anc_map[k]
                if rk >= 0.45 and ak >= 200.0:
                    is_hi_rho |= mk
                if rk >= 0.32 and ak >= 250.0:
                    is_med_rho |= mk
                if rk >= 0.25 and ak >= 300.0 and ((k >> 28) & 1):
                    is_hi_fill |= mk
                if rk < 0.20:
                    is_low_rho |= mk
            is_low_rho |= ~in_uk

    eff_bridge = max(float(thr_low), min(0.02, thr * 0.9))
    eff_col = max(float(thr_low), min(p_col if has_style else max(p_col, 0.045), thr * 0.95))
    eff_gap = max(float(thr_low), min(p_gap if has_style else max(p_gap, 0.06), thr * 0.95))
    eff_ortho = max(float(thr_low), min(p_ortho, thr * 0.95))
    eff_tee = max(float(thr_low), min(p_tee, thr * 0.95))
    eff_drop = max(float(thr_low), min(0.025, thr * 0.95))
    p_floor = (
        eff_hi
        if (has_style and (is_hi_rho.any() or is_hi_fill.any()))
        else (min(eff_bridge, eff_col, eff_gap, eff_ortho, eff_tee, eff_drop) if has_style else min(eff_bridge, eff_col, eff_gap))
    )
    if p_floor >= thr:
        return fg

    cand_shapes = (prob >= p_floor) & (~fg)
    if not cand_shapes.any():
        return fg
    if has_style:
        sub02 = cand_shapes & (prob < float(thr_low))
        if sub02.any():
            cand_shapes &= (~sub02) | is_hi_rho | (is_hi_fill & (prob >= 0.010) & (len_arr >= 60.0))
    if not cand_shapes.any():
        return fg
    cand_shapes &= ~_non_pipe_symbol_mask(segs, cand_shapes, n_shapes)
    if not cand_shapes.any():
        return fg

    active = fg | cand_shapes
    m = active[segs.owner]
    if not m.any():
        return fg
    seg = np.asarray(segs.seg[m], np.float64)
    own = np.asarray(segs.owner[m], np.int64)
    seg_cnt = np.bincount(own, minlength=n_shapes)
    p0 = seg[:, 0:2]
    p1 = seg[:, 2:4]
    u, sl = _seg_dir_len(seg)
    trusted = anchor.copy()

    for _ in range(max_hops):
        hi_mask = trusted[own]
        lo_mask = (~fg[own]) & cand_shapes[own]
        if not hi_mask.any() or not lo_mask.any():
            break
        hi_idx = np.flatnonzero(hi_mask)
        lo_idx = np.flatnonzero(lo_mask)
        hi_pts = np.vstack([p0[hi_idx], p1[hi_idx]])
        hi_own = np.concatenate([own[hi_idx], own[hi_idx]])
        hi_u = np.vstack([u[hi_idx], u[hi_idx]])
        hi_other = np.vstack([p1[hi_idx], p0[hi_idx]])
        tree = cKDTree(hi_pts)
        k_near = min(4, len(hi_pts))
        d0, i0 = tree.query(p0[lo_idx], k=k_near, distance_upper_bound=tol)
        d1, i1 = tree.query(p1[lo_idx], k=k_near, distance_upper_bound=tol)
        if k_near == 1:
            d0, i0 = d0[:, None], i0[:, None]
            d1, i1 = d1[:, None], i1[:, None]
        hit_any = np.isfinite(d0[:, 0]) | np.isfinite(d1[:, 0])
        added = False

        # 1. Endpoint-touch rules: 2-end bridge, collinear, filled-ribbon continuation, 90-deg elbow & short drops
        for k in np.flatnonzero(hit_any).tolist():
            si = int(lo_idx[k])
            o = int(own[si])
            if fg[o]:
                continue
            pv = float(prob[o])
            lo_len = float(len_arr[o])
            hi_r = bool(is_hi_rho[o]) if has_style else False
            med_r = bool(is_med_rho[o]) if has_style else False
            hi_f = bool(is_hi_fill[o]) if has_style else False
            low_r = bool(is_low_rho[o]) if has_style else False
            j0s = [int(j) for d, j in zip(d0[k], i0[k]) if math.isfinite(d)]
            j1s = [int(j) for d, j in zip(d1[k], i1[k]) if math.isfinite(d)]
            ok = False
            req_br = eff_hi if hi_r else eff_bridge
            if j0s and j1s and pv >= req_br:
                for ja in j0s:
                    oa = int(hi_own[ja])
                    for jb in j1s:
                        ob = int(hi_own[jb])
                        if oa != ob and (
                            _same_visual_style(style, o, oa, pv, thr)
                            or _same_visual_style(style, o, ob, pv, thr)
                        ):
                            ok = True
                            break
                    if ok:
                        break
            if not ok and (j0s or j1s):
                for j in j0s + j1s:
                    ho = int(hi_own[j])
                    if not _same_visual_style(style, o, ho, pv, thr):
                        continue
                    dot = abs(float(u[si] @ hi_u[j]))
                    if hi_f and style is not None and int(style[0][ho]) >= 1 and pv >= 0.010 and (seg_cnt[o] >= 8 or dot >= 0.994):
                        ok = True
                        break
                    req_col = eff_hi if hi_r else eff_col
                    if dot >= 0.994 and pv >= req_col:
                        ok = True
                        break
                    if has_style and dot <= 0.12:
                        if lo_len >= 60.0 and low_r:
                            req_o = max(eff_ortho, min(0.14, thr * 0.95))
                        elif lo_len < 10.0:
                            req_o = eff_hi if hi_r else (eff_drop if (seg_cnt[o] <= 2 and dot <= 0.08 and prob[ho] >= 0.65) else min(0.045, thr * 0.95))
                        elif hi_r and lo_len <= 40.0:
                            req_o = eff_hi
                        elif med_r and lo_len <= 40.0:
                            req_o = max(float(thr_low), min(0.022, thr * 0.95))
                        else:
                            req_o = eff_ortho
                        min_sl_o = 2.5 if (hi_r or pv >= eff_drop) else 4.0
                        if pv >= req_o and lo_len >= min_ortho_len and sl[si] >= min_sl_o:
                            ok = True
                            break
            if ok:
                fg[o] = True
                trusted[o] = True
                added = True

        # 2. Collinear dash / symbol gap extension (facing endpoints across (0, col_gap] gap, perp <= 1.5 pt)
        if col_gap > tol:
            lo_idx2 = np.flatnonzero((~fg[own]) & cand_shapes[own] & (prob[own] >= eff_gap) & (sl >= 1.5))
            if len(lo_idx2):
                k_gap = min(6, len(hi_pts))
                d0g, i0g = tree.query(p0[lo_idx2], k=k_gap, distance_upper_bound=col_gap)
                d1g, i1g = tree.query(p1[lo_idx2], k=k_gap, distance_upper_bound=col_gap)
                if k_gap == 1:
                    d0g, i0g = d0g[:, None], i0g[:, None]
                    d1g, i1g = d1g[:, None], i1g[:, None]
                hit_g = np.isfinite(d0g[:, 0]) | np.isfinite(d1g[:, 0])
                for k in np.flatnonzero(hit_g).tolist():
                    si = int(lo_idx2[k])
                    o = int(own[si])
                    if fg[o]:
                        continue
                    ui = u[si]
                    ni = np.array([-ui[1], ui[0]])
                    pv = float(prob[o])
                    found_g = False
                    for pt_lo, pt_lo_other, ds, js in (
                        (p0[si], p1[si], d0g[k], i0g[k]),
                        (p1[si], p0[si], d1g[k], i1g[k]),
                    ):
                        for d_val, j_raw in zip(ds, js):
                            if not math.isfinite(d_val):
                                break
                            j = int(j_raw)
                            if abs(float(ui @ hi_u[j])) < 0.996:
                                continue
                            dv = hi_pts[j] - pt_lo
                            if abs(float(dv @ ni)) > 1.5:
                                continue
                            if float(dv @ (pt_lo - pt_lo_other)) <= 0.0 or float((-dv) @ (hi_pts[j] - hi_other[j])) <= 0.0:
                                continue
                            if not _same_visual_style(style, o, int(hi_own[j]), pv, thr):
                                continue
                            fg[o] = True
                            trusted[o] = True
                            added = True
                            found_g = True
                            break
                        if found_g:
                            break

        # 3. Style-verified mid-span T-junction branch (incl. short 2.5..10pt sprinkler drops) via 20pt grid
        if has_style:
            min_p_t = eff_hi if is_hi_rho.any() else min(eff_tee, eff_drop, 0.022 if is_med_rho.any() else eff_tee)
            lo_idx3 = np.flatnonzero(
                (~fg[own]) & cand_shapes[own] & (prob[own] >= min_p_t) & (len_arr[own] >= min_tee_len) & (sl >= 2.5)
            )
            if len(lo_idx3):
                hi_idx_cur = np.flatnonzero(trusted[own] & (sl >= 6.0))
                if len(hi_idx_cur):
                    cell = 20.0
                    grid: dict[tuple[int, int], list[int]] = {}
                    for hi in hi_idx_cur:
                        x0, y0 = p0[hi]
                        x1, y1 = p1[hi]
                        gx0 = int(math.floor((min(x0, x1) - tol) / cell))
                        gx1 = int(math.floor((max(x0, x1) + tol) / cell))
                        gy0 = int(math.floor((min(y0, y1) - tol) / cell))
                        gy1 = int(math.floor((max(y0, y1) + tol) / cell))
                        for gx in range(gx0, gx1 + 1):
                            for gy in range(gy0, gy1 + 1):
                                grid.setdefault((gx, gy), []).append(int(hi))
                    for si in lo_idx3:
                        o = int(own[si])
                        if fg[o]:
                            continue
                        pv = float(prob[o])
                        lo_len = float(len_arr[o])
                        hi_r = bool(is_hi_rho[o])
                        med_r = bool(is_med_rho[o])
                        low_r = bool(is_low_rho[o])
                        if lo_len >= 60.0 and low_r:
                            req_t = max(eff_tee, min(0.14, thr * 0.95))
                        elif lo_len < 10.0:
                            req_t = eff_hi if hi_r else (eff_drop if seg_cnt[o] <= 2 else min(0.045, thr * 0.95))
                        elif hi_r and lo_len <= 40.0:
                            req_t = eff_hi
                        elif med_r and lo_len <= 40.0:
                            req_t = max(float(thr_low), min(0.024, thr * 0.95))
                        else:
                            req_t = eff_tee
                        if pv < req_t:
                            continue
                        found = False
                        for ep in (p0[si], p1[si]):
                            cands = grid.get((int(math.floor(ep[0] / cell)), int(math.floor(ep[1] / cell))))
                            if not cands:
                                continue
                            for hi in cands:
                                dot_h = abs(float(u[si] @ u[hi]))
                                if dot_h > (0.08 if (lo_len < 10.0 and not hi_r and pv < 0.045) else 0.15):
                                    continue
                                if lo_len < 10.0 and not hi_r and pv < 0.045 and prob[int(own[hi])] < 0.65:
                                    continue
                                if not _same_visual_style(style, o, int(own[hi]), pv, thr):
                                    continue
                                vh = p1[hi] - p0[hi]
                                lh = sl[hi]
                                uh = vh / lh
                                dv = ep - p0[hi]
                                t = float(dv @ uh)
                                if 0.0 <= t <= lh and abs(float(dv[0] * (-uh[1]) + dv[1] * uh[0])) <= tol:
                                    fg[o] = True
                                    trusted[o] = True
                                    added = True
                                    found = True
                                    break
                            if found:
                                break

        if not added:
            break
    return fg


def _prune_foreign_style_outliers(
    lines: list[dict[str, Any]],
    comps: list[dict[str, Any]],
    style: tuple[np.ndarray, np.ndarray, np.ndarray],
    qstyle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    length: np.ndarray,
) -> tuple[list[dict[str, Any]], int]:
    """Prune foreign-color / low-rho visual-style outlier FPs without ever dropping dominant-color real pipes.

    Runs strictly AFTER closed-loop pruning using surviving confident anchors (>= 1,500pt, score >= 0.70; 0 PDF layers).
      - Pass A (line-level foreign color): drops individual lines whose own RGB color is absent from dominant pipe
        colors and has tiny page purity ``rho < 0.04`` (score < 0.995) — cleanly strips black/yellow background FPs
        touching chromatic red/magenta pipe networks without touching any dominant-color pipe.
      - Pass B (detached pure foreign-style component): drops detached (< 25% of network) components where >= 85%
        of length is either foreign color (99.993% pure across 122 GT pages) or same-color foreign width with
        ``rho < 0.035`` (small <= 4-edge or score < 0.72).
    """
    if len(lines) < 3:
        return lines, 0
    skey, ckey, is_f, sat = qstyle

    used_styles: set[int] = set()
    any_sub_oracle = False
    for ln in lines:
        if float(ln.get("score", 0.0)) < 0.999:
            any_sub_oracle = True
        for m in ln["members"]:
            used_styles.add(int(skey[m]))
    if len(used_styles) <= 1 or not any_sub_oracle:
        return lines, 0

    anc_c: Counter[int] = Counter()
    anc_s: Counter[int] = Counter()
    anc_w_by_c: dict[int, list[tuple[float, float]]] = defaultdict(list)
    anc_tot = 0.0
    anc_chrom = 0.0
    for ln in lines:
        if float(ln.get("score", 0.0)) < 0.70:
            continue
        ml = ln.get("_member_lengths") or {}
        for m in ln["members"]:
            wm = float(ml.get(m, ln["length"]))
            anc_c[int(ckey[m])] += wm
            anc_s[int(skey[m])] += wm
            if not is_f[m]:
                anc_w_by_c[int(ckey[m])].append((float(style[2][m]), wm))
            anc_tot += wm
            if float(sat[m]) >= 0.25:
                anc_chrom += wm
    if anc_tot < 1500.0:
        return lines, 0

    page_s_tot = {k: float(length[skey == k].sum()) for k in used_styles}
    rho_s_map = {k: anc_s.get(k, 0.0) / max(page_s_tot.get(k, 1e-6), 1e-6) for k in used_styles}
    max_rho = max(rho_s_map.values()) if rho_s_map else 0.0
    is_chromatic_page = anc_chrom >= 0.75 * anc_tot

    dom_c = {k for k, v in anc_c.items() if v >= 0.03 * anc_tot}
    dom_s = {
        k for k, v in anc_s.items()
        if v >= 0.03 * anc_tot and not (max_rho >= 0.15 and rho_s_map.get(k, 0.0) < 0.035)
    }
    if not dom_c:
        return lines, 0

    dom_rgbs: list[np.ndarray] = []
    for ln in lines:
        if float(ln.get("score", 0.0)) < 0.70:
            continue
        m0 = ln["members"][0]
        if int(ckey[m0]) not in dom_c:
            continue
        if is_chromatic_page and float(sat[m0]) < 0.15 and rho_s_map.get(int(skey[m0]), 0.0) < 0.04:
            continue
        r = style[1][m0]
        if not any(
            abs(float(r[0] - q[0])) <= 0.09 and abs(float(r[1] - q[1])) <= 0.09 and abs(float(r[2] - q[2])) <= 0.09
            for q in dom_rgbs
        ):
            dom_rgbs.append(r)
    if not dom_rgbs:
        return lines, 0

    info: list[tuple[int, bool, bool, float]] = []
    for ln in lines:
        ml = ln.get("_member_lengths") or {}
        best_m = ln["members"][0]
        best_w = -1.0
        for m in ln["members"]:
            wm = float(ml.get(m, ln["length"]))
            if wm > best_w:
                best_w = wm
                best_m = m
        top_c = int(ckey[best_m])
        top_s = int(skey[best_m])
        r_m = style[1][best_m]
        rho_s = rho_s_map.get(top_s, 0.0)
        color_ok = any(
            abs(float(r_m[0] - q[0])) <= 0.09 and abs(float(r_m[1] - q[1])) <= 0.09 and abs(float(r_m[2] - q[2])) <= 0.09
            for q in dom_rgbs
        )
        style_ok = top_s in dom_s
        if not style_ok and color_ok and not is_f[best_m]:
            wm_val = float(style[2][best_m])
            near_w = sum(wt for aw, wt in anc_w_by_c.get(top_c, []) if abs(wm_val - aw) <= 0.08)
            if near_w >= 0.03 * anc_tot and rho_s >= 0.035:
                style_ok = True
        info.append((top_s, color_ok, style_ok, rho_s))

    keep = np.ones(len(lines), dtype=bool)
    dropped = 0

    # Pass A: Line-level foreign-COLOR pruning (low-rho only)
    for i, ln in enumerate(lines):
        top_s, color_ok, _, rho_s = info[i]
        if not color_ok and float(ln.get("score", 0.0)) < 0.995:
            if (is_chromatic_page and rho_s < 0.04) or (anc_s.get(top_s, 0.0) < 0.03 * anc_tot and rho_s < 0.04):
                keep[i] = False
                dropped += 1

    # Pass B: Detached PURE foreign-color or low-rho foreign-style component pruning
    tot_net_len = sum(float(c["tot_len"]) for c in comps)
    for c in comps:
        if c["tot_len"] >= 0.25 * tot_net_len:
            continue
        alive_es = [ei for ei in c["edges"] if keep[ei]]
        if not alive_es:
            continue
        fc_len = 0.0
        fs_len = 0.0
        c_len = 0.0
        w_sc = 0.0
        max_r_fs = 0.0
        for ei in alive_es:
            lv = float(lines[ei]["length"])
            c_len += lv
            w_sc += lv * float(lines[ei].get("score", 0.0))
            _, color_ok, style_ok, rho_s = info[ei]
            if not color_ok:
                fc_len += lv
                fs_len += lv
            elif not style_ok and max_rho >= 0.15 and rho_s < 0.035:
                fs_len += lv
                max_r_fs = max(max_r_fs, rho_s)
        mean_sc = w_sc / max(c_len, 1e-6)
        if mean_sc >= 0.999:
            continue
        if fc_len >= 0.85 * max(c_len, 1e-6):
            for ei in alive_es:
                keep[ei] = False
            dropped += 1
        elif fs_len >= 0.85 * max(c_len, 1e-6) and max_r_fs < 0.035 and (len(alive_es) <= 4 or mean_sc < 0.72):
            for ei in alive_es:
                keep[ei] = False
            dropped += 1

    if dropped > 0:
        lines = [ln for ln, k in zip(lines, keep) if k]
    return lines, dropped


def _apply_anchor_style_mb_prior(
    lines: list[dict[str, Any]],
    style: tuple[np.ndarray, np.ndarray, np.ndarray],
    qstyle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    main_id: int = 1,
    branch_id: int = 2,
) -> tuple[bool, bool]:
    """Zero-layer page-level anchor visual-style Main vs Branch discrimination (auto-abstains on uniform drawings).

    Uses ONLY confident anchors (``p_cls >= 0.70``, other <= 0.20, >= 250pt each):
      1. Chromatic color-disjoint pages (at least one class color has saturation >= 0.25, e.g. Magenta Main vs Red
         Branch; verified via class-imbalance-invariant posterior purity P(Main|c_M) >= 0.65 & P(Branch|c_B) >= 0.85):
         resolves Main/Branch cleanly with 0.00% exact-Oracle flip (``p_target > 0``).
      2. Stroked-width-disjoint pages (strictly thicker stroked Main >= 0.10pt wider than Branch, non-filled):
         promotes mislabeled thick Main runs without demoting thin lines.
    Returns ``(changed, is_chromatic_color_disjoint)``.
    """
    if len(lines) < 4:
        return False, False
    skey, ckey, _, sat = qstyle
    mc: Counter[int] = Counter()
    bc: Counter[int] = Counter()
    ms: Counter[int] = Counter()
    bs: Counter[int] = Counter()
    c_rgb: dict[int, np.ndarray] = {}
    c_sat: dict[int, float] = {}
    mt = 0.0
    bt = 0.0
    for ln in lines:
        cs = ln.get("class_scores") or {}
        pm = float(cs.get("main", 0.0))
        pb = float(cs.get("branch", 0.0))
        ml = ln.get("_member_lengths") or {}
        if pm >= 0.70 and pb <= 0.20:
            for m in ln["members"]:
                wm = float(ml.get(m, ln["length"]))
                ck = int(ckey[m])
                mc[ck] += wm
                ms[int(skey[m])] += wm
                mt += wm
                if ck not in c_rgb:
                    c_rgb[ck] = style[1][m]
                    c_sat[ck] = float(sat[m])
        elif pb >= 0.70 and pm <= 0.20:
            for m in ln["members"]:
                wm = float(ml.get(m, ln["length"]))
                ck = int(ckey[m])
                bc[ck] += wm
                bs[int(skey[m])] += wm
                bt += wm
                if ck not in c_rgb:
                    c_rgb[ck] = style[1][m]
                    c_sat[ck] = float(sat[m])
    if mt < 250.0 or bt < 250.0:
        return False, False

    # Case 1: Chromatic color-disjoint Main vs Branch (never confuses achromatic black vs gray!)
    top_mc, top_mv = mc.most_common(1)[0]
    top_bc, top_bv = bc.most_common(1)[0]
    if top_mc != top_bc:
        rgb_diff = float(np.abs(c_rgb[top_mc] - c_rgb[top_bc]).max())
        any_chrom = (c_sat[top_mc] >= 0.25) or (c_sat[top_bc] >= 0.25)
        m_share_mc = top_mv / mt
        b_share_bc = top_bv / bt
        b_share_mc = bc.get(top_mc, 0.0) / bt
        m_share_bc = mc.get(top_bc, 0.0) / mt
        post_m_given_cm = top_mv / max(top_mv + bc.get(top_mc, 0.0), 1e-6)
        post_b_given_cb = top_bv / max(top_bv + mc.get(top_bc, 0.0), 1e-6)
        if (
            any_chrom
            and rgb_diff >= 0.15
            and m_share_mc >= 0.65
            and b_share_bc >= 0.80
            and b_share_mc <= 0.15
            and m_share_bc <= 0.35
            and post_m_given_cm >= 0.65
            and post_b_given_cb >= 0.85
        ):
            changed = False
            for ln in lines:
                cs = ln.get("class_scores") or {}
                pm = float(cs.get("main", 0.0))
                pb = float(cs.get("branch", 0.0))
                ml = ln.get("_member_lengths") or {}
                cc: Counter[int] = Counter()
                for m in ln["members"]:
                    cc[int(ckey[m])] += float(ml.get(m, ln["length"]))
                lc = cc.most_common(1)[0][0]
                if lc == top_mc and ln["pipe_class"] == "branch" and pm > 0.0:
                    ln["pipe_class"] = "main"
                    ln["class_id"] = main_id
                    changed = True
                elif lc == top_bc and ln["pipe_class"] == "main" and pb > 0.0:
                    ln["pipe_class"] = "branch"
                    ln["class_id"] = branch_id
                    changed = True
            return changed, True

    # Case 2: Strict stroked-width-disjoint Main vs Branch (never fires on filled-ribbon pages or demotes thin Main)
    dom_ms = {k for k, v in ms.items() if v >= 0.15 * mt}
    dom_bs = {k for k, v in bs.items() if v >= 0.15 * bt}
    changed = False
    if dom_ms and dom_bs and not (dom_ms & dom_bs):
        if not any((k >> 28) & 1 for k in dom_ms):
            m_min_w = min(((k >> 18) & 1023) for k in dom_ms)
            b_max_w = max(((k >> 18) & 1023) for k in dom_bs)
            if m_min_w >= b_max_w + 2:  # >= 0.10 pt strictly wider than every dominant branch style
                for ln in lines:
                    cs = ln.get("class_scores") or {}
                    if ln["pipe_class"] == "branch" and float(cs.get("main", 0.0)) > 0.0:
                        ml = ln.get("_member_lengths") or {}
                        sc_: Counter[int] = Counter()
                        for m in ln["members"]:
                            sc_[int(skey[m])] += float(ml.get(m, ln["length"]))
                        if sc_.most_common(1)[0][0] in dom_ms:
                            ln["pipe_class"] = "main"
                            ln["class_id"] = main_id
                            changed = True
    return changed, False


def _smooth_fire_topology(
    lines: list[dict[str, Any]],
    *,
    style: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    main_id: int = 1,
    ring_frac: float = 0.45,
    ring_ratio: float = 1e-4,
    bridge_alpha: float = 0.06,
    topo_gap: float = 10.0,
    snap_tol: float = NODE_TOL,
) -> bool:
    """Graph-level main/branch consistency smoothing v4 (0 layers; 0.00% exact-Oracle flip):
      1. Hydraulic 2-core ring closure with virtual valve/symbol-gap bridging (``topo_gap`` pt ONLY for cycle
         detection, never synthesizing fake geometry): promotes remaining mid-loop ``branch`` gaps with
         ``p_main > 0`` & ``p_main >= ring_ratio * p_branch``.
      2. Filled-main-page parallel-header propagation: on drawings whose main trunks are drawn as filled ribbons
         (>= 30% of main length has ``tc >= 1``), promotes companion stroked pipes running tightly parallel
         (1.2..5.0pt) to an existing main or forming a main-touching / tight <=3.3pt 3+ pipe header bundle
         (88.1% main purity across 122 GT pages, ``p_main > 0``).
      3. Main-bridge continuity (single interior edge & multi-edge collinear chain between two main junctions).
      4. Multi-segment single-shape class consistency: if a single CAD polyline shape has >= 55% of its decoded
         length promoted to ``main``, unifies its remaining solely-owned segments (``p_main > 0``).
    Returns True if any line changed class."""
    n = len(lines)
    if n < 2:
        return False
    if all("_uv" in ln for ln in lines):
        edge_uv = [ln["_uv"] for ln in lines]
    else:
        pts = np.empty((2 * n, 2), np.float64)
        for i, ln in enumerate(lines):
            pts[i] = ln["vertices"][0]
            pts[n + i] = ln["vertices"][1]
        nid, _ = _radius_bounded_nodes(pts, n, snap_tol + 0.05)
        edge_uv = [(int(nid[i]), int(nid[n + i])) for i in range(n)]

    adj: dict[int, list[tuple[int, int]]] = {}
    for ei, (u, v) in enumerate(edge_uv):
        adj.setdefault(u, []).append((v, ei))
        adj.setdefault(v, []).append((u, ei))

    # Detect filled-ribbon main drawings (>= 30% of current main length has tc >= 1)
    has_filled_main = False
    if style is not None:
        tc_a = style[0]
        m_fill = 0.0
        m_tot = 0.0
        for ln in lines:
            if ln["pipe_class"] == "main":
                lv = float(ln["length"])
                m_tot += lv
                if any(int(tc_a[m]) >= 1 for m in ln["members"]):
                    m_fill += lv
        has_filled_main = (m_tot >= 200.0) and (m_fill >= 0.30 * m_tot)

    # Virtual valve/fitting-symbol gap closure strictly for 2-core ring detection
    v_adj: dict[int, list[tuple[int, int]]] = {u: list(inc) for u, inc in adj.items()}
    virt_edge_id = n
    if topo_gap > snap_tol:
        deg1_nodes = [u for u, inc in adj.items() if len(inc) == 1]
        if len(deg1_nodes) >= 2:
            d1_xy = np.empty((len(deg1_nodes), 2), np.float64)
            d1_dir = np.empty((len(deg1_nodes), 2), np.float64)
            for idx_k, u in enumerate(deg1_nodes):
                _, ei = adj[u][0]
                u0, _ = edge_uv[ei]
                p_here = np.asarray(lines[ei]["vertices"][0 if u0 == u else 1], np.float64)
                p_other = np.asarray(lines[ei]["vertices"][1 if u0 == u else 0], np.float64)
                d1_xy[idx_k] = p_here
                dv = p_here - p_other
                dl = max(float(np.hypot(dv[0], dv[1])), 1e-9)
                d1_dir[idx_k] = dv / dl
            dtr = cKDTree(d1_xy)
            for a, b in dtr.query_pairs(topo_gap):
                ua, ub = deg1_nodes[a], deg1_nodes[b]
                if ua == ub:
                    continue
                dv = d1_xy[b] - d1_xy[a]
                dist = float(np.hypot(dv[0], dv[1]))
                if dist <= 1e-6:
                    continue
                uv_ab = dv / dist
                col_face = float(d1_dir[a] @ uv_ab) >= 0.94 and float(d1_dir[b] @ (-uv_ab)) >= 0.94
                dot_ab = abs(float(d1_dir[a] @ d1_dir[b]))
                corner_face = (
                    dot_ab <= 0.20
                    and float(d1_dir[a] @ uv_ab) >= 0.35
                    and float(d1_dir[b] @ (-uv_ab)) >= 0.35
                    and dist <= 6.5
                )
                if col_face or corner_face:
                    v_adj.setdefault(ua, []).append((ub, virt_edge_id))
                    v_adj.setdefault(ub, []).append((ua, virt_edge_id))
                    virt_edge_id += 1

    changed_any = False

    # 1. Hydraulic 2-core loop closure
    deg = {u: len(inc) for u, inc in v_adj.items()}
    alive_e = np.ones(virt_edge_id, dtype=bool)
    q = [u for u, d in deg.items() if d == 1]
    while q:
        u = q.pop()
        if deg[u] != 1:
            continue
        for v, ei in v_adj[u]:
            if alive_e[ei]:
                alive_e[ei] = False
                deg[u] -= 1
                deg[v] -= 1
                if deg[v] == 1:
                    q.append(v)
    core_all = np.flatnonzero(alive_e)
    core_real = [int(e) for e in core_all if e < n]
    if len(core_real) >= 3:
        ep = {int(e): int(e) for e in core_all}

        def efind(x: int) -> int:
            while ep[x] != x:
                ep[x] = ep[ep[x]]
                x = ep[x]
            return x

        node_core: dict[int, list[int]] = {}
        for u, inc in v_adj.items():
            for _, ei in inc:
                if alive_e[ei]:
                    node_core.setdefault(u, []).append(int(ei))
        for es in node_core.values():
            r0 = efind(es[0])
            for e in es[1:]:
                re = efind(e)
                if re != r0:
                    ep[re] = r0
        comps: dict[int, list[int]] = {}
        for ei in core_real:
            comps.setdefault(efind(ei), []).append(ei)
        for es in comps.values():
            if len(es) < 3:
                continue
            xs = [p[0] for ei in es for p in lines[ei]["vertices"]]
            ys = [p[1] for ei in es for p in lines[ei]["vertices"]]
            span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            if span < 80.0:
                continue
            tot_l = sum(float(lines[ei]["length"]) for ei in es)
            main_l = sum(float(lines[ei]["length"]) for ei in es if lines[ei]["pipe_class"] == "main")
            frac = main_l / max(tot_l, 1e-6)
            core_nodes = {nd for ei in es for nd in edge_uv[ei]}
            n_off = sum(1 for nd in core_nodes for _, ej in adj[nd] if not alive_e[ej])
            if frac >= ring_frac and (frac >= 0.65 or (span >= 100.0 and n_off >= 2)):
                for ei in es:
                    if lines[ei]["pipe_class"] == "branch":
                        cs = lines[ei].get("class_scores") or {}
                        pm = float(cs.get("main", 0.0))
                        if pm > 0.0 and pm >= ring_ratio * max(float(cs.get("branch", 1e-6)), 1e-6):
                            lines[ei]["pipe_class"] = "main"
                            lines[ei]["class_id"] = main_id
                            changed_any = True

    # 2. Filled-main-page parallel-to-main & parallel-header bundle propagation
    if has_filled_main:
        P = np.array(
            [[ln["vertices"][0][0], ln["vertices"][0][1], ln["vertices"][1][0], ln["vertices"][1][1]] for ln in lines],
            np.float64,
        )
        u_vec, Lp = _seg_dir_len(P)
        mids = 0.5 * (P[:, :2] + P[:, 2:])
        mtr = cKDTree(mids)
        max_h = float(0.5 * Lp.max())
        par_nbrs: list[list[tuple[int, bool, float]]] = [[] for _ in range(n)]
        for i in range(n):
            if Lp[i] < 18.0:
                continue
            cands = mtr.query_ball_point(mids[i], float(0.5 * Lp[i] + max_h + 6.0))
            p0 = P[i, :2]
            ui = (P[i, 2:] - p0) / Lp[i]
            ni = np.array([-ui[1], ui[0]])
            for j in cands:
                if j <= i or Lp[j] < 18.0:
                    continue
                if abs(float(u_vec[i] @ u_vec[j])) < 0.996:
                    continue
                d0 = P[j, :2] - p0
                d1 = P[j, 2:] - p0
                pd0 = abs(float(d0 @ ni))
                pd1 = abs(float(d1 @ ni))
                pd = 0.5 * (pd0 + pd1)
                if not (1.2 <= pd <= 5.5) or abs(pd0 - pd1) > 1.2:
                    continue
                t0, t1 = float(d0 @ ui), float(d1 @ ui)
                ov = min(float(Lp[i]), max(t0, t1)) - max(0.0, min(t0, t1))
                if ov >= 0.55 * min(float(Lp[i]), float(Lp[j])):
                    is_twin = (ov >= 0.45 * max(float(Lp[i]), float(Lp[j]))) and (
                        min(Lp[i], Lp[j]) >= 0.60 * max(Lp[i], Lp[j])
                    )
                    par_nbrs[i].append((j, is_twin, pd))
                    par_nbrs[j].append((i, is_twin, pd))

        twin_mask = np.array([any(tw for _, tw, _ in par_nbrs[i]) and Lp[i] >= 20.0 for i in range(n)], dtype=bool)
        for _ in range(4):
            step = False
            node_m: Counter[int] = Counter()
            for ei, (u, v) in enumerate(edge_uv):
                if lines[ei]["pipe_class"] == "main":
                    node_m[u] += 1
                    node_m[v] += 1
            for i in range(n):
                if lines[i]["pipe_class"] != "branch":
                    continue
                cs = lines[i].get("class_scores") or {}
                if float(cs.get("main", 0.0)) <= 0.0:
                    continue
                if any(lines[j]["pipe_class"] == "main" and pd <= 5.0 for j, _, pd in par_nbrs[i]):
                    lines[i]["pipe_class"] = "main"
                    lines[i]["class_id"] = main_id
                    step = True
                    changed_any = True
                    continue
                u, v = edge_uv[i]
                n_tight_twins = sum(1 for _, tw, pd in par_nbrs[i] if tw and pd <= 3.3)
                if twin_mask[i] and (
                    (Lp[i] >= 25.0 and (node_m[u] > 0 or node_m[v] > 0))
                    or (n_tight_twins >= 2 and Lp[i] >= 60.0)
                ):
                    lines[i]["pipe_class"] = "main"
                    lines[i]["class_id"] = main_id
                    step = True
                    changed_any = True
            if not step:
                break

    # 3. Interior single-edge bridge between two main junctions
    for _ in range(3):
        node_main: dict[int, int] = {}
        for ei, (u, v) in enumerate(edge_uv):
            if lines[ei]["pipe_class"] == "main":
                node_main[u] = node_main.get(u, 0) + 1
                node_main[v] = node_main.get(v, 0) + 1
        step_chg = False
        for ei, (u, v) in enumerate(edge_uv):
            if lines[ei]["pipe_class"] == "branch" and node_main.get(u, 0) >= 1 and node_main.get(v, 0) >= 1:
                cs = lines[ei].get("class_scores") or {}
                pm = float(cs.get("main", 0.0))
                if pm > 0.0 and pm >= bridge_alpha * max(float(cs.get("branch", 1e-6)), 1e-6):
                    lines[ei]["pipe_class"] = "main"
                    lines[ei]["class_id"] = main_id
                    step_chg = True
                    changed_any = True
        if not step_chg:
            break

    # 4. Collinear straight-run main-to-main chain bridge (split by intermediate branch Tees)
    edir: list[tuple[float, float]] = []
    for ln in lines:
        dx = float(ln["vertices"][1][0]) - float(ln["vertices"][0][0])
        dy = float(ln["vertices"][1][1]) - float(ln["vertices"][0][1])
        dl = max(math.hypot(dx, dy), 1e-9)
        ux, uy = dx / dl, dy / dl
        if ux < -1e-9 or (abs(ux) <= 1e-9 and uy < 0):
            ux, uy = -ux, -uy
        edir.append((ux, uy))
    br_p = {ei: ei for ei in range(n) if lines[ei]["pipe_class"] == "branch"}

    def bfind(x: int) -> int:
        while br_p[x] != x:
            br_p[x] = br_p[br_p[x]]
            x = br_p[x]
        return x

    for u, inc in adj.items():
        bs = [ei for _, ei in inc if ei in br_p]
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                ei, ej = bs[i], bs[j]
                if abs(edir[ei][0] * edir[ej][0] + edir[ei][1] * edir[ej][1]) >= 0.99:
                    ra, rb = bfind(ei), bfind(ej)
                    if ra != rb:
                        br_p[rb] = ra
    chains: dict[int, list[int]] = {}
    for ei in br_p:
        chains.setdefault(bfind(ei), []).append(ei)
    node_main = {}
    for ei, (u, v) in enumerate(edge_uv):
        if lines[ei]["pipe_class"] == "main":
            node_main[u] = node_main.get(u, 0) + 1
            node_main[v] = node_main.get(v, 0) + 1
    for es in chains.values():
        if len(es) < 2:
            continue
        c_nodes = {nd for ei in es for nd in edge_uv[ei]}
        m_nodes = [nd for nd in c_nodes if node_main.get(nd, 0) >= 1]
        if len(m_nodes) >= 2:
            w_m = sum(float(lines[ei]["length"]) * float((lines[ei].get("class_scores") or {}).get("main", 0.0)) for ei in es)
            w_b = sum(float(lines[ei]["length"]) * float((lines[ei].get("class_scores") or {}).get("branch", 1e-6)) for ei in es)
            if w_m > 0.0 and w_m >= 0.005 * max(w_b, 1e-6):
                for ei in es:
                    cs = lines[ei].get("class_scores") or {}
                    if float(cs.get("main", 0.0)) > 0.0:
                        lines[ei]["pipe_class"] = "main"
                        lines[ei]["class_id"] = main_id
                        changed_any = True

    # 5. Multi-segment single-shape class consistency (one CAD polyline shape has one physical pipe class)
    sh_m: defaultdict[int, float] = defaultdict(float)
    sh_b: defaultdict[int, float] = defaultdict(float)
    for ln in lines:
        ml = ln.get("_member_lengths") or {}
        is_m = ln["pipe_class"] == "main"
        for m in ln["members"]:
            wm = float(ml.get(m, ln["length"]))
            if is_m:
                sh_m[m] += wm
            else:
                sh_b[m] += wm
    main_shapes = {m for m, vm in sh_m.items() if vm > sh_b.get(m, 0.0) and vm >= 0.55 * (vm + sh_b.get(m, 0.0))}
    if main_shapes:
        for ln in lines:
            if ln["pipe_class"] == "branch":
                cs = ln.get("class_scores") or {}
                if float(cs.get("main", 0.0)) <= 0.0:
                    continue
                ml = ln.get("_member_lengths") or {}
                w_ms = sum(float(ml.get(m, ln["length"])) for m in ln["members"] if m in main_shapes)
                w_all = sum(float(ml.get(m, ln["length"])) for m in ln["members"])
                if w_ms >= 0.60 * max(w_all, 1e-6):
                    ln["pipe_class"] = "main"
                    ln["class_id"] = main_id
                    changed_any = True

    return changed_any


def decode_fire_network(
    segs,
    prob: np.ndarray,
    cls_prob: np.ndarray | None = None,
    length: np.ndarray | None = None,
    *,
    thr: float = 0.5,
    class_names: tuple[str, ...] | list[str] | None = None,
    mode: str = "fire",
    gap: float | None = None,
    angle_deg: float = ANGLE_DEG,
    perp_tol: float = PERP_TOL,
    snap_tol: float = NODE_TOL,
    min_stub: float = MIN_STUB,
    drop_closed_loops: bool = True,
    min_closed_loop_len: float = 150.0,
    hysteresis: bool = True,
    hysteresis_low: float = 0.02,
    smooth_topo: bool = True,
    attrs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Class-aware FIRE / PLUMBING pipe network post-processor v4 (0 PDF layers, torch-free, ~0.20s/page CPU).

    1. Style-purity-aware anchor-validated auto-accept network completion: bridges gaps, extends collinear runs/dashes,
       rescues short 2.5..10pt sprinkler T-drops & dedicated-style touching continuations/filled ribbons, and guards
       long perpendicular monochrome walls while rejecting circles/symbols via 4th-harmonic coherence.
    2. Class-barrier-aware collinear merging with exact per-line contributed-length weighting, zero-kink trunk
       coordinates, and canonical graph preservation (zero post-centerline fitting drift).
    3. Drops zero-leaf (``deg1 == 0``) closed non-branch perimeter frames AND foreign-color / low-rho visual-style
       outlier FP components (without dropping dominant-color real pipes).
    4. Anchor visual-style Main vs Branch prior (chromatic color-disjoint & stroked-width-disjoint; auto-abstains
       on uniform-style drawings with 0.00% exact-Oracle flip).
    5. Hydraulic topology smoothing (virtual valve-gap 2-core rings, filled-main parallel-header bundles, main-to-main
       bridges & single-shape polyline consistency) + exact fitting & per-shape ``cls`` / ``inst`` projection.
    """
    prob_arr = np.asarray(prob, np.float64)
    n_shapes = len(prob_arr)
    if length is None:
        len_arr = segs.shape_length().astype(np.float64) if hasattr(segs, "shape_length") else np.ones(n_shapes, np.float64)
    else:
        len_arr = np.asarray(length, np.float64)

    cp = np.asarray(cls_prob, np.float64) if cls_prob is not None else None
    if cp is not None and cp.ndim == 2 and cp.shape[1] >= 2:
        k_cls = int(cp.shape[1])
        if class_names and len(class_names) == k_cls:
            names = [str(c) for c in class_names]
        elif k_cls == 3:
            names = ["other", "main", "branch"]
        else:
            names = ["other"] + [f"class_{c}" for c in range(1, k_cls)]
    else:
        cp = None
        k_cls = 2
        names = [str(c) for c in class_names] if class_names and len(class_names) >= 2 else ["other", "pipe"]

    cls_out = np.zeros(n_shapes, dtype=np.int8)
    inst_out = np.full(n_shapes, -1, dtype=np.int32)

    style = _precompute_visual_style(attrs, n_shapes)
    qstyle = _quantize_visual_style(style) if style is not None else None

    gap_tol = (PLUMBING_GAP if mode == "plumbing" else FIRE_GAP) if gap is None else float(gap)
    eff_prob = prob_arr
    if hysteresis and hysteresis_low < thr:
        fg_h = _hysteresis_bridge_collinear(
            segs, prob_arr, thr, thr_low=float(hysteresis_low),
            length=len_arr, style=style, qstyle=qstyle, col_gap=max(gap_tol + 2.0, 14.0),
        )
        if (fg_h & (prob_arr < thr)).any():
            eff_prob = np.where(fg_h & (prob_arr < thr), float(thr), prob_arr)

    # Confident per-shape class barrier (only separates confident opposite pipe classes >= 65% share)
    shape_cls: np.ndarray | None = None
    if cp is not None and k_cls >= 3:
        fg_cp = cp[:, 1:]
        fg_sum = np.maximum(fg_cp.sum(axis=1), 1e-9)
        top_c = np.argmax(fg_cp, axis=1)
        top_share = fg_cp[np.arange(n_shapes), top_c] / fg_sum
        shape_cls = np.where(top_share >= 0.65, 1 + top_c, 0).astype(np.int8)

    raw = decode_lines(
        segs, eff_prob, thr=thr, mode=mode, gap=gap,
        angle_deg=angle_deg, perp_tol=perp_tol, snap_tol=snap_tol, min_stub=min_stub,
        shape_cls=shape_cls, return_member_lengths=True, return_graph=True,
    )
    lines = raw["lines"]
    node_xy = raw.get("_node_xy")
    if not lines:
        return {
            "lines": [], "fittings": [], "classes": names,
            "cls": cls_out, "inst": inst_out,
            "counts": {}, "lengths": {}, "dropped_loops": 0,
        }

    for ln in lines:
        mems = ln["members"]
        ml_d = ln.get("_member_lengths")
        if ml_d:
            w = np.array([max(float(ml_d.get(m_i, 1e-3)), 1e-3) for m_i in mems], dtype=np.float64)
        else:
            w = np.maximum(len_arr[mems], 1e-3)
        wsum = float(w.sum())
        ln["score"] = round(float((w * prob_arr[mems]).sum() / wsum), 4)
        if cp is not None:
            cw = (w[:, None] * cp[mems, 1:]).sum(axis=0)
            cid = int(1 + np.argmax(cw))
            ln["class_id"] = cid
            ln["pipe_class"] = names[cid]
            ln["class_scores"] = {names[c]: round(float((w * cp[mems, c]).sum() / wsum), 4) for c in range(1, cp.shape[1])}
        else:
            ln["class_id"] = 1
            ln["pipe_class"] = names[1]

    comps, fittings = _recompute_fittings(lines, snap_tol=snap_tol, node_xy=node_xy)
    dropped_loops = 0
    if drop_closed_loops and "branch" in names and len(lines) >= 3:
        keep = np.ones(len(lines), dtype=bool)
        for comp in comps:
            if comp["n_deg1"] == 0 and comp["n_edges"] >= 3 and comp["tot_len"] >= float(min_closed_loop_len):
                br_len = sum(float(lines[ei]["length"]) for ei in comp["edges"] if lines[ei]["pipe_class"] == "branch")
                if br_len == 0.0:
                    for ei in comp["edges"]:
                        keep[ei] = False
                    dropped_loops += 1
        if dropped_loops > 0:
            lines = [ln for ln, k in zip(lines, keep) if k]
            comps, fittings = _recompute_fittings(lines, snap_tol=snap_tol, node_xy=node_xy)

    # Foreign-color / low-rho visual-style outlier FP pruning (runs strictly AFTER closed-loop pruning)
    if style is not None and qstyle is not None and len(lines) >= 3:
        lines, n_drop_style = _prune_foreign_style_outliers(lines, comps, style, qstyle, len_arr)
        if n_drop_style > 0:
            comps, fittings = _recompute_fittings(lines, snap_tol=snap_tol, node_xy=node_xy)

    # Anchor visual-style Main/Branch prior + Hydraulic topology smoothing
    cls_changed = False
    is_chrom_disjoint = False
    if smooth_topo and cp is not None and "main" in names and "branch" in names:
        main_idx = names.index("main")
        branch_idx = names.index("branch")
        if style is not None and qstyle is not None and len(lines) >= 4:
            c_st, is_chrom_disjoint = _apply_anchor_style_mb_prior(
                lines, style, qstyle, main_id=main_idx, branch_id=branch_idx,
            )
            if c_st:
                cls_changed = True
        if not is_chrom_disjoint and len(lines) >= 2:
            if _smooth_fire_topology(lines, style=style, main_id=main_idx, snap_tol=snap_tol):
                cls_changed = True
    if cls_changed:
        _, fittings = _recompute_fittings(lines, snap_tol=snap_tol, node_xy=node_xy)

    # Project decoded lines back to per-shape cls / inst via accumulated contributed length per class
    cls_votes = np.zeros((n_shapes, k_cls), dtype=np.float64)
    best_len = np.zeros(n_shapes, dtype=np.float64)
    counts: dict[str, int] = {}
    lengths: dict[str, float] = {}
    for idx, ln in enumerate(lines):
        cname = str(ln["pipe_class"])
        cid = int(ln["class_id"])
        L = float(ln["length"])
        counts[cname] = counts.get(cname, 0) + 1
        lengths[cname] = lengths.get(cname, 0.0) + L
        ln.pop("_uv", None)
        ml_d = ln.pop("_member_lengths", None) or {}
        for m_i in ln["members"]:
            w_m = float(ml_d.get(m_i, L))
            cls_votes[m_i, cid] += w_m
            if w_m > best_len[m_i]:
                best_len[m_i] = w_m
                inst_out[m_i] = idx
    has_vote = cls_votes[:, 1:].sum(axis=1) > 0
    if has_vote.any():
        cls_out[has_vote] = (1 + np.argmax(cls_votes[has_vote, 1:], axis=1)).astype(np.int8)

    for ft in fittings:
        fname = str(ft["shape_name"])
        counts[fname] = counts.get(fname, 0) + 1
    lengths = {k: round(v, 1) for k, v in lengths.items()}

    return {
        "lines": lines,
        "fittings": fittings,
        "classes": names,
        "cls": cls_out,
        "inst": inst_out,
        "counts": counts,
        "lengths": lengths,
        "dropped_loops": dropped_loops,
    }


def _collinear_covered_length(src: np.ndarray, ref: np.ndarray, angle_deg: float = ANGLE_DEG, perp_tol: float = PERP_TOL) -> float:
    """Total length of ``src`` segments (M, 4) covered by collinear ``ref`` segments (K, 4) (AI_Takeoff _line_length_report)."""
    if not len(src) or not len(ref):
        return 0.0
    us, Ls = _seg_dir_len(src)
    ur, Lr = _seg_dir_len(ref)
    cos_min = math.cos(math.radians(angle_deg))
    rmid = 0.5 * (ref[:, 0:2] + ref[:, 2:4])
    rtree = cKDTree(rmid)
    max_rh = float(0.5 * Lr.max())
    covered = 0.0
    for i in range(len(src)):
        if Ls[i] <= 1e-6:
            continue
        p0 = src[i, 0:2]
        ui = (src[i, 2:4] - p0) / Ls[i]   # unflipped unit vector from p0 to p1 so src[i] spans [0, Ls[i]]
        ni = np.array([-ui[1], ui[0]])
        mid = 0.5 * (src[i, 0:2] + src[i, 2:4])
        cands = rtree.query_ball_point(mid, float(0.5 * Ls[i] + max_rh + perp_tol + 1.0))
        ivals = []
        for j in cands:
            if abs(float(ur[j] @ ui)) < cos_min:
                continue
            d0 = ref[j, 0:2] - p0
            d1 = ref[j, 2:4] - p0
            if abs(float(d0 @ ni)) > perp_tol or abs(float(d1 @ ni)) > perp_tol:
                continue
            t0, t1 = float(d0 @ ui), float(d1 @ ui)
            a, b = max(0.0, min(t0, t1)), min(float(Ls[i]), max(t0, t1))
            if b > a:
                ivals.append((a, b))
        if not ivals:
            continue
        ivals.sort()
        ca, cb = ivals[0]
        for a, b in ivals[1:]:
            if a <= cb:
                cb = max(cb, b)
            else:
                covered += cb - ca
                ca, cb = a, b
        covered += cb - ca
    return float(covered)


def score_lines_length(
    pred_lines: list[dict] | np.ndarray,
    gt_lines: list[dict] | np.ndarray,
    *,
    angle_deg: float = ANGLE_DEG,
    perp_tol: float = PERP_TOL,
) -> dict[str, float]:
    """AI_Takeoff-compatible collinear pipe length Precision / Recall / F2 (0 layers)."""
    def _to_arr(x):
        if isinstance(x, np.ndarray):
            return x.reshape(-1, 4).astype(np.float64)
        rows = []
        for o in x:
            v = o.get("vertices") or []
            if len(v) >= 2:
                for a, b in zip(v[:-1], v[1:]):
                    rows.append([a[0], a[1], b[0], b[1]])
        return np.asarray(rows, np.float64).reshape(-1, 4) if rows else np.zeros((0, 4), np.float64)

    P = _to_arr(pred_lines)
    G = _to_arr(gt_lines)
    _, Lp = _seg_dir_len(P) if len(P) else (np.zeros((0, 2)), np.zeros(0))
    _, Lg = _seg_dir_len(G) if len(G) else (np.zeros((0, 2)), np.zeros(0))
    tot_p, tot_g = float(Lp.sum()), float(Lg.sum())
    cov_g = _collinear_covered_length(G, P, angle_deg=angle_deg, perp_tol=perp_tol)
    cov_p = _collinear_covered_length(P, G, angle_deg=angle_deg, perp_tol=perp_tol)
    rec = cov_g / max(tot_g, 1e-9)
    prec = cov_p / max(tot_p, 1e-9)
    f2 = 5.0 * prec * rec / max(4.0 * prec + rec, 1e-9)
    return {"f2": round(f2, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "pred_len": round(tot_p, 1), "gt_len": round(tot_g, 1)}


def postprocess_lines(
    payload: dict | str | Any,
    raw,
    *,
    thr: float | None = None,
    mode: str = "fire",
    **kw,
) -> dict[str, Any]:
    """Decoupled caller-side FIRE / PLUMBING post-processing from a raw model response (``Results``, JSON dict/str, or ``.npz``)."""
    from pathlib import Path

    from .geometry import page_segments
    from .results import Results
    from .sources import load_payload

    pay = load_payload(payload) if isinstance(payload, (str, Path)) else payload
    if isinstance(raw, Results):
        res = raw
    elif isinstance(raw, (str, Path)) and not str(raw).lstrip().startswith("{"):
        res = Results.from_npz(raw)
    else:
        res = Results.from_dict(raw)

    shapes = pay.get("shapes") or []
    segs = page_segments(shapes)
    eff_thr = float(res.threshold if thr is None else thr)
    if res.cls_prob is not None:
        if "attrs" not in kw and len(shapes) == len(res.prob):
            kw = {**kw, "attrs": _visual_style_from_shapes(shapes)}
        return decode_fire_network(
            segs, res.prob, res.cls_prob, segs.shape_length(),
            thr=eff_thr, class_names=res.class_names, mode=mode, **kw,
        )
    return decode_lines(segs, res.prob, thr=eff_thr, mode=mode, **kw)
