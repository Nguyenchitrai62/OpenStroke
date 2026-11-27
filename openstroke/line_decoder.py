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
    members: list[list[int]],
    *,
    angle_deg: float = ANGLE_DEG,
    perp_tol: float = PERP_TOL,
    gap: float = FIRE_GAP,
    node_tol: float = NODE_TOL,
) -> tuple[np.ndarray, list[list[int]]]:
    """Junction-aware collinear merging & parallel duplicate collapse (stops bridging across non-collinear branch nodes)."""
    if len(seg) <= 1:
        return seg, members
    u, L = _seg_dir_len(seg)
    cos_min = math.cos(math.radians(angle_deg))

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

    # Quantize orientations into buckets so collinear lines group cleanly
    ang = np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 180.0
    order = np.argsort(-L)   # longest segments anchor the line direction first
    used = np.zeros(n, dtype=bool)

    # Spatial index on segment midpoints for fast candidate lookup
    mids = 0.5 * (seg[:, 0:2] + seg[:, 2:4])
    mid_tree = cKDTree(mids)
    max_half = float(0.5 * L.max())

    out_s = []
    out_m: list[list[int]] = []

    for seed in order:
        if used[seed]:
            continue
        ui = u[seed]
        ni = np.array([-ui[1], ui[0]])
        origin = seg[seed, 0:2]
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

        cur_a, cur_b, w_off, w_len = ivals[0][0], ivals[0][1], ivals[0][2] * max(ivals[0][1] - ivals[0][0], 1e-6), max(ivals[0][1] - ivals[0][0], 1e-6)
        cur_mem = list(members[ivals[0][3]])
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
                cur_mem.extend(members[idx])
            else:
                mean_off = w_off / max(w_len, 1e-6)
                pa = origin + ui * cur_a + ni * mean_off
                pb = origin + ui * cur_b + ni * mean_off
                out_s.append([pa[0], pa[1], pb[0], pb[1]])
                out_m.append(sorted(set(cur_mem)))
                cur_a, cur_b, w_off, w_len = a, b, off * seglen, seglen
                cur_mem = list(members[idx])
        mean_off = w_off / max(w_len, 1e-6)
        pa = origin + ui * cur_a + ni * mean_off
        pb = origin + ui * cur_b + ni * mean_off
        out_s.append([pa[0], pa[1], pb[0], pb[1]])
        out_m.append(sorted(set(cur_mem)))

    return np.asarray(out_s, np.float64), out_m


def _snap_and_graph(
    seg: np.ndarray,
    members: list[list[int]],
    *,
    snap_tol: float = NODE_TOL,
    min_stub: float = MIN_STUB,
    perp_tol: float = PERP_TOL,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Radius-bounded endpoint snapping (no transitive chaining across parallel pipes and never collapsing a short
    dash's own two endpoints), centerline averaging of double-wall filled pipes, micro-stub pruning, and topology fittings."""
    if not len(seg):
        return [], []
    n = len(seg)
    pts = np.vstack([seg[:, 0:2], seg[:, 2:4]])
    tree = cKDTree(pts)
    pairs = sorted(
        (float(np.hypot(*(pts[a] - pts[b]))), a, b)
        for a, b in tree.query_pairs(snap_tol)
        if (a % n) != (b % n)
    )
    parent = list(range(2 * n))
    c_pts: dict[int, list[int]] = {k: [k] for k in range(2 * n)}

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
        if len({k % n for k in comb}) < len(comb):   # never collapse a segment's own two endpoints
            continue
        xy = pts[comb]
        ctr = xy.mean(axis=0)
        if float(np.hypot(*(xy - ctr).T).max()) > snap_tol:   # radius-bounded: blocks transitive chaining
            continue
        if len(c_pts[ra]) < len(c_pts[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        c_pts[ra].extend(c_pts.pop(rb))

    node_id = np.empty(2 * n, np.int64)
    node_xy = np.empty((len(c_pts), 2), np.float64)
    for idx, ks in enumerate(c_pts.values()):
        node_xy[idx] = pts[ks].mean(axis=0)
        for k in ks:
            node_id[k] = idx

    # Deduplicate overlapping edges sharing (u, v) (averaging double-wall filled pipe walls to centerline),
    # while preserving genuinely distinct parallel pipes separated by > max(perp_tol, snap_tol).
    dedup_tol = max(perp_tol, snap_tol)
    buckets: dict[tuple[int, int], list[list[Any]]] = {}
    for i in range(n):
        u, v = int(node_id[i]), int(node_id[n + i])
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        pa = node_xy[u] if float(np.hypot(*(seg[i, 0:2] - node_xy[u]))) <= 0.5 else seg[i, 0:2]
        pb = node_xy[v] if float(np.hypot(*(seg[i, 2:4] - node_xy[v]))) <= 0.5 else seg[i, 2:4]
        geom = np.array([pa[0], pa[1], pb[0], pb[1]], np.float64)
        lst = buckets.setdefault(key, [])
        mi = 0.5 * (pa + pb)
        merged = False
        for entry in lst:
            og = entry[1]
            om = 0.5 * (og[:2] + og[2:])
            if float(np.hypot(*(mi - om))) <= dedup_tol:
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
            lst.append([set(members[i]), geom])

    # Compute node degrees and drop tiny dangling stubs (< min_stub with at least one leaf endpoint)
    deg = np.zeros(len(node_xy), np.int64)
    for (u, v), lst in buckets.items():
        deg[u] += len(lst)
        deg[v] += len(lst)

    kept_edges: list[tuple[int, int, list[int], float, np.ndarray]] = []
    for (u, v), lst in buckets.items():
        for mems, geom in lst:
            L = float(np.hypot(geom[2] - geom[0], geom[3] - geom[1]))
            if L < min_stub and (deg[u] <= 1 or deg[v] <= 1):
                continue
            kept_edges.append((u, v, sorted(mems), L, geom))

    # Recompute incident edges per node for exact fitting classification
    adj: dict[int, list[int]] = {}
    for u, v, _, _, _ in kept_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    lines = [
        {
            "shape_name": "Line",
            "vertices": [
                [round(float(geom[0]), 3), round(float(geom[1]), 3)],
                [round(float(geom[2]), 3), round(float(geom[3]), 3)],
            ],
            "length": round(L, 3),
            "members": mems,
        }
        for _, _, mems, L, geom in kept_edges
    ]

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

    # Pass 2: junction-aware collinear dash/fragment bridging
    members = [[int(o)] for o in own]
    seg, members = _merge_collinear(seg, members, angle_deg=angle_deg, perp_tol=perp_tol, gap=gap_tol, node_tol=snap_tol)

    # Pass 3: re-split any merged segment touched mid-span by a bridged branch endpoint
    if len(seg) > 1:
        idx_arr = np.arange(len(seg), dtype=np.int64)
        seg, orig_idx = _split_at_t_junctions(seg, idx_arr, tol=snap_tol)
        members = [members[int(k)] for k in orig_idx]

    # Pass 4: snap endpoints, drop dangling micro-ticks, derive Elbow / Tee / Cross
    lines, fittings = _snap_and_graph(seg, members, snap_tol=snap_tol, min_stub=min_stub, perp_tol=perp_tol)
    return {"lines": lines, "fittings": fittings}


def _recompute_fittings(
    lines: list[dict[str, Any]],
    *,
    snap_tol: float = NODE_TOL,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build connected components and topology-derived fittings (Elbow / Tee / Cross) from decoded lines."""
    n = len(lines)
    if n == 0:
        return [], []
    pts = np.empty((2 * n, 2), np.float64)
    for i, ln in enumerate(lines):
        pts[i] = ln["vertices"][0]
        pts[n + i] = ln["vertices"][1]
    tree = cKDTree(pts)
    parent = list(range(2 * n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in tree.query_pairs(snap_tol + 0.05):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Canonical node coordinates & incident edges
    node_pts: dict[int, list[int]] = {}
    for k in range(2 * n):
        node_pts.setdefault(find(k), []).append(k)
    roots = list(node_pts.keys())
    root_to_id = {r: idx for idx, r in enumerate(roots)}
    node_xy = np.empty((len(roots), 2), np.float64)
    for idx, r in enumerate(roots):
        node_xy[idx] = pts[node_pts[r]].mean(axis=0)

    edge_uv: list[tuple[int, int]] = []
    adj: dict[int, list[tuple[int, int]]] = {}  # node -> [(nbr_node, edge_idx)]
    for i in range(n):
        u = root_to_id[find(i)]
        v = root_to_id[find(n + i)]
        edge_uv.append((u, v))
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
        pt = [round(float(node_xy[u, 0]), 3), round(float(node_xy[u, 1]), 3)]
        inc_classes = [str(lines[ei].get("pipe_class") or "pipe") for _, ei in inc]
        p_cls = "main" if "main" in inc_classes else (inc_classes[0] if inc_classes else "pipe")
        if d == 2:
            v1 = node_xy[inc[0][0]] - node_xy[u]
            v2 = node_xy[inc[1][0]] - node_xy[u]
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
) -> dict[str, Any]:
    """Class-aware FIRE pipe network post-processor (0 PDF layers, torch-free, ~0.15s/page CPU).

    1. Decodes picked strokes (``prob >= thr``) into canonical 2-vertex pipe lines via ``decode_lines``.
    2. Assigns each decoded line its pipe class (e.g. ``main`` / ``branch``) by length-weighted member probability.
    3. Drops zero-leaf (``deg1 == 0``) closed non-branch perimeter frames (site boundaries / title boxes; verified 0/93
       false drops across all ``fire_3cls_v3`` GT pages and +2.5–2.6 pt Main F2 on E153 Val13).
    4. Recomputes ``Elbow`` / ``Tee`` / ``Cross`` fittings on kept lines and maps kept lines back to per-shape ``cls`` / ``inst``.
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
        names = [str(c) for c in class_names] if class_names and len(class_names) >= 2 else ["other", "pipe"]

    cls_out = np.zeros(n_shapes, dtype=np.int8)
    inst_out = np.full(n_shapes, -1, dtype=np.int32)

    raw = decode_lines(
        segs, prob_arr, thr=thr, mode=mode, gap=gap,
        angle_deg=angle_deg, perp_tol=perp_tol, snap_tol=snap_tol, min_stub=min_stub,
    )
    lines = raw["lines"]
    if not lines:
        return {
            "lines": [], "fittings": [], "classes": names,
            "cls": cls_out, "inst": inst_out,
            "counts": {}, "lengths": {}, "dropped_loops": 0,
        }

    for ln in lines:
        mems = ln["members"]
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

    comps, fittings = _recompute_fittings(lines, snap_tol=snap_tol)
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
            _, fittings = _recompute_fittings(lines, snap_tol=snap_tol)

    best_len = np.zeros(n_shapes, dtype=np.float64)
    counts: dict[str, int] = {}
    lengths: dict[str, float] = {}
    for idx, ln in enumerate(lines):
        cname = str(ln["pipe_class"])
        cid = int(ln["class_id"])
        L = float(ln["length"])
        counts[cname] = counts.get(cname, 0) + 1
        lengths[cname] = lengths.get(cname, 0.0) + L
        for m in ln["members"]:
            if L > best_len[m]:
                best_len[m] = L
                inst_out[m] = idx
                cls_out[m] = cid
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

    segs = page_segments(pay.get("shapes") or [])
    eff_thr = float(res.threshold if thr is None else thr)
    if res.cls_prob is not None:
        return decode_fire_network(
            segs, res.prob, res.cls_prob, segs.shape_length(),
            thr=eff_thr, class_names=res.class_names, mode=mode, **kw,
        )
    return decode_lines(segs, res.prob, thr=eff_thr, mode=mode, **kw)
