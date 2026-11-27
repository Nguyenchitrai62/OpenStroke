"""Non-planar Endpoint-Topology cycle extractor for HVAC object post-processing (v10, F254).

Core geometric principle (user insight): build closed duct / fitting object polygons directly from
stroke-endpoint connectivity rather than planar ``shapely.polygonize`` / bitmap ink-buffer fallbacks,
so ducts and elbows crossing at different elevations (X-crossings) or sharing collinear seams stay
100% intact instead of shattering into strips or Pac-Man / shark-fin notches.

Pipeline (pure NumPy + SciPy + Shapely, torch-free, zero PDF layers):
  1. ``elementary_chains``: split polylines ONLY at sharp corners (> 22 deg) or straight-to-curve tangent
     joins, keeping every smooth Bezier/circular arc as ONE single curved edge between its true endpoints.
  2. Shared Step-0 endpoint snapping (``ep_snap = 0.25 pt`` cross-edge only) + facing collinear dash-gap
     bridging (``<= 3.5 pt``, ``perp <= 0.15 pt``) + free-end snapping (``free_snap = 1.0 pt``).
  3. Two-tier non-planar cycle extraction (ZERO X-crossing vertices!):
     - Tier-1: pure endpoint-to-endpoint graph (zero interior splits).
     - Tier-2: split chain bodies ONLY at genuine terminating ``deg == 1`` free endpoints or strictly
       one-sided non-passthrough corners (never pass-through X-crossings or collinear double-duct seams).
     - Degree-2 collinear chain contraction + exact 1/2/3/4-cycle enumerator + O(E) non-planar half-edge
       rotation-system minimal-cycle tracer (using exact local arc tangents).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon


def elementary_chains(
    sseg: np.ndarray,
    own: np.ndarray,
    curved_shape: np.ndarray | None = None,
    corner_deg: float = 22.0,
) -> list[tuple[np.ndarray, int, bool]]:
    """Decompose picked segments ``(M, 4)`` + ``owner (M,)`` into elementary wall/arc chains
    ``[(pts (K, 2), owner_id, is_curved), ...]``.

    Smooth curved arcs stay ONE curved edge between their two endpoints; polylines split only at
    sharp corners (> ``corner_deg``) or straight-to-curve tangent joins (>= 2.5 pt straight wall).
    """
    if not len(sseg):
        return []
    slen = np.hypot(sseg[:, 2] - sseg[:, 0], sseg[:, 3] - sseg[:, 1])
    keep = slen > 1e-6
    sseg = sseg[keep]
    own = own[keep]
    slen = slen[keep]
    if not len(sseg):
        return []

    cos_corner = float(np.cos(np.radians(corner_deg)))
    brk = np.r_[True, (own[1:] != own[:-1]) | (np.abs(sseg[1:, :2] - sseg[:-1, 2:]).max(1) > 1e-4)]
    starts = np.flatnonzero(brk)
    ends = np.r_[starts[1:], len(sseg)]

    dx = sseg[:, 2] - sseg[:, 0]
    dy = sseg[:, 3] - sseg[:, 1]
    ux = dx / slen
    uy = dy / slen

    chains: list[tuple[np.ndarray, int, bool]] = []
    for a, b in zip(starts.tolist(), ends.tolist()):
        o_id = int(own[a])
        is_c = bool(curved_shape[o_id]) if (curved_shape is not None and 0 <= o_id < len(curved_shape)) else False
        m = b - a
        if m == 1:
            pts = np.array([[sseg[a, 0], sseg[a, 1]], [sseg[a, 2], sseg[a, 3]]], dtype=np.float64)
            chains.append((pts, o_id, is_c))
            continue

        u_sub = np.column_stack([ux[a:b], uy[a:b]])
        l_sub = slen[a:b]
        dots = np.sum(u_sub[:-1] * u_sub[1:], axis=1)
        split_mask = dots < cos_corner
        if m >= 3:
            is_arc_seg = np.zeros(m, dtype=bool)
            turn_ok = (dots < 0.9996) & (dots >= cos_corner)
            for k in np.flatnonzero(turn_ok).tolist():
                if l_sub[k] <= 6.0 and l_sub[k + 1] <= 6.0:
                    is_arc_seg[k] = True
                    is_arc_seg[k + 1] = True
            for k in range(m - 1):
                if (not is_arc_seg[k]) and is_arc_seg[k + 1] and l_sub[k] >= 2.5:
                    split_mask[k] = True
                elif is_arc_seg[k] and (not is_arc_seg[k + 1]) and l_sub[k + 1] >= 2.5:
                    split_mask[k] = True

        sp_idx = np.flatnonzero(split_mask) + 1
        sub_starts = np.r_[0, sp_idx]
        sub_ends = np.r_[sp_idx, m]
        for sa, sb in zip(sub_starts.tolist(), sub_ends.tolist()):
            ga, gb = a + sa, a + sb
            pts = np.empty((gb - ga + 1, 2), dtype=np.float64)
            pts[:-1] = sseg[ga:gb, :2]
            pts[-1] = sseg[gb - 1, 2:]
            if len(pts) >= 3:
                arc_l = float(slen[ga:gb].sum())
                chord = float(np.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1]))
                sub_cur = (arc_l > 1.004 * max(chord, 1e-6)) or (is_c and len(pts) >= 4)
            else:
                sub_cur = False
            if (not sub_cur) and len(pts) > 2:
                v0 = pts[-1] - pts[0]
                vn = float(np.hypot(v0[0], v0[1]))
                if vn > 1e-6:
                    nx_, ny_ = -v0[1] / vn, v0[0] / vn
                    dev = np.max(np.abs((pts[1:-1, 0] - pts[0, 0]) * nx_ + (pts[1:-1, 1] - pts[0, 1]) * ny_))
                    if dev < 0.05:
                        pts = np.array([pts[0], pts[-1]], dtype=np.float64)
            chains.append((pts, o_id, sub_cur))
    return chains


def dedup_chains(chains: list[tuple[np.ndarray, int, bool]]) -> list[tuple[np.ndarray, int, bool]]:
    """Remove exact/near-identical duplicate straight segments (double-drawn CAD walls) while keeping arcs."""
    out = []
    seen = set()
    for pts, o_id, is_c in chains:
        if (not is_c) and len(pts) == 2:
            a = (round(float(pts[0, 0]) * 10), round(float(pts[0, 1]) * 10))
            b = (round(float(pts[1, 0]) * 10), round(float(pts[1, 1]) * 10))
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
        out.append((pts, o_id, is_c))
    return out


def _chain_tangent(pts: np.ndarray, at_start: bool) -> np.ndarray:
    """Unit outgoing tangent vector leaving ``pts[0]`` (if ``at_start``) or ``pts[-1]``."""
    if at_start:
        for k in range(1, len(pts)):
            v = pts[k] - pts[0]
            n = float(np.hypot(v[0], v[1]))
            if n > 1e-6:
                return v / n
    else:
        for k in range(len(pts) - 2, -1, -1):
            v = pts[k] - pts[-1]
            n = float(np.hypot(v[0], v[1]))
            if n > 1e-6:
                return v / n
    return np.array([1.0, 0.0], dtype=np.float64)


def _assemble_cycle_poly(
    edge_ids: list[int],
    edge_dirs: list[bool],
    edges_pts: list[np.ndarray],
    min_area: float = 1.0,
):
    parts = []
    for eid, fwd in zip(edge_ids, edge_dirs):
        p = edges_pts[eid] if fwd else edges_pts[eid][::-1]
        parts.append(p[:-1])
    pts = np.vstack(parts)
    if len(pts) < 3:
        return None
    x, y = pts[:, 0], pts[:, 1]
    area2 = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if abs(area2) <= 2.0 * min_area:
        return None
    g = Polygon(pts)
    if not g.is_valid or g.area <= min_area:
        return None
    return g


def contract_deg2_chains(
    nodes: np.ndarray,
    edges_uv: list[tuple[int, int]],
    edges_pts: list[np.ndarray],
    edges_cur: list[bool],
    max_turn_deg: float = 15.0,
) -> tuple[list[tuple[int, int]], list[np.ndarray], list[bool]]:
    """Contract smooth/collinear degree-2 junctions in O(E) (collapses dashed duct walls into single edges)."""
    E = len(edges_uv)
    if E < 2:
        return edges_uv, edges_pts, edges_cur
    n_nodes = len(nodes)
    inc: list[list[int]] = [[] for _ in range(n_nodes)]
    alive = [True] * E
    uv = list(edges_uv)
    pts = list(edges_pts)
    cur = list(edges_cur)

    for ei, (u, v) in enumerate(uv):
        if u != v:
            inc[u].append(ei)
            inc[v].append(ei)

    cos_str = float(np.cos(np.radians(max_turn_deg)))
    cos_cur = float(np.cos(np.radians(25.0)))

    for m in range(n_nodes):
        es = [ei for ei in inc[m] if alive[ei]]
        if len(es) != 2:
            continue
        e1, e2 = es[0], es[1]
        if e1 == e2:
            continue
        u1, v1 = uv[e1]
        u2, v2 = uv[e2]
        if v1 == m:
            a = u1
            p1 = pts[e1]
            d1_in = -_chain_tangent(p1, False)
        else:
            a = v1
            p1 = pts[e1][::-1]
            d1_in = -_chain_tangent(p1, False)
        if u2 == m:
            b = v2
            p2 = pts[e2]
            d2_out = _chain_tangent(p2, True)
        else:
            b = u2
            p2 = pts[e2][::-1]
            d2_out = _chain_tangent(p2, True)
        if a == b:
            continue
        dot = float(d1_in @ d2_out)
        thr = cos_cur if (cur[e1] or cur[e2]) else cos_str
        if dot < thr:
            continue
        merged_pts = np.vstack([p1[:-1], p2])
        is_c = bool(cur[e1] or cur[e2])
        if not is_c and len(merged_pts) > 2:
            dv = merged_pts[-1] - merged_pts[0]
            dn = float(np.hypot(dv[0], dv[1]))
            if dn > 1e-6:
                nx_, ny_ = -dv[1] / dn, dv[0] / dn
                if np.max(np.abs((merged_pts[1:-1, 0] - merged_pts[0, 0]) * nx_ + (merged_pts[1:-1, 1] - merged_pts[0, 1]) * ny_)) < 0.08:
                    merged_pts = np.array([merged_pts[0], merged_pts[-1]], dtype=np.float64)
        uv[e1] = (a, b)
        pts[e1] = merged_pts
        cur[e1] = is_c
        alive[e2] = False
        inc[b] = [e1 if x == e2 else x for x in inc[b]]
        inc[m] = []

    return (
        [uv[i] for i in range(E) if alive[i]],
        [pts[i] for i in range(E) if alive[i]],
        [cur[i] for i in range(E) if alive[i]],
    )


def extract_nonplanar_cycles(
    nodes: np.ndarray,
    edges_uv: list[tuple[int, int]],
    edges_pts: list[np.ndarray],
    edges_cur: list[bool],
    *,
    min_area: float = 1.0,
    max_halfedge_steps: int = 512,
    enum_4cycles: bool = True,
) -> list:
    """Extract valid elementary object polygons from a possibly non-planar endpoint graph:
    1-cycles + 2-cycles + exact 3/4-cycles + local-tangent half-edge rotation-system cycles."""
    if not edges_uv:
        return []
    n_nodes = len(nodes)
    E_raw = len(edges_uv)

    out_polys = []
    seen_edge_sets: set[frozenset[int]] = set()
    active_e = []
    for ei, (u, v) in enumerate(edges_uv):
        if u == v:
            pts = edges_pts[ei]
            if len(pts) >= 4:
                g = Polygon(pts[:-1])
                if g.is_valid and g.area > min_area:
                    out_polys.append(g)
                    seen_edge_sets.add(frozenset((ei,)))
        else:
            active_e.append(ei)

    if len(active_e) < 2:
        return out_polys

    deg = np.zeros(n_nodes, dtype=np.int64)
    adj_e: list[list[int]] = [[] for _ in range(n_nodes)]
    alive = np.zeros(E_raw, dtype=bool)
    for ei in active_e:
        u, v = edges_uv[ei]
        alive[ei] = True
        deg[u] += 1
        deg[v] += 1
        adj_e[u].append(ei)
        adj_e[v].append(ei)
    stack = [i for i in range(n_nodes) if deg[i] == 1]
    while stack:
        nd = stack.pop()
        if deg[nd] != 1:
            continue
        for ei in adj_e[nd]:
            if alive[ei]:
                alive[ei] = False
                u, v = edges_uv[ei]
                oth = v if u == nd else u
                deg[nd] -= 1
                deg[oth] -= 1
                if deg[oth] == 1:
                    stack.append(oth)
                break

    keep_e = np.flatnonzero(alive).tolist()
    if len(keep_e) < 2:
        return out_polys

    pair_edges: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ei in keep_e:
        u, v = edges_uv[ei]
        pair_edges[(u, v) if u < v else (v, u)].append(ei)

    # 2-cycles (parallel edges between same (u, v), e.g., two arcs or arc + straight edge)
    for (u, v), elist in pair_edges.items():
        if len(elist) >= 2:
            cap = min(len(elist), 4)
            for ia in range(cap):
                for ib in range(ia + 1, cap):
                    e1, e2 = elist[ia], elist[ib]
                    fset = frozenset((e1, e2))
                    if fset in seen_edge_sets:
                        continue
                    seen_edge_sets.add(fset)
                    g = _assemble_cycle_poly(
                        [e1, e2],
                        [edges_uv[e1][0] == u, edges_uv[e2][0] == v],
                        edges_pts,
                        min_area,
                    )
                    if g is not None:
                        out_polys.append(g)

    # Keep at most 1 straight edge + curved edges per (u, v) pair for 3/4-cycle and half-edge tracing
    he_keep = []
    for elist in pair_edges.values():
        if len(elist) == 1:
            he_keep.append(elist[0])
        else:
            has_str = False
            for ei in sorted(elist, key=lambda q: len(edges_pts[q])):
                if edges_cur[ei]:
                    he_keep.append(ei)
                elif not has_str:
                    he_keep.append(ei)
                    has_str = True

    pair_e1: dict[tuple[int, int], list[int]] = defaultdict(list)
    nbr_clean: list[list[tuple[int, int]]] = [[] for _ in range(n_nodes)]
    for ei in he_keep:
        u, v = edges_uv[ei]
        pair_e1[(u, v) if u < v else (v, u)].append(ei)
        nbr_clean[u].append((v, ei))
        nbr_clean[v].append((u, ei))

    # Exact 3-cycle & 4-cycle enumeration (immune to crossing overlaps and shared corners)
    if enum_4cycles:
        for ei in he_keep:
            u, v = edges_uv[ei]
            if len(nbr_clean[u]) > 12 or len(nbr_clean[v]) > 12:
                continue
            nu, nv = nbr_clean[u], nbr_clean[v]
            for w, e2 in nv:
                if w == u or e2 == ei:
                    continue
                k3 = (u, w) if u < w else (w, u)
                for e3 in pair_e1.get(k3, ()):
                    if e3 == ei or e3 == e2:
                        continue
                    fset = frozenset((ei, e2, e3))
                    if fset in seen_edge_sets:
                        continue
                    seen_edge_sets.add(fset)
                    g = _assemble_cycle_poly(
                        [ei, e2, e3],
                        [True, edges_uv[e2][0] == v, edges_uv[e3][0] == w],
                        edges_pts,
                        min_area,
                    )
                    if g is not None:
                        out_polys.append(g)
            for w, e2 in nv:
                if w == u or e2 == ei:
                    continue
                for x, e4 in nu:
                    if x == v or x == w or e4 == ei or e4 == e2:
                        continue
                    k3 = (w, x) if w < x else (x, w)
                    e3_list = pair_e1.get(k3)
                    if not e3_list:
                        continue
                    for e3 in e3_list:
                        if e3 in (ei, e2, e4):
                            continue
                        fset = frozenset((ei, e2, e3, e4))
                        if fset in seen_edge_sets:
                            continue
                        seen_edge_sets.add(fset)
                        g = _assemble_cycle_poly(
                            [ei, e2, e3, e4],
                            [True, edges_uv[e2][0] == v, edges_uv[e3][0] == w, edges_uv[e4][0] == x],
                            edges_pts,
                            min_area,
                        )
                        if g is not None:
                            out_polys.append(g)

    # Non-planar Half-Edge Rotation-System minimal-cycle tracer (5..max_halfedge_steps cycles)
    M = len(he_keep)
    if M >= 3:
        he_u = np.empty(2 * M, dtype=np.int64)
        he_v = np.empty(2 * M, dtype=np.int64)
        he_eid = np.empty(2 * M, dtype=np.int64)
        he_fwd = np.empty(2 * M, dtype=bool)
        he_ang = np.empty(2 * M, dtype=np.float64)
        for loc_i, ei in enumerate(he_keep):
            u, v = edges_uv[ei]
            pts = edges_pts[ei]
            t_s = _chain_tangent(pts, True)
            t_e = _chain_tangent(pts, False)
            he_u[2 * loc_i] = u
            he_v[2 * loc_i] = v
            he_eid[2 * loc_i] = ei
            he_fwd[2 * loc_i] = True
            he_ang[2 * loc_i] = np.arctan2(t_s[1], t_s[0])
            he_u[2 * loc_i + 1] = v
            he_v[2 * loc_i + 1] = u
            he_eid[2 * loc_i + 1] = ei
            he_fwd[2 * loc_i + 1] = False
            he_ang[2 * loc_i + 1] = np.arctan2(t_e[1], t_e[0])

        order = np.lexsort((he_ang, he_u))
        u_s = he_u[order]
        chg = np.flatnonzero(u_s[1:] != u_s[:-1]) + 1
        starts = np.r_[0, chg]
        ends = np.r_[chg, len(order)]

        next_he = np.empty(2 * M, dtype=np.int64)
        for a, b in zip(starts.tolist(), ends.tolist()):
            grp = order[a:b]
            k = len(grp)
            if k == 1:
                next_he[grp[0] ^ 1] = grp[0]
            else:
                for j in range(k):
                    next_he[grp[j] ^ 1] = grp[j - 1]

        visited = np.zeros(2 * M, dtype=bool)
        next_l = next_he.tolist()
        eid_l = he_eid.tolist()
        fwd_l = he_fwd.tolist()
        for h0 in range(2 * M):
            if visited[h0]:
                continue
            c_eids = []
            c_fwds = []
            h = h0
            ok = True
            while not visited[h]:
                visited[h] = True
                c_eids.append(eid_l[h])
                c_fwds.append(fwd_l[h])
                h = next_l[h]
                if len(c_eids) > max_halfedge_steps:
                    ok = False
                    break
            if not ok or h != h0 or len(c_eids) < 3:
                continue
            fset = frozenset(c_eids)
            if fset in seen_edge_sets:
                continue
            parts = [edges_pts[ei][:-1] if fw else edges_pts[ei][::-1][:-1] for ei, fw in zip(c_eids, c_fwds)]
            pts = np.vstack(parts)
            if len(pts) < 3:
                continue
            x, y = pts[:, 0], pts[:, 1]
            area2 = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
            if area2 <= 2.0 * min_area:
                continue
            g = Polygon(pts)
            if g.is_valid and g.area > min_area:
                seen_edge_sets.add(fset)
                out_polys.append(g)

    return out_polys


def _bridge_collinear_free_ends(
    nodes: np.ndarray,
    free_roots: list[int],
    root_edge_dir: dict[int, list[tuple[int, np.ndarray]]],
    max_gap: float = 3.5,
    max_perp: float = 0.15,
) -> list[tuple[np.ndarray, int, bool]]:
    """Bridge facing collinear ``deg == 1`` dash endpoints along the same straight wall BEFORE ``free_snap``
    can jump them sideways onto a nearby parallel line."""
    if len(free_roots) < 2:
        return []
    f_arr = np.asarray(free_roots, dtype=np.int64)
    f_pts = nodes[f_arr]
    f_cont = np.zeros((len(free_roots), 2), dtype=np.float64)
    f_edge = np.full(len(free_roots), -1, dtype=np.int64)
    for k, rf in enumerate(free_roots):
        ed = root_edge_dir.get(rf)
        if ed:
            f_edge[k] = ed[0][0]
            f_cont[k] = -ed[0][1]

    ftree = cKDTree(f_pts)
    pairs = []
    for i, j in ftree.query_pairs(max_gap):
        if f_edge[i] < 0 or f_edge[j] < 0 or f_edge[i] == f_edge[j]:
            continue
        if float(f_cont[i] @ f_cont[j]) > -0.97:
            continue
        dv = f_pts[j] - f_pts[i]
        dist = float(np.hypot(dv[0], dv[1]))
        if dist < 0.10:
            continue
        if float(dv @ f_cont[i]) <= 0.0 or float((-dv) @ f_cont[j]) <= 0.0:
            continue
        perp_i = abs(float(-f_cont[i, 1] * dv[0] + f_cont[i, 0] * dv[1]))
        perp_j = abs(float(-f_cont[j, 1] * (-dv[0]) + f_cont[j, 0] * (-dv[1])))
        if max(perp_i, perp_j) <= max_perp:
            pairs.append((dist, i, j))
    if not pairs:
        return []
    pairs.sort(key=lambda q: q[0])
    used: set[int] = set()
    bridges: list[tuple[np.ndarray, int, bool]] = []
    for _, i, j in pairs:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        bridges.append((np.vstack([f_pts[i], f_pts[j]]), -1, False))
    return bridges


def extract_t1_and_t2_cycles(
    chains: list[tuple[np.ndarray, int, bool]],
    *,
    ep_snap: float = 0.25,
    free_snap: float = 1.0,
    min_area: float = 1.0,
    max_halfedge_steps: int = 512,
) -> tuple[list, list]:
    """Compute shared Step-0 snapping & collinear dash bridging ONCE, then extract BOTH:
      - Tier-1: pure endpoint-to-endpoint non-planar cycles (zero interior splits, zero X-crossings);
      - Tier-2: terminating-endpoint T-junction non-planar cycles (strictly non-passthrough splits only).
    """
    chains = dedup_chains(chains)
    if len(chains) < 2:
        return [], []

    def _build_base(ch_list):
        E0 = len(ch_list)
        ep0 = np.empty((2 * E0, 2), dtype=np.float64)
        for i, (pts, _, _) in enumerate(ch_list):
            ep0[2 * i] = pts[0]
            ep0[2 * i + 1] = pts[-1]
        nd0, inv0 = np.unique(np.round(ep0, 5), axis=0, return_inverse=True)
        inv0 = inv0.ravel()
        n0 = len(nd0)
        par0 = list(range(n0))

        def rt0(i: int) -> int:
            while par0[i] != i:
                par0[i] = par0[par0[i]]
                i = par0[i]
            return i

        ne0: list[set[int]] = [set() for _ in range(n0)]
        for i in range(E0):
            ne0[int(inv0[2 * i])].add(i)
            ne0[int(inv0[2 * i + 1])].add(i)
        nt0 = cKDTree(nd0)
        if ep_snap > 0:
            pairs = sorted(
                ((float(np.hypot(*(nd0[a] - nd0[b]))), int(a), int(b)) for a, b in nt0.query_pairs(ep_snap)),
                key=lambda q: q[0],
            )
            for _, a, b in pairs:
                if not ne0[a].isdisjoint(ne0[b]):
                    continue
                ra, rb = rt0(a), rt0(b)
                if ra != rb:
                    par0[ra] = rb
        deg0 = np.zeros(n0, dtype=np.int64)
        red0: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for i in range(E0):
            u, v = rt0(int(inv0[2 * i])), rt0(int(inv0[2 * i + 1]))
            if u != v:
                deg0[u] += 1
                deg0[v] += 1
                red0[u].append((i, _chain_tangent(ch_list[i][0], True)))
                red0[v].append((i, _chain_tangent(ch_list[i][0], False)))
        return nd0, inv0, par0, rt0, ne0, nt0, deg0, red0

    nodes, inv, parent, root, node_edges, ntree, deg_r, root_edge_dir = _build_base(chains)
    free_roots_0 = [r for r in range(len(nodes)) if parent[r] == r and deg_r[r] == 1]
    bridges = _bridge_collinear_free_ends(nodes, free_roots_0, root_edge_dir, max_gap=3.5, max_perp=0.15)
    if bridges:
        chains = list(chains) + bridges
        nodes, inv, parent, root, node_edges, ntree, deg_r, root_edge_dir = _build_base(chains)

    E = len(chains)
    n_nodes = len(nodes)

    free_roots = [r for r in range(n_nodes) if parent[r] == r and deg_r[r] == 1]
    if free_roots and free_snap > ep_snap:
        f_arr = np.asarray(free_roots, dtype=np.int64)
        near_all = ntree.query_ball_point(nodes[f_arr], free_snap)
        for rf, near_list in zip(free_roots, near_all):
            best_j, best_key = -1, (2, 1e9)
            for j in near_list:
                rj = root(int(j))
                if rj == rf:
                    continue
                if not node_edges[rf].isdisjoint(node_edges[int(j)]):
                    continue
                d = float(np.hypot(*(nodes[int(j)] - nodes[rf])))
                key = (1 if deg_r[rj] == 1 else 0, d)
                if key < best_key:
                    best_key = key
                    best_j = rj
            if best_j >= 0:
                ra, rb = root(rf), root(best_j)
                if ra != rb:
                    parent[ra] = rb
                    deg_r[rb] += deg_r[ra]
                    deg_r[ra] = 0

    # Tier-1: pure endpoint-to-endpoint non-planar cycles
    base_uv = [(root(int(inv[2 * i])), root(int(inv[2 * i + 1]))) for i in range(E)]
    base_pts = [c[0] for c in chains]
    base_cur = [c[2] for c in chains]
    t1_uv, t1_pts, t1_cur = contract_deg2_chains(nodes, base_uv, base_pts, base_cur)
    t1_faces = extract_nonplanar_cycles(
        nodes, t1_uv, t1_pts, t1_cur, min_area=min_area, max_halfedge_steps=max_halfedge_steps
    )

    # Tier-2: terminating-endpoint T-junction splits (strictly non-passthrough!)
    clines = np.asarray([LineString(c[0]) for c in chains], dtype=object)
    clens = shapely.length(clines)
    ctree = shapely.STRtree(clines)

    deg_r = np.zeros(n_nodes, dtype=np.int64)
    root_edge_dir = defaultdict(list)
    for i, (u, v) in enumerate(base_uv):
        if u != v:
            deg_r[u] += 1
            deg_r[v] += 1
            root_edge_dir[u].append((i, _chain_tangent(base_pts[i], True)))
            root_edge_dir[v].append((i, _chain_tangent(base_pts[i], False)))

    splits: dict[int, list[tuple[float, int]]] = defaultdict(list)
    extra_nodes: list[np.ndarray] = []

    def _local_dir(pts: np.ndarray, s_dist: float) -> np.ndarray:
        if len(pts) == 2:
            dv = pts[1] - pts[0]
            return dv / max(float(np.hypot(dv[0], dv[1])), 1e-9)
        acc = 0.0
        for k in range(len(pts) - 1):
            dv = pts[k + 1] - pts[k]
            dl = float(np.hypot(dv[0], dv[1]))
            if acc + dl >= s_dist - 1e-6 or k == len(pts) - 2:
                return dv / max(dl, 1e-9)
            acc += dl
        return np.array([1.0, 0.0])

    # 2a. Free endpoints (deg == 1) -> T-junction on non-parallel chain body (or exact collinear <= 0.12 pt)
    free_roots = [r for r in range(n_nodes) if parent[r] == r and deg_r[r] == 1]
    if free_roots:
        f_arr = np.asarray(free_roots, dtype=np.int64)
        f_pts = shapely.points(nodes[f_arr])
        pi, ci = ctree.query(f_pts, predicate="dwithin", distance=free_snap)
        best_hit: dict[int, tuple[float, int, float, np.ndarray]] = {}
        for p_loc, c_i in zip(pi.tolist(), ci.tolist()):
            rf = free_roots[p_loc]
            if not root_edge_dir[rf]:
                continue
            own_e, dir_f = root_edge_dir[rf][0]
            if c_i == own_e or base_uv[c_i][0] == rf or base_uv[c_i][1] == rf:
                continue
            s_val = float(clines[c_i].project(f_pts[p_loc]))
            if s_val <= 0.12 or s_val >= clens[c_i] - 0.12:
                continue
            q_pt = clines[c_i].interpolate(s_val)
            q_xy = np.array([q_pt.x, q_pt.y], dtype=np.float64)
            dv = float(np.hypot(*(q_xy - nodes[rf])))
            if dv > free_snap:
                continue
            t_dir = _local_dir(base_pts[c_i], s_val)
            if abs(float(dir_f @ t_dir)) >= 0.85 and dv > 0.12:
                continue
            prev = best_hit.get(p_loc)
            if prev is None or dv < prev[0]:
                best_hit[p_loc] = (dv, c_i, s_val, q_xy)
        for p_loc, (_, c_i, s_val, q_xy) in best_hit.items():
            rf = free_roots[p_loc]
            n_new = n_nodes + len(extra_nodes)
            extra_nodes.append(q_xy)
            parent.append(n_new)
            parent[root(rf)] = n_new
            splits[c_i].append((s_val, n_new))

    # 2b. Non-free corners (deg >= 2) touching chain body within 0.15 pt (strictly one-sided, never pass-through!)
    nf_roots = [r for r in range(n_nodes) if parent[r] == r and deg_r[r] >= 2]
    if nf_roots:
        nf_arr = np.asarray(nf_roots, dtype=np.int64)
        nf_pts = shapely.points(nodes[nf_arr])
        pi, ci = ctree.query(nf_pts, predicate="dwithin", distance=0.15)
        for p_loc, c_i in zip(pi.tolist(), ci.tolist()):
            rf = nf_roots[p_loc]
            if base_uv[c_i][0] == rf or base_uv[c_i][1] == rf:
                continue
            s_val = float(clines[c_i].project(nf_pts[p_loc]))
            if s_val <= 0.12 or s_val >= clens[c_i] - 0.12:
                continue
            t_dir = _local_dir(base_pts[c_i], s_val)
            nx_, ny_ = -t_dir[1], t_dir[0]
            min_side, max_side = 1e9, -1e9
            for _, d_inc in root_edge_dir[rf]:
                sd = float(nx_ * d_inc[0] + ny_ * d_inc[1])
                if sd < min_side:
                    min_side = sd
                if sd > max_side:
                    max_side = sd
            if min_side < -0.20 and max_side > 0.20:
                continue
            if max(abs(min_side), abs(max_side)) < 0.25:
                q_pt = clines[c_i].interpolate(s_val)
                if float(np.hypot(q_pt.x - nodes[rf, 0], q_pt.y - nodes[rf, 1])) > 0.05:
                    continue
            splits[c_i].append((s_val, rf))

    if not splits:
        return t1_faces, []

    allnodes = np.vstack([nodes, np.asarray(extra_nodes)]) if extra_nodes else nodes

    def _slice_chain(pts: np.ndarray, s0_v: float, s1_v: float) -> np.ndarray:
        if len(pts) == 2:
            dv = pts[1] - pts[0]
            dl = max(float(np.hypot(dv[0], dv[1])), 1e-9)
            return np.vstack([pts[0] + (s0_v / dl) * dv, pts[0] + (s1_v / dl) * dv])
        seg_d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        cum = np.r_[0.0, np.cumsum(seg_d)]

        def _interp(sv: float) -> np.ndarray:
            if sv <= 0.0:
                return pts[0]
            if sv >= cum[-1]:
                return pts[-1]
            k = max(0, min(int(np.searchsorted(cum, sv, side="right")) - 1, len(seg_d) - 1))
            t = (sv - cum[k]) / max(seg_d[k], 1e-9)
            return pts[k] + t * (pts[k + 1] - pts[k])

        p_s, p_e = _interp(s0_v), _interp(s1_v)
        mid_m = (cum > s0_v + 1e-4) & (cum < s1_v - 1e-4)
        return np.vstack([p_s, pts[mid_m], p_e]) if mid_m.any() else np.vstack([p_s, p_e])

    new_uv, new_pts, new_cur = [], [], []
    for i, (pts, _, is_c) in enumerate(chains):
        u0, v0 = root(int(inv[2 * i])), root(int(inv[2 * i + 1]))
        cuts = splits.get(i)
        if not cuts:
            new_uv.append((u0, v0))
            new_pts.append(pts)
            new_cur.append(is_c)
        else:
            cuts.sort(key=lambda q: q[0])
            prev_s, prev_n, tot_l = 0.0, u0, float(clens[i])
            for sv, nid in cuts:
                rn = root(nid)
                if sv - prev_s > 0.05 and prev_n != rn:
                    sub = _slice_chain(pts, prev_s, sv)
                    new_uv.append((prev_n, rn))
                    new_pts.append(sub)
                    new_cur.append(is_c and len(sub) >= 3)
                    prev_s, prev_n = sv, rn
            if tot_l - prev_s > 0.05 and prev_n != v0:
                sub = _slice_chain(pts, prev_s, tot_l)
                new_uv.append((prev_n, v0))
                new_pts.append(sub)
                new_cur.append(is_c and len(sub) >= 3)

    new_uv, new_pts, new_cur = contract_deg2_chains(allnodes, new_uv, new_pts, new_cur)
    t2_faces = extract_nonplanar_cycles(
        allnodes, new_uv, new_pts, new_cur, min_area=min_area, max_halfedge_steps=max_halfedge_steps
    )
    return t1_faces, t2_faces
