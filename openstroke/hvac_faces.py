"""OpenStroke28 post-processing of its own (F148): HVAC objects from the stroke model's picks, without AI_Takeoff's logic and without layers.

The picked strokes are thickened by GAP and unioned; every closed hole of that ink is a face (a region the strokes enclose; an arc and a joint
line that end ~1 pt apart still close). Each face grows back by GAP to the strokes' centre lines and is described by its shape (size, how
rectangular, corners, share of its outline on curved strokes, ...). A classifier fitted on the GT train pages (faces of the GT strokes matched
to the GT objects) names it Duct / Elbow / Elbow vuông / Concentric Reducer / Eccentric Reducer / Chân rẽ or drops it.

    faces = page_faces(segs, picked)                      # list of shapely Polygons
    X = face_features(faces, segs, picked, curved)        # (n, F) float
"""
from __future__ import annotations

import numpy as np

GAP = 0.6
MIN_AREA = 1.0
FEATURES = (
    # Original 16 single-face geometric descriptors (keep exact order for back-compat)
    "log_area", "log_long", "log_short", "aspect", "rect_fill", "hull_fill", "corners", "arc_share", "log_perimeter",
    "compact", "n_strokes", "log_ink_inside", "touch_faces", "side_ratio", "skew_deg", "right_corners",
    # F225: 14 pure-geometry neighbourhood & T-junction topology descriptors (separates Chân rẽ vs Reducer vs Duct vs none)
    "nb_shared_cnt", "shared_frac", "nb_collinear_cnt", "nb_ortho_cnt", "nb_t_side_cnt", "nb_t_side_max_ratio",
    "nb_end_match_cnt", "nb_width_ratio_diff", "nb_max_area_ratio", "nb_min_area_ratio", "max_shared_short_ratio",
    "cusp_cnt", "par_gap_deg", "stroke_cov_frac",
)
MITRE = 1.5   # growing a face back: corners stay sharp up to 1.5 x GAP (a rectangle needs 1.41); tangent arc joints no longer grow
# 3 pt spikes (default limit 5), at most 0.9 pt (F152). A round join would cut every rectangle corner into two vertices.
RIGHT_TOL = 3.0   # degrees: a corner this close to 90 counts as right (a 10-degree reducer flank turns 95 / 85: not right)


def _chains(seg: np.ndarray, owner: np.ndarray):
    """Consecutive segments of one shape that meet end to start become one linestring (a bezier sampled into many segments is one
    curve): thickened segment by segment, every joint of a bend left a wedge notch that a face filled and then grew into a tooth (F152)."""
    import shapely

    if not len(seg):
        return np.zeros(0, dtype=object)
    brk = np.r_[True, (owner[1:] != owner[:-1]) | (np.abs(seg[1:, :2] - seg[:-1, 2:]).max(1) > 1e-6)]
    start = np.flatnonzero(brk)
    end = np.r_[start[1:], len(seg)]
    n = end - start + 1                                    # points per chain
    chain = np.repeat(np.arange(len(start)), n)
    pts = np.empty((n.sum(), 2))
    first = np.r_[0, np.cumsum(n)[:-1]]
    k = np.arange(len(seg)) - np.repeat(start, end - start) + np.repeat(first, end - start)   # position of each segment's start point
    pts[k] = seg[:, :2]
    pts[first + n - 1] = seg[end - 1, 2:]
    return shapely.linestrings(pts, indices=chain)


TOUCH = 0.05   # pt: a vertex this close to another segment's body touches it (any vertex, not only free ends)
SNAP = 1.0   # pt: stroke ends this close are one node; an end this close to another stroke's body joins it there (a T)
BRIDGE_MAX = 15.0   # pt: open ends facing each other on one line this close are joined straight (F158)
CAP_MAX, CAP_PAR, CAP_PERP = 60.0, 0.9, 0.25   # pt; cos of the two end directions (parallel); |cos| connector vs direction (square)
END_MIN = 1.0   # pt: a chain this short still has open ends (F166: merged dashes leave 2.5 pt pieces at corners; was 3)
CORNER_MAX, CORNER_COS = 4.0, 0.3   # pt along each ray to the meeting point; |cos| of the two end directions (square enough)


def _merged_chains(chains, curved_chain: np.ndarray):
    """The picked ink as drawn, whatever item drew it: overlapping pieces dissolved (shapely.unary_union) and contiguous pieces joined
    (line_merge). F166: on a dashed duct, a second item drawing the same dashes again (whole or in part) made every dash end look joined to
    its copy, so the gaps were never bridged and the duct never closed. -> (chains, curved per chain: >= half its length on a curve)."""
    import shapely

    arr = np.asarray(list(chains), dtype=object)
    if len(arr) < 2:
        return arr, np.asarray(curved_chain, bool)
    merged = shapely.get_parts(shapely.line_merge(shapely.unary_union(arr)))
    mlen = shapely.length(merged)
    keep_m = mlen > 0
    merged = merged[keep_m]
    mlen = mlen[keep_m]
    cur = arr[np.asarray(curved_chain, bool)]
    if not len(cur) or not len(merged):
        return merged, np.zeros(len(merged), bool)
    # Index buffered curves with STRtree instead of intersecting every merged chain against one giant MultiPolygon
    cur_bufs = shapely.buffer(cur, 0.05)
    ctree = shapely.STRtree(cur_bufs)
    mi, ci = ctree.query(merged, predicate="intersects")
    on = np.zeros(len(merged), dtype=np.float64)
    if len(mi):
        order = np.argsort(mi, kind="quicksort")
        mi_s, ci_s = mi[order], ci[order]
        chg = np.flatnonzero(mi_s[1:] != mi_s[:-1]) + 1
        starts = np.r_[0, chg]
        ends = np.r_[chg, len(mi_s)]
        u_m = mi_s[starts]
        cnts = ends - starts
        single = cnts == 1
        if single.any():
            sm = u_m[single]
            sc = ci_s[starts[single]]
            on[sm] = shapely.length(shapely.intersection(merged[sm], cur_bufs[sc]))
        for idx in np.flatnonzero(~single):
            m_i = u_m[idx]
            local_band = shapely.union_all(cur_bufs[ci_s[starts[idx]:ends[idx]]])
            on[m_i] = float(shapely.length(shapely.intersection(merged[m_i], local_band)))
    return merged, on >= 0.5 * mlen


def _caps(chains, curved_chain: np.ndarray):
    """Virtual end lines (F153): an elbow the model picked often has its two arcs but not the short lines closing its ends, so no face
    closes. Two chain ends, at least one of them on a curve, that point the same way (cos >= CAP_PAR), lie <= CAP_MAX pt apart and face
    each other square to that direction (|cos| <= CAP_PERP) are joined; each end at most once, nearest first."""
    import shapely
    from scipy.spatial import cKDTree

    chains, curved_chain = _merged_chains(chains, curved_chain)
    if not len(chains):
        return []
    clens = shapely.length(chains)
    pts, dirs, owner = [], [], []
    for i in np.flatnonzero(clens >= END_MIN):
        xy = np.asarray(chains[i].coords)
        if len(xy) < 2:
            continue
        for a, b in ((xy[0], xy[1]), (xy[-1], xy[-2])):
            d = a - b
            n = float(np.hypot(d[0], d[1]))
            if n > 0:
                pts.append(a); dirs.append(d / n); owner.append(int(i))
    if len(pts) < 2:
        return []
    pts, dirs, owner = np.asarray(pts), np.asarray(dirs), np.asarray(owner)
    # only ends touching no other stroke are open (F157: a tee's fillet arcs already meet the duct lines; capping them split the tee)
    tree = shapely.STRtree(chains)
    pi, li = tree.query(shapely.points(pts), predicate="dwithin", distance=SNAP)
    touched = np.zeros(len(pts), bool)
    touched[pi[li != owner[pi]]] = True
    open_idx = np.flatnonzero(~touched)
    if len(open_idx) < 2:
        return []
    # Build one cKDTree on open endpoints only (all 3 loops skip touched endpoints)
    otree = cKDTree(pts[open_idx])
    cand = []
    for ia, ib in otree.query_pairs(CAP_MAX):
        a, b = int(open_idx[ia]), int(open_idx[ib])
        if owner[a] == owner[b] or not (curved_chain[owner[a]] or curved_chain[owner[b]]):
            continue                                       # both ends open (F157; one open end only: more false caps, F159)
        v = pts[b] - pts[a]
        dist = float(np.hypot(v[0], v[1]))
        if dist < 1.0 or float(dirs[a] @ dirs[b]) < CAP_PAR:
            continue
        u = v / dist
        if abs(float(u @ dirs[a])) > CAP_PERP or abs(float(u @ dirs[b])) > CAP_PERP:
            continue
        cand.append((dist, a, b))
    # bridges first (F158): two open ends facing each other on one line (a short stroke between them the model scored low) join straight
    bridges = []
    for ia, ib in otree.query_pairs(BRIDGE_MAX):
        a, b = int(open_idx[ia]), int(open_idx[ib])
        if owner[a] == owner[b]:
            continue
        v = pts[b] - pts[a]
        dist = float(np.hypot(v[0], v[1]))
        if dist < 1e-6:
            continue
        u = v / dist
        if float(dirs[a] @ dirs[b]) <= -CAP_PAR and float(u @ dirs[a]) >= CAP_PAR and -float(u @ dirs[b]) >= CAP_PAR:
            bridges.append((dist, a, b))
    # corners (F166): two open ends roughly square to each other whose forward rays meet within CORNER_MAX of both (a dashed outline whose
    # dashes stop short of the corner) join through that point
    corners = []
    if CORNER_MAX > 0:
        for ia, ib in otree.query_pairs(2 * CORNER_MAX):
            a, b = int(open_idx[ia]), int(open_idx[ib])
            if owner[a] == owner[b] or abs(float(dirs[a] @ dirs[b])) > CORNER_COS:
                continue
            M = np.stack([dirs[a], -dirs[b]], 1)
            st = np.linalg.solve(M, pts[b] - pts[a])          # pts[a] + s dirs[a] = pts[b] + t dirs[b]
            if (st >= -1e-6).all() and (st <= CORNER_MAX).all():
                corners.append((float(st.max()), a, b, pts[a] + st[0] * dirs[a]))
    used, out = set(), []
    for dist, a, b in sorted(bridges) + sorted(cand):
        if a in used or b in used:
            continue
        used |= {a, b}
        out.append(shapely.LineString([pts[a], pts[b]]))
    for dist, a, b, x in sorted(corners, key=lambda q: q[0]):
        if a in used or b in used:
            continue
        used |= {a, b}
        out += [shapely.LineString([pts[a], x]), shapely.LineString([pts[b], x])]
    return out


def page_faces(segs, picked: np.ndarray, gap: float = GAP, curved: np.ndarray | None = None) -> list:
    """Closed regions enclosed by the picked strokes (holes of the thickened ink), grown back to the strokes' centre lines. With
    ``curved`` (per shape), open elbow ends are first closed by virtual end lines (_caps)."""
    import shapely
    from shapely.geometry import Polygon

    m = picked[segs.owner]
    if not m.any():
        return []
    own = segs.owner[m]
    sseg = segs.seg[m]
    chains = _chains(sseg, own)
    lines = list(chains)
    if curved is not None:
        brk = np.r_[True, (own[1:] != own[:-1]) | (np.abs(sseg[1:, :2] - sseg[:-1, 2:]).max(1) > 1e-6)]
        lines += _caps(chains, curved[own[np.flatnonzero(brk)]])
    ink = shapely.union_all(shapely.buffer(np.asarray(lines, dtype=object), gap, cap_style="flat", join_style="mitre", mitre_limit=MITRE))
    holes = [Polygon(r) for g in shapely.get_parts(ink) for r in g.interiors]
    out = []
    for h in holes:
        if h.area <= MIN_AREA:
            continue
        g = h.buffer(gap, join_style="mitre", mitre_limit=MITRE)
        if g.geom_type != "Polygon":   # an invalid ring can come back in pieces: keep the largest
            g = max(shapely.get_parts(g), key=lambda q: q.area)
        out.append(g)
    return out


RING_TOL = 0.5     # pt: a shape whose segments run end to start and come back to the first point draws one closed outline
SAME_IOU = 0.8     # a closed shape and a face this alike are one candidate
INSIDE = 0.8       # a face with this share of its area inside a closed shape is a piece of it (cut by a crossing stroke)


def closed_shapes(segs, picked: np.ndarray) -> list:
    """Outlines of the picked shapes that draw one closed ring on their own (user 2026-11-16: one PDF item with many commands draws one
    object; F153: ~87 % of the picked / GT stroke length on NOCOLOR and the GT pages is in such shapes)."""
    from shapely.geometry import Polygon

    import shapely
    from shapely.geometry import Polygon

    off = segs.shape_offsets
    out = []
    for s in np.flatnonzero(picked):
        a, b = off[s], off[s + 1]
        if b - a < 3:
            continue
        seg = segs.seg[a:b]
        if np.abs(seg[1:, :2] - seg[:-1, 2:]).max() > RING_TOL or np.hypot(*(seg[0, :2] - seg[-1, 2:])) > RING_TOL:
            continue
        g = Polygon(seg[:, :2]).buffer(0)
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda q: q.area)
        if g.area > MIN_AREA:
            out.append(g)
    return out


def page_candidates(segs, picked: np.ndarray, curved: np.ndarray | None = None, graph: bool = True) -> list:
    """Object candidates: the faces of the picked strokes' graph (graph_faces) and the closed picked shapes (closed_shapes). A shape that is the same
    region as a face is not repeated; faces lying inside a closed shape (pieces of it cut by a crossing duct, a label...) give way to it."""
    import shapely

    if not graph:   # F152 faces (ink buffer), the measured default until the graph faces stop over-producing reducers (F155)
        return page_faces(segs, picked)
    faces = graph_faces(segs, picked, curved=curved)   # F155: joined only end-to-end or at a T, never where strokes cross
    shapes = closed_shapes(segs, picked)
    if not shapes:
        return faces
    f_arr = np.asarray(faces, dtype=object) if faces else None
    f_areas = shapely.area(f_arr) if faces else np.zeros(0)
    ftree = shapely.STRtree(f_arr) if faces else None
    s_areas = [float(g.area) for g in shapes]
    keep_s, keep_sa, drop_f = [], [], set()
    for si, g in enumerate(shapes):
        sa = s_areas[si]
        near = ftree.query(g, predicate="intersects") if ftree is not None else []
        if len(near):
            ia = shapely.area(shapely.intersection(f_arr[near], g))
            fa = f_areas[near]
            # Inclusion-exclusion: area(A | B) = area(A) + area(B) - area(A & B) (saves a full GEOS union!)
            if np.any(ia / np.maximum(fa + sa - ia, 1e-9) >= SAME_IOU):
                continue
            drop_f.update(int(j) for j, a_int, a_f in zip(near, ia, fa) if a_int >= INSIDE * a_f)
        keep_s.append(g)
        keep_sa.append(sa)
    # one shape drawn twice (or nested copies): keep one (STRtree replaces O(N^2) scan)
    uniq = []
    if keep_s:
        ks_arr = np.asarray(keep_s, dtype=object)
        ks_tree = shapely.STRtree(ks_arr)
        ai, bi = ks_tree.query(ks_arr, predicate="intersects")
        prev_map: dict[int, list[int]] = {}
        for a, b in zip(ai.tolist(), bi.tolist()):
            if b < a:
                prev_map.setdefault(a, []).append(b)
        accepted = np.zeros(len(keep_s), dtype=bool)
        for i, g in enumerate(keep_s):
            prev = [j for j in prev_map.get(i, ()) if accepted[j]]
            if prev:
                ia = shapely.area(shapely.intersection(ks_arr[prev], g))
                ua = np.maximum(np.asarray([keep_sa[j] for j in prev]) + keep_sa[i] - ia, 1e-9)
                if np.any(ia / ua >= SAME_IOU):
                    continue
            accepted[i] = True
            uniq.append(g)
    return [f for j, f in enumerate(faces) if j not in drop_f] + uniq


def graph_faces(segs, picked: np.ndarray, curved: np.ndarray | None = None, snap: float = SNAP) -> list:
    """Smallest closed regions of the picked strokes' graph across items (F155, user 2026-11-16): joined only end to end or at a T (free
    ends, F159), never where strokes cross; open elbow ends and collinear gaps are closed first (_caps)."""
    m = picked[segs.owner]
    if not m.any():
        return []
    own = segs.owner[m]
    sseg = segs.seg[m]
    allseg = [sseg]
    if curved is not None:
        chains = _chains(sseg, own)
        brk = np.r_[True, (own[1:] != own[:-1]) | (np.abs(sseg[1:, :2] - sseg[:-1, 2:]).max(1) > 1e-6)]
        caps = _caps(chains, curved[own[np.flatnonzero(brk)]])
        if caps:
            allseg.append(np.asarray([[*c.coords[0], *c.coords[-1]] for c in caps], float))
    return _polygonize_segments(np.vstack(allseg), snap)


def _polygonize_segments(seg: np.ndarray, snap: float = SNAP) -> list:
    """Smallest closed regions of straight segments (F155 / F159). Vertices shared by segments are nodes exactly where they are; only a
    free end (used by one segment) snaps to the nearest node within snap, or else to the body of a segment within snap (a T, splitting it
    there). Segments that cross are not joined; an edge drawn twice counts once."""
    import shapely
    from scipy.spatial import cKDTree

    seg = np.asarray(seg, float)
    slen = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    keep = slen > 1e-6
    seg = seg[keep]
    slen = slen[keep]
    if len(seg) < 3:
        return []
    s0 = seg[:, :2]
    sv_norm = (seg[:, 2:] - s0) / slen[:, None]
    P = seg.reshape(-1, 2)
    nodes, inv = np.unique(np.round(P, 6), axis=0, return_inverse=True)
    inv = inv.ravel()
    n_nodes = len(nodes)
    deg = np.bincount(inv, minlength=n_nodes)
    parent = list(range(n_nodes))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lines = shapely.linestrings(seg.reshape(-1, 2, 2))
    ltree = shapely.STRtree(lines)
    splits: dict[int, list[tuple[float, int]]] = {}           # segment -> [(t along it, node)]
    extra: list[np.ndarray] = []

    free = np.flatnonzero(deg == 1)
    if len(free):
        ntree = cKDTree(nodes)
        near_all = ntree.query_ball_point(nodes[free], snap)
        no_node_free = []
        for idx_f, near_list in zip(free.tolist(), near_all):
            near = [j for j in near_list if j != idx_f]
            if near:
                nf = nodes[idx_f]
                j = min(near, key=lambda jj: (deg[jj] == 1, float(np.hypot(nodes[jj, 0] - nf[0], nodes[jj, 1] - nf[1]))))
                ra, rb = root(idx_f), root(j)
                if ra != rb:
                    parent[ra] = rb
            else:
                no_node_free.append(idx_f)
        if no_node_free:
            nf_arr = np.asarray(no_node_free, dtype=np.int64)
            f_pi, f_ki = ltree.query(shapely.points(nodes[nf_arr]), predicate="dwithin", distance=snap)
            if len(f_pi):
                f_nodes = nf_arr[f_pi]
                ok_hit = (inv[2 * f_ki] != f_nodes) & (inv[2 * f_ki + 1] != f_nodes)
                if ok_hit.any():
                    f_pi, f_ki, f_nodes = f_pi[ok_hit], f_ki[ok_hit], f_nodes[ok_hit]
                    px = nodes[f_nodes, 0]
                    py = nodes[f_nodes, 1]
                    t_proj = np.clip((px - s0[f_ki, 0]) * sv_norm[f_ki, 0] + (py - s0[f_ki, 1]) * sv_norm[f_ki, 1], 0.0, slen[f_ki])
                    qx = s0[f_ki, 0] + t_proj * sv_norm[f_ki, 0]
                    qy = s0[f_ki, 1] + t_proj * sv_norm[f_ki, 1]
                    d_hit = np.hypot(px - qx, py - qy)
                    # Best (minimum distance) segment per free end (preserve exact tie-break by first hit)
                    best_map: dict[int, tuple[float, int, float, float, float]] = {}
                    for fi_loc, k_s, d_v, t_v, qx_v, qy_v in zip(f_pi.tolist(), f_ki.tolist(), d_hit.tolist(), t_proj.tolist(), qx.tolist(), qy.tolist()):
                        prev = best_map.get(fi_loc)
                        if prev is None or d_v < prev[0]:
                            best_map[fi_loc] = (d_v, k_s, t_v, qx_v, qy_v)
                    for fi_loc in range(len(no_node_free)):
                        hit = best_map.get(fi_loc)
                        if hit is None:
                            continue
                        _, k_s, t_v, qx_v, qy_v = hit
                        f_id = no_node_free[fi_loc]
                        n_new = n_nodes + len(extra)
                        extra.append(np.array([qx_v, qy_v], dtype=np.float64))
                        splits.setdefault(k_s, []).append((t_v, n_new))
                        parent.append(n_new)
                        parent[root(f_id)] = n_new

    # any vertex lying on another segment's body (touching, <= TOUCH) splits it there: the V of a symbol meeting its base (F159)
    non_free = np.flatnonzero(deg != 1)
    if len(non_free):
        pi_nf, li = ltree.query(shapely.points(nodes[non_free]), predicate="dwithin", distance=TOUCH)
        if len(pi_nf):
            fi = non_free[pi_nf]
            ok_t = (inv[2 * li] != fi) & (inv[2 * li + 1] != fi)
            if ok_t.any():
                fi, li = fi[ok_t], li[ok_t]
                t_val = ((nodes[fi, 0] - s0[li, 0]) * sv_norm[li, 0] + (nodes[fi, 1] - s0[li, 1]) * sv_norm[li, 1])
                interior = (t_val > 1e-6) & (t_val < slen[li] - 1e-6)
                for f_id, k_s, tv in zip(fi[interior].tolist(), li[interior].tolist(), t_val[interior].tolist()):
                    splits.setdefault(k_s, []).append((tv, f_id))

    allnodes = np.vstack([nodes, np.asarray(extra)]) if extra else nodes
    edge_uv: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    seen_add = seen.add
    uv_append = edge_uv.append

    inv_l = inv.tolist()
    for k in range(len(seg)):
        a, b = inv_l[2 * k], inv_l[2 * k + 1]
        cuts = splits.get(k)
        if not cuts:
            u, v = root(a), root(b)
            if u != v:
                key = (u, v) if u < v else (v, u)
                if key not in seen:
                    seen_add(key)
                    uv_append((u, v))
        else:
            cuts.sort()
            prev = a
            for _, n_id in cuts:
                u, v = root(prev), root(n_id)
                if u != v:
                    key = (u, v) if u < v else (v, u)
                    if key not in seen:
                        seen_add(key)
                        uv_append((u, v))
                prev = n_id
            u, v = root(prev), root(b)
            if u != v:
                key = (u, v) if u < v else (v, u)
                if key not in seen:
                    seen_add(key)
                    uv_append((u, v))
    if not edge_uv:
        return []
    edges = shapely.linestrings(allnodes[np.asarray(edge_uv, dtype=np.int64)])
    out = []
    for g in shapely.get_parts(shapely.polygonize(edges)):
        g = g.buffer(0)
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda q: q.area)
        if g.area > MIN_AREA:
            out.append(g)
    return out


def _corners(g) -> int:
    import shapely

    return len(np.asarray(shapely.simplify(g, 0.5).exterior.coords)) - 1


def item_faces(segs, picked: np.ndarray) -> list:
    """-> [(region, item index, inner regions merged)]. Priority 1 (user 2026-11-16): the smallest closed regions each picked PDF item draws with its own strokes. An item drawing a duct
    piece and its reducer in one run gives two regions."""
    import shapely
    from shapely.geometry import Polygon

    off = segs.shape_offsets
    out = []
    for s in np.flatnonzero(picked):
        a, b = off[s], off[s + 1]
        if b - a < 3:
            continue
        faces = _polygonize_segments(segs.seg[a:b])
        if len(faces) >= 2 and any(_corners(g) <= 3 for g in faces):
            # inner lines (the V of a square-to-round symbol, a cross) do not make separate objects: the item's outline is one (F158)
            u = shapely.union_all([g.buffer(0.05) for g in faces]).buffer(-0.05)
            if u.geom_type == "MultiPolygon":
                u = max(u.geoms, key=lambda q: q.area)
            out.append((Polygon(u.exterior), int(s), len(faces)))
        else:
            out += [(g, int(s), 1) for g in faces]
    return out


SLIVER_ASPECT = 6.0   # long side / height: a gap between two objects is long and flat; a symbol's triangle is not
SLIVER_W = 2.0   # pt of mean thickness (area / long side): a 3-corner region this thin (a fitting's edge not quite on the duct's end line) is a gap, not an object


def merge_slivers(regions: list, return_index: bool = False):
    """Thin triangles join a neighbour (F156): the one with the most corners (a fitting: tee / elbow / reducer outline) among those
    sharing >= SHARED pt of outline, so the fitting reaches the duct's end line."""
    import shapely

    if len(regions) < 2:
        return (regions, list(range(len(regions)))) if return_index else regions
    out = list(regions)
    arr = np.asarray(out, dtype=object)
    simp_geoms = shapely.simplify(arr, 0.5)
    simp = [len(np.asarray(g.exterior.coords)) - 1 for g in simp_geoms]
    mrrs = shapely.minimum_rotated_rectangle(arr)
    areas = shapely.area(arr)
    thin = []
    for i in range(len(out)):
        if simp[i] > 3:
            continue
        rr = np.asarray(mrrs[i].exterior.coords)[:4]
        e = np.hypot(*(np.roll(rr, -1, 0) - rr).T) if len(rr) == 4 else np.zeros(4)
        long_ = max(e[0], e[1], 1e-9)
        if areas[i] / long_ < SLIVER_W and long_ * long_ / (2 * areas[i]) >= SLIVER_ASPECT:   # thin and long: a gap
            thin.append(i)
    if not thin:
        return (out, list(range(len(out)))) if return_index else out
    nb = neighbours(out)
    gone = set()
    for i in thin:
        cand = [j for j in nb[i] if j not in gone and j not in thin]
        if not cand:
            continue
        j = max(cand, key=lambda j: (simp[j], out[j].area))
        u = out[j].union(out[i].buffer(0.05)).buffer(-0.05)
        if u.geom_type == "Polygon":
            out[j] = u
            gone.add(i)
    keep = [i for i in range(len(out)) if i not in gone]
    return ([out[i] for i in keep], keep) if return_index else [out[i] for i in keep]


def priority_candidates(segs, picked: np.ndarray, curved: np.ndarray | None = None) -> tuple[list, list]:
    """Candidates in the user's order: item regions first, then the graph regions across items (graph_faces) that are not the same as an
    item region nor mostly inside one; thin gaps join a fitting (merge_slivers). -> (regions, source item per region (-1 = across items), inner regions merged per region)."""
    import shapely

    items = sorted(item_faces(segs, picked), key=lambda q: q[0].area)
    uniq, uniq_areas, src, inner = [], [], [], []
    if items:
        it_arr = np.asarray([q[0] for q in items], dtype=object)
        it_areas = shapely.area(it_arr)
        # Single STRtree over all item faces (replaces rebuilding STRtree on every iteration: O(N log N) vs O(N^2 log N))
        itree = shapely.STRtree(it_arr)
        ai, bi = itree.query(it_arr, predicate="intersects")
        prev_map: dict[int, list[int]] = {}
        for a, b in zip(ai.tolist(), bi.tolist()):
            if b < a:
                prev_map.setdefault(a, []).append(b)
        accepted = np.zeros(len(items), dtype=bool)
        for i, (g, it, n_in) in enumerate(items):
            prev = [j for j in prev_map.get(i, ()) if accepted[j]]
            if prev:
                ia = shapely.area(shapely.intersection(it_arr[prev], g))
                ua = np.maximum(it_areas[prev] + it_areas[i] - ia, 1e-9)
                if np.any(ia / ua >= SAME_IOU):
                    continue
            accepted[i] = True
            uniq.append(g); uniq_areas.append(float(it_areas[i])); src.append(it); inner.append(n_in)
    gfaces = graph_faces(segs, picked, curved=curved)
    if uniq and gfaces:
        u_arr = np.asarray(uniq, dtype=object)
        u_areas = np.asarray(uniq_areas, dtype=np.float64)
        gf_arr = np.asarray(gfaces, dtype=object)
        gf_areas = shapely.area(gf_arr)
        tree = shapely.STRtree(u_arr)
        fi, ui = tree.query(gf_arr, predicate="intersects")
        near_map: dict[int, list[int]] = {}
        if len(fi):
            ia_all = shapely.area(shapely.intersection(gf_arr[fi], u_arr[ui]))
            for f_idx, u_idx, ia_v in zip(fi.tolist(), ui.tolist(), ia_all.tolist()):
                near_map.setdefault(f_idx, []).append((u_idx, ia_v))
        for i, f in enumerate(gfaces):
            hits = near_map.get(i)
            if hits:
                fa = float(gf_areas[i])
                if any(ia_v / max(fa + u_areas[j] - ia_v, 1e-9) >= SAME_IOU for j, ia_v in hits):
                    continue
                if sum(ia_v for _, ia_v in hits) >= INSIDE * fa:
                    continue
            uniq.append(f); src.append(-1); inner.append(1)
    elif gfaces:
        for f in gfaces:
            uniq.append(f); src.append(-1); inner.append(1)
    merged, keep = merge_slivers(uniq, return_index=True)
    return merged, [src[i] for i in keep], [inner[i] for i in keep]


def _taper(v: np.ndarray) -> tuple[float, float, int]:
    """For a 4-corner outline: (short / long length of its most parallel pair of opposite sides, angle in degrees between the other pair,
    corners within RIGHT_TOL degrees of a right angle). A duct is (1, 0, 4), a concentric reducer (< 1, > 0, 0), an eccentric one (< 1, > 0, 2).
    Other outlines: (1, 0, number of right corners)."""
    n = len(v)
    if n < 3:
        return 1.0, 0.0, 0
    e = np.roll(v, -1, 0) - v
    ang = np.degrees(np.arctan2(e[:, 1], e[:, 0]))
    turn = np.abs(((np.roll(ang, -1) - ang + 180) % 360) - 180)
    right = int(np.sum(np.abs(turn - 90) <= RIGHT_TOL))
    if n != 4:
        return 1.0, 0.0, right
    L = np.hypot(e[:, 0], e[:, 1])

    def gap(a, b):  # angle between two sides as lines (0 = parallel)
        d = abs(((ang[a] - ang[b]) % 180))
        return min(d, 180 - d)

    pairs = [(0, 2), (1, 3)]
    par = min(pairs, key=lambda q: gap(*q))
    oth = pairs[1 - pairs.index(par)]
    return float(min(L[par[0]], L[par[1]]) / max(L[par[0]], L[par[1]], 1e-9)), float(gap(*oth)), right


def _mrr_axes(f) -> tuple[float, float, np.ndarray]:
    """(long_side, short_side, unit_long_axis) of face f's minimum rotated rectangle."""
    rr = np.asarray(f.minimum_rotated_rectangle.exterior.coords)[:4]
    if len(rr) < 4:
        return 0.0, 0.0, np.array([1.0, 0.0])
    v0 = rr[1] - rr[0]
    v1 = rr[2] - rr[1]
    l0 = float(np.hypot(v0[0], v0[1]))
    l1 = float(np.hypot(v1[0], v1[1]))
    if l0 >= l1:
        ax = v0 / max(l0, 1e-9)
        return l0, l1, ax
    ax = v1 / max(l1, 1e-9)
    return l1, l0, ax


def _par_gap_deg(v: np.ndarray) -> float:
    """Minimum line-angle difference (deg) between opposite sides of a 4-corner polygon (0 = parallel pair)."""
    if len(v) != 4:
        return 45.0
    e = np.roll(v, -1, 0) - v
    ang = np.degrees(np.arctan2(e[:, 1], e[:, 0]))
    def gap(a, b):
        d = abs((ang[a] - ang[b]) % 180)
        return min(d, 180 - d)
    return float(min(gap(0, 2), gap(1, 3)))


def _shared_dir(geom) -> np.ndarray | None:
    """Unit direction vector of the dominant line segment in a shared-boundary intersection geometry."""
    import shapely

    best_l, best_v = 0.0, None
    for part in shapely.get_parts(geom):
        if part.geom_type in ("LineString", "LinearRing") and part.length > best_l:
            xy = np.asarray(part.coords)
            if len(xy) >= 2:
                v = xy[-1] - xy[0]
                n = float(np.hypot(v[0], v[1]))
                if n > 1e-6:
                    best_l, best_v = float(part.length), v / n
    return best_v


def _mrr_axes_from_rect(mrr) -> tuple[float, float, np.ndarray]:
    rr = np.asarray(mrr.exterior.coords)[:4]
    if len(rr) < 4:
        return 0.0, 0.0, np.array([1.0, 0.0])
    v0 = rr[1] - rr[0]
    v1 = rr[2] - rr[1]
    l0 = float(np.hypot(v0[0], v0[1]))
    l1 = float(np.hypot(v1[0], v1[1]))
    if l0 >= l1:
        ax = v0 / max(l0, 1e-9)
        return l0, l1, ax
    ax = v1 / max(l1, 1e-9)
    return l1, l0, ax


def _cusps_from_simplified(g) -> int:
    v = np.asarray(g.exterior.coords)[:-1]
    if len(v) < 4:
        return 0
    if g.exterior.is_ccw is False:
        v = v[::-1]
    e = np.roll(v, -1, 0) - v
    ang = np.degrees(np.arctan2(e[:, 1], e[:, 0]))
    turn = ((ang - np.roll(ang, 1) + 180) % 360) - 180
    return int(np.sum(turn <= -CUSP_TURN))


def face_features(faces: list, segs, picked: np.ndarray, curved: np.ndarray) -> np.ndarray:
    """Shape + neighbourhood-topology descriptors of every face (FEATURES)."""
    import shapely

    n_faces = len(faces)
    if not n_faces:
        return np.zeros((0, len(FEATURES)))
    faces_arr = np.asarray(faces, dtype=object)
    m = picked[segs.owner]
    lines = shapely.linestrings(segs.seg[m].reshape(-1, 2, 2))
    own = segs.owner[m]
    is_curve = curved[own]
    ltree = shapely.STRtree(lines)
    ftree = shapely.STRtree(faces_arr)

    # Vectorized per-face geometry computations in Shapely C (quad_segs=16 matches Geom.buffer() default exactly)
    areas = shapely.area(faces_arr)
    mrrs = shapely.minimum_rotated_rectangle(faces_arr)
    mrr_info = [_mrr_axes_from_rect(r) for r in mrrs]
    hull_areas = shapely.area(shapely.convex_hull(faces_arr))
    bufs = shapely.buffer(faces_arr, 0.6, quad_segs=16)
    touch_bufs = shapely.buffer(faces_arr, GAP * 2, quad_segs=16)
    ext_rings = shapely.get_exterior_ring(faces_arr)
    rlens = np.maximum(shapely.length(ext_rings), 1e-9)
    bands = shapely.buffer(ext_rings, GAP + 0.3, quad_segs=16)
    simps = shapely.simplify(faces_arr, 0.5)

    # 1. Bulk outline stroke coverage, arc share, and unique stroke counts
    fi_r, li_r = ltree.query(ext_rings, predicate="dwithin", distance=GAP + 0.3)
    cov_sum = np.zeros(n_faces, dtype=np.float64)
    arc_num = np.zeros(n_faces, dtype=np.float64)
    n_strokes = np.zeros(n_faces, dtype=np.int64)
    if len(fi_r):
        cov_all = shapely.length(shapely.intersection(lines[li_r], bands[fi_r]))
        cov_sum = np.bincount(fi_r, weights=cov_all, minlength=n_faces)
        arc_num = np.bincount(fi_r, weights=cov_all * is_curve[li_r], minlength=n_faces)
        # Count distinct owning shapes per face (fi_r from STRtree.query is grouped by face; sort to be safe)
        ord_r = np.argsort(fi_r, kind="quicksort")
        fi_s, own_s = fi_r[ord_r], own[li_r[ord_r]]
        chg = np.flatnonzero(fi_s[1:] != fi_s[:-1]) + 1
        starts = np.r_[0, chg]
        ends = np.r_[chg, len(fi_s)]
        for f_i, a_i, b_i in zip(fi_s[starts].tolist(), starts.tolist(), ends.tolist()):
            n_strokes[f_i] = len(set(own_s[a_i:b_i].tolist()))

    # 2. Bulk inner ink length (faces with area > 4, computing negative buffer ONCE per big face)
    ink_in = np.zeros(n_faces, dtype=np.float64)
    big = np.flatnonzero(areas > 4.0)
    if len(big):
        inners = shapely.buffer(faces_arr[big], -GAP - 0.3, quad_segs=16)
        bi_in, li_in = ltree.query(inners, predicate="intersects")
        if len(bi_in):
            in_lens = shapely.length(shapely.intersection(lines[li_in], inners[bi_in]))
            ink_in[big] = np.bincount(bi_in, weights=in_lens, minlength=len(big))

    # 3. Bulk touching faces & shared boundary intersections
    ti, tj = ftree.query(touch_bufs, predicate="intersects")
    keep_t = ti != tj
    ti, tj = ti[keep_t], tj[keep_t]
    touch_cnt = np.bincount(ti, minlength=n_faces) if len(ti) else np.zeros(n_faces, dtype=np.int64)

    nb_sh_lens: list[list[float]] = [[] for _ in range(n_faces)]
    nb_areas_l: list[list[float]] = [[] for _ in range(n_faces)]
    collinear_a = np.zeros(n_faces, dtype=np.int64)
    ortho_a = np.zeros(n_faces, dtype=np.int64)
    t_side_a = np.zeros(n_faces, dtype=np.int64)
    t_side_max_a = np.zeros(n_faces, dtype=np.float64)
    end_match_a = np.zeros(n_faces, dtype=np.int64)

    if len(ti):
        inters = shapely.intersection(ext_rings[ti], bufs[tj])
        slens = shapely.length(inters)
        good_nb = slens >= 1.5
        if good_nb.any():
            for i_f, j_f, sl, inter in zip(ti[good_nb].tolist(), tj[good_nb].tolist(), slens[good_nb].tolist(), inters[good_nb]):
                nb_sh_lens[i_f].append(sl)
                nb_areas_l[i_f].append(float(areas[j_f]))
                ax_i = mrr_info[i_f][2]
                lj, sj, ax_j = mrr_info[j_f]
                dot_ax = abs(float(ax_i[0] * ax_j[0] + ax_i[1] * ax_j[1]))
                if dot_ax >= 0.85:
                    collinear_a[i_f] += 1
                elif dot_ax <= 0.35:
                    ortho_a[i_f] += 1
                sdir = _shared_dir(inter)
                if sdir is not None:
                    dot_j = abs(float(sdir[0] * ax_j[0] + sdir[1] * ax_j[1]))
                    # Attaching along the LONG side of an elongated neighbour (e.g. Chân rẽ on a trunk duct)
                    if lj >= 1.45 * max(sj, 1e-6) and dot_j >= 0.75 and sl < 0.75 * lj:
                        t_side_a[i_f] += 1
                        r_val = lj / max(sl, 1.0)
                        if r_val > t_side_max_a[i_f]:
                            t_side_max_a[i_f] = r_val
                    # Meeting the END CAP of a neighbour (perpendicular to neighbour's long axis, width matches)
                    elif (lj < 1.45 * max(sj, 1e-6) or dot_j <= 0.45) and 0.55 * max(sj, 1e-6) <= sl <= 1.35 * max(sj, 1e-6):
                        end_match_a[i_f] += 1

    out = np.empty((n_faces, len(FEATURES)), dtype=np.float64)
    for i in range(n_faces):
        long_, short, _ = mrr_info[i]
        area_i = float(areas[i])
        rlen = float(rlens[i])
        cs = float(cov_sum[i])
        arc = float(arc_num[i] / max(cs, 1e-9)) if cs > 0.0 else 0.0
        sg = simps[i]
        simp_v = np.asarray(sg.exterior.coords)[:-1]
        corners = len(simp_v)
        side_ratio, skew, right = _taper(simp_v)
        par_gap = _par_gap_deg(simp_v)
        n_cusps = _cusps_from_simplified(sg)
        stroke_cov = min(cs / rlen, 2.0)

        sh_lens = nb_sh_lens[i]
        nb_areas = nb_areas_l[i]
        nb_cnt = len(sh_lens)
        shared_frac = min(sum(sh_lens) / rlen, 1.5)
        if nb_cnt >= 2:
            top2 = sorted(sh_lens, reverse=True)[:2]
            w_diff = top2[1] / max(top2[0], 1e-6)
        else:
            w_diff = 1.0
        max_ar = float(np.log1p(max(nb_areas) / max(area_i, 1.0))) if nb_areas else 0.0
        min_ar = float(np.log1p(min(nb_areas) / max(area_i, 1.0))) if nb_areas else 0.0
        max_sh_short = (max(sh_lens) / max(short, 1e-6)) if sh_lens else 0.0

        out[i] = (
            np.log1p(area_i), np.log1p(long_), np.log1p(short), long_ / max(short, 1e-6), area_i / max(long_ * short, 1e-9),
            area_i / max(float(hull_areas[i]), 1e-9), min(corners, 20), arc, np.log1p(rlen),
            4 * np.pi * area_i / max(rlen ** 2, 1e-9), int(n_strokes[i]), np.log1p(ink_in[i]), int(touch_cnt[i]),
            side_ratio, skew, right,
            # F225 context features
            nb_cnt, shared_frac, int(collinear_a[i]), int(ortho_a[i]), int(t_side_a[i]), min(float(t_side_max_a[i]), 30.0),
            int(end_match_a[i]), w_diff, max_ar, min_ar, min(max_sh_short, 5.0),
            n_cusps, par_gap, stroke_cov,
        )
    return out


# ---- post-processing step (F151): faces named by a classifier, added where the logic left nothing -------------------------------------

PICK_THR = 0.3     # strokes with p >= this draw the faces (chosen on the train pages, F151)
FACE_CONF = 0.5    # a face becomes an object when its class probability >= this (chosen on the train pages, cross-fitted by document)
COVERED = 0.3      # a face more than this share inside the logic's objects belongs to the logic
CLASSES = ("none", "Duct", "Elbow", "Elbow vuông", "Concentric Reducer", "Eccentric Reducer", "Chân rẽ", "Tee")   # Tee: out of the
# 6-class benchmark (scored as nothing) but a real object of the drawings; without it a tee face was called an Elbow (F152)


def fit_classifier(npz_paths, viewer: bool = True, skip_doc: str | None = None, gt_share: float | None = 0.35):
    """HistGradientBoosting on 30-feature faces (tools/build_faces_train_e13.py / faces_train.py).
    When ``viewer=True`` and both GT and Viewer faces are present, ``gt_share`` balances GT vs Viewer sample mass
    (default 0.35 GT / 0.65 Viewer, matching ``viewer_share=0.65``; None = unweighted). -> (clf, class names)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y, pgs, classes = [], [], [], CLASSES
    for t in npz_paths:
        z = np.load(t)
        pg = z["page"].astype(str)
        keep = np.ones(len(z["y"]), bool) if viewer else ~np.char.startswith(pg, "vw_")
        if skip_doc:
            keep &= ~np.char.startswith(pg, skip_doc)
        X.append(z["X"][keep]); y.append(z["y"][keep]); pgs.append(pg[keep])
        classes = tuple(str(c) for c in z["classes"])
    X_all, y_all, pg_all = np.concatenate(X), np.concatenate(y), np.concatenate(pgs)
    sw = None
    if viewer and gt_share is not None:
        is_gt = ~np.char.startswith(pg_all, "vw_")
        if is_gt.any() and (~is_gt).any():
            sw = np.where(is_gt, gt_share / is_gt.sum(), (1.0 - gt_share) / (~is_gt).sum()) * len(y_all)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.07, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    clf.fit(X_all, y_all, sample_weight=sw)
    return clf, classes


def save_classifier(path, clf, classes) -> None:
    import joblib

    joblib.dump({"clf": clf, "classes": list(classes), "features": list(FEATURES), "gap": GAP}, path)


def load_classifier(path):
    import joblib

    m = joblib.load(path)
    if list(m["features"]) != list(FEATURES):
        raise ValueError(f"{path}: face features {m['features']} differ from this code's {list(FEATURES)}; refit it")
    return m["clf"], tuple(m["classes"])


RULE_SCORE = 0.9  # a face a shape rule confirms keeps at least this score
SHARED = 2.0      # pt of common outline: two faces are neighbours (a duct and the reducer after it)
TAPER_RATIO, TAPER_SKEW = 0.95, 4.0   # a 4-corner face narrower at one end than the other by > 5 %, flanks > 4 degrees apart
FUSE = 0.5   # F163: weight of the model's per-stroke class vote against the shape classifier (chosen on the GT train pages: 0.766 vs 0.754 / 0.738)
EDGE_KEEP, EDGE_MERGE = 0.0, 1.1   # F169: same-object link probability: a region whose outline links average below EDGE_KEEP is
# dropped; two touching Duct regions whose links across the cut average >= EDGE_MERGE are one duct (off until chosen on the train pages)
DUCT_MIN_W, DUCT_ISO_AREA = 1.0, 0.0   # F164: drop Duct objects narrower than this (pt) / lone ones smaller than this (pt^2); chosen on the GT train pages (0.766 -> 0.796; the lone-box filter added <= 0.001, off)


def neighbours(faces: list) -> list[list[int]]:
    """For each face, the faces it shares >= SHARED pt of outline with (faces grow back to the same centre line, so neighbours touch)."""
    import shapely

    if not faces:
        return []
    arr = np.asarray(faces, dtype=object)
    bufs = shapely.buffer(arr, 0.5, quad_segs=16)
    exts = shapely.get_exterior_ring(arr)
    tree = shapely.STRtree(arr)
    pi, pj = tree.query(bufs, predicate="intersects")
    keep = pi != pj
    pi, pj = pi[keep], pj[keep]
    out: list[list[int]] = [[] for _ in range(len(faces))]
    if len(pi):
        sl = shapely.length(shapely.intersection(exts[pi], bufs[pj]))
        ok = sl >= SHARED
        for i, j in zip(pi[ok].tolist(), pj[ok].tolist()):
            out[i].append(j)
    return out


CUSP_TURN = 60.0   # degrees of inward turn at one corner of the outline: the point where a Y tee's two inner arcs meet


def cusps(f) -> int:
    """Reflex corners turning >= CUSP_TURN degrees inward on the simplified outline (F154). A Y tee has one (its two inner arcs meet in a
    point), a square tee two; an elbow, a duct or a reducer none."""
    import shapely

    g = shapely.simplify(f, 0.5)
    v = np.asarray(g.exterior.coords)[:-1]
    if len(v) < 4:
        return 0
    if g.exterior.is_ccw is False:
        v = v[::-1]
    e = np.roll(v, -1, 0) - v
    ang = np.degrees(np.arctan2(e[:, 1], e[:, 0]))
    turn = ((ang - np.roll(ang, 1) + 180) % 360) - 180          # + left (convex for a ccw ring), - right (reflex)
    return int(np.sum(turn <= -CUSP_TURN))


FG_UNION_THR = 0.35  # F225: Bayes F2 foreground-union rescue when split foreground prob sum >= 0.35 beats single-class argmax=none


def _name_faces(faces, X, clf, classes):
    """Classifier names + Bayes F2 foreground-union rescue + refined shape rules (F152 / F225): inline taper between two
    end-cap neighbours -> reducer (never overriding T-attached Chân rẽ shoe-taps on a trunk duct's long side);
    Elbow with >= 3 object neighbours -> Tee. -> names, scores, why, object-neighbour lists, class probabilities."""
    import shapely

    pr = clf.predict_proba(X)
    names = [classes[int(clf.classes_[k])] for k in pr.argmax(1)]
    score = pr.max(1).astype(np.float64, copy=True)
    fg_cols = [k for k, c in enumerate(clf.classes_) if classes[int(c)] != "none"]
    if fg_cols and len(faces):
        fg_sum = pr[:, fg_cols].sum(axis=1)
        best_fg = np.argmax(pr[:, fg_cols], axis=1)
        for i in range(len(faces)):
            if names[i] == "none" and fg_sum[i] >= FG_UNION_THR:
                k_real = fg_cols[int(best_fg[i])]
                names[i] = classes[int(clf.classes_[k_real])]
                score[i] = float(fg_sum[i])
    nb = [[j for j in js if names[j] != "none"] for js in neighbours(faces)]   # neighbours the classifier calls an object
    col = {n: i for i, n in enumerate(FEATURES)}
    mrr_cache: dict[int, tuple[float, float, np.ndarray]] = {}
    why = ["face"] * len(faces)
    for i, f in enumerate(faces):
        x = X[i]
        # F225 fix: taper rule ONLY applies to inline reducers (t_side == 0) whose parallel ends both meet neighbour END CAPS,
        # NOT 45-degree Chân rẽ shoe-taps sitting on the long side wall of a trunk duct (was overwriting 59 GT5 + 92 VW8 Chân rẽ!).
        if (x[col["corners"]] == 4 and x[col["side_ratio"]] < TAPER_RATIO and x[col["skew_deg"]] >= TAPER_SKEW
                and len(nb[i]) >= 2 and x[col["nb_t_side_cnt"]] == 0):
            v = np.asarray(shapely.simplify(f, 0.5).exterior.coords)[:-1]
            e = np.roll(v, -1, 0) - v
            ang = np.degrees(np.arctan2(e[:, 1], e[:, 0]))
            d = lambda a, b: min(abs((ang[a] - ang[b]) % 180), 180 - abs((ang[a] - ang[b]) % 180))
            par = (0, 2) if d(0, 2) <= d(1, 3) else (1, 3)
            ends = [shapely.LineString([v[k], v[(k + 1) % 4]]) for k in par]
            def _meets_end(end, j):
                if end.buffer(0.6).intersection(faces[j].exterior).length < 0.5 * end.length:
                    return False
                mj = mrr_cache.get(j)
                if mj is None:
                    mj = mrr_cache[j] = _mrr_axes(faces[j])
                lj, sj, ax_j = mj
                if lj < 1.45 * max(sj, 1e-6):
                    return True
                ev = np.asarray(end.coords[-1]) - np.asarray(end.coords[0])
                el = float(np.hypot(ev[0], ev[1]))
                if el < 1e-6:
                    return False
                dot_j = abs(float((ev[0] * ax_j[0] + ev[1] * ax_j[1]) / el))
                # Reject if end runs along the long side of a much longer trunk duct (T-branch)
                return not (dot_j >= 0.75 and el < 0.75 * lj)
            if all(any(_meets_end(end, j) for j in nb[i]) for end in ends):
                names[i], why[i] = reducer_kind(f), "taper"   # F163: concentric / eccentric by its corners
                score[i] = max(score[i], RULE_SCORE)
        if names[i] == "Elbow" and (len(nb[i]) >= 3 or (x[col["arc_share"]] >= 0.1 and cusps(f) >= 1)):
            names[i], why[i] = "Tee", "three_ways"
            score[i] = max(score[i], RULE_SCORE)
    return names, score, why, nb, pr


#: General classes without suffixes (user 2026-11-16): the Viewer labels every reducer "reducer" (-> Reducer vuông tròn) and every branch
#: "branch", the GT splits concentric / eccentric / square-to-round. The class head's probabilities are summed per general class before
#: the vote; concentric vs eccentric is then decided by the region's shape (an eccentric reducer has one straight flank: 2 right corners).
VOTE_GROUPS = (("Duct", ("Duct",)), ("Elbow", ("Elbow",)), ("Elbow vuông", ("Elbow vuông",)),
               ("Reducer", ("Concentric Reducer", "Eccentric Reducer", "Reducer vuông tròn")),
               ("Chân rẽ", ("Chân rẽ", "Chân rẽ vuông tròn")), ("Tee", ("Tee",)))   # the 6 classes of the class head (user 2026-11-16)


def vote_names(faces: list, segs, picked: np.ndarray, stroke_cls: np.ndarray) -> tuple[list, np.ndarray]:
    """OpenStroke29 (F161): each region's class = the length-weighted vote of the model's class probabilities of the picked strokes on its
    outline (stroke_cls: (n shapes, N_CLASSES), column 0 = no object, not voted), summed per general class (VOTE_GROUPS). A reducer is
    named Eccentric when its outline has 2 right corners, else Concentric; a branch keeps its stronger variant. -> (names, share)."""
    import shapely

    from .objects import HVAC_CLASSES

    m = np.flatnonzero(picked[segs.owner])
    if not len(m) or not faces:
        return ["none"] * len(faces), np.zeros(len(faces))
    lines = shapely.linestrings(segs.seg[m].reshape(-1, 2, 2))
    tree = shapely.STRtree(lines)
    pc = np.asarray(stroke_cls, float)[segs.owner[m]]
    cols = [[HVAC_CLASSES.index(c) for c in members] for _, members in VOTE_GROUPS]
    names, share = [], np.zeros(len(faces))
    for i, f in enumerate(faces):
        band = f.exterior.buffer(GAP + 0.3)
        near = tree.query(band, predicate="intersects")
        if not len(near):
            names.append("none"); continue
        on = shapely.length(shapely.intersection(lines[near], band))
        v = (pc[near] * on[:, None]).sum(0)
        g = np.asarray([v[c].sum() for c in cols])
        k = int(np.argmax(g))
        if g[k] <= 0:
            names.append("none"); continue
        share[i] = g[k] / max(g.sum(), 1e-9)
        general = VOTE_GROUPS[k][0]
        if general == "Reducer":
            names.append(reducer_kind(f))
        elif general == "Chân rẽ":
            names.append(HVAC_CLASSES[max(cols[k], key=lambda c: v[c])])
        else:
            names.append(general)
    return names, share


def vote_shares(faces: list, segs, picked: np.ndarray, stroke_cls: np.ndarray) -> np.ndarray:
    """Per region, the model's class vote over VOTE_GROUPS: length-weighted class probabilities of the picked strokes on its outline,
    summed per general class and normalised to 1 (rows of 0 when no stroke lies on the outline). -> (n regions, 6)."""
    import shapely

    from .objects import HVAC_CLASSES

    G = np.zeros((len(faces), len(VOTE_GROUPS)))
    m = np.flatnonzero(picked[segs.owner])
    if not len(m) or not faces:
        return G
    lines = shapely.linestrings(segs.seg[m].reshape(-1, 2, 2))
    tree = shapely.STRtree(lines)
    pc = np.asarray(stroke_cls, float)[segs.owner[m]]
    cols = [[HVAC_CLASSES.index(c) for c in members] for _, members in VOTE_GROUPS]
    faces_arr = np.asarray(faces, dtype=object)
    bands = shapely.buffer(shapely.get_exterior_ring(faces_arr), GAP + 0.3, quad_segs=16)
    fi, li = tree.query(bands, predicate="intersects")
    if not len(fi):
        return G
    on = shapely.length(shapely.intersection(lines[li], bands[fi]))
    # Sum weighted probabilities per group for each face
    n_f = len(faces)
    for gj, col_idx in enumerate(cols):
        w_g = pc[li][:, col_idx].sum(axis=1) * on
        G[:, gj] = np.bincount(fi, weights=w_g, minlength=n_f)
    s = G.sum(axis=1, keepdims=True)
    return np.divide(G, s, out=np.zeros_like(G), where=s > 0)


def classifier_shares(pr: np.ndarray, clf, classes) -> np.ndarray:
    """The shape classifier's probabilities summed over VOTE_GROUPS (no-object column left out), rows normalised. -> (n, 6)."""
    G = np.zeros((len(pr), len(VOTE_GROUPS)))
    for k, c in enumerate(clf.classes_):
        name = classes[int(c)]
        for j, (_, members) in enumerate(VOTE_GROUPS):
            if name in members:
                G[:, j] += pr[:, k]
    s = G.sum(1, keepdims=True)
    return np.divide(G, s, out=np.zeros_like(G), where=s > 0)


def group_name(f, j: int) -> str:
    """A general class -> the name of the object: a reducer is Eccentric with 2 right corners, else Concentric."""
    import shapely

    general = VOTE_GROUPS[j][0]
    if general == "Reducer":
        return reducer_kind(f)
    return general


def reducer_kind(f) -> str:
    """Eccentric only for a tapering 4-corner outline with exactly 2 right corners (one straight flank); else Concentric (a rectangle's 4 right
    corners must not count, F163)."""
    import shapely

    simp = np.asarray(shapely.simplify(f, 0.5).exterior.coords)[:-1]
    ratio, skew, right = _taper(simp)
    return "Eccentric Reducer" if len(simp) == 4 and right == 2 and ratio < TAPER_RATIO else "Concentric Reducer"


def _obj(f, name, score, why, item=-1):
    import shapely

    g = shapely.simplify(f, 0.3)
    o = {"shape_name": name, "vertices": np.asarray(g.exterior.coords)[:-1].round(3).tolist(), "score": float(score), "refined": why,
         "face": f}
    if item >= 0:
        o["item"] = int(item)                                            # the PDF item that drew it (segment ownership, F157)
    return o


def face_objects(segs, prob: np.ndarray, curved: np.ndarray, clf, classes, thr: float = PICK_THR, conf: float = FACE_CONF) -> list[dict]:
    """Faces of the strokes with p >= thr, named by the classifier and the shape rules (_name_faces); each kept face is one object (its
    polygon, ``score``, ``refined`` = "face" or the rule's name)."""
    picked = prob >= thr
    faces = page_candidates(segs, picked, curved=curved)
    if not faces:
        return []
    names, score, why, _, _ = _name_faces(faces, face_features(faces, segs, picked, curved), clf, classes)
    return [_obj(f, names[i], score[i], why[i]) for i, f in enumerate(faces) if names[i] != "none" and score[i] >= conf]


MUST_BORDER = 0.7    # a face whose outline lies >= 70 % on picked strokes ...
MUST_FREE = 0.5      # ... of which >= 50 % is in no object yet ...
MUST_WIDTH = 60.0    # ... and no wider than this (pt; a duct, not the room between ducts) must become an object (F153)
MUST_SCORE = 0.6     # its score: kept at the usual face threshold (0.5)


def drop_spurious_ducts(objects: list[dict], min_width: float = DUCT_MIN_W, iso_area: float = DUCT_ISO_AREA) -> list[dict]:
    """F164: Duct objects that are not ducts. A Duct whose min-area box is narrower than ``min_width`` pt (slivers the must-cover step closes
    between double or cross lines), or one touching no other object (1 pt) with area < ``iso_area`` (a lone small box: symbol, tag, furniture).
    Other classes are kept."""
    import cv2
    import shapely
    from shapely.geometry import Polygon

    if not objects or (min_width <= 0 and iso_area <= 0):
        return objects
    P = [Polygon(np.asarray(o["vertices"], float)).buffer(0) if len(o.get("vertices") or []) >= 3 else Polygon() for o in objects]
    tree = shapely.STRtree(P)
    keep = []
    for i, o in enumerate(objects):
        if (o.get("shape_name") or "").startswith("Duct") and len(o.get("vertices") or []) >= 3 and P[i].area > 0:
            w, h = cv2.minAreaRect(np.asarray(o["vertices"], np.float32))[1]
            if min(w, h) < min_width:
                continue
            if iso_area > 0 and P[i].area < iso_area and not any(j != i for j in tree.query(P[i].buffer(1.0), predicate="intersects")):
                continue
        keep.append(o)
    return keep


def outline_shapes(polys: list, segs, picked: np.ndarray, band: float = GAP + 0.5, share: float = 0.5) -> list[np.ndarray]:
    """Per polygon, the picked shapes drawing its outline: >= ``share`` of the shape's length within ``band`` pt of the polygon's boundary."""
    import shapely

    m = np.flatnonzero(picked[segs.owner])
    out = [np.zeros(0, np.int64) for _ in polys]
    if not len(m) or not polys:
        return out
    lines = shapely.linestrings(segs.seg[m].reshape(-1, 2, 2))
    own = segs.owner[m]
    tot = np.bincount(own, weights=shapely.length(lines), minlength=len(picked))
    tree = shapely.STRtree(lines)
    for k, g in enumerate(polys):
        if g is None or g.is_empty:
            continue
        b = g.boundary.buffer(band)
        near = tree.query(b, predicate="intersects")
        if not len(near):
            continue
        on = np.bincount(own[near], weights=shapely.length(shapely.intersection(lines[near], b)), minlength=len(picked))
        out[k] = np.flatnonzero((on > 0) & (on >= share * np.maximum(tot, 1e-9)))
    return out


class _Links:
    """Lookup of the same-object probability of a pair of shapes (payload shape ids)."""

    def __init__(self, eij: np.ndarray, eprob: np.ndarray, n: int):
        e = np.sort(np.asarray(eij, np.int64).reshape(-1, 2), 1)
        self.n = int(max(n, int(e.max()) + 1 if len(e) else 1))
        key = e[:, 0] * self.n + e[:, 1]
        o = np.argsort(key)
        self.key, self.p = key[o], np.asarray(eprob, float)[o]

    def among(self, a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
        """Probabilities of the links inside set a (b None) or between sets a and b."""
        a = np.asarray(a, np.int64)
        b = a if b is None else np.asarray(b, np.int64)
        if not len(a) or not len(b) or not len(self.key):
            return np.zeros(0)
        x, y = np.meshgrid(a, b, indexing="ij")
        lo, hi = np.minimum(x, y).ravel(), np.maximum(x, y).ravel()
        keep = lo < hi
        q = np.unique(lo[keep] * self.n + hi[keep])
        i = np.searchsorted(self.key, q)
        i = np.clip(i, 0, len(self.key) - 1)
        hit = self.key[i] == q
        return self.p[i[hit]]


def edge_refine(objects: list[dict], segs, picked: np.ndarray, eij: np.ndarray, eprob: np.ndarray, keep: float = EDGE_KEEP,
                merge: float = EDGE_MERGE) -> list[dict]:
    """F169 (OpenStroke30 link head): on the post-processing's own regions (objects with ``refined``; the logic's objects stay as they are)
    - drop a region whose outline strokes' links (>= 2) average a same-object probability < ``keep`` (a closed shape of background strokes:
      parking bays, cabinets, whose outlines the head was trained to call different objects);
    - merge two touching Duct regions when the links between the strokes only on one and only on the other (the duct's sides across the cut)
      average >= ``merge`` and the union stays a rectangle (min-area box filled >= 0.85)."""
    import cv2
    import shapely
    from shapely.geometry import Polygon

    if not objects or eij is None or not len(eij) or (keep <= 0 and merge > 1):
        return objects
    links = _Links(eij, eprob, len(picked))
    own = [i for i, o in enumerate(objects) if o.get("refined") and len(o.get("vertices") or []) >= 3]
    polys = {i: Polygon(np.asarray(objects[i]["vertices"], float)).buffer(0) for i in own}
    outl = dict(zip(own, outline_shapes([polys[i] for i in own], segs, picked)))
    drop = set()
    if keep > 0:
        for i in own:
            p = links.among(outl[i])
            if len(p) >= 2 and p.mean() < keep:
                drop.add(i)
    groups = {i: [i] for i in own if i not in drop}
    if merge <= 1:
        ducts = [i for i in own if i not in drop and (objects[i].get("shape_name") or "").startswith("Duct")]
        tree = shapely.STRtree([polys[i] for i in ducts]) if ducts else None
        pairs = []
        for a_i, i in enumerate(ducts):
            for b_i in (tree.query(polys[i].buffer(0.6), predicate="intersects") if tree is not None else []):
                j = ducts[int(b_i)]
                if j <= i:
                    continue
                if polys[i].buffer(0.6).intersection(polys[j].buffer(0.6)).area < 1.0:
                    continue
                only_i = np.setdiff1d(outl[i], outl[j])
                only_j = np.setdiff1d(outl[j], outl[i])
                p = links.among(only_i, only_j)
                if len(p) and p.mean() >= merge:
                    pairs.append((-float(p.mean()), i, j))
        parent = {i: i for i in groups}

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        shape = {i: polys[i] for i in groups}
        for _, i, j in sorted(pairs):
            ri, rj = root(i), root(j)
            if ri == rj:
                continue
            u = shapely.union_all([shape[ri].buffer(0.4), shape[rj].buffer(0.4)]).buffer(-0.4)
            if u.geom_type != "Polygon":
                continue
            (cx, cy), (w, h), _ = cv2.minAreaRect(np.asarray(u.exterior.coords, np.float32))
            if w * h <= 0 or u.area / (w * h) < 0.85:
                continue
            parent[rj] = ri
            shape[ri] = u
            groups[ri] += groups.pop(rj)
    out = [o for i, o in enumerate(objects) if i not in polys]                       # the logic's objects
    for r, members in groups.items():
        if len(members) == 1:
            out.append(objects[r])
            continue
        g = shapely.simplify(shape[r], 0.3)
        out.append({**objects[r], "vertices": np.asarray(g.exterior.coords)[:-1].round(3).tolist(),
                    "score": max(float(objects[m].get("score", 1.0)) for m in members), "refined": "merged"})
    return out


EDGE_LINK = 0.5    # OpenStroke31 (F178): two picked strokes are one object when their same-object probability >= this (balanced loss: 0.5 is
                   # its natural cut; fixed before scoring, to be re-chosen on the GT train pages)
EDGE_SHARE = 0.6   # a face belongs to the link group drawing >= this share of its outline length; a face split between groups stays alone


def link_groups(n: int, picked: np.ndarray, eij: np.ndarray, eprob: np.ndarray, link: float = EDGE_LINK) -> np.ndarray:
    """Connected components of the picked strokes over the links with same-object probability >= ``link`` (-1 for strokes not picked)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    e = np.asarray(eij, np.int64).reshape(-1, 2)
    ok = picked[e[:, 0]] & picked[e[:, 1]] & (np.asarray(eprob, float) >= link) if len(e) else np.zeros(0, bool)
    e = e[ok]
    _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
    return np.where(picked, lab, -1)


def edge_objects(segs, prob: np.ndarray, curved: np.ndarray, clf, classes, eij: np.ndarray, eprob: np.ndarray, thr: float = PICK_THR,
                 link: float = EDGE_LINK, share: float = EDGE_SHARE, conf: float = FACE_CONF, same_name: bool = True) -> list[dict]:
    """OpenStroke31 (F178): objects from the model's same-object links instead of one object per closed face.
    1. Faces of the picked strokes (page_candidates), as in face_objects, named (_name_faces).
    2. Link groups: picked strokes joined by links with probability >= ``link`` (link_groups).
    3. Each face goes to the group drawing >= ``share`` of its outline length (a face whose outline mixes groups stays its own object);
       the faces of one group (and, with ``same_name``, of one name: the link head also joins a branch to its duct, F178) are merged: a duct
       cut in two by a crossing line becomes one region again.
    4. Each merged region is named again like a face (shape classifier + taper / three-way rules); refined = "edge"."""
    import shapely

    picked = prob >= thr
    faces = page_candidates(segs, picked, curved=curved)
    if not faces:
        return []
    fname = _name_faces(faces, face_features(faces, segs, picked, curved), clf, classes)[0] if same_name else [""] * len(faces)
    lab = link_groups(len(prob), picked, eij, eprob, link)
    L = np.bincount(segs.owner, weights=np.hypot(segs.seg[:, 2] - segs.seg[:, 0], segs.seg[:, 3] - segs.seg[:, 1]), minlength=len(prob))
    key = []
    for k, ids in enumerate(outline_shapes(faces, segs, picked)):
        g = lab[ids]
        ok = g >= 0
        if not ok.any():
            key.append(("face", k))
            continue
        w = np.bincount(g[ok], weights=L[ids][ok])
        best = int(np.argmax(w))
        key.append(("group", best, fname[k]) if w[best] >= share * w.sum() and fname[k] != "none" else ("face", k))
    members = {}
    for k, kk in enumerate(key):
        members.setdefault(kk, []).append(k)
    regions = []
    for ks in members.values():
        if len(ks) == 1:
            regions.append(faces[ks[0]])
            continue
        u = shapely.union_all([faces[k].buffer(0.4, join_style="mitre") for k in ks]).buffer(-0.4, join_style="mitre")
        regions += [p for p in shapely.get_parts(u) if p.geom_type == "Polygon" and p.area > MIN_AREA]
    names, score, why, _, _ = _name_faces(regions, face_features(regions, segs, picked, curved), clf, classes)
    return [_obj(f, names[i], score[i], "edge" if why[i] == "face" else why[i]) for i, f in enumerate(regions)
            if names[i] != "none" and score[i] >= conf]


def post_objects(*args, duct_min_width: float = DUCT_MIN_W, duct_iso_area: float = DUCT_ISO_AREA, edges=None,
                 edge_keep: float = EDGE_KEEP, edge_merge: float = EDGE_MERGE, **kw) -> list[dict]:
    """``_post_objects`` (below), then ``edge_refine`` with the model's same-object links (``edges`` = (shape-id pairs, probabilities),
    OpenStroke30 checkpoints; F169), then ``drop_spurious_ducts`` (F164)."""
    out = _post_objects(*args, **kw)
    if edges is not None and edges[0] is not None and len(edges[0]):
        segs, prob = args[1], args[2]
        out = edge_refine(out, segs, np.asarray(prob) >= kw.get("thr", PICK_THR), edges[0], edges[1], edge_keep, edge_merge)
    return drop_spurious_ducts(out, duct_min_width, duct_iso_area)


def _post_objects(objects: list[dict], segs, prob: np.ndarray, curved: np.ndarray, clf, classes, thr: float = PICK_THR,
                 conf: float = FACE_CONF, covered: float = COVERED, must: bool = True, order: str = "logic",
                 graph: bool = True, stroke_cls: np.ndarray | None = None, fuse: float = FUSE) -> list[dict]:
    """The post-processing after AI_Takeoff's logic (F153). order="logic": the logic's objects stay as they are; faces of the picked strokes (p >= thr,
    open elbow ends closed) are added where the logic left nothing:
    - a face the classifier / shape rules name with score >= conf (F151 / F152);
    - with ``must``, every other face the picked strokes enclose that nothing covers yet (MUST_*): the model picked those strokes, so they
      are an object. Its class comes from its shape: curved outline -> Elbow (Tee with >= 3 object neighbours), taper -> reducer, else the
      classifier's best object class (refined = "picked", score = MUST_SCORE).
    order="faces" (F154, user 2026-11-16): the faces come first (named, or must-cover by outline and width alone), then the logic's objects
    lying mostly outside them."""
    import shapely
    from shapely.geometry import Polygon

    picked = prob >= thr
    src = inner = None
    if order == "items":
        faces, src, inner = priority_candidates(segs, picked, curved=curved)
    else:
        faces = page_candidates(segs, picked, curved=curved, graph=graph)
    if not faces:
        return list(objects)
    X = face_features(faces, segs, picked, curved)
    names, score, why, nb, pr = _name_faces(faces, X, clf, classes)
    fused = None
    if stroke_cls is not None:   # OpenStroke29 / F163: (1 - fuse) x shape classifier + fuse x the model's per-stroke class vote
        F = (1.0 - fuse) * classifier_shares(pr, clf, classes) + fuse * vote_shares(faces, segs, picked, stroke_cls)
        fused = [group_name(f, int(np.argmax(F[i]))) if F[i].max() > 0 else "none" for i, f in enumerate(faces)]
        for i in range(len(faces)):
            if fused[i] != "none" and names[i] != "none" and why[i] not in ("taper", "three_ways"):   # the shape rules keep their say
                names[i], why[i] = fused[i], "model"
                score[i] = max(score[i], conf) if fuse >= 1.0 else score[i]
    if inner is not None:   # F158: a 4-corner symbol item with inner lines on one duct's side = square-to-round branch, not a reducer
        for i, f in enumerate(faces):
            if inner[i] >= 2 and _corners(f) == 4 and len(nb[i]) <= 1:
                names[i], score[i], why[i] = "Chân rẽ vuông tròn", max(score[i], RULE_SCORE), "symbol"
    polys = [Polygon(np.asarray(o["vertices"], float)).buffer(0) for o in objects if len(o.get("vertices") or []) >= 3]
    cover = shapely.union_all(polys) if polys else None
    m = picked[segs.owner]
    lines = shapely.linestrings(segs.seg[m].reshape(-1, 2, 2))
    mid = (segs.seg[m, :2] + segs.seg[m, 2:]) / 2
    free = ~shapely.contains_xy(cover.buffer(1.5), mid[:, 0], mid[:, 1]) if cover is not None else np.ones(len(mid), bool)
    tree = shapely.STRtree(lines)
    col = {n: i for i, n in enumerate(FEATURES)}
    obj_k = [k for k, c in enumerate(clf.classes_) if classes[int(c)] != "none"]
    faces_first = order in ("faces", "items")   # F154 / F156 (user): closed regions first, the logic fills the rest
    if faces_first:
        cover, free = None, np.ones(len(mid), bool)
        # a symbol the logic recognised (VCD, damper...: not one of the duct classes) keeps its area (F157)
        from .objects import class_of
        sym = [Polygon(np.asarray(o["vertices"], float)).buffer(0) for o in objects
               if len(o.get("vertices") or []) >= 3 and not class_of(o.get("shape_name") or "") and not str(o.get("shape_name", "")).startswith("Tee")]
        sym_u = shapely.union_all(sym) if sym else None
    out = [] if faces_first else list(objects)
    for i, f in enumerate(faces):
        if cover is not None and f.intersection(cover).area > covered * f.area:
            continue                                                       # the logic's object
        if faces_first and sym_u is not None and f.intersection(sym_u).area > 0.5 * f.area:
            continue                                                       # the logic's symbol
        if names[i] != "none" and score[i] >= conf:
            out.append(_obj(f, names[i], score[i], why[i], src[i] if src else -1))
            continue
        if not must or np.expm1(X[i, col["log_short"]]) > MUST_WIDTH:
            continue
        band = f.exterior.buffer(GAP + 0.3)
        near = tree.query(band, predicate="intersects")
        if not len(near):
            continue
        on = shapely.length(shapely.intersection(lines[near], band))
        if on.sum() < MUST_BORDER * f.exterior.length or on[free[near]].sum() < MUST_FREE * on.sum():
            continue
        x = X[i]
        if x[col["arc_share"]] >= 0.2:
            name = "Tee" if len(nb[i]) >= 3 or cusps(f) >= 1 else "Elbow"
        elif why[i] == "taper":
            name = names[i]                                                # a reducer between two objects (_name_faces)
        elif fused is not None and fused[i] != "none":
            name = fused[i]                                                # F163: classifier + model vote
        else:
            name = classes[int(clf.classes_[obj_k[int(np.argmax(pr[i, obj_k]))]])]
            if name in ("Concentric Reducer", "Eccentric Reducer"):        # a reducer needs a duct at both ends (F156: lone symbols)
                bo = [k for k in obj_k if classes[int(clf.classes_[k])] not in ("Concentric Reducer", "Eccentric Reducer")]
                name = classes[int(clf.classes_[bo[int(np.argmax(pr[i, bo]))]])]
        out.append(_obj(f, name, MUST_SCORE, "picked", src[i] if src else -1))
    if faces_first and out:   # the logic's objects where no face object is
        fu = shapely.union_all([o["face"] for o in out])
        out += [o for o, g in ((o, Polygon(np.asarray(o["vertices"], float)).buffer(0)) for o in objects if len(o.get("vertices") or []) >= 3)
                if g.area > 0 and g.intersection(fu).area <= covered * g.area]
    elif faces_first:
        out = list(objects)
    return [{k: v for k, v in o.items() if k != "face"} for o in out]


def add_faces(objects: list[dict], faces: list[dict], covered: float = COVERED) -> list[dict]:
    """The logic's objects, plus every face object lying mostly outside them (area share inside <= covered)."""
    import shapely
    from shapely.geometry import Polygon

    polys = [Polygon(np.asarray(o["vertices"], float)).buffer(0) for o in objects if len(o.get("vertices") or []) >= 3]
    cover = shapely.union_all(polys) if polys else None
    extra = [{k: v for k, v in f.items() if k != "face"} for f in faces
             if cover is None or f["face"].intersection(cover).area <= covered * f["face"].area]
    return list(objects) + extra


def segment_objects(objects: list[dict], segs, picked: np.ndarray, tol: float = 0.8) -> np.ndarray:
    """Per segment, the object it draws (-1 none), F156: an item drawing a duct and its reducer gives its segments to two objects. A picked
    segment goes to the object whose outline runs nearest its midpoint (within tol; F180: an object ends at its corners, so a dash on a
    duct's wall just past a fitting's corner stays the duct's), on a tie (EDGE_TIE pt) the object its own item drew, else the smallest
    such object; else the smallest object containing it (F157: a duct's side shared with a symbol below it stays the duct's)."""
    import shapely
    from shapely.geometry import Polygon

    out = np.full(len(segs.seg), -1, np.int64)
    polys, idx = [], []
    for i, o in enumerate(objects):
        v = o.get("vertices") or []
        if len(v) >= 3:
            g = Polygon(np.asarray(v, float)).buffer(0)
            if g.area > 0:
                polys.append(g); idx.append(i)
    if not polys:
        return out
    polys_arr = np.asarray(polys, dtype=object)
    idx_arr = np.asarray(idx, dtype=np.int64)
    area = shapely.area(polys_arr)
    bounds = shapely.boundary(polys_arr)
    m = np.flatnonzero(picked[segs.owner])
    if not len(m):
        return out
    mid = shapely.points((segs.seg[m, :2] + segs.seg[m, 2:]) / 2)
    tree = shapely.STRtree(polys_arr)
    pi, gi = tree.query(mid, predicate="dwithin", distance=tol)
    if not len(pi):
        return out
    from .objects import EDGE_TIE

    bd = shapely.distance(mid[pi], bounds[gi])
    on = bd <= tol
    item = np.asarray([objects[i].get("item", -1) for i in idx], dtype=np.int64)
    own = segs.owner[m]
    score = np.where(on, area[gi] + np.floor(bd / EDGE_TIE) * 1e12 - np.where(item[gi] == own[pi], 1e11, 0.0), area[gi] + 1e14)
    # Stable lexsort by (pi, score) picks the minimum score per segment (first hit wins exact ties, matching the original loop)
    order = np.lexsort((score, pi))
    pi_s = pi[order]
    first = np.empty(len(pi_s), dtype=bool)
    first[0] = True
    if len(pi_s) > 1:
        first[1:] = pi_s[1:] != pi_s[:-1]
    win = order[first]
    out[m[pi[win]]] = idx_arr[gi[win]]
    return out
