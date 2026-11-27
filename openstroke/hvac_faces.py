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
    # F245 (v7): 11 physical CAD topology & all-stroke interior-symbol descriptors (0 PDF layers)
    "nb_edge_overhang_max", "nb_edge_overhang_min", "end_caps_touched",
    "wide_end_overhang", "narrow_end_touched", "coaxial_ends",
    "nb_max_short_ratio", "nb_min_short_ratio",
    "all_ink_in_ratio", "has_v_crease", "free_end_unpicked_log",
    # F249 (v9): 4 pure per-face geometric interaction descriptors (zero page-constant features)
    "taper_strength", "ink_density_ratio", "nb_short_span_ratio", "hull_rect_gap",
)
FEATURES_41 = FEATURES[:41]
FEATURES_30 = FEATURES[:30]
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
END_MIN = 0.25   # pt: short dash stubs at T-junctions/collars (0.78-0.96 pt on MB TH4) still have open ends (F234; was 1.0)
CORNER_MAX, CORNER_COS = 4.5, 0.85   # pt along each ray to the meeting point; |cos| <= 0.85 covers 30-60 deg dashed reducer/branch corners (F234)
RAY_MAX = 3.5   # pt: forward ray T-extension from remaining open ends to meet a perpendicular wall/corner (closes dashed T-walls, F234)


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


def _end_dir(xy: np.ndarray, at_start: bool, min_chord: float = 1.0) -> np.ndarray | None:
    """Robust outward unit direction at chain endpoint: walk inward up to ``min_chord`` skipping < 0.25 pt CAD micro-segments
    without crossing sharp (> ~30 deg) corners on merged polylines (F234)."""
    pts = xy if at_start else xy[::-1]
    a = pts[0]
    ref_u: np.ndarray | None = None
    best_u: np.ndarray | None = None
    for k in range(1, len(pts)):
        sv = pts[k - 1] - pts[k]
        sn = float(np.hypot(sv[0], sv[1]))
        if sn <= 1e-9:
            continue
        su = sv / sn
        if ref_u is None:
            if sn >= 0.25:
                ref_u = su
        elif float(su @ ref_u) < 0.85:
            break   # do not cross a sharp corner (e.g. 0.8 pt dash stub merged with a 90-deg end wall)
        d = a - pts[k]
        n = float(np.hypot(d[0], d[1]))
        if n > 1e-9:
            best_u = d / n
            if n >= min_chord and ref_u is not None:
                return best_u
    return best_u


def _caps(chains, curved_chain: np.ndarray):
    """Virtual closing segments for open ends (F153 / F166 / F234 A5 / F245 v7):
      1. Collinear & circular-arc curved dash bridges (twin-dash safe, up to 12.5 pt curved with arc symmetry + crossing guard);
      2. Curved elbow & strict parallel straight open-ended duct virtual end caps;
      3. Angled corner ray intersections (up to CORNER_MAX, |cos| <= CORNER_COS for 30-60 deg dashed fittings);
      4. Forward ray T-extensions (up to RAY_MAX) from remaining open ends onto a perpendicular wall/corner."""
    import shapely
    from scipy.spatial import cKDTree

    chains, curved_chain = _merged_chains(chains, curved_chain)
    if not len(chains):
        return []
    clens = shapely.length(chains)
    pts, dirs, owner, clen_ep = [], [], [], []
    for i in np.flatnonzero(clens >= END_MIN):
        xy = np.asarray(chains[i].coords)
        if len(xy) < 2:
            continue
        for at_start in (True, False):
            a = xy[0] if at_start else xy[-1]
            u = _end_dir(xy, at_start, min_chord=1.0)
            if u is not None:
                pts.append(a); dirs.append(u); owner.append(int(i)); clen_ep.append(float(clens[i]))
    if len(pts) < 2:
        return []
    pts, dirs = np.asarray(pts), np.asarray(dirs)
    owner, clen_ep = np.asarray(owner, dtype=np.int64), np.asarray(clen_ep, dtype=np.float64)

    # Only ends touching no other stroke are open; for short stubs (< 2 pt) scale snap so the stub's own root wall
    # never marks its free tip touched, ignore parallel twin dashes ending at the same gap (F245 v7), and track
    # short dashes only touched by an orthogonal crossing wall (F234).
    tree = shapely.STRtree(chains)
    pt_geoms = shapely.points(pts)
    pi, li = tree.query(pt_geoms, predicate="dwithin", distance=SNAP)
    touched = np.zeros(len(pts), bool)
    touched_ortho = np.zeros(len(pts), bool)
    hits_by_p: dict[int, list[tuple[int, float]]] = {}
    if len(pi):
        keep_h = li != owner[pi]
        pi_h, li_h = pi[keep_h], li[keep_h]
        if len(pi_h):
            dists = shapely.distance(pt_geoms[pi_h], chains[li_h])
            eff = np.where(clen_ep[pi_h] < 2.0, np.minimum(SNAP, 0.45 * clen_ep[pi_h]), SNAP)
            ok_d = dists <= eff
            for p_i, l_i, dv in zip(pi_h[ok_d].tolist(), li_h[ok_d].tolist(), dists[ok_d].tolist()):
                hits_by_p.setdefault(p_i, []).append((l_i, float(dv)))
    chain_xy_cache: dict[int, np.ndarray] = {}
    for p_i, hlist in hits_by_p.items():
        real_touch = False
        all_ortho = True
        for l_i, dv in hlist:
            # Ignore tiny < 0.35 pt perpendicular CAD tick marks / corner dots attached to dash tips (2D.pdf / vw_27a3)
            if clens[l_i] < 0.35:
                continue
            xy_l = chain_xy_cache.get(l_i)
            if xy_l is None:
                xy_l = chain_xy_cache[l_i] = np.asarray(chains[l_i].coords)
            if len(xy_l) >= 2:
                sa, sb = xy_l[:-1], xy_l[1:]
                mid = 0.5 * (sa + sb)
                kn = int(np.argmin(np.hypot(mid[:, 0] - pts[p_i, 0], mid[:, 1] - pts[p_i, 1])))
                vl = sb[kn] - sa[kn]
                nl = float(np.hypot(vl[0], vl[1]))
                dp = abs(float(dirs[p_i] @ (vl / nl))) if nl > 1e-6 else 0.0
                # Parallel twin offset by 0.05..SNAP stopping at the same gap, OR a collinear dash separated by > 0.25 pt:
                if dp >= 0.85 and dv > 0.05:
                    fwd = float(np.max((xy_l - pts[p_i]) @ dirs[p_i]))
                    if fwd <= 1.0:
                        continue
                    if dv > 0.25 and abs(float(-dirs[p_i, 1] * (xy_l[0, 0] - pts[p_i, 0]) + dirs[p_i, 0] * (xy_l[0, 1] - pts[p_i, 1]))) <= 0.15:
                        continue
                real_touch = True
                if dp > 0.35:
                    all_ortho = False
            else:
                real_touch = True
                all_ortho = False
        if real_touch:
            touched[p_i] = True
            if all_ortho:
                touched_ortho[p_i] = True

    open_idx = np.flatnonzero(~touched)
    bridge_idx = np.flatnonzero((~touched) | (touched_ortho & (clen_ep <= 16.0)))
    if len(bridge_idx) < 2 and len(open_idx) < 2:
        return []

    used: set[int] = set()
    out = []

    # 1. Collinear & smooth circular-arc curved dash bridges (F158 / F234 / F245 v7)
    curved_dash_max = 12.5
    bridges = []
    if len(bridge_idx) >= 2:
        btree = cKDTree(pts[bridge_idx])
        for ia, ib in btree.query_pairs(max(BRIDGE_MAX, curved_dash_max)):
            a, b = int(bridge_idx[ia]), int(bridge_idx[ib])
            if owner[a] == owner[b]:
                continue
            if touched[a] and touched[b] and (clen_ep[a] > 10.0 or clen_ep[b] > 10.0):
                continue
            v = pts[b] - pts[a]
            dist = float(np.hypot(v[0], v[1]))
            if dist < 1e-6:
                continue
            u = v / dist
            cos_ab = float(dirs[a] @ dirs[b])
            ca = float(u @ dirs[a])
            cb = -float(u @ dirs[b])
            if dist <= BRIDGE_MAX and cos_ab <= -CAP_PAR and ca >= CAP_PAR and cb >= CAP_PAR:
                perp_a = abs(float(v[0] * (-dirs[a, 1]) + v[1] * dirs[a, 0]))
                perp_b = abs(float(v[0] * (-dirs[b, 1]) + v[1] * dirs[b, 0]))
                if max(perp_a, perp_b) > max(0.8, 0.25 * dist) and dist > 3.0:
                    continue
                bridges.append((dist, a, b))
            elif (dist <= curved_dash_max
                  and (not (touched[a] and touched[b]))
                  and (clen_ep[a] <= 16.0 or curved_chain[owner[a]])
                  and (clen_ep[b] <= 16.0 or curved_chain[owner[b]])):
                if dist <= 4.5:
                    ok_ang = (cos_ab <= -0.60 and ca >= 0.78 and cb >= 0.78 and (not touched[a]) and (not touched[b]))
                else:
                    # Circular-arc chord symmetry: dirs[a] turns toward u by same angle as u turns toward -dirs[b]
                    cross_a = float(dirs[a, 0] * u[1] - dirs[a, 1] * u[0])
                    cross_b = float(u[0] * (-dirs[b, 1]) - u[1] * (-dirs[b, 0]))
                    ok_ang = (
                        cos_ab <= -0.50 and ca >= 0.76 and cb >= 0.76
                        and (cross_a * cross_b >= -0.02)
                        and (abs(cross_a - cross_b) <= 0.25)
                    )
                if ok_ang:
                    bridges.append((dist + 0.6, a, b))
    for _, a, b in sorted(bridges):
        if a in used or b in used:
            continue
        seg_ab = shapely.LineString([pts[a], pts[b]])
        if float(seg_ab.length) > 6.0:
            hits_c = tree.query(seg_ab, predicate="crosses")
            if any(int(c) != owner[a] and int(c) != owner[b] for c in hits_c):
                continue
        used |= {a, b}
        out.append(seg_ab)

    # 2. Curved elbow + strict parallel straight open-ended duct virtual end caps (F153 / F157 / F245 v7 / F255 v11)
    # Before sealing two parallel straight open ends with a virtual cap, check whether either end has a real wall/corner
    # dead-ahead within RAY_MAX along its forward ray (e.g. MB TH4 gt#16 2.28 pt gap before a Chân rẽ corner): if so,
    # skip the virtual cap so Step 4 (forward T-ray extension) connects the dashed wall straight into the corner!
    otree = cKDTree(pts[open_idx]) if len(open_idx) >= 2 else None
    has_fwd_wall: set[int] = set()
    cand = []
    if otree is not None:
        raw_str_ends: set[int] = set()
        raw_cand = []
        for ia, ib in otree.query_pairs(CAP_MAX):
            a, b = int(open_idx[ia]), int(open_idx[ib])
            if a in used or b in used or owner[a] == owner[b]:
                continue
            is_cur = bool(curved_chain[owner[a]] or curved_chain[owner[b]])
            v = pts[b] - pts[a]
            dist = float(np.hypot(v[0], v[1]))
            if dist < 1.0:
                continue
            u = v / dist
            if is_cur:
                if float(dirs[a] @ dirs[b]) < CAP_PAR:
                    continue
                if abs(float(u @ dirs[a])) > CAP_PERP or abs(float(u @ dirs[b])) > CAP_PERP:
                    continue
                raw_cand.append((dist, a, b, False))
            else:
                # Strict parallel straight open-ended duct virtual cap (walls >= 6 pt, width 3..32 pt, ultra-parallel & square)
                if not (3.0 <= dist <= 32.0 and clen_ep[a] >= 6.0 and clen_ep[b] >= 6.0):
                    continue
                if float(dirs[a] @ dirs[b]) < 0.985:
                    continue
                if abs(float(u @ dirs[a])) > 0.12 or abs(float(u @ dirs[b])) > 0.12:
                    continue
                raw_cand.append((dist + 2.0, a, b, True))
                raw_str_ends.add(a)
                raw_str_ends.add(b)
        if raw_str_ends and RAY_MAX > 0:
            se = np.asarray(sorted(raw_str_ends), dtype=np.int64)
            sp0, sd0 = pts[se], dirs[se]
            srays = shapely.linestrings(np.stack([sp0, sp0 + RAY_MAX * sd0], axis=1))
            sri, sci = tree.query(srays, predicate="intersects")
            if len(sri):
                ok_s = sci != owner[se[sri]]
                for rl in np.unique(sri[ok_s]).tolist():
                    has_fwd_wall.add(int(se[rl]))
        for item in raw_cand:
            if item[3] and (item[1] in has_fwd_wall or item[2] in has_fwd_wall):
                continue
            cand.append(item)
    for _, a, b, is_str in sorted(cand):
        if a in used or b in used:
            continue
        seg_ab = shapely.LineString([pts[a], pts[b]])
        if is_str:
            hits_c = tree.query(seg_ab, predicate="crosses")
            if any(int(c) != owner[a] and int(c) != owner[b] for c in hits_c):
                continue
        used |= {a, b}
        out.append(seg_ab)

    # 3. Corners (F166 / F234): two open ends whose forward rays meet within CORNER_MAX
    corners = []
    if CORNER_MAX > 0 and otree is not None:
        for ia, ib in otree.query_pairs(2 * CORNER_MAX):
            a, b = int(open_idx[ia]), int(open_idx[ib])
            if a in used or b in used or owner[a] == owner[b]:
                continue
            if abs(float(dirs[a] @ dirs[b])) > CORNER_COS:
                continue
            M = np.stack([dirs[a], -dirs[b]], 1)
            if abs(float(np.linalg.det(M))) < 1e-4:
                continue
            st = np.linalg.solve(M, pts[b] - pts[a])          # pts[a] + s dirs[a] = pts[b] + t dirs[b]
            if (st >= -1e-6).all() and (st <= CORNER_MAX).all():
                corners.append((float(st.max()), a, b, pts[a] + st[0] * dirs[a]))
    for _, a, b, x in sorted(corners, key=lambda q: q[0]):
        if a in used or b in used:
            continue
        used |= {a, b}
        out += [shapely.LineString([pts[a], x]), shapely.LineString([pts[b], x])]

    # 4. Vectorized forward ray T-extension to nearest chain body/vertex (F234: closes dashed wall stopping 1.2-3.5 pt short of T-wall/corner)
    if RAY_MAX > 0 and len(open_idx):
        rem = np.asarray([int(a) for a in open_idx if int(a) not in used], dtype=np.int64)
        if len(rem):
            p0s = pts[rem]
            d0s = dirs[rem]
            ray_xy = np.stack([p0s, p0s + RAY_MAX * d0s], axis=1)
            rays = shapely.linestrings(ray_xy)
            ri, ci = tree.query(rays, predicate="intersects")
            if len(ri):
                ok_rc = ci != owner[rem[ri]]
                ri, ci = ri[ok_rc], ci[ok_rc]
                if len(ri):
                    inters = shapely.intersection(rays[ri], chains[ci])
                    best_s = np.full(len(rem), RAY_MAX + 1.0, dtype=np.float64)
                    best_pt = np.empty((len(rem), 2), dtype=np.float64)
                    has_hit = np.zeros(len(rem), dtype=bool)
                    for r_loc, inter in zip(ri.tolist(), inters):
                        if inter.is_empty:
                            continue
                        coords = shapely.get_coordinates(inter)
                        if not len(coords):
                            continue
                        p0 = p0s[r_loc]
                        da = d0s[r_loc]
                        dv = coords - p0
                        s_vals = dv[:, 0] * da[0] + dv[:, 1] * da[1]
                        valid = s_vals > 0.1
                        if valid.any():
                            k_min = int(np.argmin(np.where(valid, s_vals, 1e9)))
                            s_min = float(s_vals[k_min])
                            if s_min < best_s[r_loc]:
                                best_s[r_loc] = s_min
                                best_pt[r_loc] = coords[k_min]
                                has_hit[r_loc] = True
                    for r_loc in np.flatnonzero(has_hit).tolist():
                        out.append(shapely.LineString([p0s[r_loc], best_pt[r_loc]]))
                        used.add(int(rem[r_loc]))
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
    seg_all = segs.seg
    out = []
    for s in np.flatnonzero(picked):
        a, b = int(off[s]), int(off[s + 1])
        m = b - a
        if m < 3:
            continue
        sub = seg_all[a:b]
        # Fast path for simple single-ring closed CAD items (including curved elbows): avoids cKDTree + STRtree + polygonize
        if (np.abs(sub[1:, :2] - sub[:-1, 2:]).max() <= 1e-3
                and float(np.hypot(*(sub[0, :2] - sub[-1, 2:]))) <= 1e-3):
            g = Polygon(sub[:, :2])
            if g.is_valid and g.area > MIN_AREA:
                out.append((g, int(s), 1))
                continue
        faces = _polygonize_segments(sub)
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


def priority_candidates(
    segs,
    picked: np.ndarray,
    curved: np.ndarray | None = None,
    *,
    ep_snap: float = 0.25,
    max_halfedge_steps: int = 512,
) -> tuple[list, list, list]:
    """F254 (v10): Endpoint-Topology overlap-safe candidate generator (ZERO ``shapely.polygonize``, ZERO ``unary_union``).

    1. Decomposes picked strokes into elementary chains (arcs stay 1 edge) + open-end virtual ``_caps``.
    2. Extracts single-item closed rings (``item_faces``) and two-tier non-planar endpoint-connectivity cycles
       (Tier-1 pure endpoint cycles + Tier-2 strictly non-passthrough T-junction cycles, ignoring interior X-crossings).
    3. Fuses candidates with boundary-stroke-aware deduplication (only suppresses overlapping polygons sharing >= 30%
       exterior boundary, 100% preserving crossing ducts and twin overlapping elbows), then merges slivers and
       square-to-round symbol triangles.
    -> ``(regions, source item per region (-1 = across items), inner regions merged per region)``."""
    from collections import defaultdict

    import shapely

    from .hvac_cycles import elementary_chains, extract_t1_and_t2_cycles

    m = picked[segs.owner]
    if not m.any():
        return [], [], []
    own = segs.owner[m]
    sseg = segs.seg[m]

    chains = elementary_chains(sseg, own, curved_shape=curved)
    if curved is not None:
        raw_chains = _chains(sseg, own)
        brk = np.r_[True, (own[1:] != own[:-1]) | (np.abs(sseg[1:, :2] - sseg[:-1, 2:]).max(1) > 1e-6)]
        caps = _caps(raw_chains, curved[own[np.flatnonzero(brk)]])
        for c in caps:
            xy = np.asarray(c.coords, dtype=np.float64)
            if len(xy) >= 2:
                chains.append((xy, -1, False))

    # 1. Single-item closed shapes
    items = sorted(item_faces(segs, picked), key=lambda q: q[0].area)
    uniq, src, inner = [], [], []
    if items:
        it_arr = np.asarray([q[0] for q in items], dtype=object)
        it_areas = shapely.area(it_arr)
        itree = shapely.STRtree(it_arr)
        ai, bi = itree.query(it_arr, predicate="intersects")
        ord_m = bi < ai
        ai_f, bi_f = ai[ord_m], bi[ord_m]
        prev_map: dict[int, list[tuple[int, float]]] = defaultdict(list)
        if len(ai_f):
            ia_all = shapely.area(shapely.intersection(it_arr[ai_f], it_arr[bi_f]))
            for a_i, b_i, ia_v in zip(ai_f.tolist(), bi_f.tolist(), ia_all.tolist()):
                prev_map[a_i].append((b_i, ia_v))
        accepted = np.zeros(len(items), dtype=bool)
        for i, (g, it, n_in) in enumerate(items):
            hits = prev_map.get(i)
            if hits:
                ai_area = float(it_areas[i])
                if any(accepted[j] and ia_v / max(float(it_areas[j]) + ai_area - ia_v, 1e-9) >= SAME_IOU for j, ia_v in hits):
                    continue
            accepted[i] = True
            uniq.append(g)
            src.append(it)
            inner.append(n_in)

    # 2. Tier-1 + Tier-2 non-planar endpoint-connectivity cycles (shared Step-0 snapping, ignores X-crossings!)
    t1_faces, t2_faces = extract_t1_and_t2_cycles(
        chains, ep_snap=ep_snap, free_snap=SNAP, min_area=MIN_AREA, max_halfedge_steps=max_halfedge_steps
    )
    graph_pool = t2_faces + t1_faces
    if graph_pool:
        n_t2 = len(t2_faces)
        gp_arr = np.asarray(graph_pool, dtype=object)
        gp_areas = shapely.area(gp_arr)
        gtree = shapely.STRtree(gp_arr)
        gi, gj = gtree.query(gp_arr, predicate="intersects")
        mk = gi < gj
        gi, gj = gi[mk], gj[mk]
        drop_g: set[int] = set()
        t1_children: dict[int, list[tuple[int, float]]] = defaultdict(list)
        if len(gi):
            ia_v = shapely.area(shapely.intersection(gp_arr[gi], gp_arr[gj]))
            sa_v, sb_v = gp_areas[gi], gp_areas[gj]
            iou_v = ia_v / np.maximum(sa_v + sb_v - ia_v, 1e-9)
            same_m = iou_v >= SAME_IOU
            if same_m.any():
                drop_g.update(gj[same_m].tolist())
            cand_a = (~same_m) & (gi < n_t2) & (gj >= n_t2) & (ia_v >= 0.85 * sa_v) & (sb_v >= 1.25 * sa_v)
            cand_b = (~same_m) & (gj < n_t2) & (gi >= n_t2) & (ia_v >= 0.85 * sb_v) & (sa_v >= 1.25 * sb_v)
            if cand_a.any() or cand_b.any():
                ch_idx = np.concatenate([gi[cand_a], gj[cand_b]])
                pa_idx = np.concatenate([gj[cand_a], gi[cand_b]])
                ia_c = np.concatenate([ia_v[cand_a], ia_v[cand_b]])
                ch_exts = shapely.get_exterior_ring(gp_arr[ch_idx])
                pa_ebufs = shapely.buffer(shapely.get_exterior_ring(gp_arr[pa_idx]), 0.8, quad_segs=4)
                sh_frac = shapely.length(shapely.intersection(ch_exts, pa_ebufs)) / np.maximum(shapely.length(ch_exts), 1e-9)
                ok_ch = sh_frac >= 0.30
                for c_i, p_i, ia_0 in zip(ch_idx[ok_ch].tolist(), pa_idx[ok_ch].tolist(), ia_c[ok_ch].tolist()):
                    t1_children[p_i].append((c_i, ia_0))
        for p_idx, ch in t1_children.items():
            if p_idx in drop_g:
                continue
            valid_ch = [(c, ia) for c, ia in ch if c not in drop_g]
            if len(valid_ch) >= 2 and sum(ia for _, ia in valid_ch) >= 0.78 * float(gp_areas[p_idx]):
                drop_g.add(p_idx)
        gfaces = [f for idx, f in enumerate(graph_pool) if idx not in drop_g]
    else:
        gfaces = []

    # 3. Vectorized boundary-stroke-aware fusion with item_faces (never drops crossing objects!)
    if uniq and gfaces:
        u_arr = np.asarray(uniq, dtype=object)
        u_areas = shapely.area(u_arr)
        gf_arr = np.asarray(gfaces, dtype=object)
        gf_areas = shapely.area(gf_arr)
        tree = shapely.STRtree(u_arr)
        fi, ui = tree.query(gf_arr, predicate="intersects")
        near_map: dict[int, list[tuple[int, float]]] = defaultdict(list)
        inside_of_u: dict[int, list[int]] = defaultdict(list)
        sub_check_pairs: list[tuple[int, int]] = []
        if len(fi):
            ia_all = shapely.area(shapely.intersection(gf_arr[fi], u_arr[ui]))
            for f_idx, u_idx, ia_v in zip(fi.tolist(), ui.tolist(), ia_all.tolist()):
                near_map[f_idx].append((u_idx, ia_v))
                if ia_v >= INSIDE * float(gf_areas[f_idx]):
                    inside_of_u[u_idx].append(f_idx)
                if ia_v >= 0.82 * float(gf_areas[f_idx]):
                    sub_check_pairs.append((f_idx, u_idx))
        multi_u = set()
        for u_idx, f_list in inside_of_u.items():
            if inner[u_idx] == 1 and len(f_list) >= 2 and _corners(uniq[u_idx]) > 4:
                if sum(float(gf_areas[fi0]) for fi0 in f_list) >= 0.72 * float(u_areas[u_idx]):
                    multi_u.add(u_idx)
        sub_drop: set[int] = set()
        filt_pairs = [(fi0, ui0) for fi0, ui0 in sub_check_pairs if ui0 not in multi_u]
        if filt_pairs:
            p_fi = np.array([q[0] for q in filt_pairs], dtype=np.int64)
            p_ui = np.array([q[1] for q in filt_pairs], dtype=np.int64)
            f_ex = shapely.get_exterior_ring(gf_arr[p_fi])
            u_eb = shapely.buffer(shapely.get_exterior_ring(u_arr[p_ui]), 0.8, quad_segs=4)
            sh_r = shapely.length(shapely.intersection(f_ex, u_eb)) / np.maximum(shapely.length(f_ex), 1e-9)
            sub_drop.update(p_fi[sh_r >= 0.30].tolist())
        for i, f in enumerate(gfaces):
            if i in sub_drop:
                continue
            hits = near_map.get(i)
            if hits:
                fa = float(gf_areas[i])
                if any(ia_v / max(fa + float(u_areas[j]) - ia_v, 1e-9) >= SAME_IOU for j, ia_v in hits):
                    continue
            uniq.append(f)
            src.append(-1)
            inner.append(1)
    elif gfaces:
        for f in gfaces:
            uniq.append(f)
            src.append(-1)
            inner.append(1)

    merged, keep = merge_slivers(uniq, return_index=True)
    m_src = [src[i] for i in keep]
    m_inn = [inner[i] for i in keep]
    # F255 (v11): Absorb 3-corner inner V-crease triangles sitting >= 80% inside a larger 4..5-corner fitting
    # (even when drawn as two separate closed PDF items, e.g. Chân rẽ vuông tròn with inner V-triangle!),
    # incrementing inner[parent] so the outer fitting retains its square-to-round symbol tag cleanly.
    # Never absorb 2 symmetric diagonal halves of a mitered Elbow vuông square (each >= 35 pt², covering >= 85% of a square)!
    if len(merged) >= 2:
        m_arr = np.asarray(merged, dtype=object)
        m_areas = shapely.area(m_arr)
        m_crns = np.array([len(v) for v in _bulk_ring_coords(shapely.simplify(m_arr, 0.5))], dtype=np.int64)
        tri_idx = np.flatnonzero(m_crns == 3)
        if len(tri_idx):
            mtr = shapely.STRtree(m_arr)
            ti_loc, pj = mtr.query(m_arr[tri_idx], predicate="intersects")
            tris_in_parent: dict[int, list[tuple[int, float]]] = defaultdict(list)
            for loc_k, p_i in zip(ti_loc.tolist(), pj.tolist()):
                t_i = int(tri_idx[loc_k])
                if t_i == p_i:
                    continue
                if m_crns[p_i] not in (4, 5) or m_areas[p_i] <= 1.15 * m_areas[t_i]:
                    continue
                ia_t = float(m_arr[t_i].intersection(m_arr[p_i]).area)
                if ia_t >= 0.80 * float(m_areas[t_i]):
                    tris_in_parent[p_i].append((t_i, ia_t))
            drop_tri: set[int] = set()
            for p_i, tlist in tris_in_parent.items():
                # Protect mitered / multi-vane Elbow vuông squares (area >= 65 pt², near-square aspect <= 1.30):
                # inner V-crease triangles of Chân rẽ vuông tròn never sit inside large >= 65 pt² square mitered elbows!
                if float(m_areas[p_i]) >= 65.0:
                    ls_p, ss_p, _ = _mrr_axes(m_arr[p_i])
                    rfill_p = float(m_areas[p_i]) / max(ls_p * ss_p, 1e-9)
                    if rfill_p >= 0.85 and ls_p / max(ss_p, 1e-6) <= 1.30:
                        continue
                for t_i, _ in tlist:
                    drop_tri.add(t_i)
                m_inn[p_i] = max(m_inn[p_i] + len(tlist), 2)
            if drop_tri:
                merged = [f for idx_m, f in enumerate(merged) if idx_m not in drop_tri]
                m_src = [s for idx_m, s in enumerate(m_src) if idx_m not in drop_tri]
                m_inn = [v for idx_m, v in enumerate(m_inn) if idx_m not in drop_tri]
    return _merge_symbol_triangles(merged, m_src, m_inn)


def _merge_symbol_triangles(faces: list, src: list[int], inner: list[int]) -> tuple[list, list[int], list[int]]:
    """F237 / F241 / F245 (v7): General cross-item counterpart of ``item_faces`` inner-triangle merge (F158).

    Handles real CAD square-to-round (``vuông tròn``) transitions composed of a trapezoid + inscribed triangle(s):
      - Pass A: connected cluster of >= 2 small triangles (+ central trapezoid/quad sharing >= 50% perimeter OR
        touching >= 2 side triangles with >= 35% perimeter) whose union is a clean convex trapezoid/rectangle
        (``hull_fill >= 0.94``, ``rect_fill >= 0.62``, ``_corners in (4, 5)``, ``par_gap <= 10 deg``).
      - Pass B: strict 1 triangle + 1 compact trapezoid/quad sharing a common edge (``area <= 150``, ``_corners == 4``,
        ``hull_fill >= 0.97``, ``rect_fill >= 0.70``, ``par_gap <= 6 deg``).
    """
    import shapely
    from shapely.geometry import Polygon

    if len(faces) < 2:
        return faces, src, inner
    arr = np.asarray(faces, dtype=object)
    areas = shapely.area(arr)
    simps = shapely.simplify(arr, 0.5)
    crns = np.array([len(v) for v in _bulk_ring_coords(simps)], dtype=np.int64)
    is_tri = (crns == 3) & (areas <= 120.0)
    if int(is_tri.sum()) < 1:
        return faces, src, inner

    cand_m = ((crns == 3) & (areas <= 120.0)) | ((crns == 4) & (areas <= 160.0))
    cand_idx = np.flatnonzero(cand_m)
    if len(cand_idx) < 2:
        return faces, src, inner
    c_arr = arr[cand_idx]
    c_bufs = shapely.buffer(c_arr, 0.35, quad_segs=4)
    c_exts = shapely.get_exterior_ring(c_arr)
    c_perim = np.maximum(shapely.length(c_exts), 1e-9)
    tr = shapely.STRtree(c_arr)
    ai, bi = tr.query(c_bufs, predicate="intersects")
    mk = ai != bi
    ai, bi = ai[mk], bi[mk]
    if not len(ai):
        return faces, src, inner
    sl = shapely.length(shapely.intersection(c_exts[ai], c_bufs[bi]))
    good = sl >= 1.0
    ai, bi, sl = ai[good], bi[good], sl[good]
    c_is_tri = is_tri[cand_idx]

    quad_tri_sh = np.zeros(len(cand_idx), dtype=np.float64)
    quad_tri_cnt = np.zeros(len(cand_idx), dtype=np.int64)
    for u_i, v_i, s_v in zip(ai.tolist(), bi.tolist(), sl.tolist()):
        if (not c_is_tri[u_i]) and c_is_tri[v_i]:
            quad_tri_sh[u_i] += s_v
            quad_tri_cnt[u_i] += 1
    valid_node = c_is_tri | (quad_tri_sh >= 0.50 * c_perim) | ((quad_tri_cnt >= 2) & (quad_tri_sh >= 0.35 * c_perim))

    adj: dict[int, list[int]] = {}
    for u_i, v_i in zip(ai.tolist(), bi.tolist()):
        if valid_node[u_i] and valid_node[v_i] and (c_is_tri[u_i] or c_is_tri[v_i]):
            adj.setdefault(u_i, []).append(v_i)
            adj.setdefault(v_i, []).append(u_i)

    visited: set[int] = set()
    drop_set: set[int] = set()
    new_f: list = []
    new_s: list[int] = []
    new_in: list[int] = []
    # Pass A: >= 2 triangles (+ central trapezoid/quad)
    for u0 in np.flatnonzero(c_is_tri).tolist():
        if u0 in visited:
            continue
        comp: list[int] = []
        q = [u0]
        visited.add(u0)
        while q:
            cur_u = q.pop()
            comp.append(cur_u)
            for nb_u in adj.get(cur_u, ()):
                if nb_u not in visited:
                    visited.add(nb_u)
                    q.append(nb_u)
        if sum(1 for x in comp if c_is_tri[x]) < 2:
            continue
        orig_idxs = [int(cand_idx[x]) for x in comp]
        u = shapely.union_all([faces[k].buffer(0.08, join_style="mitre") for k in orig_idxs]).buffer(-0.08, join_style="mitre")
        if u.geom_type != "Polygon" or u.area <= MIN_AREA or u.area > 300.0:
            continue
        su = np.asarray(shapely.simplify(u, 0.5).exterior.coords)[:-1]
        nc = len(su)
        if nc not in (4, 5):
            continue
        hfill = float(u.area) / max(float(u.convex_hull.area), 1e-9)
        rfill = float(u.area) / max(float(u.minimum_rotated_rectangle.area), 1e-9)
        if hfill >= 0.94 and rfill >= 0.62 and (nc != 4 or _par_gap_deg(su) <= 10.0):
            drop_set.update(orig_idxs)
            new_f.append(Polygon(u.exterior))
            new_s.append(-1)
            new_in.append(len(orig_idxs))

    # Pass B: strict 1 triangle + 1 compact trapezoid/quad sharing a common edge
    used_p = set(drop_set)
    cands = []
    for u_i, v_i, s_v in zip(ai.tolist(), bi.tolist(), sl.tolist()):
        if not (c_is_tri[u_i] and (not c_is_tri[v_i])):
            continue
        oi_t, oi_q = int(cand_idx[u_i]), int(cand_idx[v_i])
        if oi_t in used_p or oi_q in used_p:
            continue
        if not (2.0 <= float(areas[oi_t]) <= 50.0 and float(areas[oi_q]) <= 90.0):
            continue
        if s_v < max(2.0, 0.30 * float(c_perim[u_i])):
            continue
        lq, sq, _ = _mrr_axes(faces[oi_q])
        if lq / max(sq, 1e-6) >= 1.8:
            continue
        u = shapely.union_all([faces[oi_t].buffer(0.08, join_style="mitre"), faces[oi_q].buffer(0.08, join_style="mitre")]).buffer(-0.08, join_style="mitre")
        if u.geom_type != "Polygon" or u.area > 150.0:
            continue
        su = np.asarray(shapely.simplify(u, 0.5).exterior.coords)[:-1]
        if len(su) == 4 and float(u.area) / max(float(u.convex_hull.area), 1e-9) >= 0.97 and _par_gap_deg(su) <= 6.0:
            rfill = float(u.area) / max(float(u.minimum_rotated_rectangle.area), 1e-9)
            if rfill >= 0.70:
                cands.append((rfill, oi_t, oi_q, u))
    for _, oi_t, oi_q, u in sorted(cands, key=lambda q: -q[0]):
        if oi_t in used_p or oi_q in used_p:
            continue
        used_p.add(oi_t); used_p.add(oi_q)
        drop_set.add(oi_t); drop_set.add(oi_q)
        new_f.append(Polygon(u.exterior)); new_s.append(-1); new_in.append(2)

    if not drop_set:
        return faces, src, inner
    out_f = [f for i, f in enumerate(faces) if i not in drop_set] + new_f
    out_s = [s for i, s in enumerate(src) if i not in drop_set] + new_s
    out_in = [v for i, v in enumerate(inner) if i not in drop_set] + new_in
    return out_f, out_s, out_in


def _curved_ink_faces(faces: list, src: list[int], inner: list[int], segs, picked: np.ndarray, curved: np.ndarray) -> tuple[list, list[int], list[int]]:
    """F237: Fast localized ink-hole fallback for unclosed curved Elbows missed by ``_polygonize_segments``.

    Runs ``page_faces(gap=GAP)`` ONLY on picked strokes within 35 pt of a picked curved stroke (avoiding full-page
    buffer cost) and appends only curved faces (>= 15% of perimeter on curves) that do not overlap existing candidates.
    """
    import shapely

    cur_picked = picked & curved
    if not cur_picked.any():
        return faces, src, inner
    m_pk = picked[segs.owner]
    if not m_pk.any():
        return faces, src, inner
    m_cur = cur_picked[segs.owner]
    if not m_cur.any():
        return faces, src, inner

    # Restrict ink buffer to shapes within 35 pt of any picked curved segment (fast numpy bbox pre-filter + STRtree)
    pk_seg = segs.seg[m_pk]
    pk_own = segs.owner[m_pk]
    cur_seg = segs.seg[m_cur]
    cx0 = float(min(cur_seg[:, 0].min(), cur_seg[:, 2].min())) - 36.0
    cx1 = float(max(cur_seg[:, 0].max(), cur_seg[:, 2].max())) + 36.0
    cy0 = float(min(cur_seg[:, 1].min(), cur_seg[:, 3].min())) - 36.0
    cy1 = float(max(cur_seg[:, 1].max(), cur_seg[:, 3].max())) + 36.0
    in_cbox = (
        (np.maximum(pk_seg[:, 0], pk_seg[:, 2]) >= cx0)
        & (np.minimum(pk_seg[:, 0], pk_seg[:, 2]) <= cx1)
        & (np.maximum(pk_seg[:, 1], pk_seg[:, 3]) >= cy0)
        & (np.minimum(pk_seg[:, 1], pk_seg[:, 3]) <= cy1)
    )
    if not in_cbox.any():
        return faces, src, inner
    pk_seg = pk_seg[in_cbox]
    pk_own = pk_own[in_cbox]
    pk_lines = shapely.linestrings(pk_seg.reshape(-1, 2, 2))
    cur_lines = shapely.linestrings(cur_seg.reshape(-1, 2, 2))
    ptree = shapely.STRtree(pk_lines)
    local_picked = np.zeros_like(picked, dtype=bool)
    for c0 in range(0, len(cur_lines), 512):
        _, hit_pk = ptree.query(cur_lines[c0:c0 + 512], predicate="dwithin", distance=35.0)
        if len(hit_pk):
            local_picked[pk_own[hit_pk]] = True
    if not local_picked.any():
        return faces, src, inner

    ink = page_faces(segs, local_picked, gap=GAP, curved=curved)
    if not ink:
        return faces, src, inner
    ctree = shapely.STRtree(cur_lines)
    ftree = shapely.STRtree(np.asarray(faces, dtype=object)) if faces else None
    added_f: list = []
    added_s: list[int] = []
    added_i: list[int] = []
    for f in ink:
        if f.area < 4.0 or f.area > 2500.0:
            continue
        band = f.exterior.buffer(GAP + 0.3, quad_segs=4)
        near_c = ctree.query(band, predicate="intersects")
        if not len(near_c):
            continue
        cl = float(shapely.length(shapely.intersection(cur_lines[near_c], band)).sum())
        if cl < 0.15 * max(f.exterior.length, 1e-9):
            continue
        dup = False
        if ftree is not None:
            for j in ftree.query(f, predicate="intersects"):
                ia = f.intersection(faces[int(j)]).area
                if ia >= 0.40 * min(f.area, faces[int(j)].area):
                    dup = True
                    break
        if not dup:
            added_f.append(f)
            added_s.append(-1)
            added_i.append(1)
    if not added_f:
        return faces, src, inner
    return list(faces) + added_f, list(src) + added_s, list(inner) + added_i


def _split_tangent_seams(faces: list, src: list[int], inner: list[int], segs, picked: np.ndarray, curved: np.ndarray) -> tuple[list, list[int], list[int]]:
    """F245 (v7): Split over-merged Duct+Elbow faces (missing CAD flange line) at straight <-> curved-arc tangent seams."""
    import shapely
    from shapely.geometry import LineString, Point

    if not faces:
        return faces, src, inner
    m = picked[segs.owner]
    if not m.any():
        return faces, src, inner
    cur_m = m & curved[segs.owner]
    if not cur_m.any():
        return faces, src, inner
    cur_lines = shapely.linestrings(segs.seg[cur_m].reshape(-1, 2, 2))
    ctree = shapely.STRtree(cur_lines)

    out_f, out_s, out_in = [], [], []
    for f, s_i, in_i in zip(faces, src, inner):
        fa = float(f.area)
        if fa < 60.0 or f.geom_type != "Polygon":
            out_f.append(f); out_s.append(s_i); out_in.append(in_i)
            continue
        if fa / max(float(f.convex_hull.area), 1e-9) >= 0.88:
            out_f.append(f); out_s.append(s_i); out_in.append(in_i)
            continue
        if not len(ctree.query(f.exterior.buffer(1.0, quad_segs=4), predicate="intersects")):
            out_f.append(f); out_s.append(s_i); out_in.append(in_i)
            continue
        coords = np.asarray(f.exterior.coords)[:-1]
        nv = len(coords)
        if nv < 8:
            out_f.append(f); out_s.append(s_i); out_in.append(in_i)
            continue
        ev = np.roll(coords, -1, 0) - coords
        el = np.hypot(ev[:, 0], ev[:, 1])
        keep_e = el > 1e-4
        if not keep_e.all():
            coords = coords[keep_e]
            nv = len(coords)
            if nv < 8:
                out_f.append(f); out_s.append(s_i); out_in.append(in_i)
                continue
            ev = np.roll(coords, -1, 0) - coords
            el = np.hypot(ev[:, 0], ev[:, 1])
        eu = ev / el[:, None]
        e_lines = shapely.linestrings(np.stack([coords, np.roll(coords, -1, 0)], axis=1))
        e_mid_turn = np.degrees(np.arccos(np.clip(np.sum(eu * np.roll(eu, 1, 0), axis=1), -1.0, 1.0)))
        e_next_turn = np.roll(e_mid_turn, -1)
        ei_h, _ = ctree.query(shapely.buffer(e_lines, 0.8, quad_segs=4), predicate="intersects")
        is_arc_e = np.zeros(nv, dtype=bool)
        if len(ei_h):
            is_arc_e[np.unique(ei_h)] = True
        is_arc_e |= (el <= 4.0) & (((e_mid_turn >= 2.0) & (e_mid_turn <= 35.0)) | ((e_next_turn >= 2.0) & (e_next_turn <= 35.0)))
        if not is_arc_e.any() or is_arc_e.all():
            out_f.append(f); out_s.append(s_i); out_in.append(in_i)
            continue

        str_run_len = np.zeros(nv, dtype=np.float64)
        for k in range(nv):
            if is_arc_e[k]:
                continue
            tot = float(el[k])
            j = (k - 1) % nv
            while j != k and (not is_arc_e[j]) and float(eu[j] @ eu[k]) >= 0.99:
                tot += float(el[j]); j = (j - 1) % nv
            j = (k + 1) % nv
            while j != k and (not is_arc_e[j]) and float(eu[j] @ eu[k]) >= 0.99:
                tot += float(el[j]); j = (j + 1) % nv
            str_run_len[k] = tot

        cuts = []
        ext = f.exterior
        for k in range(nv):
            prev_e = (k - 1) % nv
            if (not is_arc_e[prev_e]) and is_arc_e[k] and str_run_len[prev_e] >= 7.0 and e_mid_turn[k] <= 25.0:
                st_dir = eu[prev_e]
            elif is_arc_e[prev_e] and (not is_arc_e[k]) and str_run_len[k] >= 7.0 and e_mid_turn[k] <= 25.0:
                st_dir = eu[k]
            else:
                continue
            pt = coords[k]
            n1 = np.array([-st_dir[1], st_dir[0]])
            if f.contains(Point(pt[0] + 0.5 * n1[0], pt[1] + 0.5 * n1[1])):
                nin = n1
            elif f.contains(Point(pt[0] - 0.5 * n1[0], pt[1] - 0.5 * n1[1])):
                nin = -n1
            else:
                continue
            ray = LineString([pt + 0.05 * nin, pt + 55.0 * nin])
            inter = ray.intersection(ext)
            if inter.is_empty:
                continue
            ic = shapely.get_coordinates(inter)
            dv = ic - pt
            s_vals = dv[:, 0] * nin[0] + dv[:, 1] * nin[1]
            ok_s = (s_vals >= 2.5) & (s_vals <= 55.0)
            if not ok_s.any():
                continue
            s_min = float(s_vals[ok_s].min())
            hit_pt = pt + s_min * nin
            mid_pt = pt + 0.5 * s_min * nin
            if not f.contains(Point(mid_pt[0], mid_pt[1])):
                continue
            if any(float(np.hypot(pt[0] - cp[0], pt[1] - cp[1])) < 4.0 for cp, _ in cuts):
                continue
            cuts.append((pt, hit_pt))

        if not cuts:
            out_f.append(f); out_s.append(s_i); out_in.append(in_i)
            continue
        cut_lines = [
            LineString([a - 0.1 * (b - a) / max(np.hypot(*(b - a)), 1e-6),
                        b + 0.1 * (b - a) / max(np.hypot(*(b - a)), 1e-6)])
            for a, b in cuts
        ]
        net = shapely.unary_union([ext, *cut_lines])
        parts = [
            p.buffer(0) for p in shapely.get_parts(shapely.polygonize(shapely.get_parts(net)))
            if p.geom_type == "Polygon" and p.area > MIN_AREA and f.intersection(p).area >= 0.85 * p.area
        ]
        if len(parts) >= 2 and abs(sum(p.area for p in parts) - fa) <= 0.05 * fa:
            for p in parts:
                out_f.append(p); out_s.append(-1); out_in.append(1)
        else:
            out_f.append(f); out_s.append(s_i); out_in.append(in_i)
    return out_f, out_s, out_in


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


def _cusps_from_verts(v: np.ndarray) -> int:
    if len(v) < 4:
        return 0
    # Signed shoelace area > 0 means CCW; flip CW rings to match Shapely exterior.is_ccw
    nv = np.roll(v, -1, 0)
    if float(np.sum(v[:, 0] * nv[:, 1] - nv[:, 0] * v[:, 1])) < 0.0:
        v = v[::-1]
    e = np.roll(v, -1, 0) - v
    ang = np.degrees(np.arctan2(e[:, 1], e[:, 0]))
    turn = ((ang - np.roll(ang, 1) + 180) % 360) - 180
    return int(np.sum(turn <= -CUSP_TURN))


def _bulk_shared_dirs(geoms: np.ndarray) -> list[np.ndarray | None]:
    """Vectorized equivalent of ``[_shared_dir(g) for g in geoms]`` via bulk Shapely C calls."""
    import shapely

    n = len(geoms)
    out: list[np.ndarray | None] = [None] * n
    if not n:
        return out
    parts, idx = shapely.get_parts(geoms, return_index=True)
    if not len(parts):
        return out
    tids = shapely.get_type_id(parts)
    is_line = (tids == 1) | (tids == 2)   # LineString (1) or LinearRing (2)
    if not is_line.any():
        return out
    lparts = parts[is_line]
    lidx = idx[is_line]
    lens = shapely.length(lparts)
    # Keep the longest linear part per parent geometry
    order = np.lexsort((-lens, lidx))
    lidx_s = lidx[order]
    first = np.r_[True, lidx_s[1:] != lidx_s[:-1]]
    win = order[first]
    best_idx = lidx[win]
    best_parts = lparts[win]
    best_lens = lens[win]
    ok = best_lens > 0.0
    if not ok.any():
        return out
    best_idx = best_idx[ok]
    best_parts = best_parts[ok]
    p0 = shapely.get_coordinates(shapely.get_point(best_parts, 0))
    p1 = shapely.get_coordinates(shapely.get_point(best_parts, -1))
    dv = p1 - p0
    dn = np.hypot(dv[:, 0], dv[:, 1])
    valid = dn > 1e-6
    for k_i, v_i, n_i in zip(best_idx[valid].tolist(), dv[valid], dn[valid].tolist()):
        out[k_i] = v_i / n_i
    return out


def _bulk_ring_coords(geoms: np.ndarray) -> list[np.ndarray]:
    """Exterior ring coordinates (without closing duplicate) for an array of Polygons in ONE C call."""
    import shapely

    n = len(geoms)
    rings = shapely.get_exterior_ring(geoms)
    coords, idx = shapely.get_coordinates(rings, return_index=True)
    out = [np.empty((0, 2), dtype=np.float64) for _ in range(n)]
    if not len(idx):
        return out
    starts = np.r_[0, np.flatnonzero(idx[1:] != idx[:-1]) + 1]
    ends = np.r_[starts[1:], len(idx)]
    for g_i, a_i, b_i in zip(idx[starts].tolist(), starts.tolist(), ends.tolist()):
        if b_i - a_i >= 2:
            out[g_i] = coords[a_i:b_i - 1]
    return out


def _covered_line_inter_len(lines: np.ndarray, line_lens: np.ndarray, polys: np.ndarray,
                            li: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Exact ``shapely.length(shapely.intersection(lines[li], polys[pi]))`` with fast-path for fully covered segments."""
    import shapely

    n = len(li)
    out = np.empty(n, dtype=np.float64)
    if not n:
        return out
    cov = shapely.covers(polys[pi], lines[li])
    if cov.any():
        out[cov] = line_lens[li[cov]]
    rem = ~cov
    if rem.any():
        out[rem] = shapely.length(shapely.intersection(lines[li[rem]], polys[pi[rem]]))
    return out


def face_features(faces: list, segs, picked: np.ndarray, curved: np.ndarray) -> np.ndarray:
    """Shape + neighbourhood-topology descriptors of every face (FEATURES)."""
    import shapely

    n_faces = len(faces)
    if not n_faces:
        return np.zeros((0, len(FEATURES)))
    faces_arr = np.asarray(faces, dtype=object)
    m = picked[segs.owner]
    m_seg = segs.seg[m]
    lines = shapely.linestrings(m_seg.reshape(-1, 2, 2))
    line_lens = np.hypot(m_seg[:, 2] - m_seg[:, 0], m_seg[:, 3] - m_seg[:, 1])
    own = segs.owner[m]
    is_curve = curved[own]
    ltree = shapely.STRtree(lines)
    ftree = shapely.STRtree(faces_arr)

    # Vectorized per-face geometry computations in Shapely C (quad_segs=4 is ~3x faster with <0.02 pt diff on 0.6-1.2 pt buffers)
    areas = shapely.area(faces_arr)
    mrrs = shapely.minimum_rotated_rectangle(faces_arr)
    mrr_verts = _bulk_ring_coords(mrrs)
    mrr_info: list[tuple[float, float, np.ndarray]] = []
    for rr in mrr_verts:
        if len(rr) < 4:
            mrr_info.append((0.0, 0.0, np.array([1.0, 0.0])))
            continue
        v0 = rr[1] - rr[0]
        v1 = rr[2] - rr[1]
        l0 = float(np.hypot(v0[0], v0[1]))
        l1 = float(np.hypot(v1[0], v1[1]))
        if l0 >= l1:
            mrr_info.append((l0, l1, v0 / max(l0, 1e-9)))
        else:
            mrr_info.append((l1, l0, v1 / max(l1, 1e-9)))
    hull_areas = shapely.area(shapely.convex_hull(faces_arr))
    bufs = shapely.buffer(faces_arr, 0.6, quad_segs=4)
    touch_bufs = shapely.buffer(faces_arr, GAP * 2, quad_segs=4)
    ext_rings = shapely.get_exterior_ring(faces_arr)
    rlens = np.maximum(shapely.length(ext_rings), 1e-9)
    bands = shapely.buffer(ext_rings, GAP + 0.3, quad_segs=4)
    simps = shapely.simplify(faces_arr, 0.5)
    simp_verts = _bulk_ring_coords(simps)

    # 1. Bulk outline stroke coverage, arc share, and unique stroke counts
    fi_r, li_r = ltree.query(ext_rings, predicate="dwithin", distance=GAP + 0.3)
    cov_sum = np.zeros(n_faces, dtype=np.float64)
    arc_num = np.zeros(n_faces, dtype=np.float64)
    n_strokes = np.zeros(n_faces, dtype=np.int64)
    if len(fi_r):
        cov_all = _covered_line_inter_len(lines, line_lens, bands, li_r, fi_r)
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
        inners = shapely.buffer(faces_arr[big], -GAP - 0.3, quad_segs=4)
        bi_in, li_in = ltree.query(inners, predicate="intersects")
        if len(bi_in):
            in_lens = _covered_line_inter_len(lines, line_lens, inners, li_in, bi_in)
            ink_in[big] = np.bincount(bi_in, weights=in_lens, minlength=len(big))

    # 3. Bulk touching faces & shared boundary intersections (single pass for 30 + 11 v7 CAD topology features)
    ti, tj = ftree.query(touch_bufs, predicate="intersects")
    keep_t = ti != tj
    ti, tj = ti[keep_t], tj[keep_t]
    touch_cnt = np.bincount(ti, minlength=n_faces) if len(ti) else np.zeros(n_faces, dtype=np.int64)

    nb_sh_lens: list[list[float]] = [[] for _ in range(n_faces)]
    nb_areas_l: list[list[float]] = [[] for _ in range(n_faces)]
    nb_v7: list[list[tuple[int, float, object, np.ndarray | None]]] = [[] for _ in range(n_faces)]
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
            ig_good = inters[good_nb]
            sdirs = _bulk_shared_dirs(ig_good)
            for i_f, j_f, sl, ig, sdir in zip(ti[good_nb].tolist(), tj[good_nb].tolist(), slens[good_nb].tolist(), ig_good, sdirs):
                nb_sh_lens[i_f].append(sl)
                nb_areas_l[i_f].append(float(areas[j_f]))
                nb_v7[i_f].append((j_f, float(sl), ig, sdir))
                ax_i = mrr_info[i_f][2]
                lj, sj, ax_j = mrr_info[j_f]
                dot_ax = abs(float(ax_i[0] * ax_j[0] + ax_i[1] * ax_j[1]))
                if dot_ax >= 0.85:
                    collinear_a[i_f] += 1
                elif dot_ax <= 0.35:
                    ortho_a[i_f] += 1
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

    # 4. F245 (v7): 11 physical CAD topology & all-stroke interior-symbol descriptors (fast pure-NumPy + grid pre-filter)
    extra = np.zeros((n_faces, 11), dtype=np.float64)
    cents = shapely.get_coordinates(shapely.centroid(faces_arr))
    short_arr = np.array([q[1] for q in mrr_info], dtype=np.float64)
    edge_info: list[list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]] = []
    for v in simp_verts:
        elst = []
        if len(v) >= 3:
            for k in range(len(v)):
                a, b = v[k], v[(k + 1) % len(v)]
                dv = b - a
                dl = float(np.hypot(dv[0], dv[1]))
                if dl > 1e-6:
                    elst.append((a, b, dv / dl, dl))
        edge_info.append(elst)

    def _nb_collinear_edge_len(j_idx: int, seg_a: np.ndarray, seg_b: np.ndarray, seg_u: np.ndarray, tol: float = 0.85) -> float:
        """Pure-NumPy collinear segment-overlap check (~200x faster than Shapely buffer/intersects in Python loops)."""
        ux, uy = float(seg_u[0]), float(seg_u[1])
        ax0, ay0 = float(seg_a[0]), float(seg_a[1])
        s_end = (float(seg_b[0]) - ax0) * ux + (float(seg_b[1]) - ay0) * uy
        s_lo, s_hi = (0.0, s_end) if s_end >= 0.0 else (s_end, 0.0)
        best = 0.0
        for a, b, eu, dl in edge_info[j_idx]:
            if abs(ux * float(eu[0]) + uy * float(eu[1])) < 0.88:
                continue
            dx0, dy0 = float(a[0]) - ax0, float(a[1]) - ay0
            dx1, dy1 = float(b[0]) - ax0, float(b[1]) - ay0
            p0 = abs(-uy * dx0 + ux * dy0)
            p1 = abs(-uy * dx1 + ux * dy1)
            if min(p0, p1) > tol and (p0 > tol or p1 > tol):
                # Both endpoints outside perpendicular band: check if opposite sides (crossing)
                s0_p = -uy * dx0 + ux * dy0
                s1_p = -uy * dx1 + ux * dy1
                if s0_p * s1_p > 0.0:
                    continue
            t0 = dx0 * ux + dy0 * uy
            t1 = dx1 * ux + dy1 * uy
            if max(t0, t1) >= s_lo - tol and min(t0, t1) <= s_hi + tol:
                if dl > best:
                    best = dl
        return best

    def _edge_touch_len(j_idx: int, pa: np.ndarray, pb: np.ndarray, eu: np.ndarray, elen_val: float, tol: float = 0.75) -> float:
        """Pure-NumPy overlap length of neighbour j_idx's collinear boundary edges along edge (pa -> pb)."""
        ux, uy = float(eu[0]), float(eu[1])
        ax0, ay0 = float(pa[0]), float(pa[1])
        tot = 0.0
        for a, b, ju, dl in edge_info[j_idx]:
            if abs(ux * float(ju[0]) + uy * float(ju[1])) < 0.85:
                continue
            dx0, dy0 = float(a[0]) - ax0, float(a[1]) - ay0
            dx1, dy1 = float(b[0]) - ax0, float(b[1]) - ay0
            if min(abs(-uy * dx0 + ux * dy0), abs(-uy * dx1 + ux * dy1)) > tol:
                continue
            t0 = dx0 * ux + dy0 * uy
            t1 = dx1 * ux + dy1 * uy
            ov = min(elen_val, max(t0, t1)) - max(0.0, min(t0, t1))
            if ov > 0.0:
                tot += ov
        return tot

    for i in range(n_faces):
        nbs = nb_v7[i]
        my_s = max(short_arr[i], 1.0)
        if nbs:
            s_ratios = [short_arr[j] / my_s for j, _, _, _ in nbs]
            extra[i, 6] = min(max(s_ratios), 10.0)
            extra[i, 7] = min(min(s_ratios), 10.0)
            overhangs = []
            for j, sl, ig, sdir in nbs:
                if sdir is not None:
                    coords = shapely.get_coordinates(ig)
                    if len(coords) >= 2:
                        nb_el = _nb_collinear_edge_len(j, coords[0], coords[-1], sdir)
                        if nb_el > 0.0:
                            overhangs.append(nb_el / max(sl, 1.0))
            if overhangs:
                extra[i, 0] = min(max(overhangs), 20.0)
                extra[i, 1] = min(min(overhangs), 20.0)

        v = simp_verts[i]
        if len(v) == 4 and nbs:
            ev = np.roll(v, -1, 0) - v
            el = np.hypot(ev[:, 0], ev[:, 1])
            ang = np.degrees(np.arctan2(ev[:, 1], ev[:, 0]))
            def _eg(a_i, b_i):
                d = abs((ang[a_i] - ang[b_i]) % 180.0)
                return min(d, 180.0 - d)
            g02, g13 = _eg(0, 2), _eg(1, 3)
            if g02 <= 8.0 and g13 <= 8.0:
                par = (0, 2) if (el[0] + el[2]) <= (el[1] + el[3]) else (1, 3)
            else:
                par = (0, 2) if g02 <= g13 else (1, 3)
            wide_k, nar_k = (par[0], par[1]) if el[par[0]] >= el[par[1]] else (par[1], par[0])
            w_len = float(el[wide_k])
            n_len = float(el[nar_k])
            w_u = ev[wide_k] / max(w_len, 1e-6)
            n_u = ev[nar_k] / max(n_len, 1e-6)
            wa, wb = v[wide_k], v[(wide_k + 1) % 4]
            na, nb_p = v[nar_k], v[(nar_k + 1) % 4]
            w_nbs, n_nbs = [], []
            w_ov = 0.0
            for j, _, _, _ in nbs:
                wl_hit = _edge_touch_len(j, wa, wb, w_u, w_len)
                if wl_hit >= 0.45 * w_len:
                    nb_el = _nb_collinear_edge_len(j, wa, wb, w_u)
                    ov = nb_el / max(w_len, 1.0) if nb_el > 0.0 else 1.0
                    w_nbs.append((j, ov))
                    if ov > w_ov:
                        w_ov = ov
                nl_hit = _edge_touch_len(j, na, nb_p, n_u, n_len)
                if nl_hit >= 0.45 * n_len:
                    nb_el = _nb_collinear_edge_len(j, na, nb_p, n_u)
                    ov = nb_el / max(n_len, 1.0) if nb_el > 0.0 else 1.0
                    n_nbs.append((j, ov))
            extra[i, 2] = float((1 if w_nbs else 0) + (1 if n_nbs else 0))
            extra[i, 3] = min(w_ov, 20.0)
            extra[i, 4] = 1.0 if any(jn != jw for jw, _ in w_nbs for jn, _ in n_nbs) or (n_nbs and not w_nbs) else 0.0
            coax = 0.0
            ci = cents[i]
            for jw, ov_w in w_nbs:
                if ov_w > 1.38:
                    continue
                vw_c = cents[jw] - ci
                nw_c = float(np.hypot(vw_c[0], vw_c[1]))
                if nw_c < 1e-6:
                    continue
                vw_u = vw_c / nw_c
                for jn, ov_n in n_nbs:
                    if jn == jw or ov_n > 1.38:
                        continue
                    vn_c = cents[jn] - ci
                    nn_c = float(np.hypot(vn_c[0], vn_c[1]))
                    if nn_c < 1e-6:
                        continue
                    if float(vw_u @ (vn_c / nn_c)) < -0.25:
                        coax = 1.0
                        break
                if coax > 0.0:
                    break
            extra[i, 5] = coax

    if len(segs.seg):
        # Fast 2D occupancy-grid pre-filter in pure NumPy: skips ~95% of background strokes on dense 600k-seg A0 pages!
        all_seg = segs.seg
        fb = shapely.bounds(faces_arr)
        gx0 = float(fb[:, 0].min()) - 2.0
        gy0 = float(fb[:, 1].min()) - 2.0
        gx1 = float(fb[:, 2].max()) + 2.0
        gy1 = float(fb[:, 3].max()) + 2.0
        sx0 = np.minimum(all_seg[:, 0], all_seg[:, 2])
        sx1 = np.maximum(all_seg[:, 0], all_seg[:, 2])
        sy0 = np.minimum(all_seg[:, 1], all_seg[:, 3])
        sy1 = np.maximum(all_seg[:, 1], all_seg[:, 3])
        near_f = (sx1 >= gx0) & (sx0 <= gx1) & (sy1 >= gy0) & (sy0 <= gy1)
        if near_f.any():
            cell = 8.0
            nx = min(max(int((gx1 - gx0) / cell) + 2, 1), 4096)
            ny = min(max(int((gy1 - gy0) / cell) + 2, 1), 4096)
            inv_cx = nx / max(gx1 - gx0, 1.0)
            inv_cy = ny / max(gy1 - gy0, 1.0)
            occ = np.zeros((ny, nx), dtype=bool)
            fx0_c = np.clip(((fb[:, 0] - 2.0 - gx0) * inv_cx).astype(np.int64), 0, nx - 1)
            fx1_c = np.clip(((fb[:, 2] + 2.0 - gx0) * inv_cx).astype(np.int64), 0, nx - 1)
            fy0_c = np.clip(((fb[:, 1] - 2.0 - gy0) * inv_cy).astype(np.int64), 0, ny - 1)
            fy1_c = np.clip(((fb[:, 3] + 2.0 - gy0) * inv_cy).astype(np.int64), 0, ny - 1)
            for k_f in range(n_faces):
                occ[fy0_c[k_f]:fy1_c[k_f] + 1, fx0_c[k_f]:fx1_c[k_f] + 1] = True
            cand_all = np.flatnonzero(near_f)
            cx_a = np.clip(((sx0[cand_all] - gx0) * inv_cx).astype(np.int64), 0, nx - 1)
            cx_b = np.clip(((sx1[cand_all] - gx0) * inv_cx).astype(np.int64), 0, nx - 1)
            cy_a = np.clip(((sy0[cand_all] - gy0) * inv_cy).astype(np.int64), 0, ny - 1)
            cy_b = np.clip(((sy1[cand_all] - gy0) * inv_cy).astype(np.int64), 0, ny - 1)
            # Keep if either endpoint cell OR bbox spans > 1 cell (long segment could cross a face cell)
            hit_grid = (
                occ[cy_a, cx_a] | occ[cy_b, cx_b] | occ[cy_a, cx_b] | occ[cy_b, cx_a]
                | ((cx_b - cx_a) > 1) | ((cy_b - cy_a) > 1)
            )
            sub_idx = cand_all[hit_grid]
        else:
            sub_idx = np.zeros(0, dtype=np.int64)

        if len(sub_idx):
            sub_seg = all_seg[sub_idx]
            sub_lens = np.hypot(sub_seg[:, 2] - sub_seg[:, 0], sub_seg[:, 3] - sub_seg[:, 1])
            sub_is_unp = ~m[sub_idx]
            sub_lines = shapely.linestrings(sub_seg.reshape(-1, 2, 2))
            all_tree = shapely.STRtree(sub_lines)

            big_all = np.flatnonzero(areas > 2.5)
            if len(big_all):
                inners_all = shapely.buffer(faces_arr[big_all], -0.35, quad_segs=4)
                out_bufs_all = shapely.buffer(faces_arr[big_all], 0.6, quad_segs=4)
                bi_in, li_in = all_tree.query(inners_all, predicate="intersects")
                if len(bi_in):
                    in_lens = _covered_line_inter_len(sub_lines, sub_lens, inners_all, li_in, bi_in)
                    all_in = np.bincount(bi_in, weights=in_lens, minlength=len(big_all))
                    extra[big_all, 8] = np.minimum(all_in / rlens[big_all], 3.0)

                    ord_b = np.argsort(bi_in, kind="quicksort")
                    bi_s, li_s, il_s = bi_in[ord_b], li_in[ord_b], in_lens[ord_b]
                    chg = np.flatnonzero(bi_s[1:] != bi_s[:-1]) + 1
                    starts = np.r_[0, chg]
                    ends = np.r_[chg, len(bi_s)]
                    for b_loc, a_i, e_i in zip(bi_s[starts].tolist(), starts.tolist(), ends.tolist()):
                        fi = int(big_all[b_loc])
                        v = simp_verts[fi]
                        if len(v) != 4:
                            continue
                        hs = li_s[a_i:e_i]
                        if len(hs) < 2 or len(hs) > 24:
                            continue
                        ils = il_s[a_i:e_i]
                        ev = np.roll(v, -1, 0) - v
                        e_ang = np.degrees(np.arctan2(ev[:, 1], ev[:, 0])) % 180.0
                        ob = out_bufs_all[b_loc]
                        diag_angs = []
                        for h_i, in_l in zip(hs.tolist(), ils.tolist()):
                            fl = float(sub_lens[h_i])
                            if fl < 1.0 or in_l < 0.45 * fl:
                                continue
                            p0, p1 = sub_seg[h_i, :2], sub_seg[h_i, 2:]
                            # Fast numpy angle & corner checks BEFORE calling Shapely intersection!
                            ang = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])) % 180.0)
                            d = np.abs(e_ang - ang)
                            if float(np.minimum(d, 180.0 - d).min()) < 10.0:
                                continue
                            c0 = float(np.hypot(v[:, 0] - p0[0], v[:, 1] - p0[1]).min())
                            c1 = float(np.hypot(v[:, 0] - p1[0], v[:, 1] - p1[1]).min())
                            if c0 > 1.6 and c1 > 1.6:
                                continue
                            if float(shapely.length(shapely.intersection(sub_lines[h_i], ob))) < 0.88 * fl:
                                continue
                            diag_angs.append(ang)
                        if 2 <= len(diag_angs) <= 8:
                            a0 = diag_angs[0]
                            if any(min(abs(q - a0), 180.0 - abs(q - a0)) >= 15.0 for q in diag_angs[1:]):
                                extra[fi, 9] = 1.0

            # Reuse the single all_tree for #11 (unpicked strokes touching perimeter) — zero second STRtree!
            if sub_is_unp.any():
                ring_bufs = shapely.buffer(ext_rings, 1.2, quad_segs=4)
                fi_u, li_u = all_tree.query(ring_bufs, predicate="intersects")
                if len(fi_u):
                    unp_hit = sub_is_unp[li_u]
                    if unp_hit.any():
                        fi_u, li_u = fi_u[unp_hit], li_u[unp_hit]
                        ul = _covered_line_inter_len(sub_lines, sub_lens, ring_bufs, li_u, fi_u)
                        extra[:, 10] = np.log1p(np.bincount(fi_u, weights=ul, minlength=n_faces))

    out = np.empty((n_faces, len(FEATURES)), dtype=np.float64)
    for i in range(n_faces):
        long_, short, _ = mrr_info[i]
        area_i = float(areas[i])
        rlen = float(rlens[i])
        cs = float(cov_sum[i])
        arc = float(arc_num[i] / max(cs, 1e-9)) if cs > 0.0 else 0.0
        simp_v = simp_verts[i]
        corners = len(simp_v)
        side_ratio, skew, right = _taper(simp_v)
        par_gap = _par_gap_deg(simp_v)
        n_cusps = _cusps_from_verts(simp_v)
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

        out[i, :30] = (
            np.log1p(area_i), np.log1p(long_), np.log1p(short), long_ / max(short, 1e-6), area_i / max(long_ * short, 1e-9),
            area_i / max(float(hull_areas[i]), 1e-9), min(corners, 20), arc, np.log1p(rlen),
            4 * np.pi * area_i / max(rlen ** 2, 1e-9), int(n_strokes[i]), np.log1p(ink_in[i]), int(touch_cnt[i]),
            side_ratio, skew, right,
            # F225 context features
            nb_cnt, shared_frac, int(collinear_a[i]), int(ortho_a[i]), int(t_side_a[i]), min(float(t_side_max_a[i]), 30.0),
            int(end_match_a[i]), w_diff, max_ar, min_ar, min(max_sh_short, 5.0),
            n_cusps, par_gap, stroke_cov,
        )
    out[:, 30:41] = extra
    # F249 (v9): 4 pure per-face geometric interactions (zero page-constant features)
    out[:, 41] = np.clip((1.0 - out[:, 13]) * out[:, 14], 0.0, 90.0)                      # taper_strength
    out[:, 42] = np.minimum(out[:, 38] / np.maximum(out[:, 29], 0.25), 15.0)              # ink_density_ratio
    out[:, 43] = np.minimum(out[:, 36] / np.maximum(out[:, 37], 0.10), 20.0)              # nb_short_span_ratio
    out[:, 44] = np.clip(out[:, 5] - out[:, 4], -1.0, 1.0)                                # hull_rect_gap
    # F254 (v10): Crossing-safe log_ink_inside — subtract ONLY genuine both-end pass-through crossing strokes
    # extending >= 5.0 pt outside the face on BOTH ends so overlapping crossing ducts/elbows aren't penalized as room boxes
    has_ink = np.flatnonzero(out[:, 11] > 0.05)
    if len(has_ink) and len(m_seg):
        mlen = np.hypot(m_seg[:, 2] - m_seg[:, 0], m_seg[:, 3] - m_seg[:, 1])
        long_m = mlen > 11.0
        if long_m.any():
            l_seg = m_seg[long_m]
            l_len = mlen[long_m]
            l_lines = shapely.linestrings(l_seg.reshape(-1, 2, 2))
            ltree_p = shapely.STRtree(l_lines)
            faces_sub = np.asarray([faces[i] for i in has_ink], dtype=object)
            inners_p = shapely.buffer(faces_sub, -GAP - 0.3, quad_segs=4)
            bi_p, li_p = ltree_p.query(inners_p, predicate="intersects")
            if len(bi_p):
                d0 = shapely.distance(faces_sub[bi_p], shapely.points(l_seg[li_p, :2]))
                d1 = shapely.distance(faces_sub[bi_p], shapely.points(l_seg[li_p, 2:]))
                is_both_pass = (d0 >= 5.0) & (d1 >= 5.0)
                if is_both_pass.any():
                    bi_k, li_k = bi_p[is_both_pass], li_p[is_both_pass]
                    pass_len = _covered_line_inter_len(l_lines, l_len, inners_p, li_k, bi_k)
                    sub_pass = np.bincount(bi_k, weights=pass_len, minlength=len(has_ink))
                    out[has_ink, 11] = np.log1p(np.maximum(np.expm1(out[has_ink, 11]) - sub_pass, 0.0))
    return out


# ---- post-processing step (F151): faces named by a classifier, added where the logic left nothing -------------------------------------

PICK_THR = 0.3     # strokes with p >= this draw the faces (chosen on the train pages, F151)
FACE_CONF = 0.5    # a face becomes an object when its class probability >= this (chosen on the train pages, cross-fitted by document)
COVERED = 0.3      # a face more than this share inside the logic's objects belongs to the logic
CLASSES = ("none", "Duct", "Elbow", "Elbow vuông", "Concentric Reducer", "Eccentric Reducer", "Chân rẽ", "Tee")   # Tee: out of the
# 6-class benchmark (scored as nothing) but a real object of the drawings; without it a tee face was called an Elbow (F152)


def _predict_proba(clf, X: np.ndarray) -> np.ndarray:
    """Backward-compatible predict_proba: slices 45-feature X to 41 or 30 cols when using a legacy classifier."""
    n_in = getattr(clf, "n_features_in_", None)
    if n_in is not None and X.shape[1] > int(n_in):
        return clf.predict_proba(X[:, :int(n_in)])
    return clf.predict_proba(X)


def fit_classifier(
    npz_paths,
    viewer: bool = True,
    skip_doc: str | None = None,
    gt_share: float | None = 0.35,
    vw_in_w: float = 0.30,
    vw_noisy_w: float = 0.35,
    ensemble: bool = True,
    max_iter: int = 500,
    learning_rate: float = 0.06,
    max_leaf_nodes: int = 47,
    l2_regularization: float = 1.5,
):
    """HistGradientBoosting (or F249 v9 3-HGB diverse soft-voting ensemble) on 45-feature faces
    (tools/build_faces_train_e13.py / faces_train.py).
    When ``viewer=True`` and both GT and Viewer faces are present, ``gt_share`` balances GT vs Viewer sample mass,
    ``vw_in_w`` downweights Viewer inside-GT sub-faces labeled ``none``, and ``vw_noisy_w`` softly downweights
    contradictory Viewer V1 labels (coaxial Chân rẽ, 4-corner zero-arc Elbow, 4-right-corner Concentric Reducer).
    -> (clf, class names)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier

    X, y, pgs, in_shs, classes = [], [], [], [], CLASSES
    has_in = True
    for t in npz_paths:
        z = np.load(t)
        pg = z["page"].astype(str)
        keep = np.ones(len(z["y"]), bool) if viewer else ~np.char.startswith(pg, "vw_")
        if skip_doc:
            keep &= ~np.char.startswith(pg, skip_doc)
        X.append(z["X"][keep]); y.append(z["y"][keep]); pgs.append(pg[keep])
        if "in_sh" in z.files:
            in_shs.append(z["in_sh"][keep])
        else:
            has_in = False
        classes = tuple(str(c) for c in z["classes"])
    X_all, y_all, pg_all = np.concatenate(X), np.concatenate(y), np.concatenate(pgs)
    sw = np.ones(len(y_all), dtype=np.float64)
    if viewer and gt_share is not None:
        is_gt = ~np.char.startswith(pg_all, "vw_")
        if is_gt.any() and (~is_gt).any():
            sw = np.where(is_gt, gt_share / is_gt.sum(), (1.0 - gt_share) / (~is_gt).sum()) * len(y_all)
        if has_in and vw_in_w != 1.0:
            in_all = np.concatenate(in_shs)
            sw = np.where((~is_gt) & (y_all == 0) & (in_all >= 0.50), sw * vw_in_w, sw)
        if vw_noisy_w != 1.0 and X_all.shape[1] >= 41:
            col = {n: i for i, n in enumerate(FEATURES)}
            is_vw_noisy = (~is_gt) & (
                ((y_all == 6) & (X_all[:, col["coaxial_ends"]] >= 0.5))
                | ((y_all == 2) & (X_all[:, col["corners"]] == 4) & (X_all[:, col["arc_share"]] < 0.05))
                | ((y_all == 4) & (X_all[:, col["right_corners"]] == 4))
            )
            sw = np.where(is_vw_noisy, sw * vw_noisy_w, sw)
    if ensemble:
        clf = VotingClassifier(
            estimators=[
                ("hgb1", HistGradientBoostingClassifier(max_iter=550, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=2.0, max_features=0.80, random_state=0)),
                ("hgb2", HistGradientBoostingClassifier(max_iter=550, learning_rate=0.05, max_leaf_nodes=47, min_samples_leaf=20, l2_regularization=1.5, max_features=0.90, random_state=1)),
                ("hgb3", HistGradientBoostingClassifier(max_iter=550, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=25, l2_regularization=3.0, max_features=0.75, random_state=2)),
            ],
            voting="soft",
        )
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=20,
            l2_regularization=l2_regularization,
            random_state=0,
        )
    clf.fit(X_all, y_all, sample_weight=sw)
    return clf, classes


def save_classifier(path, clf, classes, features=None) -> None:
    import joblib

    feats = list(features) if features is not None else list(FEATURES[:getattr(clf, "n_features_in_", len(FEATURES))])
    joblib.dump({"clf": clf, "classes": list(classes), "features": feats, "gap": GAP}, path, compress=("zlib", 6))


def load_classifier(path):
    """Load 45-feature v9+, 41-feature v7/v8, or legacy 30-feature v6 classifier (100% backward compatible)."""
    import joblib

    m = joblib.load(path)
    feats = list(m["features"])
    if feats not in (list(FEATURES), list(FEATURES_41), list(FEATURES_30)):
        raise ValueError(f"{path}: face features {feats} differ from this code's {list(FEATURES)}; refit it")
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
    bufs = shapely.buffer(arr, 0.5, quad_segs=4)
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
    """Classifier names + Bayes F2 foreground-union rescue + refined shape rules (F152 / F225 / F245 v7):
    inline taper between two distinct end-cap neighbours (with anti-T-branch overhang check) -> reducer;
    L-shaped square elbow (>= 5 right corners, 1 cusp, >= 2 neighbours) -> Elbow vuông;
    Elbow with >= 3 object neighbours & >= 1 cusp -> Tee. -> names, scores, why, object-neighbour lists, class probabilities."""
    import shapely

    pr = _predict_proba(clf, X)
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
            elif names[i] != "none" and score[i] < FG_UNION_THR and fg_sum[i] >= 0.60:
                score[i] = float(fg_sum[i])
    col = {n: i for i, n in enumerate(FEATURES)}
    # F249 (v9): a crisp 4-right-corner non-curved rectangle is never a curved Elbow
    for i in range(len(faces)):
        if (names[i] == "Elbow" and X[i, col["corners"]] == 4
                and X[i, col["right_corners"]] == 4 and X[i, col["arc_share"]] < 0.25):
            names[i] = "Duct"
    nb = [[j for j in js if names[j] != "none"] for js in neighbours(faces)]   # neighbours the classifier calls an object
    mrr_cache: dict[int, tuple[float, float, np.ndarray]] = {}
    simp_cache: dict[int, np.ndarray] = {}
    why = ["face"] * len(faces)

    def _touched_nb_edge_len(end_line, j_idx: int) -> float:
        """Length of the neighbour edge collinear with end_line (detects side-tap even on short trunk ducts!)."""
        vj = simp_cache.get(j_idx)
        if vj is None:
            vj = simp_cache[j_idx] = np.asarray(shapely.simplify(faces[j_idx], 0.5).exterior.coords)[:-1]
        if len(vj) < 3:
            return 0.0
        ev = np.asarray(end_line.coords[-1]) - np.asarray(end_line.coords[0])
        el = float(np.hypot(ev[0], ev[1]))
        if el < 1e-6:
            return 0.0
        eu = ev / el
        ebuf = end_line.buffer(0.8, quad_segs=4)
        best_len = 0.0
        for k in range(len(vj)):
            a, b = vj[k], vj[(k + 1) % len(vj)]
            dv = b - a
            dl = float(np.hypot(dv[0], dv[1]))
            if dl < 1e-6:
                continue
            if abs(float(eu @ (dv / dl))) >= 0.90 and shapely.LineString([a, b]).intersects(ebuf):
                if dl > best_len:
                    best_len = dl
        return best_len

    for i, f in enumerate(faces):
        x = X[i]
        # F225 / F245 (v7) / F249 (v9): inline taper rule requires 2 DISTINCT opposite-end neighbours with anti-T-branch overhang check;
        # also allows inline coaxial reducers that have a third branch touching their flank, while protecting high-conf near-parallel Ducts
        # and classifier-confirmed Eccentric Reducers.
        ts_ok = (x[col["nb_t_side_cnt"]] == 0) or (x[col["coaxial_ends"]] >= 0.5 and x[col["wide_end_overhang"]] <= 1.38)
        do_taper = (
            x[col["corners"]] == 4 and x[col["side_ratio"]] < TAPER_RATIO and x[col["skew_deg"]] >= TAPER_SKEW
            and x[col["hull_fill"]] >= 0.90 and x[col["rect_fill"]] >= 0.50
            and f.area >= 3.0 and np.expm1(x[col["log_short"]]) >= 1.2
            and len(nb[i]) >= 2 and ts_ok
            and not (names[i] == "Duct" and score[i] >= 0.85 and x[col["skew_deg"]] < 10.0)
            and not (names[i] == "Chân rẽ" and score[i] >= 0.75)
        )
        if do_taper:
            v = np.asarray(shapely.simplify(f, 0.5).exterior.coords)[:-1]
            if len(v) == 4:
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
                    ev = np.asarray(end.coords[-1]) - np.asarray(end.coords[0])
                    el = float(np.hypot(ev[0], ev[1]))
                    if el < 1e-6:
                        return False
                    if _touched_nb_edge_len(end, j) > 1.38 * el:
                        return False   # sitting on a longer side wall = Chân rẽ shoe-tap, NOT inline reducer!
                    if lj < 1.45 * max(sj, 1e-6):
                        return True
                    dot_j = abs(float((ev[0] * ax_j[0] + ev[1] * ax_j[1]) / el))
                    return not (dot_j >= 0.75 and el < 0.75 * lj)

                m0 = [j for j in nb[i] if _meets_end(ends[0], j)]
                m1 = [j for j in nb[i] if _meets_end(ends[1], j)]
                if any(a != b for a in m0 for b in m1):
                    rk = reducer_kind(f)
                    if names[i] == "Eccentric Reducer" and score[i] >= 0.75:
                        rk = "Eccentric Reducer"
                    names[i], why[i] = rk, "taper"
                    score[i] = max(score[i], RULE_SCORE)
        # F240 / F245 / F246 (v8) / F255 (v11): 6-corner L-shaped square elbow (>= 5 right corners, 1 cusp, >= 2 nbs; OR
        # near-square 6-corner L-elbow with >= 4 right corners, 1 cusp, aspect <= 1.15, rfill >= 0.90, hfill >= 0.95, >= 1 nb)
        if x[col["corners"]] == 6 and x[col["cusp_cnt"]] == 1 and not (names[i] == "Duct" and score[i] >= 0.85) and (
            (x[col["right_corners"]] >= 5 and len(nb[i]) >= 2)
            or (names[i] in ("Duct", "Elbow") and x[col["right_corners"]] >= 4
                and x[col["aspect"]] <= 1.15 and x[col["rect_fill"]] >= 0.90
                and x[col["hull_fill"]] >= 0.95 and len(nb[i]) >= 1)
        ):
            names[i], why[i] = "Elbow vuông", "l_elbow"
            score[i] = max(score[i], RULE_SCORE)
        # F240 / F245 (v7) / F255 (v11): End-of-duct convex tapering transition with verified clean end-cap contact
        # (never override confident Chân rẽ >= 0.75: terminal bellmouth/diffuser collars are Chân rẽ, 21:1 HARM vs GAIN)
        if (names[i] in ("Chân rẽ", "Duct", "none") and not (names[i] == "Chân rẽ" and score[i] >= 0.75)
                and x[col["corners"]] == 4
                and x[col["side_ratio"]] < 0.82 and x[col["skew_deg"]] >= 14.0
                and x[col["hull_fill"]] >= 0.90 and x[col["rect_fill"]] >= 0.50
                and x[col["nb_t_side_cnt"]] == 0 and len(nb[i]) == 1
                and f.area >= 35.0 and np.expm1(x[col["log_short"]]) >= 5.0
                and x[col["nb_end_match_cnt"]] >= 1.0):
            v = np.asarray(shapely.simplify(f, 0.5).exterior.coords)[:-1]
            ok_et = True
            if len(v) == 4:
                e = np.roll(v, -1, 0) - v
                ang = np.degrees(np.arctan2(e[:, 1], e[:, 0]))
                d = lambda a, b: min(abs((ang[a] - ang[b]) % 180), 180 - abs((ang[a] - ang[b]) % 180))
                par = (0, 2) if d(0, 2) <= d(1, 3) else (1, 3)
                j0 = nb[i][0]
                ok_et = False
                for k in par:
                    end = shapely.LineString([v[k], v[(k + 1) % 4]])
                    if end.buffer(0.6).intersection(faces[j0].exterior).length >= 0.55 * end.length:
                        if _touched_nb_edge_len(end, j0) <= 1.35 * max(end.length, 1e-6):
                            ok_et = True
            if ok_et:
                names[i], why[i] = reducer_kind(f), "end_taper"
                score[i] = max(score[i], RULE_SCORE)
        # F237 / F245 (v7) / F249 (v9) / F254 (v10) / F255 (v11): physical 3-way curved junction requires >= 3 object neighbours
        # AND >= 1 reflex cusp, while protecting confident Elbows (>= 0.60: 10-to-0 clean separation on GT25!) and crisp Ducts.
        if (names[i] == "Elbow" or (x[col["arc_share"]] >= 0.15 and x[col["corners"]] >= 8)) and len(nb[i]) >= 3 and cusps(f) >= 1:
            if not (names[i] == "Elbow" and score[i] >= 0.60):
                if not (names[i] == "Elbow" and x[col["hull_fill"]] >= 0.90 and x[col["rect_fill"]] >= 0.85):
                    if not (x[col["hull_fill"]] >= 0.92 and x[col["rect_fill"]] >= 0.88):
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
    """Eccentric for a tapering 4-corner outline with one straight flank (2 right corners, or F249 v9 strongly asymmetric
    flanks where one flank is within 5 deg of perpendicular and the opposite flank is slanted >= max(2.2x, +8 deg));
    else Concentric (a rectangle's 4 right corners must not count, F163)."""
    import shapely

    simp = np.asarray(shapely.simplify(f, 0.5).exterior.coords)[:-1]
    ratio, skew, right = _taper(simp)
    if len(simp) == 4 and ratio < TAPER_RATIO:
        if right == 2:
            return "Eccentric Reducer"
        if right < 4 and skew >= 8.0:
            e = np.roll(simp, -1, 0) - simp
            ang = np.degrees(np.arctan2(e[:, 1], e[:, 0]))
            d = lambda a, b: min(abs((ang[a] - ang[b]) % 180), 180 - abs((ang[a] - ang[b]) % 180))
            par = (0, 2) if d(0, 2) <= d(1, 3) else (1, 3)
            oth = (1, 3) if par == (0, 2) else (0, 2)
            p_ang = ang[par[0]]
            dev0 = abs(90.0 - min(abs((ang[oth[0]] - p_ang) % 180), 180 - abs((ang[oth[0]] - p_ang) % 180)))
            dev1 = abs(90.0 - min(abs((ang[oth[1]] - p_ang) % 180), 180 - abs((ang[oth[1]] - p_ang) % 180)))
            d_lo, d_hi = (dev0, dev1) if dev0 <= dev1 else (dev1, dev0)
            if d_lo <= 5.0 and d_hi >= max(2.2 * d_lo, d_lo + 8.0):
                return "Eccentric Reducer"
    return "Concentric Reducer"


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


def hybrid_group_objects(
    segs,
    prob: np.ndarray,
    length: np.ndarray,
    group_objects: dict,
    face_objects: list[dict] | None = None,
    *,
    thr: float = 0.50,
    face_thr: float = 0.65,
    fg_thr: float = 0.40,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """F230 Reverse Hybrid (M1): OpenStroke41 GroupClsHead clusters + hvac_faces v2 Reducer/Split refinement (0 AI_Takeoff logic).

    Measured on E144 val (13 pages, 6-class F2):
      - Pure OpenStroke41 (face_objects=None): GT5 0.723 / VW8 0.754 (mean 0.738)
      - Reverse Hybrid M1 (with hvac_faces v2): GT5 0.835 / VW8 0.763 (mean 0.799; beats hvac_faces v2 Oracle on VW8!)

    -> (objects list[dict], cls_per_shape (N,) int8 in HVAC_CLASSES, inst_per_shape (N,) int32).
    """
    from .objects import HVAC_TO_OBJ8, OBJ8_CLASSES
    from .objects import HVAC_CLASSES, assign_objects, class_of, object_polygon

    n = len(prob)
    L = np.asarray(length, dtype=np.float64)
    ginst = np.asarray(group_objects["instance_id"], dtype=np.int64)
    gcids = np.asarray(group_objects["cluster_ids"], dtype=np.int64)
    gcprobs = np.asarray(group_objects["cluster_probs"], dtype=np.float32)

    pick = (np.asarray(prob, dtype=np.float64) >= thr) & (ginst >= 0)
    pred = np.full(n, -1, dtype=np.int64)
    pcls: dict[int, int] = {}
    pconf: dict[int, float] = {}
    ppoly: dict[int, list] = {}
    pref: dict[int, str] = {}

    for i, cid in enumerate(gcids):
        pr = gcprobs[i]
        if 1.0 - float(pr[0]) < fg_thr:
            continue
        k = int(pr[1:].argmax()) + 1
        if 1 <= k <= 6:
            m = pick & (ginst == cid)
            if m.any():
                pred[m] = int(cid)
                pcls[int(cid)] = k
                pconf[int(cid)] = float(pr[k])
                pref[int(cid)] = "group_head"

    if face_objects:
        objs6 = [o for o in face_objects if 1 <= class_of(o.get("shape_name")) <= 8]
        if objs6:
            _, finst, _ = assign_objects(segs, objs6, np.asarray(prob, dtype=np.float64) >= face_thr)
            pred_f = np.full(n, -1, dtype=np.int64)
            pcls_f: dict[int, int] = {}
            pscore_f: dict[int, float] = {}
            for oid in np.unique(finst[finst >= 0]):
                o = objs6[int(oid)]
                c8 = int(HVAC_TO_OBJ8[class_of(o.get("shape_name"))])
                if 1 <= c8 <= 6:
                    pred_f[finst == oid] = int(oid)
                    pcls_f[int(oid)] = c8
                    pscore_f[int(oid)] = float(o.get("score") or 0.7)

            lf = {oid: float(L[pred_f == oid].sum()) for oid in pcls_f}
            next_id = (max(pcls) + 1) if pcls else 0
            for cid in list(pcls.keys()):
                cm = pred == cid
                lc = float(L[cm].sum())
                ov = pred_f[cm & (pred_f >= 0)]
                if not len(ov):
                    continue
                uf, inv = np.unique(ov, return_inverse=True)
                inter = np.bincount(inv, weights=L[cm & (pred_f >= 0)])
                # 1) Split over-merged cluster when >= 2 closed faces lie >= 70% inside it
                sub_f = [int(fid) for fid, a in zip(uf, inter) if lf[int(fid)] > 0 and a >= 0.70 * lf[int(fid)]]
                if len(sub_f) >= 2:
                    pred[pred == cid] = -1
                    pcls.pop(cid, None)
                    pconf.pop(cid, None)
                    pref.pop(cid, None)
                    for fid in sub_f:
                        fm = pred_f == fid
                        pred[fm] = next_id
                        pcls[next_id] = pcls_f[fid]
                        pconf[next_id] = pscore_f[fid]
                        ppoly[next_id] = objs6[fid].get("vertices") or []
                        pref[next_id] = "face_split"
                        next_id += 1
                    continue
                # 2) 1-to-1 matched closed face (IoU >= 0.50): adopt clean polygon, override to Reducer if face sees taper/reducer
                best_i = int(np.argmax(inter))
                fid = int(uf[best_i])
                iou = inter[best_i] / max(lc + lf[fid] - inter[best_i], 1e-9)
                if iou >= 0.50:
                    ppoly[cid] = objs6[fid].get("vertices") or []
                    if pcls_f[fid] in (4, 5):
                        pcls[cid] = pcls_f[fid]
                        pconf[cid] = pscore_f[fid]
                        pref[cid] = "face_reducer"

    # Build canonical object dicts + compact inst/cls arrays (HVAC_CLASSES indices)
    obj8_to_hvac = {k: HVAC_CLASSES.index(OBJ8_CLASSES[k]) for k in range(1, 7)}
    out_objs: list[dict] = []
    cls_out = np.zeros(n, dtype=np.int8)
    inst_out = np.full(n, -1, dtype=np.int32)
    for cid in sorted(pcls.keys()):
        members = np.flatnonzero(pred == cid)
        if not len(members):
            continue
        new_oid = len(out_objs)
        k8 = pcls[cid]
        hcls = obj8_to_hvac[k8]
        verts = ppoly.get(cid)
        if not verts or len(verts) < 3:
            verts = object_polygon(segs.seg, segs.owner, members, hcls)
        out_objs.append({
            "shape_name": OBJ8_CLASSES[k8],
            "vertices": verts,
            "score": round(pconf.get(cid, 0.7), 3),
            "refined": pref.get(cid, "group_head"),
        })
        inst_out[members] = new_oid
        cls_out[members] = hcls
    return out_objs, cls_out, inst_out


def faces_first_hybrid(
    segs,
    prob: np.ndarray,
    curved: np.ndarray,
    length: np.ndarray,
    clf,
    classes,
    group_objects: dict | None = None,
    *,
    thr: float = PICK_THR,
    conf: float = 0.35,
    fp_veto_thr: float = 0.55,
    fp_veto_fscore_max: float = 0.80,
    none_rescue_thr: float = 0.55,
    cr_override_thr: float = 0.50,
    ev_override_thr: float = 0.60,
    merge_subfaces: bool = True,
    drop_isolated_fittings: bool = True,
    attrs: dict | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """F234 (A5 + C7a): Stroke-Matched Faces-First Hybrid HVAC decoder (0 PDF layers, 0 AI_Takeoff logic).

    Builds crisp planar CAD faces from picked strokes (including A5 dashed-duct closure: collinear/curved dash bridging,
    angled corners, and forward T-ray extension), classifies them with ``hvac_faces`` + geometric taper/three-way rules,
    and—when ``OpenStroke41`` ``group_objects`` are present—applies 4 precision-safe refinements via exact stroke-level
    ``assign_objects`` face↔cluster alignment (0% courtyard or trunk bleed):
      1. Same-cluster subface merge (rejoins shattered multi-item Chân rẽ / split Duct subfaces inside 1 FG cluster);
      2. Confidence-gated FP veto (drops low-conf face FPs when 1-to-1 matched cluster predicts none);
      3. Non-overlapping ``none``-face rescue (recovers faces face_clf missed when 1-to-1 cluster strongly predicts FG);
      4. Surgical ``Duct -> Chân rẽ / Elbow vuông`` override (fixes unseen Viewer branch styles without touching reducers).

    Measured on E145 val (strict 6-class ``score_pack_v2_val/gt``):
      - GT5 0.8414 (0.8534 rule_nl) / VW8 0.7756 (0.7696 rule_nl) / Mean 0.8085 (0.8115 rule_nl),
        polyIoU 0.972 / 0.919, overlap 4.8% / 6.9%, MB TH4 dashed CDC-ACMV-B.EA 0.983 (45/46, P=1.000).
    -> (objects list[dict], cls_per_shape (N,) int8 in HVAC_CLASSES, inst_per_shape (N,) int32).
    """
    import shapely

    from .objects import OBJ8_CLASSES
    from .objects import assign_objects

    prob = np.asarray(prob, dtype=np.float64)
    L = np.asarray(length, dtype=np.float64)
    n = len(prob)
    picked = prob >= thr
    if not picked.any():
        return [], np.zeros(n, dtype=np.int8), np.full(n, -1, dtype=np.int32)

    faces, src, inner = priority_candidates(segs, picked, curved=curved)
    faces, src, inner = _split_tangent_seams(faces, src, inner, segs, picked, curved)
    if not faces:
        return [], np.zeros(n, dtype=np.int8), np.full(n, -1, dtype=np.int32)

    X = face_features(faces, segs, picked, curved)
    fnames, fscore, fwhy, fnb, pr_f = _name_faces(faces, X, clf, classes)
    for i, f in enumerate(faces):
        if inner[i] >= 2 and _corners(f) == 4 and len(fnb[i]) <= 1:
            fnames[i], fscore[i], fwhy[i] = "Chân rẽ vuông tròn", max(fscore[i], RULE_SCORE), "symbol"

    n_f = len(faces)
    cid_pr: dict[int, np.ndarray] = {}
    cid_len: dict[int, float] = {}
    f_best_cid = np.full(n_f, -1, dtype=np.int64)
    f_best_iou = np.zeros(n_f, dtype=np.float64)
    f_best_sh = np.zeros(n_f, dtype=np.float64)
    f_pnone_num = np.zeros(n_f, dtype=np.float64)
    f_pnone_den = np.zeros(n_f, dtype=np.float64)
    lf = np.zeros(n_f, dtype=np.float64)
    cid_subfaces: dict[int, list[tuple[int, float, float]]] = {}

    if group_objects is not None:
        ginst_m = np.asarray(group_objects["instance_id"], dtype=np.int64)
        gcids = np.asarray(group_objects["cluster_ids"], dtype=np.int64)
        gcprobs = np.asarray(group_objects["cluster_probs"], dtype=np.float64)
        cid_pr = {int(c): gcprobs[i] for i, c in enumerate(gcids)}
        if len(gcids):
            ok_g = picked & (ginst_m >= 0)
            if ok_g.any():
                ug, inv_g = np.unique(ginst_m[ok_g], return_inverse=True)
                ul = np.bincount(inv_g, weights=L[ok_g])
                cid_len = {int(c): float(l_v) for c, l_v in zip(ug.tolist(), ul.tolist())}

        cand_objs = [
            {"shape_name": "Duct", "vertices": np.asarray(shapely.simplify(f, 0.3).exterior.coords)[:-1].tolist()}
            for f in faces
        ]
        _, finst_all, _ = assign_objects(segs, cand_objs, picked)
        lf = np.bincount(finst_all[finst_all >= 0], weights=L[finst_all >= 0], minlength=n_f)
        both = (finst_all >= 0) & (ginst_m >= 0)
        if both.any():
            max_c = int(ginst_m[both].max()) + 1
            key = finst_all[both].astype(np.int64) * max_c + ginst_m[both].astype(np.int64)
            u_k, inv_k = np.unique(key, return_inverse=True)
            inter = np.bincount(inv_k, weights=L[both])
            for k_val, a in zip(u_k.tolist(), inter.tolist()):
                fi, cid = int(k_val // max_c), int(k_val % max_c)
                cl = cid_len.get(cid, 0.0)
                if cl <= 0.0 or lf[fi] <= 0.0:
                    continue
                if cid in cid_pr:
                    f_pnone_num[fi] += a * float(cid_pr[cid][0])
                    f_pnone_den[fi] += a
                iou = a / max(lf[fi] + cl - a, 1e-9)
                sh = a / max(lf[fi], 1e-9)
                if iou > f_best_iou[fi]:
                    f_best_iou[fi] = iou
                    f_best_cid[fi] = cid
                    f_best_sh[fi] = sh
                if sh >= 0.65:
                    cid_subfaces.setdefault(cid, []).append((fi, sh, float(a)))

    merged_drop: set[int] = set()
    primary_objs: list[dict] = []
    primary_polys: list = []
    rescued_cands: list[tuple] = []
    ftree_all = shapely.STRtree(np.asarray(faces, dtype=object)) if (merge_subfaces and cid_subfaces) else None

    if merge_subfaces and cid_subfaces:
        for cid, subs in cid_subfaces.items():
            if len(subs) < 2 or cid not in cid_pr:
                continue
            pr_c = cid_pr[cid]
            if 1.0 - float(pr_c[0]) < 0.50:
                continue
            k_c = int(pr_c[1:].argmax()) + 1
            if not (1 <= k_c <= 7):
                continue
            idxs = [q[0] for q in subs]
            if sum(q[2] for q in subs) < 0.55 * cid_len.get(cid, 1e9):
                continue
            if sum(1 for i in idxs if fnames[i] not in ("none", "Duct") and float(fscore[i]) >= 0.65) >= 2:
                continue
            u = shapely.union_all([faces[i].buffer(0.4, join_style="mitre") for i in idxs]).buffer(-0.4, join_style="mitre")
            if u.geom_type != "Polygon" or u.area <= MIN_AREA:
                continue
            if k_c == 1:
                if any(fwhy[i] in ("taper", "three_ways", "l_elbow", "end_taper")
                       or fnames[i] in ("Elbow", "Elbow vuông", "Concentric Reducer", "Eccentric Reducer", "Chân rẽ")
                       for i in idxs):
                    continue
                if u.area / max(u.minimum_rotated_rectangle.area, 1e-9) < 0.85 or _corners(u) != 4:
                    continue
            elif k_c == 6:
                if u.area > 300.0 or all(fnames[i] != "none" and float(fscore[i]) >= 0.65 for i in idxs):
                    continue
                if u.area / max(u.convex_hull.area, 1e-9) < 0.85:
                    continue
            else:
                if u.area > 400.0 or all(fnames[i] != "none" for i in idxs):
                    continue
            # F237: never accept a merged union that swallows an external confident face outside idxs
            if ftree_all is not None:
                idx_set = set(idxs)
                swallows_ext = False
                for h in ftree_all.query(u, predicate="intersects"):
                    h = int(h)
                    if h in idx_set or fnames[h] == "none" or float(fscore[h]) < conf:
                        continue
                    if u.intersection(faces[h]).area >= 0.50 * faces[h].area:
                        swallows_ext = True
                        break
                if swallows_ext:
                    continue
            merged_drop.update(idxs)
            nm = reducer_kind(u) if k_c in (4, 5) else OBJ8_CLASSES[k_c]
            g = shapely.simplify(u, 0.3)
            primary_objs.append({
                "shape_name": nm,
                "vertices": np.asarray(g.exterior.coords)[:-1].round(3).tolist(),
                "score": round(float(1.0 - pr_c[0]), 3),
                "refined": "merged_cid",
            })
            primary_polys.append(u)

    # F245 / F246 (v8): Face-level mitered Elbow vuông merge (BEFORE dropping 'none' halves!) — recovers diagonal
    # turning-vane square elbows whose 2 symmetric halves (3..8 corners, area 35..450) either touch directly along the
    # 45° diagonal OR flank 1 thin double-line turning-vane strip (area <= 22 pt², aspect >= 2.2).
    if n_f >= 3:
        f_arr = np.asarray(faces, dtype=object)
        f_areas = shapely.area(f_arr)
        f_simps = _bulk_ring_coords(shapely.simplify(f_arr, 0.5))
        f_crns = np.array([len(v) for v in f_simps], dtype=np.int64)
        is_fg_f = np.array([(i not in merged_drop) and fnames[i] != "none" and float(fscore[i]) >= conf for i in range(n_f)], dtype=bool)
        col_el = {n: idx_c for idx_c, n in enumerate(FEATURES)}
        el_ok_f = np.array([
            (i not in merged_drop)
            and (
                fnames[i] in ("none", "Eccentric Reducer", "Concentric Reducer", "Chân rẽ", "Duct", "Elbow vuông")
                or (fnames[i] == "Elbow" and float(X[i, col_el["arc_share"]]) < 0.15)
            )
            and (35.0 <= float(f_areas[i]) <= 450.0)
            and (3 <= f_crns[i] <= 8)
            for i in range(n_f)
        ], dtype=bool)
        vane_ok_f = np.array([
            (i not in merged_drop)
            and (fnames[i] == "none")
            and (1.0 <= float(f_areas[i]) <= 22.0)
            and (f_crns[i] in (3, 4))
            for i in range(n_f)
        ], dtype=bool)
        if int(el_ok_f.sum()) >= 2:
            tr_m = shapely.STRtree(f_arr)
            mi, mj = tr_m.query(shapely.buffer(f_arr, 1.8, quad_segs=4), predicate="intersects")
            mmk = mi != mj
            mi, mj = mi[mmk], mj[mmk]
            nb_all_f: list[set[int]] = [set() for _ in range(n_f)]
            nb_fg_f: list[set[int]] = [set() for _ in range(n_f)]
            for a_i, b_i in zip(mi.tolist(), mj.tolist()):
                nb_all_f[a_i].add(b_i)
                if is_fg_f[b_i]:
                    nb_fg_f[a_i].add(b_i)
            # Build candidate pairs: direct neighbours + pairs linked through 1 thin diagonal turning-vane strip
            pair_strip: dict[tuple[int, int], int] = {}
            for a_i in np.flatnonzero(el_ok_f).tolist():
                for b_i in nb_all_f[a_i]:
                    if b_i > a_i and el_ok_f[b_i]:
                        pair_strip[(a_i, b_i)] = -1
            for s_i in np.flatnonzero(vane_ok_f).tolist():
                ls_s, ss_s, _ = _mrr_axes(faces[s_i])
                if ls_s / max(ss_s, 1e-6) < 2.2:
                    continue
                adj_h = sorted(h for h in nb_all_f[s_i] if el_ok_f[h])
                for ia in range(len(adj_h)):
                    for ib in range(ia + 1, len(adj_h)):
                        pair_strip.setdefault((adj_h[ia], adj_h[ib]), s_i)
            ev_cands = []
            for (a_i, b_i), s_i in pair_strip.items():
                ra = float(f_areas[a_i]) / max(float(f_areas[b_i]), 1e-6)
                if not (0.72 <= ra <= 1.38):
                    continue
                va, vb = f_simps[a_i], f_simps[b_i]
                if len(va) == 4 and _taper(va)[2] >= 4 and len(vb) == 4 and _taper(vb)[2] >= 4:
                    continue
                if faces[a_i].intersection(faces[b_i]).area >= 0.15 * min(float(f_areas[a_i]), float(f_areas[b_i])):
                    continue
                min_per = min(float(faces[a_i].length), float(faces[b_i].length))
                if s_i < 0:
                    seam_geom = faces[a_i].boundary.intersection(faces[b_i].buffer(0.45))
                    seam = float(seam_geom.length)
                    if seam < 0.22 * min_per:
                        continue
                    seam_dir = _shared_dir(seam_geom)
                    parts_u = [faces[a_i], faces[b_i]]
                    excl = {a_i, b_i}
                else:
                    sb = faces[s_i].buffer(0.45)
                    sg_a = faces[a_i].boundary.intersection(sb)
                    sa = float(sg_a.length)
                    sb_l = float(faces[b_i].boundary.intersection(sb).length)
                    if min(sa, sb_l) < 0.20 * min_per:
                        continue
                    seam_dir = _shared_dir(sg_a)
                    parts_u = [faces[a_i], faces[s_i], faces[b_i]]
                    excl = {a_i, b_i, s_i}
                if len((nb_fg_f[a_i] | nb_fg_f[b_i]) - excl) < 2:
                    continue
                both_fg = bool(nb_fg_f[a_i] - excl) and bool(nb_fg_f[b_i] - excl)
                if not both_fg:
                    # Turning vane cutting across airflow (inner half touches >= 2 ducts, outer half faces open space):
                    # strictly require twin ~45 deg diagonal miter halves (35..55 deg skew).
                    sa_sk = _taper(va)[1] if len(va) == 4 else 0.0
                    sb_sk = _taper(vb)[1] if len(vb) == 4 else 0.0
                    if not (35.0 <= sa_sk <= 55.0 and 35.0 <= sb_sk <= 55.0):
                        continue
                u = shapely.union_all([p.buffer(0.15, join_style="mitre") for p in parts_u]).buffer(-0.15, join_style="mitre")
                if u.geom_type != "Polygon" or not (75.0 <= float(u.area) <= 850.0):
                    continue
                if _corners(u) not in (4, 5, 6):
                    continue
                rfill = float(u.area) / max(float(u.minimum_rotated_rectangle.area), 1e-9)
                ls_u, ss_u, ax_u = _mrr_axes(u)
                # F255 (v11): Require genuine ~45 deg diagonal miter seam (25..65 deg to square axes),
                # preventing false merges of a Concentric Reducer + adjacent Duct sharing a 0/90 deg flange seam!
                if seam_dir is not None:
                    d_diag = abs(float(seam_dir @ ax_u))
                    if not (0.42 <= d_diag <= 0.91):
                        continue
                min_rf = 0.88 if both_fg else 0.92
                max_asp = 1.25 if both_fg else 1.12
                if rfill >= min_rf and ls_u / max(ss_u, 1e-6) <= max_asp:
                    ev_cands.append((rfill, a_i, b_i, s_i, u))
            for _, a_i, b_i, s_i, u in sorted(ev_cands, key=lambda t: -t[0]):
                if a_i in merged_drop or b_i in merged_drop or (s_i >= 0 and s_i in merged_drop):
                    continue
                merged_drop.add(a_i)
                merged_drop.add(b_i)
                if s_i >= 0:
                    merged_drop.add(s_i)
                g = shapely.simplify(u, 0.3)
                sc_ev = max(float(fscore[a_i]), float(fscore[b_i]), RULE_SCORE)
                primary_objs.append({
                    "shape_name": "Elbow vuông",
                    "vertices": np.asarray(g.exterior.coords)[:-1].round(3).tolist(),
                    "score": round(sc_ev, 3),
                    "refined": "mitered_ev",
                })
                primary_polys.append(u)

    col_f = {name: idx for idx, name in enumerate(FEATURES)}
    # Align classifier columns to OBJ8_CLASSES (0..7) for safe Bayesian class fusion
    pr_obj8 = np.zeros((n_f, 8), dtype=np.float64)
    if len(pr_f):
        obj8_idx = {name: k for k, name in enumerate(OBJ8_CLASSES)}
        for col_k, cls_id in enumerate(clf.classes_):
            k8 = obj8_idx.get(classes[int(cls_id)])
            if k8 is not None:
                pr_obj8[:, k8] += pr_f[:, col_k]

    for i in range(n_f):
        if i in merged_drop:
            continue
        f = faces[i]
        nm, sc_val, why = fnames[i], float(fscore[i]), fwhy[i]
        cid, ciou, csh = int(f_best_cid[i]), float(f_best_iou[i]), float(f_best_sh[i])
        has_match = (cid >= 0 and cid in cid_pr and ciou >= 0.50)
        pr_c = cid_pr[cid] if has_match else None

        if has_match and why not in ("taper", "three_ways", "symbol", "l_elbow", "end_taper"):
            # F237 / F255 (v11): Whole-face cluster-consensus GroupClsHead FP veto.
            # Uses length-weighted consensus across ALL clusters overlapping the face so a single fragmented
            # sub-cluster on a long dashed duct (e.g. MB TH4 gt#15) never vetoes a confident connected trunk.
            p_none_c = float(pr_c[0])
            if f_pnone_den[i] >= 0.50 * max(float(lf[i]), 1e-9):
                p_none_eff = min(p_none_c, float(f_pnone_num[i] / max(f_pnone_den[i], 1e-9)))
            else:
                p_none_eff = p_none_c
            if not (len(fnb[i]) >= 2 and sc_val >= 0.75):
                if (p_none_eff >= fp_veto_thr and sc_val < fp_veto_fscore_max) or (
                    p_none_eff >= 0.85 and (sc_val < 0.85 or len(fnb[i]) == 0)
                ):
                    continue
        hf_i = float(X[i, col_f["hull_fill"]])
        rf_i = float(X[i, col_f["rect_fill"]])
        crn_i = int(round(X[i, col_f["corners"]]))
        arc_i = float(X[i, col_f["arc_share"]])
        if nm == "none" or sc_val < conf:
            # F245 (v7) / F255 (v11): Gate cid_rescue to physically valid connected faces (len(fnb) >= 1, width >= 1.5 pt)
            if (has_match and (1.0 - float(pr_c[0])) >= none_rescue_thr
                    and len(fnb[i]) >= 1 and float(np.expm1(X[i, col_f["log_short"]])) >= 1.5
                    and not (hf_i < 0.30 and f.area < 15.0)):
                k_c = int(pr_c[1:].argmax()) + 1
                if 1 <= k_c <= 7:
                    r_nm = reducer_kind(f) if k_c in (4, 5) else OBJ8_CLASSES[k_c]
                    if r_nm == "Duct" and (hf_i < 0.72 or (crn_i == 3 and f.area < 25.0)):
                        continue
                    if r_nm in ("Concentric Reducer", "Eccentric Reducer") and (hf_i < 0.78 or rf_i < 0.45 or (crn_i == 3 and f.area < 25.0)):
                        continue
                    if r_nm == "Elbow" and crn_i == 3 and arc_i < 0.15 and f.area < 15.0:
                        continue
                    rescued_cands.append((f, r_nm, float(1.0 - pr_c[0]), "cid_rescue", src[i]))
            continue
        # F249 (v9): Reject low-confidence tiny square Duct noise boxes (e.g. 3.4x3.2 pt VAV/diffuser squares)
        if nm == "Duct" and why == "face" and sc_val < 0.52 and f.area < 16.0 and X[i, col_f["aspect"]] < 1.35:
            continue
        if group_objects is not None and why not in ("taper", "three_ways", "symbol", "l_elbow", "end_taper"):
            # F249 (v9): Gentle Bayesian class fusion (0.75 face_clf + 0.25 GroupClsHead) when matched (IoU>=0.50 or share>=0.70)
            if cid >= 0 and cid in cid_pr and (ciou >= 0.50 or csh >= 0.70):
                p_mix = 0.75 * pr_obj8[i] + 0.25 * cid_pr[cid]
                k_fg = int(np.argmax(p_mix[1:])) + 1
                new_nm = OBJ8_CLASSES[k_fg]
                if new_nm != nm and float(p_mix[k_fg]) >= 0.50:
                    if not (nm in ("Concentric Reducer", "Eccentric Reducer") and new_nm == "Duct") and new_nm != "Tee":
                        nm = reducer_kind(f) if k_fg in (4, 5) else new_nm
                        sc_val = max(sc_val, float(p_mix[k_fg]))
                        why = "bayes"
            if has_match and why != "bayes":
                p_fg_c = 1.0 - float(pr_c[0])
                p_cr = float(pr_c[6]) / max(p_fg_c, 1e-9)
                p_ev = float(pr_c[3]) / max(p_fg_c, 1e-9)
                p_rd = float(pr_c[4] + pr_c[5]) / max(p_fg_c, 1e-9)
                if nm == "Duct" and p_cr >= cr_override_thr:
                    nm, why = "Chân rẽ", "cid_cr"
                elif nm == "Duct" and p_ev >= ev_override_thr:
                    nm, why = "Elbow vuông", "cid_ev"
                elif (nm == "Chân rẽ" and p_rd >= 0.65
                      and X[i, col_f["corners"]] == 4 and X[i, col_f["nb_t_side_cnt"]] == 0):
                    nm, why = reducer_kind(f), "cid_rd"
        # F255 (v11): Physical convexity & non-triangle validity guard (67,658 GT objects: 0/34,792 Ducts & 1/987 Reducers have hf < 0.80)
        if nm == "Duct" and ((hf_i < 0.72 and (f.area < 150.0 or crn_i <= 5)) or (crn_i == 3 and f.area < 25.0 and sc_val < 0.65)):
            continue
        if nm in ("Concentric Reducer", "Eccentric Reducer") and (hf_i < 0.78 or rf_i < 0.45 or (crn_i == 3 and f.area < 25.0)):
            continue
        if nm == "Elbow" and crn_i == 3 and arc_i < 0.15 and f.area < 15.0:
            continue
        if hf_i < 0.30 and f.area < 15.0:
            continue
        if nm != "none" and sc_val >= conf:
            g = shapely.simplify(f, 0.3)
            obj = {
                "shape_name": nm,
                "vertices": np.asarray(g.exterior.coords)[:-1].round(3).tolist(),
                "score": round(sc_val, 3),
                "refined": why,
            }
            if src[i] >= 0:
                obj["item"] = int(src[i])
            primary_objs.append(obj)
            primary_polys.append(f)

    # F237 / F254 (v10) / F255 (v11): Unified Overlap & Containment Resolver on primary_objs.
    #   Case 1 (X-crossing: ia < 0.70 * min_area & iou < 0.65): ALWAYS KEEP BOTH!
    #   Case 2 (same-region dup: iou >= 0.65, with boundary share >= 0.35 for iou < 0.85): keep higher-priority/score.
    #   Case 3 (parent partitioned by >= 2 valid children covering >= 65% of parent): drop parent.
    #   Case 4 (single strictly contained sub-object S inside surviving L, ia >= 0.80 * area_S, area_L >= 1.25 * area_S):
    #     drop S if (a) 3-corner inner V-triangle, (b) low-conf carved sub-face (sc_S < 0.65 & sc_L >= sc_S + 0.05),
    #     (c) floating isolated interior damper/collar box (zero other neighbours anywhere, sc_S < 0.80, not riser/Elbow),
    #     or (d) high-boundary near-duplicate (iou >= 0.45, bsh >= 0.55, ext_nb == 0, sc_S < sc_L).
    if len(primary_objs) >= 2:
        p_arr = np.asarray(primary_polys, dtype=object)
        p_areas = shapely.area(p_arr)
        tr_p = shapely.STRtree(p_arr)
        oi, oj = tr_p.query(p_arr, predicate="intersects")
        mo = oi < oj
        oi, oj = oi[mo], oj[mo]
        drop_ov: set[int] = set()
        children_of: dict[int, list[tuple[int, float]]] = {}
        contained_pairs: list[tuple[int, int, float, float, float]] = []
        if len(oi):
            ia_v = shapely.area(shapely.intersection(p_arr[oi], p_arr[oj]))
            pos_m = ia_v > 0.0
            oi, oj, ia_v = oi[pos_m], oj[pos_m], ia_v[pos_m]
            if len(oi):
                sa_v, sb_v = p_areas[oi], p_areas[oj]
                iou_v = ia_v / np.maximum(sa_v + sb_v - ia_v, 1e-9)
                need_ch = (iou_v < 0.65) & (ia_v >= 0.75 * np.minimum(sa_v, sb_v)) & (np.maximum(sa_v, sb_v) >= 1.35 * np.minimum(sa_v, sb_v))
                need_nest = (iou_v < 0.65) & (ia_v >= 0.80 * np.minimum(sa_v, sb_v)) & (np.maximum(sa_v, sb_v) >= 1.25 * np.minimum(sa_v, sb_v))
                ext_cache: dict[int, object] = {}
                ebuf_cache: dict[int, object] = {}
                per_cache: dict[int, float] = {}

                def _get_ext(idx_p: int):
                    if idx_p not in ext_cache:
                        ex = shapely.get_exterior_ring(p_arr[idx_p])
                        ext_cache[idx_p] = ex
                        per_cache[idx_p] = max(float(shapely.length(ex)), 1e-9)
                        ebuf_cache[idx_p] = shapely.buffer(ex, 0.8, quad_segs=4)
                    return ext_cache[idx_p], per_cache[idx_p], ebuf_cache[idx_p]

                for k_i, (a, b, ia, sa, sb, iou) in enumerate(zip(oi.tolist(), oj.tolist(), ia_v.tolist(), sa_v.tolist(), sb_v.tolist(), iou_v.tolist())):
                    if iou >= 0.65:
                        if iou < 0.85:
                            ea, pa, eba = _get_ext(a)
                            eb, pb, ebb = _get_ext(b)
                            if max(float(ea.intersection(ebb).length) / pa, float(eb.intersection(eba).length) / pb) < 0.35:
                                continue
                        ra = (primary_objs[a].get("refined") in ("taper", "three_ways", "symbol", "l_elbow", "end_taper"), float(primary_objs[a].get("score", 0.0)))
                        rb = (primary_objs[b].get("refined") in ("taper", "three_ways", "symbol", "l_elbow", "end_taper"), float(primary_objs[b].get("score", 0.0)))
                        drop_ov.add(b if ra >= rb else a)
                    else:
                        if need_ch[k_i] or need_nest[k_i]:
                            s_i, l_i = (a, b) if sa <= sb else (b, a)
                            es, ps_l, _ = _get_ext(s_i)
                            _, _, ebl = _get_ext(l_i)
                            bsh = float(es.intersection(ebl).length) / ps_l
                            if need_ch[k_i] and bsh >= 0.30:
                                children_of.setdefault(l_i, []).append((s_i, ia))
                            if need_nest[k_i]:
                                contained_pairs.append((s_i, l_i, ia, iou, bsh))
        for l_i, ch in children_of.items():
            if l_i in drop_ov:
                continue
            ch_valid = [(c, ia) for c, ia in ch if c not in drop_ov]
            if len(ch_valid) >= 2 and sum(ia for _, ia in ch_valid) >= 0.65 * float(p_areas[l_i]):
                drop_ov.add(l_i)
        # Case 4: Resolve remaining single-child strictly contained duplicates (never touching surviving X-crossings!)
        rem_nested = [(s_i, l_i, ia, iou, bsh) for s_i, l_i, ia, iou, bsh in contained_pairs if s_i not in drop_ov and l_i not in drop_ov]
        if rem_nested:
            n_p = len(primary_objs)
            ni, nj = tr_p.query(shapely.buffer(p_arr, 1.5, quad_segs=4), predicate="intersects")
            nb_p: list[set[int]] = [set() for _ in range(n_p)]
            for a_n, b_n in zip(ni.tolist(), nj.tolist()):
                if a_n != b_n and a_n not in drop_ov and b_n not in drop_ov:
                    nb_p[a_n].add(b_n)
            crn_cache: dict[int, int] = {}
            for s_i, l_i, _ia, iou, bsh in rem_nested:
                if s_i in drop_ov or l_i in drop_ov:
                    continue
                if s_i not in crn_cache:
                    crn_cache[s_i] = _corners(primary_polys[s_i])
                crn_s = crn_cache[s_i]
                os_d, ol_d = primary_objs[s_i], primary_objs[l_i]
                sc_s, sc_l = float(os_d.get("score", 0.0)), float(ol_d.get("score", 0.0))
                sn_s = str(os_d.get("shape_name") or "")
                ref_s = str(os_d.get("refined") or "")
                is_rule_s = ref_s in ("taper", "three_ways", "symbol", "l_elbow", "end_taper")
                if crn_s <= 3:
                    drop_ov.add(s_i)
                    continue
                other_nbs = [k for k in nb_p[s_i] if k != l_i and k not in drop_ov]
                # Count neighbours strictly OUTSIDE L (a real stacked/crossing route exits L; interior dampers/sub-boxes do not!)
                ext_nb = sum(
                    1 for k in other_nbs
                    if primary_polys[k].intersection(primary_polys[l_i]).area < 0.50 * max(float(p_areas[k]), 1e-9)
                )
                if ext_nb > 0:
                    continue
                ar_sl = float(p_areas[s_i]) / max(float(p_areas[l_i]), 1e-9)
                # Interior damper/collar box sitting strictly inside L with zero external route connections
                # (keeps large half-ducts with ar_sl >= 0.40 like doc000199 514 pt² inside 1149 pt²)
                if not other_nbs and ar_sl < 0.40 and sn_s not in ("Chân rẽ vuông tròn", "Elbow") and sc_s < 0.72:
                    drop_ov.add(s_i)
                    continue
                if (not is_rule_s) and ar_sl < 0.35 and sn_s != "Chân rẽ vuông tròn" and sc_s < 0.58 and sc_l >= sc_s + 0.08:
                    drop_ov.add(s_i)
                    continue
                # High-overlap wall-sharing near-duplicate covering the same sub-region
                if iou >= 0.50 and bsh >= 0.60 and sc_s < sc_l:
                    drop_ov.add(s_i)
        if drop_ov:
            primary_objs = [o for i, o in enumerate(primary_objs) if i not in drop_ov]
            primary_polys = [p for i, p in enumerate(primary_polys) if i not in drop_ov]

    if rescued_cands:
        ptree = shapely.STRtree(np.asarray(primary_polys, dtype=object)) if primary_polys else None
        accepted_r: list = []
        for f, nm, sc_val, why, item_s in sorted(rescued_cands, key=lambda q: -q[2]):
            bad = False
            if ptree is not None:
                for h in ptree.query(f, predicate="intersects"):
                    hp = primary_polys[int(h)]
                    if hp.intersection(f).area > 0.30 * min(f.area, hp.area):
                        bad = True
                        break
            if not bad and accepted_r:
                for hp in accepted_r:
                    if hp.intersects(f) and hp.intersection(f).area > 0.30 * min(f.area, hp.area):
                        bad = True
                        break
            if bad:
                continue
            g = shapely.simplify(f, 0.3)
            obj = {
                "shape_name": nm,
                "vertices": np.asarray(g.exterior.coords)[:-1].round(3).tolist(),
                "score": round(sc_val, 3),
                "refined": why,
            }
            if item_s >= 0:
                obj["item"] = int(item_s)
            primary_objs.append(obj)
            accepted_r.append(f)

    out_objs = drop_spurious_ducts(primary_objs, DUCT_MIN_W, DUCT_ISO_AREA)
    # F240 (v5): Remove tiny low-confidence micro-sliver Reducer FPs (e.g. 1.9 x 3.7 pt CAD flange slivers, score < 0.50)
    if out_objs:
        from shapely.geometry import Polygon

        kept_r = []
        for o in out_objs:
            if o.get("shape_name") in ("Concentric Reducer", "Eccentric Reducer") and float(o.get("score", 1.0)) < 0.50:
                verts = o.get("vertices") or []
                if len(verts) >= 3:
                    pg = Polygon(np.asarray(verts, float)).buffer(0)
                    if pg.area > 0 and _mrr_axes(pg)[1] < 2.2:
                        continue
            kept_r.append(o)
        out_objs = kept_r
    if len(out_objs) >= 2:
        out_objs = _merge_mitered_square_elbows(out_objs)
    if drop_isolated_fittings and len(out_objs) >= 1:
        # F241 (v6b, confirmed by user 2026-11-25): Physical MEP route connectivity ("duct ko bao giờ nằm 1 mình,
        # nó phải có fitting hoặc chân rẽ gắn vào, nếu nằm đơn lẻ thì hầu hết là noise").
        # Uses pad=3.5 pt to bridge real flange/damper gaps (2.4-3.6 pt on GT25) and NEVER drops valid (Duct + Chân rẽ) pairs.
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from shapely.geometry import Polygon

        ps = [Polygon(np.asarray(o["vertices"], float)).buffer(0) if len(o.get("vertices") or []) >= 3 else Polygon()
              for o in out_objs]
        p_arr = np.asarray(ps, dtype=object)
        tr = shapely.STRtree(p_arr)
        ti, tj = tr.query(shapely.buffer(p_arr, 3.5, quad_segs=4), predicate="intersects")
        mk = ti != tj
        ti, tj = ti[mk], tj[mk]
        n_o = len(out_objs)
        has_nb = np.zeros(n_o, dtype=bool)
        if len(ti):
            has_nb[ti] = True
        _, lab = connected_components(coo_matrix((np.ones(len(ti)), (ti, tj)), shape=(n_o, n_o)), directed=False)
        n_cc = int(lab.max()) + 1
        cc_cnt = np.bincount(lab, minlength=n_cc)
        cc_has_duct = np.zeros(n_cc, dtype=bool)
        cc_has_fit = np.zeros(n_cc, dtype=bool)
        cc_max_asp = np.zeros(n_cc, dtype=np.float64)
        o_axes = [_mrr_axes(p) if p.area > 0 else (0.0, 0.0, np.array([1.0, 0.0])) for p in ps]
        o_bounds = np.array([p.bounds if p.area > 0 else (0.0, 0.0, 0.0, 0.0) for p in ps], dtype=np.float64)
        for k, o in enumerate(out_objs):
            c = int(lab[k])
            ls_k, ss_k, _ = o_axes[k]
            asp_k = ls_k / max(ss_k, 1e-6)
            if o.get("shape_name") == "Duct":
                cc_has_duct[c] = True
                if asp_k > cc_max_asp[c]:
                    cc_max_asp[c] = asp_k
            else:
                cc_has_fit[c] = True
        cc_span = np.zeros(n_cc, dtype=np.float64)
        for c in range(n_cc):
            cm = lab == c
            cc_span[c] = float(max(o_bounds[cm, 2].max() - o_bounds[cm, 0].min(), o_bounds[cm, 3].max() - o_bounds[cm, 1].min()))
        # F245 (v7) / F249 (v9): Cascade-safe check — if an object (or small duct-only cluster) looks isolated ONLY
        # because its adjacent open/dashed duct failed to close a polygon, preserve it when external picked strokes
        # of a large CC (>= 50 pt) attach to it! Also require standalone Ducts to have physical width >= 2.8 pt
        # (eliminates isolated architectural double-line wall strips).
        stroke_cc_ok = np.zeros(n_o, dtype=bool)
        suspects = [
            k for k, o in enumerate(out_objs)
            if (not has_nb[k] and (
                o.get("shape_name") != "Duct"
                or o_axes[k][1] < 2.8
                or not (
                    (o_axes[k][0] >= 18.0 and o_axes[k][0] / max(o_axes[k][1], 1e-6) >= 2.2)
                    or (ps[k].area >= 300.0 and o_axes[k][0] / max(o_axes[k][1], 1e-6) >= 1.8 and float(o.get("score", 0.0)) >= 0.85)
                )
            ))
            or (has_nb[k] and not cc_has_duct[int(lab[k])] and cc_cnt[int(lab[k])] <= 2)
            or (has_nb[k] and not cc_has_fit[int(lab[k])] and cc_span[int(lab[k])] < 60.0 and cc_max_asp[int(lab[k])] < 2.5)
        ]
        if suspects:
            spans = _picked_cc_spans(segs, picked, pad=2.5)
            m_big = picked[segs.owner] & (spans[segs.owner] >= 50.0)
            if m_big.any():
                big_lines = shapely.linestrings(segs.seg[m_big].reshape(-1, 2, 2))
                bt = shapely.STRtree(big_lines)
                for k in suspects:
                    pk_poly = ps[k]
                    if pk_poly.area <= 0:
                        continue
                    outer_b = pk_poly.buffer(3.5, quad_segs=4)
                    inner_b = pk_poly.buffer(1.0, quad_segs=4)
                    hits = bt.query(outer_b, predicate="intersects")
                    if len(hits):
                        ext_len = float(shapely.length(shapely.difference(shapely.intersection(big_lines[hits], outer_b), inner_b)).sum())
                        if ext_len >= 6.0:
                            stroke_cc_ok[k] = True
        kept_conn = []
        for k, o in enumerate(out_objs):
            nm_k = str(o.get("shape_name") or "")
            ls_k, ss_k, _ = o_axes[k]
            asp_k = ls_k / max(ss_k, 1e-6)
            c = int(lab[k])
            if not stroke_cc_ok[k]:
                if not has_nb[k]:
                    # Standalone fitting/Chân rẽ is always dropped; standalone Duct must be a genuine elongated run
                    # (>= 18 pt, asp >= 2.2, width >= 2.8 pt) or large high-conf trunk (>= 300 pt², asp >= 1.8, sc >= 0.85)
                    if nm_k != "Duct" or ss_k < 2.8:
                        continue
                    if not ((ls_k >= 18.0 and asp_k >= 2.2) or (ps[k].area >= 300.0 and asp_k >= 1.8 and float(o.get("score", 0.0)) >= 0.85)):
                        continue
                elif (not cc_has_duct[c]) and cc_cnt[c] <= 2:
                    # 2 touching FCU collars with 0 Ducts
                    continue
                elif (not cc_has_fit[c]) and cc_span[c] < 60.0 and cc_max_asp[c] < 2.5:
                    # Small detached stubby Duct-only cluster with ZERO fittings/Chân rẽ (e.g. touching office desk boxes)
                    continue
            kept_conn.append(o)
        out_objs = kept_conn
    if len(out_objs) >= 2:
        out_objs = _drop_curved_elbow_voids(out_objs, segs, picked, curved)
    if out_objs:
        _tag_square_to_round(out_objs, segs, curved)
    cached_assign = None
    if out_objs and attrs is not None:
        n_before_g = len(out_objs)
        out_objs, g_cls, g_inst = _apply_cad_noise_guardrails(
            out_objs, segs, prob, L, attrs, group_objects, thr, return_assign=True
        )
        if len(out_objs) == n_before_g:
            cached_assign = (g_cls, g_inst)
    if len(out_objs) >= 5:
        n_before_p = len(out_objs)
        out_objs = _drop_giant_room_boxes(out_objs)
        if len(out_objs) != n_before_p:
            cached_assign = None
    if not out_objs:
        return [], np.zeros(n, dtype=np.int8), np.full(n, -1, dtype=np.int32)
    if cached_assign is not None:
        cls_out, inst_out = cached_assign
    else:
        cls_out, inst_out, _ = assign_objects(segs, out_objs, picked, by_edge=True)
    return out_objs, cls_out.astype(np.int8), inst_out.astype(np.int32)


def _drop_giant_room_boxes(objects: list[dict]) -> list[dict]:
    """F249 (v9): Scale-invariant giant room / smoke-plenum-box filter (0 PDF layers).

    Drops an oversized boxy rectangle (``short >= 55 pt``, ``aspect < 2.2``) that is vastly wider than the page's
    normal duct scale (``short >= max(2.5 * page_p90_short, 55.0)`` or ``short >= 110.0``) AND is not connected
    to any comparable-width duct (``max_nb_short < 0.45 * short`` within 2.5 pt). Prevents tree-classifier constant
    extrapolation from calling giant mechanical rooms / smoke plenum boxes (e.g. 198 x 142 pt) a ``Duct`` while
    preserving zoomed detail ducts (``aspect >= 2.2``) and genuine wide trunk runs with matching-width neighbours."""
    import shapely
    from shapely.geometry import Polygon

    if len(objects) < 5:
        return objects
    ps = [
        Polygon(np.asarray(o["vertices"], float)).buffer(0) if len(o.get("vertices") or []) >= 3 else Polygon()
        for o in objects
    ]
    axes = [_mrr_axes(p) if p.area > 0 else (0.0, 0.0, None) for p in ps]
    shorts = np.array([a[1] for a in axes], dtype=np.float64)
    longs = np.array([a[0] for a in axes], dtype=np.float64)
    asps = longs / np.maximum(shorts, 1e-6)
    valid_s = shorts[shorts > 0]
    if len(valid_s) < 5:
        return objects
    p90_s = float(np.percentile(valid_s, 90))
    tr = shapely.STRtree(np.asarray(ps, dtype=object))
    drop: set[int] = set()
    for i in range(len(objects)):
        ss, asp = float(shorts[i]), float(asps[i])
        if ss < 55.0 or asp >= 2.2:
            continue
        if not (ss >= max(2.5 * p90_s, 55.0) or ss >= 110.0):
            continue
        hits = tr.query(ps[i].buffer(2.5, quad_segs=4), predicate="intersects")
        max_nb_s = max((float(shorts[int(j)]) for j in hits if int(j) != i), default=0.0)
        if max_nb_s < 0.45 * ss:
            drop.add(i)
    return [o for i, o in enumerate(objects) if i not in drop] if drop else objects


def _merge_mitered_square_elbows(objects: list[dict]) -> list[dict]:
    """F241 (v6b): Recover mitered / turning-vane ``Elbow vuông`` split along their 45° diagonal into two symmetric
    3- or 4-corner halves (commonly misclassified as a pair of ``Eccentric Reducer`` halves on C3-CH01 / Y1_10-20).
    Merges a touching pair of equal-area halves sharing a long diagonal seam whose union forms a crisp square
    (``aspect <= 1.25``, ``rect_fill >= 0.88``, ``_corners in (4, 5)``) connected to ``>= 2`` other objects."""
    import shapely
    from shapely.geometry import Polygon

    n = len(objects)
    if n < 3:
        return objects
    ps = [Polygon(np.asarray(o["vertices"], float)).buffer(0) if len(o.get("vertices") or []) >= 3 else Polygon()
          for o in objects]
    p_arr = np.asarray(ps, dtype=object)
    areas = shapely.area(p_arr)
    el_ok = np.array([
        (o.get("shape_name") in ("Eccentric Reducer", "Concentric Reducer", "Chân rẽ", "Duct"))
        and (90.0 <= float(areas[i]) <= 360.0)
        and (_corners(ps[i]) in (3, 4))
        for i, o in enumerate(objects)
    ], dtype=bool)
    if int(el_ok.sum()) < 2:
        return objects

    tr = shapely.STRtree(p_arr)
    ti, tj = tr.query(shapely.buffer(p_arr, 1.5, quad_segs=4), predicate="intersects")
    mk = ti != tj
    ti, tj = ti[mk], tj[mk]
    nb_map: list[set[int]] = [set() for _ in range(n)]
    for a, b in zip(ti.tolist(), tj.tolist()):
        nb_map[a].add(b)

    # Candidate touching pairs among eligible halves
    cand_pairs = []
    for a in np.flatnonzero(el_ok).tolist():
        for b in nb_map[a]:
            if b <= a or not el_ok[b]:
                continue
            ra = float(areas[a]) / max(float(areas[b]), 1e-6)
            if not (0.72 <= ra <= 1.38):
                continue
            # Require at least one half to look tapered/triangular (avoid merging two rectangular ducts side-by-side)
            if objects[a].get("shape_name") == "Duct" and objects[b].get("shape_name") == "Duct":
                if _corners(ps[a]) == 4 and _corners(ps[b]) == 4:
                    continue
            inter = ps[a].buffer(0.15).intersection(ps[b].buffer(0.15))
            ia = ps[a].intersection(ps[b]).area
            if ia >= 0.15 * min(float(areas[a]), float(areas[b])):
                continue
            # Shared diagonal seam length
            seam_g = ps[a].boundary.intersection(ps[b].buffer(0.35))
            seam = float(seam_g.length)
            if seam < 0.22 * min(float(ps[a].length), float(ps[b].length)):
                continue
            seam_dir = _shared_dir(seam_g)
            # Must connect to >= 2 external objects (duct corner!)
            ext_nbs = (nb_map[a] | nb_map[b]) - {a, b}
            if len(ext_nbs) < 2:
                continue
            u = shapely.union_all([ps[a].buffer(0.15, join_style="mitre"), ps[b].buffer(0.15, join_style="mitre")]).buffer(-0.15, join_style="mitre")
            if u.geom_type != "Polygon" or not (190.0 <= float(u.area) <= 680.0):
                continue
            if _corners(u) not in (4, 5, 6):
                continue
            mrr_a = max(float(u.minimum_rotated_rectangle.area), 1e-9)
            rfill = float(u.area) / mrr_a
            ls_u, ss_u, ax_u = _mrr_axes(u)
            if seam_dir is not None and not (0.42 <= abs(float(seam_dir @ ax_u)) <= 0.91):
                continue
            if rfill >= 0.88 and ls_u / max(ss_u, 1e-6) <= 1.25:
                cand_pairs.append((rfill, a, b, u))

    if not cand_pairs:
        return objects
    cand_pairs.sort(key=lambda t: -t[0])
    used: set[int] = set()
    merged: list[dict] = []
    for _, a, b, u in cand_pairs:
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        g = shapely.simplify(u, 0.3)
        sc = max(float(objects[a].get("score", 0.0)), float(objects[b].get("score", 0.0)), RULE_SCORE)
        merged.append({
            "shape_name": "Elbow vuông",
            "vertices": np.asarray(g.exterior.coords)[:-1].round(3).tolist(),
            "score": round(sc, 3),
            "refined": "mitered_ev",
        })
    if not merged:
        return objects
    return [o for i, o in enumerate(objects) if i not in used] + merged


def _picked_cc_spans(segs, picked: np.ndarray, pad: float = 2.0) -> np.ndarray:
    """Fast per-shape bbox span of the picked connected component (via bbox dwithin pad)."""
    import shapely
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(picked)
    spans = np.zeros(n, dtype=np.float64)
    pk = np.flatnonzero(picked)
    if not len(pk):
        return spans
    off = segs.shape_offsets
    seg = segs.seg
    boxes = []
    valid_pk = []
    for s in pk.tolist():
        a, b = int(off[s]), int(off[s + 1])
        if a >= b:
            continue
        sub = seg[a:b]
        boxes.append((
            float(min(sub[:, 0].min(), sub[:, 2].min())),
            float(min(sub[:, 1].min(), sub[:, 3].min())),
            float(max(sub[:, 0].max(), sub[:, 2].max())),
            float(max(sub[:, 1].max(), sub[:, 3].max())),
        ))
        valid_pk.append(s)
    if not boxes:
        return spans
    b_arr = np.asarray(boxes, dtype=np.float64)
    geoms = shapely.box(b_arr[:, 0] - pad, b_arr[:, 1] - pad, b_arr[:, 2] + pad, b_arr[:, 3] + pad)
    ii, jj = shapely.STRtree(geoms).query(geoms, predicate="intersects")
    m = ii < jj
    ii, jj = ii[m], jj[m]
    m_n = len(valid_pk)
    _, lab = connected_components(coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(m_n, m_n)), directed=False)
    for c_id in range(int(lab.max()) + 1):
        cm = lab == c_id
        sp = float(max(b_arr[cm, 2].max() - b_arr[cm, 0].min(), b_arr[cm, 3].max() - b_arr[cm, 1].min()))
        for idx_l in np.flatnonzero(cm).tolist():
            spans[valid_pk[idx_l]] = sp
    return spans


def _apply_cad_noise_guardrails(
    objects: list[dict],
    segs,
    prob: np.ndarray,
    length: np.ndarray,
    attrs: dict,
    group_objects: dict | None,
    thr: float,
    *,
    return_assign: bool = False,
):
    """F241 (v6): Layer-free CAD OOD noise guardrails using stroke widths & unpicked airflow-arrow geometry:
      1. Page-level dominant trunk-width gate: when >= 75% of high-confidence (p >= 0.95) large-CC (>= 120 pt)
         duct length shares a dominant stroke width w_dom >= 0.35 pt (e.g. 0.48 pt on VNG-2 [12]), suppress
         strictly thinner (0.01 < w <= 0.65 * w_dom, e.g. 0.24 pt furniture) non-trunk clusters (cc_span < 150 pt)
         that lack real duct fittings (no Elbow/Reducer/Tee in their object component) or have p_none >= 0.10.
      2. Sidewall grille collar filter: drop small side-attached Chân rẽ (area < 20, ss < 2.2, nb <= 1) sitting
         within 6 pt of BOTH an unpicked 5-segment zigzag airflow arrow AND an unpicked filled triangle arrowhead
         (removes 66-76% of JRP-MVAC-EAG sidewall grilles on 0875 with 0 loss across GT25 & VW8).
    """
    import shapely
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from shapely.geometry import Polygon

    from .objects import assign_objects

    if not objects:
        return (objects, None, None) if return_assign else objects
    picked = prob >= thr
    ocls, oinst, _ = assign_objects(segs, objects, picked, by_edge=True)
    ps = [Polygon(np.asarray(o["vertices"], float)).buffer(0) if len(o.get("vertices") or []) >= 3 else Polygon()
          for o in objects]
    p_arr = np.asarray(ps, dtype=object)
    n_o = len(objects)
    nb = np.zeros(n_o, dtype=np.int64)
    ti, tj = (shapely.STRtree(p_arr).query(shapely.buffer(p_arr, 1.8, quad_segs=4), predicate="intersects")
              if n_o >= 2 else (np.zeros(0, np.int64), np.zeros(0, np.int64)))
    mk = ti != tj
    ti, tj = ti[mk], tj[mk]
    if len(ti):
        nb = np.bincount(ti, minlength=n_o)
    _, obj_lab = connected_components(coo_matrix((np.ones(len(ti)), (ti, tj)), shape=(n_o, n_o)), directed=False)
    cc_has_fitting = np.zeros(int(obj_lab.max()) + 1, dtype=bool)
    for k, o in enumerate(objects):
        if o.get("shape_name") in ("Elbow", "Elbow vuông", "Concentric Reducer", "Eccentric Reducer", "Reducer vuông tròn", "Tee"):
            cc_has_fitting[int(obj_lab[k])] = True

    # Group assigned shapes by object id ONCE in O(N_fg log N_fg) instead of 2 x O(n_o * N_shapes) scans!
    valid_oi = np.flatnonzero(oinst >= 0)
    oi_order = valid_oi[np.argsort(oinst[valid_oi], kind="quicksort")] if len(valid_oi) else np.zeros(0, dtype=np.int64)
    oi_vals = oinst[oi_order] if len(oi_order) else np.zeros(0, dtype=np.int64)
    oi_ranges: dict[int, tuple[int, int]] = {}
    if len(oi_vals):
        oi_chg = np.flatnonzero(oi_vals[1:] != oi_vals[:-1]) + 1
        oi_st = np.r_[0, oi_chg]
        oi_en = np.r_[oi_chg, len(oi_vals)]
        for obj_i, a_i, b_i in zip(oi_vals[oi_st].tolist(), oi_st.tolist(), oi_en.tolist()):
            oi_ranges[obj_i] = (a_i, b_i)

    p_none = np.full(n_o, -1.0, dtype=np.float64)
    if group_objects is not None:
        ginst = np.asarray(group_objects["instance_id"], dtype=np.int64)
        c2p = {int(c): pr for c, pr in zip(group_objects["cluster_ids"].tolist(), group_objects["cluster_probs"])}
        for i, (a_i, b_i) in oi_ranges.items():
            idx_i = oi_order[a_i:b_i]
            gi_i = ginst[idx_i]
            ok_g = gi_i >= 0
            if ok_g.any():
                uc, inv = np.unique(gi_i[ok_g], return_inverse=True)
                dc = int(uc[int(np.argmax(np.bincount(inv, weights=length[idx_i[ok_g]])))])
                if dc in c2p:
                    p_none[i] = float(c2p[dc][0])

    drop: set[int] = set()

    # 1. Page-level dominant stroke-width outlier gate (v6c / F243: same-width thin-object CCs so an overhead
    #    0.48pt duct crossing over 0.24pt furniture NEVER lends its 1200pt span or Elbow to the furniture below;
    #    protects 0.00pt hairlines, real thin branch routes with Elbow/Tee or asp>=2.8, and thin collars on thick trunks)
    width_raw = attrs.get("width")
    if width_raw is not None:
        w = np.asarray(width_raw, dtype=np.float64)
        cc_spans = _picked_cc_spans(segs, picked, pad=2.0)
        hi_big = (prob >= 0.95) & (cc_spans >= 120.0) & (w > 0.01)
        if hi_big.any() and float(length[hi_big].sum()) >= 500.0:
            wb = np.round(w[hi_big] / 0.04) * 0.04
            uw, inv = np.unique(wb, return_inverse=True)
            wl = np.bincount(inv, weights=length[hi_big])
            best_k = int(np.argmax(wl))
            w_dom = float(uw[best_k])
            dom_sh = float(wl[best_k] / max(wl.sum(), 1e-9))
            if w_dom >= 0.35 and dom_sh >= 0.75:
                obj_w = np.full(n_o, -1.0, dtype=np.float64)
                for i, (a_i, b_i) in oi_ranges.items():
                    obj_w[i] = float(np.median(w[oi_order[a_i:b_i]]))
                is_thin = (obj_w > 0.01) & (obj_w <= 0.65 * w_dom)
                if is_thin.any():
                    thin_idx = np.flatnonzero(is_thin)
                    p_thin = np.asarray([ps[i] for i in thin_idx], dtype=object)
                    m_t = len(thin_idx)
                    if m_t >= 2:
                        tti, ttj = shapely.STRtree(p_thin).query(shapely.buffer(p_thin, 2.0, quad_segs=4), predicate="intersects")
                        tmk = tti != ttj
                        tti, ttj = tti[tmk], ttj[tmk]
                        _, tlab = connected_components(coo_matrix((np.ones(len(tti)), (tti, ttj)), shape=(m_t, m_t)), directed=False)
                    else:
                        tlab = np.zeros(m_t, dtype=np.int64)
                    n_tcc = int(tlab.max()) + 1
                    tcc_cnt = np.bincount(tlab, minlength=n_tcc)
                    tcc_has_elbow = np.zeros(n_tcc, dtype=bool)
                    tcc_max_asp = np.zeros(n_tcc, dtype=np.float64)
                    for loc_k, gi in enumerate(thin_idx.tolist()):
                        c_id = int(tlab[loc_k])
                        if objects[gi].get("shape_name") in ("Elbow", "Elbow vuông", "Tee", "Concentric Reducer", "Eccentric Reducer", "Reducer vuông tròn"):
                            tcc_has_elbow[c_id] = True
                        if ps[gi].area > 0:
                            ls_g, ss_g, _ = _mrr_axes(ps[gi])
                            asp_g = ls_g / max(ss_g, 1e-6)
                            if asp_g > tcc_max_asp[c_id]:
                                tcc_max_asp[c_id] = asp_g
                    thick_idx = np.flatnonzero(obj_w > 0.65 * w_dom)
                    thick_tree = shapely.STRtree(np.asarray([ps[i] for i in thick_idx], dtype=object)) if len(thick_idx) else None
                    for loc_k, gi in enumerate(thin_idx.tolist()):
                        if objects[gi].get("shape_name") not in ("Duct", "Chân rẽ", "Chân rẽ vuông tròn", "Concentric Reducer", "Eccentric Reducer"):
                            continue
                        c_id = int(tlab[loc_k])
                        if tcc_has_elbow[c_id] or tcc_max_asp[c_id] >= 2.8:
                            continue
                        if tcc_cnt[c_id] <= 2 and thick_tree is not None and ps[gi].area > 0:
                            if len(thick_tree.query(ps[gi].buffer(2.0), predicate="intersects")):
                                continue
                        drop.add(gi)

    # 2. Strict sidewall grille filter (zigzag 5-segment airflow arrow + filled triangle arrowhead within 6 pt)
    op_counts = attrs.get("op_counts")
    type_code = attrs.get("type_code")
    fill_op = attrs.get("fill_opacity")
    if op_counts is not None and (type_code is not None or fill_op is not None):
        op_l = np.asarray(op_counts)[:, 0]
        is_fill = np.zeros(len(picked), dtype=bool)
        if type_code is not None:
            is_fill |= np.asarray(type_code) >= 1
        if fill_op is not None:
            is_fill |= np.asarray(fill_op) > 0
        off = segs.shape_offsets
        seg = segs.seg
        cand_s = np.flatnonzero((~picked) & ((op_l == 3) | (op_l == 4) | (op_l == 5)))
        if len(cand_s):
            a_c, b_c = off[cand_s], off[cand_s + 1]
            has_seg = b_c > a_c
            cand_s, a_c, b_c = cand_s[has_seg], a_c[has_seg], b_c[has_seg]
        arr_boxes, zig_boxes = [], []
        if len(cand_s):
            # Vectorized bbox lookup via precomputed attrs["bbox"] (or bulk numpy slice fallback), 100% bit-exact!
            bb = attrs.get("bbox")
            if bb is not None and len(bb) == len(picked):
                c_bb = np.asarray(bb[cand_s], dtype=np.float64)
                x0_c, y0_c, x1_c, y1_c = c_bb[:, 0], c_bb[:, 1], c_bb[:, 2], c_bb[:, 3]
            else:
                x0_c = np.empty(len(cand_s), dtype=np.float64)
                y0_c = np.empty(len(cand_s), dtype=np.float64)
                x1_c = np.empty(len(cand_s), dtype=np.float64)
                y1_c = np.empty(len(cand_s), dtype=np.float64)
                for idx_c, (aa, bb_i) in enumerate(zip(a_c.tolist(), b_c.tolist())):
                    sub = seg[aa:bb_i]
                    x0_c[idx_c] = min(float(sub[:, 0].min()), float(sub[:, 2].min()))
                    x1_c[idx_c] = max(float(sub[:, 0].max()), float(sub[:, 2].max()))
                    y0_c[idx_c] = min(float(sub[:, 1].min()), float(sub[:, 3].min()))
                    y1_c[idx_c] = max(float(sub[:, 1].max()), float(sub[:, 3].max()))
            bw_c = x1_c - x0_c
            bh_c = y1_c - y0_c
            is_f_c = is_fill[cand_s]
            op_c = op_l[cand_s]
            m_arr = is_f_c & ((op_c == 3) | (op_c == 4)) & (bw_c <= 6.0) & (bh_c <= 6.0) & ((bw_c * bh_c) >= 0.15)
            m_zig = (~is_f_c) & (op_c == 5) & (np.maximum(bw_c, bh_c) >= 1.0) & (np.maximum(bw_c, bh_c) <= 8.0) & (np.minimum(bw_c, bh_c) >= 0.5)
            if m_arr.any():
                arr_boxes = shapely.box(x0_c[m_arr], y0_c[m_arr], x1_c[m_arr], y1_c[m_arr])
            if m_zig.any():
                zig_boxes = shapely.box(x0_c[m_zig], y0_c[m_zig], x1_c[m_zig], y1_c[m_zig])
        if len(arr_boxes) and len(zig_boxes):
            atree = shapely.STRtree(arr_boxes)
            ztree = shapely.STRtree(zig_boxes)
            for i, o in enumerate(objects):
                if i in drop or ps[i].area <= 0 or ps[i].area >= 20.0 or nb[i] > 1:
                    continue
                if o.get("shape_name") not in ("Chân rẽ", "Chân rẽ vuông tròn"):
                    continue
                ls, ss, _ = _mrr_axes(ps[i])
                if ss < 1.8 and (ls / max(ss, 1e-6)) >= 2.8 and len(atree.query(ps[i], predicate="dwithin", distance=6.0)) and len(ztree.query(ps[i], predicate="dwithin", distance=6.0)):
                    drop.add(i)

    if not drop:
        return (objects, ocls, oinst) if return_assign else objects
    kept = [o for i, o in enumerate(objects) if i not in drop]
    return (kept, None, None) if return_assign else kept


def _tag_square_to_round(objects: list[dict], segs, curved: np.ndarray) -> None:
    """F240 (v5) / F245 (v7): Identify square-to-round transitions (``Chân rẽ vuông tròn`` & ``Reducer vuông tròn``)
    from strictly-contained interior CAD transition creases — specifically wide-base-anchored inscribed V-triangles
    (triangle sharing a base with the trapezoid, confirmed on 258/258 GT objects), X diagonals, or interior circle arcs
    across ALL page strokes, covering ``Chân rẽ``, ``Concentric Reducer``, and ``Eccentric Reducer``.
    """
    import shapely
    from shapely.geometry import LineString, Point, Polygon

    cand_idx = [
        i for i, o in enumerate(objects)
        if o.get("shape_name") in ("Chân rẽ", "Concentric Reducer", "Eccentric Reducer") and len(o.get("vertices") or []) >= 3
    ]
    if not cand_idx or not len(segs.seg):
        return
    cand_polys: list[tuple[int, object, np.ndarray]] = []
    for oi in cand_idx:
        g = Polygon(np.asarray(objects[oi]["vertices"], float)).buffer(0)
        if g.geom_type != "Polygon" or g.area <= 2.5:
            continue
        v = np.asarray(shapely.simplify(g, 0.5).exterior.coords)[:-1]
        if len(v) != 4:
            continue
        cand_polys.append((oi, g, v))
    if not cand_polys:
        return

    all_seg = segs.seg
    fb = np.asarray([g.bounds for _, g, _ in cand_polys], dtype=np.float64)
    gx0 = float(fb[:, 0].min()) - 2.0
    gy0 = float(fb[:, 1].min()) - 2.0
    gx1 = float(fb[:, 2].max()) + 2.0
    gy1 = float(fb[:, 3].max()) + 2.0
    sx0 = np.minimum(all_seg[:, 0], all_seg[:, 2])
    sx1 = np.maximum(all_seg[:, 0], all_seg[:, 2])
    sy0 = np.minimum(all_seg[:, 1], all_seg[:, 3])
    sy1 = np.maximum(all_seg[:, 1], all_seg[:, 3])
    near_f = (sx1 >= gx0) & (sx0 <= gx1) & (sy1 >= gy0) & (sy0 <= gy1)
    if not near_f.any():
        return
    cell = 8.0
    nx = min(max(int((gx1 - gx0) / cell) + 2, 1), 4096)
    ny = min(max(int((gy1 - gy0) / cell) + 2, 1), 4096)
    inv_cx = nx / max(gx1 - gx0, 1.0)
    inv_cy = ny / max(gy1 - gy0, 1.0)
    occ = np.zeros((ny, nx), dtype=bool)
    fx0_c = np.clip(((fb[:, 0] - 2.0 - gx0) * inv_cx).astype(np.int64), 0, nx - 1)
    fx1_c = np.clip(((fb[:, 2] + 2.0 - gx0) * inv_cx).astype(np.int64), 0, nx - 1)
    fy0_c = np.clip(((fb[:, 1] - 2.0 - gy0) * inv_cy).astype(np.int64), 0, ny - 1)
    fy1_c = np.clip(((fb[:, 3] + 2.0 - gy0) * inv_cy).astype(np.int64), 0, ny - 1)
    for k_f in range(len(cand_polys)):
        occ[fy0_c[k_f]:fy1_c[k_f] + 1, fx0_c[k_f]:fx1_c[k_f] + 1] = True
    cand_all = np.flatnonzero(near_f)
    cx_a = np.clip(((sx0[cand_all] - gx0) * inv_cx).astype(np.int64), 0, nx - 1)
    cx_b = np.clip(((sx1[cand_all] - gx0) * inv_cx).astype(np.int64), 0, nx - 1)
    cy_a = np.clip(((sy0[cand_all] - gy0) * inv_cy).astype(np.int64), 0, ny - 1)
    cy_b = np.clip(((sy1[cand_all] - gy0) * inv_cy).astype(np.int64), 0, ny - 1)
    hit_grid = (
        occ[cy_a, cx_a] | occ[cy_b, cx_b] | occ[cy_a, cx_b] | occ[cy_b, cx_a]
        | ((cx_b - cx_a) > 1) | ((cy_b - cy_a) > 1)
    )
    sub_idx = cand_all[hit_grid]
    if not len(sub_idx):
        return
    seg = all_seg[sub_idx]
    seg_cur = np.asarray(curved, dtype=bool)[segs.owner[sub_idx]]
    lines = shapely.linestrings(seg.reshape(-1, 2, 2))
    tree = shapely.STRtree(lines)

    for oi, g, v in cand_polys:
        o = objects[oi]
        nm = o["shape_name"]
        in_buf = g.buffer(-0.35, quad_segs=4)
        out_buf = g.buffer(0.6, quad_segs=4)
        if in_buf.is_empty:
            continue
        hits = tree.query(in_buf, predicate="intersects")
        if not len(hits):
            continue
        ev = np.roll(v, -1, 0) - v
        elen = np.hypot(ev[:, 0], ev[:, 1])
        e_ang = np.degrees(np.arctan2(ev[:, 1], ev[:, 0])) % 180.0

        def _egap(a_i, b_i):
            d = abs((e_ang[a_i] - e_ang[b_i]) % 180.0)
            return min(d, 180.0 - d)

        g02, g13 = _egap(0, 2), _egap(1, 3)
        # On rectangles both opposite pairs are parallel -> check both orientations (independent of ring start vertex);
        # on trapezoids check the single parallel pair (wide base -> narrow opposite base).
        if g02 <= 8.0 and g13 <= 8.0:
            par_list = [(0, 2), (1, 3)]
            best_par_gap = min(g02, g13)
        else:
            par_list = [(0, 2) if g02 <= g13 else (1, 3)]
            best_par_gap = min(g02, g13)
        ext = g.exterior

        diags: list[tuple[np.ndarray, np.ndarray, float, float]] = []
        n_any_in = 0
        n_cur_in = 0
        corner_hits = 0
        for h in hits:
            h = int(h)
            p0, p1 = seg[h, :2], seg[h, 2:]
            fl = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            if fl < 1.0:
                continue
            if float(shapely.length(shapely.intersection(lines[h], out_buf))) < 0.90 * fl:
                continue
            if float(shapely.length(shapely.intersection(lines[h], in_buf))) < 0.45 * fl:
                continue
            n_any_in += 1
            if seg_cur[h]:
                n_cur_in += 1
            ang = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])) % 180.0)
            d = np.abs(e_ang - ang)
            if float(np.minimum(d, 180.0 - d).min()) < 10.0:
                continue
            if float(ext.distance(Point(p0[0], p0[1]))) > 1.5 or float(ext.distance(Point(p1[0], p1[1]))) > 1.5:
                continue
            c0 = float(np.hypot(v[:, 0] - p0[0], v[:, 1] - p0[1]).min())
            c1 = float(np.hypot(v[:, 0] - p1[0], v[:, 1] - p1[1]).min())
            if c0 <= 1.6 or c1 <= 1.6:
                corner_hits += 1
                diags.append((p0, p1, fl, ang))

        def _has_inscribed_v_triangle(strict_meet: bool = False, min_len_ratio: float = 0.80) -> bool:
            if len(diags) < 2:
                return False
            for par in par_list:
                # Try wide base first (and on near-equal rectangular sides, try both bases as the triangle's base!)
                e_a, e_b = (par[0], par[1]) if elen[par[0]] >= elen[par[1]] else (par[1], par[0])
                orientations = [(e_a, e_b), (e_b, e_a)] if elen[e_b] >= 0.90 * elen[e_a] else [(e_a, e_b)]
                for wide_e, nar_e in orientations:
                    w0, w1 = v[wide_e], v[(wide_e + 1) % 4]
                    n0, n1 = v[nar_e], v[(nar_e + 1) % 4]
                    nar_line = LineString([n0, n1])
                    wide_len = float(elen[wide_e])
                    wide_v0, wide_v1 = [], []
                    for p0, p1, fl, ang in diags:
                        for pa, pb in ((p0, p1), (p1, p0)):
                            if float(np.hypot(pa[0] - w0[0], pa[1] - w0[1])) <= 1.6 and float(nar_line.distance(Point(pb[0], pb[1]))) <= 1.8:
                                wide_v0.append((pa, pb, fl, ang))
                            elif float(np.hypot(pa[0] - w1[0], pa[1] - w1[1])) <= 1.6 and float(nar_line.distance(Point(pb[0], pb[1]))) <= 1.8:
                                wide_v1.append((pa, pb, fl, ang))
                    for _, nb0, l0, ang0 in wide_v0:
                        for _, nb1, l1, ang1 in wide_v1:
                            da = min(abs(ang0 - ang1), 180.0 - abs(ang0 - ang1))
                            if da < 15.0:
                                continue
                            d_nar = float(np.hypot(nb0[0] - nb1[0], nb0[1] - nb1[1]))
                            if strict_meet:
                                if d_nar <= 2.0 and min(l0, l1) / max(l0, l1, 1e-6) >= min_len_ratio:
                                    return True
                            else:
                                if d_nar <= 2.0 or d_nar <= 0.68 * max(wide_len, 1e-6):
                                    return True
            return False

        if nm == "Chân rẽ":
            if n_cur_in >= 2 or (_has_inscribed_v_triangle(strict_meet=False) and n_any_in <= 8):
                o["shape_name"] = "Chân rẽ vuông tròn"
            elif 2 <= len(diags) <= 8 and corner_hits >= 2 and n_any_in <= 8:
                a0 = diags[0][3]
                if any(min(abs(q[3] - a0), 180.0 - abs(q[3] - a0)) >= 15.0 for q in diags[1:]):
                    o["shape_name"] = "Chân rẽ vuông tròn"
        else:
            min_lr = 0.55 if nm == "Eccentric Reducer" else 0.80
            if n_cur_in >= 2 and n_any_in <= 5:
                o["shape_name"] = "Reducer vuông tròn"
            elif _has_inscribed_v_triangle(strict_meet=True, min_len_ratio=min_lr) and n_any_in <= 3 and best_par_gap <= 10.0:
                o["shape_name"] = "Reducer vuông tròn"


def _has_curved_shared_part(geom, min_len: float = 2.0, ratio: float = 1.03) -> bool:
    """True when a shared-boundary intersection contains a genuinely curved arc (length > ratio * chord),
    distinguishing curved flank contact (void pocket between elbows) from straight flange contact (S-bend)."""
    import shapely

    for part in shapely.get_parts(geom):
        if part.geom_type in ("LineString", "LinearRing") and part.length >= min_len:
            coords = np.asarray(part.coords, dtype=np.float64)
            if len(coords) >= 3:
                chord = float(np.hypot(coords[-1, 0] - coords[0, 0], coords[-1, 1] - coords[0, 1]))
                if float(part.length) > ratio * max(chord, 1e-6):
                    return True
    return False


def _drop_curved_elbow_voids(objects: list[dict], segs, picked: np.ndarray, curved: np.ndarray) -> list[dict]:
    """F239: Remove empty air pockets trapped between curved Elbows (cross-item ``graph_faces`` only, ``item < 0``).

    A real HVAC Elbow has curved inner/outer walls facing free air (meeting neighbours at straight flanges);
    an enclosed air pocket between two Elbows is misclassified as ``Elbow`` because its boundary borrows the
    curved arcs of neighbouring disjoint Elbows. Drops a cross-item ``Elbow`` ONLY when it shares a genuinely
    curved arc with disjoint ``Elbow`` neighbour(s) AND at least one strict void criterion holds:
      - curved wall contact with >= 2 disjoint Elbows (pocket squeezed between 2 opposing Elbows), OR
      - >= 75% of its total perimeter is shared with surrounding objects (enclosed pocket), OR
      - >= 60% of its Bezier curved stroke length is shared with disjoint Elbow neighbour(s).
    Preserves 100% of single-item Elbows, S-bends (straight shared flange), concentric parallel Elbows (1 curved
    wall shared, <75% total perimeter, <=50% curved boundary), and crossing-duct Elbows (overlapping).
    """
    import shapely
    from shapely.geometry import Polygon

    el_idx = [i for i, o in enumerate(objects) if o.get("shape_name") == "Elbow" and len(o.get("vertices") or []) >= 3]
    if len(el_idx) < 2:
        return objects
    ps = [Polygon(np.asarray(o["vertices"], float)).buffer(0) if len(o.get("vertices") or []) >= 3 else Polygon()
          for o in objects]
    arr = np.asarray(ps, dtype=object)
    areas = shapely.area(arr)
    exts = shapely.get_exterior_ring(arr)
    perim = np.maximum(shapely.length(exts), 1e-9)
    bufs = shapely.buffer(arr, 0.5, quad_segs=4)

    # All-object neighbours (for total enclosed-perimeter share)
    tr = shapely.STRtree(arr)
    pi, pj = tr.query(bufs, predicate="intersects")
    mk = pi != pj
    pi, pj = pi[mk], pj[mk]
    if not len(pi):
        return objects
    ia_all = shapely.area(shapely.intersection(arr[pi], arr[pj]))
    sh_geom = shapely.intersection(exts[pi], bufs[pj])
    sh_len = shapely.length(sh_geom)
    ok = sh_len >= 1.5
    pi, pj, ia_all, sh_geom, sh_len = pi[ok], pj[ok], ia_all[ok], sh_geom[ok], sh_len[ok]

    el_set = set(el_idx)
    all_nbs: dict[int, list[int]] = {}
    disjoint_el_nbs: dict[int, list[int]] = {}
    curved_el_nbs: dict[int, set[int]] = {}
    for a, b, ia_v, sg in zip(pi.tolist(), pj.tolist(), ia_all.tolist(), sh_geom.tolist()):
        if a not in el_set or objects[a].get("item", -1) >= 0:
            continue
        all_nbs.setdefault(a, []).append(b)
        if b in el_set and ia_v < 0.15 * min(float(areas[a]), float(areas[b])):
            disjoint_el_nbs.setdefault(a, []).append(b)
            if _has_curved_shared_part(sg):
                curved_el_nbs.setdefault(a, set()).add(b)

    if not disjoint_el_nbs:
        return objects

    m_cur = (picked & curved)[segs.owner]
    cur_lines = shapely.linestrings(segs.seg[m_cur].reshape(-1, 2, 2)) if m_cur.any() else np.empty(0, dtype=object)
    ctree = shapely.STRtree(cur_lines) if len(cur_lines) else None

    drop: set[int] = set()
    for a, el_nb in disjoint_el_nbs.items():
        if areas[a] <= 0.0:
            continue
        c_nbs = curved_el_nbs.get(a, set())
        if len(c_nbs) >= 2:
            drop.add(a)
            continue
        if c_nbs:
            nb_u_all = shapely.union_all(bufs[all_nbs[a]])
            sh_frac = float(shapely.length(shapely.intersection(exts[a], nb_u_all)) / perim[a])
            if sh_frac >= 0.75:
                drop.add(a)
                continue
        if ctree is not None:
            eb = exts[a].buffer(0.8, quad_segs=4)
            hc = ctree.query(eb, predicate="intersects")
            if len(hc):
                c_on = shapely.intersection(cur_lines[hc], eb)
                c_tot = float(shapely.length(c_on).sum())
                if c_tot > 2.0:
                    nb_u_el = shapely.union_all(bufs[el_nb])
                    if float(shapely.length(shapely.intersection(c_on, nb_u_el)).sum()) / c_tot >= 0.60:
                        drop.add(a)

    if not drop:
        return objects
    return [o for i, o in enumerate(objects) if i not in drop]


def segment_objects(objects: list[dict], segs, picked: np.ndarray, tol: float = 0.8) -> np.ndarray:
    """Per segment, the object it draws (-1 none), F156 / F237: an item drawing a duct and its reducer gives its segments to two objects.

    F237: Uses actual segment-to-outline overlap length (bulk C ``shapely.intersection``) instead of midpoint-only proximity so
    segments extending past a polygon corner into dropped strips or running across two objects never stick out as jagged colour stubs.
    A picked segment belongs to an object when >= 55% of its length runs along the object's ``tol``-buffered outline (ranking higher
    overlap share first, then ``item`` ownership, then smaller area), or else >= 70% lies inside the ``tol``-buffered polygon
    (interior symbol strokes).
    """
    import shapely
    from shapely.geometry import Polygon

    out = np.full(len(segs.seg), -1, np.int64)
    polys, idx = [], []
    for i, o in enumerate(objects):
        v = o.get("vertices") or []
        if len(v) >= 3:
            g = Polygon(np.asarray(v, float)).buffer(0)
            if g.area > 0:
                polys.append(g)
                idx.append(i)
    if not polys:
        return out
    polys_arr = np.asarray(polys, dtype=object)
    idx_arr = np.asarray(idx, dtype=np.int64)
    area = shapely.area(polys_arr)
    m = np.flatnonzero(picked[segs.owner])
    if not len(m):
        return out
    s_m = segs.seg[m]
    seg_lines = shapely.linestrings(s_m.reshape(-1, 2, 2))
    slen_raw = np.hypot(s_m[:, 2] - s_m[:, 0], s_m[:, 3] - s_m[:, 1])
    slen = np.maximum(slen_raw, 1e-9)
    poly_bufs = shapely.buffer(polys_arr, tol, quad_segs=4)
    tree = shapely.STRtree(poly_bufs)
    pi, gi = tree.query(seg_lines, predicate="intersects")
    if not len(pi):
        return out

    bound_bufs = shapely.buffer(shapely.get_exterior_ring(polys_arr), tol, quad_segs=4)
    on_len = _covered_line_inter_len(seg_lines, slen_raw, bound_bufs, pi, gi)
    on_frac = on_len / slen[pi]

    on_ok = on_frac >= 0.55
    # F239b: Only symbol-bearing classes (Chân rẽ / Reducer vuông tròn) have genuine interior X/V/circle strokes;
    # Duct / Elbow / Reducer / Tee are strictly outline-only, so interior cross-hatch or crossing lines must never attach!
    _interior_cls = frozenset(("Chân rẽ", "Chân rẽ vuông tròn", "Reducer vuông tròn"))
    allow_in = np.asarray([objects[i].get("shape_name") in _interior_cls for i in idx], dtype=bool)
    need_in = (~on_ok) & allow_in[gi]
    in_frac = np.zeros(len(pi), dtype=np.float64)
    if need_in.any():
        in_len = _covered_line_inter_len(seg_lines, slen_raw, poly_bufs, pi[need_in], gi[need_in])
        in_frac[need_in] = in_len / slen[pi[need_in]]
    in_ok = need_in & (in_frac >= 0.70)
    keep = on_ok | in_ok
    if not keep.any():
        return out
    pi, gi = pi[keep], gi[keep]
    on_ok, on_frac, in_frac = on_ok[keep], on_frac[keep], in_frac[keep]

    item = np.asarray([objects[i].get("item", -1) for i in idx], dtype=np.int64)
    own = segs.owner[m]
    # Primary rank: boundary match beats interior-only; higher overlap bucket (5% bins) beats partial corner touch;
    # item ownership breaks near-ties; smaller area breaks exact ties on shared walls.
    frac_bin = np.floor(np.where(on_ok, on_frac, in_frac) * 20.0)
    score = (
        np.where(on_ok, 0.0, 1e15)
        - frac_bin * 1e12
        - np.where(item[gi] == own[pi], 1e11, 0.0)
        + area[gi]
    )
    order = np.lexsort((score, pi))
    pi_s = pi[order]
    first = np.empty(len(pi_s), dtype=bool)
    first[0] = True
    if len(pi_s) > 1:
        first[1:] = pi_s[1:] != pi_s[:-1]
    win = order[first]
    out[m[pi[win]]] = idx_arr[gi[win]]
    return out


def postprocess_hvac(
    payload: dict | str | Path,
    raw,
    clf=None,
    classes=None,
    *,
    thr: float | None = None,
    classifier_path: str | Path | None = None,
    **kw,
) -> dict:
    """Decoupled caller-side HVAC post-processing from a raw model response (``Results``, HTTP JSON dict/str, or ``.npz``).

    Designed for two-service deployment (AI_Takeoff):
      - Service A hosts ``OpenStroke("best_deploy.pt")`` and returns ``r.to_dict(compact=True)`` (raw only);
      - Service B (caller that owns ``payload``) calls ``postprocess_hvac(payload, raw_resp, clf=..., classes=...)``.
    """
    from .build import shape_attributes
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

    if clf is None:
        cp = Path(classifier_path) if classifier_path is not None else Path(__file__).resolve().parents[1] / "models" / "hvac_faces.joblib"
        if not cp.exists():
            raise FileNotFoundError(f"HVAC face classifier not found at {cp}; pass clf/classes or classifier_path=")
        clf, classes = load_classifier(cp)

    shapes = pay.get("shapes") or []
    segs = page_segments(shapes)
    length = segs.shape_length().astype(np.float32)
    attrs = shape_attributes(shapes, segs)
    curved = np.asarray(attrs["op_counts"][:, 1] > 0, dtype=bool)
    eff_thr = float(res.threshold if thr is None else thr)
    objs, cls_out, inst_out = faces_first_hybrid(
        segs, res.prob, curved, length, clf, classes, res.groups, thr=eff_thr, attrs=attrs, **kw
    )
    return {"objects": objs, "cls": cls_out, "inst": inst_out, "threshold": eff_thr}
