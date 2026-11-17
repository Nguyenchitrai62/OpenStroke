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
FEATURES = ("log_area", "log_long", "log_short", "aspect", "rect_fill", "hull_fill", "corners", "arc_share", "log_perimeter",
            "compact", "n_strokes", "log_ink_inside", "touch_faces", "side_ratio", "skew_deg", "right_corners")
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
    merged = merged[shapely.length(merged) > 0]
    cur = arr[np.asarray(curved_chain, bool)]
    if not len(cur) or not len(merged):
        return merged, np.zeros(len(merged), bool)
    band = shapely.union_all(shapely.buffer(cur, 0.05))
    on = shapely.length(shapely.intersection(merged, band))
    return merged, on >= 0.5 * shapely.length(merged)


def _caps(chains, curved_chain: np.ndarray):
    """Virtual end lines (F153): an elbow the model picked often has its two arcs but not the short lines closing its ends, so no face
    closes. Two chain ends, at least one of them on a curve, that point the same way (cos >= CAP_PAR), lie <= CAP_MAX pt apart and face
    each other square to that direction (|cos| <= CAP_PERP) are joined; each end at most once, nearest first."""
    import shapely
    from scipy.spatial import cKDTree

    chains, curved_chain = _merged_chains(chains, curved_chain)
    pts, dirs, owner = [], [], []
    for i, c in enumerate(chains):
        xy = np.asarray(c.coords)
        if len(xy) < 2 or c.length < END_MIN:
            continue
        for a, b in ((xy[0], xy[1]), (xy[-1], xy[-2])):
            d = a - b
            n = np.hypot(*d)
            if n > 0:
                pts.append(a); dirs.append(d / n); owner.append(i)
    if len(pts) < 2:
        return []
    pts, dirs, owner = np.asarray(pts), np.asarray(dirs), np.asarray(owner)
    if len(pts) < 2:
        return []
    # only ends touching no other stroke are open (F157: a tee's fillet arcs already meet the duct lines; capping them split the tee)
    tree = shapely.STRtree(np.asarray(list(chains), dtype=object))
    pi, li = tree.query(shapely.points(pts), predicate="dwithin", distance=SNAP)
    touched = np.zeros(len(pts), bool)
    touched[pi[li != owner[pi]]] = True
    cand = []
    for a, b in cKDTree(pts).query_pairs(CAP_MAX):
        if owner[a] == owner[b] or not (curved_chain[owner[a]] or curved_chain[owner[b]]) or touched[a] or touched[b]:
            continue                                       # both ends open (F157; one open end only: more false caps, F159)
        v = pts[b] - pts[a]
        dist = np.hypot(*v)
        if dist < 1.0 or dirs[a] @ dirs[b] < CAP_PAR:
            continue
        u = v / dist
        if abs(u @ dirs[a]) > CAP_PERP or abs(u @ dirs[b]) > CAP_PERP:
            continue
        cand.append((dist, a, b))
    # bridges first (F158): two open ends facing each other on one line (a short stroke between them the model scored low) join straight
    bridges = []
    for a, b in cKDTree(pts).query_pairs(BRIDGE_MAX):
        if owner[a] == owner[b] or touched[a] or touched[b]:
            continue
        v = pts[b] - pts[a]
        dist = np.hypot(*v)
        if dist < 1e-6:
            continue
        u = v / dist
        if dirs[a] @ dirs[b] <= -CAP_PAR and u @ dirs[a] >= CAP_PAR and -(u @ dirs[b]) >= CAP_PAR:
            bridges.append((dist, a, b))
    # corners (F166): two open ends roughly square to each other whose forward rays meet within CORNER_MAX of both (a dashed outline whose
    # dashes stop short of the corner) join through that point
    corners = []
    if CORNER_MAX > 0:
        for a, b in cKDTree(pts).query_pairs(2 * CORNER_MAX):
            if owner[a] == owner[b] or touched[a] or touched[b] or abs(dirs[a] @ dirs[b]) > CORNER_COS:
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
    ftree = shapely.STRtree(faces) if faces else None
    keep_s, drop_f = [], set()
    for g in shapes:
        near = ftree.query(g, predicate="intersects") if ftree is not None else []
        same = any(faces[j].intersection(g).area / max(faces[j].union(g).area, 1e-9) >= SAME_IOU for j in near)
        if same:
            continue
        inside = [j for j in near if faces[j].intersection(g).area >= INSIDE * faces[j].area]
        keep_s.append(g)
        drop_f.update(inside)
    # one shape drawn twice (or nested copies): keep one
    uniq = []
    for g in keep_s:
        if not any(g.intersection(h).area / max(g.union(h).area, 1e-9) >= SAME_IOU for h in uniq):
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
    seg = seg[np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) > 1e-6]
    if len(seg) < 3:
        return []
    P = seg.reshape(-1, 2)
    nodes, inv = np.unique(np.round(P, 6), axis=0, return_inverse=True)
    inv = inv.ravel()
    deg = np.bincount(inv, minlength=len(nodes))
    parent = np.arange(len(nodes))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ntree = cKDTree(nodes)
    lines = shapely.linestrings(seg.reshape(-1, 2, 2))
    ltree = shapely.STRtree(lines)
    splits = {}                                               # segment -> [(t along it, node)]
    extra = []
    for f in np.flatnonzero(deg == 1):
        near = [j for j in ntree.query_ball_point(nodes[f], snap) if j != f]
        if near:
            j = min(near, key=lambda j: (deg[j] == 1, np.hypot(*(nodes[j] - nodes[f]))))   # a shared vertex before another free end
            ra, rb = root(f), root(j)
            if ra != rb:
                parent[ra] = rb
            continue
        pt = shapely.Point(nodes[f])
        best = None
        for k in ltree.query(pt, predicate="dwithin", distance=snap):
            if inv[2 * k] == f or inv[2 * k + 1] == f:
                continue
            d = lines[k].distance(pt)
            if best is None or d < best[0]:
                best = (d, int(k))
        if best is not None:
            k = best[1]
            t = lines[k].project(pt)
            q = np.asarray(lines[k].interpolate(t).coords[0])
            n = len(nodes) + len(extra)
            extra.append(q)
            splits.setdefault(k, []).append((t, n))
            parent = np.r_[parent, n]
            ra = root(f)
            parent[ra] = n
    # any vertex lying on another segment's body (touching, <= TOUCH) splits it there: the V of a symbol meeting its base (F159)
    pi, li = ltree.query(shapely.points(nodes), predicate="dwithin", distance=TOUCH)
    for f, k in zip(pi, li):
        if deg[f] == 1 or inv[2 * k] == f or inv[2 * k + 1] == f:
            continue
        pt = shapely.Point(nodes[f])
        t = lines[k].project(pt)
        if 1e-6 < t < lines[k].length - 1e-6:
            splits.setdefault(int(k), []).append((t, int(f)))
    allnodes = np.vstack([nodes] + ([np.asarray(extra)] if extra else []))
    if len(parent) < len(allnodes):
        parent = np.r_[parent, np.arange(len(parent), len(allnodes))]
    edges, seen = [], set()

    def add(u, v):
        u, v = root(u), root(v)
        if u == v:
            return
        key = (min(u, v), max(u, v))
        if key in seen:
            return
        seen.add(key)
        edges.append(shapely.LineString([allnodes[u], allnodes[v]]))

    for k in range(len(seg)):
        a, b = inv[2 * k], inv[2 * k + 1]
        cuts = sorted(splits.get(k, []))
        chain = [a] + [n for _, n in cuts] + [b]
        for u, v in zip(chain[:-1], chain[1:]):
            add(u, v)
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
    simp = [len(np.asarray(shapely.simplify(g, 0.5).exterior.coords)) - 1 for g in out]
    thin = []
    for i, g in enumerate(out):
        rr = np.asarray(g.minimum_rotated_rectangle.exterior.coords)[:4]
        e = np.hypot(*(np.roll(rr, -1, 0) - rr).T) if len(rr) == 4 else np.zeros(4)
        long_ = max(e[0], e[1], 1e-9)
        if simp[i] <= 3 and g.area / long_ < SLIVER_W and long_ * long_ / (2 * g.area) >= SLIVER_ASPECT:   # thin and long: a gap
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

    uniq, src, inner = [], [], []
    for g, it, n_in in sorted(item_faces(segs, picked), key=lambda q: q[0].area):
        if uniq:
            tree = shapely.STRtree(uniq)
            if any(g.intersection(uniq[j]).area / max(g.union(uniq[j]).area, 1e-9) >= SAME_IOU for j in tree.query(g, predicate="intersects")):
                continue
        uniq.append(g); src.append(it); inner.append(n_in)
    tree = shapely.STRtree(uniq) if uniq else None
    for f in graph_faces(segs, picked, curved=curved):
        if tree is not None:
            near = tree.query(f, predicate="intersects")
            if any(f.intersection(uniq[j]).area / max(f.union(uniq[j]).area, 1e-9) >= SAME_IOU for j in near):
                continue
            if sum(f.intersection(uniq[j]).area for j in near) >= INSIDE * f.area:
                continue
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


def face_features(faces: list, segs, picked: np.ndarray, curved: np.ndarray) -> np.ndarray:
    """Shape descriptors of every face (FEATURES)."""
    import shapely

    if not faces:
        return np.zeros((0, len(FEATURES)))
    m = picked[segs.owner]
    lines = shapely.linestrings(segs.seg[m].reshape(-1, 2, 2))
    own = segs.owner[m]
    is_curve = curved[own]
    ltree = shapely.STRtree(lines)
    ftree = shapely.STRtree(faces)
    out = []
    for f in faces:
        rr = np.asarray(f.minimum_rotated_rectangle.exterior.coords)[:4]
        e = np.hypot(*(np.roll(rr, -1, 0) - rr).T) if len(rr) == 4 else np.zeros(4)
        long_, short = float(max(e[0], e[1])), float(min(e[0], e[1]))
        ring = f.exterior
        near = ltree.query(ring, predicate="dwithin", distance=GAP + 0.3)
        band = ring.buffer(GAP + 0.3)
        cov = shapely.length(shapely.intersection(lines[near], band)) if len(near) else np.zeros(0)
        arc = float(cov[is_curve[near]].sum() / max(cov.sum(), 1e-9)) if len(near) else 0.0
        inside = ltree.query(f.buffer(-GAP - 0.3) if f.area > 4 else f, predicate="intersects")
        ink_in = float(shapely.length(shapely.intersection(lines[inside], f.buffer(-GAP - 0.3))).sum()) if len(inside) and f.area > 4 else 0.0
        touch = len(ftree.query(f.buffer(GAP * 2), predicate="intersects")) - 1
        simp = np.asarray(shapely.simplify(f, 0.5).exterior.coords)[:-1]
        corners = len(simp)
        side_ratio, skew, right = _taper(simp)
        out.append([np.log1p(f.area), np.log1p(long_), np.log1p(short), long_ / max(short, 1e-6), f.area / max(long_ * short, 1e-9),
                    f.area / max(f.convex_hull.area, 1e-9), min(corners, 20), arc, np.log1p(ring.length),
                    4 * np.pi * f.area / max(ring.length ** 2, 1e-9), len(set(own[near].tolist())), np.log1p(ink_in), touch,
                    side_ratio, skew, right])
    return np.asarray(out, dtype=np.float64)


# ---- post-processing step (F151): faces named by a classifier, added where the logic left nothing -------------------------------------

PICK_THR = 0.3     # strokes with p >= this draw the faces (chosen on the train pages, F151)
FACE_CONF = 0.5    # a face becomes an object when its class probability >= this (chosen on the train pages, cross-fitted by document)
COVERED = 0.3      # a face more than this share inside the logic's objects belongs to the logic
CLASSES = ("none", "Duct", "Elbow", "Elbow vuông", "Concentric Reducer", "Eccentric Reducer", "Chân rẽ", "Tee")   # Tee: out of the
# 6-class benchmark (scored as nothing) but a real object of the drawings; without it a tee face was called an Elbow (F152)


def fit_classifier(npz_paths, viewer: bool = True, skip_doc: str | None = None):
    """HistGradientBoosting on the faces of research/probes/faces_train.py (one or more npz). -> (clf, class names)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y, classes = [], [], CLASSES
    for t in npz_paths:
        z = np.load(t)
        pg = z["page"].astype(str)
        keep = np.ones(len(z["y"]), bool) if viewer else ~np.char.startswith(pg, "vw_")
        if skip_doc:
            keep &= ~np.char.startswith(pg, skip_doc)
        X.append(z["X"][keep]); y.append(z["y"][keep])
        classes = tuple(str(c) for c in z["classes"])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    clf.fit(np.concatenate(X), np.concatenate(y))
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
DUCT_MIN_W, DUCT_ISO_AREA = 1.0, 0.0   # F164: drop Duct objects narrower than this (pt) / lone ones smaller than this (pt^2); chosen on the GT train pages (0.766 -> 0.796; the lone-box filter added <= 0.001, off)


def neighbours(faces: list) -> list[list[int]]:
    """For each face, the faces it shares >= SHARED pt of outline with (faces grow back to the same centre line, so neighbours touch)."""
    import shapely

    tree = shapely.STRtree(faces)
    out = []
    for i, f in enumerate(faces):
        b = f.exterior
        out.append([int(j) for j in tree.query(f.buffer(0.5), predicate="intersects")
                    if j != i and b.intersection(faces[j].buffer(0.5)).length >= SHARED])
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


def _name_faces(faces, X, clf, classes):
    """Classifier names + the two shape rules (F152): taper between two object neighbours -> reducer; Elbow with >= 3 object
    neighbours -> Tee. -> names, scores, why, object-neighbour lists, class probabilities."""
    import shapely

    pr = clf.predict_proba(X)
    names = [classes[int(clf.classes_[k])] for k in pr.argmax(1)]
    score = pr.max(1)
    nb = [[j for j in js if names[j] != "none"] for js in neighbours(faces)]   # neighbours the classifier calls an object
    col = {n: i for i, n in enumerate(FEATURES)}
    why = ["face"] * len(faces)
    for i, f in enumerate(faces):
        x = X[i]
        if x[col["corners"]] == 4 and x[col["side_ratio"]] < TAPER_RATIO and x[col["skew_deg"]] >= TAPER_SKEW and len(nb[i]) >= 2:
            v = np.asarray(shapely.simplify(f, 0.5).exterior.coords)[:-1]
            e = np.roll(v, -1, 0) - v
            ang = np.degrees(np.arctan2(e[:, 1], e[:, 0]))
            d = lambda a, b: min(abs((ang[a] - ang[b]) % 180), 180 - abs((ang[a] - ang[b]) % 180))
            par = (0, 2) if d(0, 2) <= d(1, 3) else (1, 3)
            ends = [shapely.LineString([v[k], v[(k + 1) % 4]]) for k in par]
            if all(any(end.buffer(0.6).intersection(faces[j].exterior).length >= 0.5 * end.length for j in nb[i]) for end in ends):
                names[i], why[i] = reducer_kind(f), "taper"   # F163: concentric / eccentric by its corners (was always Concentric)
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
    for i, f in enumerate(faces):
        band = f.exterior.buffer(GAP + 0.3)
        near = tree.query(band, predicate="intersects")
        if not len(near):
            continue
        on = shapely.length(shapely.intersection(lines[near], band))
        v = (pc[near] * on[:, None]).sum(0)
        g = np.asarray([v[c].sum() for c in cols])
        if g.sum() > 0:
            G[i] = g / g.sum()
    return G


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


def post_objects(*args, duct_min_width: float = DUCT_MIN_W, duct_iso_area: float = DUCT_ISO_AREA, **kw) -> list[dict]:
    """``_post_objects`` (below), then ``drop_spurious_ducts`` (F164)."""
    return drop_spurious_ducts(_post_objects(*args, **kw), duct_min_width, duct_iso_area)


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
    segment goes to the object its own item drew whose outline runs within tol of its midpoint, else the smallest such object, else the
    smallest object containing it (F157: a duct's side shared with a symbol below it stays the duct's)."""
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
    area = np.asarray([g.area for g in polys])
    m = np.flatnonzero(picked[segs.owner])
    mid = shapely.points((segs.seg[m, :2] + segs.seg[m, 2:]) / 2)
    tree = shapely.STRtree(polys)
    best = np.full(len(m), np.inf)
    pi, gi = tree.query(mid, predicate="dwithin", distance=tol)
    on = shapely.distance(mid[pi], shapely.boundary(np.asarray(polys, dtype=object)[gi])) <= tol
    item = np.asarray([objects[i].get("item", -1) for i in idx])
    own = segs.owner[m]
    for k, g, ok in zip(pi, gi, on):
        a = area[g] if ok else area[g] + 1e12          # on its outline first, inside it second
        if ok and item[g] == own[k]:
            a -= 1e11                                  # ... and the object its own item drew before any other (F157)
        if a < best[k]:
            best[k] = a
            out[m[k]] = idx[g]
    return out
