"""Shape rules after AI_Takeoff's HVAC logic (F146): fix the objects it builds from the stroke model's picks, from their geometry only (no
layer, no text). Each rule is general drafting geometry, not a page-specific fix:

  thin_duct    a "Duct" whose outline is thinner than THIN_PT cannot be a duct (drawn ducts are >= ~3 pt wide); the logic gave that name to
               small branch take-off symbols -> "Chân rẽ".
  square_elbow an "Elbow" whose strokes have (almost) no arc is a square elbow -> "Elbow vuông" (the logic never names one).
  symbol_box   a small, nearly square "Duct" (long side < BOX_LONG pt, long / short < BOX_ASPECT) drawn with at most BOX_STROKES strokes is an
               equipment / diffuser box the logic took for a duct -> dropped.

    objects = refine(objects, segs, attrs, prob, thr)          # default rules: thin_duct, square_elbow
"""
from __future__ import annotations

import numpy as np

THIN_PT = 2.0
ARC_SHARE = 0.1
BOX_LONG, BOX_ASPECT, BOX_STROKES = 25.0, 2.0, 2
RULES = ("thin_duct", "square_elbow")   # symbol_box measured worse (F146: 0.841 -> 0.810 on the GT val pages), kept off


def sides(v) -> tuple[float, float]:
    """(long, short) side of the object's minimum-area rectangle, pt."""
    import shapely
    from shapely.geometry import Polygon

    p = Polygon(np.asarray(v, float))
    if not p.is_valid:
        p = p.buffer(0)
    r = np.asarray(shapely.minimum_rotated_rectangle(p).exterior.coords)[:4]
    if len(r) < 4:
        return 0.0, 0.0
    e = np.hypot(*(np.roll(r, -1, 0) - r).T)
    return float(max(e[0], e[1])), float(min(e[0], e[1]))


def refine(objects: list[dict], segs, attrs: dict, prob: np.ndarray, thr: float, rules=RULES) -> list[dict]:
    """-> new list of objects (dicts with shape_name, vertices; ``refined`` names the rule that changed one)."""
    from .objects import assign_objects, class_of, HVAC_CLASSES

    if not objects:
        return []
    picked = prob >= thr
    length = np.bincount(segs.owner, weights=segs.length, minlength=len(prob))
    ops = np.asarray(attrs["op_counts"])
    curved = ops[:, 1] > 0 if ops.ndim == 2 and ops.shape[1] > 1 else np.zeros(len(prob), bool)
    ok = [i for i, o in enumerate(objects) if len(o.get("vertices") or []) >= 3]
    _, inst, _ = assign_objects(segs, [objects[i] for i in ok], picked)
    out = []
    for j, i in enumerate(ok):
        o = dict(objects[i])
        name = HVAC_CLASSES[class_of(o.get("shape_name"))]
        m = inst == j
        n_strokes = int(m.sum())
        arc = float(length[m & curved].sum() / max(length[m].sum(), 1e-9)) if n_strokes else 0.0
        long_, short = sides(o["vertices"])
        if name == "Duct" and "thin_duct" in rules and short < THIN_PT:
            o["shape_name"], o["refined"] = "Chân rẽ", "thin_duct"
        elif name == "Elbow" and "square_elbow" in rules and n_strokes and arc < ARC_SHARE:
            o["shape_name"], o["refined"] = "Elbow vuông", "square_elbow"
        elif name == "Duct" and "symbol_box" in rules and long_ < BOX_LONG and long_ < BOX_ASPECT * max(short, 1e-6) and n_strokes <= BOX_STROKES:
            continue
        out.append(o)
    out += [objects[i] for i in range(len(objects)) if i not in set(ok)]
    return out


# ---------------------------------------------------------------------------------------------------------------------- hatch (before the logic)
HATCH_TOL = 1.0                 # pt: an end of a hatch stroke lies on a duct outline
HATCH_MIN, HATCH_MAX = 20.0, 70.0  # degrees between a hatch stroke and the outlines it spans (duct joints are perpendicular: 90)
RAIL_PARALLEL = 5.0             # degrees: the two outlines are parallel


CROSS_MIN = 3                   # a picked stroke crossed through its middle by this many picked strokes is part of a hatch grid
CROSS_END = 1.0                 # pt: a meeting this close to either end is a joint (L / T), not a crossing


def grid_strokes(segs, picked: np.ndarray) -> np.ndarray:
    """Cross-hatch grids among the picked strokes (F147): a picked stroke that at least CROSS_MIN other picked strokes cross through the
    middle of both (an X, not an L or a T). Duct outlines meet other strokes at their ends; the lines of a hatch grid pass through each other."""
    import shapely

    off = segs.shape_offsets
    idx = np.flatnonzero(picked & (np.diff(off) > 0))
    out = np.zeros(len(picked), bool)
    if len(idx) < 3:
        return out
    s = segs.seg
    pick_seg = np.flatnonzero(picked[segs.owner])
    geo = shapely.linestrings(s[pick_seg].reshape(-1, 2, 2))
    tree = shapely.STRtree(geo)
    a, b = tree.query(geo, predicate="crosses")
    keep = segs.owner[pick_seg[a]] != segs.owner[pick_seg[b]]
    a, b = a[keep], b[keep]
    if not len(a):
        return out
    pts = shapely.intersection(geo[a], geo[b])
    single = shapely.get_type_id(pts) == 0
    a, b, pts = a[single], b[single], pts[single]
    xy = shapely.get_coordinates(pts)

    def interior(i):
        e = s[pick_seg[i]]
        return (np.hypot(*(xy - e[:, :2]).T) > CROSS_END) & (np.hypot(*(xy - e[:, 2:]).T) > CROSS_END)
    real = interior(a) & interior(b)
    owners = segs.owner[pick_seg[a[real]]]
    other = segs.owner[pick_seg[b[real]]]
    pairs = np.unique(np.c_[owners, other], axis=0)
    n = np.bincount(pairs[:, 0], minlength=len(picked)) if len(pairs) else np.zeros(len(picked), int)
    out[n >= CROSS_MIN] = True
    return out


def hatch_strokes(segs, picked: np.ndarray) -> np.ndarray:
    """Insulation / fill hatch among the picked strokes (F147): a straight picked stroke whose two ends lie on two different, parallel, longer
    picked strokes (the duct's outlines) and that crosses them at 20-70 degrees. Joint lines and end caps cross at 90 degrees and stay."""
    import shapely

    off = segs.shape_offsets
    idx = np.flatnonzero(picked & (np.diff(off) > 0))
    out = np.zeros(len(picked), bool)
    if len(idx) < 3:
        return out
    one = idx[np.diff(off)[idx] == 1]                       # straight single-segment strokes: hatch candidates
    s = segs.seg
    ang = np.degrees(np.arctan2(s[:, 3] - s[:, 1], s[:, 2] - s[:, 0])) % 180
    ln = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
    rails = np.flatnonzero(picked[segs.owner])                # every picked segment can be an outline
    tree = shapely.STRtree(shapely.linestrings(s[rails].reshape(-1, 2, 2)))
    for k in one:
        j = off[k]
        a, b = shapely.points(s[j, :2]), shapely.points(s[j, 2:])
        ra = rails[tree.query(a, predicate="dwithin", distance=HATCH_TOL)]
        rb = rails[tree.query(b, predicate="dwithin", distance=HATCH_TOL)]
        ra = ra[(segs.owner[ra] != k) & (ln[ra] > ln[j])]
        rb = rb[(segs.owner[rb] != k) & (ln[rb] > ln[j])]
        if not len(ra) or not len(rb):
            continue
        done = False
        for x in ra:
            for y in rb:
                if segs.owner[x] == segs.owner[y]:
                    continue
                d = abs(ang[x] - ang[y]) % 180
                if min(d, 180 - d) > RAIL_PARALLEL:
                    continue
                c = abs(ang[j] - ang[x]) % 180
                c = min(c, 180 - c)
                if HATCH_MIN <= c <= HATCH_MAX:
                    out[k] = done = True
                    break
            if done:
                break
    return out


# ---------------------------------------------------------------------------------------------------------------------- recovery (after the logic)
REC_MIN_SHORT, REC_MIN_AREA, REC_MAX_LONG = 2.0, 20.0, 1500.0
REC_GAP = 0.6                   # pt: half the end gap a face still closes over
REC_OVERLAP = 0.2               # a face overlapping an existing object by more than this share of its area is that object's
REC_ARC = 0.2                   # share of the face's outline on curved strokes -> Elbow
REC_RECT = 0.85                 # area / minimum-rectangle area -> Duct; below (4 corners) -> reducer


def recover(objects: list[dict], segs, attrs: dict, prob: np.ndarray, thr: float) -> list[dict]:
    """Objects the logic left out (F147): the picked strokes that no object holds are closed into faces together with the picked strokes
    around them (shapely.polygonize on the noded picked segments); a face outside every existing object becomes an Elbow (curved outline),
    a Duct (rectangle) or a Concentric Reducer (4-corner, not a rectangle). Open lines (leaders, dimension lines) close nothing and stay out."""
    import shapely
    from shapely.geometry import Polygon

    from .objects import assign_objects

    picked = prob >= thr
    ok = [o for o in objects if len(o.get("vertices") or []) >= 3]
    _, inst, _ = assign_objects(segs, ok, picked)
    free = picked & (inst < 0)
    if not free.any():
        return []
    s = segs.seg
    ops = np.asarray(attrs["op_counts"])
    curved = ops[:, 1] > 0 if ops.ndim == 2 and ops.shape[1] > 1 else np.zeros(len(prob), bool)
    lines_free = shapely.linestrings(s[free[segs.owner]].reshape(-1, 2, 2))
    zone = shapely.buffer(shapely.union_all(lines_free), 30.0)          # look only around the left-out strokes
    pk = picked[segs.owner]
    near = shapely.intersects(zone, shapely.linestrings(s.reshape(-1, 2, 2))) & pk
    if not near.any():
        return []
    # faces = the closed holes of the picked strokes thickened by GAP: an arc and a joint line that end 1 pt apart still close (polygonize needs
    # exact nodes); each hole grows back by GAP to the strokes' centre lines
    ink = shapely.union_all(shapely.buffer(shapely.linestrings(s[near].reshape(-1, 2, 2)), REC_GAP, cap_style="flat"))
    holes = [Polygon(r) for g in shapely.get_parts(ink) for r in g.interiors]
    faces = [h.buffer(REC_GAP, join_style="mitre") for h in holes if h.area > 1.0]
    if not faces:
        return []
    have = [Polygon(np.asarray(o["vertices"], float)).buffer(0) for o in ok]
    tree = shapely.STRtree(have) if have else None
    free_l = shapely.union_all(lines_free)
    curve_l = shapely.union_all(shapely.linestrings(s[(curved & picked)[segs.owner]].reshape(-1, 2, 2))) if (curved & picked).any() else None
    out = []
    for f in faces:
        if not f.is_valid or f.area < REC_MIN_AREA:
            continue
        long_, short = sides(np.asarray(f.exterior.coords)[:-1])
        if short < REC_MIN_SHORT or long_ > REC_MAX_LONG:
            continue
        if tree is not None:
            hit = tree.query(f, predicate="intersects")
            if any(have[h].intersection(f).area > REC_OVERLAP * f.area for h in hit):
                continue
        ring = f.exterior
        if ring.intersection(free_l.buffer(REC_GAP + 0.3)).length < 0.5 * ring.length:   # the face must be drawn mostly by left-out strokes
            continue
        arc = ring.intersection(curve_l.buffer(REC_GAP + 0.3)).length / ring.length if curve_l is not None else 0.0
        rect = f.area / max(f.minimum_rotated_rectangle.area, 1e-9)
        corners = len(shapely.simplify(f, 0.5).exterior.coords) - 1
        if arc >= REC_ARC:
            name = "Elbow"
        elif rect >= REC_RECT:
            name = "Duct"
        elif corners == 4:
            name = "Concentric Reducer"
        else:
            continue
        out.append({"shape_name": name, "vertices": np.round(np.asarray(ring.coords)[:-1], 2).tolist(), "refined": "recovered"})
    return out
