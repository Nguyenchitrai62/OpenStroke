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
            "compact", "n_strokes", "log_ink_inside", "touch_faces")


def page_faces(segs, picked: np.ndarray, gap: float = GAP) -> list:
    """Closed regions enclosed by the picked strokes (holes of the thickened ink), grown back to the strokes' centre lines."""
    import shapely
    from shapely.geometry import Polygon

    m = picked[segs.owner]
    if not m.any():
        return []
    ink = shapely.union_all(shapely.buffer(shapely.linestrings(segs.seg[m].reshape(-1, 2, 2)), gap, cap_style="flat"))
    holes = [Polygon(r) for g in shapely.get_parts(ink) for r in g.interiors]
    return [h.buffer(gap, join_style="mitre") for h in holes if h.area > MIN_AREA]


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
        corners = len(shapely.simplify(f, 0.5).exterior.coords) - 1
        out.append([np.log1p(f.area), np.log1p(long_), np.log1p(short), long_ / max(short, 1e-6), f.area / max(long_ * short, 1e-9),
                    f.area / max(f.convex_hull.area, 1e-9), min(corners, 20), arc, np.log1p(ring.length),
                    4 * np.pi * f.area / max(ring.length ** 2, 1e-9), len(set(own[near].tolist())), np.log1p(ink_in), touch])
    return np.asarray(out, dtype=np.float64)
