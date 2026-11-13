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
