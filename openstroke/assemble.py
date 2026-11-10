"""Objects from OpenOBB boxes and their strokes: which box owns a stroke that several boxes claim (OpenStroke10 step 1, E84 / H98).

The detector returns, per box, a class, an oriented polygon, a score and (OpenOBB25) the strokes it drew. Boxes overlap: a duct box runs
under the elbow or reducer drawn on it, and near-duplicate boxes of one object survive NMS. Before E84 a contested stroke went to the
highest score, so a large duct box took the strokes of the fitting on it (E69, conf 0.3: 12 % of GT objects absorbed, 22 % of elbows).

- ``assign_members``: stroke -> box index for the model's own strokes. ``order="small"``: the smallest box wins a contested stroke (the
  GT label rule's choice, fitting over duct); ``order="score"``: the highest score wins (the old behaviour). ``dedupe``: a box whose
  strokes are mostly (>= ``overlap`` of its length) held by a higher-scored box of the same class is dropped first.
- ``dedupe_boxes``: the same suppression for boxes without strokes (the label-rule path), by polygon area instead of stroke length.
"""

from __future__ import annotations

import numpy as np


def poly_area(v: np.ndarray) -> float:
    x, y = np.asarray(v, float).T
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _members(m, n: int) -> np.ndarray:
    m = np.unique(np.asarray(m, np.int64))
    return m[(m >= 0) & (m < n)]


def assign_members(boxes: list, length: np.ndarray, order: str = "small", dedupe: bool = True, overlap: float = 0.5) -> np.ndarray:
    """``boxes``: [(class, polygon (4, 2), score, member stroke indices)]. -> owning box index per stroke (-1 = none)."""

    n = len(length)
    pi = np.full(n, -1, np.int64)
    if not boxes:
        return pi
    mem = [_members(b[3], n) for b in boxes]
    alive = np.ones(len(boxes), bool)
    if dedupe:
        held = {}  # class -> strokes already held by kept boxes of that class
        for k in sorted(range(len(boxes)), key=lambda i: -boxes[i][2]):
            m = mem[k]
            if not len(m):
                alive[k] = False
                continue
            h = held.setdefault(boxes[k][0], np.zeros(n, bool))
            if length[m][h[m]].sum() >= overlap * max(length[m].sum(), 1e-9):
                alive[k] = False
                continue
            h[m] = True
    idx = [k for k in range(len(boxes)) if alive[k]]
    key = (lambda i: -poly_area(boxes[i][1])) if order == "small" else (lambda i: boxes[i][2])
    for k in sorted(idx, key=key):  # written last wins: the smallest box (or the highest score)
        pi[mem[k]] = k
    return pi


def dedupe_boxes(boxes: list, overlap: float = 0.5) -> list[int]:
    """Indices of the boxes kept: a box with >= ``overlap`` of its area inside a higher-scored box of the same class is dropped."""

    import shapely

    polys = np.asarray([shapely.make_valid(shapely.Polygon(np.asarray(b[1], float))) for b in boxes], dtype=object)
    tree = shapely.STRtree(polys)
    kept = np.zeros(len(boxes), bool)
    for k in sorted(range(len(boxes)), key=lambda i: -boxes[i][2]):
        a = max(polys[k].area, 1e-9)
        near = [j for j in tree.query(polys[k], predicate="intersects") if kept[j] and boxes[j][0] == boxes[k][0]]
        if any(polys[k].intersection(polys[j]).area >= overlap * a for j in near):
            continue
        kept[k] = True
    return [int(k) for k in np.flatnonzero(kept)]


def filter_members(boxes: list, segs, length: np.ndarray, style: np.ndarray | None = None, clip: bool = True, tol: float = 2.0,
                   inside: float = 0.5, style_share: float = 0.2) -> list:
    """Boxes with their member strokes trimmed (E84 / H99): ``clip`` keeps strokes with >= ``inside`` of their length within the box
    polygon grown by ``tol`` pt (drops the neighbour objects a box's members spill onto); ``style`` (one id per stroke, e.g. colour +
    width + dash) keeps strokes whose style holds >= ``style_share`` of the box's member length (drops text / architecture under a box)."""

    import shapely

    n = len(length)
    off = segs.shape_offsets
    out = []
    for c, poly, s, m in boxes:
        m = _members(m, n)
        if len(m) and clip:
            cnt = off[m + 1] - off[m]
            sid = np.concatenate([np.arange(off[k], off[k + 1]) for k in m]) if cnt.sum() else np.zeros(0, np.int64)
            sg = segs.seg[sid]
            mid = 0.5 * (sg[:, :2] + sg[:, 2:])
            ln = np.hypot(*(sg[:, 2:] - sg[:, :2]).T)
            g = shapely.Polygon(np.asarray(poly, float)).buffer(tol)
            ins = shapely.contains_xy(g, mid[:, 0], mid[:, 1])
            owner = np.repeat(np.arange(len(m)), cnt)
            tot = np.bincount(owner, ln, minlength=len(m))
            inn = np.bincount(owner, ln * ins, minlength=len(m))
            m = m[(inn >= inside * np.maximum(tot, 1e-9)) | (tot <= 0)]
        if len(m) and style is not None:
            u, inv = np.unique(style[m], return_inverse=True)
            share = np.bincount(inv, length[m]) / max(length[m].sum(), 1e-9)
            m = m[share[inv] >= style_share]
        out.append((c, poly, s, m))
    return out
