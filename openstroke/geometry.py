"""Shape geometry: payload items -> polyline segments, length-weighted sample points, and coordinate checks.

All coordinates are PDF points in the display top-left frame (the frame of both the
``split_countfire`` payload ``items`` and the AI_Takeoff ground truth).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

#: Sub-segments used to flatten one cubic bezier. Elbow arcs in these drawings are at most a quarter
#: circle, where 8 chords stay within 0.1 % of the arc length.
BEZIER_STEPS = 8

#: Spacing of length samples along a segment, and the cap per segment. Labels need a length fraction, so
#: a long straight segment does not need a point every point.
SAMPLE_STEP = 1.0
MAX_SAMPLES_PER_SEGMENT = 64


def _bezier(p0, p1, p2, p3, steps: int = BEZIER_STEPS) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps + 1)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in (p0, p1, p2, p3))
    u = 1.0 - t
    return u**3 * p0 + 3 * u**2 * t * p1 + 3 * u * t**2 * p2 + t**3 * p3


def _rect_corners(rect: Sequence[Any]) -> list[tuple[float, float]]:
    if len(rect) == 2:  # [[x0, y0], [x1, y1]]
        (x0, y0), (x1, y1) = rect
    else:
        x0, y0, x1, y1 = rect[:4]
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def item_polylines(item: Sequence[Any]) -> list[np.ndarray]:
    """One payload item -> list of polylines (each an (n, 2) array, n >= 2).

    Supported ops: ``l`` (line), ``c`` (cubic bezier, sampled along the arc, not the chord),
    ``re`` (rectangle, closed), ``qu`` (quad, closed). Unknown ops yield nothing.
    """

    op = item[0]
    if op == "l":
        return [np.asarray([item[1], item[2]], dtype=np.float64)]
    if op == "c":
        return [_bezier(item[1], item[2], item[3], item[4])]
    if op == "re":
        c = _rect_corners(item[1])
        return [np.asarray(c + [c[0]], dtype=np.float64)]
    if op == "qu":
        q = [tuple(p) for p in item[1]]
        return [np.asarray(q + [q[0]], dtype=np.float64)]
    return []


@dataclass
class PageSegments:
    """Every straight segment of a page, with its owning shape.

    ``seg`` is (m, 4) float64 ``x0, y0, x1, y1``; ``owner`` is (m,) int64 shape index;
    ``shape_offsets`` is (n_shapes + 1,) so shape ``i`` owns ``seg[shape_offsets[i]:shape_offsets[i+1]]``.
    """

    seg: np.ndarray
    owner: np.ndarray
    shape_offsets: np.ndarray

    @property
    def length(self) -> np.ndarray:
        return np.hypot(self.seg[:, 2] - self.seg[:, 0], self.seg[:, 3] - self.seg[:, 1])

    def shape_length(self) -> np.ndarray:
        n = len(self.shape_offsets) - 1
        return np.bincount(self.owner, weights=self.length, minlength=n)


def page_segments(shapes: Sequence[dict]) -> PageSegments:
    chunks: list[np.ndarray] = []
    counts = np.zeros(len(shapes), dtype=np.int64)
    for i, shape in enumerate(shapes):
        k = 0
        for item in shape.get("items") or ():
            for poly in item_polylines(item):
                s = np.concatenate([poly[:-1], poly[1:]], axis=1)
                chunks.append(s)
                k += len(s)
        counts[i] = k
    seg = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 4))
    owner = np.repeat(np.arange(len(shapes), dtype=np.int64), counts)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    return PageSegments(seg=seg, owner=owner, shape_offsets=offsets)


def sample_segments(
    seg: np.ndarray, step: float = SAMPLE_STEP, cap: int = MAX_SAMPLES_PER_SEGMENT
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Midpoint-rule samples along segments.

    Returns ``points`` (p, 2), ``seg_index`` (p,) and ``weight`` (p,): each sample carries an equal share of
    its segment's length, so summing weights of "inside" samples gives the inside length. Zero-length
    segments get one sample of weight 0.
    """

    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    k = np.clip(np.ceil(length / step), 1, cap).astype(np.int64)
    idx = np.repeat(np.arange(len(seg), dtype=np.int64), k)
    start = np.concatenate([[0], np.cumsum(k)[:-1]])
    j = np.arange(len(idx), dtype=np.int64) - np.repeat(start, k)
    t = (j + 0.5) / np.repeat(k, k)
    s = seg[idx]
    pts = np.stack([s[:, 0] + (s[:, 2] - s[:, 0]) * t, s[:, 1] + (s[:, 3] - s[:, 1]) * t], axis=1)
    weight = (length / k)[idx]
    return pts, idx, weight


def polyline_points(vertices: Sequence[Sequence[float]], closed: bool, step: float = SAMPLE_STEP,
                    cap: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Samples along a vertex list (closed polygons include the closing edge). Returns points and weights."""

    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    if closed and len(v) >= 3:
        v = np.concatenate([v, v[:1]], axis=0)
    if len(v) < 2:
        return np.zeros((0, 2)), np.zeros(0)
    seg = np.concatenate([v[:-1], v[1:]], axis=1)
    pts, _, w = sample_segments(seg, step=step, cap=cap)
    return pts, w


# ---------------------------------------------------------------------------
# Coordinate check (vector-ink version of AI_Takeoff ``geometry.calibrate_rotation``)
# ---------------------------------------------------------------------------

IDENTITY = "identity"
ROT90_CCW = "rot90ccw"
ROT90_CW = "rot90cw"
ROT180 = "rot180"
FLIP_Y = "flip_y"
CANDIDATES = (IDENTITY, ROT90_CCW, ROT90_CW, ROT180, FLIP_Y)

# Same decision thresholds as AI_Takeoff BE/app/geometry.py (2026-10-20). The ink tolerance is tighter
# (0.5 pt, not AI_Takeoff's 2 px at 1.5x = 1.3 pt): on dense sheets 2 pt puts 68-75 % of *random* points
# on a stroke, the lift ceiling falls to ~1.4 and true identities were rejected (doc 20, doc 22; LEDGER F2).
# GT vertices come from the vector strokes themselves, so 0.5 pt still finds 96-100 % of them.
INK_TOLERANCE = 0.5
MIN_LIFT = 1.15
MIN_ABSOLUTE_COVERAGE = 0.50
MIN_COVERAGE_RATIO = 1.50
LIFT_MARGIN = 1.25
MAX_CALIBRATION_SAMPLES = 4000
RANDOM_SAMPLES = 2000


def transform_points(name: str, pts: np.ndarray, width: float, height: float) -> np.ndarray:
    """Map ground-truth points by a candidate transform, in a page of display size ``width`` x ``height``.

    The quarter turns map a frame of the swapped size onto this one (the unrotated page of a
    ``/Rotate 90|270`` sheet), matching AI_Takeoff's ``_rotation_function``.
    """

    x, y = pts[:, 0], pts[:, 1]
    if name == IDENTITY:
        return pts.copy()
    if name == ROT90_CCW:  # unrotated width == display height
        return np.stack([y, height - x], axis=1)
    if name == ROT90_CW:  # unrotated height == display width
        return np.stack([width - y, x], axis=1)
    if name == ROT180:
        return np.stack([width - x, height - y], axis=1)
    if name == FLIP_Y:
        return np.stack([x, height - y], axis=1)
    raise ValueError(f"unknown transform {name!r}")


def interior_samples(vertex_lists: Sequence[Sequence[Sequence[float]]], closed: Sequence[bool],
                     max_samples: int = MAX_CALIBRATION_SAMPLES) -> np.ndarray:
    """Interior points of each edge (t in 0.15..0.85), as AI_Takeoff ``_sample_along`` does.

    Differs from AI_Takeoff in one place: up to 50 (not 10) points per edge, so a page with a handful of
    GT objects still reaches the 20-sample minimum instead of being rejected for lack of evidence.
    """

    per_edge = max(2, min(50, int(np.ceil(max_samples / max(len(vertex_lists), 1)))))
    t = 0.15 + 0.7 * np.arange(per_edge) / max(per_edge - 1, 1)
    out: list[np.ndarray] = []
    total = 0
    for v, c in zip(vertex_lists, closed):
        v = np.asarray(v, dtype=np.float64).reshape(-1, 2)
        if c and len(v) >= 3:
            v = np.concatenate([v, v[:1]], axis=0)
        if len(v) < 2:
            continue
        a, b = v[:-1], v[1:]
        p = a[:, None, :] + (b - a)[:, None, :] * t[None, :, None]
        p = p.reshape(-1, 2)
        out.append(p)
        total += len(p)
        if total >= max_samples:
            break
    return np.concatenate(out, axis=0) if out else np.zeros((0, 2))


def _coverage(tree, pts: np.ndarray, width: float, height: float, tol: float,
              offpage_is_miss: bool) -> tuple[float, int]:
    import shapely

    keep = (pts[:, 0] >= 0) & (pts[:, 0] <= width) & (pts[:, 1] >= 0) & (pts[:, 1] <= height)
    total = len(pts) if offpage_is_miss else int(keep.sum())
    pts = pts[keep]
    if len(pts) == 0 or total == 0:
        return 0.0, int(len(pts))
    hits = tree.query(shapely.points(pts), predicate="dwithin", distance=tol)
    return float(len(np.unique(hits[0])) / total), int(len(pts))


def calibrate(ink_tree, gt_vertex_lists, gt_closed, width: float, height: float,
              candidates: Sequence[str] = CANDIDATES, tol: float = INK_TOLERANCE, seed: int = 12345) -> dict:
    """Pick the transform whose mapped ground truth lands best on the payload's own strokes.

    ``ink_tree`` is a shapely STRtree of the page's stroke segments (the payload, not a render, so no
    PyMuPDF). Score = coverage / random-baseline coverage in the mapped bbox ("lift"), exactly the
    AI_Takeoff rule, and the same conclusiveness test.
    """

    report: dict[str, Any] = {"method": "vector_ink_coverage", "tolerance_pt": tol, "chosen": None,
                              "conclusive": False, "reason": "", "candidates": []}
    pts = interior_samples(gt_vertex_lists, gt_closed)
    if len(pts) == 0:
        report["reason"] = "no ground-truth geometry to calibrate against"
        return report
    rng = np.random.default_rng(seed)
    scored = []
    for name in candidates:
        mapped = transform_points(name, pts, width, height)
        # GT points a transform pushes off the page count as misses. AI_Takeoff drops them instead, which
        # let a quarter turn that kept 20 % of the GT on the page win on doc 41 (LEDGER F3).
        cov, counted = _coverage(ink_tree, mapped, width, height, tol, offpage_is_miss=True)
        if counted < 20:
            scored.append((0.0, cov, 0.0, counted, name))
            continue
        # Pad the bbox by 5 % of the page: a GT made of one straight run has a zero-height bbox, where
        # random points would all fall on the run itself.
        pad = 0.05 * np.array([width, height])
        lo, hi = mapped.min(axis=0) - pad, mapped.max(axis=0) + pad
        rnd = rng.uniform(lo, hi, size=(RANDOM_SAMPLES, 2))
        base, _ = _coverage(ink_tree, rnd, width, height, tol, offpage_is_miss=False)
        lift = cov / base if base > 0 else (float("inf") if cov > 0 else 0.0)
        scored.append((lift, cov, base, counted, name))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    report["candidates"] = [{"transform": n, "lift": round(min(l, 1e6), 3), "coverage": round(c, 4),
                             "baseline": round(b, 4), "sampled": k} for l, c, b, k, n in scored]
    best_lift, best_cov, _, _, best = scored[0]
    run_lift = scored[1][0] if len(scored) > 1 else 0.0
    run_cov = scored[1][1] if len(scored) > 1 else 0.0
    decisive = best_lift >= MIN_LIFT and (
        best_lift >= run_lift * LIFT_MARGIN
        or (best_cov >= MIN_ABSOLUTE_COVERAGE and best_cov >= run_cov * MIN_COVERAGE_RATIO))
    report["chosen"] = best
    report["conclusive"] = bool(decisive)
    report["reason"] = (f"{best} put {best_cov:.1%} of GT samples within {tol:g} pt of a payload stroke, "
                        f"{best_lift:.2f}x the random baseline; runner-up lift {run_lift:.2f}")
    return report
