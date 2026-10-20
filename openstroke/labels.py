"""Label rules (AGENTS.md section 2): one label per payload shape, ``true`` (pipe network) or ``other``.

- FIRE: a shape is ``true`` when most of its length lies within ``TOL`` of a GT ``Line``.
- HVAC: a shape is ``true`` when most of its length lies inside (within ``TOL`` of) a GT ``Duct``,
  ``Elbow*``, ``Concentric Reducer*``, ``Eccentric Reducer*`` or ``Chân rẽ*`` polygon. The ``TOL`` margin
  keeps the duct walls, which sit exactly on the polygon boundary, inside.
- Everything else is ``other``. There is no ``ignore`` label.
- Length is measured along the stroke, with beziers sampled along the arc.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import shapely
from shapely import STRtree

from .geometry import PageSegments, polyline_points, sample_segments

LABEL_SCHEMA = "true_other_v2"  # v2: HVAC refinement (refine_hvac); FIRE / PLUMBING as v1
TOL = 2.0  # PDF points
TRUE_FRACTION = 0.5  # "most of its length"

FIRE_TRUE_NAMES = frozenset({"Line"})
HVAC_TRUE_RE = re.compile(r"^(Duct|Elbow|Concentric Reducer|Eccentric Reducer|Chân rẽ)")
#: PLUMBING GT draws pipes as polylines named "Ống ..." ("Ống chưa rõ loại"); fixtures (Lavabol, Xi_bet...) are other.
#: Same rule as FIRE: a shape is true when most of its length lies along a pipe line (user, 2026-10-20).
PLUMBING_TRUE_RE = re.compile(r"^Ống")
LINE_MODES = ("fire", "plumbing")


def _nfc(s: str | None) -> str:
    return unicodedata.normalize("NFC", s or "").strip()


def is_true_object(mode: str, shape_name: str | None) -> bool:
    name = _nfc(shape_name)
    if mode == "fire":
        return name in FIRE_TRUE_NAMES
    if mode == "hvac":
        return bool(HVAC_TRUE_RE.match(name))
    if mode == "plumbing":
        return bool(PLUMBING_TRUE_RE.match(name))
    raise ValueError(f"unknown mode {mode!r}")


@dataclass
class GTGeometry:
    """In-scope GT objects of one page as shapely geometries (display top-left)."""

    mode: str
    geoms: list  # LineString (fire, or 2-vertex hvac) / Polygon (hvac)
    names: list[str]
    uids: list[str]
    vertices: list[np.ndarray]
    closed: list[bool]

    def __len__(self) -> int:
        return len(self.geoms)

    def tree(self) -> STRtree:
        return STRtree(self.geoms)


def gt_geometry(mode: str, objects: Sequence[dict], transform=None) -> GTGeometry:
    """Build in-scope geometries. ``transform`` (points -> points) is applied to vertices first."""

    geoms, names, uids, verts, closed = [], [], [], [], []
    for o in objects:
        if not is_true_object(mode, o.get("shape_name")):
            continue
        v = np.asarray(o.get("vertices") or [], dtype=np.float64).reshape(-1, 2)
        if transform is not None and len(v):
            v = transform(v)
        if mode in LINE_MODES or len(v) == 2:
            if len(v) < 2:
                continue
            g, c = shapely.linestrings(v), False
        else:
            if len(v) < 3:
                continue
            g = shapely.polygons(v)
            if not g.is_valid:
                g = shapely.make_valid(g)
            c = True
        geoms.append(g)
        names.append(_nfc(o.get("shape_name")))
        uids.append(str(o.get("uid") or o.get("source_id") or len(uids)))
        verts.append(v)
        closed.append(c)
    return GTGeometry(mode=mode, geoms=geoms, names=names, uids=uids, vertices=verts, closed=closed)


def segment_lines(seg: np.ndarray):
    """(m, 4) segments -> array of shapely LineStrings."""

    coords = seg.reshape(-1, 2, 2)
    return shapely.linestrings(coords)


@dataclass
class ShapeLabels:
    label: np.ndarray  # (n,) uint8, 1 = true
    true_frac: np.ndarray  # (n,) float32, fraction of the shape's length inside the GT zone
    length: np.ndarray  # (n,) float32, stroke length in pt
    seg_inside: np.ndarray  # (m,) float32, inside length per segment


def label_shapes(segs: PageSegments, gt: GTGeometry, tol: float = TOL,
                 threshold: float = TRUE_FRACTION, chunk: int = 1_000_000) -> ShapeLabels:
    n = len(segs.shape_offsets) - 1
    seg_len = segs.length
    shape_len = np.bincount(segs.owner, weights=seg_len, minlength=n)
    seg_inside = np.zeros(len(segs.seg), dtype=np.float64)
    if len(gt) and len(segs.seg):
        tree = gt.tree()
        # Prefilter: only segments that come within tol of a GT geometry need length sampling.
        cand: list[np.ndarray] = []
        for a in range(0, len(segs.seg), chunk):
            lines = segment_lines(segs.seg[a:a + chunk])
            hit = tree.query(lines, predicate="dwithin", distance=tol)[0]
            cand.append(np.unique(hit) + a)
        cand_idx = np.concatenate(cand) if cand else np.zeros(0, dtype=np.int64)
        if len(cand_idx):
            pts, local, w = sample_segments(segs.seg[cand_idx])
            inside = np.zeros(len(pts), dtype=bool)
            for a in range(0, len(pts), chunk):
                hit = tree.query(shapely.points(pts[a:a + chunk]), predicate="dwithin", distance=tol)[0]
                inside[np.unique(hit) + a] = True
            seg_inside[cand_idx] = np.bincount(local, weights=np.where(inside, w, 0.0), minlength=len(cand_idx))
    inside_len = np.bincount(segs.owner, weights=seg_inside, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(shape_len > 0, inside_len / np.maximum(shape_len, 1e-12), 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    # A shape with no measurable length (a dot) is true when any of its points lies in the zone.
    dots = shape_len <= 0
    if dots.any() and len(gt) and len(segs.seg):
        dot_segs = np.flatnonzero(dots[segs.owner])
        if len(dot_segs):
            mids = (segs.seg[dot_segs, :2] + segs.seg[dot_segs, 2:]) / 2
            hit = gt.tree().query(shapely.points(mids), predicate="dwithin", distance=tol)[0]
            owners = np.unique(segs.owner[dot_segs[np.unique(hit)]])
            frac[owners] = 1.0
    label = (frac >= threshold).astype(np.uint8)
    return ShapeLabels(label=label, true_frac=frac.astype(np.float32), length=shape_len.astype(np.float32),
                       seg_inside=seg_inside.astype(np.float32))


#: HVAC refinement (user, 2026-10-20; LEDGER F18): a shape inside a duct polygon stays true only if most of its length is
#: within ``TOL`` of a polygon outline, or its layer has at least this share of its length inside ducts on the page.
#: Strokes of other disciplines that a duct is drawn over (architecture, structure, parking) become other.
HVAC_LAYER_SHARE = 0.5
HVAC_RULE = "inside duct polygon AND (on outline <= 2 pt OR layer >= 50 % inside ducts)"


def outline_geometry(gt: GTGeometry) -> GTGeometry:
    """The same objects as outlines: polygon boundaries, lines unchanged."""

    geoms = []
    for g, c in zip(gt.geoms, gt.closed):
        if not c:
            geoms.append(g)
            continue
        parts = [q for q in shapely.get_parts(g) if q.geom_type in ("Polygon", "MultiPolygon")]
        b = shapely.union_all([q.boundary for q in parts]) if parts else g.boundary
        geoms.append(b)
    return GTGeometry(mode=gt.mode, geoms=geoms, names=gt.names, uids=gt.uids, vertices=gt.vertices,
                      closed=[False] * len(geoms))


def refine_hvac(segs: PageSegments, gt: GTGeometry, labels: ShapeLabels, layer_id: np.ndarray,
                tol: float = TOL, layer_share: float = HVAC_LAYER_SHARE) -> tuple[ShapeLabels, dict]:
    """Apply the HVAC refinement to ``label_shapes`` output; returns the new labels and per-shape diagnostics."""

    on = label_shapes(segs, outline_geometry(gt), tol=tol)
    L = labels.length.astype(np.float64)
    lid = np.asarray(layer_id, dtype=np.int64)
    n_lay = int(lid.max()) + 1 if len(lid) else 1
    inside_len = np.bincount(lid, weights=labels.true_frac * L, minlength=n_lay)
    lay_len = np.bincount(lid, weights=L, minlength=n_lay)
    share = np.where(lay_len > 0, inside_len / np.maximum(lay_len, 1e-12), 0.0)
    keep = (on.true_frac >= TRUE_FRACTION) | (share[lid] >= layer_share)
    label = (labels.label.astype(bool) & keep).astype(np.uint8)
    return (ShapeLabels(label=label, true_frac=labels.true_frac, length=labels.length, seg_inside=labels.seg_inside),
            {"outline_frac": on.true_frac.astype(np.float32), "layer_inside_share": share[lid].astype(np.float32),
             "label_inside_only": labels.label})


def gt_outline_length(gt: GTGeometry) -> np.ndarray:
    """Per GT object: length of its line, or of its polygon outline."""

    return np.array([polyline_points(v, closed=c)[1].sum() for v, c in zip(gt.vertices, gt.closed)],
                    dtype=np.float64)


def gt_coverage(gt: GTGeometry, true_seg: np.ndarray, tol: float = TOL) -> np.ndarray:
    """Per GT object: fraction of its outline length (line, or polygon boundary) within ``tol`` of a ``true``
    stroke. Low values mean the GT has no labelled stroke underneath."""

    cov = np.zeros(len(gt), dtype=np.float64)
    if not len(gt) or not len(true_seg):
        return cov
    tree = STRtree(segment_lines(true_seg))
    all_pts, all_w, owner = [], [], []
    for i, (v, c) in enumerate(zip(gt.vertices, gt.closed)):
        p, w = polyline_points(v, closed=c)
        all_pts.append(p)
        all_w.append(w)
        owner.append(np.full(len(p), i, dtype=np.int64))
    pts = np.concatenate(all_pts)
    w = np.concatenate(all_w)
    own = np.concatenate(owner)
    inside = np.zeros(len(pts), dtype=bool)
    hit = tree.query(shapely.points(pts), predicate="dwithin", distance=tol)[0]
    inside[np.unique(hit)] = True
    tot = np.bincount(own, weights=w, minlength=len(gt))
    ins = np.bincount(own, weights=np.where(inside, w, 0.0), minlength=len(gt))
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.where(tot > 0, ins / np.maximum(tot, 1e-12), 0.0)
    return cov
