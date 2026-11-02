"""Post-processing after the model, per system (research/LEDGER.md F41, F47, F48).

- FIRE, PLUMBING: strokes along the model's confident strokes (the label rule applied to them), ``along_confident``.
  PLUMBING uses FIRE's threshold; it has not been measured.
- HVAC: the model picks layers plus stray confident strokes (``hvac_input``), AI_Takeoff's geometry logic builds the objects
  (``tools/run_hvac_logic.py``, a separate process because that logic reads PDFs with PyMuPDF), ``objects_mask`` brings them back to strokes.

Thresholds were picked on validation pages of E30 / E37.
"""

from __future__ import annotations

import numpy as np
import shapely

from .geometry import PageSegments
from .labels import GTGeometry, gt_geometry, label_shapes, refine_hvac

ALONG_THRESHOLD = {"fire": 0.55, "plumbing": 0.55, "hvac": 0.35}
HVAC_LAYER_MEAN = 0.1  # a layer goes to the logic when its length-weighted mean p reaches this (E42)
HVAC_STRAY = 0.05  # strokes outside those layers that still go to the logic (E42)
MASK_LAYER = "__MASK__"


def along_confident(segs: PageSegments, p: np.ndarray, t: float) -> np.ndarray:
    """True for the shapes with p >= t and for the shapes lying along their segments."""

    pred = p >= t
    m = pred[segs.owner]
    geoms = list(shapely.linestrings(segs.seg[m].reshape(-1, 2, 2)))
    g = GTGeometry("fire", geoms, ["Line"] * len(geoms), [""] * len(geoms), [], [])
    return label_shapes(segs, g).label.astype(bool) | pred


def hvac_input(p: np.ndarray, length: np.ndarray, layers: list[str]) -> tuple[list[str], np.ndarray]:
    """-> (layers the logic reads, shapes outside them to hand over too)."""

    names = np.asarray(layers)
    uniq, inv = np.unique(names, return_inverse=True)
    mean = np.bincount(inv, p * length, len(uniq)) / np.maximum(np.bincount(inv, length, len(uniq)), 1e-9)
    picked = mean >= HVAC_LAYER_MEAN
    return [str(x) for x in uniq[picked] if x], (p >= HVAC_STRAY) & ~picked[inv]


def objects_mask(mode: str, objects: list[dict], segs: PageSegments, layer_id: np.ndarray) -> np.ndarray:
    """The label rule on the logic's objects: which shapes they cover."""

    geo = gt_geometry(mode, objects)
    labels = label_shapes(segs, geo)
    if mode == "hvac":
        labels, _ = refine_hvac(segs, geo, labels, layer_id)
    return labels.label.astype(bool)
