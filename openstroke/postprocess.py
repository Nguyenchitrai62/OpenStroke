"""Post-processing after the model, per system (research/LEDGER.md F41, F47, F48, F52). The logic is shared; its parameters belong to the model.

- FIRE: strokes along the model's confident strokes (the label rule applied to them), ``fire_mask``, plus the layer-name and colour steps of ``fire_final`` (E55).
- PLUMBING: ``fire_mask`` with FIRE's parameters until it has pages to be calibrated on; the E55 steps are not measured on it.
- HVAC: the model picks layers plus stray confident strokes (``hvac_input``), AI_Takeoff's geometry logic builds the objects
  (``tools/run_hvac_logic.py``, a separate process because that logic reads PDFs with PyMuPDF), ``relabel.object_strokes`` brings them back to strokes, and ``hvac_mask``
  adds back what the logic dropped although the model is sure of it.
- Both: a layer + style group whose mean p is high is taken whole, so a pipe is not left in pieces; then ``tidy``.

``openstroke calibrate <run>`` picks the parameters on that run's val scores and writes ``postprocess.json`` beside its folds; ``load_params`` finds it from a
checkpoint path. ``DEFAULT_PARAMS`` were picked on E30 and only stand in for a model without its own.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .geometry import PageSegments
from .labels import GTGeometry, gt_geometry, label_shapes, refine_hvac

ALONG_THRESHOLD = {"fire": 0.55, "plumbing": 0.55, "hvac": 0.35}
HVAC_LAYER_MEAN = 0.1  # a layer goes to the logic when its length-weighted mean p reaches this (E42)
HVAC_STRAY = 0.05  # strokes outside those layers that still go to the logic (E42)
DEFAULT_PARAMS = {"fire": {"seed": 0.55, "group": 0.6}, "plumbing": {"seed": 0.55, "group": 0.6},
                  "hvac": {"layer_mean": HVAC_LAYER_MEAN, "stray": HVAC_STRAY, "sure": 0.8, "group": 0.8}}  # E51
PARAMS_FILE = "postprocess.json"
MASK_LAYER = "__MASK__"
MESH_MAX_STROKE, MESH_MIN_CLUSTER, MESH_MIN_JUNCTION = 15.0, 30, 0.5  # pt, strokes, share of stroke ends at a junction
FILL_CELL, FILL_MIN = 4.0, 4  # pt, strokes: short strokes this crowded fill a symbol
COMPACT_DIAG, COMPACT_RATIO = 10.0, 3.0  # pt, length / diagonal: one shape folded into a small box is a symbol (F59)
BRIDGE_TOL = 2.0  # pt, the label rule's tolerance
OP_CURVE = 1  # build.OPS: l, c, re, qu
# E55: words read on the FIRE val pages, checked on 16 other pages (research/probes/layer_name_probe.py). A rule outside the model: names as a model input hurt (E12, E27).
NOT_PIPE = re.compile(r"(?<![A-Z])(DIMS?|TEXT|TXT|GRID|AXIS|NOTE|ANNO)(?![A-Z])|NETTRUC|LUOI ?TRUC", re.I)
IS_PIPE = re.compile(r"PIPE|(?<![A-Z])ONG(?![A-Z])|(?<![A-Z])DN ?\d{2,3}(?!\d)|(?<![A-Z0-9])D\d{2,3}(?!\d)|SPK_(MAIN|BR)", re.I)
GREY_SPREAD, GREY_MIN_LUM, GREY_MAX_SHARE = 0.08, 0.25, 0.2  # colour spread, lightness, share of the confident strokes that may be grey for grey to be dropped


def params_file(model: str | Path) -> Path | None:
    """The run's ``postprocess.json``: beside the checkpoint, or one level up for ``<run>/fold<k>/best.pt``."""

    d = Path(model).parent
    return next((x / PARAMS_FILE for x in (d, d.parent) if (x / PARAMS_FILE).exists()), None)


def load_params(model: str | Path) -> tuple[dict, str | None]:
    """-> (parameters per system, the file they came from; None = this model has not been calibrated, the defaults stand in)."""

    f = params_file(model)
    if f is None:
        return DEFAULT_PARAMS, None
    own = json.loads(f.read_text(encoding="utf-8"))
    return {m: {**DEFAULT_PARAMS[m], **{k: v for k, v in own.get(m, {}).items() if k in DEFAULT_PARAMS[m]}} for m in DEFAULT_PARAMS}, str(f)


def style_groups(attrs: dict) -> np.ndarray:
    """(n,) group id: same layer, stroke colour, width and type."""

    rgb = np.nan_to_num(attrs["stroke_rgb"], nan=-1).round(2)
    key = np.c_[attrs["layer_id"], rgb, np.round(np.nan_to_num(attrs["width"], nan=-1), 2), attrs["type_code"]]
    return np.unique(key, axis=0, return_inverse=True)[1].ravel()


def by_group(p: np.ndarray, length: np.ndarray, group: np.ndarray, a: float) -> np.ndarray:
    """True for the shapes of groups whose length-weighted mean p reaches ``a``."""

    n = int(group.max()) + 1 if len(group) else 1
    mean = np.bincount(group, p * length, n) / np.maximum(np.bincount(group, length, n), 1e-9)
    return (mean >= a)[group]


def along_confident(segs: PageSegments, p: np.ndarray, t: float) -> np.ndarray:
    """True for the shapes with p >= t and for the shapes lying along their segments."""

    pred = p >= t
    m = pred[segs.owner]
    geoms = list(shapely.linestrings(segs.seg[m].reshape(-1, 2, 2)))
    g = GTGeometry("fire", geoms, ["Line"] * len(geoms), [""] * len(geoms), [], [])
    return label_shapes(segs, g).label.astype(bool) | pred


def _shape_ends(segs: PageSegments) -> tuple[np.ndarray, np.ndarray]:
    """First and last point of every shape, (n, 2) each; shapes without segments get NaN."""

    off = segs.shape_offsets
    n = len(off) - 1
    first, last = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    some = np.flatnonzero(np.diff(off) > 0)
    first[some] = segs.seg[off[some], :2]
    last[some] = segs.seg[off[some + 1] - 1, 2:]
    return first, last


def mesh_strokes(segs: PageSegments, group: np.ndarray, length: np.ndarray) -> np.ndarray:
    """Hatch such as honeycomb insulation, stroke by stroke: short strokes of one style connected end to end into a large cluster where most ends meet at junctions of three or more.
    A whole style group cannot be dropped: it also holds real duct outlines and flexible connectors (F54). Dashes do not meet and a chain of short strokes along an arc has no junctions, so both stay."""

    first, last = _shape_ends(segs)
    c = np.flatnonzero(np.isfinite(first[:, 0]) & (length <= MESH_MAX_STROKE))
    out = np.zeros(len(group), bool)
    if not len(c):
        return out
    q = np.floor(np.concatenate([first[c], last[c]]) / 0.5).astype(np.int64)
    idx = np.r_[np.arange(len(c)), np.arange(len(c))]
    _, inv, k = np.unique(np.c_[np.r_[group[c], group[c]], q], axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    hub = np.full(len(k), -1, np.int64)  # every end in a cell links to one stroke of that cell
    hub[inv[::-1]] = idx[::-1]
    cluster = connected_components(coo_matrix((np.ones(len(idx), bool), (idx, hub[inv])), shape=(len(c), len(c))), directed=False)[1]
    n = int(cluster.max()) + 1
    size = np.bincount(cluster, minlength=n)
    junction = np.bincount(cluster[idx[k[inv] >= 3]], minlength=n) / np.maximum(2 * size, 1)
    out[c[((size >= MESH_MIN_CLUSTER) & (junction >= MESH_MIN_JUNCTION))[cluster]]] = True
    return out


def fill_strokes(segs: PageSegments, group: np.ndarray, length: np.ndarray) -> np.ndarray:
    """Short strokes of one style crowded into one small cell: a solid dot or symbol drawn as parallel strokes. Dashes along a pipe are spaced apart."""

    out = np.zeros(len(group), bool)
    c = np.flatnonzero((length <= MESH_MAX_STROKE) & (np.diff(segs.shape_offsets) > 0))
    if not len(c):
        return out
    off = segs.shape_offsets
    mid = (segs.seg[off[c], :2] + segs.seg[off[c + 1] - 1, 2:]) / 2
    _, inv, k = np.unique(np.c_[group[c], np.floor(mid / FILL_CELL).astype(np.int64)], axis=0, return_inverse=True, return_counts=True)
    out[c] = k[inv.ravel()] >= FILL_MIN
    return out


def compact_strokes(segs: PageSegments, length: np.ndarray) -> np.ndarray:
    """One shape folded into a small box, far longer than the box is wide: a solid symbol drawn as many strokes. ``fill_strokes`` cannot see it because it looks at short strokes."""

    off = segs.shape_offsets
    out = np.zeros(len(off) - 1, bool)
    some = np.flatnonzero(np.diff(off) > 0)
    if not len(some):
        return out
    lo = np.minimum.reduceat(np.minimum(segs.seg[:, :2], segs.seg[:, 2:]), off[some])
    hi = np.maximum.reduceat(np.maximum(segs.seg[:, :2], segs.seg[:, 2:]), off[some])
    diag = np.hypot(*(hi - lo).T)
    out[some] = (diag <= COMPACT_DIAG) & (length[some] >= COMPACT_RATIO * diag)
    return out


def bridge_curves(segs: PageSegments, mask: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Chains of curve shapes whose free ends all rest on the mask: the elbow between two pipes. A closed loop (a symbol) has no free end and stays out (F54)."""

    out = np.zeros(len(mask), bool)
    first, last = _shape_ends(segs)
    c = np.flatnonzero(curve & ~mask & np.isfinite(first[:, 0]))
    m = mask[segs.owner]
    if not len(c) or not m.any():
        return out
    ends = np.concatenate([first[c], last[c]])
    idx = np.r_[np.arange(len(c)), np.arange(len(c))]
    # Coincident ends (hatching, stacked copies) made query_pairs quadratic: 54 M pairs and an out-of-memory on a Viewer page (F135).
    # Pairs are found between distinct positions (rounded to 1e-4 pt); ends sharing a position are partners of each other and are
    # joined through their position node. Same partners and chains as before up to that rounding.
    pos, at, mult = np.unique(np.round(ends, 4), axis=0, return_inverse=True, return_counts=True)
    at = at.ravel()
    upairs = cKDTree(pos).query_pairs(BRIDGE_TOL, output_type="ndarray")
    near = np.bincount(upairs.ravel(), weights=mult[upairs[:, ::-1]].ravel(), minlength=len(pos)) if len(upairs) else np.zeros(len(pos))
    partners = np.rint(near[at] + mult[at] - 1).astype(np.int64)
    k, u = len(c), len(pos)
    rows = np.r_[idx, k + upairs[:, 0]]           # shape -> its end positions, position -> position within tolerance
    cols = np.r_[k + at, k + upairs[:, 1]]
    chain = connected_components(coo_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(k + u, k + u)), directed=False)[1][:k]
    free = np.flatnonzero(partners == 0)
    if not len(free):
        return out
    hit = np.zeros(len(free), bool)
    hit[shapely.STRtree(shapely.linestrings(segs.seg[m].reshape(-1, 2, 2))).query(shapely.points(ends[free]), predicate="dwithin", distance=BRIDGE_TOL)[0]] = True
    n = int(chain.max()) + 1
    n_free = np.bincount(chain[idx[free]], minlength=n)
    n_hit = np.bincount(chain[idx[free[hit]]], minlength=n)
    out[c[((n_free >= 2) & (n_hit == n_free))[chain]]] = True
    return out


def tidy(mask: np.ndarray, segs: PageSegments, attrs: dict, group: np.ndarray, length: np.ndarray) -> np.ndarray:
    """Last step for every system: elbows between pipes in, hatch and solid symbols out (F54, E53, F59)."""

    return (mask | bridge_curves(segs, mask, attrs["op_counts"][:, OP_CURVE] > 0)) & ~mesh_strokes(segs, group, length) & ~fill_strokes(segs, group, length) & ~compact_strokes(segs, length)


def fire_mask(segs: PageSegments, p: np.ndarray, length: np.ndarray, group: np.ndarray, params: dict) -> np.ndarray:
    return along_confident(segs, p, params["seed"]) | by_group(p, length, group, params["group"])


def fire_extras(p: np.ndarray, length: np.ndarray, names: list[str], attrs: dict, segs: PageSegments) -> dict[str, np.ndarray]:
    """E55, per shape. ``rescue``: straight open strokes on a layer named as pipe, what saves a page the model misses entirely. ``drop``: layers named as dimensions, text or grid, and grey strokes on pages whose confident strokes are not grey."""

    uniq, inv = np.unique(np.asarray(names), return_inverse=True)
    not_pipe = np.array([bool(NOT_PIPE.search(n)) for n in uniq])[inv]
    is_pipe = np.array([bool(IS_PIPE.search(n)) and not NOT_PIPE.search(n) for n in uniq])[inv]
    first, last = _shape_ends(segs)
    closed = np.hypot(*(first - last).T) < 0.5
    rgb = np.nan_to_num(attrs["stroke_rgb"], nan=0.0)
    grey = (rgb.max(1) - rgb.min(1) < GREY_SPREAD) & (rgb.mean(1) >= GREY_MIN_LUM)
    conf = p >= 0.5
    if length[conf & grey].sum() >= GREY_MAX_SHARE * max(length[conf].sum(), 1e-9):
        grey = np.zeros(len(p), bool)
    return {"rescue": is_pipe & ~closed & ~(attrs["op_counts"][:, OP_CURVE] > 0), "drop": not_pipe | grey}


def fire_final(segs: PageSegments, p: np.ndarray, length: np.ndarray, group: np.ndarray, attrs: dict, names: list[str], params: dict) -> np.ndarray:
    """FIRE, whole: strokes along the model's on the layers it chose, the pipe-named layers, then ``tidy``, then what is named or coloured as not pipe out (E55)."""

    extra = fire_extras(p, length, names, attrs, segs)
    m = on_chosen_layers(fire_mask(segs, p, length, group, params), p, length, names) | extra["rescue"]
    return tidy(m, segs, attrs, group, length) & ~extra["drop"]


def hvac_mask(objects: np.ndarray, p: np.ndarray, length: np.ndarray, group: np.ndarray, params: dict) -> np.ndarray:
    """The logic's strokes, plus what it dropped although the model is sure: whole ducts scored 0.999 went missing on one page (F52)."""

    return objects | (p >= params["sure"]) | by_group(p, length, group, params["group"])


def hvac_input(p: np.ndarray, length: np.ndarray, layers: list[str], layer_mean: float = HVAC_LAYER_MEAN, stray: float = HVAC_STRAY) -> tuple[list[str], np.ndarray]:
    """-> (layers the logic reads, shapes outside them to hand over too)."""

    names = np.asarray(layers)
    uniq, inv = np.unique(names, return_inverse=True)
    if len(uniq) < 2:  # no layers to choose between: the logic reads the strokes the model believes in
        return [], p >= stray
    mean = np.bincount(inv, p * length, len(uniq)) / np.maximum(np.bincount(inv, length, len(uniq)), 1e-9)
    picked = (mean >= layer_mean) & (uniq != "")  # a nameless group is not a layer the logic can be told to read
    return [str(x) for x in uniq[picked]], (p >= stray) & ~picked[inv]


def on_chosen_layers(mask: np.ndarray, p: np.ndarray, length: np.ndarray, layers: list[str], layer_mean: float = HVAC_LAYER_MEAN) -> np.ndarray:
    """Keep only the strokes the model chose (``hvac_input``): the geometric rule also takes cars or grid lines lying along a pipe (F51). Pages with
    one layer have nothing to choose between."""

    names = np.asarray(layers)
    if len(np.unique(names)) < 2:
        return mask
    chosen, _ = hvac_input(p, length, layers, layer_mean)
    return mask & np.isin(names, chosen)


def drop_noise_layers(mask: np.ndarray, p: np.ndarray, length: np.ndarray, layer_id: np.ndarray, coverage: float, mean_p: float) -> np.ndarray:
    """Clear a whole layer when only a small part of it is in ``mask`` and the model does not believe in it (E50): cars or
    architecture lying along a duct's outline get in through the geometric rule, not through the model."""

    n = int(layer_id.max()) + 1 if len(layer_id) else 1
    total = np.maximum(np.bincount(layer_id, length, n), 1e-9)
    cov = np.bincount(layer_id[mask], length[mask], n) / total
    mean = np.bincount(layer_id, p * length, n) / total
    return mask & ~((cov < coverage) & (mean < mean_p))[layer_id]


def objects_mask(mode: str, objects: list[dict], segs: PageSegments, layer_id: np.ndarray) -> np.ndarray:
    """The label rule on the logic's objects: which shapes they cover."""

    geo = gt_geometry(mode, objects)
    labels = label_shapes(segs, geo)
    if mode == "hvac":
        labels, _ = refine_hvac(segs, geo, labels, layer_id)
    return labels.label.astype(bool)
