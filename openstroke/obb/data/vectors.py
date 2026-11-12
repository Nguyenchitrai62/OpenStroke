"""Page-level vector tokens -> per-tile tokens.

Page npz (tools/pdf_vectors.py, OpenStroke, or any CAD/SVG source), all in full-image pixel coordinates:
  tokens (N, >= 21) float32: type, 8 x (x, y) points resampled along the primitive, r, g, b in [0, 1], line width (px)
          optional columns: 21 layer (-1 = none), 22 inst (0-based line index of the object in the page's label file,
          -1 = no object), 23 pid (the primitive's id in the source page, e.g. the payload shape index)
  layer  (N,)    int32:   CAD layer index per primitive (-1 = none); layer_names (L,) optional (older pages)
Types: 0 line / stroke, 1 arc / fill, 2 circle / fill + stroke, 3 ellipse, 4 bezier curve (others reserved).

Tiles (training layout read by DotaPatches(vectors=True)): {root}/vectors/{split}/{tile}.npz with
  tokens (M, 22): type, points / tile size (tile-relative), rgb, width (px at the tile's scale), layer + 1 (0 = none)
  inst (M,) int32: index of the object in the tile's own label list (meta "objs"), -1 = none (absent: unknown)
  pid  (M,) int64: primitive id in the source page (-1 = unknown);  view: tile size (widths are px)
"""
import numpy as np

INST_UNKNOWN = -2           # model-input inst column when the data has no object ids (membership from the boxes)


def page_columns(page):
    """Page npz dict -> (tokens (N, 21) float32, layer (N,) int64, inst (N,) int64 or None, pid (N,) int64)."""
    t = np.asarray(page["tokens"], np.float32)
    if t.size == 0:
        t = np.zeros((0, 21), np.float32)
    n, c = t.shape
    if "layer" in page:
        layer = np.asarray(page["layer"]).astype(np.int64).reshape(-1)
    elif c > 21:
        layer = np.rint(t[:, 21]).astype(np.int64)
    else:
        layer = np.full(n, -1, np.int64)
    if "inst" in page:
        inst = np.asarray(page["inst"]).astype(np.int64).reshape(-1)
    else:
        inst = np.rint(t[:, 22]).astype(np.int64) if c > 22 else None
    if "pid" in page:
        pid = np.asarray(page["pid"]).astype(np.int64).reshape(-1)
    else:
        pid = np.rint(t[:, 23]).astype(np.int64) if c > 23 else np.arange(n, dtype=np.int64)
    return t[:, :21], layer, inst, pid


def _inside_any(pts, x0, y0, size):
    x, y = pts[:, 0::2] - x0, pts[:, 1::2] - y0
    return ((x >= 0) & (x <= size) & (y >= 0) & (y <= size)).any(1)


def cap_order(pts, max_tokens):
    """Deterministic token cap (OpenOBB25): indices of the primitives kept when more than max_tokens fall in a tile,
    longest first (extent of the 8 points; ties by input order), returned in input order. pts (M, 16) any unit."""
    n = len(pts)
    if not max_tokens or n <= max_tokens:
        return np.arange(n)
    p = np.asarray(pts, np.float64).reshape(n, 8, 2)
    ext = np.hypot(p[..., 0].max(1) - p[..., 0].min(1), p[..., 1].max(1) - p[..., 1].min(1))
    order = np.lexsort((np.arange(n), -ext))[:max_tokens]
    return np.sort(order)


def small_strokes(pts, min_extent):
    """OpenStroke23 pre-filter: strokes whose bounding box is smaller than min_extent input px on both sides (hatch / glyph
    fragments / dots: 62-66 % of all strokes, 0.04 % of GT object length, research/probes/noise_filter_probe.py) -> bool mask."""
    p = np.asarray(pts, np.float64).reshape(len(pts), -1, 2)
    return (p.max(1) - p.min(1)).max(1) < float(min_extent)


def cut_tile(page, x0, y0, size, rate=1.0, members=False, max_tokens=None, layers=True, stats=None, min_extent=0.0):
    """Model input of one tile: (M, 22) type, points / size (tile-relative), rgb, width / size, layer + 1 (0 = none);
    members=True adds inst (INST_UNKNOWN at inference) and pid columns -> (M, 24). The page is in full-image pixels,
    `rate` is the image scale before tiling (tile_scale / whole-image resize). max_tokens: deterministic cap
    (cap_order); stats (dict) counts "tiles", "capped", "tokens", "dropped". layers=False: layer column 0 (OpenOBB25
    does not use CAD layers)."""
    width = 24 if members else 22
    t, layer, _, pid = page_columns(page)
    if len(t) == 0:
        return np.zeros((0, width), np.float32)
    pts = t[:, 1:17] * rate
    inside = _inside_any(pts, x0, y0, size)
    if min_extent:
        inside &= ~small_strokes(pts, min_extent)
    keep = np.nonzero(inside)[0]
    if max_tokens:
        sel = cap_order(pts[keep], max_tokens)
        if stats is not None:
            stats["tiles"] = stats.get("tiles", 0) + 1
            stats["tokens"] = stats.get("tokens", 0) + len(keep)
            if len(sel) < len(keep):
                stats["capped"] = stats.get("capped", 0) + 1
                stats["dropped"] = stats.get("dropped", 0) + len(keep) - len(sel)
        keep = keep[sel]
    p = pts[keep].copy()
    p[:, 0::2] -= x0
    p[:, 1::2] -= y0
    lay = (layer[keep] + 1).clip(min=0).astype(np.float32) if layers else np.zeros(len(keep), np.float32)
    cols = [t[keep, :1], p / size, t[keep, 17:20], t[keep, 20:21] * rate / size, lay[:, None]]
    if members:
        cols += [np.full((len(keep), 1), INST_UNKNOWN, np.float32), pid[keep, None].astype(np.float32)]
    return np.concatenate(cols, 1).astype(np.float32).reshape(-1, width)


def tile_npz(page, x0, y0, size, path, rate=1.0, inst_map=None):
    """Write a tile in the training layout read by DotaPatches(vectors=True) (same as FloorPlanCAD tiles).
    rate: image scale before tiling (points and widths follow). inst_map: page inst -> tile object index (dict,
    missing / -1 = the object is not in this tile); without it (or without inst in the page) no inst is written.
    Returns the number of primitives written."""
    t, layer, inst, pid = page_columns(page)
    pts = t[:, 1:17] * rate
    keep = _inside_any(pts, x0, y0, size) if len(t) else np.zeros(0, bool)
    t, layer, pid, pts = t[keep], layer[keep], pid[keep], pts[keep].copy()
    pts[:, 0::2] -= x0
    pts[:, 1::2] -= y0
    rows = np.concatenate([t[:, :1], pts / size, t[:, 17:20], t[:, 20:21] * rate,
                           np.maximum(layer + 1, 0)[:, None].astype(np.float32)], 1)
    out = {"tokens": rows.astype(np.float32).reshape(-1, 22), "layer": layer.astype(np.int32),
           "pid": pid.astype(np.int64), "view": np.float32(size)}
    if inst is not None and inst_map is not None:
        # A positive object omitted by tile IoF is unlabelled in this tile, not real background.
        out["inst"] = np.array([int(i) if i<0 else (inst_map[int(i)] if inst_map.get(int(i),-1)>=0 else INST_UNKNOWN)
                                for i in inst[keep]],np.int32)
    np.savez_compressed(path, **out)
    return int(keep.sum())
