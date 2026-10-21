"""Per-shape token features from an E0 npz (numpy + scipy only, no torch).

The same feature vector feeds the tree baseline (E1) and model v1 (E2), so the model's gain over the
tree measures learned context, not better inputs. Coordinates are kept apart (``xy``) for the model's
positional terms; the tree gets them only through the neighbourhood statistics.

Layer ids carry no meaning across drawings: only page-level statistics *of* the layer are features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURE_VERSION = "f1"

FEATURE_NAMES = (
    # geometry of the shape itself
    "log_len", "log_bbox_w", "log_bbox_h", "log_bbox_diag", "elong", "fill_ratio", "dir_cos2", "dir_sin2",
    "dir_coherence", "log_nseg", "frac_l", "frac_c", "frac_re", "frac_qu", "closed_ends",
    # style
    "type_s", "type_f", "type_fs", "log_width", "stroke_r", "stroke_g", "stroke_b", "has_stroke",
    "fill_r", "fill_g", "fill_b", "has_fill", "stroke_opacity", "fill_opacity", "close_path", "even_odd",
    "dashed", "seqno_rank",
    # page-level statistics of the shape's layer and style (no layer identity)
    "layer_shape_share", "layer_len_share", "layer_log_count", "layer_mean_log_len", "layer_bbox_area_share",
    "page_has_layers", "style_shape_share", "style_len_share",
    # neighbourhood (scipy cKDTree on bbox centres)
    "nb10_log", "nb50_log", "nb50_same_layer", "nb50_same_style", "nb50_mean_log_len", "nn_dist_log",
)
N_FEATURES = len(FEATURE_NAMES)
#: named groups for ``feature_drop`` (model input masks). ``layer_stats``: page-level statistics of the shape's layer, a
#: shortcut that breaks on pages with one / no layer (LEDGER F28: 100 % of 7. FP_page 16's pipe length out of the train
#: range on all six); grouping by layer stays relational (LayerPool).
FEATURE_GROUPS = {
    "layer_stats": ("layer_shape_share", "layer_len_share", "layer_log_count", "layer_mean_log_len",
                    "layer_bbox_area_share", "page_has_layers"),
    "style_stats": ("style_shape_share", "style_len_share"),
    "colour": ("stroke_r", "stroke_g", "stroke_b", "fill_r", "fill_g", "fill_b"),
}


def feature_mask(drop) -> list[int]:
    """Indices of FEATURE_NAMES to hide, from group names and / or feature names."""
    out = []
    for d in drop or ():
        out += [FEATURE_NAMES.index(n) for n in FEATURE_GROUPS.get(d, (d,))]
    return sorted(set(out))


@dataclass
class PageTokens:
    stem: str
    mode: str
    x: np.ndarray  # (n, F) float32 features
    xy: np.ndarray  # (n, 2) float32 bbox centres, pt
    layer: np.ndarray  # (n,) int64 page-local layer id (grouping only)
    label: np.ndarray  # (n,) uint8
    length: np.ndarray  # (n,) float32 pt (metric weight)
    keep: np.ndarray  # (n,) bool: shapes with geometry (others are always other, never scored)


def _group_stats(key: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    return np.bincount(key, weights=values, minlength=n_groups)


def page_tokens(npz_path: str | Path, mode: str | None = None) -> PageTokens:
    import json

    from scipy.spatial import cKDTree

    z = np.load(npz_path)
    meta = json.loads(str(z["meta"]))
    n = len(z["label"])
    seg = z["segments"].astype(np.float64)
    off = z["shape_offsets"]
    owner = np.repeat(np.arange(n), np.diff(off))
    L = z["length"].astype(np.float64)
    bbox = z["bbox"].astype(np.float64)
    keep = ~np.isnan(bbox[:, 0])
    bbox = np.where(np.isnan(bbox), 0.0, bbox)
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    diag = np.hypot(w, h)
    xy = np.stack([(bbox[:, 0] + bbox[:, 2]) / 2, (bbox[:, 1] + bbox[:, 3]) / 2], axis=1)

    # length-weighted doubled-angle direction (undirected), and how coherent it is
    dx, dy = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
    sl = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)
    c2 = np.bincount(owner, weights=sl * np.cos(2 * ang), minlength=n)
    s2 = np.bincount(owner, weights=sl * np.sin(2 * ang), minlength=n)
    mag = np.hypot(c2, s2)
    Ls = np.maximum(L, 1e-9)
    dir_cos2, dir_sin2 = np.where(mag > 0, c2 / np.maximum(mag, 1e-12), 0), np.where(mag > 0, s2 / np.maximum(mag, 1e-12), 0)
    coherence = mag / Ls
    nseg = np.diff(off).astype(np.float64)
    ops = z["op_counts"].astype(np.float64)
    nops = np.maximum(ops.sum(axis=1), 1)
    # do the shape's first and last points meet (a closed outline drawn as lines)?
    first = np.where(nseg > 0, off[:-1], 0)
    last = np.where(nseg > 0, off[1:] - 1, 0)
    closed_ends = (nseg > 0) & (np.hypot(seg[first, 0] - seg[last, 2], seg[first, 1] - seg[last, 3]) < 1e-3) if len(seg) else np.zeros(n, bool)

    t = z["type_code"]
    stroke = z["stroke_rgb"].astype(np.float64)
    fill = z["fill_rgb"].astype(np.float64)
    has_stroke = ~np.isnan(stroke[:, 0])
    has_fill = ~np.isnan(fill[:, 0])
    width = np.nan_to_num(z["width"].astype(np.float64), nan=0.0)
    seq = z["seqno"].astype(np.float64)
    seq_rank = np.argsort(np.argsort(seq, kind="stable"), kind="stable") / max(n - 1, 1)

    layer = z["layer_id"].astype(np.int64)
    nl = int(layer.max()) + 1 if n else 1
    tot_len = max(L.sum(), 1e-9)
    l_cnt = _group_stats(layer, np.ones(n), nl)
    l_len = _group_stats(layer, L, nl)
    l_loglen = _group_stats(layer, np.log1p(L), nl) / np.maximum(l_cnt, 1)
    l_area = _group_stats(layer, w * h, nl)
    page_area = max((bbox[keep, 2].max() - bbox[keep, 0].min()) * (bbox[keep, 3].max() - bbox[keep, 1].min()), 1.0) if keep.any() else 1.0
    names = z["layer_names"]
    page_has_layers = float(len(names) > 1 or (len(names) == 1 and str(names[0]) not in ("", "None")))

    col = np.nan_to_num(stroke, nan=-1.0)
    skey = np.stack([layer.astype(np.float64), np.round(col, 3)[:, 0], np.round(col, 3)[:, 1], np.round(col, 3)[:, 2],
                     np.round(width, 2), t.astype(np.float64)], axis=1)
    _, style = np.unique(skey, axis=0, return_inverse=True)
    style = style.reshape(-1)
    ns = int(style.max()) + 1 if n else 1
    s_cnt = _group_stats(style, np.ones(n), ns)
    s_len = _group_stats(style, L, ns)

    # neighbourhood
    nb10 = np.zeros(n)
    nb50 = np.zeros(n)
    same_l = np.zeros(n)
    same_s = np.zeros(n)
    nb_len = np.zeros(n)
    nn = np.zeros(n)
    idx = np.flatnonzero(keep)
    if len(idx) > 1:
        tree = cKDTree(xy[idx])
        nb10[idx] = tree.query_ball_point(xy[idx], r=10.0, return_length=True) - 1
        d, j = tree.query(xy[idx], k=min(17, len(idx)))
        nn[idx] = d[:, 1]
        jj = idx[j[:, 1:]]
        within = d[:, 1:] <= 50.0
        cnt = within.sum(axis=1)
        nb50[idx] = cnt
        safe = np.maximum(cnt, 1)
        same_l[idx] = ((layer[jj] == layer[idx][:, None]) & within).sum(axis=1) / safe
        same_s[idx] = ((style[jj] == style[idx][:, None]) & within).sum(axis=1) / safe
        nb_len[idx] = (np.log1p(L[jj]) * within).sum(axis=1) / safe

    feats = [
        np.log1p(L), np.log1p(w), np.log1p(h), np.log1p(diag), np.minimum(w, h) / np.maximum(np.maximum(w, h), 1e-6),
        L / np.maximum(diag, 1e-6) / 10.0, dir_cos2, dir_sin2, coherence, np.log1p(nseg),
        ops[:, 0] / nops, ops[:, 1] / nops, ops[:, 2] / nops, ops[:, 3] / nops, closed_ends.astype(np.float64),
        (t == 0), (t == 1), (t == 2), np.log1p(width * 10), np.where(has_stroke, stroke[:, 0], 0),
        np.where(has_stroke, stroke[:, 1], 0), np.where(has_stroke, stroke[:, 2], 0), has_stroke,
        np.where(has_fill, fill[:, 0], 0), np.where(has_fill, fill[:, 1], 0), np.where(has_fill, fill[:, 2], 0), has_fill,
        np.nan_to_num(z["stroke_opacity"], nan=1.0), np.nan_to_num(z["fill_opacity"], nan=1.0), z["close_path"],
        z["even_odd"], z["dashed"], seq_rank,
        l_cnt[layer] / max(n, 1), l_len[layer] / tot_len, np.log1p(l_cnt[layer]), l_loglen[layer],
        np.minimum(l_area[layer] / page_area, 10.0), np.full(n, page_has_layers),
        s_cnt[style] / max(n, 1), s_len[style] / tot_len,
        np.log1p(nb10), np.log1p(nb50), same_l, same_s, nb_len, np.log1p(nn),
    ]
    x = np.stack([np.asarray(f, dtype=np.float64) for f in feats], axis=1).astype(np.float32)
    assert x.shape[1] == N_FEATURES
    return PageTokens(stem=meta["stem"], mode=mode or meta["mode"], x=x, xy=xy.astype(np.float32), layer=layer,
                      label=z["label"].astype(np.uint8), length=L.astype(np.float32), keep=keep)


#: Points resampled along each shape for the learned shape encoder (OpenStroke2).
SHAPE_POINTS = 16


def shape_points(segments: np.ndarray, offsets: np.ndarray, p: int = SHAPE_POINTS) -> np.ndarray:
    """(n, p, 2) float32: ``p`` points at equal arc-length steps along each shape (beziers already flattened),
    relative to the shape's bbox centre. Shapes without length repeat their first point (or 0)."""

    n = len(offsets) - 1
    out = np.zeros((n, p, 2), np.float32)
    if n == 0 or len(segments) == 0:
        return out
    seg = segments.astype(np.float64)
    sl = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(sl)])  # cum[i] = arc length before segment i
    cnt = np.diff(offsets)
    has = cnt > 0
    start, end = offsets[:-1], offsets[1:]
    L = cum[end] - cum[start]
    t = (np.arange(p) + 0.5) / p
    target = cum[start][:, None] + t[None, :] * L[:, None]  # (n, p)
    si = np.searchsorted(cum, target, side="right") - 1
    si = np.clip(si, start[:, None], np.maximum(end - 1, start)[:, None])
    si = np.clip(si, 0, len(seg) - 1)
    local = np.where(sl[si] > 0, (target - cum[si]) / np.maximum(sl[si], 1e-12), 0.0)
    local = np.clip(local, 0.0, 1.0)
    px = seg[si, 0] + (seg[si, 2] - seg[si, 0]) * local
    py = seg[si, 1] + (seg[si, 3] - seg[si, 1]) * local
    pts = np.stack([px, py], axis=-1)
    x0 = np.full(n, np.inf)
    y0 = np.full(n, np.inf)
    x1 = np.full(n, -np.inf)
    y1 = np.full(n, -np.inf)
    owner = np.repeat(np.arange(n), cnt)
    np.minimum.at(x0, owner, np.minimum(seg[:, 0], seg[:, 2]))
    np.minimum.at(y0, owner, np.minimum(seg[:, 1], seg[:, 3]))
    np.maximum.at(x1, owner, np.maximum(seg[:, 0], seg[:, 2]))
    np.maximum.at(y1, owner, np.maximum(seg[:, 1], seg[:, 3]))
    x0, y0, x1, y1 = (np.where(has, v, 0.0) for v in (x0, y0, x1, y1))
    centre = np.stack([(x0 + x1) / 2, (y0 + y1) / 2], axis=1)
    out[has] = (pts[has] - centre[has, None, :]).astype(np.float32)
    return out
