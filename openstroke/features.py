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
