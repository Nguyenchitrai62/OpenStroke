"""Training data: E0 pages -> serialized tokens -> random windows (augmented) -> batches with U-Net levels.

A page keeps only shapes with geometry, sorted by z-order of their centres. A training sample is a window
of ``window`` consecutive tokens of that order (a compact patch of the sheet). ``collate`` builds, per
window and per level, everything the models need on the CPU side (numpy), so the GPU only gathers:

- level 0: the tokens; level l >= 1: grid cells of ``LEVEL_CELLS[l]`` pt (Point Transformer V3 grid
  pooling via z-code shifts), each with its parent index from the level below and its mean position;
- per level, four serialization orders (z, z-transposed, Hilbert, Hilbert-transposed) as permutations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import os

import numpy as np

from .objects import OFFSET_SCALE
from .features import FEATURE_NAMES, FEATURE_VERSION, N_FEATURES, SHAPE_POINTS, page_tokens, page_tokens_from_arrays, shape_points
from .layernames import MAX_TOKENS, layer_table
from .serialize import ORDERS, codes, quantize, zorder

DATA_VERSION = f"{FEATURE_VERSION}p{SHAPE_POINTS}"
#: Grid cell (pt) of each pooled level. Level 0 is the shapes themselves (1 pt quantization).
LEVEL_CELLS = (1, 2, 8, 32, 128)
#: drawing flows; AI_Takeoff always knows which one a page is run in (LEDGER F14)
MODES = ("fire", "hvac", "plumbing")

_I = {n: i for i, n in enumerate(FEATURE_NAMES)}
DIR_COS, DIR_SIN = _I["dir_cos2"], _I["dir_sin2"]
#: features that are log1p(a length in pt): they follow a scale augmentation
LOG_LENGTH_FEATURES = [_I[k] for k in ("log_len", "log_bbox_w", "log_bbox_h", "log_bbox_diag", "nn_dist_log")]
LOG_WIDTH = _I["log_width"]  # log1p(10 * width)
STROKE = [_I["stroke_r"], _I["stroke_g"], _I["stroke_b"]]
FILL = [_I["fill_r"], _I["fill_g"], _I["fill_b"]]
HAS_STROKE, HAS_FILL = _I["has_stroke"], _I["has_fill"]
TYPE_COLS = [_I["type_s"], _I["type_f"], _I["type_fs"]]


def style_ids(x: np.ndarray) -> np.ndarray:
    """Page-local style group of each shape, like AI_Takeoff's synthetic ``layer_1`` (shape_color_R_G_B_width_W):
    stroke colour, fill colour, width and kind, from the raw (un-normalised) features."""

    n = len(x)
    if not n:
        return np.zeros(0, np.int64)
    sf = np.round(x[:, STROKE + FILL], 3)
    lw = np.round(x[:, LOG_WIDTH], 3)
    hsf = x[:, [HAS_STROKE, HAS_FILL]]
    tc = x[:, TYPE_COLS]
    # Fast exact lexicographic packing into 16-byte big-endian unsigned view (preserves exact unique IDs)
    sf_i = np.rint(sf * 1000.0).astype(np.int64)
    lw_i = np.rint(lw * 1000.0).astype(np.int64)
    if (
        int(sf_i.min()) >= -10000 and int(sf_i.max()) <= 50000
        and int(lw_i.min()) >= -10000 and int(lw_i.max()) <= 50000
        and np.all((hsf == 0.0) | (hsf == 1.0))
        and np.all((tc == 0.0) | (tc == 1.0))
    ):
        # 8 x >u2 = 16 bytes -> exact lexicographic match with (sf0..sf5, has_stroke, has_fill, lw, t0, t1, t2)
        packed = np.empty((n, 9), dtype=">u2")
        packed[:, :6] = (sf_i + 10000).astype(np.uint16)
        packed[:, 6] = ((hsf[:, 0].astype(np.uint16) << 1) | hsf[:, 1].astype(np.uint16))
        packed[:, 7] = (lw_i + 10000).astype(np.uint16)
        packed[:, 8] = ((tc[:, 0].astype(np.uint16) << 2) | (tc[:, 1].astype(np.uint16) << 1) | tc[:, 2].astype(np.uint16))
        view = np.ascontiguousarray(packed).view(np.dtype((np.void, packed.shape[1] * 2))).ravel()
        _, inv = np.unique(view, return_inverse=True)
        return inv.reshape(-1).astype(np.int64)
    key = np.concatenate([sf, hsf, lw[:, None], tc], axis=1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return inv.reshape(-1).astype(np.int64)


@dataclass
class Page:
    stem: str
    mode: str
    x: np.ndarray  # (n, F) float32
    xy: np.ndarray  # (n, 2) float32 centres, pt
    pts: np.ndarray  # (n, P, 2) float16, shape points relative to the centre
    layer: np.ndarray  # (n,) int64
    label: np.ndarray  # (n,) uint8
    length: np.ndarray  # (n,) float32
    index: np.ndarray  # (n,) int64: position in the payload's shape list
    ltab: np.ndarray | None = None  # (n_layers, MAX_TOKENS) hashed tokens of each layer name (layernames.py)
    style: np.ndarray | None = None  # (n,) int64 style group: colour + fill + width + type (AI_Takeoff "layer_1")
    edges: np.ndarray | None = None  # (E, 2) int64 linked token positions (graph.py: endpoint / dash gap / tee)
    ekind: np.ndarray | None = None  # (E,) int64 kind bits of each link (1 endpoint, 2 gap, 4 tee, 8 facing)
    ylayer: np.ndarray | None = None  # (n,) float32: 1 if the shape's layer is a pipe layer of the page (layer head, E24)
    lw: np.ndarray | None = None  # (n,) float32 loss weight of each shape (E36 ``coincide``: 0 for coincident true shapes)
    ocls: np.ndarray | None = None  # (n,) int64 HVAC object class of the shape (objects.HVAC_CLASSES, 0 none; E62)
    octr: np.ndarray | None = None  # (n, 2) float64 centre of that object, pt, page frame (NaN none)
    oinst: np.ndarray | None = None  # (n,) int64 GT object index of the shape (-1 none; OpenStroke30 closure / edge losses)
    pieces: dict | None = None  # OpenStroke34: pieces of the candidate shapes (attach_pieces), sorted by their shape's position
    pair_teacher: np.ndarray | None = None  # training-only probabilities in page token order (E132)

    def __post_init__(self):
        if self.style is None:
            self.style = style_ids(self.x)
        if self.ylayer is None:
            self.ylayer = layer_targets(self.layer, self.label, self.length)

    def __len__(self) -> int:
        return len(self.x)


def layer_targets(layer: np.ndarray, label: np.ndarray, length: np.ndarray, share: float = 0.5) -> np.ndarray:
    """Per shape: 1 when its layer has >= ``share`` of its length true on this page (the oracle of
    ``research/probes/layer_probe.py``), else 0."""

    lid = np.asarray(layer, np.int64)
    if not len(lid):
        return np.zeros(0, np.float32)
    L = np.asarray(length, np.float64)
    tot = np.bincount(lid, weights=L)
    tru = np.bincount(lid, weights=L * (np.asarray(label) > 0), minlength=len(tot))
    return (tru >= share * np.maximum(tot, 1e-9)).astype(np.float32)[lid]


def coincident_weights(page: Page, share: float = 0.2) -> np.ndarray:
    """E36 (LEDGER F41): loss weight 0 for a true shape whose (layer, style) group is mostly background on this page (< ``share``
    of the group's length true): a wall / grid / text stroke that only lies under a pipe line. The label rule makes it true by
    geometry, not by how it is drawn; the post-processing recovers it from the predicted pipe strokes. Every other shape 1."""

    g = np.unique(np.c_[page.layer, page.style], axis=0, return_inverse=True)[1].ravel()
    L = page.length.astype(np.float64)
    y = page.label > 0
    frac = (np.bincount(g, weights=L * y) / np.maximum(np.bincount(g, weights=L), 1e-9))[g]
    return np.where(y & (frac < share), 0.0, 1.0).astype(np.float32)


def viewer_isolated_weights(page: Page, max_duct_diag: float = 30.0, pad: float = 2.0) -> np.ndarray:
    """OpenStroke42 (F235): on Viewer Labels V1 train pages (``vw_*`` only; never GT or val), assign loss weight 0 (ignore)
    to strokes of 0-neighbor isolated fittings (``ocls != 1``) or small isolated boxes (bbox diagonal <= ``max_duct_diag`` pt).

    GT5 has 0.0% isolated fittings vs Viewer V1's 10% floating grille/diffuser boxes labeled ``Chân rẽ``; ignoring those noisy
    isolated boxes during training stops the model from scoring floating diffusers/grilles at p=0.96-1.00 on unseen OOD PDFs.
    """
    import shapely

    n = len(page)
    w = np.ones(n, dtype=np.float32)
    if n == 0 or not str(page.stem).startswith("vw_") or page.oinst is None or page.ocls is None:
        return w
    valid = (page.label > 0) & (page.oinst >= 0)
    if not valid.any():
        return w
    idx = np.flatnonzero(valid)
    ord_i = np.argsort(page.oinst[idx], kind="stable")
    idx_s = idx[ord_i]
    oi_s = page.oinst[idx_s]
    chg = np.flatnonzero(oi_s[1:] != oi_s[:-1]) + 1
    starts = np.r_[0, chg]
    ends = np.r_[chg, len(oi_s)]

    # Compute exact 2D bounding boxes from each stroke's 16 sampled points
    pts_abs = page.xy[idx_s, None, :].astype(np.float64) + page.pts[idx_s].astype(np.float64)
    s_x0 = pts_abs[..., 0].min(axis=1)
    s_y0 = pts_abs[..., 1].min(axis=1)
    s_x1 = pts_abs[..., 0].max(axis=1)
    s_y1 = pts_abs[..., 1].max(axis=1)

    boxes = []
    diags = []
    clss = []
    slices = []
    for a, b in zip(starts.tolist(), ends.tolist()):
        x0, y0 = float(s_x0[a:b].min()), float(s_y0[a:b].min())
        x1, y1 = float(s_x1[a:b].max()), float(s_y1[a:b].max())
        boxes.append(shapely.box(x0 - pad, y0 - pad, x1 + pad, y1 + pad))
        diags.append(float(np.hypot(x1 - x0, y1 - y0)))
        clss.append(int(page.ocls[idx_s[a]]))
        slices.append((a, b))
    if not boxes:
        return w
    arr = np.asarray(boxes, dtype=object)
    tree = shapely.STRtree(arr)
    ti, tj = tree.query(arr, predicate="intersects")
    has_nb = np.zeros(len(boxes), dtype=bool)
    if len(ti):
        has_nb[ti[ti != tj]] = True
    for k_i, (a, b) in enumerate(slices):
        if not has_nb[k_i] and (clss[k_i] != 1 or diags[k_i] <= max_duct_diag):
            w[idx_s[a:b]] = 0.0
    return w


def attach_graph(page: Page, npz: Path, n_shapes: int, cache: Path | None = None, *, facing: bool = False) -> Page:
    """Link graph of the page (built once by ``openstroke build-graph``, or now) as token positions."""

    from .graph import load_or_build, to_positions

    e, k = load_or_build(npz, cache, facing=facing)
    page.edges, page.ekind = to_positions(e, k, page.index, n_shapes)
    return page


def attach_objects(page: Page, raw, *, pieces: bool = True) -> Page:
    """E62: object class and centre of each shape, when the npz has them (``objects.object_dataset``). OpenStroke34: pieces too."""

    keys = raw.files if hasattr(raw, "files") else raw
    if "ocls" in keys:
        page.ocls = raw["ocls"][page.index].astype(np.int64)
        page.octr = (raw["octr"][page.index].astype(np.float64) if "octr" in keys
                     else np.full((len(page.index), 2), np.nan, dtype=np.float64))
        if "oinst" in keys:
            page.oinst = raw["oinst"][page.index].astype(np.int64)
    return attach_pieces(page, raw) if pieces else page


#: OpenStroke34: window entries holding one row per piece (kept whole by drop / reorder, like the tables)
PIECE_KEYS = ("ppos", "ppts", "plen", "pcur", "pfg", "pcls", "pinst", "pid", "pedges", "pekind")
PIECE_FEATS = 22   # piece_features: 16 normalised points, log length, curved, doubled-angle direction (2), offset to its shape (2)


def attach_pieces(page: Page, raw) -> Page:
    """OpenStroke34 (F187): the pieces of the npz (tools/make_e12.py; payload shape index -> page position), sorted by position; pieces of
    shapes the page drops are dropped, links remapped; ``pid`` = the piece's row in the npz (for saving outputs)."""
    if "pc_shape" not in raw:
        return page
    n = len(raw["shape_offsets"]) - 1
    inv = np.full(n, -1, np.int64)
    inv[page.index] = np.arange(len(page.index))
    pos = inv[raw["pc_shape"].astype(np.int64)]
    keep = np.flatnonzero(pos >= 0)
    order = keep[np.argsort(pos[keep], kind="stable")]
    new = np.full(len(pos), -1, np.int64)
    new[order] = np.arange(len(order))
    e = raw["pc_edges"].astype(np.int64).reshape(-1, 2)
    kind = raw["pc_ekind"].astype(np.int64) if "pc_ekind" in raw else np.ones(len(e), np.int64)   # 1 touch, 2 facing
    e = new[e]
    ok = (e >= 0).all(1)
    e, kind = e[ok], kind[ok]
    page.pieces = {"ppos": pos[order], "ppts": raw["pc_pts"][order].astype(np.float32), "plen": raw["pc_len"][order].astype(np.float32),
                   "pcur": raw["pc_cur"][order].astype(np.float32), "pfg": raw["pc_fg"][order].astype(np.float32),
                   "pcls": raw["pc_cls"][order].astype(np.int64), "pinst": raw["pc_inst"][order].astype(np.int64),
                   "pid": order.astype(np.int64), "pedges": e, "pekind": kind}
    return page


def window_pieces(page: Page, start: int, stop: int) -> dict:
    """Pieces whose shape lies in positions [start, stop), links between them as window-local piece indices."""
    pc = page.pieces
    if pc is None:
        return {}
    a, b = np.searchsorted(pc["ppos"], [start, stop], side="left")
    out = {k: pc[k][a:b].copy() for k in PIECE_KEYS if k not in ("pedges", "pekind")}
    e = pc["pedges"]
    m = (e >= a).all(1) & (e < b).all(1)
    out["pedges"] = e[m] - a
    out["pekind"] = pc["pekind"][m]
    return out


def piece_features(pts: np.ndarray, length: np.ndarray, curved: np.ndarray, shape_xy: np.ndarray) -> np.ndarray:
    """(P, PIECE_FEATS): the piece's points about its centre / its length (16), log length, curved, doubled-angle direction of its chord,
    offset of its centre from its shape's centre (/ 50 pt, clipped)."""
    c = pts.mean(1)
    s = np.maximum(length, 1e-3)[:, None, None]
    rel = ((pts - c[:, None]) / s).reshape(len(pts), pts.shape[1] * 2)
    d = pts[:, -1] - pts[:, 0]
    ang = np.arctan2(d[:, 1], d[:, 0])
    off = np.clip((c - shape_xy) / 50.0, -5, 5)
    return np.c_[rel, np.log1p(length)[:, None] / 3.0, curved[:, None], np.cos(2 * ang)[:, None], np.sin(2 * ang)[:, None], off].astype(np.float32)


def piece_batch(windows: list[dict], pos: np.ndarray) -> dict:
    """OpenStroke34: the pieces of a batch. ``pos``: (B, N) page position of each batch token (-1 padding). Pieces whose shape is not a
    token of the batch (dropped by augmentation, swapped away) are left out. -> pc_tok (B, P) token index (-1 pad), pc_feat, pc_c (centre,
    window frame / 100), pc_fg, pc_cls, pc_inst, pc_len, pc_id (npz row), pc_mask, pc_e (B, E, 2) local piece links, pc_emask."""
    B = len(windows)
    per = []
    for b, w in enumerate(windows):
        if "ppos" not in w or not len(w["ppos"]) or not len(w["pos"]):
            per.append(None)
            continue
        lo, hi = int(w["pos"].min()), int(w["pos"].max())
        tok = np.full(hi - lo + 1, -1, np.int64)
        row = pos[b]
        ok = (row >= lo) & (row <= hi)
        tok[row[ok] - lo] = np.flatnonzero(ok)
        xyi = np.full(hi - lo + 1, -1, np.int64)
        xyi[w["pos"] - lo] = np.arange(len(w["pos"]))
        pp = w["ppos"]
        inside = (pp >= lo) & (pp <= hi)
        t = np.where(inside, tok[np.clip(pp - lo, 0, hi - lo)], -1)
        r = np.where(inside, xyi[np.clip(pp - lo, 0, hi - lo)], -1)
        keep = (t >= 0) & (r >= 0)
        remap = np.full(len(pp), -1, np.int64)
        remap[keep] = np.arange(int(keep.sum()))
        e = remap[w["pedges"]] if len(w["pedges"]) else np.zeros((0, 2), np.int64)
        ek = w.get("pekind", np.ones(len(e), np.int64))
        if len(e):
            ok_e = (e >= 0).all(1)
            e, ek = e[ok_e], ek[ok_e]
        xy = w["xy"]
        centre = (xy.min(0) + xy.max(0)) / 2
        pts = w["ppts"][keep]
        per.append({"tok": t[keep], "feat": piece_features(pts, w["plen"][keep], w["pcur"][keep], xy[r[keep]]),
                    "c": ((pts.mean(1) - centre) / 100.0).astype(np.float32), "fg": w["pfg"][keep], "cls": w["pcls"][keep],
                    "inst": w["pinst"][keep], "len": w["plen"][keep], "id": w["pid"][keep], "e": e, "ek": ek})
    P = max([1] + [len(q["tok"]) for q in per if q is not None])
    E = max([1] + [len(q["e"]) for q in per if q is not None])
    out = {"pc_tok": np.full((B, P), -1, np.int64), "pc_feat": np.zeros((B, P, PIECE_FEATS), np.float32), "pc_c": np.zeros((B, P, 2), np.float32),
           "pc_fg": np.zeros((B, P), np.float32), "pc_cls": np.full((B, P), -1, np.int64), "pc_inst": np.full((B, P), -1, np.int64),
           "pc_len": np.zeros((B, P), np.float32), "pc_id": np.full((B, P), -1, np.int64), "pc_mask": np.zeros((B, P), bool),
           "pc_e": np.zeros((B, E, 2), np.int64), "pc_emask": np.zeros((B, E), bool), "pc_ek": np.zeros((B, E), np.int64)}
    for b, q in enumerate(per):
        if q is None:
            continue
        k = len(q["tok"])
        out["pc_tok"][b, :k] = q["tok"]
        out["pc_feat"][b, :k] = q["feat"]
        out["pc_c"][b, :k] = q["c"]
        out["pc_fg"][b, :k] = q["fg"]
        out["pc_cls"][b, :k] = q["cls"]
        out["pc_inst"][b, :k] = q["inst"]
        out["pc_len"][b, :k] = q["len"]
        out["pc_id"][b, :k] = q["id"]
        out["pc_mask"][b, :k] = True
        out["pc_e"][b, :len(q["e"])] = q["e"]
        out["pc_emask"][b, :len(q["e"])] = True
        out["pc_ek"][b, :len(q["e"])] = q["ek"]
    return out


def page_from_arrays(raw, stem: str, mode: str, graph: bool | str = False, no_layers: bool = False,
                     *, pieces: bool = True) -> Page:
    """In-memory equivalent of ``page_from_npz`` (avoids writing/reading temporary ``.npz`` & graph cache on inference)."""

    t = page_tokens_from_arrays(raw, mode, no_layers=no_layers)
    pts = shape_points(raw["segments"], raw["shape_offsets"])
    idx = np.flatnonzero(t.keep)
    if len(idx):
        idx = idx[np.argsort(zorder(quantize(t.xy[idx].astype(np.float64))), kind="stable")]
    page = Page(stem, mode, t.x[idx], t.xy[idx], pts[idx].astype(np.float16), t.layer[idx], t.label[idx],
                t.length[idx], idx, layer_table(np.array([""]) if no_layers else raw["layer_names"]))
    attach_objects(page, raw, pieces=pieces)
    if graph:
        from .graph import shape_graph_from_arrays, to_positions

        e, k = shape_graph_from_arrays(raw, facing=graph in ("continuation", "facing"))
        page.edges, page.ekind = to_positions(e, k, page.index, len(raw["shape_offsets"]) - 1)
    return page


def page_from_npz(npz: Path, stem: str, mode: str, graph: bool | str = False, graph_cache: Path | None = None,
                  no_layers: bool = False, *, pieces: bool = True) -> Page:
    """Shapes with geometry, features and arc samples, in z-order of their centres (train, val and inference)."""

    with np.load(npz) as raw:
        t = page_tokens_from_arrays(raw, mode, no_layers=no_layers)
        pts = shape_points(raw["segments"], raw["shape_offsets"])
        idx = np.flatnonzero(t.keep)
        if len(idx):
            idx = idx[np.argsort(zorder(quantize(t.xy[idx].astype(np.float64))), kind="stable")]
        page = Page(stem, mode, t.x[idx], t.xy[idx], pts[idx].astype(np.float16), t.layer[idx], t.label[idx],
                    t.length[idx], idx, layer_table(np.array([""]) if no_layers else raw["layer_names"]))
        attach_objects(page, raw, pieces=pieces)
        n_shapes = len(raw["shape_offsets"]) - 1
    return attach_graph(page, npz, n_shapes, graph_cache, facing=graph in ("continuation", "facing")) if graph else page


def load_page(data: Path, stem: str, mode: str, cache: Path, graph: bool | str = False, no_layers: bool = False,
              *, pieces: bool = True) -> Page:
    f = Path(cache) / f"{stem}.{DATA_VERSION}{'.nl' if no_layers else ''}.npz"
    npz = Path(data) / f"{stem}.npz"
    if f.exists():
        z = np.load(f)
        raw = np.load(npz)
        page = Page(stem, mode, z["x"], z["xy"], z["pts"], z["layer"], z["label"], z["length"], z["index"],
                    layer_table(np.array([""]) if no_layers else raw["layer_names"]))
        attach_objects(page, raw, pieces=pieces)
        return attach_graph(page, npz, len(raw["shape_offsets"]) - 1, cache, facing=graph in ("continuation", "facing")) if graph else page
    page = page_from_npz(npz, stem, mode, graph=graph, graph_cache=cache, no_layers=no_layers, pieces=pieces)
    Path(cache).mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + f".{os.getpid()}.tmp.npz")  # atomic: runs sharing a cache must not read a half file
    np.savez(tmp, x=page.x, xy=page.xy, pts=page.pts, layer=page.layer, label=page.label, length=page.length,
             index=page.index)
    os.replace(tmp, f)
    return page


@dataclass
class Augment:
    fliplr: float = 0.5
    rot90: bool = True
    scale: float = 0.2  # log-uniform zoom in [1/(1+s), 1+s]
    color: float = 0.5  # probability of permuting the RGB channels of stroke and fill (same permutation)
    gray: float = 0.1  # probability of turning colours to gray
    #: probability of re-drawing the window's styles: every distinct colour (stroke or fill) -> a random colour,
    #: every distinct width -> width x log-uniform [1/2, 2]. Groups stay groups, absolute style is lost (LEDGER F12)
    restyle: float = 0.0
    #: E30 ``order``: new colours keep the ORDER of the old ones' lightness (black stays darker than gray: gray strokes are 4x less
    #: often pipe, LEDGER F39) and one shared width factor keeps the order of widths; ``free``: the original fully random restyle;
    #: ``robust`` (F228): mixes order-preserving, independent per-group width resampling (including 0.00pt hairline), flat uniform width,
    #: random colours, and uniform pure-black monochrome to prevent OOD failures like 0875-MVAC-CT2-300.
    restyle_mode: str = "free"
    drop_tokens: float = 0.1  # drop up to this fraction of the window's shapes
    name_drop: float = 0.0  # probability of hiding every layer name of the window (H15: no reliance on names)
    #: OpenStroke32 (H137): probability of swapping a square region of the window for a region of another train page centred on one
    #: of its GT objects (region_swap); 0 = off
    paste: float = 0.0
    #: "region" (region_swap, E123) | "fg" (fg_swap: the window's background + another page's GT objects, E124) | "both" (one of them per window)
    paste_mode: str = "region"
    #: OpenStroke42 (F235): probability of dropping 60-95% of background shapes (simulates architecture-stripped sparse CAD drawings)
    sparse_bg: float = 0.0
    #: OpenStroke42 (F235): probability of injecting same-style isolated hard-negative boxes / X-diffusers / louver grilles (label=0)
    neg_boxes: float = 0.0


def window(page: Page, start: int, n: int) -> dict:
    sl = slice(start, min(start + n, len(page)))
    return {"x": page.x[sl].copy(), "xy": page.xy[sl].astype(np.float64), "pts": page.pts[sl].astype(np.float32),
            "layer": page.layer[sl], "label": page.label[sl], "length": page.length[sl],
            "ylayer": page.ylayer[sl], "pos": np.arange(sl.start, sl.stop),
            "lw": page.lw[sl] if page.lw is not None else np.ones(sl.stop - sl.start, np.float32),
            "mode": np.full(sl.stop - sl.start, MODES.index(page.mode) if page.mode in MODES else 0, np.int64),
            "ltab": page.ltab if page.ltab is not None else np.zeros((1, MAX_TOKENS), np.int64),
            "style": page.style[sl], **window_edges(page, sl.start, sl.stop),
            **({"pair_teacher": page.pair_teacher[sl]} if page.pair_teacher is not None else {}),
            **({"ocls": page.ocls[sl], "octr": page.octr[sl].copy()} if page.ocls is not None else {}),
            **({"oinst": page.oinst[sl]} if page.oinst is not None else {}),
            **window_pieces(page, sl.start, sl.stop)}


def window_edges(page: Page, start: int, stop: int) -> dict:
    """Links with both ends in [start, stop), as page positions (survive token drop / reorder; collate maps them)."""

    if page.edges is None:
        return {}
    e = page.edges
    m = (e[:, 0] >= start) & (e[:, 0] < stop) & (e[:, 1] >= start) & (e[:, 1] < stop)
    return {"edges": e[m], "ekind": page.ekind[m]}


#: window entries that are not one row per shape (kept whole by drop / reorder)
WINDOW_TABLES = ("ltab", "edges", "ekind", "ppos", "ppts", "plen", "pcur", "pfg", "pcls", "pinst", "pid", "pedges", "pekind")




#: OpenStroke32: side of the swapped square as a share of the window's longer extent (uniform in this range)
PASTE_SIDE = (0.2, 0.5)
#: window entries holding one row per shape that a region swap concatenates (the rest are tables, kept from the host)
ROW_KEYS = ("x", "xy", "pts", "layer", "label", "length", "ylayer", "pos", "lw", "mode", "style", "ocls", "octr", "oinst")


def region_swap(host: dict, donor: dict, rng: np.random.Generator, side: tuple = PASTE_SIDE, cap: int | None = None) -> dict:
    """OpenStroke32 (H137, F183): copy-paste of GT objects across drawings (Ghiasi et al., Simple Copy-Paste, CVPR 2021) for vector pages.
    A square of the host window (side ``side`` x its longer extent, random place) loses its shapes; the square of the donor window of the
    same size centred on a random TRUE donor shape is moved there (translation only), with its labels, object classes / centres, GT object
    ids (offset past the host's), links between kept donor shapes and per-shape style. Objects and background of one drawing then sit
    next to another drawing's, so the model cannot lean on what it memorised of a page (train pages F2 0.993 vs unseen 0.933, F182).
    The image branch draws each window from its own vectors, so image, vectors and labels change together (AGENTS 0.4). Shapes keep their
    page-level / neighbourhood features from their own page. ``cap``: at most this many shapes (host shapes dropped at random)."""

    hxy, dxy = host["xy"], donor["xy"]
    tru = np.flatnonzero(donor["label"] > 0)
    if len(hxy) < 2 or not len(tru):
        return host
    lo, hi = hxy.min(0), hxy.max(0)
    s = float(rng.uniform(*side)) * float(np.max(hi - lo))
    if s <= 0:
        return host
    hc = lo + rng.random(2) * np.maximum(hi - lo, 0)                     # centre of the host square
    dc = dxy[int(rng.choice(tru))]                                       # centre of the donor square: on a GT object's stroke
    keep_h = ~(np.abs(hxy - hc) <= s / 2).all(1)
    take_d = (np.abs(dxy - dc) <= s / 2).all(1)
    if not take_d.any():
        return host
    shift = hc - dc
    d_idx = np.flatnonzero(take_d)
    n_h = int(keep_h.sum())
    new_pos = (int(host["pos"].max()) + 1 if len(host["pos"]) else 0) + np.arange(len(d_idx))
    out = {}
    for k in ROW_KEYS:
        if k not in host:
            continue
        a = host[k][keep_h]
        if k not in donor:
            b = np.zeros((len(d_idx),) + host[k].shape[1:], host[k].dtype)
            if k == "oinst":
                b -= 1
        else:
            b = donor[k][d_idx].copy()
            if k in ("xy", "octr"):
                b = b + shift
            elif k == "pos":
                b = new_pos
            elif k == "layer":
                # F235: under no_layers=True both host and donor have layer=0 everywhere; never offset to 1 (was a 100% label leak into LayerPool!)
                if (len(host[k]) and int(host[k].max()) > 0) or (len(b) and int(b.max()) > 0):
                    b = b + (int(host[k].max()) + 1 if len(host[k]) else 0)
                else:
                    b = np.zeros_like(b)
            elif k == "style":
                b = b + (int(host[k].max()) + 1 if len(host[k]) else 0)
            elif k == "oinst":
                b = np.where(b >= 0, b + (int(host[k].max()) + 1 if len(host[k]) else 0), -1)
            elif k == "mode":
                b = np.full(len(d_idx), host[k][0] if len(host[k]) else 0, host[k].dtype)
        out[k] = np.concatenate([a, b.astype(a.dtype)])
    for k, v in host.items():
        if k not in out and k not in ("edges", "ekind"):
            out[k] = v
    if "edges" in host:
        e, ek = host["edges"], host["ekind"]
        alive = np.zeros(int(host["pos"].max()) + 1 if len(host["pos"]) else 1, bool)
        alive[host["pos"][keep_h]] = True
        hm = alive[e[:, 0]] & alive[e[:, 1]] if len(e) else np.zeros(0, bool)
        de = donor.get("edges", np.zeros((0, 2), np.int64))
        dk = donor.get("ekind", np.zeros(0, np.int64))
        if len(de):
            dmap = np.full(int(donor["pos"].max()) + 1, -1, np.int64)
            dmap[donor["pos"][d_idx]] = new_pos
            de = dmap[de]
            dm = (de >= 0).all(1)
            de, dk = de[dm], dk[dm]
        out["edges"] = np.concatenate([e[hm], de.reshape(-1, 2)]).astype(np.int64)
        out["ekind"] = np.concatenate([ek[hm], dk]).astype(np.int64)
    if cap is not None and len(out["x"]) > cap:                          # drop host shapes at random to stay inside the window
        n = len(out["x"])
        drop = rng.choice(n_h, size=min(n - cap, n_h), replace=False)
        if n - cap > n_h:                                                # still too many: donor shapes go too
            drop = np.r_[drop, n_h + rng.choice(n - n_h, size=n - cap - n_h, replace=False)]
        keep = np.ones(n, bool)
        keep[drop] = False
        out = {k: (v[keep] if k in ROW_KEYS else v) for k, v in out.items()}
    return out


FG_SHIFT = 0.25   # fg_swap: the donor's objects land at the host window's centre +- this share of its extent


def _turn_rows(w: dict, idx: np.ndarray, k: int, flip: bool, centre: np.ndarray) -> dict:
    """Rows ``idx`` of a window turned k x 90 degrees (and mirrored) about ``centre``, the way ``augment`` turns a whole window."""
    out = {}
    for key in ROW_KEYS:
        if key in w:
            out[key] = w[key][idx].copy()
    xy = out["xy"] - centre
    oc = out["octr"] - centre if "octr" in out else None
    pts, x = out["pts"], out["x"]
    if flip:
        xy[:, 0] = -xy[:, 0]
        pts[..., 0] = -pts[..., 0]
        x[:, DIR_SIN] = -x[:, DIR_SIN]
        if oc is not None:
            oc[:, 0] = -oc[:, 0]
    for _ in range(k):
        xy = np.stack([-xy[:, 1], xy[:, 0]], axis=1)
        if oc is not None:
            oc = np.stack([-oc[:, 1], oc[:, 0]], axis=1)
        pts[:] = np.stack([-pts[..., 1], pts[..., 0]], axis=-1)
        x[:, DIR_COS] = -x[:, DIR_COS]
        x[:, DIR_SIN] = -x[:, DIR_SIN]
    out["xy"] = xy + centre
    if oc is not None:
        out["octr"] = oc + centre
    return out


def fg_swap(host: dict, donor: dict, rng: np.random.Generator, cap: int | None = None) -> dict:
    """OpenStroke32 (H138, user 2026-11-18: GT objects of one page on the background of another): the host window keeps only its OTHER
    shapes; every TRUE shape of the donor window (its GT objects, links between them, object classes / centres / ids) is turned by a random
    k x 90 degrees, mirrored at random and moved so that its centre lands near the host window's centre (+- FG_SHIFT of the extent). The
    model then meets each drawing's pipes on a background it was never drawn on. Shapes keep their own page's neighbourhood features.
    ``cap``: at most this many shapes (host shapes dropped at random)."""

    tru = np.flatnonzero(donor["label"] > 0)
    oth = np.flatnonzero(host["label"] == 0)
    if not len(tru) or len(oth) < 2:
        return host
    dxy = donor["xy"][tru]
    dc = (dxy.min(0) + dxy.max(0)) / 2
    d = _turn_rows(donor, tru, int(rng.integers(0, 4)), bool(rng.random() < 0.5), dc)
    hxy = host["xy"][oth]
    lo, hi = hxy.min(0), hxy.max(0)
    target = (lo + hi) / 2 + (rng.random(2) * 2 - 1) * FG_SHIFT * (hi - lo)
    shift = target - dc
    d["xy"] = d["xy"] + shift
    if "octr" in d:
        d["octr"] = d["octr"] + shift
    keep_h = host["label"] == 0
    n_h = int(keep_h.sum())
    new_pos = (int(host["pos"].max()) + 1 if len(host["pos"]) else 0) + np.arange(len(tru))
    out = {}
    for k in ROW_KEYS:
        if k not in host:
            continue
        a = host[k][keep_h]
        if k not in d:
            b = np.zeros((len(tru),) + host[k].shape[1:], host[k].dtype)
            if k == "oinst":
                b -= 1
        else:
            b = d[k]
            if k == "pos":
                b = new_pos
            elif k == "layer":
                # F235: under no_layers=True both host and donor have layer=0 everywhere; never offset to 1 (was a 100% label leak into LayerPool!)
                if (len(host[k]) and int(host[k].max()) > 0) or (len(b) and int(b.max()) > 0):
                    b = b + (int(host[k].max()) + 1 if len(host[k]) else 0)
                else:
                    b = np.zeros_like(b)
            elif k == "style":
                b = b + (int(host[k].max()) + 1 if len(host[k]) else 0)
            elif k == "oinst":
                b = np.where(b >= 0, b + (int(host[k].max()) + 1 if len(host[k]) else 0), -1)
            elif k == "mode":
                b = np.full(len(tru), host[k][0] if len(host[k]) else 0, host[k].dtype)
        out[k] = np.concatenate([a, np.asarray(b).astype(a.dtype)])
    for k, v in host.items():
        if k not in out and k not in ("edges", "ekind"):
            out[k] = v
    if "edges" in host:
        e, ek = host["edges"], host["ekind"]
        alive = np.zeros(int(host["pos"].max()) + 1 if len(host["pos"]) else 1, bool)
        alive[host["pos"][keep_h]] = True
        hm = alive[e[:, 0]] & alive[e[:, 1]] if len(e) else np.zeros(0, bool)
        de = donor.get("edges", np.zeros((0, 2), np.int64))
        dk = donor.get("ekind", np.zeros(0, np.int64))
        if len(de):
            dmap = np.full(int(donor["pos"].max()) + 1, -1, np.int64)
            dmap[donor["pos"][tru]] = new_pos
            de = dmap[de]
            dm = (de >= 0).all(1)
            de, dk = de[dm], dk[dm]
        out["edges"] = np.concatenate([e[hm], de.reshape(-1, 2)]).astype(np.int64)
        out["ekind"] = np.concatenate([ek[hm], dk]).astype(np.int64)
    if cap is not None and len(out["x"]) > cap:
        n = len(out["x"])
        drop = rng.choice(n_h, size=min(n - cap, n_h), replace=False)
        if n - cap > n_h:                                                # still too many: donor shapes go too
            drop = np.r_[drop, n_h + rng.choice(n - n_h, size=n - cap - n_h, replace=False)]
        keep = np.ones(n, bool)
        keep[drop] = False
        out = {k: (v[keep] if k in ROW_KEYS else v) for k, v in out.items()}
    return out


def lightness(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=rgb.dtype)


def restyle_order(x: np.ndarray, rng: np.random.Generator) -> None:
    """In place, like ``restyle`` but relative style survives: distinct colours get new random tints whose lightness keeps the
    old lightness order (sorted fresh lightness values in [0.02, 0.9] assigned by rank), widths share one random factor."""

    cols = [(STROKE, x[:, HAS_STROKE] > 0.5), (FILL, x[:, HAS_FILL] > 0.5)]
    seen = np.concatenate([x[m][:, c] for c, m in cols])
    if len(seen):
        uniq, inv = np.unique(np.round(seen, 4), axis=0, return_inverse=True)
        order = np.argsort(lightness(uniq), kind="stable")
        new_l = np.sort(rng.uniform(0.02, 0.9, len(uniq)))
        tint = rng.normal(0, 0.08, (len(uniq), 3))
        tint -= lightness(tint)[:, None]  # zero-lightness tint: the lightness stays the drawn value
        new = np.empty((len(uniq), 3))
        new[order] = new_l[:, None] + tint
        new = np.clip(new, 0, 1).astype(x.dtype)[inv.reshape(-1)]
        k = 0
        for c, m in cols:
            n = int(m.sum())
            x[np.flatnonzero(m)[:, None], c] = new[k:k + n]
            k += n
    f = float(np.exp(rng.uniform(-np.log(2.0), np.log(2.0))))
    x[:, LOG_WIDTH] = np.log1p(np.expm1(x[:, LOG_WIDTH]) * f).astype(x.dtype)


def restyle(x: np.ndarray, rng: np.random.Generator) -> None:
    """In place: one random colour per distinct colour of the window (shared by stroke and fill, so a red stroke
    and a red fill stay equal) and one random factor per distinct width. Shapes that shared a style still share
    one; which colour or width an office uses for pipes no longer helps (style_probe: 33-51 % of the true length
    of folds 3 / 4 has a style never seen in train)."""
    cols = [(STROKE, x[:, HAS_STROKE] > 0.5), (FILL, x[:, HAS_FILL] > 0.5)]
    seen = np.concatenate([x[m][:, c] for c, m in cols])
    if len(seen):
        uniq, inv = np.unique(np.round(seen, 4), axis=0, return_inverse=True)
        new = rng.random((len(uniq), 3)).astype(x.dtype)[inv.reshape(-1)]
        k = 0
        for c, m in cols:
            n = int(m.sum())
            x[np.flatnonzero(m)[:, None], c] = new[k:k + n]
            k += n
    w = np.expm1(x[:, LOG_WIDTH])
    uniq, inv = np.unique(np.round(w, 4), return_inverse=True)
    f = np.exp(rng.uniform(-np.log(2.0), np.log(2.0), len(uniq)))
    x[:, LOG_WIDTH] = np.log1p(w * f[inv.reshape(-1)]).astype(x.dtype)


def restyle_robust(x: np.ndarray, rng: np.random.Generator) -> None:
    """In place (F228 / F241): OOD-robust style augmentation breaking hairline (0.00pt) and all-black shortcuts while
    preserving relative CAD line-weight hierarchy (F241: removes the 40% random per-group width inversion that taught
    E145/E148 to ignore 0.48pt duct vs 0.24pt furniture contrast on VNG-2 [12])."""

    if not len(x):
        return
    hs = x[:, HAS_STROKE] > 0.5
    # 1) Width branch first (groups by ORIGINAL (stroke_rgb, log_width) before colour changes)
    if hs.any():
        uw = float(rng.random())
        if uw < 0.75:
            # Order-preserving global width scale (75%: strong duct > furniture relative weight hierarchy for CADStem & graph_topo_v2!)
            f = float(np.exp(rng.uniform(-np.log(2.0), np.log(2.0))))
            x[hs, LOG_WIDTH] = np.log1p(np.expm1(x[hs, LOG_WIDTH]) * f).astype(x.dtype)
        elif uw < 0.95:
            # Hairline (0.00pt) invariance on random (colour, width) groups, while remaining groups keep relative order
            idx = np.flatnonzero(hs)
            rgb_q = np.rint(np.clip(x[idx][:, STROKE], 0.0, 1.0) * 1000.0).astype(np.uint64)
            lw_q = np.rint(np.clip(x[idx, LOG_WIDTH], 0.0, 10.0) * 1000.0).astype(np.uint64)
            packed = (rgb_q[:, 0] << np.uint64(34)) | (rgb_q[:, 1] << np.uint64(24)) | (rgb_q[:, 2] << np.uint64(14)) | lw_q
            uniq, inv = np.unique(packed, return_inverse=True)
            k = len(uniq)
            f = float(np.exp(rng.uniform(-np.log(2.0), np.log(2.0))))
            scaled = np.log1p(np.expm1(x[idx, LOG_WIDTH]) * f).astype(x.dtype)
            hairline_g = rng.random(k) < 0.25
            x[idx, LOG_WIDTH] = np.where(hairline_g[inv.reshape(-1)], np.float32(0.0), scaled).astype(x.dtype)
        else:
            # Flat uniform width across all stroked shapes (5%: prevents brittle reliance on width when all CAD weights equal)
            flat_w = 0.0 if rng.random() < 0.5 else float(np.exp(rng.uniform(np.log(0.12), np.log(0.96))))
            x[hs, LOG_WIDTH] = np.float32(np.log1p(flat_w * 10.0))

    # 2) Colour / lightness branch (independent of width)
    uc = float(rng.random())
    cols = [(STROKE, hs), (FILL, x[:, HAS_FILL] > 0.5)]
    if uc < 0.40:
        # Order-preserving lightness tints
        seen = np.concatenate([x[m][:, c] for c, m in cols])
        if len(seen):
            uniq, inv = np.unique(np.round(seen, 4), axis=0, return_inverse=True)
            order = np.argsort(lightness(uniq), kind="stable")
            new_l = np.sort(rng.uniform(0.02, 0.9, len(uniq)))
            tint = rng.normal(0, 0.08, (len(uniq), 3))
            tint -= lightness(tint)[:, None]
            new = np.empty((len(uniq), 3))
            new[order] = new_l[:, None] + tint
            new = np.clip(new, 0, 1).astype(x.dtype)[inv.reshape(-1)]
            k = 0
            for c, m in cols:
                n = int(m.sum())
                x[np.flatnonzero(m)[:, None], c] = new[k:k + n]
                k += n
    elif uc < 0.75:
        # Independent random colour per colour group (breaks darker == duct shortcut)
        seen = np.concatenate([x[m][:, c] for c, m in cols])
        if len(seen):
            uniq, inv = np.unique(np.round(seen, 4), axis=0, return_inverse=True)
            new = rng.random((len(uniq), 3)).astype(x.dtype)[inv.reshape(-1)]
            k = 0
            for c, m in cols:
                n = int(m.sum())
                x[np.flatnonzero(m)[:, None], c] = new[k:k + n]
                k += n
    else:
        # Uniform pure-black monochrome on all visible strokes (lightness < 0.92 keeps white wipeouts white)
        for c, m in cols:
            if m.any():
                idx = np.flatnonzero(m)
                vis = idx[lightness(x[idx][:, c]) < 0.92]
                if len(vis):
                    x[vis[:, None], c] = 0.0


def _synth_shape_row(
    segs: np.ndarray,
    rgb: np.ndarray,
    log_w: float,
    close_path: bool,
    n_pts: int,
    *,
    has_fill: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute exact (x_row (F,), xy (2,), pts (P, 2), length) for one synthetic straight-segment CAD shape matching features.py."""
    segs = np.asarray(segs, dtype=np.float64).reshape(-1, 4)
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    sl = np.hypot(dx, dy)
    L = float(sl.sum())
    pts_all = segs.reshape(-1, 2)
    x0, y0 = float(pts_all[:, 0].min()), float(pts_all[:, 1].min())
    x1, y1 = float(pts_all[:, 0].max()), float(pts_all[:, 1].max())
    bw, bh = x1 - x0, y1 - y0
    diag = float(np.hypot(bw, bh))
    xy = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)

    # Resample n_pts points at equal arc-length steps along segs, relative to xy
    if L > 1e-9:
        cum = np.concatenate([[0.0], np.cumsum(sl)])
        targets = np.linspace(0.0, L, n_pts)
        idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(segs) - 1)
        seg_l = np.maximum(sl[idx], 1e-9)
        frac = np.clip((targets - cum[idx]) / seg_l, 0.0, 1.0)
        abs_p = segs[idx, :2] + (segs[idx, 2:] - segs[idx, :2]) * frac[:, None]
        pts = (abs_p - xy[None, :]).astype(np.float32)
    else:
        pts = np.zeros((n_pts, 2), dtype=np.float32)

    ang = np.arctan2(dy, dx)
    c2 = float(np.sum(sl * np.cos(2.0 * ang)))
    s2 = float(np.sum(sl * np.sin(2.0 * ang)))
    mag = float(np.hypot(c2, s2))
    Ls = max(L, 1e-9)
    dir_cos2 = c2 / mag if mag > 1e-12 else 0.0
    dir_sin2 = s2 / mag if mag > 1e-12 else 0.0
    coh = mag / Ls
    nseg = len(segs)
    closed_ends = float(nseg > 0 and np.hypot(segs[0, 0] - segs[-1, 2], segs[0, 1] - segs[-1, 3]) < 1e-3)

    row = np.zeros(N_FEATURES, dtype=np.float32)
    row[_I["log_len"]] = np.log1p(L)
    row[_I["log_bbox_w"]] = np.log1p(bw)
    row[_I["log_bbox_h"]] = np.log1p(bh)
    row[_I["log_bbox_diag"]] = np.log1p(diag)
    row[_I["elong"]] = min(bw, bh) / max(max(bw, bh), 1e-6)
    row[_I["fill_ratio"]] = L / max(diag, 1e-6) / 10.0
    row[_I["dir_cos2"]] = dir_cos2
    row[_I["dir_sin2"]] = dir_sin2
    row[_I["dir_coherence"]] = coh
    row[_I["log_nseg"]] = np.log1p(nseg)
    row[_I["frac_l"]] = 1.0
    row[_I["closed_ends"]] = closed_ends
    if has_fill:
        row[_I["type_fs"]] = 1.0
        row[FILL] = rgb
        row[HAS_FILL] = 1.0
    else:
        row[_I["type_s"]] = 1.0
    row[_I["log_width"]] = float(log_w)
    row[STROKE] = rgb
    row[HAS_STROKE] = 1.0
    row[_I["stroke_opacity"]] = 1.0
    row[_I["fill_opacity"]] = 1.0
    row[_I["close_path"]] = float(close_path)
    return row, xy, pts, L


def inject_neg_boxes(w: dict, rng: np.random.Generator) -> dict:
    """OpenStroke44 v3 (F241): inject 6 realistic same-style hard-negative OOD CAD symbol/furniture families
    (``label=0``) into a training window with **zero feature-shortcut leaks** under ``feature_drop=layer_stats``:
      - 0: Double-walled concentric 4-line box (``op=[4,0,0,0]``, ``closePath=False``, 0.25-0.8 pt wall offset — 0875 EAD);
      - 1: Aligned row of 3-6 floating single-wall squares/rectangles (VNG-2 [12] chair/desk rows & floating grilles);
      - 2: Clustered office-desk / cubicle bay (2x2 or 2x3 touching rectangles 12-28 x 8-16 pt — VNG-2 [12] VD T1_dwg);
      - 3: Standalone FCU / equipment box (outer box + X-cross or internal louver slats / partitions);
      - 4: Adjacent sidewall grille + 5-segment zigzag airflow shaft (``op=[5,0,0,0]``) + solid filled triangle arrowhead
           (``type_fs``, ``has_fill=1`` — exact 0875 EAG/FAG/SAG symbol placed close to a real duct);
      - 5: Solid filled narrow bar / double-strip grille or framed louver fin cluster.
    Recomputes all 6 cKDTree neighbourhood ``context`` features and adds local ``g3`` graph edges."""
    from scipy.spatial import cKDTree

    tru = np.flatnonzero(w["label"] > 0)
    if len(tru) < 4 or not len(w["x"]):
        return w
    tru_xy = w["xy"][tru]
    tree = cKDTree(tru_xy)
    n_pts = w["pts"].shape[1]

    # Pre-sample styled true duct indices (prefer stroked shapes)
    tru_st = tru[w["x"][tru, HAS_STROKE] > 0.5]
    style_pool = tru_st if len(tru_st) else tru
    has_style_arr = "style" in w and len(w["style"]) == len(w["x"])

    rows_x, rows_xy, rows_pts, rows_len, rows_style = [], [], [], [], []
    local_edges: list[tuple[int, int]] = []
    n_added = 0

    def _rot(pts2: np.ndarray, theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.stack([pts2[:, 0] * c - pts2[:, 1] * s, pts2[:, 0] * s + pts2[:, 1] * c], axis=1)

    def _append_shape(
        segs_arr: np.ndarray,
        rgb: np.ndarray,
        log_w: float,
        close_p: bool,
        ref_row: np.ndarray,
        ref_st: int,
        *,
        has_fill: bool = False,
    ) -> None:
        nonlocal n_added
        rx, rxy, rpts, rl = _synth_shape_row(segs_arr, rgb, log_w, close_p, n_pts, has_fill=has_fill)
        # Copy donor style page-stats & jittered seqno_rank so there is ZERO feature shortcut under feature_drop=layer_stats
        rx[_I["style_shape_share"]] = ref_row[_I["style_shape_share"]]
        rx[_I["style_len_share"]] = ref_row[_I["style_len_share"]]
        rx[_I["seqno_rank"]] = float(np.clip(float(ref_row[_I["seqno_rank"]]) + rng.uniform(-0.05, 0.05), 0.0, 1.0))
        rows_x.append(rx); rows_xy.append(rxy); rows_pts.append(rpts); rows_len.append(rl); rows_style.append(ref_st)
        n_added += 1

    n_boxes = int(rng.integers(8, 20))
    for bi in range(n_boxes):
        ref = int(rng.choice(style_pool))
        ref_row = w["x"][ref]
        ref_st = int(w["style"][ref]) if has_style_arr else 0
        rgb = ref_row[STROKE].copy()
        log_w = float(ref_row[LOG_WIDTH])
        anchor = tru_xy[int(rng.integers(0, len(tru_xy)))]

        kind = bi % 6 if bi < 6 else int(rng.integers(0, 6))
        pad = 0.0
        sh_w = sh_l = ar_l = 0.0
        row_n = cols_n = rows_n = 1
        gap_r = 0.0

        if kind == 0:
            # Double-wall concentric box (0875 EAD)
            bw = float(rng.uniform(5.5, 22.0))
            bh = bw * float(rng.uniform(0.65, 1.0)) if rng.random() < 0.65 else bw
            pad = float(rng.uniform(0.25, max(0.80, 0.12 * min(bw, bh))))
            half_d = 0.5 * float(np.hypot(bw + 2.0 * pad, bh + 2.0 * pad))
            offset_r = half_d + float(rng.uniform(8.0, 55.0))
            min_clear = half_d + 4.0
        elif kind == 1:
            # Aligned row of 3-6 floating single-wall squares/rectangles (VNG-2 chair/desk rows; 50% at 0.5x thinner width)
            row_n = int(rng.integers(3, 7))
            bw = float(rng.uniform(5.0, 12.0))
            bh = bw * float(rng.uniform(0.70, 1.25))
            gap_r = float(rng.uniform(2.0, 7.0))
            tot_w = row_n * bw + (row_n - 1) * gap_r
            half_d = 0.5 * float(np.hypot(tot_w, bh))
            offset_r = half_d + float(rng.uniform(10.0, 60.0))
            min_clear = half_d + 4.0
            if log_w > 0.2 and rng.random() < 0.50:
                log_w = float(np.log1p(np.expm1(log_w) * float(rng.uniform(0.40, 0.60))))
        elif kind == 2:
            # Clustered office-desk / workstation row (2x2 up to 2x5 touching rectangles, span up to 140pt — VNG-2 VD T1_dwg;
            # 60% drawn at ~0.5x thinner width than ducts, 40% at same width so model rejects both thin & equal-weight desk grids)
            cols_n = int(rng.integers(2, 6))
            rows_n = 2
            bw = float(rng.uniform(12.0, 26.0))
            bh = float(rng.uniform(8.0, 15.0))
            half_d = 0.5 * float(np.hypot(cols_n * bw, rows_n * bh))
            offset_r = half_d + float(rng.uniform(12.0, 65.0))
            min_clear = half_d + 4.0
            if log_w > 0.2 and rng.random() < 0.60:
                log_w = float(np.log1p(np.expm1(log_w) * float(rng.uniform(0.40, 0.60))))
        elif kind == 3:
            # Standalone FCU / diffuser / louver box (X-cross or framed multi-slat)
            bw = float(rng.uniform(8.0, 32.0))
            bh = bw * float(rng.uniform(0.55, 1.0))
            half_d = 0.5 * float(np.hypot(bw, bh))
            offset_r = half_d + float(rng.uniform(8.0, 55.0))
            min_clear = half_d + 4.0
        elif kind == 4:
            # Unambiguous airflow grille symbol (thin louver lines + 5-segment zigzag shaft + filled triangle arrowhead,
            # F243 v4: NO plain closed rectangular side-collar box that mimicked real VIAS TOWER Chân rẽ!)
            sh_w = float(rng.uniform(0.40, 1.00))
            sh_l = float(rng.uniform(6.0, 16.0))
            ar_l = float(rng.uniform(3.5, 7.5))
            bw, bh = sh_l, sh_w * 2.4 + ar_l
            half_d = max(0.5 * sh_l, sh_w * 1.4 + ar_l)
            offset_r = half_d + float(rng.uniform(3.5, 16.0))
            min_clear = half_d + 2.5
        else:
            # Solid filled narrow bar / double-strip grille or adjacent louver fins
            bw = float(rng.uniform(8.0, 22.0))
            bh = float(rng.uniform(2.0, 10.0))
            half_d = 0.5 * float(np.hypot(bw, bh))
            offset_r = half_d + float(rng.uniform(4.0, 25.0))
            min_clear = half_d + 2.5

        placed = None
        for _ in range(5):
            ang_p = float(rng.uniform(0.0, 2.0 * np.pi))
            cand = anchor + offset_r * np.array([np.cos(ang_p), np.sin(ang_p)])
            d_nn, _ = tree.query(cand)
            if float(d_nn) >= min_clear:
                placed = cand
                break
        if placed is None:
            continue

        theta = float(rng.choice([0.0, 0.5 * np.pi])) if rng.random() < 0.85 else float(rng.uniform(0.0, np.pi))
        hx, hy = 0.5 * bw, 0.5 * bh
        corners = _rot(np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float64), theta) + placed
        wall_segs = [np.r_[corners[k], corners[(k + 1) % 4]] for k in range(4)]

        start_i = n_added
        if kind == 0:
            # Double-walled concentric 4-line box (op=[4,0,0,0], closePath=False)
            hx2, hy2 = hx + pad, hy + pad
            c2 = _rot(np.array([[-hx2, -hy2], [hx2, -hy2], [hx2, hy2], [-hx2, hy2]], dtype=np.float64), theta) + placed
            w2 = [np.r_[c2[k], c2[(k + 1) % 4]] for k in range(4)]
            _append_shape(np.stack(wall_segs, axis=0), rgb, log_w, False, ref_row, ref_st)
            _append_shape(np.stack(w2, axis=0), rgb, log_w, False, ref_row, ref_st)
            local_edges.append((start_i, start_i + 1))
        elif kind == 1:
            # Aligned row of 3-6 floating single-wall squares/rectangles
            step = bw + gap_r
            x_start = -0.5 * (row_n - 1) * step
            as_poly = bool(rng.random() < 0.5)
            for ri in range(row_n):
                cx_r = x_start + ri * step
                rc = _rot(np.array([[cx_r - hx, -hy], [cx_r + hx, -hy], [cx_r + hx, hy], [cx_r - hx, hy]], dtype=np.float64), theta) + placed
                rws = [np.r_[rc[k], rc[(k + 1) % 4]] for k in range(4)]
                s0 = n_added
                if as_poly:
                    _append_shape(np.stack(rws, axis=0), rgb, log_w, False, ref_row, ref_st)
                else:
                    for s_arr in rws:
                        _append_shape(s_arr[None, :], rgb, log_w, False, ref_row, ref_st)
                    for k in range(4):
                        local_edges.append((s0 + k, s0 + (k + 1) % 4))
        elif kind == 2:
            # Clustered office-desk / cubicle bay (2x2 or 2x3 touching rectangles)
            x_off = -0.5 * cols_n * bw
            y_off = -0.5 * rows_n * bh
            cell_starts = []
            for rr in range(rows_n):
                for cc in range(cols_n):
                    x0c, x1c = x_off + cc * bw, x_off + (cc + 1) * bw
                    y0c, y1c = y_off + rr * bh, y_off + (rr + 1) * bh
                    rc = _rot(np.array([[x0c, y0c], [x1c, y0c], [x1c, y1c], [x0c, y1c]], dtype=np.float64), theta) + placed
                    rws = np.stack([np.r_[rc[k], rc[(k + 1) % 4]] for k in range(4)], axis=0)
                    cell_starts.append(n_added)
                    _append_shape(rws, rgb, log_w, False, ref_row, ref_st)
            for rr in range(rows_n):
                for cc in range(cols_n):
                    idx_c = rr * cols_n + cc
                    if cc + 1 < cols_n:
                        local_edges.append((cell_starts[idx_c], cell_starts[idx_c + 1]))
                    if rr + 1 < rows_n:
                        local_edges.append((cell_starts[idx_c], cell_starts[idx_c + cols_n]))
        elif kind == 3:
            # Standalone FCU / diffuser / louver box (X-cross OR framed multi-slat louver)
            if rng.random() < 0.5:
                if rng.random() < 0.5:
                    _append_shape(np.stack(wall_segs, axis=0), rgb, log_w, False, ref_row, ref_st)
                    for d_seg in (np.r_[corners[0], corners[2]], np.r_[corners[1], corners[3]]):
                        _append_shape(d_seg[None, :], rgb, log_w, False, ref_row, ref_st)
                        local_edges.append((start_i, n_added - 1))
                else:
                    for s_arr in wall_segs:
                        _append_shape(s_arr[None, :], rgb, log_w, False, ref_row, ref_st)
                    for k in range(4):
                        local_edges.append((start_i + k, start_i + (k + 1) % 4))
                    for d_seg in (np.r_[corners[0], corners[2]], np.r_[corners[1], corners[3]]):
                        _append_shape(d_seg[None, :], rgb, log_w, False, ref_row, ref_st)
                        local_edges.append((start_i, n_added - 1))
                        local_edges.append((start_i + 2, n_added - 1))
            else:
                for s_arr in wall_segs:
                    _append_shape(s_arr[None, :], rgb, log_w, False, ref_row, ref_st)
                for k in range(4):
                    local_edges.append((start_i + k, start_i + (k + 1) % 4))
                n_slats = int(rng.integers(3, 7))
                for yv in np.linspace(-hy * 0.7, hy * 0.7, n_slats):
                    p01 = _rot(np.array([[-hx, yv], [hx, yv]], dtype=np.float64), theta) + placed
                    _append_shape(np.r_[p01[0], p01[1]][None, :], rgb, log_w, False, ref_row, ref_st)
                    local_edges.append((start_i + 1, n_added - 1))
                    local_edges.append((start_i + 3, n_added - 1))
        elif kind == 4:
            # F243 v4: Open double louver rail lines (NOT closed rectangular boxes that mimic VIAS TOWER Chân rẽ!)
            # + 5-segment zigzag shaft (op=[5,0,0,0]) + filled triangle arrowhead (type_fs, has_fill=1)
            h_l = 0.5 * sh_l
            for off_x in (-sh_w * 0.5, sh_w * 0.5):
                p01 = _rot(np.array([[off_x, -h_l], [off_x, h_l]], dtype=np.float64), theta) + placed
                _append_shape(np.r_[p01[0], p01[1]][None, :], rgb, log_w, False, ref_row, ref_st)
            sgn = 1.0 if rng.random() < 0.5 else -1.0
            bx = sgn * (sh_w * 1.2)
            z_len = ar_l * 0.72
            z_amp = max(0.7, ar_l * 0.14)
            # 5-segment zigzag polyline (exact CAD airflow arrow shaft on 0875 EAG!)
            z_pts = np.array([
                [bx, 0.0],
                [bx + sgn * 0.25 * z_len, 0.0],
                [bx + sgn * 0.42 * z_len, z_amp],
                [bx + sgn * 0.58 * z_len, -z_amp],
                [bx + sgn * 0.75 * z_len, 0.0],
                [bx + sgn * z_len, 0.0],
            ], dtype=np.float64)
            z_rot = _rot(z_pts, theta) + placed
            z_segs = np.stack([np.r_[z_rot[k], z_rot[k + 1]] for k in range(5)], axis=0)
            _append_shape(z_segs, rgb, log_w, False, ref_row, ref_st)
            # Solid filled triangle arrowhead (3-segment closed polygon, type_fs, has_fill=True)
            tip_x = bx + sgn * ar_l
            base_ax = bx + sgn * z_len
            aw = max(0.8, ar_l * 0.22)
            tri_pts = _rot(np.array([[tip_x, 0.0], [base_ax, aw], [base_ax, -aw]], dtype=np.float64), theta) + placed
            tri_segs = np.stack([np.r_[tri_pts[0], tri_pts[1]], np.r_[tri_pts[1], tri_pts[2]], np.r_[tri_pts[2], tri_pts[0]]], axis=0)
            _append_shape(tri_segs, rgb, log_w, True, ref_row, ref_st, has_fill=True)
            local_edges.append((start_i, start_i + 1))
            local_edges.append((start_i + 1, start_i + 2))
            local_edges.append((start_i + 2, start_i + 3))
        else:
            # Solid filled narrow bar / double-strip grille OR louver fin cluster
            if rng.random() < 0.5:
                bar_h = min(bh, float(rng.uniform(0.9, 2.5)))
                bhy = 0.5 * bar_h
                bc = _rot(np.array([[-hx, -bhy], [hx, -bhy], [hx, bhy], [-hx, bhy]], dtype=np.float64), theta) + placed
                bws = np.stack([np.r_[bc[k], bc[(k + 1) % 4]] for k in range(4)], axis=0)
                _append_shape(bws, rgb, log_w, True, ref_row, ref_st, has_fill=True)
            else:
                n_slats = int(rng.integers(4, 8))
                ys = np.linspace(-hy, hy, n_slats)
                slat_segs = []
                for yv in ys:
                    p01 = _rot(np.array([[-hx, yv], [hx, yv]], dtype=np.float64), theta) + placed
                    slat_segs.append(np.r_[p01[0], p01[1]])
                if rng.random() < 0.5:
                    slat_segs = wall_segs + slat_segs
                for s_arr in slat_segs:
                    _append_shape(s_arr[None, :], rgb, log_w, False, ref_row, ref_st)
                for k in range(start_i, n_added - 1):
                    local_edges.append((k, k + 1))

    if n_added == 0:
        return w

    new_x = np.stack(rows_x, axis=0).astype(w["x"].dtype)
    new_xy = np.stack(rows_xy, axis=0).astype(w["xy"].dtype)
    new_pts = np.stack(rows_pts, axis=0).astype(w["pts"].dtype)
    new_len = np.asarray(rows_len, dtype=w["length"].dtype)
    base_pos = int(w["pos"].max()) + 1 if len(w["pos"]) else 0
    new_pos = base_pos + np.arange(n_added, dtype=w["pos"].dtype)

    out = dict(w)
    out["xy"] = np.concatenate([w["xy"], new_xy], axis=0)
    out["pts"] = np.concatenate([w["pts"], new_pts], axis=0)
    out["length"] = np.concatenate([w["length"], new_len], axis=0)
    out["pos"] = np.concatenate([w["pos"], new_pos], axis=0)
    out["label"] = np.concatenate([w["label"], np.zeros(n_added, dtype=w["label"].dtype)], axis=0)
    out["layer"] = np.concatenate([w["layer"], np.zeros(n_added, dtype=w["layer"].dtype)], axis=0)
    if "ylayer" in w:
        out["ylayer"] = np.concatenate([w["ylayer"], np.zeros(n_added, dtype=w["ylayer"].dtype)], axis=0)
    if "lw" in w:
        out["lw"] = np.concatenate([w["lw"], np.ones(n_added, dtype=w["lw"].dtype)], axis=0)
    if "mode" in w:
        mval = w["mode"][0] if len(w["mode"]) else 0
        out["mode"] = np.concatenate([w["mode"], np.full(n_added, mval, dtype=w["mode"].dtype)], axis=0)
    if has_style_arr:
        new_st = np.asarray(rows_style, dtype=w["style"].dtype)
        out["style"] = np.concatenate([w["style"], new_st], axis=0)
    else:
        new_st = np.zeros(n_added, dtype=np.int64)
    if "ocls" in w:
        out["ocls"] = np.concatenate([w["ocls"], np.zeros(n_added, dtype=w["ocls"].dtype)], axis=0)
    if "octr" in w:
        out["octr"] = np.concatenate([w["octr"], np.full((n_added, 2), np.nan, dtype=w["octr"].dtype)], axis=0)
    if "oinst" in w:
        out["oinst"] = np.concatenate([w["oinst"], np.full(n_added, -1, dtype=w["oinst"].dtype)], axis=0)
    if "pair_teacher" in w:
        out["pair_teacher"] = np.concatenate([w["pair_teacher"], np.full(n_added, -1.0, dtype=w["pair_teacher"].dtype)], axis=0)

    # Recompute exact cKDTree neighbourhood context features (features.py:193-217) for injected shapes ONLY (<2ms)
    n_tot = len(out["xy"])
    if n_tot > 1:
        all_xy = out["xy"].astype(np.float64)
        all_tree = cKDTree(all_xy)
        nb10 = np.maximum(np.asarray(all_tree.query_ball_point(new_xy.astype(np.float64), r=10.0, return_length=True), dtype=np.float64) - 1.0, 0.0)
        k_q = min(17, n_tot)
        dq, jq = all_tree.query(new_xy.astype(np.float64), k=k_q)
        if k_q == 1:
            dq, jq = dq[:, None], jq[:, None]
        nn = dq[:, 1] if k_q > 1 else np.zeros(n_added, dtype=np.float64)
        jj = jq[:, 1:] if k_q > 1 else np.zeros((n_added, 0), dtype=np.int64)
        within = dq[:, 1:] <= 50.0 if k_q > 1 else np.zeros((n_added, 0), dtype=bool)
        cnt = within.sum(axis=1)
        safe = np.maximum(cnt, 1)
        all_layer = out["layer"]
        all_len = out["length"].astype(np.float64)
        same_l = ((all_layer[jj] == 0) & within).sum(axis=1) / safe if jj.shape[1] else np.zeros(n_added)
        if has_style_arr and jj.shape[1]:
            all_st = out["style"]
            same_s = ((all_st[jj] == new_st[:, None]) & within).sum(axis=1) / safe
        else:
            same_s = np.zeros(n_added)
        nb_len = (np.log1p(all_len[jj]) * within).sum(axis=1) / safe if jj.shape[1] else np.zeros(n_added)
        new_x[:, _I["nb10_log"]] = np.log1p(nb10).astype(new_x.dtype)
        new_x[:, _I["nb50_log"]] = np.log1p(cnt).astype(new_x.dtype)
        new_x[:, _I["nb50_same_layer"]] = same_l.astype(new_x.dtype)
        new_x[:, _I["nb50_same_style"]] = same_s.astype(new_x.dtype)
        new_x[:, _I["nb50_mean_log_len"]] = nb_len.astype(new_x.dtype)
        new_x[:, _I["nn_dist_log"]] = np.log1p(nn).astype(new_x.dtype)

    out["x"] = np.concatenate([w["x"], new_x], axis=0)
    if "edges" in w and local_edges:
        ne = np.asarray([[base_pos + a, base_pos + b] for a, b in local_edges], dtype=np.int64)
        nk = np.ones(len(ne), dtype=np.int64)
        out["edges"] = np.concatenate([w["edges"], ne], axis=0)
        out["ekind"] = np.concatenate([w["ekind"], nk], axis=0)
    return out


def augment(w: dict, aug: Augment, rng: np.random.Generator) -> dict:
    x, xy, pts = w["x"], w["xy"], w["pts"]
    oc = w.get("octr")  # E62: object centres move with the shapes
    pp = w.get("ppts")  # OpenStroke34: piece points move with the shapes
    if rng.random() < aug.fliplr:
        xy[:, 0] = -xy[:, 0]
        pts[..., 0] = -pts[..., 0]
        x[:, DIR_SIN] = -x[:, DIR_SIN]
        if oc is not None:
            oc[:, 0] = -oc[:, 0]
        if pp is not None:
            pp[..., 0] = -pp[..., 0]
    if aug.rot90:
        for _ in range(int(rng.integers(0, 4))):  # (x, y) -> (-y, x): doubled angle turns by 180 deg
            xy[:] = np.stack([-xy[:, 1], xy[:, 0]], axis=1)
            if oc is not None:
                oc[:] = np.stack([-oc[:, 1], oc[:, 0]], axis=1)
            if pp is not None:
                pp[:] = np.stack([-pp[..., 1], pp[..., 0]], axis=-1)
            pts[:] = np.stack([-pts[..., 1], pts[..., 0]], axis=-1)
            x[:, DIR_COS] = -x[:, DIR_COS]
            x[:, DIR_SIN] = -x[:, DIR_SIN]
    if aug.scale > 0:
        s = float(np.exp(rng.uniform(-np.log1p(aug.scale), np.log1p(aug.scale))))
        xy *= s
        pts *= s
        if oc is not None:
            oc *= s
        if pp is not None:
            pp *= s
            w["plen"] = w["plen"] * s
        for j in LOG_LENGTH_FEATURES:
            x[:, j] = np.log1p(np.expm1(x[:, j]) * s)
        x[:, LOG_WIDTH] = np.log1p(np.expm1(x[:, LOG_WIDTH]) * s)
    if rng.random() < aug.color:
        perm = rng.permutation(3)
        x[:, STROKE] = x[:, STROKE][:, perm]
        x[:, FILL] = x[:, FILL][:, perm]
    if rng.random() < aug.restyle:
        if aug.restyle_mode == "robust":
            restyle_robust(x, rng)
        elif aug.restyle_mode == "order":
            restyle_order(x, rng)
        else:
            restyle(x, rng)
    if rng.random() < aug.gray:
        for cols in (STROKE, FILL):
            g = x[:, cols].mean(axis=1, keepdims=True)
            x[:, cols] = g
    if aug.sparse_bg > 0 and rng.random() < aug.sparse_bg:
        bg = w["label"] == 0
        if int(bg.sum()) >= 64 and bool((w["label"] > 0).any()):
            drop_frac = float(rng.uniform(0.60, 0.95))
            keep = (~bg) | (rng.random(len(w["label"])) >= drop_frac)
            w = {k: (v if k in WINDOW_TABLES else v[keep]) for k, v in w.items()}
    if aug.neg_boxes > 0 and rng.random() < aug.neg_boxes:
        w = inject_neg_boxes(w, rng)
    if aug.drop_tokens > 0 and len(w["x"]) > 16:
        keep = rng.random(len(w["x"])) >= rng.uniform(0, aug.drop_tokens)
        w = {k: (v if k in WINDOW_TABLES else v[keep]) for k, v in w.items()}
    if aug.name_drop > 0 and "ltab" in w and rng.random() < aug.name_drop:
        w["ltab"] = np.zeros_like(w["ltab"])
    return w


def _levels(xy: np.ndarray, n_levels: int) -> list[dict]:
    """Sort a window by z-order and build the pooled levels and their orders (numpy, one window)."""

    q = quantize(xy)
    z = zorder(q)
    order = np.argsort(z, kind="stable")
    levels = []
    code = z[order]
    xy_l = xy[order]
    parent = None
    for li in range(n_levels):
        if li > 0:
            shift = 2 * int(np.log2(LEVEL_CELLS[li]))
            c = code >> shift
            new = np.concatenate([[True], c[1:] != c[:-1]])
            parent = np.cumsum(new) - 1  # level li-1 token -> level li index
            m = int(parent[-1]) + 1
            cnt = np.bincount(parent, minlength=m).astype(np.float64)
            xy_l = np.stack([np.bincount(parent, weights=xy_l[:, 0], minlength=m),
                             np.bincount(parent, weights=xy_l[:, 1], minlength=m)], axis=1) / cnt[:, None]
            code = c[new]
            code = code << shift  # keep codes in level-0 units for the next shift
        ql = quantize(xy_l, grid=float(LEVEL_CELLS[li]))
        if li == 0:
            # xy_l is already stably sorted by zorder(quantize(xy, grid=1.0)) == zorder(ql)
            p_list = [np.arange(len(xy_l), dtype=np.int64)] + [np.argsort(codes(ql, o), kind="stable") for o in ORDERS[1:]]
            perms = np.stack(p_list)
        else:
            perms = np.stack([np.argsort(codes(ql, o), kind="stable") for o in ORDERS])
        levels.append({"xy": xy_l, "perm": perms, "parent": parent})
    return levels, order


def collate(windows: list[dict], n_levels: int = len(LEVEL_CELLS)):
    """-> dict of torch tensors (see module docstring). Tokens of each window are put in z-order; ``pos``
    gives each token's position in its page (for prediction)."""

    import torch

    B = len(windows)
    per = []
    for w in windows:
        levels, order = _levels(w["xy"], n_levels)
        per.append((levels, {k: (v if k in WINDOW_TABLES else v[order]) for k, v in w.items()}))
    N = [max(len(p[0][li]["xy"]) for p in per) for li in range(n_levels)]
    P = windows[0]["pts"].shape[1]
    out = {
        "x": np.zeros((B, N[0], N_FEATURES), np.float32), "pts": np.zeros((B, N[0], P, 2), np.float32),
        "layer": np.zeros((B, N[0]), np.int64), "label": np.zeros((B, N[0]), np.float32),
        "length": np.zeros((B, N[0]), np.float32), "pos": np.full((B, N[0]), -1, np.int64),
        "mode": np.zeros(B, np.int64), "style": np.zeros((B, N[0]), np.int64),
        "ylayer": np.zeros((B, N[0]), np.float32), "lw": np.ones((B, N[0]), np.float32),
    }
    if any("pair_teacher" in w for w in windows):
        out["pair_teacher"] = np.full((B, N[0]), -1.0, np.float32)
    if any("ocls" in w for w in windows):  # E62: object class and offset to the object's centre (objects.OFFSET_SCALE pt units)
        out["ocls"] = np.zeros((B, N[0]), np.int64)
        out["ooff"] = np.zeros((B, N[0], 2), np.float32)
    if any("oinst" in w for w in windows):  # OpenStroke30: GT object of each token (-1 none / padding)
        out["oinst"] = np.full((B, N[0]), -1, np.int64)
    # links of each window as local token indices (positions -> index after drop / z-order), padded
    loc = []
    for _, w in per:
        if "edges" not in w or not len(w["pos"]):
            loc.append((np.zeros((0, 2), np.int64), np.zeros(0, np.int64)))
            continue
        lo, hi = int(w["pos"].min()), int(w["pos"].max())
        idx = np.full(hi - lo + 1, -1, np.int64)
        idx[w["pos"] - lo] = np.arange(len(w["pos"]))
        e = w["edges"]
        inside = (e >= lo).all(1) & (e <= hi).all(1)
        le = np.where(inside[:, None], idx[np.clip(e - lo, 0, hi - lo)], -1)
        ok = (le >= 0).all(1)
        loc.append((le[ok], w["ekind"][ok]))
    E = max(1, max(len(a) for a, _ in loc))
    out["edges"] = np.zeros((B, E, 2), np.int64)
    out["ekind"] = np.zeros((B, E), np.int64)
    out["emask"] = np.zeros((B, E), bool)
    for b, (a, k) in enumerate(loc):
        out["edges"][b, : len(a)] = a
        out["ekind"][b, : len(a)] = k
        out["emask"][b, : len(a)] = True
    # hashed name tokens of the layers present in each window, indexed like out["layer"] (window-local ids)
    uniq = [np.unique(w["layer"]) for _, w in per]
    out["ltok"] = np.zeros((B, max(max((len(u) for u in uniq), default=1), 1), MAX_TOKENS), np.int64)
    for li in range(n_levels):
        out[f"xy{li}"] = np.zeros((B, N[li], 2), np.float32)
        out[f"mask{li}"] = np.zeros((B, N[li]), bool)
        out[f"perm{li}"] = np.tile(np.arange(N[li], dtype=np.int64), (B, len(ORDERS), 1))
        if li > 0:
            out[f"parent{li}"] = np.zeros((B, N[li - 1]), np.int64)
    for b, (levels, w) in enumerate(per):
        n = len(w["x"])
        out["x"][b, :n] = w["x"]
        out["pts"][b, :n] = w["pts"]
        _, out["layer"][b, :n] = np.unique(w["layer"], return_inverse=True)
        if "style" in w:
            _, out["style"][b, :n] = np.unique(w["style"], return_inverse=True)
        if "ltab" in w and len(uniq[b]):
            tab = w["ltab"]
            ok = uniq[b] < len(tab)
            out["ltok"][b, : len(uniq[b])][ok] = tab[uniq[b][ok]]
        if "pair_teacher" in w:
            out["pair_teacher"][b, :n] = w["pair_teacher"]
        out["label"][b, :n] = w["label"]
        out["length"][b, :n] = w["length"]
        if "ylayer" in w:
            out["ylayer"][b, :n] = w["ylayer"]
        if "lw" in w:
            out["lw"][b, :n] = w["lw"]
        if "ocls" in out and "ocls" in w:
            out["ocls"][b, :n] = w["ocls"]
            out["ooff"][b, :n] = np.nan_to_num((w["octr"] - w["xy"]) / OFFSET_SCALE)
        if "oinst" in out and "oinst" in w:
            out["oinst"][b, :n] = w["oinst"]
        out["pos"][b, :n] = w["pos"]
        if n and "mode" in w:
            out["mode"][b] = int(w["mode"][0])
        centre = (levels[0]["xy"].min(axis=0) + levels[0]["xy"].max(axis=0)) / 2
        for li, lv in enumerate(levels):
            m = len(lv["xy"])
            out[f"xy{li}"][b, :m] = lv["xy"] - centre
            out[f"mask{li}"][b, :m] = True
            out[f"perm{li}"][b, :, :m] = lv["perm"]
            if li > 0:
                out[f"parent{li}"][b, :len(lv["parent"])] = lv["parent"]
    if any("ppos" in w for _, w in per):   # OpenStroke34
        out.update(piece_batch([w for _, w in per], out["pos"]))
    return {k: torch.from_numpy(v) for k, v in out.items()}


def page_cohort(mode: str, stem: str) -> str:
    """Cohort of a page for a dataset protocol (F213): its flow and where its labels come from, e.g. ``fire:gt``, ``plumbing:viewer``."""
    return f"{mode}:{'viewer' if str(stem).startswith('vw_') else 'gt'}"


def sampling_prob(sizes: np.ndarray, modes: list[str], plumbing_share: float | None = None, stems: list[str] | None = None,
                  viewer_share: float | None = None, cohort_share: dict | None = None) -> np.ndarray:
    """Page probability ~ sqrt(shapes) (small pages are still seen often). ``plumbing_share``: the PLUMBING pages
    together get exactly this share (195 pages of one project would otherwise take ~3/4 of the windows); the
    relative weights inside each part stay sqrt(shapes). ``cohort_share`` (a dataset protocol, F213): {cohort: share}; the
    cohorts present in ``pages`` share the windows in these proportions (renormalised), a page of an unlisted cohort is an error."""

    w = np.sqrt(np.asarray(sizes, np.float64))
    if cohort_share:
        if stems is None:
            raise ValueError("cohort_share needs the page stems")
        coh = np.array([page_cohort(m, s) for m, s in zip(modes, stems)])
        missing = sorted(set(coh) - set(cohort_share))
        if missing:
            raise ValueError(f"cohorts {missing} have no share in the dataset protocol {cohort_share}")
        present = {g: float(cohort_share[g]) for g in set(coh)}
        total = sum(present.values())
        w = w.copy()
        for g, share in present.items():
            k = coh == g
            w[k] *= (share / total) / w[k].sum()
        return w / w.sum()
    if plumbing_share is not None:
        pl = np.array([m == "plumbing" for m in modes])
        if pl.any() and (~pl).any():
            w = w.copy()
            w[pl] *= plumbing_share / w[pl].sum()
            w[~pl] *= (1 - plumbing_share) / w[~pl].sum()
    if viewer_share is not None and stems is not None:  # OpenStroke28: Viewer pages (vw_*) together get this share, GT pages the rest
        vw = np.array([str(s).startswith("vw_") for s in stems])
        if vw.any() and (~vw).any():
            w = w / w.sum()
            w[vw] *= viewer_share / w[vw].sum()
            w[~vw] *= (1 - viewer_share) / w[~vw].sum()
    return w / w.sum()


class WindowDataset:
    """Random training windows; item ``i`` is reproducible from (seed, epoch, i)."""

    def __init__(self, pages: list[Page], n_items: int, window_size: int, aug: Augment | None, seed: int = 0,
                 plumbing_share: float | None = None, raw: bool = False, viewer_share: float | None = None,
                 pdf_by_stem: dict[str, str] | None = None, cohort_share: dict | None = None):
        """``raw``: items are (window, augmentation seed); augmentation and batching happen on the GPU
        (``gpu_batch.build_batch``), the worker only cuts the window."""
        self.pages, self.n, self.window, self.aug, self.seed, self.raw = pages, n_items, window_size, aug, seed, raw
        self.epoch = 0
        sizes = np.array([len(p) for p in pages], np.float64)
        self.prob = sampling_prob(sizes, [p.mode for p in pages], plumbing_share, [p.stem for p in pages], viewer_share, cohort_share)
        if pdf_by_stem is not None:
            _, index = np.unique([pdf_by_stem[p.stem] for p in pages], return_inverse=True)
            pdf_mass = np.bincount(index, weights=self.prob)
            self.prob = self.prob / (pdf_mass[index] * len(pdf_mass))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng((self.seed, self.epoch, i))
        pi = int(rng.choice(len(self.pages), p=self.prob))
        p = self.pages[pi]
        start = int(rng.integers(0, max(len(p) - self.window, 0) + 1))
        w = window(p, start, self.window)
        if self.aug is not None and self.aug.paste > 0 and len(self.pages) > 1 and rng.random() < self.aug.paste:   # OpenStroke32
            rest = np.delete(self.prob, pi)
            qi = int(rng.choice(len(rest), p=rest / rest.sum()))
            q = self.pages[qi + (qi >= pi)]
            donor = window(q, int(rng.integers(0, max(len(q) - self.window, 0) + 1)), self.window)
            mode = self.aug.paste_mode if self.aug.paste_mode != "both" else ("region" if rng.random() < 0.5 else "fg")
            w = (fg_swap if mode == "fg" else region_swap)(w, donor, rng, cap=self.window)
        if self.raw:
            return w, int(rng.integers(0, 2**63 - 1))
        return augment(w, self.aug, rng) if self.aug is not None else w


def raw_items(items):
    """DataLoader collate for ``WindowDataset(raw=True)``: keep the windows as they are."""
    return items


def eval_windows(page: Page, window_size: int) -> list[int]:
    """Window starts covering the page with stride window/2 (overlaps are averaged)."""

    stride = max(window_size // 2, 1)
    starts = list(range(0, max(len(page) - window_size, 0) + 1, stride))
    if starts[-1] + window_size < len(page):
        starts.append(len(page) - window_size)
    return starts


def dihedral(w: dict, k: int) -> dict:
    """Test-time augmentation: one of the 8 flips / quarter turns, applied consistently to xy, pts, features."""

    w = {key: v.copy() for key, v in w.items()}
    aug = Augment(fliplr=1.0 if k >= 4 else 0.0, rot90=False, scale=0, color=0, gray=0, drop_tokens=0)
    rng = np.random.default_rng(0)
    w = augment(w, aug, rng)
    for _ in range(k % 4):
        w["xy"][:] = np.stack([-w["xy"][:, 1], w["xy"][:, 0]], axis=1)
        if "octr" in w:
            w["octr"][:] = np.stack([-w["octr"][:, 1], w["octr"][:, 0]], axis=1)
        if "ppts" in w:
            w["ppts"][:] = np.stack([-w["ppts"][..., 1], w["ppts"][..., 0]], axis=-1)
        w["pts"][:] = np.stack([-w["pts"][..., 1], w["pts"][..., 0]], axis=-1)
        w["x"][:, DIR_COS] = -w["x"][:, DIR_COS]
        w["x"][:, DIR_SIN] = -w["x"][:, DIR_SIN]
    return w


def describe(pages: list[Page]) -> str:
    n = sum(len(p) for p in pages)
    t = sum(int(p.label.sum()) for p in pages)
    return f"{len(pages)} pages, {n / 1e6:.2f}M shapes ({t / max(n, 1):.1%} true)"


def manifest_modes(data: Path) -> dict[str, str]:
    m = json.loads((Path(data) / "manifest.json").read_text(encoding="utf-8"))
    return {r["stem"]: r["mode"] for r in m["records"]}
