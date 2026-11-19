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
from .features import FEATURE_NAMES, FEATURE_VERSION, N_FEATURES, SHAPE_POINTS, page_tokens, shape_points
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

    if not len(x):
        return np.zeros(0, np.int64)
    key = np.concatenate([np.round(x[:, STROKE + FILL], 3), x[:, [HAS_STROKE, HAS_FILL]], np.round(x[:, [LOG_WIDTH]], 3),
                          x[:, TYPE_COLS]], axis=1)
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
    ekind: np.ndarray | None = None  # (E,) int64 kind bits of each link (1 endpoint, 2 gap, 4 tee)
    ylayer: np.ndarray | None = None  # (n,) float32: 1 if the shape's layer is a pipe layer of the page (layer head, E24)
    lw: np.ndarray | None = None  # (n,) float32 loss weight of each shape (E36 ``coincide``: 0 for coincident true shapes)
    ocls: np.ndarray | None = None  # (n,) int64 HVAC object class of the shape (objects.HVAC_CLASSES, 0 none; E62)
    octr: np.ndarray | None = None  # (n, 2) float64 centre of that object, pt, page frame (NaN none)
    oinst: np.ndarray | None = None  # (n,) int64 GT object index of the shape (-1 none; OpenStroke30 closure / edge losses)
    pieces: dict | None = None  # OpenStroke34: pieces of the candidate shapes (attach_pieces), sorted by their shape's position

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


def attach_graph(page: Page, npz: Path, n_shapes: int, cache: Path | None = None) -> Page:
    """Link graph of the page (built once by ``openstroke build-graph``, or now) as token positions."""

    from .graph import load_or_build, to_positions

    e, k = load_or_build(npz, cache)
    page.edges, page.ekind = to_positions(e, k, page.index, n_shapes)
    return page


def attach_objects(page: Page, raw) -> Page:
    """E62: object class and centre of each shape, when the npz has them (``objects.object_dataset``). OpenStroke34: pieces too."""

    if "ocls" in raw.files:
        page.ocls = raw["ocls"][page.index].astype(np.int64)
        page.octr = raw["octr"][page.index].astype(np.float64)
        if "oinst" in raw.files:
            page.oinst = raw["oinst"][page.index].astype(np.int64)
    return attach_pieces(page, raw)


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


def page_from_npz(npz: Path, stem: str, mode: str, graph: bool = False, graph_cache: Path | None = None,
                  no_layers: bool = False) -> Page:
    """Shapes with geometry, features and arc samples, in z-order of their centres (train, val and inference)."""

    t = page_tokens(npz, mode, no_layers=no_layers)
    raw = np.load(npz)
    pts = shape_points(raw["segments"], raw["shape_offsets"])
    idx = np.flatnonzero(t.keep)
    if len(idx):
        idx = idx[np.argsort(zorder(quantize(t.xy[idx].astype(np.float64))), kind="stable")]
    page = Page(stem, mode, t.x[idx], t.xy[idx], pts[idx].astype(np.float16), t.layer[idx], t.label[idx],
                t.length[idx], idx, layer_table(np.array([""]) if no_layers else raw["layer_names"]))
    attach_objects(page, raw)
    return attach_graph(page, npz, len(raw["shape_offsets"]) - 1, graph_cache) if graph else page


def load_page(data: Path, stem: str, mode: str, cache: Path, graph: bool = False, no_layers: bool = False) -> Page:
    f = Path(cache) / f"{stem}.{DATA_VERSION}{'.nl' if no_layers else ''}.npz"
    npz = Path(data) / f"{stem}.npz"
    if f.exists():
        z = np.load(f)
        raw = np.load(npz)
        page = Page(stem, mode, z["x"], z["xy"], z["pts"], z["layer"], z["label"], z["length"], z["index"],
                    layer_table(np.array([""]) if no_layers else raw["layer_names"]))
        attach_objects(page, raw)
        return attach_graph(page, npz, len(raw["shape_offsets"]) - 1, cache) if graph else page
    page = page_from_npz(npz, stem, mode, graph=graph, graph_cache=cache, no_layers=no_layers)
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
    #: often pipe, LEDGER F39) and one shared width factor keeps the order of widths; ``free``: the original fully random restyle
    restyle_mode: str = "free"
    drop_tokens: float = 0.1  # drop up to this fraction of the window's shapes
    name_drop: float = 0.0  # probability of hiding every layer name of the window (H15: no reliance on names)
    #: OpenStroke32 (H137): probability of swapping a square region of the window for a region of another train page centred on one
    #: of its GT objects (region_swap); 0 = off
    paste: float = 0.0
    #: "region" (region_swap, E123) | "fg" (fg_swap: the window's background + another page's GT objects, E124) | "both" (one of them per window)
    paste_mode: str = "region"


def window(page: Page, start: int, n: int) -> dict:
    sl = slice(start, min(start + n, len(page)))
    return {"x": page.x[sl].copy(), "xy": page.xy[sl].astype(np.float64), "pts": page.pts[sl].astype(np.float32),
            "layer": page.layer[sl], "label": page.label[sl], "length": page.length[sl],
            "ylayer": page.ylayer[sl], "pos": np.arange(sl.start, sl.stop),
            "lw": page.lw[sl] if page.lw is not None else np.ones(sl.stop - sl.start, np.float32),
            "mode": np.full(sl.stop - sl.start, MODES.index(page.mode) if page.mode in MODES else 0, np.int64),
            "ltab": page.ltab if page.ltab is not None else np.zeros((1, MAX_TOKENS), np.int64),
            "style": page.style[sl], **window_edges(page, sl.start, sl.stop),
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
            elif k in ("layer", "style"):
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
            elif k in ("layer", "style"):
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
        (restyle_order if aug.restyle_mode == "order" else restyle)(x, rng)
    if rng.random() < aug.gray:
        for cols in (STROKE, FILL):
            g = x[:, cols].mean(axis=1, keepdims=True)
            x[:, cols] = g
    if aug.drop_tokens > 0 and len(x) > 16:
        keep = rng.random(len(x)) >= rng.uniform(0, aug.drop_tokens)
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
        for li, lv in enumerate(levels):
            m = len(lv["xy"])
            centre = (levels[0]["xy"].min(axis=0) + levels[0]["xy"].max(axis=0)) / 2
            out[f"xy{li}"][b, :m] = lv["xy"] - centre
            out[f"mask{li}"][b, :m] = True
            out[f"perm{li}"][b, :, :m] = lv["perm"]
            if li > 0:
                out[f"parent{li}"][b, :len(lv["parent"])] = lv["parent"]
    if any("ppos" in w for _, w in per):   # OpenStroke34
        out.update(piece_batch([w for _, w in per], out["pos"]))
    return {k: torch.from_numpy(v) for k, v in out.items()}


def sampling_prob(sizes: np.ndarray, modes: list[str], plumbing_share: float | None = None, stems: list[str] | None = None,
                  viewer_share: float | None = None) -> np.ndarray:
    """Page probability ~ sqrt(shapes) (small pages are still seen often). ``plumbing_share``: the PLUMBING pages
    together get exactly this share (195 pages of one project would otherwise take ~3/4 of the windows); the
    relative weights inside each part stay sqrt(shapes)."""

    w = np.sqrt(np.asarray(sizes, np.float64))
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
                 plumbing_share: float | None = None, raw: bool = False, viewer_share: float | None = None):
        """``raw``: items are (window, augmentation seed); augmentation and batching happen on the GPU
        (``gpu_batch.build_batch``), the worker only cuts the window."""
        self.pages, self.n, self.window, self.aug, self.seed, self.raw = pages, n_items, window_size, aug, seed, raw
        self.epoch = 0
        sizes = np.array([len(p) for p in pages], np.float64)
        self.prob = sampling_prob(sizes, [p.mode for p in pages], plumbing_share, [p.stem for p in pages], viewer_share)

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
