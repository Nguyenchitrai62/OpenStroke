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

import numpy as np

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

    def __post_init__(self):
        if self.style is None:
            self.style = style_ids(self.x)

    def __len__(self) -> int:
        return len(self.x)


def attach_graph(page: Page, npz: Path, n_shapes: int, cache: Path | None = None) -> Page:
    """Link graph of the page (built once by ``openstroke build-graph``, or now) as token positions."""

    from .graph import load_or_build, to_positions

    e, k = load_or_build(npz, cache)
    page.edges, page.ekind = to_positions(e, k, page.index, n_shapes)
    return page


def page_from_npz(npz: Path, stem: str, mode: str, graph: bool = False, graph_cache: Path | None = None) -> Page:
    """Shapes with geometry, features and arc samples, in z-order of their centres (train, val and inference)."""

    t = page_tokens(npz, mode)
    raw = np.load(npz)
    pts = shape_points(raw["segments"], raw["shape_offsets"])
    idx = np.flatnonzero(t.keep)
    if len(idx):
        idx = idx[np.argsort(zorder(quantize(t.xy[idx].astype(np.float64))), kind="stable")]
    page = Page(stem, mode, t.x[idx], t.xy[idx], pts[idx].astype(np.float16), t.layer[idx], t.label[idx],
                t.length[idx], idx, layer_table(raw["layer_names"]))
    return attach_graph(page, npz, len(raw["shape_offsets"]) - 1, graph_cache) if graph else page


def load_page(data: Path, stem: str, mode: str, cache: Path, graph: bool = False) -> Page:
    f = Path(cache) / f"{stem}.{DATA_VERSION}.npz"
    npz = Path(data) / f"{stem}.npz"
    if f.exists():
        z = np.load(f)
        raw = np.load(npz)
        page = Page(stem, mode, z["x"], z["xy"], z["pts"], z["layer"], z["label"], z["length"], z["index"],
                    layer_table(raw["layer_names"]))
        return attach_graph(page, npz, len(raw["shape_offsets"]) - 1, cache) if graph else page
    page = page_from_npz(npz, stem, mode, graph=graph, graph_cache=cache)
    Path(cache).mkdir(parents=True, exist_ok=True)
    np.savez(f, x=page.x, xy=page.xy, pts=page.pts, layer=page.layer, label=page.label, length=page.length,
             index=page.index)
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
    drop_tokens: float = 0.1  # drop up to this fraction of the window's shapes
    name_drop: float = 0.0  # probability of hiding every layer name of the window (H15: no reliance on names)


def window(page: Page, start: int, n: int) -> dict:
    sl = slice(start, min(start + n, len(page)))
    return {"x": page.x[sl].copy(), "xy": page.xy[sl].astype(np.float64), "pts": page.pts[sl].astype(np.float32),
            "layer": page.layer[sl], "label": page.label[sl], "length": page.length[sl],
            "pos": np.arange(sl.start, sl.stop),
            "mode": np.full(sl.stop - sl.start, MODES.index(page.mode) if page.mode in MODES else 0, np.int64),
            "ltab": page.ltab if page.ltab is not None else np.zeros((1, MAX_TOKENS), np.int64),
            "style": page.style[sl], **window_edges(page, sl.start, sl.stop)}


def window_edges(page: Page, start: int, stop: int) -> dict:
    """Links with both ends in [start, stop), as page positions (survive token drop / reorder; collate maps them)."""

    if page.edges is None:
        return {}
    e = page.edges
    m = (e[:, 0] >= start) & (e[:, 0] < stop) & (e[:, 1] >= start) & (e[:, 1] < stop)
    return {"edges": e[m], "ekind": page.ekind[m]}


#: window entries that are not one row per shape (kept whole by drop / reorder)
WINDOW_TABLES = ("ltab", "edges", "ekind")




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
    if rng.random() < aug.fliplr:
        xy[:, 0] = -xy[:, 0]
        pts[..., 0] = -pts[..., 0]
        x[:, DIR_SIN] = -x[:, DIR_SIN]
    if aug.rot90:
        for _ in range(int(rng.integers(0, 4))):  # (x, y) -> (-y, x): doubled angle turns by 180 deg
            xy[:] = np.stack([-xy[:, 1], xy[:, 0]], axis=1)
            pts[:] = np.stack([-pts[..., 1], pts[..., 0]], axis=-1)
            x[:, DIR_COS] = -x[:, DIR_COS]
            x[:, DIR_SIN] = -x[:, DIR_SIN]
    if aug.scale > 0:
        s = float(np.exp(rng.uniform(-np.log1p(aug.scale), np.log1p(aug.scale))))
        xy *= s
        pts *= s
        for j in LOG_LENGTH_FEATURES:
            x[:, j] = np.log1p(np.expm1(x[:, j]) * s)
        x[:, LOG_WIDTH] = np.log1p(np.expm1(x[:, LOG_WIDTH]) * s)
    if rng.random() < aug.color:
        perm = rng.permutation(3)
        x[:, STROKE] = x[:, STROKE][:, perm]
        x[:, FILL] = x[:, FILL][:, perm]
    if rng.random() < aug.restyle:
        restyle(x, rng)
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
    }
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
    return {k: torch.from_numpy(v) for k, v in out.items()}


def sampling_prob(sizes: np.ndarray, modes: list[str], plumbing_share: float | None = None) -> np.ndarray:
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
    return w / w.sum()


class WindowDataset:
    """Random training windows; item ``i`` is reproducible from (seed, epoch, i)."""

    def __init__(self, pages: list[Page], n_items: int, window_size: int, aug: Augment | None, seed: int = 0,
                 plumbing_share: float | None = None, raw: bool = False):
        """``raw``: items are (window, augmentation seed); augmentation and batching happen on the GPU
        (``gpu_batch.build_batch``), the worker only cuts the window."""
        self.pages, self.n, self.window, self.aug, self.seed, self.raw = pages, n_items, window_size, aug, seed, raw
        self.epoch = 0
        sizes = np.array([len(p) for p in pages], np.float64)
        self.prob = sampling_prob(sizes, [p.mode for p in pages], plumbing_share)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng((self.seed, self.epoch, i))
        p = self.pages[int(rng.choice(len(self.pages), p=self.prob))]
        start = int(rng.integers(0, max(len(p) - self.window, 0) + 1))
        w = window(p, start, self.window)
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
