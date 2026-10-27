"""Batches built on the GPU (training speed, LEDGER F24): augmentation, z-order sort, U-Net levels, the four
serialization orders and window links, in torch on the training device. The DataLoader workers only cut windows.

Same algorithm as ``data.augment`` + ``data.collate`` (numpy); with no augmentation the batch is identical
(``tests/test_v2.py::test_gpu_batch_equals_numpy_collate``), and the random decisions are drawn from the same numpy
generator in the same order, so an augmented GPU batch equals the numpy one too (restyle aside: its new colours come
from the same draws but the group order is computed with torch).

Measured on this repo's CPU (one core, H1_KS, 4 windows of 65536 shapes): numpy augment 0.61 s + collate 0.43-0.62 s
per batch, i.e. 8 workers cap training near 4-8 steps / s, the 3.9 it/s seen on Colab (GPU waiting for the CPU).
"""

from __future__ import annotations

import numpy as np
import torch

from .data import (DIR_COS, DIR_SIN, FILL, HAS_FILL, HAS_STROKE, LEVEL_CELLS, LOG_LENGTH_FEATURES, LOG_WIDTH, STROKE,
                   WINDOW_TABLES, Augment)
from .features import N_FEATURES
from .layernames import MAX_TOKENS
from .serialize import BITS, ORDERS

_M = [(16, 0x0000FFFF0000FFFF), (8, 0x00FF00FF00FF00FF), (4, 0x0F0F0F0F0F0F0F0F), (2, 0x3333333333333333),
      (1, 0x5555555555555555)]


def quantize_t(xy: torch.Tensor, grid: float = 1.0) -> torch.Tensor:
    if not len(xy):
        return torch.zeros((0, 2), dtype=torch.int64, device=xy.device)
    q = torch.floor((xy - xy.min(dim=0, keepdim=True).values) / grid).to(torch.int64)
    return q.clamp(0, (1 << BITS) - 1)


def _spread(v: torch.Tensor) -> torch.Tensor:
    v = v.to(torch.int64)
    for s, m in _M:
        v = (v | (v << s)) & m
    return v


def zorder_t(q: torch.Tensor) -> torch.Tensor:
    return _spread(q[:, 0]) | (_spread(q[:, 1]) << 1)


def hilbert_t(q: torch.Tensor, bits: int) -> torch.Tensor:
    n = 1 << bits
    x = q[:, 0].to(torch.int64).clone()
    y = q[:, 1].to(torch.int64).clone()
    d = torch.zeros(len(q), dtype=torch.int64, device=q.device)
    s = n >> 1
    while s > 0:
        rx = ((x & s) > 0).to(torch.int64)
        ry = ((y & s) > 0).to(torch.int64)
        d += s * s * ((3 * rx) ^ ry)
        flip = (ry == 0) & (rx == 1)
        x = torch.where(flip, n - 1 - x, x)
        y = torch.where(flip, n - 1 - y, y)
        swap = ry == 0
        x, y = torch.where(swap, y, x), torch.where(swap, x, y)
        s >>= 1
    return d


def codes_t(q: torch.Tensor, order: str) -> torch.Tensor:
    if order.startswith("hilbert"):
        bits = max(1, int(q.max()).bit_length()) if len(q) else 1
        return hilbert_t(q if order == "hilbert" else q.flip(1), bits)
    return zorder_t(q if order == "z" else q.flip(1))


def _argsort(v: torch.Tensor) -> torch.Tensor:
    return torch.sort(v, stable=True).indices


def levels_t(xy: torch.Tensor, n_levels: int):
    """``data._levels`` in torch: (levels [{xy, perm (4, m), parent}], order)."""

    z = zorder_t(quantize_t(xy))
    order = _argsort(z)
    code = z[order]
    xy_l = xy[order]
    levels, parent = [], None
    for li in range(n_levels):
        if li > 0:
            shift = 2 * int(np.log2(LEVEL_CELLS[li]))
            c = code >> shift
            new = torch.ones(len(c), dtype=torch.bool, device=c.device)
            new[1:] = c[1:] != c[:-1]
            parent = torch.cumsum(new.to(torch.int64), 0) - 1
            m = int(parent[-1]) + 1
            cnt = torch.bincount(parent, minlength=m).to(xy_l.dtype)
            acc = torch.zeros((m, 2), dtype=xy_l.dtype, device=xy_l.device).index_add_(0, parent, xy_l)
            xy_l = acc / cnt[:, None]
            code = c[new] << shift
        ql = quantize_t(xy_l, grid=float(LEVEL_CELLS[li]))
        perms = torch.stack([_argsort(codes_t(ql, o)) for o in ORDERS])
        levels.append({"xy": xy_l, "perm": perms, "parent": parent})
    return levels, order


def _restyle_t(x: torch.Tensor, rng: np.random.Generator) -> None:
    cols = [(STROKE, x[:, HAS_STROKE] > 0.5), (FILL, x[:, HAS_FILL] > 0.5)]
    seen = torch.cat([x[m][:, c] for c, m in cols])
    if len(seen):
        key = torch.round(seen.double() * 1e4).to(torch.int64)
        uniq, inv = torch.unique(key, dim=0, return_inverse=True)
        new = torch.as_tensor(rng.random((len(uniq), 3)), dtype=x.dtype, device=x.device)[inv]
        k = 0
        for c, m in cols:
            n = int(m.sum())
            idx = torch.nonzero(m).reshape(-1)
            for j, col in enumerate(c):
                x[idx, col] = new[k:k + n, j]
            k += n
    w = torch.expm1(x[:, LOG_WIDTH].double())
    uniq, inv = torch.unique(torch.round(w * 1e4).to(torch.int64), return_inverse=True)
    f = torch.as_tensor(np.exp(rng.uniform(-np.log(2.0), np.log(2.0), len(uniq))), dtype=torch.float64,
                        device=x.device)
    x[:, LOG_WIDTH] = torch.log1p(w * f[inv]).to(x.dtype)


def augment_t(w: dict, aug: Augment, rng: np.random.Generator) -> dict:
    """``data.augment`` on tensors; random decisions from ``rng`` in the same order."""

    x, xy, pts = w["x"], w["xy"], w["pts"]
    if rng.random() < aug.fliplr:
        xy[:, 0] = -xy[:, 0]
        pts[..., 0] = -pts[..., 0]
        x[:, DIR_SIN] = -x[:, DIR_SIN]
    if aug.rot90:
        for _ in range(int(rng.integers(0, 4))):
            xy = torch.stack([-xy[:, 1], xy[:, 0]], dim=1)
            pts = torch.stack([-pts[..., 1], pts[..., 0]], dim=-1)
            x[:, DIR_COS] = -x[:, DIR_COS]
            x[:, DIR_SIN] = -x[:, DIR_SIN]
    if aug.scale > 0:
        s = float(np.exp(rng.uniform(-np.log1p(aug.scale), np.log1p(aug.scale))))
        xy = xy * s
        pts = pts * s
        for j in LOG_LENGTH_FEATURES:
            x[:, j] = torch.log1p(torch.expm1(x[:, j]) * s)
        x[:, LOG_WIDTH] = torch.log1p(torch.expm1(x[:, LOG_WIDTH]) * s)
    if rng.random() < aug.color:
        perm = torch.as_tensor(rng.permutation(3), device=x.device)
        st, fi = torch.as_tensor(STROKE, device=x.device), torch.as_tensor(FILL, device=x.device)
        x[:, st] = x[:, st[perm]]
        x[:, fi] = x[:, fi[perm]]
    if rng.random() < aug.restyle:
        _restyle_t(x, rng)
    if rng.random() < aug.gray:
        for cols in (STROKE, FILL):
            c = torch.as_tensor(cols, device=x.device)
            x[:, c] = x[:, c].mean(dim=1, keepdim=True)
    w = {**w, "x": x, "xy": xy, "pts": pts}
    if aug.drop_tokens > 0 and len(x) > 16:
        r = rng.random(len(x))
        keep = torch.as_tensor(r >= rng.uniform(0, aug.drop_tokens), device=x.device)
        w = {k: (v if k in WINDOW_TABLES else v[keep]) for k, v in w.items()}
    if aug.name_drop > 0 and "ltab" in w and rng.random() < aug.name_drop:
        w["ltab"] = torch.zeros_like(w["ltab"])
    return w


def to_tensors(w: dict, device) -> dict:
    out = {}
    for k, v in w.items():
        t = torch.as_tensor(np.asarray(v))
        if k == "xy":
            t = t.to(torch.float64)
        elif k == "pts":
            t = t.to(torch.float32)
        out[k] = t.to(device, non_blocking=True)
    return out


def build_batch(windows: list[dict], device, aug: Augment | None = None, seeds=None,
                n_levels: int = len(LEVEL_CELLS)) -> dict:
    """Windows (numpy, as cut by ``data.window``) -> the batch dict of ``data.collate``, built on ``device``."""

    B = len(windows)
    ws = [to_tensors(w, device) for w in windows]
    if aug is not None:
        ws = [augment_t(w, aug, np.random.default_rng(s)) for w, s in zip(ws, seeds)]
    per = []
    for w in ws:
        levels, order = levels_t(w["xy"], n_levels)
        per.append((levels, {k: (v if k in WINDOW_TABLES else v[order]) for k, v in w.items()}))
    N = [max(len(p[0][li]["xy"]) for p in per) for li in range(n_levels)]
    P = ws[0]["pts"].shape[1]
    z = lambda *s, dt=torch.float32: torch.zeros(s, dtype=dt, device=device)  # noqa: E731
    out = {"x": z(B, N[0], N_FEATURES), "pts": z(B, N[0], P, 2), "layer": z(B, N[0], dt=torch.int64),
           "label": z(B, N[0]), "length": z(B, N[0]), "pos": torch.full((B, N[0]), -1, dtype=torch.int64, device=device),
           "mode": z(B, dt=torch.int64), "style": z(B, N[0], dt=torch.int64), "ylayer": z(B, N[0]),
           "lw": torch.ones(B, N[0], device=device)}
    loc = []
    for _, w in per:
        if "edges" not in w or not len(w["pos"]):
            loc.append((z(0, 2, dt=torch.int64), z(0, dt=torch.int64)))
            continue
        lo, hi = int(w["pos"].min()), int(w["pos"].max())
        idx = torch.full((hi - lo + 1,), -1, dtype=torch.int64, device=device)
        idx[w["pos"] - lo] = torch.arange(len(w["pos"]), device=device)
        e = w["edges"]
        inside = ((e >= lo) & (e <= hi)).all(1)
        le = torch.where(inside[:, None], idx[(e - lo).clamp(0, hi - lo)], torch.full_like(e, -1))
        ok = (le >= 0).all(1)
        loc.append((le[ok], w["ekind"][ok]))
    E = max(1, max(len(a) for a, _ in loc))
    out["edges"] = z(B, E, 2, dt=torch.int64)
    out["ekind"] = z(B, E, dt=torch.int64)
    out["emask"] = z(B, E, dt=torch.bool)
    for b, (a, k) in enumerate(loc):
        out["edges"][b, : len(a)] = a
        out["ekind"][b, : len(a)] = k
        out["emask"][b, : len(a)] = True
    uniq = [torch.unique(w["layer"]) for _, w in per]
    out["ltok"] = z(B, max(max((len(u) for u in uniq), default=1), 1), MAX_TOKENS, dt=torch.int64)
    for li in range(n_levels):
        out[f"xy{li}"] = z(B, N[li], 2)
        out[f"mask{li}"] = z(B, N[li], dt=torch.bool)
        out[f"perm{li}"] = torch.arange(N[li], device=device).repeat(B, len(ORDERS), 1)
        if li > 0:
            out[f"parent{li}"] = z(B, N[li - 1], dt=torch.int64)
    for b, (levels, w) in enumerate(per):
        n = len(w["x"])
        out["x"][b, :n] = w["x"]
        out["pts"][b, :n] = w["pts"]
        out["layer"][b, :n] = torch.unique(w["layer"], return_inverse=True)[1]
        if "style" in w:
            out["style"][b, :n] = torch.unique(w["style"], return_inverse=True)[1]
        if "ltab" in w and len(uniq[b]):
            tab = w["ltab"]
            ok = uniq[b] < len(tab)
            out["ltok"][b, : len(uniq[b])][ok] = tab[uniq[b][ok]]
        out["label"][b, :n] = w["label"].to(torch.float32)
        out["length"][b, :n] = w["length"].to(torch.float32)
        if "ylayer" in w:
            out["ylayer"][b, :n] = w["ylayer"].to(torch.float32)
        if "lw" in w:
            out["lw"][b, :n] = w["lw"].to(torch.float32)
        out["pos"][b, :n] = w["pos"]
        if n and "mode" in w:
            out["mode"][b] = w["mode"][0]
        centre = (levels[0]["xy"].min(dim=0).values + levels[0]["xy"].max(dim=0).values) / 2
        for li, lv in enumerate(levels):
            m = len(lv["xy"])
            out[f"xy{li}"][b, :m] = (lv["xy"] - centre).to(torch.float32)
            out[f"mask{li}"][b, :m] = True
            out[f"perm{li}"][b, :, :m] = lv["perm"]
            if li > 0:
                out[f"parent{li}"][b, :len(lv["parent"])] = lv["parent"]
    return out
