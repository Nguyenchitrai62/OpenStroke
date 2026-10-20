"""OpenStroke v1: per-shape binary classifier over a window of serialized tokens (docs/OPENSTROKE1.md).

Token = one payload shape (``features.py``). A page is serialized in Morton (z-order) of shape centres,
as in Point Transformer V3, and cut into windows of ``N`` consecutive tokens. Inside a window:

    embed (standardize + MLP)  ->  depth x [kNN attention (k neighbours, relative-position bias)
                                            + LayerPool (mean/max per layer, zero-init)
                                            + FFN]
                               ->  head (one logit per token: true vs other)

LayerPool is the OpenOBB H16 idea (SymPoint-V2): members of one PDF layer share evidence. Its output
projection starts at zero, so at init the model is exactly the same network without it
(``tests/test_model.py::test_layerpool_starts_as_identity``).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: Positions inside a window are divided by this (pt) before entering the network.
POS_SCALE = 100.0


def morton_order(xy: np.ndarray, grid: float = 1.0) -> np.ndarray:
    """Indices that sort points by z-order of their (grid-quantized) coordinates."""

    q = np.floor((xy - xy.min(axis=0)) / grid).astype(np.uint64)
    q = np.minimum(q, np.uint64((1 << 32) - 1))

    def spread(v):  # interleave zeros between the 32 low bits
        v = (v | (v << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
        v = (v | (v << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
        v = (v | (v << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        v = (v | (v << np.uint64(2))) & np.uint64(0x3333333333333333)
        v = (v | (v << np.uint64(1))) & np.uint64(0x5555555555555555)
        return v

    code = spread(q[:, 0]) | (spread(q[:, 1]) << np.uint64(1))
    return np.argsort(code, kind="stable")


def knn(xy: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """(B, N, 2), (B, N) -> (B, N, k) neighbour indices (self included; padding never chosen when avoidable)."""

    out = []
    k = min(k, xy.shape[1])
    for b in range(xy.shape[0]):
        d = torch.cdist(xy[b].float(), xy[b].float())
        d = d.masked_fill(~mask[b][None, :], float("inf"))
        out.append(d.topk(k, dim=-1, largest=False).indices)
    return torch.stack(out, dim=0)


def gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x (B, N, C), idx (B, N, k) -> (B, N, k, C)."""

    B, N, k = idx.shape
    flat = idx.reshape(B, N * k, 1).expand(-1, -1, x.shape[-1])
    return torch.gather(x, 1, flat).reshape(B, N, k, x.shape[-1])


class KnnAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.pos = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.bias = nn.Linear(dim, heads)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, xy, idx, mask):
        B, N, C = x.shape
        H = self.heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        kn, vn = gather(k, idx), gather(v, idx)  # (B, N, K, C)
        rel = gather(xy, idx) - xy[:, :, None, :]
        rel = torch.cat([rel, rel.norm(dim=-1, keepdim=True)], dim=-1)
        pe = self.pos(rel)  # (B, N, K, C)
        qh = q.reshape(B, N, 1, H, C // H)
        kh = (kn + pe).reshape(B, N, -1, H, C // H)
        att = (qh * kh).sum(-1) / math.sqrt(C // H) + self.bias(pe)  # (B, N, K, H)
        valid = gather(mask[..., None].float(), idx)[..., 0] > 0  # (B, N, K)
        att = att.masked_fill(~valid[..., None], float("-inf"))
        att = torch.softmax(att, dim=2).nan_to_num(0.0)
        out = (att[..., None] * (vn + pe).reshape(B, N, -1, H, C // H)).sum(2).reshape(B, N, C)
        return self.proj(out)


class LayerPool(nn.Module):
    """Mean and max of token states per (window, layer), sent back to every member. Zero-init output."""

    def __init__(self, dim: int):
        super().__init__()
        self.inp = nn.Linear(dim, dim)
        self.out = nn.Linear(2 * dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, layer, mask):
        B, N, C = x.shape
        h = F.gelu(self.inp(x)).float()
        key = (layer + torch.arange(B, device=x.device)[:, None] * N).reshape(-1)  # layer ids are < N per window
        m = mask.reshape(-1).float()
        flat = h.reshape(-1, C) * m[:, None]
        tot = torch.zeros(B * N, C, device=x.device).index_add_(0, key, flat)
        cnt = torch.zeros(B * N, device=x.device).index_add_(0, key, m)
        mean = tot / cnt.clamp(min=1)[:, None]
        neg = torch.full_like(flat, -1e4)
        mx = torch.full((B * N, C), -1e4, device=x.device).scatter_reduce(
            0, key[:, None].expand(-1, C), torch.where(m[:, None] > 0, flat, neg), reduce="amax", include_self=True)
        pooled = torch.cat([mean[key], mx[key]], dim=-1).reshape(B, N, 2 * C).to(x.dtype)
        return self.out(pooled)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, layer_pool: bool):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.att = KnnAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.pool = LayerPool(dim) if layer_pool else None
        self.n3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, xy, idx, layer, mask):
        x = x + self.att(self.n1(x), xy, idx, mask)
        if self.pool is not None:
            x = x + self.pool(self.n2(x), layer, mask)
        return x + self.ffn(self.n3(x))


class OpenStroke1(nn.Module):
    def __init__(self, n_features: int, dim: int = 128, depth: int = 4, heads: int = 4, k: int = 16,
                 layer_pool: bool = True):
        super().__init__()
        self.config = {"n_features": n_features, "dim": dim, "depth": depth, "heads": heads, "k": k,
                       "layer_pool": layer_pool}
        self.k = k
        self.register_buffer("feat_mean", torch.zeros(n_features))
        self.register_buffer("feat_std", torch.ones(n_features))
        self.embed = nn.Sequential(nn.Linear(n_features, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(Block(dim, heads, layer_pool) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.as_tensor(np.maximum(std, 1e-6), dtype=torch.float32))

    def forward(self, x, xy, layer, mask):
        """x (B, N, F), xy (B, N, 2) window-centred pt, layer (B, N) window-local ids < N, mask (B, N) bool."""

        x = ((x - self.feat_mean) / self.feat_std).clamp(-10, 10)
        pos = xy / POS_SCALE
        idx = knn(pos, mask, self.k)
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, pos, idx, layer, mask)
        return self.head(self.norm(h))[..., 0]


def loss_fn(logits, label, length, mask, pos_weight: float = 2.0, alpha: float = 0.5):
    """BCE weighted by length**alpha (the metric counts length) and by ``pos_weight`` on true tokens."""

    w = (length.clamp(min=0) + 1.0) ** alpha * mask.float()
    w = w * torch.where(label > 0, pos_weight, 1.0)
    bce = F.binary_cross_entropy_with_logits(logits.float(), label.float(), reduction="none")
    return (bce * w).sum() / w.sum().clamp(min=1e-6)
