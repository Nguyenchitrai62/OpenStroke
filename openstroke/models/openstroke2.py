"""OpenStroke v2: serialized U-Net transformer over shapes (docs/OPENSTROKE2.md).

Ideas taken from the state of the art (re-implemented here, nothing imported):

- **Point Transformer V3** (Wu et al., CVPR 2024): serialize points along space-filling curves (z-order,
  Hilbert and their transposes), attend inside patches of consecutive tokens, cycle / shuffle the orders
  across blocks, grid pooling by code shifts into a U-Net, conditional positional encoding before attention.
  Here: CPE = depthwise 1-D convolution along the block's serialization (cheap stand-in for PTv3's sparse conv).
- **VecFormer / CADTransformer** (vector CAD symbol spotting): a primitive is encoded from points sampled
  along it, not from its bbox: ``ShapeEncoder`` is a PointNet over 16 arc-length samples per shape.
- **SymPoint-V2** layer feature enhancement / OpenOBB H16: ``LayerPool`` (mean / max per PDF layer,
  zero-init) at the full-resolution stages.
- **RoPE** (2-D rotary position, as in RoFormer / ViT-RoPE): relative positions inside each attention patch.
- Stochastic depth (drop path), pre-norm blocks, length-weighted BCE + Lovasz hinge (PTv3 uses CE + Lovasz).

Input: the batch dict of ``openstroke.data.collate``. Output: one logit per level-0 token.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import LEVEL_CELLS, MODES
from ..layernames import N_BUCKETS
from ..model import LayerPool


def gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x (B, N, C), idx (B, M) -> (B, M, C)."""

    return torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))


def scatter_tokens(y: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Inverse of ``gather_tokens`` for a permutation ``idx``."""

    out = torch.empty_like(y)
    return out.scatter_(1, idx[..., None].expand(-1, -1, y.shape[-1]), y)


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        keep = (torch.rand(x.shape[0], *([1] * (x.dim() - 1)), device=x.device) >= self.p).to(x.dtype)
        return x * keep / (1 - self.p)


def rope_2d(t: torch.Tensor, pos: torch.Tensor, base: float = 100.0) -> torch.Tensor:
    """Rotary embedding on (..., P, d): first half of d rotates with x, second half with y."""

    d = t.shape[-1]
    q = d // 4
    if q == 0:
        return t
    freq = base ** (-torch.arange(q, device=t.device, dtype=torch.float32) / q)  # (q,)
    out = []
    for axis, part in enumerate((t[..., : 2 * q], t[..., 2 * q: 4 * q])):
        ang = pos[..., axis:axis + 1].float() * freq  # (..., P, q)
        cos, sin = ang.cos().to(t.dtype), ang.sin().to(t.dtype)
        a, b = part[..., :q], part[..., q:]
        out.append(torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1))
    if d > 4 * q:
        out.append(t[..., 4 * q:])
    return torch.cat(out, dim=-1)


class PatchAttention(nn.Module):
    """Multi-head attention inside patches of ``patch`` consecutive tokens of a serialization order."""

    def __init__(self, dim: int, heads: int, patch: int):
        super().__init__()
        self.heads, self.patch = heads, patch
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, pos, mask, perm):
        B, N, C = x.shape
        H = self.heads
        xs = gather_tokens(x, perm)
        ps = gather_tokens(pos, perm)
        ms = torch.gather(mask, 1, perm)
        P = min(self.patch, N)
        pad = (-N) % P
        if pad:
            xs = F.pad(xs, (0, 0, 0, pad))
            ps = F.pad(ps, (0, 0, 0, pad))
            ms = F.pad(ms, (0, pad), value=False)
        nP = xs.shape[1] // P
        qkv = self.qkv(xs).reshape(B, nP, P, 3, H, C // H).permute(3, 0, 1, 4, 2, 5)  # (3, B, nP, H, P, d)
        q, k, v = qkv[0], qkv[1], qkv[2]
        pp = ps.reshape(B, nP, 1, P, 2)
        q, k = rope_2d(q, pp), rope_2d(k, pp)
        km = ms.reshape(B, nP, 1, 1, P)
        empty = ~km.any(-1, keepdim=True)
        km = km | (empty & (torch.arange(P, device=x.device) == 0))  # a fully padded patch attends to itself
        o = F.scaled_dot_product_attention(q.reshape(B * nP, H, P, -1), k.reshape(B * nP, H, P, -1),
                                           v.reshape(B * nP, H, P, -1), attn_mask=km.reshape(B * nP, 1, 1, P))
        o = o.reshape(B, nP, H, P, C // H).permute(0, 1, 3, 2, 4).reshape(B, nP * P, C)[:, :N]
        return scatter_tokens(self.proj(o), perm)


class SerialCPE(nn.Module):
    """Conditional positional encoding: depthwise conv along the serialization, then a linear layer."""

    def __init__(self, dim: int, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask, perm):
        xs = gather_tokens(x * mask[..., None], perm)
        y = self.conv(xs.transpose(1, 2)).transpose(1, 2)
        return scatter_tokens(self.norm(self.lin(y)), perm)


class Block(nn.Module):
    def __init__(self, dim, heads, patch, drop_path, layer_pool, mlp_ratio=4):
        super().__init__()
        self.cpe = SerialCPE(dim)
        self.n1 = nn.LayerNorm(dim)
        self.attn = PatchAttention(dim, heads, patch)
        self.pool = LayerPool(dim) if layer_pool else None
        self.n2 = nn.LayerNorm(dim) if layer_pool else None
        self.n3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))
        self.dp = DropPath(drop_path)

    def forward(self, x, pos, mask, perm, layer=None):
        x = x + self.cpe(x, mask, perm)
        x = x + self.dp(self.attn(self.n1(x), pos, mask, perm))
        if self.pool is not None:
            x = x + self.dp(self.pool(self.n2(x), layer, mask))
        x = x + self.dp(self.mlp(self.n3(x)))
        return x * mask[..., None]


class ShapeEncoder(nn.Module):
    """PointNet over the shape's arc-length samples (scale-normalized) + an MLP over the hand features."""

    def __init__(self, n_features: int, n_points: int, dim: int):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 64))
        self.feat = nn.Sequential(nn.Linear(n_features, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.fuse = nn.Sequential(nn.Linear(128 + dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.register_buffer("t", torch.linspace(0, 1, n_points))

    def forward(self, x_norm, pts):
        B, N, P, _ = pts.shape
        scale = pts.flatten(2).abs().amax(-1, keepdim=True).clamp(min=1e-3)  # (B, N, 1)
        p = pts / scale[..., None]
        t = self.t.to(pts.dtype).view(1, 1, P, 1).expand(B, N, P, 1)
        h = self.point(torch.cat([p, t, torch.log1p(scale)[..., None].expand(B, N, P, 1)], -1))
        g = torch.cat([h.amax(2), h.mean(2)], -1)
        return self.fuse(torch.cat([g, self.feat(x_norm)], -1))


def _init_weights(m: nn.Module) -> None:
    """Xavier weights, zero biases. PyTorch's default Linear / Conv init gives random biases that, through the
    stacked MLPs of the ShapeEncoder, outweigh the per-shape signal (token std 0.013 vs a shared offset 0.086 at
    the encoder output); every LayerNorm'd residual branch then adds a near-constant vector and the logits come out
    equal for all shapes (Colab run 2026-10-20: BCE stuck at ln 2 for 3 epochs). LEDGER section 4."""

    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class OpenStroke2(nn.Module):
    def __init__(self, n_features: int, n_points: int = 16, dims=(64, 96, 128, 192, 256), enc_depths=(2, 2, 2, 4, 2),
                 dec_depths=(1, 1, 1, 1), head_dim: int = 16, patch: int = 1024, drop_path: float = 0.3,
                 layer_pool: bool = True, shuffle_orders: bool = True, mode_input: bool = False,
                 layer_names: bool = False):
        super().__init__()
        assert len(dims) == len(enc_depths) == len(dec_depths) + 1 <= len(LEVEL_CELLS)
        self.config = dict(n_features=n_features, n_points=n_points, dims=list(dims), enc_depths=list(enc_depths),
                           dec_depths=list(dec_depths), head_dim=head_dim, patch=patch, drop_path=drop_path,
                           layer_pool=layer_pool, shuffle_orders=shuffle_orders, mode_input=mode_input,
                           layer_names=layer_names)
        self.levels = len(dims)
        self.shuffle_orders = shuffle_orders
        self.register_buffer("feat_mean", torch.zeros(n_features))
        self.register_buffer("feat_std", torch.ones(n_features))
        self.embed = ShapeEncoder(n_features, n_points, dims[0])
        total = sum(enc_depths) + sum(dec_depths)
        rates = iter(torch.linspace(0, drop_path, total).tolist())
        heads = [max(d // head_dim, 1) for d in dims]
        self.enc = nn.ModuleList(
            nn.ModuleList(Block(dims[l], heads[l], patch, next(rates), layer_pool and l == 0) for _ in range(enc_depths[l]))
            for l in range(self.levels))
        self.down = nn.ModuleList(nn.Sequential(nn.Linear(dims[l], dims[l + 1]), nn.LayerNorm(dims[l + 1]))
                                  for l in range(self.levels - 1))
        self.up_proj = nn.ModuleList(nn.Linear(dims[l + 1], dims[l]) for l in range(self.levels - 1))
        self.skip_proj = nn.ModuleList(nn.Linear(dims[l], dims[l]) for l in range(self.levels - 1))
        self.dec = nn.ModuleList(
            nn.ModuleList(Block(dims[l], heads[l], patch, next(rates), layer_pool and l == 0) for _ in range(dec_depths[l]))
            for l in range(self.levels - 1))
        self.head = nn.Sequential(nn.LayerNorm(dims[0]), nn.Linear(dims[0], dims[0]), nn.GELU(), nn.Linear(dims[0], 1))
        #: FIRE / HVAC / PLUMBING flow of the page (data.MODES), added to every token; zero-init (LEDGER F14)
        self.mode_embed = nn.Embedding(len(MODES), dims[0]) if mode_input else None
        #: hashed tokens of the PDF layer name (H15, LEDGER F19): mean embedding per layer -> every member; zero-init out
        self.name_embed = nn.Embedding(N_BUCKETS, dims[0], padding_idx=0) if layer_names else None
        self.name_proj = nn.Linear(dims[0], dims[0]) if layer_names else None
        self.apply(_init_weights)
        if self.mode_embed is not None:
            nn.init.zeros_(self.mode_embed.weight)
        if self.name_proj is not None:
            nn.init.zeros_(self.name_proj.weight)
            nn.init.zeros_(self.name_proj.bias)
        for mod in self.modules():  # LayerPool keeps its zero-init output (starts as the model without it)
            if isinstance(mod, LayerPool):
                nn.init.zeros_(mod.out.weight)
                nn.init.zeros_(mod.out.bias)

    def set_normalization(self, mean, std) -> None:
        import numpy as np

        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.as_tensor(np.maximum(std, 1e-6), dtype=torch.float32))

    def _order(self, b, level, i):
        perm = b[f"perm{level}"]  # (B, 4, N)
        k = (i + (self._shift if self.training and self.shuffle_orders else 0)) % perm.shape[1]
        return perm[:, k]

    @staticmethod
    def _pool(x, parent, mask_child, n_parent):
        """mean + max of children into parents (B, M, C)."""

        B, N, C = x.shape
        key = (parent + torch.arange(B, device=x.device)[:, None] * n_parent).reshape(-1)
        m = mask_child.reshape(-1).to(x.dtype)
        flat = x.reshape(-1, C) * m[:, None]
        tot = torch.zeros(B * n_parent, C, device=x.device, dtype=x.dtype).index_add_(0, key, flat)
        cnt = torch.zeros(B * n_parent, device=x.device, dtype=x.dtype).index_add_(0, key, m)
        mean = tot / cnt.clamp(min=1)[:, None]
        mx = torch.full((B * n_parent, C), -1e4, device=x.device, dtype=x.dtype).scatter_reduce(
            0, key[:, None].expand(-1, C), torch.where(m[:, None] > 0, flat, torch.full_like(flat, -1e4)),
            reduce="amax", include_self=True)
        mx = torch.where(cnt[:, None] > 0, mx, torch.zeros_like(mx))
        return (mean + mx).reshape(B, n_parent, C)

    def forward(self, b: dict) -> torch.Tensor:
        self._shift = int(torch.randint(0, 4, (1,))) if self.training and self.shuffle_orders else 0
        x = ((b["x"] - self.feat_mean) / self.feat_std).clamp(-10, 10)
        h = self.embed(x, b["pts"])
        if self.name_embed is not None and "ltok" in b:
            tok = b["ltok"]  # (B, n_layers, T)
            m = (tok > 0).to(h.dtype)[..., None]
            v = (self.name_embed(tok).to(h.dtype) * m).sum(2) / m.sum(2).clamp(min=1)
            per = gather_tokens(self.name_proj(v), b["layer"].clamp(max=tok.shape[1] - 1))
            h = h + per * b["mask0"][..., None].to(h.dtype)
        if self.mode_embed is not None and "mode" in b:
            h = h + self.mode_embed(b["mode"])[:, None, :] * b["mask0"][..., None].to(h.dtype)
        skips = []
        bi = 0
        for l in range(self.levels):
            mask, pos = b[f"mask{l}"], b[f"xy{l}"] / LEVEL_CELLS[l] / 8.0
            if l > 0:
                h = self._pool(self.down[l - 1](h), b[f"parent{l}"], b[f"mask{l - 1}"], mask.shape[1])
                h = h * mask[..., None]
            for blk in self.enc[l]:
                h = blk(h, pos, mask, self._order(b, l, bi), b["layer"] if l == 0 else None)
                bi += 1
            skips.append(h)
        for l in range(self.levels - 2, -1, -1):
            mask, pos = b[f"mask{l}"], b[f"xy{l}"] / LEVEL_CELLS[l] / 8.0
            up = gather_tokens(self.up_proj[l](h), b[f"parent{l + 1}"])
            h = (up + self.skip_proj[l](skips[l])) * mask[..., None]
            for blk in self.dec[l]:
                h = blk(h, pos, mask, self._order(b, l, bi), b["layer"] if l == 0 else None)
                bi += 1
        return self.head(h)[..., 0]


def lovasz_hinge(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary Lovasz hinge (Berman et al., CVPR 2018) over a flat set of tokens."""

    if logits.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, order = torch.sort(errors, descending=True)
    gt = labels[order]
    gts = gt.sum()
    inter = gts - gt.cumsum(0)
    union = gts + (1 - gt).cumsum(0)
    jac = 1.0 - inter / union.clamp(min=1e-6)
    if len(jac) > 1:
        jac = torch.cat([jac[:1], jac[1:] - jac[:-1]])
    return torch.dot(F.relu(errors_sorted), jac)


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jac = 1.0 - inter / union.clamp(min=1e-6)
    if len(jac) > 1:
        jac = torch.cat([jac[:1], jac[1:] - jac[:-1]])
    return jac


def lovasz_sigmoid(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Lovasz extension of the Jaccard loss on probabilities (binary Lovasz-Softmax, Berman et al. 2018), the
    form PTv3 trains with. Errors |y - p| are bounded in [0, 1]; the hinge form on raw logits kept BCE at ~0.62
    on a CPU check where BCE alone reached 0.42 (LEDGER section 4)."""

    if logits.numel() == 0:
        return logits.sum() * 0.0
    errors = (labels - torch.sigmoid(logits)).abs()
    errors_sorted, order = torch.sort(errors, descending=True)
    return torch.dot(errors_sorted, _lovasz_grad(labels[order]))


def loss_v2(logits, b, pos_weight=2.0, alpha=0.5, lovasz=1.0):
    """-> (total, {"bce": ..., "lovasz": ...}). BCE weighted by (length + 1)**alpha and pos_weight on true."""

    mask = b["mask0"]
    y = b["label"]
    w = (b["length"].clamp(min=0) + 1.0) ** alpha * mask.float()
    w = w * torch.where(y > 0, pos_weight, 1.0)
    bce = (F.binary_cross_entropy_with_logits(logits.float(), y.float(), reduction="none") * w).sum() / w.sum().clamp(min=1e-6)
    parts = {"bce": bce}
    total = bce
    if lovasz > 0:
        lv = torch.stack([lovasz_sigmoid(logits[i][mask[i]].float(), y[i][mask[i]].float()) for i in range(len(y))]).mean()
        parts["lovasz"] = lv
        total = total + lovasz * lv
    return total, parts
