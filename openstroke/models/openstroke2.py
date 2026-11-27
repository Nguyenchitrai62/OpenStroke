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

from ..data import DIR_COS, DIR_SIN, LEVEL_CELLS, MODES
from ..features import FEATURE_NAMES

DIR_COS_I, DIR_SIN_I, LOG_LEN_I = DIR_COS, DIR_SIN, FEATURE_NAMES.index("log_len")
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


_ROPE_FREQ_CACHE: dict[tuple, torch.Tensor] = {}


def rope_2d_qk(q_t: torch.Tensor, k_t: torch.Tensor, pos: torch.Tensor, base: float = 100.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply identical 2-D rotary embedding to ``q_t`` and ``k_t``, computing trig tables once with cached freqs."""
    d = q_t.shape[-1]
    q = d // 4
    if q == 0:
        return q_t, k_t
    key = (q_t.device, q, base)
    freq = _ROPE_FREQ_CACHE.get(key)
    if freq is None:
        freq = base ** (-torch.arange(q, device=q_t.device, dtype=torch.float32) / q)
        _ROPE_FREQ_CACHE[key] = freq
    ang = pos.float()[..., None] * freq  # (..., P, 2, q)
    cos = ang.cos().to(q_t.dtype)
    sin = ang.sin().to(q_t.dtype)

    def _apply(t: torch.Tensor) -> torch.Tensor:
        if not t.is_contiguous():
            t = t.contiguous()
        t4 = t[..., :4 * q].view(*t.shape[:-1], 2, 2, q)
        a, b = t4[..., 0, :], t4[..., 1, :]
        rot = torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-2).reshape(*t.shape[:-1], 4 * q)
        if d > 4 * q:
            return torch.cat([rot, t[..., 4 * q:]], dim=-1)
        return rot

    return _apply(q_t), _apply(k_t)


def _sdpa_patched(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ms: torch.Tensor | None,
                  all_valid: bool, B: int, nP: int, H: int, P: int) -> torch.Tensor:
    """Scaled dot-product attention over (B*nP, H, P, d) with FlashAttention-2 fast paths for unmasked patches."""
    if all_valid or ms is None:
        return F.scaled_dot_product_attention(q, k, v)
    km = ms.reshape(B * nP, 1, 1, P)
    # On CUDA with padded batches (e.g. B=4, drop_tokens=0.1), 90%+ of patches are 100% valid -> FlashAttention-2!
    if q.device.type == "cuda" and B * nP >= 4 and P >= 256:
        valid_cnt = ms.reshape(B * nP, P).sum(dim=-1)
        full = valid_cnt == P
        empty = valid_cnt == 0
        partial = ~(full | empty)
        has_full, has_part, has_emp = torch.stack([full.any(), partial.any(), empty.any()]).tolist()
        if has_full and not has_part and not has_emp:
            return F.scaled_dot_product_attention(q, k, v)
        o = torch.empty_like(v)
        if has_full:
            o[full] = F.scaled_dot_product_attention(q[full], k[full], v[full])
        if has_part:
            o[partial] = F.scaled_dot_product_attention(q[partial], k[partial], v[partial], attn_mask=km[partial])
        if has_emp:
            # Exact closed-form for an empty patch attending only to key 0: output is v[:, :, 0:1] everywhere
            o[empty] = v[empty, :, 0:1, :].expand_as(v[empty])
        return o
    empty = ~km.any(-1, keepdim=True)
    km = km | (empty & (torch.arange(P, device=q.device) == 0))  # a fully padded patch attends to itself
    return F.scaled_dot_product_attention(q, k, v, attn_mask=km)


class PatchAttention(nn.Module):
    """Multi-head attention inside patches of ``patch`` consecutive tokens of a serialization order."""

    def __init__(self, dim: int, heads: int, patch: int):
        super().__init__()
        self.heads, self.patch = heads, patch
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward_sorted(self, xs, ps, ms=None, all_valid: bool = False):
        B, N, C = xs.shape
        H = self.heads
        P = min(self.patch, N)
        pad = (-N) % P
        if pad:
            xs = F.pad(xs, (0, 0, 0, pad))
            ps = F.pad(ps, (0, 0, 0, pad))
            if ms is not None:
                ms = F.pad(ms, (0, pad), value=False)
            else:
                ms = torch.ones((B, N + pad), device=xs.device, dtype=torch.bool)
                ms[:, N:] = False
            all_valid = False
        nP = xs.shape[1] // P
        qkv = self.qkv(xs).reshape(B, nP, P, 3, H, C // H).permute(3, 0, 1, 4, 2, 5).contiguous()
        q, k = rope_2d_qk(qkv[0], qkv[1], ps.reshape(B, nP, 1, P, 2))
        v = qkv[2]
        o = _sdpa_patched(
            q.reshape(B * nP, H, P, -1),
            k.reshape(B * nP, H, P, -1),
            v.reshape(B * nP, H, P, -1),
            ms,
            all_valid,
            B, nP, H, P,
        )
        o = o.reshape(B, nP, H, P, C // H).permute(0, 1, 3, 2, 4).reshape(B, nP * P, C)[:, :N]
        return self.proj(o)

    def forward(self, x, pos, mask, perm):
        xs = gather_tokens(x, perm)
        ps = gather_tokens(pos, perm)
        ms = torch.gather(mask, 1, perm)
        all_valid = bool(ms.all()) if ms.device.type == "cpu" else False
        return scatter_tokens(self.forward_sorted(xs, ps, None if all_valid else ms, all_valid=all_valid), perm)


class SerialCPE(nn.Module):
    """Conditional positional encoding: depthwise conv along the serialization, then a linear layer."""

    def __init__(self, dim: int, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward_sorted(self, xs, ms=None):
        if ms is not None:
            xs = xs * ms[..., None]
        y = self.conv(xs.transpose(1, 2)).transpose(1, 2)
        return self.norm(self.lin(y))

    def forward(self, x, mask, perm):
        xs = gather_tokens(x * mask[..., None], perm)
        return scatter_tokens(self.forward_sorted(xs), perm)


class GraphMixer(nn.Module):
    """Token mixing along the shape link graph (OpenStroke4 level 0, in place of attention; LitePT: local operators where
    tokens are many, attention where they are few). Pre-norm input x (B, N, C); returns the update

        u_i = W_out · mean_j GELU(W_self x_i + W_nbr x_j + E(kind_ij) + W_geo geo_ij)

    i.e. a 2-layer message MLP on [x_i, x_j, link] whose first (linear) layer is split per node and whose second layer
    is moved after the mean (it commutes with it): per link only a gather, an add and a GELU, no matmul (3x faster
    than the per-link MLP on a 32k-shape window, same function class). ``g`` from ``link_context``."""

    def __init__(self, dim: int, n_kinds: int = 8):
        super().__init__()
        self.w_self = nn.Linear(dim, dim)
        self.w_nbr = nn.Linear(dim, dim, bias=False)
        self.kind = nn.Embedding(n_kinds, dim)
        self.geo = nn.Linear(4, dim, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, g):
        B, N, C = x.shape
        if g is None or not len(g["i"]):
            return torch.zeros_like(x)
        flat = x.reshape(B * N, C)
        ps, pn = self.w_self(flat), self.w_nbr(flat)
        e = (self.kind(g["kind"]) + self.geo(g["geo"])).to(ps.dtype)
        i, j = g["i"], g["j"]
        agg = torch.zeros_like(ps)
        agg.index_add_(0, i, F.gelu(ps[i] + pn[j] + e))
        agg.index_add_(0, j, F.gelu(ps[j] + pn[i] + e))
        return self.out(agg / g["deg"][:, None].to(agg.dtype)).reshape(B, N, C).to(x.dtype)


def link_context(b):
    """Flat link indices, degree and geometry of a batch (``edges`` / ``emask`` of ``collate``), or None."""

    if "edges" not in b or not b["emask"].any():
        return None
    B, N = b["mask0"].shape
    em = b["emask"]
    off = (torch.arange(B, device=em.device) * N)[:, None]
    i = (b["edges"][..., 0] + off)[em]
    j = (b["edges"][..., 1] + off)[em]
    xr = b["x"].reshape(B * N, -1).float()
    xy = b["xy0"].reshape(B * N, 2).float()
    ci, si, cj, sj = xr[i, DIR_COS_I], xr[i, DIR_SIN_I], xr[j, DIR_COS_I], xr[j, DIR_SIN_I]
    geo = torch.stack([ci * cj + si * sj, (ci * sj - si * cj).abs(), torch.log1p((xy[i] - xy[j]).norm(dim=-1)) / 5.0,
                       (xr[i, LOG_LEN_I] - xr[j, LOG_LEN_I]).abs() / 5.0], -1)
    deg = torch.zeros(B * N, device=em.device).index_add_(0, i, torch.ones_like(i, dtype=torch.float32))
    deg.index_add_(0, j, torch.ones_like(j, dtype=torch.float32))
    return {"i": i, "j": j, "kind": b["ekind"][em].clamp(0, 7), "geo": geo, "deg": deg.clamp(min=1)}


class Block(nn.Module):
    """Pre-norm block: CPE -> token mixer -> LayerPool (level 0) -> MLP. ``mixer``: "attn" (patch attention, v2),
    "conv" (no mixer beyond the CPE's serial conv: LitePT conv stage), "graph" (``GraphMixer``). New mixers go
    through a LayerScale (init 0.1) so no branch can swamp the residual stream (LEDGER F25)."""

    def __init__(self, dim, heads, patch, drop_path, layer_pool, mlp_ratio=4, mixer="attn"):
        super().__init__()
        self.mixer = mixer
        self.cpe = SerialCPE(dim)
        self.n1 = nn.LayerNorm(dim)
        self.attn = PatchAttention(dim, heads, patch) if mixer == "attn" else None
        self.graph = GraphMixer(dim) if mixer == "graph" else None
        self.gamma = nn.Parameter(torch.full((dim,), 0.1)) if mixer == "graph" else None
        self.pool = LayerPool(dim) if layer_pool else None
        self.n2 = nn.LayerNorm(dim) if layer_pool else None
        self.n3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))
        self.dp = DropPath(drop_path)

    def forward(self, x, pos, mask, perm, layer=None, g=None, n_groups: int | None = None,
                *, all_valid: bool = False, id_perm: bool = False):
        if self.attn is not None:
            # Fused permutation across SerialCPE -> LayerNorm -> PatchAttention (50% fewer gather/scatter passes, bit-exact)
            if id_perm:
                xs = x
                ps = pos
                ms = None if all_valid else mask
            else:
                xs = gather_tokens(x, perm)
                ps = gather_tokens(pos, perm)
                ms = None if all_valid else torch.gather(mask, 1, perm)
            xs = xs + self.cpe.forward_sorted(xs, ms)
            xs = xs + self.dp(self.attn.forward_sorted(self.n1(xs), ps, ms, all_valid=all_valid))
            x = xs if id_perm else scatter_tokens(xs, perm)
        else:
            x = x + self.cpe(x, mask, perm)
            if self.graph is not None:
                x = x + self.dp(self.gamma.to(x.dtype) * self.graph(self.n1(x), g))
        if self.pool is not None:
            x = x + self.dp(self.pool(self.n2(x), layer, mask, n_groups=n_groups))
        x = x + self.dp(self.mlp(self.n3(x)))
        return x if all_valid else x * mask[..., None]


class RelationBlock(nn.Module):
    """Message passing along the shape link graph (OpenStroke3, H17; LEDGER F22: 96-99 % of links join shapes of the
    same label, and with endpoint + dash-gap + tee links almost no pipe stroke is isolated). Each link (i, j) sends
    MLP([x_i, x_j, kind, geometry]) both ways; messages are averaged per shape and added through a zero-init projection,
    so the block starts as the identity. Geometry from the raw features: cos / |sin| of the angle between the two
    strokes, log centre distance, log length ratio. Attention in v2 follows space; this follows connection."""

    def __init__(self, dim: int, n_kinds: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.kind = nn.Embedding(n_kinds, dim)
        self.geo = nn.Linear(4, dim)
        self.msg = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.out = nn.Linear(dim, dim)

    def forward(self, h, b, x_raw):
        B, N, C = h.shape
        e, em = b["edges"], b["emask"]
        if not em.any():
            return h
        off = (torch.arange(B, device=h.device) * N)[:, None]
        i = (e[..., 0] + off)[em]
        j = (e[..., 1] + off)[em]
        flat = self.norm(h).reshape(B * N, C)
        xr = x_raw.reshape(B * N, -1).float()
        ci, si, cj, sj = xr[i, DIR_COS_I], xr[i, DIR_SIN_I], xr[j, DIR_COS_I], xr[j, DIR_SIN_I]
        xy = b["xy0"].reshape(B * N, 2).float()
        geo = torch.stack([ci * cj + si * sj, (ci * sj - si * cj).abs(),
                           torch.log1p((xy[i] - xy[j]).norm(dim=-1)) / 5.0,
                           (xr[i, LOG_LEN_I] - xr[j, LOG_LEN_I]).abs() / 5.0], -1)
        ef = (self.kind(b["ekind"][em].clamp(0, self.kind.num_embeddings - 1)) + self.geo(geo)).to(flat.dtype)
        mij = self.msg(torch.cat([flat[i], flat[j], ef], -1))  # to i from j
        mji = self.msg(torch.cat([flat[j], flat[i], ef], -1))  # to j from i
        agg = torch.zeros(B * N, C, device=h.device, dtype=mij.dtype)
        agg.index_add_(0, i, mij).index_add_(0, j, mji)
        deg = torch.zeros(B * N, device=h.device, dtype=mij.dtype)
        one = torch.ones_like(i, dtype=mij.dtype)
        deg.index_add_(0, i, one).index_add_(0, j, one)
        upd = self.out(agg / deg.clamp(min=1)[:, None]).reshape(B, N, C)
        return h + upd.to(h.dtype) * b["mask0"][..., None].to(h.dtype)


def edge_geometry(b, i, j, x_raw, style: bool = False):
    """Per link (flat token indices i, j): cos / |sin| of the angle between the two strokes, log centre distance, log length ratio;
    when ``style=True`` (OpenStroke46, H174), appends ``[clip(|Δlog_w|, 0, 2)/2, max|Δrgb|]`` (6D) so GraphAttention and EdgeHead
    directly see width/colour boundaries across touching CAD primitives."""

    B, N = b["mask0"].shape
    xr = x_raw.reshape(B * N, -1).float()
    ci, si, cj, sj = xr[i, DIR_COS_I], xr[i, DIR_SIN_I], xr[j, DIR_COS_I], xr[j, DIR_SIN_I]
    xy = b["xy0"].reshape(B * N, 2).float()
    parts = [
        ci * cj + si * sj,
        (ci * sj - si * cj).abs(),
        torch.log1p((xy[i] - xy[j]).norm(dim=-1)) / 5.0,
        (xr[i, LOG_LEN_I] - xr[j, LOG_LEN_I]).abs() / 5.0,
    ]
    if style:
        from ..data import LOG_WIDTH, STROKE

        rgb = xr[:, STROKE]
        parts.append((xr[i, LOG_WIDTH] - xr[j, LOG_WIDTH]).abs().clamp(max=2.0) / 2.0)
        parts.append((rgb[i] - rgb[j]).abs().amax(dim=-1).clamp(0.0, 1.0))
    return torch.stack(parts, -1)


def flat_links(b):
    """Valid links of the batch as flat token indices (i, j) into B * N, their kind bits, or None."""

    if "edges" not in b or not bool(b["emask"].any()):
        return None
    B, N = b["mask0"].shape
    e, em = b["edges"], b["emask"]
    off = (torch.arange(B, device=e.device) * N)[:, None]
    return (e[..., 0] + off)[em], (e[..., 1] + off)[em], b["ekind"][em]


class GraphAttention(nn.Module):
    """OpenStroke30 (H132): multi-head attention of each stroke over the strokes it is linked to (endpoint / dash gap / tee links of
    graph.py), with the link's kind and geometry as an attention bias and as part of the value (GAT-CADNet's edge-aware graph attention,
    GATv2-style scores). RelationBlock (OpenStroke3) averaged MLP messages; here a stroke weighs its neighbours, so one strong neighbour
    (the other side of an elbow) can carry it. Pre-norm, residual, zero-init output and FFN: starts as the identity."""

    def __init__(self, dim: int, heads: int = 4, n_kinds: int = 8, geo_dim: int = 4):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.norm = nn.LayerNorm(dim)
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.kind_b, self.geo_b = nn.Embedding(n_kinds, heads), nn.Linear(geo_dim, heads)
        self.kind_v, self.geo_v = nn.Embedding(n_kinds, dim), nn.Linear(geo_dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, h, b, x_raw, links, geo=None, bidir=None):
        if links is None:
            return h
        B, N, C = h.shape
        BN = B * N
        i0, j0, kind0 = links
        kind0_c = kind0.clamp(0, self.kind_b.num_embeddings - 1)
        if geo is None:
            geo = edge_geometry(b, i0, j0, x_raw)
        flat = self.norm(h).reshape(BN, C)
        # Fused QKV projection (1 GEMM instead of 3, identical weights/biases & autograd)
        w_qkv = torch.cat([self.q.weight, self.k.weight, self.v.weight], dim=0)
        b_qkv = torch.cat([self.q.bias, self.k.bias, self.v.bias], dim=0)
        q, k, v = F.linear(flat, w_qkv, b_qkv).split(C, dim=-1)
        q = q.view(BN, self.h, self.dh)
        k = k.view(BN, self.h, self.dh)
        inv_sq = 1.0 / math.sqrt(self.dh)
        # Edge bias & value computed ONCE per undirected edge E (2x fewer FLOPs & GPU activations than 2E bidir!)
        eb0 = (self.kind_b(kind0_c) + self.geo_b(geo)).float()
        ev0 = (self.kind_v(kind0_c) + self.geo_v(geo)).to(v.dtype)
        E = len(i0)
        cpu_chunk = (not torch.is_grad_enabled()) and (h.device.type == "cpu") and (E > 32768)
        if cpu_chunk:
            s_ij = torch.empty((E, self.h), device=h.device, dtype=torch.float32)
            s_ji = torch.empty((E, self.h), device=h.device, dtype=torch.float32)
            for s0 in range(0, E, 32768):
                s1 = min(s0 + 32768, E)
                ia, ja = i0[s0:s1], j0[s0:s1]
                ebc = eb0[s0:s1]
                s_ij[s0:s1] = (q[ia] * k[ja]).sum(-1).float() * inv_sq + ebc
                s_ji[s0:s1] = (q[ja] * k[ia]).sum(-1).float() * inv_sq + ebc
        else:
            s_ij = (q[i0] * k[j0]).sum(-1).float() * inv_sq + eb0
            s_ji = (q[j0] * k[i0]).sum(-1).float() * inv_sq + eb0
        i_both = torch.cat([i0, j0])
        s_both = torch.cat([s_ij, s_ji])
        smax = torch.full((BN, self.h), -1e4, device=h.device, dtype=s_both.dtype).scatter_reduce(
            0, i_both[:, None].expand(-1, self.h), s_both, reduce="amax", include_self=True)
        a_ij = torch.exp(s_ij - smax[i0])
        a_ji = torch.exp(s_ji - smax[j0])
        den = torch.zeros(BN, self.h, device=h.device, dtype=a_ij.dtype)
        den.index_add_(0, i0, a_ij)
        den.index_add_(0, j0, a_ji)
        den = den.clamp(min=1e-9)
        a_ij = (a_ij / den[i0]).to(v.dtype)
        a_ji = (a_ji / den[j0]).to(v.dtype)
        agg = torch.zeros(BN, self.h, self.dh, device=h.device, dtype=v.dtype)
        if cpu_chunk:
            for s0 in range(0, E, 32768):
                s1 = min(s0 + 32768, E)
                ia, ja = i0[s0:s1], j0[s0:s1]
                evc = ev0[s0:s1]
                agg.index_add_(0, ia, (v[ja] + evc).view(-1, self.h, self.dh) * a_ij[s0:s1, :, None])
                agg.index_add_(0, ja, (v[ia] + evc).view(-1, self.h, self.dh) * a_ji[s0:s1, :, None])
        else:
            agg.index_add_(0, i0, (v[j0] + ev0).view(-1, self.h, self.dh) * a_ij[..., None])
            agg.index_add_(0, j0, (v[i0] + ev0).view(-1, self.h, self.dh) * a_ji[..., None])
        m = b["mask0"][..., None].to(h.dtype)
        h = h + self.out(agg.reshape(BN, C)).reshape(B, N, C).to(h.dtype) * m
        return h + self.ffn(self.norm2(h)).to(h.dtype) * m


class EdgeHead(nn.Module):
    """OpenStroke30 (H132): per link, the logit that its two strokes draw the same object (GAT-CADNet / CADTransformer predict primitive
    pairs of one instance). Trained from the GT instance of every stroke (oinst); makes the relation features carry object membership."""

    def __init__(self, dim: int, n_kinds: int = 8, geo_dim: int = 4):
        super().__init__()
        self.kind = nn.Embedding(n_kinds, dim)
        self.geo = nn.Linear(geo_dim, dim)
        self.mlp = nn.Sequential(nn.LayerNorm(3 * dim), nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, h, b, x_raw, links, geo=None):
        B, N, C = h.shape
        i, j, kind = links
        flat = h.reshape(B * N, C)
        if geo is None:
            geo = edge_geometry(b, i, j, x_raw)
        kmax = self.kind.num_embeddings - 1
        E = len(i)
        if not torch.is_grad_enabled() and h.device.type == "cpu" and E > 16384:
            out = None
            for s0 in range(0, E, 16384):
                s1 = min(s0 + 16384, E)
                hi, hj = flat[i[s0:s1]], flat[j[s0:s1]]
                ec = self.kind(kind[s0:s1].clamp(0, kmax)) + self.geo(geo[s0:s1])
                zc = self.mlp(torch.cat([hi + hj, (hi - hj).abs(), ec.to(flat.dtype)], -1))[..., 0]
                if out is None:
                    out = torch.empty(E, device=h.device, dtype=zc.dtype)
                out[s0:s1] = zc
            return out if out is not None else flat.new_zeros(0)
        e = self.kind(kind.clamp(0, kmax)) + self.geo(geo)
        return self.mlp(torch.cat([flat[i] + flat[j], (flat[i] - flat[j]).abs(), e.to(flat.dtype)], -1))[..., 0]


class PieceHead(nn.Module):
    """OpenStroke34 (F186 / F187, H141): the object head on PIECES of strokes (openstroke.pieces: a shape's run cut at corners and meeting
    points), since one PDF item often draws several objects (64 % of the true length). Each piece starts from its shape's token, its own
    geometry and its shape's stroke confidence; ``rounds`` of message passing along pieces that touch (ends within pieces.SNAP: end to end,
    T joints) carry confidence from piece to piece (user 2026-11-19: strokes touching a confident object are likely part of it): each message
    is gated by the sender's current confidence and each round re-reads the confidence it updated. Outputs per piece: pipe logit, class
    logits (N_CLASSES), and per link the same-object logit. Works on thick (HVAC duct outlines) and thin (FIRE lines) drawings alike."""

    def __init__(self, dim: int, n_feat: int, n_cls: int, pdim: int = 64, rounds: int = 3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat, pdim), nn.GELU(), nn.Linear(pdim, pdim))
        self.region = None
        self.from_shape = nn.Linear(dim, pdim)
        self.conf_in = nn.Linear(1, pdim)
        self.geo = nn.Linear(4, pdim)
        self.msg = nn.ModuleList(nn.Sequential(nn.Linear(3 * pdim + 1, pdim), nn.GELU(), nn.Linear(pdim, pdim)) for _ in range(rounds))
        self.gate = nn.ModuleList(nn.Linear(3 * pdim + 1, 1) for _ in range(rounds))
        self.norm = nn.ModuleList(nn.LayerNorm(pdim) for _ in range(rounds))
        self.fg = nn.Linear(pdim, 1)
        self.cls = nn.Linear(pdim, n_cls)
        self.link = nn.Sequential(nn.LayerNorm(3 * pdim), nn.Linear(3 * pdim, pdim), nn.GELU(), nn.Linear(pdim, 1))

    def forward(self, h: torch.Tensor, logit: torch.Tensor, b: dict) -> dict:
        B, N, C = h.shape
        tok, m = b["pc_tok"], b["pc_mask"]
        P = tok.shape[1]
        flat_h = h.reshape(B * N, C)
        ti = (tok.clamp(min=0) + (torch.arange(B, device=h.device) * N)[:, None]).reshape(-1)
        conf = torch.sigmoid(logit.detach().float()).reshape(-1)[ti].reshape(B, P)
        z = self.inp(b["pc_feat"].to(h.dtype)) + self.from_shape(flat_h[ti].reshape(B, P, C)) + self.conf_in(conf[..., None].to(h.dtype))
        z = z * m[..., None].to(z.dtype)
        zf = z.reshape(B * P, -1)
        c = conf.reshape(-1).to(z.dtype)
        e, em = b["pc_e"], b["pc_emask"]
        off = (torch.arange(B, device=h.device) * P)[:, None]
        i = (e[..., 0] + off)[em]
        j = (e[..., 1] + off)[em]
        face = (b["pc_ek"][em] == 2) if "pc_ek" in b else torch.zeros_like(i, dtype=torch.bool)   # 1 touch, 2 facing (two walls of a duct)
        ii, jj = torch.cat([i, j]), torch.cat([j, i])                  # both directions
        ff = torch.cat([face, face])
        cc = b["pc_c"].reshape(B * P, 2).float()
        def geo(a, d, f):
            fa, fd = b["pc_feat"].reshape(B * P, -1)[a], b["pc_feat"].reshape(B * P, -1)[d]
            ca, sa, cd, sd = fa[:, 18], fa[:, 19], fd[:, 18], fd[:, 19]   # doubled-angle directions (piece_features columns 18, 19)
            dist = torch.log1p((cc[a] - cc[d]).norm(dim=-1) * 100.0) / 5.0
            return torch.stack([ca * cd + sa * sd, (sa * cd - ca * sd).abs(), dist, f.float()], -1).to(z.dtype)
        g_e = self.geo(geo(ii, jj, ff)) if len(ii) else zf.new_zeros((0, zf.shape[1]))
        deg = torch.zeros(B * P, device=h.device, dtype=zf.dtype).index_add_(0, ii, torch.ones_like(ii, dtype=zf.dtype)).clamp(min=1)[:, None]
        for msg, gate, norm in zip(self.msg, self.gate, self.norm):
            if len(ii):
                inp = torch.cat([zf[ii], zf[jj], g_e, c[jj][:, None]], -1)
                mm = torch.sigmoid(gate(inp)) * msg(inp)
                agg = torch.zeros_like(zf).index_add_(0, ii, mm.to(zf.dtype)) / deg
                zf = norm(zf + agg)
            c = torch.sigmoid(self.fg(zf)[:, 0])                        # refined confidence: the next round's gates read it
        out = {"fg": self.fg(zf)[:, 0].reshape(B, P), "cls": self.cls(zf).reshape(B, P, -1)}
        if len(i):
            out["link"] = self.link(torch.cat([zf[i] + zf[j], (zf[i] - zf[j]).abs(), self.geo(geo(i, j, face))], -1))[:, 0]
        else:
            out["link"] = zf.new_zeros(0)
        if self.region is not None:
            out["region_features"] = self.region.project(zf).reshape(B, P, -1)
        out["i"], out["j"] = i, j
        return out


def piece_loss(lp: dict, b: dict, pos_weight: float = 2.0, w_cls: float = 0.5, w_link: float = 0.5) -> torch.Tensor:
    """OpenStroke34: pipe BCE of every candidate piece (weighted by sqrt(length + 1), pos_weight on pipe pieces), class CE on pipe pieces
    with a class (GT pages; -1 = no class: Viewer pages), and the same-object loss on piece links balanced like edge_loss(pairs=True)."""
    m = b["pc_mask"]
    if not bool(m.any()):
        return lp["fg"].float().sum() * 0.0
    y = b["pc_fg"].float()
    w = (b["pc_len"].float() + 1.0).sqrt() * torch.where(y > 0, pos_weight, 1.0) * m.float()
    fg = (F.binary_cross_entropy_with_logits(lp["fg"].float(), y, reduction="none") * w).sum() / w.sum().clamp(min=1e-6)
    total = fg
    cm = m & (b["pc_cls"] > 0)
    if bool(cm.any()):
        total = total + w_cls * F.cross_entropy(lp["cls"][cm].float(), b["pc_cls"][cm])
    if len(lp["link"]):
        B, P = m.shape
        yi = (y.reshape(-1) > 0)
        inst = b["pc_inst"].reshape(-1)
        i, j = lp["i"], lp["j"]
        t = (yi[i] & yi[j] & (inst[i] == inst[j]) & (inst[i] >= 0)).float()
        bce = F.binary_cross_entropy_with_logits(lp["link"].float(), t, reduction="none")
        tt = yi[i] & yi[j]
        for sel, wt in ((tt & (t > 0), 1.0), (tt & (t == 0), 1.0), (yi[i] ^ yi[j], EDGE_TB_WEIGHT)):
            if bool(sel.any()):
                total = total + w_link * wt * bce[sel].mean()
    return total


class GroupContext(nn.Module):
    """Light attention between *groups* of shapes (H16). Every group of the window (a PDF layer, or a style group like
    AI_Takeoff's ``layer_1``: colour + width + kind) becomes one token: mean and max of its members, centroid and spread
    of their positions. Group tokens attend to each other (a few hundred per window), so the pipe network can be
    compared with the background groups it overlaps; the result goes back to every member through a zero-init
    projection, so the block starts as the identity (``test_v2_group_context_starts_as_the_model_without_it``)."""

    def __init__(self, dim: int, heads: int = 4, max_groups: int = 1024):
        super().__init__()
        self.max_groups = max_groups
        self.norm_in = nn.LayerNorm(dim)  # pre-norm: the raw residual stream made group tokens swamp the shapes (F25)
        self.inp = nn.Linear(dim, dim)
        self.tok = nn.Sequential(nn.Linear(2 * dim + 5, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out = nn.Linear(dim, dim)

    def forward(self, h, gid, mask, xy):
        B, N, C = h.shape
        gid = gid.clamp(min=0, max=self.max_groups - 1)
        G = int(gid[mask].max()) + 1 if mask.any() else 1
        key = (gid + torch.arange(B, device=h.device)[:, None] * G).reshape(-1)
        m = mask.reshape(-1).float()
        v = F.gelu(self.inp(self.norm_in(h))).float().reshape(-1, C) * m[:, None]
        cnt = torch.zeros(B * G, device=h.device).index_add_(0, key, m)
        mean = torch.zeros(B * G, C, device=h.device).index_add_(0, key, v) / cnt.clamp(min=1)[:, None]
        mx = torch.full((B * G, C), -1e4, device=h.device).scatter_reduce(
            0, key[:, None].expand(-1, C), torch.where(m[:, None] > 0, v, torch.full_like(v, -1e4)), reduce="amax",
            include_self=True)
        mx = torch.where(cnt[:, None] > 0, mx, torch.zeros_like(mx))
        p = xy.float().reshape(-1, 2) / 1000.0 * m[:, None]  # pt -> ~page units
        pm = torch.zeros(B * G, 2, device=h.device).index_add_(0, key, p) / cnt.clamp(min=1)[:, None]
        p2 = torch.zeros(B * G, 2, device=h.device).index_add_(0, key, p * p) / cnt.clamp(min=1)[:, None]
        ps = (p2 - pm * pm).clamp(min=0).sqrt()
        size = torch.log1p(cnt)[:, None] / 10.0
        t = self.tok(torch.cat([mean, mx, pm, ps, size], -1)).reshape(B, G, C)
        empty = (cnt.reshape(B, G) == 0)
        empty[:, 0] = empty[:, 0] & ~empty.all(1)  # never a fully masked row
        q = self.norm(t)
        a, _ = self.attn(q, q, q, key_padding_mask=empty, need_weights=False)
        g = self.out(self.norm(t + a))  # (B, G, C), normalised before going back to every member
        back = torch.gather(g, 1, gid[..., None].expand(-1, -1, C))
        return back.to(h.dtype) * mask[..., None].to(h.dtype)


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
        if not torch.is_grad_enabled() and pts.device.type == "cpu" and N > 4096:
            g = None
            t_vec = self.t.to(pts.dtype).view(1, 1, P, 1)
            for s in range(0, N, 4096):
                e = min(s + 4096, N)
                pc = pts[:, s:e]
                sc = pc.flatten(2).abs().amax(-1, keepdim=True).clamp(min=1e-3)
                p = pc / sc[..., None]
                tv = t_vec.expand(B, e - s, P, 1)
                ls = torch.log1p(sc)[..., None].expand(B, e - s, P, 1)
                hc = self.point(torch.cat([p, tv, ls], -1))
                if g is None:
                    g = torch.empty((B, N, hc.shape[-1] * 2), device=pts.device, dtype=hc.dtype)
                c = hc.shape[-1]
                g[:, s:e, :c] = hc.amax(2)
                g[:, s:e, c:] = hc.mean(2)
            return self.fuse(torch.cat([g, self.feat(x_norm)], -1))
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
    elif isinstance(m, nn.Embedding):
        # PyTorch's default N(0, 1) is ~10x the scale of everything else here: a link-kind / layer-name embedding of
        # that size is a large, nearly constant vector added to every message or member, and the logits collapse to
        # one value (openstroke3 / D-V3G on Colab, openstroke4 on CPU, 2026-10-21; LEDGER F26). ViT / BERT: std 0.02.
        nn.init.normal_(m.weight, std=0.02)
        if m.padding_idx is not None:
            with torch.no_grad():
                m.weight[m.padding_idx].zero_()


def _cc_spans(n_cc: int, lm, xy_m):
    """Exact per-CC center span max(xmax - xmin, ymax - ymin) via single argsort + reduceat (~6x faster than 4x np.minimum.at)."""
    import numpy as np

    sp = np.zeros(n_cc, dtype=np.float64)
    if not len(lm):
        return sp
    ord_l = np.argsort(lm, kind="quicksort")
    ls = lm[ord_l]
    chg = np.flatnonzero(ls[1:] != ls[:-1]) + 1
    st = np.empty(len(chg) + 1, dtype=np.int64)
    st[0] = 0
    st[1:] = chg
    xs = xy_m[ord_l, 0]
    ys = xy_m[ord_l, 1]
    sp[ls[st]] = np.maximum(
        np.maximum.reduceat(xs, st) - np.minimum.reduceat(xs, st),
        np.maximum.reduceat(ys, st) - np.minimum.reduceat(ys, st),
    )
    return sp


def _build_sym_csr(n: int, e):
    """Build deduplicated symmetric unweighted CSR adjacency matrix directly from undirected edges ``e``."""
    import numpy as np
    from scipy.sparse import csr_matrix

    if n == 0 or not len(e):
        return None
    u, v = e[:, 0], e[:, 1]
    keep = u != v
    if not keep.all():
        u, v = u[keep], v[keep]
    if not len(u):
        return None
    row = np.concatenate([u, v])
    col = np.concatenate([v, u])
    adj = csr_matrix((np.ones(len(row), dtype=np.int32), (row, col)), shape=(n, n))
    adj.sum_duplicates()
    adj.data.fill(1)
    return adj


def _near_pts_exp(pts_src, xy_m, radius: float = 40.0, tau: float = 8.0):
    """Exact ``exp(-d / tau)`` to nearest point in ``pts_src`` within ``radius`` (0 if none), with bbox prefilter."""
    import numpy as np
    from scipy.spatial import cKDTree
    from torch.utils.data import get_worker_info

    out = np.zeros(len(xy_m), dtype=np.float32)
    if not len(pts_src) or not len(xy_m):
        return out
    lo = pts_src.min(axis=0) - radius
    hi = pts_src.max(axis=0) + radius
    in_b = (xy_m[:, 0] >= lo[0]) & (xy_m[:, 0] <= hi[0]) & (xy_m[:, 1] >= lo[1]) & (xy_m[:, 1] <= hi[1])
    if not in_b.any():
        return out
    tree = cKDTree(pts_src)
    q_pts = xy_m[in_b]
    workers = -1 if (get_worker_info() is None and len(q_pts) >= 4096) else 1
    d, _ = tree.query(q_pts, distance_upper_bound=radius, workers=workers)
    out[in_b] = np.where(np.isfinite(d), np.exp(-d / tau), 0.0).astype(np.float32)
    return out


def _filter_edges_bi(e_raw, m, N: int):
    """Filter valid in-range active edges for one batch item."""
    e0, e1 = e_raw[:, 0], e_raw[:, 1]
    ok = (e0 >= 0) & (e0 < N) & (e1 >= 0) & (e1 < N)
    if not ok.all():
        e_raw = e_raw[ok]
        e0, e1 = e_raw[:, 0], e_raw[:, 1]
    if not m.all() and len(e_raw):
        ok2 = m[e0] & m[e1]
        e_raw = e_raw[ok2]
    return e_raw


def _filter_sw_edges(e_all, lw, rgb):
    """Filter style-consistent edges (|d_log_w| <= 0.25 & max|d_rgb| <= 0.15) without 3-col temp allocation."""
    import numpy as np

    if not len(e_all):
        return e_all
    e0, e1 = e_all[:, 0], e_all[:, 1]
    dw = np.abs(lw[e0] - lw[e1])
    dc = np.maximum(
        np.maximum(np.abs(rgb[e0, 0] - rgb[e1, 0]), np.abs(rgb[e0, 1] - rgb[e1, 1])),
        np.abs(rgb[e0, 2] - rgb[e1, 2]),
    )
    return e_all[(dw <= 0.25) & (dc <= 0.15)]


def _graph_topo_features(b: dict) -> torch.Tensor:
    """OpenStroke44 (H170): 5 exact scale/rotation-invariant ``g3`` connected-component topology features per shape:
      [log1p(cc_span)/6, log1p(cc_len)/8, log1p(cc_nodes)/5, log1p(deg)/3, is_small_box(nodes<=8 & span<35pt)].
    Computed dynamically from the batch's live ``edges`` / ``emask`` / ``xy0`` / ``length`` so values stay 100%
    exact after ``fg_swap``, ``neg_boxes`` v3, dihedral TTA, and during inference (0 cache rebuild required)."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    B, N = b["mask0"].shape
    dev = b["x"].device
    out = np.zeros((B, N, 5), dtype=np.float32)
    mask_np = b["mask0"].detach().cpu().numpy().astype(bool)
    xy_np = b["xy0"].detach().cpu().numpy().astype(np.float64)
    len_np = np.maximum(b["length"].detach().cpu().numpy().astype(np.float64), 0.0)
    has_e = "edges" in b and "emask" in b
    edges_np = b["edges"].detach().cpu().numpy() if has_e else None
    emask_np = b["emask"].detach().cpu().numpy().astype(bool) if has_e else None

    for bi in range(B):
        m = mask_np[bi]
        if not m.any():
            continue
        if has_e and emask_np[bi].any():
            e = _filter_edges_bi(edges_np[bi][emask_np[bi]], m, N)
        else:
            e = np.empty((0, 2), dtype=np.int64)

        if len(e):
            deg = (np.bincount(e[:, 0], minlength=N) + np.bincount(e[:, 1], minlength=N)).astype(np.float64)
            adj = coo_matrix((np.ones(len(e), dtype=np.float32), (e[:, 0], e[:, 1])), shape=(N, N))
            n_cc, lab = connected_components(adj, directed=False)
        else:
            deg = np.zeros(N, dtype=np.float64)
            n_cc = N
            lab = np.arange(N, dtype=np.int64)

        idx_m = np.flatnonzero(m)
        lm = lab[idx_m]
        cc_nodes = np.bincount(lm, minlength=n_cc).astype(np.float64)
        cc_len = np.bincount(lm, weights=len_np[bi, idx_m], minlength=n_cc)
        span = _cc_spans(n_cc, lm, xy_np[bi, idx_m])
        span = np.where(cc_nodes == 1, np.maximum(span, cc_len * 0.25), span)

        s_span = span[lm]
        s_len = cc_len[lm]
        s_nodes = cc_nodes[lm]
        s_deg = deg[idx_m]
        out[bi, idx_m, 0] = np.log1p(s_span) / 6.0
        out[bi, idx_m, 1] = np.log1p(s_len) / 8.0
        out[bi, idx_m, 2] = np.log1p(s_nodes) / 5.0
        out[bi, idx_m, 3] = np.log1p(s_deg) / 3.0
        out[bi, idx_m, 4] = ((s_nodes <= 8.0) & (s_span < 35.0)).astype(np.float32)

    return torch.from_numpy(out).to(device=dev)


def _graph_topo_v2_features(b: dict) -> torch.Tensor:
    """OpenStroke45 (H172): 7 style-consistent ``g3`` CC topology + relative line-weight features per shape:
      [log1p(sw_span)/6, log1p(sw_len)/8, log1p(sw_nodes)/5, log1p(sw_deg)/3, sw_is_small_box,
       rel_width_p90, is_thinner_half]."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from ..data import HAS_STROKE, LOG_WIDTH, STROKE

    B, N = b["mask0"].shape
    dev = b["x"].device
    out = np.zeros((B, N, 7), dtype=np.float32)
    mask_np = b["mask0"].detach().cpu().numpy().astype(bool)
    xy_np = b["xy0"].detach().cpu().numpy().astype(np.float32)
    len_np = np.maximum(b["length"].detach().cpu().numpy().astype(np.float32), 0.0)
    cols_5 = [LOG_WIDTH, *STROKE, HAS_STROKE]
    x5 = b["x"][..., cols_5].detach().float().cpu().numpy()
    lw_all = x5[..., 0]
    rgb_all = x5[..., 1:4]
    hs_all = x5[..., 4] > 0.5
    has_e = "edges" in b and "emask" in b
    edges_np = b["edges"].detach().cpu().numpy() if has_e else None
    emask_np = b["emask"].detach().cpu().numpy().astype(bool) if has_e else None

    for bi in range(B):
        m = mask_np[bi]
        if not m.any():
            continue
        lw = lw_all[bi]
        rgb = rgb_all[bi]
        if has_e and emask_np[bi].any():
            e = _filter_sw_edges(_filter_edges_bi(edges_np[bi][emask_np[bi]], m, N), lw, rgb)
        else:
            e = np.empty((0, 2), dtype=np.int64)

        if len(e):
            deg = (np.bincount(e[:, 0], minlength=N) + np.bincount(e[:, 1], minlength=N)).astype(np.float64)
            adj = coo_matrix((np.ones(len(e), dtype=np.float32), (e[:, 0], e[:, 1])), shape=(N, N))
            n_cc, lab = connected_components(adj, directed=False)
        else:
            deg = np.zeros(N, dtype=np.float64)
            n_cc = N
            lab = np.arange(N, dtype=np.int64)

        idx_m = np.flatnonzero(m)
        lm = lab[idx_m]
        cc_nodes = np.bincount(lm, minlength=n_cc).astype(np.float64)
        cc_len = np.bincount(lm, weights=len_np[bi, idx_m], minlength=n_cc)
        span = _cc_spans(n_cc, lm, xy_np[bi, idx_m].astype(np.float64))
        span = np.where(cc_nodes == 1, np.maximum(span, cc_len * 0.25), span)

        s_span = span[lm]
        s_len = cc_len[lm]
        s_nodes = cc_nodes[lm]
        s_deg = deg[idx_m]
        out[bi, idx_m, 0] = np.log1p(s_span) / 6.0
        out[bi, idx_m, 1] = np.log1p(s_len) / 8.0
        out[bi, idx_m, 2] = np.log1p(s_nodes) / 5.0
        out[bi, idx_m, 3] = np.log1p(s_deg) / 3.0
        out[bi, idx_m, 4] = ((s_nodes <= 8.0) & (s_span < 35.0)).astype(np.float32)

        lw_m = lw[idx_m]
        w_m = np.expm1(np.clip(lw_m, 0.0, 10.0)) / 10.0
        nz = hs_all[bi, idx_m] & (w_m > 0.01)
        if nz.any():
            p90_lw = float(np.percentile(lw_m[nz], 90))
            p90_w = float(np.expm1(p90_lw) / 10.0)
            rel_w = np.where(nz, np.clip(lw_m - p90_lw, -2.0, 2.0) / 2.0, 0.0)
            is_thin = (nz & (p90_w >= 0.30) & (w_m <= 0.65 * p90_w)).astype(np.float32)
            out[bi, idx_m, 5] = rel_w.astype(np.float32)
            out[bi, idx_m, 6] = is_thin

    return torch.from_numpy(out).to(device=dev)


def _graph_topo_v3_features(b: dict) -> torch.Tensor:
    """OpenStroke46 (H174): 12 style-consistent ``g3`` CC topology + length-weighted line-weight hierarchy +
    curved-fitting / mid-size-furniture / filled-arrowhead-louver descriptors per shape."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from ..data import HAS_FILL, HAS_STROKE, LOG_WIDTH, STROKE

    _I_FRAC_C = FEATURE_NAMES.index("frac_c")
    _I_CLOSED = FEATURE_NAMES.index("closed_ends")
    _I_LOG_NSEG = FEATURE_NAMES.index("log_nseg")
    _I_LOG_BW = FEATURE_NAMES.index("log_bbox_w")
    _I_LOG_BH = FEATURE_NAMES.index("log_bbox_h")
    _I_LOG_DIAG = FEATURE_NAMES.index("log_bbox_diag")

    B, N = b["mask0"].shape
    dev = b["x"].device
    out = np.zeros((B, N, 12), dtype=np.float32)
    mask_np = b["mask0"].detach().cpu().numpy().astype(bool)
    xy_np = b["xy0"].detach().cpu().numpy().astype(np.float32)
    len_np = np.maximum(b["length"].detach().cpu().numpy().astype(np.float32), 0.0)
    cols_12 = [LOG_WIDTH, *STROKE, HAS_STROKE, HAS_FILL, _I_FRAC_C, _I_CLOSED, _I_LOG_NSEG, _I_LOG_BW, _I_LOG_BH, _I_LOG_DIAG]
    x12 = b["x"][..., cols_12].detach().float().cpu().numpy()
    lw_all = x12[..., 0]
    rgb_all = x12[..., 1:4]
    hs_all = x12[..., 4] > 0.5
    hf_all = x12[..., 5] > 0.5
    fc_all = x12[..., 6]
    cl_all = x12[..., 7] > 0.5
    ns_all = x12[..., 8]
    bw_all = x12[..., 9]
    bh_all = x12[..., 10]
    dg_all = x12[..., 11]
    has_e = "edges" in b and "emask" in b
    edges_np = b["edges"].detach().cpu().numpy() if has_e else None
    emask_np = b["emask"].detach().cpu().numpy().astype(bool) if has_e else None

    for bi in range(B):
        m = mask_np[bi]
        if not m.any():
            continue
        lw = lw_all[bi]
        rgb = rgb_all[bi]
        if has_e and emask_np[bi].any():
            e = _filter_sw_edges(_filter_edges_bi(edges_np[bi][emask_np[bi]], m, N), lw, rgb)
        else:
            e = np.empty((0, 2), dtype=np.int64)

        if len(e):
            deg = (np.bincount(e[:, 0], minlength=N) + np.bincount(e[:, 1], minlength=N)).astype(np.float64)
            adj = coo_matrix((np.ones(len(e), dtype=np.float32), (e[:, 0], e[:, 1])), shape=(N, N))
            n_cc, lab = connected_components(adj, directed=False)
        else:
            deg = np.zeros(N, dtype=np.float64)
            n_cc = N
            lab = np.arange(N, dtype=np.int64)

        idx_m = np.flatnonzero(m)
        xy_m = xy_np[bi, idx_m].astype(np.float64)
        lm = lab[idx_m]
        len_m = len_np[bi, idx_m].astype(np.float64)
        cc_nodes = np.bincount(lm, minlength=n_cc).astype(np.float64)
        cc_len = np.bincount(lm, weights=len_m, minlength=n_cc)
        cc_cur_len = np.bincount(lm, weights=len_m * (fc_all[bi, idx_m] > 0.05), minlength=n_cc)
        cc_cur_frac = cc_cur_len / np.maximum(cc_len, 1e-6)

        span = _cc_spans(n_cc, lm, xy_m)
        span = np.where(cc_nodes == 1, np.maximum(span, cc_len * 0.25), span)

        s_span = span[lm]
        s_len = cc_len[lm]
        s_nodes = cc_nodes[lm]
        s_deg = deg[idx_m]
        s_cur = cc_cur_frac[lm]
        win_max_span = float(s_span.max()) if len(s_span) else 0.0
        rel_span = np.log1p(s_span) / max(np.log1p(win_max_span), 1.0)

        out[bi, idx_m, 0] = np.log1p(s_span) / 6.0
        out[bi, idx_m, 1] = np.log1p(s_len) / 8.0
        out[bi, idx_m, 2] = np.log1p(s_nodes) / 5.0
        out[bi, idx_m, 3] = np.log1p(s_deg) / 3.0
        out[bi, idx_m, 4] = ((s_nodes <= 8.0) & (s_span < 35.0)).astype(np.float32)

        lw_m = lw[idx_m]
        w_m = np.expm1(np.clip(lw_m, 0.0, 10.0)) / 10.0
        nz = hs_all[bi, idx_m] & (w_m > 0.01)
        if nz.any():
            lw_nz = lw_m[nz]
            L_nz = np.maximum(len_m[nz], 1e-3)
            o_w = np.argsort(lw_nz, kind="quicksort")
            cw = np.cumsum(L_nz[o_w]) / float(L_nz.sum())
            p90_lw = float(lw_nz[o_w[min(int(np.searchsorted(cw, 0.90)), len(lw_nz) - 1)]])
            p95_lw = float(lw_nz[o_w[min(int(np.searchsorted(cw, 0.95)), len(lw_nz) - 1)]])
            p95_w = float(np.expm1(p95_lw) / 10.0)
            out[bi, idx_m, 5] = np.where(nz, np.clip(lw_m - p90_lw, -2.0, 2.0) / 2.0, 0.0).astype(np.float32)
            out[bi, idx_m, 6] = (nz & (p95_w >= 0.30) & (w_m <= 0.65 * p95_w)).astype(np.float32)

        out[bi, idx_m, 7] = s_cur.astype(np.float32)
        out[bi, idx_m, 8] = rel_span.astype(np.float32)
        out[bi, idx_m, 9] = ((s_span >= 35.0) & (s_span <= 140.0) & (s_cur < 0.01) & (rel_span < 0.72)).astype(np.float32)

        diag_m = np.expm1(np.clip(dg_all[bi, idx_m], 0.0, 10.0))
        small_fill = hf_all[bi, idx_m] & (diag_m >= 0.3) & (diag_m <= 10.0)
        near_sf = _near_pts_exp(xy_m[small_fill], xy_m) if small_fill.any() else np.zeros(len(idx_m), dtype=np.float32)
        out[bi, idx_m, 10] = near_sf

        bw_m = np.expm1(np.clip(bw_all[bi, idx_m], 0.0, 10.0))
        bh_m = np.expm1(np.clip(bh_all[bi, idx_m], 0.0, 10.0))
        bshort_m = np.minimum(bw_m, bh_m)
        blong_m = np.maximum(bw_m, bh_m)
        nseg_m = np.expm1(np.clip(ns_all[bi, idx_m], 0.0, 10.0))
        out[bi, idx_m, 11] = (
            cl_all[bi, idx_m]
            & (nseg_m >= 3.5) & (nseg_m <= 4.5)
            & (bshort_m <= 1.2) & (blong_m <= 14.0)
            & (near_sf >= 0.30)
        ).astype(np.float32)

    return torch.from_numpy(out).to(device=dev)


def _kcore2_and_peel(n: int, e: np.ndarray, xy: np.ndarray, max_rounds: int = 16, *, adj=None, need_peel: bool = True):
    """O(N + E) iterative degree-1 peeling on undirected graph ``e`` -> (in_2core_big, core2_span, peel_norm).
    Uses C/SciPy CSR deduplication + sparse-frontier peeling + CSR subgraph CC (bit-exact, ~3.5x faster than list comp)."""
    import numpy as np
    from scipy.sparse.csgraph import connected_components

    if adj is None:
        if n == 0 or not len(e):
            return np.zeros(n, dtype=bool), np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float32)
        adj = _build_sym_csr(n, e)
    if n == 0 or adj is None or adj.nnz == 0:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float32)

    indptr = adj.indptr
    indices = adj.indices
    deg = np.diff(indptr).astype(np.int32)
    alive = deg > 0
    peel = np.zeros(n, dtype=np.int32) if need_peel else None
    rnd = 0
    frontier = np.flatnonzero(deg == 1)
    while len(frontier) and rnd < max_rounds:
        rnd += 1
        if need_peel:
            peel[frontier] = rnd
        alive[frontier] = False
        deg[frontier] = 0
        if len(frontier) * 8 < n:
            starts = indptr[frontier]
            ends = indptr[frontier + 1]
            lens = ends - starts
            if not int(lens.sum()):
                break
            nbs = indices[starts] if (lens == 1).all() else adj[frontier].indices
            nbs = nbs[alive[nbs]]
            if not len(nbs):
                break
            u_nb, cnt_nb = np.unique(nbs, return_counts=True)
            deg[u_nb] -= cnt_nb.astype(np.int32)
        else:
            f_vec = np.zeros(n, dtype=np.int32)
            f_vec[frontier] = 1
            hit = adj.dot(f_vec)
            u_nb = np.flatnonzero((hit > 0) & alive)
            if not len(u_nb):
                break
            deg[u_nb] -= hit[u_nb]
        zeroed = u_nb[deg[u_nb] <= 0]
        if len(zeroed):
            if need_peel:
                peel[zeroed] = rnd + 1
            alive[zeroed] = False
        frontier = u_nb[deg[u_nb] == 1]

    in_core = alive & (deg >= 2)
    peel_norm = (
        np.where(in_core, 1.0, peel.astype(np.float32) / max(float(rnd + 1), 1.0)).astype(np.float32)
        if need_peel else np.zeros(n, dtype=np.float32)
    )
    core_span = np.zeros(n, dtype=np.float64)
    if in_core.any():
        ic_idx = np.flatnonzero(in_core)
        sub_adj = adj[ic_idx][:, ic_idx]
        if sub_adj.nnz > 0:
            nc, cl = connected_components(sub_adj, directed=False)
            sp = _cc_spans(nc, cl, xy[ic_idx].astype(np.float64, copy=False))
            core_span[ic_idx] = sp[cl]
            in_core = in_core & (core_span >= 60.0)
    return in_core, core_span, peel_norm


def _graph_topo_fire_features(b: dict) -> torch.Tensor:
    """OpenStroke47 (FIRE 3-Class: 0=other, 1=main, 2=branch): 12 dynamic layer-free topological features per shape."""
    import numpy as np
    from scipy.sparse.csgraph import connected_components

    from ..data import HAS_FILL, HAS_STROKE, LOG_WIDTH, STROKE

    _I_ELONG = FEATURE_NAMES.index("elong")
    _I_FRAC_C = FEATURE_NAMES.index("frac_c")
    _I_CLOSED = FEATURE_NAMES.index("closed_ends")
    _I_LOG_DIAG = FEATURE_NAMES.index("log_bbox_diag")

    B, N = b["mask0"].shape
    dev = b["x"].device
    out = np.zeros((B, N, 12), dtype=np.float32)
    mask_np = b["mask0"].detach().cpu().numpy().astype(bool)
    xy_np = b["xy0"].detach().cpu().numpy().astype(np.float32)
    len_np = np.maximum(b["length"].detach().cpu().numpy().astype(np.float32), 0.0)
    cols = [LOG_WIDTH, *STROKE, HAS_STROKE, HAS_FILL, _I_ELONG, _I_FRAC_C, _I_CLOSED, _I_LOG_DIAG]
    xf = b["x"][..., cols].detach().float().cpu().numpy()
    lw_all, rgb_all = xf[..., 0], xf[..., 1:4]
    hs_all, hf_all = xf[..., 4] > 0.5, xf[..., 5] > 0.5
    el_all, fc_all, cl_all, dg_all = xf[..., 6], xf[..., 7], xf[..., 8] > 0.5, xf[..., 9]
    has_e = "edges" in b and "emask" in b
    edges_np = b["edges"].detach().cpu().numpy() if has_e else None
    emask_np = b["emask"].detach().cpu().numpy().astype(bool) if has_e else None

    for bi in range(B):
        m = mask_np[bi]
        if not m.any():
            continue
        lw, rgb = lw_all[bi], rgb_all[bi]
        if has_e and emask_np[bi].any():
            e_all = _filter_edges_bi(edges_np[bi][emask_np[bi]], m, N)
            e_sw = _filter_sw_edges(e_all, lw, rgb)
        else:
            e_all = e_sw = np.empty((0, 2), dtype=np.int64)

        if len(e_sw):
            deg = (np.bincount(e_sw[:, 0], minlength=N) + np.bincount(e_sw[:, 1], minlength=N)).astype(np.float64)
            adj_sw = _build_sym_csr(N, e_sw)
            n_cc, lab = connected_components(adj_sw, directed=False) if adj_sw is not None else (N, np.arange(N, dtype=np.int64))
        else:
            deg = np.zeros(N, dtype=np.float64)
            adj_sw = None
            n_cc, lab = N, np.arange(N, dtype=np.int64)

        idx_m = np.flatnonzero(m)
        xy64 = xy_np[bi].astype(np.float64)
        xy_m = xy64[idx_m]
        lm = lab[idx_m]
        len_m = len_np[bi, idx_m].astype(np.float64)
        cc_nodes = np.bincount(lm, minlength=n_cc).astype(np.float64)
        cc_len = np.bincount(lm, weights=len_m, minlength=n_cc)
        span = _cc_spans(n_cc, lm, xy_m)
        span = np.where(cc_nodes == 1, np.maximum(span, cc_len * 0.25), span)

        s_span = span[lm]
        out[bi, idx_m, 0] = np.log1p(s_span) / 6.0
        out[bi, idx_m, 1] = np.log1p(cc_len[lm]) / 8.0
        out[bi, idx_m, 2] = np.log1p(cc_nodes[lm]) / 5.0
        out[bi, idx_m, 3] = np.log1p(deg[idx_m]) / 3.0
        out[bi, idx_m, 4] = ((cc_nodes[lm] <= 8.0) & (s_span < 35.0)).astype(np.float32)

        lw_m = lw[idx_m]
        w_m = np.expm1(np.clip(lw_m, 0.0, 10.0)) / 10.0
        nz = hs_all[bi, idx_m] & (w_m > 0.01)
        if nz.any():
            lw_nz = lw_m[nz]
            L_nz = np.maximum(len_m[nz], 1e-3)
            o_w = np.argsort(lw_nz, kind="quicksort")
            cw = np.cumsum(L_nz[o_w]) / float(L_nz.sum())
            p90_lw = float(lw_nz[o_w[min(int(np.searchsorted(cw, 0.90)), len(lw_nz) - 1)]])
            out[bi, idx_m, 5] = np.where(nz, np.clip(lw_m - p90_lw, -2.0, 2.0) / 2.0, 0.0).astype(np.float32)

        out[bi, idx_m, 6] = (hf_all[bi, idx_m] & (el_all[bi, idx_m] <= 0.35) & (len_m >= 5.0)).astype(np.float32)

        if len(e_all) == len(e_sw):
            c2_sw, sp2_sw, peel_sw = _kcore2_and_peel(N, e_sw, xy64, adj=adj_sw, need_peel=True)
            c2_all, sp2_all = c2_sw, sp2_sw
        else:
            c2_all, sp2_all, _ = _kcore2_and_peel(N, e_all, xy64, need_peel=False)
            c2_sw, sp2_sw, peel_sw = _kcore2_and_peel(N, e_sw, xy64, adj=adj_sw, need_peel=True)
        out[bi, idx_m, 7] = c2_all[idx_m].astype(np.float32)
        out[bi, idx_m, 8] = c2_sw[idx_m].astype(np.float32)
        out[bi, idx_m, 9] = (np.log1p(np.maximum(sp2_all[idx_m], sp2_sw[idx_m])) / 6.0).astype(np.float32)
        out[bi, idx_m, 10] = peel_sw[idx_m]

        diag_m = np.expm1(np.clip(dg_all[bi, idx_m], 0.0, 10.0))
        small_sym = (diag_m >= 1.0) & (diag_m <= 12.0) & (cl_all[bi, idx_m] | (fc_all[bi, idx_m] > 0.30))
        if small_sym.any():
            out[bi, idx_m, 11] = _near_pts_exp(xy_m[small_sym], xy_m)

    return torch.from_numpy(out).to(device=dev)


def _graph_topo_uni_features(b: dict) -> torch.Tensor:
    """OpenStroke46 Unified (general-purpose CAD vector+graph topology, 0 layers, 0 system assumptions): 18D per shape."""
    import numpy as np
    from scipy.sparse.csgraph import connected_components

    from ..data import HAS_FILL, HAS_STROKE, LOG_WIDTH, STROKE

    _I_ELONG = FEATURE_NAMES.index("elong")
    _I_FRAC_C = FEATURE_NAMES.index("frac_c")
    _I_CLOSED = FEATURE_NAMES.index("closed_ends")
    _I_LOG_NSEG = FEATURE_NAMES.index("log_nseg")
    _I_LOG_BW = FEATURE_NAMES.index("log_bbox_w")
    _I_LOG_BH = FEATURE_NAMES.index("log_bbox_h")
    _I_LOG_DIAG = FEATURE_NAMES.index("log_bbox_diag")

    B, N = b["mask0"].shape
    dev = b["x"].device
    out = np.zeros((B, N, 18), dtype=np.float32)
    mask_np = b["mask0"].detach().cpu().numpy().astype(bool)
    xy_np = b["xy0"].detach().cpu().numpy().astype(np.float32)
    len_np = np.maximum(b["length"].detach().cpu().numpy().astype(np.float32), 0.0)
    cols = [LOG_WIDTH, *STROKE, HAS_STROKE, HAS_FILL, _I_ELONG, _I_FRAC_C, _I_CLOSED, _I_LOG_NSEG, _I_LOG_BW, _I_LOG_BH, _I_LOG_DIAG]
    xu = b["x"][..., cols].detach().float().cpu().numpy()
    lw_all, rgb_all = xu[..., 0], xu[..., 1:4]
    hs_all, hf_all = xu[..., 4] > 0.5, xu[..., 5] > 0.5
    el_all, fc_all, cl_all = xu[..., 6], xu[..., 7], xu[..., 8] > 0.5
    ns_all, bw_all, bh_all, dg_all = xu[..., 9], xu[..., 10], xu[..., 11], xu[..., 12]
    has_e = "edges" in b and "emask" in b
    edges_np = b["edges"].detach().cpu().numpy() if has_e else None
    emask_np = b["emask"].detach().cpu().numpy().astype(bool) if has_e else None

    for bi in range(B):
        m = mask_np[bi]
        if not m.any():
            continue
        lw, rgb = lw_all[bi], rgb_all[bi]
        if has_e and emask_np[bi].any():
            e_all = _filter_edges_bi(edges_np[bi][emask_np[bi]], m, N)
            e_sw = _filter_sw_edges(e_all, lw, rgb)
        else:
            e_all = e_sw = np.empty((0, 2), dtype=np.int64)

        if len(e_sw):
            deg = (np.bincount(e_sw[:, 0], minlength=N) + np.bincount(e_sw[:, 1], minlength=N)).astype(np.float64)
            adj_sw = _build_sym_csr(N, e_sw)
            n_cc, lab = connected_components(adj_sw, directed=False) if adj_sw is not None else (N, np.arange(N, dtype=np.int64))
        else:
            deg = np.zeros(N, dtype=np.float64)
            adj_sw = None
            n_cc, lab = N, np.arange(N, dtype=np.int64)

        idx_m = np.flatnonzero(m)
        xy64 = xy_np[bi].astype(np.float64)
        xy_m = xy64[idx_m]
        lm = lab[idx_m]
        len_m = len_np[bi, idx_m].astype(np.float64)
        cc_nodes = np.bincount(lm, minlength=n_cc).astype(np.float64)
        cc_len = np.bincount(lm, weights=len_m, minlength=n_cc)
        cc_cur_len = np.bincount(lm, weights=len_m * (fc_all[bi, idx_m] > 0.05), minlength=n_cc)
        cc_cur_frac = cc_cur_len / np.maximum(cc_len, 1e-6)

        span = _cc_spans(n_cc, lm, xy_m)
        span = np.where(cc_nodes == 1, np.maximum(span, cc_len * 0.25), span)

        s_span = span[lm]
        s_len = cc_len[lm]
        s_nodes = cc_nodes[lm]
        s_deg = deg[idx_m]
        s_cur = cc_cur_frac[lm]
        win_max_span = float(s_span.max()) if len(s_span) else 0.0
        rel_span = np.log1p(s_span) / max(np.log1p(win_max_span), 1.0)

        out[bi, idx_m, 0] = np.log1p(s_span) / 6.0
        out[bi, idx_m, 1] = np.log1p(s_len) / 8.0
        out[bi, idx_m, 2] = np.log1p(s_nodes) / 5.0
        out[bi, idx_m, 3] = np.log1p(s_deg) / 3.0
        out[bi, idx_m, 4] = ((s_nodes <= 8.0) & (s_span < 35.0)).astype(np.float32)

        lw_m = lw[idx_m]
        w_m = np.expm1(np.clip(lw_m, 0.0, 10.0)) / 10.0
        nz = hs_all[bi, idx_m] & (w_m > 0.01)
        if nz.any():
            lw_nz = lw_m[nz]
            L_nz = np.maximum(len_m[nz], 1e-3)
            o_w = np.argsort(lw_nz, kind="quicksort")
            cw = np.cumsum(L_nz[o_w]) / float(L_nz.sum())
            p90_lw = float(lw_nz[o_w[min(int(np.searchsorted(cw, 0.90)), len(lw_nz) - 1)]])
            p95_lw = float(lw_nz[o_w[min(int(np.searchsorted(cw, 0.95)), len(lw_nz) - 1)]])
            p95_w = float(np.expm1(p95_lw) / 10.0)
            out[bi, idx_m, 5] = np.where(nz, np.clip(lw_m - p90_lw, -2.0, 2.0) / 2.0, 0.0).astype(np.float32)
            out[bi, idx_m, 6] = (nz & (p95_w >= 0.30) & (w_m <= 0.65 * p95_w)).astype(np.float32)

        out[bi, idx_m, 7] = s_cur.astype(np.float32)
        out[bi, idx_m, 8] = rel_span.astype(np.float32)
        out[bi, idx_m, 9] = ((s_span >= 35.0) & (s_span <= 140.0) & (s_cur < 0.01) & (rel_span < 0.72)).astype(np.float32)

        diag_m = np.expm1(np.clip(dg_all[bi, idx_m], 0.0, 10.0))
        small_fill = hf_all[bi, idx_m] & (diag_m >= 0.3) & (diag_m <= 10.0)
        near_sf = _near_pts_exp(xy_m[small_fill], xy_m) if small_fill.any() else np.zeros(len(idx_m), dtype=np.float32)
        out[bi, idx_m, 10] = near_sf

        bw_m = np.expm1(np.clip(bw_all[bi, idx_m], 0.0, 10.0))
        bh_m = np.expm1(np.clip(bh_all[bi, idx_m], 0.0, 10.0))
        bshort_m = np.minimum(bw_m, bh_m)
        blong_m = np.maximum(bw_m, bh_m)
        nseg_m = np.expm1(np.clip(ns_all[bi, idx_m], 0.0, 10.0))
        out[bi, idx_m, 11] = (
            cl_all[bi, idx_m]
            & (nseg_m >= 3.5) & (nseg_m <= 4.5)
            & (bshort_m <= 1.2) & (blong_m <= 14.0)
            & (near_sf >= 0.30)
        ).astype(np.float32)

        out[bi, idx_m, 12] = (hf_all[bi, idx_m] & (el_all[bi, idx_m] <= 0.35) & (len_m >= 5.0)).astype(np.float32)

        if len(e_all) == len(e_sw):
            c2_sw, sp2_sw, peel_sw = _kcore2_and_peel(N, e_sw, xy64, adj=adj_sw, need_peel=True)
            c2_all, sp2_all = c2_sw, sp2_sw
        else:
            c2_all, sp2_all, _ = _kcore2_and_peel(N, e_all, xy64, need_peel=False)
            c2_sw, sp2_sw, peel_sw = _kcore2_and_peel(N, e_sw, xy64, adj=adj_sw, need_peel=True)
        out[bi, idx_m, 13] = c2_all[idx_m].astype(np.float32)
        out[bi, idx_m, 14] = c2_sw[idx_m].astype(np.float32)
        out[bi, idx_m, 15] = (np.log1p(np.maximum(sp2_all[idx_m], sp2_sw[idx_m])) / 6.0).astype(np.float32)
        out[bi, idx_m, 16] = peel_sw[idx_m]

        small_sym = (diag_m >= 1.0) & (diag_m <= 12.0) & (cl_all[bi, idx_m] | (fc_all[bi, idx_m] > 0.30))
        if small_sym.any():
            out[bi, idx_m, 17] = _near_pts_exp(xy_m[small_sym], xy_m)

    return torch.from_numpy(out).to(device=dev)


def topo_mode_of(model) -> str | None:
    """Topology feature mode of a model (used by DataLoader workers to precompute ``b['topo_feat']`` off GPU)."""
    if getattr(model, "topo_mlp", None) is None:
        return None
    if getattr(model, "graph_topo_uni", False):
        return "uni"
    if getattr(model, "graph_topo_fire", False):
        return "fire"
    if getattr(model, "graph_topo_v3", False):
        return "v3"
    if getattr(model, "graph_topo_v2", False):
        return "v2"
    return "v1"


def compute_graph_topo(b: dict, mode: str) -> torch.Tensor:
    """Single source of truth for dynamic graph topology features (shared by DataLoader workers and forward fallback)."""
    if mode == "uni":
        return _graph_topo_uni_features(b)
    if mode == "fire":
        return _graph_topo_fire_features(b)
    if mode == "v3":
        return _graph_topo_v3_features(b)
    if mode == "v2":
        return _graph_topo_v2_features(b)
    return _graph_topo_features(b)


class OpenStroke2(nn.Module):
    def __init__(self, n_features: int, n_points: int = 16, dims=(64, 96, 128, 192, 256), enc_depths=(2, 2, 2, 4, 2),
                 dec_depths=(1, 1, 1, 1), head_dim: int = 16, patch: int = 1024, drop_path: float = 0.3,
                 layer_pool: bool = True, shuffle_orders: bool = True, mode_input: bool = False,
                 layer_names: bool = False, group_context: bool = False, relations: int = 0, mixers=None,
                 feature_drop=None, pretrain: bool = False, layer_head: bool = False, layer_head_names: bool = False,
                 image_branch: bool = False, img_pretrained: bool = True, img_lum: bool = False, pool_by: str = "layer",
                 img_ppt: float = 1.0, img_max_px: int = 1536, img_fpn: bool = False, img_aux: bool = False, img_backbone: str = "resnet18", obj_head: bool = False, img_weights: str | None = None,
                 rel_attn: int = 0, edge_head: bool = False, edge_pairs: bool = False, piece_head: bool = False, region_head: bool = False, continuation: bool = False,
                 stroke_context: bool = False, group_head: bool = False, facing_edges: bool = False,
                 img_cad: bool = False, img_cad_v2: bool = False, graph_topo: bool = False, graph_topo_v2: bool = False,
                 graph_topo_v3: bool = False, graph_topo_fire: bool = False, graph_topo_uni: bool = False,
                 edge_style: bool = False, stroke_classes: int = 1, class_names=None):
        super().__init__()
        assert len(dims) == len(enc_depths) == len(dec_depths) + 1 <= len(LEVEL_CELLS)
        self.stroke_classes = max(int(stroke_classes or 1), 1)
        self.class_names = tuple(str(x) for x in class_names) if class_names else None
        self.config = dict(n_features=n_features, n_points=n_points, dims=list(dims), enc_depths=list(enc_depths),
                           dec_depths=list(dec_depths), head_dim=head_dim, patch=patch, drop_path=drop_path,
                           layer_pool=layer_pool, shuffle_orders=shuffle_orders, mode_input=mode_input,
                           layer_names=layer_names, group_context=group_context, relations=relations,
                           mixers=list(mixers) if mixers else None,
                           feature_drop=list(feature_drop) if feature_drop else None, pretrain=bool(pretrain),
                           layer_head=bool(layer_head), layer_head_names=bool(layer_head_names),
                           image_branch=bool(image_branch), img_pretrained=bool(img_pretrained), img_lum=bool(img_lum),
                           pool_by=str(pool_by), img_ppt=float(img_ppt), img_max_px=int(img_max_px), img_backbone=str(img_backbone), img_fpn=bool(img_fpn),
                           img_aux=bool(img_aux), obj_head=bool(obj_head), rel_attn=int(rel_attn), edge_head=bool(edge_head),
                           edge_pairs=bool(edge_pairs), piece_head=bool(piece_head), region_head=bool(region_head), continuation=bool(continuation),
                           stroke_context=bool(stroke_context), group_head=bool(group_head), facing_edges=bool(facing_edges),
                           img_cad=bool(img_cad), img_cad_v2=bool(img_cad_v2), graph_topo=bool(graph_topo),
                           graph_topo_v2=bool(graph_topo_v2), graph_topo_v3=bool(graph_topo_v3),
                           graph_topo_fire=bool(graph_topo_fire), graph_topo_uni=bool(graph_topo_uni),
                           edge_style=bool(edge_style), stroke_classes=int(self.stroke_classes))
        if self.class_names is not None:
            self.config["class_names"] = list(self.class_names)
        mixers = list(mixers) if mixers else ["attn"] * len(dims)
        assert len(mixers) == len(dims) and all(m in ("attn", "conv", "graph") for m in mixers), mixers
        assert all(m != "graph" for m in mixers[1:]), "graph mixing needs shape-level links: level 0 only"
        self.mixers = mixers
        self.levels = len(dims)
        self.shuffle_orders = shuffle_orders
        self.register_buffer("feat_mean", torch.zeros(n_features))
        self.register_buffer("feat_std", torch.ones(n_features))
        from ..features import feature_mask

        keep = torch.ones(n_features)
        keep[feature_mask(feature_drop)] = 0.0
        #: hidden input features are fixed at their train mean (0 after normalisation): the model cannot use them
        self.register_buffer("feat_keep", keep)
        self.embed = ShapeEncoder(n_features, n_points, dims[0])
        total = sum(enc_depths) + sum(dec_depths)
        rates = iter(torch.linspace(0, drop_path, total).tolist())
        heads = [max(d // head_dim, 1) for d in dims]
        self.enc = nn.ModuleList(
            nn.ModuleList(Block(dims[l], heads[l], patch, next(rates), layer_pool and l == 0, mixer=mixers[l])
                          for _ in range(enc_depths[l]))
            for l in range(self.levels))
        self.down = nn.ModuleList(nn.Sequential(nn.Linear(dims[l], dims[l + 1]), nn.LayerNorm(dims[l + 1]))
                                  for l in range(self.levels - 1))
        self.up_proj = nn.ModuleList(nn.Linear(dims[l + 1], dims[l]) for l in range(self.levels - 1))
        self.skip_proj = nn.ModuleList(nn.Linear(dims[l], dims[l]) for l in range(self.levels - 1))
        self.dec = nn.ModuleList(
            nn.ModuleList(Block(dims[l], heads[l], patch, next(rates), layer_pool and l == 0, mixer=mixers[l])
                          for _ in range(dec_depths[l]))
            for l in range(self.levels - 1))
        self.head = nn.Sequential(nn.LayerNorm(dims[0]), nn.Linear(dims[0], dims[0]), nn.GELU(), nn.Linear(dims[0], self.stroke_classes))
        #: FIRE / HVAC / PLUMBING flow of the page (data.MODES), added to every token; zero-init (LEDGER F14)
        self.mode_embed = nn.Embedding(len(MODES), dims[0]) if mode_input else None
        #: hashed tokens of the PDF layer name (H15, LEDGER F19): mean embedding per layer -> every member; zero-init out
        self.name_embed = nn.Embedding(N_BUCKETS, dims[0], padding_idx=0) if layer_names else None
        self.name_proj = nn.Linear(dims[0], dims[0]) if layer_names else None
        #: H16: attention between layer groups and between style groups, before the head (sees the full U-Net context)
        self.gc_layer = GroupContext(dims[0]) if group_context else None
        self.gc_style = GroupContext(dims[0]) if group_context else None
        #: OpenStroke3 (H17): ``relations`` rounds of message passing along the link graph, before the head
        self.rel = nn.ModuleList(RelationBlock(dims[0]) for _ in range(int(relations)))
        #: OpenStroke30 (H132): ``rel_attn`` graph-attention blocks along the link graph after the decoder, and a same-object link head
        #: OpenStroke42 (F235): ``facing_edges=True`` expands link kind embeddings 8 -> 16 to carry facing parallel-wall bit 8
        n_edge_kinds = 16 if facing_edges else 8
        n_edge_geo = 6 if edge_style else 4
        self.edge_style = bool(edge_style)
        self.gat = nn.ModuleList(GraphAttention(dims[0], n_kinds=n_edge_kinds, geo_dim=n_edge_geo) for _ in range(int(rel_attn)))
        self.edge_mlp = EdgeHead(dims[0], n_kinds=n_edge_kinds, geo_dim=n_edge_geo) if edge_head else None
        #: OpenStroke31 (H136): the link head's loss balances same- and other-object links between two true strokes (edge_loss pairs=)
        self.edge_pairs = bool(edge_pairs)
        #: OpenStroke34 (H141): object head on pieces of strokes, with confidence passed along touching pieces
        from ..data import PIECE_FEATS
        from ..objects import N_CLASSES
        self.piece = PieceHead(dims[0], PIECE_FEATS, N_CLASSES) if piece_head else None
        self.last_piece = None
        self.last_edge = None
        #: E19 masked-shape pretraining: masked tokens get one learned embedding instead of their own; the recon head
        #: predicts their normalised input features from the context. Unused (no batch key ``mae``) when fine-tuning.
        self.mask_token = nn.Parameter(torch.zeros(dims[0])) if pretrain else None
        self.recon = (nn.Sequential(nn.LayerNorm(dims[0]), nn.Linear(dims[0], dims[0]), nn.GELU(),
                                    nn.Linear(dims[0], n_features)) if pretrain else None)
        self.apply(_init_weights)
        if self.mask_token is not None:
            nn.init.normal_(self.mask_token, std=0.02)
        #: E24 layer head: mean + max of the final token features over each layer of the window -> "pipe layer" logit.
        #: Built after the rest (same initial weights as without it); the shape head is unchanged.
        self.layer_mlp = None
        self.last_layer = None
        if layer_head:
            self.layer_mlp = nn.Sequential(nn.LayerNorm(dims[0]), nn.Linear(dims[0], dims[0]), nn.GELU(),
                                           nn.Linear(dims[0], 1))
            self.layer_mlp.apply(_init_weights)
        #: E27: the layer head (only) also reads the hashed tokens of the layer name; zero-init (starts as E24)
        self.lh_name = nn.Embedding(N_BUCKETS, dims[0], padding_idx=0) if layer_head and layer_head_names else None
        self.lh_name_proj = nn.Linear(dims[0], dims[0]) if self.lh_name is not None else None
        if self.lh_name is not None:
            nn.init.normal_(self.lh_name.weight, std=0.02)
            nn.init.zeros_(self.lh_name_proj.weight)
            nn.init.zeros_(self.lh_name_proj.bias)
        #: E62: HVAC object class of each shape and the offset to its object's centre (objects.py); its own head, the pipe logit is unchanged
        self.obj_mlp = None
        self.last_obj = None
        if obj_head:
            from ..objects import N_CLASSES

            self.obj_mlp = nn.Sequential(nn.LayerNorm(dims[0]), nn.Linear(dims[0], dims[0]), nn.GELU(), nn.Linear(dims[0], N_CLASSES + 2))
            self.obj_mlp.apply(_init_weights)
        #: E28: the window drawn as an image, read by a pretrained CNN at each shape's points; zero-init (starts as without)
        self.img = None
        if stroke_context and not image_branch:
            raise ValueError("stroke_context requires an image branch")
        if (img_cad or img_cad_v2) and not image_branch:
            raise ValueError("img_cad / img_cad_v2 requires an image branch")
        if image_branch:
            from .image_branch import ImageBranch

            self.img = ImageBranch(dims[0], pretrained=img_pretrained, lum=img_lum, ppt=img_ppt, max_px=img_max_px,
                                   fpn=img_fpn, aux=img_aux, backbone=img_backbone, weights=img_weights,
                                   stroke_context=stroke_context, img_cad=img_cad, img_cad_v2=img_cad_v2)
        #: OpenStroke44 (5D) / OpenStroke45 (7D) / OpenStroke46 legacy (12D) / OpenStroke47 legacy (12D) / Unified (18D): zero-init topology MLP
        self.topo_mlp = None
        self.graph_topo_v2 = bool(graph_topo_v2)
        self.graph_topo_v3 = bool(graph_topo_v3)
        self.graph_topo_fire = bool(graph_topo_fire)
        self.graph_topo_uni = bool(graph_topo_uni)
        if graph_topo_uni or graph_topo_fire or graph_topo_v3 or graph_topo_v2 or graph_topo:
            in_topo = 18 if graph_topo_uni else (12 if (graph_topo_fire or graph_topo_v3) else (7 if graph_topo_v2 else 5))
            hid_topo = 64 if graph_topo_uni else (48 if (graph_topo_fire or graph_topo_v3) else 32)
            self.topo_mlp = nn.Sequential(nn.Linear(in_topo, hid_topo), nn.GELU(), nn.Linear(hid_topo, dims[0]))
            self.topo_mlp.apply(_init_weights)
            nn.init.zeros_(self.topo_mlp[-1].weight)
            nn.init.zeros_(self.topo_mlp[-1].bias)
        #: E31: LayerPool over the page-local style groups (colour + fill + width + kind, AI_Takeoff ``layer_1``) instead of
        #: the PDF layer field, which some drawings do not have
        self.pool_by = pool_by
        for rb in self.rel:
            nn.init.zeros_(rb.out.weight)
            nn.init.zeros_(rb.out.bias)
        for ga in self.gat:   # zero-init: the blocks start as the identity (model = the checkpoint it starts from)
            for lin in (ga.out, ga.ffn[-1]):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        for gc in (self.gc_layer, self.gc_style):
            if gc is not None:
                nn.init.zeros_(gc.out.weight)
                nn.init.zeros_(gc.out.bias)
        if self.mode_embed is not None:
            nn.init.zeros_(self.mode_embed.weight)
        if self.name_proj is not None:
            nn.init.zeros_(self.name_proj.weight)
            nn.init.zeros_(self.name_proj.bias)
        for mod in self.modules():  # LayerPool keeps its zero-init output (starts as the model without it)
            if isinstance(mod, LayerPool):
                nn.init.zeros_(mod.out.weight)
                nn.init.zeros_(mod.out.bias)

        if region_head:
            if self.piece is None:
                raise ValueError("region_head requires piece_head")
            from .region_decoder import RegionDecoder
            self.piece.region = RegionDecoder(self.piece.fg.in_features, N_CLASSES)

        self.continuation = None
        if continuation:
            from ..continuation import ContinuationHead
            self.continuation = ContinuationHead(dims[0])

        #: OpenStroke41 (H162): object-level 8-class + FP-rejection head pooled over grouped stroke clusters
        self.group_head = None
        self.last_h = None
        if group_head:
            from .group_head import GroupClsHead
            self.group_head = GroupClsHead(dims[0])

    def set_normalization(self, mean, std) -> None:
        import numpy as np

        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.as_tensor(np.maximum(std, 1e-6), dtype=torch.float32))

    def _order_idx(self, b, level, i) -> int:
        perm = b[f"perm{level}"]  # (B, 4, N)
        return (i + (self._shift if self.training and self.shuffle_orders else 0)) % perm.shape[1]

    def _order(self, b, level, i):
        return b[f"perm{level}"][:, self._order_idx(b, level, i)]

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
        x = ((b["x"] - self.feat_mean) / self.feat_std).clamp(-10, 10) * self.feat_keep
        h = self.embed(x, b["pts"])
        if self.name_embed is not None and "ltok" in b:
            tok = b["ltok"]  # (B, n_layers, T)
            m = (tok > 0).to(h.dtype)[..., None]
            v = (self.name_embed(tok).to(h.dtype) * m).sum(2) / m.sum(2).clamp(min=1)
            per = gather_tokens(self.name_proj(v), b["layer"].clamp(max=tok.shape[1] - 1))
            h = h + per * b["mask0"][..., None].to(h.dtype)
        if self.mode_embed is not None and "mode" in b:
            h = h + self.mode_embed(b["mode"])[:, None, :] * b["mask0"][..., None].to(h.dtype)
        mae = b.get("mae") if self.mask_token is not None else None
        if mae is not None:
            h = torch.where(mae[..., None], self.mask_token.to(h.dtype), h)
        if self.img is not None:  # after the mask token: a masked shape still sees the image around it (drawn without it)
            h = h + self.img(b).to(h.dtype)
        if self.topo_mlp is not None:
            t_feat = b.get("topo_feat")
            if t_feat is None:
                t_feat = compute_graph_topo(b, topo_mode_of(self))
            h = h + self.topo_mlp(t_feat.to(device=h.device, dtype=h.dtype)) * b["mask0"][..., None].to(h.dtype)
        g = link_context(b) if "graph" in self.mixers else None
        grp = b["style"] if self.pool_by == "style" and "style" in b else b["layer"]
        has_pool0 = (any(blk.pool is not None for blk in self.enc[0])
                     or (self.levels > 1 and any(blk.pool is not None for blk in self.dec[0])))
        if has_pool0 and grp.numel():
            # Avoid a GPU->CPU sync when no_layers guarantees grp == 0 everywhere (openstroke28..45)
            n_grp = 1 if (self.config.get("no_layers") and self.pool_by != "style") else (int(grp.max()) + 1)
        else:
            n_grp = None
        skips = []
        all_valids = []
        bi = 0
        for l in range(self.levels):
            mask, pos = b[f"mask{l}"], b[f"xy{l}"] / LEVEL_CELLS[l] / 8.0
            av = bool(mask.all()) if mask.device.type == "cpu" else False
            all_valids.append(av)
            if l > 0:
                h = self._pool(self.down[l - 1](h), b[f"parent{l}"], b[f"mask{l - 1}"], mask.shape[1])
                if not av:
                    h = h * mask[..., None]
            for blk in self.enc[l]:
                k = self._order_idx(b, l, bi)
                perm = b[f"perm{l}"][:, k]
                h = blk(h, pos, mask, perm, grp if l == 0 else None, g if l == 0 else None, n_grp if l == 0 else None,
                        all_valid=av, id_perm=(l == 0 and k == 0))
                bi += 1
            skips.append(h)
        for l in range(self.levels - 2, -1, -1):
            mask, pos = b[f"mask{l}"], b[f"xy{l}"] / LEVEL_CELLS[l] / 8.0
            av = all_valids[l]
            up = gather_tokens(self.up_proj[l](h), b[f"parent{l + 1}"])
            h = up + self.skip_proj[l](skips[l])
            if not av:
                h = h * mask[..., None]
            for blk in self.dec[l]:
                k = self._order_idx(b, l, bi)
                perm = b[f"perm{l}"][:, k]
                h = blk(h, pos, mask, perm, grp if l == 0 else None, g if l == 0 else None, n_grp if l == 0 else None,
                        all_valid=av, id_perm=(l == 0 and k == 0))
                bi += 1
        for rb in self.rel:
            if "edges" in b:
                h = rb(h, b, b["x"])
        links = flat_links(b) if (len(self.gat) or self.edge_mlp is not None or self.continuation is not None) else None
        link_geo = edge_geometry(b, links[0], links[1], b["x"], style=self.edge_style) if links is not None else None
        for ga in self.gat:
            h = ga(h, b, b["x"], links, geo=link_geo)
        self.last_edge = None
        if self.edge_mlp is not None and links is not None:
            self.last_edge = (self.edge_mlp(h, b, b["x"], links, geo=link_geo), links[0], links[1])
        if self.gc_layer is not None:
            m0, xy0 = b["mask0"], b["xy0"]
            h = h + self.gc_layer(h, b["layer"], m0, xy0) + self.gc_style(h, b.get("style", b["layer"]), m0, xy0)
        if mae is not None:
            return self.recon(h)
        if self.layer_mlp is not None:
            n_l = int(b["layer"].max()) + 1
            g = self._pool(h, b["layer"], b["mask0"], n_l)
            if self.lh_name is not None and "ltok" in b:
                tok = b["ltok"][:, :n_l]
                m = (tok > 0).to(g.dtype)[..., None]
                v = (self.lh_name(tok).to(g.dtype) * m).sum(2) / m.sum(2).clamp(min=1)
                g = g + self.lh_name_proj(v)
            self.last_layer = self.layer_mlp(g)[..., 0]  # (B, n_l)
        if self.obj_mlp is not None:
            self.last_obj = self.obj_mlp(h)  # (B, N, classes + 2)
        self.last_h = h if self.group_head is not None else None
        logit = self.head(h)[..., 0] if self.stroke_classes == 1 else self.head(h)
        if self.continuation is not None:
            logit = self.continuation(h, logit, links, link_geo)
        self.last_piece = self.piece(h, logit, b) if self.piece is not None and "pc_tok" in b else None
        return logit

    def target_features(self, b: dict) -> torch.Tensor:
        """What the recon head predicts: the normalised input features (hidden groups stay 0)."""

        return ((b["x"] - self.feat_mean) / self.feat_std).clamp(-10, 10) * self.feat_keep


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


def lovasz_sigmoid_batch(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sync-free batched binary Lovasz-Softmax over (B, N) with boolean validity mask (bit-exact to per-item lovasz_sigmoid)."""
    if logits.numel() == 0:
        return logits.sum() * 0.0
    err = (labels - torch.sigmoid(logits)).abs()
    err_pad = torch.where(mask, err, err.new_full((), -1.0))
    err_s, order = torch.sort(err_pad, dim=1, descending=True)
    m_s = torch.gather(mask, 1, order)
    gt = torch.gather(labels, 1, order) * m_s.to(labels.dtype)
    gts = gt.sum(dim=1, keepdim=True)
    inter = gts - gt.cumsum(dim=1)
    union = gts + (1.0 - gt).cumsum(dim=1)
    jac = 1.0 - inter / union.clamp(min=1e-6)
    if jac.shape[1] > 1:
        jac = torch.cat([jac[:, :1], jac[:, 1:] - jac[:, :-1]], dim=1)
    return (err_s * m_s.to(err_s.dtype) * jac).sum(dim=1).mean()


def closure_loss(logits, b, tau: float = 1.0):
    """OpenStroke30 (H133): every GT object must be picked whole. Per object of a window (true strokes sharing ``oinst``, >= 2 of them), a
    soft minimum of their logits, -tau * logsumexp(-z / tau) (<= the weakest stroke's logit), gets BCE towards 1: the loss sits on the
    stroke an object is missing (4 strokes found 3), not on strokes already picked. Mean over objects."""

    if "oinst" not in b or logits.ndim != 2:
        return logits.float().sum() * 0.0
    B, N = logits.shape
    m = (b["label"] > 0) & b["mask0"] & (b["oinst"] >= 0)
    if "lw" in b:
        m = m & (b["lw"] > 0)
    if not bool(m.any()):
        return logits.float().sum() * 0.0
    stride = b["oinst"].max().clamp(min=0) + 2
    key = (b["oinst"] + 1 + (torch.arange(B, device=logits.device) * stride)[:, None])[m]
    uk, inv = torch.unique(key, return_inverse=True)
    z = -logits.float()[m] / tau
    mx = torch.full((len(uk),), -1e4, device=z.device).scatter_reduce(0, inv, z, reduce="amax", include_self=True)
    s = torch.zeros(len(uk), device=z.device).index_add_(0, inv, torch.exp(z - mx[inv]))
    cnt = torch.zeros(len(uk), device=z.device).index_add_(0, inv, torch.ones_like(z))
    smin = -tau * (mx + torch.log(s))
    ok = (cnt >= 2).to(smin.dtype)
    return (F.softplus(-smin) * ok).sum() / ok.sum().clamp(min=1.0)


#: OpenStroke31 (H136): weight of the true-other links next to the two balanced true-true terms of the pairs loss
EDGE_TB_WEIGHT = 0.25


def edge_loss(last_edge, b, bg_weight: float = 0.1, pairs: bool = False):
    """OpenStroke30 (H132): BCE of the same-object link head (sync-free GPU/CPU branchless reduction, exact parity)."""

    if last_edge is None or "oinst" not in b:
        return b["label"].float().sum() * 0.0
    z, i, j = last_edge
    y = (b["label"] > 0).reshape(-1)
    o = b["oinst"].reshape(-1)
    lw_ok = (b["lw"].reshape(-1) > 0) if "lw" in b else torch.ones_like(y)
    keep_e = lw_ok[i] & lw_ok[j]
    t = (y[i] & y[j] & (o[i] == o[j]) & (o[i] >= 0)).float()
    bce = F.binary_cross_entropy_with_logits(z.float(), t, reduction="none")
    if not pairs:
        w = torch.where(y[i] | y[j], 1.0, bg_weight) * keep_e.float()
        return (bce * w).sum() / w.sum().clamp(min=1e-6)
    tt = keep_e & y[i] & y[j]
    total = z.float().sum() * 0.0
    for sel, wt in ((tt & (t > 0), 1.0), (tt & (t == 0), 1.0), (keep_e & (y[i] ^ y[j]), EDGE_TB_WEIGHT)):
        sf = sel.float()
        total = total + wt * ((bce * sf).sum() / sf.sum().clamp(min=1.0))
    return total


def lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Multi-class Lovasz-Softmax loss (Berman et al., CVPR 2018) over a flat set of tokens (N, C) & labels (N,).
    F253: on empty foreground classes (e.g. negative pages or windows without ``main``), apply top-32 FP suppression
    instead of skipping so false positives on empty classes still get strong gradients."""
    if probs.numel() == 0:
        return probs.sum() * 0.0
    C = probs.shape[1]
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0 and c > 0:
            k = min(32, probs.shape[0])
            losses.append(probs[:, c].topk(k).values.mean())
            continue
        err = (fg - probs[:, c]).abs()
        err_sorted, order = torch.sort(err, descending=True)
        losses.append(torch.dot(err_sorted, _lovasz_grad(fg[order])))
    return torch.stack(losses).mean() if losses else probs.sum() * 0.0


def loss_multiclass(logits: torch.Tensor, b: dict, pos_weight: float = 2.0, alpha: float = 0.5, lovasz: float = 1.0,
                    class_weights: torch.Tensor | None = None, label_smoothing: float = 0.03):
    """Dynamic K-class stroke loss (0=other, 1..K-1=foreground classes): length-weighted multi-class CE (with mild
    label_smoothing=0.03 to prevent 99.5%-background logit overconfidence & threshold drift to 0.05) + Lovasz-Softmax
    (with top-32 FP suppression on empty classes) + auxiliary foreground Lovasz (p_fg = 1 - p_other vs y > 0)."""
    mask = b["mask0"]
    if "lw" in b:
        mask = mask & (b["lw"] > 0)
    y = b["label"].long().clamp(0, logits.shape[-1] - 1)
    C = logits.shape[-1]
    if class_weights is not None and len(class_weights) == C:
        cls_w = class_weights.to(device=logits.device, dtype=torch.float32)
    else:
        cls_w = logits.new_ones(C, dtype=torch.float32)
        if C == 3:
            cls_w[1] = max(float(pos_weight) * 2.25, 3.0)
            cls_w[2] = max(float(pos_weight) * 1.5, 2.0)
        elif C == 2:
            cls_w[1] = float(pos_weight)
        elif C > 3:
            cls_w[1:] = max(float(pos_weight) * 1.8, 2.5)
    w = (b["length"].float().clamp(min=0) + 1.0) ** alpha * mask.float() * cls_w[y]
    if "lw" in b:
        w = w * b["lw"].float()
    per = F.cross_entropy(
        logits.float().reshape(-1, C), y.reshape(-1), reduction="none", label_smoothing=float(label_smoothing)
    ).reshape_as(y)
    ce = (per * w).sum() / w.sum().clamp(min=1e-6)
    parts = {"bce": ce}
    total = ce
    if lovasz > 0:
        probs = torch.softmax(logits.float(), dim=-1)
        lvs = []
        for bi in range(len(y)):
            m_i = mask[bi]
            if not bool(m_i.any()):
                continue
            p_i = probs[bi][m_i]
            y_i = y[bi][m_i]
            lv_mc = lovasz_softmax_flat(p_i, y_i)
            # Auxiliary binary foreground Lovasz on p_fg = 1 - p_other (top-32 FP suppression on negative windows)
            fg_i = (y_i > 0).float()
            p_fg_i = 1.0 - p_i[:, 0]
            if fg_i.sum() == 0:
                lv_fg = p_fg_i.topk(min(32, p_fg_i.shape[0])).values.mean()
            else:
                err_fg = (fg_i - p_fg_i).abs()
                es, ord_fg = torch.sort(err_fg, descending=True)
                lv_fg = torch.dot(es, _lovasz_grad(fg_i[ord_fg]))
            lvs.append(0.7 * lv_mc + 0.3 * lv_fg)
        lv = torch.stack(lvs).mean() if lvs else ce * 0.0
        parts["lovasz"] = lv
        total = total + lovasz * lv
    return total, parts


def loss_v2(logits, b, pos_weight=2.0, alpha=0.5, lovasz=1.0, poly=0.0, class_weights=None):
    """-> (total, {"bce": ..., "lovasz": ...}). BCE weighted by (length + 1)**alpha and pos_weight on true.
    Automatically dispatches to ``loss_multiclass`` when ``logits`` has shape ``(B, N, C)`` (dynamic K-class output).
    ``poly`` (Poly-1, Leng et al., ICLR 2022): adds poly * (1 - p_t) to each token's BCE, so a correct answer keeps earning
    as it grows more confident, where BCE's gradient fades; a confident wrong answer stays punished by BCE's unbounded term."""
    if logits.ndim == 3 and logits.shape[-1] > 1:
        return loss_multiclass(logits, b, pos_weight=pos_weight, alpha=alpha, lovasz=lovasz, class_weights=class_weights)

    mask = b["mask0"]
    y = b["label"]
    w = (b["length"].clamp(min=0) + 1.0) ** alpha * mask.float()
    w = w * torch.where(y > 0, pos_weight, 1.0)
    if "lw" in b:  # E36: coincident true shapes carry no loss
        w = w * b["lw"]
        mask = mask & (b["lw"] > 0)
    per = F.binary_cross_entropy_with_logits(logits.float(), y.float(), reduction="none")
    if poly > 0:
        pt = torch.where(y > 0, torch.sigmoid(logits.float()), 1.0 - torch.sigmoid(logits.float()))
        per = per + poly * (1.0 - pt)
    bce = (per * w).sum() / w.sum().clamp(min=1e-6)
    parts = {"bce": bce}
    total = bce
    if lovasz > 0:
        lv = lovasz_sigmoid_batch(logits.float(), y.float(), mask)
        parts["lovasz"] = lv
        total = total + lovasz * lv
    return total, parts
