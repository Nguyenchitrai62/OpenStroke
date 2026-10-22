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

    def forward(self, x, pos, mask, perm, layer=None, g=None):
        x = x + self.cpe(x, mask, perm)
        if self.attn is not None:
            x = x + self.dp(self.attn(self.n1(x), pos, mask, perm))
        elif self.graph is not None:
            x = x + self.dp(self.gamma.to(x.dtype) * self.graph(self.n1(x), g))
        if self.pool is not None:
            x = x + self.dp(self.pool(self.n2(x), layer, mask))
        x = x + self.dp(self.mlp(self.n3(x)))
        return x * mask[..., None]


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


class OpenStroke2(nn.Module):
    def __init__(self, n_features: int, n_points: int = 16, dims=(64, 96, 128, 192, 256), enc_depths=(2, 2, 2, 4, 2),
                 dec_depths=(1, 1, 1, 1), head_dim: int = 16, patch: int = 1024, drop_path: float = 0.3,
                 layer_pool: bool = True, shuffle_orders: bool = True, mode_input: bool = False,
                 layer_names: bool = False, group_context: bool = False, relations: int = 0, mixers=None,
                 feature_drop=None, pretrain: bool = False, layer_head: bool = False):
        super().__init__()
        assert len(dims) == len(enc_depths) == len(dec_depths) + 1 <= len(LEVEL_CELLS)
        self.config = dict(n_features=n_features, n_points=n_points, dims=list(dims), enc_depths=list(enc_depths),
                           dec_depths=list(dec_depths), head_dim=head_dim, patch=patch, drop_path=drop_path,
                           layer_pool=layer_pool, shuffle_orders=shuffle_orders, mode_input=mode_input,
                           layer_names=layer_names, group_context=group_context, relations=relations,
                           mixers=list(mixers) if mixers else None,
                           feature_drop=list(feature_drop) if feature_drop else None, pretrain=bool(pretrain),
                           layer_head=bool(layer_head))
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
        self.head = nn.Sequential(nn.LayerNorm(dims[0]), nn.Linear(dims[0], dims[0]), nn.GELU(), nn.Linear(dims[0], 1))
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
        for rb in self.rel:
            nn.init.zeros_(rb.out.weight)
            nn.init.zeros_(rb.out.bias)
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
        g = link_context(b) if "graph" in self.mixers else None
        skips = []
        bi = 0
        for l in range(self.levels):
            mask, pos = b[f"mask{l}"], b[f"xy{l}"] / LEVEL_CELLS[l] / 8.0
            if l > 0:
                h = self._pool(self.down[l - 1](h), b[f"parent{l}"], b[f"mask{l - 1}"], mask.shape[1])
                h = h * mask[..., None]
            for blk in self.enc[l]:
                h = blk(h, pos, mask, self._order(b, l, bi), b["layer"] if l == 0 else None, g if l == 0 else None)
                bi += 1
            skips.append(h)
        for l in range(self.levels - 2, -1, -1):
            mask, pos = b[f"mask{l}"], b[f"xy{l}"] / LEVEL_CELLS[l] / 8.0
            up = gather_tokens(self.up_proj[l](h), b[f"parent{l + 1}"])
            h = (up + self.skip_proj[l](skips[l])) * mask[..., None]
            for blk in self.dec[l]:
                h = blk(h, pos, mask, self._order(b, l, bi), b["layer"] if l == 0 else None, g if l == 0 else None)
                bi += 1
        for rb in self.rel:
            if "edges" in b:
                h = rb(h, b, b["x"])
        if self.gc_layer is not None:
            m0, xy0 = b["mask0"], b["xy0"]
            h = h + self.gc_layer(h, b["layer"], m0, xy0) + self.gc_style(h, b.get("style", b["layer"]), m0, xy0)
        if mae is not None:
            return self.recon(h)
        if self.layer_mlp is not None:
            n_l = int(b["layer"].max()) + 1
            self.last_layer = self.layer_mlp(self._pool(h, b["layer"], b["mask0"], n_l))[..., 0]  # (B, n_l)
        return self.head(h)[..., 0]

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
