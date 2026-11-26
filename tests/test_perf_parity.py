"""Strict numerical parity & autograd equivalence tests for all CPU/GPU performance optimizations.

Verifies zero accuracy loss (atol <= 1e-4 on FP32, bit-exact on discrete outputs) across:
  - ImageBranch: proj-before-sample linearity, CPU AVX2/FBGEMM F.embedding_bag bilinear sampling,
    channels_last (CPU + CUDA), sync-free binary scatter_reduce.
  - OpenStroke2: contiguous RoPE, fused CPE+PatchAttention perm, identity perm / all_valid skips,
    FlashAttention-2 patch splitting (full/partial/empty), chunked ShapeEncoder, undirected/fused-QKV
    GraphAttention, chunked EdgeHead, fast CSR _kcore2_and_peel & all 5 graph_topo modes.
  - Sync-free losses: lovasz_sigmoid_batch, closure_loss, edge_loss, _edge_cluster_stats, group_cls_loss.
  - Preprocessing & DataLoader: fast 1-seg shape_points, algebraic doubled-angle dir, _unique_rgb4 restyle,
    single-pass _rot90_arrays, shared-memory WindowDataset.epoch with persistent_workers=True.
  - Real checkpoint compatibility: E150 (openstroke45) & openstroke46.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch = pytest.importorskip("torch")
torch.set_num_threads(1)


def test_sample_mean_cpu_embedding_bag_exact_parity():
    import torch.nn.functional as F

    from openstroke.models.image_branch import _sample_mean, _sample_mean_cpu_eb

    torch.manual_seed(0)
    for (B, C, H, W, N, P), cl in [
        ((2, 64, 32, 48, 256, 16), False),
        ((2, 64, 64, 64, 512, 16), True),
        ((1, 16, 17, 19, 128, 8), True),
    ]:
        feat = torch.randn(B, C, H, W, dtype=torch.float32)
        if cl:
            feat = feat.contiguous(memory_format=torch.channels_last)
        grid = torch.rand(B, N, P, 2) * 2.4 - 1.2  # includes OOB < -1 and > 1
        # Inject exact boundary values -1.0, 0.0, +1.0
        grid[0, 0, :, 0] = -1.0
        grid[0, 1, :, 1] = 1.0
        grid[0, 2, :, :] = 0.0
        ref = F.grid_sample(feat, grid, align_corners=False).mean(-1).transpose(1, 2)
        got_eb = _sample_mean_cpu_eb(feat, grid)
        got_disp = _sample_mean(feat, grid)
        assert (ref - got_eb).abs().max().item() < 1e-5
        assert torch.equal(got_eb, got_disp)


def test_image_branch_proj_before_sample_and_channels_last_forward_backward_parity():
    import torch.nn.functional as F

    from openstroke.data import HAS_FILL, HAS_STROKE, LOG_WIDTH, STROKE
    from openstroke.features import FEATURE_NAMES, N_FEATURES
    from openstroke.models.image_branch import ImageBranch, enable_image_channels_last, fuse_image_branch, render_multi

    for cad_v2 in (False, True):
        torch.manual_seed(42 + int(cad_v2))
        B, N, P = 2, 96, 16
        xy0 = torch.randn(B, N, 2) * 60.0
        pts = torch.randn(B, N, P, 2) * 8.0
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[1, -12:] = False
        x = torch.zeros(B, N, N_FEATURES)
        x[..., HAS_STROKE] = 1.0
        x[..., LOG_WIDTH] = torch.rand(B, N) * 2.0
        x[..., STROKE] = torch.rand(B, N, 3)
        x[..., HAS_FILL] = (torch.rand(B, N) > 0.75).float()
        x[..., FEATURE_NAMES.index("frac_c")] = (torch.rand(B, N) > 0.8).float()
        b = {"x": x, "xy0": xy0, "pts": pts, "mask0": mask}

        img_opt = ImageBranch(
            dim=16, pretrained=False, ppt=1.0, max_px=128, lum=True, fpn=True,
            img_cad=not cad_v2, img_cad_v2=cad_v2,
        )
        with torch.no_grad():
            for p in img_opt.parameters():
                if p.ndim >= 1 and float(p.abs().sum()) == 0.0:
                    torch.nn.init.normal_(p, std=0.05)

        img_ref = copy.deepcopy(img_opt)

        # Reference forward using classic post-sample proj(cat([grid_sample(f2), grid_sample(f3)])) + proj_fpn(grid_sample(p))
        def ref_forward(im, batch):
            from openstroke.data import FILL

            m = batch["mask0"] > 0
            xf = batch["x"].float()
            rgb = torch.where(xf[..., HAS_STROKE:HAS_STROKE + 1] > 0.5, xf[..., STROKE], xf[..., FILL])
            dark = (1.0 - (rgb * im.lum_w).sum(-1)).clamp(0.05, 1.0)
            abs_pts = (batch["xy0"][:, :, None, :] + batch["pts"]).float()
            w_val = (xf[..., LOG_WIDTH] / 3.0).clamp(0.0, 1.0) * (xf[..., HAS_STROKE] > 0.5).float()
            f_val = xf[..., HAS_FILL]
            chans = [(dark, False), (w_val, False), (f_val, True)]
            if im.img_cad_v2:
                chans.append(((xf[..., FEATURE_NAMES.index("frac_c")] > 0.05).float(), True))
            maps, scale, S = render_multi(batch["xy0"], batch["pts"], m, chans, im.ppt, im.max_px, abs_pts=abs_pts)
            ink, w_map, f_map = maps[0], maps[1], maps[2]
            c_map = maps[3] if im.img_cad_v2 else None
            xi = ((1.0 - ink).expand(-1, 3, -1, -1) - im.mean) / im.std
            f1 = im.stem(xi)
            f2 = im.layer2(f1)
            f3 = im.layer3(f2)
            grid = (abs_pts * (scale / (S / 2))[:, None, None, None]).clamp(-1, 1)
            g2 = F.grid_sample(f2.float(), grid, align_corners=False).mean(-1).transpose(1, 2)
            g3 = F.grid_sample(f3.float(), grid, align_corners=False).mean(-1).transpose(1, 2)
            out = im.proj(torch.cat([g2, g3], dim=-1))
            p = im.lat[3](im.layer4(f3))
            for lat, f in zip(im.lat[2::-1], (f3, f2, f1)):
                p = lat(f) + F.interpolate(p, size=f.shape[-2:], mode="nearest")
            p = im.smooth(p)
            cad_in = torch.cat([w_map, f_map, c_map], dim=1) if im.img_cad_v2 else torch.cat([w_map, f_map], dim=1)
            p = p + im.cad_stem(cad_in.to(p.dtype), p.shape[-2:])
            gp = F.grid_sample(p.float(), grid, align_corners=False).mean(-1).transpose(1, 2)
            out = out + im.proj_fpn(gp)
            return out * m[..., None]

        # 1) Training forward + backward gradient parity (with channels_last enabled on opt)
        class _Wrap(torch.nn.Module):
            def __init__(self, im):
                super().__init__()
                self.img = im

        w_opt = _Wrap(img_opt)
        enable_image_channels_last(w_opt)
        img_opt.train()
        img_ref.train()
        y_ref = ref_forward(img_ref, b)
        y_opt = img_opt(b)
        assert (y_ref - y_opt).abs().max().item() < 1e-4
        y_ref.square().mean().backward()
        y_opt.square().mean().backward()
        for (n1, p1), (n2, p2) in zip(img_ref.named_parameters(), img_opt.named_parameters()):
            assert n1 == n2
            assert p1.grad is not None and p2.grad is not None
            assert (p1.grad - p2.grad).abs().max().item() < 1e-4, f"grad mismatch on {n1}"

        # 2) Inference fused + channels_last + CPU embedding_bag parity
        img_opt.eval()
        img_ref.eval()
        fuse_image_branch(w_opt, channels_last=True)
        with torch.inference_mode():
            yi_ref = ref_forward(img_ref, b)
            yi_opt = img_opt(b)
        assert (yi_ref - yi_opt).abs().max().item() < 1e-4


def test_rope_fused_block_and_sdpa_patch_splitting_parity():
    import torch.nn.functional as F

    from openstroke.models.openstroke2 import Block, rope_2d, rope_2d_qk

    torch.manual_seed(3)
    B, N, C = 2, 130, 32
    x = torch.randn(B, N, C, requires_grad=True)
    pos = torch.randn(B, N, 2)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[1, 90:] = False  # leaves full patches, 1 partial patch, and empty patches with patch=32!
    perm = torch.stack([torch.randperm(N) for _ in range(B)])

    # 1) rope_2d_qk vs rope_2d (including non-contiguous & odd d not divisible by 4)
    for d in (16, 18):
        q = torch.randn(B, 4, 2, 32, d).transpose(2, 3).transpose(2, 3)  # contiguous
        k = torch.randn(B, 4, 32, 2, d).transpose(2, 3)                  # non-contiguous
        pp = torch.randn(B, 4, 1, 32, 2)
        rq0, rk0 = rope_2d(q, pp), rope_2d(k, pp)
        rq1, rk1 = rope_2d_qk(q, k, pp)
        assert (rq0 - rq1).abs().max().item() < 1e-6
        assert (rk0 - rk1).abs().max().item() < 1e-6

    # 2) Fused Block vs unfused reference (both id_perm=False/True and all_valid=False/True)
    blk = Block(C, heads=2, patch=32, drop_path=0.0, layer_pool=False, mixer="attn").eval()
    for av, ip in [(False, False), (True, False), (True, True), (False, True)]:
        m_use = torch.ones(B, N, dtype=torch.bool) if av else mask
        p_use = torch.arange(N).expand(B, N) if ip else perm
        x1 = x.detach().clone().requires_grad_(True)
        x2 = x.detach().clone().requires_grad_(True)
        # Unfused reference
        h_ref = x1 + blk.cpe(x1, m_use, p_use)
        h_ref = h_ref + blk.attn(blk.n1(h_ref), pos, m_use, p_use)
        h_ref = (h_ref + blk.mlp(blk.n3(h_ref))) * m_use[..., None]
        # Fused optimized
        h_opt = blk(x2, pos, m_use, p_use, all_valid=av, id_perm=ip)
        assert (h_ref - h_opt).abs().max().item() < 1e-5
        h_ref.square().mean().backward()
        h_opt.square().mean().backward()
        assert (x1.grad - x2.grad).abs().max().item() < 1e-5

    # 3) Explicit check of empty/partial/full patch SDPA math (exact formula used on CUDA FlashAttn-2 split)
    nP, H, P, d = 6, 2, 16, 16
    q = torch.randn(nP, H, P, d)
    k = torch.randn(nP, H, P, d)
    v = torch.randn(nP, H, P, d)
    ms = torch.zeros(nP, P, dtype=torch.bool)
    ms[0:2, :] = True        # full
    ms[2:4, :7] = True       # partial
    # ms[4:6] all False      # empty
    km = ms.reshape(nP, 1, 1, P)
    empty = ~km.any(-1, keepdim=True)
    km_ref = km | (empty & (torch.arange(P) == 0))
    o_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=km_ref)

    valid_cnt = ms.sum(dim=-1)
    full_m, empty_m = valid_cnt == P, valid_cnt == 0
    part_m = ~(full_m | empty_m)
    o_split = torch.empty_like(v)
    o_split[full_m] = F.scaled_dot_product_attention(q[full_m], k[full_m], v[full_m])
    o_split[part_m] = F.scaled_dot_product_attention(q[part_m], k[part_m], v[part_m], attn_mask=km[part_m])
    o_split[empty_m] = v[empty_m, :, 0:1, :].expand_as(v[empty_m])
    assert (o_ref - o_split).abs().max().item() < 1e-6


def test_shape_encoder_gat_and_edge_head_parity():
    from openstroke.features import N_FEATURES
    from openstroke.models.openstroke2 import EdgeHead, GraphAttention, ShapeEncoder

    torch.manual_seed(5)
    # 1) ShapeEncoder chunked (>4096) vs unchunked reference
    enc = ShapeEncoder(N_FEATURES, 16, 32).eval()
    x_n = torch.randn(1, 4500, N_FEATURES)
    pts = torch.randn(1, 4500, 16, 2)
    with torch.enable_grad():
        # unchunked path runs whenever grad is enabled
        ref_se = enc(x_n, pts).detach()
    with torch.inference_mode():
        opt_se = enc(x_n, pts)
    assert (ref_se - opt_se).abs().max().item() < 1e-5

    # 2) GraphAttention undirected+fused-QKV (both training grad & CPU chunked >32768!) vs naive bidir reference
    ga = GraphAttention(32, heads=4, n_kinds=8, geo_dim=6).eval()
    with torch.no_grad():
        torch.nn.init.normal_(ga.out.weight, std=0.05)
        torch.nn.init.normal_(ga.ffn[-1].weight, std=0.05)

    def ga_naive(mod, h, mask0, i0, j0, kind0, geo0):
        B, N, C = h.shape
        kind = kind0.clamp(0, mod.kind_b.num_embeddings - 1)
        i, j = torch.cat([i0, j0]), torch.cat([j0, i0])
        kind, geo = torch.cat([kind, kind]), torch.cat([geo0, geo0])
        flat = mod.norm(h).reshape(B * N, C)
        q = mod.q(flat).view(-1, mod.h, mod.dh)
        k = mod.k(flat).view(-1, mod.h, mod.dh)
        v = mod.v(flat)
        s = (q[i] * k[j]).sum(-1).float() / math.sqrt(mod.dh) + (mod.kind_b(kind) + mod.geo_b(geo)).float()
        smax = torch.full((B * N, mod.h), -1e4, dtype=s.dtype).scatter_reduce(
            0, i[:, None].expand(-1, mod.h), s, reduce="amax", include_self=True)
        a = torch.exp(s - smax[i])
        den = torch.zeros(B * N, mod.h, dtype=a.dtype).index_add_(0, i, a)
        a = a / den[i].clamp(min=1e-9)
        val = (v[j] + mod.kind_v(kind).to(v.dtype) + mod.geo_v(geo).to(v.dtype)).view(-1, mod.h, mod.dh)
        agg = torch.zeros(B * N, mod.h, mod.dh, dtype=val.dtype).index_add_(0, i, val * a[..., None].to(val.dtype))
        m = mask0[..., None].to(h.dtype)
        h = h + mod.out(agg.reshape(B * N, C)).reshape(B, N, C).to(h.dtype) * m
        return h + mod.ffn(mod.norm2(h)).to(h.dtype) * m

    # Training forward + backward parity
    h1 = torch.randn(2, 128, 32, requires_grad=True)
    h2 = h1.detach().clone().requires_grad_(True)
    mask0 = torch.ones(2, 128, dtype=torch.bool)
    mask0[1, 100:] = False
    i0 = torch.randint(0, 200, (600,))
    j0 = torch.randint(0, 200, (600,))
    k0 = torch.randint(0, 8, (600,))
    geo0 = torch.randn(600, 6)
    ga_ref = copy.deepcopy(ga)
    y_ref = ga_naive(ga_ref, h1, mask0, i0, j0, k0, geo0)
    y_opt = ga(h2, {"mask0": mask0}, None, (i0, j0, k0), geo=geo0)
    assert (y_ref - y_opt).abs().max().item() < 1e-5
    y_ref.square().mean().backward()
    y_opt.square().mean().backward()
    assert (h1.grad - h2.grad).abs().max().item() < 1e-5
    for (n1, p1), (n2, p2) in zip(ga_ref.named_parameters(), ga.named_parameters()):
        assert (p1.grad - p2.grad).abs().max().item() < 1e-5, f"GAT grad mismatch {n1}"

    # Large CPU chunked inference parity (E > 32768 for GAT, E > 16384 for EdgeHead)
    h_big = torch.randn(1, 2000, 32)
    m_big = torch.ones(1, 2000, dtype=torch.bool)
    ib = torch.randint(0, 2000, (36000,))
    jb = torch.randint(0, 2000, (36000,))
    kb = torch.randint(0, 8, (36000,))
    gb = torch.randn(36000, 6)
    with torch.inference_mode():
        y_gb_ref = ga_naive(ga, h_big, m_big, ib, jb, kb, gb)
        y_gb_opt = ga(h_big, {"mask0": m_big}, None, (ib, jb, kb), geo=gb)
    assert (y_gb_ref - y_gb_opt).abs().max().item() < 1e-4

    eh = EdgeHead(32, n_kinds=8, geo_dim=6).eval()
    with torch.enable_grad():
        y_eh_ref = eh(h_big, {"mask0": m_big}, None, (ib, jb, kb), geo=gb).detach()
    with torch.inference_mode():
        y_eh_opt = eh(h_big, {"mask0": m_big}, None, (ib, jb, kb), geo=gb)
    assert (y_eh_ref - y_eh_opt).abs().max().item() < 1e-5


def test_sync_free_losses_forward_and_backward_parity():
    import torch.nn.functional as F

    from openstroke.models.group_head import GroupClsHead, _edge_cluster_stats, group_cls_loss
    from openstroke.models.openstroke2 import EDGE_TB_WEIGHT, closure_loss, edge_loss, lovasz_sigmoid, lovasz_sigmoid_batch

    torch.manual_seed(11)
    # 1) lovasz_sigmoid_batch vs per-item lovasz_sigmoid (mixed masks, all-zero label row, empty row!)
    B, N = 4, 64
    z1 = torch.randn(B, N, requires_grad=True)
    z2 = z1.detach().clone().requires_grad_(True)
    y = (torch.rand(B, N) > 0.7).float()
    y[2] = 0.0  # all-negative row
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[0, 50:] = False
    mask[1, 33:] = False
    mask[3, :] = False  # completely empty row!

    lv_ref = torch.stack([lovasz_sigmoid(z1[i][mask[i]], y[i][mask[i]]) for i in range(B)]).mean()
    lv_opt = lovasz_sigmoid_batch(z2, y, mask)
    assert (lv_ref - lv_opt).abs().item() < 1e-6
    lv_ref.backward()
    lv_opt.backward()
    assert (z1.grad - z2.grad).abs().max().item() < 1e-6

    # 2) edge_loss (pairs=True & False) against explicit boolean-indexed reference
    E = 200
    ez1 = torch.randn(E, requires_grad=True)
    ez2 = ez1.detach().clone().requires_grad_(True)
    i = torch.randint(0, B * N, (E,))
    j = torch.randint(0, B * N, (E,))
    oinst = torch.where(y > 0, torch.randint(0, 6, (B, N)), torch.full((B, N), -1))
    lw = torch.ones(B, N)
    lw[0, :10] = 0.0
    b = {"label": y, "oinst": oinst, "lw": lw, "mask0": mask}

    for pairs in (False, True):
        ez1.grad = ez2.grad = None
        # Reference
        yf = (y > 0).reshape(-1)
        of = oinst.reshape(-1)
        keep = (lw.reshape(-1) > 0)[i] & (lw.reshape(-1) > 0)[j]
        zk, ik, jk = ez1[keep], i[keep], j[keep]
        t = (yf[ik] & yf[jk] & (of[ik] == of[jk]) & (of[ik] >= 0)).float()
        bce = F.binary_cross_entropy_with_logits(zk, t, reduction="none")
        if not pairs:
            w = torch.where(yf[ik] | yf[jk], 1.0, 0.1)
            l_ref = (bce * w).sum() / w.sum().clamp(min=1e-6)
        else:
            tt = yf[ik] & yf[jk]
            l_ref = zk.sum() * 0.0
            for sel, wt in ((tt & (t > 0), 1.0), (tt & (t == 0), 1.0), (yf[ik] ^ yf[jk], EDGE_TB_WEIGHT)):
                if bool(sel.any()):
                    l_ref = l_ref + wt * bce[sel].mean()
        l_opt = edge_loss((ez2, i, j), b, pairs=pairs)
        assert (l_ref - l_opt).abs().item() < 1e-6
        l_ref.backward()
        l_opt.backward()
        assert (ez1.grad - ez2.grad).abs().max().item() < 1e-6

    # 3) closure_loss edge case (single-stroke objects only -> exact 0) & normal case
    c_loss = closure_loss(z2, b)
    assert torch.isfinite(c_loss)
    b_single = {**b, "oinst": torch.arange(B * N).reshape(B, N)}
    assert float(closure_loss(z2, b_single).detach()) == 0.0

    # 4) _edge_cluster_stats parity
    cid_full = torch.tensor([-1, 0, 0, 1, 1, 2, -1, 2])
    fg_m = cid_full >= 0
    ei = torch.tensor([0, 1, 1, 2, 3, 5])
    ej = torch.tensor([1, 2, 3, 4, 5, 7])
    ep = torch.tensor([0.9, 0.8, 0.2, 0.7, 0.6, 0.4])
    ext, int_m = _edge_cluster_stats(cid_full, 3, ei, ej, ep, fg_m)
    assert torch.allclose(int_m, torch.tensor([0.8, 0.5, 0.4]))
    assert torch.allclose(ext, torch.tensor([2.0, 3.0, 1.0]))

    # 5) group_cls_loss smoke & grad
    gh = GroupClsHead(16)
    h = torch.randn(B, N, 16, requires_grad=True)
    b_gh = {
        **b,
        "xy0": torch.randn(B, N, 2) * 50.0,
        "length": torch.rand(B, N) * 20.0 + 1.0,
        "x": torch.randn(B, N, 47),
        "ocls": torch.where(y > 0, torch.randint(1, 7, (B, N)), torch.zeros(B, N, dtype=torch.long)),
    }
    gl = group_cls_loss(gh, h, z2.detach(), (ez2.detach(), i, j), b_gh, torch.ones(8))
    assert torch.isfinite(gl)
    gl.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def _first_pos_collate(xs):
    return [int(x["pos"][0]) for x in xs]


def test_data_features_and_persistent_workers_parity():
    from openstroke.data import DIR_COS, DIR_SIN, FILL, HAS_FILL, HAS_STROKE, LOG_WIDTH, STROKE, Page, WindowDataset, _rot90_arrays, _unique_rgb4
    from openstroke.features import N_FEATURES, shape_points

    rng = np.random.default_rng(123)
    # 1) shape_points fast 1-seg path vs reference general searchsorted
    segs = rng.uniform(-100, 100, (120, 4)).astype(np.float64)
    segs[5, 2:] = segs[5, :2]  # zero-length 1-seg shape!
    # Mix of 0-seg, 1-seg, 2-seg, 5-seg shapes
    counts = [0, 1, 1, 1, 2, 1, 5, 0, 1, 3, 1, 1]
    rem = 120 - sum(counts)
    counts += [1] * rem
    offsets = np.concatenate([[0], np.cumsum(counts)])

    def shape_points_ref(segments, off, p=16):
        n = len(off) - 1
        out = np.zeros((n, p, 2), np.float32)
        seg = segments.astype(np.float64)
        sl = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
        cum = np.concatenate([[0.0], np.cumsum(sl)])
        cnt = np.diff(off)
        has = cnt > 0
        start, end = off[:-1], off[1:]
        L = cum[end] - cum[start]
        t = (np.arange(p) + 0.5) / p
        target = cum[start][:, None] + t[None, :] * L[:, None]
        si = np.clip(np.searchsorted(cum, target, side="right") - 1, start[:, None], np.maximum(end - 1, start)[:, None])
        si = np.clip(si, 0, len(seg) - 1)
        local = np.clip(np.where(sl[si] > 0, (target - cum[si]) / np.maximum(sl[si], 1e-12), 0.0), 0.0, 1.0)
        pts = np.stack([seg[si, 0] + (seg[si, 2] - seg[si, 0]) * local, seg[si, 1] + (seg[si, 3] - seg[si, 1]) * local], axis=-1)
        starts_h = start[has]
        x0 = np.minimum.reduceat(np.minimum(seg[:, 0], seg[:, 2]), starts_h)
        y0 = np.minimum.reduceat(np.minimum(seg[:, 1], seg[:, 3]), starts_h)
        x1 = np.maximum.reduceat(np.maximum(seg[:, 0], seg[:, 2]), starts_h)
        y1 = np.maximum.reduceat(np.maximum(seg[:, 1], seg[:, 3]), starts_h)
        centre = np.stack([(x0 + x1) * 0.5, (y0 + y1) * 0.5], axis=1)
        out[has] = (pts[has] - centre[:, None, :]).astype(np.float32)
        return out

    assert np.allclose(shape_points(segs, offsets), shape_points_ref(segs, offsets), atol=1e-5)

    # 2) _unique_rgb4 exact match
    seen = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0, 0.12345], size=(500, 3)).astype(np.float32)
    u0, i0 = np.unique(np.round(seen, 4), axis=0, return_inverse=True)
    u1, i1 = _unique_rgb4(seen)
    assert np.allclose(u0, u1, atol=1e-6) and np.array_equal(i0.reshape(-1), i1)

    # 3) _rot90_arrays exact match for k=0..3
    for k in range(4):
        xy0 = rng.normal(size=(40, 2))
        pts0 = rng.normal(size=(40, 16, 2)).astype(np.float32)
        x0 = rng.normal(size=(40, N_FEATURES)).astype(np.float32)
        oc0 = rng.normal(size=(40, 2))
        xy1, pts1, x1, oc1 = xy0.copy(), pts0.copy(), x0.copy(), oc0.copy()
        for _ in range(k):
            xy0[:] = np.stack([-xy0[:, 1], xy0[:, 0]], axis=1)
            oc0[:] = np.stack([-oc0[:, 1], oc0[:, 0]], axis=1)
            pts0[:] = np.stack([-pts0[..., 1], pts0[..., 0]], axis=-1)
            x0[:, DIR_COS] = -x0[:, DIR_COS]
            x0[:, DIR_SIN] = -x0[:, DIR_SIN]
        _rot90_arrays(k, xy1, pts1, x1, oc=oc1)
        assert np.allclose(xy0, xy1) and np.allclose(pts0, pts1) and np.allclose(x0, x1) and np.allclose(oc0, oc1)

    # 4) WindowDataset shared-memory epoch across persistent DataLoader workers
    p = Page(
        stem="s0", mode="hvac",
        x=np.zeros((200, N_FEATURES), np.float32),
        xy=rng.uniform(0, 100, (200, 2)).astype(np.float32),
        pts=np.zeros((200, 16, 2), np.float16),
        layer=np.zeros(200, np.int64),
        label=np.zeros(200, np.uint8),
        length=np.ones(200, np.float32),
        index=np.arange(200, dtype=np.int64),
    )
    ds = WindowDataset([p], n_items=4, window_size=32, aug=None, seed=9)
    dl = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=2, persistent_workers=True, collate_fn=_first_pos_collate)
    ds.epoch = 0
    e0_w = [v for batch in dl for v in batch]
    ds.epoch = 1
    e1_w = [v for batch in dl for v in batch]
    # Exact match to single-process sequential WindowDataset!
    ds.epoch = 0
    e0_ref = [int(ds[i]["pos"][0]) for i in range(4)]
    ds.epoch = 1
    e1_ref = [int(ds[i]["pos"][0]) for i in range(4)]
    assert e0_w == e0_ref and e1_w == e1_ref and e0_w != e1_w


def test_graph_topo_all_modes_and_kcore_exact_parity():
    from openstroke.data import HAS_FILL, HAS_STROKE, LOG_WIDTH, STROKE
    from openstroke.features import FEATURE_NAMES, N_FEATURES
    from openstroke.models.openstroke2 import (
        _graph_topo_features,
        _graph_topo_fire_features,
        _graph_topo_uni_features,
        _graph_topo_v2_features,
        _graph_topo_v3_features,
        _kcore2_and_peel,
    )

    rng = np.random.default_rng(77)
    B, N = 2, 160
    x = np.zeros((B, N, N_FEATURES), dtype=np.float32)
    x[..., HAS_STROKE] = 1.0
    x[..., STROKE] = rng.choice([0.0, 0.5, 1.0], size=(B, N, 3)).astype(np.float32)
    x[..., LOG_WIDTH] = rng.choice([0.0, np.log1p(2.4), np.log1p(4.8)], size=(B, N)).astype(np.float32)
    x[..., HAS_FILL] = (rng.random((B, N)) > 0.85).astype(np.float32)
    x[..., FEATURE_NAMES.index("elong")] = rng.uniform(0.05, 1.0, (B, N)).astype(np.float32)
    x[..., FEATURE_NAMES.index("frac_c")] = (rng.random((B, N)) > 0.8).astype(np.float32)
    x[..., FEATURE_NAMES.index("closed_ends")] = (rng.random((B, N)) > 0.7).astype(np.float32)
    x[..., FEATURE_NAMES.index("log_nseg")] = np.log1p(rng.integers(1, 6, (B, N))).astype(np.float32)
    x[..., FEATURE_NAMES.index("log_bbox_w")] = np.log1p(rng.uniform(0.5, 25.0, (B, N))).astype(np.float32)
    x[..., FEATURE_NAMES.index("log_bbox_h")] = np.log1p(rng.uniform(0.5, 25.0, (B, N))).astype(np.float32)
    x[..., FEATURE_NAMES.index("log_bbox_diag")] = np.log1p(rng.uniform(0.5, 30.0, (B, N))).astype(np.float32)
    xy = rng.uniform(0, 500, (B, N, 2)).astype(np.float32)
    length = rng.uniform(2, 60, (B, N)).astype(np.float32)
    mask = np.ones((B, N), dtype=bool)
    mask[1, 130:] = False
    # Build edges with loops, chains, duplicates, self-loops
    e0 = np.vstack([
        np.column_stack([np.arange(0, 40), np.arange(1, 41)]),
        np.array([[40, 0], [0, 1], [1, 0], [5, 5]]),  # loop + dup + self-loop
        np.column_stack([np.arange(50, 90), np.arange(51, 91)]),
    ])
    e1 = np.column_stack([np.arange(0, 60), np.arange(1, 61)])
    E = max(len(e0), len(e1))
    edges = np.zeros((B, E, 2), dtype=np.int64)
    emask = np.zeros((B, E), dtype=bool)
    edges[0, :len(e0)] = e0; emask[0, :len(e0)] = True
    edges[1, :len(e1)] = e1; emask[1, :len(e1)] = True
    b = {
        "x": torch.from_numpy(x),
        "xy0": torch.from_numpy(xy),
        "length": torch.from_numpy(length),
        "mask0": torch.from_numpy(mask),
        "edges": torch.from_numpy(edges),
        "emask": torch.from_numpy(emask),
    }

    t1 = _graph_topo_features(b)
    t2 = _graph_topo_v2_features(b)
    t3 = _graph_topo_v3_features(b)
    tf = _graph_topo_fire_features(b)
    tu = _graph_topo_uni_features(b)
    assert t1.shape == (B, N, 5) and torch.isfinite(t1).all()
    assert t2.shape == (B, N, 7) and torch.isfinite(t2).all()
    assert t3.shape == (B, N, 12) and torch.isfinite(t3).all()
    assert tf.shape == (B, N, 12) and torch.isfinite(tf).all()
    assert tu.shape == (B, N, 18) and torch.isfinite(tu).all()
    # Uni[0..11] == v3[0..11] and Uni[12..17] == fire[6..11] bit-exact!
    assert torch.allclose(tu[..., :12], t3, atol=1e-6)
    assert torch.allclose(tu[..., 12:], tf[..., 6:], atol=1e-6)
    # Monochrome fast-path (e_all == e_sw)
    b_mono = dict(b, x=b["x"].clone())
    b_mono["x"][..., STROKE] = 0.0
    b_mono["x"][..., LOG_WIDTH] = float(np.log1p(4.8))
    tu_m = _graph_topo_uni_features(b_mono)
    assert torch.isfinite(tu_m).all()
    assert torch.equal(tu_m[..., 13], tu_m[..., 14])  # in_2core_all == in_2core_sw when all edges match style!

    # Reference BFS peeling vs fast CSR _kcore2_and_peel
    xy_k = np.array([[0, 0], [100, 0], [100, 100], [0, 100], [140, 0], [180, 0], [220, 0]], dtype=np.float32)
    ek = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [1, 4], [4, 5], [5, 6], [1, 0], [2, 2]], dtype=np.int64)
    c2, sp2, peel = _kcore2_and_peel(7, ek, xy_k)
    assert c2[:4].all() and not c2[4:].any()
    assert np.allclose(sp2[:4], 100.0) and np.allclose(sp2[4:], 0.0)
    assert peel[0] == 1.0 and 0 < peel[6] < peel[5] < peel[4] < 1.0


def test_real_checkpoint_and_openstroke46_parity(tmp_path):
    from openstroke.features import FEATURE_NAMES, N_FEATURES
    from openstroke.infer import Predictor, _load_checkpoint_into_model
    from openstroke.models import build
    from openstroke.models.image_branch import fuse_image_branch

    # Synthetic CAD payload (duct rectangle + elbow + background)
    pay = {
        "shapes": [
            {"items": [["l", [0, 0], [120, 0]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0"},
            {"items": [["l", [0, 20], [120, 20]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0"},
            {"items": [["l", [120, 0], [120, 20]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0"},
            {"items": [["l", [0, 20], [0, 0]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0"},
            {"items": [["l", [200, 100], [215, 100]]], "type": "s", "color": [0.5, 0.5, 0.5], "width": 0.24, "layer": "0"},
            {"items": [["l", [215, 100], [215, 115]]], "type": "s", "color": [0.5, 0.5, 0.5], "width": 0.24, "layer": "0"},
        ],
        "document_metadata": {"page_width": 400.0, "page_height": 300.0},
    }

    # Verify unfused NCHW vs fused+CL Predictor parity end-to-end (prob, edges, GroupClsHead) on openstroke45 & openstroke46
    for arch in ("openstroke45", "openstroke46"):
        torch.manual_seed(17 if arch == "openstroke45" else 23)
        m = build(arch, N_FEATURES, dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32,
                  img_pretrained=False, img_max_px=128, edge_head=True, edge_pairs=True, group_head=True).eval()
        with torch.no_grad():
            for p in m.parameters():
                if p.ndim >= 2 and float(p.abs().sum()) == 0.0:
                    torch.nn.init.normal_(p, std=0.05)
        ckpt = tmp_path / f"{arch}.pt"
        torch.save({
            "model": m.state_dict(), "config": dict(m.config),
            "feature_names": list(FEATURE_NAMES), "window": 256, "threshold": 0.0, "tta": 1,
        }, ckpt)

        p_unfused = Predictor(ckpt, device="cpu", threads=1)
        m_raw, _ = _load_checkpoint_into_model(ckpt, torch.device("cpu"), fuse=False)
        p_unfused.model = m_raw
        p_unfused.threshold = 0.0
        p_fused = Predictor(ckpt, device="cpu", threads=1)
        p_fused.threshold = 0.0

        r0 = p_unfused.predict_payload(pay)
        r1 = p_fused.predict_payload(pay)
        assert np.abs(r0["prob"] - r1["prob"]).max() < 1e-4
        assert np.array_equal(r0["edges"][0], r1["edges"][0])
        assert np.abs(r0["edges"][1] - r1["edges"][1]).max() < 1e-4
        assert np.array_equal(r0["group_objects"]["instance_id"], r1["group_objects"]["instance_id"])
        assert len(r0["group_objects"]["cluster_probs"]) > 0
        assert np.abs(r0["group_objects"]["cluster_probs"] - r1["group_objects"]["cluster_probs"]).max() < 1e-4

    # OpenStroke46 direct forward unfused vs fused+CL parity
    m46_ref = build("openstroke46", N_FEATURES, dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32, img_pretrained=False, img_max_px=128).eval()
    with torch.no_grad():
        for p in m46_ref.parameters():
            if p.ndim >= 2 and float(p.abs().sum()) == 0.0:
                torch.nn.init.normal_(p, std=0.05)
    m46_fus = copy.deepcopy(m46_ref)
    fuse_image_branch(m46_fus, channels_last=True)
    from openstroke.data import collate, page_from_arrays, window
    from openstroke.infer import payload_arrays

    arr, _ = payload_arrays(pay, mode="raw")
    pg = page_from_arrays(arr, "p", mode="raw", graph=True, no_layers=True)
    b = collate([window(pg, 0, 64)])
    with torch.inference_mode():
        y0 = m46_ref(b)
        y1 = m46_fus(b)
    assert (y0 - y1).abs().max().item() < 1e-4

    # Optional real checkpoint parity when path is provided via environment variable
    env_ckpt = os.environ.get("OPENSTROKE_PARITY_CKPT", "").strip()
    if env_ckpt and Path(env_ckpt).exists():
        p0 = Predictor(env_ckpt, device="cpu", threads=1)
        m_raw, _ = _load_checkpoint_into_model(env_ckpt, torch.device("cpu"), fuse=False)
        p0.model = m_raw
        p0.threshold = 0.0
        p1 = Predictor(env_ckpt, device="cpu", threads=1)
        p1.threshold = 0.0
        r0 = p0.predict_payload(pay)
        r1 = p1.predict_payload(pay)
        assert np.abs(r0["prob"] - r1["prob"]).max() < 1e-4
        assert np.array_equal(r0["edges"][0], r1["edges"][0])
        assert np.abs(r0["edges"][1] - r1["edges"][1]).max() < 1e-4
        assert np.array_equal(r0["group_objects"]["instance_id"], r1["group_objects"]["instance_id"])


def test_supplement_fg_edges_facing_and_cpu_runtime_parity():
    from openstroke.data import page_from_arrays
    from openstroke.features import N_FEATURES
    from openstroke.infer import _configure_cpu_runtime, payload_arrays
    from openstroke.models import build
    from openstroke.models.group_head import supplement_fg_edges

    _configure_cpu_runtime()  # idempotent & safe on Windows/Linux

    # Parallel duct walls + multi-segment shapes to exercise facing_edges=True vectorized supplement_fg_edges
    pay = {
        "shapes": [
            {"items": [["l", [0, 0], [80, 0]], ["l", [80, 0], [160, 0]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0"},
            {"items": [["l", [0, 24], [160, 24]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0"},
            {"items": [["l", [200, 0], [200, 10]]], "type": "s", "color": [0.5, 0.5, 0.5], "width": 0.2, "layer": "0"},
            {"items": [["l", [0, 60], [160, 60]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0"},
            {"items": [["l", [0, 84], [160, 84]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0"},
        ],
        "document_metadata": {"page_width": 300.0, "page_height": 200.0},
    }
    arr, info = payload_arrays(pay, mode="raw")
    pg = page_from_arrays(arr, "p", mode="raw", graph="facing", no_layers=True)
    m = build("openstroke46", N_FEATURES, dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32,
              img_pretrained=False, img_max_px=64, facing_edges=True, edge_head=True).eval()
    h_page = np.random.default_rng(7).normal(size=(len(pg), 16)).astype(np.float16)
    p_page = np.array([0.9, 0.9, 0.9, 0.1, 0.9], dtype=np.float32)

    ep, el = supplement_fg_edges(
        m, h_page, pg, p_page, 0.5,
        np.zeros((0, 2), np.int64), np.zeros(0, np.float32),
        info["segments"], len(pay["shapes"]), torch.device("cpu"),
    )
    assert len(ep) >= 2 and len(ep) == len(el) and np.isfinite(el).all()
    # Verify vectorized fg segment slice + length matches original per-shape loop bit-exact
    segs = info["segments"]
    fg_mask = np.zeros(len(pay["shapes"]), dtype=bool)
    fg_mask[pg.index[p_page >= 0.5]] = True
    fg_ids = np.flatnonzero(fg_mask)
    off_all = segs.shape_offsets
    lens_ref = np.array([
        float(np.hypot(segs.seg[off_all[s]:off_all[s + 1], 2] - segs.seg[off_all[s]:off_all[s + 1], 0],
                       segs.seg[off_all[s]:off_all[s + 1], 3] - segs.seg[off_all[s]:off_all[s + 1], 1]).sum())
        for s in fg_ids
    ], dtype=np.float64)
    segs_ref = np.vstack([segs.seg[off_all[s]:off_all[s + 1]] for s in fg_ids])
    sm = fg_mask[segs.owner]
    counts = off_all[fg_ids + 1] - off_all[fg_ids]
    fg_segs = segs.seg[sm]
    seg_lens = np.hypot(fg_segs[:, 2] - fg_segs[:, 0], fg_segs[:, 3] - fg_segs[:, 1]).astype(np.float64)
    lens_vec = np.bincount(np.repeat(np.arange(len(fg_ids), dtype=np.int64), counts), weights=seg_lens, minlength=len(fg_ids))
    assert np.array_equal(segs_ref, fg_segs)
    assert np.allclose(lens_ref, lens_vec, atol=1e-12)
