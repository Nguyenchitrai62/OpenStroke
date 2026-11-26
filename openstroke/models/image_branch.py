"""E28: the image the vectors draw, read by a pretrained CNN at the shapes' own points (docs: research/HYPOTHESES.md E28).

A person reads a drawing by looking at it (a pipe run with its sprinklers and size labels, a dimension line with its ticks, a title
block) and then picks the exact stroke. Here every window is drawn into an image from its own tokens: the polyline through each shape's
16 arc-length samples, black ink on white, in the window's own frame (so flips / quarter turns / scale of the vector augmentation
apply to the image for free, and image and vectors line up exactly). A ResNet18 (torchvision, BSD; ImageNet weights, frozen
BatchNorm) gives stride-8 and stride-16 features; each shape gets the mean feature at its 16 points, projected (zero-init) and added
to its token. No PyMuPDF, no separate render step.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FPN_DIM = 128


def _raster_coords(xy0: torch.Tensor, pts: torch.Tensor, mask: torch.Tensor, ppt: float = 1.0, max_px: int = 1536,
                   max_samples: int = 4096, *, abs_pts: torch.Tensor | None = None):
    """Compute shared rasterized pixel indices once for one or more per-shape value channels."""
    B, N, P, _ = pts.shape
    a = abs_pts if abs_pts is not None else (xy0[:, :, None, :] + pts).float()  # (B, N, P, 2) absolute, window frame
    m = mask[:, :, None].expand(B, N, P)
    ext = torch.where(m[..., None], a.abs(), a.new_zeros(())).flatten(1).amax(1).clamp(min=1.0)  # (B,) half extent, pt
    scale = torch.minimum(torch.full_like(ext, ppt), (max_px / 2 - 2) / ext)
    S = int(min(max_px, math.ceil(float((2 * ext * scale).max()) / 32 + 1) * 32))
    px = a * scale[:, None, None, None] + S / 2  # (B, N, P, 2) pixel coords (x right, y down)
    p0, p1 = px[:, :, :-1], px[:, :, 1:]
    seg_ok = mask[:, :, None].expand(B, N, P - 1).reshape(B, -1)
    p0, p1 = p0.reshape(B, -1, 2), p1.reshape(B, -1, 2)
    length = (p1 - p0).norm(dim=-1)
    k = (length.ceil().clamp(min=1, max=max_samples)).long() * seg_ok  # samples per segment (0 for padding)
    bi = torch.arange(B, device=pts.device)[:, None].expand_as(k)
    k, bi, q0, q1 = k.reshape(-1), bi.reshape(-1), p0.reshape(-1, 2), p1.reshape(-1, 2)
    keep = k > 0
    k, bi, q0, q1 = k[keep], bi[keep], q0[keep], q1[keep]
    if len(k) == 0:
        return None, None, keep, scale, S, (B, N, P)
    seg = torch.repeat_interleave(torch.arange(len(k), device=pts.device), k)
    start = torch.cumsum(k, 0) - k
    t = (torch.arange(len(seg), device=pts.device) - start[seg]).float() / k[seg].clamp(min=1).float()
    p = q0[seg] + (q1[seg] - q0[seg]) * t[:, None]
    u = p[:, 0].round().long().clamp(0, S - 1)
    v = p[:, 1].round().long().clamp(0, S - 1)
    pix = bi[seg] * (S * S) + v * S + u
    return pix, seg, keep, scale, S, (B, N, P)


def _scatter_channel(pix, seg, keep, S: int, shape_bnp: tuple[int, int, int], device, val: torch.Tensor | None,
                     binary: bool = False) -> torch.Tensor:
    B, N, P = shape_bnp
    img = torch.zeros(B * S * S, device=device, dtype=torch.float32)
    if pix is None:
        return img.view(B, 1, S, S)
    if val is None:
        img.index_fill_(0, pix, 1.0)
    elif binary:
        sv = (val[:, :, None].expand(B, N, P - 1).reshape(-1)[keep] > 0.5)
        if bool(sv.any()):
            img.index_fill_(0, pix[sv[seg]], 1.0)
    else:
        sv = val.float()[:, :, None].expand(B, N, P - 1).reshape(-1)[keep]
        img.scatter_reduce_(0, pix, sv[seg], reduce="amax")  # the darkest / thickest stroke wins a pixel
    return img.view(B, 1, S, S)


def render(xy0: torch.Tensor, pts: torch.Tensor, mask: torch.Tensor, ppt: float = 1.0, max_px: int = 1536,
           max_samples: int = 4096, ink: torch.Tensor | None = None, *, abs_pts: torch.Tensor | None = None):
    """Ink image of each window. ``xy0`` (B, N, 2) shape centres, ``pts`` (B, N, P, 2) points relative to them, both in pt in the
    window frame. ``ink`` (B, N): darkness of each shape in [0, 1] (E30: gray strokes are drawn gray), default 1 (black).
    -> (img (B, 1, S, S) ink in [0, 1], scale (B,) px per pt). One square size S for the batch (multiple of 32);
    each window keeps ``ppt`` px / pt unless it would not fit in ``max_px``."""
    pix, seg, keep, scale, S, bnp = _raster_coords(xy0, pts, mask, ppt, max_px, max_samples, abs_pts=abs_pts)
    return _scatter_channel(pix, seg, keep, S, bnp, pts.device, ink), scale, S


def render_multi(xy0: torch.Tensor, pts: torch.Tensor, mask: torch.Tensor, channels: list[tuple[torch.Tensor | None, bool]],
                 ppt: float = 1.0, max_px: int = 1536, max_samples: int = 4096, *, abs_pts: torch.Tensor | None = None):
    """Rasterize multiple per-shape value channels sharing ONE pixel-coordinate pass (exact parity with repeated ``render``,
    ~2.5-3x faster for OpenStroke44/45 ``img_cad=True``)."""
    pix, seg, keep, scale, S, bnp = _raster_coords(xy0, pts, mask, ppt, max_px, max_samples, abs_pts=abs_pts)
    maps = [_scatter_channel(pix, seg, keep, S, bnp, pts.device, val, binary=is_bin) for val, is_bin in channels]
    return maps, scale, S


def fuse_conv_frozen_bn(conv: nn.Conv2d, bn: nn.Module) -> nn.Conv2d:
    """Algebraically fold a FrozenBatchNorm2d (eval affine scale + shift) into its preceding Conv2d (exact parity)."""
    with torch.no_grad():
        scale = bn.weight * torch.rsqrt(bn.running_var + bn.eps)
        bias = bn.bias - bn.running_mean * scale
        fused = nn.Conv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
            conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode,
            device=conv.weight.device, dtype=conv.weight.dtype,
        )
        fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
        if conv.bias is not None:
            fused.bias.copy_(conv.bias * scale + bias)
        else:
            fused.bias.copy_(bias)
    return fused


def fuse_image_branch(model: nn.Module, channels_last: bool = False) -> nn.Module:
    """Inference optimization (exact parity, as OpenOBB ``fuse_model``): fold all Conv2d + FrozenBatchNorm2d pairs
    inside ``ImageBranch`` (ResNet18) into fused Conv2d ops and optionally switch 2D conv weights to ``channels_last``."""
    from torchvision.ops.misc import FrozenBatchNorm2d

    img = getattr(model, "img", None)
    if img is None:
        return model
    if not getattr(img, "_fused", False) and img.backbone_name != "hgnetv2_b5":
        # Stem: [conv1, bn1, relu, maxpool, layer1]
        if isinstance(img.stem[0], nn.Conv2d) and isinstance(img.stem[1], FrozenBatchNorm2d):
            img.stem[0] = fuse_conv_frozen_bn(img.stem[0], img.stem[1])
            img.stem[1] = nn.Identity()
        for stage in (img.stem[4], img.layer2, img.layer3, img.layer4):
            if stage is None:
                continue
            for blk in stage:
                if hasattr(blk, "conv1") and isinstance(blk.bn1, FrozenBatchNorm2d):
                    blk.conv1 = fuse_conv_frozen_bn(blk.conv1, blk.bn1)
                    blk.bn1 = nn.Identity()
                if hasattr(blk, "conv2") and isinstance(blk.bn2, FrozenBatchNorm2d):
                    blk.conv2 = fuse_conv_frozen_bn(blk.conv2, blk.bn2)
                    blk.bn2 = nn.Identity()
                ds = getattr(blk, "downsample", None)
                if ds is not None and len(ds) >= 2 and isinstance(ds[0], nn.Conv2d) and isinstance(ds[1], FrozenBatchNorm2d):
                    ds[0] = fuse_conv_frozen_bn(ds[0], ds[1])
                    ds[1] = nn.Identity()
        img._fused = True
    if channels_last:
        img._channels_last = True
        for name in ("stem", "layer2", "layer3", "layer4", "hg", "lat", "smooth", "cad_stem"):
            sub = getattr(img, name, None)
            if sub is not None:
                sub.to(memory_format=torch.channels_last)
    return model


def _sample_mean(feat: torch.Tensor, grid: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """Chunked ``F.grid_sample(feat, grid).mean(-1).transpose(1, 2)``: identical output, 25x lower peak memory on 65k windows."""
    n = grid.shape[1]
    if n <= chunk:
        return F.grid_sample(feat, grid, align_corners=False).mean(-1).transpose(1, 2)
    return torch.cat(
        [F.grid_sample(feat, grid[:, s:s + chunk], align_corners=False).mean(-1).transpose(1, 2)
         for s in range(0, n, chunk)],
        dim=1,
    )


class StrokeContext(nn.Module):
    """Read FPN context along a shape before pooling it to one token."""

    def __init__(self, dim: int):
        super().__init__()
        channels, hidden = 12, 24
        self.reduce = nn.Conv2d(FPN_DIM, channels, 1)
        self.encode = nn.Sequential(nn.Linear(7 * channels + 2, hidden), nn.GELU(),
                                    nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Linear(2 * hidden, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feature, xy, points, mask, scale, size):
        field = self.reduce(feature).float()
        if points.shape[2] < 2 or points.shape[1] == 0:
            return field.new_zeros((*mask.shape, self.out.out_features))

        # Use symmetric segment positions so reversing a shape only swaps sample order and the sides.
        segments = points.shape[2] - 1
        left = (round((segments - 1) / 8), round(3 * (segments - 1) / 8))
        indices = torch.tensor((*left, segments - 1 - left[1], segments - 1 - left[0]),
                               device=points.device)
        offset = field.new_tensor((0, -4, 4, -12, 12, -32, 32))
        near_end = (indices.to(field.dtype) + 0.5) / segments
        near_end = torch.minimum(near_end, 1 - near_end)
        outputs = []
        for start in range(0, points.shape[1], 8192):
            end = start + 8192
            pts = points[:, start:end].float()
            a, b = pts[:, :, indices], pts[:, :, indices + 1]
            tangent = b - a
            length = tangent.norm(dim=-1, keepdim=True)
            normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1) / length.clamp_min(1e-6)
            center = xy[:, start:end, None].float() + (a + b) * 0.5
            grid = center[..., None, :] + normal[..., None, :] * offset[:, None]
            grid = (grid * (scale / (size / 2))[:, None, None, None, None]).flatten(2, 3)
            hits = F.grid_sample(field, grid, align_corners=False)
            # (B, shapes, positions, center/left/right, channels).
            hits = hits.permute(0, 2, 3, 1).unflatten(2, (len(indices), len(offset)))
            lhs, rhs = hits[..., 1::2, :], hits[..., 2::2, :]
            position = near_end.view(1, 1, -1, 1).expand_as(length)
            stats = torch.cat((hits[..., 0, :], ((lhs + rhs) * 0.5).flatten(-2),
                               (lhs - rhs).abs().flatten(-2), position, length.log1p()), dim=-1)
            tokens = self.encode(stats)
            valid = mask[:, start:end, None, None] & (length > 1e-6)
            mean = tokens.masked_fill(~valid, 0).sum(2) / valid.sum(2).clamp_min(1)
            peak = tokens.masked_fill(~valid, torch.finfo(tokens.dtype).min).amax(2)
            present = valid.any(2)
            peak = peak.masked_fill(~present, 0)
            outputs.append(self.out(torch.cat((mean.to(tokens.dtype), peak), dim=-1)) * present)
        return torch.cat(outputs, dim=1)


def obb_backbone_state(path) -> dict:
    """HGNetv2 weights for the image branch: a backbone state dict (E48 ``hgnetv2_b5_wall_color.pt``) or a whole OpenOBB checkpoint
    (``{"model": state, ...}``, OpenStroke38: an HVAC detector's backbone), whose ``backbone.*`` entries are taken."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:   # an OpenOBB training checkpoint also pickles its args
        state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        m = state["model"]
        state = m.state_dict() if hasattr(m, "state_dict") else m
        state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
        if not state:
            raise ValueError(f"{path}: an OpenOBB checkpoint without backbone.* weights")
    return {k: v.float() for k, v in state.items() if not k.endswith("num_batches_tracked")}


class CADStem(nn.Module):
    """OpenStroke44 (H170): lightweight zero-init 2-channel CAD attribute CNN stem operating on pixel-aligned
    ``[norm_log_width, has_fill]`` rasters at stride 1 -> 2 -> 4, fused directly into the FPN stride-4 map ``p``.
    Preserves exact Step-0 parity with pretrained ResNet18-FPN checkpoints (``E145``) while exposing line-weight
    hierarchy (e.g. 0.48 pt ducts vs 0.24 pt furniture) and filled symbols (arrowheads, solid bars)."""

    def __init__(self, out_dim: int = FPN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_dim, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cad2: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        y = self.net(cad2)
        if y.shape[-2:] != target_hw:
            y = F.interpolate(y, size=target_hw, mode="bilinear", align_corners=False)
        return y


class ImageBranch(nn.Module):
    def __init__(self, dim: int, pretrained: bool = True, ppt: float = 1.0, max_px: int = 1536, lum: bool = False,
                 fpn: bool = False, aux: bool = False, backbone: str = "resnet18", weights: str | None = None,
                 stroke_context: bool = False, img_cad: bool = False):
        super().__init__()
        from torchvision.ops.misc import FrozenBatchNorm2d

        #: E48: HGNetv2 (OpenOBB's backbone; ``weights`` = a state dict from a floorplan-trained OpenOBB checkpoint)
        self.backbone_name = backbone
        if backbone == "hgnetv2_b5":
            from .hgnetv2 import HGNetv2

            net = HGNetv2("B5", use_lab=False, return_idx=[0, 1, 2, 3], freeze_at=-1, freeze_norm=True, pretrained=False)
            if weights:
                net.load_state_dict(obb_backbone_state(weights))
            self.hg = net
            chans = (128, 512, 1024, 2048)
        else:
            from torchvision.models import ResNet18_Weights, resnet18

            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None, norm_layer=FrozenBatchNorm2d)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)  # stride 4
            self.layer2, self.layer3 = net.layer2, net.layer3  # stride 8 (128 ch), 16 (256 ch)
            chans = (64, 128, 256, 512)
        self.proj = nn.Linear(chans[1] + chans[2], dim)
        nn.init.zeros_(self.proj.weight)  # starts as the model without an image
        nn.init.zeros_(self.proj.bias)
        self.ppt, self.max_px, self.lum = ppt, max_px, lum
        #: E37: layer4 + FPN to stride 4, one fused map read at every stroke point
        self.layer4 = net.layer4 if fpn and backbone != "hgnetv2_b5" else None
        self.use_fpn = fpn
        if stroke_context and not fpn:
            raise ValueError("stroke_context requires img_fpn")
        if img_cad and not fpn:
            raise ValueError("img_cad requires img_fpn")
        self.stroke_context = StrokeContext(dim) if stroke_context else None
        self.cad_stem = CADStem(FPN_DIM) if img_cad else None
        self.lat = nn.ModuleList(nn.Conv2d(c, FPN_DIM, 1) for c in chans) if fpn else None
        self.smooth = nn.Sequential(nn.Conv2d(FPN_DIM, FPN_DIM, 3, padding=1), nn.GELU()) if fpn else None
        self.proj_fpn = nn.Linear(FPN_DIM, dim) if fpn else None
        if fpn:
            nn.init.zeros_(self.proj_fpn.weight)
            nn.init.zeros_(self.proj_fpn.bias)
        #: E38: dense "pipe ink" logits on the stride-4 map, trained on the rasterised labels (training only)
        assert fpn or not aux, "img_aux needs img_fpn"
        self.aux_head = (nn.Sequential(nn.Conv2d(FPN_DIM, FPN_DIM // 2, 3, padding=1), nn.GELU(), nn.Conv2d(FPN_DIM // 2, 1, 1))
                         if aux else None)
        #: (logits, target, valid) of the last training forward, each (B, 1, S/4, S/4); read and cleared by the trainer
        self.last_aux = None
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.register_buffer("lum_w", torch.tensor([0.299, 0.587, 0.114]), persistent=False)

    def forward(self, b: dict) -> torch.Tensor:
        mask = b["mask0"] > 0
        drawn = mask & ~b["mae"] if "mae" in b else mask  # masked pretraining: the hidden shapes are not drawn either
        dark = None
        xf = b["x"].float() if (self.lum or self.cad_stem is not None) else None
        if self.lum:  # E30: each shape drawn with its own darkness (stroke colour, else fill colour), black = 1
            from ..data import FILL, HAS_STROKE, STROKE

            rgb = torch.where(xf[..., HAS_STROKE:HAS_STROKE + 1] > 0.5, xf[..., STROKE], xf[..., FILL])
            dark = (1.0 - (rgb * self.lum_w).sum(-1)).clamp(0.05, 1.0)
        abs_pts = (b["xy0"][:, :, None, :] + b["pts"]).float()
        need_aux = self.use_fpn and self.aux_head is not None and self.training and "label" in b and "mae" not in b
        if self.use_fpn and self.cad_stem is not None:
            from ..data import HAS_FILL, HAS_STROKE, LOG_WIDTH

            w_val = (xf[..., LOG_WIDTH] / 3.0).clamp(0.0, 1.0) * (xf[..., HAS_STROKE] > 0.5).float()
            f_val = xf[..., HAS_FILL]
            chans = [(dark, False), (w_val, False), (f_val, True)]
            if need_aux:
                chans.append((b["label"].float(), False))
            maps, scale, S = render_multi(b["xy0"], b["pts"], drawn, chans, self.ppt, self.max_px, abs_pts=abs_pts)
            ink, w_map, f_map = maps[0], maps[1], maps[2]
            lab_map = maps[3] if need_aux else None
        elif need_aux:
            maps, scale, S = render_multi(b["xy0"], b["pts"], drawn, [(dark, False), (b["label"].float(), False)],
                                          self.ppt, self.max_px, abs_pts=abs_pts)
            ink, lab_map = maps[0], maps[1]
            w_map = f_map = None
        else:
            ink, scale, S = render(b["xy0"], b["pts"], drawn, self.ppt, self.max_px, ink=dark, abs_pts=abs_pts)
            w_map = f_map = lab_map = None
        x = (1.0 - ink).expand(-1, 3, -1, -1)  # black ink on white
        use_cl = bool(getattr(self, "_channels_last", False)) and x.device.type == "cuda"
        f4 = None
        if self.backbone_name == "hgnetv2_b5":  # as OpenOBB: [0, 1], no mean / std
            if use_cl:
                x = x.contiguous(memory_format=torch.channels_last)
            f1, f2, f3, f4 = self.hg(x)
        else:
            x = (x - self.mean) / self.std  # ImageNet-normalised
            if use_cl:
                x = x.contiguous(memory_format=torch.channels_last)
            f1 = self.stem(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
        grid = (abs_pts * (scale / (S / 2))[:, None, None, None]).clamp(-1, 1)  # (B, N, P, 2) in [-1, 1], x then y as grid_sample wants
        g = torch.cat([_sample_mean(f2.float(), grid), _sample_mean(f3.float(), grid)], dim=-1)  # (B, N, 384)
        out = self.proj(g)
        if self.use_fpn:
            p = self.lat[3](f4 if f4 is not None else self.layer4(f3))
            for lat, f in zip(self.lat[2::-1], (f3, f2, f1)):  # top-down: stride 32 -> 16 -> 8 -> 4
                p = lat(f) + F.interpolate(p, size=f.shape[-2:], mode="nearest")
            p = self.smooth(p)
            if self.cad_stem is not None:
                cad2 = torch.cat([w_map, f_map], dim=1).to(p.dtype)
                if use_cl:
                    cad2 = cad2.contiguous(memory_format=torch.channels_last)
                p = p + self.cad_stem(cad2, p.shape[-2:])
            gp = _sample_mean(p.float(), grid)  # (B, N, FPN_DIM)
            out = out + self.proj_fpn(gp)
            if self.stroke_context is not None:
                out = out + self.stroke_context(p, b["xy0"], b["pts"], mask, scale, S).to(out.dtype)
            self.last_aux = None
            if need_aux and lab_map is not None:
                k = S // p.shape[-1]
                self.last_aux = (self.aux_head(p), F.max_pool2d(lab_map, k), F.max_pool2d((ink > 0).float(), k))
        return out * mask[..., None]


def aux_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, pos_weight: float = 2.0) -> torch.Tensor:
    """E38: BCE of the dense head on inked pixels only (blank paper is trivially "other"), pipe pixels weighted like true shapes."""

    w = valid * torch.where(target > 0.5, pos_weight, 1.0)
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    return (bce * w).sum() / w.sum().clamp(min=1.0)
