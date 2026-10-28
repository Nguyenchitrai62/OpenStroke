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


def render(xy0: torch.Tensor, pts: torch.Tensor, mask: torch.Tensor, ppt: float = 1.0, max_px: int = 1536,
           max_samples: int = 4096, ink: torch.Tensor | None = None):
    """Ink image of each window. ``xy0`` (B, N, 2) shape centres, ``pts`` (B, N, P, 2) points relative to them, both in pt in the
    window frame. ``ink`` (B, N): darkness of each shape in [0, 1] (E30: gray strokes are drawn gray), default 1 (black).
    -> (img (B, 1, S, S) ink in [0, 1], scale (B,) px per pt). One square size S for the batch (multiple of 32);
    each window keeps ``ppt`` px / pt unless it would not fit in ``max_px``."""

    B, N, P, _ = pts.shape
    a = (xy0[:, :, None, :] + pts).float()  # (B, N, P, 2) absolute, window frame
    m = mask[:, :, None].expand(B, N, P)
    ext = torch.where(m[..., None], a.abs(), torch.zeros_like(a)).flatten(1).amax(1).clamp(min=1.0)  # (B,) half extent, pt
    scale = torch.minimum(torch.full_like(ext, ppt), (max_px / 2 - 2) / ext)
    S = int(min(max_px, math.ceil(float((2 * ext * scale).max()) / 32 + 1) * 32))
    px = a * scale[:, None, None, None] + S / 2  # (B, N, P, 2) pixel coords (x right, y down)
    p0, p1 = px[:, :, :-1], px[:, :, 1:]
    seg_ok = (mask[:, :, None] & torch.ones_like(p0[..., 0], dtype=torch.bool)).reshape(B, -1)
    p0, p1 = p0.reshape(B, -1, 2), p1.reshape(B, -1, 2)
    length = (p1 - p0).norm(dim=-1)
    k = (length.ceil().clamp(min=1, max=max_samples)).long() * seg_ok  # samples per segment (0 for padding)
    img = torch.zeros(B * S * S, device=pts.device, dtype=torch.float32)
    bi = torch.arange(B, device=pts.device)[:, None].expand_as(k)
    val = (ink.float() if ink is not None else torch.ones(B, N, device=pts.device))[:, :, None].expand(B, N, P - 1)
    k, bi, q0, q1, val = k.reshape(-1), bi.reshape(-1), p0.reshape(-1, 2), p1.reshape(-1, 2), val.reshape(-1)
    keep = k > 0
    k, bi, q0, q1, val = k[keep], bi[keep], q0[keep], q1[keep], val[keep]
    seg = torch.repeat_interleave(torch.arange(len(k), device=pts.device), k)
    start = torch.cumsum(k, 0) - k
    t = (torch.arange(len(seg), device=pts.device) - start[seg]).float() / k[seg].clamp(min=1).float()
    p = q0[seg] + (q1[seg] - q0[seg]) * t[:, None]
    u = p[:, 0].round().long().clamp(0, S - 1)
    v = p[:, 1].round().long().clamp(0, S - 1)
    img.scatter_reduce_(0, bi[seg] * S * S + v * S + u, val[seg], reduce="amax")  # the darkest stroke wins a pixel
    return img.view(B, 1, S, S), scale, S


class ImageBranch(nn.Module):
    def __init__(self, dim: int, pretrained: bool = True, ppt: float = 1.0, max_px: int = 1536, lum: bool = False,
                 fpn: bool = False, aux: bool = False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        from torchvision.ops.misc import FrozenBatchNorm2d

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None, norm_layer=FrozenBatchNorm2d)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)  # stride 4
        self.layer2, self.layer3 = net.layer2, net.layer3  # stride 8 (128 ch), 16 (256 ch)
        self.proj = nn.Linear(128 + 256, dim)
        nn.init.zeros_(self.proj.weight)  # starts as the model without an image
        nn.init.zeros_(self.proj.bias)
        self.ppt, self.max_px, self.lum = ppt, max_px, lum
        #: E37: layer4 + FPN to stride 4, one fused map read at every stroke point
        self.layer4 = net.layer4 if fpn else None
        self.lat = nn.ModuleList(nn.Conv2d(c, FPN_DIM, 1) for c in (64, 128, 256, 512)) if fpn else None
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

    def forward(self, b: dict) -> torch.Tensor:
        mask = b["mask0"] > 0
        drawn = mask & ~b["mae"] if "mae" in b else mask  # masked pretraining: the hidden shapes are not drawn either
        dark = None
        if self.lum:  # E30: each shape drawn with its own darkness (stroke colour, else fill colour), black = 1
            from ..data import FILL, HAS_STROKE, STROKE

            x = b["x"].float()
            rgb = torch.where(x[..., HAS_STROKE:HAS_STROKE + 1] > 0.5, x[..., STROKE], x[..., FILL])
            w = torch.tensor([0.299, 0.587, 0.114], device=x.device)
            dark = (1.0 - (rgb * w).sum(-1)).clamp(0.05, 1.0)
        ink, scale, S = render(b["xy0"], b["pts"], drawn, self.ppt, self.max_px, ink=dark)
        x = ((1.0 - ink).expand(-1, 3, -1, -1) - self.mean) / self.std  # black ink on white, ImageNet-normalised
        f1 = self.stem(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        a = (b["xy0"][:, :, None, :] + b["pts"]).float() * scale[:, None, None, None]  # pixel offset from the image centre
        grid = (a / (S / 2)).clamp(-1, 1)  # (B, N, P, 2) in [-1, 1], x then y as grid_sample wants
        g2 = F.grid_sample(f2.float(), grid, align_corners=False)  # (B, 128, N, P)
        g3 = F.grid_sample(f3.float(), grid, align_corners=False)  # (B, 256, N, P)
        g = torch.cat([g2, g3], 1).mean(-1).transpose(1, 2)  # (B, N, 384)
        out = self.proj(g)
        if self.layer4 is not None:
            p = self.lat[3](self.layer4(f3))
            for lat, f in zip(self.lat[2::-1], (f3, f2, f1)):  # top-down: stride 32 -> 16 -> 8 -> 4
                p = lat(f) + F.interpolate(p, size=f.shape[-2:], mode="nearest")
            p = self.smooth(p)
            gp = F.grid_sample(p.float(), grid, align_corners=False).mean(-1).transpose(1, 2)  # (B, N, FPN_DIM)
            out = out + self.proj_fpn(gp)
            self.last_aux = None
            if self.aux_head is not None and self.training and "label" in b and "mae" not in b:
                lab, _, _ = render(b["xy0"], b["pts"], drawn, self.ppt, self.max_px, ink=b["label"].float())
                k = S // p.shape[-1]
                self.last_aux = (self.aux_head(p), F.max_pool2d(lab, k), F.max_pool2d((ink > 0).float(), k))
        return out * mask[..., None]


def aux_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, pos_weight: float = 2.0) -> torch.Tensor:
    """E38: BCE of the dense head on inked pixels only (blank paper is trivially "other"), pipe pixels weighted like true shapes."""

    w = valid * torch.where(target > 0.5, pos_weight, 1.0)
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    return (bce * w).sum() / w.sum().clamp(min=1.0)
