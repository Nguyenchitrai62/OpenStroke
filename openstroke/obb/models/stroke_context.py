"""OpenStroke20: deep, image-aware context for the stroke tokens of OpenOBB25 (docs/OPENSTROKE20.md).

E69's stroke tokens come from a 2-layer, 128-d transformer that sees only geometry (0.57M parameters, 1.1 % of the model) and
the model under-fits even its training pages (object F2 0.63). This module stacks wider attention blocks on those tokens,
each stroke also reading the P3 image features at its 8 sample points. Attention runs inside windows of `window` strokes
along a Morton order of the stroke centres, alternating x/y-first orders and half-window shifts (serialized attention, idea
of Point Transformer V3; own code). The output projection starts at zero: tokens, boxes and memberships equal E69's.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .stroke_instances import serialized_order


class DeepStrokeContext(nn.Module):
    def __init__(self, d, img_channels, width=256, blocks=6, window=128, heads=8):
        super().__init__()
        self.window = int(window)
        self.inp = nn.Linear(d, width)
        self.img = nn.Linear(img_channels, width)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation="gelu",
                                                                batch_first=True, norm_first=True) for _ in range(blocks)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, tok, vec, mask, img):
        """tok (B, M, d), vec (B, M, >=17) points in tile units, mask (B, M), img (B, C, H, W) = P3 -> tok + residual."""
        b, m, _ = tok.shape
        if m == 0 or not self.blocks:
            return tok
        pts = vec[..., 1:17].float().reshape(b, m, 8, 2)
        grid = (pts * 2 - 1).reshape(b, m * 8, 1, 2).to(img.dtype)
        sample = F.grid_sample(img, grid, align_corners=False).squeeze(-1).reshape(b, -1, m, 8).mean(-1).transpose(1, 2)
        h = self.inp(tok) + self.img(sample.to(tok.dtype))
        centre = pts.mean(2)
        out = torch.zeros_like(h)
        for i in range(b):
            rows = mask[i].nonzero(as_tuple=True)[0]
            n = len(rows)
            if n == 0:
                continue
            x = h[i, rows]
            orders = [serialized_order(centre[i, rows], False), serialized_order(centre[i, rows], True)]
            for k, block in enumerate(self.blocks):
                order = orders[k % 2]
                shift = (self.window // 2) if (k // 2) % 2 else 0
                if shift and n > self.window:
                    order = torch.roll(order, shifts=shift)
                pad = (-n) % self.window
                packed = F.pad(x[order], (0, 0, 0, pad)).reshape(-1, self.window, x.shape[-1])
                valid = (torch.arange(n + pad, device=x.device) < n).reshape(-1, self.window)
                y = block(packed, src_key_padding_mask=~valid).reshape(-1, x.shape[-1])[:n]
                x = torch.empty_like(y).index_copy_(0, order, y)
            out[i, rows] = x
        return tok + self.out(self.norm(out)).to(tok.dtype)
