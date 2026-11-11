"""OpenStroke17: residual refinement of OpenOBB25 (E69) membership logits (docs/OPENSTROKE17.md).

E69 scores every (box, stroke) pair on its own: one box never sees the other boxes of the tile, so overlapping boxes claim the
same strokes and a duct cut by its box loses the strokes outside. The refiner lets the boxes of a tile attend to each other
(self-attention) and to the strokes near them (cross-attention restricted to each box's gate), and adds a correction to E69's
logits. Its output layer starts at zero, so a model built from E69 returns exactly E69's memberships before training.
"""
import torch
from torch import nn


class MemberRefiner(nn.Module):
    def __init__(self, d_box, d_tok, num_classes, d=128, layers=2, heads=4):
        super().__init__()
        self.q = nn.Linear(d_box, d)
        self.cls = nn.Embedding(max(num_classes, 1), d)
        self.base = nn.Linear(2, d)                       # mean / max of the box's own E69 probabilities over its gate
        self.t = nn.Linear(d_tok, d)
        self.null = nn.Parameter(torch.zeros(1, 1, d))    # always-visible memory slot: a box with an empty gate attends to it
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d, heads, 2 * d, dropout=0.0, activation="gelu",
                                                                batch_first=True, norm_first=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.embed = nn.Linear(d, d)
        self.scale = d ** -0.5
        nn.init.zeros_(self.embed.weight)
        nn.init.zeros_(self.embed.bias)

    def forward(self, h, labels, base, gate, tok):
        """One image: h (K, d_box) box features, labels (K,), base (K, M) E69 logits, gate (K, M) bool, tok (M, d_tok)
        -> refined logits (K, M) float32 (-30 outside the gate, as E69)."""
        if base.numel() == 0:
            return base
        k, m = base.shape
        p = base.detach().float().sigmoid() * gate
        cnt = gate.float().sum(1, keepdim=True).clamp_min(1)
        stats = torch.cat([p.sum(1, keepdim=True) / cnt, p.amax(1, keepdim=True)], 1)
        q = self.q(h.float()) + self.cls(labels.long().clamp(0, self.cls.num_embeddings - 1)) + self.base(stats)
        mem = self.t(tok.float())
        memory = torch.cat([self.null[0], mem], 0)[None]
        block = torch.cat([gate.new_zeros(k, 1), ~gate], 1)
        block = block.repeat(self.layers[0].multihead_attn.num_heads, 1, 1) if self.layers else block
        x = q[None]
        for layer in self.layers:
            x = layer(x, memory, memory_mask=block)
        delta = (self.embed(self.norm(x[0])) @ mem.t()) * self.scale
        return (base.float() + delta.float()).masked_fill(~gate, -30.0)
