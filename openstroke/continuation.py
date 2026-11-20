"""One-hop confidence refinement for the frozen E114 stroke classifier."""
import torch
from torch import nn


ANCHOR_PROB = 0.20
MAX_SHIFT = 6.0


class ContinuationHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 24))
        self.kind = nn.Embedding(16, 8)
        self.message = nn.Sequential(nn.Linear(62, 32), nn.GELU(), nn.Linear(32, 16))
        self.update = nn.Sequential(nn.Linear(58, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, tokens, logit, links, geometry):
        # Base scores always survive. Connect the empty case to the trainable parameters.
        if links is None:
            return logit + self.update[-1].weight.sum().to(logit.dtype) * 0
        a, b, kind = links
        receiver, sender = torch.cat((a, b)), torch.cat((b, a))
        prob = logit.detach().float().reshape(-1).sigmoid()
        supported = prob[sender] >= ANCHOR_PROB
        if not supported.any():
            return logit + self.update[-1].weight.sum().to(logit.dtype) * 0
        receiver, sender = receiver[supported], sender[supported]
        ids, group = torch.unique(receiver, return_inverse=True)
        h = self.project(tokens.flatten(0, 1))
        edge = self.kind(torch.cat((kind, kind))[supported])
        pair = torch.cat((h[receiver], h[sender], edge, torch.cat((geometry, geometry))[supported].to(h.dtype),
                          prob[receiver, None].to(h.dtype), prob[sender, None].to(h.dtype)), dim=1)
        msg = self.message(pair).float()
        mass = msg.new_zeros((len(ids),)).index_add_(0, group, torch.ones_like(group, dtype=msg.dtype))
        mean = msg.new_zeros((len(ids), 16)).index_add_(0, group, msg) / mass[:, None]
        peak = msg.new_full((len(ids), 16), -torch.inf).scatter_reduce_(
            0, group[:, None].expand_as(msg), msg, reduce="amax")
        local = torch.cat((h[ids].float(), mean, peak, prob[ids, None], mass.log1p()[:, None]), dim=1)
        shift = MAX_SHIFT * self.update(local).squeeze(1).float().tanh()
        return logit.reshape(-1).float().index_add(0, ids, shift).reshape_as(logit)
