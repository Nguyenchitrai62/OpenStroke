"""Frozen 3976da0 reference for loss/gradient/AdamW equivalence checks only."""
import torch
from torch.nn import functional as F
from openstroke.obb.models.openobb25 import INST_UNKNOWN, members_from_boxes

def legacy_member_loss(self, out, targets, size):
        side = out["side"]
        tok, vec, mask = side["tok"], side["vec"], side["mask"]
        picks = self.member_targets(out, targets, size)
        bce = dice = tok.sum() * 0.0
        n = 0
        cand = [(b, p) for b, p in enumerate(picks) if p is not None and bool(mask[b].any())]
        if not cand:
            return {"member_bce": bce, "member_dice": dice}
        K = max(len(p[0]) for _, p in cand)
        boxes = vec.new_zeros((len(cand), K, 5))
        boxes[..., 2:4] = 1.0
        for i, (b, (fb, _)) in enumerate(cand):
            boxes[i, :len(fb)] = fb
        bi = torch.tensor([b for b, _ in cand], device=vec.device)
        h = self.box_hidden(out["f4"][bi], out["feats"][0][bi], boxes)
        for i, (b, (fb, j)) in enumerate(cand):
            labels = targets[b]["labels"].to(vec.device)[j]
            lg, gate, rows = self._members_of(h[i, :len(fb)], labels, fb, tok[b], vec[b], mask[b], size,
                                               out["f4"][b])
            inst = vec[b, rows, 22] if vec.shape[-1] > 22 else torch.full_like(rows, INST_UNKNOWN, dtype=vec.dtype)
            if bool((inst == INST_UNKNOWN).all()):                   # no object ids: geometric membership
                gt = targets[b]["boxes"].float().to(vec.device)[j]
                tgt = members_from_boxes(vec[b, rows, 1:17].float().view(-1, 8, 2), gt).float()
            else:
                tgt = (inst.round().long()[None, :] == j[:, None]).float()
            g = gate.float()
            cnt = g.sum(1)
            keep = cnt > 0
            if not bool(keep.any()):
                continue
            lb = F.binary_cross_entropy_with_logits(lg, tgt, reduction="none") * g
            bce = bce + (lb.sum(1)[keep] / cnt[keep]).sum()
            pr = lg.sigmoid() * g
            tg = tgt * g
            dice = dice + (1 - (2 * (pr * tg).sum(1) + 1) / (pr.sum(1) + tg.sum(1) + 1))[keep].sum()
            n += int(keep.sum())
        n = max(n, 1)
        return {"member_bce": bce / n, "member_dice": dice / n}
