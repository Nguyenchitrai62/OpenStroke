"""Training-only ranking for same-style errors made by the input checkpoint."""

import torch
from torch.nn import functional as F

from .compare_stroke_checkpoints import THRESHOLD


LOSS_WEIGHT = 0.10
MARGIN = 1.0
MIN_F2_GAIN = 0.01
MAX_RECALL_DROP = 0.01


def rank_loss(logits, batch):
    """Pair each missed pipe stroke with the highest-scored false positive in its style and window."""
    windows, n = logits.shape
    size = windows * n
    device = logits.device
    teacher = batch["pair_teacher"].flatten()
    label = batch["label"].flatten() > 0
    valid = (batch["mask0"] & (batch["length"] > 0) & (batch["lw"] > 0)).flatten() & (teacher >= 0)
    key = (batch["style"] + (torch.arange(windows, device=device) * n)[:, None]).flatten()
    negative = valid & ~label & (teacher >= THRESHOLD)
    missed = valid & label & (teacher < THRESHOLD)

    peak = teacher.new_full((size,), -1.0).scatter_reduce_(
        0, key, teacher.masked_fill(~negative, -1.0), reduce="amax"
    )
    ids = torch.arange(size, device=device)
    # In a tie, choose the earliest token deterministically.
    best = ids.new_full((size,), size).scatter_reduce_(
        0, key, ids.masked_fill(~negative | (teacher != peak[key]), size), reduce="amin"
    )
    other = best[key]
    paired = missed & (other < size)
    flat = logits.flatten().float()
    penalty = F.softplus(MARGIN + flat[other[paired]] - flat[paired])
    return penalty.sum() / max(1, len(penalty)), len(penalty)
