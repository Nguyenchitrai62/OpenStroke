"""Exponential moving average of weights (YOLO ``ModelEMA``, as in OpenOBB ``engine.py``)."""

from __future__ import annotations

import copy
import math

import torch


class ModelEMA:
    def __init__(self, model, decay: float = 0.9995, warmups: int = 1000):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.warmups, self.updates = decay, warmups, 0

    @torch.no_grad()
    def update(self, model):
        # e*d + m*(1-d) with mul/add (not lerp): lerp turns inf into NaN (OpenOBB, 2026-10-08)
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmups))
        msd = model.state_dict()
        fe, fm = [], []
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                fe.append(v)
                fm.append(msd[k].detach())
            else:
                v.copy_(msd[k])
        torch._foreach_mul_(fe, d)
        torch._foreach_add_(fe, fm, alpha=1 - d)

    @torch.no_grad()
    def repair(self, model, log=print):
        msd = model.state_dict()
        bad = [k for k, v in self.module.state_dict().items() if v.dtype.is_floating_point and not torch.isfinite(v).all()]
        if bad:
            log(f"WARNING: {len(bad)} EMA tensors are NaN/inf (e.g. {bad[:3]}); re-synced from the training model")
            sd = self.module.state_dict()
            for k in bad:
                sd[k].copy_(torch.nan_to_num(msd[k], nan=0.0, posinf=0.0, neginf=0.0))
        return bad

    def state_dict(self):
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd["module"])
        self.updates = sd["updates"]
