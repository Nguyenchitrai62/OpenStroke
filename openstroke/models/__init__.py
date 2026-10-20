"""Model registry: ``model=openstroke1 | openstroke2`` (same batch dict in, one logit per token out)."""

from __future__ import annotations

import torch.nn as nn

MODELS = ("openstroke1", "openstroke2")


class OpenStroke1Batch(nn.Module):
    """v1 (kNN attention, docs/OPENSTROKE1.md) behind the batch-dict interface of the engine."""

    def __init__(self, n_features: int, dim: int = 128, depth: int = 4, heads: int = 4, k: int = 16,
                 layer_pool: bool = True, **_):
        super().__init__()
        from ..model import OpenStroke1

        self.net = OpenStroke1(n_features, dim=dim, depth=depth, heads=heads, k=k, layer_pool=layer_pool)
        self.config = {"arch": "openstroke1", **self.net.config}

    def set_normalization(self, mean, std):
        self.net.set_normalization(mean, std)

    def forward(self, b):
        return self.net(b["x"], b["xy0"], b["layer"], b["mask0"])


def build(name: str, n_features: int, **cfg) -> nn.Module:
    if name == "openstroke1":
        keep = {k: cfg[k] for k in ("dim", "depth", "heads", "k", "layer_pool") if k in cfg}
        return OpenStroke1Batch(n_features, **keep)
    if name == "openstroke2":
        from .openstroke2 import OpenStroke2

        keep = {k: cfg[k] for k in ("n_points", "dims", "enc_depths", "dec_depths", "head_dim", "patch", "drop_path",
                                    "layer_pool", "shuffle_orders", "mode_input") if k in cfg}
        m = OpenStroke2(n_features, **keep)
        m.config = {"arch": "openstroke2", **m.config}
        return m
    raise ValueError(f"model={name!r}: expected one of {MODELS}")


def from_config(config: dict) -> nn.Module:
    cfg = dict(config)
    name = cfg.pop("arch", "openstroke1")  # checkpoints of the first trainer (v1) had no "arch"
    n = cfg.pop("n_features")
    return build(name, n, **cfg)


def load_weights(model: nn.Module, state: dict) -> None:
    """Load a state dict, accepting v1 checkpoints of the first trainer (keys without the ``net.`` prefix)."""

    if isinstance(model, OpenStroke1Batch) and not any(k.startswith("net.") for k in state):
        model.net.load_state_dict(state)
    else:
        model.load_state_dict(state)


def loss(name: str, logits, batch, pos_weight: float, alpha: float, lovasz: float):
    from .openstroke2 import loss_v2

    return loss_v2(logits, batch, pos_weight=pos_weight, alpha=alpha, lovasz=lovasz)
