"""Model registry: ``model=openstroke1 | openstroke2 | openstroke3`` (same batch dict in, one logit per token out).
openstroke3 = the openstroke2 network with 2 RelationBlocks (docs/OPENSTROKE3.md)."""

from __future__ import annotations

import torch.nn as nn

MODELS = ("openstroke1", "openstroke2", "openstroke3", "openstroke4")
#: OpenStroke4 (docs/OPENSTROKE4.md): LitePT-style split. Level 0 (shapes) mixes along the link graph, level 1 (2 pt
#: cells) only by the serial conv, attention only at levels 2-4 where tokens are few.
OPENSTROKE4_MIXERS = ("graph", "conv", "attn", "attn", "attn")


def openstroke4_mixers(n_levels: int) -> tuple:
    """Graph at level 0, serial conv at level 1 when there are >= 4 levels, attention above (5 levels: the default)."""
    m = ["graph"] + (["conv"] if n_levels >= 4 else [])
    return tuple(m + ["attn"] * (n_levels - len(m)))


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
    if name in ("openstroke2", "openstroke3", "openstroke4"):
        from .openstroke2 import OpenStroke2

        if name == "openstroke3" and not cfg.get("relations"):
            cfg = {**cfg, "relations": 2}
        if name == "openstroke4" and not cfg.get("mixers"):
            cfg = {**cfg, "mixers": openstroke4_mixers(len(cfg.get("dims") or (64, 96, 128, 192, 256)))}

        keep = {k: cfg[k] for k in ("n_points", "dims", "enc_depths", "dec_depths", "head_dim", "patch", "drop_path",
                                    "layer_pool", "shuffle_orders", "mode_input", "layer_names", "group_context", "relations", "mixers",
                                    "feature_drop", "pretrain", "layer_head", "layer_head_names") if k in cfg}
        m = OpenStroke2(n_features, **keep)
        m.config = {"arch": name, **m.config}
        return m
    raise ValueError(f"model={name!r}: expected one of {MODELS}")


def from_config(config: dict) -> nn.Module:
    cfg = dict(config)
    name = cfg.pop("arch", "openstroke1")  # checkpoints of the first trainer (v1) had no "arch"
    n = cfg.pop("n_features")
    return build(name, n, **cfg)


def load_weights(model: nn.Module, state: dict, new_heads: bool = False) -> None:
    """Load a state dict, accepting v1 checkpoints of the first trainer (keys without the ``net.`` prefix).
    ``new_heads``: the model may have heads the checkpoint lacks (``layer_mlp``, ``lh_name*``: fine-tuning a pretrained
    or older checkpoint with a layer head); those keep their fresh init, every other key must match."""

    if isinstance(model, OpenStroke1Batch) and not any(k.startswith("net.") for k in state):
        model.net.load_state_dict(state)
        return
    own = model.state_dict()
    if "feat_keep" in own and "feat_keep" not in state:  # checkpoints from before feature_drop: every feature kept
        state = {**state, "feat_keep": own["feat_keep"]}
    if new_heads:
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad = [k for k in missing if not k.startswith(("layer_mlp.", "lh_name"))]
        if bad or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing {bad[:5]}, unexpected {unexpected[:5]}")
        return
    model.load_state_dict(state)


def loss(name: str, logits, batch, pos_weight: float, alpha: float, lovasz: float):
    from .openstroke2 import loss_v2

    return loss_v2(logits, batch, pos_weight=pos_weight, alpha=alpha, lovasz=lovasz)
