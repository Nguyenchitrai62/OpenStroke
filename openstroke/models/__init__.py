"""Model registry: ``model=openstroke1 | openstroke2 | openstroke3`` (same batch dict in, one logit per token out).
openstroke3 = the openstroke2 network with 2 RelationBlocks (docs/OPENSTROKE3.md)."""

from __future__ import annotations

import re

import torch.nn as nn

#: Research versions (docs/OPENSTROKE<N>.md), as OpenOBB names its networks: each is the OpenStroke2 backbone with fixed switches. The name is
#: what a checkpoint's ``config["arch"]`` says, so any version can be rebuilt, compared or rolled back to. Checkpoints saved before these names
#: (arch "openstroke2") are recognised by their switches (train.describe_arch).
ARCHS = {
    "openstroke28": dict(image_branch=True, img_lum=True, img_fpn=True, no_layers=True),                # E114: strokes, no layers, image branch + FPN
    "openstroke29": dict(image_branch=True, img_lum=True, img_fpn=True, no_layers=True, obj_head=True),  # E116 / E117: + per-stroke class head
    "openstroke30": dict(image_branch=True, img_lum=True, img_fpn=True, no_layers=True, rel_attn=2, edge_head=True),   # E120: + link-graph attention, link head
}
MODELS = ("openstroke1", "openstroke2", "openstroke3", "openstroke4", *ARCHS)


def canonical_arch(name) -> str:
    """A checkpoint's arch name -> this code's name: the research and the product (renamed) copies of the package save their own prefix,
    and a checkpoint of either loads in both. ``None`` (first trainer) -> openstroke1."""
    return re.sub(r"^(?:open|hicas)stroke", "openstroke", str(name or "openstroke1"))


def base_arch(name) -> str:
    """The network class a name builds: a research version (ARCHS) is the openstroke2 backbone with its switches."""
    name = canonical_arch(name)
    return "openstroke2" if name in ARCHS else name
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
    name = canonical_arch(name)
    if name == "openstroke1":
        keep = {k: cfg[k] for k in ("dim", "depth", "heads", "k", "layer_pool") if k in cfg}
        return OpenStroke1Batch(n_features, **keep)
    if name in ("openstroke2", "openstroke3", "openstroke4") or name in ARCHS:
        from .openstroke2 import OpenStroke2

        if name in ARCHS:   # a research version: its switches, unless the config says otherwise (a head added on top)
            cfg = {**ARCHS[name], **{k: v for k, v in cfg.items() if v is not None}}

        if name == "openstroke3" and not cfg.get("relations"):
            cfg = {**cfg, "relations": 2}
        if name == "openstroke4" and not cfg.get("mixers"):
            cfg = {**cfg, "mixers": openstroke4_mixers(len(cfg.get("dims") or (64, 96, 128, 192, 256)))}

        keep = {k: cfg[k] for k in ("n_points", "dims", "enc_depths", "dec_depths", "head_dim", "patch", "drop_path",
                                    "layer_pool", "shuffle_orders", "mode_input", "layer_names", "group_context", "relations", "mixers",
                                    "feature_drop", "pretrain", "layer_head", "layer_head_names", "image_branch", "img_pretrained", "img_lum",
                                    "pool_by", "img_ppt", "img_max_px", "img_fpn", "img_aux", "img_backbone", "img_weights", "obj_head", "rel_attn", "edge_head") if k in cfg}
        m = OpenStroke2(n_features, **keep)
        m.config = {"arch": name, **m.config}
        return m
    raise ValueError(f"model={name!r}: expected one of {MODELS}")


def from_config(config: dict, weights_follow: bool = False) -> nn.Module:
    """``weights_follow``: a full state dict is loaded right after, so pretrained sub-networks (E28 image branch) need not be
    downloaded."""
    cfg = dict(config)
    if weights_follow and cfg.get("image_branch"):
        cfg["img_pretrained"] = False
    name = cfg.pop("arch", "openstroke1")  # checkpoints of the first trainer (v1) had no "arch"
    n = cfg.pop("n_features")
    cfg.pop("no_layers", None)  # OpenStroke28: how pages are read, not a network option
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
        bad = [k for k in missing if not k.startswith(("layer_mlp.", "lh_name", "img.", "mode_embed.", "obj_mlp.", "gat.", "edge_mlp."))]
        if bad or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing {bad[:5]}, unexpected {unexpected[:5]}")
        return
    model.load_state_dict(state)


def loss(name: str, logits, batch, pos_weight: float, alpha: float, lovasz: float, poly: float = 0.0):
    from .openstroke2 import loss_v2

    return loss_v2(logits, batch, pos_weight=pos_weight, alpha=alpha, lovasz=lovasz, poly=poly)
