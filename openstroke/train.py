"""``openstroke train``: cross-validation by PDF on the E0 dataset (YOLO / OpenOBB-style CLI).

    openstroke train data=/content/dataset_local/e0 model=openstroke2 epochs=100

Each validation fold trains on the other folds and writes ``runs/train/fold<k>/`` (``best.pt``, ``last.pt``,
``results.csv``, ``results.png``, ``result.json``, ``val_scores.npz``); the run writes ``train.log``,
``args.json`` and ``summary.json``. Model defaults (window, batch, lr0...) depend on ``model=``
(``DEFAULTS``); anything passed on the command line wins.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .features import FEATURE_VERSION, PageTokens, page_tokens
from .folds import load as load_folds
from .folds import resolve


def cached_tokens(data: Path, stem: str, cache: Path, mode: str) -> PageTokens:
    """Hand features only (tree baseline)."""

    f = Path(cache) / f"{stem}.{FEATURE_VERSION}.npz"
    if f.exists():
        z = np.load(f)
        return PageTokens(stem=stem, mode=mode, x=z["x"], xy=z["xy"], layer=z["layer"], label=z["label"],
                          length=z["length"], keep=z["keep"])
    t = page_tokens(Path(data) / f"{stem}.npz", mode)
    Path(cache).mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + f".{os.getpid()}.tmp.npz")  # atomic: runs sharing a cache must not read a half file
    np.savez(tmp, x=t.x, xy=t.xy, layer=t.layer, label=t.label, length=t.length, keep=t.keep)
    os.replace(tmp, f)
    return t


#: Per-model defaults (docs/OPENSTROKE1.md, docs/OPENSTROKE2.md §4, docs/OPENSTROKE3.md: why these values).
#: ``model=`` is required, as in YOLO: it names the architecture (or a run folder / .pt to start from).
DEFAULTS = {
    "openstroke1": dict(window=4096, batch=8, lr0=1e-3, epochs=30, lovasz=0.0, drop_path=0.0, val_batch=8,
                        restyle=0.0, relations=0),
    "openstroke2": dict(window=65536, batch=4, lr0=2e-3, epochs=100, lovasz=1.0, drop_path=0.3, val_batch=2,
                        restyle=0.5, relations=0),
    # v2 + message passing along the shape link graph (2 RelationBlocks); unstable when combined (LEDGER F25)
    "openstroke3": dict(window=65536, batch=4, lr0=2e-3, epochs=100, lovasz=1.0, drop_path=0.3, val_batch=2,
                        restyle=0.5, relations=2),
    # LitePT-style: graph mixing at level 0, serial conv at level 1, attention at levels 2-4 (docs/OPENSTROKE4.md)
    "openstroke4": dict(window=65536, batch=4, lr0=2e-3, epochs=100, lovasz=1.0, drop_path=0.3, val_batch=2,
                        restyle=0.5, relations=0),
}
#: ``model=<research version>`` (user 2026-11-17: one architecture name per research card, as OpenOBB): the network (models.ARCHS) and the
#: training recipe of that card's run. openstroke28 = E114 (docs/OPENSTROKE28.md), openstroke29 = E117 (OPENSTROKE29), openstroke30 = E120
#: (OPENSTROKE30), openstroke31 = E122 (OPENSTROKE31). Every checkpoint saves its version in ``config["arch"]``; older checkpoints are named by their switches (describe_arch).
from .models import ARCHS  # noqa: E402

_RECIPE = dict(no_layers=True, feature_drop="layer_stats", restyle_mode="order")
PRESETS = {
    "openstroke28": ("openstroke2", {**ARCHS["openstroke28"], **_RECIPE}),
    "openstroke29": ("openstroke2", {**ARCHS["openstroke29"], **_RECIPE}),
    "openstroke30": ("openstroke2", {**ARCHS["openstroke30"], **_RECIPE, "closure": 0.5}),
    "openstroke31": ("openstroke2", {**ARCHS["openstroke31"], **_RECIPE, "closure": 0.5}),
}
ALIASES = {"openstroke-v1": "openstroke28", "openstroke-v2": "openstroke30"}   # names used in earlier commands
#: Training recipe a checkpoint passes on when a run starts from it (its args.json values; the command line still wins). Paths, epochs,
#: run names, the device and pretraining are never inherited.
INHERIT = ("batch", "window", "val_batch", "lr0", "lrf", "cos_lr", "warmup_epochs", "weight_decay", "clip", "ema", "patience", "steps",
           "tta", "amp", "pos_weight", "poly", "alpha", "lovasz", "fliplr", "rot90", "scale", "color", "gray", "restyle", "restyle_mode",
           "drop_tokens", "modes", "no_layers", "viewer_share", "plumbing_share", "feature_drop", "pool_by", "layer_loss", "obj_loss", "edge_loss", "closure",
           "coincide", "img_ppt", "img_max_px", "img_aux", "mode_input", "drop_path")
#: Architecture switches read back from a checkpoint's config (what the network really is), for the log and args.json.
ARCH_KEYS = ("image_branch", "img_lum", "img_fpn", "img_backbone", "no_layers", "layer_head", "obj_head", "mode_input", "layer_names",
             "group_context", "relations", "pool_by", "rel_attn", "edge_head", "edge_pairs")
MODEL_HELP = ("a research version (openstroke28 | openstroke29 | openstroke30 | openstroke31), a .pt or a run folder to start from (architecture and training "
              "recipe come from it; each fold of a run folder starts from its fold<k>/best.pt), or an old research name: openstroke1 | 2 | 3 | 4")


def describe_arch(config: dict) -> str:
    """Research version of a checkpoint's network (models.ARCHS): its ``arch`` when it is one, else the version with the most of its switches
    (checkpoints saved as "openstroke2"), also when a named version carries a newer version's switches; switches beyond that version
    follow (+ obj_head ...)."""
    def label(k, v):
        return k if v is True else f"{k}={int(v)}"   # counts, e.g. rel_attn=2 (OpenStroke30)

    def want(name):
        return {label(k, v) for k, v in ARCHS[name].items() if v}

    net = ("image_branch", "img_lum", "img_fpn", "no_layers", "layer_head", "obj_head", "mode_input", "layer_names", "group_context", "relations",
           "rel_attn", "edge_head", "edge_pairs")
    on = {label(k, config[k]) for k in net if config.get(k)}
    from .models import canonical_arch

    base = canonical_arch(config.get("arch"))
    best = None
    if base in ARCHS:   # a research version fine-tuned with more switches is the newer version they make (openstroke28 + rel_attn ... -> 30)
        on |= want(base)
        best = base
    if base == "openstroke2" or base in ARCHS:
        for name in ARCHS:
            if want(name) <= on and (best is None or len(want(name)) > len(want(best))):
                best = name
    if best is not None:
        extra = sorted(on - want(best))
        return best + (" + " + " + ".join(extra) if extra else "")
    return base + (" + " + " + ".join(sorted(on)) if on else "")


def _switch(v):
    """A switch read from the command line ("2", "true", 2, True) -> int / bool, so describe_arch sees the same values as a saved config."""
    if isinstance(v, str):
        s = v.strip().lower()
        return int(s) if s.lstrip("-").isdigit() else s in ("1", "true", "yes", "on")
    return v


def run_version(args) -> str:
    """Research version of the network this run trains (the name its checkpoints save), e.g. openstroke30 when E114 (openstroke28) is
    fine-tuned with rel_attn=2 + edge_head; user 2026-11-18: the log must name the run's version, not the source's or the backbone's."""
    keys = ("image_branch", "img_lum", "img_fpn", "no_layers", "layer_head", "obj_head", "mode_input", "layer_names", "group_context", "relations",
            "rel_attn", "edge_head", "edge_pairs")
    return describe_arch({"arch": args.arch, **{k: _switch(getattr(args, k, None)) for k in keys}})


def arch_name(config: dict) -> str:
    """The research version a newly trained checkpoint saves as its ``arch`` (no extras; old research bases stay as they are). A research
    version fine-tuned with another version's switches saves that version (OpenStroke31 started from E114 / E121)."""
    from .models import canonical_arch

    arch = canonical_arch(config.get("arch"))
    return describe_arch(config).split(" + ")[0] if arch == "openstroke2" or arch in ARCHS else arch


def _checkpoint_of(model: str) -> Path | None:
    p = Path(model)
    ck = p / "fold0" / "best.pt" if p.is_dir() else p
    return ck if ck.exists() else None


def add_args(ap) -> None:
    a = ap.add_argument
    a("--data", default="datasets/e0")
    a("--folds", default=None, help="fold file (default: <data>/folds.json)")
    a("--fold", default="all", help="'all' or validation fold ids, e.g. 0,2")
    a("--modes", default="all", help="pages used, train and val: all | a comma list of fire, hvac, plumbing ('both' = all)")
    a("--model", default=None, help="required: " + MODEL_HELP)
    a("--project", default="runs")
    a("--name", default="train", help="run folder project/name; taken -> name2, name3...")
    a("--out", default=None, help="explicit run folder (overrides project/name)")
    a("--exist-ok", action="store_true")
    a("--resume", default=None, help="true (latest run) or a run folder: continue unfinished folds from last.pt")
    a("--cache", default=None, help="feature cache (default: <project>/cache)")
    # schedule
    a("--epochs", type=int, default=None)
    a("--steps", type=int, default=200, help="optimizer steps per epoch")
    a("--batch", type=int, default=None, help="windows per step")
    a("--window", type=int, default=None, help="shapes per window (train and val)")
    a("--val-batch", type=int, default=None)
    a("--lr0", "--lr", dest="lr0", type=float, default=None)
    a("--lrf", type=float, default=0.01, help="final lr = lr0 * lrf")
    a("--cos-lr", default="true")
    a("--warmup-epochs", type=float, default=3.0)
    a("--weight-decay", type=float, default=0.05)
    a("--clip", type=float, default=1.0)
    a("--ema", type=float, default=0.999)
    a("--patience", type=int, default=30)
    a("--val-every", type=int, default=1)
    a("--tta", default="true", help="final validation of best.pt with the 8 flips / quarter turns")
    a("--amp", default="true")
    a("--workers", type=int, default=8)
    a("--gpu-batch", default="false",
      help="build batches on the GPU (true / false / auto). Off by default: on G4 it was slower (LEDGER F28)")
    a("--feature-drop", default="", help="hide input features: comma list of groups (layer_stats, style_stats, colour) "
                                       "or feature names")
    a("--seed", type=int, default=0)
    a("--pretrain", type=float, default=0.0,
      help="E19: masked-shape pretraining with this mask ratio (e.g. 0.4) instead of the label loss; fine-tune after with model=<this run>")
    a("--parallel", type=int, default=1,
      help="run up to N folds at once, one process each on the same GPU (same results as one by one)")
    a("--child", type=int, default=None, help=argparse.SUPPRESS)  # internal: one fold of a parallel run
    a("--device", default="auto")
    # loss
    a("--pos-weight", type=float, default=2.0)
    a("--poly", type=float, default=0.0, help="E49: Poly-1 weight on BCE, rewards a correct answer as it grows more confident")
    a("--alpha", type=float, default=0.5, help="BCE weight = (length + 1) ** alpha")
    a("--lovasz", type=float, default=None, help="Lovasz (on probabilities) weight")
    # augmentation
    a("--fliplr", type=float, default=0.5)
    a("--rot90", default="true")
    a("--scale", type=float, default=0.2)
    a("--color", type=float, default=0.5)
    a("--gray", type=float, default=0.1)
    a("--restyle", type=float, default=None, help="probability of random colours / widths per style group (F12)")
    a("--drop-tokens", type=float, default=0.1)
    # architecture
    a("--layer-pool", default="true")
    a("--mode-input", default="false", help="give the model the page's flow (fire / hvac / plumbing), zero-init")
    a("--layer-names", default="false", help="give the model the PDF layer names (hashed tokens), zero-init (H15)")
    a("--name-drop", type=float, default=0.3, help="with layer_names: probability of hiding a window's layer names")
    a("--layer-head", default="false", help="E24: second head, one 'pipe layer' logit per layer of the window")
    a("--layer-loss", type=float, default=0.5, help="weight of the layer-head BCE")
    a("--img-ppt", type=float, default=1.0, help="E34: image pixels per pt (capped by --img-max-px)")
    a("--img-max-px", type=int, default=1536, help="E34: largest image side, px")
    a("--img-fpn", default="false", help="E37: image branch with layer4 + FPN to stride 4 (zero-init, starts as without)")
    a("--img-backbone", default="resnet18", help="E48: image backbone, resnet18 | hgnetv2_b5 (OpenOBB's, Apache-2.0)")
    a("--img-weights", default=None, help="E48: state dict of a new image backbone (a floorplan-trained OpenOBB one), not stored in the checkpoint config")
    a("--obj-head", default="false", help="E62: HVAC object class + centre offset of every shape (docs/OPENSTROKE7.md)")
    a("--obj-loss", type=float, default=1.0, help="E62: weight of the object loss")
    a("--rel-attn", type=int, default=0, help="OpenStroke30: graph-attention blocks along the stroke link graph after the decoder, zero-init (H132)")
    a("--edge-head", default="false", help="OpenStroke30: same-object logit per linked stroke pair, trained from GT instances (H132)")
    a("--edge-loss", type=float, default=0.5, help="OpenStroke30: weight of the same-object link loss")
    a("--edge-pairs", default="false", help="OpenStroke31: the link loss balances same- / other-object links between two true strokes (H136)")
    a("--closure", type=float, default=0.0, help="OpenStroke30: weight of the closure loss (soft-min of each GT object's stroke logits -> 1; H133)")
    a("--img-aux", type=float, default=0.0, help="E38: weight of the dense pipe-ink loss on the FPN map (needs img_fpn)")
    a("--img-lum", default="false", help="E30: the image branch draws each shape with its own darkness (gray strokes gray)")
    a("--coincide", type=float, default=0.0, help="E36: no loss on true shapes whose (layer, style) group is < this share true")
    a("--restyle-mode", default="free", help="E30: free (random colours / widths) | order (keep lightness and width order)")
    a("--pool-by", default="layer", help="E31: layer (PDF layer field) | style (colour + fill + width + kind groups, layer_1)")
    a("--image-branch", default="false", help="E28: draw each window as an image, pretrained CNN features at every shape's points")
    a("--img-pretrained", default="true", help="E28: ImageNet weights for the image backbone (false: random init, for offline tests)")
    a("--layer-head-names", default="false", help="E27: the layer head also reads the layer name tokens (not the shape head)")
    a("--group-context", default="false", help="attention between layer groups and style groups (layer_1), zero-init (H16)")
    a("--relations", type=int, default=None,
      help="OpenStroke3: rounds of message passing along the shape link graph (endpoint / dash gap / tee), zero-init (H17)")
    a("--no-layers", default="false",
      help="OpenStroke28: every shape on one unnamed layer (input = drawing commands, styles, image); no layer statistics or pooling")
    a("--viewer-share", type=float, default=None,
      help="OpenStroke28: Viewer pages (vw_*) together get this share of the training windows, GT pages the rest")
    a("--plumbing-share", type=float, default=None,
      help="PLUMBING pages together get this share of training windows (default: by size, like every page)")
    a("--drop-path", type=float, default=None)
    a("--patch", type=int, default=1024)
    a("--dims", default="64,96,128,192,256")
    a("--enc-depths", default="2,2,2,4,2")
    a("--dec-depths", default="1,1,1,1")
    a("--dim", type=int, default=128, help="openstroke1 width")
    a("--depth", type=int, default=4, help="openstroke1 depth")
    a("--k", type=int, default=16, help="openstroke1 neighbours")
    # back-compat switches of the first CLI
    a("--no-layer-pool", action="store_true", help=argparse_suppress())


def argparse_suppress():
    import argparse

    return argparse.SUPPRESS


def _bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise SystemExit(f"expected true/false, got {v!r}")


def _parser_defaults():
    ap = argparse.ArgumentParser()
    add_args(ap)
    return ap.parse_args([])


def _boolish(v) -> bool:
    try:
        return _bool(v)
    except SystemExit:
        return False


def _ints(v):
    return [int(x) for x in str(v).split(",")] if isinstance(v, str) else list(v)


def finalize(args):
    if not args.model:
        res = str(getattr(args, "resume", None) or "").lower()
        if res in ("", "false", "0", "no"):
            raise SystemExit("model= is required (as in YOLO): " + MODEL_HELP +
                             "\n  e.g. openstroke train model=openstroke3 data=...")
        run = latest_run(Path(args.project), args.name) if res in ("true", "1", "yes") else Path(args.resume)
        args.model = json.loads((run / "args.json").read_text(encoding="utf-8"))["model"]
    explicit = set(getattr(args, "_explicit", None) or ())
    args.start = None
    args.model = ALIASES.get(str(args.model), args.model)
    if args.model in PRESETS:   # a research version: its switches and recipe unless the command line sets them
        arch, flags = PRESETS[args.model]
        for k, v in flags.items():
            if k not in explicit:
                setattr(args, k, v)
    elif args.model in DEFAULTS:
        arch = args.model
    else:  # start from a checkpoint: its architecture and its training recipe (OpenOBB-style)
        import torch

        ck_path = _checkpoint_of(args.model)
        if ck_path is None:
            raise SystemExit(f"model={args.model}: expected " + MODEL_HELP)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        cfg = dict(ck["config"])
        from .models import base_arch

        arch = base_arch(cfg.get("arch"))   # research version (openstroke30 ...) -> its backbone's defaults; either repo's prefix
        prev = dict(ck.get("args") or {})
        took = {}
        for k in INHERIT:
            if k not in explicit and prev.get(k) is not None:
                setattr(args, k, prev[k])
                took[k] = prev[k]
        for k in ARCH_KEYS:   # what the network is (the config), unless the command adds a head
            if k in cfg and k not in explicit:   # anything the command line sets wins (a new head, another image backbone...)
                setattr(args, k, cfg[k])
        args.start = {"checkpoint": str(ck_path), "architecture": describe_arch(cfg), "run": Path(prev.get("out") or prev.get("name") or "").name or None,
                      "epoch": None if ck.get("epoch") is None else int(ck["epoch"]) + 1, "val_f2": ck.get("val_f2") or ck.get("best"),
                      "threshold": ck.get("threshold"), "recipe": took,
                      "new_heads": [k for k in ("obj_head", "layer_head", "edge_head", "rel_attn") if k in explicit and _switch(getattr(args, k, None)) and not cfg.get(k)]}
    for k, v in DEFAULTS[arch].items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    args.arch = arch
    if not isinstance(args.feature_drop, list):
        args.feature_drop = [f.strip() for f in str(args.feature_drop or "").split(",") if f.strip()]
    for k in ("cos_lr", "tta", "amp", "rot90", "layer_pool", "mode_input", "layer_names", "group_context", "layer_head", "layer_head_names", "image_branch", "img_pretrained", "img_lum", "img_fpn", "obj_head", "no_layers", "edge_head", "edge_pairs"):
        setattr(args, k, _bool(getattr(args, k)))
    if args.no_layer_pool:
        args.layer_pool = False
    args.dims, args.enc_depths, args.dec_depths = _ints(args.dims), _ints(args.enc_depths), _ints(args.dec_depths)
    return args


def _cache_one(job):
    from .data import load_page

    data, stem, mode, cache, graph, no_layers = job
    load_page(data, stem, mode, cache, graph=graph, no_layers=no_layers)   # writes the cache file
    return stem


def build_cache(data: Path, entries, cache: Path, graph: bool, no_layers: bool, say, workers: int | None = None) -> None:
    """Features of the pages missing from the cache, computed in parallel processes, with progress (speed, user 2026-11-16: the first
    run on a fresh cache took 4 min one page at a time, silently). Same files as load_page writes; pages already cached are skipped."""
    from concurrent.futures import ProcessPoolExecutor

    from .data import DATA_VERSION

    todo = [(data, s, m, cache, graph, no_layers) for s, _, m in entries
            if not (Path(cache) / f"{s}.{DATA_VERSION}{'.nl' if no_layers else ''}.npz").exists()]
    if not todo:
        return
    n = workers or max(1, min(8, (os.cpu_count() or 2) - 1, len(todo)))
    say(f"features {len(todo)} pages to compute ({len(entries) - len(todo)} cached), {n} processes -> {cache}")
    t0 = time.time()
    if n == 1:
        for i, j in enumerate(todo, 1):
            _cache_one(j)
            if i % 20 == 0 or i == len(todo):
                say(f"         {i}/{len(todo)} pages ({time.time() - t0:.0f}s)")
        return
    with ProcessPoolExecutor(n) as ex:
        for i, _ in enumerate(ex.map(_cache_one, todo), 1):
            if i % 20 == 0 or i == len(todo):
                say(f"         {i}/{len(todo)} pages ({time.time() - t0:.0f}s)")


def next_run_dir(path: Path) -> Path:
    """``runs/train`` if unused, else ``runs/train2``, ``train3``... (as YOLO / OpenOBB)."""

    if not (path / "train.log").exists():
        return path
    i = 2
    while (path.parent / f"{path.name}{i}" / "train.log").exists():
        i += 1
    return path.parent / f"{path.name}{i}"


def latest_run(project: Path, name: str) -> Path:
    runs = [p for p in project.glob(f"{name}*") if (p / "train.log").exists()]
    if not runs:
        raise SystemExit(f"resume: no run under {project}/{name}*")
    return max(runs, key=lambda p: (p / "train.log").stat().st_mtime)


def init_checkpoint(model_arg: str, fold: int) -> Path | None:
    """``model=`` -> checkpoint to start fold ``fold`` from (None: a fresh model). A run folder gives each fold
    its own ``fold<k>/best.pt``, so no fold starts from weights that saw its validation pages."""

    if model_arg in DEFAULTS:
        return None
    p = Path(model_arg)
    if p.is_dir():
        ck = p / f"fold{fold}" / "best.pt"
        if not ck.exists():
            raise FileNotFoundError(f"model={model_arg}: no {ck}")
        return ck
    if p.suffix == ".pt" and p.exists():
        return p
    raise ValueError(f"model={model_arg!r}: expected one of {tuple(DEFAULTS)}, a run folder or an existing .pt")


def _child_args(args):
    """A fold process of ``parallel=N``: the parent's finalized arguments, one fold, the parent's run folder."""

    out = Path(args.out)
    prev = json.loads((out / "args.json").read_text(encoding="utf-8"))
    # a run started by an older version lacks newer options: take their defaults (their old behaviour)
    ap = argparse.ArgumentParser()
    add_args(ap)
    ns = argparse.Namespace(**{**vars(ap.parse_args([])), **prev})
    for k in ("cos_lr", "tta", "amp", "rot90", "layer_pool", "mode_input", "layer_names", "group_context", "layer_head", "layer_head_names", "image_branch", "img_pretrained", "img_lum", "img_fpn", "obj_head", "no_layers", "edge_head", "edge_pairs"):
        setattr(ns, k, _bool(getattr(ns, k)))
    ns.child, ns.fold, ns.parallel, ns.workers = args.child, str(args.child), 1, args.workers
    ns.resume, ns.out, ns.exist_ok = bool(args.resume), str(out), True
    return ns


def _run_parallel(args, run, out_root, say) -> None:
    """One ``openstroke train --child k`` process per fold, at most ``args.parallel`` at a time. Each writes
    ``fold<k>/train.log`` and ``fold<k>/console.log``; this process prints each fold's validation rows."""

    import csv

    n = max(1, min(args.parallel, len(run)))
    workers = max(2, min(args.workers, (os.cpu_count() or 8) // n - 1))
    say(f"parallel {n} fold processes on one device, {workers} data workers each "
        f"(per fold: {out_root}/fold<k>/train.log, console.log)")
    queue, live, seen, failed = list(run), {}, {}, []
    while queue or live:
        while queue and len(live) < n:
            f = queue.pop(0)
            fo = out_root / f"fold{f}"
            fo.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "openstroke", "train", "--child", str(f), "--out", str(out_root),
                   "--workers", str(workers)] + (["--resume", "true"] if args.resume else [])
            con = open(fo / "console.log", "a", encoding="utf-8")
            pkg_root = str(Path(__file__).resolve().parents[1])
            env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [pkg_root, os.environ.get("PYTHONPATH")]))}
            live[f] = (subprocess.Popen(cmd, stdout=con, stderr=subprocess.STDOUT, env=env), con)
            say(f"fold {f + 1}: started (pid {live[f][0].pid})")
        time.sleep(5)
        for f in sorted(live):
            csvp = out_root / f"fold{f}" / "results.csv"
            if csvp.exists():
                rows = list(csv.DictReader(open(csvp, encoding="utf-8")))
                for r in rows[seen.get(f, 0):]:
                    if r.get("F2"):
                        say(f"fold {f + 1}  epoch {r['epoch']:>3}/{args.epochs}  bce {float(r['bce_loss']):.4f}  "
                            f"P {float(r['P']):.3f} R {float(r['R']):.3f} F2 {float(r['F2']):.3f} "
                            f"F2@0.5 {float(r['F2@0.5']):.3f} thr {float(r['thr']):.2f}  {r['epoch_s']}s")
                seen[f] = len(rows)
        for f in sorted(live):
            proc, con = live[f]
            if proc.poll() is None:
                continue
            con.close()
            del live[f]
            tail = (out_root / f"fold{f}" / "console.log").read_text(encoding="utf-8", errors="replace").splitlines()
            if proc.returncode == 0:
                start = max((i for i, l in enumerate(tail) if l.startswith(f"Fold {f + 1} best.pt")), default=len(tail))
                block = []
                for line in tail[start:]:
                    if line.startswith("fold ") and " done (" in line:
                        break
                    block.append(line)
                say("\n".join(block))
            else:
                failed.append(f)
                say(f"fold {f + 1}: FAILED (exit {proc.returncode}); last lines of console.log:\n" + "\n".join(tail[-15:]))
    if failed:
        say(f"WARNING: folds {[f + 1 for f in failed]} failed; rerun with resume=true (finished folds are skipped)")


def main_train(args) -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if getattr(args, "child", None) is not None:
        args = _child_args(args)
    if not hasattr(args, "arch"):
        args = finalize(args)
    from .data import load_page
    from .engine.trainer import Trainer, summarize

    data = Path(args.data)
    project = Path(args.project)
    child = getattr(args, "child", None)
    if child is not None:
        out_root = Path(args.out)
    elif args.resume and str(args.resume).lower() not in ("false", "0", "no"):
        out_root = latest_run(project, args.name) if str(args.resume).lower() in ("true", "1", "yes") else Path(args.resume)
        args.resume = True
        prev = json.loads((out_root / "args.json").read_text(encoding="utf-8"))
        # every option of the run being resumed (an option newer than the run keeps its default = old behaviour);
        # only how this session runs it may change
        for k, v in prev.items():
            if k not in ("resume", "parallel", "child", "workers", "device", "out", "exist_ok", "val_batch"):
                setattr(args, k, v)
    else:
        args.resume = False
        out_root = Path(args.out) if args.out else project / args.name
        if not args.exist_ok:
            out_root = next_run_dir(out_root)
    data = Path(args.data)  # after a resume restored the run's own options (it used to stay at the default data)
    out_root.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else data.parent / "cache"   # next to the dataset (local disk on Colab), not in project
    log_dir = out_root / f"fold{child}" if child is not None else out_root
    log_dir.mkdir(parents=True, exist_ok=True)
    logf = open(log_dir / "train.log", "a", encoding="utf-8")

    def log(s):  # file only
        logf.write(s + "\n")
        logf.flush()

    def say(s):  # console + file
        print(s, flush=True)
        log(s)

    folds_path = Path(args.folds) if args.folds else data / "folds.json"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    folds = load_folds(folds_path)
    entries, warn = resolve(manifest, folds)
    if str(args.modes) not in ("all", "both"):
        keep = {m.strip() for m in str(args.modes).split(",") if m.strip()}
        bad = keep - {"fire", "hvac", "plumbing"}
        if bad:
            raise SystemExit(f"modes={args.modes}: unknown {sorted(bad)} (fire, hvac, plumbing)")
        entries = [e for e in entries if e[2] in keep]
    if child is None:
        (out_root / "args.json").write_text(json.dumps(vars(args), indent=1, default=str), encoding="utf-8")
    st = getattr(args, "start", None)
    version = run_version(args)
    if st:   # what is being loaded, in plain words (user 2026-11-16)
        say(f"start   {st['checkpoint']}")
        say(f"        weights of {st['architecture']}" + (f" | run {st['run']}" if st.get('run') else "") +
            (f" | epoch {st['epoch']}" if st.get('epoch') else "") + (f" | val F2 {float(st['val_f2']):.3f}" if st.get('val_f2') else "") +
            f"  ->  trains {version}" + (f" (new: {', '.join(st['new_heads'])})" if st.get('new_heads') else ""))
        if st.get("recipe"):   # only what differs from the defaults: the rest is the usual recipe
            base = vars(_parser_defaults()) | DEFAULTS.get(args.arch, {})
            diff = {k: v for k, v in st["recipe"].items() if base.get(k) != v and not (isinstance(v, bool) and str(base.get(k)).lower() == str(v).lower())}
            say("        recipe from the checkpoint" + (": " + " ".join(f"{k}={','.join(v) if isinstance(v, list) else v}" for k, v in diff.items())
                                                         if diff else ": the defaults"))
    elif args.model in PRESETS:
        say(f"start   new {args.model} ({PRESETS[args.model][0]} + " + " + ".join(k if v is True else f"{k}={v}" for k, v in PRESETS[args.model][1].items()
                                                                            if k in ARCH_KEYS and v) + ")")
    say(f"OpenStroke train  {out_root}  {version}  model={args.model}  data={data}  folds={folds_path.name} "
        f"[{folds.get('status')}]")
    if folds.get("status") != "confirmed":
        say("WARNING: fold file is a proposal, not confirmed by the user: numbers are provisional")
    dups = [w.split("(", 1)[1].split(")", 1)[0] for w in warn if "duplicate_choice" in w]
    if dups:
        say(f"WARNING: {len(dups)} pages with two GT versions skipped (no duplicate_choice): {', '.join(dups)}")
    for w in warn:
        log("WARNING: " + w)
    say(f"train   epochs={args.epochs} steps={args.steps} batch={args.batch} window={args.window} lr0={args.lr0} "
        f"lrf={args.lrf} {'cos' if args.cos_lr else 'linear'} warmup={args.warmup_epochs}ep wd={args.weight_decay} "
        f"ema={args.ema} clip={args.clip} patience={args.patience} amp={args.amp}")
    say(f"loss    BCE (length+1)^{args.alpha} x pos_weight {args.pos_weight}" + (f" + Lovasz {args.lovasz}" if args.lovasz else ""))
    say(f"augment flip {args.fliplr} | rot90 {args.rot90} | scale ±{args.scale} | color perm {args.color} | gray "
        f"{args.gray} | restyle {args.restyle} | drop shapes ≤{args.drop_tokens}" + (" | TTA x8 on final val" if args.tta else ""))
    say(f"inputs  mode_input={args.mode_input}  layer_names={args.layer_names} (name_drop {args.name_drop})  "
        f"group_context={args.group_context}  relations={args.relations}  feature_drop={args.feature_drop or '-'}  "
        f"plumbing_share={args.plumbing_share}  no_layers={getattr(args, 'no_layers', False)}  viewer_share={getattr(args, 'viewer_share', None)}")
    t0 = time.time()
    pages = {}
    graph = bool(getattr(args, "relations", 0)) or args.arch == "openstroke4" or bool(getattr(args, "rel_attn", 0)) or bool(getattr(args, "edge_head", False))
    no_layers = bool(getattr(args, "no_layers", False))
    build_cache(data, entries, cache, graph, no_layers, say)
    for s, _, mode in entries:
        pages[s] = load_page(data, s, mode, cache, graph=graph, no_layers=no_layers)
    say(f"data    {len(pages)} pages, {sum(len(p) for p in pages.values()) / 1e6:.2f}M shapes "
        f"(features {time.time() - t0:.0f}s, cache {cache})")
    say(f"save    {out_root}/fold<k>/ (best.pt, last.pt, results.csv/png, result.json), summary.json")
    fold_ids = sorted({f for _, f, _ in entries if f >= 0})  # fold -1 = train only (pseudo-labelled pages, E24)
    run = fold_ids if str(args.fold) == "all" else [int(f) for f in str(args.fold).split(",")]
    trainer = Trainer(args, log, say)
    results = {}
    if child is None and getattr(args, "parallel", 1) > 1 and len(run) > 1:
        _run_parallel(args, run, out_root, say)
        for f in run:
            rj = out_root / f"fold{f}" / "result.json"
            if rj.exists():
                results[f] = json.loads(rj.read_text(encoding="utf-8"))
        run = []
    for f in run:
        fo = out_root / f"fold{f}"
        if args.resume and (fo / "result.json").exists():
            results[f] = json.loads((fo / "result.json").read_text(encoding="utf-8"))
            say(f"fold {f + 1}: done earlier, skipped")
            continue
        val = [pages[s] for s, ff, _ in entries if ff == f]
        train = [pages[s] for s, ff, _ in entries if ff != f]
        results[f] = trainer.run_fold(f, len(fold_ids), train, val, fo, init=init_checkpoint(args.model, f))
    if child is not None:
        say(f"fold {child + 1} done ({(time.time() - t0) / 3600:.2f} h)")
        logf.close()
        return 0
    summary = summarize(results, pages, out_root, say)
    summary["fold_status"] = folds.get("status")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    say(f"\nResults saved to {out_root} ({(time.time() - t0) / 3600:.2f} h)")
    logf.close()
    return 0
