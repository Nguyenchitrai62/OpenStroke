"""Deployment checkpoint: ``openstroke strip model=best.pt`` -> ``best_deploy.pt`` with only what inference needs.

Mirrors ``openobb/strip.py``: strips training run paths, optimizer/scheduler state, ~80 training args, validation
metrics and epoch metadata. Keeps only the EMA weights (optionally in float16, cutting file size in half), the
minimal architecture ``config``, ``feature_names``, ``window``, ``threshold``, ``tta``, and package ``version``.
Loads transparently with ``OpenStroke(path)`` and ``Predictor(path)`` under any filename.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .models import canonical_arch

# Paths / internal strings that must never leak into a shipped deploy checkpoint
LEAKS = ("/content", "/root", "\\users\\", "mydrive", "runs/", "drive/", "f:\\", "c:\\")


def _ema_state(ck: dict) -> dict:
    ema = ck.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"]
    return ck["model"]


def strip(model: str | Path, out: str | Path | None = None, half: bool = False) -> str:
    """Training checkpoint (.pt) -> clean deploy checkpoint path (default ``<stem>_deploy.pt`` next to it).

    ``half=True`` stores floating-point weights in float16 (~50% smaller on disk; cast back to float32 on load).
    """
    from . import __version__

    src = Path(model)
    if not src.exists():
        raise FileNotFoundError(src)
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if "config" not in ck or "model" not in ck:
        raise ValueError(f"{src}: not an OpenStroke checkpoint (missing 'config' / 'model')")

    cfg = dict(ck["config"])
    cfg["arch"] = canonical_arch(cfg.get("arch"))
    # Strip any local training weight path inside config (e.g. img_weights)
    cfg.pop("img_weights", None)

    sd = {}
    for k, v in _ema_state(ck).items():
        t = v.detach().cpu()
        if half and t.is_floating_point():
            t = t.half()
        sd[k] = t.clone()

    args = ck.get("args") or {}
    window = int(ck.get("window") or args.get("window") or (4096 if cfg["arch"] == "openstroke1" else 65536))
    dep = {
        "model": sd,
        "config": cfg,
        "window": window,
        "threshold": float(ck.get("threshold", 0.5)),
        "tta": int(ck.get("tta", 1)),
        "version": str(ck.get("version") or __version__),
        "deploy": True,
    }
    if ck.get("feature_names") is not None:
        dep["feature_names"] = list(ck["feature_names"])
    # Preserve single training mode ONLY for legacy mode_input=True checkpoints
    if cfg.get("mode_input"):
        m_arg = str(ck.get("default_mode") or args.get("modes") or "").strip().lower()
        if m_arg in ("fire", "hvac", "plumbing"):
            dep["default_mode"] = m_arg

    meta_text = json.dumps({k: v for k, v in dep.items() if k != "model"}, default=str).lower()
    found = [w for w in LEAKS if w in meta_text]
    if found:
        raise SystemExit(f"deploy checkpoint metadata still contains local paths {found}: {dep['config']}")

    dst = Path(out) if out else src.with_name(f"{src.stem}_deploy.pt")
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dep, dst)
    mb = lambda p: p.stat().st_size / (1024 * 1024)  # noqa: E731
    print(f"[strip] {src.name} ({mb(src):.1f} MB) -> {dst} ({mb(dst):.1f} MB, arch={cfg['arch']}, half={bool(half)})")
    return str(dst)
