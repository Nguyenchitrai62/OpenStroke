"""Raw stroke probabilities of a trained model on EVERY page of a dataset (train and val), for tuning post-processing locally (F149).

    openstroke scores model=runs/e114/fold0/best.pt data=/content/os28/e8            -> <model dir>/all_scores.npz

The npz has the layout of ``val_scores.npz`` (``<stem>`` float16 probabilities of the page's kept tokens, ``<stem>__index`` their
shape ids) plus ``__meta`` (JSON: each page's fold and split as the run saw it, the run's val fold, threshold, TTA). Same windows,
stride and TTA as the run's final val, so the val pages reproduce ``val_scores.npz``. Train pages were seen by the model: their
scores are optimistic and serve only to fit post-processing; conclusions stay on the val pages.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def score_all(model: Path, data: Path, out: Path | None = None, folds: Path | None = None, cache: Path | None = None,
              tta: int | None = None, batch: int | None = None, device: str = "auto", log=print) -> Path:
    import torch

    from .data import load_page
    from .engine.trainer import predict_page
    from .folds import load as load_folds, resolve
    from .models import from_config, load_weights

    ck = torch.load(model, map_location="cpu", weights_only=False)
    a = ck.get("args") or {}
    run_folds = Path(a["folds"]).name if a.get("folds") else "folds.json"
    folds = Path(folds) if folds else data / run_folds
    entries, _ = resolve(json.loads((data / "manifest.json").read_text(encoding="utf-8")), load_folds(folds))
    modes = str(a.get("modes", "all"))
    if modes not in ("all", "both"):
        keep = {m.strip() for m in modes.split(",") if m.strip()}
        entries = [e for e in entries if e[2] in keep]
    val_fold = int(str(a.get("fold", "0")).split(",")[0]) if str(a.get("fold", "0")) != "all" else None
    cache = Path(cache) if cache else data.parent / "cache"
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if dev.type == "cuda" and a.get("amp", True) else None
    m = from_config(ck["config"], weights_follow=True)
    load_weights(m, ck["model"])   # as the run's final val (finish_fold)
    m.to(dev).eval()
    tta = int(tta or ck.get("tta") or 1)
    batch = int(batch or a.get("val_batch") or 2)
    window = int(a.get("window", 65536))
    no_layers = bool(a.get("no_layers", False))
    graph = bool(a.get("relations", 0)) or a.get("arch") == "openstroke4"
    out = Path(out) if out else Path(model).with_name("all_scores.npz")
    arrays, meta = {}, {"model": str(model), "data": str(data), "folds": folds.name, "val_fold": val_fold,
                        "threshold": ck.get("threshold"), "tta": tta, "window": window, "pages": {}}
    t0 = time.time()
    for i, (stem, fold, mode) in enumerate(entries):
        p = load_page(data, stem, mode, cache, graph=graph, no_layers=no_layers)
        with torch.no_grad():
            sc = predict_page(m, p, window, dev, batch, tta, amp_dtype)
        arrays[stem] = sc.astype(np.float16)
        arrays[f"{stem}__index"] = p.index
        split = "val" if fold == val_fold else "train"
        meta["pages"][stem] = {"fold": int(fold), "split": split, "mode": mode, "shapes": int(len(p))}
        log(f"[{i + 1}/{len(entries)}] {stem} {split} {len(p)} shapes ({time.time() - t0:.0f}s)")
    arrays["__meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8)
    np.savez_compressed(out, **arrays)
    log(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB, {len(entries)} pages, TTA x{tta})")
    return out


def read_meta(z) -> dict:
    return json.loads(bytes(np.asarray(z["__meta"], np.uint8)).decode("utf-8"))
