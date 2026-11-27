"""Raw stroke probabilities of a trained model on every GT page of a dataset (train and val), for tuning post-processing locally (F149).

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
              tta: int | None = None, batch: int | None = None, device: str = "auto", limit: int | None = None, viewer: bool = False,
              split: str = "all", log=print, *, stroke_dtype=np.float16) -> Path:
    """Each page is saved to ``<out>.parts/<stem>.npz`` as soon as it is scored (a stopped run continues from there; parts of another
    TTA are rescored) and the parts are joined into ``out`` at the end. Only the fully labelled GT pages by default; ``viewer=True`` adds the
    Viewer pages (partial "Labels V1" labels), after the GT pages. ``limit``: first N pages only."""
    import torch

    from .data import load_page
    from .engine.trainer import predict_page
    from .folds import load as load_folds, resolve
    from .models import canonical_arch, from_config, load_weights

    ck = torch.load(model, map_location="cpu", weights_only=False)
    a = ck.get("args") or {}
    run_folds = Path(a["folds"]).name if a.get("folds") else "folds.json"
    folds = Path(folds) if folds else data / run_folds
    entries, _ = resolve(json.loads((data / "manifest.json").read_text(encoding="utf-8")), load_folds(folds))
    modes = str(a.get("modes", "all"))
    if modes not in ("all", "both"):
        keep = {m.strip() for m in modes.split(",") if m.strip()}
        entries = [e for e in entries if e[2] in keep]
    cache = Path(cache) if cache else data.parent / "cache"
    from . import gpu

    dev = gpu.resolve_device(device)
    gpu.configure(dev, train=False)
    amp_dtype = gpu.amp_dtype(dev) if a.get("amp", True) else None   # bf16 where native, fp16 (inf / NaN -> fp32) on Turing
    m = from_config(ck["config"], weights_follow=True)
    load_weights(m, (ck.get("ema") or {}).get("module") or ck["model"])   # best.pt: EMA weights; last.pt: its "ema" entry (the scored ones)
    m.to(dev).eval()
    tta = int(tta or ck.get("tta") or 1)
    batch = int(batch or a.get("val_batch") or 2)
    window = int(a.get("window", 65536))
    no_layers = bool(a.get("no_layers", False))
    graph = (bool(a.get("relations", 0)) or canonical_arch(a.get("arch")) == "openstroke4" or bool(ck["config"].get("rel_attn", 0))
             or bool(ck["config"].get("edge_head")))
    if ck["config"].get("continuation"):
        graph = "continuation"
    elif ck["config"].get("facing_edges"):
        graph = "facing"
    out = Path(out) if out else Path(model).with_name("all_scores.npz")
    fold_arg = str(a.get("fold", "0"))
    val_folds = ({int(f) for _, f, _ in entries if f >= 0} if fold_arg == "all" else {int(x) for x in fold_arg.split(",")})
    val_fold = sorted(val_folds)[0] if len(val_folds) == 1 else None
    arrays, meta = {}, {"model": str(model), "data": str(data), "folds": folds.name, "val_fold": val_fold,
                        "threshold": ck.get("threshold"), "tta": tta, "window": window, "pages": {}}
    entries = sorted((e for e in entries if viewer or not e[0].startswith("vw_")), key=lambda e: e[0].startswith("vw_"))
    if split != "all":   # only the run's val pages (or only its train pages)
        entries = [e for e in entries if (e[1] in val_folds) == (split == "val")]
    entries = entries[:limit]
    parts = out.with_name(out.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, (stem, fold, mode) in enumerate(entries):
        page_split = "val" if fold in val_folds else "train"
        f = parts / f"{stem}.npz"
        if f.exists():
            with np.load(f) as z:
                if z["score"].dtype == stroke_dtype and int(z["tta"]) == tta and ("graph" not in z.files and graph != "continuation"
                                            or "graph" in z.files and str(z["graph"]) == str(graph)):
                    arrays[stem], arrays[f"{stem}__index"] = z["score"], z["index"]
                    if "oprob" in z.files:
                        arrays[f"{stem}__oprob"] = z["oprob"]
                    if "eij" in z.files:
                        arrays[f"{stem}__eij"], arrays[f"{stem}__elog"] = z["eij"], z["elog"]
                    meta["pages"][stem] = {"fold": int(fold), "split": page_split, "mode": mode, "shapes": int(len(z["index"]))}
                    continue
        p = load_page(data, stem, mode, cache, graph=graph, no_layers=no_layers, pieces=getattr(m, "piece", None) is not None)
        t1 = time.time()
        pc_out = [] if getattr(m, "obj_mlp", None) is not None else None
        e_out = [] if getattr(m, "edge_mlp", None) is not None else None
        with torch.no_grad():
            sc = predict_page(m, p, window, dev, batch, tta, amp_dtype, obj_out=pc_out, edge_out=e_out)
        arrays[stem] = sc.astype(stroke_dtype)
        arrays[f"{stem}__index"] = p.index
        extra = {}
        if pc_out:   # OpenStroke29: per-stroke class probabilities, same pass and views as the stroke scores
            from .objects import N_CLASSES
            arrays[f"{stem}__oprob"] = extra["oprob"] = pc_out[0][:, :N_CLASSES].astype(np.float16)
        if e_out:   # OpenStroke30 / F169: same-object logit per linked stroke pair (payload shape ids)
            arrays[f"{stem}__eij"] = extra["eij"] = p.index[e_out[0][0]].astype(np.int32)
            arrays[f"{stem}__elog"] = extra["elog"] = e_out[0][1].astype(np.float16)
        np.savez(f, score=arrays[stem], index=p.index, tta=tta, graph=str(graph), **extra)
        meta["pages"][stem] = {"fold": int(fold), "split": page_split, "mode": mode, "shapes": int(len(p))}
        log(f"[{i + 1}/{len(entries)}] {stem} {page_split} {len(p)} shapes, model {time.time() - t1:.0f}s ({time.time() - t0:.0f}s)")
    arrays["__meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8)
    np.savez_compressed(out, **arrays)
    log(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB, {len(entries)} pages, TTA x{tta})")
    return out


def read_meta(z) -> dict:
    return json.loads(bytes(np.asarray(z["__meta"], np.uint8)).decode("utf-8"))
