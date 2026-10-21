"""``openstroke bench``: time one training step and peak GPU memory for a model / batch / window, on a
synthetic sheet (no data needed). Use it on the Colab GPU to choose ``batch`` and ``window`` before a long run.

    openstroke bench model=openstroke2 batch=4,8,16 window=65536
"""

from __future__ import annotations

import time

import numpy as np


def synthetic_window(n: int, rng, extent: float = 2400.0) -> dict:
    from .features import N_FEATURES, SHAPE_POINTS

    return {"x": rng.normal(size=(n, N_FEATURES)).astype(np.float32),
            "xy": rng.uniform(0, extent, size=(n, 2)),
            "pts": rng.normal(scale=5, size=(n, SHAPE_POINTS, 2)).astype(np.float32),
            "layer": rng.integers(0, 60, n), "label": (rng.random(n) < 0.15).astype(np.uint8),
            "length": rng.exponential(10, n).astype(np.float32), "pos": np.arange(n)}


def bench(arch: str, batches: list[int], windows: list[int], iters: int = 5, device: str = "auto", **model_cfg) -> list[dict]:
    import torch

    from .data import collate
    from .features import N_FEATURES
    from .models import build, loss

    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    rng = np.random.default_rng(0)
    rows = []
    for w in windows:
        for bsz in batches:
            model = build(arch, N_FEATURES, **model_cfg).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            t_col = time.time()
            batch = collate([synthetic_window(w, rng) for _ in range(bsz)])
            t_col = time.time() - t_col
            b = {k: v.to(dev) for k, v in batch.items()}
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            try:
                times = []
                for i in range(iters + 1):
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    t = time.time()
                    with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=dev.type == "cuda"):
                        logits = model(b)
                    total, _ = loss(arch, logits, b, 2.0, 0.5, 0.0 if arch == "openstroke1" else 1.0)
                    opt.zero_grad(set_to_none=True)
                    total.backward()
                    opt.step()
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    if i:  # first iteration warms up kernels
                        times.append(time.time() - t)
                s = float(np.median(times))
                mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else float("nan")
                row = {"window": w, "batch": bsz, "s_per_step": s, "shapes_per_s": w * bsz / s, "peak_gb": mem,
                       "collate_s": t_col}
            except torch.OutOfMemoryError:
                row = {"window": w, "batch": bsz, "s_per_step": float("nan"), "shapes_per_s": 0.0, "peak_gb": float("inf"),
                       "collate_s": t_col}
            rows.append(row)
            print(f"{arch} window={w:>6} batch={bsz:>3}: {row['s_per_step']:.3f} s/step, "
                  f"{row['shapes_per_s'] / 1e3:,.0f}k shapes/s, peak {row['peak_gb']:.1f} GB, collate {t_col:.2f} s"
                  + ("  (OOM)" if row["peak_gb"] == float("inf") else ""), flush=True)
            del model, opt, b, batch
    return rows


def bench_real(data, models: list[str], batch: int = 4, window: int = 65536, iters: int = 5, device: str = "auto",
               stems=None) -> list[dict]:
    """Real pages (with link graphs): time a batch built by one CPU worker (numpy augment + collate) and on the GPU
    (``gpu_batch``), then one optimizer step of each model on it. What the training speed really is made of."""

    import json
    from pathlib import Path

    import torch

    from .data import Augment, augment, collate, load_page, window as cut
    from .features import N_FEATURES
    from .gpu_batch import build_batch
    from .models import build, loss

    data = Path(data)
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = dev.type == "cuda"
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    recs = [r for r in json.loads((data / "manifest.json").read_text(encoding="utf-8"))["records"] if r["status"] == "ok"]
    recs = sorted(recs, key=lambda r: -r["stats"]["shapes"])[:3] if not stems else [r for r in recs if r["stem"] in stems]
    pages = [load_page(data, r["stem"], r["mode"], data.parent / "bench_cache", graph=True) for r in recs]
    aug = Augment(restyle=0.5)
    rng = np.random.default_rng(0)
    ws = [cut(pages[i % len(pages)], 0, window) for i in range(batch)]
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)

    def timed(fn, n=iters):
        out, ts = None, []
        for i in range(n + 1):
            sync()
            t = time.time()
            out = fn()
            sync()
            if i:
                ts.append(time.time() - t)
        return out, float(np.median(ts))

    _, t_cpu = timed(lambda: collate([augment({k: v.copy() for k, v in w.items()}, aug, rng) for w in ws]), n=2)
    b, t_gpu = timed(lambda: build_batch(ws, dev, aug, [int(s) for s in rng.integers(0, 2**31, len(ws))]))
    print(f"batch of {batch} x {window} shapes from {[r['filename'][:20] for r in recs]}:")
    print(f"  one CPU worker (numpy augment + collate): {t_cpu:.3f} s  -> 8 workers cap ~{8 / t_cpu:.1f} steps/s")
    print(f"  on {dev.type} (gpu_batch):                   {t_gpu:.3f} s")
    rows = []
    for arch in models:
        model = build(arch, N_FEATURES).to(dev)
        model.set_normalization(np.concatenate([p.x[:50000] for p in pages]).mean(0),
                                np.concatenate([p.x[:50000] for p in pages]).std(0))
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        def step():
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp):
                logits = model(b)
            total, _ = loss(arch, logits, b, 2.0, 0.5, 1.0)
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

        _, t = timed(step)
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else float("nan")
        rows.append({"model": arch, "s_per_step": t, "peak_gb": mem})
        print(f"  {arch}: {t:.3f} s / step (fwd + bwd + AdamW), peak {mem:.1f} GB  -> model alone ~{1 / t:.1f} steps/s")
        del model, opt
    return rows
