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
                    total, _ = loss(arch, logits, b, 2.0, 0.5, 1.0 if arch == "openstroke2" else 0.0)
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
