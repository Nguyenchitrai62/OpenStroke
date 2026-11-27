"""End-of-training deploy speed: ``torch.compile`` off vs ``compile=True`` on the run's val pages.

Loads the stripped deploy checkpoint the way a service does (``OpenStroke(ckpt, compile=...)``: fused Conv+BN,
channels_last, per-GPU AMP) and times the model stage (``Results.speed["inference"]``) on the already-loaded val
pages, TTA x1, as ``predict`` runs by default. Feature building and post-processing do not change with ``compile``,
so they are not timed here. Never raises: a failure is one line in the log.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np


def _pass(m, pages, sync) -> tuple[list[float], list[np.ndarray]]:
    """One TTA x1 pass over ``pages`` with the heads ``OpenStroke.predict`` runs; per-page ms and probabilities."""
    from .engine.trainer import predict_page

    has_edge = getattr(m.model, "edge_mlp", None) is not None
    has_group = getattr(m.model, "group_head", None) is not None
    has_cls = getattr(m.model, "obj_mlp", None) is not None
    has_scls = int(getattr(m.model, "stroke_classes", 1)) > 1
    ms, probs = [], []
    for page in pages:
        sync()
        t = time.perf_counter()
        p = m._run_guarded(
            lambda bs: predict_page(
                m.model, page, m.window, m.device, batch=bs, tta=1, amp_dtype=m.amp_dtype,
                obj_out=[] if has_cls else None, edge_out=[] if has_edge else None,
                group_out=[] if has_group else None, cls_out=[] if has_scls else None,
            ),
            m.batch,
        )
        sync()
        ms.append((time.perf_counter() - t) * 1e3)
        probs.append(np.asarray(p, np.float32))
    return ms, probs


def compile_speed_report(ckpt, pages, device=None, say=print, steady_passes: int = 2) -> dict[str, Any] | None:
    """Time ``ckpt`` on ``pages`` with compile off, then ``compile=True``; print a table and return the numbers.

    Off: one warm-up pass (cuDNN / allocator), then ``steady_passes`` timed passes. On: the first pass (includes the
    1-3 min compilation) is reported on its own, then ``steady_passes`` timed passes. ``max|dp|`` is the largest
    probability difference between the two settings (parity check)."""
    pages = [p for p in pages if len(p)]
    if not pages:
        return None
    try:
        import torch

        from .infer import OpenStroke

        def sync():
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize()

        out: dict[str, Any] = {"checkpoint": str(ckpt), "pages": len(pages),
                               "shapes": int(sum(len(p) for p in pages)), "tta": 1}

        m = OpenStroke(ckpt, device=device, compile=False)
        out["device"] = m.runtime
        _pass(m, pages, sync)
        runs = [_pass(m, pages, sync) for _ in range(max(1, steady_passes))]
        off_ms = np.mean([r[0] for r in runs], axis=0)
        off_p = runs[-1][1]
        del m

        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mc = OpenStroke(ckpt, device=device, compile=True)
        if not mc.compile:
            why = next((str(w.message) for w in caught if "compile" in str(w.message)), "compile not applied")
            say(f"Deploy speed ({len(pages)} val pages, TTA x1, {out['device']}): compile off "
                f"{off_ms.mean():.0f} ms/page; compile=True unavailable here ({why})")
            out.update(off_ms_per_page=float(off_ms.mean()), compile="unavailable", reason=why)
            return out
        try:
            t = time.perf_counter()
            first_ms, _ = _pass(mc, pages, sync)
            first_s = time.perf_counter() - t
            runs = [_pass(mc, pages, sync) for _ in range(max(1, steady_passes))]
        except Exception as exc:   # compilation that fails at the first call (Triton / inductor error)
            why = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
            say(f"Deploy speed ({len(pages)} val pages, TTA x1, {out['device']}): compile off "
                f"{off_ms.mean():.0f} ms/page; compile=True failed at the first call ({why})")
            out.update(off_ms_per_page=float(off_ms.mean()), compile="failed", reason=why)
            return out
        on_ms = np.mean([r[0] for r in runs], axis=0)
        on_p = runs[-1][1]
        del mc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dp = max(float(np.abs(a - b).max()) for a, b in zip(off_p, on_p))
        say(f"\nDeploy speed: {ckpt.name if hasattr(ckpt, 'name') else ckpt}, {len(pages)} val pages, TTA x1, "
            f"model stage only ({out['device']})")
        say(f"  {'page':<34}{'shapes':>9}{'off ms':>9}{'on ms':>9}{'speedup':>9}{'1st call ms':>13}")
        for p, a, b, f in zip(pages, off_ms, on_ms, first_ms):
            say(f"  {p.stem[:33]:<34}{len(p):>9}{a:>9.0f}{b:>9.0f}{a / b:>8.2f}x{f:>13.0f}")
        tot_off, tot_on = float(off_ms.sum()), float(on_ms.sum())
        say(f"  {'total':<34}{out['shapes']:>9}{tot_off:>9.0f}{tot_on:>9.0f}{tot_off / tot_on:>8.2f}x{sum(first_ms):>13.0f}")
        say(f"  compile=True: {tot_off / tot_on:.2f}x on the model stage after a {first_s:.0f} s first pass "
            f"(compilation); max|dp| vs off {dp:.1e}")
        out.update(off_ms_per_page=float(off_ms.mean()), on_ms_per_page=float(on_ms.mean()),
                   speedup=tot_off / tot_on, first_pass_s=first_s, max_abs_dprob=dp, compile="default",
                   per_page=[{"page": p.stem, "shapes": len(p), "off_ms": float(a), "on_ms": float(b),
                              "first_ms": float(f)} for p, a, b, f in zip(pages, off_ms, on_ms, first_ms)])
        return out
    except Exception as exc:  # the report must never break a finished training run
        say(f"Deploy speed report skipped: {type(exc).__name__}: {exc}")
        return None
