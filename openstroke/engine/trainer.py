"""Cross-validation trainer with YOLO / OpenOBB conventions.

Per validation fold: AdamW (no weight decay on norms / biases), linear warmup then cosine (or linear) decay
from ``lr0`` to ``lr0 * lrf``, AMP (bf16 when the GPU has it, else fp16 + GradScaler), gradient clipping,
EMA weights for validation and checkpoints, non-finite steps skipped, ``best.pt`` (EMA, by val F2),
``last.pt`` (everything, for ``resume``), ``results.csv`` / ``results.png``, early stop after
``patience`` epochs without a better F2. The run ends with a per-page table and a summary over folds.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..data import Augment, Page, WindowDataset, collate, describe, dihedral, eval_windows, window
from ..features import FEATURE_NAMES, N_FEATURES
from ..metrics import length_counts, per_page, prf, sweep
from .console import VAL_HEAD, EpochBar, dur, table, val_row
from .ema import ModelEMA
from .report import append_results, plot_results


def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in n or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_factor(it: int, total: int, warmup: int, lrf: float, cos: bool) -> float:
    warm = min(1.0, (it + 1) / max(warmup, 1))
    prog = min(max(it - warmup, 0) / max(total - warmup, 1), 1.0)
    f = lrf + (1 - lrf) * (0.5 * (1 + math.cos(math.pi * prog)) if cos else (1 - prog))
    return warm * f


def to_device(b, device):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


def predict_page(model, page: Page, window_size: int, device, batch: int = 4, tta: int = 1, amp_dtype=None):
    """Probabilities for every token of ``page``: windows with stride window/2 and ``tta`` dihedral views,
    all averaged."""

    import torch

    total = np.zeros(len(page), np.float64)
    count = np.zeros(len(page), np.float64)
    items = [(s, k) for s in eval_windows(page, window_size) for k in range(tta)]
    model.eval()
    with torch.no_grad():
        for a in range(0, len(items), batch):
            ws = [dihedral(window(page, s, window_size), k) if k else window(page, s, window_size)
                  for s, k in items[a:a + batch]]
            b = to_device(collate(ws), device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float16,
                                enabled=device.type == "cuda"):
                prob = torch.sigmoid(model(b).float())
            prob = prob.cpu().numpy()
            pos = b["pos"].cpu().numpy()
            m = pos >= 0
            np.add.at(total, pos[m], prob[m])
            np.add.at(count, pos[m], 1.0)
    return total / np.maximum(count, 1)


def evaluate(model, pages, cfg, device, amp_dtype, tta=1):
    scores = [predict_page(model, p, cfg.window, device, cfg.val_batch, tta, amp_dtype) for p in pages]
    sw = sweep(scores, [p.label for p in pages], [p.length for p in pages])
    at05 = next(r for r in sw["curve"] if abs(r["threshold"] - 0.5) < 1e-9)
    return scores, sw["best"], at05


class Trainer:
    def __init__(self, cfg, log, say):
        self.cfg, self.log, self.say = cfg, log, say

    def _device(self):
        import torch

        c = self.cfg
        dev = torch.device(c.device if c.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if bf16 else torch.float16
        return dev, amp_dtype, dev.type == "cuda" and c.amp

    def build_model(self, train_pages, init=None):
        import torch

        from ..models import build, from_config, load_weights

        c = self.cfg
        if init is not None:
            ck = torch.load(init, map_location="cpu", weights_only=False)
            model = from_config(ck["config"])
            load_weights(model, ck["model"])
            self.log(f"init from {init}")
            return model
        model = build(c.arch, N_FEATURES, dims=c.dims, enc_depths=c.enc_depths, dec_depths=c.dec_depths,
                      patch=c.patch, drop_path=c.drop_path, layer_pool=c.layer_pool, dim=c.dim, depth=c.depth, k=c.k,
                      mode_input=getattr(c, "mode_input", False), layer_names=getattr(c, "layer_names", False),
                      group_context=getattr(c, "group_context", False))
        rng = np.random.default_rng(0)
        xs = np.concatenate([p.x[rng.choice(len(p), min(len(p), 100_000), replace=False)] for p in train_pages])
        model.set_normalization(xs.mean(axis=0), xs.std(axis=0))
        return model

    def run_fold(self, fold, n_folds, train_pages, val_pages, out: Path, init=None):
        import torch

        from ..models import loss as loss_fn

        c = self.cfg
        out.mkdir(parents=True, exist_ok=True)
        device, amp_dtype, amp = self._device()
        torch.manual_seed(c.seed + fold)
        last = out / "last.pt"
        model = self.build_model(train_pages, init).to(device)
        ema = ModelEMA(model, decay=c.ema, warmups=max(c.steps, 100))
        opt = torch.optim.AdamW(param_groups(model, c.weight_decay), lr=c.lr0, betas=(0.9, 0.999))
        scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
        start_epoch, it, best, best_epoch = 0, 0, -1.0, -1
        if c.resume and last.exists():
            ck = torch.load(last, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            ema.load_state_dict(ck["ema"])
            opt.load_state_dict(ck["opt"])
            start_epoch, it, best, best_epoch = ck["epoch"] + 1, ck["iter"], ck["best"], ck["best_epoch"]
            self.say(f"resume  fold {fold} from epoch {start_epoch + 1} (best F2 {best:.4f} at epoch {best_epoch + 1})")
        nparams = sum(p.numel() for p in model.parameters())
        names = ", ".join(sorted({p.stem.split("_gt")[0] for p in val_pages}))
        self.say(f"\nFold {fold + 1}/{n_folds}  val: {describe(val_pages)} [{names}]  |  train: {describe(train_pages)}")
        if val_pages:
            tl = sum(float(p.length[p.label > 0].sum()) for p in val_pages)
            pi = tl / max(sum(float(p.length.sum()) for p in val_pages), 1e-9)
            self.say(f"baseline val F2 if every shape is called true (no learning): {5 * pi / (4 * pi + 1):.3f} "
                     f"(true = {pi:.1%} of length)")
        if fold == 0 or start_epoch == 0:
            self.say(f"model   {c.arch} {nparams / 1e6:.2f}M params, " + json.dumps(model.config))
        ds = WindowDataset(train_pages, c.steps * c.batch, c.window,
                           Augment(fliplr=c.fliplr, rot90=c.rot90, scale=c.scale, color=c.color, gray=c.gray, restyle=c.restyle,
                                   name_drop=getattr(c, "name_drop", 0.0) if getattr(c, "layer_names", False) else 0.0,
                                   drop_tokens=c.drop_tokens), seed=c.seed + 1000 * fold,
                           plumbing_share=getattr(c, "plumbing_share", None))
        dl = torch.utils.data.DataLoader(ds, batch_size=c.batch, shuffle=False, num_workers=c.workers,
                                         collate_fn=collate, pin_memory=device.type == "cuda",
                                         persistent_workers=False, prefetch_factor=4 if c.workers else None)
        total_iters = c.epochs * c.steps
        warmup = int(c.warmup_epochs * c.steps)
        cols = ["bce", "lovasz"] if c.lovasz > 0 else ["bce"]
        history = []
        for epoch in range(start_epoch, c.epochs):
            ds.epoch = epoch
            model.train()
            t0 = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            sums, n, shapes, bad = {}, 0, 0, 0
            bar = EpochBar(epoch, c.epochs, c.steps, cols, c.window, header=True)
            for bi, b in enumerate(dl):
                for g in opt.param_groups:
                    g["lr"] = c.lr0 * lr_factor(it, total_iters, warmup, c.lrf, c.cos_lr)
                b = to_device(b, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                    logits = model(b)
                total, parts = loss_fn(c.arch, logits, b, c.pos_weight, c.alpha, c.lovasz)
                if not torch.isfinite(total):
                    bad += 1
                    opt.zero_grad(set_to_none=True)
                    it += 1
                    continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip)
                if not torch.isfinite(gn):
                    bad += 1
                    scaler.update()
                    it += 1
                    continue
                scaler.step(opt)
                scaler.update()
                ema.update(model)
                for k, v in parts.items():
                    sums[k] = sums.get(k, 0.0) + float(v.detach())
                n += 1
                it += 1
                shapes += int(b["mask0"].sum())
                bar.update(bi + 1, {k: sums[k] / n for k in sums}, int(b["mask0"].sum()),
                           torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else None,
                           force=bi + 1 == c.steps)
            bar.close()
            dt = time.time() - t0
            rec = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], "epoch_s": round(dt, 1),
                   "shapes_s": round(shapes / max(dt, 1e-9)),
                   "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if device.type == "cuda" else ""}
            rec.update({f"{k}_loss": round(v / max(n, 1), 5) for k, v in sums.items()})
            if bad:
                self.say(f"WARNING: {bad} non-finite steps skipped in epoch {epoch + 1}")
            if val_pages and ((epoch + 1) % c.val_every == 0 or epoch + 1 == c.epochs):
                tv = time.time()
                ema.repair(model, self.log)
                _, bst, at05 = evaluate(ema.module, val_pages, c, device, amp_dtype)
                rec.update({"P": round(bst["precision"], 4), "R": round(bst["recall"], 4), "F2": round(bst["f2"], 4),
                            "F2@0.5": round(at05["f2"], 4), "thr": bst["threshold"]})
                print(VAL_HEAD + f"  ({dur(time.time() - tv)})")
                print(val_row("all", len(val_pages), sum(len(p) for p in val_pages),
                              sum(float(p.length[p.label > 0].sum()) for p in val_pages),
                              bst["precision"], bst["recall"], bst["f2"], at05["f2"], bst["threshold"]), flush=True)
                if bst["f2"] > best:
                    best, best_epoch = bst["f2"], epoch
                    torch.save({"model": ema.module.state_dict(), "config": ema.module.config, "epoch": epoch,
                                "threshold": bst["threshold"], "val_f2": best, "feature_names": FEATURE_NAMES,
                                "args": vars(c)}, out / "best.pt")
                    self.log(f"best.pt <- epoch {epoch + 1} (val F2 {best:.4f} @ {bst['threshold']:.2f})")
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                        "iter": it, "best": best, "best_epoch": best_epoch, "config": model.config, "args": vars(c)}, last)
            history.append(rec)
            append_results(out, rec)
            plot_results(out)
            self.log(f"epoch {epoch + 1}: " + json.dumps(rec))
            if c.patience and best_epoch >= 0 and epoch - best_epoch >= c.patience:
                self.say(f"Stopping early: no better val F2 for {c.patience} epochs (best {best:.4f} at epoch "
                         f"{best_epoch + 1}).")
                break
        return self.finish_fold(fold, model, val_pages, out, device, amp_dtype, best, best_epoch, nparams)

    def finish_fold(self, fold, model, val_pages, out, device, amp_dtype, best, best_epoch, nparams):
        import torch

        from ..models import from_config, load_weights

        c = self.cfg
        result = {"fold": fold, "val": [p.stem for p in val_pages], "params": nparams, "best_f2": best,
                  "best_epoch": best_epoch + 1}
        if not val_pages or not (out / "best.pt").exists():
            (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
            return result
        ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        m = from_config(ck["config"])
        load_weights(m, ck["model"])
        m.to(device)
        tta = 8 if c.tta else 1
        scores, bst, at05 = evaluate(m, val_pages, c, device, amp_dtype, tta=tta)
        thr = bst["threshold"]
        ck["threshold"] = thr
        ck["tta"] = tta
        torch.save(ck, out / "best.pt")
        labels = [p.label for p in val_pages]
        lengths = [p.length for p in val_pages]
        stems = [p.stem for p in val_pages]
        result.update({"tta": tta, "threshold": thr, "val_at_threshold": bst, "val_at_0.5": at05,
                       "pages": per_page(scores, labels, lengths, stems, thr),
                       "pages_at_0.5": per_page(scores, labels, lengths, stems, 0.5)})
        np.savez_compressed(out / "val_scores.npz", **{s: sc.astype(np.float32) for s, sc in zip(stems, scores)},
                            **{f"{p.stem}__index": p.index for p in val_pages})
        (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        self.say(f"\nFold {fold + 1} best.pt (epoch {best_epoch + 1}){' + TTA x8' if tta > 1 else ''}, threshold {thr:.2f}")
        rows = [[p["stem"], f"{p['recall']:.3f}", f"{p['precision']:.3f}", f"{p['f2']:.3f}",
                 f"{p['true_length_share']:.1%}", f"{p['pred_length_share']:.1%}"] for p in result["pages"]]
        rows.append(["all", f"{bst['recall']:.3f}", f"{bst['precision']:.3f}", f"{bst['f2']:.3f}", "", ""])
        self.say(table(rows, ["page", "R", "P", "F2", "true", "pred"]))
        return result


def summarize(results: dict[int, dict], pages: dict[str, Page], out: Path, say) -> dict:
    """Micro over every fold's validation pages (each at its fold's threshold, and at 0.5)."""

    summary = {}
    for key, thr_of in (("at_fold_threshold", lambda r: r["threshold"]), ("at_0.5", lambda r: 0.5)):
        acc: dict[str, list[float]] = {}
        for f, r in results.items():
            fo = out / f"fold{f}"
            if "threshold" not in r or not (fo / "val_scores.npz").exists():
                continue
            z = np.load(fo / "val_scores.npz")
            for s in r["val"]:
                p = pages[s]
                tp, fp, fn = length_counts(z[s] >= thr_of(r), p.label, p.length)
                for k in ("all", p.mode):
                    a = acc.setdefault(k, [0.0, 0.0, 0.0])
                    a[0] += tp
                    a[1] += fp
                    a[2] += fn
        summary[key] = {k: prf(*v) for k, v in acc.items()}
    rows = []
    for f, r in sorted(results.items()):
        if "val_at_threshold" in r:
            v = r["val_at_threshold"]
            rows.append([f"fold {f + 1}", len(r["val"]), r["best_epoch"], f"{r['threshold']:.2f}", f"{v['recall']:.3f}",
                         f"{v['precision']:.3f}", f"{v['f2']:.3f}"])
    for k, v in summary.get("at_fold_threshold", {}).items():
        rows.append([k, "", "", "", f"{v['recall']:.3f}", f"{v['precision']:.3f}", f"{v['f2']:.3f}"])
    say("\nCross-validation summary (micro over every fold's val pages, length-weighted)")
    say(table(rows, ["", "pages", "best_ep", "thr", "R", "P", "F2"]))
    for k, v in summary.get("at_0.5", {}).items():
        say(f"SUMMARY at_0.5 {k}: R {v['recall']:.3f} P {v['precision']:.3f} F2 {v['f2']:.3f}")
    for k, v in summary.get("at_fold_threshold", {}).items():
        say(f"SUMMARY {k}: R {v['recall']:.3f} P {v['precision']:.3f} F2 {v['f2']:.3f}")
    worst = sorted((p for r in results.values() for p in r.get("pages", [])), key=lambda p: p["f2"])
    if worst:
        say("Pages by F2 (worst first):")
        say(table([[p["stem"], pages[p["stem"]].mode, f"{p['recall']:.3f}", f"{p['precision']:.3f}", f"{p['f2']:.3f}",
                    f"{p['true_length_share']:.1%}", f"{p['pred_length_share']:.1%}"] for p in worst],
                  ["page", "mode", "R", "P", "F2", "true", "pred"]))
    return summary
