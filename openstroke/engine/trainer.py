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

from ..data import Augment, Page, WindowDataset, coincident_weights, collate, describe, dihedral, eval_windows, raw_items, window
from ..features import FEATURE_NAMES, N_FEATURES
from ..gpu_batch import build_batch
from ..metrics import length_counts, per_page, prf, sweep
from .console import VAL_HEAD, EpochBar, dur, table, val_row
from .ema import ModelEMA
from .report import append_results, plot_results


def param_groups(model, weight_decay):
    """AdamW groups; the pretrained image backbone (E28 ``img.``, not its new ``img.proj``) learns at 0.1 x lr."""

    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nd = p.ndim < 2 or "norm" in n or n.endswith(".bias")
        slow = n.startswith("img.") and not n.startswith("img.proj")
        groups.setdefault((nd, slow), []).append(p)
    return [{"params": ps, "weight_decay": 0.0 if nd else weight_decay, "lr_scale": 0.1 if slow else 1.0}
            for (nd, slow), ps in groups.items()]


def lr_factor(it: int, total: int, warmup: int, lrf: float, cos: bool) -> float:
    warm = min(1.0, (it + 1) / max(warmup, 1))
    prog = min(max(it - warmup, 0) / max(total - warmup, 1), 1.0)
    f = lrf + (1 - lrf) * (0.5 * (1 + math.cos(math.pi * prog)) if cos else (1 - prog))
    return warm * f


def use_gpu_batch(cfg, device) -> bool:
    v = str(getattr(cfg, "gpu_batch", "auto")).lower()
    return device.type == "cuda" if v == "auto" else v in ("1", "true", "yes")


def to_device(b, device):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


def predict_page(model, page: Page, window_size: int, device, batch: int = 4, tta: int = 1, amp_dtype=None,
                 layer_out: list | None = None):
    """Probabilities for every token of ``page``: windows with stride window/2 and ``tta`` dihedral views,
    all averaged. With a layer head and ``layer_out`` given, also appends each token's layer probability (its layer's
    logit in each window it was in, averaged the same way)."""

    import torch

    total = np.zeros(len(page), np.float64)
    count = np.zeros(len(page), np.float64)
    ltotal = np.zeros(len(page), np.float64)
    use_layer = layer_out is not None and getattr(model, "layer_mlp", None) is not None
    items = [(s, k) for s in eval_windows(page, window_size) for k in range(tta)]
    model.eval()
    with torch.no_grad():
        for a in range(0, len(items), batch):
            ws = [dihedral(window(page, s, window_size), k) if k else window(page, s, window_size)
                  for s, k in items[a:a + batch]]
            b = build_batch(ws, device) if device.type == "cuda" else to_device(collate(ws), device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float16,
                                enabled=device.type == "cuda"):
                prob = torch.sigmoid(model(b).float())
            prob = prob.cpu().numpy()
            pos = b["pos"].cpu().numpy()
            m = pos >= 0
            np.add.at(total, pos[m], prob[m])
            np.add.at(count, pos[m], 1.0)
            if use_layer:
                lp = torch.sigmoid(model.last_layer.float())
                tok = torch.gather(lp, 1, b["layer"].clamp(max=lp.shape[1] - 1)).cpu().numpy()
                np.add.at(ltotal, pos[m], tok[m])
    if use_layer:
        layer_out.append(ltotal / np.maximum(count, 1))
    return total / np.maximum(count, 1)


def evaluate(model, pages, cfg, device, amp_dtype, tta=1, layer_out: list | None = None):
    scores = [predict_page(model, p, cfg.window, device, cfg.val_batch, tta, amp_dtype, layer_out) for p in pages]
    sw = sweep(scores, [p.label for p in pages], [p.length for p in pages])
    at05 = next(r for r in sw["curve"] if abs(r["threshold"] - 0.5) < 1e-9)
    return scores, sw["best"], at05


def layer_loss(logits, b):
    """BCE of the layer head: one target per (window, layer) = whether that layer is a pipe layer of its page
    (``data.layer_targets``), weighted by sqrt of the layer's length in the window."""

    import torch

    B, n_l = logits.shape
    m = b["mask0"] > 0
    key = (b["layer"] + torch.arange(B, device=logits.device)[:, None] * n_l)[m]
    z = lambda: torch.zeros(B * n_l, device=logits.device, dtype=torch.float32)  # noqa: E731
    cnt = z().index_add_(0, key, torch.ones_like(key, dtype=torch.float32))
    y = z().index_add_(0, key, b["ylayer"][m].float()) / cnt.clamp(min=1)
    w = z().index_add_(0, key, b["length"][m].float()).sqrt()
    ok = cnt > 0
    w = w[ok] / w[ok].mean().clamp(min=1e-6)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits.reshape(-1)[ok].float(), y[ok], weight=w, pos_weight=torch.tensor(3.0, device=logits.device))


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
            cfg = dict(ck["config"])
            # heads asked for on the command line but absent from the checkpoint start fresh (E26: pretrained -> two heads)
            extra = {k: True for k in ("layer_head", "layer_head_names", "image_branch", "img_lum")
                     if getattr(c, k, False) and not cfg.get(k)}
            if str(getattr(c, "pool_by", "layer")) != cfg.get("pool_by", "layer"):  # E31: re-group LayerPool (same weights)
                extra["pool_by"] = str(getattr(c, "pool_by", "layer"))
            if extra.get("image_branch"):
                extra["img_pretrained"] = bool(getattr(c, "img_pretrained", True))
                extra["img_ppt"] = float(getattr(c, "img_ppt", 1.0))  # E34: resolution of the new image branch
                extra["img_max_px"] = int(getattr(c, "img_max_px", 1536))
            model = from_config({**cfg, **extra}, weights_follow=not extra.get("image_branch"))
            load_weights(model, ck["model"], new_heads=bool(extra))
            self.log(f"init from {init}")
            return model
        model = build(c.arch, N_FEATURES, dims=c.dims, enc_depths=c.enc_depths, dec_depths=c.dec_depths,
                      patch=c.patch, drop_path=c.drop_path, layer_pool=c.layer_pool, dim=c.dim, depth=c.depth, k=c.k,
                      mode_input=getattr(c, "mode_input", False), layer_names=getattr(c, "layer_names", False),
                      group_context=getattr(c, "group_context", False), relations=getattr(c, "relations", 0),
                      feature_drop=getattr(c, "feature_drop", None) or None,
                      pretrain=float(getattr(c, "pretrain", 0) or 0) > 0,
                      layer_head=bool(getattr(c, "layer_head", False)),
                      layer_head_names=bool(getattr(c, "layer_head_names", False)),
                      image_branch=bool(getattr(c, "image_branch", False)),
                      img_pretrained=bool(getattr(c, "img_pretrained", True)),
                      img_lum=bool(getattr(c, "img_lum", False)), pool_by=str(getattr(c, "pool_by", "layer")),
                      img_ppt=float(getattr(c, "img_ppt", 1.0)), img_max_px=int(getattr(c, "img_max_px", 1536)))
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
        aug = Augment(fliplr=c.fliplr, rot90=c.rot90, scale=c.scale, color=c.color, gray=c.gray, restyle=c.restyle,
                      restyle_mode=str(getattr(c, "restyle_mode", "free")),
                      name_drop=getattr(c, "name_drop", 0.0)
                      if getattr(c, "layer_names", False) or getattr(c, "layer_head_names", False) else 0.0,
                      drop_tokens=c.drop_tokens)
        gpu_batch = use_gpu_batch(c, device)
        co = float(getattr(c, "coincide", 0) or 0)
        for p in train_pages:  # E36: no loss on true shapes that only lie under a pipe (their group is mostly background)
            p.lw = coincident_weights(p, co) if co > 0 else None
        if co > 0:
            n0 = sum(float(p.length[p.lw == 0].sum()) for p in train_pages)
            n1 = sum(float(p.length[p.label > 0].sum()) for p in train_pages)
            self.say(f"coincide {co}: {n0 / max(n1, 1e-9):.1%} of the true length of the train pages carries no loss")
        ds = WindowDataset(train_pages, c.steps * c.batch, c.window, aug, seed=c.seed + 1000 * fold,
                           plumbing_share=getattr(c, "plumbing_share", None), raw=gpu_batch)
        dl = torch.utils.data.DataLoader(ds, batch_size=c.batch, shuffle=False, num_workers=c.workers,
                                         collate_fn=raw_items if gpu_batch else collate,
                                         pin_memory=device.type == "cuda" and not gpu_batch,
                                         persistent_workers=False, prefetch_factor=4 if c.workers else None)
        if fold == 0 or start_epoch == 0:
            self.say(f"batches built on {'the GPU (gpu_batch)' if gpu_batch else 'CPU workers'}")
        total_iters = c.epochs * c.steps
        warmup = int(c.warmup_epochs * c.steps)
        mae_ratio = float(getattr(c, "pretrain", 0) or 0)  # E19: masked-shape pretraining instead of the label loss
        cols = ["mse"] if mae_ratio > 0 else (["bce", "lovasz"] if c.lovasz > 0 else ["bce"])
        if mae_ratio <= 0 and getattr(c, "layer_head", False):
            cols = cols + ["layer"]
        if mae_ratio > 0:
            self.say(f"pretrain masked shapes: {mae_ratio:.0%} of tokens hidden, recon of their features (labels unused)")
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
                    g["lr"] = c.lr0 * lr_factor(it, total_iters, warmup, c.lrf, c.cos_lr) * g.get("lr_scale", 1.0)
                if gpu_batch:
                    b = build_batch([w for w, _ in b], device, aug, [s for _, s in b])
                else:
                    b = to_device(b, device)
                if mae_ratio > 0:
                    b["mae"] = (torch.rand(b["mask0"].shape, device=device) < mae_ratio) & (b["mask0"] > 0)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                    logits = model(b)
                if mae_ratio > 0:
                    err = ((logits.float() - model.target_features(b)) ** 2).mean(-1)
                    total = err[b["mae"]].mean()
                    parts = {"mse": total}
                else:
                    total, parts = loss_fn(c.arch, logits, b, c.pos_weight, c.alpha, c.lovasz)
                    if getattr(model, "layer_mlp", None) is not None and "ylayer" in b:
                        lb = layer_loss(model.last_layer, b)
                        total = total + float(getattr(c, "layer_loss", 0.5)) * lb
                        parts = {**parts, "layer": lb}
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
            if mae_ratio > 0:  # no labels to score: keep the latest EMA weights as the start for fine-tuning
                best, best_epoch = -1.0, epoch
                torch.save({"model": ema.module.state_dict(), "config": ema.module.config, "epoch": epoch,
                            "threshold": 0.5, "val_f2": -1.0, "feature_names": FEATURE_NAMES, "args": vars(c)},
                           out / "best.pt")
            elif val_pages and ((epoch + 1) % c.val_every == 0 or epoch + 1 == c.epochs):
                tv = time.time()
                ema.repair(model, self.log)
                vscores, bst, at05 = evaluate(ema.module, val_pages, c, device, amp_dtype)
                rec.update({"P": round(bst["precision"], 4), "R": round(bst["recall"], 4), "F2": round(bst["f2"], 4),
                            "F2@0.5": round(at05["f2"], 4), "thr": bst["threshold"]})
                print(VAL_HEAD + f"  ({dur(time.time() - tv)})")
                print(val_row("all", len(val_pages), sum(len(p) for p in val_pages),
                              sum(float(p.length[p.label > 0].sum()) for p in val_pages),
                              bst["precision"], bst["recall"], bst["f2"], at05["f2"], bst["threshold"]), flush=True)
                if len(val_pages) <= 12:  # small val (dev split): every page, to see which one collapses
                    for pg, sc, row in zip(val_pages, vscores, per_page(vscores, [q.label for q in val_pages],
                                                                         [q.length for q in val_pages],
                                                                         [q.stem for q in val_pages], bst["threshold"])):
                        r05 = prf(*length_counts(sc >= 0.5, pg.label, pg.length))
                        print(val_row(pg.stem.split("_gt")[0][:12], 1, len(pg), float(pg.length[pg.label > 0].sum()),
                                      row["precision"], row["recall"], row["f2"], r05["f2"], bst["threshold"]), flush=True)
                        rec[f"F2:{pg.stem}"] = round(row["f2"], 4)
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
            if mae_ratio <= 0 and c.patience and best_epoch >= 0 and epoch - best_epoch >= c.patience:
                self.say(f"Stopping early: no better val F2 for {c.patience} epochs (best {best:.4f} at epoch "
                         f"{best_epoch + 1}).")
                break
        return self.finish_fold(fold, model, [] if mae_ratio > 0 else val_pages, out, device, amp_dtype, best,
                                best_epoch, nparams)

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
        m = from_config(ck["config"], weights_follow=True)
        load_weights(m, ck["model"])
        m.to(device)
        tta = 8 if c.tta else 1
        lscores = [] if getattr(m, "layer_mlp", None) is not None else None
        scores, bst, at05 = evaluate(m, val_pages, c, device, amp_dtype, tta=tta, layer_out=lscores)
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
                            **{f"{p.stem}__index": p.index for p in val_pages},
                            **({f"{s}__layer": ls.astype(np.float32) for s, ls in zip(stems, lscores)} if lscores else {}))
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
    if not rows:  # a pretraining run (E19) scores nothing
        say("\nno scored validation (pretraining run: fine-tune it with model=<this run folder>)")
        return summary
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
