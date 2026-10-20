"""Train / evaluate OpenStroke v1 with cross-validation by PDF (folds.json).

``py -m openstroke train --data datasets/e0 --folds datasets/e0/folds.json --fold all --out runs/v1``

For each validation fold: fit on the other folds, choose the threshold by length-weighted F2 on the
validation pages, write ``fold<k>/best.pt`` + ``fold<k>/result.json``. ``summary.json`` then sums
(micro) every fold's validation pages, at 0.5 and at each fold's own threshold.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, FEATURE_VERSION, N_FEATURES, PageTokens, page_tokens
from .folds import load as load_folds
from .folds import resolve
from .metrics import per_page, prf, sweep

DIR_COS, DIR_SIN = FEATURE_NAMES.index("dir_cos2"), FEATURE_NAMES.index("dir_sin2")


def cached_tokens(data: Path, stem: str, cache: Path, mode: str) -> PageTokens:
    f = cache / f"{stem}.{FEATURE_VERSION}.npz"
    if f.exists():
        z = np.load(f)
        return PageTokens(stem=stem, mode=mode, x=z["x"], xy=z["xy"], layer=z["layer"], label=z["label"],
                          length=z["length"], keep=z["keep"])
    t = page_tokens(data / f"{stem}.npz", mode)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(f, x=t.x, xy=t.xy, layer=t.layer, label=t.label, length=t.length, keep=t.keep)
    return t


@dataclass
class Serialized:
    """A page reduced to scored tokens (with geometry) in Morton order."""

    stem: str
    mode: str
    x: np.ndarray
    xy: np.ndarray
    layer: np.ndarray
    label: np.ndarray
    length: np.ndarray
    index: np.ndarray  # position of each token in the original shape list


def serialize(t: PageTokens) -> Serialized:
    from .model import morton_order

    idx = np.flatnonzero(t.keep)
    order = idx[morton_order(t.xy[idx])] if len(idx) else idx
    return Serialized(t.stem, t.mode, t.x[order], t.xy[order], t.layer[order], t.label[order], t.length[order], order)


def window_arrays(p: Serialized, start: int, n: int, rng: np.random.Generator | None = None):
    sl = slice(start, min(start + n, len(p.x)))
    x = p.x[sl].copy()
    xy = p.xy[sl].astype(np.float64)
    xy = xy - (xy.min(axis=0) + xy.max(axis=0)) / 2
    if rng is not None:  # random flip / quarter turn; doubled-angle direction features follow
        if rng.random() < 0.5:
            xy[:, 0] = -xy[:, 0]
            x[:, DIR_SIN] = -x[:, DIR_SIN]
        if rng.random() < 0.5:
            xy = np.stack([-xy[:, 1], xy[:, 0]], axis=1)
            x[:, DIR_COS] = -x[:, DIR_COS]
            x[:, DIR_SIN] = -x[:, DIR_SIN]
    _, layer = np.unique(p.layer[sl], return_inverse=True)
    return x, xy.astype(np.float32), layer.reshape(-1), p.label[sl], p.length[sl]


def collate(items, n: int):
    import torch

    B = len(items)
    X = np.zeros((B, n, N_FEATURES), np.float32)
    XY = np.zeros((B, n, 2), np.float32)
    LY = np.zeros((B, n), np.int64)
    Y = np.zeros((B, n), np.float32)
    LEN = np.zeros((B, n), np.float32)
    M = np.zeros((B, n), bool)
    for b, (x, xy, ly, y, ln) in enumerate(items):
        m = len(x)
        X[b, :m], XY[b, :m], LY[b, :m], Y[b, :m], LEN[b, :m], M[b, :m] = x, xy, ly, y, ln, True
    return tuple(torch.from_numpy(a) for a in (X, XY, LY, Y, LEN, M))


def predict_page(model, p: Serialized, n: int, device, batch: int = 8) -> np.ndarray:
    """Probabilities for every token: windows of ``n`` with stride ``n/2``, overlaps averaged."""

    import torch

    total = np.zeros(len(p.x), np.float64)
    count = np.zeros(len(p.x), np.float64)
    stride = max(n // 2, 1)
    starts = list(range(0, max(len(p.x) - n, 0) + 1, stride))
    if starts and starts[-1] + n < len(p.x):
        starts.append(len(p.x) - n)
    model.eval()
    with torch.no_grad():
        for a in range(0, len(starts), batch):
            chunk = starts[a:a + batch]
            items = [window_arrays(p, s, n) for s in chunk]
            X, XY, LY, _, _, M = (t.to(device) for t in collate(items, n))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prob = torch.sigmoid(model(X, XY, LY, M).float()).cpu().numpy()
            for s, pr, it in zip(chunk, prob, items):
                m = len(it[0])
                total[s:s + m] += pr[:m]
                count[s:s + m] += 1
    return total / np.maximum(count, 1)


ARCHS = ("openstroke1",)


def init_checkpoint(model_arg: str, fold: int) -> Path | None:
    """``model=`` -> checkpoint to start fold ``fold`` from, or None for a fresh model.

    ``openstroke1`` trains from scratch. A run folder (``runs/train``) gives each fold its *own*
    ``fold<k>/best.pt``, so no fold starts from weights that saw its validation pages. A single ``.pt``
    is used for every fold (the caller warns: that leaks validation pages into training).
    """

    if model_arg in ARCHS:
        return None
    p = Path(model_arg)
    if p.is_dir():
        ck = p / f"fold{fold}" / "best.pt"
        if not ck.exists():
            raise FileNotFoundError(f"model={model_arg}: no {ck}")
        return ck
    if p.suffix == ".pt" and p.exists():
        return p
    raise ValueError(f"model={model_arg!r}: expected one of {ARCHS}, a run folder or an existing .pt")


def run_fold(pages: dict[str, Serialized], train: list[str], val: list[str], args, out: Path, log,
             fold: int = 0) -> dict:
    import torch

    from .model import OpenStroke1, loss_fn

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    ck_path = init_checkpoint(getattr(args, "model", "openstroke1"), fold)
    if ck_path is None:
        model = OpenStroke1(N_FEATURES, dim=args.dim, depth=args.depth, heads=args.heads, k=args.k,
                            layer_pool=not args.no_layer_pool).to(device)
        xs = np.concatenate([pages[s].x for s in train])
        model.set_normalization(xs.mean(axis=0), xs.std(axis=0))
        del xs
    else:  # fine-tune: architecture and feature normalization come from the checkpoint
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        if ck.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"{ck_path}: feature version {ck.get('feature_version')} != {FEATURE_VERSION}")
        model = OpenStroke1(**ck["config"]).to(device)
        model.load_state_dict(ck["model"])
        log(f"init from {ck_path}")
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * args.steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / max(args.warmup, 1)) * 0.5 * (1 + math.cos(math.pi * min(i / total_steps, 1.0))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    sizes = np.array([len(pages[s].x) for s in train], np.float64)
    prob = np.sqrt(sizes) / np.sqrt(sizes).sum()  # small pages still get seen
    c = model.config
    log(f"model OpenStroke1 dim={c['dim']} depth={c['depth']} k={c['k']} layer_pool={c['layer_pool']} "
        f"params={nparams / 1e6:.2f}M device={device} train_pages={len(train)} val_pages={len(val)} "
        f"train_tokens={int(sizes.sum()):,}")
    best = {"f2": -1.0}
    history = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        losses = []
        for _ in range(args.steps):
            items = []
            for _ in range(args.batch):
                p = pages[train[rng.choice(len(train), p=prob)]]
                start = int(rng.integers(0, max(len(p.x) - args.window, 0) + 1))
                items.append(window_arrays(p, start, args.window, rng))
            X, XY, LY, Y, LEN, M = (t.to(device, non_blocking=True) for t in collate(items, args.window))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(X, XY, LY, M)
            loss = loss_fn(logits, Y, LEN, M, pos_weight=args.pos_weight, alpha=args.alpha)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()))
        row = {"epoch": ep + 1, "loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1)}
        if val and ((ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs):
            scores = [predict_page(model, pages[s], args.window, device) for s in val]
            sw = sweep(scores, [pages[s].label for s in val], [pages[s].length for s in val])
            row.update({"val_" + k: v for k, v in sw["best"].items()})
            at05 = next(r for r in sw["curve"] if abs(r["threshold"] - 0.5) < 1e-9)
            row["val_f2@0.5"] = at05["f2"]
            if sw["best"]["f2"] > best["f2"]:
                best = {**sw["best"], "epoch": ep + 1}
                torch.save({"model": model.state_dict(), "config": model.config, "threshold": sw["best"]["threshold"],
                            "feature_version": FEATURE_VERSION, "feature_names": FEATURE_NAMES, "args": vars(args)},
                           out / "best.pt")
        history.append(row)
        log("epoch {epoch:3d} loss {loss:.4f} ".format(**row)
            + (f"val R {row['val_recall']:.3f} P {row['val_precision']:.3f} F2 {row['val_f2']:.3f} "
               f"@t={row['val_threshold']:.2f} (F2@0.5 {row['val_f2@0.5']:.3f}) " if "val_f2" in row else "")
            + f"{row['seconds']}s")
    torch.save({"model": model.state_dict(), "config": model.config, "feature_version": FEATURE_VERSION,
                "feature_names": FEATURE_NAMES, "args": vars(args)}, out / "last.pt")
    result = {"train": train, "val": val, "params": nparams, "history": history, "best": best}
    if val and (out / "best.pt").exists():
        ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        scores = [predict_page(model, pages[s], args.window, device) for s in val]
        result["val_pages"] = per_page(scores, [pages[s].label for s in val], [pages[s].length for s in val], val,
                                       ck["threshold"])
        result["val_pages_at_0.5"] = per_page(scores, [pages[s].label for s in val], [pages[s].length for s in val], val, 0.5)
        np.savez_compressed(out / "val_scores.npz", **{s: sc.astype(np.float32) for s, sc in zip(val, scores)},
                            **{f"{s}__index": pages[s].index for s in val})
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def summarize(results: dict[int, dict]) -> dict:
    """Per-page rows of every fold, at the fold's threshold and at 0.5."""

    return {key: [{"fold": f, **p} for f, r in sorted(results.items()) for p in r.get(key, [])]
            for key in ("val_pages", "val_pages_at_0.5")}


def next_run_dir(path: Path) -> Path:
    """``runs/train`` if unused, else ``runs/train2``, ``train3``... (as YOLO / OpenOBB)."""

    if not (path / "train.log").exists():
        return path
    i = 2
    while (path.parent / f"{path.name}{i}" / "train.log").exists():
        i += 1
    return path.parent / f"{path.name}{i}"


def main_train(args) -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    folds_path = args.folds or str(data / "folds.json")
    folds = load_folds(folds_path)
    model_arg = getattr(args, "model", "openstroke1")
    out_root = next_run_dir(Path(args.out)) if not getattr(args, "exist_ok", False) else Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    logf = open(out_root / "train.log", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    entries, warn = resolve(manifest, folds)
    log(f"openstroke train: data={data} folds={folds_path} model={model_arg} status={folds.get('status')} pages={len(entries)} "
        f"feature_version={FEATURE_VERSION}")
    if folds.get("status") != "confirmed":
        log("WARNING: fold file is not confirmed by the user (status != 'confirmed'); numbers are provisional")
    for w in warn:
        log("WARNING: " + w)
    if model_arg not in ARCHS and Path(model_arg).suffix == ".pt":
        log("WARNING: model=<one .pt> starts every fold from the same weights; folds whose pages it was trained on "
            "are no longer held out. Use model=<run folder> to start each fold from its own best.pt.")
    cache = Path(args.cache) if args.cache else out_root / "cache"
    pages = {}
    for stem, _, mode in entries:
        pages[stem] = serialize(cached_tokens(data, stem, cache, mode))
    fold_ids = sorted({f for _, f, _ in entries})
    run = fold_ids if args.fold == "all" else [int(f) for f in str(args.fold).split(",")]
    results = {}
    for f in run:
        val = [s for s, ff, _ in entries if ff == f]
        train = [s for s, ff, _ in entries if ff != f]
        if args.modes != "both":
            train = [s for s in train if pages[s].mode == args.modes]
            val = [s for s in val if pages[s].mode == args.modes]
        fo = out_root / f"fold{f}"
        fo.mkdir(parents=True, exist_ok=True)
        log(f"=== fold {f}: val {val}")
        results[f] = run_fold(pages, train, val, args, fo, log, fold=f)
    # micro over all validation pages, recomputed from the saved scores
    summary = {"folds": run, "args": vars(args), "fold_status": folds.get("status")}
    for label, use_fold_t in (("at_fold_threshold", True), ("at_0.5", False)):
        tp = fp = fn = 0.0
        per_mode: dict[str, list[float]] = {}
        for f in run:
            fo = out_root / f"fold{f}"
            if not (fo / "val_scores.npz").exists():
                continue
            import torch

            t = torch.load(fo / "best.pt", map_location="cpu", weights_only=False)["threshold"] if use_fold_t else 0.5
            z = np.load(fo / "val_scores.npz")
            for s in results[f]["val"]:
                pr = z[s] >= t
                y = pages[s].label.astype(bool)
                L = pages[s].length.astype(np.float64)
                a, b, c = L[pr & y].sum(), L[pr & ~y].sum(), L[~pr & y].sum()
                tp, fp, fn = tp + a, fp + b, fn + c
                m = per_mode.setdefault(pages[s].mode, [0.0, 0.0, 0.0])
                m[0] += a
                m[1] += b
                m[2] += c
        summary[label] = {"all": prf(tp, fp, fn), **{k: prf(*v) for k, v in per_mode.items()}}
        log(f"SUMMARY {label}: " + "  ".join(f"{k}: R {v['recall']:.3f} P {v['precision']:.3f} F2 {v['f2']:.3f}"
                                             for k, v in summary[label].items()))
    summary["pages"] = summarize(results)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for f in run:
        for p in results[f].get("val_pages", []):
            log(f"PAGE fold{f} {p['stem']}: R {p['recall']:.3f} P {p['precision']:.3f} F2 {p['f2']:.3f} "
                f"true {p['true_length_share']:.1%} pred {p['pred_length_share']:.1%}")
    return 0


def add_args(ap) -> None:
    ap.add_argument("--data", default="datasets/e0")
    ap.add_argument("--folds", default=None, help="fold file (default: <data>/folds.json)")
    ap.add_argument("--model", default="openstroke1",
                    help="'openstroke1' (new), a run folder (each fold starts from its fold<k>/best.pt) or a .pt")
    ap.add_argument("--fold", default="all", help="'all' or comma-separated validation fold ids")
    ap.add_argument("--modes", default="both", choices=("both", "fire", "hvac"))
    ap.add_argument("--out", default="runs/train", help="run folder; taken -> train2, train3...")
    ap.add_argument("--exist-ok", action="store_true", help="write into --out even if it already holds a run")
    ap.add_argument("--cache", default="runs/cache", help="feature cache shared by runs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=200, help="optimizer steps per epoch")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--no-layer-pool", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5, help="loss weight = (length + 1) ** alpha")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
