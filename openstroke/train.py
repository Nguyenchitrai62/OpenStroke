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
    np.savez(f, x=t.x, xy=t.xy, layer=t.layer, label=t.label, length=t.length, keep=t.keep)
    return t


#: Per-model defaults (docs/OPENSTROKE1.md, docs/OPENSTROKE2.md §4: why these values).
DEFAULTS = {
    "openstroke1": dict(window=4096, batch=8, lr0=1e-3, epochs=30, lovasz=0.0, drop_path=0.0, val_batch=8,
                        restyle=0.0),
    "openstroke2": dict(window=65536, batch=4, lr0=2e-3, epochs=100, lovasz=1.0, drop_path=0.3, val_batch=2,
                        restyle=0.5),
}


def add_args(ap) -> None:
    a = ap.add_argument
    a("--data", default="datasets/e0")
    a("--folds", default=None, help="fold file (default: <data>/folds.json)")
    a("--fold", default="all", help="'all' or validation fold ids, e.g. 0,2")
    a("--modes", default="both", choices=("both", "fire", "hvac"))
    a("--model", default="openstroke2",
      help="openstroke1 | openstroke2 (new), a run folder (each fold starts from its fold<k>/best.pt) or a .pt")
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
    a("--seed", type=int, default=0)
    a("--parallel", type=int, default=1,
      help="run up to N folds at once, one process each on the same GPU (same results as one by one)")
    a("--child", type=int, default=None, help=argparse.SUPPRESS)  # internal: one fold of a parallel run
    a("--device", default="auto")
    # loss
    a("--pos-weight", type=float, default=2.0)
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
    a("--group-context", default="false", help="attention between layer groups and style groups (layer_1), zero-init (H16)")
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


def _ints(v):
    return [int(x) for x in str(v).split(",")] if isinstance(v, str) else list(v)


def finalize(args):
    arch = args.model if args.model in DEFAULTS else None
    if arch is None:  # fine-tune: architecture from the checkpoint
        import torch

        p = Path(args.model)
        ck = p / "fold0" / "best.pt" if p.is_dir() else p
        if not ck.exists():
            raise SystemExit(f"model={args.model}: expected openstroke1 | openstroke2, a run folder or a .pt")
        arch = torch.load(ck, map_location="cpu", weights_only=False)["config"].get("arch", "openstroke1")
    for k, v in DEFAULTS[arch].items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    args.arch = arch
    for k in ("cos_lr", "tta", "amp", "rot90", "layer_pool", "mode_input", "layer_names", "group_context"):
        setattr(args, k, _bool(getattr(args, k)))
    if args.no_layer_pool:
        args.layer_pool = False
    args.dims, args.enc_depths, args.dec_depths = _ints(args.dims), _ints(args.enc_depths), _ints(args.dec_depths)
    return args


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
    for k in ("cos_lr", "tta", "amp", "rot90", "layer_pool", "mode_input", "layer_names", "group_context"):
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
    out_root.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else project / "cache"
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
    if args.modes != "both":
        entries = [e for e in entries if e[2] == args.modes]
    if child is None:
        (out_root / "args.json").write_text(json.dumps(vars(args), indent=1, default=str), encoding="utf-8")
    say(f"OpenStroke train  {out_root}  model={args.model} ({args.arch})  data={data}  folds={folds_path.name} "
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
        f"group_context={args.group_context}  "
        f"plumbing_share={args.plumbing_share}")
    t0 = time.time()
    pages = {}
    for s, _, mode in entries:
        pages[s] = load_page(data, s, mode, cache)
    say(f"data    {len(pages)} pages, {sum(len(p) for p in pages.values()) / 1e6:.2f}M shapes "
        f"(features {time.time() - t0:.0f}s, cache {cache})")
    say(f"save    {out_root}/fold<k>/ (best.pt, last.pt, results.csv/png, result.json), summary.json")
    fold_ids = sorted({f for _, f, _ in entries})
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
