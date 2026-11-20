"""Paired stroke comparison on the frozen hvac10b validation cohort. Inference only."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from .folds import load as load_folds, resolve
from .metrics import length_counts, prf
from .region_data import gt_entries
from .score_all import read_meta, score_all


THRESHOLD = 0.20  # E129 epoch-zero GT-selected threshold, fixed for both models.


def length_weighted_ap(score, label, length):
    order = np.argsort(-score, kind="stable")
    ranked, weight = score[order], length[order].astype(float)
    truth = np.cumsum(weight * (label[order] > 0))
    if not len(order) or truth[-1] <= 0:
        return 0.0
    # Add each tied score as one step in the PR curve.
    end = np.flatnonzero(np.r_[ranked[:-1] != ranked[1:], True])
    tp = truth[end]
    precision = tp / np.maximum(np.cumsum(weight)[end], 1e-9)
    return float(np.dot(np.diff(np.r_[0, tp]) / truth[-1], precision))


def report(data, records, base_path, last_path):
    totals, aps = {}, {}
    with np.load(base_path, allow_pickle=False) as first, np.load(last_path, allow_pickle=False) as last:
        archives = (first, last)
        stems = {r["stem"] for r in records}
        for sc in archives:
            meta = read_meta(sc)
            if set(meta["pages"]) != stems or meta["tta"] != 1 or any(p["split"] != "val" for p in meta["pages"].values()):
                raise ValueError("Expected complete hvac10b validation scores at TTA x1")
        print(f"Length-weighted stroke F2/P/R at shared threshold {THRESHOLD:.2f}, plus average precision over all thresholds")
        for rec in records:
            stem = rec["stem"]
            idx = first[f"{stem}__index"]
            if not np.array_equal(idx, last[f"{stem}__index"]):
                raise ValueError(f"{stem}: payload IDs differ between runs")
            with np.load(data / rec["npz"], allow_pickle=False) as source:
                label, length = source["label"][idx], source["length"][idx]
            counts = np.array([length_counts(sc[stem] >= THRESHOLD, label, length) for sc in archives])
            cohort = "Viewer" if stem.startswith("vw_") else "GT"
            totals[cohort] = totals.get(cohort, np.zeros((2, 3))) + counts
            ap = [length_weighted_ap(sc[stem], label, length) for sc in archives]
            aps.setdefault(cohort, []).append(ap)
            print(f"{cohort:6s} {stem}  {comparison(counts)}  AP {ap[0]:.3f} -> {ap[1]:.3f}")
    for group in ("GT", "Viewer"):
        a, b = np.mean(aps[group], axis=0)
        print(f"{group} micro  {comparison(totals[group])}  mean page AP {a:.3f} -> {b:.3f}")


def comparison(counts):
    base, tuned = (prf(*row) for row in counts)

    def display(row):
        return f"{row['f2']:.3f}/{row['precision']:.3f}/{row['recall']:.3f}"
    return f"E114 {display(base)} -> E129 last {display(tuned)}  delta F2 {tuned['f2'] - base['f2']:+.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("e114", type=Path)
    parser.add_argument("e129_last", type=Path)
    args = parser.parse_args()
    if not args.e114.is_file() or not args.e129_last.is_file() or args.e129_last.name != "last.pt":
        parser.error("Pass the original E114 best.pt and E129's last.pt")
    import torch

    if not torch.cuda.is_available():
        parser.error("Use a GPU runtime for scoring both checkpoints")
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    folds = args.data / "folds.json"
    entries, _ = resolve(manifest, load_folds(folds))
    gt, _ = gt_entries(args.data, manifest, entries, require_pieces=False)
    gt_val = {s for s, f, _ in gt if f == 0}
    viewer_val = {s for s, f, mode in entries if f == 0 and mode == "hvac" and s.startswith("vw_")}
    if len(viewer_val) != 8:
        raise ValueError("Expected the eight historical Viewer validation pages")
    val = gt_val | viewer_val
    records = [r for r in manifest["records"] if r["status"] == "ok" and r["stem"] in val]
    # Separate temporary parts prevent stale scores from a prior checkpoint or another cohort.
    with tempfile.TemporaryDirectory(prefix="openstroke_e129_compare_") as folder:
        outputs = [
            score_all(checkpoint, args.data, Path(folder) / name, folds=folds, cache=args.data.parent / "cache" / "hvac_gt",
                      tta=1, device="cuda",
                      viewer=True, split="val", log=lambda msg: print(msg, flush=True))
            for checkpoint, name in ((args.e114, "e114.npz"), (args.e129_last, "e129_last.npz"))
        ]
        report(args.data, records, *outputs)


if __name__ == "__main__":
    main()
