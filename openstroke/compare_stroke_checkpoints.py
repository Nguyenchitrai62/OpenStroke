"""Paired stroke comparison on the frozen hvac10b validation cohort. Inference only."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .folds import load as load_folds, resolve
from .metrics import length_counts, prf
from .region_data import gt_entries
from .score_all import read_meta, score_all


THRESHOLD = 0.20  # E129 epoch-zero GT-selected threshold, fixed for both models.
VIEWER_VALIDATION = frozenset((
    "vw_2f3d386b8610_p0004",
    "vw_579f78f50745_p0002",
    "vw_62b1ac808de5_p0002",
    "vw_62b1ac808de5_p0003",
    "vw_e241de8b8f63_p0001",
    "vw_e241de8b8f63_p0002",
    "vw_e3fe1f60a5ca_p0002",
    "vw_f4afa52daee9_p0004",
))


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


def paired_metrics(counts, aps):
    names = ("baseline", "candidate")
    result = {key: {k: float(v) for k, v in prf(*row).items()} for key, row in zip(names, counts)}
    for key, ap in zip(names, aps):
        result[key]["ap"] = float(ap)
    result["delta_f2"] = result["candidate"]["f2"] - result["baseline"]["f2"]
    return result


def comparison(result, baseline_label, candidate_label):
    def display(row):
        return f"{row['f2']:.3f}/{row['precision']:.3f}/{row['recall']:.3f}"

    return (f"{baseline_label} {display(result['baseline'])} -> {candidate_label} {display(result['candidate'])}"
            f"  delta F2 {result['delta_f2']:+.3f}")


def report(data, records, base_path, candidate_path, baseline_label, candidate_label, log):
    totals, aps, pages, groups = {}, {}, {}, {}
    with np.load(base_path, allow_pickle=False) as first, np.load(candidate_path, allow_pickle=False) as candidate:
        archives = (first, candidate)
        stems = {r["stem"] for r in records}
        for sc in archives:
            meta = read_meta(sc)
            if set(meta["pages"]) != stems or meta["tta"] != 1 or any(p["split"] != "val" for p in meta["pages"].values()):
                raise ValueError("Expected complete hvac10b validation scores at TTA x1")
        log(f"Length-weighted stroke F2/P/R at shared threshold {THRESHOLD:.2f}, plus average precision over all thresholds")
        for rec in records:
            stem = rec["stem"]
            idx = first[f"{stem}__index"]
            if not np.array_equal(idx, candidate[f"{stem}__index"]):
                raise ValueError(f"{stem}: payload IDs differ between runs")
            with np.load(data / rec["npz"], allow_pickle=False) as source:
                label, length = source["label"][idx], source["length"][idx]
            counts = np.array([length_counts(sc[stem] >= THRESHOLD, label, length) for sc in archives])
            cohort = "Viewer" if stem.startswith("vw_") else "GT"
            totals[cohort] = totals.get(cohort, np.zeros((2, 3))) + counts
            ap = [length_weighted_ap(sc[stem], label, length) for sc in archives]
            aps.setdefault(cohort, []).append(ap)
            row = pages[stem] = {"cohort": cohort, **paired_metrics(counts, ap)}
            log(f"{cohort:6s} {stem}  {comparison(row, baseline_label, candidate_label)}  AP {ap[0]:.3f} -> {ap[1]:.3f}")
    for group in ("GT", "Viewer"):
        ap = np.mean(aps[group], axis=0)
        row = groups[group] = {"pages": len(aps[group]), **paired_metrics(totals[group], ap)}
        log(f"{group} micro  {comparison(row, baseline_label, candidate_label)}  mean page AP {ap[0]:.3f} -> {ap[1]:.3f}")
    return {"groups": groups, "pages": pages}


def evaluate(data: Path, baseline: Path, candidate: Path, *, out: Path | None = None,
             folds: Path | None = None, cache: Path | None = None, device: str = "cuda",
             baseline_label: str = "E114", candidate_label: str = "candidate", log=print) -> Path | None:
    """Score the same cohort on CUDA and, if requested, save archives and the report together."""
    import torch

    for path in (baseline, candidate, data / "manifest.json", folds or data / "folds.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Use a GPU runtime for scoring both checkpoints")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    folds = folds or data / "folds.json"
    entries, _ = resolve(manifest, load_folds(folds))
    gt, proof = gt_entries(data, manifest, entries, require_pieces=False)
    gt_val = {s for s, f, _ in gt if f == 0}
    viewer_val = {s for s, f, mode in entries if f == 0 and mode == "hvac" and s.startswith("vw_")}
    if viewer_val != VIEWER_VALIDATION:
        raise ValueError("Expected the eight historical Viewer validation pages")
    val = gt_val | viewer_val
    records = [r for r in manifest["records"] if r["status"] == "ok" and r["stem"] in val]
    log(f"Paired stroke evaluation on hvac10b (GT5 + Viewer8 with partial labels): {baseline_label} vs {candidate_label}")
    # Separate temporary parts prevent stale scores from a prior checkpoint or another cohort.
    with tempfile.TemporaryDirectory(prefix="openstroke_compare_") as folder:
        outputs = [
            score_all(checkpoint, data, Path(folder) / name, folds=folds, cache=cache or data.parent / "cache" / "hvac_gt",
                      tta=1, device=device, viewer=True, split="val", log=log)
            for checkpoint, name in ((baseline, "e114.npz" if baseline_label == "E114" else "baseline.npz"),
                                     (candidate, "candidate.npz"))
        ]
        metrics = report(data, records, *outputs, baseline_label, candidate_label, log)
        if out is None:
            return None
        out.mkdir(parents=True, exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix="comparison_", dir=out.resolve()))
        for archive in outputs:
            shutil.copyfile(archive, destination / archive.name)
        payload = {
            "protocol": "hvac10b.stroke_comparison.v1",
            "threshold": THRESHOLD,
            "tta": 1,
            "split": "validation",
            "gt_split_sha256": proof["sha256"],
            "viewer_labels": "partial",
            "data": str(data),
            "folds": str(folds),
            "baseline": {"label": baseline_label, "checkpoint": str(baseline), "scores": outputs[0].name},
            "candidate": {"label": candidate_label, "checkpoint": str(candidate), "scores": outputs[1].name},
            "object_evaluation": {"status": "pending", "requires": "AI_Takeoff HVAC postprocessing"},
            **metrics,
        }
        # Publish the completion marker only after both archives are on Drive.
        staged = destination / "report.json.tmp"
        staged.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        staged.replace(destination / "report.json")
        log(f"Saved comparison report and raw stroke scores: {destination}")
        log("Object F2 pending AI_Takeoff HVAC postprocessing")
        return destination / "report.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("e114", type=Path, help="baseline checkpoint (E114 by default)")
    parser.add_argument("candidate", type=Path, help="stroke model checkpoint, for example E129 last.pt or E130 best.pt")
    parser.add_argument("--baseline-label", default="E114")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--out", type=Path, help="save the report and both score archives in a new subdirectory here")
    args = parser.parse_args()
    for name in ("manifest.json", "folds.json"):
        if not (args.data / name).is_file():
            parser.error(f"Missing {args.data / name}. Mount Drive and extract e12 in cell 3 of this Colab runtime.")
    try:
        evaluate(args.data, args.e114, args.candidate, out=args.out, baseline_label=args.baseline_label,
                 candidate_label=args.candidate_label, log=lambda msg: print(msg, flush=True))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
