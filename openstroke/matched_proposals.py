"""Prepare the GT-only E133 ablation without modifying the original e12 dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .folds import load as load_folds, resolve
from .region_data import GT_SPLIT, gt_entries

THRESHOLD = 0.75
TTA = 8
E122_SHA256 = "c71e7c7f7697124810a605e8f8b9da7088bfd93b8e857534c73e52ce9e468d36"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cohort(data):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    entries, _ = resolve(manifest, load_folds(data / "folds.json"))
    gt, proof = gt_entries(data, manifest, entries)
    val_seed = manifest.get("e12", {}).get("val_seed", {})
    if (val_seed.get("threshold") != THRESHOLD or
            not str(val_seed.get("scores", "")).replace(chr(92), "/").endswith("e122/fold0/val_scores.npz")):
        raise ValueError("Expected the original e12 rebuilt using E122 validation proposals")
    return manifest, {stem: fold for stem, fold, _ in gt}, [list(row) for row in proof["rows"]]


def pack(data, source, output):
    """Export existing GT polygons for train only, preserving indices used by e12."""
    manifest, split, rows = cohort(data)
    source_pages = {r["stem"]: r for r in load_folds(source / "pages.json")}
    records = [r for r in manifest["records"] if r["status"] == "ok" and r["stem"] in split]
    artifact = {"cohort": GT_SPLIT, "source": "ai_takeoff_gt", "train": [r for r in rows if r[-1] == "train"],
                "folds": sha256(data / "folds.json"), "npz": {r["npz"]: sha256(data / r["npz"]) for r in records}}
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        tmp = Path(scratch) / output.name
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(artifact))
            for stem, _, page, _ in artifact["train"]:
                record = source_pages[stem]
                if record.get("pseudo") or record["split"] != "train" or record["page_number"] != page:
                    raise ValueError(f"{stem}: expected an AI_Takeoff GT train page in the score pack")
                objects = load_folds(source / "gt" / f"{stem}.json")
                if not objects or any(o["group"] not in {"primary", "reference"} for o in objects):
                    raise ValueError(f"{stem}: unexpected original GT object groups")
                # Retain polygon ordering. No raw payload, layer names, or object metadata is needed.
                reduced = [{"shape_name": o.get("shape_name"), "vertices": o["vertices"]} for o in objects]
                archive.writestr(f"gt/{stem}.json", json.dumps(reduced, ensure_ascii=False))
        tmp.replace(output)
    print(f"Packed {len(artifact['train'])} GT train polygon files into {output} ({output.stat().st_size:,} bytes)")


def rebuild(source, record, scores, labels, out, rng):
    from .piece_candidates import labelled_pieces

    stem = record["stem"]
    with np.load(source, allow_pickle=False) as q:
        data = {k: q[k] for k in q.files if not k.startswith("pc_")}
        original_objects = len(np.unique(q["pc_inst"][q["pc_fg"] > 0]))
    idx, prob = scores[f"{stem}__index"], scores[stem]
    seed = np.zeros(len(data["label"]), dtype=bool)
    if idx.ndim != 1 or prob.shape != idx.shape or idx.dtype.kind not in "iu" or not len(idx):
        raise ValueError(f"{stem}: malformed stroke scores")
    if idx.min() < 0 or idx.max() >= len(seed) or len(np.unique(idx)) != len(idx) or not np.isfinite(prob).all():
        raise ValueError(f"{stem}: unexpected stroke score payload IDs")
    seed[idx] = prob >= THRESHOLD
    pieces, stats = labelled_pieces(data, labels, False, rng, seed)
    objects = len(np.unique(pieces["pc_inst"][pieces["pc_fg"] > 0]))
    data["pc_pick_thr"] = np.float32(THRESHOLD)
    np.savez_compressed(out, **data, **pieces)
    print(f"{stem}: {seed.sum():,} picked strokes, {stats['pieces']:,} pieces, "
          f"{stats['fg']:,} positive, GT with pieces {original_objects} -> {objects}", flush=True)


def prepare(data, labels, teacher, out):
    from .score_all import read_meta, score_all

    if out.exists():
        raise FileExistsError(f"{out} already exists. Reuse that prepared dataset or pick a new output path.")
    if sha256(teacher) != E122_SHA256:
        raise ValueError(f"{teacher}: expected the original E122 best.pt that seeded e12 validation")
    manifest, split, rows = cohort(data)
    train = {s for s, fold in split.items() if fold == -1}
    records = [r for r in manifest["records"] if r["status"] == "ok" and r["stem"] in split]
    with zipfile.ZipFile(labels) as archive, tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        pack_info = json.loads(archive.read("manifest.json"))
        expected = {"manifest.json"} | {f"gt/{s}.json" for s in train}
        if (pack_info["cohort"] != GT_SPLIT or pack_info["source"] != "ai_takeoff_gt" or
                pack_info["train"] != [r for r in rows if r[-1] == "train"] or
                set(archive.namelist()) != expected or len(archive.namelist()) != len(expected)):
            raise ValueError("Expected a polygon pack for exactly the 19 frozen AI_Takeoff train pages")
        if (sha256(data / "folds.json") != pack_info["folds"] or
                any(sha256(data / r["npz"]) != pack_info["npz"][r["npz"]] for r in records)):
            raise ValueError("The base files differ from the frozen e12 used for this experiment")
        scratch = Path(scratch)
        print("E122 candidate scores on AI_Takeoff GT train only, fold 0, TTA x8, threshold 0.75", flush=True)
        scores = score_all(teacher, data, scratch / "scores.npz", folds=data / "folds.json", cache=scratch / "cache",
                           device="cuda", tta=TTA, split="train", stroke_dtype=np.float32,
                           log=lambda line: print(line, flush=True))
        new = scratch / "data"
        new.mkdir()
        rng = np.random.default_rng(0)
        with np.load(scores, allow_pickle=False) as score:
            meta = read_meta(score)
            if (meta["tta"] != TTA or set(meta["pages"]) != train or
                    any(p["split"] != "train" or p["mode"] != "hvac" for p in meta["pages"].values())):
                raise ValueError("Expected E122 scores for exactly the 19 GT train pages")
            for r in records:
                src, dst = data / r["npz"], new / r["npz"]
                if r["stem"] in train:
                    objects = json.loads(archive.read(f"gt/{r['stem']}.json"))
                    rebuild(src, r, score, objects, dst, rng)
                else:
                    shutil.copyfile(src, dst)
        shutil.copyfile(data / "folds.json", new / "folds.json")
        manifest["records"] = records
        manifest["physical_pages"] = [p for p in manifest["physical_pages"] if any(s in split for s in p["gt_variants"])]
        manifest["e12"]["pieces"] = "Prediction-seeded train and val; GT is used only for target assignment"
        manifest["e12"]["val_seed"]["pages"] = len(split) - len(train)
        manifest["e12"]["train_seed"] = {"teacher_sha256": E122_SHA256, "tta": TTA, "threshold": THRESHOLD,
                                         "pages": len(train), "rng_seed": 0, "labels_sha256": sha256(labels)}
        (new / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        cohort(new)
        new.rename(out)
    print(f"Ready: {out} | 19 AI_Takeoff GT train, 5 unchanged GT val, 0 Viewer | split {GT_SPLIT}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name, paths in (("pack", ("data", "source", "output")), ("prepare", ("data", "labels", "teacher", "out"))):
        command = sub.add_parser(name)
        for path in paths:
            command.add_argument(path, type=Path)
        command.set_defaults(run=pack if name == "pack" else prepare)
    args = vars(parser.parse_args())
    run = args.pop("run")
    args.pop("action")
    run(**args)


if __name__ == "__main__":
    main()
