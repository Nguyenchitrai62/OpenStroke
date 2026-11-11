"""Full-page GT from the uploaded dataset's raw vector IDs and label-row classes.

No separate pack, image-box IoU, layer heuristics or polygon re-assignment. Length uses
an optional exact `length` array, otherwise the same eight sampled points as the model.
"""
from pathlib import Path

import numpy as np

from openstroke.assemble import assign_members
from openstroke.objects import HVAC_CLASSES, object_counts
from openstroke.obb.object_score import CONFS, PIPELINE_PAGES, report


def companion(path, kind, suffix):
    parts = list(Path(path).parts)
    positions = [i for i, part in enumerate(parts) if part == "images"]
    if not positions:
        raise ValueError(f"Vector evaluation needs an images/<split> dataset layout: {path}")
    parts[positions[-1]] = kind
    return Path(*parts).with_suffix(suffix)


class DatasetVectorGT:
    def __init__(self, images, names, comparison_pages=PIPELINE_PAGES):
        self.pages = {}
        self.coverage = {}
        self.comparison_pages = tuple(comparison_pages)
        canonical = {name.casefold(): i for i, name in enumerate(HVAC_CLASSES)}
        try:
            classes = [canonical[name.replace("_", " ").casefold()] for name in names]
        except KeyError as exc:
            raise ValueError(f"Unknown HVAC class in vector dataset: {exc}") from exc
        exact = True
        files = sorted(p for p in Path(images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        for image in files:
            vector, label = companion(image, "vectors", ".npz"), companion(image, "labels", ".txt")
            if not vector.is_file() or not label.is_file():
                raise FileNotFoundError(f"Vector GT requires both {vector} and {label}")
            rows = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            try:
                ids = [int(row[0]) for row in rows]
                if any(len(row) != 9 or not 0 <= c < len(classes) for row, c in zip(rows, ids)):
                    raise ValueError("invalid class or label row")
                gt_class = {i: classes[c] for i, c in enumerate(ids)}
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Invalid vector GT class rows: {label}") from exc
            with np.load(vector) as data:
                tok = np.asarray(data["tokens"])
                if tok.ndim != 2 or tok.shape[1] < 24 or not np.isfinite(tok[:, :24]).all():
                    raise ValueError(f"Vector GT needs finite 24-column tokens with inst/pid: {vector}")
                inst, pid = tok[:, 22], tok[:, 23]
                if (np.any(inst != np.rint(inst)) or np.any(inst < -1) or np.any(inst >= len(rows))
                        or np.any(pid != np.rint(pid)) or np.any(pid < 0) or len(np.unique(pid)) != len(pid)):
                    raise ValueError(f"Invalid GT object/primitive IDs: {vector}")
                inst, pid = inst.astype(np.int64), pid.astype(np.int64)
                if "length" in data:
                    length = np.asarray(data["length"], np.float64).copy()
                else:
                    pts = tok[:, 1:17].astype(np.float64).reshape(-1, 8, 2)
                    length = np.linalg.norm(np.diff(pts, axis=1), axis=2).sum(axis=1)
                    exact = False
            if length.shape != (len(tok),) or not np.isfinite(length).all() or (length < 0).any():
                raise ValueError(f"Invalid vector lengths: {vector}")
            represented = len(np.unique(inst[inst >= 0]))
            self.coverage[image.stem] = dict(label_objects=len(rows), objects_with_strokes=represented,
                                              objects_without_strokes=len(rows)-represented)
            self.pages[image.stem] = (length, inst, gt_class, {int(p): i for i, p in enumerate(pid)})
        missing = set(self.comparison_pages) - {stem[:9] for stem in self.pages}
        if missing:
            raise ValueError(f"Dataset lacks fixed comparison pages: {sorted(missing)}")
        if not self.pages:
            raise ValueError(f"No full-page vector GT in {images}")
        self.protocol = "dataset_vectors.exact_length.v1" if exact else "dataset_vectors.sampled8.v1"

    def evaluate(self, detections, how, names):
        missing = self.pages.keys() - detections.keys()
        if missing:
            raise ValueError(f"Vector predictions lack dataset pages: {sorted(missing)}")
        class_ids = {name.replace("_", " ").casefold(): i for i, name in enumerate(HVAC_CLASSES)}
        result = {conf: {} for conf in CONFS}
        stroke_lengths = {conf: {} for conf in CONFS}
        for stem, (length, gt, gt_class, at) in self.pages.items():
            objects = []
            for pred in detections[stem]:
                if (pred.get("source") or "members") != (how if how in ("graph",) or how.startswith("seeded") else "members"):
                    continue
                cls = int(pred["class_id"])
                if not 0 <= cls < len(names):
                    raise ValueError(f"Invalid predicted class for {stem}: {cls}")
                members = pred.get("members") or []
                unknown = set(members) - at.keys()
                if unknown:
                    raise ValueError(f"Prediction has unknown primitive IDs for {stem}: {sorted(unknown)[:5]}")
                objects.append((class_ids[names[cls].replace("_", " ").casefold()], np.empty((0, 2)),
                                float(pred["score"]), [at[p] for p in members]))
            for conf in CONFS:
                keep = [obj for obj in objects if obj[2] >= conf]
                pred = assign_members(keep, length, order="score", dedupe=False)
                pred_cls = np.asarray([obj[0] for obj in keep], np.int64)
                # Background-only rows cannot affect counts; avoid repeated scans of the whole PDF for every object.
                relevant = (pred >= 0) | (gt >= 0)
                stroke_lengths[conf][stem] = (length[(pred >= 0) & (gt >= 0)].sum(),
                                              length[pred >= 0].sum(), length[gt >= 0].sum())
                result[conf][stem] = object_counts(pred[relevant], pred_cls, gt[relevant], gt_class, length[relevant])
        summary = report(result, how, log=lambda _: None, comparison_pages=self.comparison_pages, reference=False)
        values = stroke_lengths[summary["best_conf"]]
        total = np.sum([v for stem, v in values.items() if stem[:9] in self.comparison_pages], axis=0)
        tp, pred_length, gt_length = total
        p, rec = float(tp / max(pred_length, 1e-9)), float(tp / max(gt_length, 1e-9))
        summary.update(protocol=self.protocol, gt_source="dataset_vectors", page_count=len(self.pages), gt_coverage=self.coverage,
                       stroke_length=dict(P=p, R=rec, F2=5*p*rec/max(4*p+rec,1e-9)))
        return summary
