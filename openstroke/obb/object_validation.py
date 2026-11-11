"""Full-page vector object F2 (stroke membership, not box IoU) used to select EMA checkpoints during training."""
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from openstroke.obb.object_score import PIPELINE_PAGES, open_pack, report, score


class VectorEpochEvaluator:
    def __init__(self, pack, images, names, args):
        self.temporary = TemporaryDirectory(prefix="openstroke-epoch-eval-")
        self.root = Path(self.temporary.name)
        try:
            self.images = Path(images)
            self.names = list(names)
            self.args = dict(args)
            self.how = "panoptic" if self.args.get("stroke_panoptic") else "instances" if self.args.get("instance_queries") else "graph" if self.args.get("stroke_graph") else "members"
            self.dataset_gt = None
            self.train_gt = None
            if not pack:
                from openstroke.obb.vector_validation import DatasetVectorGT
                self.dataset_gt = DatasetVectorGT(self.images, self.names)
                self.protocol = self.dataset_gt.protocol
                self._train_pages(int(self.args.get("train_f2_pages", 3) or 0))
                return
            self.protocol = "score_pack.exact.v1"
            self.pack = open_pack(pack, temporary=self.root / "pack")
            pack_names = json.loads((self.pack / "names.json").read_text(encoding="utf-8"))
            normalize = lambda values: [name.replace("_", " ") for name in values]
            if normalize(pack_names) != normalize(self.names):
                raise ValueError("score_pack class names/order differ from the model")
            self.cache = {}
            records = json.loads((self.pack / "pages.json").read_text(encoding="utf-8"))
            page_ids = {row["stem"][:9] for row in records}
            missing = set(PIPELINE_PAGES) - page_ids
            if missing:
                raise ValueError(f"score_pack lacks comparison pages: {sorted(missing)}")
            stems = {p.stem for p in self.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
            required = {row["stem"] for row in records if row["stem"][:9] in PIPELINE_PAGES}
            if not required <= stems:
                raise ValueError(f"val images missing score_pack pages: {sorted(required - stems)}")
        except BaseException:
            self.close()
            raise

    def evaluate(self, module, epoch):
        path = self.root / "epoch.pt"
        torch.save({"model": module.state_dict(), "args": self.args, "classes": self.names, "epoch": epoch}, path)
        return self.evaluate_checkpoint(path)

    def evaluate_checkpoint(self, path):
        summary = self._evaluate_checkpoint(path)
        if getattr(self, "train_gt", None) is not None:
            summary["train"] = self.train_score(path, summary.get("how") or self.how)
        return summary

    def _evaluate_checkpoint(self, path):
        from openstroke.obb.cli import predict

        devices=[torch.cuda.current_device()] if torch.cuda.is_available() and torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            raw = predict(str(path), str(self.images), conf=0.05, save=False,
                          batch=min(8, int(self.args["batch"])), verbose=False,
                          tile_scale=self.args.get("scale"), imgsz=self.args.get("eval_img"),
                          compile=self.args.get("eval_compile"), seeded_edges=self.seeded_edges())
        detections = {Path(name).stem: rows for name, rows in raw.items()}
        if self.seeded_edges():  # OpenStroke13 / E92: every decoding path from the same prediction, best seeded one is primary
            return self.paths(detections)
        if self.args.get("stroke_panoptic") and self.dataset_gt is not None:   # OpenStroke19: best panoptic@t primary, members beside
            from openstroke.obb.models.stroke_panoptic import PANOPTIC_FG
            hows = [f"panoptic@{t:g}" for t in PANOPTIC_FG] + ["members"]
            res = {how: self._one(detections, how) for how in hows}
            best = max(hows[:-1], key=lambda h: res[h]["F2"])
            summary = dict(res[best])
            summary["how"] = best
            summary["paths"] = {h: {k: r[k] for k in ("F2", "P", "R", "best_conf")} for h, r in res.items()}
            sl = res[best].get("stroke_length") or {}
            if sl:
                summary["paths"][best + " strokes"] = {"F2": sl["F2"], "P": sl["P"], "R": sl["R"], "best_conf": 0.0}
            return summary
        if self.dataset_gt is not None:
            return self.dataset_gt.evaluate(detections, self.how, self.names)
        expected = {row["stem"] for row in json.loads((self.pack / "pages.json").read_text(encoding="utf-8"))
                    if row["stem"][:9] in PIPELINE_PAGES}
        if not expected <= detections.keys():
            raise ValueError(f"object validation lacks pages: {sorted(expected - detections.keys())}")
        summary = report(score(self.pack, detections, self.how, log=lambda _: None, cache=self.cache),
                         self.how, log=lambda _: None)
        if self.how == "members":  # F109: the same boxes with the layer-free label rule as light post-processing
            rule = report(score(self.pack, detections, "rule_nl", log=lambda _: None), "rule_nl", log=lambda _: None)
            summary["paths"] = {h: {k: float(r[k]) for k in ("F2", "P", "R", "best_conf")}
                                for h, r in (("members", summary), ("rule_nl", rule))}
        if not all(math.isfinite(summary[k]) for k in ("F2", "P", "R", "best_conf")):
            raise ValueError("object validation returned a non-finite score")
        summary["protocol"] = self.protocol
        return summary

    def _train_pages(self, n):
        """Diagnostic: the same F2 on n train pages (distinct documents) every epoch, to tell under- from over-fitting.
        Pages are linked into a private folder in the companion layout (images / vectors / labels)."""
        import os
        import shutil

        from openstroke.obb.vector_validation import DatasetVectorGT, companion
        src = self.images.parent / "train"
        if n <= 0 or not src.is_dir():
            return
        pages, seen = [], set()
        for image in sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")):
            if image.stem[:9] in seen or not companion(image, "vectors", ".npz").is_file():
                continue
            seen.add(image.stem[:9])
            pages.append(image)
            if len(pages) == n:
                break
        if not pages:
            return
        root = self.root / "train_pages"
        for image in pages:
            for kind, path in (("images", image), ("vectors", companion(image, "vectors", ".npz")),
                               ("labels", companion(image, "labels", ".txt"))):
                dst = root / kind / "train" / path.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(path, dst)
                except OSError:
                    shutil.copy2(path, dst)
        self.train_images = root / "images" / "train"
        self.train_gt = DatasetVectorGT(self.train_images, self.names, comparison_pages=tuple(p.stem[:9] for p in pages))

    def train_score(self, path, how):
        from openstroke.obb.cli import predict

        devices = [torch.cuda.current_device()] if torch.cuda.is_available() and torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            raw = predict(str(path), str(self.train_images), conf=0.05, save=False,
                          batch=min(8, int(self.args["batch"])), verbose=False,
                          tile_scale=self.args.get("scale"), imgsz=self.args.get("eval_img"),
                          compile=self.args.get("eval_compile"), seeded_edges=self.seeded_edges())
        r = self.train_gt.evaluate({Path(name).stem: rows for name, rows in raw.items()}, how, self.names)
        return {k: float(r[k]) for k in ("F2", "P", "R")}

    def seeded_edges(self):
        if str(self.args.get("graph_decode") or "") != "seeded" or not self.args.get("stroke_graph"):
            return None
        from openstroke.obb.seeded import SEEDED_EDGES
        return SEEDED_EDGES

    def _one(self, detections, how):
        if self.dataset_gt is not None:
            return self.dataset_gt.evaluate(detections, how, self.names)
        summary = report(score(self.pack, detections, how, log=lambda _: None, cache=self.cache), how, log=lambda _: None)
        summary["protocol"] = self.protocol
        return summary

    def paths(self, detections):
        """members (boxes + their strokes), graph (components) and seeded@t; the primary result is the best seeded@t."""
        hows = ["members", "graph"] + [f"seeded@{t:g}" for t in self.seeded_edges()]
        res = {how: self._one(detections, how) for how in hows}
        best = max((h for h in hows if h.startswith("seeded")), key=lambda h: res[h]["F2"])
        summary = dict(res[best])
        summary["how"] = best
        summary["paths"] = {h: {k: r[k] for k in ("F2", "P", "R", "best_conf")} for h, r in res.items()}
        if not all(math.isfinite(summary[k]) for k in ("F2", "P", "R", "best_conf")):
            raise ValueError("object validation returned a non-finite score")
        return summary

    def close(self):
        self.temporary.cleanup()


GraphEpochEvaluator = VectorEpochEvaluator  # compatibility for callers of the v11/v12 evaluator
