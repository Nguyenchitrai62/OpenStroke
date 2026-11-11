"""Full-page graph object F2 used to select EMA checkpoints during training."""
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from openstroke.obb.object_score import PIPELINE_PAGES, open_pack, report, score


class GraphEpochEvaluator:
    def __init__(self, pack, images, names, args):
        self.temporary = TemporaryDirectory(prefix="openstroke-epoch-eval-")
        self.root = Path(self.temporary.name)
        try:
            self.pack = open_pack(pack, temporary=self.root / "pack")
            self.images = Path(images)
            self.names = list(names)
            self.args = dict(args)
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
        from openstroke.obb.cli import predict

        raw = predict(str(path), str(self.images), conf=0.05, save=False,
                      batch=min(8, int(self.args["batch"])), verbose=False)
        detections = {Path(name).stem: rows for name, rows in raw.items()}
        expected = {row["stem"] for row in json.loads((self.pack / "pages.json").read_text(encoding="utf-8"))
                    if row["stem"][:9] in PIPELINE_PAGES}
        if not expected <= detections.keys():
            raise ValueError(f"object validation lacks pages: {sorted(expected - detections.keys())}")
        summary = report(score(self.pack, detections, "graph", log=lambda _: None, cache=self.cache),
                         "graph", log=lambda _: None)
        if not all(math.isfinite(summary[k]) for k in ("F2", "P", "R", "best_conf")):
            raise ValueError("object validation returned a non-finite score")
        return summary

    def close(self):
        self.temporary.cleanup()
