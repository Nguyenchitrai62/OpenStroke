"""The test app's runner for the vector HVAC detector (``openstroke obb``: OpenOBB25 / OpenStroke9, e.g. E69, E83): objects and their strokes.

A page goes the way the training pages were made (``tools/make_obb_dataset.py``): drawn from its own vectors at ``SCALE`` px per pt, one
vector token per stroke beside the image (no layers), then ``openstroke.obb`` predicts boxes with the strokes each box drew (``members``).
Strokes claimed by several boxes go to the highest score (``openstroke.assemble``). Everything stays inside OpenStroke; OpenOBB is not used.

The page result has, per payload shape, the owning object (-1 none) and that object's score; the viewer's threshold is the detection
confidence: with the highest score owning a stroke, hiding objects under a threshold gives exactly the assignment made at that threshold.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

SCALE = 4.0  # px per pt of the training pages (obb_hvac7: pages.json scale 4.0)
LOW_CONF = 0.05  # boxes kept for the viewer; its threshold hides the rest
DEFAULT_CONF = 0.2  # best object F2 of E69 with the model's own strokes (F81)


def is_obb_checkpoint(path: str | Path) -> bool:
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    return "config" not in ck and "classes" in ck


class ObbPagePredictor:
    kind = "obb"

    def __init__(self, checkpoint: str | Path, threads: int | None = None):
        import torch

        if threads:
            torch.set_num_threads(threads)
        self.path = Path(checkpoint)
        ck = torch.load(self.path, map_location="cpu", weights_only=False)
        self.names = list(ck["classes"].values()) if isinstance(ck["classes"], dict) else list(ck["classes"])
        self.args = ck.get("args") or {}
        self.version = ck.get("version")
        self.epoch = ck.get("epoch")
        self.map = ck.get("val_mAP50_95")
        self.pool_error = None
        del ck

    def describe(self) -> dict:
        arch = f"OpenOBB25 vector ({self.version})" + (" + endpoint members" if self.args.get("endpoint_members") else "")
        return {"checkpoint": str(self.path), "arch": arch, "device": "cpu", "window": 1024, "threshold": DEFAULT_CONF,
                "tta_used_in_val": 1, "val_f2": None, "val_map50_95": self.map, "epoch": None if self.epoch is None else int(self.epoch) + 1,
                "mode_input": False, "kind": "obb", "classes": self.names}

    def predict_payload(self, payload: dict, on_stage=None) -> dict:
        import cv2

        from make_obb_dataset import render, vector_tokens
        from openstroke.assemble import assign_members
        from openstroke.infer import payload_npz
        from openstroke.obb.cli import predict

        t0 = time.time()
        if on_stage:
            on_stage("features", 0, 0)
        shapes = payload.get("shapes") or []
        with tempfile.TemporaryDirectory() as td:
            info = payload_npz(payload, Path(td) / "strokes.npz", "hvac")
            w, h = info["width"], info["height"]
            img = render(shapes, w, h, SCALE)
            png = Path(td) / "page.png"
            cv2.imwrite(str(png), img)
            np.savez_compressed(Path(td) / "page.npz", tokens=vector_tokens(shapes, [], SCALE))  # beside the image: found by the loader
            t1 = time.time()
            if on_stage:
                on_stage("model", 0, 0)
            dets = next(iter(predict(str(self.path), str(png), conf=LOW_CONF, tile_scale=1, save=False, batch=1).values()), [])
        n = int(info["shapes"])
        L = np.asarray(info["length"], np.float64)
        boxes = [(int(q["class_id"]), np.asarray(q["poly"], float).reshape(4, 2) / SCALE, float(q["score"]), q.get("members") or []) for q in dets]
        own = assign_members(boxes, L, order="score", dedupe=False)
        prob = np.where(own >= 0, np.asarray([b[2] for b in boxes] + [0.0])[own], 0.0)
        objects = [{"cls": c, "name": self.names[c] if c < len(self.names) else str(c), "score": round(s, 3),
                    "poly": np.round(v, 2).tolist(), "shapes": int((own == k).sum())} for k, (c, v, s, _) in enumerate(boxes)]
        return {"prob": prob, "obj": own, "objects": objects, "info": info, "kept": n,
                "seconds": {"features": round(t1 - t0, 1), "model": round(time.time() - t1, 1)}}
