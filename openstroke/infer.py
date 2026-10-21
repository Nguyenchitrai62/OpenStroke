"""Inference on a page without ground truth: ``split_countfire`` payload -> the training input -> one probability per shape.

The payload is the JSON that AI_Takeoff's SDK receives (``tools/extract_payload.py`` writes it from a PDF with
AI_Takeoff's ``pdf_etl``). It goes through the same steps as a training page (``build`` arrays with no GT ->
``features`` -> ``data.page_from_npz``), so the model sees exactly what it saw in training.

    from openstroke.infer import Predictor, load_payload
    pred = Predictor("runs/train/fold0/best.pt")
    out = pred.predict_payload(load_payload("page-0001.v0.json.gz"), mode="fire")
    out["prob"]   # (n_shapes,) in payload order; shapes without geometry get 0
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from .sources import load_payload  # noqa: F401  (re-exported for callers)

#: window used when a checkpoint does not record its training arguments (first v1 trainer)
DEFAULT_WINDOW = {"openstroke1": 4096, "openstroke2": 65536, "openstroke3": 65536, "openstroke4": 65536}


def payload_npz(payload: dict, path: Path, mode: str, stem: str = "page") -> dict[str, Any]:
    """Write the E0 npz fields of a payload page with every label ``other`` (no GT); returns page info."""

    from .build import COORDINATE_FRAME, SCHEMA_VERSION, shape_attributes
    from .geometry import page_segments
    from .labels import LABEL_SCHEMA, gt_geometry, label_shapes

    shapes = payload.get("shapes") or []
    meta = payload.get("document_metadata") or {}
    segs = page_segments(shapes)
    labels = label_shapes(segs, gt_geometry(mode, []))
    attrs = shape_attributes(shapes, segs)
    meta_json = json.dumps({"stem": stem, "mode": mode, "schema_version": SCHEMA_VERSION, "label_schema": LABEL_SCHEMA,
                            "coordinate_frame": COORDINATE_FRAME, "source": "payload (inference, no GT)"})
    np.savez(path, meta=np.array(meta_json), label=labels.label, true_frac=labels.true_frac, length=labels.length,
             segments=segs.seg.astype(np.float32), shape_offsets=segs.shape_offsets, seg_inside=labels.seg_inside,
             qa_suspicious_other=np.zeros(len(shapes), np.uint8), gt_class=np.array([], dtype=str),
             gt_coverage=np.zeros(0, np.float32), **attrs)
    return {"shapes": len(shapes), "width": float(meta.get("page_width") or 0), "height": float(meta.get("page_height") or 0),
            "segments": segs, "length": labels.length, "attrs": attrs}


class Predictor:
    """A trained checkpoint (``best.pt`` of any fold, v1 or v2) ready to score payload pages."""

    def __init__(self, checkpoint: str | Path, device: str = "auto", threads: int | None = None):
        import torch

        from .models import from_config, load_weights

        if threads:
            torch.set_num_threads(threads)
        self.path = Path(checkpoint)
        ck = torch.load(self.path, map_location="cpu", weights_only=False)
        self.config = dict(ck["config"])
        self.arch = self.config.get("arch", "openstroke1")
        self.model = from_config(self.config)
        load_weights(self.model, ck["model"])
        dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.model.to(self.device).eval()
        args = ck.get("args") or {}
        self.window = int(args.get("window") or DEFAULT_WINDOW.get(self.arch, 65536))
        self.threshold = float(ck.get("threshold", 0.5))
        self.tta = int(ck.get("tta", 1))
        self.val_f2 = ck.get("val_f2")
        self.epoch = ck.get("epoch")
        self.mode_input = bool(self.config.get("mode_input", False))
        self.feature_names = ck.get("feature_names")

    def describe(self) -> dict:
        return {"checkpoint": str(self.path), "arch": self.arch, "device": str(self.device), "window": self.window,
                "threshold": self.threshold, "tta_used_in_val": self.tta, "val_f2": self.val_f2,
                "epoch": None if self.epoch is None else int(self.epoch) + 1, "mode_input": self.mode_input}

    def predict_payload(self, payload: dict, mode: str = "fire", tta: int = 1) -> dict[str, Any]:
        """-> {"prob": (n,) per payload shape, "info": page info, "seconds": {...}}."""

        import torch

        from .data import page_from_npz
        from .engine.trainer import predict_page
        from .features import FEATURE_NAMES

        if self.feature_names is not None and tuple(self.feature_names) != tuple(FEATURE_NAMES):
            raise ValueError("checkpoint was trained on another feature set than this code version")
        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "page.npz"
            info = payload_npz(payload, npz, mode)
            page = page_from_npz(npz, "page", mode, graph=bool(self.config.get("relations", 0))
                                 or "graph" in (self.config.get("mixers") or []))
        t1 = time.time()
        prob = np.zeros(info["shapes"], np.float32)
        if len(page):
            amp = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else None
            p = predict_page(self.model, page, self.window, self.device, batch=1, tta=tta, amp_dtype=amp)
            prob[page.index] = p
        return {"prob": prob, "info": info, "kept": int(len(page)),
                "seconds": {"features": round(t1 - t0, 2), "model": round(time.time() - t1, 2)}}
