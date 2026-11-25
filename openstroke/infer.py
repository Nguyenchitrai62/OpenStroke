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


def payload_arrays(payload: dict, mode: str, stem: str = "page") -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build the E0 npz arrays of a payload page in memory (every label ``other``, no GT); returns (arrays, page info)."""

    from .build import COORDINATE_FRAME, SCHEMA_VERSION, shape_attributes
    from .geometry import page_segments
    from .labels import LABEL_SCHEMA

    shapes = payload.get("shapes") or []
    meta = payload.get("document_metadata") or {}
    n = len(shapes)
    segs = page_segments(shapes)
    length = segs.shape_length().astype(np.float32)
    attrs = shape_attributes(shapes, segs)
    meta_dict = {"stem": stem, "mode": mode, "schema_version": SCHEMA_VERSION, "label_schema": LABEL_SCHEMA,
                 "coordinate_frame": COORDINATE_FRAME, "source": "payload (inference, no GT)"}
    arrays = {
        "meta": np.array(json.dumps(meta_dict)),
        "label": np.zeros(n, np.uint8),
        "true_frac": np.zeros(n, np.float32),
        "length": length,
        "segments": segs.seg.astype(np.float32),
        "shape_offsets": segs.shape_offsets,
        "seg_inside": np.zeros(len(segs.seg), np.float32),
        "qa_suspicious_other": np.zeros(n, np.uint8),
        "gt_class": np.array([], dtype=str),
        "gt_coverage": np.zeros(0, np.float32),
        **attrs,
    }
    info = {"shapes": n, "width": float(meta.get("page_width") or 0), "height": float(meta.get("page_height") or 0),
            "segments": segs, "length": length, "attrs": attrs}
    return arrays, info


def payload_npz(payload: dict, path: Path, mode: str, stem: str = "page", compress: bool = False) -> dict[str, Any]:
    """Write the E0 npz fields of a payload page with every label ``other`` (no GT); returns page info."""

    arrays, info = payload_arrays(payload, mode, stem=stem)
    (np.savez_compressed if compress else np.savez)(path, **arrays)
    return info


GPU_BATCH = 4


class Predictor:
    """A trained checkpoint (``best.pt`` of any fold, v1 or v2) ready to score payload pages."""

    def __init__(self, checkpoint: str | Path, device: str = "auto", threads: int | None = None, workers: int = 1,
                 batch: int | None = None):
        import torch

        from .models import from_config, load_weights

        if threads:
            torch.set_num_threads(threads)
        self.path = Path(checkpoint)
        ck = torch.load(self.path, map_location="cpu", weights_only=False)
        if "config" not in ck:  # an OpenOBB detector (openstroke obb train: E69, E83 ...) is not a stroke model
            kind = f"OpenOBB {ck.get('version', '')} (HVAC không layer: hộp + nét)" if "classes" in ck else "không phải model nét OpenStroke"
            raise ValueError(f"{self.path.name}: checkpoint {kind}. App này chỉ chạy model phân loại nét (openstroke train, ví dụ E54). "
                             "Model OpenOBB: chấm bằng tools/obb_predict_val.py rồi tools/obb_score_pack.py.")
        self.config = dict(ck["config"])
        from .models import canonical_arch

        self.arch = canonical_arch(self.config.get("arch"))
        self.model = from_config(self.config, weights_follow=True)
        load_weights(self.model, (ck.get("ema") or {}).get("module") or ck["model"])   # last.pt: its EMA weights, the ones val scored
        dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.model.to(self.device).eval()
        self.batch = int(batch) if batch is not None else (1 if self.device.type == "cpu" else GPU_BATCH)  # a GPU runs windows together in a batch, not in several processes (F21)
        self.pool = None
        self.pool_error = None
        needs_inproc = (getattr(self.model, "obj_mlp", None) is not None
                        or getattr(self.model, "edge_mlp", None) is not None
                        or getattr(self.model, "group_head", None) is not None)
        if self.device.type == "cpu" and workers > 1 and not needs_inproc:
            from .parallel import WindowPool

            self.pool = WindowPool(self.path, workers, threads or torch.get_num_threads())
        args = ck.get("args") or {}
        self.window = int(args.get("window") or DEFAULT_WINDOW.get(self.arch, 65536))
        self.threshold = float(ck.get("threshold", 0.5))
        self.tta = int(ck.get("tta", 1))
        self.val_f2 = ck.get("val_f2")
        self.epoch = ck.get("epoch")
        self.mode_input = bool(self.config.get("mode_input", False))
        self.feature_names = ck.get("feature_names")

    def describe(self) -> dict:
        from .train import describe_arch

        return {"checkpoint": str(self.path), "arch": describe_arch(self.config), "device": str(self.device), "window": self.window,
                "threshold": self.threshold, "tta_used_in_val": self.tta, "val_f2": self.val_f2,
                "epoch": None if self.epoch is None else int(self.epoch) + 1, "mode_input": self.mode_input}

    def predict_payload(self, payload: dict, mode: str = "fire", tta: int = 1, on_stage=None, *, group_strokes: bool = False) -> dict[str, Any]:
        """-> {"prob": (n,) per payload shape, "info": page info, "seconds": {...}}. ``on_stage(name, done, total)``: "features", then
        "model" with windows done / total. With group_strokes=True (same-object edge head), instance_id
        indexes groups in payload order, with -1 for shapes below the checkpoint threshold."""

        import torch

        from .data import page_from_arrays
        from .engine.trainer import predict_page
        from .features import FEATURE_NAMES

        if self.feature_names is not None and tuple(self.feature_names) != tuple(FEATURE_NAMES):
            raise ValueError("checkpoint was trained on another feature set than this code version")
        if group_strokes and getattr(self.model, "edge_mlp", None) is None:
            raise ValueError("group_strokes needs a checkpoint with a same-object edge head")
        t0 = time.time()
        if on_stage:
            on_stage("features", 0, 0)
        arrays, info = payload_arrays(payload, mode)
        graph = (bool(self.config.get("relations", 0)) or "graph" in (self.config.get("mixers") or [])
                 or bool(self.config.get("rel_attn", 0)) or bool(self.config.get("edge_head"))
                 or bool(self.config.get("group_head")))
        if self.config.get("continuation"):
            graph = "continuation"
        elif self.config.get("facing_edges"):
            graph = "facing"
        page = page_from_arrays(arrays, "page", mode, graph=graph, no_layers=bool(self.config.get("no_layers")))
        t1 = time.time()
        prob = np.zeros(info["shapes"], np.float32)
        if on_stage:
            on_stage("model", 0, 0)
        pc_out = e_out = g_out = None
        if len(page):
            on_batch = (lambda done, total: on_stage("model", done, total)) if on_stage else None
            has_cls = getattr(self.model, "obj_mlp", None) is not None
            has_edge = getattr(self.model, "edge_mlp", None) is not None
            has_group = getattr(self.model, "group_head", None) is not None
            p = None
            if self.pool is not None and not (has_cls or has_edge or has_group):
                with tempfile.TemporaryDirectory() as td:
                    npz = Path(td) / "page.npz"
                    np.savez(npz, **arrays)
                    p = self.pool.predict(page, npz, mode, graph, self.window, tta, on_batch)
                if p is None and self.pool:  # free what the other processes hold first: the usual cause is memory (F53)
                    self.pool_error = self.pool.error
                    self.pool.close()
                    self.pool = None
            if p is None:   # in-process model (always used on CUDA or when class / link / group heads are present)
                amp = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else None
                pc_out = [] if has_cls else None
                e_out = [] if has_edge else None
                g_out = [] if has_group else None
                try:
                    p = predict_page(self.model, page, self.window, self.device, batch=self.batch, tta=tta, amp_dtype=amp,
                                     on_batch=on_batch, obj_out=pc_out, edge_out=e_out, group_out=g_out)
                except torch.cuda.OutOfMemoryError:  # a smaller GPU: one window at a time from now on
                    torch.cuda.empty_cache()
                    self.batch = 1
                    pc_out = [] if has_cls else None
                    e_out = [] if has_edge else None
                    g_out = [] if has_group else None
                    p = predict_page(self.model, page, self.window, self.device, batch=1, tta=tta, amp_dtype=amp, on_batch=on_batch,
                                     obj_out=pc_out, edge_out=e_out, group_out=g_out)
            prob[page.index] = p
        oprob = None
        if pc_out:   # OpenStroke29: class of every stroke, from the same pass
            from .objects import N_CLASSES
            oprob = np.zeros((info["shapes"], N_CLASSES), np.float32)
            oprob[page.index] = pc_out[0][:, :N_CLASSES]
        edges = None
        if e_out:   # OpenStroke30 / F169: (shape ids (E, 2), same-object probability (E,))
            edges = (page.index[e_out[0][0]], 1.0 / (1.0 + np.exp(-e_out[0][1].astype(np.float64))))
        extra = {}
        if g_out and len(page):   # OpenStroke41: full-page H142 mutex instances + GroupClsHead 8-class probs
            from .models.group_head import supplement_fg_edges
            from .stroke_groups import stroke_instances

            p_page = prob[page.index]
            e_pairs, e_logits = (e_out[0] if e_out else (np.zeros((0, 2), np.int64), np.zeros(0, np.float32)))
            e_pairs, e_logits = supplement_fg_edges(
                self.model, g_out[0][0], page, p_page, self.threshold, e_pairs, e_logits,
                info.get("segments"), info["shapes"], self.device,
            )
            if len(e_pairs):
                edges = (page.index[e_pairs], 1.0 / (1.0 + np.exp(-np.clip(e_logits.astype(np.float64), -20.0, 20.0))))
            inst_p = (stroke_instances(p_page, self.threshold, e_pairs, e_logits)
                      if len(e_pairs) else np.where(p_page >= self.threshold, np.arange(len(p_page), dtype=np.int64), -1))
            cids, cprobs = self.model.group_head.classify_page_clusters(
                g_out[0][0], page, p_page, self.threshold, (e_pairs, e_logits) if len(e_pairs) else None, inst_p, self.device
            )
            full_inst = np.full(info["shapes"], -1, np.int64)
            full_inst[page.index] = inst_p
            scls = np.zeros((info["shapes"], g_out[0][1].shape[1]), np.float32)
            scls[page.index] = g_out[0][1]
            extra["group_objects"] = {"instance_id": full_inst, "cluster_ids": cids, "cluster_probs": cprobs, "stroke_cls": scls}
        if self.config.get("region_head"):
            from .piece_candidates import predict_regions
            amp = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else None
            try:
                extra = predict_regions(self.model, page, info, prob, self.window, self.device, self.batch, amp)
            except torch.cuda.OutOfMemoryError:
                if self.batch == 1:
                    raise
                self.batch = 1
                torch.cuda.empty_cache()
                extra = predict_regions(self.model, page, info, prob, self.window, self.device, 1, amp)
        if group_strokes:
            from .stroke_groups import stroke_instances

            pairs, logits = (edges[0], e_out[0][1]) if e_out else (np.empty((0, 2), np.int64), np.empty(0))
            extra["instance_id"] = stroke_instances(prob, self.threshold, pairs, logits)
        return {**extra, "prob": prob, "oprob": oprob, "edges": edges, "info": info, "kept": int(len(page)),
                "seconds": {"features": round(t1 - t0, 2), "model": round(time.time() - t1, 2)}}
