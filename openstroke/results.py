"""Raw prediction results with an OpenOBB-style interface for standalone model hosting and deployment.

    from openstroke import OpenStroke
    model = OpenStroke("best.pt")
    for r in model.predict(payload):
        r.prob          # (N,) float32 raw probability per payload shape (0 for shapes without geometry)
        r.valid         # (N,) bool mask of shapes scored by the model
        r.edges         # (pairs (E, 2), prob (E,)) | None (models with edge_head)
        r.groups        # {instance_id, cluster_ids, cluster_probs, stroke_cls, names} | None (models with group_head)
        r.threshold     # float reference threshold calibrated on val (NOT applied to r.prob)
        r.speed         # {"preprocess": ms, "inference": ms, "postprocess": ms}
        r.to_dict()     # lightweight JSON-serializable raw payload for microservice deployment
        r.save_npz("out.npz")

Post-processing (building HVAC polygons or FIRE/PLUMBING line topologies from raw stroke probabilities) is strictly
separate from this result object so the model service can be deployed independently.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class Strokes:
    """Raw per-shape predictions of one drawing page in exact ``payload['shapes']`` order."""

    def __init__(
        self,
        prob: np.ndarray,
        valid: np.ndarray,
        *,
        edges: tuple[np.ndarray, np.ndarray] | None = None,
        groups: dict[str, Any] | None = None,
        oprob: np.ndarray | None = None,
    ):
        self.prob = np.asarray(prob, dtype=np.float32)
        self.valid = np.asarray(valid, dtype=bool)
        self.edges = edges
        self.groups = groups
        self.oprob = oprob

    def __len__(self) -> int:
        return int(len(self.prob))

    @property
    def kept(self) -> int:
        return int(self.valid.sum())

    def selected(self, threshold: float) -> np.ndarray:
        """Indices of valid shapes with ``prob >= threshold``."""
        return np.flatnonzero(self.valid & (self.prob >= float(threshold)))

    def __repr__(self) -> str:
        return f"openstroke.results.Strokes(shapes={len(self)}, valid={self.kept})"


class Results:
    """Raw inference result of one page (OpenOBB-style interface, strictly raw — zero post-processing)."""

    def __init__(
        self,
        strokes: Strokes,
        path: str = "payload0.json",
        *,
        threshold: float = 0.5,
        arch: str = "",
        speed: dict[str, float] | None = None,
        info: dict[str, Any] | None = None,
        save_dir: str | None = None,
        extra: dict[str, Any] | None = None,
    ):
        self.strokes = strokes
        self.path = str(path)
        self.threshold = float(threshold)
        self.arch = str(arch)
        self.speed = dict(speed or {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0})
        self.info = info
        self.save_dir = save_dir
        self.extra = dict(extra or {})

    # ---- direct shortcuts (r.prob == r.strokes.prob)
    @property
    def prob(self) -> np.ndarray:
        return self.strokes.prob

    @property
    def valid(self) -> np.ndarray:
        return self.strokes.valid

    @property
    def edges(self) -> tuple[np.ndarray, np.ndarray] | None:
        return self.strokes.edges

    @property
    def groups(self) -> dict[str, Any] | None:
        return self.strokes.groups

    @property
    def oprob(self) -> np.ndarray | None:
        return self.strokes.oprob

    @property
    def kept(self) -> int:
        return self.strokes.kept

    def __len__(self) -> int:
        return len(self.strokes)

    def selected(self, threshold: float | None = None) -> np.ndarray:
        """Shape indices with ``valid & (prob >= threshold)`` (defaults to ``self.threshold``)."""
        return self.strokes.selected(self.threshold if threshold is None else threshold)

    def verbose(self) -> str:
        n_sel = int((self.valid & (self.prob >= self.threshold)).sum())
        return f"{len(self):,} shapes ({self.kept:,} valid, {n_sel:,} >= {self.threshold:.2f}), "

    def to_dict(
        self,
        *,
        include_edges: bool = True,
        include_groups: bool = True,
        include_oprob: bool = True,
        decimals: int = 5,
    ) -> dict[str, Any]:
        """Lightweight JSON-serializable raw response dict for HTTP / worker microservice hosting."""
        out: dict[str, Any] = {
            "path": self.path,
            "arch": self.arch,
            "n_shapes": len(self),
            "kept": self.kept,
            "threshold": round(self.threshold, 4),
            "speed_ms": {k: round(float(v), 2) for k, v in self.speed.items()},
            "prob": np.round(self.prob, decimals).tolist(),
            "valid": self.valid.tolist(),
        }
        if include_edges and self.edges is not None:
            pairs, eprob = self.edges
            out["edges"] = {
                "pairs": np.asarray(pairs, dtype=np.int64).tolist(),
                "prob": np.round(np.asarray(eprob, dtype=np.float32), decimals).tolist(),
            }
        if include_groups and self.groups is not None:
            g = self.groups
            out["groups"] = {
                "instance_id": np.asarray(g["instance_id"], dtype=np.int64).tolist(),
                "cluster_ids": np.asarray(g["cluster_ids"], dtype=np.int64).tolist(),
                "cluster_probs": np.round(np.asarray(g["cluster_probs"], dtype=np.float32), decimals).tolist(),
                "names": list(g.get("names") or ()),
            }
        if include_oprob and self.oprob is not None:
            out["oprob"] = np.round(np.asarray(self.oprob, dtype=np.float32), decimals).tolist()
        return out

    def to_json(self, *, indent: int | None = None, **kw) -> str:
        return json.dumps(self.to_dict(**kw), indent=indent, ensure_ascii=False)

    def save_json(self, path: str | Path, *, indent: int | None = None, **kw) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(indent=indent, **kw), encoding="utf-8")
        return str(p)

    def save_npz(self, path: str | Path, *, compress: bool = True) -> str:
        """Compact binary raw output (.npz) for decoupled downstream post-processing modules."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "prob": self.prob.astype(np.float32),
            "valid": self.valid.astype(bool),
            "threshold": np.array(self.threshold, dtype=np.float32),
            "meta": np.array(json.dumps({"path": self.path, "arch": self.arch, "speed_ms": self.speed}, ensure_ascii=False)),
        }
        if self.edges is not None:
            arrays["edge_pairs"] = np.asarray(self.edges[0], dtype=np.int32)
            arrays["edge_prob"] = np.asarray(self.edges[1], dtype=np.float32)
        if self.groups is not None:
            arrays["group_instance_id"] = np.asarray(self.groups["instance_id"], dtype=np.int32)
            arrays["group_cluster_ids"] = np.asarray(self.groups["cluster_ids"], dtype=np.int32)
            arrays["group_cluster_probs"] = np.asarray(self.groups["cluster_probs"], dtype=np.float32)
            if "stroke_cls" in self.groups:
                arrays["group_stroke_cls"] = np.asarray(self.groups["stroke_cls"], dtype=np.float16)
        if self.oprob is not None:
            arrays["oprob"] = np.asarray(self.oprob, dtype=np.float32)
        (np.savez_compressed if compress else np.savez)(p, **arrays)
        return str(p)

    def to_legacy_dict(self) -> dict[str, Any]:
        """Convert to the legacy ``Predictor.predict_payload`` dict shape for full backward compatibility."""
        out: dict[str, Any] = {
            **self.extra,
            "prob": self.prob,
            "valid": self.valid,
            "oprob": self.oprob,
            "edges": self.edges,
            "info": self.info,
            "kept": self.kept,
            "seconds": {
                "features": round(self.speed.get("preprocess", 0.0) / 1e3, 3),
                "model": round((self.speed.get("inference", 0.0) + self.speed.get("postprocess", 0.0)) / 1e3, 3),
            },
        }
        if self.groups is not None:
            out["group_objects"] = self.groups
        return out

    def __repr__(self) -> str:
        return f"openstroke.results.Results(path={self.path!r}, shapes={len(self)}, kept={self.kept}, arch={self.arch!r})"
