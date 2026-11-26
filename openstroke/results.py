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
        cls_prob: np.ndarray | None = None,
        class_names: tuple[str, ...] | None = None,
    ):
        self.prob = np.asarray(prob, dtype=np.float32)
        self.valid = np.asarray(valid, dtype=bool)
        self.edges = edges
        self.groups = groups
        self.oprob = oprob
        self.cls_prob = np.asarray(cls_prob, dtype=np.float32) if cls_prob is not None else None
        self.class_names = tuple(class_names) if class_names is not None else None

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
    def cls_prob(self) -> np.ndarray | None:
        return self.strokes.cls_prob

    @property
    def class_names(self) -> tuple[str, ...] | None:
        return self.strokes.class_names

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
        compact: bool = False,
        min_prob: float = 0.05,
        decimals: int = 5,
    ) -> dict[str, Any]:
        """JSON-serializable raw response dict for HTTP / worker microservice hosting.

        ``compact=True`` sends sparse foreground arrays (only shapes with ``prob >= min_prob`` and invalid indices),
        cutting HTTP payload size ~10-30x on large A0 drawings (230k+ shapes); ``Results.from_dict(...)`` reconstructs
        both dense and compact responses transparently into identical full ``(N,)`` arrays.
        """
        n = len(self)
        out: dict[str, Any] = {
            "path": self.path,
            "arch": self.arch,
            "n_shapes": n,
            "kept": self.kept,
            "threshold": round(self.threshold, 4),
            "speed_ms": {k: round(float(v), 2) for k, v in self.speed.items()},
        }
        if compact:
            out["format"] = "sparse"
            out["min_prob"] = float(min_prob)
            fg = self.valid & (self.prob >= float(min_prob))
            fg_idx = np.flatnonzero(fg)
            out["prob_idx"] = fg_idx.tolist()
            out["prob_val"] = np.round(self.prob[fg_idx], decimals).tolist()
            out["invalid_idx"] = np.flatnonzero(~self.valid).tolist()
        else:
            out["format"] = "dense"
            out["prob"] = np.round(self.prob, decimals).tolist()
            out["valid"] = self.valid.tolist()

        if include_edges and self.edges is not None:
            pairs = np.asarray(self.edges[0], dtype=np.int64).reshape(-1, 2)
            eprob = np.asarray(self.edges[1], dtype=np.float32)
            if compact and len(pairs):
                keep_e = (self.prob[pairs[:, 0]] >= float(min_prob)) & (self.prob[pairs[:, 1]] >= float(min_prob))
                pairs, eprob = pairs[keep_e], eprob[keep_e]
            out["edges"] = {
                "pairs": pairs.tolist(),
                "prob": np.round(eprob, decimals).tolist(),
            }
        if include_groups and self.groups is not None:
            g = self.groups
            inst = np.asarray(g["instance_id"], dtype=np.int64)
            gd: dict[str, Any] = {
                "cluster_ids": np.asarray(g["cluster_ids"], dtype=np.int64).tolist(),
                "cluster_probs": np.round(np.asarray(g["cluster_probs"], dtype=np.float32), decimals).tolist(),
                "names": list(g.get("names") or ()),
            }
            if compact:
                ai = np.flatnonzero(inst >= 0)
                gd["inst_idx"] = ai.tolist()
                gd["inst_val"] = inst[ai].tolist()
            else:
                gd["instance_id"] = inst.tolist()
            out["groups"] = gd
        if include_oprob and self.oprob is not None:
            op = np.asarray(self.oprob, dtype=np.float32)
            if compact:
                fg_idx = np.flatnonzero(self.valid & (self.prob >= float(min_prob)))
                out["oprob_idx"] = fg_idx.tolist()
                out["oprob_val"] = np.round(op[fg_idx], decimals).tolist()
                out["oprob_cols"] = int(op.shape[1]) if op.ndim == 2 else 0
            else:
                out["oprob"] = np.round(op, decimals).tolist()
        if self.cls_prob is not None:
            cp = np.asarray(self.cls_prob, dtype=np.float32)
            if self.class_names is not None:
                out["class_names"] = list(self.class_names)
            if compact:
                fg_idx = np.flatnonzero(self.valid & (self.prob >= float(min_prob)))
                out["cls_prob_idx"] = fg_idx.tolist()
                out["cls_prob_val"] = np.round(cp[fg_idx], decimals).tolist()
                out["cls_prob_cols"] = int(cp.shape[1]) if cp.ndim == 2 else 0
            else:
                out["cls_prob"] = np.round(cp, decimals).tolist()
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str) -> "Results":
        """Reconstruct a ``Results`` object from a raw API response dict or JSON string (dense or compact sparse)."""
        d = json.loads(data) if isinstance(data, str) else dict(data)
        n = int(d["n_shapes"])
        if d.get("format") == "sparse" or "prob_idx" in d:
            prob = np.zeros(n, dtype=np.float32)
            p_idx = np.asarray(d.get("prob_idx") or (), dtype=np.int64)
            if len(p_idx):
                prob[p_idx] = np.asarray(d.get("prob_val") or (), dtype=np.float32)
            valid = np.ones(n, dtype=bool)
            inv = np.asarray(d.get("invalid_idx") or (), dtype=np.int64)
            if len(inv):
                valid[inv] = False
        else:
            prob = np.asarray(d["prob"], dtype=np.float32)
            valid = np.asarray(d["valid"], dtype=bool) if "valid" in d else np.ones(len(prob), dtype=bool)

        edges = None
        if d.get("edges") is not None:
            ed = d["edges"]
            pairs = np.asarray(ed.get("pairs") or (), dtype=np.int64).reshape(-1, 2)
            eprob = np.asarray(ed.get("prob") or (), dtype=np.float32)
            edges = (pairs, eprob)

        groups = None
        if d.get("groups") is not None:
            gd = d["groups"]
            if "inst_idx" in gd:
                inst = np.full(n, -1, dtype=np.int64)
                ai = np.asarray(gd["inst_idx"], dtype=np.int64)
                if len(ai):
                    inst[ai] = np.asarray(gd["inst_val"], dtype=np.int64)
            else:
                inst = np.asarray(gd.get("instance_id") or (), dtype=np.int64)
            cids = np.asarray(gd.get("cluster_ids") or (), dtype=np.int64)
            cprobs = np.asarray(gd.get("cluster_probs") or (), dtype=np.float32)
            if cprobs.ndim == 1:
                cprobs = cprobs.reshape(0, len(gd.get("names") or ()) or 8)
            groups = {
                "instance_id": inst,
                "cluster_ids": cids,
                "cluster_probs": cprobs,
                "names": tuple(gd.get("names") or ()),
            }

        oprob = None
        if d.get("oprob") is not None:
            oprob = np.asarray(d["oprob"], dtype=np.float32)
        elif d.get("oprob_idx") is not None:
            nc = int(d.get("oprob_cols") or 0)
            oprob = np.zeros((n, nc), dtype=np.float32)
            oi = np.asarray(d["oprob_idx"], dtype=np.int64)
            if len(oi) and nc:
                oprob[oi] = np.asarray(d["oprob_val"], dtype=np.float32)

        cls_prob = None
        class_names = tuple(d["class_names"]) if d.get("class_names") is not None else None
        if d.get("cls_prob") is not None:
            cls_prob = np.asarray(d["cls_prob"], dtype=np.float32)
        elif d.get("cls_prob_idx") is not None:
            nc = int(d.get("cls_prob_cols") or (len(class_names) if class_names else 3))
            cls_prob = np.zeros((n, nc), dtype=np.float32)
            if nc > 0:
                cls_prob[:, 0] = 1.0
            ci = np.asarray(d["cls_prob_idx"], dtype=np.int64)
            if len(ci) and nc:
                cls_prob[ci] = np.asarray(d["cls_prob_val"], dtype=np.float32)

        return cls(
            Strokes(prob, valid, edges=edges, groups=groups, oprob=oprob, cls_prob=cls_prob, class_names=class_names),
            path=str(d.get("path") or "payload0.json"),
            threshold=float(d.get("threshold", 0.5)),
            arch=str(d.get("arch") or ""),
            speed=d.get("speed_ms"),
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "Results":
        """Load a raw prediction ``.npz`` written by ``Results.save_npz``."""
        with np.load(path, allow_pickle=False) as z:
            prob = z["prob"].astype(np.float32)
            valid = z["valid"].astype(bool) if "valid" in z else np.ones(len(prob), dtype=bool)
            thr = float(z["threshold"]) if "threshold" in z else 0.5
            meta = json.loads(str(z["meta"])) if "meta" in z else {}
            edges = (z["edge_pairs"].astype(np.int64), z["edge_prob"].astype(np.float32)) if "edge_pairs" in z else None
            groups = None
            if "group_instance_id" in z:
                groups = {
                    "instance_id": z["group_instance_id"].astype(np.int64),
                    "cluster_ids": z["group_cluster_ids"].astype(np.int64),
                    "cluster_probs": z["group_cluster_probs"].astype(np.float32),
                }
                if "group_stroke_cls" in z:
                    groups["stroke_cls"] = z["group_stroke_cls"].astype(np.float32)
            oprob = z["oprob"].astype(np.float32) if "oprob" in z else None
            cls_prob = z["cls_prob"].astype(np.float32) if "cls_prob" in z else None
            class_names = tuple(meta["class_names"]) if meta.get("class_names") else None
        return cls(
            Strokes(prob, valid, edges=edges, groups=groups, oprob=oprob, cls_prob=cls_prob, class_names=class_names),
            path=str(meta.get("path") or Path(path).name),
            threshold=thr,
            arch=str(meta.get("arch") or ""),
            speed=meta.get("speed_ms"),
        )

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
        meta_d: dict[str, Any] = {"path": self.path, "arch": self.arch, "speed_ms": self.speed}
        if self.class_names is not None:
            meta_d["class_names"] = list(self.class_names)
        arrays: dict[str, Any] = {
            "prob": self.prob.astype(np.float32),
            "valid": self.valid.astype(bool),
            "threshold": np.array(self.threshold, dtype=np.float32),
            "meta": np.array(json.dumps(meta_d, ensure_ascii=False)),
        }
        if self.cls_prob is not None:
            arrays["cls_prob"] = np.asarray(self.cls_prob, dtype=np.float32)
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
        if self.cls_prob is not None:
            out["cls_prob"] = self.cls_prob
            if self.class_names is not None:
                out["class_names"] = self.class_names
        return out

    def __repr__(self) -> str:
        return f"openstroke.results.Results(path={self.path!r}, shapes={len(self)}, kept={self.kept}, arch={self.arch!r})"
