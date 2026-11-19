"""Length-weighted metrics of the ``true`` label (AGENTS.md section 4): recall first, F2 to pick thresholds."""

from __future__ import annotations

import numpy as np


def prf(tp: float, fp: float, fn: float, beta: float = 2.0) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r) if p + r > 0 else 0.0
    return {"precision": p, "recall": r, f"f{beta:g}": f}


def length_counts(pred: np.ndarray, label: np.ndarray, length: np.ndarray) -> tuple[float, float, float]:
    pred = pred.astype(bool)
    lab = label.astype(bool)
    L = length.astype(np.float64)
    return float(L[pred & lab].sum()), float(L[pred & ~lab].sum()), float(L[~pred & lab].sum())


def sweep(scores: list[np.ndarray], labels: list[np.ndarray], lengths: list[np.ndarray],
          thresholds: np.ndarray | None = None, beta: float = 2.0, *, equal_pages: bool = False) -> dict:
    """Length-weighted metrics and one common threshold, selected by micro F2 or mean page F2."""

    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for t in thresholds:
        tp = fp = fn = 0.0
        page_scores = []
        for s, y, L in zip(scores, labels, lengths):
            a, b, c = length_counts(s >= t, y, L)
            tp, fp, fn = tp + a, fp + b, fn + c
            if equal_pages:
                page_scores.append(prf(a, b, c, beta)[f"f{beta:g}"])
        row = {"threshold": float(t), **prf(tp, fp, fn, beta)}
        if page_scores:
            row.update(macro_f2=float(np.mean(page_scores)), worst_f2=min(page_scores))
        rows.append(row)
    key = f"f{beta:g}"
    best = max(rows, key=lambda r: (r.get("macro_f2", r[key]), r.get("worst_f2", r[key])))
    return {"best": best, "curve": rows}


def per_page(scores, labels, lengths, stems, threshold: float, beta: float = 2.0) -> list[dict]:
    out = []
    for s, y, L, stem in zip(scores, labels, lengths, stems):
        tp, fp, fn = length_counts(s >= threshold, y, L)
        out.append({"stem": stem, **prf(tp, fp, fn, beta), "true_length_share": float(L[y.astype(bool)].sum() / max(L.sum(), 1e-9)),
                    "pred_length_share": float(L[s >= threshold].sum() / max(L.sum(), 1e-9))})
    return out
