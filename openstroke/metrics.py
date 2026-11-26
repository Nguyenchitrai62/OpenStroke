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


#: Canonical 3-class FIRE stroke taxonomy (0=other, 1=main, 2=branch)
FIRE3_CLASSES = ("other", "main", "branch")


def multiclass_metrics(
    cls_probs: list[np.ndarray],
    labels: list[np.ndarray],
    lengths: list[np.ndarray],
    *,
    names: tuple[str, ...] = FIRE3_CLASSES,
    beta: float = 2.0,
    fg_threshold: float | None = None,
) -> dict:
    """Length-weighted per-class P/R/F2 for multi-class stroke models (e.g. FIRE 3-class: other/main/branch).

    When ``fg_threshold`` is None, predicts ``argmax(cls_prob)`` per stroke; when set, predicts class 0 if
    ``1 - p_other < fg_threshold`` else ``1 + argmax(p[:, 1:])``. Returns ``{"per_class": ..., "pipe": ...,
    "macro_fg_f2": ..., "fitness": ...}`` where ``fitness = 0.40 * pipe_F2 + 0.60 * macro_fg_f2``."""
    n_cls = len(names)
    tp = np.zeros(n_cls, dtype=np.float64)
    fp = np.zeros(n_cls, dtype=np.float64)
    fn = np.zeros(n_cls, dtype=np.float64)
    p_tp = p_fp = p_fn = 0.0

    for pr, y, L in zip(cls_probs, labels, lengths):
        pr = np.asarray(pr, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        Lf = np.asarray(L, dtype=np.float64)
        if fg_threshold is None:
            pred = pr.argmax(axis=1)
        else:
            p_fg = 1.0 - pr[:, 0]
            pred = np.where(p_fg >= fg_threshold, 1 + pr[:, 1:].argmax(axis=1), 0)
        for c in range(n_cls):
            pm = pred == c
            ym = y == c
            tp[c] += float(Lf[pm & ym].sum())
            fp[c] += float(Lf[pm & ~ym].sum())
            fn[c] += float(Lf[~pm & ym].sum())
        a, b, e = length_counts(pred > 0, y > 0, Lf)
        p_tp += a; p_fp += b; p_fn += e

    key = f"f{beta:g}"
    per_cls = {names[c]: prf(tp[c], fp[c], fn[c], beta) for c in range(n_cls)}
    fg_f2s = [per_cls[names[c]][key] for c in range(1, n_cls) if (tp[c] + fn[c]) > 0]
    macro_fg = float(np.mean(fg_f2s)) if fg_f2s else 0.0
    pipe_res = prf(p_tp, p_fp, p_fn, beta)
    fitness = 0.40 * float(pipe_res[key]) + 0.60 * macro_fg
    return {
        "per_class": per_cls,
        "pipe": pipe_res,
        "macro_fg_f2": macro_fg,
        "fitness": fitness,
    }
