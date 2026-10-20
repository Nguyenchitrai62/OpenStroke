"""E1 baselines (AGENTS.md section 5), on the same pages and metric as the model.

- ``layer_ceiling``: with GT labels, the best set of whole layers per page (greedy prefix by layer
  precision, the F-beta optimum for a ranking by precision). Upper bound of any layer-selection cascade.
- ``gbdt``: gradient boosting (scikit-learn, BSD) on the hand features of ``features.py``, CV by fold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import prf, sweep


def layer_ceiling(data: Path, stems: list[str], beta: float = 2.0) -> dict:
    rows = []
    tp = fp = fn = 0.0
    for stem in stems:
        z = np.load(data / f"{stem}.npz")
        L = z["length"].astype(np.float64)
        y = z["label"].astype(bool)
        layer = z["layer_id"].astype(np.int64)
        nl = int(layer.max()) + 1
        t_l = np.bincount(layer, weights=np.where(y, L, 0.0), minlength=nl)
        a_l = np.bincount(layer, weights=L, minlength=nl)
        order = np.argsort(-(t_l / np.maximum(a_l, 1e-9)))
        total_true = t_l.sum()
        ctp = np.cumsum(t_l[order])
        cfp = np.cumsum(a_l[order] - t_l[order])
        best = (0.0, 0.0, 0.0, total_true, 0)
        for i in range(nl):
            r = prf(ctp[i], cfp[i], total_true - ctp[i], beta)
            if r[f"f{beta:g}"] > best[0]:
                best = (r[f"f{beta:g}"], ctp[i], cfp[i], total_true - ctp[i], i + 1)
        _, a, b, c, k = best
        tp, fp, fn = tp + a, fp + b, fn + c
        names = z["layer_names"]
        rows.append({"stem": stem, "layers": int(nl), "chosen_layers": int(k), **prf(a, b, c, beta),
                     "chosen": [str(names[j]) for j in order[:k]][:8]})
    return {"all": prf(tp, fp, fn, beta), "pages": rows}


def gbdt(data: Path, folds_path: Path, out: Path, cache: Path | None = None, max_train: int = 3_000_000,
         seed: int = 0, log=print) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    from .folds import load, resolve
    from .metrics import per_page
    from .train import cached_tokens

    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    folds = load(folds_path)
    entries, warn = resolve(manifest, folds)
    for w in warn:
        log("WARNING: " + w)
    cache = cache or out / "cache"
    toks = {s: cached_tokens(data, s, cache, m) for s, _, m in entries}
    rng = np.random.default_rng(seed)
    results = {}
    tp = fp = fn = 0.0
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted({f for _, f, _ in entries}):
        train = [s for s, ff, _ in entries if ff != f]
        val = [s for s, ff, _ in entries if ff == f]
        X = np.concatenate([toks[s].x[toks[s].keep] for s in train])
        Y = np.concatenate([toks[s].label[toks[s].keep] for s in train])
        W = np.concatenate([np.sqrt(toks[s].length[toks[s].keep] + 1.0) for s in train])
        if len(X) > max_train:
            pick = rng.choice(len(X), max_train, replace=False)
            X, Y, W = X[pick], Y[pick], W[pick]
        W = W * np.where(Y > 0, 2.0, 1.0)  # same pos_weight as model v1
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                             early_stopping=False, random_state=seed)
        clf.fit(X, Y, sample_weight=W)
        scores = []
        for s in val:
            t = toks[s]
            sc = np.zeros(len(t.label))
            if t.keep.any():
                sc[t.keep] = clf.predict_proba(t.x[t.keep])[:, 1]
            scores.append(sc)
        sw = sweep(scores, [toks[s].label for s in val], [toks[s].length for s in val])
        th = sw["best"]["threshold"]
        pages = per_page(scores, [toks[s].label for s in val], [toks[s].length for s in val], val, th)
        for s, sc in zip(val, scores):
            y = toks[s].label.astype(bool)
            L = toks[s].length.astype(np.float64)
            p = sc >= th
            tp, fp, fn = tp + L[p & y].sum(), fp + L[p & ~y].sum(), fn + L[~p & y].sum()
        results[f] = {"best": sw["best"], "pages": pages}
        log(f"GBDT fold {f}: R {sw['best']['recall']:.3f} P {sw['best']['precision']:.3f} F2 {sw['best']['f2']:.3f} "
            f"@t={th:.2f} train_tokens={len(X):,}")
        for p in pages:
            log(f"  PAGE {p['stem']}: R {p['recall']:.3f} P {p['precision']:.3f} F2 {p['f2']:.3f}")
    summary = {"all_at_fold_threshold": prf(tp, fp, fn), "folds": results, "fold_status": folds.get("status")}
    log(f"GBDT SUMMARY: R {summary['all_at_fold_threshold']['recall']:.3f} P "
        f"{summary['all_at_fold_threshold']['precision']:.3f} F2 {summary['all_at_fold_threshold']['f2']:.3f}")
    (out / "gbdt_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
