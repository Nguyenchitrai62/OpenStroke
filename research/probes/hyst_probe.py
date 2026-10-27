"""E29: hysteresis + connected-component post-processing on the link graph (research/HYPOTHESES.md E29).

Rule ``hyst(t_hi, t_lo, L_min)``: keep a shape when p >= t_hi, or when p >= t_lo and it is linked (through the g3 link graph, over
shapes with p >= t_lo) to a shape with p >= t_hi; then drop every kept connected component shorter than L_min pt. Parameters are
picked on the TUNE runs (5-fold val scores) and applied unchanged to the TEST runs. ``none`` = the model with one pooled threshold.

    py -3.12 research/probes/hyst_probe.py --tune e24 e26 --test e28
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.metrics import length_counts, prf  # noqa: E402

DATA = Path("datasets/e3")
T_HI = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
T_LO = [0.03, 0.05, 0.08, 0.12, 0.16, 0.2, 0.3]
L_MIN = [0, 50, 150, 400, 1000]


def load_run(run):
    pages = []
    for k in range(5):
        z = np.load(f"runs_colab/{run}/fold{k}/val_scores.npz")
        for s in [f for f in z.files if not f.endswith(("__index", "__layer"))]:
            a = np.load(DATA / f"{s}.npz")
            y, L = a["label"].astype(bool), a["length"].astype(np.float64)
            p = np.zeros(len(y))
            p[z[f"{s}__index"]] = z[s]
            e = np.load(sorted(DATA.glob(f"{s}.graph.*.npz"))[0])["edges"].astype(np.int64)
            pages.append((s, y, L, p, e))
    return pages


def hyst(p, L, e, t_hi, t_lo, l_min):
    n = len(p)
    cand = p >= t_lo
    if not cand.any():
        return cand
    ee = e[cand[e[:, 0]] & cand[e[:, 1]]]
    g = coo_matrix((np.ones(len(ee)), (ee[:, 0], ee[:, 1])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    strong = np.zeros(comp.max() + 1, bool)
    strong[np.unique(comp[cand & (p >= t_hi)])] = True
    keep = cand & strong[comp]
    if l_min > 0:
        clen = np.bincount(comp[keep], weights=L[keep], minlength=comp.max() + 1)
        keep &= clen[comp] >= l_min
    return keep


def pooled(pages, fn):
    c = np.sum([length_counts(fn(p, L, e), y, L) for _, y, L, p, e in pages], 0)
    return prf(*c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    a = ap.parse_args()
    tune = [pg for r in a.tune for pg in load_run(r)]
    best, best_f2 = None, -1
    for t_hi, t_lo, l_min in itertools.product(T_HI, T_LO, L_MIN):
        if t_lo > t_hi:
            continue
        m = pooled(tune, lambda p, L, e: hyst(p, L, e, t_hi, t_lo, l_min))
        if m["f2"] > best_f2:
            best, best_f2 = (t_hi, t_lo, l_min), m["f2"]
    thr_tune = max(np.round(np.arange(0.02, 0.96, 0.01), 2), key=lambda t: pooled(tune, lambda p, L, e: p >= t)["f2"])
    print(f"tune {a.tune}: none (thr {thr_tune:.2f}) F2 {pooled(tune, lambda p, L, e: p >= thr_tune)['f2']:.3f} | "
          f"hyst {best} F2 {best_f2:.3f}")
    for run in a.test:
        test = load_run(run)
        thr = max(np.round(np.arange(0.02, 0.96, 0.01), 2), key=lambda t: pooled(test, lambda p, L, e: p >= t)["f2"])
        m0 = pooled(test, lambda p, L, e: p >= thr)
        m0t = pooled(test, lambda p, L, e: p >= thr_tune)
        m1 = pooled(test, lambda p, L, e: hyst(p, L, e, *best))
        print(f"test {run}: none@own-best-thr {thr:.2f} F2 {m0['f2']:.3f} R {m0['recall']:.3f} P {m0['precision']:.3f} | "
              f"none@tune-thr {thr_tune:.2f} F2 {m0t['f2']:.3f} | hyst{best} F2 {m1['f2']:.3f} R {m1['recall']:.3f} P {m1['precision']:.3f}")
        for s, y, L, p, e in test:
            k0, k1 = p >= thr, hyst(p, L, e, *best)
            f0, f1 = prf(*length_counts(k0, y, L)), prf(*length_counts(k1, y, L))
            print(f"   {s[:22]:22s} none {f0['f2']:.3f} (P {f0['precision']:.2f})  hyst {f1['f2']:.3f} (P {f1['precision']:.2f})")


if __name__ == "__main__":
    main()
