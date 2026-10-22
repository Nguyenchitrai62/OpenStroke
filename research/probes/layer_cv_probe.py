"""E24 scoring: layer selection of the two-head model over every GT page (val of its own fold), next to the AI_Takeoff
selector on the same pages (pages with real layers only: the pipeline's ``layer_1`` pages have no comparable layer list).

Selectors: ``head`` = layer-head probability (per token, ``val_scores.npz`` ``<stem>__layer``), ``agg`` = length-weighted mean
shape probability of the layer, ``pipeline`` = AI_Takeoff ``result.json`` layers. A layer is selected when its score >= t; ``kept``
and ``purity`` as in ``layer_probe.py``. For each model selector: the largest ``kept`` with pooled purity >= the pipeline's, at
one global threshold, and the pages that keep < 0.5 there.

    py -3.12 research/probes/layer_cv_probe.py datasets/e3 runs_colab/e24
"""

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer_probe import BE, selected_layers  # noqa: E402

THRS = np.round(np.arange(0.02, 0.96, 0.01), 2)


def main(data, run):
    data, run = Path(data), Path(run)
    man = {r["stem"]: r for r in json.loads((data / "manifest.json").read_text(encoding="utf-8"))["records"]}
    con = sqlite3.connect((BE / "data" / "ai_takeoff.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    jobs = defaultdict(list)
    for jid, doc, mode in con.execute("select id, document_id, mode from jobs where status = 'succeeded'"):
        jobs[(doc, mode)].append(jid)
    pages = []
    for fd in sorted(run.glob("fold*")):
        z = np.load(fd / "val_scores.npz")
        for stem in [f for f in z.files if not f.endswith(("__index", "__layer"))]:
            r = man[stem]
            a = np.load(data / r["npz"])
            y, L, lid = a["label"].astype(np.float64), a["length"].astype(np.float64), a["layer_id"].astype(np.int64)
            n = len(y)
            p = np.zeros(n)
            p[z[f"{stem}__index"]] = z[stem]
            q = np.zeros(n)
            if f"{stem}__layer" in z.files:
                q[z[f"{stem}__index"]] = z[f"{stem}__layer"]
            tot = np.bincount(lid, weights=L)
            tru = np.bincount(lid, weights=L * y, minlength=len(tot))
            W = np.maximum(tot, 1e-9)
            sc = {"agg": np.bincount(lid, weights=L * p, minlength=len(tot)) / W,
                  "head": np.bincount(lid, weights=L * q, minlength=len(tot)) / W}
            js = jobs.get((r["document_id"], r["mode"]))
            sel, field = selected_layers(max(js), r["page_number"]) if js else (None, None)
            if sel is None or field != "layer" or tru.sum() <= 0:
                continue
            names = [str(s) for s in a["layer_names"]]
            pipe = np.array([nm in sel for nm in names] + [False] * max(0, len(tot) - len(names)))[: len(tot)]
            pages.append({"stem": stem, "file": r["filename"][:30], "mode": r["mode"], "fold": fd.name, "tot": tot,
                          "tru": tru, "pipe": pipe, **sc})
    T = sum(pg["tru"].sum() for pg in pages)

    def pooled(mask_of):
        tp = sum(pg["tru"][mask_of(pg)].sum() for pg in pages)
        sl = sum(pg["tot"][mask_of(pg)].sum() for pg in pages)
        return tp / T, tp / max(sl, 1e-9)

    pk, pp = pooled(lambda pg: pg["pipe"])
    print(f"{len(pages)} GT pages with real layers (val of their own fold)\n")
    print(f"pipeline selector:  kept {pk:.3f}  purity {pp:.3f}")
    best = {}
    for name in ("head", "agg"):
        curve = [(t, *pooled(lambda pg, t=t: pg[name] >= t)) for t in THRS]
        ok = [c for c in curve if c[2] >= pp]
        t, k, pu = max(ok, key=lambda c: c[1]) if ok else max(curve, key=lambda c: c[2])
        best[name] = t
        hi = max((c for c in curve if c[1] >= 0.95), key=lambda c: c[2], default=None)
        print(f"model {name:4s}:  at purity >= pipeline ({pp:.3f}): thr {t:.2f} kept {k:.3f} purity {pu:.3f}"
              + (f" | at kept >= 0.95: thr {hi[0]:.2f} purity {hi[2]:.3f}" if hi else ""))
    print(f"\n{'page':24s} {'mode':5s} {'file':30s} {'fold':6s}  pipe kept/pur   head kept/pur   agg kept/pur")
    low = defaultdict(list)
    for pg in pages:
        row = []
        for name, m in (("pipe", pg["pipe"]), ("head", pg["head"] >= best["head"]), ("agg", pg["agg"] >= best["agg"])):
            k = pg["tru"][m].sum() / pg["tru"].sum()
            u = pg["tru"][m].sum() / max(pg["tot"][m].sum(), 1e-9)
            row.append(f"{k:.3f}/{u:.3f}")
            if k < 0.5:
                low[name].append(pg["stem"])
        print(f"{pg['stem']:24s} {pg['mode']:5s} {pg['file']:30s} {pg['fold']:6s}  " + "   ".join(row))
    for name in ("pipe", "head", "agg"):
        print(f"pages kept < 0.5 by {name}: {len(low[name])} {low[name]}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
