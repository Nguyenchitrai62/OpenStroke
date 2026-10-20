"""Does the drawing style of pipe strokes carry over between PDFs? A lookup table, no learning.

For each proposed fold: from the train pages, the length share of `true` per style key; score each val shape
with the share of its key (unseen key -> 0); sweep the threshold as the trainer does. Two keys:
  full    = (stroke rgb, fill rgb, width, type)        what a model can memorise per office
  nocolor = (width, type, close_path, has fill, has stroke)
  nocolor+mode = nocolor and the page's mode (FIRE / HVAC): does knowing the task help? (the models do not get it)
Also the share of val true length whose key never occurs in train ("unseen").

A model that only memorises style cannot beat this table on val; if the table is far below the per-page layer
ceiling, cross-PDF transfer has to come from geometry and context, not style.

Run: py -3.12 research/probes/style_probe.py datasets/e0 > research/probes/style_probe.log
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.metrics import prf, sweep  # noqa: E402


def keys(z, kind, mode=None):
    s = np.nan_to_num(z["stroke_rgb"], nan=-1.0)
    f = np.nan_to_num(z["fill_rgb"], nan=-1.0)
    w = np.round(z["width"].astype(np.float64), 2)
    t = z["type_code"].astype(np.int64)
    if kind == "full":
        cols = [np.round(s * 255), np.round(f * 255), w[:, None], t[:, None]]
    else:  # nocolor, nocolor+mode
        cols = [w[:, None], t[:, None], z["close_path"][:, None].astype(np.float64),
                (f[:, 0] >= 0)[:, None].astype(np.float64), (s[:, 0] >= 0)[:, None].astype(np.float64)]
    if kind.endswith("+mode"):
        cols.append(np.full((len(w), 1), 1.0 if mode == "fire" else 0.0))
    a = np.concatenate(cols, axis=1)
    return [tuple(r) for r in a]


def main(data):
    data = Path(data)
    folds = json.loads((data / "folds_proposal.json").read_text(encoding="utf8"))
    man = json.loads((data / "manifest.json").read_text(encoding="utf8"))
    dup = {(p["document_sha256"], p["page_number"]) for p in man["physical_pages"] if p.get("duplicate_gt")}
    fold_of = {(g["sha256"], pg): g["fold"] for g in folds["groups"] for pg in g["pages"]}
    stems = defaultdict(list)
    for r in man["records"]:
        key = (r["document_sha256"], r["page_number"])
        if r.get("status") == "ok" and key not in dup and key in fold_of:
            stems[fold_of[key]].append(r["stem"])
    pages = {s: np.load(data / f"{s}.npz") for f in stems for s in stems[f]}
    mode = {r["stem"]: r["mode"] for r in man["records"]}

    for kind in ("full", "nocolor", "nocolor+mode"):
        K = {s: keys(z, kind, mode[s]) for s, z in pages.items()}
        allS, allY, allL = [], [], []
        print(f"=== key={kind}")
        for k in sorted(stems):
            tot, tru = defaultdict(float), defaultdict(float)
            for f in stems:
                if f == k:
                    continue
                for s in stems[f]:
                    z = pages[s]
                    L, y = z["length"].astype(np.float64), z["label"].astype(bool)
                    for key, l, t in zip(K[s], L, y):
                        tot[key] += l
                        if t:
                            tru[key] += l
            S, Y, Ls = [], [], []
            unseen_true = all_true = 0.0
            for s in stems[k]:
                z = pages[s]
                L, y = z["length"].astype(np.float64), z["label"].astype(bool)
                sc = np.array([tru[key] / tot[key] if tot.get(key) else 0.0 for key in K[s]])
                seen = np.array([key in tot for key in K[s]])
                unseen_true += L[y & ~seen].sum()
                all_true += L[y].sum()
                S.append(sc), Y.append(y), Ls.append(L)
                b = sweep([sc], [y], [L])["best"]
                print(f"  fold {k} {s} {mode[s]:5s} unseen_true {L[y & ~seen].sum() / max(L[y].sum(), 1e-9):5.1%}"
                      f"  F2 {b['f2']:.3f} (R {b['recall']:.3f} P {b['precision']:.3f} t {b['threshold']:.2f})")
            b = sweep(S, Y, Ls)["best"]
            print(f"fold {k}: unseen_true {unseen_true / all_true:5.1%}  F2 {b['f2']:.3f} (R {b['recall']:.3f} "
                  f"P {b['precision']:.3f} t {b['threshold']:.2f})")
            # micro over folds at each fold's own threshold, as the trainer reports
            t = b["threshold"]
            for sc, y, L in zip(S, Y, Ls):
                allS.append((sc >= t).astype(float)), allY.append(y), allL.append(L)
        tp = sum(float(L[(s > 0) & y].sum()) for s, y, L in zip(allS, allY, allL))
        fp = sum(float(L[(s > 0) & ~y].sum()) for s, y, L in zip(allS, allY, allL))
        fn = sum(float(L[(s == 0) & y].sum()) for s, y, L in zip(allS, allY, allL))
        r = prf(tp, fp, fn)
        print(f"MICRO key={kind}: F2 {r['f2']:.3f} (R {r['recall']:.3f} P {r['precision']:.3f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "datasets/e0")
