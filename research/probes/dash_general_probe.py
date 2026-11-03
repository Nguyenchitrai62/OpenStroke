"""Is the features step what hurts dashed pipes? (1) Dashes against solid strokes among the true strokes, on every val page of a run, from saved scores. (2) How far MB TH4's true dashes fall outside the range of true strokes on that fold's train pages, feature by feature. No model run.

    py -3.12 research/probes/dash_general_probe.py e54 3 datasets/e6 doc000196_p0001_gt123
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dash_trace_probe import dash_runs  # noqa: E402
from openstroke.features import FEATURE_GROUPS, FEATURE_NAMES, page_tokens  # noqa: E402
from openstroke.folds import load, resolve  # noqa: E402
from openstroke.postprocess import style_groups  # noqa: E402
from openstroke.relabel import STYLE_KEYS, page_segments_of  # noqa: E402

MAX_TRAIN_PAGES = 24


def share(m, p, L, t=0.5):
    return float(L[m & (p >= t)].sum() / max(L[m].sum(), 1e-9))


def dashes_of(d):
    segs = page_segments_of(d)
    return dash_runs(segs.seg, segs.shape_offsets, style_groups({k: d[k] for k in STYLE_KEYS})) >= 0


def part1(run, data):
    tot = {"dash": np.zeros(2), "solid": np.zeros(2)}
    for f in sorted(Path(f"runs_colab/{run}").glob("fold*")):
        z = np.load(f / "val_scores.npz")
        for pg in json.loads((f / "result.json").read_text(encoding="utf-8"))["pages"]:
            s = pg["stem"]
            d = np.load(data / f"{s}.npz")
            d = {k: d[k] for k in ("label", "length", "segments", "shape_offsets", *STYLE_KEYS)}
            y, L = d["label"].astype(bool), d["length"].astype(np.float64)
            p = np.zeros(len(y))
            p[z[f"{s}__index"]] = z[s]
            m = dashes_of(d)
            a, b = y & m, y & ~m
            tot["dash"] += [L[a & (p >= 0.5)].sum(), L[a].sum()]
            tot["solid"] += [L[b & (p >= 0.5)].sum(), L[b].sum()]
            if L[a].sum() > 0.05 * L[y].sum():
                print(f"  {s[:9]} dashes {L[a].sum() / L[y].sum():.0%} of true: p>=0.5 dash {share(a, p, L):.2f}, solid {share(b, p, L):.2f}")
    print(f"{run} all val pages: true length with p>=0.5: dash {tot['dash'][0] / max(tot['dash'][1], 1):.3f} of {tot['dash'][1]:.0f} pt, solid {tot['solid'][0] / max(tot['solid'][1], 1):.3f} of {tot['solid'][1]:.0f} pt")


def part2(run, fold, data, stem):
    args = json.loads(Path(f"runs_colab/{run}/args.json").read_text(encoding="utf-8"))
    man = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    rows, _ = resolve(man, load(data / "folds_cv_pseudo.json"))
    train = [s for s, k, mode in rows if k != int(fold) and mode in ("fire", "hvac") and s != "doc000153_p0001_gt103"]
    rng = np.random.default_rng(0)
    pick = [train[i] for i in rng.permutation(len(train))[:MAX_TRAIN_PAGES]]
    fd = args.get("feature_drop") or []
    dropped = {n for g in (fd.split(",") if isinstance(fd, str) else fd) if g for n in FEATURE_GROUPS.get(g, ())}
    xs = []
    for s in pick:
        t = page_tokens(data / f"{s}.npz", "hvac")
        xs.append(t.x[t.keep & t.label.astype(bool)])
    x = np.concatenate(xs)
    lo, hi = np.percentile(x, 0.5, axis=0), np.percentile(x, 99.5, axis=0)
    z = np.load(data / f"{stem}.npz")
    d = {k: z[k] for k in ("label", "length", "segments", "shape_offsets", *STYLE_KEYS)}
    t = page_tokens(data / f"{stem}.npz", "hvac")
    y = d["label"].astype(bool) & t.keep
    m = dashes_of(d)
    X = t.x
    print(f"{len(pick)} train pages of fold {fold}, {len(x)} true strokes; dropped features {sorted(dropped)}")
    for name, q in (("true dashes", y & m), ("true solid", y & ~m)):
        out = (X[q] < lo) | (X[q] > hi)
        order = np.argsort(-out.mean(0))
        print(f" {name}: {int(q.sum())} strokes; any kept feature outside range {np.mean(out[:, [i for i, n in enumerate(FEATURE_NAMES) if n not in dropped]].any(1)):.2f}; most outside:", [(FEATURE_NAMES[i], round(float(out[:, i].mean()), 2)) for i in order[:8] if out[:, i].mean() > 0.01])
    q = y & m
    print(" true dashes, median against train median:", [(FEATURE_NAMES[FEATURE_NAMES.index(n)], round(float(np.median(X[q][:, FEATURE_NAMES.index(n)])), 2), round(float(np.median(x[:, FEATURE_NAMES.index(n)])), 2)) for n in ("log_len", "log_bbox_diag", "nb10_log", "nb50_log", "nn_dist_log", "nb50_same_style", "style_len_share", "log_width")])


if __name__ == "__main__":
    run, fold, data, stem = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
    if "--skip-scores" not in sys.argv:
        part1(run, data)
    part2(run, fold, data, stem)
