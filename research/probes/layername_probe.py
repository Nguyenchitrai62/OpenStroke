"""Do layer *names* carry meaning across drawings? (AGENTS.md section 6 assumed not: layer id = grouping only.)

No learning beyond counting, like ``style_probe.py``. Per fold: from the train pages, length-weighted naive-Bayes
log-odds of ``true`` for every token of a layer name (words split on non-alphanumerics, lower-cased, accents removed,
plus their character 3-grams). A val shape's score is the summed log-odds of its layer's tokens (unseen tokens 0;
pages without layer names get the prior). Threshold swept per fold as the trainer does; micro over folds.

    py -3.12 research/probes/layername_probe.py datasets/e1 datasets/e1/folds_plumbing.json > research/probes/layername_probe.log
"""

import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.folds import resolve  # noqa: E402
from openstroke.metrics import prf, sweep  # noqa: E402


def tokens(name: str) -> set[str]:
    s = unicodedata.normalize("NFKD", name.lower()).encode("ascii", "ignore").decode()
    words = [w for w in re.split(r"[^a-z0-9]+", s) if w and not w.isdigit()]
    out = {f"w:{w}" for w in words}
    for w in words:
        p = f"^{w}$"
        out |= {f"c:{p[i:i + 3]}" for i in range(len(p) - 2)}
    return out


def main(data, folds_path):
    data = Path(data)
    man = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    entries, _ = resolve(man, json.loads(Path(folds_path).read_text(encoding="utf-8")))
    pages = {}
    for stem, fold, mode in entries:
        z = np.load(data / f"{stem}.npz")
        L = z["length"].astype(np.float64)
        y = z["label"].astype(bool)
        lid = z["layer_id"].astype(np.int64)
        names = [str(n) for n in z["layer_names"]]
        n = len(names)
        lay_len = np.bincount(lid, weights=L, minlength=n)
        lay_true = np.bincount(lid, weights=L * y, minlength=n)
        pages[stem] = dict(fold=fold, mode=mode, L=L, y=y, lid=lid, names=names, lay_len=lay_len, lay_true=lay_true,
                           tok=[tokens(nm) for nm in names])
    folds = sorted({p["fold"] for p in pages.values()})
    allS, allY, allL, by_mode = [], [], [], defaultdict(lambda: [0.0, 0.0, 0.0])
    for k in folds:
        pos, neg = defaultdict(float), defaultdict(float)
        P = N = 0.0
        for p in pages.values():
            if p["fold"] == k:
                continue
            for t_set, a, t in zip(p["tok"], p["lay_len"], p["lay_true"]):
                for tok in t_set:
                    pos[tok] += t
                    neg[tok] += a - t
            P += p["lay_true"].sum()
            N += p["lay_len"].sum() - p["lay_true"].sum()
        a = 1.0  # smoothing in "pt"
        lo = {tok: np.log((pos[tok] + a) / (P + a)) - np.log((neg[tok] + a) / (N + a)) for tok in set(pos) | set(neg)}
        prior = np.log(P / N)
        S, Y, Ls, modes = [], [], [], []
        for stem, p in pages.items():
            if p["fold"] != k:
                continue
            lay_score = np.array([prior + sum(lo.get(t, 0.0) for t in ts) for ts in p["tok"]])
            sc = 1 / (1 + np.exp(-np.clip(lay_score, -30, 30)))[p["lid"]]
            S.append(sc), Y.append(p["y"]), Ls.append(p["L"]), modes.append(p["mode"])
        b = sweep(S, Y, Ls)["best"]
        print(f"fold {k}: F2 {b['f2']:.3f} (R {b['recall']:.3f} P {b['precision']:.3f} t {b['threshold']:.2f}) pages {len(S)}")
        for sc, y, L, m in zip(S, Y, Ls, modes):
            hit = sc >= b["threshold"]
            for key in ("all", m):
                v = by_mode[key]
                v[0] += L[hit & y].sum()
                v[1] += L[hit & ~y].sum()
                v[2] += L[~hit & y].sum()
    for key, v in sorted(by_mode.items()):
        r = prf(*v)
        print(f"MICRO {key:8s}: F2 {r['f2']:.3f} (R {r['recall']:.3f} P {r['precision']:.3f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
