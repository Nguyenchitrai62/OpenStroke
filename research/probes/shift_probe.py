"""Which input features put a validation page outside what training saw? (why some dev pages collapse, F28)

For every hand feature (``features.FEATURE_NAMES``): the length-weighted 1st..99th percentile range of the feature over
the *true* shapes of the train pages; for each val page, the share of its true length whose value lies outside that
range. A feature with a large share on a page that collapses (and small on pages that improve) is a shortcut the
model can learn on train and that breaks on that page.

    py -3.12 research/probes/shift_probe.py datasets/e2 datasets/e2/folds_dev.json fire,hvac > research/probes/shift_probe.log
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.data import load_page  # noqa: E402
from openstroke.features import FEATURE_NAMES  # noqa: E402
from openstroke.folds import resolve  # noqa: E402


def wq(v, w, qs):
    o = np.argsort(v)
    c = np.cumsum(w[o]) / w.sum()
    return np.interp(qs, c, v[o])


def main(data, folds, modes):
    data = Path(data)
    man = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    entries, _ = resolve(man, json.loads(Path(folds).read_text(encoding="utf-8")))
    keep = set(modes.split(","))
    entries = [e for e in entries if e[2] in keep]
    pages = {s: load_page(data, s, m, Path("runs/cache_e2")) for s, _, m in entries}
    fname = {r["stem"]: r["filename"] for r in man["records"]}
    tr = [pages[s] for s, f, _ in entries if f != 0]
    xs = np.concatenate([p.x[p.label > 0] for p in tr])
    ws = np.concatenate([p.length[p.label > 0] for p in tr]).astype(np.float64) + 1e-6
    lo = np.array([wq(xs[:, j], ws, 0.01) for j in range(xs.shape[1])])
    hi = np.array([wq(xs[:, j], ws, 0.99) for j in range(xs.shape[1])])
    rows = []
    for s, f, m in entries:
        if f != 0:
            continue
        p = pages[s]
        t = p.label > 0
        x, L = p.x[t], p.length[t].astype(np.float64)
        out = ((x < lo - 1e-6) | (x > hi + 1e-6))
        share = (out * L[:, None]).sum(0) / max(L.sum(), 1e-9)
        rows.append((s, m, fname[s], share))
    print("share of each val page's TRUE length outside the train-true 1-99 % range, per feature (top 8 per page)")
    for s, m, fn, share in rows:
        top = np.argsort(-share)[:8]
        print(f"\n{s} {m} {fn[:45]}")
        print("   " + "  ".join(f"{FEATURE_NAMES[j]} {share[j]:.0%}" for j in top if share[j] > 0.02))
    print("\nmean over val pages, by feature (top 15)")
    mean = np.mean([r[3] for r in rows], axis=0)
    for j in np.argsort(-mean)[:15]:
        print(f"   {FEATURE_NAMES[j]:22s} {mean[j]:.1%}  " + " ".join(f"{r[3][j]:.0%}" for r in rows))


if __name__ == "__main__":
    main(*sys.argv[1:4])
