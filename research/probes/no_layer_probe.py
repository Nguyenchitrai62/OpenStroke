"""E55: no layers at all, only colour, width and type. The best possible choice of those groups given the labels (a ceiling), and each run's FIRE mask with layers taken out of the post-processing.
The scores themselves came from a model that saw layer ids as groups, so the mask is an optimistic stand-in for a PDF with no layers. No model run.

    py -3.12 research/probes/no_layer_probe.py e37 datasets/e6
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.metrics import length_counts, prf  # noqa: E402
from openstroke.postprocess import OP_CURVE, along_confident, bridge_curves, by_group, compact_strokes, fill_strokes, mesh_strokes  # noqa: E402
from openstroke.relabel import page_segments_of  # noqa: E402

SEEDS = (0.45, 0.55, 0.65, 0.75)
GROUP = (0.6, 0.8, 0.95, 2.0)
GRID = [(t, a) for t in SEEDS for a in GROUP]
THRESHOLDS = np.arange(0.05, 0.96, 0.05)


def plain_groups(d):
    rgb = np.nan_to_num(d["stroke_rgb"], nan=-1).round(2)
    key = np.c_[rgb, np.round(np.nan_to_num(d["width"], nan=-1), 2), d["type_code"]]
    return np.unique(key, axis=0, return_inverse=True)[1].ravel()


def page(job):
    run, fold, stem, data = job
    z = np.load(f"runs_colab/{run}/fold{fold}/val_scores.npz")
    d = np.load(Path(data) / f"{stem}.npz")
    d = {k: d[k] for k in d.files}
    mode = json.loads(str(d["meta"]))["mode"]
    y, L = d["label"].astype(bool), d["length"].astype(np.float64)
    p = np.zeros(len(y))
    p[z[f"{stem}__index"]] = z[stem]
    g = plain_groups(d)
    n = int(g.max()) + 1
    best = (np.bincount(g[y], L[y], n) >= 0.5 * np.bincount(g, L, n))[g]
    out = {"stem": stem, "mode": mode, "groups": n, "true_groups": int(np.unique(g[y]).size), "layers": int(d["layer_id"].max()) + 1, "best": length_counts(best, y, L),
           "raw": {round(float(t), 2): length_counts(p >= t, y, L) for t in THRESHOLDS}}
    if mode == "fire":
        segs = page_segments_of(d)
        curve = d["op_counts"][:, OP_CURVE] > 0
        junk = mesh_strokes(segs, g, L) | fill_strokes(segs, g, L) | compact_strokes(segs, L)
        along = {t: along_confident(segs, p, t) for t in SEEDS}
        groups = {a: by_group(p, L, g, a) for a in GROUP}
        out["mask"] = {q: length_counts(((along[q[0]] | groups[q[1]]) | bridge_curves(segs, along[q[0]] | groups[q[1]], curve)) & ~junk, y, L) for q in GRID}
    return out


def patient(job):
    for wait in (0, 60, 180, 300, 600):
        time.sleep(wait)
        try:
            with ProcessPoolExecutor(1) as ex:
                return ex.submit(page, job).result()
        except (RuntimeError, MemoryError, BrokenProcessPool) as e:
            print("retry", job[2], str(e)[:60], flush=True)
    raise RuntimeError(job[2])


def main(run, data):
    jobs = [(run, f.name[4:], pg["stem"], data) for f in sorted(Path(f"runs_colab/{run}").glob("fold*")) for pg in json.loads((f / "result.json").read_text(encoding="utf-8"))["pages"]]
    rows = [patient(j) for j in jobs]
    for mode in ("fire", "hvac"):
        r = [x for x in rows if x["mode"] == mode]
        if not r:
            continue
        m = prf(*np.sum([x["best"] for x in r], 0))
        raw = max((prf(*np.sum([x["raw"][round(float(t), 2)] for x in r], 0)) for t in THRESHOLDS), key=lambda q: q["f2"])
        line = f"{mode}: {len(r)} pages, median {int(np.median([x['groups'] for x in r]))} colour+width groups | best choice F2 {m['f2']:.3f} R {m['recall']:.3f} P {m['precision']:.3f} | {run} raw F2 {raw['f2']:.3f}"
        if mode == "fire":
            f2 = {q: prf(*np.sum([x["mask"][q] for x in r], 0)) for q in GRID}
            q = max(GRID, key=lambda q: f2[q]["f2"])
            line += f" | mask without layers F2 {f2[q]['f2']:.3f} R {f2[q]['recall']:.3f} P {f2[q]['precision']:.3f} at {q}, low recall {sum(prf(*x['mask'][q])['recall'] < 0.8 for x in r)}"
        print(line)
        for x in r:
            a = prf(*x["best"])
            extra = ""
            if mode == "fire":
                b = prf(*x["mask"][q])
                extra = f"| mask F2 {b['f2']:.3f} R {b['recall']:.2f} P {b['precision']:.2f}"
            print(f"  {x['stem'][:9]} layers {x['layers']:4d} groups {x['groups']:3d} true in {x['true_groups']:3d} | best {a['f2']:.3f} (R {a['recall']:.2f} P {a['precision']:.2f}) {extra}")


if __name__ == "__main__":
    main(*sys.argv[1:])
