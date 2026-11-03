"""E55 / H72: HVAC post-processing on saved scores and stored logic objects, on the new labels. No model or logic run. One page per worker: memory is short with the app running.

    py -3.12 research/probes/hvac_e55_probe.py e37 datasets/e6
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openstroke.metrics import length_counts, prf  # noqa: E402
from openstroke.postprocess import OP_CURVE, bridge_curves, by_group, compact_strokes, fill_strokes, mesh_strokes, style_groups  # noqa: E402
from openstroke.relabel import STYLE_KEYS, object_strokes, page_segments_of  # noqa: E402
from postproc_e55_probe import F2_MARGIN, GREY_MAX_SHARE, GREY_MIN_LUM, GREY_SPREAD, NOT_PIPE, PAGE_RECALL  # noqa: E402

LOGIC = Path("runs_colab/logic_hvac2")
SURE = (0.7, 0.8, 0.9, 0.95, 2.0)
GROUP = (0.6, 0.8, 0.95, 2.0)
GRID = [(t, a) for t in SURE for a in GROUP]
ARMS = [(), ("name",), ("grey",), ("name", "grey")]


def page_counts(job):
    run, fold, stem, data = job
    d = np.load(Path(data) / f"{stem}.npz")
    d = {k: d[k] for k in d.files}
    f = LOGIC / stem / f"{run}_union0.1.json"
    if json.loads(str(d["meta"]))["mode"] != "hvac" or not f.exists():
        return None
    z = np.load(f"runs_colab/{run}/fold{fold}/val_scores.npz")
    y, L = d["label"].astype(bool), d["length"].astype(np.float64)
    p = np.zeros(len(y))
    p[z[f"{stem}__index"]] = z[stem]
    segs = page_segments_of(d)
    attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
    group = style_groups(attrs)
    obj = object_strokes("hvac", json.loads(f.read_text(encoding="utf-8"))["objects"], segs, attrs, L)
    junk = mesh_strokes(segs, group, L) | fill_strokes(segs, group, L) | compact_strokes(segs, L)
    curve = attrs["op_counts"][:, OP_CURVE] > 0
    names = [str(x) for x in d["layer_names"]]
    lay = d["layer_id"].astype(np.int64)
    not_pipe = np.array([bool(NOT_PIPE.search(n)) for n in names])[lay]
    rgb = np.nan_to_num(d["stroke_rgb"], nan=0.0)
    grey = (rgb.max(1) - rgb.min(1) < GREY_SPREAD) & (rgb.mean(1) >= GREY_MIN_LUM)
    conf = p >= 0.5
    grey = grey if L[conf & grey].sum() < GREY_MAX_SHARE * max(L[conf].sum(), 1e-9) else np.zeros(len(y), bool)
    groups = {a: by_group(p, L, group, a) for a in GROUP}

    def mask(q, steps):
        m = obj | (p >= q[0]) | groups[q[1]]
        m = (m | bridge_curves(segs, m, curve)) & ~junk
        if "name" in steps:
            m = m & ~not_pipe
        if "grey" in steps:
            m = m & ~grey
        return m

    q0 = (2.0, 2.0)
    top = np.argsort(-np.bincount(lay[mask(q0, ()) & ~y], L[mask(q0, ()) & ~y], len(names)))[:3]
    return stem, {s: {q: length_counts(mask(q, s), y, L) for q in GRID} for s in ARMS}, [names[i][:26] for i in top]


def fit(rows, steps):
    counts = {q: [r[1][steps][q] for r in rows] for q in GRID}
    f2 = {q: prf(*np.sum(c, 0))["f2"] for q, c in counts.items()}
    low = {q: sum(prf(*x)["recall"] < PAGE_RECALL for x in c) for q, c in counts.items()}
    near = [q for q in GRID if f2[q] >= max(f2.values()) - F2_MARGIN]
    best = min(near, key=lambda q: (low[q], -f2[q]))
    return best, counts[best], low[best]


def patient(job):
    """One page at a time in its own process, waiting when the machine is out of memory for a while."""

    for wait in (0, 60, 180, 300, 600):
        time.sleep(wait)
        try:
            with ProcessPoolExecutor(1) as ex:
                out = ex.submit(page_counts, job).result()
            print("done", job[2], flush=True)
            return out
        except (RuntimeError, MemoryError, BrokenProcessPool) as e:
            print("retry", job[2], str(e)[:60], flush=True)
    raise RuntimeError(f"out of memory on {job[2]}")


def main(run, data):
    jobs = [(run, f.name[4:], pg["stem"], data) for f in sorted(Path(f"runs_colab/{run}").glob("fold*")) for pg in json.loads((f / "result.json").read_text(encoding="utf-8"))["pages"]]
    rows = [r for r in map(patient, jobs) if r]
    pipe = {x["stem"]: x["counts"] for x in json.loads(Path("runs_colab/e55/pipeline.json").read_text(encoding="utf-8")) if x["counts"]}
    have = [r for r in rows if r[0] in pipe]
    m = prf(*np.sum([pipe[r[0]] for r in have], 0))
    print(f"{len(rows)} pages with logic objects; pipeline on {len(have)} of them F2 {m['f2']:.3f} R {m['recall']:.3f} P {m['precision']:.3f}", flush=True)
    out = {}
    for steps in ARMS:
        q, c, low = fit(rows, steps)
        m = prf(*np.sum(c, 0))
        print(f"{'+'.join(steps) or 'base':12s} sure {q[0]} group {q[1]} F2 {m['f2']:.3f} R {m['recall']:.3f} P {m['precision']:.3f} low recall {low}", flush=True)
        out["+".join(steps) or "base"] = {"params": list(q), "pages": {r[0]: [float(x) for x in k] for r, k in zip(rows, c)}}
    Path(f"runs_colab/e55/{run}_hvac_arms.json").write_text(json.dumps(out), encoding="utf-8")
    q, c, _ = fit(rows, ())
    for r, k in zip(rows, c):
        a, b = prf(*k), prf(*pipe[r[0]]) if r[0] in pipe else None
        print(f"  {r[0][:9]} F2 {a['f2']:.3f} R {a['recall']:.2f} P {a['precision']:.2f} | pipeline {b['f2'] if b else float('nan'):.3f} | most FP {r[2]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
