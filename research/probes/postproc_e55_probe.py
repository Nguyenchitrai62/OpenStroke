"""E55 / H72: FIRE post-processing steps on saved scores, on the new labels. No model run. Each arm refits seed and group for its own steps.

    py -3.12 research/probes/postproc_e55_probe.py e37 datasets/e6
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.metrics import length_counts, prf  # noqa: E402
from openstroke.postprocess import GREY_MAX_SHARE, GREY_MIN_LUM, GREY_SPREAD, HVAC_LAYER_MEAN, IS_PIPE, NOT_PIPE, OP_CURVE, along_confident, bridge_curves, by_group, compact_strokes, fill_strokes, hvac_input, mesh_strokes, style_groups  # noqa: E402
from openstroke.relabel import STYLE_KEYS, page_segments_of  # noqa: E402

SEEDS = (0.45, 0.55, 0.65, 0.75)
GROUP = (0.6, 0.8, 0.95, 2.0)
F2_MARGIN, PAGE_RECALL = 0.02, 0.8
ALONE = 50.0  # pt, a connected cluster shorter than this is not a pipe network
GRID = [(t, a) for t in SEEDS for a in GROUP]
ARMS = [(), ("name",), ("grey",), ("rescue",), ("name", "grey", "rescue"), ("name", "grey", "rescue", "trust")]
TRUST, TRUST_P = 0.5, 0.9  # the pipe-named layers hold this much of the mask's length: outside them keep only p above this


def apart(mask, segs, length, p, limit):
    """Clusters of strokes in the mask that touch nothing else and stay short."""

    idx = np.flatnonzero(mask[segs.owner])
    if not len(idx):
        return np.zeros(len(mask), bool)
    lines = shapely.linestrings(segs.seg[idx].reshape(-1, 2, 2))
    a, b = shapely.STRtree(lines).query(lines, predicate="dwithin", distance=2.0)
    label = connected_components(coo_matrix((np.ones(len(a), bool), (a, b)), shape=(len(idx), len(idx))), directed=False)[1]
    size = np.bincount(label, np.hypot(*(segs.seg[idx, 2:] - segs.seg[idx, :2]).T))
    out = np.zeros(len(mask), bool)
    out[segs.owner[idx[size[label] < limit]]] = True
    return out


def load(job):
    run, fold, stem, data = job
    z = np.load(f"runs_colab/{run}/fold{fold}/val_scores.npz")
    d = np.load(Path(data) / f"{stem}.npz")
    d = {k: d[k] for k in d.files}
    if json.loads(str(d["meta"]))["mode"] != "fire":
        return None
    y, L = d["label"].astype(bool), d["length"].astype(np.float64)
    p = np.zeros(len(y))
    p[z[f"{stem}__index"]] = z[stem]
    segs = page_segments_of(d)
    attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
    group = style_groups(attrs)
    names = np.array([str(x) for x in d["layer_names"]])
    lay = d["layer_id"].astype(np.int64)
    rgb = np.nan_to_num(d["stroke_rgb"], nan=0.0)
    grey = (rgb.max(1) - rgb.min(1) < GREY_SPREAD) & (rgb.mean(1) >= GREY_MIN_LUM)
    conf = p >= 0.5
    grey_drop = grey if L[conf & grey].sum() < GREY_MAX_SHARE * max(L[conf].sum(), 1e-9) else np.zeros(len(y), bool)
    not_pipe = np.array([bool(NOT_PIPE.search(n)) for n in names])[lay]
    is_pipe = np.array([bool(IS_PIPE.search(n)) and not NOT_PIPE.search(n) for n in names])[lay]
    # circles and closed marks on a pipe layer are not pipe
    first, last = segs.seg[d["shape_offsets"][:-1].clip(max=len(segs.seg) - 1), :2], segs.seg[(d["shape_offsets"][1:] - 1).clip(min=0), 2:]
    closed = np.hypot(*(first - last).T) < 0.5
    curve = d["op_counts"][:, OP_CURVE] > 0
    rescue = is_pipe & ~closed & ~curve
    chosen = np.isin(names[lay], hvac_input(p, L, list(names[lay]), HVAC_LAYER_MEAN)[0]) if len(np.unique(lay)) > 1 else np.ones(len(y), bool)
    junk = mesh_strokes(segs, group, L) | fill_strokes(segs, group, L) | compact_strokes(segs, L)
    return dict(stem=stem, y=y, L=L, p=p, segs=segs, group=group, curve=curve, junk=junk, chosen=chosen, grey=grey_drop, not_pipe=not_pipe, rescue=rescue,
                along={t: along_confident(segs, p, t) for t in SEEDS})


def mask_of(g, q, steps):
    m = g["along"][q[0]] | by_group(g["p"], g["L"], g["group"], q[1])
    m = m & g["chosen"]
    if "trust" in steps and g["L"][g["rescue"]].sum() >= TRUST * max(g["L"][m].sum(), 1e-9):
        m = m & (g["p"] >= TRUST_P)  # the layer names can be believed on this page: elsewhere only what the model is very sure of
    if "rescue" in steps:
        m = m | g["rescue"]
    m = m | bridge_curves(g["segs"], m, g["curve"])
    m = m & ~g["junk"]
    if "name" in steps:
        m = m & ~g["not_pipe"]
    if "grey" in steps:
        m = m & ~g["grey"]
    if "apart" in steps:
        m = m & ~apart(m, g["segs"], g["L"], g["p"], ALONE)
    return m


def page_counts(job):
    """Counts for every arm and threshold of one page: a page's arrays stay in its worker, memory is short with the app running."""

    g = load(job)
    if g is None:
        return None
    return g["stem"], {steps: {q: length_counts(mask_of(g, q, steps), g["y"], g["L"]) for q in GRID} for steps in ARMS}


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
    out = {}
    for steps in ARMS:
        q, c, low = fit(rows, steps)
        m = prf(*np.sum(c, 0))
        print(f"{'+'.join(steps) or 'base':26s} seed {q[0]} group {q[1]} F2 {m['f2']:.3f} R {m['recall']:.3f} P {m['precision']:.3f} low recall {low}", flush=True)
        out["+".join(steps) or "base"] = {"params": list(q), "pages": {r[0]: [float(x) for x in k] for r, k in zip(rows, c)}}
    Path("runs_colab/e55").mkdir(exist_ok=True)
    Path(f"runs_colab/e55/{run}_fire_arms.json").write_text(json.dumps(out), encoding="utf-8")
    for k in ("base", "name+grey+rescue", "name+grey+rescue+trust"):
        print(k)
        for s, v in out[k]["pages"].items():
            m = prf(*v)
            print(f"  {s[:9]} F2 {m['f2']:.3f} R {m['recall']:.2f} P {m['precision']:.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
