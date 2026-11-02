"""E43: the model scores the logic's objects: drop objects whose strokes the model does not believe in (research/HYPOTHESES.md E43).

No model and no logic run: objects cached by ``logic_hvac_probe.py`` (arm union0.1), saved val scores. An object's score is the length-weighted
mean p of the shapes whose midpoint lies inside it (buffered by 2 pt, the label rule's tolerance). Objects below tau are dropped; tau is picked
on TUNE and applied to TEST.

    py -3.12 research/probes/objectveto_probe.py --tune e30 --test e37
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.geometry import PageSegments  # noqa: E402
from openstroke.labels import gt_geometry, label_shapes, refine_hvac  # noqa: E402
from openstroke.metrics import length_counts, prf  # noqa: E402

CACHE = Path("runs_colab/logic_hvac")
TAU = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)


def scores_of(run):
    out = {}
    for k in range(5):
        z = np.load(f"runs_colab/{run}/fold{k}/val_scores.npz")
        for s in [f for f in z.files if not f.endswith(("__index", "__layer"))]:
            out[s] = (z, s)
    return out


def page_counts(run, stem, sc):
    d = np.load(f"datasets/e3/{stem}.npz")
    z, s = sc[stem]
    p = np.zeros(len(d["label"]))
    p[z[f"{s}__index"]] = z[s]
    L = d["length"].astype(np.float64)
    off = d["shape_offsets"].astype(np.int64)
    seg = d["segments"].astype(np.float64)
    owner = np.repeat(np.arange(len(off) - 1), np.diff(off))
    segs = PageSegments(seg=seg, owner=owner, shape_offsets=off)
    objs = json.loads((CACHE / stem / f"{run}_union0.1.json").read_text(encoding="utf-8"))["objects"]
    geo = gt_geometry("hvac", objs)
    keep_names = [o for o in objs if len(o.get("vertices") or []) >= 2]
    mids = shapely.points((seg[:, :2] + seg[:, 2:]) / 2)
    sl = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    tree = shapely.STRtree(mids)
    polys = [shapely.make_valid(shapely.polygons(np.asarray(o["vertices"], float))) if len(o["vertices"]) >= 3 else shapely.linestrings(np.asarray(o["vertices"], float))
             for o in keep_names]
    obj_i, seg_i = tree.query([g.buffer(2.0) for g in polys], predicate="contains") if polys else (np.zeros(0, int), np.zeros(0, int))
    num = np.bincount(obj_i, p[owner[seg_i]] * sl[seg_i], len(polys))
    den = np.bincount(obj_i, sl[seg_i], len(polys))
    obj_score = num / np.maximum(den, 1e-9)
    res = {}
    for tau in TAU:
        kept = [o for o, x in zip(keep_names, obj_score) if x >= tau]
        g = gt_geometry("hvac", kept)
        lab, _ = refine_hvac(segs, g, label_shapes(segs, g), d["layer_id"])
        res[tau] = length_counts(lab.label.astype(bool), d["label"].astype(bool), L)
    print(f"  {run} {stem[:21]} objects {len(geo.geoms)} " + " ".join(f"{t}:{prf(*c)['f2']:.3f}" for t, c in res.items()), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="e30")
    ap.add_argument("--test", default="e37")
    a = ap.parse_args()
    stems = sorted(json.loads((CACHE / f"{a.test}_counts.json").read_text(encoding="utf-8")))
    tot = {}
    for run in (a.tune, a.test):
        sc = scores_of(run)
        rows = [page_counts(run, s, sc) for s in stems]
        tot[run] = {t: np.sum([r[t] for r in rows], 0) for t in TAU}
        for t in TAU:
            m = prf(*tot[run][t])
            print(f"{run} tau {t:<5} F2 {m['f2']:.3f}  R {m['recall']:.3f}  P {m['precision']:.3f}", flush=True)
    best = max(TAU, key=lambda t: prf(*tot[a.tune][t])["f2"])
    m, b = prf(*tot[a.test][best]), prf(*tot[a.test][0.0])
    print(f"tau picked on {a.tune}: {best}; {a.test}: F2 {m['f2']:.3f} R {m['recall']:.3f} P {m['precision']:.3f}; without veto F2 {b['f2']:.3f} R {b['recall']:.3f} P {b['precision']:.3f}")


if __name__ == "__main__":
    main()
