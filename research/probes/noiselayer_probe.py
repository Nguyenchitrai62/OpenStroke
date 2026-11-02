"""E50: after post-processing, drop whole noise layers (research/HYPOTHESES.md E50; user 2026-11-02, cars under a duct kept as pipe).

No model and no logic run: saved val scores, objects cached by ``logic_hvac_probe.py``. HVAC mask = the objects' label rule, FIRE mask = strokes along p >= 0.55.
A layer is noise when only a small part of it is in the mask and the model does not believe in it: coverage < c and length-weighted mean p < a.

    py -3.12 research/probes/noiselayer_probe.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.geometry import PageSegments  # noqa: E402
from openstroke.metrics import length_counts, prf  # noqa: E402
from openstroke.postprocess import ALONG_THRESHOLD, along_confident, drop_noise_layers, objects_mask  # noqa: E402

DATA = Path("datasets/e3")
GRID = [(c, a) for c in (0.05, 0.1, 0.2, 0.3) for a in (0.1, 0.2, 0.3)]
CACHES = {"e30": Path("runs_colab/logic_hvac"), "e37": Path("runs_colab/logic_hvac2")}


def pages(run):
    out = []
    for k in range(5):
        res = json.load(open(f"runs_colab/{run}/fold{k}/result.json"))
        z = np.load(f"runs_colab/{run}/fold{k}/val_scores.npz")
        for pg in res["pages"]:
            s = pg["stem"]
            d = np.load(DATA / f"{s}.npz")
            mode = json.loads(str(d["meta"]))["mode"]
            obj = CACHES[run] / s / f"{run}_union0.1.json"
            if mode == "hvac" and not obj.exists():
                continue
            p = np.zeros(len(d["label"]))
            p[z[f"{s}__index"]] = z[s]
            off = d["shape_offsets"].astype(np.int64)
            segs = PageSegments(seg=d["segments"].astype(np.float64), owner=np.repeat(np.arange(len(off) - 1), np.diff(off)), shape_offsets=off)
            lay = d["layer_id"].astype(np.int64)
            mask = objects_mask("hvac", json.loads(obj.read_text(encoding="utf-8"))["objects"], segs, lay) if mode == "hvac" else along_confident(segs, p, ALONG_THRESHOLD["fire"])
            out.append(dict(stem=s, mode=mode, p=p, y=d["label"].astype(bool), L=d["length"].astype(np.float64), lay=lay, mask=mask))
            print(f"{run} {s[:21]} {mode}", flush=True)
    return out


def score(ps, c=None, a=None):
    f = (lambda g: g["mask"]) if c is None else (lambda g: drop_noise_layers(g["mask"], g["p"], g["L"], g["lay"], c, a))
    return np.sum([length_counts(f(g), g["y"], g["L"]) for g in ps], 0)


def show(name, cnt):
    m = prf(*cnt)
    print(f"  {name:26s} F2 {m['f2']:.3f}  R {m['recall']:.3f}  P {m['precision']:.3f}", flush=True)


def main():
    tune, test = pages("e30"), pages("e37")
    for mode in ("hvac", "fire"):
        tu, te = [g for g in tune if g["mode"] == mode], [g for g in test if g["mode"] == mode]
        print(f"{mode}: tune {len(tu)} pages, test {len(te)} pages")
        show("e30 no filter", score(tu))
        for c, a in GRID:
            show(f"e30 c {c} a {a}", score(tu, c, a))
        best = max(GRID, key=lambda q: prf(*score(tu, *q))["f2"])
        print(f"  picked on e30: c {best[0]} a {best[1]}")
        show("e37 no filter", score(te))
        show("e37 filter", score(te, *best))
        for c, a in GRID:
            show(f"e37 c {c} a {a}", score(te, c, a))


if __name__ == "__main__":
    main()
