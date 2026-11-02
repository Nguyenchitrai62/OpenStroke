"""User 2026-11-02: the new pipeline misses many curve commands; can it handle pages without layers? Saved E37 val scores and the ``chain_probe`` cache, no model run.

Curve = a shape with at least one cubic bezier item. Rows: how curves are labelled on pipe layers (share of the layer's length true >= 50 %), mean p, recall of the model alone and after post-processing.
Then the pages with no layers to choose between.

    py -3.12 research/probes/curve_layer_probe.py e37
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.metrics import length_counts, prf  # noqa: E402
from openstroke.postprocess import by_group, load_params, style_groups  # noqa: E402

OP_C = 1  # build.OPS: l, c, re, qu


def main(run):
    params, _ = load_params(f"runs_colab/{run}/fold0/best.pt")
    tally = {}
    rows = []
    for k in range(5):
        for pg in json.load(open(f"runs_colab/{run}/fold{k}/result.json"))["pages"]:
            s = pg["stem"]
            f = Path(f"runs_colab/chain_cache/{run}/{s}.npz")
            if not f.exists():
                continue
            g = np.load(f)
            d = np.load(f"datasets/e3/{s}.npz")
            mode = str(g["mode"])
            p, L, y, lay = g["p"], g["L"], g["y"], g["lay"]
            group = style_groups({x: d[x] for x in ("layer_id", "stroke_rgb", "width", "type_code")})
            par = params[mode]
            base = g["obj"] if mode == "hvac" else g["t0.55"]
            rule = base | by_group(p, L, group, par["group"])
            if mode == "hvac":
                rule = rule | (p >= par["sure"])
            n = lay.max() + 1
            share = np.bincount(lay[y], L[y], n) / np.maximum(np.bincount(lay, L, n), 1e-9)
            pipe_layer = (share >= 0.5)[lay]
            curve = d["op_counts"][:, OP_C] > 0
            for name, c in (("curve", curve), ("no curve", ~curve)):
                t = tally.setdefault((mode, name), np.zeros(9))
                t += [L[c].sum(), L[c & y].sum(), L[c & pipe_layer].sum(), L[c & pipe_layer & y].sum(), (p[c & y] * L[c & y]).sum(), L[c & y & (p >= 0.5)].sum(), L[c & y & base].sum(), L[c & y & rule].sum(), L[c & ~y & rule].sum()]
            layers = len(np.unique(lay))
            names = [str(x) for x in d["layer_names"]]
            if layers < 2 or (layers <= 3 and share.max() < 0.5):
                rows.append((s[:9], mode, layers, names[:3], len(y), prf(*length_counts(rule, y, L))))
    print("system kind | true % of all | true % on pipe layers | true mean p | recall p>=0.5 | recall old post | recall new post | precision new | true length k pt")
    for (mode, name), t in sorted(tally.items()):
        print(f"{mode:5s} {name:9s} {t[1] / t[0] * 100:6.1f} {t[3] / max(t[2], 1) * 100:6.1f} {t[4] / max(t[1], 1):6.3f} {t[5] / max(t[1], 1):6.3f} {t[6] / max(t[1], 1):6.3f} {t[7] / max(t[1], 1):6.3f} {t[7] / max(t[7] + t[8], 1):6.3f} {t[1] / 1e3:8.0f}")
    print("\npages with no real layers:")
    for r in rows:
        print(f"  {r[0]} {r[1]} layers {r[2]} {r[3]} shapes {r[4]} F2 {r[5]['f2']:.3f} R {r[5]['recall']:.3f} P {r[5]['precision']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
