"""E55: how far would deciding by whole group go? Per page, the best possible choice of style groups (layer, colour, width, type) or of layers, given the labels, and what E37's mean score per group manages. No model run.

    py -3.12 research/probes/group_oracle_probe.py e37 datasets/e6
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.metrics import length_counts, prf  # noqa: E402
from openstroke.postprocess import style_groups  # noqa: E402
from openstroke.relabel import STYLE_KEYS  # noqa: E402

THRESHOLDS = np.arange(0.05, 0.96, 0.05)


def oracle(y, L, group):
    n = int(group.max()) + 1
    t, a = np.bincount(group[y], L[y], n), np.bincount(group, L, n)
    return (t >= 0.5 * a)[group]


def main(run, data):
    rows = []
    for f in sorted(Path(f"runs_colab/{run}").glob("fold*")):
        z = np.load(f / "val_scores.npz")
        for pg in json.loads((f / "result.json").read_text(encoding="utf-8"))["pages"]:
            s = pg["stem"]
            d = np.load(Path(data) / f"{s}.npz")
            d = {k: d[k] for k in ("meta", "label", "length", *STYLE_KEYS)}
            y, L = d["label"].astype(bool), d["length"].astype(np.float64)
            p = np.zeros(len(y))
            p[z[f"{s}__index"]] = z[s]
            g = style_groups(d)
            n = int(g.max()) + 1
            mean = np.bincount(g, p * L, n) / np.maximum(np.bincount(g, L, n), 1e-9)
            rows.append({"stem": s, "mode": json.loads(str(d["meta"]))["mode"], "groups": n, "layers": int(d["layer_id"].max()) + 1,
                         "style": length_counts(oracle(y, L, g), y, L), "layer": length_counts(oracle(y, L, d["layer_id"].astype(np.int64)), y, L),
                         "raw": {round(float(t), 2): length_counts(p >= t, y, L) for t in THRESHOLDS},
                         "mean": {round(float(t), 2): length_counts((mean >= t)[g], y, L) for t in THRESHOLDS}})
    for mode in ("fire", "hvac"):
        r = [x for x in rows if x["mode"] == mode]
        line = f"{mode}: {len(r)} pages, median {int(np.median([x['groups'] for x in r]))} style groups, {int(np.median([x['layers'] for x in r]))} layers |"
        for k in ("style", "layer"):
            m = prf(*np.sum([x[k] for x in r], 0))
            line += f" best {k} choice F2 {m['f2']:.3f} R {m['recall']:.3f} P {m['precision']:.3f} |"
        for k in ("raw", "mean"):
            m = max((prf(*np.sum([x[k][round(float(t), 2)] for x in r], 0)) for t in THRESHOLDS), key=lambda q: q["f2"])
            line += f" E37 {k} F2 {m['f2']:.3f} |"
        print(line)
        for x in r:
            a, b = prf(*x["style"]), prf(*x["layer"])
            print(f"  {x['stem'][:9]} groups {x['groups']:4d} layers {x['layers']:4d} | best style {a['f2']:.3f} (R {a['recall']:.2f} P {a['precision']:.2f}) | best layer {b['f2']:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
