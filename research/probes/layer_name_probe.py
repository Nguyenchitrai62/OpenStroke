"""E55 / H72: are the layer-name rules true beyond the pages they were read on? FIRE pages with no score in the run. No model run.

    py -3.12 research/probes/layer_name_probe.py e37 datasets/e6
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openstroke.metrics import length_counts, prf  # noqa: E402
from openstroke.postprocess import OP_CURVE, compact_strokes  # noqa: E402
from openstroke.relabel import page_segments_of  # noqa: E402
from postproc_e55_probe import GREY_MIN_LUM, GREY_SPREAD, IS_PIPE, NOT_PIPE  # noqa: E402


def main(run, data):
    data = Path(data)
    seen = {pg["stem"] for f in Path(f"runs_colab/{run}").glob("fold*") for pg in json.loads((f / "result.json").read_text(encoding="utf-8"))["pages"]}
    man = json.loads((data / "manifest.json").read_text(encoding="utf-8"))["records"]
    rows = [r for r in man if r.get("status") == "ok" and r["mode"] == "fire" and r["stem"] not in seen and r["stem"] != "doc000020_p0001_gt10"]
    print(len(rows), "FIRE pages not scored (all but 333.pdf labelled from pipeline output: a weak check)")
    total, lost, grey_true = np.zeros(3), 0.0, 0.0
    for r in rows:
        z = np.load(data / r["npz"])
        d = {k: z[k] for k in z.files}
        y, L = d["label"].astype(bool), d["length"].astype(np.float64)
        names = [str(x) for x in d["layer_names"]]
        lay = d["layer_id"].astype(np.int64)
        segs = page_segments_of(d)
        drop = np.array([bool(NOT_PIPE.search(n)) for n in names])[lay]
        keep = np.array([bool(IS_PIPE.search(n)) and not NOT_PIPE.search(n) for n in names])[lay]
        first, last = segs.seg[d["shape_offsets"][:-1].clip(max=len(segs.seg) - 1), :2], segs.seg[(d["shape_offsets"][1:] - 1).clip(min=0), 2:]
        rescue = keep & ~(np.hypot(*(first - last).T) < 0.5) & ~(d["op_counts"][:, OP_CURVE] > 0) & ~compact_strokes(segs, L)
        c = length_counts(rescue, y, L)
        total += c
        lost += L[y & drop].sum()
        rgb = np.nan_to_num(d["stroke_rgb"], nan=0.0)
        grey = (rgb.max(1) - rgb.min(1) < GREY_SPREAD) & (rgb.mean(1) >= GREY_MIN_LUM)
        g = L[y & grey].sum() / max(L[y].sum(), 1)
        grey_true += L[y & grey].sum() if g < 0.2 else 0.0
        m = prf(*c)
        print(f"  {r['stem'][:9]} {r['filename'][:24]:24s} names alone R {m['recall']:.2f} P {m['precision']:.2f} F2 {m['f2']:.2f}; true that is grey {g:.3f}; {sorted({names[i][:22] for i in np.unique(lay[rescue])})[:3]}")
    m = prf(*total)
    t = total[0] + total[2]
    print(f"all: names alone R {m['recall']:.3f} P {m['precision']:.3f} F2 {m['f2']:.3f}; true lost to drop names {lost / t:.4f}; true lost to the grey step {grey_true / t:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
