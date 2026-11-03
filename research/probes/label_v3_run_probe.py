"""E54 / H70: a run on labels v3, raw and after the app's post-processing, page by page. Saved scores and logic objects, no model or logic run.

    py -3.12 -m openstroke calibrate runs_colab/e54 --data datasets/e5 --objects runs_colab/logic_hvac2
    py -3.12 research/probes/label_v3_run_probe.py e54
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.metrics import length_counts, prf  # noqa: E402
from openstroke.postprocess import fire_mask, hvac_mask, load_params, on_chosen_layers, style_groups, tidy  # noqa: E402
from openstroke.relabel import STYLE_KEYS, object_strokes, page_segments_of  # noqa: E402

DATA = Path("datasets/e5")
LOGIC = Path("runs_colab/logic_hvac2")
THRESHOLDS = np.arange(0.05, 0.96, 0.05)


def one(job):
    run, fold, stem = job
    z = np.load(f"runs_colab/{run}/fold{fold}/val_scores.npz")
    d = np.load(DATA / f"{stem}.npz")
    d = {k: d[k] for k in d.files}
    mode = json.loads(str(d["meta"]))["mode"]
    y, L = d["label"].astype(bool), d["length"].astype(np.float64)
    p = np.zeros(len(y))
    p[z[f"{stem}__index"]] = z[stem]
    segs = page_segments_of(d)
    attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
    group = style_groups(attrs)
    par = load_params(f"runs_colab/{run}/fold{fold}/best.pt")[0][mode]
    if mode == "hvac":
        objs = json.loads((LOGIC / stem / f"{run}_union0.1.json").read_text(encoding="utf-8"))["objects"]
        final = tidy(hvac_mask(object_strokes("hvac", objs, segs, attrs, L), p, L, group, par), segs, attrs, group, L)
    else:
        names = [str(d["layer_names"][i]) for i in d["layer_id"]]
        final = on_chosen_layers(tidy(fire_mask(segs, p, L, group, par), segs, attrs, group, L), p, L, names)
    return {"stem": stem, "mode": mode, "final": length_counts(final, y, L), "raw": {round(float(t), 2): length_counts(p >= t, y, L) for t in THRESHOLDS}}


def main(run):
    jobs = [(run, f.name[4:], pg["stem"]) for f in sorted(Path(f"runs_colab/{run}").glob("fold*")) for pg in json.loads((f / "result.json").read_text(encoding="utf-8"))["pages"]]
    with ProcessPoolExecutor(3) as ex:
        rows = list(ex.map(one, jobs))
    for mode in ("hvac", "fire", None):
        r = [x for x in rows if mode is None or x["mode"] == mode]
        m = prf(*np.sum([x["final"] for x in r], 0))
        rec = [prf(*x["final"])["recall"] for x in r]
        raw = max((prf(*np.sum([x["raw"][round(float(t), 2)] for x in r], 0)) | {"t": round(float(t), 2)} for t in THRESHOLDS), key=lambda q: q["f2"])
        print(f"{mode or 'all'}: {len(r)} pages | raw F2 {raw['f2']:.3f} R {raw['recall']:.3f} P {raw['precision']:.3f} at {raw['t']} | final F2 {m['f2']:.3f} R {m['recall']:.3f} P {m['precision']:.3f}, pages recall < 0.8: {sum(a < 0.8 for a in rec)}")
    for x in rows:
        m = prf(*x["final"])
        print(f"  {x['stem'][:9]} {x['mode']:5s} F2 {m['f2']:.3f} R {m['recall']:.2f} P {m['precision']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
