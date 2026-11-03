"""E55 / H73: the pipeline kept, the model as its guard: when most of the pipeline's strokes are not in the model's mask, the page takes the model's mask. Stored objects and saved scores, no model or logic run. AI_Takeoff is only read.

    py -3.12 research/probes/guard_e55_probe.py e37 datasets/e6
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
from label_v3_probe import job_of  # noqa: E402
from openstroke.metrics import length_counts, prf  # noqa: E402
from openstroke.postprocess import OP_CURVE, bridge_curves, compact_strokes, fill_strokes, mesh_strokes, style_groups  # noqa: E402
from openstroke.relabel import STYLE_KEYS, object_strokes, page_segments_of  # noqa: E402
from postproc_e55_probe import load, mask_of  # noqa: E402

LOGIC = Path("runs_colab/logic_hvac2")
FIRE_PARAMS, FIRE_STEPS = (0.55, 0.6), ("name", "grey", "rescue")
AGREE = 0.5  # share of the pipeline's stroke length the model's mask must also hold


def page(job):
    run, fold, stem, data, rec = job
    z = np.load(Path(data) / f"{stem}.npz")
    d = {k: z[k] for k in z.files}
    y, L = d["label"].astype(bool), d["length"].astype(np.float64)
    segs = page_segments_of(d)
    attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
    if rec["mode"] == "fire":
        model = mask_of(load((run, fold, stem, data)), FIRE_PARAMS, FIRE_STEPS)
    else:
        f = LOGIC / stem / f"{run}_union0.1.json"
        if not f.exists():
            return None
        group = style_groups(attrs)
        m = object_strokes("hvac", json.loads(f.read_text(encoding="utf-8"))["objects"], segs, attrs, L)
        model = (m | bridge_curves(segs, m, attrs["op_counts"][:, OP_CURVE] > 0)) & ~(mesh_strokes(segs, group, L) | fill_strokes(segs, group, L) | compact_strokes(segs, L))
    j = job_of(rec["document_id"], rec["mode"], rec["page_number"])
    pipe = object_strokes(rec["mode"], j[0], segs, attrs, L) if j else np.zeros(len(y), bool)
    agree = float(L[pipe & model].sum() / max(L[pipe].sum(), 1e-9))
    return {"stem": stem, "mode": rec["mode"], "agree": agree, "reverse": float(L[pipe & model].sum() / max(L[model].sum(), 1e-9)), "pipe": length_counts(pipe, y, L), "model": length_counts(model, y, L)}


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
    man = {r["stem"]: r for r in json.loads((Path(data) / "manifest.json").read_text(encoding="utf-8"))["records"] if r.get("status") == "ok"}
    jobs = [(run, f.name[4:], pg["stem"], data, man[pg["stem"]]) for f in sorted(Path(f"runs_colab/{run}").glob("fold*")) for pg in json.loads((f / "result.json").read_text(encoding="utf-8"))["pages"]]
    rows = [r for r in map(patient, jobs) if r]
    Path(f"runs_colab/e55/{run}_guard.json").write_text(json.dumps(rows, default=float), encoding="utf-8")
    for mode in ("fire", "hvac", None):
        r = [x for x in rows if mode is None or x["mode"] == mode]
        line = f"{mode or 'all'}: {len(r)} pages"
        for name, pick in (("pipeline", lambda x: x["pipe"]), ("model", lambda x: x["model"]), ("guarded", lambda x: x["pipe"] if x["agree"] >= AGREE else x["model"])):
            c = [pick(x) for x in r]
            m = prf(*np.sum(c, 0))
            line += f" | {name} F2 {m['f2']:.3f} R {m['recall']:.3f} P {m['precision']:.3f} low {sum(prf(*k)['recall'] < 0.8 for k in c)}"
        print(line)
    for x in sorted(rows, key=lambda x: x["agree"]):
        a, b = prf(*x["pipe"]), prf(*x["model"])
        print(f"  {x['stem'][:9]} {x['mode']:5s} agree {x['agree']:.2f} reverse {x['reverse']:.2f} | pipeline {a['f2']:.3f} model {b['f2']:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
