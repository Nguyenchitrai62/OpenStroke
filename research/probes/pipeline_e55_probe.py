"""E55: the pipeline's objects and each run's HVAC objects on the new labels, page by page, same rule as the labels. Stored objects, no model or logic run. AI_Takeoff is only read.

    py -3.12 research/probes/pipeline_e55_probe.py datasets/e6
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_v3_probe import job_of  # noqa: E402
from openstroke.metrics import length_counts, prf  # noqa: E402
from openstroke.relabel import STYLE_KEYS, object_strokes, page_segments_of  # noqa: E402

CV_RUN = "e37"


def one(job):
    stem, rec, data = job
    z = np.load(Path(data) / f"{stem}.npz")
    d = {k: z[k] for k in z.files}
    y, L = d["label"].astype(bool), d["length"].astype(np.float64)
    j = job_of(rec["document_id"], rec["mode"], rec["page_number"])
    if j is None:
        return {"stem": stem, "mode": rec["mode"], "counts": None}
    segs = page_segments_of(d)
    attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
    return {"stem": stem, "mode": rec["mode"], "counts": [float(x) for x in length_counts(object_strokes(rec["mode"], j[0], segs, attrs, L), y, L)]}


def main(data):
    man = {r["stem"]: r for r in json.loads((Path(data) / "manifest.json").read_text(encoding="utf-8"))["records"] if r.get("status") == "ok"}
    stems = [pg["stem"] for f in sorted(Path(f"runs_colab/{CV_RUN}").glob("fold*")) for pg in json.loads((f / "result.json").read_text(encoding="utf-8"))["pages"]]
    stems.append("doc000153_p0001_gt103")
    with ProcessPoolExecutor(3) as ex:
        rows = list(ex.map(one, [(s, man[s], data) for s in stems]))
    Path("runs_colab/e55").mkdir(exist_ok=True)
    Path("runs_colab/e55/pipeline.json").write_text(json.dumps(rows), encoding="utf-8")
    for mode in ("hvac", "fire"):
        r = [x for x in rows if x["mode"] == mode and x["counts"] and x["stem"] != "doc000153_p0001_gt103"]
        m = prf(*np.sum([x["counts"] for x in r], 0))
        low = sum(prf(*x["counts"])["recall"] < 0.8 for x in r)
        print(f"pipeline {mode}: {len(r)} pages F2 {m['f2']:.3f} R {m['recall']:.3f} P {m['precision']:.3f} low recall {low}")
    for x in rows:
        if x["counts"]:
            m = prf(*x["counts"])
            print(f"  {x['stem'][:9]} {x['mode']:5s} F2 {m['f2']:.3f} R {m['recall']:.2f} P {m['precision']:.2f}")
        else:
            print(f"  {x['stem'][:9]} {x['mode']:5s} no pipeline run")


if __name__ == "__main__":
    main(*sys.argv[1:])
