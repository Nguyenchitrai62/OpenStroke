"""E53: the pipeline and E37 scored on labels v3, 31 out-of-fold pages. Reporting only: E37 was trained on v2. Saved scores and stored objects, no model or logic run. AI_Takeoff is only read.

Arms: each system's objects under the v2 rule (every stroke inside) and the v3 rule (strokes making up the object); E37's app mask (the whole of ``app/page_worker.py`` after the model and the logic).

    py -3.12 research/probes/label_v3_score_probe.py
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_v3_probe import job_of  # noqa: E402
from openstroke.labels import gt_geometry, label_shapes  # noqa: E402
from openstroke.metrics import length_counts, prf  # noqa: E402
from openstroke.postprocess import fire_mask, hvac_mask, load_params, objects_mask, on_chosen_layers, tidy  # noqa: E402
from openstroke.relabel import STYLE_KEYS, labels_v3, page_segments_of  # noqa: E402

DATA = Path("datasets/e5")
CACHE = Path("runs_colab/chain_cache/e37")
LOGIC = Path("runs_colab/logic_hvac2")


def v3_rule(d, mode, objects, segs):
    geo = gt_geometry(mode, objects)
    return labels_v3({**d, "label_inside_only": label_shapes(segs, geo).label}, mode, geo)[0]


def one(job):
    stem, rec = job
    mode = rec["mode"]
    z = np.load(DATA / f"{stem}.npz")
    d = {k: z[k] for k in z.files}
    g = np.load(CACHE / f"{stem}.npz")
    segs = page_segments_of(d)
    y2, y3, L, group, lay = d["label_v2"].astype(bool), d["label"].astype(bool), d["length"].astype(np.float64), g["group"], d["layer_id"].astype(np.int64)
    p = g["p"]
    par = load_params("runs_colab/e37/fold0/best.pt")[0][mode]
    attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
    names = [str(d["layer_names"][i]) for i in lay]
    arms = {}
    if mode == "hvac":
        f = LOGIC / stem / "e37_union0.1.json"
        if f.exists():
            objs = json.loads(f.read_text(encoding="utf-8"))["objects"]
            arms["E37 objects, v2 rule"] = objects_mask("hvac", objs, segs, lay)
            arms["E37 objects, v3 rule"] = v3_rule(d, "hvac", objs, segs)
            arms["E37 app mask"] = tidy(hvac_mask(arms["E37 objects, v3 rule"], p, L, group, par), segs, attrs, group, L)
    else:
        arms["E37 app mask"] = on_chosen_layers(tidy(fire_mask(segs, p, L, group, par), segs, attrs, group, L), p, L, names)
    job = job_of(rec["document_id"], mode, rec["page_number"])
    if job is not None:
        objs = job[0]
        arms["pipeline objects, v2 rule"] = objects_mask(mode, objs, segs, lay)
        arms["pipeline objects, v3 rule"] = v3_rule(d, mode, objs, segs)
    return {"stem": stem, "mode": mode, "true2": float(L[y2].sum()), "true3": float(L[y3].sum()),
            "arms": {k: {"v2": length_counts(m, y2, L), "v3": length_counts(m, y3, L)} for k, m in arms.items()}}


def main():
    man = {r["stem"]: r for r in json.load(open(DATA / "manifest.json", encoding="utf-8"))["records"] if r.get("status") == "ok"}
    stems = [f.stem for f in sorted(CACHE.glob("*.npz"))]
    with ProcessPoolExecutor(3) as ex:
        rows = list(ex.map(one, [(s, man[s]) for s in stems]))
    json.dump(rows, open("runs_colab/label_v3_score.json", "w", encoding="utf-8"), indent=1, default=float)
    for mode in ("hvac", "fire"):
        r = [x for x in rows if x["mode"] == mode]
        print(f"\n{mode}: {len(r)} pages, true length v3 / v2 {sum(x['true3'] for x in r) / sum(x['true2'] for x in r):.3f}")
        names = sorted({k for x in r for k in x["arms"]})
        common = [x for x in r if len(x["arms"]) == max(len(y["arms"]) for y in r)]
        print(f" on the {len(common)} pages every arm has")
        for k in names:
            line = f"  {k:28s}"
            for lab in ("v2", "v3"):
                m = prf(*np.sum([x["arms"][k][lab] for x in common], axis=0))
                line += f" | {lab} F2 {m['f2']:.3f} R {m['recall']:.3f} P {m['precision']:.3f}"
            print(line)
        worst = {k: sorted((prf(*x["arms"][k]["v3"])["recall"], x["stem"][:9]) for x in common)[:3] for k in names}
        for k, v in worst.items():
            print("  lowest recall", k, [(round(a, 2), b) for a, b in v])


if __name__ == "__main__":
    main()
