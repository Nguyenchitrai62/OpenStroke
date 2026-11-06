"""E62 / H79 (version A): objects straight from the stroke model's outputs on its val pages, scored like the pipeline and OpenOBB.

Strokes the model calls pipe (p >= t, no other post-processing) take their most likely object class; strokes of one class whose predicted
object centres lie within ``eps`` of each other are one object (``objects.decode_objects``). Matched to GT objects by shared strokes. Writes
the final JSON per page (class, polygon, the payload shape indices). No model run.

    py -3.12 research/probes/objects_model_probe.py runs_colab/e62nl datasets/e7nl
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.objects import HVAC_CLASSES, N_CLASSES, decode_objects, object_counts, object_polygon  # noqa: E402
from openstroke.relabel import page_segments_of  # noqa: E402

THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6)
EPS = (4.0, 6.0, 10.0)
PRE = (0.5, 6.0)  # written before the run: p >= 0.5, centres within 6 pt
PIPELINE_PAGES = ("doc000022", "doc000030", "doc000180", "doc000189", "doc000190", "doc000200", "doc000201")


def f1(c):
    tp, npred, ngt = c[1:].sum(0)
    return 2 * tp / max(npred + ngt, 1), tp / max(npred, 1), tp / max(ngt, 1)


def main(run, data):
    run, data = Path(run), Path(data)
    res = {}
    (run / "final_json").mkdir(exist_ok=True)
    for fold in sorted(run.glob("fold*")):
        z = np.load(fold / "val_scores.npz")
        for pg in json.loads((fold / "result.json").read_text(encoding="utf-8"))["pages"]:
            s = pg["stem"]
            if f"{s}__oprob" not in z.files:
                continue
            d = np.load(data / f"{s}.npz")
            d = {k: d[k] for k in d.files}
            n = len(d["label"])
            idx = z[f"{s}__index"]
            p = np.zeros(n)
            p[idx] = z[s]
            prob = np.zeros((n, N_CLASSES))
            prob[idx] = z[f"{s}__oprob"]
            off = np.zeros((n, 2))
            off[idx] = z[f"{s}__ooff"]
            L = d["length"].astype(np.float64)
            xy = d["bbox"].reshape(-1, 2, 2).mean(1)
            gi = d["oinst"].astype(np.int64)
            gcls = {int(i): int(c) for i, c in zip(gi[gi >= 0], d["ocls"][gi >= 0])}
            cls = prob.argmax(1)
            for t in THRESHOLDS:
                for e in EPS:
                    obj, oc = decode_objects(xy, cls, off, (p >= t) & (cls > 0), e)
                    res.setdefault((t, e), {})[s] = object_counts(obj, oc if len(oc) else np.zeros(1, np.int64), gi, gcls, L).tolist()
                    if (t, e) == PRE:
                        segs = page_segments_of(d)
                        out = [{"shape_name": HVAC_CLASSES[int(oc[k])], "vertices": object_polygon(segs.seg, segs.owner, np.flatnonzero(obj == k), int(oc[k])),
                                "shapes": np.flatnonzero(obj == k).tolist()} for k in range(len(oc))]
                        (run / "final_json" / f"{s}.json").write_text(json.dumps({"page": s, "coordinate_system": "display_top_left", "objects": out}, ensure_ascii=False), encoding="utf-8")
            print("scored", s, flush=True)
    for name, keyf in (("7 pages with a pipeline run", lambda s: s[:9] in PIPELINE_PAGES), ("all val pages", lambda s: True)):
        print(f"\n{name}")
        for q in sorted(res):
            c = np.sum([v for s, v in res[q].items() if keyf(s)], 0)
            fo, po, ro = f1(c)
            low = sum(np.asarray(v)[1:, 0].sum() / max(np.asarray(v)[1:, 2].sum(), 1) < 0.5 for s, v in res[q].items() if keyf(s))
            print(f"  p>={q[0]:.1f} eps {q[1]:4.1f}pt{' (written before)' if q == PRE else ''}: objects F1 {fo:.3f} (P {po:.3f} R {ro:.3f}), pages with object recall < 0.5: {low}")
        c = np.sum([v for s, v in res[PRE].items() if keyf(s)], 0)
        print("  per class (written before): " + ", ".join(f"{HVAC_CLASSES[k]} {2 * c[k, 0] / max(c[k, 1] + c[k, 2], 1):.2f} ({c[k, 2]})" for k in range(1, N_CLASSES) if c[k, 2]))


if __name__ == "__main__":
    main(*sys.argv[1:])
