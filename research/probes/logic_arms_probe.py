"""E62 / E75 analysis: objects from any stored logic arm in runs_colab/logic_hvac2 (<stem>/<arm>.json), scored at object level.
scored at object level like the pipeline and the detectors. No model or logic run. AI_Takeoff is only read.

    py -3.12 research/probes/logic_arms_probe.py gt gtstyle gtlayer ctrl e37_union0.1
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.objects import HVAC_CLASSES, N_CLASSES, assign_objects, object_counts  # noqa: E402
from openstroke.relabel import STYLE_KEYS, object_strokes, page_segments_of  # noqa: E402

PAGES = ("doc000022", "doc000030", "doc000180", "doc000189", "doc000190", "doc000200", "doc000201")
DATA = Path("datasets/e7")


def main(*runs):
    for run in runs:
        tot = np.zeros((N_CLASSES, 3), np.int64)
        per = []
        for d in sorted(Path("runs_colab/logic_hvac2").iterdir()):
            f = d / f"{run}.json"
            if d.name[:9] not in PAGES or not f.exists():
                continue
            z = np.load(DATA / f"{d.name}.npz")
            z = {k: z[k] for k in z.files}
            segs = page_segments_of(z)
            L = z["length"].astype(np.float64)
            attrs = {k: z[k] for k in STYLE_KEYS} | {"op_counts": z["op_counts"]}
            objs = [o for o in json.loads(f.read_text(encoding="utf-8"))["objects"] if o.get("vertices")]
            pm = object_strokes("hvac", objs, segs, attrs, L)
            pc, pi, _ = assign_objects(segs, objs, pm)
            pcls = np.zeros(max(int(pi.max()) + 1, 1), np.int64)
            pcls[pi[pi >= 0]] = pc[pi >= 0]
            gi = z["oinst"].astype(np.int64)
            gcls = {int(i): int(c) for i, c in zip(gi[gi >= 0], z["ocls"][gi >= 0])}
            c = object_counts(pi.astype(np.int64), pcls, gi, gcls, L)
            tot += c
            per.append((d.name[:9], round(2 * c[1:, 0].sum() / max(c[1:, 1].sum() + c[1:, 2].sum(), 1), 3)))
        tp, npred, ngt = tot[1:].sum(0)
        print(f"{run} + logic, {len(per)} pages: objects F1 {2 * tp / max(npred + ngt, 1):.3f} (P {tp / max(npred, 1):.3f} R {tp / max(ngt, 1):.3f}) {per}")
        for k in range(1, N_CLASSES):
            if tot[k, 2] or tot[k, 1]:
                t, n, g = tot[k]
                print(f"   {HVAC_CLASSES[k]:20s} GT {g:5d} pred {n:5d} TP {t:5d}  P {t / max(n, 1):.3f}  R {t / max(g, 1):.3f}  F1 {2 * t / max(n + g, 1):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
