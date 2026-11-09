"""E77 (0 CU): put the pipeline's stored objects and a detector's boxes in ``runs_colab/logic_hvac2/<stem>/<arm>.json`` so that
``logic_arms_probe.py`` scores every approach with one scorer and one GT instance set (P / R / F1 per class).

    py -3.12 research/probes/arms_export.py runs_colab/e69 0.3
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from logic_hvac_probe import CACHE, pipeline_job, stored_objects  # noqa: E402
from openstroke.objects import HVAC_CLASSES  # noqa: E402

PAGES = ("doc000022", "doc000030", "doc000180", "doc000189", "doc000190", "doc000200", "doc000201")


def main(run, conf):
    run, conf = Path(run), float(conf)
    dets = json.loads((run / "val_dets.json").read_text(encoding="utf-8"))
    names = [l.split(": ", 1)[1].strip() for l in Path("datasets/obb_hvac7/data.yaml").read_text(encoding="utf-8").splitlines() if l.startswith("  ")]
    scale = {r["stem"]: r["scale"] for r in json.loads(Path("datasets/obb_hvac7/pages.json").read_text(encoding="utf-8"))}
    for d in sorted(CACHE.iterdir()):
        stem = d.name
        if stem[:9] not in PAGES:
            continue
        job = pipeline_job(int(stem[3:9]), int(stem[11:15]))
        if job:
            (d / "stored.json").write_text(json.dumps({"objects": stored_objects(job[0], int(stem[11:15]))}), encoding="utf-8")
        objs = [{"shape_name": names[int(q["class_id"])], "vertices": (np.asarray(q["poly"], float).reshape(4, 2) / scale[stem]).tolist()}
                for q in dets.get(stem, []) if q["score"] >= conf]
        (d / f"{run.name}_c{conf:g}.json").write_text(json.dumps({"objects": objs}), encoding="utf-8")
        print(stem, "pipeline job", job[0] if job else None, len(objs), "boxes")


if __name__ == "__main__":
    main(*sys.argv[1:3])
