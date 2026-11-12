"""Score a saved prediction folder (final_json_*: objects with shape_name / vertices / score / shapes) with the current product metric
(6 classes, layer-free rule and members, comparison pages = object_score.PIPELINE_PAGES). 0 GPU.

    py -3.12 research/probes/rescore_saved.py <score_pack folder> <final_json folder>
"""
import json
import sys
from pathlib import Path

import numpy as np

from openstroke.obb import object_score as osc


def main(pack, preds):
    pack, preds = Path(pack), Path(preds)
    names = [n.replace("_", " ") for n in json.loads((pack / "names.json").read_text(encoding="utf-8"))]
    scale = {r["stem"]: r["scale"] for r in json.loads((pack / "pages.json").read_text(encoding="utf-8"))}
    dets = {s: [{"class_id": names.index(o["shape_name"]), "score": o["score"], "members": o["shapes"],
                 "poly": (np.asarray(o["vertices"], float) * scale[s]).ravel().tolist()}
                for o in json.loads((preds / f"{s}.json").read_text(encoding="utf-8"))["objects"]] for s in scale if (preds / f"{s}.json").exists()}
    for how in ("members", "rule_nl"):
        r = osc.report(osc.score(pack, dets, how, log=lambda _: None, merge=True), how, log=lambda _: None)
        print(f"{how:8s} 6 classes, pages {','.join(p[3:9] for p in osc.PIPELINE_PAGES)}: F2 {r['F2']:.3f} R {r['R']:.3f} P {r['P']:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
