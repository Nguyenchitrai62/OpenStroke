"""Does E69's 'stroke rule' gain (0.584 vs members 0.513, score-pack protocol) depend on CAD layers? 0 GPU, saved predictions:
E69's boxes (final_json_members) scored with how=members, how=rule, and how=rule with every stroke's layer_id set to one value
(the layer-free product case: the rule's style groups then use colour / width / type only).

    py -3.12 research/probes/rule_no_layer_probe.py <score_pack folder> <final_json_members folder>
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
    dets = {}
    for stem in scale:
        objs = json.loads((preds / f"{stem}.json").read_text(encoding="utf-8"))["objects"]
        dets[stem] = [{"class_id": names.index(o["shape_name"]), "score": o["score"], "members": o["shapes"],
                       "poly": (np.asarray(o["vertices"], float) * scale[stem]).ravel().tolist()} for o in objs]
    for how in ("members", "rule"):
        print(how, "F2 %.3f" % osc.report(osc.score(pack, dets, how, log=lambda _: None), how, log=lambda _: None)["F2"])
    real = np.load

    def no_layer(path, *a, **k):
        z = real(path, *a, **k)
        d = {key: z[key] for key in z.files}
        if "layer_id" in d:
            d["layer_id"] = np.zeros_like(d["layer_id"])
        return _Z(d)

    class _Z(dict):
        files = property(lambda self: list(self))
        def __enter__(self): return self
        def __exit__(self, *a): return False

    osc.np.load = no_layer
    try:
        print("rule, no layers F2 %.3f" % osc.report(osc.score(pack, dets, "rule", log=lambda _: None), "rule", log=lambda _: None)["F2"])
    finally:
        osc.np.load = real


if __name__ == "__main__":
    main(*sys.argv[1:])
