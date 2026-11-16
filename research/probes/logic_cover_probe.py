"""F151: per page, how much of the model's confident picks (p >= 0.3, by length) a logic arm's objects cover: a page-level signal of the
logic collapsing (Viewer / NOCOLOR-style drawings). 0 GPU.
    py -3.12 research/probes/logic_cover_probe.py <score_pack> <val_scores.npz> <logic cache> <arm>
"""
import json, sys
from pathlib import Path
import numpy as np, shapely
from shapely.geometry import Polygon
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.relabel import page_segments_of


def cover(objects, segs, prob, L, thr=0.3):
    hi = prob >= thr
    if not hi.any():
        return 1.0
    polys = [Polygon(np.asarray(o["vertices"], float)).buffer(2.0) for o in objects if len(o.get("vertices") or []) >= 3]
    if not polys:
        return 0.0
    u = shapely.union_all(polys)
    m = hi[segs.owner]
    mid = (segs.seg[m, :2] + segs.seg[m, 2:]) / 2
    inside = shapely.contains_xy(u, mid[:, 0], mid[:, 1])
    w = segs.length[m]
    return float((w * inside).sum() / max(w.sum(), 1e-9))


if __name__ == "__main__":
    pack, scores, cache, arm = Path(sys.argv[1]), np.load(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    for r in json.loads((pack / "pages.json").read_text(encoding="utf-8")):
        st = r["stem"]
        if f"{st}__index" not in scores.files or not (cache / st / f"{arm}.json").exists():
            continue
        d = dict(np.load(pack / "npz" / f"{st}.npz"))
        segs = page_segments_of(d)
        p = np.zeros(len(d["label"])); p[scores[f"{st}__index"]] = scores[st]
        objs = json.loads((cache / st / f"{arm}.json").read_text(encoding="utf-8"))["objects"]
        print(f"{st[:22]} objects {len(objs):5d} cover of p>=0.3 {cover(objs, segs, p, d['length']):.1%}", flush=True)
