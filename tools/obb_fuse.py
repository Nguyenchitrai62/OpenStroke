"""E71: fuse the val detections of several OpenOBB runs (``val_dets.json``, page-image pixels) into one, for ``obb_score_probe.py``.

Per page and class, boxes of all runs are clustered greedily by rotated IoU (``--iou``, best score first); a cluster keeps the geometry of
its best box and scores sum(score) / n_runs, so a box all runs agree on keeps its score and a box only one run sees is damped. No model
run; boxes only (``members`` of the best box are kept).

    py -3.12 tools/obb_fuse.py runs_colab/e71 runs_colab/e68a runs_colab/e70 [--iou 0.5]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import shapely


def fuse(page_dets: list[list[dict]], iou: float) -> list[dict]:
    n = len(page_dets)
    items = [(q, shapely.Polygon(np.asarray(q["poly"], float).reshape(4, 2)).buffer(0)) for dets in page_dets for q in dets]
    out = []
    for cls in {int(q["class_id"]) for q, _ in items}:
        cand = sorted([x for x in items if int(x[0]["class_id"]) == cls], key=lambda x: -float(x[0]["score"]))
        if not cand:
            continue
        tree = shapely.STRtree([p for _, p in cand])
        used = np.zeros(len(cand), bool)
        for i, (q, p) in enumerate(cand):
            if used[i]:
                continue
            group = [i]
            for j in tree.query(p):
                if j != i and not used[j]:
                    o = p.intersection(cand[j][1]).area / max(p.union(cand[j][1]).area, 1e-9)
                    if o >= iou:
                        group.append(int(j))
            used[group] = True
            out.append({**q, "score": float(sum(float(cand[k][0]["score"]) for k in group)) / n, "fused": len(group)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()
    dets = [json.loads((Path(r) / "val_dets.json").read_text(encoding="utf-8")) for r in a.runs]
    pages = set().union(*dets)
    res = {s: fuse([d.get(s, []) for d in dets], a.iou) for s in sorted(pages)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "val_dets.json").write_text(json.dumps(res), encoding="utf-8")
    print({s: len(v) for s, v in res.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
