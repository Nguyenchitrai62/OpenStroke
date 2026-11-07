"""E75 (0 CU): rescore detector boxes with a no-layer stroke model. Each box takes the length-weighted mean p ("is a duct-system stroke") of
the stroke segments whose midpoint lies inside it, from the stroke model's out-of-fold val outputs (E62nl, layers erased), and its score becomes
score * pmean ** b. Boxes on equipment, architecture or text get little support and drop below the threshold. Writes a val_dets.json for
``obb_score_probe.py`` per b. No model run.

    py -3.12 research/probes/obb_rescore_probe.py runs_colab/e69 runs_colab/e62nl datasets/e7nl datasets/obb_hvac7 0.5,1,2
"""

import json
import sys
from pathlib import Path

import numpy as np
import shapely


def main(det_run, stroke_run, stroke_data, obb, powers):
    det_run, stroke_run, stroke_data, obb = map(Path, (det_run, stroke_run, stroke_data, obb))
    dets = json.loads((det_run / "val_dets.json").read_text(encoding="utf-8"))
    scale = {r["stem"]: r["scale"] for r in json.loads((obb / "pages.json").read_text(encoding="utf-8"))}
    prob = {}
    for f in sorted(stroke_run.glob("fold*/val_scores.npz")):
        z = np.load(f)
        for k in z.files:
            if "__" not in k:
                prob[k] = (z[k], z[f"{k}__index"])
    out = {float(b): {} for b in powers.split(",")}
    support = []
    for stem, ds in dets.items():
        if stem not in prob:
            print("no stroke scores for", stem)
            for b in out:
                out[b][stem] = ds
            continue
        d = np.load(stroke_data / f"{stem}.npz")
        n = len(d["length"])
        p = np.zeros(n)
        p[prob[stem][1]] = prob[stem][0]
        seg = d["segments"].astype(np.float64)  # (m, 4) x0 y0 x1 y1 in pt; a long polyline crosses many objects, so vote per segment
        owner = np.repeat(np.arange(n), np.diff(d["shape_offsets"]))
        L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
        ps = p[owner]
        cen = shapely.points(((seg[:, :2] + seg[:, 2:]) / 2) * scale[stem])
        tree = shapely.STRtree(cen)
        for q in ds:
            poly = shapely.Polygon(np.asarray(q["poly"], float).reshape(4, 2)).buffer(2)
            idx = tree.query(poly, predicate="contains")
            pm = float((ps[idx] * L[idx]).sum() / max(L[idx].sum(), 1e-9)) if len(idx) else 0.0
            support.append(pm)
            for b in out:
                out[b].setdefault(stem, []).append({**q, "score": float(q["score"]) * pm ** b, "stroke_p": pm})
    s = np.array(support)
    print(f"box support: median {np.median(s):.2f}, < 0.2: {(s < 0.2).mean():.2f}, > 0.8: {(s > 0.8).mean():.2f}")
    for b, res in out.items():
        dst = det_run.parent / f"{det_run.name}_r{b:g}"
        dst.mkdir(exist_ok=True)
        (dst / "val_dets.json").write_text(json.dumps(res), encoding="utf-8")
        print("wrote", dst)


if __name__ == "__main__":
    main(*sys.argv[1:6])
