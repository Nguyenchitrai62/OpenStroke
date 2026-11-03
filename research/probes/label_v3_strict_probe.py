"""E53 / H69: labels v3 against v2 on the GT pages. Coverage of the GT outlines, ink per GT outline, what left and what came in, by layer. No model run. AI_Takeoff is only read.

    py -3.12 research/probes/label_v3_strict_probe.py
"""

import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.build import transform_points  # noqa: E402
from openstroke.labels import gt_coverage, gt_geometry, gt_outline_length  # noqa: E402
from openstroke.relabel import labels_v3, page_segments_of  # noqa: E402
from openstroke.sources import load_gt_pages  # noqa: E402

DB = r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3"
DATA = Path("datasets/e3")


def one(job):
    rec, objects = job
    s, mode = rec["stem"], rec["mode"]
    d = np.load(DATA / f"{s}.npz")
    d = {k: d[k] for k in d.files}
    w, h = rec["page_size"]["payload"]
    t = rec.get("gt_transform_applied")
    geo = gt_geometry(mode, objects, transform=None if t == "identity" else (lambda v: transform_points(t, v, w, h)))
    segs = page_segments_of(d)
    L = d["length"].astype(np.float64)
    v2 = d["label"].astype(bool)
    v3, why = labels_v3(d, mode, geo)
    outline = gt_outline_length(geo)
    cov = lambda m: float((gt_coverage(geo, segs.seg[m[segs.owner]]) * outline).sum())  # noqa: E731
    names = [str(x) for x in d["layer_names"]]
    lay = d["layer_id"]
    lost, new = Counter(), Counter()
    for i in np.flatnonzero(v2 & ~v3):
        lost[names[lay[i]]] += float(L[i])
    for i in np.flatnonzero(v3 & ~v2):
        new[names[lay[i]]] += float(L[i])
    return {"stem": s, "mode": mode, "outline": float(outline.sum()), "cov2": cov(v2), "cov3": cov(v3), "len2": float(L[v2].sum()), "len3": float(L[v3].sum()),
            "stray": float(L[why["stray"] & v2].sum()), "mesh": float(L[why["mesh"] & v2].sum()), "bridged": float(L[why["bridged"] & ~v2].sum()), "lost": lost.most_common(6), "new": new.most_common(4)}


def main():
    man = {r["stem"]: r for r in json.load(open(DATA / "manifest.json", encoding="utf-8"))["records"] if r.get("status") == "ok"}
    jobs = [(man[p.stem], p.objects) for p in load_gt_pages(DB) if p.stem in man and p.mode in ("fire", "hvac")]
    print("GT pages", len(jobs), flush=True)
    with ProcessPoolExecutor(4) as ex:
        rows = list(ex.map(one, jobs))
    json.dump(rows, open("runs_colab/label_v3_strict.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for mode in ("hvac", "fire"):
        r = [x for x in rows if x["mode"] == mode]
        o = sum(x["outline"] for x in r)
        s = lambda k: sum(x[k] for x in r)  # noqa: E731
        print(f"\n{mode}: {len(r)} pages, coverage v2 {s('cov2') / o:.3f} v3 {s('cov3') / o:.3f}; ink per outline v2 {s('len2') / o:.2f} v3 {s('len3') / o:.2f} ({1 - s('len3') / s('len2'):+.1%} less); stray {s('stray') / s('len2'):.1%} mesh {s('mesh') / s('len2'):.1%} of v2; curves added {s('bridged') / s('len2'):.1%}")
        lost, new = Counter(), Counter()
        for x in r:
            lost.update(dict(x["lost"]))
            new.update(dict(x["new"]))
        print(" most lost:", [(k[:30].encode("ascii", "replace").decode(), round(v / 1e3, 1)) for k, v in lost.most_common(14)])
        print(" most added:", [(k[:30].encode("ascii", "replace").decode(), round(v / 1e3, 1)) for k, v in new.most_common(6)])
        for x in sorted(r, key=lambda x: (x["cov3"] - x["cov2"]) / max(x["outline"], 1))[:8]:
            print(f"  {x['stem'][:24]:24s} coverage {x['cov2'] / x['outline']:.3f} -> {x['cov3'] / x['outline']:.3f}  ink {x['len2'] / x['outline']:.2f} -> {x['len3'] / x['outline']:.2f}")


if __name__ == "__main__":
    main()
