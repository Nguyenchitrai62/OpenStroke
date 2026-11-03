"""E55: on the FIRE GT pages, which style groups of true strokes the GT lines do not need: coverage lost if the group is taken out. AI_Takeoff is only read.

    py -3.12 research/probes/label_redundant_probe.py
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.labels import gt_coverage, gt_geometry, gt_outline_length  # noqa: E402
from openstroke.postprocess import compact_strokes, style_groups  # noqa: E402
from openstroke.relabel import STYLE_KEYS, page_segments_of  # noqa: E402
from openstroke.sources import load_gt_pages  # noqa: E402

DB = r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3"
DATA = Path("datasets/e5")
NOT_CHOSEN = {"doc000020_p0001_gt10"}


def one(job):
    rec, objects = job
    s = rec["stem"]
    z = np.load(DATA / f"{s}.npz")
    d = {k: z[k] for k in z.files}
    geo = gt_geometry(rec["mode"], objects)
    segs = page_segments_of(d)
    L = d["length"].astype(np.float64)
    y = d["label"].astype(bool) & ~compact_strokes(segs, L)
    g = style_groups({k: d[k] for k in STYLE_KEYS})
    outline = gt_outline_length(geo)
    cov = lambda m: float((gt_coverage(geo, segs.seg[m[segs.owner]]) * outline).sum())  # noqa: E731
    base = cov(y)
    names = [str(x) for x in d["layer_names"]]
    rows = []
    ids = np.unique(g[y])
    w = np.bincount(g[y], L[y])
    for i in sorted(ids, key=lambda i: -w[i])[:8]:
        m = g == i
        rows.append((names[d["layer_id"][m][0]][:30], round(float(L[m & y].sum())), round(float(L[m].sum())), round((base - cov(y & ~m)) / outline.sum(), 3)))
    return {"stem": s, "file": rec["filename"][:24], "outline": round(float(outline.sum())), "ink": round(float(L[y].sum()) / outline.sum(), 2), "cov": round(base / outline.sum(), 3), "rows": rows}


def main():
    man = {r["stem"]: r for r in json.load(open(DATA / "manifest.json", encoding="utf-8"))["records"] if r.get("status") == "ok"}
    jobs = [(man[p.stem], p.objects) for p in load_gt_pages(DB) if p.stem in man and p.mode == "fire" and p.stem not in NOT_CHOSEN]
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(one, jobs):
            print(r["stem"][:9], r["file"], "outline", r["outline"], "ink", r["ink"], "cov", r["cov"])
            for x in r["rows"]:
                print("   layer", x[0].ljust(30), "true", x[1], "of group", x[2], "coverage lost if out", x[3])


if __name__ == "__main__":
    main()
