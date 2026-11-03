"""E55 / H71: labels v3 with compact symbols out, against labels v3 (``datasets/e5``), on the GT pages. Coverage of the GT outlines, ink per GT outline, which layers the true strokes sit on. No model run. AI_Takeoff is only read.

    py -3.12 research/probes/label_v3b_probe.py
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.build import transform_points  # noqa: E402
from openstroke.labels import gt_coverage, gt_geometry, gt_outline_length  # noqa: E402
from openstroke.relabel import labels_v3, page_segments_of  # noqa: E402
from openstroke.sources import load_gt_pages  # noqa: E402

DB = r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3"
DATA = Path("datasets/e5")
NOT_CHOSEN = {"doc000020_p0001_gt10"}


def one(job):
    rec, objects = job
    s, mode = rec["stem"], rec["mode"]
    z = np.load(DATA / f"{s}.npz")
    d = {k: z[k] for k in z.files}
    w, h = rec["page_size"]["payload"]
    t = rec.get("gt_transform_applied")
    geo = gt_geometry(mode, objects, transform=None if t == "identity" else (lambda v: transform_points(t, v, w, h)))
    segs = page_segments_of(d)
    L = d["length"].astype(np.float64)
    old = d["label"].astype(bool)
    t0 = time.time()
    new = labels_v3(d, mode, geo)[0]
    secs = time.time() - t0
    outline = gt_outline_length(geo)
    cov = lambda m: float((gt_coverage(geo, segs.seg[m[segs.owner]]) * outline).sum())  # noqa: E731
    names = [str(x) for x in d["layer_names"]]
    lay = d["layer_id"].astype(np.int64)
    n = len(names)
    tot, tr = np.bincount(lay, L, n), np.bincount(lay[new], L[new], n)
    return {"stem": s, "file": rec["filename"][:26], "mode": mode, "outline": float(outline.sum()), "cov_old": cov(old), "cov_new": cov(new), "len_old": float(L[old].sum()), "len_new": float(L[new].sum()),
            "secs": round(secs, 1),
            "layers": [(names[i][:34], round(float(tr[i])), round(float(tot[i]))) for i in np.argsort(-tr)[:6] if tr[i] > 0],
            "lost": [(names[i][:34], round(float(x))) for i, x in sorted(enumerate(np.bincount(lay[old & ~new], L[old & ~new], n)), key=lambda q: -q[1])[:5] if x > 0]}


def main():
    man = {r["stem"]: r for r in json.load(open(DATA / "manifest.json", encoding="utf-8"))["records"] if r.get("status") == "ok"}
    jobs = [(man[p.stem], p.objects) for p in load_gt_pages(DB) if p.stem in man and p.mode in ("fire", "hvac") and p.stem not in NOT_CHOSEN]
    with ProcessPoolExecutor(4) as ex:
        rows = list(ex.map(one, jobs))
    json.dump(rows, open("runs_colab/label_v3b.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for mode in ("fire", "hvac"):
        r = [x for x in rows if x["mode"] == mode]
        o = sum(x["outline"] for x in r)
        s = lambda k: sum(x[k] for x in r)  # noqa: E731
        print(f"{mode}: {len(r)} pages, coverage {s('cov_old') / o:.4f} -> {s('cov_new') / o:.4f}; ink per outline {s('len_old') / o:.2f} -> {s('len_new') / o:.2f}")
        for x in sorted(r, key=lambda x: -x["len_new"] / max(x["outline"], 1)):
            print(f"  {x['stem'][:9]} {x['file']:26s} cov {x['cov_old'] / x['outline']:.3f}->{x['cov_new'] / x['outline']:.3f} ink {x['len_old'] / x['outline']:.2f}->{x['len_new'] / x['outline']:.2f} {x['secs']}s lost {x['lost'][:4]}")


if __name__ == "__main__":
    main()
