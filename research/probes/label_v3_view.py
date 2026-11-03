"""E53: labels v3 against v2 by eye, in the window of a page with the most length that changed. Green kept, red only in v2, blue only in v3, thin yellow the GT outline, grey the rest.

    py -3.12 research/probes/label_v3_view.py runs_colab/viz doc000196_p0001_gt123 doc000054_p0001_gt29
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.build import transform_points  # noqa: E402
from openstroke.labels import gt_geometry, outline_geometry  # noqa: E402
from openstroke.relabel import labels_v3, page_segments_of  # noqa: E402
from openstroke.sources import load_gt_pages  # noqa: E402

DB = r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3"
DATA = Path("datasets/e3")
WINDOW = 500


def main(out, *stems):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    man = {r["stem"]: r for r in json.load(open(DATA / "manifest.json", encoding="utf-8"))["records"]}
    pages = {p.stem: p for p in load_gt_pages(DB)}
    for s in stems:
        rec = man[s]
        d = np.load(DATA / f"{s}.npz")
        d = {k: d[k] for k in d.files}
        w, h = rec["page_size"]["payload"]
        t = rec.get("gt_transform_applied")
        geo = gt_geometry(rec["mode"], pages[s].objects, transform=None if t == "identity" else (lambda v: transform_points(t, v, w, h)))
        v2 = d["label"].astype(bool)
        v3, why = labels_v3(d, rec["mode"], geo)
        segs = page_segments_of(d)
        seg, own = segs.seg, segs.owner
        mid = (seg[:, :2] + seg[:, 2:]) / 2
        ln = np.hypot(*(seg[:, 2:] - seg[:, :2]).T)
        q = np.floor(mid / WINDOW).astype(np.int64)
        ids = q[:, 0] * 100000 + q[:, 1]
        c = (v2 != v3)[own]
        u, inv = np.unique(ids[c], return_inverse=True)
        b = u[np.bincount(inv, ln[c]).argmax()]
        x0, y0 = (b // 100000) * WINDOW, (b % 100000) * WINDOW
        ins = (mid[:, 0] >= x0) & (mid[:, 0] < x0 + WINDOW) & (mid[:, 1] >= y0) & (mid[:, 1] < y0 + WINDOW)
        fig, ax = plt.subplots(figsize=(13, 13), dpi=85)
        cls = np.where((v2 & v3)[own], 1, np.where((v2 & ~v3)[own], 2, np.where((v3 & ~v2)[own], 3, 0)))
        for k, col, lw in ((0, "#d0d0d0", 0.3), (2, "#e01010", 0.7), (1, "#1a9a1a", 1.0), (3, "#2060ff", 1.2)):
            ax.add_collection(LineCollection(seg[ins & (cls == k)].reshape(-1, 2, 2), colors=col, linewidths=lw))
        for g, v in zip(outline_geometry(geo).geoms, geo.vertices):
            v = np.vstack([v, v[:1]]) if rec["mode"] == "hvac" and len(v) > 2 else v
            ax.plot(v[:, 0], v[:, 1], color="#e0b000", lw=0.5, alpha=0.8)
        ax.set_xlim(x0, x0 + WINDOW)
        ax.set_ylim(y0 + WINDOW, y0)
        ax.set_aspect("equal")
        ax.axis("off")
        L = d["length"]
        ax.set_title(f"{rec['filename'][:50]} {rec['mode']}: true {L[v2].sum() / 1e3:.0f}k -> {L[v3].sum() / 1e3:.0f}k pt")
        fig.tight_layout()
        fig.savefig(out / f"v3_{s[:9]}.png")
        plt.close(fig)
        print(s, rec["filename"], "saved", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
