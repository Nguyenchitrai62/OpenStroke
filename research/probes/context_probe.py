"""How much context does a token need? Pipe runs vs the area one serialized window covers.

For each E0 page: (1) the bbox diagonal of z-order windows of 4096 / 16384 / 65536 shapes;
(2) connected pipe runs = true shapes joined when within 2 pt, and their bbox diagonal;
(3) the share of true length lying in runs longer than a window's diagonal.

Run: py -3.12 research/probes/context_probe.py datasets/e0 > research/probes/context_probe.log
"""

import json
import sys
from pathlib import Path

import numpy as np
import shapely
from shapely import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.serialize import quantize, zorder  # noqa: E402

WINDOWS = (4096, 16384, 65536)


def components(n, pairs):
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs.T:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def main(data):
    data = Path(data)
    m = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for r in m["records"]:
        if r["status"] != "ok" or r.get("duplicate_gt"):
            continue
        z = np.load(data / r["npz"])
        bbox = z["bbox"].astype(np.float64)
        keep = ~np.isnan(bbox[:, 0])
        xy = np.stack([(bbox[:, 0] + bbox[:, 2]) / 2, (bbox[:, 1] + bbox[:, 3]) / 2], 1)[keep]
        order = np.argsort(zorder(quantize(xy)))
        xs = xy[order]
        ext = {}
        for w in WINDOWS:
            d = []
            for s in range(0, max(len(xs) - w, 0) + 1, max(w // 2, 1)):
                c = xs[s:s + w]
                d.append(np.hypot(*(c.max(0) - c.min(0))))
            ext[w] = float(np.median(d))
        # pipe runs
        lab = z["label"].astype(bool)
        off = z["shape_offsets"]
        seg = z["segments"].astype(np.float64)
        ts = np.flatnonzero(lab)
        geoms = [shapely.multilinestrings(seg[off[i]:off[i + 1]].reshape(-1, 2, 2)) for i in ts]
        tree = STRtree(geoms)
        pairs = tree.query(geoms, predicate="dwithin", distance=2.0)
        comp = components(len(ts), pairs)
        L = z["length"].astype(np.float64)[ts]
        b = bbox[ts]
        runs = {}
        for c in np.unique(comp):
            mm = comp == c
            bb = b[mm]
            runs[c] = (L[mm].sum(), float(np.hypot(bb[:, 2].max() - bb[:, 0].min(), bb[:, 3].max() - bb[:, 1].min())))
        rl = np.array([v[0] for v in runs.values()])
        rd = np.array([v[1] for v in runs.values()])
        share = {w: float(rl[rd > ext[w]].sum() / rl.sum()) for w in WINDOWS}
        page_diag = float(np.hypot(*(xy.max(0) - xy.min(0))))
        wl = np.average(rd, weights=rl)
        rows.append((r["stem"], r["mode"], len(xy), page_diag, ext, len(runs), wl, share))
        print(f"{r['stem']} {r['mode']} shapes={len(xy):,} page_diag={page_diag:.0f} "
              + " ".join(f"win{w}={ext[w]:.0f}" for w in WINDOWS)
              + f" runs={len(runs)} run_diag(len-weighted)={wl:.0f} "
              + " ".join(f"true_len_in_runs>win{w}={share[w]:.0%}" for w in WINDOWS), flush=True)
    for w in WINDOWS:
        print(f"MEDIAN win{w} diag {np.median([r[4][w] for r in rows]):.0f} pt; "
              f"true length in runs longer than the window: median {np.median([r[7][w] for r in rows]):.0%}")
    print(f"MEDIAN length-weighted run diag {np.median([r[6] for r in rows]):.0f} pt, page diag "
          f"{np.median([r[3] for r in rows]):.0f} pt")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "datasets/e0")
