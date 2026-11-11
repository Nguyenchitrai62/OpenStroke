"""Ceiling of the OpenStroke15 tile stitching (0 GPU, no model): a perfect model inside every 1024 px tile returns each GT
object's strokes in that tile as one instance; merge_instances joins the pieces into page objects; the native scorer gives
F2 object. Also: the same with the 8192-token cap per tile, and with a looser merge rule (same class + any shared stroke).

    py -3.12 research/probes/stitch_ceiling.py <GT-only dataset folder>
"""
import sys
from pathlib import Path

import numpy as np
import yaml

from openstroke.obb.data.vectors import page_columns
from openstroke.obb.instance_decode import merge_instances
from openstroke.obb.vector_validation import DatasetVectorGT


def starts(n, size=1024, gap=200):
    if n <= size:
        return [0]
    s = list(range(0, n - size, size - gap))
    return s + [n - size]


def pieces(page, cap=None, size=1024):
    t, _, inst, pid = page_columns(page)
    pts = t[:, 1:17].reshape(-1, 8, 2)
    length = np.linalg.norm(np.diff(pts, axis=1), axis=2).sum(1)
    w, h = pts[..., 0].max() + 1, pts[..., 1].max() + 1
    out = []
    for y0 in starts(int(np.ceil(h)), size):
        for x0 in starts(int(np.ceil(w)), size):
            inside = ((pts[..., 0] >= x0) & (pts[..., 0] < x0 + size) & (pts[..., 1] >= y0) & (pts[..., 1] < y0 + size)).any(1)
            rows = np.flatnonzero(inside)
            if cap and len(rows) > cap:
                rows = rows[np.argsort(-length[rows])[:cap]]
            for o in np.unique(inst[rows]):
                if o >= 0:
                    out.append((x0, y0, int(o), sorted(pid[rows][inst[rows] == o].tolist())))
    return out


def loose_merge(dets, names, page):
    """Union-find: same class and at least one shared stroke."""
    parent = list(range(len(dets)))
    find = lambda i: i if parent[i] == i else find(parent[i])
    owner = {}
    for i, d in enumerate(dets):
        for p in d[4]:
            j = owner.setdefault((d[1], p), i)
            if j != i:
                parent[find(i)] = find(j)
    groups = {}
    for i, d in enumerate(dets):
        groups.setdefault(find(i), set()).update(d[4])
    first = {find(i): d for i, d in enumerate(dets)}
    return merge_instances([(first[g][0], first[g][1], 1.0, None, sorted(m)) for g, m in groups.items()], names, page)


def main(folder):
    folder = Path(folder)
    names = list(yaml.safe_load((folder / "data.yaml").read_text(encoding="utf-8"))["names"].values())
    gt = DatasetVectorGT(folder / "images" / "val", names)
    rows = {}
    for stem in gt.pages:
        with np.load(folder / "vectors" / "val" / f"{stem}.npz") as z:
            page = {k: z[k] for k in z.files}
        labels = [int(l.split()[0]) for l in (folder / "labels" / "val" / f"{stem}.txt").read_text().splitlines() if l.strip()]
        rows[stem] = (page, labels)
    for title, cap, merge in (("whole objects (no tiles)", None, None), ("tiles, current stitching", None, "cur"),
                              ("tiles + 8192 cap, current stitching", 8192, "cur"), ("tiles, any shared stroke", None, "loose"),
                              ("tiles + 8192 cap, any shared stroke", 8192, "loose")):
        dets, npieces = {}, 0
        for stem, (page, labels) in rows.items():
            if merge is None:
                _, _, inst, pid = page_columns(page)
                raw = [("p", labels[o], 1.0, None, sorted(pid[inst == o].tolist())) for o in np.unique(inst[inst >= 0])]
                dets[stem] = merge_instances(raw, names, page)
                continue
            ps = pieces(page, cap)
            npieces += len(ps)
            raw = [(f"img__1__{x0}___{y0}", labels[o], 1.0, None, m) for x0, y0, o, m in ps]
            dets[stem] = merge_instances(raw, names, page) if merge == "cur" else loose_merge(raw, names, page)
        r = gt.evaluate(dets, "instances", names)
        print(f"{title:38s} F2 {r['F2']:.3f}  R {r['R']:.3f}  P {r['P']:.3f}  (pieces {npieces})", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
