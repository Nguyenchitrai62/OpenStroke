"""Measure OBB and labelled-stroke coverage of a tile recipe, without running a model.

Usage: python research/probes/obb_tile_coverage.py datasets/obb_hvac7 --split val --scales 1,0.5,1.5 --vectors
"""
import argparse
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import shapely
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.obb.data.prepare import _to_dota, find_splits  # noqa: E402
from openstroke.obb.data.split import window_starts  # noqa: E402
from openstroke.obb.data.vectors import cap_order, page_columns  # noqa: E402
from openstroke.obb.eval.dota import parse_dota_txt  # noqa: E402


def page_coverage(item, scales, size, gap, cap, vectors):
    path, labels = item[:2]
    with Image.open(path) as image:
        width, height = image.size
    coords = np.asarray([obj[0] for obj in parse_dota_txt(labels)], dtype=np.float64).reshape(-1, 4, 2)
    seen = np.zeros(len(coords), bool)
    totals = np.zeros(3)
    if vectors:
        if len(item) < 4:
            raise ValueError(f"Missing vectors for {path}")
        with np.load(item[2]) as page:
            tokens, _, inst, _ = page_columns(page)
        if inst is None:
            raise ValueError(f"Missing object IDs for {path}")
        mapping = np.full(max(item[3], default=-1) + 1, -1, dtype=np.int64)
        for row, obj in item[3].items():
            mapping[row] = obj
        known = (inst >= 0) & (inst < len(mapping))
        inst[known] = mapping[inst[known]]
        inst[~known] = -1
        points = tokens[:, 1:17].reshape(-1, 8, 2)
        lengths = np.linalg.norm(np.diff(points, axis=1), axis=2).sum(axis=1)
        positive = (inst >= 0) & (inst < len(coords))
        stroke_seen = np.zeros(len(points), bool)
        totals[2] = lengths[positive].sum()
    for scale in scales:
        w, h = max(1, round(width * scale)), max(1, round(height * scale))
        poly = shapely.polygons(coords * scale)
        bad = ~shapely.is_valid(poly)
        if bad.any():
            poly[bad] = shapely.convex_hull(poly[bad])
        areas = shapely.area(poly)
        box = shapely.bounds(poly)
        if vectors:
            scaled = points * scale
            lo, hi = scaled.min(axis=1), scaled.max(axis=1)
        for y in window_starts(h, size, size - gap):
            for x in window_starts(w, size, size - gap):
                x1, y1 = min(w, x + size), min(h, y + size)
                idx = np.flatnonzero((box[:, 2] > x) & (box[:, 0] < x1) & (box[:, 3] > y) & (box[:, 1] < y1) & (areas > 0))
                iof = shapely.area(shapely.intersection(poly[idx], shapely.box(x, y, x1, y1))) / areas[idx]
                objects = idx[iof >= 0.7]
                seen[objects] = True
                if vectors and len(objects):
                    # The candidate filter must match tile_npz before the length-based cap.
                    cand = np.flatnonzero((hi[:, 0] >= x) & (lo[:, 0] <= x + size) &
                                          (hi[:, 1] >= y) & (lo[:, 1] <= y + size))
                    p = scaled[cand]
                    inside = ((p >= [x, y]) & (p <= [x + size, y + size])).all(axis=2).any(axis=1)
                    cand = cand[inside]
                    eligible = np.isin(inst[cand], objects)
                    totals[0] += lengths[cand[eligible]].sum()
                    sel = cap_order(scaled[cand].reshape(-1, 16), cap)
                    retained = cand[sel[eligible[sel]]]
                    totals[1] += lengths[retained].sum()
                    stroke_seen[retained] = True
    covered = lengths[stroke_seen].sum() if vectors else 0.0
    return int(seen.sum()), len(seen), covered, totals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--split", choices=("train", "val"), default="val")
    ap.add_argument("--scales", default="1")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--gap", type=int, default=200)
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--vectors", action="store_true")
    a = ap.parse_args()
    scales = [float(s) for s in a.scales.split(",")]
    names, splits = find_splits(a.root)
    if not names or not splits[a.split]:
        raise ValueError(f"Missing names or images for {a.split}")
    found = total = 0
    covered = 0.0
    lengths = np.zeros(3)
    with TemporaryDirectory(prefix="openstroke-tile-coverage-") as temp:
        items, _, _, _ = _to_dota(splits[a.split], names, Path(temp))
        for item in items:
            n, t, length, stats = page_coverage(item, scales, a.size, a.gap, a.cap, a.vectors)
            found, total = found + n, total + t
            covered += length
            lengths += stats
    print(f"{a.split} scales={a.scales}: {found}/{total} objects in at least one tile, {total - found} lost")
    if a.vectors:
        print(f"  unique GT stroke length supervised after cap: {covered / max(lengths[2], 1e-9):.3%}; "
              f"eligible GT stroke length lost to cap across tile visits: {1 - lengths[1] / max(lengths[0], 1e-9):.3%}")


if __name__ == "__main__":
    main()
