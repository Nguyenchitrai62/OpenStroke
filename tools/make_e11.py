"""OpenStroke30 dataset e11 (F173): e10 + square elbows whose outer wall and miter diagonal are one PDF item.

Label rule v3 calls a shape true when >= 50 % of its length lies along a GT edge. A square elbow (Elbow vuông) is often drawn as ONE item: the
outer wall(s) plus the 45-degree miter line from the outer to the inner corner. The GT polygon is the square without the miter, so the wall is
on the edge but the diagonal is not, the item stays under 50 % and the whole item - wall included - is labelled `other` (F173: 39 / 39 such
items on the GT pages, 32 / 38 on the Viewer pages). The model then learns to leave the elbow's outer wall out (user 2026-11-17).

Added: a straight item (no curve command) with >= 2 segments, >= 95 % of its length inside a GT Elbow vuông (2 pt margin), >= max(MIN_EDGE pt,
EDGE_SHARE of its length) on that elbow's outline (<= EDGE_TOL pt) and a segment at 35-55 degrees to an outline segment. Class / instance /
centre recomputed (assign_objects, general classes). Nothing else changes. Data only; GT is read, never written.

    py -3.12 tools/make_e11.py datasets/e10 datasets/score_pack_e8_all/gt datasets/e11 [datasets/openstroke_e11_data.zip]
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openstroke.objects import assign_objects, class_of, general_class  # noqa: E402
from openstroke.relabel import page_segments_of  # noqa: E402

ELBOW_SQ = 3          # objects.HVAC_CLASSES index of Elbow vuông
EDGE_TOL = 2.0        # pt from the elbow's outline
MIN_EDGE, EDGE_SHARE = 5.0, 0.25
DIAG = (35.0, 55.0)   # degrees between the miter and an outline segment


def wall_and_miter(d: dict, objects: list[dict]) -> np.ndarray:
    """Boolean (n shapes,): items labelled other that draw a square elbow's outer wall and its miter (rule in the module docstring)."""
    import shapely

    y = d["label"].astype(bool)
    curved = d["op_counts"][:, 1] > 0
    out = np.zeros(len(y), bool)
    elbows = []
    for o in objects:
        v = np.asarray(o.get("vertices") or [], float)
        if len(v) >= 3 and class_of(o.get("shape_name")) == ELBOW_SQ:
            parts = [q for q in shapely.get_parts(shapely.make_valid(shapely.Polygon(v))) if q.geom_type == "Polygon" and q.area > 0]
            if parts:
                elbows.append(max(parts, key=lambda q: q.area))
    if not elbows:
        return out
    segs = page_segments_of(d)
    off, S = segs.shape_offsets, segs.seg
    tree = shapely.STRtree(shapely.linestrings(S.reshape(-1, 2, 2)))
    for poly in elbows:
        inside, edge = poly.buffer(EDGE_TOL), poly.boundary.buffer(EDGE_TOL)
        for sh in np.unique(segs.owner[tree.query(inside, predicate="intersects")]):
            if y[sh] or curved[sh] or off[sh + 1] - off[sh] < 2:
                continue
            ss = S[off[sh]:off[sh + 1]]
            ls = shapely.linestrings(ss.reshape(-1, 2, 2))
            n = shapely.length(ls)
            if shapely.length(shapely.intersection(ls, inside)).sum() < 0.95 * n.sum():
                continue
            on = (shapely.length(shapely.intersection(ls, edge)) >= 0.8 * n) & (n > 1.0)
            if not on.any() or n[on].sum() < max(MIN_EDGE, EDGE_SHARE * n.sum()):
                continue
            dv = ss[:, 2:] - ss[:, :2]
            a = np.degrees(np.arctan2(dv[:, 1], dv[:, 0]))
            rest = ~on & (n > 1.0)
            if not rest.any():
                continue
            diff = np.abs(((a[rest][:, None] - a[on][None, :]) + 90) % 180 - 90)
            out[sh] = bool(((diff >= DIAG[0]) & (diff <= DIAG[1])).any())
    return out


def main(src, gt_dir, out, zip_out=None):
    src, gt_dir, out = Path(src), Path(gt_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    added = {"gt": [0, 0.0], "vw": [0, 0.0]}
    for i, r in enumerate(manifest["records"], 1):
        if r.get("status") != "ok":
            continue
        st = r["stem"]
        with np.load(src / r["npz"]) as z:
            d = {k: z[k] for k in z.files}
        f = gt_dir / f"{st}.json"
        objects = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        add = wall_and_miter(d, objects)
        g = "vw" if st.startswith("vw_") else "gt"
        added[g][0] += int(add.sum())
        added[g][1] += float(d["length"][add].sum())
        if add.any():
            d["label"] = (d["label"].astype(bool) | add).astype(d["label"].dtype)
            cls, inst, ctr = assign_objects(page_segments_of(d), objects, d["label"].astype(bool))
            d["ocls"], d["oinst"], d["octr"] = general_class(cls).astype(cls.dtype), inst, ctr
            print(f"[{i}/{len(manifest['records'])}] {st}: +{int(add.sum())} wall + miter items", flush=True)
        tmp = out / f"{st}.tmp.npz"
        np.savez_compressed(tmp, **d)
        tmp.replace(out / r["npz"])
    for f in src.iterdir():
        if f.suffix == ".json" and f.name != "manifest.json":
            shutil.copy2(f, out / f.name)
    manifest["e11"] = {"rule": "tools/make_e11.py wall_and_miter", "added": {g: {"items": a[0], "length": a[1]} for g, a in added.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for g, a in added.items():
        print(f"{g}: +{a[0]} items, +{a[1]:.0f} pt")
    if zip_out:
        with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_STORED) as z:
            for f in sorted(out.iterdir()):
                if f.suffix in (".npz", ".json"):
                    z.write(f, f"{out.name}/{f.name}")
        print(f"zip {zip_out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
