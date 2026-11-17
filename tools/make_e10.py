"""OpenStroke30 dataset e10 (F165): e9g + the curve strokes that close an HVAC object labelled true.

Label rule v3 keeps a stroke only when >= 50 % of its length lies within 2 pt of a GT edge at <= 20 degrees. GT / Viewer polygons draw an
elbow's arc as a few chords, so arcs of elbows / tees / branches often fail it and are labelled `other` (F165: on the Viewer pages, curve
strokes on those outlines labelled other are longer than the ones labelled true); the model then learns to skip them. Added here: a curve
stroke (a `c` item) labelled other whose BOTH ends lie within ``TOUCH`` pt of a true stroke and whose length lies >= ``ON_EDGE`` within
``EDGE_TOL`` pt of the outline of a GT Elbow / Elbow vuông / Tee (objects drawn with arcs). Flexible-duct ribs (also the first rib at a
duct's end, which lies on a Duct outline), door swings, revision clouds and symbols fail the end or outline test. Class / instance / centre (ocls, oinst, octr) recomputed with openstroke.objects.assign_objects on the new labels, general
classes (as e9g). Nothing else changes (same strokes, folds, manifest records). Data only; GT is read, never written.

    py -3.12 tools/make_e10.py datasets/e9g datasets/score_pack_e8_all/gt datasets/e10 [datasets/openstroke_e10_data.zip]
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

TOUCH = 1.0      # pt: an end of the curve on a true stroke
EDGE_TOL = 3.0   # pt from a GT object's outline
ON_EDGE = 0.8    # share of the curve's length near that outline
ARC_CLASSES = (2, 3, 9)   # objects.HVAC_CLASSES: Elbow, Elbow vuông, Tee (outlines with real arcs; a duct's end next to a flexible duct is not)


def closing_curves(d: dict, objects: list[dict]) -> np.ndarray:
    """Boolean (n shapes,): curve strokes labelled other that close an HVAC object (rule in the module docstring)."""
    import shapely
    from scipy.spatial import cKDTree

    y = d["label"].astype(bool)
    curved = d["op_counts"][:, 1] > 0
    out = np.zeros(len(y), bool)
    polys = []
    for o in objects:
        v = np.asarray(o.get("vertices") or [], float)
        if len(v) >= 3 and class_of(o.get("shape_name")) in ARC_CLASSES:
            p = shapely.make_valid(shapely.Polygon(v))
            polys += [q.exterior for q in shapely.get_parts(p) if q.geom_type == "Polygon" and q.area > 0]
    cand0 = curved & ~y
    if not polys or not cand0.any() or not y.any():
        return out
    segs = page_segments_of(d)
    own, S = segs.owner, segs.seg.reshape(-1, 2, 2)
    idx = np.arange(len(own))
    first = np.full(len(y), -1)
    last = np.full(len(y), -1)
    first[own[::-1]] = idx[::-1]
    last[own] = idx
    cand = np.flatnonzero(cand0 & (first >= 0))
    tree = cKDTree(S[y[own]].reshape(-1, 2))
    da, _ = tree.query(S[first[cand], 0])
    db, _ = tree.query(S[last[cand], 1])
    cand = cand[(da <= TOUCH) & (db <= TOUCH)]
    if not len(cand):
        return out
    edge = shapely.union_all([b.buffer(EDGE_TOL) for b in polys])
    m = np.isin(own, cand)
    lines = shapely.linestrings(S[m])
    near = np.bincount(own[m], weights=shapely.length(shapely.intersection(lines, edge)), minlength=len(y))
    tot = np.bincount(own[m], weights=shapely.length(lines), minlength=len(y))
    out[cand] = near[cand] >= ON_EDGE * np.maximum(tot[cand], 1e-9)
    return out


def main(src, gt_dir, out, zip_out=None):
    src, gt_dir, out = Path(src), Path(gt_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    added = {"gt": [0, 0.0, 0.0], "vw": [0, 0.0, 0.0]}   # strokes, length added, true length before
    for i, r in enumerate(manifest["records"], 1):
        if r.get("status") != "ok":
            continue
        st = r["stem"]
        with np.load(src / r["npz"]) as z:
            d = {k: z[k] for k in z.files}
        f = gt_dir / f"{st}.json"
        objects = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        add = closing_curves(d, objects)
        g = "vw" if st.startswith("vw_") else "gt"
        L = d["length"].astype(np.float64)
        added[g][0] += int(add.sum())
        added[g][1] += float(L[add].sum())
        added[g][2] += float(L[d["label"].astype(bool)].sum())
        if add.any():
            d["label"] = (d["label"].astype(bool) | add).astype(d["label"].dtype)
            cls, inst, ctr = assign_objects(page_segments_of(d), objects, d["label"].astype(bool))
            d["ocls"], d["oinst"], d["octr"] = general_class(cls).astype(cls.dtype), inst, ctr
        tmp = out / f"{st}.tmp.npz"
        np.savez_compressed(tmp, **d)
        tmp.replace(out / r["npz"])
        print(f"[{i}/{len(manifest['records'])}] {st}: +{int(add.sum())} curve strokes ({L[add].sum():.0f} pt)", flush=True)
    for f in src.iterdir():
        if f.suffix == ".json" and f.name != "manifest.json":
            shutil.copy2(f, out / f.name)
    manifest["e10"] = {"rule": "tools/make_e10.py closing_curves", "touch_pt": TOUCH, "edge_tol_pt": EDGE_TOL, "on_edge": ON_EDGE,
                       "added": {g: {"strokes": a[0], "length": a[1], "share_of_true_length": a[1] / max(a[2], 1e-9)} for g, a in added.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for g, a in added.items():
        print(f"{g}: +{a[0]} curve strokes, +{a[1]:.0f} pt = {a[1] / max(a[2], 1e-9):.2%} of the true length")
    if zip_out:
        with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_STORED) as z:
            for f in sorted(out.iterdir()):
                if f.suffix in (".npz", ".json"):
                    z.write(f, f"{out.name}/{f.name}")
        print(f"zip {zip_out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
