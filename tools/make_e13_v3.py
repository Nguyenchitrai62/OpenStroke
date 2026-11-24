"""Build object-clean HVAC dataset e13_v3 from e13_v2 + relabel_v2 (0 ML, pure CAD geometry).

Fixes two train-only object-level inconsistencies diagnosed in F222 & F224:
  1. Splits unsplit multi-face ``source='agent'`` connected-component blobs (>= 4 shapes) on the 168 Viewer TRAIN pages
     into individual planar objects (Duct / Elbow / Reducer / Chân rẽ / Tee) using pure Shapely ``page_faces``,
     keeping 100 % IDENTICAL stroke ``label`` while sharpening ``oinst`` object boundaries for ``edge_head``.
  2. Standardizes Viewer ``Reducer`` objects on the 168 TRAIN pages into ``Concentric Reducer`` vs ``Eccentric Reducer``
     using the exact geometric ``openstroke.hvac_faces.reducer_kind`` rule (4 corners & 2 right corners -> Eccentric),
     matching AI_Takeoff GT's 6-class taxonomy in ``ocls``.

All 25 GT pages (including doc000183_p0001_gt170) and all 8 Viewer8 VAL pages are kept 100 % BYTE-IDENTICAL to e13_v2
so every validation metric is strictly apples-to-apples with E114..E142.

Usage:
    py -3.12 tools/make_e13_v3.py [--src datasets/e13_v2] [--relabel datasets/relabel_v2]
                                  [--out-relabel datasets/relabel_v3] [--out datasets/e13_v3]
                                  [--zip datasets/openstroke_e13_v3_data.zip]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from openstroke.hvac_faces import page_faces, reducer_kind  # noqa: E402
from openstroke.objects import class_of  # noqa: E402
from openstroke.relabel import page_segments_of  # noqa: E402

VAL8_STEMS = {
    "vw_2f3d386b8610_p0004",
    "vw_579f78f50745_p0002",
    "vw_62b1ac808de5_p0002",
    "vw_62b1ac808de5_p0003",
    "vw_e241de8b8f63_p0001",
    "vw_e241de8b8f63_p0002",
    "vw_e3fe1f60a5ca_p0002",
    "vw_f4afa52daee9_p0004",
}


def object_centres(bbox: np.ndarray, oinst: np.ndarray) -> np.ndarray:
    c = np.full((len(oinst), 2), np.nan, np.float32)
    for k in np.unique(oinst[oinst >= 0]):
        m = oinst == k
        b = bbox[m]
        c[m] = [(b[:, 0].min() + b[:, 2].max()) / 2, (b[:, 1].min() + b[:, 3].max()) / 2]
    return c


def classify_face_geom(poly: Polygon, sarr: np.ndarray, curved: np.ndarray) -> str:
    """Pure geometric 6-class label for a planar face (zero ML)."""
    has_curve = bool(curved[sarr].any())
    mrr = poly.minimum_rotated_rectangle
    coords = np.asarray(mrr.exterior.coords)
    e1 = float(np.hypot(*(coords[1] - coords[0])))
    e2 = float(np.hypot(*(coords[2] - coords[1])))
    w_min, l_max = min(e1, e2), max(e1, e2)
    fill_r = float(poly.area / max(mrr.area, 1e-6))
    simp = poly.simplify(max(0.8, 0.08 * w_min), preserve_topology=True)
    n_v = len(simp.exterior.coords) - 1 if simp.geom_type == "Polygon" else 0
    if has_curve and fill_r < 0.88:
        return "Elbow"
    if n_v == 4 and 0.55 <= fill_r <= 0.88 and l_max / max(w_min, 1e-3) <= 3.2:
        sc = np.asarray(simp.exterior.coords)
        elens = sorted(float(np.hypot(*(sc[k + 1] - sc[k]))) for k in range(4))
        if elens[0] / max(elens[-1], 1e-3) < 0.78 and abs(elens[1] - elens[2]) / max(elens[2], 1e-3) < 0.25:
            return reducer_kind(poly)
    if l_max < 14.0 and w_min < 10.0:
        return "Chân rẽ"
    if n_v >= 6 and fill_r < 0.75:
        return "Tee"
    return "Duct"


def poly_from_shapes(seg: np.ndarray, offs: np.ndarray, sarr: np.ndarray) -> list[list[float]]:
    pts = []
    for sid in sarr:
        ss = seg[offs[sid]:offs[sid + 1]]
        if len(ss):
            pts.append(ss[:, :2])
            pts.append(ss[:, 2:4])
    if not pts:
        return []
    xy = np.vstack(pts)
    hull = shapely.MultiPoint(xy).convex_hull
    if hull.geom_type == "Polygon" and hull.area > 0:
        return np.asarray(hull.exterior.coords[:-1], float).round(2).tolist()
    x0, y0 = xy.min(0)
    x1, y1 = xy.max(0)
    return [[round(float(x0), 2), round(float(y0), 2)], [round(float(x1), 2), round(float(y0), 2)],
            [round(float(x1), 2), round(float(y1), 2)], [round(float(x0), 2), round(float(y1), 2)]]


def process_train_page(stem: str, src_npz: Path, rel_dir: Path, out_rel_dir: Path, out_npz: Path) -> dict:
    with np.load(src_npz) as z:
        d = {k: z[k].copy() for k in z.files if not k.startswith("pc_")}
    rj = json.loads((rel_dir / "relabel.json").read_text(encoding="utf-8"))
    objs = rj["pages"][0]["objects"]

    n = len(d["length"])
    seg = d["segments"].astype(np.float64)
    offs = d["shape_offsets"].astype(np.int64)
    L = d["length"].astype(np.float64)
    curved = d["op_counts"][:, 1] > 0
    label_orig = d["label"].copy()
    oinst_orig = d["oinst"].copy()
    ocls_orig = d["ocls"].copy()

    # Map existing oinst -> shape indices
    by_oid: dict[int, list[int]] = {}
    for sid in np.flatnonzero((label_orig > 0) & (oinst_orig >= 0)):
        by_oid.setdefault(int(oinst_orig[sid]), []).append(int(sid))

    psegs = None
    new_objs = []
    new_oinst = np.full(n, -1, dtype=d["oinst"].dtype)
    new_ocls = np.zeros(n, dtype=d["ocls"].dtype)
    blobs_split = 0
    blobs_into = 0
    red_conc = 0
    red_ecc = 0

    for oid, o in enumerate(objs):
        slist = by_oid.get(oid, [])
        if not slist:
            continue
        cls_name = str(o.get("shape_name") or "Duct")
        src_tag = str(o.get("source") or "viewer")

        # 1. Split multi-face agent blobs (>= 4 shapes)
        did_split = False
        if src_tag == "agent" and len(slist) >= 4:
            if psegs is None:
                psegs = page_segments_of(d)
            picked = np.zeros(n, dtype=bool)
            picked[slist] = True
            faces = page_faces(psegs, picked, curved=curved)
            if len(faces) >= 2:
                f_bounds = [f.boundary for f in faces]
                f_shapes: list[list[int]] = [[] for _ in faces]
                unassigned: list[int] = []
                for sid in slist:
                    ss = seg[offs[sid]:offs[sid + 1]]
                    if not len(ss):
                        continue
                    mp = shapely.multipoints(0.5 * (ss[:, :2] + ss[:, 2:4]))
                    best_fi, best_d = -1, 4.0
                    for fi, fb in enumerate(f_bounds):
                        dist = float(fb.distance(mp))
                        if dist < best_d:
                            best_d, best_fi = dist, fi
                    if best_fi >= 0:
                        f_shapes[best_fi].append(sid)
                    else:
                        unassigned.append(sid)
                nonempty = [fi for fi in range(len(faces)) if f_shapes[fi]]
                for sid in unassigned:
                    ss = seg[offs[sid]:offs[sid + 1]]
                    if len(ss) and nonempty:
                        mp = shapely.multipoints(0.5 * (ss[:, :2] + ss[:, 2:4]))
                        best_fi = min(nonempty, key=lambda fi: float(faces[fi].distance(mp)))
                        f_shapes[best_fi].append(sid)
                parts = [(faces[fi], f_shapes[fi]) for fi in range(len(faces)) if f_shapes[fi]]
                if len(parts) >= 2:
                    did_split = True
                    blobs_split += 1
                    blobs_into += len(parts)
                    for poly, ps in parts:
                        sarr = np.asarray(ps, np.int64)
                        cname = classify_face_geom(poly, sarr, curved)
                        if cname == "Concentric Reducer":
                            red_conc += 1
                        elif cname == "Eccentric Reducer":
                            red_ecc += 1
                        nid = len(new_objs)
                        g = shapely.simplify(poly, 0.3)
                        verts = np.asarray(g.exterior.coords[:-1], float).round(2).tolist() if g.geom_type == "Polygon" else poly_from_shapes(seg, offs, sarr)
                        new_objs.append({
                            "shape_name": cname,
                            "kind": "polygon",
                            "group": "primary",
                            "coordinate_system": "display_top_left",
                            "vertices": verts,
                            "source": "agent_split",
                        })
                        cid = class_of(cname) or 1
                        new_oinst[sarr] = nid
                        new_ocls[sarr] = cid

        if not did_split:
            # 2. Standardize bare "Reducer" -> Concentric / Eccentric Reducer via reducer_kind
            if cls_name in ("Reducer", "Reducer vuông tròn"):
                pts = o.get("vertices") or []
                poly = Polygon(np.asarray(pts, float)).buffer(0) if len(pts) >= 3 else None
                if poly is not None and poly.geom_type == "Polygon" and poly.area > 0:
                    cls_name = reducer_kind(poly)
                else:
                    cls_name = "Concentric Reducer"
                if cls_name == "Eccentric Reducer":
                    red_ecc += 1
                else:
                    red_conc += 1
            nid = len(new_objs)
            o2 = dict(o)
            o2["shape_name"] = cls_name
            new_objs.append(o2)
            sarr = np.asarray(slist, np.int64)
            cid = class_of(cls_name) or int(ocls_orig[sarr[0]]) or 1
            new_oinst[sarr] = nid
            new_ocls[sarr] = cid

    # Safety assertion: 100% identical stroke label!
    new_label = (new_oinst >= 0).astype(d["label"].dtype)
    assert np.array_equal(new_label > 0, label_orig > 0), f"{stem}: stroke label changed!"

    d["label"] = label_orig
    d["oinst"] = new_oinst
    d["ocls"] = new_ocls
    d["octr"] = object_centres(d["bbox"], new_oinst).astype(d["octr"].dtype)
    np.savez_compressed(out_npz, **d)

    out_rel_dir.mkdir(parents=True, exist_ok=True)
    rj_out = json.loads(json.dumps(rj))
    rj_out["pages"][0]["objects"] = new_objs
    (out_rel_dir / "relabel.json").write_text(json.dumps(rj_out, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out_rel_dir / "relabel_strokes.npz", label=d["label"], oinst=new_oinst, ocls=new_ocls)

    return {
        "old_objs": len(objs),
        "new_objs": len(new_objs),
        "blobs_split": blobs_split,
        "blobs_into": blobs_into,
        "red_conc": red_conc,
        "red_ecc": red_ecc,
        "true_shapes": int((label_orig > 0).sum()),
        "cls_counts": dict(Counter(o["shape_name"] for o in new_objs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/e13_v2")
    ap.add_argument("--relabel", default="datasets/relabel_v2")
    ap.add_argument("--out-relabel", default="datasets/relabel_v3")
    ap.add_argument("--out", default="datasets/e13_v3")
    ap.add_argument("--zip", default="datasets/openstroke_e13_v3_data.zip")
    a = ap.parse_args()

    t0 = time.time()
    src, rel, out_rel, out = Path(a.src), Path(a.relabel), Path(a.out_relabel), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    out_rel.mkdir(parents=True, exist_ok=True)
    man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))

    tot = Counter()
    cls_tot = Counter()
    trn_vw = 0
    for r in man["records"]:
        if r.get("status") != "ok":
            continue
        st, fn = r["stem"], r["npz"]
        dst = out / fn
        if dst.exists():
            dst.unlink()
        if st.startswith("vw_") and st not in VAL8_STEMS and (rel / st / "relabel.json").exists():
            info = process_train_page(st, src / fn, rel / st, out_rel / st, dst)
            trn_vw += 1
            tot["old_objs"] += info["old_objs"]
            tot["new_objs"] += info["new_objs"]
            tot["blobs_split"] += info["blobs_split"]
            tot["blobs_into"] += info["blobs_into"]
            tot["red_conc"] += info["red_conc"]
            tot["red_ecc"] += info["red_ecc"]
            cls_tot.update(info["cls_counts"])
            if "relabel" in r:
                r["relabel"] = {**r["relabel"], "v3_objects": info["new_objs"], "v3_blobs_split": info["blobs_split"]}
            if trn_vw % 25 == 0:
                print(f"  processed {trn_vw}/168 Viewer train pages ({time.time()-t0:.1f}s)...", flush=True)
        else:
            # GT pages (25) and VAL8 pages (8): keep 100% byte-identical via hardlink/copy
            try:
                os.link(src / fn, dst)
            except OSError:
                shutil.copy2(src / fn, dst)
            if st in VAL8_STEMS and (rel / st / "relabel.json").exists():
                (out_rel / st).mkdir(parents=True, exist_ok=True)
                shutil.copy2(rel / st / "relabel.json", out_rel / st / "relabel.json")
                shutil.copy2(rel / st / "relabel_strokes.npz", out_rel / st / "relabel_strokes.npz")

    man["e13_v3"] = {
        "tool": "tools/make_e13_v3.py",
        "from": str(src),
        "viewer_train_pages_processed": trn_vw,
        "blobs_split": tot["blobs_split"],
        "blobs_split_into": tot["blobs_into"],
        "train_objects_before": tot["old_objs"],
        "train_objects_after": tot["new_objs"],
        "reducers_concentric": tot["red_conc"],
        "reducers_eccentric": tot["red_ecc"],
    }
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for f in src.glob("folds*.json"):
        shutil.copy2(f, out / f.name)

    print(f"\n=== e13_v3 BUILD COMPLETE ({time.time()-t0:.1f}s) ===", flush=True)
    print(f"  Viewer train pages: {trn_vw}", flush=True)
    print(f"  Agent blobs split : {tot['blobs_split']} -> {tot['blobs_into']} individual planar objects", flush=True)
    print(f"  Viewer train objs : {tot['old_objs']} -> {tot['new_objs']} (+{tot['new_objs']-tot['old_objs']})", flush=True)
    print(f"  Reducers resolved : Concentric={tot['red_conc']}, Eccentric={tot['red_ecc']}", flush=True)
    print(f"  Class breakdown   : {dict(cls_tot)}", flush=True)

    if a.zip:
        zpath = Path(a.zip)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
            for f in sorted(out.iterdir()):
                zf.write(f, f"{out.name}/{f.name}")
        print(f"  ZIP saved         : {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
