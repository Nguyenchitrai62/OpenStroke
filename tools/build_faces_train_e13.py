"""Build 45-feature face classifier training dataset from ALL 188 clean TRAIN pages of e13_v4 (0 GPU, 0 val leakage).

Extracts candidate planar faces (page_candidates) from the GT/clean strokes on the 188 TRAIN pages ONLY
(20 AI_Takeoff GT train pages incl. doc000183_p0001_gt170 + 168 relabel_v3 Viewer train pages; strictly excludes
all 13 validation pages of folds.json!). Matches each face to GT objects (polygon IoU >= 0.5) using the unified
6-class + Tee taxonomy (with Concentric vs Eccentric Reducer already resolved geometrically in e13_v3).

Usage:
    py -3.12 tools/build_faces_train_e13.py --data datasets/e13_v3 --relabel datasets/relabel_v3
                                            --out datasets/e13_v3/faces_train_e13v3.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from shapely import STRtree
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openstroke.folds import load as load_folds, resolve  # noqa: E402
from openstroke.hvac_faces import (  # noqa: E402
    CLASSES,
    FEATURES,
    _curved_ink_faces,
    _split_tangent_seams,
    face_features,
    priority_candidates,
)
from openstroke.objects import HVAC_CLASSES, base6, class_of  # noqa: E402
from openstroke.obb.object_score import merge6_any  # noqa: E402
from openstroke.relabel import page_segments_of  # noqa: E402

DB = r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3"


def name6(n: str | None) -> str | None:
    if str(n or "").startswith("Tee"):
        return "Tee"
    b = base6(n) if class_of(n or "") else None
    return HVAC_CLASSES[merge6_any(b)] if b is not None else None


def _worker(job: tuple[str, str, list[tuple[str, list]]]):
    stem, npz_path, raw_objs = job
    g = []
    for n, verts in raw_objs:
        if len(verts) < 3:
            continue
        p = Polygon(np.asarray(verts, float)).buffer(0)
        if p.geom_type == "MultiPolygon":
            p = max(p.geoms, key=lambda q: q.area)
        if p.area > 0:
            g.append((n, p))
    with np.load(npz_path) as z:
        d = {k: z[k] for k in ("segments", "shape_offsets", "label", "op_counts")}
    segs = page_segments_of(d)
    picked = d["label"].astype(bool)
    curved = d["op_counts"][:, 1] > 0
    # Match exact v7 inference candidate generator (priority_candidates + _curved_ink_faces + _split_tangent_seams)
    faces, src, inner = priority_candidates(segs, picked, curved=curved)
    faces, src, inner = _curved_ink_faces(faces, src, inner, segs, picked, curved)
    faces, src, inner = _split_tangent_seams(faces, src, inner, segs, picked, curved)
    if not faces:
        return stem, np.zeros((0, len(FEATURES)), np.float64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    X = face_features(faces, segs, picked, curved)
    tree = STRtree([p for _, p in g]) if g else None
    y = np.zeros(len(faces), np.int64)
    in_sh = np.zeros(len(faces), np.float32)
    for i, f in enumerate(faces):
        best, bn, best_in = 0.0, "none", 0.0
        fa = max(float(f.area), 1e-9)
        if tree is not None:
            for k in tree.query(f, predicate="intersects"):
                k = int(k)
                ia = f.intersection(g[k][1]).area
                iou = ia / max(fa + g[k][1].area - ia, 1e-9)
                sh = ia / fa
                if iou > best:
                    best, bn = iou, g[k][0]
                if sh > best_in:
                    best_in = sh
        if best >= 0.5 and bn in CLASSES:
            y[i] = CLASSES.index(bn)
        in_sh[i] = float(best_in)
    return stem, X, y, in_sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/e13_v3")
    ap.add_argument("--relabel", default="datasets/relabel_v3")
    ap.add_argument("--out", default="datasets/e13_v3/faces_train_e13v3.npz")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    t0 = time.time()
    data, rel = Path(a.data), Path(a.relabel)
    man = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    folds = load_folds(data / "folds.json")
    entries, _ = resolve(man, folds)

    # STRICT TRAIN ONLY (fold == -1) — ZERO validation pages!
    train_stems = [st for st, fl, _ in entries if fl == -1]
    val_stems = {st for st, fl, _ in entries if fl >= 0}
    print(f"Extracting {len(FEATURES)}-feature faces on {len(train_stems)} TRAIN pages ONLY (excluding {len(val_stems)} val pages)...", flush=True)

    from openstroke.sources import load_gt_pages

    gt_map = {p.stem: p for p in load_gt_pages(DB)}
    jobs = []
    for st in train_stems:
        assert st not in val_stems
        raw_objs = []
        if st.startswith("vw_"):
            rj = json.loads((rel / st / "relabel.json").read_text(encoding="utf-8"))
            for o in rj["pages"][0]["objects"]:
                nm = name6(o.get("shape_name"))
                if nm is not None and len(o.get("vertices") or []) >= 3:
                    raw_objs.append((nm, o["vertices"]))
        else:
            if st not in gt_map:
                raise SystemExit(f"Missing GT page {st} in AI_Takeoff DB")
            for o in gt_map[st].objects:
                nm = name6(o.get("shape_name"))
                if nm is not None and len(o.get("vertices") or []) >= 3:
                    raw_objs.append((nm, o["vertices"]))
        jobs.append((st, str(data / f"{st}.npz"), raw_objs))

    Xs, Ys, Is, Ps = [], [], [], []
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for st, x, y, ish in ex.map(_worker, jobs):
            done += 1
            if len(y):
                Xs.append(x)
                Ys.append(y)
                Is.append(ish)
                Ps.extend([st] * len(y))
            if done % 25 == 0 or done == len(jobs):
                print(f"  {done:3d}/{len(jobs)} pages ({time.time()-t0:.1f}s), total faces={sum(len(v) for v in Ys):,}", flush=True)

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(Ys, axis=0)
    in_all = np.concatenate(Is, axis=0)
    p_all = np.asarray(Ps)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X_all, y=y_all, in_sh=in_all, page=p_all, features=np.asarray(FEATURES), classes=np.asarray(CLASSES))
    cnts = {CLASSES[i]: int(c) for i, c in enumerate(np.bincount(y_all, minlength=len(CLASSES)))}
    print(f"Saved {out}: {len(y_all):,} faces ({time.time()-t0:.1f}s) | classes={cnts}", flush=True)


if __name__ == "__main__":
    main()
