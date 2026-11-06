"""E63: an OpenOBB dataset of HVAC objects from the GT (docs/OPENSTROKE7.md, research/HYPOTHESES.md H78).

Each HVAC GT page is drawn from its own vectors (payload strokes and fills, real colours and widths; no PyMuPDF, no layers) at ``--scale`` px
per pt, and every GT object in scope becomes an oriented box (smallest rotated rectangle of its polygon) in OpenOBB's 8-point format. The split
follows the PDF folds (``--val-folds``); a page's GT transform comes from the built dataset or, for a page built after it, from the same
vector-ink calibration. A PDF with two GT versions the user has not chosen between gives its latest edited version (``--duplicates latest``,
user 2026-11-06: use all the data). ``--pseudo``: train pages also from AI_Takeoff's pipeline runs (``normalized.json`` primary objects) on
pages with no GT, never from a val PDF. AI_Takeoff is only read.

    py -3.12 tools/make_obb_dataset.py --out datasets/obb_hvac2 --val-folds 3,4 --duplicates latest --pseudo
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import shapely
from shapely import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openstroke.build import shape_attributes  # noqa: E402
from openstroke.labels import segment_lines  # noqa: E402
from openstroke.features import shape_points  # noqa: E402
from openstroke.relabel import object_strokes  # noqa: E402
from openstroke.folds import load as load_folds  # noqa: E402
from openstroke.geometry import IDENTITY, calibrate, page_segments, transform_points  # noqa: E402
from openstroke.labels import gt_geometry  # noqa: E402
from openstroke.objects import HVAC_CLASSES, assign_objects, class_of  # noqa: E402
from openstroke.sources import load_gt_pages, latest_payload_path  # noqa: E402

BE = Path(r"F:\Source_code\AI_Takeoff\BE")
NAMES = list(HVAC_CLASSES[1:])  # OpenOBB class id = objects class - 1


def payload_of(page, extra: Path) -> Path | None:
    p = latest_payload_path(BE / "data" / "drawings", page.document_id, page.page_number)
    if p is None:
        q = extra / f"document-{page.document_id:06d}" / f"page-{page.page_number:04d}.v0.json.gz"
        p = q if q.exists() else None
    return p


def render(shapes: list[dict], width: float, height: float, scale: float) -> np.ndarray:
    """The page as the PDF draws it, from its vectors: fills first, then strokes, each in its own colour and width (at least 1 px)."""

    img = np.full((int(round(height * scale)), int(round(width * scale)), 3), 255, np.uint8)
    segs = page_segments(shapes)
    off = segs.shape_offsets
    for i, s in enumerate(shapes):
        a, b = off[i], off[i + 1]
        if a == b:
            continue
        pts = np.round(segs.seg[a:b] * scale).astype(np.int32)
        fill = s.get("fill")
        if fill and s.get("type") in ("f", "fs") and b - a >= 2:
            ring = np.vstack([pts[:, :2], pts[-1:, 2:]])
            cv2.fillPoly(img, [ring], tuple(int(255 * c) for c in fill[:3][::-1]), cv2.LINE_AA)
    for i, s in enumerate(shapes):
        a, b = off[i], off[i + 1]
        if a == b or s.get("type") == "f":
            continue
        col = tuple(int(255 * c) for c in (s.get("color") or (0, 0, 0))[:3][::-1])
        w = max(1, int(round(float(s.get("width") or 0) * scale)))
        pts = np.round(segs.seg[a:b] * scale).astype(np.int32)
        for x0, y0, x1, y1 in pts:
            cv2.line(img, (x0, y0), (x1, y1), col, w, cv2.LINE_AA)
    return img


def vector_tokens(shapes: list[dict], boxes_kept: list[tuple[int, np.ndarray, dict]], scale: float) -> np.ndarray:
    """E64 tokens for OpenOBB25, one row per primitive, pixel coordinates of the page image: type, 8 arc-length points, stroke rgb, width px,
    layer (always -1: the model must not rely on layers), inst (line index of its object in the page's label file, or -1), pid (shape index)."""

    segs = page_segments(shapes)
    off = segs.shape_offsets
    n = len(shapes)
    rel = shape_points(segs.seg, off, 8).astype(np.float64)
    has = np.diff(off) > 0
    cen = np.zeros((n, 2))
    idx = np.flatnonzero(has)
    if len(idx):
        lo = np.minimum.reduceat(np.minimum(segs.seg[:, :2], segs.seg[:, 2:]), off[idx])
        hi = np.maximum.reduceat(np.maximum(segs.seg[:, :2], segs.seg[:, 2:]), off[idx])
        cen[idx] = (lo + hi) / 2
    pts = (rel + cen[:, None, :]) * scale
    attrs = shape_attributes(shapes, segs)
    length = np.bincount(segs.owner, weights=segs.length, minlength=n)
    objs = [o for _, _, o in boxes_kept]
    mask = object_strokes("hvac", objs, segs, attrs, length) if objs else np.zeros(n, bool)
    _, inst, _ = assign_objects(segs, objs, mask)
    rgb = np.nan_to_num(attrs["stroke_rgb"], nan=0.0)
    tok = np.c_[attrs["type_code"].astype(np.float64), pts.reshape(n, 16), rgb, np.nan_to_num(attrs["width"], nan=0.0) * scale,
                np.full(n, -1.0), inst.astype(np.float64), np.arange(n, dtype=np.float64)]
    return tok[has].astype(np.float32)


def boxes(objects: list[dict], transform, width: float, height: float, keep_objects: bool = False) -> list:
    out = []
    for o in objects:
        c = class_of(o.get("shape_name"))
        v = np.asarray(o.get("vertices") or [], float)
        if not c or len(v) < 2:
            continue
        if transform != IDENTITY:
            v = transform_points(transform, v, width, height)
        r = shapely.minimum_rotated_rectangle(shapely.multipoints(v))
        q = shapely.get_coordinates(r)[:4]
        if len(q) == 4:
            out.append((c - 1, q, {**o, "vertices": v.tolist()}) if keep_objects else (c - 1, q))
    return out


class PseudoPage:
    """A page labelled by AI_Takeoff's pipeline run, shaped like ``sources.GTPage`` for this tool."""

    def __init__(self, doc, filename, sha, page, width, height, objects, job):
        self.document_id, self.filename, self.sha256, self.page_number = doc, filename, sha, page
        self.page_width, self.page_height, self.objects, self.mode, self.pseudo = width, height, objects, "hvac", True
        self.updated_at = ""
        self.stem = f"doc{doc:06d}_p{page:04d}_ps{job}"


def pseudo_pages(gt_pages, fold_of, fold_by_name, val_folds) -> list:
    import sqlite3

    con = sqlite3.connect((BE / "data" / "ai_takeoff.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    jobs = con.execute("select j.id, j.document_id, d.filename, d.sha256 from jobs j join documents d on d.id = j.document_id "
                       "where j.mode = 'hvac' and j.status = 'succeeded' order by j.id desc").fetchall()
    con.close()
    has_gt = {(p.sha256, p.page_number) for p in gt_pages} | {(p.filename, p.page_number) for p in gt_pages}
    out, seen = [], set()
    for job, doc, filename, sha in jobs:
        if fold_of.get(sha, fold_by_name.get(filename, -1)) in val_folds:
            continue
        f = BE / "data" / "runs" / f"job-{job:06d}" / "normalized.json"
        if not f.exists():
            continue
        for q in json.loads(f.read_text(encoding="utf-8")).get("pages") or []:
            n = int(q.get("page_number", 0))
            objs = [o for o in q.get("objects") or [] if o.get("group") in ("primary", "reference")]
            if not objs or (sha, n) in has_gt or (filename, n) in has_gt or (doc, n) in seen:
                continue
            seen.add((doc, n))  # newest job of the page only
            out.append(PseudoPage(doc, filename, sha, n, q.get("page_width"), q.get("page_height"), objs, job))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/obb_hvac"))
    ap.add_argument("--built", type=Path, default=Path("datasets/e6"), help="dataset whose manifest gives each page's GT transform")
    ap.add_argument("--folds", type=Path, default=Path("datasets/e6/folds_cv_pseudo.json"))
    ap.add_argument("--val-folds", default="3,4")
    ap.add_argument("--scale", type=float, default=4.0, help="px per pt")
    ap.add_argument("--payloads", type=Path, default=Path("datasets/payloads"))
    ap.add_argument("--duplicates", default="skip", help="skip | latest: a PDF with two GT versions and no user choice")
    ap.add_argument("--pseudo", action="store_true", help="add pipeline-labelled train pages (no GT, not a val PDF)")
    ap.add_argument("--vectors", action="store_true", help="E64: also write vectors/{split}/{stem}.npz (OpenOBB25 tokens)")
    a = ap.parse_args()
    val_folds = {int(x) for x in a.val_folds.split(",") if x.strip()}
    man = {r["stem"]: r for r in json.loads((a.built / "manifest.json").read_text(encoding="utf-8"))["records"] if r.get("status") == "ok"}
    folds = load_folds(a.folds)
    fold_of = {g["sha256"]: int(g["fold"]) for g in folds["groups"]}
    fold_by_name = {}
    for r in man.values():
        if r["document_sha256"] in fold_of:
            fold_by_name.setdefault(r["filename"], fold_of[r["document_sha256"]])
    pages = [p for p in load_gt_pages(BE / "data" / "ai_takeoff.sqlite3") if p.mode == "hvac"]
    count = {}
    for p in pages:
        count[(p.sha256, p.page_number)] = count.get((p.sha256, p.page_number), 0) + 1
    chosen = set((folds.get("duplicate_choice") or {}).values())
    if a.duplicates == "latest":
        latest = {}
        for p in pages:
            k = (p.sha256, p.page_number)
            if count[k] > 1 and (k not in latest or str(p.updated_at) > str(latest[k].updated_at)):
                latest[k] = p
        chosen |= {p.stem for p in latest.values()}
    pages = pages + (pseudo_pages(pages, fold_of, fold_by_name, val_folds) if a.pseudo else [])
    rows = []
    for split in ("train", "val"):
        (a.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        if a.vectors:
            (a.out / "vectors" / split).mkdir(parents=True, exist_ok=True)
    for p in pages:
        if count.get((p.sha256, p.page_number), 1) > 1 and p.stem not in chosen:
            print("skip, two GT versions:", p.stem, p.filename)
            continue
        fold = fold_of.get(p.sha256, fold_by_name.get(p.filename, -1))  # a re-uploaded PDF keeps the fold of its old copy
        path = payload_of(p, a.payloads)
        if path is None:
            print("skip, no payload:", p.stem)
            continue
        payload = json.load(gzip.open(path, "rt", encoding="utf-8"))
        shapes = payload.get("shapes") or []
        meta = payload.get("document_metadata") or {}
        width, height = float(meta.get("page_width") or p.page_width), float(meta.get("page_height") or p.page_height)
        if p.stem in man:
            t = man[p.stem].get("gt_transform_applied") or IDENTITY
        else:
            segs = page_segments(shapes)
            gt0 = gt_geometry("hvac", p.objects)
            cal = calibrate(STRtree(segment_lines(segs.seg)), gt0.vertices, gt0.closed, width, height)
            if not cal.get("conclusive"):
                print("skip, GT does not land on the strokes:", p.stem, cal.get("reason"))
                continue
            t = cal["chosen"]
        split = "val" if fold in val_folds else "train"
        img = render(shapes, width, height, a.scale)
        h, w = img.shape[:2]
        bx = boxes(p.objects, t, width, height, keep_objects=True)
        if a.vectors:
            np.savez_compressed(a.out / "vectors" / split / f"{p.stem}.npz", tokens=vector_tokens(shapes, bx, a.scale))
        bx = [(c, q) for c, q, _ in bx]
        cv2.imwrite(str(a.out / "images" / split / f"{p.stem}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        with open(a.out / "labels" / split / f"{p.stem}.txt", "w", encoding="utf-8") as f:
            for c, q in bx:
                xy = q * a.scale / np.array([w, h])
                f.write(f"{c} " + " ".join(f"{v:.6f}" for v in np.clip(xy, 0, 1).ravel()) + "\n")
        rows.append({"pseudo": bool(getattr(p, "pseudo", False)), "stem": p.stem, "filename": p.filename, "document_id": p.document_id, "page_number": p.page_number, "fold": fold, "split": split,
                     "payload": str(path), "transform": t, "scale": a.scale, "size_pt": [width, height], "size_px": [w, h], "objects": len(bx)})
        print(f"{split:5s} fold {fold:2d} {p.stem} {p.filename[:30]} {w}x{h}px {len(bx)} objects", flush=True)
    (a.out / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)), encoding="utf-8")
    (a.out / "pages.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for split in ("train", "val"):
        r = [x for x in rows if x["split"] == split]
        print(f"{split}: {len(r)} pages, {sum(x['objects'] for x in r)} objects")
    return 0


if __name__ == "__main__":
    sys.exit(main())
