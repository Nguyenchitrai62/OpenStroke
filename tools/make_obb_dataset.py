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
import hashlib
import json
import re
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
        fill = s.get("fill")
        if fill and s.get("type") in ("f", "fs") and b - a >= 2:
            # one ring per subpath: a title-block border is an outer and an inner rectangle in one fill, and one polygon through
            # both painted the whole sheet black (22 of 51 pages before E68). fillPoly with several rings keeps the holes.
            q = segs.seg[a:b] * scale
            cut = np.flatnonzero(np.hypot(*(q[1:, :2] - q[:-1, 2:]).T) > 0.5) + 1
            rings = [np.round(np.vstack([r[:, :2], r[-1:, 2:]])).astype(np.int32) for r in np.split(q, cut) if len(r) >= 2]
            if rings:
                cv2.fillPoly(img, rings, tuple(int(255 * c) for c in fill[:3][::-1]), cv2.LINE_AA)
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


def restyle(shapes: list[dict], rng: np.random.Generator) -> list[dict]:
    """E76: the same page in another drawing style. Every style group (stroke colour + width) gets one random colour (dark enough to read on
    white) and a width x0.5-2; every non-white fill colour another random light-to-mid colour. Geometry and labels are untouched: a duct must
    be recognised by its shape, not by the colour or pen one project happened to use."""

    def colour(lo, hi):
        if rng.random() < 0.4:  # grey
            v = float(rng.uniform(lo, hi))
            return [v, v, v]
        c = rng.uniform(lo, hi, 3)
        c[rng.integers(3)] = rng.uniform(0, lo) if lo > 0 else 0.0
        return c.tolist()

    strokes, fills, out = {}, {}, []
    for sh in shapes:
        k = (tuple(np.round(sh.get("color") or (0, 0, 0), 2)), round(float(sh.get("width") or 0), 2))
        if k not in strokes:
            strokes[k] = (colour(0.0, 0.55), float(rng.uniform(0.5, 2.0)))
        c, wm = strokes[k]
        new = dict(sh, color=c, width=float(sh.get("width") or 0) * wm)
        if sh.get("fill"):
            f = tuple(np.round(sh["fill"][:3], 2))
            if f not in fills:
                fills[f] = list(f) if min(f) >= 0.95 else colour(0.35, 0.95)  # white wipeouts stay white
            new["fill"] = fills[f]
        out.append(new)
    return out


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_CAP = 22.0  # cap height in px of FONT at scale 1


def ascii_text(t: str) -> str:
    """Vietnamese and symbols to the ASCII that FONT has: diacritics dropped, Đ -> D, Ø / Φ -> D (a round size), the rest -> ?."""

    import unicodedata

    t = t.replace("Đ", "D").replace("đ", "d")
    t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    return "".join(c if ord(c) < 128 else ("D" if c in "ØøΦφ∅" else "?") for c in t)


def drawn_already(shapes: list[dict], spans: list[dict]) -> np.ndarray:
    """(n_spans,) True where the page already draws that text as strokes (CAD text exported as glyph outlines): drawing it again doubles it."""

    if not spans:
        return np.zeros(0, bool)
    segs = page_segments(shapes)
    short = segs.length < 8
    tree = STRtree(shapely.linestrings(segs.seg[short].reshape(-1, 2, 2)))
    boxes_ = shapely.box(*np.array([sp.get("b") or [*sp["o"], *sp["o"]] for sp in spans], float).T)
    return np.array([len(tree.query(b)) >= 3 for b in boxes_])


def draw_text(img: np.ndarray, spans: list[dict], scale: float) -> np.ndarray:
    """E67: the PDF's own text (``tools/page_text.py``) drawn in black at its baseline, size and reading direction: size labels (400x250,
    Ø200) are what tells a round branch or a size change apart, and a page drawn from its strokes alone has none of them."""

    h, w = img.shape[:2]
    for sp in spans:
        t = ascii_text(sp["t"])
        px = float(sp.get("s") or 0) * scale
        if px < 4 or not t.strip():
            continue
        dx, dy = sp["d"]
        fs = 0.6 * px / FONT_CAP  # the box holds ascent + descent; caps fill ~60 % of it
        if sp.get("b"):  # and the text must not run past its box along the reading direction
            bx0, by0, bx1, by1 = sp["b"]
            run = (abs((bx1 - bx0) * dx) + abs((by1 - by0) * dy)) * scale
            w1 = cv2.getTextSize(t, FONT, 1.0, 1)[0][0]
            fs = min(fs, 0.95 * run / max(w1, 1))
        th = max(1, int(round(fs * 1.6)))
        (tw, tht), base = cv2.getTextSize(t, FONT, fs, th)
        ox, oy = sp["o"][0] * scale, sp["o"][1] * scale
        if abs(dy) < 1e-3 and dx > 0:
            cv2.putText(img, t, (int(round(ox)), int(round(oy))), FONT, fs, (0, 0, 0), th, cv2.LINE_AA)
            continue
        pad = th + 2
        patch = np.full((tht + base + 2 * pad, tw + 2 * pad), 255, np.uint8)
        cv2.putText(patch, t, (pad, pad + tht), FONT, fs, 0, th, cv2.LINE_AA)
        # patch (u, v) -> page: origin + u' * d + v' * n, with (u', v') relative to the baseline start (pad, pad + tht), n = d rotated +90
        nx, ny = -dy, dx
        m = np.array([[dx, nx, ox - dx * pad - nx * (pad + tht)], [dy, ny, oy - dy * pad - ny * (pad + tht)]], np.float32)
        corners = np.array([[0, 0, 1], [patch.shape[1], 0, 1], [0, patch.shape[0], 1], [patch.shape[1], patch.shape[0], 1]], np.float32) @ m.T
        x0, y0 = np.floor(corners.min(0)).astype(int)
        x1, y1 = np.ceil(corners.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        m2 = m.copy()
        m2[:, 2] -= (x0, y0)
        roi = cv2.warpAffine(patch, m2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=255)
        img[y0:y1, x0:x1] = np.minimum(img[y0:y1, x0:x1], roi[:, :, None])
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


class TeacherPage(PseudoPage):
    """E65: a corpus page labelled by the stroke model's layer choice + AI_Takeoff's HVAC logic (``tools/teacher_*.py``); payload frame."""

    def __init__(self, row, objects, payload, width, height):
        super().__init__(-1, Path(row["pdf"]).name, row["sha256"], row["page"], width, height, objects, 0)
        self.stem, self.payload_path, self.teacher = f"t_{row['key']}", payload, True


def fingerprint(shapes: list[dict]) -> set:
    """Endpoints (2 pt grid) of the page's strokes of 30 pt or more: a copy of a page shares most of them."""

    segs = page_segments(shapes)
    long = segs.length >= 30
    q = np.round(segs.seg[long].reshape(-1, 2) / 2).astype(np.int64)
    return set(map(tuple, q))


def teacher_pages(corpus: Path, guard: list[set], min_objects: int, max_overlap: float = 0.3) -> list:
    rows = {r["key"]: r for r in json.loads((corpus / "pages.json").read_text(encoding="utf-8"))}
    out = []
    for f in sorted((corpus / "objects").glob("*.json")):
        k = f.stem
        objs = [o for o in json.loads(f.read_text(encoding="utf-8"))["objects"] if class_of(o.get("shape_name")) and len(o.get("vertices") or []) >= 3]
        if len(objs) < min_objects:
            print("skip, teacher found", len(objs), "objects:", k)
            continue
        pay = corpus / "pages" / f"{k}.json.gz"
        payload = json.load(gzip.open(pay, "rt", encoding="utf-8"))
        fp = fingerprint(payload.get("shapes") or [])
        hit = max((len(fp & g) / max(min(len(fp), len(g)), 1) for g in guard), default=0.0)
        if hit > max_overlap:
            print(f"skip, shares {hit:.2f} of its long strokes with a GT / val page:", k)
            continue
        meta = payload.get("document_metadata") or {}
        out.append(TeacherPage(rows[k], objs, pay, float(meta.get("page_width") or 0), float(meta.get("page_height") or 0)))
    print("teacher pages", len(out))
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
    ap.add_argument("--text", type=Path, default=None, help="E67: dir of text spans per page (tools/page_text.py), drawn into the image")
    ap.add_argument("--teacher", type=Path, default=None, help="E65: corpus dir with teacher objects (tools/teacher_logic.py), train only")
    ap.add_argument("--exclude-train", default="", help="regex on the PDF file name: train pages to leave out (same project as a val PDF)")
    ap.add_argument("--restyle", type=int, default=0, help="E76: extra renders of every train page in random drawing styles")
    ap.add_argument("--teacher-min", type=int, default=15, help="drop teacher pages with fewer objects (not a duct plan, or the teacher failed)")
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
    if a.teacher:
        guard = []
        for p in pages:
            if getattr(p, "pseudo", False):
                continue
            q = payload_of(p, a.payloads)
            if q is not None:
                guard.append(fingerprint(json.load(gzip.open(q, "rt", encoding="utf-8")).get("shapes") or []))
        pages = pages + teacher_pages(a.teacher, guard, a.teacher_min)
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
        path = getattr(p, "payload_path", None) or payload_of(p, a.payloads)
        if path is None:
            print("skip, no payload:", p.stem)
            continue
        payload = json.load(gzip.open(path, "rt", encoding="utf-8"))
        shapes = payload.get("shapes") or []
        meta = payload.get("document_metadata") or {}
        width, height = float(meta.get("page_width") or p.page_width), float(meta.get("page_height") or p.page_height)
        if getattr(p, "teacher", False):
            t = IDENTITY  # the logic answered in the payload's own frame
        elif p.stem in man:
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
        if split == "train" and a.exclude_train and re.search(a.exclude_train, p.filename or ""):
            print("skip, same project as a val PDF:", p.stem, p.filename)
            continue
        img = render(shapes, width, height, a.scale)
        if a.text:
            tf = a.text / f"{p.stem}.json"
            if tf.exists():
                spans = json.loads(tf.read_text(encoding="utf-8"))
                spans = [sp for sp, d in zip(spans, drawn_already(shapes, spans)) if not d]
                img = draw_text(img, spans, a.scale)
            else:
                print("no text file:", p.stem)
        h, w = img.shape[:2]
        bx = boxes(p.objects, t, width, height, keep_objects=True)
        if a.vectors:
            np.savez_compressed(a.out / "vectors" / split / f"{p.stem}.npz", tokens=vector_tokens(shapes, bx, a.scale))
        bx = [(c, q) for c, q, _ in bx]
        cv2.imwrite(str(a.out / "images" / split / f"{p.stem}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        lines = []
        for c, q in bx:
            xy = q * a.scale / np.array([w, h])
            lines.append(f"{c} " + " ".join(f"{v:.6f}" for v in np.clip(xy, 0, 1).ravel()) + "\n")
        (a.out / "labels" / split / f"{p.stem}.txt").write_text("".join(lines), encoding="utf-8")
        if split == "train" and a.restyle:
            rng = np.random.default_rng(int(hashlib.sha256(p.stem.encode()).hexdigest()[:8], 16))
            for r in range(a.restyle):
                cv2.imwrite(str(a.out / "images" / split / f"{p.stem}_rs{r}.png"), render(restyle(shapes, rng), width, height, a.scale),
                            [cv2.IMWRITE_PNG_COMPRESSION, 6])
                (a.out / "labels" / split / f"{p.stem}_rs{r}.txt").write_text("".join(lines), encoding="utf-8")
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
