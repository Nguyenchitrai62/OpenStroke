"""OpenStroke15: an OpenOBB25 dataset (page images + vector tokens + oriented boxes) from the Viewer HVAC project #19 "Labels V1" objects
(docs/OPENSTROKE15.md). No layers anywhere: layers are neither drawn nor tokenised; the objects come from the Viewer boxes.

Two steps, two Pythons:

    <AI_Takeoff venv python> tools/viewer_obb_dataset.py extract <export dir> [<export dir> ...] --out datasets/viewer_hvac19/payloads
        every page with >= --min-labels "Labels V1" objects -> the same split_countfire payload AI_Takeoff builds (make_payloads.build)
    py -3.12 tools/viewer_obb_dataset.py build <export dir> [...] --payloads datasets/viewer_hvac19/payloads --out datasets/viewer_obb1
        render (4 px / pt), vector tokens with the object of every stroke (label rule), DOTA labels, pages.json, data.yaml

Class names follow this repo's (so a fine-tune onto the 9 GT classes carries them over by name): duct -> Duct, elbow -> Elbow,
elbow square -> Elbow vuông, reducer -> Concentric Reducer, branch -> Chân rẽ, tee -> Tee. "ống kèm máy", Z and cross are left out.
Guards: documents of the same project as a val page are excluded (``--exclude``); a page sharing > 30 % of its long-stroke endpoints with
a val page is dropped (``fingerprint``); a page whose boxes do not land on its strokes is dropped (``calibrate``). Whole documents are held out
as this dataset's own val split (``--val-docs``). AI_Takeoff and the Viewer are only read.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

CLASS_MAP = {"duct": "Duct", "elbow": "Elbow", "elbow square": "Elbow vuông", "reducer": "Concentric Reducer", "branch": "Chân rẽ", "tee": "Tee"}
NAMES = ["Duct", "Elbow", "Elbow vuông", "Concentric Reducer", "Chân rẽ", "Tee"]
EXCLUDE = (r"DC4-MEP-MV", r"04\. MECHANICAL")  # same projects as the val pages DC4-MEP-MV [40].pdf and 04. MECHANICAL [58].pdf
SET = "Labels V1"


def doc_of(name: str) -> str:
    return re.sub(r"__pages_\d+-\d+", "", name).rsplit(".", 1)[0]


def pages_of(export_dirs, min_labels, exclude):
    """-> [(export dir, pdf record, page record)] with >= min_labels objects, outside the excluded documents."""
    out = []
    for d in export_dirs:
        d = Path(d)
        lab = json.loads((d / "labels.json").read_text(encoding="utf-8"))
        for p in lab["pdfs"]:
            if any(re.search(x, p["name"]) for x in exclude):
                continue
            for pg in p["pages"]:
                objs = (pg["labels"].get(SET) or {}).get("objects") or []
                if len([o for o in objs if o["class"] in CLASS_MAP]) >= min_labels:
                    out.append((d, p, pg))
    return out


def key_of(p, pg) -> str:
    return f"{p['sha256'][:12]}_p{int(pg['page']):04d}"


def _extract(job):
    d, pdf_file, page, out = job
    from make_payloads import build

    if Path(out).exists():
        return out, "kept"
    payload = build(str(Path(d) / pdf_file), int(page))
    tmp = Path(str(out) + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=1) as fh:
        json.dump(payload, fh)
    tmp.replace(out)
    return out, "built"


def extract(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(d), p["file"], pg["page"], str(out / f"{key_of(p, pg)}.json.gz")) for d, p, pg in pages_of(a.exports, a.min_labels, EXCLUDE)]
    print(f"{len(jobs)} pages to extract", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (f, how) in enumerate(ex.map(_extract, jobs), 1):
            if i % 20 == 0 or how == "built":
                print(f"[{i}/{len(jobs)}] {how} {Path(f).name}", flush=True)
    return 0


def build(a):
    import cv2
    import numpy as np
    from shapely import STRtree

    from make_obb_dataset import IDENTITY, boxes, calibrate, fingerprint, page_segments, render, segment_lines, transform_points, vector_tokens

    out = Path(a.out)
    for s in ("train", "val"):
        for f in ("images", "labels", "vectors"):
            (out / f / s).mkdir(parents=True, exist_ok=True)
    guard = []  # fingerprints of the GT val pages (leak check)
    for r in json.loads((ROOT / a.val_pages).read_text(encoding="utf-8")):
        if r["split"] == "val":
            with gzip.open(ROOT / r["payload"].replace("\\", "/"), "rt", encoding="utf-8") as fh:
                guard.append(fingerprint(json.load(fh).get("shapes") or []))
    val_docs = [re.compile(x) for x in a.val_docs]
    rows, skipped = [], {"no payload": 0, "leak": 0, "calibration": 0, "no boxes": 0}
    for d, p, pg in pages_of(a.exports, a.min_labels, EXCLUDE):
        stem = key_of(p, pg)
        f = Path(a.payloads) / f"{stem}.json.gz"
        if not f.exists():
            skipped["no payload"] += 1
            continue
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        shapes = payload.get("shapes") or []
        meta = payload.get("document_metadata") or {}
        width, height = float(meta.get("page_width") or pg["width_pt"]), float(meta.get("page_height") or pg["height_pt"])
        fp = fingerprint(shapes)
        if fp and any(len(fp & g) > 0.3 * max(len(fp), 1) for g in guard):
            skipped["leak"] += 1
            print("skip, shares strokes with a val page:", p["name"], pg["page"], flush=True)
            continue
        objects = [{"shape_name": CLASS_MAP[o["class"]], "vertices": o["points"]} for o in pg["labels"][SET]["objects"] if o["class"] in CLASS_MAP]
        segs = page_segments(shapes)
        cal = calibrate(STRtree(segment_lines(segs.seg)), [o["vertices"] for o in objects], [True] * len(objects), width, height)
        if not cal.get("conclusive"):
            skipped["calibration"] += 1
            print("skip, boxes do not land on the strokes:", p["name"], pg["page"], cal.get("reason"), flush=True)
            continue
        t = cal["chosen"]
        bx = boxes(objects, t, width, height, keep_objects=True)
        if not bx:
            skipped["no boxes"] += 1
            continue
        split = "val" if any(rx.search(p["name"]) for rx in val_docs) else "train"
        img = render(shapes, width, height, a.scale)
        h, w = img.shape[:2]
        np.savez_compressed(out / "vectors" / split / f"{stem}.npz", tokens=vector_tokens(shapes, bx, a.scale))
        cv2.imwrite(str(out / "images" / split / f"{stem}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        lines = []
        for c, q, _ in bx:
            k = NAMES.index(CLASS_MAP_INV[c])
            xy = q * a.scale / np.array([w, h])
            lines.append(f"{k} " + " ".join(f"{v:.6f}" for v in np.clip(xy, 0, 1).ravel()) + "\n")
        (out / "labels" / split / f"{stem}.txt").write_text("".join(lines), encoding="utf-8")
        rows.append({"stem": stem, "filename": p["name"], "document": doc_of(p["name"]), "page_number": int(pg["page"]), "split": split,
                     "payload": str(f), "transform": t, "scale": a.scale, "size_pt": [width, height], "size_px": [w, h], "objects": len(bx),
                     "labels_status": pg["labels"][SET].get("status")})
        print(f"{split:5s} {stem} {p['name'][:40]} p{pg['page']} {len(bx)} objects", flush=True)
    (out / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)), encoding="utf-8")
    (out / "pages.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for s in ("train", "val"):
        r = [x for x in rows if x["split"] == s]
        print(f"{s}: {len(r)} pages, {sum(x['objects'] for x in r)} objects, {len({x['document'] for x in r})} documents")
    print("skipped:", skipped)
    return 0


def _class_map_inv():
    from openstroke.objects import class_of

    return {class_of(n) - 1: n for n in NAMES}


CLASS_MAP_INV = {}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("exports", nargs="+")
    e.add_argument("--out", default="datasets/viewer_hvac19/payloads")
    e.add_argument("--min-labels", type=int, default=20)
    e.add_argument("--workers", type=int, default=4)
    b = sub.add_parser("build")
    b.add_argument("exports", nargs="+")
    b.add_argument("--payloads", default="datasets/viewer_hvac19/payloads")
    b.add_argument("--out", default="datasets/viewer_obb1")
    b.add_argument("--min-labels", type=int, default=20)
    b.add_argument("--scale", type=float, default=4.0)
    b.add_argument("--val-pages", default="datasets/obb_hvac7/pages.json", help="GT val pages for the leak check")
    b.add_argument("--val-docs", nargs="*", default=["^VIAS TOWER"],
                   help="Viewer documents held out as this dataset's val (one small project: full-page val runs every epoch)")
    a = ap.parse_args()
    if a.cmd == "build":
        CLASS_MAP_INV.update(_class_map_inv())
        return build(a)
    return extract(a)


if __name__ == "__main__":
    sys.exit(main())
