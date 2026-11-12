"""OpenStroke15: an OpenOBB25 dataset (page images + vector tokens + oriented boxes) from the Viewer HVAC project #19 "Labels V1" objects
(docs/OPENSTROKE15.md). No layers anywhere: layers are neither drawn nor tokenised; the objects come from the Viewer boxes.

Two steps, two Pythons:

    <AI_Takeoff venv python> tools/viewer_obb_dataset.py extract <export dir> [<export dir> ...] --out datasets/viewer_hvac19/payloads
        every page with >= --min-labels "Labels V1" objects -> the same split_countfire payload AI_Takeoff builds (make_payloads.build)
    py -3.12 tools/viewer_obb_dataset.py build <export dir> [...] --labels <fresh labels.json> --payloads datasets/viewer_hvac19/payloads
        --gt datasets/obb_hvac8r --out datasets/obb_hvac9v
        the GT dataset's pages linked unchanged (its val = the official GT val, the main F2) + Viewer pages added to train: render (4 px / pt),
        vector tokens with the object of every stroke (label rule), DOTA labels with the GT's class list, pages.json, gt_provenance.json

Class names are the GT's: duct -> Duct, elbow -> Elbow, elbow square -> Elbow vuông, reducer -> Reducer vuông tròn (the Viewer does not
record concentric / eccentric; the 6-class set and the scorer treat it as a reducer of either base), branch -> Chân rẽ, tee -> Tee.
"ống kèm máy", Z and cross are left out. Guards: documents of the same project as a GT page are excluded (EXCLUDE, ``--exclude``); a page
sharing > 30 % of its long-stroke endpoints with any GT page is dropped (``fingerprint``); a page whose boxes do not land on its strokes is
dropped (``calibrate``); a page with more than ``--max-bare`` of its objects without any stroke is dropped. At most ``--max-pages-per-doc``
pages per document (variety over volume). AI_Takeoff and the Viewer are only read.
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

CLASS_MAP = {"duct": "Duct", "elbow": "Elbow", "elbow square": "Elbow vuông", "reducer": "Reducer vuông tròn", "branch": "Chân rẽ", "tee": "Tee"}
EXCLUDE = (r"DC4-MEP-MV", r"04\. MECHANICAL")  # same projects as the val pages DC4-MEP-MV [40].pdf and 04. MECHANICAL [58].pdf
SET = "Labels V1"


def doc_of(name: str) -> str:
    return re.sub(r"__pages_\d+-\d+", "", name).rsplit(".", 1)[0]


def pages_of(export_dirs, min_labels, exclude, labels=None):
    """-> [(export dir, pdf record, page record)] with >= min_labels objects, outside the excluded documents. labels: one fresh
    labels.json (a labels-only export) read instead of each export's own; the export dir is then the one holding the PDF."""
    out = []
    sources = [(Path(labels), None)] if labels else [(Path(d) / "labels.json", Path(d)) for d in export_dirs]
    for lf, d in sources:
        lab = json.loads(lf.read_text(encoding="utf-8"))
        for p in lab["pdfs"]:
            if any(re.search(x, p["name"]) for x in exclude):
                continue
            here = d or next((Path(x) for x in export_dirs if (Path(x) / p["file"]).exists()), Path(export_dirs[0]))
            for pg in p["pages"]:
                objs = (pg["labels"].get(SET) or {}).get("objects") or []
                if len([o for o in objs if o["class"] in CLASS_MAP]) >= min_labels:
                    out.append((here, p, pg))
    return out


def capped(cand, cap):
    """At most `cap` pages per document, the most-labelled first (variety over volume)."""
    by_doc = {}
    for d, p, pg in cand:
        by_doc.setdefault(doc_of(p["name"]), []).append((d, p, pg))
    out = []
    for doc, items in sorted(by_doc.items()):
        items.sort(key=lambda x: -len((x[2]["labels"].get(SET) or {}).get("objects") or []))
        out += items[:cap] if cap else items
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
    jobs = [(str(d), p["file"], pg["page"], str(out / f"{key_of(p, pg)}.json.gz")) for d, p, pg in capped(pages_of(a.exports, a.min_labels, EXCLUDE, labels=a.labels), a.max_pages_per_doc)]
    print(f"{len(jobs)} pages to extract", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (f, how) in enumerate(ex.map(_extract, jobs), 1):
            if i % 20 == 0 or how == "built":
                print(f"[{i}/{len(jobs)}] {how} {Path(f).name}", flush=True)
    return 0


def build(a):
    """Viewer pages -> train pages of a combined dataset next to the GT one (--gt): the GT train / val files are linked unchanged
    (val stays the official GT val, the main F2), the Viewer pages are added to train with the GT's class list. F135."""
    import os
    import shutil

    import cv2
    import numpy as np
    import yaml
    from shapely import STRtree

    from make_obb_dataset import boxes, calibrate, fingerprint, page_segments, render, segment_lines, vector_tokens

    gt = Path(a.gt)
    out = Path(a.out)
    cfg = yaml.safe_load((gt / "data.yaml").read_text(encoding="utf-8"))
    names = cfg["names"]
    names = [names[k] for k in sorted(names, key=int)] if isinstance(names, dict) else list(names)
    for s in ("train", "val"):
        for f in ("images", "labels", "vectors"):
            (out / f / s).mkdir(parents=True, exist_ok=True)
    gt_rows = json.loads((gt / "pages.json").read_text(encoding="utf-8"))
    proof = json.loads((gt / "gt_provenance.json").read_text(encoding="utf-8"))
    for r in gt_rows:  # GT pages unchanged (hardlinks)
        for kind, suf in (("images", ".png"), ("labels", ".txt"), ("vectors", ".npz")):
            src, dst = gt / kind / r["split"] / (r["stem"] + suf), out / kind / r["split"] / (r["stem"] + suf)
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
    repo = gt.resolve().parents[1]
    guard = []  # every GT page (train and val): a Viewer page sharing its strokes is a duplicate or a leak
    for r in gt_rows:
        with gzip.open(repo / r["payload"].replace(chr(92), "/"), "rt", encoding="utf-8") as fh:
            guard.append(fingerprint(json.load(fh).get("shapes") or []))
    chosen = capped(pages_of(a.exports, a.min_labels, EXCLUDE + tuple(a.exclude), labels=a.labels), a.max_pages_per_doc)
    rows, skipped = [], {"no payload": 0, "leak / duplicate": 0, "calibration": 0, "no boxes": 0, "objects without strokes": 0}

    def one(d, p, pg, stem):
        f = Path(a.payloads) / f"{key_of(p, pg)}.json.gz"
        if not f.exists():
            return {"skip": "no payload"}
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        shapes = payload.get("shapes") or []
        meta = payload.get("document_metadata") or {}
        width, height = float(meta.get("page_width") or pg["width_pt"]), float(meta.get("page_height") or pg["height_pt"])
        fp = fingerprint(shapes)
        if fp and any(len(fp & g) > 0.3 * max(len(fp), 1) for g in guard):
            print("skip, shares strokes with a GT page:", p["name"], pg["page"], flush=True)
            return {"skip": "leak / duplicate"}
        objects = [{"shape_name": CLASS_MAP[o["class"]], "vertices": o["points"]} for o in pg["labels"][SET]["objects"] if o["class"] in CLASS_MAP]
        segs = page_segments(shapes)
        cal = calibrate(STRtree(segment_lines(segs.seg)), [o["vertices"] for o in objects], [True] * len(objects), width, height)
        if not cal.get("conclusive"):
            print("skip, boxes do not land on the strokes:", p["name"], pg["page"], cal.get("reason"), flush=True)
            return {"skip": "calibration"}
        t = cal["chosen"]
        bx = boxes(objects, t, width, height, keep_objects=True)
        if not bx:
            return {"skip": "no boxes"}
        tok = vector_tokens(shapes, bx, a.scale)
        inst = np.rint(tok[:, 22]).astype(np.int64)
        bare = len(bx) - len(np.unique(inst[inst >= 0]))
        if bare > a.max_bare * len(bx):
            print(f"skip, {bare} / {len(bx)} objects have no stroke:", p["name"], pg["page"], flush=True)
            return {"skip": "objects without strokes"}
        img = render(shapes, width, height, a.scale)
        h, w = img.shape[:2]
        np.savez_compressed(out / "vectors" / "train" / f"{stem}.npz", tokens=tok)
        cv2.imwrite(str(out / "images" / "train" / f"{stem}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        lines = []
        for c, q, o in bx:
            k = names.index(o["shape_name"])
            xy = q * a.scale / np.array([w, h])
            lines.append(f"{k} " + " ".join(f"{v:.6f}" for v in np.clip(xy, 0, 1).ravel()) + chr(10))
        (out / "labels" / "train" / f"{stem}.txt").write_text("".join(lines), encoding="utf-8")
        row = {"stem": stem, "source": "viewer_hvac19", "filename": p["name"], "document": doc_of(p["name"]),
                     "page_number": int(pg["page"]), "split": "train", "payload": str(f), "transform": t, "scale": a.scale,
                     "size_pt": [width, height], "size_px": [w, h], "objects": len(bx), "objects_without_strokes": int(bare),
                     "labels_status": pg["labels"][SET].get("status")}
        print(f"train {stem} {p['name'][:40]} p{pg['page']} {len(bx)} objects ({bare} without strokes)", flush=True)
        return {"row": row}

    import gc

    (out / "_rows").mkdir(exist_ok=True)
    for d, p, pg in chosen:
        stem = "vw_" + key_of(p, pg)
        side = out / "_rows" / f"{stem}.json"  # resumable: a page already built (or skipped) is not done again
        if side.exists():
            rec = json.loads(side.read_text(encoding="utf-8"))
        else:
            try:
                rec = one(d, p, pg, stem)
            except MemoryError:
                print("skip, out of memory:", p["name"], pg["page"], flush=True)
                rec = {"skip": "out of memory"}
            gc.collect()
            if rec.get("skip") != "out of memory":  # retried on the next run
                side.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        if "row" in rec:
            rows.append(rec["row"])
        else:
            skipped[rec["skip"]] = skipped.get(rec["skip"], 0) + 1
    (out / "data.yaml").write_text(yaml.safe_dump(dict(cfg, path=".", train="images/train", val="images/val"), allow_unicode=True,
                                                  sort_keys=False), encoding="utf-8")
    (out / "pages.json").write_text(json.dumps(gt_rows + rows, ensure_ascii=False, indent=1), encoding="utf-8")
    recs = list(proof["records"]) + [{"stem": r["stem"], "split": "train", "source": "viewer_hvac19", "filename": r["filename"],
                                      "page_number": r["page_number"], "labels_status": r["labels_status"]} for r in rows]
    (out / "gt_provenance.json").write_text(json.dumps(dict(proof, records=recs, viewer=dict(
        project="Viewer HVAC #19 Labels V1", labels=str(a.labels), class_map=CLASS_MAP, max_pages_per_doc=a.max_pages_per_doc,
        min_labels=a.min_labels, max_bare=a.max_bare, skipped=skipped)), ensure_ascii=False, indent=1), encoding="utf-8")
    docs = {r["document"] for r in rows}
    print(f"GT: {sum(r['split'] == 'train' for r in gt_rows)} train / {sum(r['split'] == 'val' for r in gt_rows)} val pages; "
          f"Viewer: {len(rows)} train pages, {sum(r['objects'] for r in rows)} objects, {len(docs)} documents")
    print("skipped:", skipped)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("exports", nargs="+")
    e.add_argument("--labels", help="one fresh labels.json instead of each export's own")
    e.add_argument("--max-pages-per-doc", type=int, default=12)
    e.add_argument("--out", default="datasets/viewer_hvac19/payloads")
    e.add_argument("--min-labels", type=int, default=20)
    e.add_argument("--workers", type=int, default=4)
    b = sub.add_parser("build")
    b.add_argument("exports", nargs="+")
    b.add_argument("--payloads", default="datasets/viewer_hvac19/payloads")
    b.add_argument("--out", default="datasets/obb_hvac9v")
    b.add_argument("--gt", default="datasets/obb_hvac8r", help="GT dataset: its pages are kept (val = official GT val), Viewer pages join train")
    b.add_argument("--labels", help="one fresh labels.json (labels-only export) instead of each export's own")
    b.add_argument("--max-pages-per-doc", type=int, default=12, help="most-labelled pages kept per document (variety over volume)")
    b.add_argument("--max-bare", type=float, default=0.25, help="drop a page when more than this share of its objects has no stroke")
    b.add_argument("--exclude", nargs="*", default=[], help="more document-name regexes to leave out")
    b.add_argument("--min-labels", type=int, default=20)
    b.add_argument("--scale", type=float, default=4.0)
    a = ap.parse_args()
    if a.cmd == "build":
        return build(a)
    return extract(a)


if __name__ == "__main__":
    sys.exit(main())
