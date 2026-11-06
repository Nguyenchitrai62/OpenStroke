"""E65: add the teacher-labelled corpus pages to an OpenOBB dataset as train pages (on the VM, next to the payloads: rendering there
saves uploading page images). Same drawing and labels as ``make_obb_dataset.py`` (vectors only, no layers); pages with fewer than
``--min`` teacher objects or sharing more than ``--overlap`` of their long strokes with a val page are left out.

    python tools/obb_teacher_pages.py <obb_dir> <corpus_dir> [--min 15] [--scale 4]
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_obb_dataset import IDENTITY, boxes, class_of, fingerprint, render  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("obb")
    ap.add_argument("corpus")
    ap.add_argument("--min", type=int, default=15)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--overlap", type=float, default=0.3)
    a = ap.parse_args()
    obb, c = Path(a.obb), Path(a.corpus)
    guard = []
    for f in sorted((c / "guard").glob("*.json.gz")):  # payloads of the val pages
        guard.append(fingerprint(json.load(gzip.open(f, "rt", encoding="utf-8")).get("shapes") or []))
    rows = json.loads((obb / "pages.json").read_text(encoding="utf-8"))
    rows = [r for r in rows if not r["stem"].startswith("t_")]
    n_obj = 0
    for f in sorted((c / "objects").glob("*.json")):
        k = f.stem
        objs = [o for o in json.loads(f.read_text(encoding="utf-8"))["objects"] if class_of(o.get("shape_name")) and len(o.get("vertices") or []) >= 3]
        if len(objs) < a.min:
            print("skip,", len(objs), "objects:", k)
            continue
        payload = json.load(gzip.open(c / "pages" / f"{k}.json.gz", "rt", encoding="utf-8"))
        shapes = payload.get("shapes") or []
        fp = fingerprint(shapes)
        hit = max((len(fp & g) / max(min(len(fp), len(g)), 1) for g in guard), default=0.0)
        if hit > a.overlap:
            print(f"skip, shares {hit:.2f} with a val page:", k)
            continue
        meta = payload.get("document_metadata") or {}
        w_pt, h_pt = float(meta.get("page_width") or 0), float(meta.get("page_height") or 0)
        img = render(shapes, w_pt, h_pt, a.scale)
        h, w = img.shape[:2]
        bx = boxes(objs, IDENTITY, w_pt, h_pt)
        stem = f"t_{k}"
        cv2.imwrite(str(obb / "images" / "train" / f"{stem}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        with open(obb / "labels" / "train" / f"{stem}.txt", "w", encoding="utf-8") as fh:
            for cl, q in bx:
                xy = q * a.scale / np.array([w, h])
                fh.write(f"{cl} " + " ".join(f"{v:.6f}" for v in np.clip(xy, 0, 1).ravel()) + "\n")
        rows.append({"pseudo": True, "teacher": True, "stem": stem, "filename": k, "fold": -1, "split": "train", "scale": a.scale,
                     "transform": IDENTITY, "size_pt": [w_pt, h_pt], "size_px": [w, h], "objects": len(bx)})
        n_obj += len(bx)
        print("train", stem, f"{w}x{h}px", len(bx), "objects", flush=True)
    (obb / "pages.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    t = [r for r in rows if r.get("teacher")]
    print(f"teacher pages {len(t)}, {n_obj} objects; train pages now {sum(r['split'] == 'train' for r in rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
