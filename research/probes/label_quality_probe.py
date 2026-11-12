"""Label quality of a set of HVAC pages, the same yardstick for the Viewer "Labels V1" boxes and the AI_Takeoff GT (F135). 0 GPU, no model.

Per page, from the page's own vectors (payload) and its objects (6 product classes):
  - objects, class counts; degenerate (area < 0.5 pt^2 or short side < 0.25 pt); duplicates (same class, IoU >= 0.9);
    conflicts (different classes, IoU >= 0.5);
  - stroke support: objects with no member stroke under the training label rule (relabel.object_strokes + objects.assign_objects, as the
    dataset builder does) -> the model can never return them;
  - completeness proxy: "object styles" = (stroke colour, width) groups holding >= 10 % of the page's member-stroke length; the share of their
    long strokes (>= 10 pt) lying outside every object polygon (+2 pt). Ducts drawn in that style but left without a box raise it; walls
    drawn in the duct style raise it too, so it is compared between sources, not read as an absolute.

    py -3.12 research/probes/label_quality_probe.py viewer <labels.json> <payload dir> [--per-doc N] [--limit N] [--out json]
    py -3.12 research/probes/label_quality_probe.py gt <dataset pages.json> [--out json]
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from openstroke.build import shape_attributes  # noqa: E402
from openstroke.geometry import IDENTITY, calibrate, page_segments, transform_points  # noqa: E402
from openstroke.labels import segment_lines  # noqa: E402
from openstroke.objects import HVAC_CLASSES, assign_objects, base6  # noqa: E402
from openstroke.relabel import object_strokes  # noqa: E402

VIEWER = {"duct": "Duct", "elbow": "Elbow", "elbow square": "Elbow vuông", "reducer": "Concentric Reducer", "branch": "Chân rẽ", "tee": "Tee"}
DB = "file:F:/Source_code/AI_Takeoff/BE/data/ai_takeoff.sqlite3?mode=ro"


def poly(v):
    v = np.asarray(v, float)
    if len(v) < 3:
        return None
    p = Polygon(v)
    return p if p.is_valid else p.buffer(0)


def short_side(p):
    c = np.asarray(p.minimum_rotated_rectangle.exterior.coords)[:4]
    if len(c) < 3:
        return 0.0
    e = np.hypot(*(np.roll(c, -1, 0) - c).T)
    return float(e.min())


def page_quality(shapes, objects, width, height, transform=None):
    """objects: [{"shape_name", "vertices"}] in pt (display top-left). -> dict of counts, or None if the boxes do not land on the strokes."""
    segs = page_segments(shapes)
    if transform is None:
        cal = calibrate(STRtree(segment_lines(segs.seg)), [o["vertices"] for o in objects], [True] * len(objects), width, height)
        if not cal.get("conclusive"):
            return None
        transform = cal["chosen"]
    if transform != IDENTITY:
        objects = [dict(o, vertices=transform_points(transform, np.asarray(o["vertices"], float), width, height).tolist()) for o in objects]
    polys = [poly(o["vertices"]) for o in objects]
    cls = [o["shape_name"] for o in objects]
    r = Counter(objects=len(objects))
    r.update(f"class {c}" for c in cls)
    ok = []
    for i, p in enumerate(polys):
        if p is None or p.area < 0.5 or short_side(p) < 0.25:
            r["degenerate"] += 1
        else:
            ok.append(i)
    tree = STRtree([polys[i] for i in ok])
    for a, i in enumerate(ok):
        for b in tree.query(polys[i]):
            j = ok[int(b)]
            if j <= i:
                continue
            inter = polys[i].intersection(polys[j]).area
            iou = inter / max(polys[i].union(polys[j]).area, 1e-9)
            if cls[i] == cls[j] and iou >= 0.9:
                r["duplicate"] += 1
            elif cls[i] != cls[j] and iou >= 0.5:
                r["conflict"] += 1
    n = len(shapes)
    attrs = shape_attributes(shapes, segs)
    length = np.bincount(segs.owner, weights=segs.length, minlength=n)
    objs = [objects[i] for i in ok]
    mask = object_strokes("hvac", objs, segs, attrs, length) if objs else np.zeros(n, bool)
    _, inst, _ = assign_objects(segs, objs, mask)
    have = set(np.unique(inst[inst >= 0]).tolist())
    r["no_stroke"] = len(objs) - len(have) + r["degenerate"]
    r["member_len"] = float(length[inst >= 0].sum())
    # completeness proxy
    rgb = np.round(np.nan_to_num(attrs["stroke_rgb"], nan=0.0), 2)
    w = np.round(np.nan_to_num(attrs["width"], nan=0.0), 2)
    key = np.unique(np.c_[rgb, w], axis=0, return_inverse=True)[1].ravel()
    mem_by = np.bincount(key, weights=np.where(inst >= 0, length, 0.0))
    styles = np.flatnonzero(mem_by >= 0.1 * max(mem_by.sum(), 1e-9))
    cand = np.isin(key, styles) & (length >= 10.0)
    if cand.any() and objs:
        area = shapely.union_all([polys[i].buffer(2.0) for i in ok])
        lines = segment_lines(segs.seg)
        own = segs.owner
        sel = np.flatnonzero(cand[own])
        inside = shapely.contains(area, lines[sel]) if len(sel) else np.zeros(0, bool)
        tot = float(segs.length[sel].sum())
        r["style_len"] = tot
        r["style_out_len"] = float(segs.length[sel][~inside].sum())
    return r


def viewer_pages(labels, payloads, limit, per_doc=0):
    lab = json.loads(Path(labels).read_text(encoding="utf-8"))
    out = []
    for p in lab["pdfs"]:
        for pg in p["pages"]:
            objs = [o for o in ((pg["labels"].get("Labels V1") or {}).get("objects") or []) if o["class"] in VIEWER]
            if len(objs) < 20:
                continue
            f = Path(payloads) / f"{p['sha256'][:12]}_p{int(pg['page']):04d}.json.gz"
            if not f.exists():
                continue
            doc = re.sub(r"__pages_\d+-\d+", "", p["name"]).rsplit(".", 1)[0]
            out.append((f"{doc} p{pg['page']}", f, [{"shape_name": VIEWER[o["class"]], "vertices": o["points"]} for o in objs], None,
                        pg.get("width_pt"), pg.get("height_pt")))
    if per_doc:  # the most-labelled pages of every document
        by = {}
        for x in sorted(out, key=lambda x: -len(x[2])):
            by.setdefault(x[0].rsplit(" p", 1)[0], []).append(x)
        out = [x for v in by.values() for x in v[:per_doc]]
    return out[:limit] if limit else out


def gt_pages(pages_json):
    con = sqlite3.connect(DB, uri=True)
    out = []
    for r in json.loads(Path(pages_json).read_text(encoding="utf-8")):
        gid = int(r["stem"].rsplit("_gt", 1)[1])
        pj = json.loads(con.execute("select pages_json from ground_truths where id = ?", (gid,)).fetchone()[0])
        pg = next(x for x in pj if int(x.get("page_number") or 1) == int(r["page_number"]))
        objs = []
        for o in pg.get("objects") or []:
            b = base6(o.get("shape_name")) if o.get("group") in ("primary", "reference") else None
            if b is not None:
                objs.append({"shape_name": HVAC_CLASSES[b], "vertices": o.get("vertices")})
        out.append((r["stem"], ROOT.parent / "OpenStroke" / r["payload"].replace("\\", "/"), objs, r.get("transform") or IDENTITY,
                    r["size_pt"][0], r["size_pt"][1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["viewer", "gt"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-doc", type=int, default=0, help="viewer: at most this many pages per document (the most-labelled)")
    ap.add_argument("--out")
    a = ap.parse_args()
    pages = viewer_pages(a.args[0], a.args[1], a.limit, a.per_doc) if a.source == "viewer" else gt_pages(a.args[0])
    rows, tot = [], Counter()
    for name, f, objs, t, w, h in pages:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            pay = json.load(fh)
        meta = pay.get("document_metadata") or {}
        w, h = float(meta.get("page_width") or w), float(meta.get("page_height") or h)
        try:
            r = page_quality(pay.get("shapes") or [], objs, w, h, t)
        except MemoryError:
            tot["pages out of memory"] += 1
            print(f"{name}: out of memory, skipped", flush=True)
            continue
        if r is None:
            tot["pages not calibrated"] += 1
            print(f"{name}: boxes do not land on the strokes", flush=True)
            continue
        tot.update(r)
        tot["pages"] += 1
        so = r["style_out_len"] / max(r["style_len"], 1e-9)
        rows.append(dict(page=name, **{k: v for k, v in r.items()}))
        print(f"{name[:60]:60s} obj {r['objects']:5d} | no stroke {r['no_stroke'] / max(r['objects'], 1):5.1%} | dup {r['duplicate']:3d} "
              f"| conflict {r['conflict']:3d} | degen {r['degenerate']:3d} | object-style length outside boxes {so:5.1%}", flush=True)
    n = max(tot["objects"], 1)
    print(f"\nTOTAL {a.source}: pages {tot['pages']} (+{tot['pages not calibrated']} not calibrated), objects {tot['objects']}")
    print(f"  no member stroke {tot['no_stroke'] / n:.1%} | duplicates {tot['duplicate'] / n:.2%} | conflicts {tot['conflict'] / n:.2%} | "
          f"degenerate {tot['degenerate'] / n:.2%} | object-style length outside boxes {tot['style_out_len'] / max(tot['style_len'], 1e-9):.1%}")
    print("  classes: " + ", ".join(f"{k[6:]} {v}" for k, v in sorted(tot.items()) if k.startswith("class ")))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(total=tot, pages=rows), ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
