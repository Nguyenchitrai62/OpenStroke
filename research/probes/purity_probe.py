"""Measure: shape structure (items per shape, dashes) and how pure a true/other label is per shape vs per item."""
import collections
import glob
import gzip
import json
import re
import sqlite3
import sys

from shapely import STRtree
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

DB = "data/ai_takeoff.sqlite3"
FIRE_TRUE = {"Line"}
HVAC_TRUE_RE = re.compile(r"^(Duct|Elbow|Concentric Reducer|Eccentric Reducer|Chân rẽ)")
TOL = 2.0  # PDF points


def latest_payload(doc, page):
    best = None
    for f in glob.glob(f"data/drawings/document-{doc:06d}/page-{page:04d}.v*.json.gz"):
        if ".view." in f:
            continue
        v = int(re.search(r"\.v(\d+)\.json", f).group(1))
        if best is None or v > best[0]:
            best = (v, f)
    return json.load(gzip.open(best[1]))


def segments(item):
    op = item[0]
    if op == "l":
        return [(item[1], item[2])]
    if op == "c":
        return [(item[1], item[4])]
    if op == "re":
        x0, y0, x1, y1 = item[1][:4] if len(item[1]) == 4 else (*item[1][0], *item[1][1])
        c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return [(c[k], c[(k + 1) % 4]) for k in range(4)]
    if op == "qu":
        c = item[1]
        return [(c[k], c[(k + 1) % 4]) for k in range(4)]
    return []


def main(doc, page):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    mode, pj = con.execute("select mode,pages_json from ground_truths where document_id=?", (doc,)).fetchone()
    objs = [o for p in json.loads(pj) if p["page_number"] == page for o in p.get("objects") or []
            if o.get("group") in ("primary", "reference")]
    geoms = []
    for o in objs:
        name = o.get("shape_name") or ""
        v = o.get("vertices") or []
        if mode == "fire" and name in FIRE_TRUE and len(v) >= 2:
            geoms.append(LineString(v).buffer(TOL))
        elif mode == "hvac" and HVAC_TRUE_RE.match(name) and len(v) >= 3:
            geoms.append(Polygon(v).buffer(TOL))
    zone = unary_union(geoms)
    pay = latest_payload(doc, page)
    rot = pay["document_metadata"].get("rotation")
    shapes = pay["shapes"]
    # label every segment
    seg_true = []  # per shape: list of (len, is_true)
    all_lines = []
    owner = []
    for si, s in enumerate(shapes):
        for it in s.get("items") or []:
            for a, b in segments(it):
                ls = LineString([a, b])
                all_lines.append(ls)
                owner.append(si)
    tree = STRtree(all_lines)
    cand = set(tree.query(zone, predicate="intersects").tolist()) if not zone.is_empty else set()
    per_shape = collections.defaultdict(lambda: [0.0, 0.0])  # true_len, total_len
    for i, ls in enumerate(all_lines):
        L = max(ls.length, 1e-6)
        t = 0.0
        if i in cand:
            t = ls.intersection(zone).length / L
        is_true = t >= 0.5
        per_shape[owner[i]][0] += L if is_true else 0.0
        per_shape[owner[i]][1] += L
    n_true_shape = mixed = 0
    mixed_len = tot_len = true_len = 0.0
    for si, (tl, al) in per_shape.items():
        tot_len += al
        true_len += tl
        frac = tl / al if al else 0
        if frac >= 0.5:
            n_true_shape += 1
        if 0.1 < frac < 0.9:
            mixed += 1
            mixed_len += min(tl, al - tl)  # length that a whole-shape label gets wrong
    one = sum(1 for s in shapes if len(s.get("items") or []) == 1)
    dashed = sum(1 for s in shapes if str(s.get("dashes")) not in ("[] 0", "None", ""))
    layers = len({s.get("layer") for s in shapes})
    print(f"doc{doc} {mode} rot={rot} shapes={len(shapes)} 1-item={one / len(shapes):.1%} dashed-attr={dashed} "
          f"layers={layers} gt_true_objs={len(geoms)}")
    print(f"   true shapes={n_true_shape} ({n_true_shape / len(shapes):.1%}), true length={true_len / tot_len:.1%}; "
          f"mixed shapes={mixed}, length mislabelled by whole-shape label={mixed_len / max(true_len, 1e-6):.2%} "
          f"of true length")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        d, p = arg.split(":")
        main(int(d), int(p))
