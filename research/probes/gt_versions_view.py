"""Two GT versions of one page, for the user to choose (AGENTS section 3): a local page with pan and zoom over the page's strokes. GT read-only.

    py -3.12 research/probes/gt_versions_view.py runs_colab/viz/gt333 doc000020_p0001_gt10 doc000153_p0001_gt103 PTN-PIPE
"""

import json
import sys
from pathlib import Path

import numpy as np
from shapely import LineString, MultiLineString, STRtree, linestrings
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.relabel import page_segments_of  # noqa: E402
from openstroke.sources import load_gt_pages  # noqa: E402

DB = r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3"
DATA = Path("datasets/e5")
TOL = 2.0  # pt, same tolerance as the label rule
CELL = 60.0  # pt, window of one jump button
JUMPS = 10


def lines_of(page):
    return [np.asarray(o["vertices"], float) for o in page.objects if o.get("shape_name") == "Line" and len(o.get("vertices") or []) >= 2]


def others_of(page):
    return [{"name": o["shape_name"], "v": np.round(np.asarray(o["vertices"], float), 2).tolist()} for o in page.objects if o.get("shape_name") != "Line" and len(o.get("vertices") or []) >= 2]


def shared_mask(lines, other):
    """Per line: is most of it within TOL of the other version's lines."""

    zone = unary_union(MultiLineString([LineString(v) for v in other]).buffer(TOL))
    out = []
    for v in lines:
        g = LineString(v)
        out.append(g.length == 0 or g.intersection(zone).length / g.length >= 0.5)
    return np.array(out)


def ink_share(lines, seg):
    """Share of a version's line length that has a stroke of the page under it."""

    tree = STRtree(linestrings(seg.reshape(-1, 2, 2)))
    pts = []
    for v in lines:
        g = LineString(v)
        pts += [g.interpolate(t) for t in np.arange(0.5, max(g.length, 1.0), 1.0)]
    return len(np.unique(tree.query(pts, predicate="dwithin", distance=TOL)[0])) / max(len(pts), 1)


def jumps(items):
    """Windows with the most line length found in only one version, largest first, not overlapping."""

    score = {}
    for v in items:
        for p, q in zip(v[:-1], v[1:]):
            n = max(int(np.hypot(*(q - p)) // 5), 1)
            for t in (np.arange(n) + 0.5) / n:
                c = tuple(np.floor((p + (q - p) * t) / CELL).astype(int))
                score[c] = score.get(c, 0.0) + np.hypot(*(q - p)) / n
    out, used = [], set()
    for c, s in sorted(score.items(), key=lambda kv: -kv[1]):
        if any(abs(c[0] - u[0]) <= 1 and abs(c[1] - u[1]) <= 1 for u in used):
            continue
        used.add(c)
        out.append({"x": c[0] * CELL, "y": c[1] * CELL, "w": CELL, "len": round(s)})
        if len(out) == JUMPS:
            break
    return out


def main(out, stem_a, stem_b, pipe_layer=""):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pages = {p.stem: p for p in load_gt_pages(DB)}
    pa, pb = pages[stem_a], pages[stem_b]
    assert pa.sha256 == pb.sha256 and pa.page_number == pb.page_number
    d = np.load(DATA / f"{stem_b}.npz")
    d = {k: d[k] for k in d.files}
    segs = page_segments_of(d)
    seg = segs.seg
    hot = np.array([bool(pipe_layer) and pipe_layer in str(n) for n in d["layer_names"]])[d["layer_id"]][segs.owner]
    la, lb = lines_of(pa), lines_of(pb)
    sa, sb = shared_mask(la, lb), shared_mask(lb, la)
    only = [v for v, s in zip(la, sa) if not s] + [v for v, s in zip(lb, sb) if not s]
    length = lambda ls: sum(LineString(v).length for v in ls)  # noqa: E731
    stats = {
        k: {"doc": p.document_id, "gt": p.gt_id, "updated_at": p.updated_at, "objects": len(p.objects), "lines": len(ls), "line_length": round(length(ls)), "only_here": round(length([v for v, x in zip(ls, s) if not x])), "only_here_lines": int((~s).sum()), "on_ink": round(float(ink_share(ls, seg)), 3)}
        for k, p, ls, s in (("a", pa, la, sa), ("b", pb, lb, sb))
    }
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    data = {
        "file": pa.filename,
        "w": float(pa.page_width or seg[:, [0, 2]].max()),
        "h": float(pa.page_height or seg[:, [1, 3]].max()),
        "seg": np.round(seg[~hot], 2).ravel().tolist(),
        "pipe": np.round(seg[hot], 2).ravel().tolist(),
        "pipe_name": pipe_layer,
        "a": [{"v": np.round(v, 2).tolist(), "s": bool(x)} for v, x in zip(la, sa)],
        "b": [{"v": np.round(v, 2).tolist(), "s": bool(x)} for v, x in zip(lb, sb)],
        "ao": others_of(pa),
        "bo": others_of(pb),
        "stats": stats,
        "jumps": jumps(only),
    }
    html = (Path(__file__).parent / "gt_versions_view.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(html.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    print(out / "index.html", f"{(out / 'index.html').stat().st_size / 1e6:.1f} MB", len(seg), "segments")


if __name__ == "__main__":
    main(*sys.argv[1:])
