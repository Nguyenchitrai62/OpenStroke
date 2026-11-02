"""E45: a run's out-of-fold scores through AI_Takeoff's real FIRE growth logic, with chains of the model's strokes in place of the line detector
(research/HYPOTHESES.md E45). No model and no HTTP call. Needs AI_Takeoff's Python for ``extract_fire``; AI_Takeoff is only read:

    PYTHONDONTWRITEBYTECODE=1 <AI_Takeoff python> research/probes/logic_fire_probe.py --tune e30 --test e37

Steps on a page: keep the layers with length-weighted mean p >= AGG (``layer`` when the page has real layers, else the style field ``layer_1``);
chain the line candidates with p >= t that are collinear with gaps <= GAP pt into runs of at least MIN_RUN pt; hand the runs to ``run_auto_accept`` as validated
"connect" detections; score the resulting lines with the label rule of FIRE.
"""

import argparse
import glob
import gzip
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.geometry import PageSegments  # noqa: E402
from openstroke.labels import gt_geometry, label_shapes  # noqa: E402
from openstroke.metrics import length_counts, prf  # noqa: E402

BE = Path(r"F:\Source_code\AI_Takeoff\BE")
DATA = Path("datasets/e3")
AGG = 0.1
GAP = 12.0
MIN_RUN = 20.0
SEEDS = (0.1, 0.3, 0.5)
TAU = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)  # E46: drop a line whose strokes the model scores below this


def stored_objects(document_id, page_number):
    con = sqlite3.connect(BE.joinpath("data/ai_takeoff.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    jobs = sorted((r[0] for r in con.execute("select id from jobs where document_id = ? and mode = 'fire' and status = 'succeeded'", (document_id,))), reverse=True)
    con.close()
    for jid in jobs:
        f = BE / "data" / "runs" / f"job-{jid:06d}" / "normalized.json"
        if f.exists():
            pages = json.loads(f.read_text(encoding="utf-8")).get("pages") or []
            return [o for p in pages if int(p.get("page_number", -1)) == page_number for o in p.get("objects") or [] if o.get("group") in ("primary", "reference")]
    return None


def chains(ctx, seqno_p, t):
    groups = {}
    for key, c in ctx.line_candidates_by_key.items():
        if max((seqno_p.get(n, 0.0) for n in c.seqnos), default=0.0) < t:
            continue
        dx, dy = c.point_b.x - c.point_a.x, c.point_b.y - c.point_a.y
        ln = (dx * dx + dy * dy) ** 0.5
        if ln < 1e-6:
            continue
        ux, uy = dx / ln, dy / ln
        if ux < 0 or (abs(ux) < 1e-9 and uy < 0):
            ux, uy = -ux, -uy
        g = (c.layer_name, round(np.degrees(np.arctan2(uy, ux))), round((c.point_a.y * ux - c.point_a.x * uy) / 1.5))
        s0, s1 = c.point_a.x * ux + c.point_a.y * uy, c.point_b.x * ux + c.point_b.y * uy
        a, b = (c.point_a, c.point_b) if s0 <= s1 else (c.point_b, c.point_a)
        groups.setdefault(g, []).append((min(s0, s1), max(s0, s1), key, a, b, c.layer_name))
    dets = []
    for v in groups.values():
        v.sort(key=lambda x: x[0])
        run, end = [v[0]], v[0][1]
        runs = []
        for x in v[1:]:
            if x[0] - end <= GAP:
                run.append(x)
                end = max(end, x[1])
            else:
                runs.append(run)
                run, end = [x], x[1]
        runs.append(run)
        for r in runs:
            first, last = min(r, key=lambda x: x[0]), max(r, key=lambda x: x[1])
            if last[1] - first[0] < MIN_RUN:
                continue
            layer = first[5]
            pts = [{"x": first[3].x, "y": first[3].y, "layerName": layer}, {"x": last[4].x, "y": last[4].y, "layerName": layer}]
            dets.append({"class_name": "connect", "postprocess": {"validated": True, "layer_name": layer, "points": pts, "manual_points": pts,
                                                                  "matched_line_keys": [x[2] for x in r]}})
    return dets


def page_scores(run, fold, rec):
    from extract_fire.auto_accept import build_line_query_context, run_auto_accept
    from extract_fire.extract_line import build_compact_line_detections, prepare_line_extract_page_json

    stem, doc, page = rec["stem"], rec["document_id"], rec["page_number"]
    f = sorted(glob.glob(f"datasets/payloads/document-{doc:06d}/page-{page:04d}*.json.gz"))
    if not f:
        return None, "no payload"
    pay = json.load(gzip.open(f[-1], "rt", encoding="utf-8"))
    sh = pay["shapes"]
    d = np.load(DATA / f"{stem}.npz")
    if len(sh) != len(d["label"]):
        return None, "shape count differs"
    z = np.load(f"runs_colab/{run}/fold{fold}/val_scores.npz")
    p = np.zeros(len(sh))
    p[z[f"{stem}__index"]] = z[stem]
    L, y = d["length"].astype(np.float64), d["label"].astype(bool)
    off = d["shape_offsets"]
    segs = PageSegments(seg=d["segments"].astype(np.float64), owner=np.repeat(np.arange(len(off) - 1), np.diff(off)), shape_offsets=off)

    def score(objs):
        return length_counts(label_shapes(segs, gt_geometry("fire", objs)).label.astype(bool), y, L)

    field = "layer" if len({s.get("layer") for s in sh}) > 1 else "layer_1"
    names = np.array([str(s.get(field) or "") for s in sh])
    u, inv = np.unique(names, return_inverse=True)
    agg = np.bincount(inv, p * L, len(u)) / np.maximum(np.bincount(inv, L, len(u)), 1e-9)
    pj = prepare_line_extract_page_json(pay, visible_layers=[str(x) for x in u[agg >= AGG] if x], layer_field=field)["page_json"]
    seqno_p = {}
    for s, q in zip(sh, p):
        seqno_p[s.get("seqno")] = max(seqno_p.get(s.get("seqno"), 0.0), float(q))
    ctx = build_line_query_context(pj)
    res = {"plain": None}
    seg = d["segments"].astype(np.float64)
    owner = np.repeat(np.arange(len(off) - 1), np.diff(off))
    mids = shapely.points((seg[:, :2] + seg[:, 2:]) / 2)
    seg_len = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    tree = shapely.STRtree(mids)
    stored = stored_objects(doc, page)
    if stored is not None:
        res["stored"] = score(stored)
    for t in SEEDS:
        dets = chains(ctx, seqno_p, t)
        out = run_auto_accept(pj, {"detections": dets}, context=ctx)
        objs = [{"shape_name": "Line", "vertices": x["points"]} for x in build_compact_line_detections(out) if x["class_name"] == "connect" and len(x["points"]) >= 2]
        lines = [shapely.linestrings(np.asarray(o["vertices"], float)) for o in objs]
        oi, si = tree.query([g.buffer(2.0) for g in lines], predicate="contains") if lines else (np.zeros(0, int), np.zeros(0, int))
        num = np.bincount(oi, p[owner[si]] * seg_len[si], len(lines))
        den = np.bincount(oi, seg_len[si], len(lines))
        line_score = num / np.maximum(den, 1e-9)
        for tau in TAU:
            res[f"t{t}_v{tau}"] = score([o for o, x in zip(objs, line_score) if x >= tau])
    del res["plain"]
    return res, field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="e30")
    ap.add_argument("--test", default="e37")
    a = ap.parse_args()
    logging.disable(logging.INFO)
    man = {r["stem"]: r for r in json.load(open(DATA / "manifest.json", encoding="utf-8"))["records"] if r.get("status") == "ok"}
    arms = ["stored", *[f"t{t}_v{tau}" for t in SEEDS for tau in TAU]]
    total = {}
    for run in (a.tune, a.test):
        rows = {}
        for k in range(5):
            for pg in json.load(open(f"runs_colab/{run}/fold{k}/result.json"))["pages"]:
                rec = man[pg["stem"]]
                if rec["mode"] != "fire":
                    continue
                try:
                    res, note = page_scores(run, k, rec)
                except Exception as e:  # one page must not lose the others; a failed page is reported, never scored as 0
                    res, note = None, f"failed: {type(e).__name__}: {str(e)[:160]}"
                if res:
                    rows[pg["stem"]] = res
                brief = "" if not res else " ".join(f"{x} {prf(*v)['f2']:.3f}" for x, v in res.items() if x == "stored" or x.endswith("_v0.0") or x.endswith("_v0.4"))
                print(f"{run} {pg['stem'][:21]:21s} {note:10s} {brief}", flush=True)
        total[run] = {x: np.sum([r[x] for r in rows.values() if x in r], 0) for x in arms}
        print(f"{run}: {len(rows)} pages")
        for x in arms:
            m = prf(*total[run][x])
            print(f"  {x:7s} F2 {m['f2']:.3f}  R {m['recall']:.3f}  P {m['precision']:.3f}", flush=True)
    best = max(arms[1:], key=lambda x: prf(*total[a.tune][x])["f2"])
    m = prf(*total[a.test][best])
    print(f"picked on {a.tune}: {best}; {a.test}: F2 {m['f2']:.3f}  R {m['recall']:.3f}  P {m['precision']:.3f}")


if __name__ == "__main__":
    main()
