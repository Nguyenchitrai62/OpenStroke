"""E40: a run's out-of-fold masks through AI_Takeoff's real HVAC geometry logic, against the pipeline's own layers (research/HYPOTHESES.md E40).

No model and no HTTP call: ``extract_hvac.detection.run_detection`` on the payload, with the text the AI_Takeoff run stored. Objects are scored
with the same label rule as ``pipeline_baseline.py``. Needs AI_Takeoff's Python for ``extract_hvac``; AI_Takeoff is only read, results and the
per-arm cache go to ``runs_colab/logic_hvac/``:

    PYTHONDONTWRITEBYTECODE=1 <AI_Takeoff python> research/probes/logic_hvac_probe.py e30 --workers 3
"""

import argparse
import glob
import gzip
import json
import re
import sqlite3
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.geometry import PageSegments  # noqa: E402
from openstroke.labels import gt_geometry, label_shapes, refine_hvac  # noqa: E402
from openstroke.metrics import length_counts, prf  # noqa: E402

BE = Path(r"F:\Source_code\AI_Takeoff\BE")
DATA = Path("datasets/e3")
CACHE = Path("runs_colab/logic_hvac2")  # E44; the first cache (E40-E43, one text source) is runs_colab/logic_hvac
THRESHOLDS = ()  # E40 tuned 0.05, 0.1, 0.2, 0.3
AGG = (0.1,)  # E42: a layer is selected when its length-weighted mean p reaches this; 0.1 was fixed on E30
UNION_T = 0.05
MASK_LAYER = "__MASK__"


def pipeline_job(document_id, page_number):
    """Latest succeeded HVAC run of the page that stored its result: -> (job id, page result) or None."""

    con = sqlite3.connect(BE.joinpath("data/ai_takeoff.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    jobs = sorted((r[0] for r in con.execute("select id from jobs where document_id = ? and mode = 'hvac' and status = 'succeeded'",
                                             (document_id,))), reverse=True)
    con.close()
    for jid in jobs:
        f = BE / "data" / "runs" / f"job-{jid:06d}" / "result.json"
        if not f.exists() or not f.with_name("normalized.json").exists():  # a run still in progress has no normalized.json yet
            continue
        for p in json.loads(f.read_text(encoding="utf-8")).get("pages") or []:
            if int(p.get("page_number", -1)) == page_number and p.get("result"):
                return jid, p["result"]
    return None


def page_text(document_id, page_number):
    """The two text sources the real run merges, from ``model_boxes``: same frame as the payload, so rotated pages work too."""

    con = sqlite3.connect(BE.joinpath("data/ai_takeoff.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    out = {}
    for arg, source in (("ocr", "ocr"), ("vector_text", "text")):
        rows = con.execute("select text, x0, y0, x1, y1, payload_json from model_boxes where document_id = ? and page_number = ? and source = ? order by id",
                           (document_id, page_number, source)).fetchall()
        texts, boxes = [], []
        for t, x0, y0, x1, y1, payload in rows:
            t = t or (json.loads(payload).get("text") if payload else None)
            if t:
                texts.append(t)
                boxes.append([x0, y0, x1, y1])
        if texts:
            out[arg] = {"rec_texts": texts, "rec_boxes": boxes, "textline_orientation_angles": [0] * len(texts), "list_seqno": []}
    con.close()
    return out


def stored_objects(job, page_number):
    f = BE / "data" / "runs" / f"job-{job:06d}" / "normalized.json"
    pages = json.loads(f.read_text(encoding="utf-8")).get("pages") or []
    return [o for p in pages if int(p.get("page_number", -1)) == page_number for o in p.get("objects") or [] if o.get("group") in ("primary", "reference")]


def payload_of(document_id, page_number):
    f = sorted(glob.glob(f"datasets/payloads/document-{document_id:06d}/page-{page_number:04d}*.json.gz"))
    return json.load(gzip.open(f[-1], "rt", encoding="utf-8")) if f else None


def run_arm(cache_file, shapes, layers, text, area):
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    from extract_hvac.detection import run_detection

    t = time.time()
    out = run_detection({"drawings": shapes, "layers": layers, "page_area": area, **text})
    objs = [{"shape_name": o.get("shape_name"), "vertices": o.get("vertices")} for o in out.detected_shapes]
    res = {"objects": objs, "seconds": round(time.time() - t, 1)}
    cache_file.write_text(json.dumps(res), encoding="utf-8")
    return res


def score(objs, z, segs):
    geo = gt_geometry("hvac", objs)
    lab = label_shapes(segs, geo)
    lab, _ = refine_hvac(segs, geo, lab, z["layer_id"])
    return length_counts(lab.label.astype(bool), z["label"].astype(bool), z["length"].astype(np.float64))


def one_page(run, fold, rec):
    stem, doc, page = rec["stem"], rec["document_id"], rec["page_number"]
    out_dir = CACHE / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    job = pipeline_job(doc, page)
    pay = payload_of(doc, page)
    if job is None or pay is None:
        return stem, None, f"no {'job' if job is None else 'payload'}"
    jid, res = job
    z = np.load(DATA / f"{stem}.npz")
    shapes = pay["shapes"]
    if len(shapes) != len(z["label"]):
        return stem, None, f"payload has {len(shapes)} shapes, npz {len(z['label'])}"
    meta = pay.get("document_metadata") or {}
    text = page_text(doc, page)
    area = float(meta.get("page_width") or 0) * float(meta.get("page_height") or 0) or None
    off = z["shape_offsets"]
    segs = PageSegments(seg=z["segments"].astype(np.float64), owner=np.repeat(np.arange(len(off) - 1), np.diff(off)), shape_offsets=off)
    counts = {}
    layers = res.get("selected_ong_gio_layers") or res.get("layers") or []
    if layers:
        counts["ctrl"] = score(run_arm(out_dir / "ctrl.json", shapes, layers, text, area)["objects"], z, segs)
    counts["stored"] = score(stored_objects(jid, page), z, segs)

    def with_mask(mask):
        d = [dict(s) for s in shapes]
        for s, m in zip(d, mask):
            if m:
                s["layer"] = MASK_LAYER
        return d

    counts["gt"] = score(run_arm(out_dir / "gt.json", with_mask(z["label"].astype(bool)), [MASK_LAYER], text, area)["objects"], z, segs)
    sc = np.load(f"runs_colab/{run}/fold{fold}/val_scores.npz")
    p = np.zeros(len(shapes))
    p[sc[f"{stem}__index"]] = sc[stem]
    for t in THRESHOLDS:
        c = out_dir / f"{run}_t{t}.json"
        counts[f"mask{t}"] = score(run_arm(c, with_mask(p >= t), [MASK_LAYER], text, area)["objects"], z, segs)
    L = z["length"].astype(np.float64)
    names = np.array([str(s.get("layer") or "") for s in shapes])
    uniq, inv = np.unique(names, return_inverse=True)
    agg = np.bincount(inv, p * L, len(uniq)) / np.maximum(np.bincount(inv, L, len(uniq)), 1e-9)
    for a in AGG:
        sel = agg >= a
        chosen = [str(x) for x in uniq[sel] if x]
        if chosen:
            counts[f"layers{a}"] = score(run_arm(out_dir / f"{run}_layers{a}.json", shapes, chosen, text, area)["objects"], z, segs)
        else:
            counts[f"layers{a}"] = length_counts(np.zeros(len(L), bool), z["label"].astype(bool), L)
        extra = (p >= UNION_T) & ~sel[inv]
        counts[f"union{a}"] = score(run_arm(out_dir / f"{run}_union{a}.json", with_mask(extra), [*chosen, MASK_LAYER], text, area)["objects"], z, segs)
    return stem, counts, f"job {jid}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", default=None, help="comma list of stems (debug)")
    a = ap.parse_args()
    man = {r["stem"]: r for r in json.load(open(DATA / "manifest.json", encoding="utf-8"))["records"] if r.get("status") == "ok"}
    todo = []
    for k in sorted(int(f.name[4:]) for f in Path(f"runs_colab/{a.run}").glob("fold*")):
        for pg in json.load(open(f"runs_colab/{a.run}/fold{k}/result.json"))["pages"]:
            if man[pg["stem"]]["mode"] == "hvac" and (not a.only or pg["stem"] in a.only.split(",")):
                todo.append((a.run, k, {x: man[pg["stem"]][x] for x in ("stem", "document_id", "page_number")}))
    print(f"{len(todo)} HVAC pages, run {a.run}", flush=True)
    results = {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one_page, *t): t[2]["stem"] for t in todo}
        for f in as_completed(futs):
            stem = futs[f]
            try:
                _, counts, note = f.result()
            except Exception as e:  # one page must not lose the others; a failed page is reported, never silently scored as 0
                counts, note = None, f"failed: {type(e).__name__}: {str(e)[:200]}"
            results[stem] = counts
            brief = "" if counts is None else " ".join(f"{k} {prf(*v)['f2']:.3f}" for k, v in counts.items())
            print(f"{stem:24s} {note:30s} {brief}", flush=True)
    done = {s: c for s, c in results.items() if c}
    arms = [x for x in ("stored", "ctrl", *[f"mask{t}" for t in THRESHOLDS], *[f"layers{g}" for g in AGG], *[f"union{g}" for g in AGG], "gt")
            if all(x in c for c in done.values())]
    f2 = lambda v: prf(*v)["f2"]  # noqa: E731
    for label, keep in (("valid (ctrl >= stored - 0.05)", lambda c: f2(c["ctrl"]) >= f2(c["stored"]) - 0.05), ("not valid", lambda c: f2(c["ctrl"]) < f2(c["stored"]) - 0.05)):
        part = [c for c in done.values() if "ctrl" in c and keep(c)]
        print(f"{label}: {len(part)} pages")
        for x in arms:
            if part:
                m = prf(*np.sum([c[x] for c in part], 0))
                print(f"  {x:10s} F2 {m['f2']:.3f}  R {m['recall']:.3f}  P {m['precision']:.3f}  pages F2 < 0.5: {sum(f2(c[x]) < 0.5 for c in part)}")
    print(f"\nscored {len(done)} / {len(todo)} pages; arms on every scored page")
    for x in arms:
        m = prf(*np.sum([c[x] for c in done.values()], 0))
        low = sum(prf(*c[x])["f2"] < 0.5 for c in done.values())
        print(f"  {x:10s} F2 {m['f2']:.3f}  R {m['recall']:.3f}  P {m['precision']:.3f}  pages F2 < 0.5: {low}")
    skipped = [s for s, c in results.items() if not c]
    if skipped:
        print("not scored:", skipped)
    (CACHE / f"{a.run}_counts.json").write_text(json.dumps({s: {k: list(v) for k, v in c.items()} for s, c in done.items()}), encoding="utf-8")


if __name__ == "__main__":
    main()
