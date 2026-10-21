"""E1.2: the current AI_Takeoff pipeline, scored with the model's metric (the bar to beat, user 2026-10-21).

For every GT page: the latest succeeded AI_Takeoff job of that document and mode (``BE/data/runs/job-NNNNNN/
normalized.json``, read only), its predicted objects of that page labelled onto the page's shapes with the *same* rule
as the ground truth (``labels``: FIRE / PLUMBING along lines, HVAC inside ducts + v2 refinement) -> the pipeline's
shape mask; then recall / precision / F2 by length against the GT labels (``datasets/e2``). Also prints AI_Takeoff's own
object-level score when it is in the ``evaluations`` table.

    py -3.12 research/probes/pipeline_baseline.py datasets/e2 datasets/e2/folds_dev.json > research/probes/pipeline_baseline.log
"""

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.geometry import PageSegments  # noqa: E402
from openstroke.labels import gt_geometry, label_shapes, refine_hvac  # noqa: E402
from openstroke.metrics import length_counts, prf  # noqa: E402

BE = Path(r"F:\Source_code\AI_Takeoff\BE")


def main(data, folds=None):
    data = Path(data)
    man = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    dev = set()
    if folds:
        from openstroke.folds import resolve

        entries, _ = resolve(man, json.loads(Path(folds).read_text(encoding="utf-8")))
        dev = {s for s, f, _ in entries if f == 0}
    con = sqlite3.connect((BE / "data" / "ai_takeoff.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    jobs = defaultdict(list)
    for jid, doc, mode in con.execute("select id, document_id, mode from jobs where status = 'succeeded'"):
        jobs[(doc, mode)].append(jid)
    acc = defaultdict(lambda: [0.0, 0.0, 0.0])
    print(f"{'page':24s} {'mode':8s} {'file':40s} {'job':>5s}  R      P      F2    | pipeline objects")
    for r in man["records"]:
        if r.get("status") != "ok":
            continue
        cand = sorted(jobs.get((r["document_id"], r["mode"]), []))
        if not cand:
            print(f"{r['stem']:24s} {r['mode']:8s} no pipeline job")
            continue
        jid = cand[-1]
        nf = BE / "data" / "runs" / f"job-{jid:06d}" / "normalized.json"
        if not nf.exists():
            print(f"{r['stem']:24s} {r['mode']:8s} job {jid}: no normalized.json")
            continue
        norm = json.loads(nf.read_text(encoding="utf-8"))
        objs = [o for p in norm.get("pages") or [] if int(p.get("page_number", -1)) == r["page_number"]
                for o in p.get("objects") or [] if o.get("group") in ("primary", "reference")]
        z = np.load(data / r["npz"])
        off = z["shape_offsets"]
        segs = PageSegments(seg=z["segments"].astype(np.float64), owner=np.repeat(np.arange(len(off) - 1), np.diff(off)),
                            shape_offsets=off)
        geo = gt_geometry(r["mode"], objs)
        lab = label_shapes(segs, geo)
        if r["mode"] == "hvac":
            lab, _ = refine_hvac(segs, geo, lab, z["layer_id"])
        pred = lab.label.astype(bool)
        y, L = z["label"], z["length"]
        tp, fp, fn = length_counts(pred, y, L)
        m = prf(tp, fp, fn)
        tag = "DEV " if r["stem"] in dev else ""
        print(f"{r['stem']:24s} {r['mode']:8s} {r['filename'][:40]:40s} {jid:5d}  {m['recall']:.3f}  {m['precision']:.3f}  "
              f"{m['f2']:.3f} | {len(geo)} in-scope objects {tag}")
        keys = ["all", r["mode"]] + (["dev", f"dev-{r['mode']}"] if r["stem"] in dev else [])
        for k in keys:
            a = acc[k]
            a[0] += tp
            a[1] += fp
            a[2] += fn
    print("\nMICRO (length-weighted, model metric)")
    for k, v in sorted(acc.items()):
        m = prf(*v)
        print(f"  {k:12s} R {m['recall']:.3f}  P {m['precision']:.3f}  F2 {m['f2']:.3f}")
    print("\nAI_Takeoff object-level evaluations (evaluate.py, its own metric)")
    for eid, gt, mj in con.execute("select id, ground_truth_id, metrics_json from evaluations"):
        o = json.loads(mj)["overall"]
        doc, mode, fn = con.execute("select g.document_id, g.mode, d.filename from ground_truths g join documents d "
                                    "on d.id = g.document_id where g.id = ?", (gt,)).fetchone()
        print(f"  eval {eid} {mode:8s} {fn[:40]:40s} F2 {o['f2']:.3f} (R {o['recall']:.3f} P {o['precision']:.3f})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
