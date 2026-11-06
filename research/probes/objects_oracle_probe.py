"""E62: can HVAC objects come straight from per-stroke class + centre offset? On the HVAC GT pages, objects matched by shared strokes:
- decode with the true class and true offsets (the decoder's own ceiling), and with offsets off by Gaussian noise;
- the pipeline's objects (what AI_Takeoff draws today), turned into strokes by the same rule.
No model run. AI_Takeoff is only read. ``Y1_10-20.pdf`` (doc202) is left out: its stored pipeline run was overwritten (user: the file is broken).

    py -3.12 research/probes/objects_oracle_probe.py datasets/e6
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_v3_probe import job_of  # noqa: E402
from openstroke.geometry import IDENTITY, transform_points  # noqa: E402
from openstroke.objects import HVAC_CLASSES, N_CLASSES, assign_objects, decode_objects, object_counts  # noqa: E402
from openstroke.relabel import STYLE_KEYS, object_strokes, page_segments_of  # noqa: E402
from openstroke.sources import load_gt_pages  # noqa: E402

DB = r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3"
NOISE = (0.0, 5.0, 15.0)
SKIP = ("doc000202", "doc000020_p0001_gt10")


def page(job):
    data, rec, objects = job
    z = np.load(Path(data) / rec["npz"])
    d = {k: z[k] for k in z.files}
    w, h = rec["page_size"]["payload"]
    t = rec.get("gt_transform_applied")
    if t and t != IDENTITY:
        objects = [{**o, "vertices": transform_points(t, np.asarray(o["vertices"], float), w, h).tolist()} for o in objects if len(o.get("vertices") or [])]
    segs = page_segments_of(d)
    y, L = d["label"].astype(bool), d["length"].astype(np.float64)
    cls, inst, ctr = assign_objects(segs, objects, y)
    gt_cls_of = {int(i): int(c) for i, c in zip(inst[inst >= 0], cls[inst >= 0])}
    off = np.where(np.isnan(ctr), 0.0, ctr - d["bbox"].reshape(-1, 2, 2).mean(1)) if "bbox" in d else None
    xy = d["bbox"].reshape(-1, 2, 2).mean(1)
    rng = np.random.default_rng(0)
    row = {"stem": rec["stem"], "classed": float(L[y & (cls > 0)].sum() / max(L[y].sum(), 1e-9)), "objects": len(gt_cls_of)}
    for s in NOISE:
        pred, pcls = decode_objects(xy, cls, off + rng.normal(0, s, off.shape) * (s > 0), cls > 0)
        row[f"oracle{s:g}"] = object_counts(pred, pcls, inst, gt_cls_of, L).tolist()
    j = job_of(rec["document_id"], "hvac", rec["page_number"])
    if j:
        attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
        objs = [o for o in j[0] if o.get("vertices")]
        pm = object_strokes("hvac", objs, segs, attrs, L)
        pcls_s, pinst, _ = assign_objects(segs, objs, pm)
        pcls = np.zeros(max(int(pinst.max()) + 1, 1), np.int64)
        pcls[pinst[pinst >= 0]] = pcls_s[pinst >= 0]
        row["pipeline"] = object_counts(pinst.astype(np.int64), pcls, inst, gt_cls_of, L).tolist()
    return row


def patient(job):
    for wait in (0, 60, 180, 300):
        time.sleep(wait)
        try:
            with ProcessPoolExecutor(1) as ex:
                return ex.submit(page, job).result()
        except (RuntimeError, MemoryError, BrokenProcessPool) as e:
            print("retry", job[1]["stem"], str(e)[:60], flush=True)
    raise RuntimeError(job[1]["stem"])


def show(name, rows, key):
    c = np.sum([r[key] for r in rows if key in r], 0)
    line = f"{name:12s} pages {sum(key in r for r in rows):2d} |"
    tp, npred, ngt = c[1:].sum(0)
    line += f" all P {tp / max(npred, 1):.3f} R {tp / max(ngt, 1):.3f} F1 {2 * tp / max(npred + ngt, 1):.3f} |"
    for k in range(1, N_CLASSES):
        if c[k, 2]:
            line += f" {HVAC_CLASSES[k]} {2 * c[k, 0] / max(c[k, 1] + c[k, 2], 1):.2f} ({c[k, 2]})"
    print(line)


def main(data):
    man = {r["stem"]: r for r in json.loads((Path(data) / "manifest.json").read_text(encoding="utf-8"))["records"] if r.get("status") == "ok"}
    gt = {p.stem: p for p in load_gt_pages(DB)}
    jobs = [(data, man[s], gt[s].objects) for s in sorted(man) if man[s]["mode"] == "hvac" and not man[s].get("pseudo") and s in gt and not s.startswith(SKIP)]
    rows = [patient(j) for j in jobs]
    Path("runs_colab/e62").mkdir(parents=True, exist_ok=True)
    Path("runs_colab/e62/oracle.json").write_text(json.dumps(rows), encoding="utf-8")
    print(f"true length with an object class: {np.mean([r['classed'] for r in rows]):.3f} (page mean); GT objects {sum(r['objects'] for r in rows)}")
    for s in NOISE:
        show(f"oracle ±{s:g}pt", rows, f"oracle{s:g}")
    show("pipeline", rows, "pipeline")
    have = [r for r in rows if "pipeline" in r]
    show("oracle same", have, "oracle5")


if __name__ == "__main__":
    main(*sys.argv[1:])
