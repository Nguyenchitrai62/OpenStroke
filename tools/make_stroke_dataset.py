"""OpenStroke28: stroke dataset `e8` (the stroke trainer's layout: <stem>.npz + manifest.json + folds.json) from a re-split OBB dataset
(`obb_hvac10b`, openstroke.obb.resplit). Every page, GT or Viewer, is labelled from its payload with the same label rule
(relabel.object_strokes: the strokes that make up the objects), as the score pack does; layers are kept in the npz (the logic needs
them; `no_layers=true` hides them from the model). Folds: the val pages of the split = fold 0, train pages = fold -1 (train only), the
pages the split dropped are excluded. Data only; AI_Takeoff (DB, payloads) and the Viewer export are only read.

    py -3.12 tools/make_stroke_dataset.py datasets/obb_hvac10b datasets/viewer_hvac19/exports_20261113/labels.json datasets/e8
"""
from __future__ import annotations

import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
DB = r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3"


def build_one(row, gobj, sha, repo, out):
    from openstroke.build import shape_attributes
    from openstroke.geometry import IDENTITY, page_segments, transform_points
    from openstroke.relabel import object_strokes

    stem = row["stem"]
    with gzip.open(repo / row["payload"].replace("\\", "/"), "rt", encoding="utf-8") as fh:
        shapes = json.load(fh).get("shapes") or []
    w, h = row["size_pt"]
    if row.get("transform", IDENTITY) != IDENTITY:
        gobj = [{**o, "vertices": transform_points(row["transform"], np.asarray(o["vertices"], float), w, h).tolist()} for o in gobj]
    segs = page_segments(shapes)
    attrs = shape_attributes(shapes, segs)
    length = np.bincount(segs.owner, weights=segs.length, minlength=len(shapes))
    label = object_strokes("hvac", gobj, segs, attrs, length).astype(np.uint8)
    meta = {"stem": stem, "mode": "hvac", "document_sha256": sha, "page_number": int(row["page_number"]),
            "source": "viewer_hvac19" if stem.startswith("vw_") else "ai_takeoff_gt", "label_rule": "relabel.object_strokes"}
    tmp = out / f"{stem}.tmp.npz"   # atomic: an interrupted build leaves no half file that a rerun would skip
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), label=label, length=length.astype(np.float32),
                        segments=segs.seg.astype(np.float32), shape_offsets=segs.shape_offsets, **attrs)
    tmp.replace(out / f"{stem}.npz")
    return stem, len(shapes), int(label.sum()), float(length[label > 0].sum() / max(length.sum(), 1e-9))


def main(obb, labels_json, out, workers="4"):
    from openstroke.sources import load_gt_pages
    from viewer_obb_dataset import CLASS_MAP, SET

    obb, out = Path(obb), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    repo = obb.resolve().parents[1]
    rows = json.loads((obb / "pages.json").read_text(encoding="utf-8"))
    gt = {p.stem: p for p in load_gt_pages(DB)}
    lab = json.loads(Path(labels_json).read_text(encoding="utf-8"))
    pdf_of = {p["sha256"][:12]: p for p in lab["pdfs"]}
    jobs = []
    for r in rows:
        if r["stem"].startswith("vw_"):
            key, page = r["stem"][3:].rsplit("_p", 1)
            pdf = pdf_of[key]
            pg = next(x for x in pdf["pages"] if int(x["page"]) == int(page))
            gobj = [{"shape_name": CLASS_MAP[o["class"]], "vertices": o["points"]}
                    for o in (pg["labels"].get(SET) or {}).get("objects") or [] if o["class"] in CLASS_MAP]
            sha, name = pdf["sha256"], pdf["name"]
        else:
            g = gt[r["stem"]]
            gobj = [o for o in g.objects if len(o.get("vertices") or [])]
            sha, name = g.sha256, g.filename
        jobs.append((r, gobj, sha, name))
    done, failed = {}, []
    todo = []
    for r, gobj, sha, name in jobs:   # resumable: a page whose npz exists is not built again
        if (out / f"{r['stem']}.npz").exists():
            done[r["stem"]] = (r, sha, name)
        else:
            todo.append((r, gobj, sha, name))
    print(f"{len(done)} pages already built, {len(todo)} to build", flush=True)
    with ProcessPoolExecutor(int(workers)) as ex:
        futs = {ex.submit(build_one, r, gobj, sha, repo, out): (r, sha, name) for r, gobj, sha, name in todo}
        for f in as_completed(futs):
            try:
                stem, n, k, share = f.result()
            except Exception as e:  # noqa: BLE001  (e.g. out of memory: rerun to build the rest)
                failed.append(futs[f][0]["stem"])
                print(f"{futs[f][0]['stem']}: FAILED {type(e).__name__}", flush=True)
                continue
            done[stem] = futs[f]
            print(f"{stem}: {n} shapes, {k} label strokes ({share:.1%} of length)", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} pages failed ({failed[:5]}...): run again to build them")
    records, physical, groups = [], [], {}
    for stem, (r, sha, name) in sorted(done.items()):
        records.append({"stem": stem, "status": "ok", "document_sha256": sha, "page_number": int(r["page_number"]), "filename": name,
                        "mode": "hvac", "npz": f"{stem}.npz", "split": r["split"]})
        physical.append({"document_sha256": sha, "page_number": int(r["page_number"]), "filename": name, "mode": "hvac",
                         "gt_variants": [stem]})
        fold = 0 if r["split"] == "val" else -1
        g = groups.setdefault(sha, {"sha256": sha, "filename": name, "mode": "hvac", "pages": [], "fold": fold})
        if g["fold"] != fold:
            raise SystemExit(f"{name}: pages on both sides of the split")
        g["pages"].append(int(r["page_number"]))
    plan = json.loads((obb / "gt_provenance.json").read_text(encoding="utf-8")).get("split_plan") or {}
    (out / "manifest.json").write_text(json.dumps({"schema_version": "openstroke.e8.v1", "generator": "tools/make_stroke_dataset.py",
                                                   "source": str(obb), "split_plan": plan.get("name"), "records": records,
                                                   "physical_pages": physical}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "folds.json").write_text(json.dumps({"status": "confirmed", "k": 1,
                                                "note": f"OpenStroke28: split {plan.get('name')} (fold 0 = val: 5 GT + 8 Viewer pages; "
                                                        "fold -1 = train only). Chosen by the agent under the user's split authority (AGENTS 0.3).",
                                                "groups": list(groups.values())}, ensure_ascii=False, indent=1), encoding="utf-8")
    n_val = sum(r["split"] == "val" for r in records)
    print(f"{out}: {len(records)} pages ({n_val} val, {len(records) - n_val} train), {len(groups)} PDFs")


if __name__ == "__main__":
    main(*sys.argv[1:])
