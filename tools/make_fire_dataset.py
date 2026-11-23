"""FIRE stroke dataset `fire1` (F187) for the stroke model (openstroke31 / OpenStroke33): the same layout as the HVAC set `e11`
(<stem>.npz + manifest.json + folds.json), so FIRE and HVAC train in one framework.

Source: the AI_Takeoff GT pages of the local folder trai_nc -> FIRE (DB `folders` 19 and its sub-folder `test` 20), groups primary +
reference only; read-only (DB opened mode=ro). Payloads: datasets/payloads/document-NNNNNN (tools/make_payloads.py, the same generator as
AI_Takeoff's cache). Per page:

- coordinate check (geometry.calibrate on the payload's own strokes; an inconclusive page is rejected, a non-identity frame is applied);
- label = relabel.object_strokes("fire", ...): label rule v3 + E55 (the strokes along a GT ``Line``), the rule `e6` used for these pages;
- ``oinst`` / ``octr`` = objects.assign_lines: the GT Line each true stroke draws (closure / edge losses group by it);
- ``fcls`` = objects.FIRE_CLASSES index (1 = Line); ``ocls`` = 0 everywhere (the HVAC class head has no FIRE class: its loss skips these
  pages; giving FIRE a class of its own is an architecture change for a version card).

Viewer: the only FIRE project (#34 Line-Fire) holds 11 JPG renders of TOWER Z35.1 (the project of GT doc163), all in_progress: no
vectors and the same project as a GT page, so none is used (F187). Folds: PROJECTS below (one group per project, a duplicate drawing
on one side), fold 0 = val, -1 = train; chosen before any run (AGENTS 0.3). Data only; nothing in AI_Takeoff is written.

    py -3.12 tools/make_fire_dataset.py datasets/fire1 [datasets/openstroke_fire1_data.zip]
"""
from __future__ import annotations

import gzip
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB = Path(r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3")
FOLDERS = (19, 20)                 # trai_nc / FIRE and FIRE / test
PAYLOADS = ROOT / "datasets" / "payloads"
#: (sha256:page) -> the GT variant kept when a page has two (user 2026-11-03 for 333.pdf, AGENTS 3)
DUPLICATE_CHOICE = {"6070d055f03f29e55842ac71cf6289c8cedd9fefa55f3db03b1819b5a8c310fb:1": "doc000153_p0001_gt103"}
#: document id -> project group. Same project = same side of the split. Evidence (F187): doc143 / doc150 draw block A of one project
#: (xref "Block A- AR B1 / B2", layer names Jaccard 0.64), doc151 / doc153 share the PTN-* pipe layers seen nowhere else (F31, F59);
#: doc163 / doc164 (TOWER Z35.1 / U35.2) are grouped on their names only (doc163 has no layers), kept together to be safe.
#: Fingerprints are checked again at build time (``check_groups``).
PROJECTS = {143: "block_a", 150: "block_a", 149: "fp_0875", 151: "ptn", 153: "ptn", 154: "fp_page16", 158: "thap_a",
            162: "l30m", 163: "tower35", 164: "tower35"}
VAL_PROJECTS = ("fp_0875", "l30m", "tower35")   # 4 val pages, 3 projects, 3 drawing styles; the rest train
LONG = 20.0       # pt: strokes whose endpoints fingerprint a drawing
SAME_DRAWING = 0.3


def fire_pages():
    """GT pages of the FIRE folders, one variant per physical page."""
    from openstroke.sources import connect_ro, load_gt_pages

    con = connect_ro(DB)
    try:
        docs = {int(r[0]) for r in con.execute(f"select id from documents where folder_id in ({','.join(map(str, FOLDERS))})")}
    finally:
        con.close()
    pages = [p for p in load_gt_pages(DB) if p.document_id in docs and p.mode == "fire"]
    by_key, dropped = {}, []
    for p in pages:
        by_key.setdefault(f"{p.sha256}:{p.page_number}", []).append(p)
    out = []
    for key, ps in sorted(by_key.items()):
        if len(ps) > 1:
            pick = DUPLICATE_CHOICE.get(key)
            if pick is None:
                raise SystemExit(f"{ps[0].filename}: {len(ps)} GT variants, add one to DUPLICATE_CHOICE")
            keep = [p for p in ps if p.stem == pick]
            dropped += [(p.stem, f"duplicate GT, kept {pick}") for p in ps if p.stem != pick]
            ps = keep
        out.append(ps[0])
    return sorted(out, key=lambda p: p.stem), dropped


def build_page(gp, out: Path) -> dict:
    from shapely import STRtree

    from openstroke.build import shape_attributes
    from openstroke.geometry import IDENTITY, calibrate, page_segments, transform_points
    from openstroke.labels import gt_coverage, gt_geometry, segment_lines
    from openstroke.objects import FIRE_CLASSES, assign_lines
    from openstroke.relabel import object_strokes
    from openstroke.sources import file_sha256, latest_payload_path

    rec = {"stem": gp.stem, "gt_id": gp.gt_id, "document_id": gp.document_id, "filename": gp.filename, "document_sha256": gp.sha256,
           "page_number": gp.page_number, "mode": "fire", "gt_updated_at": gp.updated_at, "source": "ai_takeoff_gt"}
    path = latest_payload_path(PAYLOADS, gp.document_id, gp.page_number)
    if path is None:
        return rec | {"status": "missing_payload"}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    shapes = payload.get("shapes") or []
    meta = payload.get("document_metadata") or {}
    w = float(meta.get("page_width") or gp.page_width or 0)
    h = float(meta.get("page_height") or gp.page_height or 0)
    rec["payload"] = {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
    rec["page_size"] = {"payload": [w, h], "gt": [gp.page_width, gp.page_height]}
    segs = page_segments(shapes)
    geo0 = gt_geometry("fire", gp.objects)
    cal = calibrate(STRtree(segment_lines(segs.seg)), geo0.vertices, geo0.closed, w, h)
    rec["calibration"] = {k: cal[k] for k in ("chosen", "conclusive", "reason")}
    if not cal.get("conclusive"):
        return rec | {"status": "rejected_calibration"}
    t = cal["chosen"]
    objs = [o for o in gp.objects if len(o.get("vertices") or [])]
    if t != IDENTITY:
        objs = [{**o, "vertices": transform_points(t, np.asarray(o["vertices"], float), w, h).tolist()} for o in objs]
    rec["gt_transform_applied"] = t
    attrs = shape_attributes(shapes, segs)
    length = np.bincount(segs.owner, weights=segs.length, minlength=len(shapes))
    label = object_strokes("fire", objs, segs, attrs, length)
    geo = gt_geometry("fire", objs)
    cov = gt_coverage(geo, segs.seg[label[segs.owner]])
    glen = np.array([float(g.length) for g in geo.geoms])
    inst, ctr = assign_lines(segs, objs, label)
    fcls = np.where(inst >= 0, FIRE_CLASSES.index("Line"), 0).astype(np.int8)
    n_line = sum(o.get("shape_name") == "Line" and len(o["vertices"]) >= 2 for o in objs)
    got = np.unique(inst[inst >= 0])
    L = length.astype(np.float64)
    rec["objects"] = {"by_class": {}, "line_objects": int(n_line), "line_objects_with_strokes": int(len(got)),
                      "line_coverage": len(got) / max(n_line, 1),
                      "lines_half_covered": int((cov >= 0.5).sum()),
                      "line_length_covered": float((cov * glen).sum() / max(glen.sum(), 1e-9)),
                      "true_length_with_object": float(L[label & (inst >= 0)].sum() / max(L[label].sum(), 1e-9))}
    for o in gp.objects:
        k = str(o.get("shape_name"))
        rec["objects"]["by_class"][k] = rec["objects"]["by_class"].get(k, 0) + 1
    rec["stats"] = {"shapes": len(shapes), "true_shapes": int(label.sum()), "true_length_share": float(L[label].sum() / max(L.sum(), 1e-9))}
    meta_np = {"stem": gp.stem, "mode": "fire", "document_sha256": gp.sha256, "page_number": gp.page_number, "source": "ai_takeoff_gt",
               "label_rule": "relabel.object_strokes", "object_rule": "objects.assign_lines", "fire_classes": list(FIRE_CLASSES)}
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"{gp.stem}.tmp.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta_np, ensure_ascii=False)), label=label.astype(np.uint8),
                        length=length.astype(np.float32), segments=segs.seg.astype(np.float32), shape_offsets=segs.shape_offsets, **attrs,
                        ocls=np.zeros(len(shapes), np.int8), oinst=inst, octr=ctr, fcls=fcls)
    tmp.replace(out / f"{gp.stem}.npz")
    return rec | {"status": "ok", "npz": f"{gp.stem}.npz"}


def fingerprint(npz: Path) -> tuple[set, set]:
    """(endpoints of the long strokes rounded to 1 pt, layer names) of a built page."""
    with np.load(npz) as z:
        s, off, L = z["segments"], z["shape_offsets"], z["length"]
        names = {str(x) for x in z["layer_names"]} - {"", "0"}   # no layers / the default layer say nothing about a project
    keep = np.flatnonzero(L >= LONG)
    a = s[off[keep]][:, :2]
    b = s[off[keep + 1] - 1][:, 2:]
    return {tuple(p) for p in np.round(np.r_[a, b]).astype(np.int64)}, names


def check_groups(out: Path, records: list[dict]) -> list[dict]:
    """Pairs of pages that look like one drawing (shared endpoints) or one project (shared layer names); a same-drawing pair across
    two project groups stops the build."""
    fp = {r["stem"]: fingerprint(out / r["npz"]) for r in records}
    proj = {r["stem"]: PROJECTS[r["document_id"]] for r in records}
    pairs = []
    stems = sorted(fp)
    for i, a in enumerate(stems):
        for b in stems[i + 1:]:
            ea, eb = fp[a][0], fp[b][0]
            ends = len(ea & eb) / max(min(len(ea), len(eb)), 1)
            la, lb = fp[a][1], fp[b][1]
            layers = len(la & lb) / max(len(la | lb), 1)
            if ends >= 0.05 or layers >= 0.3:
                pairs.append({"a": a, "b": b, "shared_endpoints": round(ends, 3), "layer_jaccard": round(layers, 3),
                              "same_group": proj[a] == proj[b]})
            if ends >= SAME_DRAWING and proj[a] != proj[b]:
                raise SystemExit(f"{a} / {b} share {ends:.0%} of their long-stroke endpoints but sit in groups {proj[a]} / {proj[b]}")
    return pairs


def main(out, zip_out=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pages, dropped = fire_pages()
    records = []
    for gp in pages:
        if gp.document_id not in PROJECTS:
            raise SystemExit(f"{gp.stem} ({gp.filename}): no project group in PROJECTS")
        r = build_page(gp, out)
        records.append(r)
        o = r.get("objects", {})
        print(f"{r['stem']}: {r['status']} {r.get('calibration', {}).get('chosen')} lines owning strokes {o.get('line_objects_with_strokes')}/"
              f"{o.get('line_objects')}, >= half covered {o.get('lines_half_covered')}, Line length covered "
              f"{o.get('line_length_covered', 0):.3f}, true length with object {o.get('true_length_with_object', 0):.3f}", flush=True)
    ok = [r for r in records if r["status"] == "ok"]
    pairs = check_groups(out, ok)
    groups = {}
    for r in ok:
        p = PROJECTS[r["document_id"]]
        r["project"] = p
        r["split"] = "val" if p in VAL_PROJECTS else "train"
        g = groups.setdefault(r["document_sha256"], {"sha256": r["document_sha256"], "filename": r["filename"], "mode": "fire",
                                                     "project": p, "pages": [], "fold": 0 if r["split"] == "val" else -1})
        g["pages"].append(r["page_number"])
    physical = [{"document_sha256": r["document_sha256"], "page_number": r["page_number"], "filename": r["filename"], "mode": "fire",
                 "gt_variants": [r["stem"]]} for r in ok]
    from openstroke.objects import FIRE_CLASSES, HVAC_CLASSES

    tot = {k: sum(r["objects"][k] for r in ok) for k in ("line_objects", "line_objects_with_strokes", "lines_half_covered")}
    manifest = {"schema_version": "openstroke.e8.v1", "generator": "tools/make_fire_dataset.py", "source": f"AI_Takeoff GT folders {FOLDERS}",
                "split_plan": "fire1", "records": records, "physical_pages": physical, "excluded": [{"stem": s, "reason": why} for s, why in dropped],
                "viewer": "none: project #34 Line-Fire = 11 JPG renders of TOWER Z35.1 (GT doc163's project), in_progress, no vectors",
                "objects": {"fire_classes": list(FIRE_CLASSES), "fire_key": "fcls", "hvac_classes": list(HVAC_CLASSES),
                            "ocls": "0 on every FIRE stroke (no FIRE class in the HVAC class head)", "rule": "openstroke/objects.py assign_lines",
                            **tot, "line_coverage": tot["line_objects_with_strokes"] / max(tot["line_objects"], 1),
                            "note": "line_objects_with_strokes: Lines owning >= 1 true stroke (a PDF stroke running across several GT Lines split at "
                                    "tees belongs to one of them only); lines_half_covered: Lines with >= 50 % of their length within 2 pt of a "
                                    "true stroke (labels.gt_coverage)"},
                "similar_pages": pairs,
                # F213: read by the trainer (train.apply_protocol): one cohort, threshold and best.pt on the FIRE GT val pages
                "protocol": {"name": "fire1", "sampling": None, "select": ["fire:gt"]}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    n_val = sum(r["split"] == "val" for r in ok)
    (out / "folds.json").write_text(json.dumps({"status": "confirmed", "k": 1, "split_plan": "fire1",
                                                "note": f"fire1: fold 0 = val ({n_val} GT pages, projects {', '.join(VAL_PROJECTS)}), fold -1 = train "
                                                        f"({len(ok) - n_val} GT pages). One project per side. Chosen by the agent under the user's "
                                                        "split authority (AGENTS 0.3), before any run.",
                                                "duplicate_choice": DUPLICATE_CHOICE, "groups": list(groups.values())},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    for p in pairs:
        print("similar:", p)
    print(f"{out}: {len(ok)} pages ({n_val} val, {len(ok) - n_val} train), {len(groups)} PDFs, {len(set(PROJECTS[r['document_id']] for r in ok))} "
          f"projects; Line objects with strokes {tot['line_objects_with_strokes']}/{tot['line_objects']}")
    if zip_out:
        with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_STORED) as zf:   # npz are already compressed
            for f in sorted(out.iterdir()):
                if f.suffix in (".npz", ".json") and ".tmp." not in f.name:
                    zf.write(f, f"{out.name}/{f.name}")
        print(f"{zip_out}: {Path(zip_out).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:])
