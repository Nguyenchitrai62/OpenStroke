"""Thin-line pipe dataset `thin1` (F188): FIRE and PLUMBING pipes are the same case for the stroke model (single thin lines; user
2026-11-19), so one dataset in the e11 layout holds

- FIRE GT: the 11 AI_Takeoff pages of fire1 (tools/make_fire_dataset.py: folders trai_nc / FIRE, same projects and val);
- PLUMBING GT: every AI_Takeoff GT page of mode plumbing (folder PLUMBING: GT38_* / GTG_* extracts of the CTN PDFs), groups primary +
  reference, names ``ỐNG`` / ``ÔNG`` spelled ``Ống`` as build._pipeline_name does;
- Viewer PLUMBING: project #38 PLUMBING_PDF, label set ``line_junction``, pages with status ``completed`` only; classes line1 .. line 8
  are pipe polylines (-> ``Ống <class>``); ``Mặt bằng`` (plan frame) and ``duct`` are not pipes and are left out. Stems ``vw_<sha12>_pNNNN``.

Every page: coordinate check (geometry.calibrate; inconclusive -> rejected), label rule v3 + E55 (relabel.object_strokes, mode fire /
plumbing), ``oinst`` / ``octr`` = objects.assign_lines, ``fcls`` = LINE_CLASSES (1 = pipe line), ``ocls`` = 0 (no HVAC class).
Duplicates: a Viewer page that draws the same strokes as a GT page (>= DUPLICATE of the long-stroke endpoints) is dropped, the GT is
kept (DUPLICATE); pages sharing >= MERGE join one split group; pairs >= SAME_PROJECT across the split are reported. Viewer quality per page: share of its pipe objects with >= 50 % of their length
under true strokes; a page below MIN_COVERED is dropped. On pages a Viewer page and a GT page both cover, the Viewer labels are
compared with the GT's (``viewer_vs_gt``). Folds: one group per source PDF (CTN_04_500_NN, CTN_DEV_NNN-NNN; FIRE: fire1 projects),
fold 0 = val, -1 = train, chosen before any run. AI_Takeoff and the Viewer are only read.

    <AI_Takeoff venv python> tools/make_thin_dataset.py extract <viewer export dir> datasets/viewer_plumbing38/payloads [workers]
    py -3.12 tools/make_thin_dataset.py build <viewer export dir> datasets/viewer_plumbing38/payloads datasets/thin1 [zip]
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

LABEL_SET = "line_junction"
PIPE_CLASS = re.compile(r"^line ?\d+$")
#: val PLUMBING source groups (each with GT extracts). Chosen from the drawing-sharing structure, before any run (F188): at MERGE the
#: CTN files form 6 blocks (CTN_04_500_01-07 + 09; DEV 001 + 051; DEV 101 + 251; DEV 151 + 201; 04_500_10; 04_500_08); val = the last three.
VAL_PLUMBING = ("CTN_DEV_151-200", "CTN_DEV_201-250", "CTN_04_500_10", "CTN_04_500_08")
SAME_PROJECT = 0.3    # share of long-stroke endpoints: two floors of one building share its grid / walls (reported)
MERGE = 0.7           # pages sharing this much are one drawing / revision -> one split group (at 0.3 every CTN page chains into one, F188)
DUPLICATE = 0.95      # the same drawing (true twins share >= 0.97, F188): keep one copy, GT before Viewer
MAX_VAL_SHARE = 0.35  # of the PLUMBING pages: a val group merged into a large project stops the build
#: F213, read by the trainer (train.apply_protocol): FIRE (7 train pages, the target) 40 % of the windows, PLUMBING GT 30 %, Viewer 30 %
#: (its labels miss ~24 % of the pipes, F188); threshold and best.pt on the mean stroke F2 of the FIRE and PLUMBING GT val pages
#: (Viewer val is reported, not selected on). Fixed before any run.
PROTOCOL = {"name": "thin1", "sampling": {"fire:gt": 0.4, "plumbing:gt": 0.3, "plumbing:viewer": 0.3}, "select": ["fire:gt", "plumbing:gt"]}
MIN_COVERED = 0.5
SOURCE_PDF = re.compile(r"(CTN_(?:04_500_\d+|DEV_\d+-\d+))")


def key_of(sha: str, page: int) -> str:
    return f"{sha[:12]}_p{int(page):04d}"


def viewer_pages(export_dir: Path) -> list[tuple[dict, dict, list[dict]]]:
    """-> [(pdf record, page record, pipe objects)] of the completed line_junction pages."""
    lab = json.loads((export_dir / "labels.json").read_text(encoding="utf-8"))
    out = []
    for p in lab["pdfs"]:
        for pg in p["pages"]:
            L = pg["labels"].get(LABEL_SET) or {}
            if L.get("status") != "completed":
                continue
            objs = [{"shape_name": f"Ống {o['class']}", "vertices": o["points"], "group": "primary", "viewer_id": o.get("id")}
                    for o in L.get("objects") or [] if PIPE_CLASS.match(str(o["class"])) and len(o.get("points") or []) >= 2]
            if objs:
                out.append((p, pg, objs))
    return out


def _extract(job):
    pdf, page, out = job
    from make_payloads import build

    if Path(out).exists():
        return out, "kept"
    payload = build(pdf, int(page))
    tmp = Path(out + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=1) as fh:
        json.dump(payload, fh)
    tmp.replace(out)
    return out, "built"


def extract(export_dir, out, workers="4"):
    export_dir, out = Path(export_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(export_dir / p["file"]), int(pg["page"]), str(out / f"{key_of(p['sha256'], pg['page'])}.json.gz"))
            for p, pg, _ in viewer_pages(export_dir)]
    print(f"{len(jobs)} completed Viewer pages", flush=True)
    with ProcessPoolExecutor(int(workers)) as ex:
        for i, (f, how) in enumerate(ex.map(_extract, jobs), 1):
            if i % 10 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {how} {Path(f).name}", flush=True)


def _pipe_name(obj: dict) -> dict:
    name = unicodedata.normalize("NFC", obj.get("shape_name") or "").strip()
    if name.casefold().startswith(("ống", "ông ")):
        return {**obj, "shape_name": "Ống" + name[3:]}
    return obj


def is_pipe(mode: str):
    return (lambda n: n == "Line") if mode == "fire" else (lambda n: n.startswith("Ống"))


def build_page(job: dict) -> dict:
    """One page -> npz in job['out'] + its manifest record."""
    from shapely import STRtree

    from openstroke.build import shape_attributes
    from openstroke.geometry import IDENTITY, calibrate, page_segments, transform_points
    from openstroke.labels import gt_coverage, gt_geometry, segment_lines
    from openstroke.objects import LINE_CLASSES, assign_lines
    from openstroke.relabel import object_strokes
    from openstroke.sources import file_sha256

    out, mode, stem, path = Path(job["out"]), job["mode"], job["stem"], Path(job["payload"])
    rec = dict(job["rec"])
    if not path.exists():
        return rec | {"status": "missing_payload"}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    shapes = payload.get("shapes") or []
    meta = payload.get("document_metadata") or {}
    w = float(meta.get("page_width") or job["size"][0] or 0)
    h = float(meta.get("page_height") or job["size"][1] or 0)
    rec["payload"] = {"path": str(path.resolve().relative_to(ROOT)), "sha256": file_sha256(path)}
    rec["page_size"] = {"payload": [w, h], "labels": list(job["size"])}
    segs = page_segments(shapes)
    objs = [o for o in job["objects"] if len(o.get("vertices") or [])]
    geo0 = gt_geometry(mode, objs)
    if not len(geo0.geoms):
        return rec | {"status": "no_pipe_objects"}
    cal = calibrate(STRtree(segment_lines(segs.seg)), geo0.vertices, geo0.closed, w, h)
    rec["calibration"] = {k: cal[k] for k in ("chosen", "conclusive", "reason")}
    if not cal.get("conclusive"):
        return rec | {"status": "rejected_calibration"}
    t = cal["chosen"]
    if t != IDENTITY:
        objs = [{**o, "vertices": transform_points(t, np.asarray(o["vertices"], float), w, h).tolist()} for o in objs]
    rec["gt_transform_applied"] = t
    attrs = shape_attributes(shapes, segs)
    length = np.bincount(segs.owner, weights=segs.length, minlength=len(shapes))
    label = object_strokes(mode, objs, segs, attrs, length)
    geo = gt_geometry(mode, objs)
    cov = gt_coverage(geo, segs.seg[label[segs.owner]])
    glen = np.array([float(g.length) for g in geo.geoms])
    inst, ctr = assign_lines(segs, objs, label, names=is_pipe(mode))
    L = length.astype(np.float64)
    got = np.unique(inst[inst >= 0])
    rec["objects"] = {"by_class": dict(Counter(str(o.get("shape_name")) for o in job["all_objects"])), "line_objects": len(geo.geoms),
                      "line_objects_with_strokes": int(len(got)), "lines_half_covered": int((cov >= 0.5).sum()),
                      "line_half_covered_share": float((cov >= 0.5).mean()),
                      "line_length_covered": float((cov * glen).sum() / max(glen.sum(), 1e-9)),
                      "true_length_with_object": float(L[label & (inst >= 0)].sum() / max(L[label].sum(), 1e-9))}
    rec["stats"] = {"shapes": len(shapes), "true_shapes": int(label.sum()), "true_length": float(L[label].sum()),
                    "true_length_share": float(L[label].sum() / max(L.sum(), 1e-9))}
    meta_np = {"stem": stem, "mode": mode, "document_sha256": rec["document_sha256"], "page_number": rec["page_number"], "source": rec["source"],
               "label_rule": "relabel.object_strokes", "object_rule": "objects.assign_lines", "line_classes": list(LINE_CLASSES)}
    tmp = out / f"{stem}.tmp.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta_np, ensure_ascii=False)), label=label.astype(np.uint8),
                        length=length.astype(np.float32), segments=segs.seg.astype(np.float32), shape_offsets=segs.shape_offsets, **attrs,
                        ocls=np.zeros(len(shapes), np.int8), oinst=inst, octr=ctr,
                        fcls=np.where(inst >= 0, LINE_CLASSES.index("Line"), 0).astype(np.int8))
    tmp.replace(out / f"{stem}.npz")
    return rec | {"status": "ok", "npz": f"{stem}.npz"}


def jobs_of(export_dir: Path, vw_payloads: Path, out: Path) -> tuple[list[dict], list[dict]]:
    from make_fire_dataset import PROJECTS, VAL_PROJECTS, fire_pages

    from openstroke.sources import latest_payload_path, load_gt_pages

    payloads = ROOT / "datasets" / "payloads"
    jobs, dropped = [], []
    fire, fdrop = fire_pages()
    dropped += [{"stem": s, "reason": r} for s, r in fdrop]
    for gp in fire:
        p = PROJECTS[gp.document_id]
        jobs.append({"stem": gp.stem, "mode": "fire", "objects": gp.objects, "all_objects": gp.objects, "size": (gp.page_width, gp.page_height),
                     "payload": str(latest_payload_path(payloads, gp.document_id, gp.page_number) or payloads / "missing"),
                     "group": f"fire:{p}", "val": p in VAL_PROJECTS,
                     "rec": {"stem": gp.stem, "gt_id": gp.gt_id, "document_id": gp.document_id, "filename": gp.filename,
                             "document_sha256": gp.sha256, "page_number": gp.page_number, "mode": "fire", "source": "ai_takeoff_gt"}})
    keys = {}
    for gp in load_gt_pages(ROOT.parent / "AI_Takeoff" / "BE" / "data" / "ai_takeoff.sqlite3"):
        if gp.mode != "plumbing":
            continue
        k = f"{gp.sha256}:{gp.page_number}"
        if k in keys:
            raise SystemExit(f"{gp.filename} p{gp.page_number}: two PLUMBING GT variants ({keys[k]}, {gp.stem})")
        keys[k] = gp.stem
        m = SOURCE_PDF.search(gp.filename)
        src = m.group(1) if m else gp.filename
        objs = [_pipe_name(o) for o in gp.objects]
        jobs.append({"stem": gp.stem, "mode": "plumbing", "objects": objs, "all_objects": objs, "size": (gp.page_width, gp.page_height),
                     "payload": str(latest_payload_path(payloads, gp.document_id, gp.page_number) or payloads / "missing"),
                     "group": f"plumbing:{src}", "val": src in VAL_PLUMBING,
                     "rec": {"stem": gp.stem, "gt_id": gp.gt_id, "document_id": gp.document_id, "filename": gp.filename,
                             "document_sha256": gp.sha256, "page_number": gp.page_number, "mode": "plumbing", "source": "ai_takeoff_gt"}})
    for p, pg, objs in viewer_pages(export_dir):
        stem = f"vw_{key_of(p['sha256'], pg['page'])}"
        m = SOURCE_PDF.search(p["name"])
        src = m.group(1) if m else p["name"]
        jobs.append({"stem": stem, "mode": "plumbing", "objects": objs, "all_objects": objs, "size": (pg.get("width_pt"), pg.get("height_pt")),
                     "payload": str(vw_payloads / f"{key_of(p['sha256'], pg['page'])}.json.gz"),
                     "group": f"plumbing:{src}", "val": src in VAL_PLUMBING,
                     "rec": {"stem": stem, "filename": p["name"], "document_sha256": p["sha256"], "page_number": int(pg["page"]),
                             "mode": "plumbing", "source": "viewer_plumbing38", "viewer_status": "completed"}})
    for j in jobs:
        j["out"] = str(out)
    return jobs, dropped


def fingerprints(out: Path, stems: list[str]) -> dict[str, set]:
    from make_fire_dataset import LONG

    fp = {}
    for s in stems:
        with np.load(out / f"{s}.npz") as z:
            seg, off, L = z["segments"], z["shape_offsets"], z["length"]
        keep = np.flatnonzero(L >= LONG)
        pts = np.r_[seg[off[keep]][:, :2], seg[off[keep + 1] - 1][:, 2:]]
        fp[s] = {tuple(p) for p in np.round(pts).astype(np.int64)}
    return fp


def shared_pairs(fp: dict[str, set]) -> list[tuple[str, str, float]]:
    """Pairs of pages sharing >= SAME_PROJECT of the smaller page's long-stroke endpoints (inverted index)."""
    index = defaultdict(list)
    for s, pts in fp.items():
        for p in pts:
            index[p].append(s)
    count = Counter()
    for ss in index.values():
        if 1 < len(ss) <= 50:
            for i, a in enumerate(ss):
                for b in ss[i + 1:]:
                    count[(a, b) if a < b else (b, a)] += 1
    out = []
    for (a, b), n in count.items():
        share = n / max(min(len(fp[a]), len(fp[b])), 1)
        if share >= SAME_PROJECT:
            out.append((a, b, round(share, 3)))
    return out


def viewer_vs_gt(out: Path, vw: str, gt: str, tol: float = 1.0) -> dict:
    """True strokes of a Viewer page against the GT page of the same drawing (same coordinates): length of each, and the share of the
    GT's true length within ``tol`` pt of a Viewer true stroke (recall of the Viewer labels against the GT) and the other way round."""
    import shapely

    def true_lines(stem):
        with np.load(out / f"{stem}.npz") as z:
            off, seg, y = z["shape_offsets"], z["segments"], z["label"].astype(bool)
        own = np.repeat(np.arange(len(off) - 1), np.diff(off))
        s = seg[y[own]].astype(np.float64)
        return shapely.linestrings(s.reshape(-1, 2, 2))

    a, b = true_lines(vw), true_lines(gt)
    if not len(a) or not len(b):
        return {"viewer_true_length": float(shapely.length(a).sum()), "gt_true_length": float(shapely.length(b).sum())}
    ua, ub = shapely.union_all(a).buffer(tol), shapely.union_all(b).buffer(tol)
    la, lb = shapely.length(a), shapely.length(b)
    return {"viewer_true_length": float(la.sum()), "gt_true_length": float(lb.sum()),
            "gt_true_covered_by_viewer": float(shapely.length(shapely.intersection(b, ua)).sum() / max(lb.sum(), 1e-9)),
            "viewer_true_covered_by_gt": float(shapely.length(shapely.intersection(a, ub)).sum() / max(la.sum(), 1e-9))}


def build(export_dir, vw_payloads, out, zip_out=None, workers="3"):
    export_dir, vw_payloads, out = Path(export_dir), Path(vw_payloads), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs, dropped = jobs_of(export_dir, vw_payloads, out)
    by_stem = {j["stem"]: j for j in jobs}
    done = {}
    prev = out / "records.partial.json"   # resumable: records of pages built by an earlier run (failed / missing pages are retried)
    if prev.exists():
        done = {r["stem"]: r for r in json.loads(prev.read_text(encoding="utf-8"))
                if (r["status"] in ("ok", "duplicate") and (out / f"{r['stem']}.npz").exists())
                or r["status"] not in ("ok", "duplicate", "missing_payload", "failed")}
    todo = [j for j in jobs if j["stem"] not in done]
    print(f"{len(jobs)} pages ({sum(j['mode'] == 'fire' for j in jobs)} FIRE GT, "
          f"{sum(j['rec']['source'] == 'ai_takeoff_gt' and j['mode'] == 'plumbing' for j in jobs)} PLUMBING GT, "
          f"{sum(j['stem'].startswith('vw_') for j in jobs)} Viewer); {len(todo)} to build", flush=True)
    with ProcessPoolExecutor(int(workers)) as ex:   # large pages need several GB: few workers
        futs = {ex.submit(build_page, j): j for j in todo}
        for i, f in enumerate(as_completed(futs), 1):
            j = futs[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001  (out of memory on a large page: rerun to retry it)
                r = j["rec"] | {"status": "failed", "error": f"{type(e).__name__}: {e}"[:200]}
            done[r["stem"]] = r
            o = r.get("objects", {})
            print(f"[{i}/{len(todo)}] {r['stem']} {r['status']} lines {o.get('lines_half_covered')}/{o.get('line_objects')} covered", flush=True)
            prev.write_text(json.dumps(list(done.values()), ensure_ascii=False, default=float), encoding="utf-8")
    failed = [s for s, r in done.items() if r["status"] in ("failed", "missing_payload")]
    if failed:
        raise SystemExit(f"{len(failed)} pages failed or have no payload ({failed[:5]}): run again")
    records = [done[j["stem"]] for j in jobs]
    for r in records:
        if r["status"] == "ok" and r["stem"].startswith("vw_") and r["objects"]["line_half_covered_share"] < MIN_COVERED:
            r["status"] = "rejected_coverage"
    known = [r for r in records if r["status"] in ("ok", "duplicate")]   # duplicates of an earlier run take part in the checks again
    ok = [r for r in records if r["status"] == "ok"]
    dropped += [{"stem": r["stem"], "reason": r["status"]} for r in records if r["status"] not in ("ok", "duplicate")]
    pairs = shared_pairs(fingerprints(out, [r["stem"] for r in known]))
    ok = known
    is_vw = lambda s: s.startswith("vw_")  # noqa: E731
    compare, gone = [], {}
    rec_of = {r["stem"]: r for r in ok}
    for a, b, share in sorted(pairs, key=lambda x: -x[2]):
        if share < DUPLICATE or a in gone or b in gone:
            continue
        if is_vw(a) != is_vw(b):
            vw, gt = (a, b) if is_vw(a) else (b, a)
            compare.append({"viewer": vw, "gt": gt, "shared_endpoints": share, **viewer_vs_gt(out, vw, gt)})
            gone[vw] = f"same drawing as GT page {gt} (GT kept)"
        else:   # two copies of one drawing from the same kind of source: keep the one with more true length
            keep, drop = sorted((a, b), key=lambda s: -rec_of[s]["stats"]["true_length"])
            gone[drop] = f"same drawing as {keep} (kept)"
    for s, why in sorted(gone.items()):
        dropped.append({"stem": s, "reason": why})
    ok = [r for r in ok if r["stem"] not in gone]
    parent = {r["stem"]: by_stem[r["stem"]]["group"] for r in ok}   # union of source-PDF groups that share a drawing
    root = {g: g for g in set(parent.values())}

    def find(g):
        while root[g] != g:
            root[g] = root[root[g]]
            g = root[g]
        return g

    merged = []
    for a, b, share in pairs:
        if share >= MERGE and a in parent and b in parent and find(parent[a]) != find(parent[b]):
            merged.append({"a": a, "b": b, "shared_endpoints": share, "groups": [parent[a], parent[b]]})
            root[find(parent[a])] = find(parent[b])
    comp_val = defaultdict(bool)
    for r in ok:
        comp_val[find(parent[r["stem"]])] |= by_stem[r["stem"]]["val"]
    plumb = [r for r in ok if r["mode"] == "plumbing"]
    n_val_pl = sum(comp_val[find(parent[r["stem"]])] for r in plumb)
    if n_val_pl > MAX_VAL_SHARE * max(len(plumb), 1):
        raise SystemExit(f"val holds {n_val_pl} / {len(plumb)} PLUMBING pages: a val group joined a large project, choose others")
    groups = {}
    for r in ok:
        comp = find(parent[r["stem"]])
        r["group"], r["split_group"] = parent[r["stem"]], comp
        r["split"] = "val" if comp_val[comp] else "train"
        g = groups.setdefault(r["document_sha256"], {"sha256": r["document_sha256"], "filename": r["filename"], "mode": r["mode"],
                                                     "group": comp, "pages": [], "fold": 0 if r["split"] == "val" else -1})
        if g["fold"] != (0 if r["split"] == "val" else -1):
            raise SystemExit(f"{r['filename']}: pages on both sides")
        g["pages"].append(r["page_number"])
    physical = [{"document_sha256": r["document_sha256"], "page_number": r["page_number"], "filename": r["filename"], "mode": r["mode"],
                 "gt_variants": [r["stem"]]} for r in ok]

    def cohort(r):
        return "viewer" if is_vw(r["stem"]) else f"gt_{r['mode']}"

    summary = defaultdict(lambda: defaultdict(float))
    for r in ok:
        c = summary[f"{cohort(r)}:{r['split']}"]
        c["pages"] += 1
        for k in ("line_objects", "line_objects_with_strokes", "lines_half_covered"):
            c[k] += r["objects"][k]
        c["true_shapes"] += r["stats"]["true_shapes"]
    from openstroke.objects import HVAC_CLASSES, LINE_CLASSES

    manifest = {"schema_version": "openstroke.e8.v1", "generator": "tools/make_thin_dataset.py", "split_plan": "thin1", "records": ok,
                "physical_pages": physical, "excluded": dropped, "summary": {k: dict(v) for k, v in sorted(summary.items())},
                "viewer": {"project": 38, "label_set": LABEL_SET, "status": "completed", "pipe_classes": "line1 .. line 8 (Mặt bằng, duct left out)"},
                "objects": {"line_classes": list(LINE_CLASSES), "line_key": "fcls", "hvac_classes": list(HVAC_CLASSES),
                            "ocls": "0 on every stroke (no thin-line class in the HVAC class head)", "rule": "openstroke/objects.py assign_lines"},
                "protocol": PROTOCOL, "viewer_vs_gt": compare, "merged_groups": merged,
                "cross_split_similar": [{"a": a, "b": b, "shared_endpoints": sh} for a, b, sh in pairs
                                        if a in parent and b in parent and comp_val[find(parent[a])] != comp_val[find(parent[b])]]}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    from make_fire_dataset import DUPLICATE_CHOICE

    n_val = sum(r["split"] == "val" for r in ok)
    (out / "folds.json").write_text(json.dumps({"status": "confirmed", "k": 1, "split_plan": "thin1", "duplicate_choice": DUPLICATE_CHOICE,
                                                "note": f"thin1: fold 0 = val ({n_val} pages: FIRE fire1 val + PLUMBING groups {', '.join(VAL_PLUMBING)}), "
                                                        f"fold -1 = train ({len(ok) - n_val}). One source PDF group (and every group sharing a "
                                                        "drawing with it) per side. Chosen by the agent under AGENTS 0.3, before any run.",
                                                "groups": list(groups.values())}, ensure_ascii=False, indent=1), encoding="utf-8")
    for s, why in gone.items():   # duplicates stay on disk (status "duplicate"): not rebuilt, not zipped
        done[s] = done[s] | {"status": "duplicate", "reason": why}
    prev.write_text(json.dumps(list(done.values()), ensure_ascii=False, default=float), encoding="utf-8")
    for k, v in sorted(summary.items()):
        print(k, {a: int(b) for a, b in v.items()})
    print(f"dropped {Counter(d['reason'] for d in dropped)}; merged groups {len(merged)}; viewer-vs-GT pairs {len(compare)}")
    if zip_out:
        with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_STORED) as zf:   # the manifest's pages only (rejected pages stay local)
            for name in ["manifest.json", "folds.json", *sorted(r["npz"] for r in ok)]:
                zf.write(out / name, f"{out.name}/{name}")
        print(f"{zip_out}: {Path(zip_out).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    {"extract": extract, "build": build}[sys.argv[1]](*sys.argv[2:])
