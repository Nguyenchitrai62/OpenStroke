"""E0 dataset: GT + payload -> one npz per (GT, page) with per-shape labels, and a self-describing manifest.

Run: ``py -3.12 -m openstroke build-e0`` (see ``openstroke/cli.py``).
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from shapely import STRtree

from . import __version__
from .geometry import IDENTITY, PageSegments, calibrate, page_segments, transform_points
from .labels import (HVAC_RULE, HVAC_TRUE_RE, LABEL_SCHEMA, refine_hvac, PLUMBING_TRUE_RE, TOL, TRUE_FRACTION, FIRE_TRUE_NAMES, gt_coverage, gt_outline_length,
                     gt_geometry, is_true_object, segment_lines, label_shapes)
from .sources import (GT_GROUPS, GTPage, file_sha256, group_by_page, latest_payload_path, load_gt_pages,
                      load_payload, payload_version)

SCHEMA_VERSION = "openstroke.e0.v1"
COORDINATE_FRAME = "display_top_left_pt"
TYPE_CODES = {"s": 0, "f": 1, "fs": 2}
OPS = ("l", "c", "re", "qu")

#: QA: a stroke style counts as "pipe-like" when at least this share of its length is true, and an
#: ``other`` shape of that style is "suspicious" when its midpoint is farther than ``FAR_PT`` from any GT.
PIPE_STYLE_SHARE = 0.5
PIPE_STYLE_MIN_TRUE = 20
FAR_PT = 25.0
GT_UNCOVERED = 0.2  # GT objects with less outline coverage than this have no stroke underneath
CELL_PT = 200.0


def _rgb(v) -> list[float]:
    if v is None:
        return [np.nan, np.nan, np.nan]
    v = list(v)[:3]
    return [float(x) for x in v] + [np.nan] * (3 - len(v))


def _num(v, default=np.nan) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def shape_attributes(shapes: list[dict], segs: PageSegments) -> dict[str, np.ndarray]:
    """Per-shape descriptors kept next to the labels so training never needs the payload JSON."""

    n = len(shapes)
    layers: dict[str, int] = {}
    layer_id = np.zeros(n, dtype=np.int32)
    type_code = np.full(n, 3, dtype=np.int8)
    stroke = np.zeros((n, 3), dtype=np.float32)
    fill = np.zeros((n, 3), dtype=np.float32)
    width = np.zeros(n, dtype=np.float32)
    stroke_op = np.zeros(n, dtype=np.float32)
    fill_op = np.zeros(n, dtype=np.float32)
    close = np.zeros(n, dtype=np.uint8)
    even_odd = np.zeros(n, dtype=np.uint8)
    dashed = np.zeros(n, dtype=np.uint8)
    seqno = np.zeros(n, dtype=np.int64)
    op_counts = np.zeros((n, len(OPS)), dtype=np.int32)
    for i, s in enumerate(shapes):
        layer_id[i] = layers.setdefault(str(s.get("layer")), len(layers))
        type_code[i] = TYPE_CODES.get(s.get("type"), 3)
        stroke[i] = _rgb(s.get("color"))
        fill[i] = _rgb(s.get("fill"))
        width[i] = _num(s.get("width"))
        stroke_op[i] = _num(s.get("stroke_opacity"))
        fill_op[i] = _num(s.get("fill_opacity"))
        close[i] = bool(s.get("closePath"))
        even_odd[i] = bool(s.get("even_odd"))
        dashed[i] = str(s.get("dashes")) not in ("[] 0", "None", "")
        seqno[i] = int(_num(s.get("seqno"), -1))
        for it in s.get("items") or ():
            if it and it[0] in OPS:
                op_counts[i, OPS.index(it[0])] += 1
    seg = segs.seg
    bbox = np.full((n, 4), np.nan, dtype=np.float64)
    if len(seg):
        x = np.minimum(seg[:, 0], seg[:, 2]), np.maximum(seg[:, 0], seg[:, 2])
        y = np.minimum(seg[:, 1], seg[:, 3]), np.maximum(seg[:, 1], seg[:, 3])
        has = np.diff(segs.shape_offsets) > 0
        starts = segs.shape_offsets[:-1][has]
        bbox[has, 0] = np.minimum.reduceat(x[0], starts)
        bbox[has, 1] = np.minimum.reduceat(y[0], starts)
        bbox[has, 2] = np.maximum.reduceat(x[1], starts)
        bbox[has, 3] = np.maximum.reduceat(y[1], starts)
    return {
        "layer_id": layer_id, "layer_names": np.array(list(layers), dtype=str), "type_code": type_code,
        "stroke_rgb": stroke, "fill_rgb": fill, "width": width, "stroke_opacity": stroke_op,
        "fill_opacity": fill_op, "close_path": close, "even_odd": even_odd, "dashed": dashed, "seqno": seqno,
        "op_counts": op_counts, "bbox": bbox.astype(np.float32),
    }


def _style_keys(attrs: dict[str, np.ndarray]) -> np.ndarray:
    """Integer id of (layer, stroke colour, width, type) per shape."""

    col = np.nan_to_num(attrs["stroke_rgb"], nan=-1.0)
    key = np.stack([attrs["layer_id"].astype(np.float64), np.round(col[:, 0], 3), np.round(col[:, 1], 3),
                    np.round(col[:, 2], 3), np.round(np.nan_to_num(attrs["width"], nan=-1.0), 2),
                    attrs["type_code"].astype(np.float64)], axis=1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return inv.reshape(-1)


def _style_label(attrs, idx_shape: int) -> str:
    names = attrs["layer_names"]
    c = attrs["stroke_rgb"][idx_shape]
    col = "none" if np.isnan(c).any() else "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c)
    w = attrs["width"][idx_shape]
    t = {0: "s", 1: "f", 2: "fs"}.get(int(attrs["type_code"][idx_shape]), "?")
    return f"{names[attrs['layer_id'][idx_shape]]} | {col} | w={w:.2f} | {t}"


def qa_page(segs: PageSegments, labels, attrs, gt, gt_cov: np.ndarray) -> dict[str, Any]:
    """Per-page QA numbers (AGENTS.md section 8, E0)."""

    n = len(labels.label)
    L = labels.length.astype(np.float64)
    true = labels.label.astype(bool)
    tot_len = float(L.sum())
    frac = labels.true_frac
    mixed = (frac > 0.1) & (frac < 0.9)
    q: dict[str, Any] = {
        "shapes": int(n),
        "segments": int(len(segs.seg)),
        "single_item_shapes": int((attrs["op_counts"].sum(axis=1) == 1).sum()),
        "true_shapes": int(true.sum()),
        "true_shape_share": float(true.mean()) if n else 0.0,
        "true_length_share": float(L[true].sum() / tot_len) if tot_len else 0.0,
        "true_length_pt": float(L[true].sum()),
        "total_length_pt": tot_len,
        "mixed_shapes_10_90": int(mixed.sum()),
        # length a whole-shape label gets wrong on mixed shapes, relative to the true length
        "mixed_mislabel_length_share": float(np.minimum(frac * L, (1 - frac) * L)[mixed].sum()
                                             / max(L[true].sum(), 1e-9)),
        "layers": int(len(attrs["layer_names"])),
        "dashed_attr_shapes": int(attrs["dashed"].sum()),
    }
    # GT covered by true strokes
    names = np.array(gt.names)
    q["gt_objects"] = int(len(gt))
    q["gt_by_class"] = {k: int(v) for k, v in zip(*np.unique(names, return_counts=True))} if len(gt) else {}
    gl = gt_outline_length(gt)
    q["gt_outline_length_pt"] = float(gl.sum())
    q["gt_length_coverage"] = float((gt_cov * gl).sum() / gl.sum()) if gl.sum() else 0.0
    unc = np.flatnonzero(gt_cov < GT_UNCOVERED)
    q["gt_uncovered"] = int(len(unc))
    q["gt_uncovered_length_share"] = float(gl[unc].sum() / gl.sum()) if gl.sum() else 0.0
    q["gt_uncovered_examples"] = [
        {"uid": gt.uids[i], "class": gt.names[i], "coverage": round(float(gt_cov[i]), 3),
         "bbox": [round(float(x), 1) for x in shapely.bounds(gt.geoms[i])]}
        for i in unc[np.argsort(-gl[unc])][:10]]
    # Suspicious "other": same style as many true strokes, but far from every GT object.
    style = _style_keys(attrs)
    ns = int(style.max()) + 1 if n else 0
    s_true = np.bincount(style, weights=np.where(true, L, 0.0), minlength=ns)
    s_all = np.bincount(style, weights=L, minlength=ns)
    s_true_n = np.bincount(style, weights=true.astype(float), minlength=ns)
    pipe_style = (s_true >= PIPE_STYLE_SHARE * np.maximum(s_all, 1e-9)) & (s_true_n >= PIPE_STYLE_MIN_TRUE)
    cand = np.flatnonzero(~true & pipe_style[style] & ~np.isnan(attrs["bbox"][:, 0]))
    susp = np.zeros(0, dtype=np.int64)
    if len(cand) and len(gt):
        b = attrs["bbox"][cand].astype(np.float64)
        mid = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)
        near = gt.tree().query(shapely.points(mid), predicate="dwithin", distance=FAR_PT)[0]
        far = np.ones(len(cand), dtype=bool)
        far[np.unique(near)] = False
        susp = cand[far]
    other_len = float(L[~true].sum())
    q["suspicious_other_shapes"] = int(len(susp))
    q["suspicious_other_length_pt"] = float(L[susp].sum())
    q["suspicious_vs_true_length"] = float(L[susp].sum() / max(L[true].sum(), 1e-9))
    q["suspicious_share_of_other_length"] = float(L[susp].sum() / other_len) if other_len else 0.0
    by_style = []
    if len(susp):
        ss = style[susp]
        for sk in np.unique(ss)[np.argsort(-np.bincount(ss, weights=L[susp])[np.unique(ss)])][:5]:
            members = susp[ss == sk]
            by_style.append({"style": _style_label(attrs, int(members[0])), "shapes": int(len(members)),
                             "length_pt": round(float(L[members].sum()), 1),
                             "style_true_length_share": round(float(s_true[sk] / s_all[sk]), 3)})
        b = attrs["bbox"][susp].astype(np.float64)
        cx = np.floor((b[:, 0] + b[:, 2]) / 2 / CELL_PT).astype(np.int64)
        cy = np.floor((b[:, 1] + b[:, 3]) / 2 / CELL_PT).astype(np.int64)
        cells: dict[tuple[int, int], list[int]] = {}
        for k, (a, c) in enumerate(zip(cx, cy)):
            cells.setdefault((int(a), int(c)), []).append(k)
        top = sorted(cells.items(), key=lambda kv: -L[susp[kv[1]]].sum())[:8]
        q["suspicious_regions"] = [
            {"bbox": [round(float(b[m, 0].min()), 1), round(float(b[m, 1].min()), 1),
                      round(float(b[m, 2].max()), 1), round(float(b[m, 3].max()), 1)],
             "shapes": len(m), "length_pt": round(float(L[susp[m]].sum()), 1)} for _, m in top]
    else:
        q["suspicious_regions"] = []
    q["suspicious_by_style"] = by_style
    q["suspicious_index"] = susp  # removed before JSON
    return q


def find_payload(gt_page: GTPage, drawings_dir: Path, extra_dirs=()) -> tuple[Path | None, str]:
    """AI_Takeoff's own cache first; else a payload written by ``tools/make_payloads.py`` (same generator)."""

    path = latest_payload_path(drawings_dir, gt_page.document_id, gt_page.page_number)
    if path is not None:
        return path, "ai_takeoff_cache"
    for d in extra_dirs:
        path = latest_payload_path(d, gt_page.document_id, gt_page.page_number)
        if path is not None:
            return path, "openstroke_generated"
    return None, ""


def build_gt_page(gt_page: GTPage, drawings_dir: Path, out_dir: Path, extra_dirs=()) -> dict[str, Any]:
    """Label one GT page, write its npz and return its manifest record."""

    t0 = time.time()
    rec: dict[str, Any] = {
        "stem": gt_page.stem, "gt_id": gt_page.gt_id, "document_id": gt_page.document_id,
        "filename": gt_page.filename, "document_sha256": gt_page.sha256, "page_number": gt_page.page_number,
        "mode": gt_page.mode, "gt_updated_at": gt_page.updated_at, "status": "ok",
        "gt_objects_primary_reference": len(gt_page.objects),
        "gt_objects_in_scope": sum(is_true_object(gt_page.mode, o.get("shape_name")) for o in gt_page.objects),
    }
    path, origin = find_payload(gt_page, drawings_dir, extra_dirs)
    if path is None:
        rec["status"] = "missing_payload"
        return rec
    rec["payload"] = {"path": str(path), "version": payload_version(path), "sha256": file_sha256(path), "origin": origin}
    payload = load_payload(path)
    shapes = payload.get("shapes") or []
    meta = payload.get("document_metadata") or {}
    width = float(meta.get("page_width") or gt_page.page_width or 0)
    height = float(meta.get("page_height") or gt_page.page_height or 0)
    rec["page_size"] = {"payload": [width, height], "gt": [gt_page.page_width, gt_page.page_height],
                        "payload_rotation": meta.get("rotation")}
    segs = page_segments(shapes)
    # Coordinate check: does the GT land on the payload's own strokes untransformed?
    gt0 = gt_geometry(gt_page.mode, gt_page.objects)
    ink = STRtree(segment_lines(segs.seg))
    cal = calibrate(ink, gt0.vertices, gt0.closed, width, height)
    rec["calibration"] = cal
    chosen = cal.get("chosen")
    if not cal.get("conclusive"):
        rec["status"] = "rejected_calibration"
        return rec
    if chosen != IDENTITY:
        gt_geo = gt_geometry(gt_page.mode, gt_page.objects,
                             transform=lambda v: transform_points(chosen, v, width, height))
    else:
        gt_geo = gt0
    rec["gt_transform_applied"] = chosen
    labels = label_shapes(segs, gt_geo)
    attrs = shape_attributes(shapes, segs)
    extra = {}
    if gt_page.mode == "hvac":
        labels, diag = refine_hvac(segs, gt_geo, labels, attrs["layer_id"])
        extra = {"hvac_outline_frac": diag["outline_frac"], "hvac_layer_inside_share": diag["layer_inside_share"],
                 "label_inside_only": diag["label_inside_only"]}
        rec["hvac_refine"] = {
            "rule": HVAC_RULE,
            "true_len_inside_only": float(labels.length[diag["label_inside_only"].astype(bool)].sum()),
            "true_len": float(labels.length[labels.label.astype(bool)].sum())}
    true_seg = segs.seg[labels.label[segs.owner].astype(bool)]
    cov = gt_coverage(gt_geo, true_seg)
    qa = qa_page(segs, labels, attrs, gt_geo, cov)
    susp = qa.pop("suspicious_index")
    rec["stats"] = qa
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{gt_page.stem}.npz"
    meta_json = json.dumps({k: rec[k] for k in ("stem", "gt_id", "document_id", "document_sha256", "page_number",
                                                  "mode", "payload", "gt_transform_applied")}
                           | {"schema_version": SCHEMA_VERSION, "label_schema": LABEL_SCHEMA,
                              "coordinate_frame": COORDINATE_FRAME}, ensure_ascii=False)
    suspicious = np.zeros(len(shapes), dtype=np.uint8)
    suspicious[susp] = 1
    np.savez_compressed(
        npz, meta=np.array(meta_json), label=labels.label, true_frac=labels.true_frac, length=labels.length,
        segments=segs.seg.astype(np.float32), shape_offsets=segs.shape_offsets,
        seg_inside=labels.seg_inside, qa_suspicious_other=suspicious,
        gt_class=np.array(gt_geo.names, dtype=str), gt_coverage=cov.astype(np.float32), **attrs, **extra)
    rec["npz"] = npz.name
    rec["npz_bytes"] = npz.stat().st_size
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def compare_pair(a: dict, b: dict, out_dir: Path, b_page: GTPage | None = None) -> dict[str, Any]:
    """How far two GT versions of one physical page disagree, in shape labels.

    When the two documents' payloads differ (another payload version of the same PDF), B's GT is labelled
    again on A's payload so both are compared on the same strokes.
    """

    res: dict[str, Any] = {"a": a["stem"], "b": b["stem"], "a_objects_in_scope": a.get("gt_objects_in_scope"),
                           "b_objects_in_scope": b.get("gt_objects_in_scope")}
    if a.get("status") != "ok" or b.get("status") != "ok":
        res["comparable"] = False
        res["reason"] = f"status {a.get('status')} / {b.get('status')}"
        return res
    za, zb = np.load(out_dir / a["npz"]), np.load(out_dir / b["npz"])
    res["a_gt_by_class"] = a["stats"]["gt_by_class"]
    res["b_gt_by_class"] = b["stats"]["gt_by_class"]
    res["a_true_length_pt"] = a["stats"]["true_length_pt"]
    res["b_true_length_pt"] = b["stats"]["true_length_pt"]
    same = (len(za["label"]) == len(zb["label"]) and za["segments"].shape == zb["segments"].shape
            and np.allclose(za["segments"], zb["segments"]))
    res["same_payload_shapes"] = bool(same)
    L = za["length"].astype(np.float64)
    ta = za["label"].astype(bool)
    if same:
        tb = zb["label"].astype(bool)
    elif b_page is not None:
        off = za["shape_offsets"]
        segs = PageSegments(seg=za["segments"].astype(np.float64), owner=np.repeat(np.arange(len(off) - 1), np.diff(off)),
                            shape_offsets=off)
        tb = label_shapes(segs, gt_geometry(b_page.mode, b_page.objects)).label.astype(bool)
        res["b_relabelled_on_a_payload"] = True
        res["b_true_length_on_a_payload_pt"] = float(L[tb].sum())
    else:
        res["comparable"] = False
        res["reason"] = "the two documents' payloads differ shape-by-shape; only totals are comparable"
        return res
    union = L[ta | tb].sum()
    res["comparable"] = True
    res["shapes_true_a_only"] = int((ta & ~tb).sum())
    res["shapes_true_b_only"] = int((tb & ~ta).sum())
    res["length_true_a_only_pt"] = float(L[ta & ~tb].sum())
    res["length_true_b_only_pt"] = float(L[tb & ~ta].sum())
    res["length_iou"] = float(L[ta & tb].sum() / union) if union else 1.0
    return res


def build_all(ai_takeoff_be: Path, out_dir: Path, only: set[str] | None = None, log=print,
              payload_dirs=(Path("datasets/payloads"),)) -> dict[str, Any]:
    db = ai_takeoff_be / "data" / "ai_takeoff.sqlite3"
    drawings = ai_takeoff_be / "data" / "drawings"
    pages = load_gt_pages(db)
    groups = group_by_page(pages)
    records: list[dict] = []
    physical: list[dict] = []
    for (sha, pno), variants in groups.items():
        entry = {"document_sha256": sha, "page_number": pno, "filename": variants[0].filename,
                 "mode": variants[0].mode, "gt_variants": [v.stem for v in variants],
                 "duplicate_gt": len(variants) > 1,
                 "decision": "pending_user_choice" if len(variants) > 1 else "single_gt"}
        physical.append(entry)
        for v in variants:
            if only and v.stem not in only and str(v.document_id) not in only:
                continue
            log(f"[{len(records) + 1}] {v.stem} {v.mode} {v.filename}")
            rec = build_gt_page(v, drawings, out_dir, extra_dirs=payload_dirs)
            rec["duplicate_gt"] = len(variants) > 1
            records.append(rec)
            s = rec.get("stats", {})
            log(f"    {rec['status']} cal={rec.get('calibration', {}).get('chosen')} "
                f"shapes={s.get('shapes')} true={s.get('true_shape_share', 0):.1%} "
                f"len={s.get('true_length_share', 0):.1%} gtcov={s.get('gt_length_coverage', 0):.1%} "
                f"{rec.get('seconds')}s")
    by_stem = {r["stem"]: r for r in records}
    gt_by_stem = {p.stem: p for p in pages}
    pairs = []
    for e in physical:
        if e["duplicate_gt"] and all(s in by_stem for s in e["gt_variants"]):
            a, b = (by_stem[s] for s in e["gt_variants"][:2])
            pairs.append({"filename": e["filename"], "page_number": e["page_number"],
                          **compare_pair(a, b, out_dir, b_page=gt_by_stem.get(b["stem"]))})
    st = db.stat()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "generator": f"openstroke {__version__} (openstroke.build.build_all)",
        "coordinate_frame": COORDINATE_FRAME,
        "label": {
            "schema": LABEL_SCHEMA, "unit": "payload shape (split_countfire shapes[i])", "values": {"1": "true", "0": "other"},
            "fire_true_classes": sorted(FIRE_TRUE_NAMES), "hvac_true_class_regex": HVAC_TRUE_RE.pattern, "hvac_rule": HVAC_RULE,
            "plumbing_true_class_regex": PLUMBING_TRUE_RE.pattern,
            "tolerance_pt": TOL, "true_if_length_fraction_ge": TRUE_FRACTION, "gt_groups": list(GT_GROUPS),
            "bezier": "sampled along the arc (8 chords)", "ignore_label": None,
        },
        "source": {"db": str(db), "db_size": st.st_size,
                   "db_mtime": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                   "drawings": str(drawings), "payload_dirs": [str(d) for d in payload_dirs],
                   "access": "read-only (sqlite mode=ro)"},
        "npz_fields": {
            "label": "(n,) uint8 per shape", "true_frac": "(n,) float32 length fraction in GT zone",
            "length": "(n,) float32 pt", "segments": "(m,4) float32 x0,y0,x1,y1, beziers flattened",
            "shape_offsets": "(n+1,) shape i owns segments[o[i]:o[i+1]]", "seg_inside": "(m,) inside length",
            "bbox": "(n,4)", "layer_id/layer_names": "layer index (page-local, grouping only)",
            "type_code": "0 s, 1 f, 2 fs, 3 other", "stroke_rgb/fill_rgb": "(n,3) nan if none",
            "width/stroke_opacity/fill_opacity/close_path/even_odd/dashed/seqno/op_counts(l,c,re,qu)": "payload attrs",
            "qa_suspicious_other": "QA flag only, not a label", "gt_class/gt_coverage": "per in-scope GT object",
            "meta": "JSON string",
        },
        "physical_pages": physical,
        "records": records,
        "duplicate_pairs": pairs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float),
                                           encoding="utf-8")
    return manifest


class PseudoPage(GTPage):
    """A page labelled from the AI_Takeoff pipeline's own output (``runs/job-*/normalized.json``) instead of a human
    GT (E8, user 2026-10-21: learn from the pipeline, fine-tune on GT). ``gt_id`` holds the job id."""

    @property
    def stem(self) -> str:
        return f"doc{self.document_id:06d}_p{self.page_number:04d}_ps{self.gt_id}"


def _pipeline_name(mode: str, obj: dict) -> dict:
    """The pipeline writes PLUMBING line names upper case and often with the typo "ÔNG" (``ỐNG DẪN…``, ``ÔNG CẤP…``);
    human GT always starts with "Ống" (0 of 19,338 GT objects differ). Map them onto the GT spelling."""

    import unicodedata

    name = unicodedata.normalize("NFC", obj.get("shape_name") or "").strip()
    if mode == "plumbing" and name.casefold().startswith(("ống", "ông ")):
        return {**obj, "shape_name": "Ống" + name[3:]}
    return obj


def build_pseudo(ai_takeoff_be: Path, pages_json: Path, out_dir: Path, log=print,
                 payload_dirs=(Path("datasets/payloads"),)) -> dict[str, Any]:
    """Pages listed in ``pages_json`` ({document_id, page_number, mode, job}) -> npz labelled from the pipeline's
    objects with the same rules and coordinate check as the GT build; manifest with ``pseudo`` records."""

    db = ai_takeoff_be / "data" / "ai_takeoff.sqlite3"
    drawings = ai_takeoff_be / "data" / "drawings"
    runs = ai_takeoff_be / "data" / "runs"
    from .sources import connect_ro

    con = connect_ro(db)
    docs = {d: (fn, sha) for d, fn, sha in con.execute("select id, filename, sha256 from documents")}
    con.close()
    want = json.loads(Path(pages_json).read_text(encoding="utf-8"))
    norm_cache: dict[int, dict] = {}
    records, physical = [], []
    for i, w in enumerate(want, 1):
        jid, doc, pno, mode = int(w["job"]), int(w["document_id"]), int(w["page_number"]), str(w["mode"])
        if jid not in norm_cache:
            norm_cache[jid] = json.loads((runs / f"job-{jid:06d}" / "normalized.json").read_text(encoding="utf-8"))
        npage = next((p for p in norm_cache[jid].get("pages") or [] if int(p.get("page_number", -1)) == pno), {})
        objs = [_pipeline_name(mode, o) for o in npage.get("objects") or [] if o.get("group") in GT_GROUPS]
        fn, sha = docs[doc]
        page = PseudoPage(gt_id=jid, document_id=doc, mode=mode, filename=fn, sha256=sha, page_number=pno,
                          page_width=npage.get("page_width"), page_height=npage.get("page_height"),
                          updated_at="", objects=objs)
        log(f"[{i}/{len(want)}] {page.stem} {mode} {fn}")
        rec = build_gt_page(page, drawings, out_dir, extra_dirs=payload_dirs)
        rec["pseudo"] = {"job": jid, "source": "AI_Takeoff pipeline runs/job-*/normalized.json"}
        records.append(rec)
        physical.append({"document_sha256": sha, "page_number": pno, "filename": fn, "mode": mode,
                         "gt_variants": [page.stem], "duplicate_gt": False, "decision": "pseudo_label"})
        s = rec.get("stats", {})
        log(f"    {rec['status']} shapes={s.get('shapes')} true len={s.get('true_length_share', 0):.1%} {rec.get('seconds')}s")
    manifest = {"schema_version": SCHEMA_VERSION, "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "generator": f"openstroke {__version__} (openstroke.build.build_pseudo)", "coordinate_frame": COORDINATE_FRAME,
                "label": {"schema": LABEL_SCHEMA, "source": "pipeline output (pseudo labels)", "hvac_rule": HVAC_RULE},
                "physical_pages": physical, "records": records, "duplicate_pairs": []}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return manifest


def merge_datasets(parts: list[Path], out_dir: Path, log=print) -> dict[str, Any]:
    """One dataset folder from several (e2 GT + pseudo): records / physical pages concatenated, npz and link graphs
    hard-linked (copied if links are not possible). A (sha256, page) present twice keeps the first part's pages."""

    import os
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    records, physical, seen = [], [], set()
    for part in parts:
        m = json.loads((part / "manifest.json").read_text(encoding="utf-8"))
        stems, n = set(), 0
        ok = {r["stem"] for r in m["records"] if r.get("status") == "ok"}
        for p in m["physical_pages"]:  # first usable occurrence wins, also inside one part (one PDF uploaded twice)
            key = (p["document_sha256"], p["page_number"])
            if key in seen or not ok & set(p["gt_variants"]):
                continue
            seen.add(key)
            physical.append(p)
            stems |= set(p["gt_variants"])
            n += 1
        for r in m["records"]:
            if r["stem"] not in stems:
                continue
            records.append(r)
            if r.get("status") != "ok":
                continue
            for f in [part / r["npz"], *part.glob(r["npz"].replace(".npz", ".graph.*.npz"))]:
                dst = out_dir / f.name
                if not dst.exists():
                    try:
                        os.link(f, dst)
                    except OSError:
                        shutil.copy2(f, dst)
        log(f"{part}: {n} physical pages")
    manifest = {"schema_version": SCHEMA_VERSION, "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "generator": f"openstroke {__version__} (openstroke.build.merge_datasets)", "parts": [str(p) for p in parts],
                "coordinate_frame": COORDINATE_FRAME, "physical_pages": physical, "records": records, "duplicate_pairs": []}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return manifest


def _unlabeled_one(src: str, dst: str, mode: str, stem: str) -> int:
    from .infer import payload_npz
    from .sources import load_payload

    return payload_npz(load_payload(src), Path(dst), mode, stem, compress=True)["shapes"]


def build_unlabeled(payload_dir: Path, out_dir: Path, mode: str = "fire", log=print, workers: int = 6) -> dict[str, Any]:
    """Payloads of unlabeled pages (``tools/make_unlabeled.py``, its ``index.json``) -> npz with every label ``other``
    and a manifest whose records carry ``unlabeled: true``: for self-supervised pretraining only (E19). A supervised fold
    file never lists them, so ``folds.resolve`` skips them there."""

    from concurrent.futures import ProcessPoolExecutor

    index = json.loads((payload_dir / "index.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    records, physical, futs = [], [], {}
    with ProcessPoolExecutor(workers) as ex:
        for e in index:
            stem = f"ul_{e['sha256'][:12]}_p{e['page_number']:04d}"
            src = payload_dir / e["payload"]
            rec = {"stem": stem, "document_sha256": e["sha256"], "page_number": e["page_number"], "mode": mode,
                   "filename": Path(e["file"]).name, "unlabeled": True, "status": "ok", "npz": f"{stem}.npz"}
            if not src.exists():
                rec["status"] = "missing_payload"
            elif not (out_dir / rec["npz"]).exists():
                futs[stem] = ex.submit(_unlabeled_one, str(src), str(out_dir / rec["npz"]), mode, stem)
            records.append(rec)
            physical.append({"document_sha256": e["sha256"], "page_number": e["page_number"], "filename": rec["filename"],
                             "mode": mode, "gt_variants": [stem], "duplicate_gt": False, "decision": "unlabeled"})
        for i, rec in enumerate(records, 1):
            if rec["stem"] in futs:
                try:
                    rec["shapes"] = futs[rec["stem"]].result()
                except Exception as ex_:  # a broken page must not stop the build
                    rec["status"] = f"error: {ex_!r}"[:200]
            log(f"[{i}/{len(records)}] {rec['stem']} {rec['status']} {rec.get('shapes', '')}")
    manifest = {"schema_version": SCHEMA_VERSION, "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "generator": f"openstroke {__version__} (openstroke.build.build_unlabeled)",
                "coordinate_frame": COORDINATE_FRAME, "physical_pages": physical, "records": records,
                "duplicate_pairs": []}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return manifest
