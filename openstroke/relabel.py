"""Labels v3 (research/HYPOTHESES.md E53): the strokes that make up a GT object, not every stroke lying inside it.

A stroke counts when most of it follows an edge of a GT object (``labels.along_gt_edges``). Everything else on the page comes from the built npz; only the GT objects are read again.
"""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Mapping

import numpy as np

from .build import _pipeline_name
from .geometry import IDENTITY, PageSegments, transform_points
from .labels import GTGeometry, TRUE_FRACTION, along_gt_edges, gt_coverage, gt_geometry, gt_outline_length, label_shapes
from .postprocess import OP_CURVE, bridge_curves, compact_strokes, fill_strokes, mesh_strokes, style_groups
from .sources import GT_GROUPS, load_gt_pages

LABEL_SCHEMA_V3 = "true_other_v3b"  # v3 plus compact symbols and groups the GT does not need out (E55)
GROUP_SHARE = 0.1  # F51: a style or layer with less of its length on the GT is not what the GT was drawn on
STYLE_KEYS = ("layer_id", "stroke_rgb", "width", "type_code")
NEEDED_SHARE, NEEDED_GAIN = 0.5, 0.1  # E55: a group mostly off the GT must add this share of its true length to the GT's coverage


def page_segments_of(d: Mapping) -> PageSegments:
    off = d["shape_offsets"].astype(np.int64)
    return PageSegments(seg=d["segments"].astype(np.float64), owner=np.repeat(np.arange(len(off) - 1), np.diff(off)), shape_offsets=off)


def _share(mask: np.ndarray, length: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Per shape: the share of its group's length that is in ``mask``."""

    n = int(group.max()) + 1 if len(group) else 1
    return (np.bincount(group[mask], length[mask], n) / np.maximum(np.bincount(group, length, n), 1e-9))[group]


def unneeded_groups(segs: PageSegments, geo: GTGeometry, mask: np.ndarray, length: np.ndarray, group: np.ndarray) -> np.ndarray:
    """The part of ``mask`` the GT does not need: groups mostly off the GT whose strokes in ``mask`` add almost nothing to its coverage, such as a grid line or a wall lying along a pipe drawn on its own layer. Worst group first, so of two groups that repeat each other one stays."""

    outline = gt_outline_length(geo)
    cov = lambda m: float((gt_coverage(geo, segs.seg[m[segs.owner]]) * outline).sum())  # noqa: E731
    n = int(group.max()) + 1 if len(group) else 1
    share = np.bincount(group[mask], length[mask], n) / np.maximum(np.bincount(group, length, n), 1e-9)
    keep = mask.copy()
    base = cov(keep)
    for i in sorted(np.flatnonzero((share > 0) & (share < NEEDED_SHARE)), key=lambda i: share[i]):
        without = keep & (group != i)
        now = cov(without)
        if base - now < NEEDED_GAIN * length[keep & (group == i)].sum():
            keep, base = without, now
    return mask & ~keep


def labels_v3(d: Mapping, mode: str, geo: GTGeometry) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """-> (label, the reason each v2-true or new shape got its label)."""

    length = d["length"].astype(np.float64)
    segs = page_segments_of(d)
    curve = d["op_counts"][:, OP_CURVE] > 0
    on_gt = along_gt_edges(segs, geo) >= TRUE_FRACTION
    zone = d["label_inside_only"].astype(bool) if mode == "hvac" else None
    if zone is not None:
        on_gt &= zone
    group = style_groups({k: d[k] for k in STYLE_KEYS})
    stray = on_gt & (_share(on_gt, length, group) < GROUP_SHARE)
    mesh = mesh_strokes(segs, group, length)
    fill = fill_strokes(segs, group, length) | compact_strokes(segs, length)
    core = on_gt & ~stray & ~mesh & ~fill
    core &= ~unneeded_groups(segs, geo, core, length, group)
    bridged = bridge_curves(segs, core, curve) & ~mesh & ~fill
    bridged &= _share(core, length, d["layer_id"].astype(np.int64)) >= GROUP_SHARE
    if zone is not None:
        bridged &= zone
    return core | bridged, {"on_gt": on_gt, "stray": stray, "mesh": mesh, "fill": fill, "bridged": bridged}


def object_strokes(mode: str, objects: list[dict], segs: PageSegments, attrs: Mapping, length: np.ndarray) -> np.ndarray:
    """The strokes that make up a system's objects: what a GT of those objects would label true."""

    geo = gt_geometry(mode, objects)
    d = {k: attrs[k] for k in (*STYLE_KEYS, "op_counts")} | {"length": length, "segments": segs.seg, "shape_offsets": segs.shape_offsets}
    if mode == "hvac":
        d["label_inside_only"] = label_shapes(segs, geo).label
    return labels_v3(d, mode, geo)[0]


def _page_objects(rec: dict, ai_takeoff_be: Path, gt_pages: dict) -> list[dict]:
    """The objects a built page was labelled from: its GT, or for a pseudo page its pipeline run."""

    if not rec.get("pseudo"):
        return gt_pages[rec["stem"]].objects
    run = ai_takeoff_be / "data" / "runs" / f"job-{rec['pseudo']['job']:06d}" / "normalized.json"
    page = next(p for p in json.loads(run.read_text(encoding="utf-8")).get("pages") or [] if int(p.get("page_number", -1)) == rec["page_number"])
    return [_pipeline_name(rec["mode"], o) for o in page.get("objects") or [] if o.get("group") in GT_GROUPS]


def _relabel_page(job: tuple) -> dict:
    src, out, rec, objects = job
    z = np.load(src / rec["npz"])
    d = {k: z[k] for k in z.files}
    w, h = rec["page_size"]["payload"]
    t = rec.get("gt_transform_applied")
    geo = gt_geometry(rec["mode"], objects, transform=None if t == IDENTITY else (lambda v: transform_points(t, v, w, h)))
    label, _ = labels_v3(d, rec["mode"], geo)
    meta = json.loads(str(d["meta"])) | {"label_schema": LABEL_SCHEMA_V3}
    tmp = out / f"{rec['stem']}.tmp.npz"
    np.savez_compressed(tmp, **{**d, "meta": np.array(json.dumps(meta, ensure_ascii=False)), "label": label.astype(np.uint8), "label_v2": d["label"]})
    tmp.replace(out / rec["npz"])
    L = d["length"].astype(np.float64)
    return {"stem": rec["stem"], "mode": rec["mode"], "true_v2": float(L[d["label"].astype(bool)].sum()), "true_v3": float(L[label].sum())}


def relabel_dataset(src: Path, out: Path, ai_takeoff_be: Path, workers: int = 4, log=print) -> dict:
    """``src`` -> ``out`` with labels v3: the same pages, fold files and graphs, other arrays untouched. ``src`` is not changed; AI_Takeoff is only read. Finished pages are kept, so a stopped run resumes."""

    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    gt_pages = {p.stem: p for p in load_gt_pages(ai_takeoff_be / "data" / "ai_takeoff.sqlite3")}
    records = [r for r in manifest["records"] if r.get("status") == "ok"]
    jobs = [(src, out, r, _page_objects(r, ai_takeoff_be, gt_pages)) for r in records if not (out / r["npz"]).exists()]
    log(f"{len(records)} pages, {len(jobs)} to do")
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(_relabel_page, jobs), 1):
            log(f"[{i}/{len(jobs)}] {r['stem']} {r['mode']} true {r['true_v2'] / 1e3:.0f}k -> {r['true_v3'] / 1e3:.0f}k pt")
    for f in src.iterdir():
        target = out / f.name
        if target.exists() or (f.suffix == ".npz" and ".graph." not in f.name) or f.is_dir() or f.name == "manifest.json":
            continue
        try:
            os.link(f, target)  # graphs are large and do not depend on labels
        except OSError:
            shutil.copy2(f, target)
    manifest["label"] = {**manifest.get("label", {}), "schema": LABEL_SCHEMA_V3, "rule": "research/HYPOTHESES.md E53, E55", "relabelled_from": str(src)}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return manifest
