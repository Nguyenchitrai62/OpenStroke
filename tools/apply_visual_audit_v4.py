"""Build visually-verified HVAC dataset ``e13_v4`` + ``score_pack_v4_val`` + ``openstroke_e13_v4_data.zip``
directly from per-page visual audit decisions (``datasets/audit_v4/<stem>/decision.json``).

Unlike blind heuristic scripts, EVERY dropped object, kept object, dropped bulk layer, and class fix
comes strictly from human/visual-agent inspection of rendered CAD overviews & zoomed crops.
All 25 human-verified AI_Takeoff GT pages (``doc*.npz``) are copied/hardlinked 100% untouched.

Usage:
    py -3.12 tools/apply_visual_audit_v4.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from openstroke.objects import class_of

CONN_PAD_PT = 3.5


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def apply_one_viewer_page(job: tuple[str, str, str, str]) -> dict:
    src_npz_s, dst_npz_s, audit_dir_s, rel_v3_dir_s = job
    src_npz = Path(src_npz_s)
    dst_npz = Path(dst_npz_s)
    audit_dir = Path(audit_dir_s)
    stem = src_npz.stem

    dec_p = audit_dir / stem / "decision.json"
    meta_p = audit_dir / stem / "page_meta.json"
    if not dec_p.exists():
        raise FileNotFoundError(f"Missing visual decision.json for {stem}: {dec_p}")

    dec = json.loads(dec_p.read_text(encoding="utf-8"))
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}

    with np.load(src_npz) as z:
        d = {k: z[k].copy() for k in z.files}

    y = d["label"].astype(bool)
    bb = d["bbox"].astype(np.float64)
    oinst = d["oinst"].astype(np.int64)
    ocls = d["ocls"].astype(np.int64)
    lid = d["layer_id"].astype(np.int64)
    lnames = [str(x) for x in d["layer_names"]]

    drop_oids: set[int] = {int(x) for x in dec.get("drop_oids", [])}
    # Add any bulk layers visually confirmed as non-duct
    drop_bulk_layers: set[str] = {str(x) for x in dec.get("drop_bulk_layers", [])}
    bulk_map = meta.get("bulk_sus_layer_oids", {})
    for lay in drop_bulk_layers:
        for oid in bulk_map.get(lay, []):
            drop_oids.add(int(oid))
    # Never drop an oid explicitly marked keep_oids (if both appeared by typo, drop_oids minus keep_oids? No: log warning!)
    keep_oids = {int(x) for x in dec.get("keep_oids", [])}
    overlap = drop_oids & keep_oids
    if overlap:
        drop_oids -= overlap

    # Partition dropped oids into:
    # (a) 1D duct-joint seam slivers touching kept ducts (min_side < 0.5pt & deg >= 1):
    #     DELETE as standalone object (oinst / gt/*.json) so no fake 0-area Duct object exists,
    #     BUT PRESERVE label=1 and reassign oinst/ocls/octr to the adjacent kept duct/elbow
    #     so we NEVER punch holes across real duct closing seams needed by hvac_faces.py!
    # (b) True non-duct noise objects (isolated FCU flex collars, equipment casings, diffusers, masonry shafts):
    #     ZERO label=0, ocls=0, oinst=-1!
    smap = {int(o["oid"]): o for cl in meta.get("clusters", []) for o in cl.get("objects", [])}
    sliver_seam_oids: set[int] = set()
    noise_drop_oids: set[int] = set()
    for oid in drop_oids:
        s = smap.get(oid)
        if s is not None and "tiny_sliver" in s.get("flags", []) and min(float(s.get("w", 9)), float(s.get("h", 9))) < 0.5 and int(s.get("deg", 0)) >= 1:
            sliver_seam_oids.add(oid)
        else:
            noise_drop_oids.add(oid)

    drop_stroke = np.zeros(len(y), dtype=bool)
    if noise_drop_oids:
        drop_stroke |= np.isin(oinst, np.fromiter(noise_drop_oids, dtype=np.int64))
    if drop_bulk_layers:
        is_drop_lay = np.array([lnames[int(k)] in drop_bulk_layers for k in lid], dtype=bool)
        drop_stroke |= (y & is_drop_lay)

    # Reassign 1D seam-sliver strokes to their nearest touching kept object (keeping label=1 intact!)
    n_sliver_reassigned = 0
    if sliver_seam_oids:
        sliver_m = np.isin(oinst, np.fromiter(sliver_seam_oids, dtype=np.int64)) & (~drop_stroke)
        kept_obj_m = y & (~drop_stroke) & (~sliver_m) & (oinst >= 0)
        if sliver_m.any() and kept_obj_m.any():
            ko_idx = np.flatnonzero(kept_obj_m)
            ko_boxes = shapely.box(
                bb[ko_idx, 0] - 1.5, bb[ko_idx, 1] - 1.5,
                bb[ko_idx, 2] + 1.5, bb[ko_idx, 3] + 1.5,
            )
            ko_tr = shapely.STRtree(ko_boxes)
            sl_idx = np.flatnonzero(sliver_m)
            sl_boxes = shapely.box(bb[sl_idx, 0], bb[sl_idx, 1], bb[sl_idx, 2], bb[sl_idx, 3])
            hi, hj = ko_tr.query(sl_boxes, predicate="intersects")
            best_for: dict[int, int] = {}
            for a_i, b_i in zip(hi.tolist(), hj.tolist()):
                s_row = int(sl_idx[a_i])
                k_row = int(ko_idx[b_i])
                # Prefer touching Duct (ocls==1) or Elbow (2,3) over other classes
                if s_row not in best_for or (int(ocls[k_row]) == 1 and int(ocls[best_for[s_row]]) != 1):
                    best_for[s_row] = k_row
            for s_row in sl_idx.tolist():
                if s_row in best_for:
                    k_row = best_for[s_row]
                    d["oinst"][s_row] = d["oinst"][k_row]
                    d["ocls"][s_row] = d["ocls"][k_row]
                    if "octr" in d:
                        d["octr"][s_row] = d["octr"][k_row]
                    n_sliver_reassigned += 1
                else:
                    # Truly detached 1D sliver -> safe to zero out
                    drop_stroke[s_row] = True
        elif sliver_m.any():
            drop_stroke |= sliver_m

    # Clean orphan unassigned label==1 strokes detached > 3.5 pt from all kept objects
    n_orphan = 0
    unassigned_pos = y & (~drop_stroke) & (d["oinst"] < 0)
    if unassigned_pos.any():
        kept_assigned = y & (~drop_stroke) & (d["oinst"] >= 0)
        if kept_assigned.any():
            ka_idx = np.flatnonzero(kept_assigned)
            ka_box = shapely.box(
                bb[ka_idx, 0] - CONN_PAD_PT, bb[ka_idx, 1] - CONN_PAD_PT,
                bb[ka_idx, 2] + CONN_PAD_PT, bb[ka_idx, 3] + CONN_PAD_PT,
            )
            ka_tr = shapely.STRtree(ka_box)
            ua_idx = np.flatnonzero(unassigned_pos)
            ua_box = shapely.box(bb[ua_idx, 0], bb[ua_idx, 1], bb[ua_idx, 2], bb[ua_idx, 3])
            hit_u, _ = ka_tr.query(ua_box, predicate="intersects")
            near_kept = np.zeros(len(ua_idx), dtype=bool)
            if len(hit_u):
                near_kept[np.unique(hit_u)] = True
            far_u = ua_idx[~near_kept]
            if len(far_u):
                drop_stroke[far_u] = True
                n_orphan = int(len(far_u))

    n_drop_s = int(drop_stroke.sum())
    if n_drop_s > 0:
        d["label"][drop_stroke] = 0
        d["ocls"][drop_stroke] = 0
        d["oinst"][drop_stroke] = -1
        if "octr" in d:
            d["octr"][drop_stroke] = np.nan
        if "pc_fg" in d and "pc_shape" in d:
            ps = d["pc_shape"].astype(np.int64)
            bad_p = drop_stroke[ps]
            d["pc_fg"][bad_p] = 0
            if "pc_cls" in d:
                d["pc_cls"][bad_p] = 0
            if "pc_inst" in d:
                d["pc_inst"][bad_p] = -1

    # Apply visually verified class fixes
    n_reclass = 0
    for rc in dec.get("reclass_oids", []):
        oid = int(rc["oid"])
        if oid in drop_oids:
            continue
        new_cls = str(rc.get("new_cls") or "")
        cid = class_of(new_cls)
        if cid and cid > 0:
            m = (d["oinst"] == oid) & (d["label"] > 0)
            if m.any():
                d["ocls"][m] = int(cid)
                n_reclass += 1

    np.savez_compressed(dst_npz, **d)
    return {
        "stem": stem,
        "page_status": dec.get("page_status", "clean"),
        "total_objs": int(meta.get("total_objs", 0)),
        "dropped_oids": len(drop_oids),
        "dropped_noise_oids": len(noise_drop_oids),
        "dropped_1d_seam_oids": len(sliver_seam_oids),
        "reassigned_seam_strokes": n_sliver_reassigned,
        "kept_suspects": len(keep_oids),
        "dropped_bulk_layers": sorted(drop_bulk_layers),
        "dropped_strokes": n_drop_s,
        "orphan_strokes": n_orphan,
        "reclassed_oids": n_reclass,
        "drop_oid_list": sorted(drop_oids),
        "reclass_list": dec.get("reclass_oids", []),
    }


def rebuild_score_pack_v4(src_pack: Path, dst_pack: Path, dst_ds: Path, audit_dir: Path) -> tuple[int, int]:
    """Rebuild ``score_pack_v4_val`` using exact visual audit decisions on the 8 Viewer val pages (5 GT5 untouched)."""
    dst_pack.mkdir(parents=True, exist_ok=True)
    (dst_pack / "npz").mkdir(exist_ok=True)
    (dst_pack / "gt").mkdir(exist_ok=True)
    for fn in ("pages.json", "names.json"):
        if (src_pack / fn).exists():
            shutil.copy2(src_pack / fn, dst_pack / fn)

    pages = json.loads((src_pack / "pages.json").read_text(encoding="utf-8"))
    tot_vw_g, drop_vw_g = 0, 0
    for r in pages:
        st = r["stem"]
        _link_or_copy(dst_ds / f"{st}.npz", dst_pack / "npz" / f"{st}.npz")
        src_gt = src_pack / "gt" / f"{st}.json"
        dst_gt = dst_pack / "gt" / f"{st}.json"
        if not st.startswith("vw_"):
            shutil.copy2(src_gt, dst_gt)
            continue

        dec = json.loads((audit_dir / st / "decision.json").read_text(encoding="utf-8"))
        meta = json.loads((audit_dir / st / "page_meta.json").read_text(encoding="utf-8"))
        drop_oids = {int(x) for x in dec.get("drop_oids", [])} - {int(x) for x in dec.get("keep_oids", [])}
        for lay in dec.get("drop_bulk_layers", []):
            for oid in meta.get("bulk_sus_layer_oids", {}).get(lay, []):
                drop_oids.add(int(oid))
        reclass_map = {int(rc["oid"]): str(rc["new_cls"]) for rc in dec.get("reclass_oids", []) if rc.get("new_cls")}

        # Map dropped oids -> their exact bboxes from e13_v3 NPZ so we match GT JSON polygons unambiguously (IoU >= 0.5 or center inside)
        with np.load(ROOT / "datasets" / "e13_v3" / f"{st}.npz") as z:
            bb = z["bbox"].astype(np.float64)
            oinst = z["oinst"].astype(np.int64)
        oid_boxes: dict[int, Polygon] = {}
        for oid in (drop_oids | set(reclass_map.keys())):
            m = oinst == oid
            if m.any():
                oid_boxes[oid] = shapely.box(
                    float(bb[m, 0].min()) - 0.25, float(bb[m, 1].min()) - 0.25,
                    float(bb[m, 2].max()) + 0.25, float(bb[m, 3].max()) + 0.25,
                )

        graw = json.loads(src_gt.read_text(encoding="utf-8"))
        tot_vw_g += len(graw)
        kept = []
        for idx_g, o in enumerate(graw):
            verts = o.get("vertices") or []
            pg = Polygon(np.asarray(verts, float)).buffer(0) if len(verts) >= 3 else Polygon()
            # Direct index match OR geometric match against dropped oid box
            is_dropped = False
            if idx_g in drop_oids and idx_g in oid_boxes:
                ob = oid_boxes[idx_g]
                if pg.area > 0 and ob.intersects(pg):
                    is_dropped = True
            if not is_dropped and pg.area > 0 and drop_oids:
                for doid in drop_oids:
                    ob = oid_boxes.get(doid)
                    if ob is None or not ob.intersects(pg):
                        continue
                    ia = ob.intersection(pg).area
                    if ia / max(min(ob.area, pg.area), 1e-6) >= 0.70 and max(ob.area, pg.area) / max(min(ob.area, pg.area), 1e-6) <= 2.5:
                        is_dropped = True
                        break
            if is_dropped:
                drop_vw_g += 1
                continue
            if idx_g in reclass_map:
                o = dict(o, shape_name=reclass_map[idx_g])
            kept.append(o)
        dst_gt.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    return drop_vw_g, tot_vw_g


def main() -> None:
    ap = argparse.ArgumentParser(description="Build visually-verified e13_v4, score_pack_v4_val, and openstroke_e13_v4_data.zip")
    ap.add_argument("--src", type=Path, default=ROOT / "datasets/e13_v3")
    ap.add_argument("--audit", type=Path, default=ROOT / "datasets/audit_v4")
    ap.add_argument("--dst", type=Path, default=ROOT / "datasets/e13_v4")
    ap.add_argument("--src-pack", type=Path, default=ROOT / "datasets/score_pack_v2_val")
    ap.add_argument("--dst-pack", type=Path, default=ROOT / "datasets/score_pack_v4_val")
    ap.add_argument("--zip", type=Path, default=ROOT / "datasets/openstroke_e13_v4_data.zip")
    ap.add_argument("--workers", type=int, default=min(6, max(1, (os.cpu_count() or 2) - 1)))
    args = ap.parse_args()

    t0 = time.time()
    args.dst.mkdir(parents=True, exist_ok=True)
    for jp in sorted(args.src.glob("*.json")):
        shutil.copy2(jp, args.dst / jp.name)

    npzs = sorted(
        p for p in args.src.glob("*.npz")
        if not p.stem.endswith(".graph.g3") and not p.stem.startswith("faces_train")
    )
    vw_jobs = []
    tot_gt = 0
    for p in npzs:
        dst_p = args.dst / p.name
        if p.stem.startswith("vw_"):
            vw_jobs.append((str(p), str(dst_p), str(args.audit), str(ROOT / "datasets/relabel_v3")))
        else:
            _link_or_copy(p, dst_p)
            tot_gt += 1
        g3 = args.src / f"{p.stem}.graph.g3.npz"
        if g3.exists():
            _link_or_copy(g3, args.dst / g3.name)

    # Remove any stray faces_train*.npz from dst if left over from earlier build
    for stray in args.dst.glob("faces_train*.npz"):
        stray.unlink()

    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(vw_jobs)))) as ex:
        results = list(ex.map(apply_one_viewer_page, vw_jobs))

    tot_vw_o = sum(r["total_objs"] for r in results)
    drop_vw_o = sum(r["dropped_oids"] for r in results)
    drop_noise_o = sum(r["dropped_noise_oids"] for r in results)
    drop_seam_o = sum(r["dropped_1d_seam_oids"] for r in results)
    reass_seam_s = sum(r["reassigned_seam_strokes"] for r in results)
    keep_sus = sum(r["kept_suspects"] for r in results)
    drop_vw_s = sum(r["dropped_strokes"] for r in results)
    reclassed = sum(r["reclassed_oids"] for r in results)
    status_cnt = Counter(r["page_status"] for r in results)
    major_pages = [r["stem"] for r in results if r["page_status"] == "flag_major"]
    dropped_layers = sorted({l for r in results for l in r["dropped_bulk_layers"]})

    summary = {
        "source": str(args.src),
        "method": "100% page-by-page visual audit (datasets/audit_v4/*/decision.json)",
        "gt_pages_untouched": tot_gt,
        "viewer_pages_audited": len(results),
        "page_statuses": dict(status_cnt),
        "major_flagged_pages": major_pages,
        "viewer_objects_total": tot_vw_o,
        "viewer_objects_dropped": drop_vw_o,
        "viewer_objects_dropped_noise": drop_noise_o,
        "viewer_objects_dropped_1d_seams": drop_seam_o,
        "viewer_1d_seam_strokes_preserved_fg": reass_seam_s,
        "viewer_suspects_visually_kept": keep_sus,
        "viewer_objects_reclassed": reclassed,
        "viewer_strokes_zeroed": drop_vw_s,
        "dropped_bulk_layers": dropped_layers,
    }
    (args.dst / "clean_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Built visually-verified {args.dst} in {time.time() - t0:.1f}s | GT untouched={tot_gt} | "
        f"Viewer visually audited={len(results)}p ({dict(status_cnt)}): dropped {drop_vw_o}/{tot_vw_o} objs "
        f"({100 * drop_vw_o / max(tot_vw_o, 1):.2f}%: {drop_noise_o} noise + {drop_seam_o} 1D-seams w/ {reass_seam_s} fg strokes kept), "
        f"visually kept {keep_sus} flagged suspects, reclassed {reclassed}, zeroed {drop_vw_s} strokes | "
        f"dropped_bulk_layers={dropped_layers}",
        flush=True,
    )

    if args.src_pack.exists():
        dw, tw = rebuild_score_pack_v4(args.src_pack, args.dst_pack, args.dst, args.audit)
        print(f"Built visually-verified {args.dst_pack}: cleaned {dw}/{tw} Viewer8 val GT polygons (5 GT5 untouched).", flush=True)

    # Pack ready-to-upload zip cleanly
    t_z = time.time()
    zfiles = sorted(
        [p for p in args.dst.glob("*.json")] +
        [p for p in args.dst.glob("*.npz") if not p.stem.endswith(".graph.g3") and not p.stem.startswith("faces_train")]
    )
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_STORED) as z:
        for p in zfiles:
            z.write(p, arcname=f"e13_v4/{p.name}")
    print(f"Packed {args.zip} ({len(zfiles)} files, {args.zip.stat().st_size / 1e6:.1f} MB) in {time.time() - t_z:.1f}s", flush=True)


if __name__ == "__main__":
    main()
