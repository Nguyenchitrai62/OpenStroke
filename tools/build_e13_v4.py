"""Build cleaned HVAC dataset ``e13_v4`` from ``e13_v3`` (F241 / OpenStroke44, user 2026-11-25).

Applies strictly physical MEP route-connectivity cleaning to **Viewer pages ONLY** (``vw_*.npz``), leaving all 25
human-verified AI_Takeoff GT pages (``doc*.npz``) **100% untouched** (copied/hardlinked verbatim):
  - Keeps real connected return-air duct routes (including ``0874-MVAC`` ``JRP-MVAC-RAG`` which visual inspection
    confirmed are full connected ``Duct + Elbow + Reducer + Chân rẽ`` routes — NEVER drops by layer name).
  - Enforces physical MEP route connectivity (using ``CONN_PAD_PT = 3.5 pt`` bbox padding to bridge flange/dash gaps;
    confirmed by user 2026-11-25: "duct ko bao giờ nằm 1 mình, nó phải có fitting hoặc chân rẽ gắn vào, nếu nằm đơn lẻ
    thì hầu hết là noise"):
      1. ``R1_iso_fitting``: standalone fitting / ``Chân rẽ`` with 0 connected neighbors (e.g., FCU flex collars on
         VIAS TOWER & ACU train pages; 0 / 638 on all 25 human GT pages);
      2. ``R2_iso_duct``: standalone ``Duct`` with 0 connected neighbors (no fitting, no ``Chân rẽ``, no other duct
         attached — removes standalone FCU plenum boxes & floating short stubs);
      3. ``R3_nonroute_cc``: tiny detached component with NO Duct (``<= 2`` touching FCU collars) OR a detached
         ``Duct``-only component with ZERO fittings/``Chân rẽ`` and ``span < 60 pt`` (preserves all legitimate
         ``(Duct + Chân rẽ)`` and ``(Duct + Fitting)`` routes);
      4. ``R4_orphan_pos_strokes``: unassigned ``label==1, oinst==-1`` strokes detached ``> 3.5 pt`` from all kept objects.
  - Optionally rebuilds ``score_pack_v4_val`` from ``score_pack_v2_val`` using the exact same physical rule on the 8
    Viewer val GT JSONs + cleaned NPZs (5 GT5 pages untouched).

Self-contained (zero SQLite / external DB dependency): runs locally OR directly on Colab in ~20 seconds:
    python tools/build_e13_v4.py --src /content/os41/e13_v3 --dst /content/os41/e13_v4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import Polygon

CONN_PAD_PT = 3.5
NOFIT_DUCT_CC_MAX_SPAN = 60.0
NODUCT_FIT_CC_MAX_OBJS = 2


def _classify_dropped_objects(
    o_bb: np.ndarray,
    o_is_duct: np.ndarray,
) -> tuple[np.ndarray, list[str | None]]:
    """Apply the exact physical MEP route-connectivity rule to ``n_o`` object bounding boxes."""
    n_o = len(o_bb)
    if n_o == 0:
        return np.zeros(0, dtype=bool), []
    geoms = shapely.box(
        o_bb[:, 0] - CONN_PAD_PT,
        o_bb[:, 1] - CONN_PAD_PT,
        o_bb[:, 2] + CONN_PAD_PT,
        o_bb[:, 3] + CONN_PAD_PT,
    )
    tr = shapely.STRtree(geoms)
    ii, jj = tr.query(geoms, predicate="intersects")
    mk = ii != jj
    ii, jj = ii[mk], jj[mk]
    deg = np.bincount(ii, minlength=n_o) if len(ii) else np.zeros(n_o, dtype=np.int64)
    _, lab = connected_components(coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(n_o, n_o)), directed=False)

    n_cc = int(lab.max()) + 1
    cc_cnt = np.bincount(lab, minlength=n_cc)
    cc_has_duct = np.zeros(n_cc, dtype=bool)
    cc_has_fit = np.zeros(n_cc, dtype=bool)
    cc_span = np.zeros(n_cc, dtype=np.float64)
    for k in range(n_o):
        c = int(lab[k])
        if o_is_duct[k]:
            cc_has_duct[c] = True
        else:
            cc_has_fit[c] = True
    for c in range(n_cc):
        cm = lab == c
        cc_span[c] = float(max(o_bb[cm, 2].max() - o_bb[cm, 0].min(), o_bb[cm, 3].max() - o_bb[cm, 1].min()))

    dropped = np.zeros(n_o, dtype=bool)
    why_list: list[str | None] = [None] * n_o
    for k in range(n_o):
        c = int(lab[k])
        why = None
        if deg[k] == 0:
            why = "R2_iso_duct" if o_is_duct[k] else "R1_iso_fitting"
        elif (not cc_has_duct[c]) and cc_cnt[c] <= NODUCT_FIT_CC_MAX_OBJS:
            why = "R3_noduct_fit_cc"
        elif (not cc_has_fit[c]) and cc_span[c] < NOFIT_DUCT_CC_MAX_SPAN:
            why = "R3_nofit_duct_cc"
        if why is not None:
            dropped[k] = True
            why_list[k] = why
    return dropped, why_list


def _find_dropped_object_ids(
    bb: np.ndarray,
    L: np.ndarray,
    oinst: np.ndarray,
    ocls: np.ndarray,
    lid: np.ndarray | None = None,
    lnames: list[str] | None = None,
) -> tuple[set[int], Counter, int]:
    """Identify physically invalid standalone Viewer objects (isolated fittings, standalone ducts, non-route islands)."""
    m_obj = oinst >= 0
    if not m_obj.any():
        return set(), Counter(), 0
    u_ids = np.unique(oinst[m_obj])
    n_o = len(u_ids)
    o_bb = np.zeros((n_o, 4), dtype=np.float64)
    o_is_duct = np.zeros(n_o, dtype=bool)
    for k, oid in enumerate(u_ids.tolist()):
        m = oinst == oid
        o_bb[k, 0] = float(bb[m, 0].min())
        o_bb[k, 1] = float(bb[m, 1].min())
        o_bb[k, 2] = float(bb[m, 2].max())
        o_bb[k, 3] = float(bb[m, 3].max())
        uc, inv = np.unique(ocls[m], return_inverse=True)
        cls_k = int(uc[int(np.argmax(np.bincount(inv, weights=L[m])))])
        o_is_duct[k] = (cls_k == 1)

    dropped, why_list = _classify_dropped_objects(o_bb, o_is_duct)
    drop_ids: set[int] = set()
    reasons: Counter = Counter()
    for k in range(n_o):
        if dropped[k]:
            drop_ids.add(int(u_ids[k]))
            reasons[str(why_list[k])] += 1
    return drop_ids, reasons, n_o


def clean_viewer_npz(src_p: Path, dst_p: Path) -> tuple[int, int, int, Counter]:
    """Clean one Viewer ``vw_*.npz`` file and write to ``dst_p`` -> (n_objs, n_dropped_objs, n_dropped_strokes, reasons)."""
    with np.load(src_p) as z:
        d = {k: z[k].copy() for k in z.files}

    y = d["label"].astype(bool)
    L = d["length"].astype(np.float64)
    bb = d["bbox"].astype(np.float64)
    oinst = d["oinst"].astype(np.int64)
    ocls = d["ocls"].astype(np.int64)

    drop_ids, reasons, n_o = _find_dropped_object_ids(bb, L, oinst, ocls)

    drop_stroke = np.zeros(len(y), dtype=bool)
    if drop_ids:
        drop_stroke |= np.isin(oinst, np.fromiter(drop_ids, dtype=np.int64))

    # Also clean orphan unassigned label==1 strokes (oinst < 0) detached > 3.5 pt from all kept objects
    unassigned_pos = y & (~drop_stroke) & (oinst < 0)
    if unassigned_pos.any():
        kept_assigned = y & (~drop_stroke) & (oinst >= 0)
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
                reasons["R4_orphan_pos_strokes"] += int(len(far_u))

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

    np.savez_compressed(dst_p, **d)
    return n_o, len(drop_ids), n_drop_s, reasons


def _worker_clean(job: tuple[str, str]) -> tuple[int, int, int, dict[str, int]]:
    no, nd, ns, rs = clean_viewer_npz(Path(job[0]), Path(job[1]))
    return no, nd, ns, dict(rs)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_score_pack_v4(src_pack: Path, dst_pack: Path, dst_ds: Path) -> None:
    """Create ``score_pack_v4_val`` from ``score_pack_v2_val`` with cleaned Viewer8 val NPZs & GT JSONs (GT5 untouched)."""
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
        graw = json.loads(src_gt.read_text(encoding="utf-8"))
        polys = [Polygon(np.asarray(o["vertices"], float)).buffer(0) if len(o.get("vertices") or []) >= 3 else Polygon() for o in graw]
        valid = [i for i, p in enumerate(polys) if p.area > 0]
        if not valid:
            dst_gt.write_text(json.dumps(graw, ensure_ascii=False), encoding="utf-8")
            continue
        v_bb = np.array([polys[i].bounds for i in valid], dtype=np.float64)
        v_is_duct = np.array([(str(graw[i].get("shape_name") or "") == "Duct") for i in valid], dtype=bool)
        dropped, _ = _classify_dropped_objects(v_bb, v_is_duct)
        drop_idx = {valid[k] for k in range(len(valid)) if dropped[k]}
        tot_vw_g += len(graw)
        drop_vw_g += len(drop_idx)
        kept = [o for i, o in enumerate(graw) if i not in drop_idx]
        dst_gt.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    print(f"Built {dst_pack}: cleaned {drop_vw_g}/{tot_vw_g} Viewer8 val GT polygons (5 GT5 pages untouched).", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build cleaned e13_v4 HVAC dataset from e13_v3 (Viewer MEP connectivity cleanup).")
    ap.add_argument("--src", type=Path, default=Path("datasets/e13_v3"))
    ap.add_argument("--dst", type=Path, default=Path("datasets/e13_v4"))
    ap.add_argument("--src-pack", type=Path, default=None, help="Optional score_pack_v2_val dir to also build score_pack_v4_val")
    ap.add_argument("--dst-pack", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=min(6, max(1, (os.cpu_count() or 2) - 1)))
    args = ap.parse_args()

    t0 = time.time()
    args.dst.mkdir(parents=True, exist_ok=True)
    for jp in sorted(args.src.glob("*.json")):
        shutil.copy2(jp, args.dst / jp.name)

    tot_gt, tot_vw_o, drop_vw_o, drop_vw_s = 0, 0, 0, 0
    reasons_all: Counter = Counter()

    npzs = sorted(p for p in args.src.glob("*.npz") if not p.stem.endswith(".graph.g3"))
    vw_jobs: list[tuple[str, str]] = []
    for p in npzs:
        dst_p = args.dst / p.name
        if p.stem.startswith("vw_"):
            vw_jobs.append((str(p), str(dst_p)))
        else:
            _link_or_copy(p, dst_p)
            tot_gt += 1
        g3 = args.src / f"{p.stem}.graph.g3.npz"
        if g3.exists():
            _link_or_copy(g3, args.dst / g3.name)

    n_w = max(1, min(int(args.workers), len(vw_jobs))) if vw_jobs else 1
    if n_w == 1:
        results = [_worker_clean(j) for j in vw_jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_w) as ex:
            results = list(ex.map(_worker_clean, vw_jobs))

    for no, nd, ns, rs in results:
        tot_vw_o += no
        drop_vw_o += nd
        drop_vw_s += ns
        reasons_all.update(rs)

    summary = {
        "source": str(args.src),
        "gt_pages_untouched": tot_gt,
        "viewer_pages_cleaned": len(vw_jobs),
        "viewer_objects_total": tot_vw_o,
        "viewer_objects_dropped": drop_vw_o,
        "viewer_strokes_zeroed": drop_vw_s,
        "reasons": dict(reasons_all),
        "params": {
            "conn_pad_pt": CONN_PAD_PT,
            "nofit_duct_cc_max_span": NOFIT_DUCT_CC_MAX_SPAN,
            "noduct_fit_cc_max_objs": NODUCT_FIT_CC_MAX_OBJS,
        },
    }
    (args.dst / "clean_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Built {args.dst} in {time.time() - t0:.1f}s ({n_w}w) | GT untouched={tot_gt} | "
        f"Viewer cleaned={len(vw_jobs)} pages: dropped {drop_vw_o}/{tot_vw_o} objs "
        f"({100 * drop_vw_o / max(tot_vw_o, 1):.2f}%), zeroed {drop_vw_s} strokes | {dict(reasons_all)}",
        flush=True,
    )
    if args.src_pack and args.dst_pack and args.src_pack.exists():
        build_score_pack_v4(args.src_pack, args.dst_pack, args.dst)


if __name__ == "__main__":
    main()
