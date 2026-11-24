"""Fast, stroke-exact relabeling toolkit for Viewer HVAC pages (100% compatible with tools/relabel_kit.py).

Writes the exact same files as relabel_kit.py:
  <page>/info.json
  <page>/relabel_state.json
  <page>/relabel.json
  <page>/relabel_strokes.npz   (label, oinst, ocls with openstroke.objects.HVAC_CLASSES indices)
Plus clean visual verification artifacts:
  <page>/overview.png, tile_r*_c*.png, layer_check.png, diff.png, final_strokes.png

Key speedups over relabel_kit.py:
  - Rich `layers` table (clean_pt after stripping hatch/fill, old_pt, cur_pt, median stroke length).
  - `layer-viz <id1,id2,...>`: color-coded visual check of multiple candidate layers in one image.
  - Safe `ductlayers <id1,id2,...>`: guards against accidentally wiping major duct layers; prunes off-layer strokes live.
  - `candidates`: pure-geometry connected components of unlabelled clean duct-layer strokes (C1, C2, ...) separated from hatch/fill/flex.
  - `tiles`: renders clean full-plan overview + zoomed overlapping tiles (no giant ID boxes blocking lines) in one call.
  - Batch editing: `accept-comp`, `add-boxes`, `remove-box` (single or multi-box, `--partial` stroke-level trim), `remove-poly`, `keep-only-box`, `add-all-clean`.
  - `qa` & `save`: automatic QA checks + diff.png + final_strokes.png.

Usage:
    py -3.12 tools/relabel_fast.py <command> <page_dir> [args...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from openstroke.postprocess import compact_strokes, fill_strokes, mesh_strokes, style_groups  # noqa: E402
from openstroke.relabel import STYLE_KEYS, page_segments_of  # noqa: E402
import relabel_kit as rk  # noqa: E402

DATA = ROOT / "datasets" / "e12"
CLASSES = rk.CLASSES
TO_IDX = rk.TO_IDX
FROM_IDX = rk.FROM_IDX
COLOR = rk.COLOR

DUCT_KW = re.compile(
    r"(DUCT|SAD\b|RAD\b|EAD\b|FAD\b|OAD\b|SED\b|MAD\b|PAD\b|TED\b|KEAD|SEAD|SFAD|SMOKE|ACONSUPP|ACONRETN|ACONPREC|MEVS|FITTING)",
    re.I,
)
LAYER_PALETTE = [
    "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1",
    "#D81B60", "#7CB342", "#F4511E", "#3949AB", "#00897B", "#C0CA33",
]


def _manifest_map() -> dict[str, dict]:
    man = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    return {r["stem"]: r for r in man["records"] if r.get("status") == "ok"}


def ensure_info(d: Path) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    p = d / "info.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    stem = d.name
    rec = _manifest_map().get(stem)
    if rec is None:
        raise SystemExit(f"stem {stem!r} not found in {DATA / 'manifest.json'}")
    info = {
        "stem": stem,
        "pdf": rec["filename"],
        "page_number": int(rec["page_number"]),
        "document_sha256": rec["document_sha256"],
    }
    p.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


def load_page(d: Path):
    info = ensure_info(d)
    z = np.load(DATA / f"{info['stem']}.npz")
    return info, z


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_state(d: Path) -> dict:
    p = d / "relabel_state.json"
    if not p.exists():
        raise SystemExit(f"no relabel_state.json in {d}: run `init {d}` first")
    return json.loads(p.read_text(encoding="utf-8"))


def save_state(d: Path, st: dict) -> None:
    atomic_write_text(d / "relabel_state.json", json.dumps(st, ensure_ascii=False))


def v3_bad_mask(z) -> np.ndarray:
    """Deterministic hatch/fill/compact-symbol filter (same v3 functions as openstroke.relabel)."""
    L = z["length"].astype(np.float64)
    segs = page_segments_of(z)
    grp = style_groups({k: z[k] for k in STYLE_KEYS})
    return mesh_strokes(segs, grp, L) | fill_strokes(segs, grp, L) | compact_strokes(segs, L)


def shape_endpoints(z):
    """(n, 2) start and (n, 2) end points of every shape."""
    off = z["shape_offsets"].astype(np.int64)
    seg = z["segments"].astype(np.float64)
    n = len(off) - 1
    nseg = np.diff(off)
    valid = nseg > 0
    s0 = np.zeros((n, 2), np.float64)
    s1 = np.zeros((n, 2), np.float64)
    if valid.any():
        s0[valid] = seg[off[:-1][valid], :2]
        s1[valid] = seg[off[1:][valid] - 1, 2:]
    return s0, s1


def connected_components_shapes(z, shape_ids: np.ndarray, snap: float = 2.5) -> list[np.ndarray]:
    """Fast spatial-hash connected components on shape endpoints (within `snap` pt)."""
    m = len(shape_ids)
    if m == 0:
        return []
    off = z["shape_offsets"].astype(np.int64)
    seg = z["segments"].astype(np.float64)
    # Collect endpoints of every segment belonging to shape_ids (catches T-junctions & multi-seg polylines cleanly)
    pts_x, pts_y, owners = [], [], []
    for local_i, sh in enumerate(shape_ids.tolist()):
        a, b = int(off[sh]), int(off[sh + 1])
        if b <= a:
            continue
        sub = seg[a:b]
        # Start of first seg, end of last seg, plus intermediate vertices if small
        if len(sub) <= 4:
            xy = np.vstack([sub[:, :2], sub[-1:, 2:]])
        else:
            xy = np.vstack([sub[0:1, :2], sub[-1:, 2:], sub[::max(1, len(sub) // 4), :2]])
        pts_x.append(xy[:, 0])
        pts_y.append(xy[:, 1])
        owners.append(np.full(len(xy), local_i, np.int32))
    if not pts_x:
        return []
    px = np.concatenate(pts_x)
    py = np.concatenate(pts_y)
    ow = np.concatenate(owners)

    parent = list(range(m))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    inv = 1.0 / snap
    gx = np.floor(px * inv).astype(np.int64)
    gy = np.floor(py * inv).astype(np.int64)
    buckets: dict[tuple[int, int], list[int]] = {}
    for k in range(len(px)):
        buckets.setdefault((int(gx[k]), int(gy[k])), []).append(k)

    r2 = snap * snap
    for (cx, cy), idxs in buckets.items():
        # Connect within same cell and 4 forward neighbor cells
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1), (1, -1)):
            other = buckets.get((cx + dx, cy + dy))
            if not other:
                continue
            if dx == 0 and dy == 0:
                if len(idxs) > 40:
                    # High-density cell: union all shapes hitting this tiny 2.5pt cell
                    base = ow[idxs[0]]
                    for k in idxs[1:]:
                        union(int(base), int(ow[k]))
                    continue
                for a_i in range(len(idxs)):
                    ka = idxs[a_i]
                    oa = int(ow[ka])
                    xa, ya = px[ka], py[ka]
                    for b_i in range(a_i + 1, len(idxs)):
                        kb = idxs[b_i]
                        ob = int(ow[kb])
                        if oa == ob:
                            continue
                        if (px[kb] - xa) ** 2 + (py[kb] - ya) ** 2 <= r2:
                            union(oa, ob)
            else:
                if len(idxs) * len(other) > 1600:
                    # Dense crossing: check bounding-box/cell overlap quickly
                    a_arr = np.array(idxs)
                    b_arr = np.array(other)
                    # subsample or vectorize in chunks
                    for ka in idxs[:40]:
                        d2 = (px[b_arr] - px[ka]) ** 2 + (py[b_arr] - py[ka]) ** 2
                        for kb in b_arr[d2 <= r2]:
                            union(int(ow[ka]), int(ow[kb]))
                    continue
                for ka in idxs:
                    oa = int(ow[ka])
                    xa, ya = px[ka], py[ka]
                    for kb in other:
                        ob = int(ow[kb])
                        if oa == ob:
                            continue
                        if (px[kb] - xa) ** 2 + (py[kb] - ya) ** 2 <= r2:
                            union(oa, ob)

    groups: dict[int, list[int]] = {}
    for local_i in range(m):
        groups.setdefault(find(local_i), []).append(int(shape_ids[local_i]))
    return [np.array(v, np.int64) for v in groups.values()]


def compute_candidates(d: Path, z, st: dict, min_pt: float = 25.0):
    """Compute unlabelled duct-layer candidate components, separating flex/micro clutter."""
    n = len(z["length"])
    L = z["length"].astype(np.float64)
    own = rk.owners(st, n)
    dm = rk.duct_mask(z, st)
    if dm is None:
        dm = np.ones(n, bool)
    bad = v3_bad_mask(z)
    stroke_only = z["type_code"] == 0
    free_clean = dm & stroke_only & (own < 0) & ~bad & (L >= 1.5)
    free_ids = np.flatnonzero(free_clean)
    comps = connected_components_shapes(z, free_ids, snap=2.5)

    # Build quick spatial lookup of labelled endpoints to check touches_labelled
    lab_ids = np.flatnonzero(own >= 0)
    lab_cells: set[tuple[int, int]] = set()
    if len(lab_ids):
        s0, s1 = shape_endpoints(z)
        lpts = np.vstack([s0[lab_ids], s1[lab_ids]])
        gx = np.floor(lpts[:, 0] / 3.0).astype(np.int64)
        gy = np.floor(lpts[:, 1] / 3.0).astype(np.int64)
        for x, y in zip(gx.tolist(), gy.tolist()):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    lab_cells.add((x + dx, y + dy))

    s0, s1 = shape_endpoints(z)
    op_curve = z["op_counts"][:, 1] > 0
    lid = z["layer_id"].astype(np.int64)
    names = [str(x) for x in z["layer_names"]]
    bb_all = z["bbox"].astype(np.float64)

    duct_cands = []
    flex_cands = []
    for sh in comps:
        tot = float(L[sh].sum())
        if tot < min_pt:
            continue
        med_len = float(np.median(L[sh]))
        max_len = float(L[sh].max())
        curve_len_share = float(L[sh][op_curve[sh]].sum() / max(tot, 1e-9))
        # Check if touching labelled
        pts = np.vstack([s0[sh], s1[sh]])
        touches = any((int(np.floor(x / 3.0)), int(np.floor(y / 3.0))) in lab_cells for x, y in pts[::max(1, len(pts) // 200)])
        b = bb_all[sh]
        bbox = [round(float(b[:, 0].min()), 1), round(float(b[:, 1].min()), 1),
                round(float(b[:, 2].max()), 1), round(float(b[:, 3].max()), 1)]
        top_lid = int(np.bincount(lid[sh], weights=L[sh]).argmax())
        rec = {
            "shapes": sh.tolist(),
            "pt": round(tot, 1),
            "n_shapes": int(len(sh)),
            "med_len": round(med_len, 1),
            "max_len": round(max_len, 1),
            "curve_share": round(curve_len_share, 2),
            "touches": bool(touches),
            "bbox": bbox,
            "layer": names[top_lid],
        }
        # Flex-duct / wavy / tiny-text heuristic: many tiny segments (med < 3.5 pt and max < 10 pt) or high-curve tiny chain
        is_flex = (len(sh) >= 12 and med_len < 3.5 and max_len < 12.0) or (len(sh) >= 20 and med_len < 4.5 and curve_len_share > 0.5)
        if is_flex:
            flex_cands.append(rec)
        else:
            duct_cands.append(rec)

    duct_cands.sort(key=lambda r: -r["pt"])
    flex_cands.sort(key=lambda r: -r["pt"])
    out = {}
    for k, r in enumerate(duct_cands, 1):
        r["cid"] = f"C{k}"
        out[r["cid"]] = r
    for k, r in enumerate(flex_cands, 1):
        r["cid"] = f"F{k}"
        out[r["cid"]] = r
    atomic_write_text(d / "candidates.json", json.dumps({"duct": duct_cands, "flex": flex_cands}, ensure_ascii=False))
    return duct_cands, flex_cands, bad


# ---------------- COMMANDS ----------------

def cmd_init(d: Path, reset: bool = False) -> None:
    info, z = load_page(d)
    p = d / "relabel_state.json"
    if p.exists() and not reset:
        st = load_state(d)
        print(f"existing state kept: {len(st['objects'])} objects, duct_layers={st.get('duct_layers')}")
        return
    old_dl = load_state(d).get("duct_layers") if p.exists() else None
    oinst, ocls = z["oinst"], z["ocls"]
    objs = {}
    for k, inst in enumerate(sorted(set(oinst[oinst >= 0].tolist())), 1):
        sh = np.flatnonzero(oinst == inst)
        c = FROM_IDX.get(int(np.bincount(ocls[sh].clip(0)).argmax()))
        if c:
            objs[str(k)] = {"cls": c, "shapes": sh.tolist(), "src": "viewer"}
    st = {"stem": info["stem"], "next": len(objs) + 1, "objects": objs, "notes": [], "log": []}
    if old_dl:
        st["duct_layers"] = old_dl
    save_state(d, st)
    print(f"initialized {info['stem']} ({info['pdf']} p{info['page_number']}): {len(objs)} viewer objects")


def cmd_layers(d: Path, show_all: bool = False) -> None:
    info, z = load_page(d)
    st = load_state(d) if (d / "relabel_state.json").exists() else None
    lid = z["layer_id"].astype(np.int64)
    L = z["length"].astype(np.float64)
    tc = z["type_code"]
    old_y = z["label"] > 0
    cur_y = (rk.owners(st, len(L)) >= 0) if st else old_y
    chosen = set((st or {}).get("duct_layers") or [])
    bad = v3_bad_mask(z)
    names = [str(x) for x in z["layer_names"]]

    rows = []
    for k, nm in enumerate(names):
        m = (lid == k) & (tc == 0)
        if not m.any():
            continue
        tot = float(L[m].sum())
        clean = float(L[m & ~bad].sum())
        old_pt = float(L[m & old_y].sum())
        cur_pt = float(L[m & cur_y].sum())
        med = float(np.median(L[m]))
        kw = bool(DUCT_KW.search(nm))
        if show_all or old_pt > 20 or cur_pt > 20 or kw or k in chosen or tot > 3000:
            rows.append((max(old_pt, cur_pt), kw, tot, k, nm, int(m.sum()), clean, old_pt, cur_pt, med))
    rows.sort(key=lambda r: (r[3] in chosen, r[0], r[1], r[2]), reverse=True)
    print(f"Page {info['stem']} | {info['pdf']} p{info['page_number']} | {len(names)} total layers")
    print(f"{'id':>3} {'strokes':>7} {'stroke_pt':>9} {'clean_pt':>9} {'old_pt':>8} {'cur_pt':>8} {'med':>5}  name")
    for _, kw, tot, k, nm, cnt, clean, old_pt, cur_pt, med in rows[:60]:
        tag = "  <- DUCT" if k in chosen else ("  [kw]" if kw else "")
        print(f"{k:>3} {cnt:>7} {tot:>9.0f} {clean:>9.0f} {old_pt:>8.0f} {cur_pt:>8.0f} {med:>5.1f}  {nm}{tag}")


def cmd_layer_viz(d: Path, ids_arg: str, box=None, out_name: str = "layer_check.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    info, z = load_page(d)
    ids = [int(x) for x in ids_arg.replace(" ", "").split(",") if x != ""]
    n = len(z["length"])
    seg, owner = rk.shape_xy(z, n)
    lid = z["layer_id"].astype(np.int64)
    tc = z["type_code"]
    L = z["length"].astype(float)
    names = [str(x) for x in z["layer_names"]]
    sel_all = np.isin(lid, ids) & (tc == 0)

    if box:
        x0, y0, x1, y1 = box
    else:
        ms = sel_all[owner]
        s2 = seg[ms] if ms.any() else seg
        x0, y0 = float(s2[:, [0, 2]].min()) - 25, float(s2[:, [1, 3]].min()) - 25
        x1, y1 = float(s2[:, [0, 2]].max()) + 25, float(s2[:, [1, 3]].max()) + 25

    m = (np.maximum(seg[:, 0], seg[:, 2]) >= x0) & (np.minimum(seg[:, 0], seg[:, 2]) <= x1) & \
        (np.maximum(seg[:, 1], seg[:, 3]) >= y0) & (np.minimum(seg[:, 1], seg[:, 3]) <= y1)
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    fig, ax = plt.subplots(figsize=(16, min(16, 16 * h / w)) if w >= h else (min(16, 16 * w / h), 16), dpi=115)
    lines = seg.reshape(-1, 2, 2)
    bg = m & ~sel_all[owner]
    if bg.any():
        ax.add_collection(LineCollection(lines[bg], colors=[(0.75, 0.75, 0.75, 0.4)], linewidths=0.3))
    legend_parts = []
    for idx_i, k in enumerate(ids):
        col = LAYER_PALETTE[idx_i % len(LAYER_PALETTE)]
        mk = m & ((lid == k) & (tc == 0))[owner]
        pt_k = float(L[(lid == k) & (tc == 0)].sum())
        if mk.any():
            ax.add_collection(LineCollection(lines[mk], colors=[col], linewidths=1.3))
        ax.plot([], [], color=col, lw=3, label=f"{k}: {names[k]} ({pt_k:.0f}pt)")
        legend_parts.append(f"{k}={names[k]}")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92, ncol=1 if len(ids) <= 8 else 2)
    ax.set_title(f"{info['stem']} layer-viz ({len(ids)} layers)", fontsize=10)
    out = d / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out.resolve())


def cmd_ductlayers(d: Path, ids_arg: str, prune: bool = True, force: bool = False) -> None:
    info, z = load_page(d)
    st = load_state(d)
    ids = sorted({int(v) for v in ids_arg.replace(" ", "").split(",") if v != ""})
    lid = z["layer_id"].astype(np.int64)
    L = z["length"].astype(np.float64)
    tc = z["type_code"]
    names = [str(x) for x in z["layer_names"]]
    old_y = z["label"] > 0

    # Safety check: warn about omitted duct-like layers with significant old labels
    omitted = []
    for k, nm in enumerate(names):
        if k in ids:
            continue
        oy = float(L[(lid == k) & (tc == 0) & old_y].sum())
        if oy >= 500 or (DUCT_KW.search(nm) and oy >= 200):
            omitted.append((k, nm, round(oy)))
    if omitted:
        print(f"WARNING: omitted layers with old labels: {omitted}")

    # Check how much of current state would be pruned
    own = rk.owners(st, len(L))
    cur_mask = own >= 0
    cur_pt = float(L[cur_mask].sum())
    keep_mask = np.isin(lid, np.asarray(ids, np.int64))
    dropped_pt = float(L[cur_mask & ~keep_mask].sum())
    if prune and cur_pt > 0 and (dropped_pt / cur_pt) > 0.50 and not force:
        raise SystemExit(
            f"SAFETY GUARD: ductlayers {ids} would drop {dropped_pt:.0f}/{cur_pt:.0f} pt ({100*dropped_pt/cur_pt:.1f}%) "
            f"of current labels! Verify layer IDs (e.g. with layer-viz) or pass --force."
        )

    st["duct_layers"] = ids
    st["log"].append(["ductlayers", ids])
    removed_objs = 0
    if prune:
        new_objs = {}
        for oid, ob in st["objects"].items():
            sh = np.asarray(ob["shapes"], np.int64)
            sh = sh[keep_mask[sh]]
            if len(sh):
                ob["shapes"] = sh.tolist()
                new_objs[oid] = ob
            else:
                removed_objs += 1
        st["objects"] = new_objs
    save_state(d, st)
    print(
        f"duct_layers={ids} ({[names[i] for i in ids if i < len(names)]}) | "
        f"pruned {removed_objs} off-layer objects, {dropped_pt:.0f} pt dropped | {len(st['objects'])} objects left"
    )


def cmd_candidates(d: Path, min_pt: float = 25.0, max_show: int = 35) -> None:
    info, z = load_page(d)
    st = load_state(d)
    duct_cands, flex_cands, _ = compute_candidates(d, z, st, min_pt=min_pt)
    tot_d = sum(c["pt"] for c in duct_cands)
    tot_f = sum(c["pt"] for c in flex_cands)
    print(f"{info['stem']}: {len(duct_cands)} duct candidates ({tot_d:.0f} pt), {len(flex_cands)} flex/micro clusters ({tot_f:.0f} pt)")
    for c in duct_cands[:max_show]:
        t = "TOUCH" if c["touches"] else "     "
        print(f"  {c['cid']:>4} {c['pt']:7.0f}pt {c['n_shapes']:4d}sh med={c['med_len']:5.1f} {t} bbox={c['bbox']}  {c['layer']}")


def _render_view(d: Path, z, st: dict, box, out: Path, show_ids: bool = False, show_cand: bool = True,
                 by_class: bool = False, show_raw: bool = False, duct_cands=None, flex_cands=None, bad_mask=None,
                 dpi: int = 115, title_extra: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    n = len(z["length"])
    seg, owner = rk.shape_xy(z, n)
    own = rk.owners(st, n)
    dm = rk.duct_mask(z, st)
    if dm is None:
        dm = np.ones(n, bool)
    if bad_mask is None:
        bad_mask = v3_bad_mask(z)

    x0, y0, x1, y1 = box
    m = (np.maximum(seg[:, 0], seg[:, 2]) >= x0) & (np.minimum(seg[:, 0], seg[:, 2]) <= x1) & \
        (np.maximum(seg[:, 1], seg[:, 3]) >= y0) & (np.minimum(seg[:, 1], seg[:, 3]) <= y1)
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    fig_w, fig_h = (16, max(6, min(16, 16 * h / w))) if w >= h else (max(6, min(16, 16 * w / h)), 16)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    lines = seg.reshape(-1, 2, 2)

    is_lab = own >= 0
    is_dl = dm & (z["type_code"] == 0)

    # 1. Background non-duct-layer strokes (very faint)
    bg = m & (~is_lab[owner]) & (~is_dl[owner])
    if bg.any():
        ax.add_collection(LineCollection(lines[bg], colors=[(0.72, 0.72, 0.72, 0.45)], linewidths=0.3))

    # 2. Unlabelled duct-layer hatch/clutter/flex
    cand_shape = np.zeros(n, bool)
    flex_shape = np.zeros(n, bool)
    if duct_cands:
        for c in duct_cands:
            cand_shape[np.asarray(c["shapes"], np.int64)] = True
    if flex_cands:
        for c in flex_cands:
            flex_shape[np.asarray(c["shapes"], np.int64)] = True

    clutter = m & (~is_lab[owner]) & is_dl[owner] & (~cand_shape[owner])
    if clutter.any():
        c_col = (0.88, 0.65, 0.35, 0.65) if show_raw else (0.55, 0.55, 0.62, 0.55)
        ax.add_collection(LineCollection(lines[clutter], colors=[c_col], linewidths=0.45))

    # 3. Unlabelled clean duct candidates in vivid BLUE/CYAN
    if cand_shape.any():
        cm = m & cand_shape[owner]
        if cm.any():
            ax.add_collection(LineCollection(lines[cm], colors=[(0.0, 0.48, 1.0, 1.0)], linewidths=1.35))

    # 4. Labelled duct strokes in vivid GREEN (or per-class colors)
    if not by_class:
        lm = m & is_lab[owner]
        if lm.any():
            ax.add_collection(LineCollection(lines[lm], colors=[(0.0, 0.76, 0.25, 1.0)], linewidths=1.35))
    else:
        for cls_name in CLASSES:
            ids_c = [int(oid) for oid, ob in st["objects"].items() if ob["cls"] == cls_name]
            if not ids_c:
                continue
            lm = m & np.isin(own[owner], ids_c)
            if lm.any():
                ax.add_collection(LineCollection(lines[lm], colors=[COLOR[cls_name]], linewidths=1.45))

    # Optional object IDs (use when zoomed in!)
    if show_ids:
        for oid, ob in st["objects"].items():
            sel = m & (own[owner] == int(oid))
            if not sel.any():
                continue
            pts = lines[sel].reshape(-1, 2)
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                ax.text(cx, cy, oid, fontsize=7.5, color="black", ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.08", fc="white", ec=COLOR[ob["cls"]], lw=0.7, alpha=0.82))

    # Optional candidate C# badges
    if show_cand and duct_cands:
        for c in duct_cands[:45]:
            bx0, by0, bx1, by1 = c["bbox"]
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                ax.text(cx, cy, c["cid"], fontsize=8, color="white", fontweight="bold", ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.12", fc="#0052CC", ec="white", lw=0.6, alpha=0.9))

    span = max(w, h)
    step = next(s for s in (5, 10, 20, 50, 100, 200, 500, 1000, 2000) if span / s <= 16)
    ax.set_xticks(np.arange(np.floor(x0 / step) * step, x1 + step, step))
    ax.set_yticks(np.arange(np.floor(y0 / step) * step, y1 + step, step))
    ax.grid(True, color="#90CAF9", lw=0.35, alpha=0.55)
    ax.tick_params(labelsize=8, top=True, labeltop=True, right=True, labelright=True)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_title(
        f"{st['stem']} ({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f}) grid={step}pt | "
        f"GREEN=labelled  BLUE=unlabelled duct candidates (C#)  grey=other {title_extra}",
        fontsize=9,
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def drawing_bounds(z, st: dict) -> tuple[float, float, float, float]:
    n = len(z["length"])
    seg, owner = rk.shape_xy(z, n)
    own = rk.owners(st, n)
    dm = rk.duct_mask(z, st)
    mask = (own >= 0) | ((dm & (z["type_code"] == 0)) if dm is not None else False)
    ms = mask[owner]
    s2 = seg[ms] if ms.any() else seg
    # Clip extreme 0.2% outliers (title block dots) only if they expand bbox wildly
    xs = np.concatenate([s2[:, 0], s2[:, 2]])
    ys = np.concatenate([s2[:, 1], s2[:, 3]])
    x0, x1 = float(xs.min()) - 20, float(xs.max()) + 20
    y0, y1 = float(ys.min()) - 20, float(ys.max()) + 20
    return x0, y0, x1, y1


def cmd_tiles(d: Path, nx: int = 2, ny: int = 2, by_class: bool = False, show_ids: bool = False, show_cand: bool = True, box=None) -> None:
    info, z = load_page(d)
    st = load_state(d)
    duct_cands, flex_cands, bad = compute_candidates(d, z, st, min_pt=25.0)
    x0, y0, x1, y1 = tuple(box) if box else drawing_bounds(z, st)
    ov = d / "overview.png"
    _render_view(d, z, st, (x0, y0, x1, y1), ov, show_ids=False, show_cand=show_cand, by_class=by_class,
                 duct_cands=duct_cands, flex_cands=flex_cands, bad_mask=bad, dpi=110, title_extra="OVERVIEW")
    paths = [str(ov.resolve())]
    w, h = (x1 - x0) / nx, (y1 - y0) / ny
    pad_x, pad_y = 0.05 * w, 0.05 * h
    for r in range(ny):
        for c in range(nx):
            bx0 = max(x0, x0 + c * w - pad_x)
            bx1 = min(x1, x0 + (c + 1) * w + pad_x)
            by0 = max(y0, y0 + r * h - pad_y)
            by1 = min(y1, y0 + (r + 1) * h + pad_y)
            out = d / f"tile_r{r}_c{c}.png"
            _render_view(d, z, st, (bx0, by0, bx1, by1), out, show_ids=show_ids, show_cand=show_cand, by_class=by_class,
                         duct_cands=duct_cands, flex_cands=flex_cands, bad_mask=bad, dpi=115,
                         title_extra=f"TILE r{r}c{c}")
            paths.append(str(out.resolve()))
    print(f"rendered {len(paths)} images ({len(duct_cands)} duct candidates, {sum(c['pt'] for c in duct_cands):.0f} pt):")
    for p in paths:
        print(" ", p)


def cmd_show(d: Path, box, show_ids: bool = False, show_cand: bool = True, by_class: bool = False,
             show_raw: bool = False, out_name: str = "zoom.png") -> None:
    info, z = load_page(d)
    st = load_state(d)
    duct_cands, flex_cands, bad = compute_candidates(d, z, st, min_pt=20.0)
    if not box:
        box = drawing_bounds(z, st)
    out = d / out_name
    _render_view(d, z, st, box, out, show_ids=show_ids, show_cand=show_cand, by_class=by_class, show_raw=show_raw,
                 duct_cands=duct_cands, flex_cands=flex_cands, bad_mask=bad, dpi=125)
    print(out.resolve())


def expand_ids(tokens: list[str], prefix: str = "") -> list[str]:
    out = []
    for tok in tokens:
        for part in tok.split(","):
            part = part.strip()
            if not part:
                continue
            m = re.match(rf"^({re.escape(prefix)})?(\d+)-({re.escape(prefix)})?(\d+)$", part, re.I)
            if m:
                a, b = int(m.group(2)), int(m.group(4))
                for v in range(min(a, b), max(a, b) + 1):
                    out.append(f"{prefix}{v}" if prefix else str(v))
            else:
                out.append(part.upper() if prefix else part)
    return out


def cmd_accept_comp(d: Path, cls_name: str, cids: list[str], dry: bool = False) -> None:
    info, z = load_page(d)
    st = load_state(d)
    cls = rk.canon(cls_name)
    cf = d / "candidates.json"
    if not cf.exists():
        compute_candidates(d, z, st)
    cdata = json.loads(cf.read_text(encoding="utf-8"))
    by_cid = {c["cid"]: c for c in cdata["duct"] + cdata["flex"]}
    own = rk.owners(st, len(z["length"]))
    L = z["length"].astype(float)
    want = expand_ids(cids, prefix="C")
    added_objs, added_sh, added_pt = 0, 0, 0.0
    for cid in want:
        c = by_cid.get(cid)
        if not c:
            print(f"  skip unknown {cid}")
            continue
        sh = np.asarray(c["shapes"], np.int64)
        sh = sh[own[sh] < 0]
        if not len(sh):
            print(f"  {cid}: already labelled")
            continue
        pt = float(L[sh].sum())
        added_objs += 1
        added_sh += len(sh)
        added_pt += pt
        if not dry:
            oid = str(st["next"])
            st["next"] += 1
            st["objects"][oid] = {"cls": cls, "shapes": sh.tolist(), "src": "agent"}
            own[sh] = int(oid)
            st["log"].append(["accept_comp", oid, cls, cid])
            print(f"  object {oid} <- {cid} ({cls}, {len(sh)} shapes, {pt:.0f} pt)")
        else:
            print(f"  [dry] {cid} ({cls}, {len(sh)} shapes, {pt:.0f} pt)")
    if not dry and added_objs:
        save_state(d, st)
    print(f"{'[dry] ' if dry else ''}accepted {added_objs} components, {added_sh} shapes, {added_pt:.0f} pt")


def cmd_add_all_clean(d: Path, cls_name: str = "Duct", only_touching: bool = False, min_pt: float = 20.0, dry: bool = False) -> None:
    info, z = load_page(d)
    st = load_state(d)
    cls = rk.canon(cls_name)
    duct_cands, _, _ = compute_candidates(d, z, st, min_pt=min_pt)
    own = rk.owners(st, len(z["length"]))
    L = z["length"].astype(float)
    added_objs, added_sh, added_pt = 0, 0, 0.0
    for c in duct_cands:
        if only_touching and not c["touches"]:
            continue
        sh = np.asarray(c["shapes"], np.int64)
        sh = sh[own[sh] < 0]
        if not len(sh):
            continue
        added_objs += 1
        added_sh += len(sh)
        added_pt += float(L[sh].sum())
        if not dry:
            oid = str(st["next"])
            st["next"] += 1
            st["objects"][oid] = {"cls": cls, "shapes": sh.tolist(), "src": "agent"}
            own[sh] = int(oid)
            st["log"].append(["add_all_clean", oid, cls, c["cid"]])
    if not dry and added_objs:
        save_state(d, st)
    print(f"{'[dry] ' if dry else ''}added {added_objs} clean components, {added_sh} shapes, {added_pt:.0f} pt")


def _parse_boxes_or_poly(args: list[str]) -> list[np.ndarray]:
    """Accept either one or more 'x0,y0,x1,y1' boxes OR a single box/polygon given as 'x,y x,y [x,y ...]'."""
    if all(len(a.split(",")) == 4 for a in args):
        polys = []
        for a in args:
            x0, y0, x1, y1 = [float(v) for v in a.split(",")]
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            polys.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float))
        return polys
    return [rk.parse_region(args)]


def cmd_add(d: Path, cls_name: str, region_args: list[str], dry: bool = False, raw: bool = False) -> None:
    info, z = load_page(d)
    st = load_state(d)
    cls = rk.canon(cls_name)
    polys = _parse_boxes_or_poly(region_args)
    n = len(z["length"])
    own = rk.owners(st, n)
    dm = rk.duct_mask(z, st)
    bad = np.zeros(n, bool) if raw else v3_bad_mask(z)
    L = z["length"].astype(float)
    tot_objs, tot_sh, tot_pt = 0, 0, 0.0
    for poly_xy in polys:
        free = (own < 0) & ~bad
        keep, msg = rk.select_strokes(z, poly_xy, free, dm)
        if not len(keep):
            print(f"  region empty: {msg}")
            continue
        pt = float(L[keep].sum())
        tot_objs += 1
        tot_sh += len(keep)
        tot_pt += pt
        if dry:
            print(f"  [dry] {cls}: {msg}")
        else:
            oid = str(st["next"])
            st["next"] += 1
            st["objects"][oid] = {"cls": cls, "shapes": keep.tolist(), "src": "agent"}
            own[keep] = int(oid)
            st["log"].append(["add", oid, cls, poly_xy.tolist()])
            print(f"  object {oid} {cls}: {msg}")
    if not dry and tot_objs:
        save_state(d, st)
    print(f"{'[dry] ' if dry else ''}added {tot_objs} objects, {tot_sh} shapes, {tot_pt:.0f} pt")


def cmd_remove_regions(d: Path, region_args: list[str], partial: bool = False, dry: bool = False, keep_inside: bool = False) -> None:
    import shapely

    info, z = load_page(d)
    st = load_state(d)
    polys = [shapely.make_valid(shapely.Polygon(p)) for p in _parse_boxes_or_poly(region_args)]
    region = shapely.unary_union(polys)
    n = len(z["length"])
    seg, owner = rk.shape_xy(z, n)
    mid = (seg[:, :2] + seg[:, 2:]) / 2
    sl = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    ins_seg = shapely.contains_xy(region, mid[:, 0], mid[:, 1])
    ins_len = np.bincount(owner, weights=sl * ins_seg, minlength=n)
    tot_len = np.bincount(owner, weights=sl, minlength=n)
    sh_in = (tot_len > 0) & (ins_len >= 0.5 * tot_len)

    L = z["length"].astype(float)
    rem_objs, trim_objs, rem_pt = 0, 0, 0.0
    new_objs = {}
    for oid, ob in st["objects"].items():
        sh = np.asarray(ob["shapes"], np.int64)
        if not len(sh):
            continue
        if keep_inside:
            drop_mask = ~sh_in[sh]
        elif partial:
            drop_mask = sh_in[sh]
        else:
            frac = float(L[sh][sh_in[sh]].sum() / max(float(L[sh].sum()), 1e-9))
            drop_mask = np.ones(len(sh), bool) if frac >= 0.60 else np.zeros(len(sh), bool)

        if not drop_mask.any():
            new_objs[oid] = ob
            continue
        dpt = float(L[sh[drop_mask]].sum())
        rem_pt += dpt
        kept = sh[~drop_mask]
        if len(kept):
            trim_objs += 1
            if not dry:
                ob["shapes"] = kept.tolist()
            new_objs[oid] = ob
        else:
            rem_objs += 1

    if not dry and (rem_objs or trim_objs):
        st["objects"] = new_objs
        st["log"].append(["keep_only" if keep_inside else ("trim_box" if partial else "remove_box"), region_args])
        save_state(d, st)
    mode_s = "keep-only-box" if keep_inside else ("trim(--partial)" if partial else "remove-box")
    print(f"{'[dry] ' if dry else ''}{mode_s}: removed {rem_objs} objects, trimmed {trim_objs} objects, dropped {rem_pt:.0f} pt | {len(new_objs)} objects left")


def cmd_remove_ids(d: Path, id_args: list[str]) -> None:
    st = load_state(d)
    ids = expand_ids(id_args)
    n_rm = 0
    for i in ids:
        if st["objects"].pop(i, None) is not None:
            st["log"].append(["remove", i])
            n_rm += 1
    save_state(d, st)
    print(f"removed {n_rm} objects | {len(st['objects'])} objects left")


def cmd_remove_all(d: Path) -> None:
    st = load_state(d)
    n_prev = len(st["objects"])
    st["objects"] = {}
    st["log"].append(["remove_all", n_prev])
    save_state(d, st)
    print(f"cleared all {n_prev} objects (duct_layers={st.get('duct_layers')})")


def cmd_cls_box(d: Path, cls_name: str, region_args: list[str]) -> None:
    import shapely

    _, z = load_page(d)
    st = load_state(d)
    cls = rk.canon(cls_name)
    region = shapely.unary_union([shapely.make_valid(shapely.Polygon(p)) for p in _parse_boxes_or_poly(region_args)])
    n = len(z["length"])
    seg, owner = rk.shape_xy(z, n)
    mid = (seg[:, :2] + seg[:, 2:]) / 2
    sl = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    ins_len = np.bincount(owner, weights=sl * shapely.contains_xy(region, mid[:, 0], mid[:, 1]), minlength=n)
    tot_len = np.bincount(owner, weights=sl, minlength=n)
    sh_in = (tot_len > 0) & (ins_len >= 0.5 * tot_len)
    L = z["length"].astype(float)
    changed = 0
    for oid, ob in st["objects"].items():
        sh = np.asarray(ob["shapes"], np.int64)
        if len(sh) and L[sh][sh_in[sh]].sum() >= 0.6 * max(L[sh].sum(), 1e-9):
            ob["cls"] = cls
            changed += 1
    save_state(d, st)
    print(f"reclassified {changed} objects -> {cls}")


def render_final_strokes(d: Path, z, label: np.ndarray, ocls: np.ndarray) -> Path:
    """Clean vector stroke image: duct strokes colored vividly by class, everything else light grey."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    n = len(z["length"])
    seg, owner = rk.shape_xy(z, n)
    t_mask = label > 0
    ms = t_mask[owner]
    s2 = seg[ms] if ms.any() else seg
    x0, y0 = float(s2[:, [0, 2]].min()) - 25, float(s2[:, [1, 3]].min()) - 25
    x1, y1 = float(s2[:, [0, 2]].max()) + 25, float(s2[:, [1, 3]].max()) + 25
    m = (np.maximum(seg[:, 0], seg[:, 2]) >= x0) & (np.minimum(seg[:, 0], seg[:, 2]) <= x1) & \
        (np.maximum(seg[:, 1], seg[:, 3]) >= y0) & (np.minimum(seg[:, 1], seg[:, 3]) <= y1)
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    fig, ax = plt.subplots(figsize=(18, max(7, min(18, 18 * h / w))) if w >= h else (max(7, min(18, 18 * w / h)), 18), dpi=115)
    lines = seg.reshape(-1, 2, 2)
    bg = m & ~t_mask[owner]
    if bg.any():
        ax.add_collection(LineCollection(lines[bg], colors=[(0.72, 0.72, 0.72, 0.45)], linewidths=0.3))
    for idx_val, cname in FROM_IDX.items():
        cm = m & t_mask[owner] & (ocls[owner] == idx_val)
        if cm.any():
            ax.add_collection(LineCollection(lines[cm], colors=[COLOR[cname]], linewidths=1.35))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    L = z["length"].astype(float)
    ax.set_title(
        f"{d.name} FINAL STROKES | true={L[t_mask].sum():,.0f} pt ({int(t_mask.sum())} shapes) | "
        f"Duct=green Elbow=orange SqElbow=magenta Reducer=blue ChanRe=gold Tee=red",
        fontsize=10,
    )
    out = d / "final_strokes.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def cmd_qa(d: Path) -> dict:
    info, z = load_page(d)
    st = load_state(d)
    n = len(z["length"])
    L = z["length"].astype(np.float64)
    own = rk.owners(st, n)
    cur_y = own >= 0
    old_y = z["label"] > 0
    dm = rk.duct_mask(z, st)
    off_dl_pt = float(L[cur_y & ~dm].sum()) if dm is not None else 0.0
    duct_cands, flex_cands, bad = compute_candidates(d, z, st, min_pt=25.0)
    hatch_in_cur = float(L[cur_y & bad].sum())
    kept = float(L[cur_y & old_y].sum())
    added = float(L[cur_y & ~old_y].sum())
    removed = float(L[~cur_y & old_y].sum())
    unlab_cand_pt = sum(c["pt"] for c in duct_cands)
    res = {
        "stem": info["stem"],
        "objects": len(st["objects"]),
        "duct_layers": st.get("duct_layers") or [],
        "old_pt": round(float(L[old_y].sum())),
        "new_pt": round(float(L[cur_y].sum())),
        "kept_pt": round(kept),
        "added_pt": round(added),
        "removed_pt": round(removed),
        "off_dl_pt": round(off_dl_pt),
        "hatch_in_lab_pt": round(hatch_in_cur),
        "unlab_cand_count": len(duct_cands),
        "unlab_cand_pt": round(unlab_cand_pt),
        "top_cands": [(c["cid"], round(c["pt"]), c["bbox"]) for c in duct_cands[:5]],
    }
    print(json.dumps(res, ensure_ascii=False))
    return res


def cmd_missed_viz(d: Path, extra_layers: list[int] | None = None, out_name: str = "missed_check.png") -> None:
    """Render GREEN=labelled, RED=touching same-style unlabelled on duct_layers, ORANGE=far same-style unlabelled, BLUE=extra_layers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    info, z = load_page(d)
    st = load_state(d)
    n = len(z["length"])
    seg, owner = rk.shape_xy(z, n)
    own = rk.owners(st, n)
    lab = own >= 0
    L = z["length"].astype(np.float64)
    lid = z["layer_id"]
    rgb = np.round(z["stroke_rgb"], 2)
    width = np.round(z["width"], 2)
    dl = set(st.get("duct_layers") or [])

    style_len: dict[tuple, float] = {}
    for i in np.flatnonzero(lab):
        k = (int(lid[i]), tuple(rgb[i].tolist()), float(width[i]))
        style_len[k] = style_len.get(k, 0.0) + float(L[i])
    tot_lab = float(L[lab].sum())
    major = {k for k, v in style_len.items() if v >= max(80.0, 0.01 * tot_lab)}

    cell = 4.0
    lab_segs = seg[lab[owner]]
    lab_cells: set[tuple[int, int]] = set()
    if len(lab_segs):
        ep = np.vstack([lab_segs[:, :2], lab_segs[:, 2:4]])
        gx, gy = np.floor(ep[:, 0] / cell).astype(np.int32), np.floor(ep[:, 1] / cell).astype(np.int32)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                lab_cells.update(zip((gx + dx).tolist(), (gy + dy).tolist()))

    offs = z["shape_offsets"].astype(np.int64)
    touch = np.zeros(n, bool)
    far = np.zeros(n, bool)
    for i in np.flatnonzero((~lab) & np.isin(lid, list(dl)) & (L >= 2.0)):
        if (int(lid[i]), tuple(rgb[i].tolist()), float(width[i])) not in major:
            continue
        s0, s1 = int(offs[i]), int(offs[i + 1])
        hit = False
        if s1 > s0:
            ss = seg[s0:s1]
            for pt in (ss[0, :2], ss[-1, 2:4]):
                if (int(np.floor(pt[0] / cell)), int(np.floor(pt[1] / cell))) in lab_cells:
                    hit = True
                    break
        if hit:
            touch[i] = True
        else:
            far[i] = True

    x0, y0, x1, y1 = drawing_bounds(z, st)
    m = (np.maximum(seg[:, 0], seg[:, 2]) >= x0) & (np.minimum(seg[:, 0], seg[:, 2]) <= x1) & \
        (np.maximum(seg[:, 1], seg[:, 3]) >= y0) & (np.minimum(seg[:, 1], seg[:, 3]) <= y1)
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    fig, ax = plt.subplots(figsize=(18, max(7, min(18, 18 * h / w))) if w >= h else (max(7, min(18, 18 * w / h)), 18), dpi=115)
    lines = seg.reshape(-1, 2, 2)
    bg = m & ~(lab | touch | far)[owner]
    if bg.any():
        ax.add_collection(LineCollection(lines[bg], colors=[(0.78, 0.78, 0.78, 0.4)], linewidths=0.3))
    if lab[owner].any():
        ax.add_collection(LineCollection(lines[m & lab[owner]], colors=["#00B84C"], linewidths=1.3))
    if far[owner].any():
        ax.add_collection(LineCollection(lines[m & far[owner]], colors=["#FF9100"], linewidths=1.6))
    if touch[owner].any():
        ax.add_collection(LineCollection(lines[m & touch[owner]], colors=["#E50000"], linewidths=1.9))
    if extra_layers:
        white = (rgb[:, 0] > 0.95) & (rgb[:, 1] > 0.95) & (rgb[:, 2] > 0.95)
        em = (~lab) & np.isin(lid, extra_layers) & (~white)
        if em[owner].any():
            ax.add_collection(LineCollection(lines[m & em[owner]], colors=["#0055FF"], linewidths=1.9))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_title(
        f"{d.name} MISSED CHECK | lab={tot_lab:,.0f}pt | RED(touch)={L[touch].sum():,.0f}pt ({int(touch.sum())}sh) "
        f"ORANGE(far)={L[far].sum():,.0f}pt" + (f" BLUE(extra={extra_layers})" if extra_layers else ""),
        fontsize=10,
    )
    out = d / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"missed-viz -> {out.resolve()} (touch={L[touch].sum():.0f}pt, far={L[far].sum():.0f}pt)")


def cmd_inspect_objs(d: Path, oids: list[str], out_name: str = "inspect_objs.png") -> None:
    """Print info and render zoomed crop highlighting specific object IDs in RED on top of GREEN neighbours."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    _, z = load_page(d)
    st = load_state(d)
    n = len(z["length"])
    seg, owner = rk.shape_xy(z, n)
    own = rk.owners(st, n)
    L = z["length"].astype(np.float64)
    bbox = z["bbox"]

    target_sh = np.zeros(n, bool)
    boxes = []
    for oid in oids:
        ob = st["objects"].get(str(oid))
        if not ob:
            print(f"  oid={oid}: NOT FOUND")
            continue
        sh = np.asarray(ob["shapes"], np.int64)
        target_sh[sh] = True
        b = [float(bbox[sh, 0].min()), float(bbox[sh, 1].min()), float(bbox[sh, 2].max()), float(bbox[sh, 3].max())]
        boxes.append((oid, ob["cls"], b))
        print(f"  oid={oid:4s} cls={ob['cls']:12s} src={ob.get('src',''):6s} sh={len(sh):4d} len={L[sh].sum():7.0f}pt bbox=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] size={b[2]-b[0]:.0f}x{b[3]-b[1]:.0f}")
    if not boxes:
        return
    x0 = min(b[0] for _, _, b in boxes) - 60.0
    y0 = min(b[1] for _, _, b in boxes) - 60.0
    x1 = max(b[2] for _, _, b in boxes) + 60.0
    y1 = max(b[3] for _, _, b in boxes) + 60.0
    m = (np.maximum(seg[:, 0], seg[:, 2]) >= x0) & (np.minimum(seg[:, 0], seg[:, 2]) <= x1) & \
        (np.maximum(seg[:, 1], seg[:, 3]) >= y0) & (np.minimum(seg[:, 1], seg[:, 3]) <= y1)
    fig, ax = plt.subplots(figsize=(14, 10), dpi=115)
    lines = seg.reshape(-1, 2, 2)
    other_lab = (own >= 0) & (~target_sh)
    bg = m & (own[owner] < 0)
    if bg.any():
        ax.add_collection(LineCollection(lines[bg], colors=[(0.75, 0.75, 0.75, 0.55)], linewidths=0.4))
    if (m & other_lab[owner]).any():
        ax.add_collection(LineCollection(lines[m & other_lab[owner]], colors=["#00B84C"], linewidths=1.4))
    if (m & target_sh[owner]).any():
        ax.add_collection(LineCollection(lines[m & target_sh[owner]], colors=["#FF0000"], linewidths=2.4))
    for oid, cname, b in boxes:
        ax.text((b[0] + b[2]) / 2, b[1] - 3, f"{oid}:{cname}", color="#CC0000", fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="red", alpha=0.85))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_title(f"{d.name} INSPECT oids={','.join(oids)} (RED=target, GREEN=other labelled)", fontsize=10)
    out = d / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"inspect-objs -> {out.resolve()}")


def cmd_split_giant(d: Path, oids: list[str], dry: bool = False) -> None:
    """Split oversized multi-fitting object(s) into individual CAD objects using pure planar face geometry (zero ML)."""
    import shapely
    from openstroke.hvac_faces import page_faces
    from openstroke.relabel import page_segments_of

    _, z = load_page(d)
    st = load_state(d)
    n = len(z["length"])
    seg, owner = rk.shape_xy(z, n)
    offs = z["shape_offsets"].astype(np.int64)
    L = z["length"].astype(np.float64)
    curved = z["op_counts"][:, 1] > 0
    z_dict = {k: z[k] for k in z.files}
    psegs = page_segments_of(z_dict)

    for oid in oids:
        ob = st["objects"].get(str(oid))
        if not ob:
            print(f"split-giant: oid {oid} not found")
            continue
        sh = np.asarray(ob["shapes"], np.int64)
        if len(sh) < 2:
            print(f"split-giant: oid {oid} has only {len(sh)} shape, cannot split")
            continue

        picked = np.zeros(n, dtype=bool)
        picked[sh] = True
        faces = page_faces(psegs, picked, curved=curved)
        if len(faces) <= 1:
            print(f"split-giant: oid {oid} ({len(sh)} shapes, {L[sh].sum():.0f}pt) yielded only {len(faces)} planar face(s) — split manually by box")
            continue

        # Assign each shape in `sh` to the planar face whose boundary is closest to the shape
        f_bounds = [f.boundary for f in faces]
        f_shapes: list[list[int]] = [[] for _ in faces]
        unassigned: list[int] = []
        for sid in sh:
            ss = seg[offs[sid]:offs[sid + 1]]
            if not len(ss):
                continue
            mids = 0.5 * (ss[:, :2] + ss[:, 2:4])
            mp = shapely.multipoints(mids)
            best_fi, best_d = -1, 4.0
            for fi, fb in enumerate(f_bounds):
                dist = float(fb.distance(mp))
                if dist < best_d:
                    best_d = dist
                    best_fi = fi
            if best_fi >= 0:
                f_shapes[best_fi].append(int(sid))
            else:
                unassigned.append(int(sid))

        # Attach any unassigned shapes to nearest non-empty face
        nonempty = [fi for fi in range(len(faces)) if f_shapes[fi]]
        for sid in unassigned:
            ss = seg[offs[sid]:offs[sid + 1]]
            if not len(ss) or not nonempty:
                continue
            mp = shapely.multipoints(0.5 * (ss[:, :2] + ss[:, 2:4]))
            best_fi = min(nonempty, key=lambda fi: float(faces[fi].distance(mp)))
            f_shapes[best_fi].append(int(sid))

        # Pure geometric classification for each face
        new_objs = []
        for fi, slist in enumerate(f_shapes):
            if not slist:
                continue
            poly = faces[fi]
            sarr = np.asarray(slist, np.int64)
            has_curve = bool(curved[sarr].any())
            mrr = poly.minimum_rotated_rectangle
            coords = np.asarray(mrr.exterior.coords)
            e1 = float(np.hypot(*(coords[1] - coords[0])))
            e2 = float(np.hypot(*(coords[2] - coords[1])))
            w_min, l_max = min(e1, e2), max(e1, e2)
            fill_r = float(poly.area / max(mrr.area, 1e-6))
            simp = poly.simplify(max(0.8, 0.08 * w_min), preserve_topology=True)
            n_v = len(simp.exterior.coords) - 1 if simp.geom_type == "Polygon" else 0
            # Classify purely from geometry:
            if has_curve and fill_r < 0.88:
                cls_name = "Elbow"
            elif n_v == 4 and 0.55 <= fill_r <= 0.88 and l_max / max(w_min, 1e-3) <= 3.2:
                # Check if opposite sides have different lengths (trapezoid = Reducer)
                sc = np.asarray(simp.exterior.coords)
                elens = sorted(float(np.hypot(*(sc[k + 1] - sc[k]))) for k in range(4))
                if elens[0] / max(elens[-1], 1e-3) < 0.78 and abs(elens[1] - elens[2]) / max(elens[2], 1e-3) < 0.25:
                    cls_name = "Reducer"
                else:
                    cls_name = "Duct"
            elif l_max < 14.0 and w_min < 10.0:
                cls_name = "Chân rẽ"
            elif n_v >= 6 and fill_r < 0.75:
                cls_name = "Tee"
            else:
                cls_name = "Duct"
            new_objs.append((cls_name, slist, float(L[sarr].sum())))

        from collections import Counter
        summary = dict(Counter(c for c, _, _ in new_objs))
        print(f"split-giant oid={oid} ({len(sh)}sh, {L[sh].sum():.0f}pt) -> {len(new_objs)} objects {summary}" + (" [DRY]" if dry else ""))
        if not dry and len(new_objs) > 1:
            st["objects"].pop(str(oid), None)
            new_ids = []
            for cls_name, slist, _ in new_objs:
                nid = str(st["next"])
                st["next"] += 1
                st["objects"][nid] = {"cls": cls_name, "shapes": slist, "src": "agent"}
                new_ids.append(nid)
            st["log"].append(["split-giant", str(oid), new_ids])
    if not dry:
        save_state(d, st)


def cmd_save(d: Path, no_images: bool = False) -> None:
    import relabel_diff

    rk.save(d)
    info, z = load_page(d)
    rs = np.load(d / "relabel_strokes.npz")
    if not no_images:
        relabel_diff.main(str(d))
        fp = render_final_strokes(d, z, rs["label"], rs["ocls"])
        print(f"final_strokes -> {fp.resolve()}")
    cmd_qa(d)


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    cmd, d = argv[0], Path(argv[1])
    rest = argv[2:]

    def pop_flag(flag: str) -> bool:
        if flag in rest:
            rest.remove(flag)
            return True
        return False

    def pop_opt(flag: str, default=None):
        if flag in rest:
            i = rest.index(flag)
            val = rest[i + 1]
            del rest[i:i + 2]
            return val
        return default

    if cmd == "init":
        cmd_init(d, reset=pop_flag("--reset"))
    elif cmd == "layers":
        cmd_layers(d, show_all=pop_flag("--all"))
    elif cmd == "layer-viz":
        out_name = pop_opt("--out", "layer_check.png")
        ids_arg = rest[0]
        box = [float(v) for v in rest[1:5]] if len(rest) >= 5 else None
        cmd_layer_viz(d, ids_arg, box, out_name=out_name)
    elif cmd == "ductlayers":
        force = pop_flag("--force")
        no_prune = pop_flag("--no-prune")
        cmd_ductlayers(d, ",".join(rest), prune=not no_prune, force=force)
    elif cmd == "candidates":
        min_pt = float(pop_opt("--min-pt", "25"))
        max_show = int(pop_opt("--max-show", "35"))
        cmd_candidates(d, min_pt=min_pt, max_show=max_show)
    elif cmd == "tiles":
        nx = int(pop_opt("--nx", "2"))
        ny = int(pop_opt("--ny", "2"))
        by_class = pop_flag("--by-class")
        show_ids = pop_flag("--ids")
        no_cand = pop_flag("--no-cand")
        box = [float(v) for v in rest[:4]] if len(rest) >= 4 else None
        cmd_tiles(d, nx=nx, ny=ny, by_class=by_class, show_ids=show_ids, show_cand=not no_cand, box=box)
    elif cmd == "show":
        show_ids = pop_flag("--ids")
        by_class = pop_flag("--by-class")
        no_cand = pop_flag("--no-cand")
        show_raw = pop_flag("--raw")
        out_name = pop_opt("--out", "zoom.png")
        box = [float(v) for v in rest[:4]] if len(rest) >= 4 else None
        cmd_show(d, box, show_ids=show_ids, show_cand=not no_cand, by_class=by_class, show_raw=show_raw, out_name=out_name)
    elif cmd == "list":
        box = [float(v) for v in rest[:4]] if len(rest) >= 4 else None
        rk.list_(d, box)
    elif cmd == "accept-comp":
        dry = pop_flag("--dry")
        cmd_accept_comp(d, rest[0], rest[1:], dry=dry)
    elif cmd == "add-all-clean":
        dry = pop_flag("--dry")
        only_touching = pop_flag("--only-touching")
        cls_name = pop_opt("--class", "Duct")
        min_pt = float(pop_opt("--min-pt", "20"))
        cmd_add_all_clean(d, cls_name=cls_name, only_touching=only_touching, min_pt=min_pt, dry=dry)
    elif cmd == "add":
        dry = pop_flag("--dry")
        raw = pop_flag("--raw")
        cmd_add(d, rest[0], rest[1:], dry=dry, raw=raw)
    elif cmd == "remove":
        cmd_remove_ids(d, rest)
    elif cmd == "remove-box":
        dry = pop_flag("--dry")
        partial = pop_flag("--partial")
        cmd_remove_regions(d, rest, partial=partial, dry=dry, keep_inside=False)
    elif cmd == "remove-poly":
        dry = pop_flag("--dry")
        partial = pop_flag("--partial")
        cmd_remove_regions(d, rest, partial=partial, dry=dry, keep_inside=False)
    elif cmd == "keep-only-box":
        dry = pop_flag("--dry")
        cmd_remove_regions(d, rest, partial=True, dry=dry, keep_inside=True)
    elif cmd == "remove-all":
        cmd_remove_all(d)
    elif cmd == "cls":
        rk.setcls(d, rest[0], rest[1])
    elif cmd == "cls-box":
        cmd_cls_box(d, rest[0], rest[1:])
    elif cmd == "restore":
        rk.restore(d, rest)
    elif cmd == "restoreclass":
        rk.restore_class(d, rest)
    elif cmd == "note":
        st = load_state(d)
        st["notes"].append(" ".join(rest))
        save_state(d, st)
    elif cmd == "qa":
        cmd_qa(d)
    elif cmd == "missed-viz":
        out_name = pop_opt("--out", "missed_check.png")
        layers_opt = pop_opt("--layers", "")
        extra = [int(x) for x in layers_opt.split(",") if x.strip()] if layers_opt else None
        cmd_missed_viz(d, extra_layers=extra, out_name=out_name)
    elif cmd == "inspect-objs":
        out_name = pop_opt("--out", "inspect_objs.png")
        cmd_inspect_objs(d, rest, out_name=out_name)
    elif cmd == "split-giant":
        dry = pop_flag("--dry")
        cmd_split_giant(d, rest, dry=dry)
    elif cmd == "save":
        cmd_save(d, no_images=pop_flag("--no-images"))
    else:
        raise SystemExit(f"unknown command {cmd!r}\n" + (__doc__ or ""))


if __name__ == "__main__":
    main(sys.argv[1:])
