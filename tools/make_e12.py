"""OpenStroke34 dataset e12 (F187): datasets/e11 + the pieces (openstroke.pieces) of every CANDIDATE shape of each page, with piece labels.

Pieces are cut only by the ends of the true shapes (at inference: the picked strokes), not by hatching touching a wall.
Candidates (what the piece head sees; at inference: the strokes the stroke model picks + the strokes touching them):
  - the true shapes;
  - shapes touching a true shape (an end within pieces.SNAP of a true shape's segment, or a true shape's end on theirs): the strokes the
    piece head may have to bring in (user 2026-11-19: strokes touching a confident object are likely part of it) or keep out;
  - a sample of other shapes drawn in a true shape's style (hard negatives), at most HARD_MAX.
Piece labels: object = the GT polygon whose outline is nearest along the piece (pieces.piece_objects, F180); pc_fg = the piece's shape is
true and >= half of the piece lies on that object; pc_inst = the object's index (-1 when not fg); pc_cls = its general class (GT pages;
-1 on Viewer pages, whose class conventions differ, F162 / F167). Shapes are payload indices. Data only.

    py -3.12 tools/make_e12.py datasets/e11 datasets/score_pack_e8_all datasets/e12 [datasets/openstroke_e12_data.zip] [--limit N]
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openstroke.objects import class_of, general_class  # noqa: E402
from openstroke.pieces import SNAP, facing_edges, piece_objects, shape_pieces  # noqa: E402
from openstroke.relabel import page_segments_of  # noqa: E402

HARD_SHARE, HARD_MAX = 0.05, 2000


def candidates(d: dict, segs, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    import shapely
    from scipy.spatial import cKDTree

    y = d["label"].astype(bool)
    n = len(y)
    true = np.flatnonzero(y)
    if not len(true):
        return true, {"true": 0, "touch": 0, "hard": 0}
    off = segs.shape_offsets
    first = np.full(n, -1, np.int64); last = np.full(n, -1, np.int64)
    has = off[1:] > off[:-1]
    first[has] = off[:-1][has]; last[has] = off[1:][has] - 1
    ends = np.r_[segs.seg[first[has], :2], segs.seg[last[has], 2:]]
    end_owner = np.r_[np.flatnonzero(has), np.flatnonzero(has)]
    tm = y[segs.owner]
    tlines = shapely.linestrings(segs.seg[tm].reshape(-1, 2, 2))
    tree = shapely.STRtree(tlines)
    pi, _ = tree.query(shapely.points(ends), predicate="dwithin", distance=SNAP)   # ends of any shape on a true segment
    touch = set(end_owner[np.unique(pi)].tolist())
    t_ends = ends[y[end_owner]]                                                     # true ends on any shape's segment
    all_lines = shapely.linestrings(segs.seg.reshape(-1, 2, 2))
    atree = shapely.STRtree(all_lines)
    _, li = atree.query(shapely.points(t_ends), predicate="dwithin", distance=SNAP)
    touch |= set(segs.owner[np.unique(li)].tolist())
    touch -= set(true.tolist())
    # hard negatives: shapes drawn in a true style (colour + width + type)
    key = np.c_[np.round(d["stroke_rgb"], 2), np.round(d["width"], 2), d["type_code"]]
    tk = {tuple(r) for r in key[true]}
    same = np.flatnonzero(~y & np.array([tuple(r) in tk for r in key]))
    same = np.setdiff1d(same, np.fromiter(touch, np.int64, len(touch)))
    k = min(HARD_MAX, int(np.ceil(HARD_SHARE * len(same))))
    hard = rng.choice(same, size=k, replace=False) if k and len(same) else np.zeros(0, np.int64)
    cand = np.unique(np.r_[true, np.fromiter(touch, np.int64, len(touch)), hard]).astype(np.int64)
    return cand, {"true": int(len(true)), "touch": int(len(touch)), "hard": int(len(hard))}


def with_facing(arr: dict) -> dict:
    """Touch links (kind 1) + facing links between parallel pieces (kind 2, pieces.facing_edges: the two walls of a duct), one row each."""
    touch = arr["pc_edges"].astype(np.int64).reshape(-1, 2)
    face = facing_edges(arr["pc_pts"], arr["pc_len"].astype(float), arr["pc_cur"].astype(bool))
    key_t = {tuple(e) for e in touch.tolist()}
    face = np.asarray([e for e in face.tolist() if tuple(e) not in key_t], np.int64).reshape(-1, 2)
    arr = dict(arr)
    arr["pc_edges"] = np.r_[touch, face].astype(np.int32)
    arr["pc_ekind"] = np.r_[np.ones(len(touch)), np.full(len(face), 2)].astype(np.uint8)
    return arr


def page_arrays(d: dict, objects: list[dict], viewer: bool, rng: np.random.Generator) -> tuple[dict, dict]:
    segs = page_segments_of(d)
    cand, info = candidates(d, segs, rng)
    pc = shape_pieces(segs, cand, others=np.flatnonzero(d["label"] > 0))   # only pipe strokes cut pieces (inference: the picked strokes)
    obj, share = piece_objects(pc, objects)
    y = d["label"].astype(bool)
    fg = (len(pc) > 0) & y[pc.shape] & (obj >= 0) & (share >= 0.5) if len(pc) else np.zeros(0, bool)
    if len(pc):
        cls = np.array([class_of(objects[o].get("shape_name")) if o >= 0 else 0 for o in obj], np.int64)
        cls = general_class(cls)
        cls = np.where(fg & (cls > 0), cls, -1) if not viewer else np.full(len(pc), -1)
    else:
        cls = np.zeros(0, np.int64)
    arr = {"pc_shape": pc.shape.astype(np.int32), "pc_pts": pc.pts.astype(np.float32), "pc_len": pc.length.astype(np.float32),
           "pc_cur": pc.curved.astype(np.uint8), "pc_edges": pc.edges.astype(np.int32), "pc_fg": fg.astype(np.uint8),
           "pc_cls": cls.astype(np.int8), "pc_inst": np.where(fg, obj, -1).astype(np.int32)}
    arr = with_facing(arr)
    info.update(pieces=int(len(pc)), fg=int(fg.sum()), edges=int((arr["pc_ekind"] == 1).sum()), facing=int((arr["pc_ekind"] == 2).sum()))
    return arr, info


def main(src, pack, out, *rest):
    src, pack, out = Path(src), Path(pack), Path(out)
    limit = int(rest[rest.index("--limit") + 1]) if "--limit" in rest else None
    zip_out = next((a for a in rest if a.endswith(".zip")), None)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    recs = [r for r in manifest["records"] if r.get("status") == "ok"][:limit]
    tot = {"pieces": 0, "fg": 0, "true": 0, "touch": 0, "hard": 0}
    upgrade = "--upgrade" in rest   # add facing links to an existing e12 (pieces kept)
    for i, r in enumerate(recs, 1):
        st = r["stem"]
        dst = out / r["npz"]
        if upgrade and dst.exists():
            with np.load(dst) as z:
                d = {k: z[k] for k in z.files}
            if "pc_ekind" in d:
                continue
            d.update(with_facing({k: d[k] for k in d if k.startswith("pc_")}))
            tmp = out / f"{st}.tmp.npz"
            np.savez_compressed(tmp, **d)
            tmp.replace(dst)
            print(f"[{i}/{len(recs)}] {st}: facing links {int((d['pc_ekind'] == 2).sum())}", flush=True)
            continue
        if dst.exists():
            continue
        t0 = time.time()
        with np.load(src / r["npz"]) as z:
            d = {k: z[k] for k in z.files}
        f = pack / "gt" / f"{st}.json"
        objects = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        arr, info = page_arrays(d, objects, st.startswith("vw_"), rng)
        for k in tot:
            tot[k] += info.get(k, 0)
        tmp = out / f"{st}.tmp.npz"
        np.savez_compressed(tmp, **d, **arr)
        tmp.replace(dst)
        print(f"[{i}/{len(recs)}] {st}: {info} ({time.time() - t0:.0f}s)", flush=True)
    shutil.copyfile(src / "folds.json", out / "folds.json")
    manifest["e12"] = {"pieces": "openstroke/pieces.py shape_pieces on candidates (true + touching + style hard negatives)", **tot}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("total", tot)
    if zip_out:
        with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_STORED) as zf:
            for f in sorted(out.iterdir()):
                zf.write(f, f"{out.name}/{f.name}")
        print(f"{zip_out}: {Path(zip_out).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:])
