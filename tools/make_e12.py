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
from openstroke.pieces import piece_objects  # noqa: E402
from openstroke.relabel import page_segments_of  # noqa: E402

from openstroke.piece_candidates import candidate_pieces, candidates, with_facing  # noqa: E402, F401


def page_arrays(d: dict, objects: list[dict], viewer: bool, rng: np.random.Generator, pick: np.ndarray | None = None) -> tuple[dict, dict]:
    """``pick``: the strokes that seed candidates and cut pieces; None = the pipe label (train pages), else the stage-1 picks (val pages)."""
    segs = page_segments_of(d)
    seed = d["label"] > 0 if pick is None else np.asarray(pick, bool)
    pc, info = candidate_pieces(d, segs, rng, seed)
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
    # --val-picks <val_scores.npz> <thr>: rebuild the val pages from a stage-1 model's picks (prob >= thr) instead of their labels
    vp = rest.index("--val-picks") if "--val-picks" in rest else -1
    picks = np.load(rest[vp + 1]) if vp >= 0 else None
    pick_thr = float(rest[vp + 2]) if vp >= 0 else None
    if picks is not None:
        recs = [r for r in recs if r.get("split") == "val"]
        missing = [r["stem"] for r in recs if r["stem"] not in picks.files]
        if missing:
            sys.exit(f"--val-picks: no scores for {missing}")
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
        if dst.exists() and picks is None:
            continue
        t0 = time.time()
        with np.load(src / r["npz"]) as z:
            d = {k: z[k] for k in z.files}
        f = pack / "gt" / f"{st}.json"
        objects = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        pick = None
        if picks is not None:   # val_scores rows follow the loaded page; <stem>__index maps them back to npz rows
            s = np.zeros(len(d["label"]))
            s[picks[f"{st}__index"]] = picks[st].astype(float)
            pick = s >= pick_thr
        if pick is not None:
            d = {k: v for k, v in d.items() if not k.startswith("pc_")}
            d["pc_pick_thr"] = np.float32(pick_thr)
        arr, info = page_arrays(d, objects, st.startswith("vw_"), rng, pick)
        for k in tot:
            tot[k] += info.get(k, 0)
        tmp = out / f"{st}.tmp.npz"
        np.savez_compressed(tmp, **d, **arr)
        tmp.replace(dst)
        print(f"[{i}/{len(recs)}] {st}: {info} ({time.time() - t0:.0f}s)", flush=True)
    shutil.copyfile(src / "folds.json", out / "folds.json")
    prev = out / "manifest.json"
    old = json.loads(prev.read_text(encoding="utf-8")).get("e12", {}) if prev.exists() else {}
    old = {k: v for k, v in old.items() if k not in ("pieces", "val_seed")}
    manifest["e12"] = {"pieces": "openstroke/pieces.py shape_pieces on candidates (seed + touching + style hard negatives); seed = pipe "
                                 "label on train pages, stage-1 picks on val pages", **(old if picks is not None else tot)}
    if picks is not None:
        manifest["e12"]["val_seed"] = {"scores": str(rest[vp + 1]), "threshold": pick_thr, "pages": len(recs)}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("total", tot)
    if zip_out:
        with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_STORED) as zf:
            for f in sorted(out.iterdir()):
                zf.write(f, f"{out.name}/{f.name}")
        print(f"{zip_out}: {Path(zip_out).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:])
