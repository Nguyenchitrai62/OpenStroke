"""OpenStroke candidate geometry shared by e12 generation and inference on unlabelled PDFs."""
from __future__ import annotations

import numpy as np

from .pieces import SNAP, facing_edges, shape_pieces

HARD_SHARE, HARD_MAX = 0.05, 2000


def candidates(d: dict, segs, rng: np.random.Generator, pick: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Picked shapes (``pick``; default the pipe label), shapes whose end lies on a picked stroke or that a picked end lies on, and hard
    negatives drawn in a picked style. Val pages pass the stage-1 model's picks: no val label shapes their inputs (AGENTS 0.4)."""
    import shapely

    y = d["label"].astype(bool) if pick is None else np.asarray(pick, bool)
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


def candidate_pieces(d, segs, rng, seed):
    ids, info = candidates(d, segs, rng, seed)
    return shape_pieces(segs, ids, others=np.flatnonzero(seed)), info


def labelled_pieces(d: dict, objects: list[dict], viewer: bool, rng: np.random.Generator,
                    pick: np.ndarray | None = None) -> tuple[dict, dict]:
    """``pick``: the strokes that seed candidates and cut pieces; None = the pipe label (train pages), else the stage-1 picks (val pages)."""
    from .objects import class_of, general_class
    from .pieces import piece_objects
    from .relabel import page_segments_of

    segs = page_segments_of(d)
    seed = d["label"] > 0 if pick is None else np.asarray(pick, bool)
    pc, info = candidate_pieces(d, segs, rng, seed)
    obj, share = piece_objects(pc, objects)
    y = d["label"].astype(bool)
    fg = y[pc.shape] & (obj >= 0) & (share >= 0.5)
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


def predict_regions(model, page, info, prob, window_size, device, batch, amp):
    """Second pass on pieces cut from the model's own stroke picks; retain exact subsegment provenance."""
    from .data import attach_pieces
    from .engine.trainer import predict_page
    from .objects import HVAC_CLASSES

    segs = info["segments"]
    pc, _ = candidate_pieces(info["attrs"], segs, np.random.default_rng(0), prob >= 0.75)
    objects = []
    if len(pc):
        geom = with_facing(dict(pc_shape=pc.shape, pc_pts=pc.pts, pc_len=pc.length, pc_cur=pc.curved, pc_edges=pc.edges))
        geom.update(shape_offsets=segs.shape_offsets, pc_inst=np.full(len(pc), -1), pc_cls=np.full(len(pc), -1),
                    pc_fg=np.zeros(len(pc)))
        attach_pieces(page, geom)
        outputs = []
        predict_page(model, page, window_size, device, batch=batch, tta=1, amp_dtype=amp, piece_out=outputs)
        q = outputs[0]
        keep = np.flatnonzero(q["region_id"] >= 0)
        ordered = keep[np.argsort(q["region_id"][keep], kind="stable")]
        if len(ordered):
            for rows in np.split(ordered, np.flatnonzero(np.diff(q["region_id"][ordered])) + 1):
                objects.append({"class": HVAC_CLASSES[q["region_cls"][rows[0]]],
                                "piece_ids": q["pid"][rows].tolist()})
    return {"objects": objects, "piece_shape": pc.shape, "piece_start": pc.seg_start,
            "piece_stop": pc.seg_stop, "piece_segments": pc.sub, "piece_length": pc.length}
