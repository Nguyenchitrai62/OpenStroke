"""OpenStroke34: pieces of strokes (cut at corners, straight / arc changes and where another stroke ends on them), piece labels, pieces in
windows / batches, the piece head with confidence passed along touching pieces, and a tiny training run (CPU)."""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")


def _shapes():
    """Shape 0: one polyline item = duct wall (0,0)->(30,0), a quarter arc to (40,10), a wall (40,10)->(40,60); shape 1: a joint line ending
    on the middle of the long wall at (40, 35); shape 2: an unrelated line."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_e0 import shape

    poly = [("l", (0, 0), (30, 0)), ("c", (30, 0), (35.5, 0), (40, 4.5), (40, 10)), ("l", (40, 10), (40, 60))]
    return [shape(*poly), shape(("l", (40, 35), (70, 35))), shape(("l", (100, 100), (120, 100)))]


def test_pieces_cut_at_arcs_and_where_another_stroke_ends():
    from openstroke.geometry import page_segments
    from openstroke.pieces import shape_pieces

    segs = page_segments(_shapes())
    pc = shape_pieces(segs, [0, 1])
    mine = np.flatnonzero(pc.shape == 0)
    assert len(mine) == 4                                    # wall | arc | wall up to the joint | wall after it
    assert pc.curved[mine].tolist() == [False, True, False, False]
    lens = sorted(np.round(pc.length[mine]).tolist())
    assert lens == [16.0, 25.0, 25.0, 30.0]                 # arc ~15.7 pt; the 50 pt wall cut at (40, 35) -> 25 / 25; first wall 30
    # pieces that meet are linked (wall-arc, arc-wall, wall-wall at the cut, the joint line at the T)
    linked = {tuple(e) for e in pc.edges.tolist()}
    joint = int(np.flatnonzero(pc.shape == 1)[0])
    assert any(joint in e for e in linked) and len(linked) >= 4
    assert pc.pts.shape[1:] == (8, 2)


def test_piece_objects_follow_the_nearest_outline():
    from openstroke.geometry import page_segments
    from openstroke.pieces import piece_objects, shape_pieces

    segs = page_segments(_shapes())
    pc = shape_pieces(segs, [0])
    duct = {"shape_name": "Duct", "vertices": [[-1, -1], [30, -1], [30, 10], [-1, 10]]}
    elbow = {"shape_name": "Elbow", "vertices": [[30, -1], [41, -1], [41, 10], [30, 10]]}
    duct2 = {"shape_name": "Duct", "vertices": [[30, 10], [41, 10], [41, 61], [30, 61]]}
    obj, share = piece_objects(pc, [duct, elbow, duct2])
    assert obj.tolist()[:2] == [0, 1] and set(obj.tolist()[2:]) == {2} and (share > 0.5).all()


def _page_with_pieces(tmp_path):
    from test_model import _synthetic

    from openstroke.data import load_page
    from openstroke.pieces import shape_pieces
    from openstroke.relabel import page_segments_of

    data, manifest = _synthetic(tmp_path)
    for f in data.glob("*.npz"):
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        y = d["label"] > 0
        n = len(y)
        d["oinst"] = np.where(y, np.arange(n) // 4, -1).astype(np.int32)
        d["ocls"] = np.where(y, 1, 0).astype(np.int8)
        d["octr"] = np.zeros((n, 2), np.float32)
        segs = page_segments_of(d)
        cand = np.unique(np.r_[np.flatnonzero(y), np.arange(0, n, 7)])
        pc = shape_pieces(segs, cand, others=np.flatnonzero(y))
        fg = y[pc.shape]
        d.update(pc_shape=pc.shape.astype(np.int32), pc_pts=pc.pts.astype(np.float32), pc_len=pc.length.astype(np.float32),
                 pc_cur=pc.curved.astype(np.uint8), pc_edges=pc.edges.astype(np.int32), pc_fg=fg.astype(np.uint8),
                 pc_cls=np.where(fg, 1, -1).astype(np.int8), pc_inst=np.where(fg, pc.shape // 4, -1).astype(np.int32))
        np.savez(f, **d)
    stem = sorted(p.stem for p in data.glob("*.npz"))[0]
    return data, manifest, load_page(data, stem, "hvac", tmp_path / "cache", graph=True)


def test_batches_map_each_piece_to_its_shape_token(tmp_path):
    from openstroke.data import Augment, augment, collate, window

    data, manifest, page = _page_with_pieces(tmp_path)
    assert page.pieces is not None and len(page.pieces["ppos"]) > 0
    rng = np.random.default_rng(0)
    w = augment(window(page, 10, 150), Augment(fliplr=1.0, scale=0.2, drop_tokens=0.1), rng)
    b = collate([w, window(page, 0, 100)])
    tok, m = b["pc_tok"].numpy(), b["pc_mask"].numpy()
    pos = b["pos"].numpy()
    for bi in range(2):
        t = tok[bi][m[bi]]
        got = pos[bi][t]
        ids = b["pc_id"].numpy()[bi][m[bi]]
        want = page.pieces["ppos"][np.searchsorted(page.pieces["pid"], ids)] if np.all(np.diff(page.pieces["pid"]) > 0) else None
        assert (got >= 0).all()
        if want is not None:
            assert np.array_equal(got, want)
    e, em = b["pc_e"].numpy(), b["pc_emask"].numpy()
    assert (e[em] < m.sum(1).max()).all()


def test_piece_head_passes_confidence_along_links_and_trains():
    torch = pytest.importorskip("torch")
    from openstroke.data import PIECE_FEATS
    from openstroke.models.openstroke2 import PieceHead, piece_loss

    torch.manual_seed(0)
    B, N, C, P = 1, 6, 16, 4
    head = PieceHead(C, PIECE_FEATS, 10, pdim=16, rounds=3)
    h = torch.randn(B, N, C, requires_grad=True)
    logit = torch.tensor([[6.0, -6.0, -6.0, -6.0, 0.0, 0.0]])
    b = {"pc_tok": torch.tensor([[0, 1, 2, 3]]), "pc_mask": torch.ones(B, P, dtype=torch.bool), "pc_feat": torch.randn(B, P, PIECE_FEATS),
         "pc_c": torch.zeros(B, P, 2), "pc_e": torch.tensor([[[0, 1], [1, 2], [0, 0]]]), "pc_emask": torch.tensor([[True, True, False]]),
         "pc_fg": torch.tensor([[1.0, 1.0, 1.0, 0.0]]), "pc_cls": torch.tensor([[1, 2, 1, -1]]), "pc_inst": torch.tensor([[0, 1, 1, -1]]),
         "pc_len": torch.ones(B, P)}
    out = head(h, logit, b)
    assert out["fg"].shape == (B, P) and out["cls"].shape == (B, P, 10) and out["link"].shape == (2,)
    loss = piece_loss(out, b)
    loss.backward()
    assert torch.isfinite(loss) and h.grad is not None and float(h.grad.abs().sum()) > 0
    # piece 3 has no link: changing the confidence of shape 0 changes linked pieces 1 / 2, not piece 3
    with torch.no_grad():
        a = head(h, logit, b)["fg"]
        logit2 = logit.clone(); logit2[0, 0] = -6.0
        c = head(h, logit2, b)["fg"]
    assert not torch.allclose(a[0, 1], c[0, 1]) and torch.allclose(a[0, 3], c[0, 3])


def test_training_with_the_piece_head_saves_piece_outputs(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from openstroke.cli import main
    from openstroke.folds import propose

    data, manifest, _ = _page_with_pieces(tmp_path)
    (data / "folds.json").write_text(json.dumps(propose(manifest, k=2)))
    common = [f"data={data}", "fold=0", "steps=2", "batch=2", "window=128", "val_batch=2", "workers=0", "device=cpu", "tta=false",
              f"project={tmp_path / 'runs'}", "epochs=1"]
    assert main(["train", *common, "model=openstroke2", "dims=16,32", "enc_depths=1,1", "dec_depths=1", "patch=32", "edge_head=true",
                 "piece_head=true", "name=v34"]) == 0
    run = tmp_path / "runs" / "v34" / "fold0"
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["piece_head"] is True
    z = np.load(run / "val_scores.npz")
    stem = json.loads((run / "result.json").read_text())["val"][0]
    assert len(z[f"{stem}__pid"]) == len(z[f"{stem}__pfg"]) > 0 and z[f"{stem}__pcls"].shape[1] > 1
    assert z[f"{stem}__pe"].shape[1] == 2 and len(z[f"{stem}__pe"]) == len(z[f"{stem}__plink"])
    csv = (run / "results.csv").read_text(encoding="utf-8")
    assert "piece" in csv


def test_facing_links_join_the_two_walls_of_a_duct():
    """F187: the two walls of a duct do not touch; facing links join nearly parallel, overlapping pieces a duct width apart, not pieces in
    line or far apart."""
    from openstroke.pieces import facing_edges

    pts = np.stack([np.linspace([0, 0], [50, 0], 8), np.linspace([5, 20], [45, 20], 8),       # two walls 20 pt apart
                    np.linspace([60, 0], [90, 0], 8), np.linspace([0, 300], [50, 300], 8)])  # in line with wall 0; far away
    length = np.hypot(*(pts[:, -1] - pts[:, 0]).T)
    e = facing_edges(pts, length, np.zeros(4, bool))
    assert e.tolist() == [[0, 1]]


def test_crossing_strokes_neither_cut_nor_link():
    """User 2026-11-19: pieces touch only at ends. Layers drawn over each other cross a duct wall without belonging to it: a stroke crossing
    another (X) cuts nothing and links nothing; a stroke ending on another (T) cuts it and links to both halves."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_e0 import shape

    from openstroke.geometry import page_segments
    from openstroke.pieces import shape_pieces

    wall = shape(("l", (0, 0), (100, 0)))
    cross = shape(("l", (50, -20), (50, 20)))           # X through the wall
    tee = shape(("l", (80, 0), (80, 30)))                # T: ends on the wall
    pc = shape_pieces(page_segments([wall, cross]), [0, 1])
    assert len(pc) == 2 and len(pc.edges) == 0
    pc = shape_pieces(page_segments([wall, tee]), [0, 1])
    assert sorted(np.round(pc.length[pc.shape == 0]).tolist()) == [20.0, 80.0]
    t = int(np.flatnonzero(pc.shape == 1)[0])
    assert sum(t in e for e in pc.edges.tolist()) == 2
