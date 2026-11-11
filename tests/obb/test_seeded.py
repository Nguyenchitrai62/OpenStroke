"""OpenStroke13 / E92: detector boxes as seeds grown along the stroke graph inside the box (docs/OPENSTROKE13.md, H107)."""
import sys
from pathlib import Path

import numpy as np

from openstroke.obb.models.stroke_graph import GraphPage
from openstroke.obb.seeded import seeded_objects
from openstroke.obb.vector_validation import DatasetVectorGT

sys.path.insert(0, str(Path(__file__).parent))
from test_vector_validation import make_page  # noqa: E402


def _vec(rows):
    """rows: (x0, y0, x1, y1) per primitive (page px) -> vector npz dict with pid = 10 + index."""
    tok = np.zeros((len(rows), 24), np.float32)
    for i, (x0, y0, x1, y1) in enumerate(rows):
        tok[i, 1:17] = np.linspace([x0, y0], [x1, y1], 8).ravel()
    tok[:, 22] = -1
    tok[:, 23] = 10 + np.arange(len(rows))
    return {"tokens": tok}


def _page(edges, background=()):
    gp = GraphPage(2)
    pid = np.arange(10, 16)
    prob = np.tile([0.1, 0.8, 0.1], (6, 1))
    for b in background:
        prob[b - 10] = [0.9, 0.05, 0.05]
    e = np.asarray([[a - 10, b - 10] for a, b, _ in edges], np.int64).reshape(-1, 2)
    gp.add(pid, prob, e, np.asarray([q for _, _, q in edges], float))
    return gp


ROWS = [(0, 0, 10, 0), (10, 0, 20, 0), (20, 0, 30, 0), (30, 0, 40, 0), (200, 0, 210, 0), (12, 0, 18, 0)]
BOX = {"class": "Duct", "class_id": 0, "score": 0.9, "poly": [0, -5, 25, -5, 25, 5, 0, 5], "members": [10]}


def test_grows_inside_the_box_only():
    gp = _page([(10, 11, 0.95), (11, 12, 0.9), (12, 13, 0.95), (13, 14, 0.99)])
    out = seeded_objects(gp, [BOX], _vec(ROWS), ["Duct", "Elbow"], edge_thr=0.5, margin=2.0)
    assert len(out) == 1 and out[0]["source"] == "seeded" and out[0]["class_id"] == 0 and out[0]["score"] == 0.9
    assert out[0]["members"] == [10, 11, 12]          # 13 (middle at x 35) and 14 lie outside the box grown by 2 px
    assert out[0]["seed_members"] == 1


def test_edge_threshold_and_background_strokes_stop_growth():
    gp = _page([(10, 11, 0.6), (11, 12, 0.95), (10, 15, 0.99)], background=(15,))
    assert seeded_objects(gp, [BOX], _vec(ROWS), ["Duct", "Elbow"], edge_thr=0.7, margin=2.0)[0]["members"] == [10]
    assert seeded_objects(gp, [BOX], _vec(ROWS), ["Duct", "Elbow"], edge_thr=0.5, margin=2.0)[0]["members"] == [10, 11, 12]


def test_box_without_known_members_is_skipped():
    gp = _page([])
    assert seeded_objects(gp, [dict(BOX, members=[999])], _vec(ROWS), ["Duct", "Elbow"]) == []


def test_scorer_picks_rows_by_source(tmp_path):
    images, _, stem = make_page(tmp_path)
    gt = DatasetVectorGT(images, ["Duct", "Elbow"], comparison_pages=("doc000022",))
    rows = [dict(class_id=0, score=0.8, members=[10, 11], source="seeded@0.7"), dict(class_id=1, score=0.8, members=[30], source="seeded@0.7"),
            dict(class_id=0, score=0.8, members=[99], source="seeded@0.5"),
            dict(class_id=1, score=0.8, members=[30]), dict(class_id=0, score=0.8, members=[10, 11], source="graph")]
    assert gt.evaluate({stem: rows}, "seeded@0.7", ["Duct", "Elbow"])["R"] == 1.0
    assert gt.evaluate({stem: rows}, "seeded@0.5", ["Duct", "Elbow"])["R"] == 0.0   # a background stroke only
    assert gt.evaluate({stem: rows}, "members", ["Duct", "Elbow"])["R"] == 0.5      # only the source-less box row (the elbow)
    assert gt.evaluate({stem: rows}, "graph", ["Duct", "Elbow"])["R"] == 0.5        # only the graph row (the duct)


def test_validator_scores_every_path_and_keeps_the_best_seeded(tmp_path, monkeypatch):
    import functools

    import openstroke.obb.vector_validation as vv
    from openstroke.obb.object_validation import VectorEpochEvaluator

    monkeypatch.setattr(vv, "DatasetVectorGT", functools.partial(vv.DatasetVectorGT, comparison_pages=("doc000022",)))
    images, _, stem = make_page(tmp_path)
    ev = VectorEpochEvaluator(None, images, ["Duct", "Elbow"], {"stroke_graph": True, "graph_decode": "seeded", "batch": 8})
    try:
        assert ev.seeded_edges() == (0.3, 0.5, 0.7, 0.9)
        rows = [dict(class_id=0, score=0.8, members=[10, 11], source="seeded@0.9"), dict(class_id=1, score=0.8, members=[30], source="seeded@0.9"),
                dict(class_id=0, score=0.8, members=[99], source="seeded@0.5"), dict(class_id=0, score=0.8, members=[99], source="seeded@0.7"),
                dict(class_id=1, score=0.8, members=[30]), dict(class_id=0, score=0.8, members=[10, 11], source="graph")]
        s = ev.paths({stem: rows})
        assert s["how"] == "seeded@0.9" and s["F2"] == 1.0
        assert set(s["paths"]) == {"members", "graph", "seeded@0.3", "seeded@0.5", "seeded@0.7", "seeded@0.9"}
        assert s["paths"]["members"]["R"] == 0.5 and s["paths"]["seeded@0.5"]["R"] == 0.0
    finally:
        ev.close()
    off = VectorEpochEvaluator(None, images, ["Duct", "Elbow"], {"stroke_graph": True, "batch": 8})
    try:
        assert off.seeded_edges() is None                     # default decoding is unchanged
    finally:
        off.close()


def test_edge_loss_at_the_real_base_rate():
    import torch
    import torch.nn.functional as F

    from openstroke.obb.models.stroke_graph import graph_loss

    cls, ct = torch.zeros(4, 3), torch.tensor([0, 1, 1, 2])
    lg, et = torch.tensor([2.0, -1.0, 0.5, -3.0, 1.0]), torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0])
    _, le = graph_loss(cls, lg, ct, et, pos_weight=20.0)
    assert torch.allclose(le, F.binary_cross_entropy_with_logits(lg, et, pos_weight=torch.tensor(20.0)))
    _, balanced = graph_loss(cls, lg, ct, et)                       # pos_weight 0: the v10-v12 balanced means, unchanged
    lb = F.binary_cross_entropy_with_logits(lg, et, reduction="none")
    assert torch.allclose(balanced, lb[et > 0.5].mean() + lb[et <= 0.5].mean())


def test_graph_page_mean_pooling():
    for pool, want in (("max", 0.9), ("mean", 0.5)):
        gp = GraphPage(2, pool=pool)
        c = np.tile([0.1, 0.8, 0.1], (2, 1))
        gp.add(np.array([10, 11]), c, np.array([[0, 1]]), np.array([0.9]))
        gp.add(np.array([10, 11]), c, np.array([[0, 1]]), np.array([0.1]))
        assert abs(gp.edges[(10, 11)] - want) < 1e-9


def test_model_trains_with_edge_pos_weight():
    import torch

    from openstroke.obb.engine import ctx_batch
    from openstroke.obb.models.openobb25 import OpenOBB25, OpenOBB25Loss
    from test_openobb25 import G, IMG, NC, _batch

    torch.manual_seed(0)
    m = OpenOBB25("s", num_classes=NC, img_size=IMG, max_det=50, groups=G, vec_dim=32, vec_layers=1, stroke_graph=True,
                  graph_edge_pos_weight=20.0).train()
    assert m.graph_edge_pos_weight == 20.0
    tg = _batch()
    x = torch.rand(2, 3, IMG, IMG)
    ctx = ctx_batch(tg, torch.device("cpu"))
    crit = OpenOBB25Loss(img_size=IMG, member_weight=1.0, graph_weight=1.0)
    opt = torch.optim.Adam(list(m.graph.parameters()) + list(m.vec.parameters()), lr=3e-3)
    hist = []
    for _ in range(15):
        losses = crit(m(x, [{k: v for k, v in t.items() if k != "vec"} for t in tg], ctx=ctx), tg)
        loss = losses["graph_cls"] + losses["graph_edge"]
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
    assert hist[-1] < hist[0] * 0.7, hist
