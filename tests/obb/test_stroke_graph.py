"""OpenStroke10: primitive classes + same-object edges on the vector tokens; objects = connected components (docs/OPENSTROKE10.md)."""
import numpy as np
import torch

from openstroke.obb.engine import ctx_batch
from openstroke.obb.models.openobb25 import OpenOBB25, OpenOBB25Loss, graph_stats, member_summary
from openstroke.obb.models.stroke_graph import GraphPage, graph_targets, knn_pairs, pair_geometry

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_openobb25 import G, IMG, NC, _batch  # noqa: E402

torch.set_num_threads(1)


def test_knn_pairs_and_geometry():
    line = lambda x0, y0, x1, y1: torch.from_numpy(np.linspace([x0, y0], [x1, y1], 8)).float()
    pts = torch.stack([line(0, 0, 50, 0), line(50, 0, 100, 0), line(0, 10, 50, 10), line(500, 500, 520, 500)])
    e = knn_pairs(pts, k=4, radius=20.0)
    pairs = {tuple(p) for p in e.tolist()}
    assert (0, 1) in pairs and (0, 2) in pairs and all(3 not in p for p in pairs)   # the far stroke has no candidate
    g = pair_geometry(pts, torch.tensor([[0, 1], [0, 2]]), 20.0)
    assert g.shape == (2, 6)
    assert g[0, 1] < 1e-3                                  # end to end
    assert abs(float(g[1, 2]) - 1.0) < 1e-4                # parallel walls
    assert abs(float(g[1, 4]) - 0.5) < 1e-3                # 10 px apart sideways / radius 20


def test_graph_targets():
    inst = torch.tensor([0, 0, 1, -1, 1])
    labels = torch.tensor([2, 0])
    e = torch.tensor([[0, 1], [1, 2], [2, 4], [3, 4]])
    cls, same = graph_targets(inst, labels, e)
    assert cls.tolist() == [3, 3, 1, 0, 1]
    assert same.tolist() == [1, 0, 1, 0]


def _pair_graph():
    torch.manual_seed(0)
    a = OpenOBB25("s", num_classes=NC, img_size=IMG, max_det=50, groups=G, vec_dim=32, vec_layers=1).eval()
    b = OpenOBB25("s", num_classes=NC, img_size=IMG, max_det=50, groups=G, vec_dim=32, vec_layers=1, stroke_graph=True).eval()
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("graph.") for k in missing)
    return a, b


def test_graph_head_leaves_the_detector_unchanged():
    a, b = _pair_graph()
    x = torch.rand(2, 3, IMG, IMG)
    ctx = ctx_batch(_batch(), torch.device("cpu"))
    with torch.no_grad():
        oa, ob = a(x, ctx=ctx), b(x, ctx=ctx)
    for k in ("pred_boxes", "pred_logits", "member_logits"):
        assert torch.equal(oa[k], ob[k]), k
    assert "graph" not in oa and len(ob["graph"]) == 2
    cls, e, lg, rows = ob["graph"][0]
    assert cls.shape == (len(rows), NC + 1) and lg.shape == (len(e),)
    assert torch.allclose(cls, torch.zeros_like(cls))      # zero-initialised class head: uniform at step 0
    acc = {}
    graph_stats(ob, _batch(), acc)
    d, line = member_summary(acc)
    assert "graph" in d and "Graph (OpenStroke10" in line


def test_graph_loss_finite_and_decreases():
    torch.manual_seed(0)
    m = OpenOBB25("s", num_classes=NC, img_size=IMG, max_det=50, groups=G, vec_dim=32, vec_layers=1, stroke_graph=True).train()
    tg = _batch()
    x = torch.rand(2, 3, IMG, IMG)
    ctx = ctx_batch(tg, torch.device("cpu"))
    crit = OpenOBB25Loss(img_size=IMG, member_weight=1.0, graph_weight=1.0)
    opt = torch.optim.Adam(list(m.graph.parameters()) + list(m.vec.parameters()), lr=3e-3)
    hist = []
    for step in range(15):
        out = m(x, [{k: v for k, v in t.items() if k != "vec"} for t in tg], ctx=ctx)
        losses = crit(out, tg)
        assert torch.isfinite(losses["graph_cls"]) and torch.isfinite(losses["graph_edge"])
        loss = losses["graph_cls"] + losses["graph_edge"]
        opt.zero_grad()
        loss.backward()
        if step == 0:
            assert m.graph.cls[-1].weight.grad.abs().sum() > 0 and m.graph.edge[-1].weight.grad.abs().sum() > 0
        opt.step()
        hist.append(loss.item())
    assert hist[-1] < hist[0] * 0.7, hist


def test_graph_page_components():
    gp = GraphPage(3)
    c = lambda k: np.eye(4)[k] * 0.9 + 0.025                  # class probabilities (index 0 = no object)
    gp.add(np.array([10, 11, 12, 13, 14]), np.stack([c(1), c(1), c(3), c(3), c(0)]),
           np.array([[0, 1], [1, 2], [2, 3], [3, 4]]), np.array([0.9, 0.1, 0.8, 0.9]))
    gp.add(np.array([12, 13]), np.stack([c(3), c(3)]), np.array([[0, 1]]), np.array([0.95]))   # second tile: max kept
    objs = sorted(gp.objects({10: 1, 11: 1, 12: 1, 13: 1, 14: 1}), key=lambda o: o["members"])
    assert [o["members"] for o in objs] == [[10, 11], [12, 13]]                    # 14 is not an object stroke
    assert [o["class_id"] for o in objs] == [0, 2]
