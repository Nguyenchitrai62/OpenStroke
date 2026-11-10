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


def test_v11_starts_as_v10_and_keeps_positives_for_training():
    torch.manual_seed(0)
    opts = dict(num_classes=NC, img_size=IMG, max_det=50, groups=G, vec_dim=32, vec_layers=1, stroke_graph=True)
    v10 = OpenOBB25("s", **opts).eval()
    v11 = OpenOBB25("s", **opts, graph_foreground=True).eval()
    missing, unexpected = v11.load_state_dict(v10.state_dict(), strict=False)
    assert set(missing) == {"graph.foreground.weight", "graph.foreground.bias"} and not unexpected

    tg = _batch()
    ctx = ctx_batch(tg, torch.device("cpu"))
    x = torch.rand(2, 3, IMG, IMG)
    with torch.no_grad():
        before, after = v10(x, ctx=ctx), v11(x, ctx=ctx)
    for key in ("pred_boxes", "pred_logits", "member_logits"):
        assert torch.equal(before[key], after[key]), key
    for old, new in zip(before["graph"], after["graph"]):
        assert all(torch.equal(a, b) for a, b in zip(old, new))

    with torch.no_grad():
        v11.graph.foreground.bias.fill_(-1)
        rejected = v11(x, ctx=ctx)
        assert all(len(e) == 0 and (cls.softmax(-1)[:, 0] >= 0.5).all() for cls, e, _, _ in rejected["graph"])
        acc = {}
        graph_stats(rejected, tg, acc)
        stats, _ = member_summary(acc)
        assert stats["graph"]["gate_R_length"] == 0
        assert stats["graph"]["edge_candidates"] == 0

    v11.train()
    out = v11(x, [{k: v for k, v in t.items() if k != "vec"} for t in tg], ctx=ctx)
    losses = OpenOBB25Loss(img_size=IMG)(out, tg)
    assert torch.isfinite(losses["graph_foreground"]) and losses["graph_edge"] > 0
    losses["graph_foreground"].backward()
    assert v11.graph.foreground.weight.grad.abs().sum() > 0


def test_v11_checkpoint_survives_strip_and_inference(tmp_path):
    from openstroke.obb.cli import ARCH_KEYS, check_keys
    from openstroke.obb.predict import _load_model
    from openstroke.obb.strip import strip

    opts = dict(num_classes=NC, img_size=IMG, max_det=50, groups=G, vec_dim=32, vec_layers=1, stroke_graph=True)
    model = OpenOBB25("s", **opts, graph_foreground=True, graph_object_class=True).eval()
    with torch.no_grad():
        model.graph.foreground.bias.fill_(0.75)
    ckpt = tmp_path / "best.pt"
    args = dict(arch="v25", size="s", img=IMG, groups=G, vec_dim=32, vec_layers=1, num_top=50,
                stroke_graph=True, graph_foreground=True, graph_object_class=True, scales="1,0.5")
    switches = {"graph_foreground", "graph_hard_negative", "graph_object_class"}
    check_keys({key: True for key in switches})
    assert switches <= set(ARCH_KEYS)
    torch.save({"model": model.state_dict(), "args": args, "classes": ["a", "b", "c"]}, ckpt)
    restored, _, dep = _load_model(strip(ckpt), torch.device("cpu"))
    assert dep["graph_foreground"] is True and dep["graph_object_class"] is True and dep["scales"] == "1,0.5"
    assert torch.equal(restored.graph.foreground.bias, model.graph.foreground.bias)
    x = torch.rand(2, 3, IMG, IMG)
    ctx = ctx_batch(_batch(), torch.device("cpu"))
    with torch.no_grad():
        out = restored(x, ctx=ctx)
    assert out["graph"] and (out["graph_foreground_logits"][0] == 0.75).all()

    from openstroke.obb.predict import graph_objects
    gp = GraphPage(NC)
    vec = _batch()[0]["vec"][:2].numpy()
    gp.add(vec[:, 23].astype(int), np.array([[0.1, 0.6, 0.2, 0.1]] * 2),
           np.array([[0, 1]]), np.array([0.9]))
    page = {"tokens": vec}
    classes = ["a", "b", "c"]
    assert graph_objects(gp, page, classes, head=restored.graph) == graph_objects(gp, page, classes)
    with torch.no_grad():
        restored.graph.object_class[-1].bias[2] = 4
    assert graph_objects(gp, page, classes, head=restored.graph)[0]["class_id"] == 2


def test_v11_hard_edges_and_object_class_learn():
    from openstroke.obb.models.stroke_graph import graph_loss

    logits = torch.zeros(3, requires_grad=True)
    _, edge_loss = graph_loss(torch.zeros(2, 2), logits, torch.tensor([0, 1]), torch.tensor([1.0, 0.0, 0.0]),
                              hard_negative=torch.tensor([False, True, False]), known_edge=torch.ones(3, dtype=torch.bool))
    edge_loss.backward()
    assert torch.allclose(logits.grad, torch.tensor([-0.5, 0.375, 0.125]))

    torch.manual_seed(0)
    model = OpenOBB25("s", num_classes=NC, img_size=IMG, max_det=50, groups=G, vec_dim=32, vec_layers=1,
                      stroke_graph=True, graph_foreground=True, graph_hard_negative=True, graph_object_class=True).train()
    tg = _batch()
    ctx = ctx_batch(tg, torch.device("cpu"))
    out = model(torch.rand(2, 3, IMG, IMG), [{k: v for k, v in t.items() if k != "vec"} for t in tg], ctx=ctx)
    losses = OpenOBB25Loss(img_size=IMG)(out, tg)
    assert torch.isfinite(losses["graph_object_class"]) and losses["graph_object_class"] > 0
    losses["graph_object_class"].backward()
    assert model.graph.object_class[-1].weight.grad.abs().sum() > 0
