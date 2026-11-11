"""V12 acceptance checks: one probability, real gradients, layer invariance and portable deployment."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openstroke.obb.engine import ctx_batch
from openstroke.obb.models.openobb25 import OpenOBB25, OpenOBB25Loss
from openstroke.obb.models.stroke_graph import soft_f2_loss
sys.path.insert(0, str(Path(__file__).parent))
from test_openobb25 import G, IMG, NC, _batch

torch.set_num_threads(1)
OPTS = dict(num_classes=NC, img_size=IMG, max_det=50, groups=G, vec_dim=32, vec_layers=1, stroke_graph=True)


def test_zero_residual_matches_learned_v10_and_layer_ids_do_not_matter():
    torch.manual_seed(42)
    old = OpenOBB25("s", **OPTS).eval()
    with torch.no_grad():
        old.graph.cls[-1].weight.normal_(0, 0.1)
        old.graph.edge[-1].weight.normal_(0, 0.1)
    new = OpenOBB25("s", **OPTS, graph_unified=True, graph_f2=True).eval()
    missing, extra = new.load_state_dict(old.state_dict(), strict=False)
    assert set(missing) == {"graph.foreground.weight", "graph.foreground.bias"} and not extra
    tg = _batch()
    x = torch.rand(2, 3, IMG, IMG)
    ctx = ctx_batch(tg, torch.device("cpu"))
    other = {k: v.clone() if torch.is_tensor(v) else v for k, v in ctx.items()}
    other["vec"][..., 21] = torch.arange(other["vec"].shape[1]) + 19
    with torch.no_grad():
        before, after, changed = old(x, ctx=ctx), new(x, ctx=ctx), new(x, ctx=other)
    for key in ("pred_boxes", "pred_logits", "member_logits"):
        assert torch.equal(before[key], after[key])
        assert torch.equal(after[key], changed[key])
    for control, treated, layer_changed in zip(before["graph"], after["graph"], changed["graph"]):
        assert all(torch.equal(a, b) for a, b in zip(control, treated))
        assert all(torch.equal(a, b) for a, b in zip(treated, layer_changed))
    # Every stroke has one shared HVAC probability, and the same candidate set in both modes.
    tok, pts = torch.randn(4, 32), torch.randn(4, 8, 2)
    new.graph.foreground.bias.data.fill_(-10)
    train = new.graph(tok, pts, train_foreground=torch.ones(4, dtype=torch.bool))
    infer = new.graph(tok, pts)
    assert all(torch.equal(a, b) for a, b in zip(train, infer))
    cls, edges, _, fg = infer
    assert len(edges) == 6  # low foreground does not change kNN or erase supervision
    assert torch.allclose(fg.sigmoid(), 1 - cls.softmax(-1)[:, 0], atol=1e-6)


def test_soft_f2_formula_length_weights_unknown_and_empty_pages():
    p = torch.tensor([0.5, 0.25, 0.9, 0.999])
    y = torch.tensor([True, False, False, True])
    known = torch.tensor([True, True, True, False])
    length = torch.tensor([4., 1., 2., 1000.])
    logit = p.logit().requires_grad_()
    loss = soft_f2_loss(logit, y, known, length)
    assert float(loss.detach()) == pytest.approx(1 - 10 / (10 + 8 + 2.05), abs=1e-6)
    loss.backward()
    assert logit.grad[0] < 0 and (logit.grad[1:3] > 0).all() and logit.grad[3] == 0
    negative = torch.zeros(3, requires_grad=True)
    nl = soft_f2_loss(negative, torch.zeros(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool), torch.ones(3))
    nl.backward()
    assert torch.isfinite(nl) and (negative.grad > 0).all()
    assert soft_f2_loss(negative, y[:3], torch.zeros(3, dtype=torch.bool), length[:3]) == 0
    # Formula retains a larger cost for a missed positive than an equally sized false positive.
    a = soft_f2_loss(torch.tensor([-3., -3.]), torch.tensor([True, False]), known[:2], torch.ones(2))
    b = soft_f2_loss(torch.tensor([3., 3.]), torch.tensor([True, False]), known[:2], torch.ones(2))
    assert a > b


def test_real_model_loss_backpropagates_to_shared_probability():
    model = OpenOBB25("s", **OPTS, graph_unified=True, graph_f2=True).train()
    tg = _batch()
    ctx = ctx_batch(tg, torch.device("cpu"))
    out = model(torch.rand(2, 3, IMG, IMG), [{k:v for k,v in t.items() if k != "vec"} for t in tg], ctx=ctx)
    losses = OpenOBB25Loss(img_size=IMG)(out, tg)
    assert "graph_f2" in losses and all(torch.isfinite(v) for v in losses.values())
    losses["graph_f2"].backward()
    for param in (model.graph.foreground.weight, model.graph.cls[-1].weight):
        assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="requires graph_unified"):
        OpenOBB25("s", **OPTS, graph_f2=True)


def test_strip_load_app_and_predict_keep_the_same_object_calibration(tmp_path):
    from openstroke.obb.strip import strip
    from openstroke.obb.predict import _load_model, conf_thresholds
    from openstroke.obb.cli import ARCH_KEYS, check_keys
    from app.obb_page import ObbPagePredictor
    model = OpenOBB25("s", **OPTS, graph_unified=True, graph_f2=True).eval()
    args = dict(arch="v25", size="s", img=IMG, groups=G, vec_dim=32, vec_layers=1, num_top=50,
                stroke_graph=True, graph_unified=True, graph_f2=True, scales="1,0.5")
    check_keys(dict(graph_unified=True, graph_f2=True))
    assert {"graph_unified", "graph_f2"} <= set(ARCH_KEYS)
    source = tmp_path / "source.pt"
    score = dict(best_conf=0.35, F2=0.61, P=0.5, R=0.65, protocol="dataset_vectors.sampled8.v1", paths="/content/drive/private")
    torch.save(dict(model=model.state_dict(), args=args, classes=["a","b","c"], object_score=score), source)
    deploy = strip(source, out=tmp_path / "renamed.pt")
    ck = torch.load(deploy, weights_only=False)
    assert "paths" not in ck["object_score"]
    assert ck["object_score"]["protocol"] == "dataset_vectors.sampled8.v1"
    restored, names, saved = _load_model(deploy, torch.device("cpu"))
    assert restored.graph.unified and restored.graph_f2 and saved["scales"] == "1,0.5"
    assert conf_thresholds(names, saved, None) == [0.35] * 3
    assert conf_thresholds(names, saved, 0.1) == [0.1] * 3
    app = ObbPagePredictor(deploy, threads=1)
    assert app.conf == 0.35 and app.val_f2 == 0.61 and "OpenStroke12" in app.describe()["arch"]


def test_public_api_uses_page_graph_and_saved_scales_threshold_not_detector_boxes(monkeypatch):
    from openstroke.obb.model import OpenOBB
    from openstroke.obb import predict as pp
    graph = SimpleNamespace(object_class=None)
    dummy = SimpleNamespace(graph=graph, member=object(), vec=object())
    api = OpenOBB.__new__(OpenOBB)
    api.model, api.device, api.compile, api._names = dummy, torch.device("cpu"), False, {0:"a"}
    api.args = dict(img=128, scale=1., scales="1,0.5", fit=False, stroke_graph=True, object_conf=0.3)
    monkeypatch.setattr(pp, "page_vectors", lambda *args: {"tokens": np.zeros((1,24))})
    calls=[]
    def image(*args, **kw):
        calls.append(kw["scale"])
        assert kw["graph"] is not None
        return []  # The page graph still returns objects when detector boxes are empty.
    monkeypatch.setattr(pp, "predict_image", image)
    poly=[0.,0.,10.,0.,10.,10.,0.,10.]
    objects=[dict(class_id=0, score=s, poly=poly, members=[i]) for i,s in enumerate([0.1,0.4,0.8])]
    monkeypatch.setattr(pp, "graph_objects", lambda *args, **kw: objects)
    result=api.predict(np.zeros((128,128,3),np.uint8), verbose=False, max_det=1)[0]
    assert calls == [1.,0.5] and len(result.obb) == 2
    assert result.members == [[2],[1]]
    assert result.obb.conf.tolist() == pytest.approx([0.8,0.4])


def test_app_erases_layers_and_keeps_graph_empty_instead_of_box_fallback(tmp_path, monkeypatch):
    from app.obb_page import ObbPagePredictor
    from openstroke.obb import cli
    import app.obb_page as page_app
    import make_obb_dataset as dataset
    app = ObbPagePredictor.__new__(ObbPagePredictor)
    app.path, app.args, app.names = tmp_path / "model.pt", dict(stroke_graph=True, graph_unified=True), ["duct"]
    info = dict(width=20., height=20., shapes=2, length=np.array([10., 20.]))
    def features(payload, *args):
        assert [s["layer"] for s in payload["shapes"]] == ["", ""]
        return info
    monkeypatch.setattr(page_app, "vector_page_info", features)
    monkeypatch.setattr(dataset, "render", lambda *args: np.zeros((80,80,3), np.uint8))
    monkeypatch.setattr(dataset, "vector_tokens", lambda *args: np.zeros((2,24), np.float32))
    # Even high-confidence detector boxes must not fill an empty graph.
    monkeypatch.setattr(cli, "predict", lambda *args, **kw: {"page": [dict(class_id=0, score=0.9, poly=[0]*8, members=[0,1])]})
    payload = dict(shapes=[dict(layer="duct"), dict(layer="architecture")])
    result = app.predict_payload(payload)
    assert result["objects"] == [] and (result["prob"] == 0).all()
    assert [s["layer"] for s in payload["shapes"]] == ["duct", "architecture"]
    assert "OpenStroke12" in result["source"]
