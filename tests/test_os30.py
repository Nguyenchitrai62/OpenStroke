"""OpenStroke30: graph attention along the link graph, same-object link head, closure loss, GT instances in batches (CPU, tiny)."""

import os

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from test_model import _synthetic  # noqa: E402

from openstroke.data import collate, load_page, window  # noqa: E402
from openstroke.features import N_FEATURES  # noqa: E402
from openstroke.models import build, load_weights  # noqa: E402
from openstroke.models.openstroke2 import OpenStroke2, closure_loss, edge_loss  # noqa: E402

TINY = dict(dims=(16, 32, 32), enc_depths=(1, 1, 1), dec_depths=(1, 1), patch=16, drop_path=0.1)


def _model(**kw):
    torch.manual_seed(0)
    return OpenStroke2(N_FEATURES, **{**TINY, **kw})


def _graph_page(tmp_path):
    data, _ = _synthetic(tmp_path)
    page = load_page(data, "doc000000_p0001_gt0", "fire", tmp_path / "cache", graph=True)
    assert page.edges is not None and len(page.edges)
    return page


def test_graph_attention_and_edge_head_start_as_the_model_without_them(tmp_path):
    page = _graph_page(tmp_path)
    b = collate([window(page, 0, 200)])
    base, new = _model(), _model(rel_attn=2, edge_head=True)
    load_weights(new, base.state_dict(), new_heads=True)          # what a run started from a checkpoint does
    for m in (base, new):
        m.set_normalization(page.x.mean(0), page.x.std(0))
        m.eval()
    with torch.no_grad():
        assert torch.allclose(base(b), new(b), atol=1e-6)
        z, i, j = new.last_edge
        assert z.shape == i.shape == j.shape and len(z) == int(b["emask"].sum()) and torch.isfinite(z).all()
        for ga in new.gat:
            torch.nn.init.normal_(ga.out.weight, std=0.1)
        y = new(b)
        assert not torch.allclose(base(b), y) and torch.isfinite(y).all()
        w = window(page, 0, 120)                                   # padding of other windows changes nothing
        a1 = new(collate([w]))[0, :120]
        padded = collate([w, window(page, 0, 300)])
        a2 = new({k: v[:1] for k, v in padded.items()})[0, :120]
    assert torch.allclose(a1, a2, atol=1e-4)


def test_graph_attention_weighs_neighbours_and_ignores_unlinked_tokens(tmp_path):
    """A token's update comes only from tokens it is linked to: changing an unlinked token leaves it unchanged."""
    page = _graph_page(tmp_path)
    m = _model(rel_attn=1)
    m.set_normalization(page.x.mean(0), page.x.std(0))
    for lin in (m.gat[0].out, m.gat[0].ffn[-1]):
        torch.nn.init.normal_(lin.weight, std=0.2)
    from openstroke.models.openstroke2 import flat_links
    b = collate([window(page, 0, 200)])
    i, j, k = flat_links(b)
    h = torch.randn(1, b["mask0"].shape[1], 16)
    out1 = m.gat[0](h, b, b["x"], (i, j, k))
    linked = set(i.tolist()) | set(j.tolist())
    alone = [t for t in range(int(b["mask0"].sum())) if t not in linked]
    other = [t for t in range(int(b["mask0"].sum())) if t in linked]
    assert alone and other
    h2 = h.clone()
    h2[0, alone[0]] += 5.0
    out2 = m.gat[0](h2, b, b["x"], (i, j, k))
    assert torch.allclose(out1[0, other], out2[0, other], atol=1e-5)


def test_closure_loss_sits_on_the_weakest_stroke_of_an_object():
    """Object 0 = tokens 0..3 (true), token 3 missed (logit -4); object 1 = tokens 4..5 picked; token 6 alone, token 7 other."""
    z = torch.tensor([[4.0, 4.0, 4.0, -4.0, 3.0, 3.0, -2.0, -5.0]], requires_grad=True)
    b = {"label": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]]), "mask0": torch.ones(1, 8, dtype=torch.bool),
         "oinst": torch.tensor([[0, 0, 0, 0, 1, 1, 2, -1]])}
    loss = closure_loss(z, b)
    loss.backward()
    g = z.grad[0]
    assert g[3] < -0.3 and (g[:3].abs() < 0.05).all()            # pushes the missed stroke up, not the picked ones
    assert g[6] == 0 and g[7] == 0                                 # one-stroke objects and other strokes: no closure loss
    z2 = torch.tensor([[4.0, 4.0, 4.0, 4.0, 3.0, 3.0, -2.0, -5.0]])
    assert closure_loss(z2, b) < 0.1 * float(loss)                 # a complete object costs ~nothing


def test_edge_loss_targets_same_object_links():
    b = {"label": torch.tensor([[1, 1, 1, 0]]), "oinst": torch.tensor([[0, 0, 1, -1]])}
    i, j = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])        # same object / two objects / true-other
    good = edge_loss((torch.tensor([6.0, -6.0, -6.0]), i, j), b)
    bad = edge_loss((torch.tensor([-6.0, 6.0, 6.0]), i, j), b)
    assert good < 0.01 < 1.0 < bad


def test_batches_carry_gt_instances(tmp_path):
    page = _graph_page(tmp_path)
    page.ocls = np.zeros(len(page), np.int64)
    page.octr = np.zeros((len(page), 2))
    page.oinst = np.arange(len(page), dtype=np.int64) // 3
    b = collate([window(page, 10, 50)])
    pos = b["pos"][0]
    ok = pos >= 0
    assert (b["oinst"][0][ok] == pos[ok] // 3).all() and (b["oinst"][0][~ok] == -1).all()


def test_build_keeps_the_new_switches():
    m = build("openstroke2", N_FEATURES, **TINY, rel_attn=2, edge_head=True)
    assert m.config["rel_attn"] == 2 and m.config["edge_head"] is True and len(m.gat) == 2 and m.edge_mlp is not None


def test_training_with_graph_attention_edge_head_and_closure(tmp_path):
    """Tiny CPU run through the real trainer: the new losses are computed, logged and the checkpoint keeps the switches; a second run
    started from that checkpoint (architecture from it, as model=<.pt>) loads it."""
    import json

    from openstroke.cli import main
    from openstroke.folds import propose

    data, manifest = _synthetic(tmp_path)
    for f in data.glob("*.npz"):                                   # GT instances: groups of 4 true shapes
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        n = len(d["label"])
        y = d["label"] > 0
        d["oinst"] = np.where(y, np.arange(n) // 4, -1).astype(np.int32)
        d["ocls"] = np.where(y, 1, 0).astype(np.int8)
        d["octr"] = np.zeros((n, 2), np.float32)
        np.savez(f, **d)
    (data / "folds.json").write_text(json.dumps(propose(manifest, k=2)))
    common = [f"data={data}", "fold=0", "steps=2", "batch=2", "window=128", "val_batch=2", "workers=0", "device=cpu", "tta=false",
              f"project={tmp_path / 'runs'}"]
    assert main(["train", *common, "model=openstroke2", "dims=16,32", "enc_depths=1,1", "dec_depths=1", "patch=32", "epochs=1",
                 "rel_attn=1", "edge_head=true", "closure=0.5", "name=v30"]) == 0
    run = tmp_path / "runs" / "v30" / "fold0"
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["rel_attn"] == 1 and ck["config"]["edge_head"] is True
    log = (run / "train.log").read_text(encoding="utf-8") if (run / "train.log").exists() else ""
    csv = (run / "results.csv").read_text(encoding="utf-8")
    assert "close" in csv + log and "edge" in csv + log
    assert main(["train", *common, f"model={run / 'best.pt'}", "epochs=1", "name=again"]) == 0
