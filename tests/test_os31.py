"""OpenStroke31: the link head's loss balanced on true-true pairs (edge_pairs, H136), the openstroke31 name (CPU, tiny)."""

import os

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from openstroke.models.openstroke2 import EDGE_TB_WEIGHT, edge_loss  # noqa: E402

# tokens 0-3: one duct (true, object 0); 4-5: the next duct in a row (true, object 1); 6, 7: background
B = {"label": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]), "oinst": torch.tensor([[0, 0, 0, 0, 1, 1, -1, -1]])}
# links: 4 same-object (0-1, 1-2, 2-3, 4-5), 1 between the two ducts (3-4), 1 true-other (5-6), 1 other-other (6-7)
I = torch.tensor([0, 1, 2, 4, 3, 5, 6])
J = torch.tensor([1, 2, 3, 5, 4, 6, 7])


def _grad(z, pairs):
    z = z.clone().requires_grad_(True)
    edge_loss((z, I, J), B, pairs=pairs).backward()
    return z.grad


def test_pairs_loss_weighs_the_boundary_between_two_objects_like_all_same_object_links():
    z = torch.zeros(7)
    g = _grad(z, pairs=True)
    assert torch.isclose(g[:4].abs().sum(), g[4].abs(), rtol=1e-5)          # 4 same-object links together = the one boundary link
    assert torch.isclose(g[5].abs(), EDGE_TB_WEIGHT * g[4].abs(), rtol=1e-5)  # true-other link: EDGE_TB_WEIGHT of a term
    assert g[6] == 0                                                          # other-other: no loss
    old = _grad(z, pairs=False)
    assert g[4].abs() / g.abs().sum() > 2 * old[4].abs() / old.abs().sum()    # the boundary link's share of the gradient grows


def test_pairs_loss_targets():
    good = torch.tensor([6.0, 6.0, 6.0, 6.0, -6.0, -6.0, 3.0])
    bad = -good
    assert edge_loss((good, I, J), B, pairs=True) < 0.01 < 1.0 < edge_loss((bad, I, J), B, pairs=True)
    only_tt = {"label": torch.ones(1, 8, dtype=torch.long), "oinst": torch.zeros(1, 8, dtype=torch.long)}
    v = edge_loss((torch.zeros(7), I, J), only_tt, pairs=True)                 # a missing term (no other-object link) is skipped
    assert torch.isclose(v, torch.tensor(np.log(2.0), dtype=v.dtype), atol=1e-5)


def test_openstroke31_is_openstroke30_with_the_pairs_loss():
    from openstroke.models import ARCHS, build, from_config
    from openstroke.train import PRESETS, arch_name, describe_arch

    assert {k: v for k, v in ARCHS["openstroke31"].items() if k != "edge_pairs"} == ARCHS["openstroke30"]
    assert PRESETS["openstroke31"][1]["closure"] == 0.5 and PRESETS["openstroke31"][1]["edge_pairs"] is True
    e114 = {"arch": "openstroke2", **ARCHS["openstroke28"]}
    assert describe_arch({**e114, "rel_attn": 2, "edge_head": True, "edge_pairs": True}) == "openstroke31"
    named = {"arch": "openstroke28", **ARCHS["openstroke28"], "rel_attn": 2, "edge_head": True, "edge_pairs": True}
    assert describe_arch(named) == "openstroke31" and arch_name(named) == "openstroke31"   # E114 saved by name, fine-tuned as v31
    assert describe_arch({"arch": "openstroke30", **ARCHS["openstroke30"], "edge_pairs": False}) == "openstroke30"
    tiny = dict(dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32, img_pretrained=False)
    m = build("openstroke31", 47, **tiny)
    assert m.config["arch"] == "openstroke31" and m.config["edge_pairs"] is True and m.edge_pairs and m.edge_mlp is not None
    m2 = from_config(dict(m.config, n_features=47), weights_follow=True)
    assert m2.edge_pairs and set(m2.state_dict()) == set(build("openstroke30", 47, **tiny).state_dict())   # same weights as v30


def test_training_with_edge_pairs_uses_the_pairs_loss_and_keeps_the_switch(tmp_path, monkeypatch):
    """Tiny CPU run through the real trainer: the link loss is called with pairs=True, the checkpoint keeps edge_pairs, and a run started
    from it (model=<.pt>) inherits it."""
    import json

    from test_model import _synthetic

    import openstroke.models.openstroke2 as os2
    from openstroke.cli import main
    from openstroke.folds import propose

    data, manifest = _synthetic(tmp_path)
    for f in data.glob("*.npz"):
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        n = len(d["label"])
        y = d["label"] > 0
        d["oinst"] = np.where(y, np.arange(n) // 4, -1).astype(np.int32)
        d["ocls"] = np.where(y, 1, 0).astype(np.int8)
        d["octr"] = np.zeros((n, 2), np.float32)
        np.savez(f, **d)
    (data / "folds.json").write_text(json.dumps(propose(manifest, k=2)))
    seen = []
    real = os2.edge_loss
    monkeypatch.setattr(os2, "edge_loss", lambda *a, **k: (seen.append(k.get("pairs")), real(*a, **k))[1])
    common = [f"data={data}", "fold=0", "steps=2", "batch=2", "window=128", "val_batch=2", "workers=0", "device=cpu", "tta=false",
              f"project={tmp_path / 'runs'}"]
    assert main(["train", *common, "model=openstroke2", "dims=16,32", "enc_depths=1,1", "dec_depths=1", "patch=32", "epochs=1",
                 "rel_attn=1", "edge_head=true", "edge_pairs=true", "closure=0.5", "name=v31"]) == 0
    assert seen and all(p is True for p in seen)
    run = tmp_path / "runs" / "v31" / "fold0"
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["edge_pairs"] is True and ck["config"]["edge_head"] is True
    seen.clear()
    assert main(["train", *common, f"model={run / 'best.pt'}", "epochs=1", "name=again"]) == 0
    assert seen and all(p is True for p in seen)


class _AlwaysDuct:
    """Stand-in for the face classifier: every region is a Duct with probability 0.9."""
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.tile([0.1, 0.9], (len(X), 1))


def _cut_duct():
    """A duct 0..60 x 0..20 drawn as two halves cut by a line at x = 30 (shapes 0-6)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_e0 import line, shape

    from openstroke.geometry import page_segments

    shapes = [shape(line(0, 0, 30, 0)), shape(line(30, 0, 60, 0)), shape(line(60, 0, 60, 20)), shape(line(60, 20, 30, 20)),
              shape(line(30, 20, 0, 20)), shape(line(0, 20, 0, 0)), shape(line(30, 0, 30, 20))]
    eij = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [1, 6], [3, 6], [4, 6]])
    return page_segments(shapes), eij


def test_link_groups_join_picked_strokes_over_strong_links():
    from openstroke.hvac_faces import link_groups

    picked = np.array([True, True, True, False])
    lab = link_groups(4, picked, np.array([[0, 1], [1, 2], [2, 3]]), np.array([0.9, 0.2, 0.9]), link=0.5)
    assert lab[0] == lab[1] != lab[2] and lab[3] == -1


def test_edge_objects_merge_a_cut_duct_only_when_its_halves_are_linked():
    from openstroke.hvac_faces import edge_objects

    segs, eij = _cut_duct()
    prob = np.ones(7)
    curved = np.zeros(7, bool)
    clf, classes = _AlwaysDuct(), ["none", "Duct"]
    strong = np.full(len(eij), 0.9)
    one = edge_objects(segs, prob, curved, clf, classes, eij, strong, thr=0.5, link=0.5, conf=0.5)
    assert len(one) == 1 and one[0]["shape_name"] == "Duct" and one[0]["refined"] == "edge"
    xs = [v[0] for v in one[0]["vertices"]]
    assert min(xs) < 1 and max(xs) > 59
    weak = strong.copy()
    weak[[0, 2, 3]] = 0.1     # the sides across the cut (0-1, 3-4) and 2-3: the two halves are different objects
    weak[[6, 7, 8, 9]] = 0.1  # the cut line belongs to neither side strongly
    two = edge_objects(segs, prob, curved, clf, classes, eij, weak, thr=0.5, link=0.5, conf=0.5)
    assert len(two) == 2
