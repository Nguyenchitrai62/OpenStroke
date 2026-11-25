"""OpenStroke42 (H165 / F235): fg_swap no-layer leak fix, sparse_bg, neg_boxes, viewer_clean_iso,
StrokeContext + facing_edges (n_kinds=16) weight parity from OpenStroke41, and end-to-end CPU smoke tests."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch = pytest.importorskip("torch")
torch.set_num_threads(1)


def _synth_win(n: int = 120, x0: float = 0.0, label_every: int = 10, pos0: int = 0, with_layers: bool = False, seed: int = 0):
    from openstroke.data import HAS_STROKE, LOG_WIDTH, STROKE
    from openstroke.features import FEATURE_NAMES, N_FEATURES
    from openstroke.layernames import MAX_TOKENS

    rng = np.random.default_rng(seed)
    xy = np.stack([x0 + 25.0 * (np.arange(n) % 12), 25.0 * (np.arange(n) // 12)], axis=1).astype(np.float64)
    lab = (np.arange(n) % label_every == 0).astype(np.uint8)
    oinst = np.where(lab > 0, np.cumsum(lab) // 2, -1).astype(np.int64)
    pos = pos0 + np.arange(n, dtype=np.int64)
    edges = np.stack([pos[:-1], pos[1:]], axis=1)
    x = np.zeros((n, N_FEATURES), dtype=np.float32)
    x[:, HAS_STROKE] = 1.0
    x[:, FEATURE_NAMES.index("type_s")] = 1.0
    x[:, LOG_WIDTH] = np.log1p(0.5 * 10.0)
    x[:, STROKE] = np.where(lab[:, None] > 0, [0.1, 0.2, 0.8], [0.5, 0.5, 0.5])
    pts = np.zeros((n, 16, 2), dtype=np.float32)
    pts[:, :, 0] = np.linspace(-5.0, 5.0, 16)[None, :]
    layer = (np.arange(n, dtype=np.int64) % 3) if with_layers else np.zeros(n, dtype=np.int64)
    return {
        "x": x, "xy": xy, "pts": pts, "layer": layer, "label": lab,
        "length": np.full(n, 10.0, dtype=np.float32), "ylayer": np.zeros(n, dtype=np.float32),
        "pos": pos, "lw": np.ones(n, dtype=np.float32), "mode": np.ones(n, dtype=np.int64),
        "ltab": np.zeros((1, MAX_TOKENS), dtype=np.int64), "style": np.arange(n, dtype=np.int64) % 3,
        "edges": edges, "ekind": np.ones(len(edges), dtype=np.int64),
        "ocls": np.where(lab > 0, 1, 0).astype(np.int64), "octr": xy.copy(), "oinst": oinst,
    }


def test_fg_swap_and_region_swap_have_zero_layer_leak_under_no_layers():
    """F235 Bug #1: under no_layers=True (all layer==0), fg_swap & region_swap must keep layer==0 everywhere."""
    from openstroke.data import fg_swap, region_swap

    host = _synth_win(120, 0.0, label_every=10, pos0=0, with_layers=False, seed=0)
    donor = _synth_win(120, 500.0, label_every=6, pos0=500, with_layers=False, seed=1)

    out_fg = fg_swap(host, donor, np.random.default_rng(3))
    assert (out_fg["label"] > 0).any() and (out_fg["label"] == 0).any()
    assert (out_fg["layer"] == 0).all(), "fg_swap leaked donor label into layer id under no_layers=True!"

    for sd in range(10):
        out_rg = region_swap(host, donor, np.random.default_rng(sd))
        assert (out_rg["layer"] == 0).all(), "region_swap leaked donor region into layer id under no_layers=True!"

    # When real PDF layers exist (max > 0), donor layers stay cleanly separated past host layers
    host_l = _synth_win(120, 0.0, label_every=10, pos0=0, with_layers=True, seed=0)
    donor_l = _synth_win(120, 500.0, label_every=6, pos0=500, with_layers=True, seed=1)
    out_fg_l = fg_swap(host_l, donor_l, np.random.default_rng(3))
    new = out_fg_l["pos"] > host_l["pos"].max()
    assert out_fg_l["layer"][new].min() > host_l["layer"].max()


def test_sparse_bg_drops_only_background_and_keeps_all_true_strokes():
    from openstroke.data import Augment, augment

    w = _synth_win(200, 0.0, label_every=10, seed=4)
    n_fg0 = int((w["label"] > 0).sum())
    n_bg0 = int((w["label"] == 0).sum())
    aug = Augment(fliplr=0, rot90=False, scale=0, color=0, gray=0, restyle=0, drop_tokens=0, sparse_bg=1.0)
    out = augment(w, aug, np.random.default_rng(9))
    assert int((out["label"] > 0).sum()) == n_fg0
    n_bg1 = int((out["label"] == 0).sum())
    assert 0.02 * n_bg0 <= n_bg1 <= 0.50 * n_bg0
    n = len(out["x"])
    assert all(len(out[k]) == n for k in ("xy", "pts", "layer", "label", "length", "pos", "lw", "ocls", "oinst"))


def test_neg_boxes_injects_same_style_non_overlapping_symbols_with_exact_strict_features():
    from openstroke.data import Augment, HAS_STROKE, LOG_WIDTH, STROKE, augment, collate
    from openstroke.features import FEATURE_NAMES

    w = _synth_win(160, 0.0, label_every=12, seed=7)
    n0 = len(w["x"])
    tru_xy0 = w["xy"][w["label"] > 0].copy()
    aug = Augment(fliplr=0, rot90=False, scale=0, color=0, gray=0, restyle=0, drop_tokens=0, neg_boxes=1.0)
    out = augment(w, aug, np.random.default_rng(11))
    n1 = len(out["x"])
    assert n1 > n0, "neg_boxes should inject synthetic hard-negative strokes"
    syn = slice(n0, n1)
    assert (out["label"][syn] == 0).all()
    assert (out["ocls"][syn] == 0).all()
    assert (out["oinst"][syn] == -1).all()
    assert (out["layer"][syn] == 0).all()
    assert (out["lw"][syn] == 1.0).all()
    # Exact colour copied from true strokes ([0.1, 0.2, 0.8]) and width <= donor (neg_boxes v4 includes ~0.5x thinner desk rows)
    assert np.allclose(out["x"][syn, STROKE], [0.1, 0.2, 0.8], atol=1e-5)
    assert (out["x"][syn, LOG_WIDTH] <= np.log1p(5.0) + 1e-5).all() and (out["x"][syn, LOG_WIDTH] > 0.0).all()
    assert (out["x"][syn, HAS_STROKE] == 1.0).all()
    assert (out["x"][syn, FEATURE_NAMES.index("frac_l")] == 1.0).all()
    assert (out["length"][syn] > 1.0).all()
    assert np.isfinite(out["x"][syn]).all() and np.isfinite(out["pts"][syn]).all()
    # Minimum clearance from real true strokes
    dists = np.hypot(*(out["xy"][syn, None, :] - tru_xy0[None, :, :]).transpose(2, 0, 1)).min(axis=1)
    assert dists.min() >= 1.5
    # Collates cleanly with edges
    b = collate([out])
    assert b["x"].shape[1] == n1 and bool(b["emask"].any())


def test_viewer_isolated_weights_and_lw_masked_losses():
    from openstroke.data import Page, viewer_isolated_weights
    from openstroke.features import N_FEATURES
    from openstroke.models.group_head import GroupClsHead, group_cls_loss
    from openstroke.models.openstroke2 import closure_loss, edge_loss

    # Build a Viewer page with:
    #   obj 0 (Duct, ocls=1) + obj 1 (Chân rẽ, ocls=7) touching at x=0..30
    #   obj 2 (isolated Chân rẽ, ocls=7) far away at x=500 (must get lw=0!)
    #   obj 3 (isolated small Duct, ocls=1, diag=14pt) at x=800 (must get lw=0!)
    #   obj 4 (isolated LONG Duct, ocls=1, diag=60pt) at x=1200 (must keep lw=1!)
    xy = np.array([
        [10, 0], [20, 0],     # obj 0 (Duct)
        [25, 0],              # obj 1 (Chân rẽ touching obj 0)
        [500, 0], [506, 0],   # obj 2 (isolated Chân rẽ)
        [800, 0],             # obj 3 (isolated small Duct, diag ~10pt)
        [1200, 0],            # obj 4 (isolated long Duct, diag ~60pt)
        [2000, 0],            # bg stroke
    ], dtype=np.float32)
    pts = np.zeros((len(xy), 16, 2), dtype=np.float16)
    for i, half in enumerate([10, 10, 5, 4, 4, 5, 30, 5]):
        pts[i, :, 0] = np.linspace(-half, half, 16)
    x = np.zeros((len(xy), N_FEATURES), dtype=np.float32)
    label = np.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8)
    oinst = np.array([0, 0, 1, 2, 2, 3, 4, -1], dtype=np.int64)
    ocls = np.array([1, 1, 7, 7, 7, 1, 1, 0], dtype=np.int64)
    pg_vw = Page("vw_test_p0001", "hvac", x, xy, pts, np.zeros(len(xy), np.int64), label,
                 np.full(len(xy), 10.0, np.float32), np.arange(len(xy), dtype=np.int64), ocls=ocls, oinst=oinst)
    w_vw = viewer_isolated_weights(pg_vw)
    assert w_vw[:3].tolist() == [1.0, 1.0, 1.0]       # connected objects kept
    assert w_vw[3:6].tolist() == [0.0, 0.0, 0.0]      # isolated Chân rẽ + small isolated Duct ignored!
    assert w_vw[6] == 1.0 and w_vw[7] == 1.0          # long isolated Duct & bg kept

    # GT page (doc000*) MUST NEVER be masked!
    pg_gt = Page("doc000180_p0001", "hvac", x, xy, pts, np.zeros(len(xy), np.int64), label,
                 np.full(len(xy), 10.0, np.float32), np.arange(len(xy), dtype=np.int64), ocls=ocls, oinst=oinst)
    assert (viewer_isolated_weights(pg_gt) == 1.0).all()

    # Verify closure_loss, edge_loss, and group_cls_loss ignore lw==0 strokes
    z = torch.tensor([[3.0, 3.0, 3.0, -6.0, -6.0, -6.0, 3.0, -3.0]], requires_grad=True)
    b = {
        "label": torch.tensor(label)[None], "mask0": torch.ones(1, len(xy), dtype=torch.bool),
        "oinst": torch.tensor(oinst)[None], "ocls": torch.tensor(ocls)[None],
        "lw": torch.tensor(w_vw)[None], "xy0": torch.tensor(xy)[None],
        "length": torch.full((1, len(xy)), 10.0), "x": torch.tensor(x)[None],
    }
    cl = closure_loss(z, b)
    cl.backward()
    assert z.grad[0, 3].item() == 0.0 and z.grad[0, 4].item() == 0.0  # ignored obj 2 gets 0 closure gradient!

    el = edge_loss((torch.tensor([5.0], requires_grad=True), torch.tensor([3]), torch.tensor([4])), b, pairs=True)
    assert el.detach().item() == 0.0  # edge between two lw==0 strokes carries 0 loss!

    gh = GroupClsHead(8)
    h = torch.randn(1, len(xy), 8, requires_grad=True)
    gl = group_cls_loss(gh, h, z.detach(), None, b, torch.ones(8))
    assert torch.isfinite(gl)


def test_openstroke42_loads_openstroke41_weights_with_exact_parity_and_facing_grads(tmp_path):
    from test_model import _synthetic

    from openstroke.data import collate, load_page, window
    from openstroke.models import ARCHS, build, load_weights
    from openstroke.train import PRESETS, arch_name, describe_arch

    assert ARCHS["openstroke42"]["stroke_context"] is True
    assert ARCHS["openstroke42"]["facing_edges"] is True
    assert PRESETS["openstroke42"][1]["feature_drop"] == "strict"
    assert PRESETS["openstroke42"][1]["sparse_bg"] == 0.3
    assert PRESETS["openstroke42"][1]["neg_boxes"] == 0.5
    assert PRESETS["openstroke42"][1]["viewer_clean_iso"] is True
    assert describe_arch({"arch": "openstroke41", **ARCHS["openstroke41"], "stroke_context": True, "facing_edges": True}) == "openstroke42"
    assert arch_name({"arch": "openstroke41", **ARCHS["openstroke41"], "stroke_context": True, "facing_edges": True}) == "openstroke42"

    tiny = dict(dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32, img_pretrained=False)
    m41 = build("openstroke41", 47, **tiny).eval()
    m42 = build("openstroke42", 47, **tiny).eval()
    assert m41.gat[0].kind_b.num_embeddings == 8 and m42.gat[0].kind_b.num_embeddings == 16
    assert m41.edge_mlp.kind.num_embeddings == 8 and m42.edge_mlp.kind.num_embeddings == 16
    load_weights(m42, m41.state_dict(), new_heads=True)
    assert m42.img.stroke_context is not None
    # Rows 8..15 zero-initialized
    assert (m42.gat[0].kind_b.weight[8:] == 0).all()
    assert (m42.edge_mlp.kind.weight[8:] == 0).all()

    data, _ = _synthetic(tmp_path)
    stem = sorted(p.stem for p in data.glob("*.npz"))[0]
    pg = load_page(data, stem, "hvac", tmp_path / "cache", graph=True, no_layers=True)
    b = collate([window(pg, 0, 128)])
    with torch.no_grad():
        y41 = m41(b)
        e41 = m41.last_edge[0].clone() if m41.last_edge is not None else None
        y42 = m42(b)
        e42 = m42.last_edge[0].clone() if m42.last_edge is not None else None
    assert torch.allclose(y41, y42, atol=1e-5), "OpenStroke42 must match OpenStroke41 stroke logits at step 0!"
    if e41 is not None:
        assert torch.allclose(e41, e42, atol=1e-5), "OpenStroke42 must match OpenStroke41 edge logits at step 0!"

    # Test finite gradients through StrokeContext AND facing edges (node 0 has >=2 neighbours with ekind=1 and ekind=8)
    m42.train()
    for ga in m42.gat:
        torch.nn.init.normal_(ga.out.weight, std=0.1)
    edges_mix = torch.tensor([[[0, 1], [0, 2], [1, 2]]], dtype=torch.long)
    ek_mix = torch.tensor([[1, 8, 8]], dtype=torch.long)
    em_mix = torch.tensor([[True, True, True]], dtype=torch.bool)
    b_fac = {**b, "edges": edges_mix, "ekind": ek_mix, "emask": em_mix}
    out = m42(b_fac)
    loss = out.square().mean() + (m42.last_edge[0].square().mean() if m42.last_edge is not None else 0.0)
    loss.backward()
    assert m42.img.stroke_context.out.weight.grad is not None and torch.isfinite(m42.img.stroke_context.out.weight.grad).all()
    assert m42.gat[0].kind_v.weight.grad is not None and m42.gat[0].kind_v.weight.grad[8].abs().sum() > 0
    assert m42.gat[0].kind_b.weight.grad is not None and m42.gat[0].kind_b.weight.grad[8].abs().sum() > 0
    assert m42.edge_mlp.kind.weight.grad is not None and m42.edge_mlp.kind.weight.grad[8].abs().sum() > 0


def test_openstroke42_trainer_smoke_from_scratch_finetune_and_infer(tmp_path):
    """Full CPU smoke test: train OpenStroke42 from scratch, fine-tune from OpenStroke41 with strict+facing+stroke_context, and infer."""
    from test_model import _synthetic

    from openstroke.cli import main
    from openstroke.features import FEATURE_GROUPS, FEATURE_NAMES
    from openstroke.folds import propose
    from openstroke.infer import Predictor

    data, manifest = _synthetic(tmp_path)
    for f in data.glob("*.npz"):
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        n = len(d["label"])
        y = d["label"] > 0
        inst = np.where(y, np.arange(n) // 4, -1).astype(np.int32)
        cls_cycle = np.array([1, 2, 4, 7], dtype=np.int8)
        d["oinst"] = inst
        d["ocls"] = np.where(y, cls_cycle[np.maximum(inst, 0) % len(cls_cycle)], 0).astype(np.int8)
        d["octr"] = np.zeros((n, 2), np.float32)
        np.savez(f, **d)
    (data / "folds.json").write_text(json.dumps(propose(manifest, k=2)))

    common = [
        f"data={data}", "fold=0", "steps=2", "batch=2", "window=128", "val_batch=2",
        "workers=0", "device=cpu", "tta=false", f"project={tmp_path / 'runs'}",
    ]
    # 1. Train OpenStroke42 from scratch
    assert main([
        "train", *common, "model=openstroke42", "dims=16,32", "enc_depths=1,1", "dec_depths=1",
        "patch=32", "img_pretrained=false", "epochs=1", "name=v42_new",
    ]) == 0
    run42 = tmp_path / "runs" / "v42_new" / "fold0"
    ck42 = torch.load(run42 / "best.pt", map_location="cpu", weights_only=False)
    assert ck42["config"]["arch"] == "openstroke42"
    assert ck42["config"]["stroke_context"] is True and ck42["config"]["facing_edges"] is True
    assert ck42["config"]["feature_drop"] == ["strict"]
    keep = ck42["model"]["feat_keep"].numpy()
    ctx_idx = [FEATURE_NAMES.index(n) for n in FEATURE_GROUPS["strict"]]
    assert (keep[ctx_idx] == 0).all()

    # 2. Train tiny OpenStroke41, then fine-tune into OpenStroke42 (exact E145 -> E146 Colab upgrade path!)
    assert main([
        "train", *common, "model=openstroke41", "dims=16,32", "enc_depths=1,1", "dec_depths=1",
        "patch=32", "img_pretrained=false", "epochs=1", "name=v41_src",
    ]) == 0
    src41 = tmp_path / "runs" / "v41_src" / "fold0" / "best.pt"
    assert main([
        "train", *common, f"model={src41}",
        "stroke_context=true", "facing_edges=true", "feature_drop=strict",
        "sparse_bg=0.3", "neg_boxes=0.5", "viewer_clean_iso=true", "scale=0.35", "restyle=0.8",
        "epochs=1", "name=v42_ft",
    ]) == 0
    run_ft = tmp_path / "runs" / "v42_ft" / "fold0"
    ck_ft = torch.load(run_ft / "best.pt", map_location="cpu", weights_only=False)
    assert ck_ft["config"]["arch"] == "openstroke42"
    assert ck_ft["config"]["stroke_context"] is True and ck_ft["config"]["facing_edges"] is True
    assert ck_ft["config"]["feature_drop"] == ["strict"]
    assert (ck_ft["model"]["feat_keep"].numpy()[ctx_idx] == 0).all()

    # 3. Verify Predictor.predict_payload works cleanly on an OpenStroke42 checkpoint (with facing graph + supplement_fg_edges)
    pred = Predictor(run_ft / "best.pt", device="cpu", threads=1, workers=1)
    payload = {
        "shapes": [
            {"items": [["l", [0, 0], [60, 0]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0", "rect": [0, 0, 60, 0]},
            {"items": [["l", [0, 15], [60, 15]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0", "rect": [0, 15, 60, 15]},
            {"items": [["l", [60, 0], [60, 15]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0", "rect": [60, 0, 60, 15]},
            {"items": [["l", [0, 15], [0, 0]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0", "rect": [0, 0, 0, 15]},
        ],
        "document_metadata": {"page_width": 200.0, "page_height": 200.0},
    }
    out = pred.predict_payload(payload, mode="hvac", tta=1)
    assert out["prob"].shape == (4,) and np.isfinite(out["prob"]).all()
    assert "group_objects" in out and out["group_objects"]["cluster_probs"].shape[1] == 8
