"""OpenStroke43 (H168 / H169 / F238): clean g3 graph (NO facing_edges) + StrokeContext zero-init parity
from OpenStroke41 + context-matched neg_boxes v2 (4 unambiguous non-duct CAD symbol families with full
cKDTree neighbourhood & style stats recomputation, zero feature-shortcut leak under layer_stats)
+ blended select_by=object smoke tests."""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch = pytest.importorskip("torch")
torch.set_num_threads(1)


def _synth_win_with_context(n: int = 180, seed: int = 0):
    from openstroke.data import HAS_STROKE, LOG_WIDTH, STROKE
    from openstroke.features import FEATURE_NAMES, N_FEATURES
    from openstroke.layernames import MAX_TOKENS

    xy = np.stack([22.0 * (np.arange(n) % 15), 22.0 * (np.arange(n) // 15)], axis=1).astype(np.float64)
    lab = (np.arange(n) % 10 == 0).astype(np.uint8)
    oinst = np.where(lab > 0, np.cumsum(lab) // 2, -1).astype(np.int64)
    pos = np.arange(n, dtype=np.int64)
    edges = np.stack([pos[:-1], pos[1:]], axis=1)
    x = np.zeros((n, N_FEATURES), dtype=np.float32)
    x[:, HAS_STROKE] = 1.0
    x[:, FEATURE_NAMES.index("type_s")] = 1.0
    x[:, FEATURE_NAMES.index("stroke_opacity")] = 1.0
    x[:, FEATURE_NAMES.index("fill_opacity")] = 1.0
    x[:, LOG_WIDTH] = np.log1p(0.84 * 10.0)
    x[:, STROKE] = np.where(lab[:, None] > 0, [0.05, 0.05, 0.05], [0.6, 0.6, 0.6])
    # Realistic donor page-stats for true strokes (distinct from 0 so we verify copying!)
    x[lab > 0, FEATURE_NAMES.index("style_shape_share")] = 0.14
    x[lab > 0, FEATURE_NAMES.index("style_len_share")] = 0.27
    x[lab > 0, FEATURE_NAMES.index("seqno_rank")] = 0.62
    style = np.where(lab > 0, 7, 2).astype(np.int64)
    pts = np.zeros((n, 16, 2), dtype=np.float32)
    pts[:, :, 0] = np.linspace(-5.0, 5.0, 16)[None, :]
    return {
        "x": x, "xy": xy, "pts": pts, "layer": np.zeros(n, dtype=np.int64), "label": lab,
        "length": np.full(n, 10.0, dtype=np.float32), "ylayer": np.zeros(n, dtype=np.float32),
        "pos": pos, "lw": np.ones(n, dtype=np.float32), "mode": np.ones(n, dtype=np.int64),
        "ltab": np.zeros((1, MAX_TOKENS), dtype=np.int64), "style": style,
        "edges": edges, "ekind": np.ones(len(edges), dtype=np.int64),
        "ocls": np.where(lab > 0, 1, 0).astype(np.int64), "octr": xy.copy(), "oinst": oinst,
    }


def test_openstroke43_registry_and_exact_step0_parity_from_openstroke41(tmp_path):
    from test_model import _synthetic

    from openstroke.data import collate, load_page, window
    from openstroke.models import ARCHS, build, load_weights
    from openstroke.train import PRESETS, arch_name, describe_arch

    assert ARCHS["openstroke43"]["stroke_context"] is True
    assert ARCHS["openstroke43"].get("facing_edges", False) is False
    assert ARCHS["openstroke43"]["group_head"] is True
    preset_cfg = PRESETS["openstroke43"][1]
    assert preset_cfg["feature_drop"] == "layer_stats"
    assert preset_cfg["neg_boxes"] == 0.35
    assert preset_cfg["lr0"] == 0.0005
    assert preset_cfg["warmup_epochs"] == 1.0
    assert preset_cfg.get("facing_edges", False) is False
    assert preset_cfg.get("sparse_bg", 0.0) == 0.0
    assert preset_cfg.get("viewer_clean_iso", False) is False

    # describe_arch / arch_name must distinguish openstroke41, openstroke43 (no facing_edges), and openstroke42 (+ facing_edges)
    cfg43 = {"arch": "openstroke41", **ARCHS["openstroke41"], "stroke_context": True}
    assert describe_arch(cfg43) == "openstroke43"
    assert arch_name(cfg43) == "openstroke43"
    cfg42 = {**cfg43, "facing_edges": True}
    assert describe_arch(cfg42) == "openstroke42"

    tiny = dict(dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32, img_pretrained=False)
    m41 = build("openstroke41", 47, **tiny).eval()
    m43 = build("openstroke43", 47, **tiny).eval()
    # Graph tables stay clean g3 (n_kinds=8, 0 resized weights!)
    assert m43.gat[0].kind_b.num_embeddings == 8
    assert m43.edge_mlp.kind.num_embeddings == 8
    assert m43.img.stroke_context is not None
    load_weights(m43, m41.state_dict(), new_heads=True)

    data, _ = _synthetic(tmp_path)
    stem = sorted(p.stem for p in data.glob("*.npz"))[0]
    pg = load_page(data, stem, "hvac", tmp_path / "cache", graph=True, no_layers=True)
    b = collate([window(pg, 0, 128)])
    with torch.no_grad():
        y41 = m41(b)
        e41 = m41.last_edge[0].clone() if m41.last_edge is not None else None
        h41 = m41.last_h.clone()
        y43 = m43(b)
        e43 = m43.last_edge[0].clone() if m43.last_edge is not None else None
        h43 = m43.last_h.clone()
    assert torch.allclose(y41, y43, atol=1e-5), "OpenStroke43 must match OpenStroke41 stroke logits at step 0!"
    assert torch.allclose(h41, h43, atol=1e-5), "OpenStroke43 must match OpenStroke41 hidden features at step 0!"
    if e41 is not None:
        assert torch.allclose(e41, e43, atol=1e-5), "OpenStroke43 must match OpenStroke41 edge logits at step 0!"

    # Finite gradients through StrokeContext
    m43.train()
    out = m43(b)
    loss = out.square().mean() + (m43.last_edge[0].square().mean() if m43.last_edge is not None else 0.0)
    loss.backward()
    assert m43.img.stroke_context.out.weight.grad is not None
    assert torch.isfinite(m43.img.stroke_context.out.weight.grad).all()
    assert m43.img.stroke_context.out.weight.grad.abs().sum() > 0


def test_neg_boxes_v2_recomputes_context_features_with_zero_shortcut_leak():
    from openstroke.data import Augment, HAS_STROKE, LOG_WIDTH, STROKE, augment
    from openstroke.features import FEATURE_NAMES

    w = _synth_win_with_context(180, seed=3)
    n0 = len(w["x"])
    tru_xy0 = w["xy"][w["label"] > 0].copy()
    aug = Augment(fliplr=0, rot90=False, scale=0, color=0, gray=0, restyle=0, drop_tokens=0, neg_boxes=1.0)

    seen_multi_seg = False
    seen_single_seg = False
    for sd in range(8):
        out = augment({k: v.copy() if isinstance(v, np.ndarray) else v for k, v in w.items()}, aug, np.random.default_rng(sd))
        n1 = len(out["x"])
        assert n1 > n0
        syn = slice(n0, n1)
        xs = out["x"][syn]
        assert (out["label"][syn] == 0).all()
        assert (out["ocls"][syn] == 0).all()
        assert (out["oinst"][syn] == -1).all()
        assert (out["layer"][syn] == 0).all()
        assert (out["lw"][syn] == 1.0).all()
        # Exact donor style, opacity, and page-level style statistics copied (zero shortcut!)
        assert (out["style"][syn] == 7).all()
        assert np.allclose(xs[:, STROKE], [0.05, 0.05, 0.05], atol=1e-5)
        assert (xs[:, LOG_WIDTH] <= np.log1p(8.4) + 1e-5).all() and (xs[:, LOG_WIDTH] > 0.0).all()
        assert (xs[:, HAS_STROKE] == 1.0).all()
        assert (xs[:, FEATURE_NAMES.index("stroke_opacity")] == 1.0).all()
        assert (xs[:, FEATURE_NAMES.index("fill_opacity")] == 1.0).all()
        assert np.allclose(xs[:, FEATURE_NAMES.index("style_shape_share")], 0.14, atol=1e-5)
        assert np.allclose(xs[:, FEATURE_NAMES.index("style_len_share")], 0.27, atol=1e-5)
        seq = xs[:, FEATURE_NAMES.index("seqno_rank")]
        assert ((seq >= 0.569) & (seq <= 0.671)).all()
        # All 6 cKDTree context features must be populated (NOT stuck at 0.0!) and finite
        assert np.isfinite(xs).all() and np.isfinite(out["pts"][syn]).all()
        assert (xs[:, FEATURE_NAMES.index("nb50_log")] > 0.0).all()
        assert (xs[:, FEATURE_NAMES.index("nb50_same_layer")] == 1.0).all()
        assert (xs[:, FEATURE_NAMES.index("nb50_same_style")] > 0.0).all()
        assert (xs[:, FEATURE_NAMES.index("nb50_mean_log_len")] > 0.5).all()
        assert (xs[:, FEATURE_NAMES.index("nb10_log")] > 0.0).any()
        assert (xs[:, FEATURE_NAMES.index("nn_dist_log")] > 0.0).any()
        # Strict non-overlap clearance from real true strokes (including kind 3 arrow tips)
        dists = np.hypot(*(out["xy"][syn, None, :] - tru_xy0[None, :, :]).transpose(2, 0, 1)).min(axis=1)
        assert dists.min() >= 1.5
        nseg_vals = np.expm1(xs[:, FEATURE_NAMES.index("log_nseg")])
        if (nseg_vals >= 3.5).any():
            seen_multi_seg = True
        if (nseg_vals <= 1.2).any():
            seen_single_seg = True
    assert seen_multi_seg and seen_single_seg, "Expected both multi-segment (op=[4,0,0,0] / arrow) and single-segment symbols"


def test_openstroke43_trainer_smoke_finetune_from_41_and_infer(tmp_path):
    """End-to-end CPU smoke test of the exact E145 -> E148 upgrade path + Predictor."""
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
    # 1. Source OpenStroke41 checkpoint (simulates E145)
    assert main([
        "train", *common, "model=openstroke41", "dims=16,32", "enc_depths=1,1", "dec_depths=1",
        "patch=32", "img_pretrained=false", "epochs=1", "name=v41_src",
    ]) == 0
    src41 = tmp_path / "runs" / "v41_src" / "fold0" / "best.pt"

    # 2. Exact E148 fine-tune flags -> must produce openstroke43, facing_edges=False, feature_drop=['layer_stats']
    assert main([
        "train", *common, f"model={src41}",
        "stroke_context=true", "neg_boxes=0.35", "lr0=0.0005", "warmup_epochs=1.0",
        "epochs=1", "name=v43_ft",
    ]) == 0
    run43 = tmp_path / "runs" / "v43_ft" / "fold0"
    ck43 = torch.load(run43 / "best.pt", map_location="cpu", weights_only=False)
    assert ck43["config"]["arch"] == "openstroke43"
    assert ck43["config"]["stroke_context"] is True
    assert ck43["config"]["facing_edges"] is False
    assert ck43["config"]["feature_drop"] == ["layer_stats"]
    keep = ck43["model"]["feat_keep"].numpy()
    ls_idx = [FEATURE_NAMES.index(n) for n in FEATURE_GROUPS["layer_stats"]]
    ctx_idx = [FEATURE_NAMES.index(n) for n in FEATURE_GROUPS["context"]]
    assert (keep[ls_idx] == 0).all()
    assert (keep[ctx_idx] == 1).all(), "OpenStroke43 must keep context features active (no strict drop!)"

    # 3. Verify Predictor.predict_payload works cleanly on OpenStroke43 (g3 graph)
    pred = Predictor(run43 / "best.pt", device="cpu", threads=1, workers=1)
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
