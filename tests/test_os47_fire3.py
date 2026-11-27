"""OpenStroke47 (H176 / openstroke_fire3): Dedicated FIRE 3-class (0=other, 1=main, 2=branch) model tests:
  - 12D _graph_topo_fire_features (2-core big loops vs tiny symbol loops, tree peeling depth, filled strip, sprinkler proximity)
  - Multi-class loss (length-weighted CE + Lovasz-Softmax + auxiliary pipe-fg Lovasz)
  - End-to-end 3-class train/val (3-class metrics + best.pt selection)
  - Raw deploy API roundtrip (OpenStroke -> Results.cls_prob / class_names -> to_dict(compact=True) / from_dict / save_npz / from_npz)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch = pytest.importorskip("torch")
torch.set_num_threads(1)


def test_graph_topo_fire_detects_2core_loop_tree_backbone_filled_strip_and_sprinklers():
    from openstroke.data import HAS_FILL, HAS_STROKE, LOG_WIDTH, STROKE
    from openstroke.features import FEATURE_NAMES, N_FEATURES
    from openstroke.models.openstroke2 import _graph_topo_fire_features, _graph_topo_uni_features

    # Build a synthetic FIRE drawing with:
    #   0..3: large 400x200 pt main loop (4 nodes in a cycle -> in_2core = 1, core_span = 400!)
    #         node 0 is also a CAD Global-Width filled thick strip (has_fill=1, elong=0.05, len=400)
    #   4..6: tree branch off node 1 -> node 4 (trunk stub) -> node 5 (sub-branch) -> node 6 (leaf at sprinkler)
    #   7..9: tiny 5x5 pt closed sprinkler symbol loop at node 6 (span=5 < 60pt -> NOT in_2core_big!)
    N = 10
    x = np.zeros((1, N, N_FEATURES), dtype=np.float32)
    xy = np.zeros((1, N, 2), dtype=np.float32)
    length = np.zeros((1, N), dtype=np.float32)

    i_el = FEATURE_NAMES.index("elong")
    i_fc = FEATURE_NAMES.index("frac_c")
    i_cl = FEATURE_NAMES.index("closed_ends")
    i_dg = FEATURE_NAMES.index("log_bbox_diag")

    x[0, :, HAS_STROKE] = 1.0
    x[0, :, STROKE] = 0.0
    x[0, :, LOG_WIDTH] = np.log1p(0.15 * 10.0)  # all SAME 0.15pt black (like L30M-LUMI!)

    # Main loop 0-1-2-3-0 (400 x 200 pt)
    xy[0, 0:4] = [[0, 0], [400, 0], [400, 200], [0, 200]]
    length[0, 0:4] = [400, 200, 400, 200]
    x[0, 0, HAS_FILL] = 1.0
    x[0, 0, i_el] = 0.05  # filled thick polyline strip!

    # Branch chain 1 - 4 - 5 - 6
    xy[0, 4:7] = [[450, 0], [500, 0], [550, 0]]
    length[0, 4:7] = [50, 50, 50]

    # Tiny 5x5 sprinkler head closed loop 7-8-9-7 right at leaf node 6 (552, 0)
    xy[0, 7:10] = [[552, 0], [557, 0], [554, 5]]
    length[0, 7:10] = [5, 5, 5]
    x[0, 7:10, i_cl] = 1.0
    x[0, 7:10, i_fc] = 1.0
    x[0, 7:10, i_dg] = np.log1p(5.0)

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # large loop
        [1, 4], [4, 5], [5, 6],          # tree branch
        [7, 8], [8, 9], [9, 7],          # tiny symbol loop
    ]
    b = {
        "x": torch.from_numpy(x),
        "xy0": torch.from_numpy(xy),
        "length": torch.from_numpy(length),
        "mask0": torch.ones((1, N), dtype=torch.bool),
        "edges": torch.tensor([edges], dtype=torch.long),
        "emask": torch.ones((1, len(edges)), dtype=torch.bool),
    }

    tf = _graph_topo_fire_features(b)[0].numpy()
    tu = _graph_topo_uni_features(b)[0].numpy()
    assert tf.shape == (N, 12) and tu.shape == (N, 18)
    # Unified 18D dims [12..17] carry identical loop/peel/filled-strip/symbol signals as legacy 12D dims [6..11]
    assert np.allclose(tu[:, 12:18], tf[:, 6:12], atol=1e-5)
    # Filled strip detector flags node 0 only
    assert tf[0, 6] == 1.0 and tf[1, 6] == 0.0
    # Large main loop (0..3) is in_2core_all & in_2core_sw with big core_span (~400pt)
    assert np.all(tf[0:4, 7] == 1.0) and np.all(tf[0:4, 8] == 1.0)
    assert np.expm1(tf[0, 9] * 6.0) >= 350.0
    # Tiny sprinkler symbol loop (7..9, span=5pt) is EXCLUDED from in_2core_big!
    assert np.all(tf[7:10, 7] == 0.0) and np.all(tf[7:10, 8] == 0.0)
    # Tree branch (4..6) is NOT in 2-core, and peeling depth increases monotonically from leaf 6 -> 5 -> 4 -> loop 1!
    assert np.all(tf[4:7, 7] == 0.0)
    assert tf[6, 10] < tf[5, 10] < tf[4, 10] < tf[1, 10]
    # Leaf branch 6 near sprinkler symbol gets high near_small_sym, far main loop node 0 gets ~0!
    assert tf[6, 11] > 0.70 and tf[0, 11] < 0.05


def test_openstroke47_fire3_multiclass_loss_train_and_deploy_api_roundtrip(tmp_path):
    import argparse
    from test_model import _synthetic

    from openstroke.cli import main, yolo_args
    from openstroke.folds import propose
    from openstroke.infer import OpenStroke
    from openstroke.metrics import FIRE3_CLASSES, multiclass_metrics
    from openstroke.models import ARCHS, build, loss as loss_fn
    from openstroke.results import Results
    from openstroke.train import add_args, finalize, run_version

    assert ARCHS["openstroke47"]["stroke_classes"] == 3
    assert ARCHS["openstroke47"]["graph_topo_fire"] is True
    assert ARCHS["openstroke47"]["img_cad_v2"] is True
    assert ARCHS["openstroke47"]["img_max_px"] == 2048
    assert ARCHS["openstroke47"]["edge_style"] is True

    # CLI alias openstroke_fire3 -> unified openstroke46 backbone with auto dynamic classes
    ap = argparse.ArgumentParser()
    add_args(ap)
    raw = yolo_args(["data=datasets/fire_3cls_v1", "model=openstroke_fire3", "epochs=25"])
    args_c = ap.parse_args(raw)
    args_c._explicit = {t[2:].replace("-", "_") for t in raw if t.startswith("--")}
    args_c = finalize(args_c)
    assert run_version(args_c) == "openstroke46"
    assert args_c.stroke_classes == 0 and args_c.graph_topo_uni is True

    # Verify multiclass_metrics helper
    cp = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], dtype=np.float32)
    mc = multiclass_metrics([cp], [np.array([0, 1, 2])], [np.array([10.0, 20.0, 30.0])])
    assert mc["per_class"]["main"]["f2"] == 1.0 and mc["per_class"]["branch"]["f2"] == 1.0
    assert mc["pipe"]["f2"] == 1.0 and mc["fitness"] == 1.0

    # Synthetic 3-class FIRE dataset (0=other, 1=main, 2=branch) with manifest class_names
    data, manifest = _synthetic(tmp_path)
    manifest["class_names"] = ["other", "main", "branch"]
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for f in data.glob("*.npz"):
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        y_bin = d["label"].astype(np.uint8)
        fg = np.flatnonzero(y_bin > 0)
        y3 = np.zeros_like(y_bin)
        y3[fg[: len(fg) // 3]] = 1  # main
        y3[fg[len(fg) // 3 :]] = 2  # branch
        d["label"] = y3
        np.savez(f, **d)
    (data / "folds.json").write_text(json.dumps(propose(manifest, k=2)))

    # End-to-end 3-class train + val using ONE unified model=openstroke46 (auto-detects K=3 + drops unused HVAC heads!)
    assert main([
        "train", f"data={data}", "fold=0", "model=openstroke46", "modes=fire", "img_max_px=128",
        "dims=16,32", "enc_depths=1,1", "dec_depths=1", "patch=32", "img_pretrained=false",
        "steps=2", "batch=2", "window=128", "val_batch=2", "workers=0", "device=cpu", "tta=false",
        "epochs=1", f"project={tmp_path / 'runs'}", "name=fire3_smoke",
    ]) == 0

    run_dir = tmp_path / "runs" / "fire3_smoke" / "fold0"
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["arch"] == "openstroke46"
    assert ck["config"]["graph_topo_uni"] is True
    assert ck["config"]["stroke_classes"] == 3
    assert ck["config"]["class_names"] == ["other", "main", "branch"]
    assert ck["config"]["group_head"] is False and ck["config"]["edge_head"] is False
    res_json = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert "multiclass" in res_json and "main" in res_json["multiclass"]["per_class"]
    vs = np.load(run_dir / "val_scores.npz")
    assert any(k.endswith("__cls_prob") for k in vs.files)

    # Raw Deploy API roundtrip (OpenStroke -> Results -> compact JSON dict & npz -> Results)
    m = OpenStroke(run_dir / "best.pt", device="cpu", threads=1, workers=1)
    payload = {
        "page": {"width": 500, "height": 500},
        "shapes": [
            {"items": [["l", [10, 10], [200, 10]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0"},
            {"items": [["l", [200, 10], [200, 150]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0"},
            {"items": [["l", [200, 150], [10, 150]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0"},
            {"items": [["l", [10, 150], [10, 10]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0"},
            {"items": [["l", [200, 80], [260, 80]]], "type": "s", "color": [0, 0, 0], "width": 0.24, "layer": "0"},
        ],
    }
    r = m.predict(payload)[0]
    assert r.prob.shape == (5,)
    assert r.cls_prob is not None and r.cls_prob.shape == (5, 3)
    assert r.class_names == FIRE3_CLASSES
    assert np.allclose(r.cls_prob.sum(axis=1), 1.0, atol=1e-4)
    assert np.allclose(r.prob, 1.0 - r.cls_prob[:, 0], atol=1e-4)

    # Compact JSON microservice roundtrip
    d_raw = r.to_dict(compact=True, min_prob=0.0)
    r_json = Results.from_dict(d_raw)
    assert r_json.cls_prob is not None and r_json.class_names == FIRE3_CLASSES
    assert np.allclose(r_json.cls_prob, r.cls_prob, atol=1e-4)

    # NPZ binary roundtrip
    npz_p = r.save_npz(tmp_path / "fire3_raw.npz")
    r_npz = Results.from_npz(npz_p)
    assert r_npz.cls_prob is not None and r_npz.class_names == FIRE3_CLASSES
    assert np.allclose(r_npz.cls_prob, r.cls_prob, atol=1e-5)

    # Dynamic K=4 classes + cross-dataset transfer learning directly from the 3-class checkpoint!
    data4 = tmp_path / "data4"
    data4.mkdir()
    m4 = {**manifest, "class_names": ["other", "main", "branch", "hydrant"]}
    (data4 / "manifest.json").write_text(json.dumps(m4), encoding="utf-8")
    (data4 / "folds.json").write_text(json.dumps(propose(m4, k=2)))
    for f in data.glob("*.npz"):
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        y = d["label"].copy()
        br = np.flatnonzero(y == 2)
        if len(br) >= 2:
            y[br[: len(br) // 3]] = 3  # hydrant
        d["label"] = y
        np.savez(data4 / f.name, **d)

    assert main([
        "train", f"data={data4}", "fold=0", f"model={run_dir / 'best.pt'}", "modes=fire", "img_max_px=128",
        "steps=2", "batch=2", "window=128", "val_batch=2", "workers=0", "device=cpu", "tta=false",
        "epochs=1", f"project={tmp_path / 'runs'}", "name=dyn4_transfer",
    ]) == 0
    ck4 = torch.load(tmp_path / "runs" / "dyn4_transfer" / "fold0" / "best.pt", map_location="cpu", weights_only=False)
    assert ck4["config"]["arch"] == "openstroke46"
    assert ck4["config"]["stroke_classes"] == 4
    assert ck4["config"]["class_names"] == ["other", "main", "branch", "hydrant"]
    r4 = OpenStroke(tmp_path / "runs" / "dyn4_transfer" / "fold0" / "best.pt", device="cpu", threads=1, workers=1).predict(payload)[0]
    assert r4.cls_prob is not None and r4.cls_prob.shape == (5, 4)
    assert r4.class_names == ("other", "main", "branch", "hydrant")
    assert np.allclose(r4.cls_prob.sum(axis=1), 1.0, atol=1e-4)


def test_e153_fire_equal_width_and_wide_scale_augmentation():
    from openstroke.data import Augment, HAS_STROKE, LOG_LENGTH_FEATURES, LOG_WIDTH, N_FEATURES, STROKE, augment

    N = 12
    x0 = np.zeros((N, N_FEATURES), dtype=np.float32)
    x0[:, HAS_STROKE] = 1.0
    x0[:4, LOG_WIDTH] = np.log1p(0.96 * 10.0)
    x0[4:, LOG_WIDTH] = np.log1p(0.24 * 10.0)
    x0[:, STROKE] = 0.0
    for j in LOG_LENGTH_FEATURES:
        x0[:, j] = np.log1p(50.0)
    w0 = {
        "x": x0,
        "xy": np.stack([np.arange(N, dtype=np.float64) * 20.0, np.zeros(N, dtype=np.float64)], axis=1),
        "pts": np.ones((N, 16, 2), dtype=np.float32),
        "layer": np.zeros(N, dtype=np.int64),
        "label": np.array([1] * 4 + [2] * 8, dtype=np.uint8),
        "length": np.full(N, 50.0, dtype=np.float32),
        "ylayer": np.zeros(N, dtype=np.float32),
        "pos": np.arange(N, dtype=np.int64),
        "style": np.array([0] * 4 + [1] * 8, dtype=np.int64),
    }

    # 1. restyle_mode="fire" flattens wM == wB far more often (~35%) than "robust" (5%) while preserving groups otherwise
    aug_fire = Augment(fliplr=0.0, rot90=False, scale=0.0, color=0.0, gray=0.0, restyle=1.0, restyle_mode="fire", drop_tokens=0.0)
    n_flat = 0
    for sd in range(100):
        out = augment({k: v.copy() for k, v in w0.items()}, aug_fire, np.random.default_rng(sd))
        if np.allclose(out["x"][:, LOG_WIDTH], out["x"][0, LOG_WIDTH], atol=1e-5):
            n_flat += 1
    assert 20 <= n_flat <= 50, f"Expected ~35/100 flat equal-width trials for restyle_mode='fire', got {n_flat}"

    # 2. scale=1.0 scales coordinates and log-length/width features smoothly in [0.5x, 2.0x] (covers A1->A3 shrink)
    aug_scale = Augment(fliplr=0.0, rot90=False, scale=1.0, color=0.0, gray=0.0, restyle=0.0, drop_tokens=0.0)
    saw_small = saw_large = False
    for sd in range(40):
        out = augment({k: v.copy() for k, v in w0.items()}, aug_scale, np.random.default_rng(sd))
        s_eff = float(out["xy"][1, 0] / 20.0)
        assert 0.499 <= s_eff <= 2.001
        if s_eff < 0.65:
            saw_small = True
        if s_eff > 1.50:
            saw_large = True
        assert np.allclose(np.expm1(out["x"][:, LOG_LENGTH_FEATURES[0]]), 50.0 * s_eff, rtol=1e-4)
    assert saw_small and saw_large


def test_e154_style_share_leak_closure_and_empty_class_lovasz_fp_penalty():
    """F253: restyle_robust recomputes style_shape/len_share when styles collapse, and loss_multiclass penalizes empty-class FPs."""
    from openstroke.data import HAS_STROKE, LOG_WIDTH, N_FEATURES, STROKE, _I, restyle_robust
    from openstroke.models.openstroke2 import loss_multiclass, lovasz_softmax_flat

    # 1. Style-share leak closure: when widths & colours collapse to monochrome equal-width, style_shape_share MUST become 1.0
    N = 10
    x = np.zeros((N, N_FEATURES), dtype=np.float32)
    x[:, HAS_STROKE] = 1.0
    x[:, _I["type_s"]] = 1.0
    x[:, _I["log_len"]] = np.log1p(20.0)
    x[:2, STROKE] = [1.0, 0.0, 0.0]
    x[:2, LOG_WIDTH] = np.log1p(4.0)
    x[2:, STROKE] = [0.0, 0.0, 0.0]
    x[2:, LOG_WIDTH] = np.log1p(1.0)
    x[:2, _I["style_shape_share"]] = 0.005  # pre-restyle rare pipe shortcut!
    x[2:, _I["style_shape_share"]] = 0.995
    x[:2, _I["nb50_same_style"]] = 0.10
    x[:, _I["seqno_rank"]] = np.linspace(0.0, 0.9, N)

    saw_mono_flat = False
    for sd in range(120):
        xc = x.copy()
        restyle_robust(xc, np.random.default_rng(sd), flat_p=0.35)
        all_black = np.allclose(xc[:, STROKE], 0.0, atol=1e-5)
        all_same_w = np.allclose(xc[:, LOG_WIDTH], xc[0, LOG_WIDTH], atol=1e-5)
        if all_black and all_same_w:
            saw_mono_flat = True
            assert np.allclose(xc[:, _I["style_shape_share"]], 1.0, atol=1e-4)
            assert np.allclose(xc[:, _I["style_len_share"]], 1.0, atol=1e-4)
            assert float(xc[0, _I["nb50_same_style"]]) >= 0.85
    assert saw_mono_flat

    # 2. Empty-class FP penalty: negative page (all label=0) with high FP prob on main/branch MUST incur strong Lovasz loss
    probs_fp = torch.tensor([[0.1, 0.8, 0.1]] * 16, dtype=torch.float32)
    probs_ok = torch.tensor([[0.98, 0.01, 0.01]] * 16, dtype=torch.float32)
    y_neg = torch.zeros(16, dtype=torch.long)
    assert float(lovasz_softmax_flat(probs_fp, y_neg)) > float(lovasz_softmax_flat(probs_ok, y_neg)) + 0.25

    logits_fp = torch.tensor([[[0.0, 3.0, 0.0]] * 16], dtype=torch.float32, requires_grad=True)
    b = {"mask0": torch.ones((1, 16), dtype=torch.bool), "label": torch.zeros((1, 16)), "length": torch.full((1, 16), 20.0)}
    tot, parts = loss_multiclass(logits_fp, b)
    assert float(parts["lovasz"].detach()) > 0.30
    tot.backward()
    assert float(logits_fp.grad[..., 1].mean()) > 0.0  # gradient pushes main FP logit DOWN!

    # 3. inject_neg_boxes preserves uint8 3-class labels (new hard-negative CAD boxes get label=0)
    from types import SimpleNamespace
    from openstroke.data import WindowDataset, inject_neg_boxes

    w3 = {
        "x": x.copy(),
        "xy": np.stack([np.linspace(0, 200, N), np.zeros(N)], axis=1).astype(np.float32),
        "pts": np.zeros((N, 16, 2), dtype=np.float32),
        "length": np.full(N, 20.0, dtype=np.float32),
        "pos": np.arange(N, dtype=np.int64),
        "label": np.array([1, 1, 2, 2, 2, 0, 0, 0, 0, 0], dtype=np.uint8),
        "layer": np.zeros(N, dtype=np.int64),
        "style": np.arange(N, dtype=np.int64),
    }
    w3_out = inject_neg_boxes(w3, np.random.default_rng(7))
    assert w3_out["label"].dtype == np.uint8 and len(w3_out["label"]) > N
    assert np.array_equal(w3_out["label"][:N], w3["label"]) and np.all(w3_out["label"][N:] == 0)

    # 4. pdf_balance equalizes positive project weights without letting diverse negative projects starve positive share
    class _P:
        def __init__(self, stem, n, pos):
            self.stem, self.mode, self._n = stem, "fire", n
            self.label = np.ones(n, dtype=np.uint8) if pos else np.zeros(n, dtype=np.uint8)
        def __len__(self):
            return self._n

    ps = [_P("a1", 100, True), _P("a2", 100, True), _P("a3", 100, True), _P("b1", 100, True), _P("n1", 100, False)]
    pdf_map = {"a1": "projA", "a2": "projA", "a3": "projA", "b1": "projB", "n1": "projN"}
    ds = WindowDataset(ps, 10, 64, None, pdf_by_stem=pdf_map)
    assert abs(float(ds.prob[:3].sum()) - float(ds.prob[3])) < 1e-6   # projA (3p) == projB (1p) == 0.40 each!
    assert abs(float(ds.prob[4]) - 0.20) < 1e-6                      # neg stays at natural 1/5 = 0.20 (not 1/3!)
