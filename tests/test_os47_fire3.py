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
    from openstroke.models.openstroke2 import _graph_topo_fire_features

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
    assert tf.shape == (N, 12)
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

    # CLI alias openstroke_fire3 -> openstroke47
    ap = argparse.ArgumentParser()
    add_args(ap)
    raw = yolo_args(["data=datasets/fire_3cls_v1", "model=openstroke_fire3", "epochs=25"])
    args_c = ap.parse_args(raw)
    args_c._explicit = {t[2:].replace("-", "_") for t in raw if t.startswith("--")}
    args_c = finalize(args_c)
    assert run_version(args_c) == "openstroke47"
    assert args_c.stroke_classes == 3 and args_c.graph_topo_fire is True

    # Verify multiclass_metrics helper
    cp = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], dtype=np.float32)
    mc = multiclass_metrics([cp], [np.array([0, 1, 2])], [np.array([10.0, 20.0, 30.0])])
    assert mc["per_class"]["main"]["f2"] == 1.0 and mc["per_class"]["branch"]["f2"] == 1.0
    assert mc["pipe"]["f2"] == 1.0 and mc["fitness"] == 1.0

    # Synthetic 3-class FIRE dataset (0=other, 1=main, 2=branch)
    data, manifest = _synthetic(tmp_path)
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

    # End-to-end 3-class train + val
    assert main([
        "train", f"data={data}", "fold=0", "model=openstroke47", "modes=fire", "img_max_px=128",
        "dims=16,32", "enc_depths=1,1", "dec_depths=1", "patch=32", "img_pretrained=false",
        "steps=2", "batch=2", "window=128", "val_batch=2", "workers=0", "device=cpu", "tta=false",
        "epochs=1", f"project={tmp_path / 'runs'}", "name=fire3_smoke",
    ]) == 0

    run_dir = tmp_path / "runs" / "fire3_smoke" / "fold0"
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["arch"] == "openstroke47"
    assert ck["config"]["stroke_classes"] == 3
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
