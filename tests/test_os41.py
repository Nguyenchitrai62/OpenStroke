"""OpenStroke41 (H162): GroupClsHead, rotation-invariant cluster descriptors, 6-class object scoring, weight parity with OpenStroke31,
and tiny CPU end-to-end trainer smoke test with select_by=object."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch = pytest.importorskip("torch")
torch.set_num_threads(1)


def test_cluster_descriptors_are_rotation_and_flip_invariant():
    from openstroke.features import FEATURE_NAMES, N_FEATURES
    from openstroke.models.group_head import N_GEOM, cluster_descriptors

    rng = np.random.default_rng(7)
    N, C = 12, 16
    h = torch.tensor(rng.normal(size=(N, C)), dtype=torch.float32)
    xy = torch.tensor(rng.uniform(-100, 100, size=(N, 2)), dtype=torch.float32)
    length = torch.tensor(rng.uniform(2, 40, size=(N,)), dtype=torch.float32)
    x_raw = torch.zeros((N, N_FEATURES), dtype=torch.float32)
    ang = rng.uniform(0, np.pi, size=(N,))
    x_raw[:, FEATURE_NAMES.index("dir_cos2")] = torch.tensor(np.cos(2 * ang), dtype=torch.float32)
    x_raw[:, FEATURE_NAMES.index("dir_sin2")] = torch.tensor(np.sin(2 * ang), dtype=torch.float32)
    x_raw[:, FEATURE_NAMES.index("frac_c")] = torch.tensor(rng.uniform(0, 1, size=(N,)), dtype=torch.float32)
    x_raw[:, FEATURE_NAMES.index("closed_ends")] = torch.tensor(rng.integers(0, 2, size=(N,)), dtype=torch.float32)
    x_raw[:, FEATURE_NAMES.index("log_bbox_w")] = torch.tensor(rng.uniform(0.5, 3.0, size=(N,)), dtype=torch.float32)
    x_raw[:, FEATURE_NAMES.index("log_bbox_h")] = torch.tensor(rng.uniform(0.5, 3.0, size=(N,)), dtype=torch.float32)
    prob = torch.tensor(rng.uniform(0.5, 0.99, size=(N,)), dtype=torch.float32)
    cid = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)

    f0 = cluster_descriptors(h, xy, length, x_raw, prob, cid, 3)
    assert f0.shape == (3, 2 * C + N_GEOM)
    assert torch.isfinite(f0).all()

    # 90-degree rotation: (x, y) -> (-y, x), dir_cos2/sin2 negated, bbox w <-> h
    xy_rot = torch.stack([-xy[:, 1], xy[:, 0]], dim=1)
    xr_rot = x_raw.clone()
    xr_rot[:, FEATURE_NAMES.index("dir_cos2")] *= -1.0
    xr_rot[:, FEATURE_NAMES.index("dir_sin2")] *= -1.0
    iw, ih = FEATURE_NAMES.index("log_bbox_w"), FEATURE_NAMES.index("log_bbox_h")
    xr_rot[:, iw], xr_rot[:, ih] = x_raw[:, ih].clone(), x_raw[:, iw].clone()
    f_rot = cluster_descriptors(h, xy_rot, length, xr_rot, prob, cid, 3)
    assert torch.allclose(f0, f_rot, atol=1e-4)

    # Horizontal flip: x -> -x, dir_sin2 negated
    xy_fl = torch.stack([-xy[:, 0], xy[:, 1]], dim=1)
    xr_fl = x_raw.clone()
    xr_fl[:, FEATURE_NAMES.index("dir_sin2")] *= -1.0
    f_fl = cluster_descriptors(h, xy_fl, length, xr_fl, prob, cid, 3)
    assert torch.allclose(f0, f_fl, atol=1e-4)


def test_openstroke41_loads_openstroke31_weights_with_exact_stroke_and_edge_parity():
    from openstroke.models import ARCHS, build, load_weights
    from openstroke.train import PRESETS, arch_name, describe_arch

    assert {k: v for k, v in ARCHS["openstroke41"].items() if k != "group_head"} == ARCHS["openstroke31"]
    assert ARCHS["openstroke41"]["group_head"] is True
    assert PRESETS["openstroke41"][1]["select_by"] == "object"
    assert describe_arch({"arch": "openstroke31", **ARCHS["openstroke31"], "group_head": True}) == "openstroke41"
    assert arch_name({"arch": "openstroke31", **ARCHS["openstroke31"], "group_head": True}) == "openstroke41"

    tiny = dict(dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32, img_pretrained=False)
    m31 = build("openstroke31", 47, **tiny).eval()
    m41 = build("openstroke41", 47, **tiny).eval()
    load_weights(m41, m31.state_dict(), new_heads=True)
    assert m41.group_head is not None

    # Exact stroke and edge logit parity before any OpenStroke41 gradient step
    from openstroke.data import collate, window
    from openstroke.features import N_FEATURES
    from test_model import _synthetic
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        data, _ = _synthetic(Path(td))
        from openstroke.data import load_page
        stem = sorted(p.stem for p in data.glob("*.npz"))[0]
        pg = load_page(data, stem, "hvac", Path(td) / "cache", graph=True, no_layers=True)
        b = collate([window(pg, 0, 128)])
        with torch.no_grad():
            y31 = m31(b)
            e31 = m31.last_edge[0].clone() if m31.last_edge is not None else None
            y41 = m41(b)
            e41 = m41.last_edge[0].clone() if m41.last_edge is not None else None
        assert torch.allclose(y31, y41, atol=1e-5)
        if e31 is not None:
            assert torch.allclose(e31, e41, atol=1e-5)
        assert m41.last_h is not None and m41.last_h.shape == (1, y41.shape[1], 16)


def test_group_counts_6class_handles_fp_rejection_tee_and_any_reducer():
    from openstroke.object_eval import group_counts_6class

    # 4 GT objects:
    #   inst 0: Duct (ocls=1), strokes 0,1
    #   inst 1: Reducer vuông tròn / ANY_REDUCER (ocls=6), strokes 2,3
    #   inst 2: Chân rẽ vuông tròn (ocls=8 -> merged to Chân rẽ=6), strokes 4,5
    #   inst 3: Tee (ocls=9 -> excluded from 6-class scope!), strokes 6,7
    # Background strokes 8,9 (FP cluster predicted as none -> rejected!)
    page = SimpleNamespace(
        label=np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], np.int64),
        length=np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], np.float64),
        oinst=np.array([0, 0, 1, 1, 2, 2, 3, 3, -1, -1], np.int64),
        ocls=np.array([1, 1, 6, 6, 8, 8, 9, 9, 0, 0], np.int64),
    )
    prob = np.array([0.9] * 10, np.float64)
    inst = np.array([10, 10, 11, 11, 12, 12, 13, 13, 14, 14], np.int64)
    cids = np.array([10, 11, 12, 13, 14], np.int64)
    # Predictions: 10->Duct(1), 11->Eccentric Reducer(5, must match ANY_REDUCER!), 12->Chân rẽ(6), 13->Tee(7, dropped), 14->none(0, FP rejected)
    cprobs = np.zeros((5, 8), np.float32)
    cprobs[0, 1] = 1.0
    cprobs[1, 5] = 1.0
    cprobs[2, 6] = 1.0
    cprobs[3, 7] = 1.0
    cprobs[4, 0] = 1.0

    by_cls = {}
    cnt = group_counts_6class(page, prob, 0.5, inst, cids, cprobs, by_class=by_cls)
    assert cnt.tolist() == [3.0, 3.0, 3.0]   # 3/3 perfect match, Tee & FP none cleanly ignored!
    assert by_cls["Duct"].tolist() == [1.0, 1.0, 1.0]
    assert by_cls["Eccentric Reducer"].tolist() == [1.0, 1.0, 1.0]
    assert by_cls["Chân rẽ"].tolist() == [1.0, 1.0, 1.0]


def test_openstroke41_trainer_smoke_select_by_object_and_saves_group_outputs(tmp_path):
    """Tiny CPU end-to-end training of OpenStroke41 with select_by=object, verifying grp_loss, 6cls eval, val_scores.npz, and fine-tune inheritance."""
    from test_model import _synthetic

    from openstroke.cli import main
    from openstroke.folds import propose

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
    assert main([
        "train", *common, "model=openstroke41", "dims=16,32", "enc_depths=1,1", "dec_depths=1",
        "patch=32", "img_pretrained=false", "epochs=1", "name=v41",
    ]) == 0
    run = tmp_path / "runs" / "v41" / "fold0"
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["group_head"] is True and ck["config"]["arch"] == "openstroke41"

    z = np.load(run / "val_scores.npz")
    val_stems = json.loads((run / "result.json").read_text(encoding="utf-8"))["val"]
    for st in val_stems:
        assert f"{st}__scls" in z.files and z[f"{st}__scls"].shape[1] == 8
        assert f"{st}__ginst" in z.files
        assert f"{st}__gcids" in z.files
        assert f"{st}__gcprobs" in z.files and z[f"{st}__gcprobs"].shape[1] == 8
    res = json.loads((run / "result.json").read_text(encoding="utf-8"))
    assert "6cls" in res["objects"] and "mx" in res["objects"]

    # Also test starting from an openstroke31 checkpoint and adding group_head=true + select_by=object
    assert main([
        "train", *common, "model=openstroke31", "dims=16,32", "enc_depths=1,1", "dec_depths=1",
        "patch=32", "img_pretrained=false", "epochs=1", "name=v31base",
    ]) == 0
    base_pt = tmp_path / "runs" / "v31base" / "fold0" / "best.pt"
    assert main([
        "train", *common, f"model={base_pt}", "group_head=true", "select_by=object", "epochs=1", "name=v41_ft",
    ]) == 0
    ck_ft = torch.load(run_ft := (tmp_path / "runs" / "v41_ft" / "fold0" / "best.pt"), map_location="cpu", weights_only=False)
    assert ck_ft["config"]["group_head"] is True and ck_ft["config"]["arch"] == "openstroke41"

    # Also verify Predictor.predict_payload returns group_objects cleanly on an OpenStroke41 checkpoint
    from openstroke.infer import Predictor
    pred = Predictor(run_ft, device="cpu", threads=1, workers=1)
    payload = {
        "shapes": [
            {"items": [["l", [0, 0], [50, 0]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0", "rect": [0, 0, 50, 0]},
            {"items": [["l", [50, 0], [50, 20]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0", "rect": [50, 0, 50, 20]},
            {"items": [["l", [50, 20], [0, 20]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0", "rect": [0, 20, 50, 20]},
            {"items": [["l", [0, 20], [0, 0]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0", "rect": [0, 0, 0, 20]},
        ],
        "document_metadata": {"page_width": 200.0, "page_height": 200.0},
    }
    out = pred.predict_payload(payload, mode="hvac", tta=1)
    assert "group_objects" in out
    go = out["group_objects"]
    assert go["instance_id"].shape == (4,) and go["stroke_cls"].shape == (4, 8)
    assert go["cluster_probs"].ndim == 2 and go["cluster_probs"].shape[1] == 8


def test_restyle_robust_breaks_hairline_and_all_black_shortcuts():
    """F228: restyle_robust must produce mixed 0.00pt hairline + thick widths, flat widths, and pure-black monochrome
    while keeping white wipeouts (>=0.92 lightness) white and non-stroked shapes at width 0."""
    from openstroke.data import HAS_FILL, HAS_STROKE, LOG_WIDTH, STROKE, FILL, restyle_robust
    from openstroke.features import N_FEATURES
    from openstroke.train import PRESETS

    assert PRESETS["openstroke41"][1]["restyle_mode"] == "robust"
    assert PRESETS["openstroke41"][1]["restyle"] == 0.7

    rng = np.random.default_rng(42)
    base = np.zeros((10, N_FEATURES), dtype=np.float32)
    base[:8, HAS_STROKE] = 1.0
    base[8:, HAS_FILL] = 1.0
    # 3 distinct stroked style groups + 1 white wipeout
    base[:3, STROKE] = [0.0, 0.0, 0.8]
    base[:3, LOG_WIDTH] = np.log1p(0.84 * 10.0)
    base[3:6, STROKE] = [0.0, 0.6, 0.0]
    base[3:6, LOG_WIDTH] = np.log1p(0.72 * 10.0)
    base[6:7, STROKE] = [0.5, 0.5, 0.5]
    base[6:7, LOG_WIDTH] = np.log1p(0.24 * 10.0)
    base[7:8, STROKE] = [0.98, 0.98, 0.98]  # white wipeout
    base[7:8, LOG_WIDTH] = np.log1p(0.12 * 10.0)
    base[8:, FILL] = [0.4, 0.4, 0.4]

    saw_mixed_hairline = saw_flat_width = saw_all_black = False
    for _ in range(120):
        x = base.copy()
        restyle_robust(x, rng)
        assert (x[8:, LOG_WIDTH] == 0.0).all()  # non-stroked stays 0
        w_st = x[:7, LOG_WIDTH]
        if (w_st == 0.0).any() and (w_st > 0.0).any():
            saw_mixed_hairline = True
        if np.allclose(x[:8, LOG_WIDTH], x[0, LOG_WIDTH]):
            saw_flat_width = True
        if np.allclose(x[:7, STROKE], 0.0):
            saw_all_black = True
            # White wipeout MUST remain white even in all-black branch
            assert float(x[7, STROKE].mean()) >= 0.92
    assert saw_mixed_hairline and saw_flat_width and saw_all_black
    # Empty window safety
    restyle_robust(np.zeros((0, N_FEATURES), dtype=np.float32), rng)


def test_group_class_weights_counts_objects_not_stroke_length():
    """F230: 1 giant 1000-pt Duct + 1 tiny 10-pt Concentric Reducer must get equal object-count weight (not 10x skewed)."""
    from openstroke.models.group_head import group_class_weights

    page = SimpleNamespace(
        label=np.array([1, 1, 1, 1], np.uint8),
        oinst=np.array([0, 0, 1, 1], np.int32),
        ocls=np.array([1, 1, 4, 4], np.int8),   # 1 Duct (2000pt), 1 Concentric Reducer (20pt)
        length=np.array([1000.0, 1000.0, 10.0, 10.0], np.float32),
    )
    w = group_class_weights([page])
    assert abs(w[1] - w[4]) < 1e-6


def test_hybrid_group_objects_pure_and_reverse_hybrid_m1():
    """F230: verify pure end-to-end GroupClsHead mode AND Reverse Hybrid M1 (Reducer override + over-merged cluster split)."""
    from openstroke.geometry import PageSegments
    from openstroke.hvac_faces import hybrid_group_objects

    # 3 rectangles of 4 segments each (12 shapes, 1 segment per shape):
    #   shapes 0..3: box [0,0]-[40,20]  (GroupClsHead calls Chân rẽ, hvac_faces calls Concentric Reducer -> M1 overrides!)
    #   shapes 4..7: box [50,0]-[90,20] + shapes 8..11: box [100,0]-[140,20] (over-merged into ONE cluster 1 -> M1 splits into 2!)
    def rect_segs(x0, y0, x1, y1):
        return [[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]]

    seg = np.array(rect_segs(0, 0, 40, 20) + rect_segs(50, 0, 90, 20) + rect_segs(100, 0, 140, 20), dtype=np.float64)
    segs = PageSegments(seg=seg, owner=np.arange(12), shape_offsets=np.arange(13))
    prob = np.full(12, 0.9, dtype=np.float32)
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])

    # Cluster 0 = shapes 0..3 (predicted Chân rẽ=6), Cluster 1 = shapes 4..11 (over-merged Duct=1)
    ginst = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    cprobs = np.zeros((2, 8), dtype=np.float32)
    cprobs[0, 6] = 0.85   # Chân rẽ
    cprobs[1, 1] = 0.90   # Duct
    go = {"instance_id": ginst, "cluster_ids": np.array([0, 1]), "cluster_probs": cprobs}

    # 1) Pure end-to-end mode (face_objects=None): 2 objects (Chân rẽ, Duct)
    objs_pure, cls_p, inst_p = hybrid_group_objects(segs, prob, length, go, face_objects=None)
    assert [o["shape_name"] for o in objs_pure] == ["Chân rẽ", "Duct"]
    assert set(inst_p.tolist()) == {0, 1}

    # 2) Reverse Hybrid M1 with 3 closed faces: Reducer on [0..40], Duct on [50..90], Elbow on [100..140]
    faces = [
        {"shape_name": "Concentric Reducer", "vertices": [[0, 0], [40, 0], [40, 20], [0, 20]], "score": 0.9},
        {"shape_name": "Duct", "vertices": [[50, 0], [90, 0], [90, 20], [50, 20]], "score": 0.9},
        {"shape_name": "Elbow", "vertices": [[100, 0], [140, 0], [140, 20], [100, 20]], "score": 0.9},
    ]
    objs_m1, cls_m, inst_m = hybrid_group_objects(segs, prob, length, go, face_objects=faces)
    assert sorted(o["shape_name"] for o in objs_m1) == ["Concentric Reducer", "Duct", "Elbow"]
    assert len(objs_m1) == 3 and set(inst_m.tolist()) == {0, 1, 2}
