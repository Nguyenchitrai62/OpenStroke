"""OpenStroke45 (H172 / H173 / F243): style-consistent g3 CC topology + relative line-weight (graph_topo_v2, 7-dim)
+ neg_boxes v4 (no closed rectangular side-collar boxes; 2x2..2x5 thinner-weight workstation rows)
+ restyle_robust 75/20/5 + hvac_faces v6c same-width CAD guardrail."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch = pytest.importorskip("torch")
torch.set_num_threads(1)


def test_openstroke45_registry_step0_parity_and_backward_compat(tmp_path):
    from test_model import _synthetic

    from openstroke.data import collate, load_page, window
    from openstroke.models import ARCHS, build, load_weights
    from openstroke.train import PRESETS, arch_name, describe_arch

    assert ARCHS["openstroke45"]["img_cad"] is True
    assert ARCHS["openstroke45"]["img_max_px"] == 2048
    assert ARCHS["openstroke45"]["graph_topo_v2"] is True
    assert ARCHS["openstroke45"].get("graph_topo", False) is False
    assert ARCHS["openstroke45"]["rel_attn"] == 4
    assert ARCHS["openstroke45"]["group_head"] is True
    assert ARCHS["openstroke45"].get("stroke_context", False) is False
    assert ARCHS["openstroke45"].get("facing_edges", False) is False

    preset_cfg = PRESETS["openstroke45"][1]
    assert preset_cfg["graph_topo_v2"] is True
    assert preset_cfg["neg_boxes"] == 0.45
    assert preset_cfg["lr0"] == 0.0005

    cfg45 = {"arch": "openstroke41", **ARCHS["openstroke41"], "img_cad": True, "graph_topo_v2": True, "rel_attn": 4, "img_max_px": 2048}
    assert describe_arch(cfg45) == "openstroke45"
    assert arch_name(cfg45) == "openstroke45"

    tiny = dict(dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32, img_pretrained=False)
    m41 = build("openstroke41", 47, **tiny).eval()
    m45 = build("openstroke45", 47, **tiny).eval()
    assert m45.topo_mlp is not None and m45.topo_mlp[0].in_features == 7
    assert m45.img.cad_stem is not None
    assert len(m45.gat) == 4

    load_weights(m45, m41.state_dict(), new_heads=True)

    data, _ = _synthetic(tmp_path)
    stem = sorted(p.stem for p in data.glob("*.npz"))[0]
    pg = load_page(data, stem, "hvac", tmp_path / "cache", graph=True, no_layers=True)
    b = collate([window(pg, 0, 128)])
    m45.img.max_px = m41.img.max_px
    with torch.no_grad():
        y41, h41 = m41(b), m41.last_h.clone()
        e41 = m41.last_edge[0].clone() if m41.last_edge is not None else None
        y45, h45 = m45(b), m45.last_h.clone()
        e45 = m45.last_edge[0].clone() if m45.last_edge is not None else None
    assert torch.allclose(y41, y45, atol=1e-5), "OpenStroke45 must match OpenStroke41 stroke logits at step 0!"
    assert torch.allclose(h41, h45, atol=1e-5), "OpenStroke45 must match OpenStroke41 hidden features at step 0!"
    if e41 is not None:
        assert torch.allclose(e41, e45, atol=1e-5), "OpenStroke45 must match OpenStroke41 edge logits at step 0!"

    # Gradients flow cleanly through 7-dim topo_mlp, CADStem, and gat.2..3
    m45.train()
    torch.nn.init.normal_(m45.img.proj_fpn.weight, std=0.05)
    out = m45(b)
    loss = out.square().mean() + (m45.last_edge[0].square().mean() if m45.last_edge is not None else 0.0)
    loss.backward()
    for name, p in [
        ("topo_mlp[-1]", m45.topo_mlp[-1].weight),
        ("cad_stem.net[-1]", m45.img.cad_stem.net[-1].weight),
        ("gat[2].ffn[-1]", m45.gat[2].ffn[-1].weight),
        ("gat[3].ffn[-1]", m45.gat[3].ffn[-1].weight),
    ]:
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Invalid grad on {name}"
    assert m45.topo_mlp[-1].weight.grad.abs().sum() > 0


def test_openstroke45_loads_real_e145_and_backward_loads_e149_if_present():
    from openstroke.models import build, load_weights

    ck41_p = Path(r"F:\Downloads\e145_os41_robust\fold0\best.pt")
    if ck41_p.exists():
        ck = torch.load(ck41_p, map_location="cpu", weights_only=False)
        cfg = dict(ck["config"])
        m45 = build(
            "openstroke45", cfg["n_features"],
            dims=cfg["dims"], enc_depths=cfg["enc_depths"], dec_depths=cfg["dec_depths"],
            patch=cfg["patch"], pretrain=cfg.get("pretrain", False), img_pretrained=False, img_max_px=2048,
        )
        load_weights(m45, ck["model"], new_heads=True)
        assert m45.topo_mlp[0].in_features == 7 and len(m45.gat) == 4

    ck44_p = Path(r"F:\Downloads\e149_os44_cad_topo_e13v4\fold0\best.pt")
    if ck44_p.exists():
        ck = torch.load(ck44_p, map_location="cpu", weights_only=False)
        cfg = dict(ck["config"])
        m44 = build(
            "openstroke44", cfg["n_features"],
            dims=cfg["dims"], enc_depths=cfg["enc_depths"], dec_depths=cfg["dec_depths"],
            patch=cfg["patch"], pretrain=cfg.get("pretrain", False), img_pretrained=False, img_max_px=cfg.get("img_max_px", 2048),
        )
        m44.load_state_dict(ck["model"])
        assert m44.topo_mlp[0].in_features == 5


def test_graph_topo_v2_isolates_thin_desk_touching_thick_duct_and_protects_hairlines():
    from openstroke.data import HAS_STROKE, LOG_WIDTH, STROKE
    from openstroke.features import N_FEATURES
    from openstroke.models.openstroke2 import _graph_topo_features, _graph_topo_v2_features

    # 3 strokes of a 0.48pt overhead duct run (span = 600pt): indices 0, 1, 2
    # 2 strokes of a 0.24pt office desk (span = 20pt) touching duct node 1 via g3 edge (1, 3): indices 3, 4
    # 2 strokes of a 0.00pt hairline duct run (span = 400pt): indices 5, 6
    N = 7
    x = np.zeros((1, N, N_FEATURES), dtype=np.float32)
    x[0, :, HAS_STROKE] = 1.0
    x[0, :, STROKE] = 0.0  # all black (like VNG-2 [12] & 0875!)
    x[0, 0:3, LOG_WIDTH] = np.log1p(0.48 * 10.0)
    x[0, 3:5, LOG_WIDTH] = np.log1p(0.24 * 10.0)
    x[0, 5:7, LOG_WIDTH] = 0.0  # 0.00pt hairline!

    xy = np.array([[[0, 0], [300, 0], [600, 0], [300, 0], [320, 0], [0, 200], [400, 200]]], dtype=np.float32)
    length = np.array([[300, 300, 300, 20, 20, 400, 400]], dtype=np.float32)
    edges = np.array([[[0, 1], [1, 2], [1, 3], [3, 4], [5, 6]]], dtype=np.int64)
    b = {
        "x": torch.from_numpy(x),
        "xy0": torch.from_numpy(xy),
        "length": torch.from_numpy(length),
        "mask0": torch.ones((1, N), dtype=torch.bool),
        "edges": torch.from_numpy(edges),
        "emask": torch.ones((1, 5), dtype=torch.bool),
    }

    t1 = _graph_topo_features(b)[0].numpy()
    t2 = _graph_topo_v2_features(b)[0].numpy()

    # Old v1 leaked the 600pt duct span into the 0.24pt desk!
    assert np.isclose(t1[3, 0], t1[0, 0], atol=1e-5)

    # New v2 breaks the cross-width edge (1, 3) -> desk has tiny span (~20pt) while duct keeps 600pt!
    duct_span = np.expm1(t2[0, 0] * 6.0)
    desk_span = np.expm1(t2[3, 0] * 6.0)
    hair_span = np.expm1(t2[5, 0] * 6.0)
    assert np.isclose(duct_span, 600.0, atol=1.0)
    assert np.isclose(desk_span, 20.0, atol=1.0)
    assert np.isclose(hair_span, 400.0, atol=1.0)
    assert t2[3, 4] == 1.0 and t2[0, 4] == 0.0   # desk flagged as sw_is_small_box, duct not
    assert t2[3, 5] < -0.15 and t2[3, 6] == 1.0  # desk flagged rel_width_p90 < 0 and is_thinner_half = 1
    assert t2[0, 6] == 0.0                       # thick duct is_thinner_half = 0
    assert t2[5, 5] == 0.0 and t2[5, 6] == 0.0   # 0.00pt hairline stays strictly neutral!


def test_neg_boxes_v4_has_thin_desk_rows_and_no_closed_collar_boxes():
    from test_os44 import _synth_win_v44

    from openstroke.data import Augment, HAS_FILL, LOG_WIDTH, augment
    from openstroke.features import FEATURE_NAMES

    w = _synth_win_v44(200)
    n0 = len(w["x"])
    aug = Augment(fliplr=0, rot90=False, scale=0, color=0, gray=0, restyle=0, drop_tokens=0, neg_boxes=1.0)

    saw_thinner = False
    saw_same_w = False
    saw_zigzag_with_filled_tri = False
    ref_lw = float(np.log1p(0.48 * 10.0))
    for sd in range(15):
        out = augment({k: v.copy() if isinstance(v, np.ndarray) else v for k, v in w.items()}, aug, np.random.default_rng(sd))
        xs = out["x"][n0:]
        lw = xs[:, LOG_WIDTH]
        if (lw < ref_lw - 0.25).any():
            saw_thinner = True
        if np.isclose(lw, ref_lw, atol=1e-4).any():
            saw_same_w = True
        nseg = np.expm1(xs[:, FEATURE_NAMES.index("log_nseg")])
        if ((nseg >= 4.8) & (nseg <= 5.2)).any() and (xs[:, HAS_FILL] > 0.5).any():
            saw_zigzag_with_filled_tri = True
    assert saw_thinner and saw_same_w, "neg_boxes v4 must generate both ~0.5x thinner and equal-width negatives"
    assert saw_zigzag_with_filled_tri


def test_hvac_faces_v6c_same_width_guardrail_drops_desk_touching_duct_keeps_thin_duct():
    from openstroke.geometry import page_segments
    from openstroke.hvac_faces import _apply_cad_noise_guardrails

    def _rect_shapes(x0, y0, x1, y1, w):
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return [
            {"items": [["l", list(pts[k]), list(pts[(k + 1) % 4])]], "type": "s", "color": [0, 0, 0], "width": w, "layer": "0"}
            for k in range(4)
        ]

    def _rect_poly(x0, y0, x1, y1):
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    # 1) Thick 0.48pt overhead duct run (length ~2040pt -> dom_sh >= 75%, span 1000pt)
    shapes = _rect_shapes(0, 0, 1000, 20, 0.48)
    # 2) Stubby 0.24pt 2x2 desk grid (4 touching 20x14 boxes, tcc_cnt=4 > 2) directly touching the 0.48pt duct at y=20!
    shapes += _rect_shapes(100, 20, 120, 34, 0.24)
    shapes += _rect_shapes(120, 20, 140, 34, 0.24)
    shapes += _rect_shapes(100, 34, 120, 48, 0.24)
    shapes += _rect_shapes(120, 34, 140, 48, 0.24)
    # 3) Lone 0.24pt thin takeoff collar (tcc_cnt=1 <= 2) touching the 0.48pt trunk -> protected!
    shapes += _rect_shapes(220, 20, 232, 28, 0.24)
    # 4) Real 0.24pt thin elongated branch duct (100x12, asp=8.3 >= 2.8) touching the 0.48pt trunk -> protected!
    shapes += _rect_shapes(300, -100, 312, 0, 0.24)

    segs = page_segments(shapes)
    n = len(shapes)
    prob = np.full(n, 0.98, dtype=np.float32)
    length = np.array([
        max(abs(s["items"][0][1][0] - s["items"][0][2][0]), abs(s["items"][0][1][1] - s["items"][0][2][1]))
        for s in shapes
    ], dtype=np.float64)
    attrs = {"width": np.array([s["width"] for s in shapes], dtype=np.float64)}
    objs = [
        {"shape_name": "Duct", "vertices": _rect_poly(0, 0, 1000, 20), "score": 0.99},
        {"shape_name": "Duct", "vertices": _rect_poly(100, 20, 120, 34), "score": 0.95},
        {"shape_name": "Duct", "vertices": _rect_poly(120, 20, 140, 34), "score": 0.95},
        {"shape_name": "Duct", "vertices": _rect_poly(100, 34, 120, 48), "score": 0.95},
        {"shape_name": "Duct", "vertices": _rect_poly(120, 34, 140, 48), "score": 0.95},
        {"shape_name": "Chân rẽ", "vertices": _rect_poly(220, 20, 232, 28), "score": 0.96},
        {"shape_name": "Duct", "vertices": _rect_poly(300, -100, 312, 0), "score": 0.96},
    ]

    kept = _apply_cad_noise_guardrails(objs, segs, prob, length, attrs, None, thr=0.65)
    kept_x0s = sorted(round(min(v[0] for v in o["vertices"]), 1) for o in kept)
    # Overhead 0.48pt trunk (0), lone collar (220), and elongated thin branch duct (300) MUST survive;
    # the 2x2 stubby 0.24pt desk grid (100, 120) touching the trunk MUST be dropped!
    assert kept_x0s == [0.0, 220.0, 300.0], f"Expected trunk+collar+branch only, got x0s={kept_x0s}"


def test_openstroke45_trainer_smoke_and_cli_dry_parse(tmp_path):
    import argparse
    from test_model import _synthetic

    from openstroke.cli import main, yolo_args
    from openstroke.folds import propose
    from openstroke.infer import Predictor
    from openstroke.train import add_args, finalize, run_version

    # Verify E150c (preset) CLI dry-parse
    ap = argparse.ArgumentParser()
    add_args(ap)
    raw = yolo_args([
        "data=datasets/e13_v4", "folds=datasets/e13_v4/folds_gt20.json",
        "model=openstroke45", "modes=hvac", "lr0=0.002", "warmup_epochs=2.0", "epochs=15",
    ])
    args_c = ap.parse_args(raw)
    args_c._explicit = {t[2:].replace("-", "_") for t in raw if t.startswith("--")}
    args_c = finalize(args_c)
    assert run_version(args_c) == "openstroke45"
    assert args_c.graph_topo_v2 is True and args_c.graph_topo is False

    # Smoke train openstroke41 -> fine-tune to openstroke45 -> infer
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
        "patch=32", "img_pretrained=false", "epochs=1", "name=v41_src",
    ]) == 0
    src41 = tmp_path / "runs" / "v41_src" / "fold0" / "best.pt"

    assert main([
        "train", *common, f"model={src41}",
        "img_cad=true", "img_max_px=2048", "graph_topo_v2=true", "rel_attn=4",
        "neg_boxes=0.45", "lr0=0.0005", "warmup_epochs=1.0",
        "epochs=1", "name=v45_ft",
    ]) == 0
    run45 = tmp_path / "runs" / "v45_ft" / "fold0"
    ck45 = torch.load(run45 / "best.pt", map_location="cpu", weights_only=False)
    assert ck45["config"]["arch"] == "openstroke45"
    assert ck45["config"]["img_cad"] is True
    assert ck45["config"]["graph_topo_v2"] is True
    assert ck45["config"].get("graph_topo", False) is False
    assert ck45["config"]["rel_attn"] == 4

    pred = Predictor(run45 / "best.pt", device="cpu", threads=1, workers=1)
    payload = {
        "shapes": [
            {"items": [["l", [0, 0], [60, 0]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0", "rect": [0, 0, 60, 0]},
            {"items": [["l", [0, 15], [60, 15]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0", "rect": [0, 15, 60, 15]},
            {"items": [["l", [60, 0], [60, 15]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0", "rect": [60, 0, 60, 15]},
            {"items": [["l", [0, 15], [0, 0]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0", "rect": [0, 0, 0, 15]},
        ],
        "document_metadata": {"page_width": 200.0, "page_height": 200.0},
    }
    out = pred.predict_payload(payload, mode="hvac", tta=1)
    assert out["prob"].shape == (4,) and np.isfinite(out["prob"]).all()
    assert "group_objects" in out and out["group_objects"]["cluster_probs"].shape[1] == 8
