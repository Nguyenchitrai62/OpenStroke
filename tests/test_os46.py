"""OpenStroke46 Unified (H174 + H177): ONE general-purpose CAD vector+image model (0 layers, 0 system assumptions):
  - 18D universal CAD-graph topology (_graph_topo_uni_features: style CC + length-weighted width + louver/arrowhead + filled strip + 2-core loop + tree peeling + terminal symbol)
  - 6D edge_style (GAT & EdgeHead) + 3-channel dilated CADStemV2 (img_cad_v2) + neg_boxes v5
  - Dynamic output head (auto nc + class_names from dataset) + cross-system transfer learning (HVAC binary -> FIRE 3-class -> N-class)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch = pytest.importorskip("torch")
torch.set_num_threads(1)


def test_openstroke46_registry_step0_parity_and_backward_compat(tmp_path):
    from test_model import _synthetic

    from openstroke.data import collate, load_page, window
    from openstroke.models import ARCHS, build, load_weights
    from openstroke.train import PRESETS, arch_name, describe_arch

    assert ARCHS["openstroke46"]["img_cad_v2"] is True
    assert ARCHS["openstroke46"].get("img_cad", False) is False
    assert ARCHS["openstroke46"]["img_max_px"] == 2048
    assert ARCHS["openstroke46"]["graph_topo_uni"] is True
    assert ARCHS["openstroke46"]["edge_style"] is True
    assert ARCHS["openstroke46"]["rel_attn"] == 4

    preset_cfg = PRESETS["openstroke46"][1]
    assert preset_cfg["graph_topo_uni"] is True
    assert preset_cfg["img_cad_v2"] is True
    assert preset_cfg["edge_style"] is True
    assert preset_cfg["neg_boxes"] == 0.45
    assert preset_cfg["stroke_classes"] == 0  # dynamic auto-detect from dataset

    cfg46 = {
        "arch": "openstroke41", **ARCHS["openstroke41"],
        "img_cad_v2": True, "graph_topo_uni": True, "edge_style": True, "rel_attn": 4, "img_max_px": 2048,
    }
    assert arch_name(cfg46) == "openstroke46"
    assert describe_arch({"arch": "openstroke2", **ARCHS["openstroke46"]}) == "openstroke46"

    tiny = dict(dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32, img_pretrained=False)
    m41 = build("openstroke41", 47, **tiny).eval()
    m45 = build("openstroke45", 47, **tiny).eval()
    m46 = build("openstroke46", 47, edge_head=True, edge_pairs=True, group_head=True, **tiny).eval()
    assert m46.topo_mlp is not None and m46.topo_mlp[0].in_features == 18
    assert m46.img.img_cad_v2 is True and m46.img.cad_stem is not None
    assert len(m46.gat) == 4 and m46.gat[0].geo_b.in_features == 6 and m46.edge_mlp.geo.in_features == 6

    # Step-0 parity from OpenStroke41 (expands 4D -> 6D edge geometry with zero cols 4..5!)
    load_weights(m46, m41.state_dict(), new_heads=True)
    assert torch.all(m46.gat[0].geo_b.weight[:, 4:] == 0)
    assert torch.all(m46.edge_mlp.geo.weight[:, 4:] == 0)

    data, _ = _synthetic(tmp_path)
    stem = sorted(p.stem for p in data.glob("*.npz"))[0]
    pg = load_page(data, stem, "hvac", tmp_path / "cache", graph=True, no_layers=True)
    b = collate([window(pg, 0, 128)])
    m46.img.max_px = m41.img.max_px
    with torch.no_grad():
        y41, h41 = m41(b), m41.last_h.clone()
        e41 = m41.last_edge[0].clone() if m41.last_edge is not None else None
        y46, h46 = m46(b), m46.last_h.clone()
        e46 = m46.last_edge[0].clone() if m46.last_edge is not None else None
    assert torch.allclose(y41, y46, atol=1e-5), "Unified OpenStroke46 must match OpenStroke41 stroke logits at step 0!"
    assert torch.allclose(h41, h46, atol=1e-5), "Unified OpenStroke46 must match OpenStroke41 hidden features at step 0!"
    if e41 is not None:
        assert torch.allclose(e41, e46, atol=1e-5), "Unified OpenStroke46 must match OpenStroke41 edge logits at step 0!"

    # Gradients flow cleanly through 18D topo_mlp, CADStemV2 (dw2/dw4/proj), 6D edge_style cols 4..5, and gat.2..3
    m46.train()
    torch.nn.init.normal_(m46.img.proj_fpn.weight, std=0.05)
    out = m46(b)
    loss = out.square().mean() + (m46.last_edge[0].square().mean() if m46.last_edge is not None else 0.0)
    loss.backward()
    for name, p in [
        ("topo_mlp[-1]", m46.topo_mlp[-1].weight),
        ("cad_stem.proj", m46.img.cad_stem.proj.weight),
        ("gat[0].geo_b", m46.gat[0].geo_b.weight),
        ("gat[2].ffn[-1]", m46.gat[2].ffn[-1].weight),
        ("edge_mlp.geo", m46.edge_mlp.geo.weight),
    ]:
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Invalid grad on {name}"
    assert m46.topo_mlp[-1].weight.grad.abs().sum() > 0

    # Upgrading from OpenStroke45 (automatically handles 7D->18D topo_mlp & 2ch->3ch cad_stem resize!)
    m46_b = build("openstroke46", 47, edge_head=True, edge_pairs=True, group_head=True, **tiny).eval()
    load_weights(m46_b, m45.state_dict(), new_heads=True)

    # Cross-system transfer: HVAC binary (edge_head + group_head, 1 logit) -> 4-class general vector model (0 extra heads, 4 logits)!
    m46_4c = build("openstroke46", 47, stroke_classes=4, class_names=["other", "main", "branch", "riser"], **tiny).eval()
    load_weights(m46_4c, m46.state_dict(), new_heads=True)
    assert torch.allclose(m46_4c.enc[0][0].cpe.conv.weight, m46.enc[0][0].cpe.conv.weight)
    assert torch.allclose(m46_4c.head[1].weight, m46.head[1].weight)
    assert m46_4c.head[3].out_features == 4


def test_graph_topo_v3_length_weighted_p90_and_louver_arrowhead_signals():
    from openstroke.data import HAS_FILL, HAS_STROKE, LOG_WIDTH, STROKE
    from openstroke.features import FEATURE_NAMES, N_FEATURES
    from openstroke.models.openstroke2 import _graph_topo_uni_features, _graph_topo_v2_features, _graph_topo_v3_features

    # Simulate VNG-2 distribution: 92% by COUNT of short 0.24pt furniture shapes (92 shapes x 5pt = 460pt)
    # vs 4 long 0.48pt duct shapes (4 x 250pt = 1000pt -> 68% by LENGTH!) + 2 hairline 0.00pt ducts
    # + 1 ultra-thin closed louver bar (0.4x6.5pt, op_l=4) 5pt away from a small filled triangle arrowhead!
    n_furn = 92
    n_duct = 4
    n_hair = 2
    n_louver = 1
    n_arr = 1
    N = n_furn + n_duct + n_hair + n_louver + n_arr
    x = np.zeros((1, N, N_FEATURES), dtype=np.float32)
    xy = np.zeros((1, N, 2), dtype=np.float32)
    length = np.zeros((1, N), dtype=np.float32)

    i_fc = FEATURE_NAMES.index("frac_c")
    i_cl = FEATURE_NAMES.index("closed_ends")
    i_ns = FEATURE_NAMES.index("log_nseg")
    i_bw = FEATURE_NAMES.index("log_bbox_w")
    i_bh = FEATURE_NAMES.index("log_bbox_h")
    i_dg = FEATURE_NAMES.index("log_bbox_diag")

    x[0, :, HAS_STROKE] = 1.0
    x[0, :, STROKE] = 0.0
    # 0..91: 0.24pt short furniture pieces in a 60pt mid-size straight bay
    x[0, :n_furn, LOG_WIDTH] = np.log1p(0.24 * 10.0)
    xy[0, :n_furn, 0] = np.linspace(0, 60, n_furn)
    length[0, :n_furn] = 5.0
    # 92..95: 0.48pt long duct run (span=750pt, with 1 curved elbow!)
    d0 = n_furn
    x[0, d0:d0 + n_duct, LOG_WIDTH] = np.log1p(0.48 * 10.0)
    x[0, d0 + 1, i_fc] = 1.0  # curved elbow in duct CC!
    xy[0, d0:d0 + n_duct, 0] = [0, 250, 500, 750]
    xy[0, d0:d0 + n_duct, 1] = 200.0
    length[0, d0:d0 + n_duct] = 250.0
    # 96..97: 0.00pt hairline duct run (span=400pt)
    h0 = d0 + n_duct
    x[0, h0:h0 + n_hair, LOG_WIDTH] = 0.0
    xy[0, h0:h0 + n_hair, 0] = [0, 400]
    xy[0, h0:h0 + n_hair, 1] = 400.0
    length[0, h0:h0 + n_hair] = 400.0
    # 98: ultra-thin closed louver bar (6.5 x 0.4pt, 4 segs, closed) at (500, 210)
    l0 = h0 + n_hair
    x[0, l0, LOG_WIDTH] = np.log1p(0.48 * 10.0)
    x[0, l0, i_cl] = 1.0
    x[0, l0, i_ns] = np.log1p(4.0)
    x[0, l0, i_bw] = np.log1p(6.5)
    x[0, l0, i_bh] = np.log1p(0.4)
    x[0, l0, i_dg] = np.log1p(6.51)
    xy[0, l0] = [500.0, 210.0]
    length[0, l0] = 13.8
    # 99: small filled triangle arrowhead (diag=2.5pt) 5pt away at (500, 215)
    a0 = l0 + 1
    x[0, a0, HAS_FILL] = 1.0
    x[0, a0, i_dg] = np.log1p(2.5)
    xy[0, a0] = [500.0, 215.0]
    length[0, a0] = 6.0

    # Edges: furniture chain + duct chain + hairline chain
    edges = [[k, k + 1] for k in range(n_furn - 1)] + [[d0 + k, d0 + k + 1] for k in range(n_duct - 1)] + [[h0, h0 + 1]]
    b = {
        "x": torch.from_numpy(x),
        "xy0": torch.from_numpy(xy),
        "length": torch.from_numpy(length),
        "mask0": torch.ones((1, N), dtype=torch.bool),
        "edges": torch.tensor([edges], dtype=torch.long),
        "emask": torch.ones((1, len(edges)), dtype=torch.bool),
    }

    t2 = _graph_topo_v2_features(b)[0].numpy()
    t3 = _graph_topo_v3_features(b)[0].numpy()
    tu = _graph_topo_uni_features(b)[0].numpy()
    assert tu.shape == (N, 18)
    assert np.allclose(tu[:, :12], t3, atol=1e-5)

    # Old v2 count-p90 collapsed to 0.24pt -> is_thinner_half was BLIND (0.0) on furniture!
    assert t2[0, 6] == 0.0
    # New v3 length-weighted p90/p95 sees 0.48pt duct -> furniture flagged thin AND mid_box!
    assert t3[0, 5] < -0.15 and t3[0, 6] == 1.0, f"Expected thin flag on furniture, got {t3[0, 5:7]}"
    assert t3[0, 7] == 0.0 and t3[0, 9] == 1.0, f"Expected straight mid-box flag on 60pt desk bay, got {t3[0, 7:10]}"
    # Duct CC has curved_frac = 0.25, rel_span = 1.0, NOT thin, NOT mid_box!
    assert t3[d0, 6] == 0.0 and t3[d0, 7] > 0.20 and t3[d0, 8] > 0.99 and t3[d0, 9] == 0.0
    # 0.00pt hairline stays strictly neutral!
    assert t3[h0, 5] == 0.0 and t3[h0, 6] == 0.0
    # Ultra-thin louver bar 5pt from filled arrowhead triggers near_small_fill & is_thin_louver_sig!
    assert t3[l0, 10] > 0.50 and t3[l0, 11] == 1.0
    assert t3[d0, 11] == 0.0


def test_neg_boxes_v5_generates_thin_louver_bar_and_border_hatch():
    from test_os44 import _synth_win_v44

    from openstroke.data import Augment, HAS_FILL, augment
    from openstroke.features import FEATURE_NAMES

    w = _synth_win_v44(200)
    n0 = len(w["x"])
    aug = Augment(fliplr=0, rot90=False, scale=0, color=0, gray=0, restyle=0, drop_tokens=0, neg_boxes=1.0)
    i_cl = FEATURE_NAMES.index("closed_ends")
    i_ns = FEATURE_NAMES.index("log_nseg")
    i_bw = FEATURE_NAMES.index("log_bbox_w")
    i_bh = FEATURE_NAMES.index("log_bbox_h")

    saw_thin_closed_bar = False
    saw_hatch_ticks = False
    for sd in range(25):
        out = augment({k: v.copy() if isinstance(v, np.ndarray) else v for k, v in w.items()}, aug, np.random.default_rng(sd))
        xs = out["x"][n0:]
        nseg = np.expm1(xs[:, i_ns])
        bw = np.expm1(xs[:, i_bw])
        bh = np.expm1(xs[:, i_bh])
        bshort = np.minimum(bw, bh)
        blong = np.maximum(bw, bh)
        if ((xs[:, i_cl] > 0.5) & (nseg >= 3.5) & (nseg <= 4.5) & (bshort <= 1.1) & (blong <= 14.5) & (xs[:, HAS_FILL] < 0.5)).any():
            saw_thin_closed_bar = True
        if int(((nseg >= 0.8) & (nseg <= 1.2) & (blong >= 3.0) & (blong <= 11.0)).sum()) >= 10:
            saw_hatch_ticks = True
    assert saw_thin_closed_bar, "neg_boxes v5 must generate ultra-thin closed louver bars (0.35-1.05pt)"
    assert saw_hatch_ticks, "neg_boxes v5 must generate Family 6 BORDER-HATCH parallel ticks"


def test_openstroke46_smoke_train_supplement_fg_edges_and_cli(tmp_path):
    import argparse
    from test_model import _synthetic

    from openstroke.cli import main, yolo_args
    from openstroke.folds import propose
    from openstroke.infer import Predictor
    from openstroke.train import add_args, finalize, run_version

    # Verify E151c (preset) CLI dry-parse
    ap = argparse.ArgumentParser()
    add_args(ap)
    raw = yolo_args([
        "data=datasets/e13_v4", "folds=datasets/e13_v4/folds_gt20.json",
        "model=openstroke46", "modes=hvac", "lr0=0.002", "warmup_epochs=2.0", "epochs=20",
    ])
    args_c = ap.parse_args(raw)
    args_c._explicit = {t[2:].replace("-", "_") for t in raw if t.startswith("--")}
    args_c = finalize(args_c)
    assert run_version(args_c).startswith("openstroke46")
    assert args_c.graph_topo_uni is True and args_c.img_cad_v2 is True and args_c.edge_style is True
    assert args_c.graph_topo_v2 is False and args_c.img_cad is False

    # Smoke train openstroke41 -> fine-tune to openstroke46 -> Predictor infer (exercises supplement_fg_edges with 6D geo!)
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
        "img_cad_v2=true", "img_max_px=128", "graph_topo_uni=true", "edge_style=true", "rel_attn=4",
        "neg_boxes=0.45", "lr0=0.0005", "warmup_epochs=1.0",
        "epochs=1", "name=v46_ft",
    ]) == 0
    run46 = tmp_path / "runs" / "v46_ft" / "fold0"
    ck46 = torch.load(run46 / "best.pt", map_location="cpu", weights_only=False)
    assert ck46["config"]["arch"] == "openstroke46"
    assert ck46["config"]["img_cad_v2"] is True
    assert ck46["config"]["graph_topo_uni"] is True
    assert ck46["config"]["edge_style"] is True
    assert ck46["config"]["rel_attn"] == 4

    pred = Predictor(run46 / "best.pt", device="cpu", threads=1, workers=1)
    pred.threshold = 0.0
    payload = {
        "page": {"width": 500, "height": 500},
        "shapes": [
            {"items": [["l", [10, 10], [200, 10]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0"},
            {"items": [["l", [200, 10], [200, 30]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0"},
            {"items": [["l", [200, 30], [10, 30]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0"},
            {"items": [["l", [10, 30], [10, 10]]], "type": "s", "color": [0, 0, 0], "width": 0.48, "layer": "0"},
        ],
    }
    res = pred.predict_payload(payload, mode="hvac")
    assert len(res["prob"]) == 4 and np.isfinite(res["prob"]).all()
    assert res["group_objects"] is not None

    # Cross-system CLI transfer: HVAC binary openstroke46 (with group_head + edge_head) -> 3-class FIRE dataset (auto-resizes head 1->3 & drops unused HVAC heads!)
    fire3_dir = tmp_path / "fire3_xfer"
    fire3_dir.mkdir()
    mf3 = {**manifest, "class_names": ["other", "main", "branch"]}
    (fire3_dir / "manifest.json").write_text(json.dumps(mf3), encoding="utf-8")
    (fire3_dir / "folds.json").write_text(json.dumps(propose(mf3, k=2)))
    for f in data.glob("*.npz"):
        with np.load(f) as z:
            d = {k: z[k] for k in z.files if k not in ("oinst", "ocls", "octr")}
        fg = np.flatnonzero(d["label"] > 0)
        y3 = np.zeros_like(d["label"], dtype=np.uint8)
        y3[fg[: len(fg) // 3]] = 1
        y3[fg[len(fg) // 3 :]] = 2
        d["label"] = y3
        np.savez(fire3_dir / f.name, **d)
    assert main([
        "train", f"data={fire3_dir}", "fold=0", f"model={run46 / 'best.pt'}",
        "img_max_px=128", "steps=2", "batch=2", "window=128", "val_batch=2",
        "workers=0", "device=cpu", "tta=false", "epochs=1",
        f"project={tmp_path / 'runs'}", "name=hvac46_to_fire3",
    ]) == 0
    ck_x = torch.load(tmp_path / "runs" / "hvac46_to_fire3" / "fold0" / "best.pt", map_location="cpu", weights_only=False)
    assert ck_x["config"]["arch"] == "openstroke46"
    assert ck_x["config"]["graph_topo_uni"] is True
    assert ck_x["config"]["stroke_classes"] == 3
    assert ck_x["config"]["class_names"] == ["other", "main", "branch"]
    assert ck_x["config"]["group_head"] is False and ck_x["config"]["edge_head"] is False


def test_worker_precomputed_topo_feat_exact_parity():
    import pickle
    from functools import partial
    from test_os44 import _synth_win_v44

    from openstroke.data import collate
    from openstroke.features import N_FEATURES
    from openstroke.models.openstroke2 import OpenStroke2, _graph_topo_uni_features, _kcore2_and_peel, topo_mode_of

    # Picklability for multiprocessing DataLoader workers
    fn = partial(collate, topo_mode="uni")
    fn2 = pickle.loads(pickle.dumps(fn))

    wins = [_synth_win_v44(160), _synth_win_v44(140)]
    b_plain = collate(wins)
    b_pre = fn2(wins)
    assert "topo_feat" not in b_plain
    assert "topo_feat" in b_pre and b_pre["topo_feat"].shape == (2, b_plain["mask0"].shape[1], 18)

    t_live = _graph_topo_uni_features(b_plain)
    assert torch.equal(b_pre["topo_feat"], t_live), "Worker-precomputed topo_feat must be bit-exact to live forward!"

    torch.manual_seed(7)
    m = OpenStroke2(
        N_FEATURES, dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32,
        image_branch=True, img_pretrained=False, img_fpn=True, img_cad_v2=True, img_max_px=128,
        rel_attn=2, edge_style=True, graph_topo_uni=True, stroke_classes=3,
    ).eval()
    assert topo_mode_of(m) == "uni"
    # Give topo_mlp non-zero final weights so topo_feat actually drives logits!
    with torch.no_grad():
        m.topo_mlp[-1].weight.fill_(0.05)
        z_live = m(b_plain)
        z_pre = m(b_pre)
    assert torch.allclose(z_live, z_pre, atol=1e-6), f"Max diff {(z_live - z_pre).abs().max().item()}"

    # Edge cases for CSR _kcore2_and_peel: duplicate edges + self-loops + large loop + leaf chain
    xy = np.array([[0.0, 0.0], [80.0, 0.0], [80.0, 80.0], [0.0, 80.0], [100.0, 0.0], [120.0, 0.0]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 0], [1, 2], [2, 3], [3, 0], [0, 0], [1, 4], [4, 5]], dtype=np.int64)
    c2, sp2, peel = _kcore2_and_peel(6, edges, xy)
    assert c2[:4].all() and not c2[4:].any()
    assert sp2[0] == 80.0 and peel[0] == 1.0 and peel[5] < peel[4] < 1.0
