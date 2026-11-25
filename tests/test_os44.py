"""OpenStroke44 (H170 / H171 / F241): CADStem (img_cad, 2048px) + g3 CC topology (graph_topo) + 4-layer GAT (rel_attn=4)
+ width-order-preserving restyle_robust + neg_boxes v3 (6 OOD CAD families incl. filled arrowheads & office desks)
+ Viewer MEP route-connectivity cleanup (tools/build_e13_v4.py)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch = pytest.importorskip("torch")
torch.set_num_threads(1)


def _synth_win_v44(n: int = 180):
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
    x[:, LOG_WIDTH] = np.where(lab > 0, np.log1p(0.48 * 10.0), np.log1p(0.24 * 10.0)).astype(np.float32)
    x[:, STROKE] = np.where(lab[:, None] > 0, [0.05, 0.05, 0.05], [0.6, 0.6, 0.6])
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


def test_openstroke44_registry_naming_and_exact_step0_parity_from_openstroke41(tmp_path):
    from test_model import _synthetic

    from openstroke.data import collate, load_page, window
    from openstroke.models import ARCHS, build, load_weights
    from openstroke.train import PRESETS, arch_name, describe_arch

    assert ARCHS["openstroke44"]["img_cad"] is True
    assert ARCHS["openstroke44"]["img_max_px"] == 2048
    assert ARCHS["openstroke44"]["graph_topo"] is True
    assert ARCHS["openstroke44"]["rel_attn"] == 4
    assert ARCHS["openstroke44"].get("stroke_context", False) is False
    assert ARCHS["openstroke44"].get("facing_edges", False) is False
    assert ARCHS["openstroke44"]["group_head"] is True

    preset_cfg = PRESETS["openstroke44"][1]
    assert preset_cfg["feature_drop"] == "layer_stats"
    assert preset_cfg["neg_boxes"] == 0.45
    assert preset_cfg["lr0"] == 0.0005
    assert preset_cfg["img_max_px"] == 2048

    # Upgrading openstroke41 with img_cad + graph_topo + rel_attn=4 must cleanly name itself "openstroke44" (NO stray "+ rel_attn=2"!)
    cfg44 = {"arch": "openstroke41", **ARCHS["openstroke41"], "img_cad": True, "graph_topo": True, "rel_attn": 4, "img_max_px": 2048}
    assert describe_arch(cfg44) == "openstroke44"
    assert arch_name(cfg44) == "openstroke44"

    tiny = dict(dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32, img_pretrained=False)
    m41 = build("openstroke41", 47, **tiny).eval()
    m44 = build("openstroke44", 47, **tiny).eval()
    assert m44.img.cad_stem is not None
    assert m44.img.stroke_context is None
    assert m44.topo_mlp is not None
    assert len(m44.gat) == 4
    assert m44.gat[0].kind_b.num_embeddings == 8
    assert m44.edge_mlp.kind.num_embeddings == 8

    load_weights(m44, m41.state_dict(), new_heads=True)

    data, _ = _synthetic(tmp_path)
    stem = sorted(p.stem for p in data.glob("*.npz"))[0]
    pg = load_page(data, stem, "hvac", tmp_path / "cache", graph=True, no_layers=True)
    b = collate([window(pg, 0, 128)])
    # Match image resolution on synthetic check so step-0 parity compares identical FPN grids
    m44.img.max_px = m41.img.max_px
    with torch.no_grad():
        y41 = m41(b)
        e41 = m41.last_edge[0].clone() if m41.last_edge is not None else None
        h41 = m41.last_h.clone()
        y44 = m44(b)
        e44 = m44.last_edge[0].clone() if m44.last_edge is not None else None
        h44 = m44.last_h.clone()
    assert torch.allclose(y41, y44, atol=1e-5), "OpenStroke44 must match OpenStroke41 stroke logits at step 0!"
    assert torch.allclose(h41, h44, atol=1e-5), "OpenStroke44 must match OpenStroke41 hidden features at step 0!"
    if e41 is not None:
        assert torch.allclose(e41, e44, atol=1e-5), "OpenStroke44 must match OpenStroke41 edge logits at step 0!"

    # Finite non-zero gradients through CADStem, topo_mlp, and expanded gat.2..3
    # (Simulate a trained E145 ImageBranch where img.proj_fpn is non-zero; fresh m41 has zero-init img.proj_fpn)
    m44.train()
    torch.nn.init.normal_(m44.img.proj_fpn.weight, std=0.05)
    out = m44(b)
    loss = out.square().mean() + (m44.last_edge[0].square().mean() if m44.last_edge is not None else 0.0)
    loss.backward()
    for name, p in [
        ("cad_stem.net[-1]", m44.img.cad_stem.net[-1].weight),
        ("topo_mlp[-1]", m44.topo_mlp[-1].weight),
        ("gat[2].ffn[-1]", m44.gat[2].ffn[-1].weight),
        ("gat[3].ffn[-1]", m44.gat[3].ffn[-1].weight),
    ]:
        assert p.grad is not None, f"Missing grad on {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad on {name}"
    assert m44.img.cad_stem.net[-1].weight.grad.abs().sum() > 0
    assert m44.topo_mlp[-1].weight.grad.abs().sum() > 0


def test_openstroke44_loads_real_e145_checkpoint_if_present():
    ck_p = Path(r"F:\Downloads\e145_os41_robust\fold0\best.pt")
    if not ck_p.exists():
        pytest.skip("Local E145 checkpoint not present")
    from openstroke.models import build, load_weights

    ck = torch.load(ck_p, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    m44 = build(
        "openstroke44", cfg["n_features"],
        dims=cfg["dims"], enc_depths=cfg["enc_depths"], dec_depths=cfg["dec_depths"],
        patch=cfg["patch"], pretrain=cfg.get("pretrain", False), img_pretrained=False, img_max_px=2048,
    )
    load_weights(m44, ck["model"], new_heads=True)
    assert len(m44.gat) == 4 and m44.img.cad_stem is not None and m44.topo_mlp is not None


def test_restyle_robust_v44_preserves_width_order_and_hairline_invariance():
    from openstroke.data import LOG_WIDTH, restyle_robust

    w0 = _synth_win_v44(120)
    lab = w0["label"] > 0
    saw_preserved = 0
    saw_hairline = 0
    saw_flat = 0
    for sd in range(80):
        x = w0["x"].copy()
        restyle_robust(x, np.random.default_rng(sd))
        wd = x[lab, LOG_WIDTH]
        wb = x[~lab, LOG_WIDTH]
        if (wd > wb[0] + 1e-4).all() and (wb > 0).all():
            saw_preserved += 1
        elif (wd == 0).any() or (wb == 0).any():
            saw_hairline += 1
        elif np.allclose(x[:, LOG_WIDTH], x[0, LOG_WIDTH]):
            saw_flat += 1
        # Must NEVER invert non-zero background width above non-zero duct width!
        if (wd > 0).all() and (wb > 0).all():
            assert wd.mean() >= wb.mean() - 1e-5, "restyle_robust v44 must never invert thick duct < thin background!"
    assert saw_preserved >= 35
    assert saw_hairline >= 6
    assert saw_flat >= 1


def test_neg_boxes_v3_all_6_families_and_zero_shortcut_leak():
    from openstroke.data import Augment, HAS_FILL, HAS_STROKE, LOG_WIDTH, STROKE, augment
    from openstroke.features import FEATURE_NAMES

    w = _synth_win_v44(180)
    n0 = len(w["x"])
    aug = Augment(fliplr=0, rot90=False, scale=0, color=0, gray=0, restyle=0, drop_tokens=0, neg_boxes=1.0)

    saw_filled_arrow_or_bar = False
    saw_zigzag_5seg = False
    for sd in range(10):
        out = augment({k: v.copy() if isinstance(v, np.ndarray) else v for k, v in w.items()}, aug, np.random.default_rng(sd))
        n1 = len(out["x"])
        assert n1 > n0
        syn = slice(n0, n1)
        xs = out["x"][syn]
        assert (out["label"][syn] == 0).all()
        assert (out["style"][syn] == 7).all()
        assert np.allclose(xs[:, STROKE], [0.05, 0.05, 0.05], atol=1e-5)
        assert (xs[:, LOG_WIDTH] <= np.log1p(4.8) + 1e-5).all() and (xs[:, LOG_WIDTH] > 0.0).all()
        assert (xs[:, HAS_STROKE] == 1.0).all()
        assert np.isfinite(xs).all()
        assert (xs[:, FEATURE_NAMES.index("nb50_log")] > 0.0).all()
        if (xs[:, HAS_FILL] > 0.5).any():
            saw_filled_arrow_or_bar = True
        nseg = np.expm1(xs[:, FEATURE_NAMES.index("log_nseg")])
        if ((nseg >= 4.8) & (nseg <= 5.2)).any():
            saw_zigzag_5seg = True
    assert saw_filled_arrow_or_bar, "neg_boxes v3 must generate filled triangle arrowheads / bars"
    assert saw_zigzag_5seg, "neg_boxes v3 must generate 5-segment zigzag airflow arrows"


def test_build_e13_v4_physical_mep_connectivity_rule():
    from tools.build_e13_v4 import _classify_dropped_objects

    # 0: Duct connected to 1: Elbow (KEEP route!)
    # 2: Duct connected to 3: Chân rẽ (KEEP 2-obj duct+chân rẽ pair!)
    # 4: Isolated Chân rẽ (DROP R1_iso_fitting!)
    # 5: Isolated Duct (DROP R2_iso_duct!)
    # 6, 7: 2 touching Chân rẽ with 0 Ducts (DROP R3_noduct_fit_cc!)
    # 8, 9: 2 touching short Ducts with 0 fittings (span < 60pt -> DROP R3_nofit_duct_cc!)
    o_bb = np.array([
        [0, 0, 50, 12],      # 0 Duct
        [50, 0, 65, 15],     # 1 Elbow (touches 0)
        [100, 0, 115, 6],    # 2 small Duct
        [115, 0, 121, 6],    # 3 Chân rẽ (touches 2)
        [200, 0, 206, 10],   # 4 iso Chân rẽ
        [250, 0, 280, 15],   # 5 iso Duct
        [320, 0, 326, 8],    # 6 FCU collar A
        [327, 0, 333, 8],    # 7 FCU collar B (touches 6, 0 Ducts)
        [400, 0, 418, 12],   # 8 desk box A (Duct)
        [418, 0, 436, 12],   # 9 desk box B (Duct, touches 8, 0 fittings, span 36 < 60)
    ], dtype=np.float64)
    o_is_duct = np.array([True, False, True, False, False, True, False, False, True, True], dtype=bool)
    dropped, why = _classify_dropped_objects(o_bb, o_is_duct)
    assert dropped[:4].tolist() == [False, False, False, False]
    assert dropped[4:].tolist() == [True, True, True, True, True, True]
    assert why[4] == "R1_iso_fitting"
    assert why[5] == "R2_iso_duct"
    assert why[6] == "R3_noduct_fit_cc" and why[7] == "R3_noduct_fit_cc"
    assert why[8] == "R3_nofit_duct_cc" and why[9] == "R3_nofit_duct_cc"


def test_openstroke44_trainer_smoke_finetune_from_41_and_infer(tmp_path):
    from test_model import _synthetic

    from openstroke.cli import main
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
    assert main([
        "train", *common, "model=openstroke41", "dims=16,32", "enc_depths=1,1", "dec_depths=1",
        "patch=32", "img_pretrained=false", "epochs=1", "name=v41_src",
    ]) == 0
    src41 = tmp_path / "runs" / "v41_src" / "fold0" / "best.pt"

    assert main([
        "train", *common, f"model={src41}",
        "img_cad=true", "graph_topo=true", "rel_attn=4", "img_max_px=2048",
        "neg_boxes=0.45", "lr0=0.0005", "warmup_epochs=1.0",
        "epochs=1", "name=v44_ft",
    ]) == 0
    run44 = tmp_path / "runs" / "v44_ft" / "fold0"
    ck44 = torch.load(run44 / "best.pt", map_location="cpu", weights_only=False)
    assert ck44["config"]["arch"] == "openstroke44"
    assert ck44["config"]["img_cad"] is True
    assert ck44["config"]["graph_topo"] is True
    assert ck44["config"]["rel_attn"] == 4
    assert ck44["config"]["img_max_px"] == 2048
    assert ck44["config"].get("stroke_context", False) is False
    assert ck44["config"].get("facing_edges", False) is False

    pred = Predictor(run44 / "best.pt", device="cpu", threads=1, workers=1)
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


def test_apply_visual_audit_v4_preserves_1d_seam_fg_and_reclasses(tmp_path):
    from tools.apply_visual_audit_v4 import apply_one_viewer_page

    # 3 strokes:
    #  row 0: kept Duct oid=10 (0..50 x 0..10)
    #  row 1: 1D seam sliver oid=11 touching Duct 10 at x=50 (50..50 x 0..10) -> DROP as object, KEEP label=1 & merge into oid=10!
    #  row 2: standalone FCU collar oid=12 (200..205 x 200..208) -> DROP completely (label=0, oinst=-1)!
    #  row 3: mislabeled ChanRe oid=13 (50..65 x 0..10) -> reclass to Concentric Reducer (ocls=4)!
    src = tmp_path / "vw_test_p0001.npz"
    dst = tmp_path / "out_vw_test_p0001.npz"
    audit = tmp_path / "audit"
    (audit / "vw_test_p0001").mkdir(parents=True)
    np.savez(
        src,
        label=np.array([1, 1, 1, 1], dtype=np.int8),
        bbox=np.array([
            [0.0, 0.0, 50.0, 10.0],
            [50.0, 0.0, 50.0, 10.0],
            [200.0, 200.0, 205.0, 208.0],
            [50.0, 0.0, 65.0, 10.0],
        ], dtype=np.float32),
        oinst=np.array([10, 11, 12, 13], dtype=np.int32),
        ocls=np.array([1, 1, 7, 7], dtype=np.int8),
        octr=np.array([[25, 5], [50, 5], [202, 204], [57, 5]], dtype=np.float32),
        layer_id=np.array([0, 0, 0, 0], dtype=np.int32),
        layer_names=np.array(["DUCT"]),
    )
    (audit / "vw_test_p0001" / "page_meta.json").write_text(json.dumps({
        "total_objs": 4,
        "clusters": [
            {"cid": "C1", "oids": [11], "objects": [{"oid": 11, "w": 0.0, "h": 10.0, "deg": 2, "flags": ["tiny_sliver"]}]},
            {"cid": "C2", "oids": [12], "objects": [{"oid": 12, "w": 5.0, "h": 8.0, "deg": 0, "flags": ["iso_fit"]}]},
        ],
    }), encoding="utf-8")
    (audit / "vw_test_p0001" / "decision.json").write_text(json.dumps({
        "page_status": "fix",
        "drop_oids": [11, 12],
        "keep_oids": [],
        "drop_bulk_layers": [],
        "reclass_oids": [{"oid": 13, "old_cls": "Chân rẽ", "new_cls": "Concentric Reducer"}],
    }), encoding="utf-8")

    res = apply_one_viewer_page((str(src), str(dst), str(audit), str(tmp_path)))
    assert res["dropped_noise_oids"] == 1
    assert res["dropped_1d_seam_oids"] == 1
    assert res["reassigned_seam_strokes"] == 1
    assert res["reclassed_oids"] == 1

    with np.load(dst) as z:
        assert z["label"].tolist() == [1, 1, 0, 1]
        assert z["oinst"].tolist() == [10, 10, -1, 13]
        assert z["ocls"].tolist() == [1, 1, 0, 4]

