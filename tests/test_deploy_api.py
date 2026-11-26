"""Tests for OpenOBB-style deployment API (no mode required, strictly raw output, separate post-processing),
``strip`` deploy checkpoints, CLI ``predict`` / ``strip`` / ``profile``, and exact parity of inference speedups."""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch = pytest.importorskip("torch")
torch.set_num_threads(1)


def _sample_payload(n_lines: int = 24, include_empty: bool = True) -> dict:
    rng = np.random.default_rng(7)
    shapes = []
    for k in range(n_lines):
        x0, y0 = float(rng.uniform(10, 180)), float(rng.uniform(10, 180))
        dx, dy = float(rng.uniform(-20, 20)), float(rng.uniform(-20, 20))
        shapes.append({
            "items": [["l", [x0, y0], [x0 + dx, y0 + dy]]],
            "type": "s",
            "color": [0.0, 0.0, 0.0],
            "width": 0.48 if k % 3 == 0 else 0.24,
            "layer": "0",
        })
    if include_empty:
        shapes.append({"items": [], "type": "s", "color": [0.0, 0.0, 0.0], "width": 0.24, "layer": "0"})
    return {"shapes": shapes, "document_metadata": {"page_width": 200.0, "page_height": 200.0}}


def _save_tiny_checkpoint(path: Path, arch: str = "openstroke45") -> Path:
    from openstroke.features import FEATURE_NAMES, N_FEATURES
    from openstroke.models import build

    m = build(arch, N_FEATURES, dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32, img_pretrained=False).eval()
    # Non-zero CADStem / FPN projections so fusion & render_multi parity tests exercise real non-trivial weights
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 2 and float(p.abs().sum()) == 0.0:
                torch.nn.init.normal_(p, std=0.05)
    ck = {
        "model": m.state_dict(),
        "config": dict(m.config),
        "args": {"window": 256, "modes": "hvac", "data": "/content/os44/e13_v4"},
        "feature_names": list(FEATURE_NAMES),
        "threshold": 0.65,
        "tta": 8,
        "val_f2": 0.945,
        "epoch": 14,
    }
    torch.save(ck, path)
    return path


def test_render_multi_and_fuse_image_branch_exact_parity():
    from openstroke.data import HAS_FILL, HAS_STROKE, LOG_WIDTH, STROKE
    from openstroke.features import N_FEATURES
    from openstroke.models import build
    from openstroke.models.image_branch import fuse_image_branch, render, render_multi

    torch.manual_seed(0)
    B, N, P = 2, 64, 16
    xy0 = torch.randn(B, N, 2) * 80.0
    pts = torch.randn(B, N, P, 2) * 8.0
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[0, -5:] = False
    x = torch.zeros(B, N, N_FEATURES)
    x[..., HAS_STROKE] = 1.0
    x[..., LOG_WIDTH] = torch.rand(B, N) * 2.5
    x[..., STROKE] = torch.rand(B, N, 3)
    x[..., HAS_FILL] = (torch.rand(B, N) > 0.8).float()

    # 1) render_multi must match repeated render bit-for-bit
    w_val = (x[..., LOG_WIDTH] / 3.0).clamp(0.0, 1.0)
    f_val = x[..., HAS_FILL]
    abs_pts = xy0[:, :, None, :] + pts
    m0, s0, S0 = render(xy0, pts, mask, ppt=1.0, max_px=256, ink=w_val, abs_pts=abs_pts)
    m1, _, _ = render(xy0, pts, mask, ppt=1.0, max_px=256, ink=f_val, abs_pts=abs_pts)
    maps, sm, Sm = render_multi(xy0, pts, mask, [(w_val, False), (f_val, True)], ppt=1.0, max_px=256, abs_pts=abs_pts)
    assert Sm == S0 and torch.equal(sm, s0)
    assert torch.equal(maps[0], m0)
    assert torch.equal(maps[1], m1)

    # 2) Conv+FrozenBN fusion must match unfused ImageBranch output
    m_ref = build("openstroke45", N_FEATURES, dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32, img_pretrained=False).eval()
    with torch.no_grad():
        torch.nn.init.normal_(m_ref.img.proj.weight, std=0.05)
        torch.nn.init.normal_(m_ref.img.proj_fpn.weight, std=0.05)
        torch.nn.init.normal_(m_ref.img.cad_stem.net[-1].weight, std=0.05)
    m_ref.img.max_px = 128

    m_fused = build("openstroke45", N_FEATURES, dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32, img_pretrained=False).eval()
    m_fused.load_state_dict(m_ref.state_dict())
    m_fused.img.max_px = 128
    fuse_image_branch(m_fused, channels_last=False)

    b = {"x": x, "xy0": xy0, "pts": pts, "mask0": mask}
    with torch.inference_mode():
        y_ref = m_ref.img(b)
        y_fus = m_fused.img(b)
    assert torch.allclose(y_ref, y_fus, atol=1e-4, rtol=1e-4)


def test_openstroke_api_no_mode_raw_outputs_and_strip_parity(tmp_path):
    from openstroke import OpenStroke, Predictor, Results, Strokes
    from openstroke.strip import strip

    ckpt = _save_tiny_checkpoint(tmp_path / "best.pt", arch="openstroke45")
    pay = _sample_payload(24, include_empty=True)
    p_json = tmp_path / "page_01.json"
    p_gz = tmp_path / "page_02.json.gz"
    p_json.write_text(json.dumps(pay), encoding="utf-8")
    with gzip.open(p_gz, "wt", encoding="utf-8") as f:
        json.dump(pay, f)

    # Load model -> infer WITHOUT any mode -> raw Results
    model = OpenStroke(ckpt, device="cpu", threads=1)
    assert "openstroke45" in model.info(verbose=False)
    assert model.warmup(32) >= 0.0

    # Dict input, shorthand call, file path, gz path, folder, stream=True
    res_dict = model(pay)
    assert len(res_dict) == 1
    r = res_dict[0]
    assert isinstance(r, Results) and isinstance(r.strokes, Strokes)
    assert r.prob.shape == (25,) and r.prob.dtype == np.float32
    assert r.valid.shape == (25,) and r.valid.dtype == bool
    assert bool(r.valid[-1]) is False and float(r.prob[-1]) == 0.0
    assert r.kept == 24
    assert r.info is None   # lightweight raw deploy mode does not hold shapely/segments unless requested
    assert set(r.speed.keys()) == {"preprocess", "inference", "postprocess"}
    assert r.edges is not None and r.groups is not None
    assert "names" in r.groups and len(r.groups["names"]) == 8

    # JSON-serializable microservice payload + npz/json persistence
    d_raw = r.to_dict()
    assert d_raw["n_shapes"] == 25 and d_raw["kept"] == 24 and len(d_raw["prob"]) == 25
    assert json.loads(r.to_json())["n_shapes"] == 25
    npz_out = Path(r.save_npz(tmp_path / "out.npz"))
    assert npz_out.exists()
    with np.load(npz_out) as z:
        assert np.allclose(z["prob"], r.prob) and np.array_equal(z["valid"], r.valid)

    # Folder + stream=True yields both .json and .json.gz identically
    res_dir = list(model.predict(tmp_path, stream=True))
    assert len(res_dir) == 2
    for rd in res_dir:
        assert np.allclose(rd.prob, r.prob, atol=1e-5)

    # Fast stroke-only serving (include_groups=False, include_edges=False) gives identical prob
    r_fast = model.predict(pay, include_groups=False, include_edges=False)[0]
    assert r_fast.groups is None and r_fast.edges is None
    assert np.allclose(r_fast.prob, r.prob, atol=1e-5)

    # Legacy Predictor works both WITH and WITHOUT mode and matches OpenStroke prob
    pred = Predictor(ckpt, device="cpu", threads=1)
    leg_no_mode = pred.predict_payload(pay)
    leg_with_mode = pred.predict_payload(pay, mode="hvac")
    assert np.allclose(leg_no_mode["prob"], r.prob, atol=1e-5)
    assert np.allclose(leg_with_mode["prob"], r.prob, atol=1e-5)
    assert leg_no_mode["info"] is not None

    # Stripped deploy checkpoints (fp32 & fp16) drop all training leaks and preserve predictions
    dep32 = Path(strip(ckpt, out=tmp_path / "dep32.pt", half=False))
    dep16 = Path(strip(ckpt, out=tmp_path / "dep16.pt", half=True))
    assert dep16.stat().st_size < dep32.stat().st_size
    ck16 = torch.load(dep16, map_location="cpu", weights_only=False)
    assert ck16["deploy"] is True and "args" not in ck16 and "val_f2" not in ck16
    assert "/content" not in json.dumps({k: v for k, v in ck16.items() if k != "model"})
    # F249: half=True keeps 1D norm/stat/bias buffers in FP32 while storing >=2D weight matrices in FP16
    sd16 = ck16["model"]
    f1d = [t for t in sd16.values() if t.is_floating_point() and t.ndim < 2]
    f2d = [t for t in sd16.values() if t.is_floating_point() and t.ndim >= 2]
    assert f1d and all(t.dtype == torch.float32 for t in f1d)
    assert f2d and all(t.dtype == torch.float16 for t in f2d)

    r32 = OpenStroke(dep32, device="cpu", threads=1).predict(pay)[0]
    r16 = OpenStroke(dep16, device="cpu", threads=1).predict(pay)[0]
    assert np.allclose(r32.prob, r.prob, atol=1e-5)
    assert np.allclose(r16.prob, r.prob, atol=1e-2)

    # Input validation
    with pytest.raises(TypeError):
        model.predict(12345)
    with pytest.raises(ValueError):
        model.predict({"no_shapes": []})


def test_openstroke_deploy_imports_no_postprocessing_modules(tmp_path):
    """Standalone deployment inference must never pull in HVAC/FIRE post-processing or sklearn."""
    ckpt = _save_tiny_checkpoint(tmp_path / "best.pt", arch="openstroke45")
    p_json = tmp_path / "p.json"
    p_json.write_text(json.dumps(_sample_payload(12)), encoding="utf-8")

    code = (
        "import sys; from openstroke import OpenStroke; "
        f"m = OpenStroke(r'{ckpt}', device='cpu', threads=1); "
        f"r = m.predict(r'{p_json}')[0]; "
        "bad = [k for k in sys.modules if any(x in k for x in ('hvac_faces', 'hvac_refine', 'line_decoder', 'sklearn'))]; "
        "assert not bad, f'Post-processing imported during raw inference: {bad}'; "
        "print('OK', len(r.prob), r.kept)"
    )
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert "OK 13 12" in proc.stdout


def test_cli_predict_strip_and_profile(tmp_path):
    from openstroke.cli import main

    ckpt = _save_tiny_checkpoint(tmp_path / "best.pt", arch="openstroke45")
    p_json = tmp_path / "page.json"
    p_json.write_text(json.dumps(_sample_payload(16)), encoding="utf-8")

    dep = tmp_path / "deploy.pt"
    assert main(["strip", f"model={ckpt}", f"out={dep}", "half=true"]) == 0
    assert dep.exists()

    out_proj = tmp_path / "runs"
    assert main([
        "predict", f"model={dep}", f"source={p_json}", "device=cpu", "threads=1",
        "save=true", "save_json=true", f"project={out_proj}", "name=p1", "verbose=false",
    ]) == 0
    assert (out_proj / "p1" / "page.npz").exists()
    assert (out_proj / "p1" / "page.json").exists()

    assert main(["profile", f"model={dep}", f"source={p_json}", "device=cpu", "threads=1"]) == 0


def test_two_service_http_json_roundtrip_and_separate_postprocessing(tmp_path):
    """End-to-end two-service contract: Service A returns compact JSON raw -> Service B reconstructs & runs post-proc."""
    from openstroke import OpenStroke, Results
    from openstroke.line_decoder import postprocess_lines

    ckpt = _save_tiny_checkpoint(tmp_path / "best.pt", arch="openstroke45")
    pay = _sample_payload(24, include_empty=True)

    # Service A (standalone model server — raw only, no mode)
    r = OpenStroke(ckpt, device="cpu", threads=1).predict(pay)[0]
    dense_str = r.to_json(compact=False)
    compact_str = r.to_json(compact=True, min_prob=0.05)

    # Service B (API caller) reconstructs Results from JSON or .npz
    rd = Results.from_dict(dense_str)
    rc = Results.from_dict(compact_str)
    rnpz = Results.from_npz(r.save_npz(tmp_path / "raw.npz"))
    assert np.allclose(rd.prob, r.prob, atol=1e-4) and np.array_equal(rd.valid, r.valid)
    assert np.array_equal(rc.valid, r.valid)
    fg = r.valid & (r.prob >= 0.05)
    assert np.allclose(rc.prob[fg], r.prob[fg], atol=1e-4)
    assert np.allclose(rnpz.prob, r.prob, atol=1e-6) and np.array_equal(rnpz.valid, r.valid)
    assert rd.groups is not None and rc.groups is not None
    assert np.array_equal(rc.groups["instance_id"], r.groups["instance_id"])

    # Caller runs decoupled post-processing directly on (payload, raw_json)
    lp = postprocess_lines(pay, compact_str, thr=0.0, mode="fire")
    assert "lines" in lp and "fittings" in lp and len(lp["lines"]) > 0
