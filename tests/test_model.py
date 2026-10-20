"""Model v1, features, folds, metrics and the train loop on a tiny synthetic dataset (CPU, one thread)."""

import json
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from openstroke.features import N_FEATURES, page_tokens  # noqa: E402
from openstroke.folds import propose, resolve  # noqa: E402
from openstroke.metrics import prf, sweep  # noqa: E402
from openstroke.model import OpenStroke1, knn, loss_fn, morton_order  # noqa: E402


def _batch(B=2, N=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, N_FEATURES, generator=g)
    xy = torch.rand(B, N, 2, generator=g) * 300
    layer = torch.randint(0, 5, (B, N), generator=g)
    mask = torch.ones(B, N, dtype=torch.bool)
    if B > 1:
        mask[1, 50:] = False
    y = (torch.rand(B, N, generator=g) > 0.7).float()
    length = torch.rand(B, N, generator=g) * 50
    return x, xy, layer, mask, y, length


def test_morton_order_is_a_spatially_local_permutation():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1000, size=(5000, 2))
    o = morton_order(xy)
    assert sorted(o.tolist()) == list(range(5000))
    step = np.hypot(*np.diff(xy[o], axis=0).T)
    assert np.median(step) < 0.2 * np.median(np.hypot(*np.diff(xy, axis=0).T))


def test_knn_skips_padding_and_includes_self():
    xy = torch.tensor([[[0.0, 0], [1, 0], [2, 0], [100, 100]]])
    mask = torch.tensor([[True, True, True, False]])
    idx = knn(xy, mask, 3)
    assert (idx[0, :, 0] == torch.tensor([0, 1, 2, 0])).all() or (idx[0, :3, 0] == torch.arange(3)).all()
    assert not (idx[0, :3] == 3).any()


def test_forward_backward_finite_and_every_part_gets_gradient():
    torch.manual_seed(0)
    model = OpenStroke1(N_FEATURES, dim=32, depth=2, heads=4, k=8)
    x, xy, layer, mask, y, length = _batch()
    logits = model(x, xy, layer, mask)
    assert logits.shape == (2, 64) and torch.isfinite(logits).all()
    loss = loss_fn(logits, y, length, mask)
    assert torch.isfinite(loss)
    loss.backward()
    # LayerPool output is zero-init: its own weight gets gradient on the first step
    for name, p in model.named_parameters():
        assert p.grad is not None, name
    assert model.blocks[0].pool.out.weight.grad.abs().sum() > 0


def test_layerpool_starts_as_identity():
    """Zero-init: at init the model equals the same weights without LayerPool."""

    torch.manual_seed(0)
    with_pool = OpenStroke1(N_FEATURES, dim=32, depth=2, k=8, layer_pool=True)
    without = OpenStroke1(N_FEATURES, dim=32, depth=2, k=8, layer_pool=False)
    missing = without.load_state_dict(with_pool.state_dict(), strict=False)
    assert not missing.missing_keys
    x, xy, layer, mask, _, _ = _batch()
    with torch.no_grad():
        assert torch.allclose(with_pool(x, xy, layer, mask), without(x, xy, layer, mask), atol=1e-6)


def test_padding_does_not_change_valid_outputs():
    torch.manual_seed(0)
    model = OpenStroke1(N_FEATURES, dim=32, depth=2, k=8).eval()
    for blk in model.blocks:  # make LayerPool active so its masking is tested too
        torch.nn.init.normal_(blk.pool.out.weight, std=0.1)
    x, xy, layer, mask, _, _ = _batch(B=1, N=40)
    xp = torch.cat([x, torch.randn(1, 24, N_FEATURES)], 1)
    xyp = torch.cat([xy, torch.rand(1, 24, 2)], 1)
    lp = torch.cat([layer, torch.zeros(1, 24, dtype=torch.long)], 1)
    mp = torch.cat([mask, torch.zeros(1, 24, dtype=torch.bool)], 1)
    with torch.no_grad():
        a = model(x, xy, layer, mask)
        b = model(xp, xyp, lp, mp)[:, :40]
    assert torch.allclose(a, b, atol=1e-5)


def test_metrics_length_weighted():
    assert prf(8, 2, 2)["recall"] == 0.8
    s = [np.array([0.9, 0.2, 0.6])]
    y = [np.array([1, 0, 0])]
    L = [np.array([10.0, 1.0, 1.0])]
    best = sweep(s, y, L)["best"]
    assert best["recall"] == 1.0 and best["precision"] == 1.0 and best["threshold"] > 0.6


# --- synthetic E0 dataset and the train loop ---------------------------------------------------------------

def _write_page(out: Path, stem: str, seed: int):
    rng = np.random.default_rng(seed)
    n = 300
    xs = rng.uniform(0, 500, n)
    ys = rng.uniform(0, 500, n)
    label = (rng.random(n) < 0.3).astype(np.uint8)
    seg = np.stack([xs, ys, xs + 5, ys + np.where(label > 0, 0, 3)], axis=1).astype(np.float32)
    nan3 = np.full((n, 3), np.nan, np.float32)
    meta = {"stem": stem, "mode": "fire", "page_number": 1}
    np.savez_compressed(
        out / f"{stem}.npz", meta=np.array(json.dumps(meta)), label=label,
        true_frac=label.astype(np.float32), length=np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]),
        segments=seg, shape_offsets=np.arange(n + 1), layer_id=label.astype(np.int32),
        layer_names=np.array(["other", "pipe"]), type_code=np.zeros(n, np.int8),
        stroke_rgb=np.where(label[:, None] > 0, 1.0, 0.0).astype(np.float32) * np.ones((n, 3), np.float32),
        fill_rgb=nan3, width=np.where(label > 0, 0.5, 0.2).astype(np.float32), stroke_opacity=np.ones(n, np.float32),
        fill_opacity=np.ones(n, np.float32), close_path=np.zeros(n, np.uint8), even_odd=np.zeros(n, np.uint8),
        dashed=np.zeros(n, np.uint8), seqno=np.arange(n), op_counts=np.tile([1, 0, 0, 0], (n, 1)).astype(np.int32),
        bbox=np.stack([seg[:, 0], np.minimum(seg[:, 1], seg[:, 3]), seg[:, 2], np.maximum(seg[:, 1], seg[:, 3])], 1))


def _synthetic(tmp_path: Path):
    data = tmp_path / "e0"
    data.mkdir()
    records, physical = [], []
    for i in range(4):
        stem = f"doc{i:06d}_p0001_gt{i}"
        _write_page(data, stem, i)
        records.append({"stem": stem, "status": "ok", "npz": f"{stem}.npz", "mode": "fire",
                        "stats": {"true_length_pt": 100.0 + i}})
        physical.append({"document_sha256": f"sha{i}", "page_number": 1, "filename": f"f{i}.pdf", "mode": "fire",
                         "gt_variants": [stem], "duplicate_gt": False})
    manifest = {"records": records, "physical_pages": physical}
    (data / "manifest.json").write_text(json.dumps(manifest))
    return data, manifest


def test_features_shape_and_finite(tmp_path):
    data, _ = _synthetic(tmp_path)
    t = page_tokens(data / "doc000000_p0001_gt0.npz")
    assert t.x.shape == (300, N_FEATURES) and np.isfinite(t.x).all() and t.keep.all()


def test_folds_group_by_pdf_and_skip_unresolved_duplicates():
    manifest = {
        "records": [{"stem": s, "status": "ok", "stats": {"true_length_pt": 1.0}} for s in ("a1", "a2", "b", "c")],
        "physical_pages": [
            {"document_sha256": "A", "page_number": 1, "filename": "a.pdf", "mode": "hvac", "gt_variants": ["a1", "a2"],
             "duplicate_gt": True},
            {"document_sha256": "B", "page_number": 1, "filename": "b.pdf", "mode": "fire", "gt_variants": ["b"], "duplicate_gt": False},
            {"document_sha256": "B", "page_number": 2, "filename": "b.pdf", "mode": "fire", "gt_variants": ["c"], "duplicate_gt": False},
        ]}
    f = propose(manifest, k=2)
    assert f["status"] == "proposal" and len(f["groups"]) == 2  # one group per PDF
    entries, warn = resolve(manifest, f)
    assert [e[0] for e in entries] == ["b", "c"] and len({e[1] for e in entries}) == 1 and warn
    f["duplicate_choice"] = {"A:1": "a2"}
    entries, _ = resolve(manifest, f)
    assert [e[0] for e in entries] == ["a2", "b", "c"]


def test_train_loop_runs_and_learns(tmp_path):
    from openstroke.train import main_train

    data, manifest = _synthetic(tmp_path)
    folds = propose(manifest, k=2)
    (data / "folds.json").write_text(json.dumps(folds))
    args = Namespace(data=str(data), folds=str(data / "folds.json"), fold="0", modes="both", out=str(tmp_path / "run"),
                     cache=None, epochs=4, steps=10, batch=2, window=128, dim=32, depth=1, heads=4, k=8,
                     no_layer_pool=False, lr=3e-3, weight_decay=0.0, warmup=1, pos_weight=2.0, alpha=0.5,
                     eval_every=2, seed=0, device="cpu")
    assert main_train(args) == 0
    s = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert s["at_fold_threshold"]["all"]["f2"] > 0.8  # style separates the classes perfectly
    assert (tmp_path / "run" / "fold0" / "best.pt").exists()
    assert "WARNING: fold file is not confirmed" in (tmp_path / "run" / "train.log").read_text()


def test_yolo_style_arguments():
    from openstroke.cli import yolo_args

    assert yolo_args(["data=/content/e0", "epochs=30", "layer_pool=false", "exist_ok=true", "--batch", "4"]) == [
        "--data", "/content/e0", "--epochs", "30", "--no-layer-pool", "--exist-ok", "--batch", "4"]
    assert yolo_args(["layer_pool=true", "model=runs/train"]) == ["--model", "runs/train"]
    with pytest.raises(SystemExit):
        yolo_args(["layer_pool=maybe"])


def test_finetune_from_run_folder_uses_each_folds_own_checkpoint(tmp_path):
    from openstroke.train import init_checkpoint

    for f in (0, 1):
        (tmp_path / "run" / f"fold{f}").mkdir(parents=True)
        (tmp_path / "run" / f"fold{f}" / "best.pt").write_bytes(b"x")
    assert init_checkpoint("openstroke1", 0) is None
    assert init_checkpoint(str(tmp_path / "run"), 1) == tmp_path / "run" / "fold1" / "best.pt"
    with pytest.raises(FileNotFoundError):
        init_checkpoint(str(tmp_path / "run"), 2)
    with pytest.raises(ValueError):
        init_checkpoint("openstroke9", 0)


def test_train_cli_yolo_style_and_finetune(tmp_path):
    from openstroke.cli import main

    data, manifest = _synthetic(tmp_path)
    (data / "folds.json").write_text(json.dumps(propose(manifest, k=2)))
    common = [f"data={data}", "fold=0", "epochs=1", "steps=2", "batch=2", "window=128", "dim=32", "depth=1", "k=8",
              "eval_every=1", "device=cpu", f"cache={tmp_path / 'cache'}"]
    assert main(["train", *common, f"out={tmp_path / 'runs' / 'train'}"]) == 0
    assert main(["train", *common, f"out={tmp_path / 'runs' / 'train'}", f"model={tmp_path / 'runs' / 'train'}"]) == 0
    log = (tmp_path / "runs" / "train2" / "train.log").read_text()  # second run auto-numbered
    assert "init from" in log and "fold0" in log
