"""OpenStroke v2, data pipeline (windows, augmentation, U-Net levels) and the YOLO-style engine (CPU, tiny)."""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from test_model import _synthetic  # noqa: E402

from openstroke.data import Augment, augment, collate, dihedral, eval_windows, load_page, window  # noqa: E402
from openstroke.features import N_FEATURES, shape_points  # noqa: E402
from openstroke.models import build, from_config, load_weights  # noqa: E402
from openstroke.models.openstroke2 import OpenStroke2, loss_v2, lovasz_hinge  # noqa: E402
from openstroke.serialize import hilbert, zorder  # noqa: E402

TINY = dict(dims=(16, 32, 32), enc_depths=(1, 1, 1), dec_depths=(1, 1), patch=16, drop_path=0.1)


def _page(tmp_path):
    data, manifest = _synthetic(tmp_path)
    return load_page(data, "doc000000_p0001_gt0", "fire", tmp_path / "cache"), data, manifest


def test_curves_are_bijections_and_hilbert_is_continuous():
    g = np.array([(x, y) for x in range(16) for y in range(16)])
    for f in (lambda q: hilbert(q, 4), zorder):
        assert sorted(f(g).tolist()) == list(range(256))
    o = np.argsort(hilbert(g, 4))
    assert (np.abs(np.diff(g[o], axis=0)).sum(1) == 1).all()


def test_shape_points_follow_the_arc():
    seg = np.array([[0, 0, 10, 0], [10, 0, 10, 10]], float)
    p = shape_points(seg, np.array([0, 2]), 4)[0]
    assert np.allclose(p, [[-2.5, -5], [2.5, -5], [5, -2.5], [5, 2.5]])


def test_collate_levels_are_consistent(tmp_path):
    page, _, _ = _page(tmp_path)
    ws = [window(page, 0, 200), window(page, 50, 120)]
    b = collate(ws)
    for li in range(4):
        n = b[f"mask{li}"].sum(1)
        perm = b[f"perm{li}"]
        for i in range(2):
            for k in range(4):  # valid tokens first, each exactly once
                assert sorted(perm[i, k, : n[i]].tolist()) == list(range(int(n[i])))
        if li:
            par = b[f"parent{li}"]
            prev = b[f"mask{li - 1}"].sum(1)
            for i in range(2):
                p = par[i, : prev[i]]
                assert int(p.max()) + 1 == int(n[i]) and (torch.diff(p) >= 0).all()  # contiguous cells
    assert sorted(b["pos"][0][b["mask0"][0]].tolist()) == list(range(200))
    assert n.tolist()  # coarser levels are smaller
    assert int(b["mask3"].sum()) < int(b["mask0"].sum())


def test_augment_keeps_tokens_consistent(tmp_path):
    page, _, _ = _page(tmp_path)
    w = window(page, 0, 300)
    rng = np.random.default_rng(1)
    a = augment({k: v.copy() for k, v in w.items()}, Augment(drop_tokens=0.0, scale=0.3), rng)
    assert len(a["x"]) == 300 and np.isfinite(a["x"]).all()
    # quarter turns keep pairwise distances up to the zoom
    d0 = np.hypot(*(w["xy"][1] - w["xy"][0]))
    d1 = np.hypot(*(a["xy"][1] - a["xy"][0]))
    assert 1 / 1.31 < d1 / max(d0, 1e-9) < 1.31
    v = dihedral(w, 5)
    assert np.allclose(np.hypot(*(v["xy"][1] - v["xy"][0])), d0)
    drop = augment({k: v.copy() for k, v in w.items()}, Augment(drop_tokens=0.5), np.random.default_rng(2))
    assert len(drop["x"]) < 300


def _model(**kw):
    torch.manual_seed(0)
    m = OpenStroke2(N_FEATURES, **{**TINY, **kw})
    return m


def test_v2_forward_backward_and_gradients(tmp_path):
    page, _, _ = _page(tmp_path)
    b = collate([window(page, 0, 200), window(page, 30, 150)])
    m = _model()
    m.train()
    logits = m(b)
    assert logits.shape == b["mask0"].shape and torch.isfinite(logits).all()
    total, parts = loss_v2(logits, b)
    assert torch.isfinite(total) and set(parts) == {"bce", "lovasz"}
    total.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing


def test_v2_layerpool_starts_as_identity(tmp_path):
    page, _, _ = _page(tmp_path)
    b = collate([window(page, 0, 200)])
    a = _model(layer_pool=True).eval()
    c = _model(layer_pool=False).eval()
    res = c.load_state_dict(a.state_dict(), strict=False)
    assert not res.missing_keys
    with torch.no_grad():
        assert torch.allclose(a(b), c(b), atol=1e-5)


def test_v2_padding_does_not_change_outputs(tmp_path):
    page, _, _ = _page(tmp_path)
    m = _model().eval()
    for blk in m.enc[0]:  # make LayerPool active so its masking is tested too
        torch.nn.init.normal_(blk.pool.out.weight, std=0.1)
    w = window(page, 0, 120)
    alone = collate([w])
    padded = collate([w, window(page, 0, 300)])
    with torch.no_grad():
        a = m(alone)[0, :120]
        b = m({k: v[:1] for k, v in padded.items()})[0, :120]
    assert torch.allclose(a, b, atol=1e-4)


def test_lovasz_is_zero_for_a_confident_correct_prediction():
    y = torch.tensor([1.0, 0, 1, 0])
    assert float(lovasz_hinge(torch.tensor([5.0, -5, 5, -5]), y)) == 0.0
    assert float(lovasz_hinge(torch.tensor([-5.0, 5, -5, 5]), y)) > 1.0


def test_eval_windows_cover_the_page():
    class P:
        def __len__(self):
            return 1000

    s = eval_windows(P(), 256)
    covered = np.zeros(1000, bool)
    for a in s:
        covered[a:a + 256] = True
    assert covered.all()


def test_v1_checkpoint_of_the_first_trainer_still_loads():
    from openstroke.model import OpenStroke1

    old = OpenStroke1(N_FEATURES, dim=32, depth=1, k=8)
    config = dict(old.config)  # no "arch": written by the first trainer
    m = from_config(config)
    load_weights(m, old.state_dict())
    assert torch.equal(m.net.head.weight, old.head.weight)


def _cli(tmp_path, *extra):
    from openstroke.cli import main

    data, manifest = _synthetic(tmp_path)
    from openstroke.folds import propose

    (data / "folds.json").write_text(json.dumps(propose(manifest, k=2)))
    args = [f"data={data}", "fold=0", "steps=2", "batch=2", "window=128", "val_batch=2", "workers=0", "device=cpu",
            f"project={tmp_path / 'runs'}", "dims=16,32", "enc_depths=1,1", "dec_depths=1", "patch=32", *extra]
    return main(["train", *args])


def test_engine_end_to_end_resume_and_finetune(tmp_path, capsys):
    assert _cli(tmp_path, "model=openstroke2", "epochs=2", "tta=true") == 0
    run = tmp_path / "runs" / "train"
    out = capsys.readouterr().out
    assert "GPU_mem" in out and "bce_loss" in out and "Cross-validation summary" in out
    for f in ("results.csv", "results.png", "best.pt", "last.pt", "result.json", "val_scores.npz"):
        assert (run / "fold0" / f).exists(), f
    res = json.loads((run / "fold0" / "result.json").read_text())
    assert res["tta"] == 8 and 0 <= res["val_at_threshold"]["f2"] <= 1
    rows = (run / "fold0" / "results.csv").read_text().splitlines()
    assert len(rows) == 3 and "F2" in rows[0]
    # resume an unfinished fold: continues from last.pt at epoch 3
    (run / "fold0" / "result.json").unlink()
    args = json.loads((run / "args.json").read_text())
    args["epochs"] = 3
    (run / "args.json").write_text(json.dumps(args))
    assert _cli(tmp_path, f"resume={run}") == 0
    assert "resume  fold 0 from epoch 3" in (run / "train.log").read_text()
    assert len((run / "fold0" / "results.csv").read_text().splitlines()) == 4
    # fine-tune: each fold from its own best.pt, written to a new auto-numbered run
    assert _cli(tmp_path, f"model={run}", "epochs=1") == 0
    log2 = (tmp_path / "runs" / "train2" / "train.log").read_text()
    assert "init from" in log2 and "fold0" in log2


def test_engine_runs_v1(tmp_path):
    assert _cli(tmp_path, "model=openstroke1", "epochs=1", "dim=32", "depth=1", "k=8", "tta=false") == 0
    ck = torch.load(tmp_path / "runs" / "train" / "fold0" / "best.pt", weights_only=False)
    assert ck["config"]["arch"] == "openstroke1"
    m = build("openstroke1", N_FEATURES, dim=32, depth=1, k=8)
    load_weights(m, ck["model"])
