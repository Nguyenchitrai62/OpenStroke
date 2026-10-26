"""OpenStroke v2, data pipeline (windows, augmentation, U-Net levels) and the YOLO-style engine (CPU, tiny)."""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from test_model import _synthetic  # noqa: E402

from openstroke.data import (FILL, HAS_FILL, HAS_STROKE, LOG_WIDTH, STROKE, Augment, augment, collate, dihedral,  # noqa: E402
                             eval_windows, load_page, restyle, window)
from openstroke.features import N_FEATURES, shape_points  # noqa: E402
from openstroke.models import build, from_config, load_weights  # noqa: E402
from openstroke.models.openstroke2 import OpenStroke2, loss_v2, lovasz_hinge, lovasz_sigmoid  # noqa: E402
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


def test_restyle_keeps_style_groups_and_drops_absolute_style():
    x = np.zeros((7, N_FEATURES), np.float32)
    red, blue = [1, 0, 0], [0, 0, 1]
    for i, c in [(0, red), (1, red), (2, red), (3, blue), (4, blue)]:
        x[i, STROKE], x[i, HAS_STROKE] = c, 1
    x[5, FILL], x[5, HAS_FILL] = red, 1  # same colour as the red strokes, as a fill
    x[:, LOG_WIDTH] = np.log1p(np.array([0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.5]) * 10)
    before = x.copy()
    restyle(x, np.random.default_rng(3))
    s = x[:, STROKE]
    assert np.allclose(s[0], s[1]) and np.allclose(s[0], s[2]) and np.allclose(s[3], s[4])
    assert np.allclose(x[5, FILL], s[0])  # one map for stroke and fill colours
    assert not np.allclose(s[0], s[3]) and not np.allclose(s[0], red)
    assert np.allclose(x[6], before[6]) or np.allclose(x[6, LOG_WIDTH], x[2, LOG_WIDTH])  # no colour: only width moves
    assert np.allclose(x[6, STROKE], 0) and np.allclose(x[:5, FILL], 0)
    w = np.expm1(x[:, LOG_WIDTH])
    assert np.isclose(w[0], w[1]) and np.isclose(w[0], w[3]) and np.isclose(w[2], w[4]) and np.isclose(w[2], w[6])
    assert ((s >= 0) & (s <= 1)).all() and np.isfinite(x).all()


def test_augment_restyle_runs_on_a_window(tmp_path):
    page, _, _ = _page(tmp_path)
    w = window(page, 0, 300)
    a = augment({k: v.copy() for k, v in w.items()}, Augment(fliplr=0.0, rot90=False, scale=0.0, color=0.0, gray=0.0, restyle=1.0, drop_tokens=0.0), np.random.default_rng(0))
    assert a["x"].shape == w["x"].shape and np.isfinite(a["x"]).all()
    keep = [j for j in range(N_FEATURES) if j not in STROKE + FILL + [LOG_WIDTH]]
    assert np.allclose(a["x"][:, keep], w["x"][:, keep])  # only style columns move


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


def test_style_groups_follow_colour_width_and_kind(tmp_path):
    from openstroke.data import style_ids

    page, _, _ = _page(tmp_path)
    x = page.x.copy()
    g = style_ids(x)
    assert g.shape == (len(x),) and g.min() == 0
    x[:5, STROKE] = [0.123, 0.456, 0.789]  # a new colour = a new group, the same for those five
    g2 = style_ids(x)
    assert len(set(g2[:5])) == 1 and g2[0] not in set(g2[5:])
    b = collate([window(page, 0, 200)])
    assert int(b["style"][0].max()) + 1 == len(np.unique(page.style[:200]))


def test_v2_group_context_starts_as_the_model_without_it(tmp_path):
    page, _, _ = _page(tmp_path)
    b = collate([window(page, 0, 200)])
    base, withg = _model(), _model(group_context=True)
    withg.load_state_dict(base.state_dict(), strict=False)
    for m in (base, withg):
        m.set_normalization(page.x.mean(0), page.x.std(0))
        m.eval()
    with torch.no_grad():
        assert torch.allclose(base(b), withg(b), atol=1e-6)
        for gc in (withg.gc_layer, withg.gc_style):
            torch.nn.init.normal_(gc.out.weight, std=0.1)
        y = withg(b)
        assert not torch.allclose(base(b), y) and torch.isfinite(y).all()
        # padding invariance with the group attention active
        w = window(page, 0, 120)
        a = withg(collate([w]))[0, :120]
        padded = collate([w, window(page, 0, 300)])
        c = withg({k: v[:1] for k, v in padded.items()})[0, :120]
    assert torch.allclose(a, c, atol=1e-4)


def test_link_graph_kinds():
    from openstroke.graph import all_edges

    seg = np.array([[0, 0, 10, 0],      # shape 0
                    [10, 0, 20, 0],     # shape 1: touches 0 end to end
                    [24, 0, 30, 0],     # shape 2: collinear dash 4 pt after 1
                    [15, 0, 15, 10],    # shape 3: tee into the middle of 1
                    [50, 50, 60, 60],   # shape 4: alone
                    [26, 3, 26, 9]], float)  # shape 5: near 2 but neither touching nor collinear
    e, k = all_edges(seg, np.arange(6))
    got = {tuple(x): int(b) for x, b in zip(e.tolist(), k.tolist())}
    assert got[(0, 1)] & 1 and got[(1, 2)] & 2 and got[(1, 3)] & 4
    assert not any(4 in pair for pair in got) and (2, 5) not in got


def test_collate_maps_links_to_window_tokens(tmp_path):
    from openstroke.data import load_page

    data, _ = _synthetic(tmp_path)
    page = load_page(data, "doc000000_p0001_gt0", "fire", tmp_path / "cache", graph=True)
    assert page.edges is not None and len(page.edges) > 0
    w = window(page, 20, 200)
    a = augment({k: v.copy() for k, v in w.items()}, Augment(drop_tokens=0.4, fliplr=0, rot90=False, scale=0),
                np.random.default_rng(0))
    b = collate([a])
    pos = b["pos"][0].numpy()
    page_pairs = {tuple(sorted(x)) for x in page.edges.tolist()}
    loc = b["edges"][0][b["emask"][0]].numpy()
    assert len(loc) > 0
    assert all(tuple(sorted((int(pos[i]), int(pos[j])))) in page_pairs for i, j in loc)
    kept = set(pos[pos >= 0].tolist())
    want = sum(1 for i, j in page.edges.tolist() if i in kept and j in kept)
    assert len(loc) == want  # every link between two kept tokens survives


def test_v3_relations_start_as_the_model_without_them(tmp_path):
    from openstroke.data import load_page

    data, _ = _synthetic(tmp_path)
    page = load_page(data, "doc000000_p0001_gt0", "fire", tmp_path / "cache", graph=True)
    b = collate([window(page, 0, 200)])
    base, withr = _model(), _model(relations=2)
    withr.load_state_dict(base.state_dict(), strict=False)
    for m in (base, withr):
        m.set_normalization(page.x.mean(0), page.x.std(0))
        m.eval()
    with torch.no_grad():
        assert torch.allclose(base(b), withr(b), atol=1e-6)
        for rb in withr.rel:
            torch.nn.init.normal_(rb.out.weight, std=0.1)
        y = withr(b)
        assert not torch.allclose(base(b), y) and torch.isfinite(y).all()
        w = window(page, 0, 120)
        a1 = withr(collate([w]))[0, :120]
        padded = collate([w, window(page, 0, 300)])
        a2 = withr({k: v[:1] for k, v in padded.items()})[0, :120]
    assert torch.allclose(a1, a2, atol=1e-4)


def test_v2_logits_depend_on_the_shape_at_init(tmp_path):
    """Regression (Colab 2026-10-20): default init made every shape's logit the same (std 0.006), BCE stayed at ln 2."""
    page, _, _ = _page(tmp_path)
    m = _model()
    m.set_normalization(page.x.mean(0), page.x.std(0))
    m.eval()
    b = collate([window(page, 0, 300)])
    with torch.no_grad():
        out = m(b)[0][b["mask0"][0]]
    assert float(out.std()) > 0.1


def test_plumbing_share_caps_the_plumbing_pages():
    from openstroke.data import sampling_prob

    sizes = np.array([100_000] * 3 + [25_000] * 30)
    modes = ["fire", "hvac", "hvac"] + ["plumbing"] * 30
    natural = sampling_prob(sizes, modes)
    assert natural[3:].sum() > 0.7  # 30 small pages of one project would take most windows
    p = sampling_prob(sizes, modes, plumbing_share=0.3)
    assert np.isclose(p[3:].sum(), 0.3) and np.isclose(p.sum(), 1)
    assert np.allclose(p[:3] / p[0], natural[:3] / natural[0])  # FIRE / HVAC keep their relative weights
    assert np.allclose(sampling_prob(sizes[:3], modes[:3], 0.3), natural[:3] / natural[:3].sum())


def test_v2_mode_input_starts_as_the_model_without_it(tmp_path):
    page, _, _ = _page(tmp_path)
    b = collate([window(page, 0, 200)])
    assert b["mode"].tolist() == [0]  # fire
    base, withm = _model(), _model(mode_input=True)
    withm.load_state_dict(base.state_dict(), strict=False)
    for m in (base, withm):
        m.set_normalization(page.x.mean(0), page.x.std(0))
        m.eval()
    with torch.no_grad():
        assert torch.allclose(base(b), withm(b), atol=1e-6)
        withm.mode_embed.weight.normal_()
        b2 = dict(b, mode=torch.tensor([2]))
        assert not torch.allclose(withm(b), withm(b2))


def test_layer_name_tokens_share_discipline_words():
    from openstroke.layernames import MAX_TOKENS, layer_table, token_ids, tokens

    assert "w:acmv" in tokens("360D-ACMV-B.SA") and "w:thoat" in tokens("Xref-Block A|THOÁT NƯỚC")
    a, b = set(token_ids("INNO-ACMV-B.EA")), set(token_ids("360D-ACMV-B.SA"))
    assert a & b and not (set(token_ids("A-WALL")) & {0})
    t = layer_table(["", "M-DUCT", "M-DUCT"])
    assert t.shape == (3, MAX_TOKENS) and (t[0] == 0).all() and (t[1] == t[2]).all() and (t[1] > 0).any()


def test_collate_carries_the_layer_names_of_each_window(tmp_path):
    page, _, _ = _page(tmp_path)
    assert page.ltab is not None and page.ltab.shape[0] >= int(page.layer.max()) + 1
    w = window(page, 0, 200)
    b = collate([w])
    u = np.unique(w["layer"])
    assert b["ltok"].shape[1] == len(u)
    assert np.array_equal(b["ltok"][0].numpy(), page.ltab[u])
    hidden = augment({k: v.copy() for k, v in w.items()}, Augment(name_drop=1.0, drop_tokens=0.5), np.random.default_rng(0))
    assert not hidden["ltab"].any() and len(hidden["x"]) < 200


def test_v2_layer_names_start_as_the_model_without_them(tmp_path):
    page, _, _ = _page(tmp_path)
    b = collate([window(page, 0, 200)])
    base, withn = _model(), _model(layer_names=True)
    withn.load_state_dict(base.state_dict(), strict=False)
    for m in (base, withn):
        m.set_normalization(page.x.mean(0), page.x.std(0))
        m.eval()
    with torch.no_grad():
        assert torch.allclose(base(b), withn(b), atol=1e-6)
        withn.name_proj.weight.normal_()
        other = dict(b, ltok=torch.zeros_like(b["ltok"]))
        assert not torch.allclose(withn(b), withn(other))


def test_lovasz_is_zero_for_a_confident_correct_prediction():
    y = torch.tensor([1.0, 0, 1, 0])
    assert float(lovasz_hinge(torch.tensor([5.0, -5, 5, -5]), y)) == 0.0
    assert float(lovasz_hinge(torch.tensor([-5.0, 5, -5, 5]), y)) > 1.0
    assert float(lovasz_sigmoid(torch.tensor([9.0, -9, 9, -9]), y)) < 1e-3
    assert 0.9 < float(lovasz_sigmoid(torch.tensor([-9.0, 9, -9, 9]), y)) <= 1.0  # bounded, unlike the hinge


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


def test_engine_parallel_folds_match_the_serial_run(tmp_path, capsys):
    assert _cli(tmp_path, "model=openstroke2", "epochs=1", "tta=false", "fold=all", "parallel=2", "seed=3") == 0
    run = tmp_path / "runs" / "train"
    out = capsys.readouterr().out
    assert "parallel 2 fold processes" in out and "Cross-validation summary" in out
    for f in (0, 1):
        assert (run / f"fold{f}" / "result.json").exists() and (run / f"fold{f}" / "train.log").exists()
    s = json.loads((run / "summary.json").read_text())
    assert "all" in s["at_fold_threshold"]
    par = json.loads((run / "fold1" / "result.json").read_text())["val_at_threshold"]
    assert _cli(tmp_path, "model=openstroke2", "epochs=1", "tta=false", "fold=1", "seed=3") == 0  # same fold, serial
    ser = json.loads((tmp_path / "runs" / "train2" / "fold1" / "result.json").read_text())["val_at_threshold"]
    assert abs(par["f2"] - ser["f2"]) < 1e-6


def test_parallel_resume_of_a_run_started_by_an_older_version(tmp_path, capsys):
    assert _cli(tmp_path, "model=openstroke2", "epochs=1", "tta=false", "fold=all") == 0
    run = tmp_path / "runs" / "train"
    args = json.loads((run / "args.json").read_text())
    for k in ("parallel", "child", "mode_input", "plumbing_share", "restyle"):
        args.pop(k, None)  # options that did not exist when the run started
    args["epochs"] = 2
    (run / "args.json").write_text(json.dumps(args))
    (run / "fold1" / "result.json").unlink()  # fold 2 unfinished
    assert _cli(tmp_path, f"resume={run}", "parallel=2") == 0
    out = capsys.readouterr().out
    assert "fold 1: done earlier" in (run / "fold0" / "console.log").read_text() and "FAILED" not in out
    assert "resume  fold 1 from epoch 2" in (run / "fold1" / "train.log").read_text()
    assert len((run / "fold1" / "results.csv").read_text().splitlines()) == 3


def test_predictor_scores_a_payload_with_a_trained_checkpoint(tmp_path):
    from test_e0 import line, shape

    from openstroke.infer import Predictor

    assert _cli(tmp_path, "model=openstroke2", "epochs=1", "tta=false") == 0
    ck = tmp_path / "runs" / "train" / "fold0" / "best.pt"
    rng = np.random.default_rng(0)
    shapes = [shape(line(*rng.uniform(0, 500, 4)), layer=f"L{i % 3}") for i in range(150)]
    shapes.append({"items": [], "type": "s", "layer": "L0"})  # no geometry: never scored
    payload = {"shapes": shapes, "document_metadata": {"page_width": 500.0, "page_height": 500.0}}
    pred = Predictor(ck, device="cpu")
    assert pred.describe()["arch"] == "openstroke2" and pred.window == 128
    out = pred.predict_payload(payload, mode="fire")
    p = out["prob"]
    assert p.shape == (151,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert p[-1] == 0 and out["kept"] == 150 and p[:150].std() > 0
    assert np.allclose(pred.predict_payload(payload, mode="fire")["prob"], p)  # deterministic


def test_engine_runs_with_mode_and_layer_names_and_the_predictor_reads_them(tmp_path):
    from test_e0 import line, shape

    from openstroke.infer import Predictor

    assert _cli(tmp_path, "model=openstroke2", "epochs=1", "tta=false", "mode_input=true", "layer_names=true",
                "plumbing_share=0.3", "group_context=true", "relations=2") == 0
    run = tmp_path / "runs" / "train"
    assert "layer_names=True" in (run / "train.log").read_text(encoding="utf-8")
    pred = Predictor(run / "fold0" / "best.pt", device="cpu")
    assert pred.config["layer_names"] and pred.config["mode_input"] and pred.config["group_context"]
    assert pred.config["relations"] == 2
    shapes = [shape(line(i, 0, i, 50), layer="M-DUCT" if i % 2 else "A-WALL") for i in range(0, 200, 4)]
    out = pred.predict_payload({"shapes": shapes, "document_metadata": {"page_width": 200.0, "page_height": 60.0}}, "hvac")
    assert out["prob"].shape == (50,) and np.isfinite(out["prob"]).all()


def test_model_is_required_and_openstroke3_is_v2_with_relations(tmp_path, capsys):
    from openstroke.cli import main

    data, manifest = _synthetic(tmp_path)
    from openstroke.folds import propose

    (data / "folds.json").write_text(json.dumps(propose(manifest, k=2)))
    with pytest.raises(SystemExit) as e:
        main(["train", f"data={data}", f"project={tmp_path / 'runs'}", "device=cpu"])
    assert "model= is required" in str(e.value) and "openstroke3" in str(e.value)
    assert _cli(tmp_path, "model=openstroke3", "epochs=1", "tta=false") == 0
    ck = torch.load(tmp_path / "runs" / "train" / "fold0" / "best.pt", weights_only=False)
    assert ck["config"]["arch"] == "openstroke3" and ck["config"]["relations"] == 2
    m = from_config(ck["config"])
    load_weights(m, ck["model"])
    assert len(m.rel) == 2


def test_modes_list_filters_pages(tmp_path, capsys):
    assert _cli(tmp_path, "model=openstroke2", "epochs=1", "tta=false", "modes=fire,hvac") == 0
    with pytest.raises(SystemExit):
        _cli(tmp_path, "model=openstroke2", "epochs=1", "modes=fire,plumb")


def _same_batch(a, b):
    assert set(a) == set(b), set(a) ^ set(b)
    for k in a:
        x, y = a[k].cpu(), b[k].cpu()
        assert x.shape == y.shape and x.dtype == y.dtype, (k, x.shape, y.shape, x.dtype, y.dtype)
        assert torch.allclose(x.double(), y.double(), atol=1e-5) if x.is_floating_point() else torch.equal(x, y), k


def test_gpu_batch_equals_numpy_collate(tmp_path):
    from openstroke.data import load_page
    from openstroke.gpu_batch import build_batch

    data, _ = _synthetic(tmp_path)
    page = load_page(data, "doc000000_p0001_gt0", "fire", tmp_path / "cache", graph=True)
    ws = [window(page, 0, 200), window(page, 50, 150)]
    _same_batch(collate([{k: v.copy() for k, v in w.items()} for w in ws]), build_batch(ws, "cpu"))
    # augmented: same random draws, same order -> the same batch (restyle off: its group order is torch's)
    aug = Augment(restyle=0.0, gray=0.5, color=0.7, drop_tokens=0.3, name_drop=0.5)
    ref = collate([augment({k: v.copy() for k, v in w.items()}, aug, np.random.default_rng(s)) for w, s in zip(ws, (3, 4))])
    _same_batch(ref, build_batch(ws, "cpu", aug=aug, seeds=(3, 4)))


def test_gpu_restyle_keeps_groups(tmp_path):
    from openstroke.data import load_page
    from openstroke.gpu_batch import build_batch

    data, _ = _synthetic(tmp_path)
    page = load_page(data, "doc000000_p0001_gt0", "fire", tmp_path / "cache")
    w = window(page, 0, 300)
    plain = build_batch([w], "cpu")
    b = build_batch([w], "cpu", aug=Augment(fliplr=0, rot90=False, scale=0, color=0, gray=0, restyle=1.0, drop_tokens=0),
                    seeds=(1,))
    a, c = plain["x"][0, :300, STROKE].numpy(), b["x"][0, :300, STROKE].numpy()
    _, ia = np.unique(a.round(4), axis=0, return_inverse=True)
    _, ic = np.unique(c.round(4), axis=0, return_inverse=True)
    assert len(set(zip(ia.ravel(), ic.ravel()))) == len(set(ia.ravel()))  # same groups, new colours


def test_engine_trains_with_gpu_batch_path_on_cpu(tmp_path):
    assert _cli(tmp_path, "model=openstroke3", "epochs=1", "tta=false", "gpu_batch=true", "restyle=1.0") == 0
    log = (tmp_path / "runs" / "train" / "train.log").read_text(encoding="utf-8")
    assert "batches built on the GPU" in log


def test_v4_mixers_padding_and_links_matter(tmp_path):
    from openstroke.data import load_page

    data, _ = _synthetic(tmp_path)
    page = load_page(data, "doc000000_p0001_gt0", "fire", tmp_path / "cache", graph=True)
    torch.manual_seed(0)
    m = OpenStroke2(N_FEATURES, **{**TINY, "mixers": ("graph", "conv", "attn")}).eval()
    m.set_normalization(page.x.mean(0), page.x.std(0))
    assert m.enc[0][0].attn is None and m.enc[0][0].graph is not None and m.enc[1][0].attn is None
    assert m.enc[2][0].attn is not None
    w = window(page, 0, 120)
    with torch.no_grad():
        a = m(collate([w]))[0, :120]
        padded = collate([w, window(page, 0, 300)])
        c = m({k: v[:1] for k, v in padded.items()})[0, :120]
        assert torch.allclose(a, c, atol=1e-4)
        b = collate([w])
        nolinks = dict(b, emask=torch.zeros_like(b["emask"]))
        assert not torch.allclose(m(b), m(nolinks))  # the link graph changes the prediction
    with pytest.raises(AssertionError):
        OpenStroke2(N_FEATURES, **{**TINY, "mixers": ("conv", "graph", "attn")})


def test_engine_runs_openstroke4(tmp_path):
    from openstroke.infer import Predictor
    from test_e0 import line, shape

    assert _cli(tmp_path, "model=openstroke4", "epochs=1", "tta=false", "gpu_batch=true") == 0
    run = tmp_path / "runs" / "train"
    ck = torch.load(run / "fold0" / "best.pt", weights_only=False)
    assert ck["config"]["arch"] == "openstroke4" and ck["config"]["mixers"][0] == "graph"
    pred = Predictor(run / "fold0" / "best.pt", device="cpu")
    shapes = [shape(line(i, 0, i + 4, 0)) for i in range(0, 200, 5)]
    out = pred.predict_payload({"shapes": shapes, "document_metadata": {"page_width": 210.0, "page_height": 10.0}}, "fire")
    assert out["prob"].shape == (40,) and np.isfinite(out["prob"]).all()


def test_feature_drop_hides_features_from_the_model(tmp_path):
    from openstroke.features import FEATURE_NAMES, feature_mask

    page, _, _ = _page(tmp_path)
    idx = feature_mask(["layer_stats"])
    assert [FEATURE_NAMES[i] for i in idx][:2] == ["layer_shape_share", "layer_len_share"] and len(idx) == 6
    torch.manual_seed(0)
    m = OpenStroke2(N_FEATURES, **{**TINY, "feature_drop": ["layer_stats"]}).eval()
    m.set_normalization(page.x.mean(0), page.x.std(0))
    b = collate([window(page, 0, 200)])
    b2 = dict(b, x=b["x"].clone())
    b2["x"][..., idx] = 123.0  # any value: the model must not see it
    with torch.no_grad():
        assert torch.allclose(m(b), m(b2))
        b3 = dict(b, x=b["x"].clone())
        b3["x"][..., FEATURE_NAMES.index("log_width")] += 3.0
        assert not torch.allclose(m(b), m(b3))
    assert _cli(tmp_path, "model=openstroke2", "epochs=1", "tta=false", "feature_drop=layer_stats") == 0
    ck = torch.load(tmp_path / "runs" / "train" / "fold0" / "best.pt", weights_only=False)
    assert ck["config"]["feature_drop"] == ["layer_stats"]


def test_checkpoint_from_before_feature_drop_loads():
    m = OpenStroke2(N_FEATURES, **TINY)
    state = {k: v for k, v in m.state_dict().items() if k != "feat_keep"}
    m2 = OpenStroke2(N_FEATURES, **TINY)
    load_weights(m2, state)
    assert torch.equal(m2.feat_keep, torch.ones(N_FEATURES))


def test_engine_runs_v1(tmp_path):
    assert _cli(tmp_path, "model=openstroke1", "epochs=1", "dim=32", "depth=1", "k=8", "tta=false") == 0
    ck = torch.load(tmp_path / "runs" / "train" / "fold0" / "best.pt", weights_only=False)
    assert ck["config"]["arch"] == "openstroke1"
    m = build("openstroke1", N_FEATURES, dim=32, depth=1, k=8)
    load_weights(m, ck["model"])


def test_v2_masked_pretraining_head_and_finetune(tmp_path, capsys):
    page, _, _ = _page(tmp_path)
    b = collate([window(page, 0, 200)])
    m = _model(pretrain=True)
    plain = _model()
    m.load_state_dict({**plain.state_dict(), **{k: v for k, v in m.state_dict().items() if k not in plain.state_dict()}})
    m.eval()
    plain.eval()
    with torch.no_grad():
        assert torch.allclose(m(b), plain(b), atol=1e-5)  # no ``mae`` key: the classifier, unchanged
        b["mae"] = torch.rand(b["mask0"].shape) < 0.4
        rec = m(b)
    assert rec.shape == (*b["mask0"].shape, N_FEATURES) and m.target_features(b).shape == rec.shape
    # engine: a pretraining run writes best.pt without a scored val, and fine-tuning starts from it
    assert _cli(tmp_path, "model=openstroke2", "epochs=1", "pretrain=0.4", "name=pre") == 0
    out = capsys.readouterr().out
    assert "mse_loss" in out and "no scored validation" in out
    assert _cli(tmp_path, f"model={tmp_path / 'runs' / 'pre'}", "epochs=1", "name=ft") == 0
    assert "init from" in (tmp_path / "runs" / "ft" / "train.log").read_text()


def test_layer_targets_and_layer_head(tmp_path, capsys):
    from openstroke.data import layer_targets

    lay = np.array([0, 0, 1, 1, 2])
    lab = np.array([1, 1, 0, 1, 0])
    L = np.array([1.0, 1.0, 3.0, 1.0, 1.0])
    assert layer_targets(lay, lab, L).tolist() == [1, 1, 0, 0, 0]  # layer 1: 1 / 4 true by length
    page, _, _ = _page(tmp_path)
    b = collate([window(page, 0, 200), window(page, 30, 150)])
    assert b["ylayer"].shape == b["mask0"].shape
    base, two = _model(), _model(layer_head=True)
    common = {k: v for k, v in two.state_dict().items() if k in base.state_dict()}
    assert all(torch.equal(v, base.state_dict()[k]) for k, v in common.items())  # same start as without the head
    base.eval()
    two.eval()
    with torch.no_grad():
        assert torch.allclose(base(b), two(b))
    assert two.last_layer.shape == (2, int(b["layer"].max()) + 1)
    from openstroke.engine.trainer import layer_loss

    two.train()
    two(b)
    ll = layer_loss(two.last_layer, b)
    assert torch.isfinite(ll)
    ll.backward()
    # engine: the head trains, and the final val writes a per-token layer probability next to the shape scores
    assert _cli(tmp_path, "model=openstroke2", "epochs=1", "layer_head=true", "name=lh") == 0
    assert "layer_loss" in capsys.readouterr().out
    z = np.load(tmp_path / "runs" / "lh" / "fold0" / "val_scores.npz")
    assert any(k.endswith("__layer") for k in z.files)


def test_layer_head_names_start_as_e24_and_pretrained_init_gets_new_heads(tmp_path, capsys):
    page, _, _ = _page(tmp_path)
    b = collate([window(page, 0, 200)])
    e24, e27 = _model(layer_head=True), _model(layer_head=True, layer_head_names=True)
    e27.load_state_dict({**e27.state_dict(), **e24.state_dict()})
    e24.eval()
    e27.eval()
    with torch.no_grad():
        e24(b)
        e27(b)
    assert torch.allclose(e24.last_layer, e27.last_layer)  # zero-init name branch: same layer logits at start
    # a checkpoint without the layer head (e.g. masked pretraining) fine-tunes into a two-head model
    assert _cli(tmp_path, "model=openstroke2", "epochs=1", "pretrain=0.4", "name=pre") == 0
    ck = tmp_path / "runs" / "pre" / "fold0" / "best.pt"
    assert _cli(tmp_path, f"model={ck}", "epochs=1", "layer_head=true", "layer_head_names=true", "name=ft") == 0
    assert "layer_loss" in capsys.readouterr().out


def test_render_draws_the_shapes_where_grid_sample_reads_them():
    import torch.nn.functional as F

    from openstroke.models.image_branch import render

    xy0 = torch.tensor([[[0.0, 0.0], [30.0, -20.0]]])  # two shapes, window frame (pt)
    t = torch.linspace(-10, 10, 16)
    pts = torch.stack([torch.stack([t, torch.zeros(16)], -1), torch.stack([torch.zeros(16), t], -1)])[None]  # h / v lines
    ink, scale, S = render(xy0, pts, torch.tensor([[True, True]]))
    assert ink.shape == (1, 1, S, S) and S % 32 == 0 and float(scale) == 1.0
    assert 30 <= float(ink.sum()) <= 50  # ~2 x 20 px of ink
    a = (xy0[:, :, None, :] + pts) * scale[:, None, None, None]
    read = F.grid_sample(ink, (a / (S / 2)).clamp(-1, 1), align_corners=False)  # (1, 1, 2, 16)
    assert float(read.mean()) > 0.3  # the points land on their own ink (bilinear, 1 px lines)


def test_image_branch_starts_as_the_model_without_it(tmp_path):
    page, _, _ = _page(tmp_path)
    b = collate([window(page, 0, 200)])
    base, two = _model(layer_head=True), _model(layer_head=True, image_branch=True, img_pretrained=False)
    two.load_state_dict({**two.state_dict(), **base.state_dict()})
    base.eval()
    two.eval()
    with torch.no_grad():
        assert torch.allclose(base(b), two(b), atol=1e-6)
    two.train()
    two(b).sum().backward()
    assert two.img.proj.weight.grad is not None and float(two.img.proj.weight.grad.abs().sum()) > 0
