"""OpenStroke28: the stroke model without layers (no_layers) and the GT / Viewer page share (viewer_share)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_model import _synthetic  # noqa: E402

from openstroke.data import load_page, sampling_prob  # noqa: E402
from openstroke.features import FEATURE_NAMES, page_tokens  # noqa: E402


def test_no_layers_page_sees_one_unnamed_layer(tmp_path):
    data, _ = _synthetic(tmp_path)
    npz = data / "doc000000_p0001_gt0.npz"
    with_l, without = page_tokens(npz), page_tokens(npz, no_layers=True)
    assert len(np.unique(with_l.layer)) == 2 and (without.layer == 0).all()
    i = FEATURE_NAMES.index
    assert (without.x[:, i("page_has_layers")] == 0).all() and (without.x[:, i("layer_shape_share")] == 1).all()
    geo = [i(n) for n in ("log_len", "stroke_r", "log_width", "nb10_log")]
    assert np.array_equal(with_l.x[:, geo], without.x[:, geo])          # drawing commands and styles unchanged
    p = load_page(data, "doc000000_p0001_gt0", "fire", tmp_path / "cache", no_layers=True)
    q = load_page(data, "doc000000_p0001_gt0", "fire", tmp_path / "cache")
    assert (p.layer == 0).all() and len(np.unique(q.layer)) == 2       # separate cache files
    assert {f.name.count(".nl.") for f in (tmp_path / "cache").iterdir()} == {0, 1}


def test_viewer_share_splits_the_windows():
    sizes = np.array([100.0, 400.0, 900.0, 100.0, 100.0])
    stems = ["doc1", "doc2", "vw_a", "vw_b", "vw_c"]
    p = sampling_prob(sizes, ["hvac"] * 5, None, stems, 0.65)
    assert np.isclose(p[2:].sum(), 0.65) and np.isclose(p[:2].sum(), 0.35)
    assert np.isclose(p[1] / p[0], 2.0) and np.isclose(p[2] / p[3], 3.0)   # sqrt(size) kept inside each part
    assert np.allclose(sampling_prob(sizes, ["hvac"] * 5), np.sqrt(sizes) / np.sqrt(sizes).sum())


def test_checkpoint_config_with_no_layers_builds():
    from openstroke.models import build, from_config
    m = build("openstroke2", len(FEATURE_NAMES), dims=(16, 32, 32), enc_depths=(1, 1, 1), dec_depths=(1, 1), patch=16)
    cfg = {**m.config, "no_layers": True}
    assert from_config(json.loads(json.dumps(cfg))) is not None


def test_trainer_no_layers_reaches_the_checkpoint_and_predictor_ignores_layers(tmp_path):
    from test_e0 import line, shape
    from test_v2 import _cli

    from openstroke.infer import Predictor

    assert _cli(tmp_path, "model=openstroke2", "epochs=1", "tta=false", "no_layers=true", "viewer_share=0.5") == 0
    run = tmp_path / "runs" / "train"
    assert "no_layers=True" in (run / "train.log").read_text(encoding="utf-8")
    pred = Predictor(run / "fold0" / "best.pt", device="cpu")
    assert pred.config["no_layers"]
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 500, (120, 4))
    a = [shape(line(*v), layer=f"L{i % 3}") for i, v in enumerate(xy)]
    b = [shape(line(*v), layer="WRONG" if i % 2 else "0") for i, v in enumerate(xy)]
    meta = {"page_width": 500.0, "page_height": 500.0}
    pa = pred.predict_payload({"shapes": a, "document_metadata": meta}, "hvac")["prob"]
    pb = pred.predict_payload({"shapes": b, "document_metadata": meta}, "hvac")["prob"]
    assert np.allclose(pa, pb)          # the layer a stroke was drawn on does not change its score


def test_scores_every_page_and_val_matches_the_run(tmp_path):
    """F149: `openstroke scores` scores train and val pages; its val pages reproduce the run's own val_scores.npz."""
    from test_v2 import _cli

    from openstroke.cli import main
    from openstroke.score_all import read_meta

    assert _cli(tmp_path, "model=openstroke2", "epochs=1", "tta=false", "no_layers=true") == 0
    fold = tmp_path / "runs" / "train" / "fold0"
    data = next(p.parent for p in tmp_path.rglob("manifest.json"))
    assert main(["scores", "--model", str(fold / "best.pt"), "--data", str(data), "--cache", str(tmp_path / "runs" / "cache"),
                 "--batch", "2"]) == 0
    z, v = np.load(fold / "all_scores.npz"), np.load(fold / "val_scores.npz")
    meta = read_meta(z)
    splits = {s: r["split"] for s, r in meta["pages"].items()}
    assert set(splits.values()) == {"train", "val"}
    for s, split in splits.items():
        assert len(z[s]) == len(z[f"{s}__index"])
        if split == "val":
            assert np.array_equal(z[f"{s}__index"], v[f"{s}__index"])
            assert np.allclose(z[s].astype(np.float32), v[s], atol=2e-3)


def test_class_head_reaches_the_predictor_and_scores(tmp_path):
    """OpenStroke29: a checkpoint with the per-stroke class head gives oprob (n, classes) in the app and __oprob in `openstroke scores`."""
    from test_e0 import line, shape
    from test_v2 import _cli

    from openstroke.cli import main
    from openstroke.infer import Predictor
    from openstroke.objects import N_CLASSES

    assert _cli(tmp_path, "model=openstroke2", "epochs=1", "tta=false", "no_layers=true", "obj_head=true") == 0
    fold = tmp_path / "runs" / "train" / "fold0"
    pred = Predictor(fold / "best.pt", device="cpu")
    xy = np.random.default_rng(0).uniform(0, 500, (60, 4))
    out = pred.predict_payload({"shapes": [shape(line(*v)) for v in xy], "document_metadata": {"page_width": 500.0, "page_height": 500.0}}, "hvac")
    assert out["oprob"].shape == (60, N_CLASSES)
    data = next(p.parent for p in tmp_path.rglob("manifest.json"))
    assert main(["scores", "--model", str(fold / "best.pt"), "--data", str(data), "--cache", str(tmp_path / "runs" / "cache")]) == 0
    z = np.load(fold / "all_scores.npz")
    stems = [k for k in z.files if not k.startswith("__") and "__" not in k]
    assert all(z[f"{s}__oprob"].shape == (len(z[s]), N_CLASSES) for s in stems)


def test_a_run_started_from_a_checkpoint_takes_its_architecture_and_recipe(tmp_path):
    """User 2026-11-16 (OpenOBB-style CLI): model=<best.pt> + data / epochs only -> the network and the training recipe of that run."""
    from test_v2 import _cli

    from openstroke.cli import main

    assert _cli(tmp_path, "model=openstroke2", "epochs=1", "tta=false", "no_layers=true", "viewer_share=0.5", "restyle_mode=order") == 0
    first = tmp_path / "runs" / "train"
    data = next(p.parent for p in tmp_path.rglob("manifest.json"))
    assert main(["train", f"model={first / 'fold0' / 'best.pt'}", f"data={data}", "epochs=1", "steps=2", "workers=0", "device=cpu",
                 "fold=0", "obj_head=true", f"project={tmp_path / 'runs'}", "name=second"]) == 0
    second = tmp_path / "runs" / "second"
    a = json.loads((second / "args.json").read_text(encoding="utf-8"))
    assert a["no_layers"] is True and a["viewer_share"] == 0.5 and a["restyle_mode"] == "order" and a["window"] == 128
    assert a["obj_head"] is True and a["epochs"] == 1
    log = (second / "train.log").read_text(encoding="utf-8")
    assert "start   " in log and "weights of openstroke2 + no_layers" in log and "(new: obj_head)" in log
    assert "trains openstroke2 + no_layers + obj_head" in log   # user 2026-11-18: the log names the run's network, not the source's
    assert "viewer_share=0.5" in log


def test_research_versions_name_their_networks():
    """User 2026-11-17: one architecture name per research card (as OpenOBB): openstroke28 (E114), openstroke29 (E117), openstroke30 (E120).
    Checkpoints saved as "openstroke2" are named by their switches; a new run saves its version in config["arch"] and it rebuilds."""
    from openstroke.models import ARCHS, build, from_config
    from openstroke.train import ALIASES, PRESETS, arch_name, describe_arch

    old = {"arch": "openstroke2", **ARCHS["openstroke28"]}
    assert describe_arch(old) == "openstroke28" and arch_name(old) == "openstroke28"
    assert describe_arch({**old, "obj_head": True}) == "openstroke29"
    assert describe_arch({**old, "rel_attn": 2, "edge_head": True}) == "openstroke30"
    assert describe_arch({**old, "rel_attn": 2, "edge_head": True, "obj_head": True}) == "openstroke30 + obj_head"
    assert describe_arch({"arch": "openstroke30", **ARCHS["openstroke30"]}) == "openstroke30"
    assert describe_arch({"arch": "openstroke2", "image_branch": True, "img_lum": True, "img_fpn": True}) == "openstroke2 + image_branch + img_fpn + img_lum"   # reads layers: not 28
    assert describe_arch({"arch": "openstroke2"}) == "openstroke2"
    assert ALIASES["openstroke-v1"] == "openstroke28" and set(PRESETS) == set(ARCHS)
    tiny = dict(dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32, img_pretrained=False)
    m = build("openstroke30", 47, **tiny)
    assert m.config["arch"] == "openstroke30" and len(m.gat) == 2 and m.edge_mlp is not None and m.img is not None
    m2 = from_config(dict(m.config, n_features=47), weights_follow=True)
    assert type(m2) is type(m) and len(m2.gat) == 2


def test_class_probabilities_come_from_the_same_pass_as_the_stroke_scores(tmp_path):
    """Speed (user 2026-11-16): predict_page returns the class head's probabilities from its own forward pass; they match the separate
    one-view pass (predict_objects) when tta=1, and the stroke scores do not change."""
    from test_v2 import _cli

    import torch

    from openstroke.data import load_page
    from openstroke.engine.trainer import predict_objects, predict_page
    from openstroke.models import from_config, load_weights

    assert _cli(tmp_path, "model=openstroke2", "epochs=1", "tta=false", "obj_head=true") == 0
    ck = torch.load(tmp_path / "runs" / "train" / "fold0" / "best.pt", map_location="cpu", weights_only=False)
    m = from_config(ck["config"], weights_follow=True)
    load_weights(m, ck["model"])
    data = next(p.parent for p in tmp_path.rglob("manifest.json"))
    page = load_page(data, "doc000000_p0001_gt0", "fire", tmp_path / "c")
    dev = torch.device("cpu")
    a = predict_page(m, page, 128, dev, 2)
    out = []
    b = predict_page(m, page, 128, dev, 2, obj_out=out)
    sep, off = predict_objects(m, page, 128, dev, 2)
    from openstroke.objects import N_CLASSES
    assert np.allclose(a, b) and np.allclose(out[0][:, :N_CLASSES], sep, atol=1e-5) and np.allclose(out[0][:, N_CLASSES:], off, atol=1e-3)


def test_feature_cache_is_built_in_parallel_with_the_same_files(tmp_path):
    from openstroke.data import load_page
    from openstroke.train import build_cache

    data, _ = _synthetic(tmp_path)
    entries = [(p.stem, 0, "fire") for p in data.glob("*.npz")]
    logs = []
    build_cache(data, entries, tmp_path / "par", False, True, logs.append, workers=2)
    build_cache(data, entries, tmp_path / "seq", False, True, logs.append, workers=1)
    for s, _, _ in entries:
        a = load_page(data, s, "fire", tmp_path / "par", no_layers=True)
        b = load_page(data, s, "fire", tmp_path / "seq", no_layers=True)
        assert np.array_equal(a.x, b.x) and np.array_equal(a.index, b.index)
    assert any("pages to compute" in l for l in logs)


def test_class_accuracy_merges_square_to_round_variants_and_weights_by_length():
    from types import SimpleNamespace

    from openstroke.engine.trainer import class_accuracy
    from openstroke.objects import HVAC_CLASSES, N_CLASSES

    duct, red, red_sq = (HVAC_CLASSES.index(n) for n in ("Duct", "Concentric Reducer", "Reducer vuông tròn"))
    p = SimpleNamespace(label=np.array([1, 1, 1, 0]), ocls=np.array([duct, red_sq, duct, 0]), length=np.array([3.0, 1.0, 1.0, 9.0]))
    pc = np.zeros((4, N_CLASSES))
    pc[0, duct] = pc[1, red] = pc[2, red] = pc[3, duct] = 1.0
    acc, per = class_accuracy([p], [pc])
    assert abs(acc - 4 / 5) < 1e-9 and per["Duct"] == 0.75 and per["Concentric Reducer"] == 1.0


def test_checkpoints_named_by_research_version_or_product_prefix_load_and_train_on(tmp_path):
    """A checkpoint saved with arch "openstroke30" (or the product copy's "hicasstroke30") rebuilds, and `model=<that .pt>` resolves the
    backbone's defaults (finalize used to look DEFAULTS up by the research name)."""
    import argparse

    import torch

    from openstroke.models import ARCHS, base_arch, build, canonical_arch, from_config
    from openstroke.train import add_args, describe_arch, finalize

    assert canonical_arch("hicasstroke30") == "openstroke30" and canonical_arch(None) == "openstroke1"
    assert base_arch("openstroke30") == "openstroke2" and base_arch("hicasstroke28") == "openstroke2" and base_arch("openstroke4") == "openstroke4"
    tiny = dict(dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32, img_pretrained=False)
    m = build("hicasstroke30", 47, **tiny)
    assert m.config["arch"] == "openstroke30" and len(m.gat) == 2
    for prefix in ("openstroke30", "hicasstroke30"):
        cfg = dict(m.config, arch=prefix, n_features=47)
        assert describe_arch(cfg) == "openstroke30"
        assert type(from_config(cfg, weights_follow=True)) is type(m)
        ck = tmp_path / f"{prefix}.pt"
        torch.save({"config": cfg, "model": m.state_dict(), "args": {"lr0": 0.001, "closure": 0.5}}, ck)
        ap = argparse.ArgumentParser()
        add_args(ap)
        args = ap.parse_args(["--model", str(ck), "--data", str(tmp_path)])
        args._explicit = ["model", "data"]
        finalize(args)
        assert args.arch == "openstroke2" and args.rel_attn == 2 and args.edge_head is True and args.closure == 0.5 and args.window == 65536
    assert set(ARCHS) == {"openstroke28", "openstroke29", "openstroke30", "openstroke31", "openstroke34", "openstroke35", "openstroke36",
                          "openstroke37"}
