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
