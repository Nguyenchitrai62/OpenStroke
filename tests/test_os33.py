"""OpenStroke33: page-context features can be hidden (feature group "context"), and a feature mask on the command line now applies when a
run starts from a checkpoint (it used to be ignored, F185)."""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")


def test_context_group_names_every_page_dependent_feature():
    from openstroke.features import FEATURE_GROUPS, FEATURE_NAMES, feature_mask

    ctx = FEATURE_GROUPS["context"]
    assert set(ctx) <= set(FEATURE_NAMES) and len(ctx) == 9
    assert {"seqno_rank", "style_len_share", "nb50_same_style", "nn_dist_log"} <= set(ctx)
    assert sorted(feature_mask(["layer_stats", "context"])) == sorted(FEATURE_NAMES.index(n) for n in ctx + FEATURE_GROUPS["layer_stats"])


def test_feature_drop_applies_when_training_from_a_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from test_model import _synthetic

    from openstroke.cli import main
    from openstroke.features import FEATURE_GROUPS, FEATURE_NAMES
    from openstroke.folds import propose

    data, manifest = _synthetic(tmp_path)
    (data / "folds.json").write_text(json.dumps(propose(manifest, k=2)))
    common = [f"data={data}", "fold=0", "steps=2", "batch=2", "window=128", "val_batch=2", "workers=0", "device=cpu", "tta=false",
              f"project={tmp_path / 'runs'}", "epochs=1"]
    assert main(["train", *common, "model=openstroke2", "dims=16,32", "enc_depths=1,1", "dec_depths=1", "patch=32",
                 "feature_drop=layer_stats", "name=a"]) == 0
    first = tmp_path / "runs" / "a" / "fold0" / "best.pt"
    assert main(["train", *common, f"model={first}", "feature_drop=layer_stats,context", "name=b"]) == 0
    ck = torch.load(tmp_path / "runs" / "b" / "fold0" / "best.pt", map_location="cpu", weights_only=False)
    assert set(ck["config"]["feature_drop"]) == {"layer_stats", "context"}
    keep = ck["model"]["feat_keep"].numpy()
    ctx = [FEATURE_NAMES.index(n) for n in FEATURE_GROUPS["context"]]
    assert (keep[ctx] == 0).all() and keep[FEATURE_NAMES.index("log_len")] == 1
    second = tmp_path / "runs" / "b" / "fold0" / "best.pt"
    assert main(["train", *common, f"model={second}", "name=c"]) == 0          # no mask on the command: the checkpoint's stays
    ck = torch.load(tmp_path / "runs" / "c" / "fold0" / "best.pt", map_location="cpu", weights_only=False)
    assert set(ck["config"]["feature_drop"]) == {"layer_stats", "context"} and (ck["model"]["feat_keep"].numpy()[ctx] == 0).all()
