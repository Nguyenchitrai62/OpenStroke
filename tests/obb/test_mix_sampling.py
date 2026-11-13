"""OpenStroke27: GT / Viewer tile mix and a fixed number of tiles per epoch."""
import json

import numpy as np
import pytest

from openstroke.obb import cli
from openstroke.obb.data.dota import mix_weights


def _items(n_gt, n_vw):
    return ([{"name": f"doc{i}_p1__1__0___0", "src": f"doc{i}_p1"} for i in range(n_gt)]
            + [{"name": f"vw_{i}__1__0___0", "src": f"vw_{i}"} for i in range(n_vw)])


def test_mix_weights_shares_and_relative_weights():
    items = _items(3, 9)
    w0 = np.ones(12)
    w0[0] = 2.0                                     # e.g. a repeat factor inside the GT group
    w, n = mix_weights(items, w0, "vw_", 0.7)
    assert n == 9
    assert w[3:].sum() == pytest.approx(0.7) and w[:3].sum() == pytest.approx(0.3)
    assert w[0] / w[1] == pytest.approx(2.0)        # relative weights kept within a group
    assert np.allclose(w[3:], w[3])


def test_mix_weights_needs_both_groups_and_a_share_in_0_1():
    with pytest.raises(SystemExit):
        mix_weights(_items(3, 0), np.ones(3), "vw_", 0.7)
    with pytest.raises(ValueError):
        mix_weights(_items(3, 3), np.ones(6), "vw_", 1.0)


def test_cli_knows_the_mix_keys():
    cli.check_keys({"mix_share": 0.7, "mix_prefix": "vw_", "epoch_tiles": 4800})


def test_epoch_tiles_becomes_max_iters_for_the_batch(tmp_path, monkeypatch):
    from openstroke.obb import train as trainer
    seen = {}
    monkeypatch.setattr(trainer, "main", lambda argv: seen.setdefault("argv", argv))
    opts = {"batch": 32, "epoch_tiles": 4800, "epochs": 10, "epoch_lr": True, "nbs": 32}
    cli._run_trainer(tmp_path, tmp_path, opts, n_train=21416, epochs=10, warmup_epochs=0.5, user_ema=set())
    assert opts["max_iters"] == 150
    argv = seen["argv"]
    assert argv[argv.index("--max-iters") + 1] == "150"
    saved = json.loads((tmp_path / "openobb_args.json").read_text())
    assert saved["epoch_tiles"] == 4800 and saved["warmup"] == 75     # warmup 0.5 epoch of the capped epoch
