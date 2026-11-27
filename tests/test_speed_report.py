"""End-of-training deploy speed report (compile off vs compile=True), on CPU with a tiny checkpoint."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from test_deploy_api import _sample_payload, _save_tiny_checkpoint  # noqa: E402


def _pages(ckpt, n=2):
    from openstroke.data import page_from_arrays
    from openstroke.infer import OpenStroke, payload_arrays

    cfg = OpenStroke(ckpt, device="cpu").config
    graph = bool(cfg.get("relations", 0) or cfg.get("rel_attn", 0) or cfg.get("edge_head") or cfg.get("group_head")
                 or "graph" in (cfg.get("mixers") or []))
    out = []
    for i in range(n):
        arrays, _ = payload_arrays(_sample_payload(20 + 8 * i, include_empty=False))
        out.append(page_from_arrays(arrays, f"p{i}", mode="raw", graph=graph, no_layers=bool(cfg.get("no_layers"))))
    return out


def test_cpu_reports_compile_unavailable(tmp_path):
    from openstroke.speed_report import compile_speed_report

    ck = _save_tiny_checkpoint(tmp_path / "m.pt")
    lines = []
    res = compile_speed_report(ck, _pages(ck), device="cpu", say=lines.append, steady_passes=1)
    assert res is not None and res["compile"] == "unavailable"
    assert res["off_ms_per_page"] > 0
    assert any("compile=True unavailable" in ln for ln in lines)


def test_table_and_parity_when_compile_applies(tmp_path, monkeypatch):
    import openstroke.infer as infer
    from openstroke.speed_report import compile_speed_report

    monkeypatch.setattr(infer, "_compile_submodules", lambda model, device, mode: mode is not None)
    ck = _save_tiny_checkpoint(tmp_path / "m.pt")
    pages = _pages(ck)
    lines = []
    res = compile_speed_report(ck, pages, device="cpu", say=lines.append, steady_passes=1)
    assert res["compile"] == "default" and len(res["per_page"]) == len(pages)
    assert res["max_abs_dprob"] == pytest.approx(0.0, abs=1e-6)
    assert res["speedup"] > 0 and res["first_pass_s"] >= 0
    text = "\n".join(lines)
    assert "Deploy speed" in text and "total" in text and "max|dp|" in text


def test_never_raises(tmp_path):
    from openstroke.speed_report import compile_speed_report

    lines = []
    assert compile_speed_report(tmp_path / "missing.pt", _pages(_save_tiny_checkpoint(tmp_path / "m.pt")),
                                device="cpu", say=lines.append) is None
    assert lines and lines[0].startswith("Deploy speed report skipped")
