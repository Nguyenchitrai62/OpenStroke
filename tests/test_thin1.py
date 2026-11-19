"""Thin-line dataset thin1 (F188, tools/make_thin_dataset.py): FIRE + PLUMBING pipe lines, Viewer pages parsed and de-duplicated
against the GT, the built dataset loads like e11 (CPU, no training)."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
THIN1 = ROOT / "datasets" / "thin1"


def test_plumbing_names_and_viewer_pipe_classes(tmp_path):
    import make_thin_dataset as mt

    assert mt._pipe_name({"shape_name": "ÔNG NƯỚC NÓNG-HỒI"})["shape_name"] == "Ống NƯỚC NÓNG-HỒI"
    assert mt._pipe_name({"shape_name": "ỐNG CẤP NƯỚC NÓNG"})["shape_name"] == "Ống CẤP NƯỚC NÓNG"
    assert mt._pipe_name({"shape_name": "Elbow"})["shape_name"] == "Elbow"
    assert mt.is_pipe("plumbing")("Ống chưa rõ loại") and not mt.is_pipe("plumbing")("Hố thoát nước")
    assert mt.is_pipe("fire")("Line") and not mt.is_pipe("fire")("Sprinkler")
    page = lambda status, objs: {"page": 3, "width_pt": 100, "height_pt": 50, "labels": {"line_junction": {"status": status, "objects": objs}}}  # noqa: E731
    objs = [{"class": "line1", "points": [[0, 0], [10, 0]]}, {"class": "line 8", "points": [[0, 5], [10, 5]]},
            {"class": "Mặt bằng", "points": [[0, 0], [9, 0], [9, 9], [0, 9]]}, {"class": "duct", "points": [[0, 0], [5, 5]]},
            {"class": "line2", "points": [[1, 1]]}]
    lab = {"pdfs": [{"name": "CTN_DEV_001-050_LAYERED.pdf", "sha256": "ab" * 32, "file": "x.pdf",
                     "pages": [page("completed", objs), page("in_progress", objs[:1])]}]}
    (tmp_path / "labels.json").write_text(json.dumps(lab), encoding="utf-8")
    got = mt.viewer_pages(tmp_path)
    assert len(got) == 1 and [o["shape_name"] for o in got[0][2]] == ["Ống line1", "Ống line 8"]
    assert mt.SOURCE_PDF.search("GT38_CTN_04_500_01_p38_46.pdf").group(1) == "CTN_04_500_01"
    assert mt.SOURCE_PDF.search("GTG_CTN_DEV_001-050_p1-4.pdf").group(1) == "CTN_DEV_001-050"


def test_assign_lines_takes_a_name_predicate():
    from openstroke.geometry import PageSegments
    from openstroke.objects import assign_lines

    seg = np.array([[0, 0.5, 50, 0.5], [0, 20.5, 50, 20.5]], float)
    segs = PageSegments(seg=seg, owner=np.arange(2), shape_offsets=np.arange(3))
    objs = [{"shape_name": "Ống chưa rõ loại", "vertices": [[0, 0], [60, 0]]}, {"shape_name": "Hố thoát nước", "vertices": [[0, 20], [60, 20]]}]
    inst, _ = assign_lines(segs, objs, np.ones(2, bool), names=lambda n: n.startswith("Ống"))
    assert inst.tolist() == [0, -1]


def test_shared_pairs_finds_the_same_drawing():
    import make_thin_dataset as mt

    a = {(i, 0) for i in range(100)}
    fp = {"gt": a, "vw_x": a | {(500, 1)}, "other": {(i, 7) for i in range(100)}}
    assert [(p[0], p[1]) for p in mt.shared_pairs(fp)] == [("gt", "vw_x")]


@pytest.mark.skipif(not (THIN1 / "manifest.json").exists(), reason="datasets/thin1 not built (tools/make_thin_dataset.py)")
def test_built_thin1_loads_and_keeps_groups_on_one_side():
    from openstroke.data import attach_objects, page_from_npz
    from openstroke.folds import resolve

    m = json.loads((THIN1 / "manifest.json").read_text(encoding="utf-8"))
    folds = json.loads((THIN1 / "folds.json").read_text(encoding="utf-8"))
    ok = m["records"]
    assert {r["mode"] for r in ok} == {"fire", "plumbing"} and any(r["stem"].startswith("vw_") for r in ok)
    side = {}
    for g in folds["groups"]:
        side.setdefault(g["group"], set()).add(g["fold"])
    assert all(len(s) == 1 for s in side.values())
    entries, _ = resolve(m, folds)
    assert sorted(e[0] for e in entries) == sorted(r["stem"] for r in ok)
    val = {e[0] for e in entries if e[1] == 0}
    assert any(not s.startswith("vw_") for s in val) and {r["mode"] for r in ok if r["stem"] in val} == {"fire", "plumbing"}
    gone = {d["stem"] for d in m["excluded"]}
    assert not gone & {r["stem"] for r in ok}
    for r in sorted(ok, key=lambda r: (THIN1 / r["npz"]).stat().st_size)[:2]:
        npz = THIN1 / r["npz"]
        p = page_from_npz(npz, r["stem"], r["mode"], no_layers=True)
        with np.load(npz) as raw:
            attach_objects(p, raw)
            assert not raw["ocls"].any() and ((raw["fcls"] > 0) == (raw["oinst"] >= 0)).all()
        assert p.oinst is not None and len(p.oinst) == len(p)
