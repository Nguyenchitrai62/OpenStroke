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


def test_cohort_sampling_gives_each_cohort_its_share():
    from openstroke.data import page_cohort, sampling_prob

    stems = ["doc1_p0001_gt1", "doc2_p0001_gt2", "doc3_p0001_gt3", "vw_ab_p0001", "vw_cd_p0002"]
    modes = ["fire", "plumbing", "plumbing", "plumbing", "plumbing"]
    p = sampling_prob(np.array([100, 400, 900, 10000, 40000.0]), modes, stems=stems,
                      cohort_share={"fire:gt": 0.4, "plumbing:gt": 0.3, "plumbing:viewer": 0.3, "hvac:gt": 0.5})
    coh = [page_cohort(m, s) for m, s in zip(modes, stems)]
    share = {g: sum(q for q, c in zip(p, coh) if c == g) for g in set(coh)}
    assert abs(share["fire:gt"] - 0.4) < 1e-9 and abs(share["plumbing:gt"] - 0.3) < 1e-9 and abs(share["plumbing:viewer"] - 0.3) < 1e-9
    assert abs(p[1] / p[2] - 2 / 3) < 1e-9   # sqrt(shapes) inside a cohort
    with pytest.raises(ValueError):
        sampling_prob(np.ones(2), ["fire", "hvac"], stems=["a", "b"], cohort_share={"fire:gt": 1.0})


def test_protocol_drops_the_checkpoint_data_options_but_not_the_command_line():
    from types import SimpleNamespace

    from openstroke.train import apply_protocol

    proto = {"name": "thin1", "sampling": {"fire:gt": 0.5, "plumbing:gt": 0.5}, "select": ["fire:gt"]}
    said = []
    a = SimpleNamespace(modes="hvac", viewer_share=0.65, plumbing_share=None, gt_only=False, _explicit=["data"])
    apply_protocol(a, {"protocol": proto}, said.append)
    assert a.modes == "all" and a.viewer_share is None and a.protocol == proto and "not taken from the checkpoint" in said[0]
    b = SimpleNamespace(modes="fire", viewer_share=None, plumbing_share=0.2, gt_only=False, _explicit=["modes", "plumbing_share"])
    apply_protocol(b, {"protocol": proto}, said.append)
    assert b.modes == "fire" and b.plumbing_share == 0.2 and b.protocol["sampling"] is None
    c = SimpleNamespace(modes="hvac", viewer_share=0.65, plumbing_share=None, gt_only=False, _explicit=[])
    apply_protocol(c, {}, said.append)   # HVAC datasets: unchanged
    assert c.modes == "hvac" and c.viewer_share == 0.65 and c.protocol is None
    with pytest.raises(ValueError):
        apply_protocol(SimpleNamespace(modes="all", viewer_share=None, plumbing_share=None, gt_only=True, _explicit=[]),
                       {"protocol": proto}, said.append)


def test_cohort_select_weighs_cohorts_equally():
    from types import SimpleNamespace

    from openstroke.engine.trainer import cohort_select

    class P(SimpleNamespace):
        def __len__(self):
            return len(self.label)

    fire = P(stem="doc1_p1_gt1", mode="fire", label=np.array([1, 0], np.uint8), length=np.ones(2, np.float32))
    plum = P(stem="doc2_p1_gt2", mode="plumbing", label=np.array([1] * 2 + [0] * 98, np.uint8), length=np.ones(100, np.float32))
    vw = P(stem="vw_x_p0001", mode="plumbing", label=np.array([0, 1], np.uint8), length=np.ones(2, np.float32))
    s_fire = np.array([0.9, 0.3])                  # right only above 0.3
    s_plum = np.r_[[0.6, 0.6], np.full(98, 0.1)]   # right for any threshold in (0.1, 0.6]
    best, at05 = cohort_select([s_fire, s_plum, np.array([0.9, 0.1])], [fire, plum, vw], ["fire:gt", "plumbing:gt"])
    assert 0.3 < best["threshold"] <= 0.6 and abs(best["f2"] - 1.0) < 1e-9
    assert set(best["cohorts"]) == {"fire:gt", "plumbing:gt", "plumbing:viewer"} and best["selected"] == ["fire:gt", "plumbing:gt"]
    assert best["cohorts"]["plumbing:viewer"]["f2"] == 0.0 and at05["threshold"] == 0.5


def test_object_eval_counts_line_objects_on_fire_pages():
    from types import SimpleNamespace

    from openstroke.object_eval import gt_objects, summarize

    page = SimpleNamespace(mode="fire", label=np.array([1, 1, 0], np.uint8), oinst=np.array([0, 1, -1]), ocls=np.zeros(3, np.int64))
    g, n = gt_objects(page)
    assert n == 2 and g.tolist() == [0, 1, -1]
    hvac = SimpleNamespace(mode="hvac", label=page.label, oinst=page.oinst, ocls=np.array([1, 0, 0]))
    assert gt_objects(hvac)[1] == 1
    s = summarize({"a": np.array([1, 1, 2]), "vw_b": np.array([0, 1, 1])}, {"a": "fire:gt", "vw_b": "plumbing:viewer"})
    assert set(s) == {"fire:gt", "plumbing:viewer", "all"} and s["fire:gt"]["matched"] == 1


def test_training_with_a_dataset_protocol(tmp_path, capsys):
    """Tiny CPU run through the real trainer on FIRE + PLUMBING pages with a protocol: cohort sampling and cohort rows in the log."""
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from test_model import _synthetic

    from openstroke.cli import main

    data, manifest = _synthetic(tmp_path)
    for r, mode in zip(manifest["records"], ("fire", "fire", "plumbing", "plumbing")):
        r["mode"] = mode
    for p, mode in zip(manifest["physical_pages"], ("fire", "fire", "plumbing", "plumbing")):
        p["mode"] = mode
    for f in data.glob("*.npz"):
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        y = d["label"] > 0
        d["oinst"] = np.where(y, np.arange(len(y)) // 3, -1).astype(np.int32)
        d["ocls"] = np.zeros(len(y), np.int8)
        d["octr"] = np.zeros((len(y), 2), np.float32)
        np.savez(f, **d)
    manifest["protocol"] = {"name": "tiny", "sampling": {"fire:gt": 0.5, "plumbing:gt": 0.5}, "select": ["fire:gt", "plumbing:gt"]}
    (data / "manifest.json").write_text(json.dumps(manifest))
    folds = {"status": "confirmed", "k": 1, "groups": [{"sha256": f"sha{i}", "filename": f"f{i}.pdf", "mode": m, "pages": [1],
                                                       "fold": 0 if i in (0, 3) else -1}
                                                      for i, m in enumerate(("fire", "fire", "plumbing", "plumbing"))]}
    (data / "folds.json").write_text(json.dumps(folds))   # val: one FIRE and one PLUMBING page
    assert main(["train", f"data={data}", "fold=0", "steps=2", "batch=2", "window=128", "val_batch=2", "workers=0", "device=cpu",
                 "tta=false", f"project={tmp_path / 'runs'}", "model=openstroke2", "dims=16,32", "enc_depths=1,1", "dec_depths=1",
                 "patch=32", "epochs=1", "modes=all", "name=proto"]) == 0
    out = capsys.readouterr().out
    assert "protocol tiny" in out and "sampling fire:gt 50%" in out
    log = (tmp_path / "runs" / "proto" / "train.log").read_text(encoding="utf-8")
    assert "fire:gt" in log and "SUMMARY select (mean of fire:gt, plumbing:gt)" in log
    # a FIRE-only run, then a run started from it: the checkpoint's modes=fire is not taken, its weights are scored first
    assert main(["train", f"data={data}", "fold=0", "steps=2", "batch=2", "window=128", "val_batch=2", "workers=0", "device=cpu",
                 "tta=false", f"project={tmp_path / 'runs'}", "model=openstroke2", "dims=16,32", "enc_depths=1,1", "dec_depths=1",
                 "patch=32", "epochs=1", "modes=fire", "name=fire_only"]) == 0
    first = tmp_path / "runs" / "fire_only" / "fold0" / "best.pt"
    assert main(["train", f"data={data}", "fold=0", "steps=2", "batch=2", "window=128", "val_batch=2", "workers=0", "device=cpu",
                 f"project={tmp_path / 'runs'}", f"model={first}", "epochs=1", "name=from_ckpt"]) == 0
    log = (tmp_path / "runs" / "from_ckpt" / "train.log").read_text(encoding="utf-8")
    assert "not taken from the checkpoint: {'modes': 'fire'}" in log and "Before fine-tune" in log and "plumbing:gt: F2" in log
