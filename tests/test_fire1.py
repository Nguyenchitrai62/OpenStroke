"""FIRE stroke dataset fire1 (F187, tools/make_fire_dataset.py): Line objects of the true strokes (objects.assign_lines), FIRE pages train
through the openstroke31 losses (closure, edge pairs) with no HVAC class, and the built dataset loads like e11 (CPU, tiny)."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
FIRE1 = ROOT / "datasets" / "fire1"


def _segs(rows):
    from openstroke.geometry import PageSegments

    seg = np.asarray(rows, float)
    off = np.arange(len(seg) + 1)
    return PageSegments(seg=seg, owner=np.arange(len(seg)), shape_offsets=off)


def test_assign_lines_gives_each_true_stroke_its_line():
    from openstroke.objects import assign_lines

    objects = [{"shape_name": "Sprinkler", "vertices": [[0, 0], [1, 1], [0, 1]]},
               {"shape_name": "Line", "vertices": [[0, 0], [100, 0]]},
               {"shape_name": "Line", "vertices": [[100, 0], [100, 80]]},
               {"shape_name": "Tee", "vertices": [[99, -1], [101, -1], [101, 1]]}]
    segs = _segs([[10, 0.5, 60, 0.5],      # along line 1
                  [100.5, 20, 100.5, 70],  # along line 2
                  [95, 0, 100, 10],        # mostly near line 2's start / line 1's end: the nearer one wins by length
                  [10, 30, 60, 30],        # far from every line
                  [20, 0, 40, 0]])         # on line 1 but not eligible
    eligible = np.array([1, 1, 1, 1, 0], bool)
    inst, ctr = assign_lines(segs, objects, eligible)
    assert inst.tolist()[:2] == [1, 2] and inst[3] == -1 and inst[4] == -1
    assert inst[2] in (1, 2)
    assert np.allclose(ctr[0], [50, 0]) and np.allclose(ctr[1], [100, 40]) and np.isnan(ctr[3]).all()


def test_fire_pages_without_hvac_class_train_with_closure_and_edge_pairs(tmp_path):
    """FIRE npz as fire1 writes it (ocls 0, oinst = Line, fcls): the openstroke31 losses run on it through the real trainer."""
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from test_model import _synthetic

    import openstroke.models.openstroke2 as os2
    from openstroke.cli import main
    from openstroke.folds import propose

    data, manifest = _synthetic(tmp_path)
    for f in data.glob("*.npz"):
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        n = len(d["label"])
        y = d["label"] > 0
        d["oinst"] = np.where(y, np.arange(n) // 3, -1).astype(np.int32)
        d["ocls"] = np.zeros(n, np.int8)
        d["octr"] = np.where(y[:, None], np.full((n, 2), 50.0), np.nan).astype(np.float32)
        d["fcls"] = y.astype(np.int8)
        np.savez(f, **d)
    (data / "folds.json").write_text(json.dumps(propose(manifest, k=2)))
    calls = {"closure": 0, "edge": 0}
    real_c, real_e = os2.closure_loss, os2.edge_loss
    os2.closure_loss = lambda *a, **k: (calls.__setitem__("closure", calls["closure"] + 1), real_c(*a, **k))[1]
    os2.edge_loss = lambda *a, **k: (calls.__setitem__("edge", calls["edge"] + 1), real_e(*a, **k))[1]
    try:
        assert main(["train", f"data={data}", "fold=0", "steps=2", "batch=2", "window=128", "val_batch=2", "workers=0", "device=cpu",
                     "tta=false", f"project={tmp_path / 'runs'}", "model=openstroke2", "dims=16,32", "enc_depths=1,1", "dec_depths=1",
                     "patch=32", "epochs=1", "rel_attn=1", "edge_head=true", "edge_pairs=true", "closure=0.5", "feature_drop=strict",
                     "name=fire"]) == 0
    finally:
        os2.closure_loss, os2.edge_loss = real_c, real_e
    assert calls["closure"] > 0 and calls["edge"] > 0


@pytest.mark.skipif(not (FIRE1 / "manifest.json").exists(), reason="datasets/fire1 not built (tools/make_fire_dataset.py)")
def test_built_fire1_loads_like_e11_and_keeps_projects_on_one_side():
    from openstroke.data import attach_objects, page_from_npz
    from openstroke.objects import FIRE_CLASSES

    m = json.loads((FIRE1 / "manifest.json").read_text(encoding="utf-8"))
    folds = json.loads((FIRE1 / "folds.json").read_text(encoding="utf-8"))
    ok = [r for r in m["records"] if r["status"] == "ok"]
    assert ok and all(r["mode"] == "fire" and not r["stem"].startswith("vw_") for r in ok)
    side = {}
    for g in folds["groups"]:
        side.setdefault(g["project"], set()).add(g["fold"])
    assert all(len(s) == 1 for s in side.values()) and {0, -1} <= {g["fold"] for g in folds["groups"]}
    assert m["objects"]["fire_classes"] == list(FIRE_CLASSES)
    from openstroke.folds import resolve

    entries, _ = resolve(m, folds)                      # what the trainer reads: every page once, val = fold 0
    assert sorted(e[0] for e in entries) == sorted(r["stem"] for r in ok)
    assert {e[0] for e in entries if e[1] == 0} == {r["stem"] for r in ok if r["split"] == "val"}
    for r in sorted(ok, key=lambda r: (FIRE1 / r["npz"]).stat().st_size)[:2]:   # the two smallest pages
        npz = FIRE1 / r["npz"]
        with np.load(npz) as z:
            assert {"ocls", "oinst", "octr", "fcls"} <= set(z.files) and not z["ocls"].any()
            y = z["label"] > 0
            assert ((z["fcls"] > 0) == (z["oinst"] >= 0)).all() and not (z["oinst"][~y] >= 0).any()
        p = page_from_npz(npz, r["stem"], "fire", no_layers=True)
        with np.load(npz) as raw:
            attach_objects(p, raw)
        assert p.oinst is not None and len(p.oinst) == len(p) and (p.label > 0).any()
        L = p.length.astype(float)
        yy = p.label > 0
        assert L[yy & (p.oinst >= 0)].sum() >= 0.9 * L[yy].sum()
