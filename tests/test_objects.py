"""E62: HVAC objects from per-stroke class and centre offset (openstroke/objects.py) and the model's object head."""

import json

import numpy as np

from openstroke.geometry import page_segments
from openstroke.objects import HVAC_CLASSES, OFFSET_SCALE, assign_objects, class_of, decode_objects, object_counts
from test_e0 import line, shape


def _duct_and_elbow():
    duct = {"shape_name": "Duct", "vertices": [[0, 0], [100, 0], [100, 10], [0, 10]]}
    elbow = {"shape_name": "Elbow 90", "vertices": [[100, 0], [110, 0], [110, 10], [100, 10]]}
    shapes = [shape(line(0, 0, 100, 0)), shape(line(0, 10, 100, 10)), shape(line(100, 0, 110, 0)), shape(line(110, 0, 110, 10)), shape(line(300, 0, 400, 0))]
    return [duct, elbow], shapes


def test_class_names_follow_ai_takeoff():
    assert [HVAC_CLASSES[class_of(n)] for n in ("Duct", "Elbow 45", "Elbow vuông", "Concentric Reducer vuông tròn", "Eccentric Reducer", "Chân rẽ vuông tròn", "Chân rẽ", "Line")] == [
        "Duct", "Elbow", "Elbow vuông", "Reducer vuông tròn", "Eccentric Reducer", "Chân rẽ vuông tròn", "Chân rẽ", "other"]


def test_assign_objects_gives_each_stroke_its_object_and_the_fitting_wins_where_they_touch():
    objects, shapes = _duct_and_elbow()
    segs = page_segments(shapes)
    eligible = np.array([True, True, True, True, False])
    cls, inst, ctr = assign_objects(segs, objects, eligible)
    assert list(inst) == [0, 0, 1, 1, -1]
    assert [HVAC_CLASSES[c] for c in cls] == ["Duct", "Duct", "Elbow", "Elbow", "other"]
    assert np.allclose(ctr[0], [50, 5]) and np.allclose(ctr[2], [105, 5]) and np.isnan(ctr[4]).all()


def test_decode_with_true_offsets_gives_back_the_objects_and_counts_match():
    objects, shapes = _duct_and_elbow()
    segs = page_segments(shapes)
    cls, inst, ctr = assign_objects(segs, objects, np.array([True, True, True, True, False]))
    xy = np.array([[50, 0], [50, 10], [105, 0], [110, 5], [350, 0]], float)
    off = np.nan_to_num(ctr - xy)
    pred, pcls = decode_objects(xy, cls, off, cls > 0)
    length = np.array([100, 100, 10, 10, 100], float)
    counts = object_counts(pred, pcls, inst.astype(np.int64), {0: 1, 1: 2}, length)
    assert counts[1].tolist() == [1, 1, 1] and counts[2].tolist() == [1, 1, 1]
    wrong = object_counts(pred, pcls[::-1].copy(), inst.astype(np.int64), {0: 1, 1: 2}, length)
    assert wrong[:, 0].sum() == 0  # the right strokes with the wrong class match nothing


def test_windows_carry_object_targets_and_offsets_turn_with_the_window():
    from openstroke.data import Augment, Page, augment, collate, window

    n = 4
    page = Page("s", "hvac", np.zeros((n, 47), np.float32), np.array([[0, 0], [10, 0], [20, 0], [30, 0]], np.float32),
                np.zeros((n, 16, 2), np.float16), np.zeros(n, np.int64), np.ones(n, np.uint8), np.ones(n, np.float32), np.arange(n))
    page.ocls = np.array([1, 1, 2, 0], np.int64)
    page.octr = np.array([[5, 0], [5, 0], [20, 10], [np.nan, np.nan]])
    w = window(page, 0, n)
    w = augment(w, Augment(fliplr=0, rot90=False, scale=0, color=0, gray=0, drop_tokens=0), np.random.default_rng(0))
    b = collate([w])
    order = b["pos"][0].numpy()
    off = dict(zip(order.tolist(), (b["ooff"][0].numpy() * OFFSET_SCALE).tolist()))
    assert np.allclose(off[0], [5, 0]) and np.allclose(off[2], [0, 10]) and off[3] == [0, 0]
    assert dict(zip(order.tolist(), b["ocls"][0].tolist())) == {0: 1, 1: 1, 2: 2, 3: 0}
    w2 = augment(window(page, 0, n), Augment(fliplr=1.0, rot90=False, scale=0, color=0, gray=0, drop_tokens=0), np.random.default_rng(0))
    b2 = collate([w2])
    off2 = dict(zip(b2["pos"][0].tolist(), (b2["ooff"][0].numpy() * OFFSET_SCALE).tolist()))
    assert np.allclose(off2[0], [-5, 0])  # mirrored with the shapes


def test_object_head_leaves_the_pipe_logit_as_it_was(tmp_path):
    import torch

    from openstroke.data import collate, window
    from openstroke.models import build, load_weights
    from test_v2 import _page

    torch.manual_seed(0)
    a = build("openstroke2", 47, dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32)
    torch.manual_seed(0)
    b = build("openstroke2", 47, dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32, obj_head=True)
    load_weights(b, a.state_dict(), new_heads=True)
    a.eval()
    b.eval()
    batch = collate([window(_page(tmp_path)[0], 0, 200)])
    with torch.no_grad():
        assert torch.allclose(a(batch), b(batch), atol=1e-6)
        assert b.last_obj.shape[-1] == len(HVAC_CLASSES) + 2


def test_training_with_the_object_head_saves_object_outputs(tmp_path):
    from test_v2 import _cli

    assert _cli(tmp_path, "model=openstroke2", "epochs=1", "tta=false", "obj_head=true", "name=obj") == 0
    run = tmp_path / "runs" / "obj"
    z = np.load(run / "fold0" / "val_scores.npz")
    stem = json.loads((run / "fold0" / "result.json").read_text())["val"][0]
    assert z[f"{stem}__oprob"].shape[1] == len(HVAC_CLASSES) and z[f"{stem}__ooff"].shape[1] == 2
