"""F148 / F151: faces of the picked strokes, the face classifier step and how faces join the logic's objects."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_e0 import line, shape  # noqa: E402

from openstroke.geometry import page_segments  # noqa: E402
from openstroke.hvac_faces import CLASSES, FEATURES, add_faces, face_features, face_objects, page_faces  # noqa: E402


def _duct(x, y, w=100.0, h=20.0, gap=0.0):
    """A rectangle of four strokes; ``gap`` leaves the last corner open by that much."""
    return [shape(line(x, y, x + w, y)), shape(line(x + w, y, x + w, y + h)), shape(line(x + w, y + h, x, y + h)),
            shape(line(x, y + h, x, y + gap))]


def test_faces_close_small_gaps_and_grow_back_to_the_centre_lines():
    shapes = _duct(0, 0) + _duct(200, 0, gap=0.5) + _duct(400, 0, gap=5.0)
    segs = page_segments(shapes)
    faces = page_faces(segs, np.ones(len(shapes), bool))
    assert len(faces) == 2                                   # a 0.5 pt opening at a corner still closes, 5 pt does not
    assert all(abs(f.area - 2000.0) < 60 for f in faces)     # grown back to the 100 x 20 outline
    X = face_features(faces, segs, np.ones(len(shapes), bool), np.zeros(len(shapes), bool))
    assert X.shape == (2, len(FEATURES))


class _Always:
    """A stand-in classifier: every face is a Duct with probability p."""
    def __init__(self, p):
        self.p, self.classes_ = p, np.arange(len(CLASSES))

    def predict_proba(self, X):
        out = np.full((len(X), len(CLASSES)), (1 - self.p) / (len(CLASSES) - 1))
        out[:, 1] = self.p
        return out


def test_face_objects_keep_confident_faces_and_skip_the_logics_area():
    shapes = _duct(0, 0) + _duct(200, 0)
    segs = page_segments(shapes)
    prob = np.full(len(shapes), 0.9)
    curved = np.zeros(len(shapes), bool)
    assert face_objects(segs, prob, curved, _Always(0.4), CLASSES) == []          # below the class-probability threshold
    assert face_objects(segs, np.full(len(shapes), 0.1), curved, _Always(0.9), CLASSES) == []   # strokes below the pick threshold
    faces = face_objects(segs, prob, curved, _Always(0.9), CLASSES)
    assert [f["shape_name"] for f in faces] == ["Duct", "Duct"]
    logic = [{"shape_name": "Duct", "vertices": [[-1, -1], [101, -1], [101, 21], [-1, 21]]}]
    out = add_faces(logic, faces)
    assert len(out) == 2 and out[0] is logic[0] and out[1]["refined"] == "face" and "face" not in out[1]
    assert min(v[0] for v in out[1]["vertices"]) > 150      # the face kept is the one the logic did not cover


def test_quad_is_drawn_round_its_outline_not_as_a_bowtie():
    from openstroke.geometry import item_polylines

    ul, ur, ll, lr = [0, 0], [10, 0], [0, 5], [10, 5]
    (pl,) = item_polylines(["qu", [ul, ur, ll, lr]])
    assert pl.tolist() == [ul, ur, lr, ll, ul]
