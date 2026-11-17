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


class _Names:
    """A stand-in classifier: fixed class names per face, decided by the face's x position."""
    def __init__(self, by_x):
        self.by_x, self.classes_ = by_x, np.arange(len(CLASSES))

    def predict_proba(self, X):   # X rows follow the faces' order: page_faces sorts nothing, so use the area to tell them apart
        out = np.zeros((len(X), len(CLASSES)))
        for i, a in enumerate(np.expm1(X[:, 0])):
            out[i, CLASSES.index(self.by_x(a))] = 0.9
        return out


def test_taper_between_two_ducts_is_a_reducer():
    # duct (0..100, 20 high) - trapezoid (100..120, 20 -> 14 high) - duct (120..220, 14 high)
    shapes = [shape(line(0, 0, 100, 0)), shape(line(100, 0, 100, 20)), shape(line(100, 20, 0, 20)), shape(line(0, 20, 0, 0)),
              shape(line(100, 0, 120, 3)), shape(line(100, 20, 120, 17)), shape(line(120, 3, 120, 17)),
              shape(line(120, 3, 220, 3)), shape(line(220, 3, 220, 17)), shape(line(220, 17, 120, 17))]
    segs = page_segments(shapes)
    clf = _Names(lambda area: "Duct" if area > 1000 else "none")      # the classifier misses the reducer
    out = face_objects(segs, np.full(len(shapes), 0.9), np.zeros(len(shapes), bool), clf, CLASSES)
    names = sorted(o["shape_name"] for o in out)
    assert names == ["Concentric Reducer", "Duct", "Duct"]
    assert next(o for o in out if o["shape_name"] == "Concentric Reducer")["refined"] == "taper"


def test_a_curve_drawn_as_many_segments_leaves_no_teeth():
    # a quarter elbow: outer arc r=40, inner arc r=20 (beziers sampled into many segments), closed by two straight ends
    from openstroke.geometry import item_polylines
    k = 0.5523
    outer = ["c", [40, 0], [40, 40 * k], [40 * k, 40], [0, 40]]
    inner = ["c", [20, 0], [20, 20 * k], [20 * k, 20], [0, 20]]
    shapes = [shape(outer), shape(inner), shape(line(20, 0, 40, 0)), shape(line(0, 20, 0, 40))]
    segs = page_segments(shapes)
    assert len(item_polylines(outer)[0]) > 4
    (f,) = page_faces(segs, np.ones(len(shapes), bool))
    ring = np.hypot(*np.asarray(f.exterior.coords).T)
    assert ring.min() > 20 - 0.2 and ring.max() < 40 + 1.0      # no tooth pokes inside the inner arc or outside the outer one


def test_open_elbow_is_closed_by_virtual_end_lines():
    """The model often picks the two arcs of an elbow but not the short lines at its ends: with ``curved`` the ends are joined."""
    k = 0.5523
    outer = ["c", [40, 0], [40, 40 * k], [40 * k, 40], [0, 40]]
    inner = ["c", [20, 0], [20, 20 * k], [20 * k, 20], [0, 20]]
    shapes = [shape(outer), shape(inner)]
    segs = page_segments(shapes)
    picked = np.ones(2, bool)
    assert page_faces(segs, picked) == []
    (f,) = page_faces(segs, picked, curved=np.ones(2, bool))
    assert abs(f.area - np.pi * (40 ** 2 - 20 ** 2) / 4) < 40


def test_a_closed_shape_cut_by_a_crossing_duct_stays_one_candidate():
    from openstroke.hvac_faces import page_candidates

    duct = {"items": [line(0, 0, 100, 0), line(100, 0, 100, 20), line(100, 20, 0, 20), line(0, 20, 0, 0)], "type": "s",
            "color": [1, 0, 0], "width": 0.5, "layer": "P", "closePath": False, "dashes": "[] 0", "seqno": 0}
    crossing = [shape(line(40, -30, 40, 50)), shape(line(60, -30, 60, 50))]   # a duct passing over it
    segs = page_segments([duct] + crossing)
    picked = np.ones(3, bool)
    assert len(page_faces(segs, picked)) == 3                    # the outline cut into three pieces
    cands = page_candidates(segs, picked)
    assert any(abs(c.area - 2000) < 30 for c in cands) and not any(c.area < 1000 for c in cands)


def test_picked_strokes_that_close_a_region_become_an_object_even_if_the_classifier_says_none():
    from openstroke.hvac_faces import post_objects

    shapes = _duct(0, 0)
    segs = page_segments(shapes)
    clf = _Names(lambda area: "none")
    curved = np.zeros(len(shapes), bool)
    prob = np.full(len(shapes), 0.9)
    out = post_objects([], segs, prob, curved, clf, CLASSES)
    assert [o["refined"] for o in out] == ["picked"] and out[0]["shape_name"] != "none"
    assert post_objects([], segs, prob, curved, clf, CLASSES, must=False) == []
    logic = [{"shape_name": "Duct", "vertices": [[-1, -1], [101, -1], [101, 21], [-1, 21]]}]
    assert post_objects(logic, segs, prob, curved, clf, CLASSES) == logic   # the logic already has it


def test_graph_faces_join_ends_and_tees_but_not_crossings():
    """User 2026-11-16: a closed region is made by strokes whose ends touch; strokes that cross belong to different duct runs."""
    from openstroke.hvac_faces import graph_faces

    h = _duct(0, 0)                                                       # 100 x 20
    v = [shape(line(40, -30, 60, -30)), shape(line(60, -30, 60, 50)), shape(line(60, 50, 40, 50)), shape(line(40, 50, 40, -30))]
    segs = page_segments(h + v)
    areas = sorted(round(f.area) for f in graph_faces(segs, np.ones(8, bool)))
    assert areas == [1600, 2000]                                          # two whole ducts, one over the other, not 3 + 2 pieces
    # a branch duct whose two sides end on the main duct's side (a T) closes against it
    br = [shape(line(30, 20, 30, 60)), shape(line(50, 20, 50, 60)), shape(line(30, 60, 50, 60))]
    segs = page_segments(h + br)
    areas = sorted(round(f.area) for f in graph_faces(segs, np.ones(7, bool)))
    assert areas == [800, 2000]
    # ends 0.8 pt apart still meet
    segs = page_segments(_duct(0, 0, gap=0.8))
    assert len(graph_faces(segs, np.ones(4, bool))) == 1


def test_an_item_drawing_a_duct_and_its_reducer_gives_two_objects_and_segments_split_between_them():
    from openstroke.hvac_faces import item_faces, segment_objects

    item = {"items": [line(0, 0, 60, 0), line(60, 0, 60, 10), line(60, 10, 67, 9.5), line(67, 9.5, 67, 0.5), line(67, 0.5, 60, 0),
                      line(60, 0, 60, 10), line(60, 10, 0, 10), line(0, 10, 0, 0)],
            "type": "s", "color": [1, 0, 0], "width": 0.5, "layer": "P", "closePath": False, "dashes": "[] 0", "seqno": 0}
    segs = page_segments([item])
    regions = sorted((g for g, _, _ in item_faces(segs, np.ones(1, bool))), key=lambda g: g.area)
    assert [round(g.area) for g in regions] == [66, 600]
    objs = [{"shape_name": "Concentric Reducer", "vertices": np.asarray(regions[0].exterior.coords)[:-1].tolist()},
            {"shape_name": "Duct", "vertices": np.asarray(regions[1].exterior.coords)[:-1].tolist()}]
    owner = segment_objects(objs, segs, np.ones(1, bool))
    assert set(owner[[2, 3, 4]]) == {0} and set(owner[[0, 6, 7]]) == {1}       # flanks + far end -> reducer, duct sides -> duct


def test_a_thin_gap_between_a_fitting_and_a_duct_end_joins_the_fitting():
    from shapely.geometry import Polygon

    from openstroke.hvac_faces import merge_slivers

    duct = Polygon([(0, -40), (20, -40), (20, 0), (0, 0)])
    gap = Polygon([(0, 0), (20, 0), (0, 2)])                              # the fitting's top edge slants 2 pt off the duct's end
    tee = Polygon([(0, 2), (20, 0), (30, 10), (25, 20), (10, 15), (-5, 20), (-10, 10)])
    out = merge_slivers([duct, gap, tee])
    assert len(out) == 2 and any(abs(g.area - (tee.area + gap.area)) < 6 for g in out)   # +- the 0.05 pt seam


def test_regions_take_the_class_their_outline_strokes_vote_for():
    """OpenStroke29: the model's per-stroke class probabilities name each closed region (length-weighted vote of its outline strokes)."""
    from openstroke.hvac_faces import post_objects, vote_names
    from openstroke.objects import HVAC_CLASSES, N_CLASSES

    shapes = _duct(0, 0) + _duct(200, 0)
    segs = page_segments(shapes)
    picked = np.ones(len(shapes), bool)
    faces = page_faces(segs, picked)
    pc = np.zeros((len(shapes), N_CLASSES))
    pc[:4, HVAC_CLASSES.index("Duct")] = 0.9
    pc[4:, HVAC_CLASSES.index("Chân rẽ")] = 0.8
    names, share = vote_names(faces, segs, picked, pc)
    assert sorted(names) == ["Chân rẽ", "Duct"] and (share > 0.9).all()
    out = post_objects([], segs, np.full(len(shapes), 0.9), np.zeros(len(shapes), bool), _Names(lambda a: "none"), CLASSES,
                       order="items", stroke_cls=pc)
    assert sorted(o["shape_name"] for o in out) == ["Chân rẽ", "Duct"] and {o["refined"] for o in out} <= {"model", "picked"}


def test_reducer_probability_split_over_three_labels_still_wins_and_shape_picks_eccentric():
    """Viewer 'reducer' and GT Concentric / Eccentric are one general class in the vote; the shape decides concentric vs eccentric."""
    from openstroke.hvac_faces import vote_names
    from openstroke.objects import HVAC_CLASSES, N_CLASSES

    def quad(pts):
        return [shape(line(*pts[k], *pts[(k + 1) % 4])) for k in range(4)]

    conc = quad([(0, 0), (20, 3), (20, 17), (0, 20)])                 # both flanks slanted
    ecc = quad([(100, 0), (120, 0), (120, 14), (100, 20)])            # one flank straight: 2 right corners
    shapes = conc + ecc
    segs = page_segments(shapes)
    picked = np.ones(len(shapes), bool)
    faces = sorted(page_faces(segs, picked), key=lambda f: f.centroid.x)
    pc = np.zeros((len(shapes), N_CLASSES))
    pc[:, HVAC_CLASSES.index("Duct")] = 0.4                            # Duct beats each reducer label alone ...
    for c in ("Concentric Reducer", "Eccentric Reducer", "Reducer vuông tròn"):
        pc[:, HVAC_CLASSES.index(c)] = 0.2                             # ... but not their sum (0.6)
    names, _ = vote_names(faces, segs, picked, pc)
    assert names == ["Concentric Reducer", "Eccentric Reducer"]


def test_general_classes_make_gt_and_viewer_labels_agree():
    from openstroke.objects import HVAC_CLASSES, general_class

    names = ["Duct", "Concentric Reducer", "Eccentric Reducer", "Reducer vuông tròn", "Chân rẽ", "Chân rẽ vuông tròn", "Tee"]
    out = [HVAC_CLASSES[k] for k in general_class(np.array([HVAC_CLASSES.index(n) for n in names]))]
    assert out == ["Duct", "Reducer vuông tròn", "Reducer vuông tròn", "Reducer vuông tròn", "Chân rẽ", "Chân rẽ", "Tee"]


def test_reducer_kind_by_corners():
    """F163: a tapering outline with one straight flank (2 right corners) is Eccentric, a symmetric taper Concentric; a rectangle's 4 right
    corners do not make it Eccentric."""
    from shapely.geometry import Polygon

    from openstroke.hvac_faces import reducer_kind

    assert reducer_kind(Polygon([(0, 0), (20, 0), (20, 10), (0, 4)])) == "Eccentric Reducer"
    assert reducer_kind(Polygon([(0, 0), (20, -3), (20, 13), (0, 10)])) == "Concentric Reducer"
    assert reducer_kind(Polygon([(0, 0), (20, 0), (20, 10), (0, 10)])) == "Concentric Reducer"


def test_drop_spurious_ducts_slivers_and_lone_small_boxes():
    """F164: a Duct narrower than min_width goes, a lone small Duct goes, a Duct touching another object stays, other classes stay."""
    from openstroke.hvac_faces import drop_spurious_ducts

    box = lambda x, y, w, h, n: {"shape_name": n, "vertices": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]}
    run = [box(0, 0, 40, 8, "Duct"), box(40, 0, 10, 8, "Elbow")]          # a duct with its elbow
    sliver = box(0, 20, 40, 0.5, "Duct")
    lone = box(100, 100, 6, 6, "Duct")
    lone_elbow = box(200, 200, 6, 6, "Elbow")
    out = drop_spurious_ducts(run + [sliver, lone, lone_elbow], min_width=1.0, iso_area=150)
    assert out == run + [lone_elbow]
    assert drop_spurious_ducts(run + [sliver, lone], min_width=0, iso_area=0) == run + [sliver, lone]


def test_dashed_duct_closes_even_when_a_second_item_redraws_part_of_its_end_line():
    """F166: a duct drawn with dashes (4.6 pt dash, 2.3 pt gap) on all four sides; a second item draws two dashes of one end line again,
    one of them together with a piece of the side. The gaps are bridged and the duct is one closed region (before: the copies made the
    dash ends look joined and the end line stayed open)."""
    from openstroke.hvac_faces import graph_faces

    def dashed(x0, y0, x1, y1, dash=4.6, gap=2.3):
        L = np.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        out, t = [], 0.0
        while t < L - 1e-9:
            e = min(t + dash, L)
            out.append(shape(line(x0 + ux * t, y0 + uy * t, x0 + ux * e, y0 + uy * e)))
            t = e + gap
        return out

    shapes = dashed(0, 0, 60, 0) + dashed(60, 0, 60, 20) + dashed(60, 20, 0, 20) + dashed(0, 20, 0, 0)
    shapes += [shape(line(6.9, 0, 11.5, 0)), shape(line(0, 4.6, 0, 0), line(0, 0, 4.6, 0))]   # the copies
    segs = page_segments(shapes)
    faces = graph_faces(segs, np.ones(len(shapes), bool), curved=np.zeros(len(shapes), bool))
    big = [f for f in faces if f.area > 600]
    assert len(big) == 1 and abs(big[0].area - 1200) < 60
