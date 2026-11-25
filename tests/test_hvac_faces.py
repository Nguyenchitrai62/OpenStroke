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


def _duct_and_reducer():
    """F180: a duct 0..30 x 0..10 and a reducer 30..40 after it; shape 0 a dash on the duct's top wall just before the reducer's corner,
    shape 1 the shared end line x = 30, shape 2 the reducer's flank."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_e0 import line, shape

    from openstroke.geometry import page_segments

    shapes = [shape(line(28.6, 0, 29.8, 0)), shape(line(30, 0, 30, 10)), shape(line(30, 0, 40, 2))]
    objects = [{"shape_name": "Duct", "vertices": [[0, 0], [30, 0], [30, 10], [0, 10]]},
               {"shape_name": "Concentric Reducer", "vertices": [[30, 0], [40, 2], [40, 8], [30, 10]]}]
    return page_segments(shapes), objects


def test_strokes_end_at_an_objects_corners():
    """User 2026-11-18: a dash on the duct's wall just past the reducer's corner belongs to the duct, not to the smaller reducer; a line on
    both outlines (the shared end line) still goes to the fitting."""
    from openstroke.hvac_faces import segment_objects
    from openstroke.objects import assign_objects

    segs, objects = _duct_and_reducer()
    eligible = np.ones(3, bool)
    old = assign_objects(segs, objects, eligible)[1]
    new = assign_objects(segs, objects, eligible, by_edge=True)[1]
    assert old[0] == 1                       # the default (labels, scoring): the smallest object within 2 pt
    assert new.tolist() == [0, 1, 1]         # by_edge: the duct's dash, the shared line and the flank to the reducer
    seg_obj = segment_objects(objects, segs, eligible)
    assert seg_obj[segs.owner == 0].tolist() == [0] and seg_obj[segs.owner == 2].tolist() == [1]


def test_dashed_duct_closes_through_crossing_wall_and_short_t_gap():
    """F234 (A5): a dashed duct whose side dashes stop 2.2 pt short of a perpendicular end wall (forward T-ray),
    has a 0.8 pt stub at one end, and is crossed orthogonally by a solid duct wall 0.3 pt from a dash end still closes."""
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

    # Dashed duct 0..60 x 0..20: starts with a 0.8 pt stub at x=0..0.8 (below old END_MIN=1.0),
    # and ends with a dash stopping at x=57.8 (2.2 pt short of solid perpendicular end wall x=60).
    top = [shape(line(0.0, 0.0, 0.8, 0.0))] + dashed(3.1, 0.0, 56.0, 0.0) + [shape(line(56.0, 0.0, 57.8, 0.0))]
    bot = [shape(line(0.0, 20.0, 0.8, 20.0))] + dashed(3.1, 20.0, 56.0, 20.0) + [shape(line(56.0, 20.0, 57.8, 20.0))]
    left_end = [shape(line(0.0, 0.0, 0.0, 20.0))]
    right_end = [shape(line(60.0, -10.0, 60.0, 30.0))]   # perpendicular trunk wall (not a free end at y=0 / y=20!)
    # Crossing solid vertical duct passing 0.3 pt from a dash end along x = 21.8 (inside gap)
    crossing = [shape(line(21.8, -30.0, 21.8, 50.0)), shape(line(35.0, -30.0, 35.0, 50.0)),
                shape(line(21.8, -30.0, 35.0, -30.0)), shape(line(21.8, 50.0, 35.0, 50.0))]
    shapes = top + bot + left_end + right_end + crossing
    segs = page_segments(shapes)
    faces = graph_faces(segs, np.ones(len(shapes), bool), curved=np.zeros(len(shapes), bool))
    assert any(abs(f.area - 1200.0) < 60.0 for f in faces)


def test_faces_first_hybrid_overrides_branch_rescues_none_and_vetoes_fp():
    """F234 (C7a): faces_first_hybrid uses 1-to-1 stroke-matched GroupClsHead clusters to:
      (1) override Duct -> Chân rẽ, (2) rescue a 'none' face (Elbow touching a Duct),
      (3) preserve Tee (not dropped), (4) veto a low-conf FP face, and (5) drop an isolated floating Reducer."""
    from openstroke.hvac_faces import faces_first_hybrid

    # Four regions:
    # 0..3:   0..100 x 0..20 (area=2000)   -> clf says Duct (0.90), cluster 10 says Chân rẽ -> overridden to Chân rẽ
    # 4..7:   100..130 x 0..30 (area=900)  -> touches 0..3; clf says none (0.90), cluster 11 says Elbow -> rescued as Elbow
    # 8..11:  130..160 x 0..25 (area=750)  -> touches 4..7; clf says Tee (0.90) -> kept as Tee
    # 12..15: 300..320 x 0..20 (area=400)  -> clf says Duct (0.55), cluster 13 says none -> vetoed (dropped)
    # 16..19: 500..515 x 0..15 (area=225)  -> isolated floating Concentric Reducer -> dropped by physical guard
    shapes = (
        _duct(0, 0, w=100, h=20)
        + _duct(100, 0, w=30, h=30)
        + _duct(130, 0, w=30, h=25)
        + _duct(300, 0, w=20, h=20)
        + _duct(500, 0, w=15, h=15)
    )
    segs = page_segments(shapes)
    n = len(shapes)
    prob = np.full(n, 0.9, dtype=np.float64)
    curved = np.zeros(n, dtype=bool)
    L = np.bincount(segs.owner, weights=np.hypot(segs.seg[:, 2] - segs.seg[:, 0], segs.seg[:, 3] - segs.seg[:, 1]), minlength=n)

    class _Clf:
        classes_ = np.arange(len(CLASSES))

        def predict_proba(self, X):
            out = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
            for i, a in enumerate(np.expm1(X[:, 0])):
                if a > 1500:
                    out[i, CLASSES.index("Duct")] = 0.90; out[i, 0] = 0.10
                elif a > 820:
                    out[i, 0] = 0.90; out[i, CLASSES.index("Elbow")] = 0.10
                elif a > 600:
                    out[i, CLASSES.index("Tee")] = 0.90; out[i, 0] = 0.10
                elif a > 300:
                    out[i, CLASSES.index("Duct")] = 0.55; out[i, 0] = 0.45
                else:
                    out[i, CLASSES.index("Concentric Reducer")] = 0.90; out[i, 0] = 0.10
            return out

    # OBJ8_CLASSES: 0=none, 1=Duct, 2=Elbow, 3=Elbow vuông, 4=Conc, 5=Ecc, 6=Chân rẽ, 7=Tee
    ginst = np.array([10] * 4 + [11] * 4 + [12] * 4 + [13] * 4 + [14] * 4, dtype=np.int64)
    gcids = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    gcprobs = np.array([
        [0.05, 0.10, 0.0, 0.0, 0.0, 0.0, 0.85, 0.0],   # cluster 10 -> Chân rẽ
        [0.10, 0.05, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0],   # cluster 11 -> Elbow
        [0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90],   # cluster 12 -> Tee
        [0.80, 0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # cluster 13 -> none (FP veto)
        [0.05, 0.05, 0.0, 0.0, 0.90, 0.0, 0.0, 0.0],   # cluster 14 -> isolated Reducer (dropped)
    ], dtype=np.float32)
    group_objects = {"instance_id": ginst, "cluster_ids": gcids, "cluster_probs": gcprobs}

    objs, cls_out, inst_out = faces_first_hybrid(segs, prob, curved, L, _Clf(), CLASSES, group_objects, thr=0.5)
    got = sorted((o["shape_name"], o["refined"]) for o in objs)
    assert got == [("Chân rẽ", "cid_cr"), ("Elbow", "cid_rescue"), ("Tee", "face")]
    assert (inst_out[12:] == -1).all() and (cls_out[:12] > 0).all()


def test_f237_cross_item_symbol_triangles_merge_to_single_quad():
    """F237: a square-to-round branch drawn as separate shapes (outer square + inscribed diamond creating 4 corner triangles + 1 inner quad,
    the exact (3,3,3,3,4) pattern seen on real CAD drawings) merges into ONE crisp 4-corner polygon with inner == 5."""
    from openstroke.hvac_faces import priority_candidates

    # Outer square 0..10 x 0..10 (4 separate sides) + inscribed diamond connecting midpoints (4 separate diagonals -> 4 triangles + 1 inner quad)
    sym_shapes = _duct(0, 0, w=10, h=10) + [
        shape(line(5, 0, 10, 5)),
        shape(line(10, 5, 5, 10)),
        shape(line(5, 10, 0, 5)),
        shape(line(0, 5, 5, 0)),
    ]
    segs = page_segments(sym_shapes)
    faces, _, inner = priority_candidates(segs, np.ones(len(sym_shapes), bool), curved=np.zeros(len(sym_shapes), bool))
    assert len(faces) == 1
    assert abs(faces[0].area - 100.0) < 1.0
    assert inner[0] == 5


def test_f237_segment_objects_prevents_protruding_stub_bleed():
    """F237: a segment sticking past a polygon corner into a dropped thin strip (<55% on outline) is NOT assigned to the polygon,
    eliminating visual 'horns/antennae' while true outline segments stay cleanly assigned."""
    from openstroke.hvac_faces import segment_objects

    # Trapezoid y=0..10, x=0..20 (top edge), x=0..25 (bottom edge); plus a 16pt vertical line sticking 6pt ABOVE (y=-6..10)
    # and a 4pt vertical stub strictly ABOVE the top-right corner (y=-4..0, touching corner x=20,y=0)
    objs = [{"shape_name": "Chân rẽ", "vertices": [[0, 0], [20, 0], [25, 10], [0, 10]]}]
    shapes = [
        shape(line(0, 0, 20, 0)),     # 0: exact top wall -> assigned (0)
        shape(line(20, 0, 25, 10)),   # 1: exact chamfer flank -> assigned (0)
        shape(line(20, -4, 20, 0)),   # 2: external vertical stub touching corner -> MUST be -1!
        shape(line(0, -10, 0, 10)),   # 3: 20pt line with only 50% on polygon -> MUST be -1 (not stolen!)
    ]
    segs = page_segments(shapes)
    sobj = segment_objects(objs, segs, np.ones(len(shapes), bool))
    assert sobj[segs.owner == 0].tolist() == [0]
    assert sobj[segs.owner == 1].tolist() == [0]
    assert sobj[segs.owner == 2].tolist() == [-1]
    assert sobj[segs.owner == 3].tolist() == [-1]


def test_f237_fp_veto_preserves_high_conf_connected_duct():
    """F237: a high-confidence connected face (score=0.95, nb>=1, e.g. dashed duct on MB TH4) is NOT dropped even if GroupClsHead predicts none>=0.85."""
    from openstroke.hvac_faces import faces_first_hybrid

    shapes = _duct(0, 0, w=80, h=20) + _duct(80, 0, w=40, h=20)
    segs = page_segments(shapes)
    n = len(shapes)
    prob = np.full(n, 0.9, dtype=np.float64)
    curved = np.zeros(n, dtype=bool)
    L = np.bincount(segs.owner, weights=np.hypot(segs.seg[:, 2] - segs.seg[:, 0], segs.seg[:, 3] - segs.seg[:, 1]), minlength=n)

    class _Clf:
        classes_ = np.arange(len(CLASSES))

        def predict_proba(self, X):
            out = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
            out[:, CLASSES.index("Duct")] = 0.95
            out[:, 0] = 0.05
            return out

    # Cluster 10 (duct 0) wrongly says none=0.92; because face score=0.95 and nb=1, safe veto preserves it!
    go = {
        "instance_id": np.array([10] * 4 + [11] * 4, dtype=np.int64),
        "cluster_ids": np.array([10, 11], dtype=np.int64),
        "cluster_probs": np.array([
            [0.92, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.05, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32),
    }
    objs, _, _ = faces_first_hybrid(segs, prob, curved, L, _Clf(), CLASSES, go, thr=0.5)
    assert len(objs) == 2 and all(o["shape_name"] == "Duct" for o in objs)


def test_f239_drop_curved_elbow_void_between_two_elbows_preserves_sbend_and_concentric():
    """F239: empty air pocket squeezed between 2 opposing Elbows (polygonalized arcs, no Bezier required) is dropped,
    while both real Elbows, S-bend Elbows (straight shared flange), and concentric parallel Elbows are preserved."""
    from openstroke.hvac_faces import _drop_curved_elbow_voids

    def _arc_ring(cx, cy, r_in, r_out, a0, a1, steps=10):
        th = np.linspace(a0, a1, steps)
        outer = [[cx + r_out * np.cos(t), cy + r_out * np.sin(t)] for t in th]
        inner = [[cx + r_in * np.cos(t), cy + r_in * np.sin(t)] for t in th[::-1]]
        return outer + inner

    # 1. Exact double-elbow crossing pocket from user screenshot:
    #    - Big upper-right Elbow centered at (0, 20): inner radius 20, outer radius 40, angles [0, pi/2] -> inner arc from (20,20) to (0,40)
    #    - Small lower-left Elbow centered at (20, 40): inner radius 20, outer radius 32, angles [pi, 1.5*pi] -> inner arc from (0,40) to (20,20)
    #    - Lens-shaped empty void pocket enclosed between their two inner arcs!
    e_big = _arc_ring(0.0, 20.0, 20.0, 40.0, 0.0, 0.5 * np.pi)
    e_sml = _arc_ring(20.0, 40.0, 20.0, 32.0, np.pi, 1.5 * np.pi)
    th = np.linspace(0.0, 0.5 * np.pi, 10)
    arc1 = [[20.0 * np.cos(t), 20.0 + 20.0 * np.sin(t)] for t in th]
    th2 = np.linspace(np.pi, 1.5 * np.pi, 10)
    arc2 = [[20.0 + 20.0 * np.cos(t), 40.0 + 20.0 * np.sin(t)] for t in th2]
    void_lens = arc1 + arc2[1:-1]

    segs = page_segments(_duct(0, 0, w=20, h=20))
    picked = np.ones(4, bool)
    curved = np.zeros(4, bool)

    objs_double = [
        {"shape_name": "Elbow", "vertices": e_big, "score": 0.9},
        {"shape_name": "Elbow", "vertices": e_sml, "score": 0.9},
        {"shape_name": "Elbow", "vertices": void_lens, "score": 0.7},  # trapped empty pocket!
    ]
    out_double = _drop_curved_elbow_voids(objs_double, segs, picked, curved)
    assert len(out_double) == 2
    assert out_double[0]["vertices"] == e_big and out_double[1]["vertices"] == e_sml

    # 2. S-bend (2 real Elbows meeting along a straight vertical flange at x=20) -> BOTH kept!
    e_s1 = _arc_ring(20.0, 0.0, 10.0, 25.0, 0.5 * np.pi, np.pi)
    e_s2 = _arc_ring(20.0, 35.0, 10.0, 25.0, 1.5 * np.pi, 2.0 * np.pi)
    objs_sbend = [
        {"shape_name": "Elbow", "vertices": e_s1, "score": 0.9},
        {"shape_name": "Elbow", "vertices": e_s2, "score": 0.9},
    ]
    assert len(_drop_curved_elbow_voids(objs_sbend, segs, picked, curved)) == 2

    # 3. Two concentric parallel Elbows sharing 1 curved wall (outer/inner walls remain free air) -> BOTH kept!
    e_c1 = _arc_ring(0.0, 0.0, 15.0, 30.0, 0.0, 0.5 * np.pi)
    e_c2 = _arc_ring(0.0, 0.0, 30.0, 45.0, 0.0, 0.5 * np.pi)
    objs_conc = [
        {"shape_name": "Elbow", "vertices": e_c1, "score": 0.9},
        {"shape_name": "Elbow", "vertices": e_c2, "score": 0.9},
    ]
    assert len(_drop_curved_elbow_voids(objs_conc, segs, picked, curved)) == 2
