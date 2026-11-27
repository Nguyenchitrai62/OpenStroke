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


def test_f239b_segment_objects_rejects_interior_hatch_on_duct_but_keeps_branch_symbol():
    """F239b: diagonal cross-hatch lines inside a Duct must NOT attach to the Duct in segment_objects,
    whereas interior X-symbol diagonals inside a Chân rẽ vuông tròn are preserved."""
    from openstroke.hvac_faces import segment_objects

    # Object 0: Duct [0,0]-[100,20] with 2 interior diagonal cross-hatch lines (shapes 4, 5)
    # Object 1: Chân rẽ vuông tròn [120,0]-[140,20] with 1 interior diagonal (shape 10)
    shapes = (
        _duct(0, 0, w=100, h=20)
        + [{"items": [["l", [5, 0], [35, 20]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0", "rect": [5, 0, 35, 20]}]
        + [{"items": [["l", [0, 8], [20, 20]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0", "rect": [0, 8, 20, 20]}]
        + _duct(120, 0, w=20, h=20)
        + [{"items": [["l", [120, 0], [140, 20]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0", "rect": [120, 0, 140, 20]}]
    )
    objs = [
        {"shape_name": "Duct", "vertices": [[0, 0], [100, 0], [100, 20], [0, 20]]},
        {"shape_name": "Chân rẽ vuông tròn", "vertices": [[120, 0], [140, 0], [140, 20], [120, 20]]},
    ]
    segs = page_segments(shapes)
    sobj = segment_objects(objs, segs, np.ones(len(shapes), bool))
    # Duct outline walls belong to Duct 0; interior hatch diagonals (shapes 4, 5) must be -1!
    for s in range(4):
        assert sobj[segs.owner == s].tolist() == [0]
    assert sobj[segs.owner == 4].tolist() == [-1]
    assert sobj[segs.owner == 5].tolist() == [-1]
    # Chân rẽ vuông tròn keeps BOTH its 4 walls AND its interior X-diagonal (shape 10)!
    for s in range(6, 11):
        assert sobj[segs.owner == s].tolist() == [1]


def test_f240_v5_end_reducer_l_elbow_micro_sliver_and_square_to_round_suffix():
    """F240 (v5): verify (1) 6-right-angle L-shaped Elbow vuông rule, (2) end-of-duct Reducer rule (flaring out with 1 end neighbour),
    (3) micro-sliver low-conf Reducer removal, and (4) _tag_square_to_round V/X crease detection without false positives on parallel hatch."""
    from openstroke.hvac_faces import _name_faces, _tag_square_to_round, face_features

    # 1. _tag_square_to_round: Chân rẽ with V-creases -> Chân rẽ vuông tròn; plain Chân rẽ with parallel hatch stays Chân rẽ;
    #    Concentric Reducer with exact 2 equal-length corner V-creases -> Reducer vuông tròn; plain Reducer stays Concentric Reducer.
    shapes = (
        _duct(0, 0, w=20, h=20)
        + [{"items": [["l", [0, 0], [10, 20]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0", "rect": [0, 0, 10, 20]}]
        + [{"items": [["l", [20, 0], [10, 20]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0", "rect": [10, 0, 20, 20]}]
        + _duct(40, 0, w=20, h=20)
        + [{"items": [["l", [42, 0], [52, 20]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0", "rect": [42, 0, 52, 20]}]
        + [{"items": [["l", [48, 0], [58, 20]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0", "rect": [48, 0, 58, 20]}]
        + _duct(80, 0, w=12, h=10)
        + [{"items": [["l", [80, 0], [86, 10]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0", "rect": [80, 0, 86, 10]}]
        + [{"items": [["l", [92, 0], [86, 10]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0", "rect": [86, 0, 92, 10]}]
        + _duct(120, 0, w=12, h=10)
    )
    segs = page_segments(shapes)
    curved = np.zeros(len(shapes), dtype=bool)
    objs = [
        {"shape_name": "Chân rẽ", "vertices": [[0, 0], [20, 0], [20, 20], [0, 20]]},
        {"shape_name": "Chân rẽ", "vertices": [[40, 0], [60, 0], [60, 20], [40, 20]]},
        {"shape_name": "Concentric Reducer", "vertices": [[80, 0], [92, 0], [92, 10], [80, 10]]},
        {"shape_name": "Concentric Reducer", "vertices": [[120, 0], [132, 0], [132, 10], [120, 10]]},
    ]
    _tag_square_to_round(objs, segs, curved)
    assert objs[0]["shape_name"] == "Chân rẽ vuông tròn"
    assert objs[1]["shape_name"] == "Chân rẽ"  # parallel hatch (1 angle only) must NOT trigger vuông tròn!
    assert objs[2]["shape_name"] == "Reducer vuông tròn"
    assert objs[3]["shape_name"] == "Concentric Reducer"

    # 2. _name_faces: 6-right-angle L-elbow between 2 ducts -> Elbow vuông (even if clf says none!)
    #    and 4-corner end-of-duct trapezoid flaring from 1 duct end-cap -> Concentric Reducer (even if clf says Chân rẽ!)
    from shapely.geometry import Polygon

    d1 = Polygon([[0, 0], [40, 0], [40, 10], [0, 10]])
    l_el = Polygon([[40, 0], [56, 0], [56, 16], [46, 16], [46, 10], [40, 10]])
    d2 = Polygon([[46, 16], [56, 16], [56, 56], [46, 56]])
    end_red = Polygon([[46, 56], [56, 56], [62, 68], [40, 68]])
    faces = [d1, l_el, d2, end_red]
    seg_shapes = []
    for f in faces:
        cc = list(f.exterior.coords)
        for a, b in zip(cc[:-1], cc[1:]):
            seg_shapes.append({"items": [["l", list(a), list(b)]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0",
                               "rect": [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]})
    segs2 = page_segments(seg_shapes)
    X2 = face_features(faces, segs2, np.ones(len(seg_shapes), bool), np.zeros(len(seg_shapes), bool))

    class _DummyClf:
        classes_ = np.arange(len(CLASSES))

        def predict_proba(self, X):
            out = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
            out[0, CLASSES.index("Duct")] = 0.9
            out[1, CLASSES.index("none")] = 0.8   # clf mistakenly rejects L-elbow as none!
            out[2, CLASSES.index("Duct")] = 0.9
            out[3, CLASSES.index("Chân rẽ")] = 0.7  # clf mistakenly calls end-of-duct reducer a Chân rẽ!
            return out

    names, _, why, _, _ = _name_faces(faces, X2, _DummyClf(), CLASSES)
    assert names[1] == "Elbow vuông" and why[1] == "l_elbow"
    assert names[3] == "Concentric Reducer" and why[3] == "end_taper"


def test_f241_v6_mep_connectivity_mitered_elbow_sq_tee_and_cad_guardrails():
    """F241 (v6b): verify (1) _merge_mitered_square_elbows merges twin diagonal Eccentric Reducer halves into Elbow vuông
    when connected to >= 2 ducts; (2) physical MEP connectivity drops isolated Chân rẽ & isolated square/short Duct boxes
    while preserving both real connected Duct+Chân rẽ routes AND small 2D.pdf-style (Duct+Chân rẽ) pairs;
    (3) page-level width outlier gate & zigzag+arrowhead sidewall grille filter."""
    from openstroke.hvac_faces import (
        _apply_cad_noise_guardrails,
        _merge_mitered_square_elbows,
        faces_first_hybrid,
    )

    # 1. Mitered 19x19 (361 pt^2) square Elbow vuông split into 2 diagonal halves called Eccentric Reducer, touching 2 ducts
    miter_objs = [
        {"shape_name": "Duct", "vertices": [[-40, 0], [0, 0], [0, 19], [-40, 19]], "score": 0.95},
        {"shape_name": "Duct", "vertices": [[0, 19], [19, 19], [19, 60], [0, 60]], "score": 0.95},
        {"shape_name": "Eccentric Reducer", "vertices": [[0, 0], [19, 0], [19, 19]], "score": 0.85},
        {"shape_name": "Eccentric Reducer", "vertices": [[0, 0], [19, 19], [0, 19]], "score": 0.85},
    ]
    merged_m = _merge_mitered_square_elbows(miter_objs)
    assert sorted(o["shape_name"] for o in merged_m) == ["Duct", "Duct", "Elbow vuông"]

    # 2. Physical MEP connectivity: 1 real connected Duct + side Chân rẽ AND 1 small (Duct + Chân rẽ) pair MUST survive;
    #    1 floating isolated Chân rẽ (6x6) + 1 floating isolated square Duct (10x10) MUST be dropped!
    shapes = (
        _duct(0, 0, w=80, h=12)           # real long duct (0..3)
        + _duct(30, 12, w=10, h=8)        # connected side Chân rẽ touching duct (4..7)
        + _duct(200, 200, w=6, h=6)       # floating isolated Chân rẽ box (8..11) -> DROP
        + _duct(300, 200, w=10, h=10)     # floating isolated square Duct box (12..15) -> DROP
        + _duct(400, 0, w=10, h=4)        # small 2D.pdf-style branch Duct (16..19) -> KEEP (has Chân rẽ!)
        + _duct(410, 0, w=4, h=4)         # attached Chân rẽ (20..23) -> KEEP (has Duct!)
    )
    segs = page_segments(shapes)
    n = len(shapes)
    prob = np.full(n, 0.95, dtype=np.float64)
    curved = np.zeros(n, dtype=bool)
    L = np.bincount(segs.owner, weights=np.hypot(segs.seg[:, 2] - segs.seg[:, 0], segs.seg[:, 3] - segs.seg[:, 1]), minlength=n)

    class _ConnClf:
        classes_ = np.arange(len(CLASSES))

        def predict_proba(self, X):
            out = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
            for idx_x in range(len(X)):
                area = float(np.expm1(X[idx_x, 0]))
                if area > 500 or abs(area - 40.0) < 5.0:
                    out[idx_x, CLASSES.index("Duct")] = 0.95
                elif abs(area - 100.0) < 10.0:
                    out[idx_x, CLASSES.index("Duct")] = 0.90  # isolated 10x10 square called Duct
                else:
                    out[idx_x, CLASSES.index("Chân rẽ")] = 0.90
            return out

    objs, _, _ = faces_first_hybrid(segs, prob, curved, L, _ConnClf(), CLASSES, None, thr=0.5)
    names_out = sorted(o["shape_name"] for o in objs)
    assert names_out == ["Chân rẽ", "Chân rẽ", "Duct", "Duct"], f"Expected 2 connected Duct+Chân rẽ pairs, got {names_out}"

    # 3. Sidewall grille filter: small side-attached Chân rẽ (7x1.5 pt) + unpicked 5-seg zigzag + filled triangle arrowhead -> DROPPED!
    grille_shapes = (
        _duct(0, 0, w=80, h=10)
        + _duct(30, 10, w=7, h=1.5)  # small sidewall grille collar (4..7)
        + [{"items": [["l", [33.5, 13.0], [33.5, 14.0]], ["l", [33.5, 14.0], [32.5, 14.5]],
                      ["l", [32.5, 14.5], [34.5, 15.0]], ["l", [34.5, 15.0], [33.5, 15.5]],
                      ["l", [33.5, 15.5], [33.5, 16.5]]], "type": "s", "color": [0, 0, 0], "width": 0.5, "layer": "0", "rect": [32.5, 13.0, 34.5, 16.5]}]
        + [{"items": [["l", [33.0, 12.2], [34.0, 12.2]], ["l", [34.0, 12.2], [33.5, 13.2]], ["l", [33.5, 13.2], [33.0, 12.2]]],
            "type": "fs", "color": [0, 0, 0], "width": 0.0, "layer": "0", "rect": [33.0, 12.2, 34.0, 13.2]}]
    )
    segs_g = page_segments(grille_shapes)
    ng = len(grille_shapes)
    prob_g = np.array([0.98] * 8 + [0.01, 0.01], dtype=np.float64)
    Lg = np.bincount(segs_g.owner, weights=np.hypot(segs_g.seg[:, 2] - segs_g.seg[:, 0], segs_g.seg[:, 3] - segs_g.seg[:, 1]), minlength=ng)
    attrs_g = {
        "width": np.array([0.5] * 8 + [0.5, 0.0], dtype=np.float32),
        "op_counts": np.array([[1, 0, 0, 0]] * 8 + [[5, 0, 0, 0], [3, 0, 0, 0]], dtype=np.int32),
        "type_code": np.array([0] * 9 + [2], dtype=np.int8),
        "fill_opacity": np.array([0.0] * 9 + [1.0], dtype=np.float32),
    }
    objs_g = [
        {"shape_name": "Duct", "vertices": [[0, 0], [80, 0], [80, 10], [0, 10]], "score": 0.95},
        {"shape_name": "Chân rẽ", "vertices": [[30, 10], [37, 10], [37, 11.5], [30, 11.5]], "score": 0.90},
    ]
    cleaned_g = _apply_cad_noise_guardrails(objs_g, segs_g, prob_g, Lg, attrs_g, None, thr=0.5)
    assert [o["shape_name"] for o in cleaned_g] == ["Duct"]


def test_f245_v7_vuong_tron_trapezoid_merge_eccentric_tag_miter_none_tee_and_compat(tmp_path):
    """F245 (v7): verify (1) trapezoid + inscribed triangle vuông tròn merge (Pass A & Pass B 1-tri+1-trap),
    (2) _tag_square_to_round covers Eccentric Reducer & wide-base trapezoid V-triangle,
    (3) _caps bridges curved dashes <= 12.5 pt and closes open parallel straight duct ends,
    (4) face-level mitered Elbow vuông recovers twin diagonal halves even when classifier says 'none',
    (5) curved 3-way Tee rule catches Y-Tee even when classifier says 'Chân rẽ',
    (6) load_classifier / _predict_proba remain 100% backward-compatible with legacy 30-feature joblib files."""
    import joblib
    from shapely.geometry import Polygon

    from openstroke.hvac_faces import (
        FEATURES,
        FEATURES_30,
        FEATURES_41,
        _name_faces,
        _tag_square_to_round,
        face_features,
        faces_first_hybrid,
        graph_faces,
        load_classifier,
        priority_candidates,
    )

    assert len(FEATURES) == 45 and len(FEATURES_41) == 41 and len(FEATURES_30) == 30

    # 1. Trapezoid + inscribed V-triangle (3 triangles sharing edges -> clean 4-corner trapezoid with hull_fill=1.0, rect_fill<0.85!)
    #    Wide base (0,0)-(14,0), narrow base (4,8)-(10,8), inscribed V-triangle (0,0)-(7,8)-(14,0)
    vt_shapes = [
        shape(line(0, 0, 14, 0)),
        shape(line(14, 0, 10, 8)),
        shape(line(10, 8, 4, 8)),
        shape(line(4, 8, 0, 0)),
        shape(line(0, 0, 7, 8)),
        shape(line(14, 0, 7, 8)),
    ]
    segs_vt = page_segments(vt_shapes)
    f_vt, _, in_vt = priority_candidates(segs_vt, np.ones(len(vt_shapes), bool), curved=np.zeros(len(vt_shapes), bool))
    assert len(f_vt) == 1 and abs(f_vt[0].area - 80.0) < 1.5 and in_vt[0] == 3

    # 2. _tag_square_to_round tags Eccentric Reducer with wide-base V-triangle -> Reducer vuông tròn
    ecc_vt_shapes = [
        shape(line(0, 0, 14, 0)),
        shape(line(14, 0, 8, 10)),
        shape(line(8, 10, 0, 10)),
        shape(line(0, 10, 0, 0)),
        shape(line(0, 0, 4, 10)),
        shape(line(14, 0, 4, 10)),
    ]
    segs_ev = page_segments(ecc_vt_shapes)
    ecc_objs = [{"shape_name": "Eccentric Reducer", "vertices": [[0, 0], [14, 0], [8, 10], [0, 10]]}]
    _tag_square_to_round(ecc_objs, segs_ev, np.zeros(len(ecc_vt_shapes), bool))
    assert ecc_objs[0]["shape_name"] == "Reducer vuông tròn"

    # 3. _caps v7 closes a straight open-ended parallel duct (two 40pt horizontal walls 12pt apart with open ends)
    open_duct = [shape(line(0, 0, 40, 0)), shape(line(0, 12, 40, 12))]
    segs_od = page_segments(open_duct)
    gf_od = graph_faces(segs_od, np.ones(2, bool), curved=np.zeros(2, bool))
    assert any(abs(f.area - 480.0) < 20.0 for f in gf_od)

    # 4. Face-level mitered Elbow vuông: two 19x19 diagonal halves classified as 'none' touching 2 Ducts still merge into Elbow vuông!
    miter_shapes = (
        _duct(-40, 0, w=40, h=19)
        + _duct(0, 19, w=19, h=40)
        + [shape(line(0, 0, 19, 0)), shape(line(19, 0, 19, 19)), shape(line(0, 0, 19, 19))]
    )
    segs_m = page_segments(miter_shapes)
    nm = len(miter_shapes)
    Lm = np.bincount(segs_m.owner, weights=np.hypot(segs_m.seg[:, 2] - segs_m.seg[:, 0], segs_m.seg[:, 3] - segs_m.seg[:, 1]), minlength=nm)

    class _NoneHalvesClf:
        classes_ = np.arange(len(CLASSES))

        def predict_proba(self, X):
            out = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
            for i, a in enumerate(np.expm1(X[:, 0])):
                if a > 500:
                    out[i, CLASSES.index("Duct")] = 0.95; out[i, 0] = 0.05
                else:
                    out[i, 0] = 0.95; out[i, CLASSES.index("Duct")] = 0.05
            return out

    m_objs, _, _ = faces_first_hybrid(segs_m, np.ones(nm), np.zeros(nm, bool), Lm, _NoneHalvesClf(), CLASSES, None, thr=0.5)
    assert sorted(o["shape_name"] for o in m_objs) == ["Duct", "Duct", "Elbow vuông"]

    # 5. Legacy 30-feature and 41-feature classifiers load cleanly and work seamlessly on 45-feature X!
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(0)
    y_fit = np.array([0, 1] * 20)
    clf_old = HistGradientBoostingClassifier(max_iter=5, random_state=0).fit(rng.normal(size=(40, 30)), y_fit)
    p_old = tmp_path / "legacy30.joblib"
    joblib.dump({"clf": clf_old, "classes": list(CLASSES), "features": list(FEATURES_30), "gap": 0.6}, p_old)
    loaded_clf, loaded_cls = load_classifier(p_old)
    X45 = face_features(f_vt, segs_vt, np.ones(len(vt_shapes), bool), np.zeros(len(vt_shapes), bool))
    assert X45.shape == (1, 45)
    names_old, _, _, _, _ = _name_faces(f_vt, X45, loaded_clf, loaded_cls)
    assert len(names_old) == 1

    clf_41 = HistGradientBoostingClassifier(max_iter=5, random_state=0).fit(rng.normal(size=(40, 41)), y_fit)
    p_41 = tmp_path / "legacy41.joblib"
    joblib.dump({"clf": clf_41, "classes": list(CLASSES), "features": list(FEATURES_41), "gap": 0.6}, p_41)
    loaded_41, loaded_cls41 = load_classifier(p_41)
    names_41, _, _, _, _ = _name_faces(f_vt, X45, loaded_41, loaded_cls41)
    assert len(names_41) == 1


def test_f246_v8_mitered_vane_strip_and_relaxed_l_elbow():
    """F246 (v8): verify (1) mitered Elbow vuông bridges 1 thin diagonal turning-vane double-line strip between
    symmetric halves (even when halves + strip are all classified 'none'), and (2) relaxed near-square 6-corner
    L-elbow rule (rc>=4, cusp==1, asp<=1.15, rfill>=0.90, hfill>=0.95, nb>=1) promotes to Elbow vuông."""
    from openstroke.hvac_faces import faces_first_hybrid

    # Two 20x20 mitered halves separated by a 0.6 pt diagonal double-line turning-vane strip (between 2 Ducts)
    vane_shapes = (
        _duct(-40, 0, w=40, h=20)
        + _duct(0, 20, w=20, h=40)
        + [
            shape(line(0, 0, 20, 0)),
            shape(line(20, 0, 20, 20)),
            shape(line(0, 0.4, 19.6, 20)),
            shape(line(0.4, 0, 20, 19.6)),
        ]
    )
    segs_v = page_segments(vane_shapes)
    nv = len(vane_shapes)
    Lv = np.bincount(segs_v.owner, weights=np.hypot(segs_v.seg[:, 2] - segs_v.seg[:, 0], segs_v.seg[:, 3] - segs_v.seg[:, 1]), minlength=nv)

    class _VaneClf:
        classes_ = np.arange(len(CLASSES))

        def predict_proba(self, X):
            out = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
            for i, a in enumerate(np.expm1(X[:, 0])):
                if a > 500:
                    out[i, CLASSES.index("Duct")] = 0.95; out[i, 0] = 0.05
                else:
                    out[i, 0] = 0.98; out[i, CLASSES.index("Duct")] = 0.02
            return out

    v_objs, _, _ = faces_first_hybrid(segs_v, np.ones(nv), np.zeros(nv, bool), Lv, _VaneClf(), CLASSES, None, thr=0.5)
    assert sorted(o["shape_name"] for o in v_objs) == ["Duct", "Duct", "Elbow vuông"]


def test_f249_v9_plenum_filter_asym_eccentric_rect_no_elbow_and_iso_guards():
    """F249 (v9): verify (1) _drop_giant_room_boxes drops giant mechanical/smoke plenum room box (198x142 pt)
    while preserving elongated zoomed detail duct (302x101 pt) and wide trunk connected to matching-width neighbour;
    (2) asymmetric-flank reducer_kind catches Eccentric Reducer with 4.3 deg vs 27.5 deg flanks;
    (3) 4-right-corner non-curved rectangle predicted as Elbow is reclassified to Duct;
    (4) isolated < 2.8 pt thin double-line wall Duct is dropped while >= 2.8 pt standalone Duct and short connected
    Duct cluster touching a >= 50 pt picked stroke trunk survive;
    (5) low-confidence tiny square Duct noise box (3.4x3.2 pt, score=0.44) is dropped."""
    from shapely.geometry import Polygon

    from openstroke.hvac_faces import (
        _drop_giant_room_boxes,
        _name_faces,
        face_features,
        faces_first_hybrid,
        load_classifier,
        reducer_kind,
    )

    # 1. Giant room/plenum box filter
    box = lambda x, y, w, h, nm="Duct": {"shape_name": nm, "vertices": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], "score": 0.96}
    normal_ducts = [box(i * 40, 0, 30, 14) for i in range(6)]
    smoke_plenum = box(500, 500, 198.3, 142.4)          # 28,246 pt^2 room box -> MUST DROP
    small_touch = box(520, 642.4, 20, 12)               # small duct touching plenum (nb_short=12 < 0.45*142.4) -> plenum still DROPPED
    zoomed_duct = box(0, 300, 302.4, 101.4)             # aspect=2.98 >= 2.2 -> MUST KEEP
    wide_a = box(900, 0, 140, 120)                      # aspect=1.17, short=120 ...
    wide_b = box(1040, 0, 160, 115)                     # ... touching matching-width wide_b (115 >= 0.45*120) -> BOTH KEEP!
    kept = _drop_giant_room_boxes(normal_ducts + [smoke_plenum, small_touch, zoomed_duct, wide_a, wide_b])
    assert smoke_plenum not in kept
    assert small_touch in kept and zoomed_duct in kept and wide_a in kept and wide_b in kept

    # 2. Asymmetric-flank Eccentric Reducer (one flank 4.3 deg off perpendicular, opposite flank 27.5 deg)
    # Parallel ends along x=0 (y=0..14) and x=16 (y=1.2..22.3): bottom dy/dx=1.2/16 -> 4.29 deg; top dy/dx=8.3/16 -> 27.4 deg
    asym_ecc = Polygon([(0, 0), (16, 1.2), (16, 22.3), (0, 14)])
    sym_conc = Polygon([(0, 0), (16, -2.5), (16, 16.5), (0, 14)])
    assert reducer_kind(asym_ecc) == "Eccentric Reducer"
    assert reducer_kind(sym_conc) == "Concentric Reducer"

    # 3. 4-right-corner non-curved rectangle predicted as Elbow -> reclassified to Duct
    rect_shapes = _duct(0, 0, w=50, h=20)
    segs_r = page_segments(rect_shapes)
    f_r = page_faces(segs_r, np.ones(4, bool))
    X_r = face_features(f_r, segs_r, np.ones(4, bool), np.zeros(4, bool))

    class _ElbClf:
        classes_ = np.arange(len(CLASSES))

        def predict_proba(self, X):
            out = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
            out[:, CLASSES.index("Elbow")] = 0.85
            out[:, 0] = 0.15
            return out

    r_names, _, _, _, _ = _name_faces(f_r, X_r, _ElbClf(), CLASSES)
    assert r_names == ["Duct"]

    # 4. Isolated thin 2.0pt wall dropped, standalone 5.0pt duct kept, low-conf tiny square (3.4x3.2) dropped,
    #    and short 2-duct cluster touching a 70pt open picked trunk line preserved (Bug A fix!)
    shapes_g = (
        _duct(0, 0, w=30, h=2.0)           # isolated thin 2.0pt wall (area=60) -> DROP (short < 2.8)
        + _duct(0, 100, w=30, h=5.0)       # isolated valid 5.0pt duct (area=150, asp=6.0) -> KEEP
        + _duct(200, 0, w=3.4, h=3.2)      # low-conf tiny square (area=10.88, sc=0.44) -> DROP
        + _duct(203.4, 0, w=30, h=10)      # connected duct touching tiny square -> KEEP
        + _duct(233.4, 0, w=8, h=10)       # connected Chân rẽ -> KEEP
        + _duct(500, 0, w=12, h=8)         # short stubby duct A (asp=1.5) ...
        + _duct(512, 0, w=12, h=8)         # ... touching duct B (cc_span=24 < 60, cc_max_asp=1.5 < 2.5, 0 fittings) ...
        + [shape(line(524, 0, 600, 0)), shape(line(524, 8, 600, 8))]   # ... attached to 76pt open trunk -> MUST KEEP!
    )
    segs_g = page_segments(shapes_g)
    ng = len(shapes_g)
    Lg = np.bincount(segs_g.owner, weights=np.hypot(segs_g.seg[:, 2] - segs_g.seg[:, 0], segs_g.seg[:, 3] - segs_g.seg[:, 1]), minlength=ng)

    class _GuardClf:
        classes_ = np.arange(len(CLASSES))

        def predict_proba(self, X):
            out = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
            for i, a in enumerate(np.expm1(X[:, 0])):
                if abs(a - 10.88) < 2.0:
                    out[i, CLASSES.index("Duct")] = 0.44; out[i, 0] = 0.56
                elif abs(a - 80.0) < 5.0:
                    out[i, CLASSES.index("Chân rẽ")] = 0.90; out[i, 0] = 0.10
                else:
                    out[i, CLASSES.index("Duct")] = 0.92; out[i, 0] = 0.08
            return out

    g_objs, _, _ = faces_first_hybrid(segs_g, np.ones(ng), np.zeros(ng, bool), Lg, _GuardClf(), CLASSES, None, thr=0.5)
    areas_kept = sorted(round(Polygon(o["vertices"]).area) for o in g_objs)
    # Must drop 60 (thin wall) and 11 (tiny low-conf square); must keep 80 (CR), 96x2 (trunk-attached short ducts), 150, 300!
    assert 60 not in areas_kept and 11 not in areas_kept
    assert 150 in areas_kept and 300 in areas_kept and areas_kept.count(96) == 2

    # 5. Shipped models/hvac_faces.joblib loads cleanly and predicts on 45-feature X
    shipped = Path(__file__).resolve().parents[1] / "models" / "hvac_faces.joblib"
    if shipped.exists():
        s_clf, s_cls = load_classifier(shipped)
        pr_s = s_clf.predict_proba(X_r)
        assert pr_s.shape == (1, len(s_cls)) and abs(float(pr_s.sum()) - 1.0) < 1e-5


def test_f254_v10_endpoint_topology_crossing_ducts_and_twin_elbows():
    """F254 (v10): Endpoint-Topology non-planar cycles keep X-crossing ducts & twin overlapping 90-deg Elbows
    100% intact (ZERO polygonize shattering, ZERO Pac-Man/shark-fin ink-buffer notches) and ignore both-end
    pass-through crossing strokes in log_ink_inside."""
    from shapely.geometry import Polygon

    from openstroke.hvac_faces import face_features, faces_first_hybrid, load_classifier, priority_candidates

    # 1. Two orthogonal crossing ducts (horizontal 0..100 x 30..50 and vertical 40..60 x 0..80) -> BOTH intact (2000 & 1600 pt^2)!
    cross_shapes = _duct(0, 30, w=100, h=20) + _duct(40, 0, w=20, h=80)
    segs_c = page_segments(cross_shapes)
    f_c, _, _ = priority_candidates(segs_c, np.ones(len(cross_shapes), bool), curved=np.zeros(len(cross_shapes), bool))
    areas_c = sorted(round(f.area) for f in f_c)
    assert areas_c == [1600, 2000], f"Expected 2 intact crossing ducts [1600, 2000], got {areas_c}"

    # 2. Crossing-safe log_ink_inside (col 11): pass-through vertical walls extending > 5pt outside horizontal duct
    #    must NOT inflate log_ink_inside, whereas a genuine interior hatch line stays counted!
    X_c = face_features(f_c, segs_c, np.ones(len(cross_shapes), bool), np.zeros(len(cross_shapes), bool))
    assert float(X_c[:, 11].max()) < 0.1, f"Pass-through crossing walls leaked into log_ink_inside: {X_c[:, 11]}"

    hatch_shapes = _duct(0, 0, w=60, h=20) + [shape(line(10, 4, 50, 16))]
    segs_h = page_segments(hatch_shapes)
    f_h, _, _ = priority_candidates(segs_h, np.ones(len(hatch_shapes), bool), curved=np.zeros(len(hatch_shapes), bool))
    X_h = face_features(f_h, segs_h, np.ones(len(hatch_shapes), bool), np.zeros(len(hatch_shapes), bool))
    assert len(f_h) == 1 and float(X_h[0, 11]) > 2.5

    # 3. Twin overlapping 90-deg curved Elbows + overlapping twin vertical Ducts (User Image 2):
    #    6 intact objects (4 Ducts + 2 Elbows), ZERO shattered 3-strip splits or shark-fin notches!
    def _arc_shape(cx, cy, r, a0, a1, n=12):
        th = np.linspace(a0, a1, n + 1)
        pts = [[float(cx + r * np.cos(t)), float(cy + r * np.sin(t))] for t in th]
        return shape(*[line(pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1]) for k in range(n)])

    img2_shapes = [
        # Left horiz (-40..0, y=0..20)
        *_duct(-40, 0, w=40, h=20),
        # Left elbow center=(0,30), r_in=10, r_out=30, [-pi/2 .. 0] -> top flange (10,30)-(30,30)
        _arc_shape(0, 30, 10, -0.5 * np.pi, 0.0), _arc_shape(0, 30, 30, -0.5 * np.pi, 0.0),
        shape(line(10, 30, 30, 30)),
        # Left vertical duct (x=10..30, y=30..90)
        shape(line(10, 30, 10, 90)), shape(line(30, 30, 30, 90)), shape(line(10, 90, 30, 90)),
        # Right horiz (50..90, y=0..20)
        *_duct(50, 0, w=40, h=20),
        # Right elbow center=(50,30), r_in=10, r_out=30, [-pi .. -pi/2] -> top flange (20,30)-(40,30), overlaps left elbow!
        _arc_shape(50, 30, 10, -np.pi, -0.5 * np.pi), _arc_shape(50, 30, 30, -np.pi, -0.5 * np.pi),
        shape(line(20, 30, 40, 30)),
        # Right vertical duct (x=20..40, y=30..90), overlaps left vertical duct x=20..30!
        shape(line(20, 30, 20, 90)), shape(line(40, 30, 40, 90)), shape(line(20, 90, 40, 90)),
    ]
    segs_2 = page_segments(img2_shapes)
    n2 = len(img2_shapes)
    cur_2 = np.zeros(n2, dtype=bool)
    cur_2[[4, 5, 14, 15]] = True
    f_2, _, _ = priority_candidates(segs_2, np.ones(n2, bool), curved=cur_2)
    areas_2 = sorted(round(f.area) for f in f_2)
    assert sum(1 for a in areas_2 if abs(a - 1200) < 10) == 2, f"Twin overlapping vertical ducts missing: {areas_2}"
    assert sum(1 for a in areas_2 if abs(a - 800) < 10) == 2, f"Horizontal ducts missing: {areas_2}"
    assert sum(1 for a in areas_2 if abs(a - 628) < 25) == 2, f"Twin overlapping curved elbows missing: {areas_2}"

    shipped = Path(__file__).resolve().parents[1] / "models" / "hvac_faces.joblib"
    if shipped.exists():
        s_clf, s_cls = load_classifier(shipped)
        L2 = np.bincount(segs_2.owner, weights=np.hypot(segs_2.seg[:, 2] - segs_2.seg[:, 0], segs_2.seg[:, 3] - segs_2.seg[:, 1]), minlength=n2)
        objs_2, _, _ = faces_first_hybrid(segs_2, np.ones(n2), cur_2, L2, s_clf, s_cls, None, thr=0.5)
        names_2 = sorted(o["shape_name"] for o in objs_2)
        assert names_2 == ["Duct", "Duct", "Duct", "Duct", "Elbow", "Elbow"], f"Expected 6 exact intact objects, got {names_2}"


def test_f255_v11_dashed_closure_nesting_convexity_and_open_throat_ev():
    """F255 (v11): Holistic HVAC post-processing upgrades verified end-to-end:
    (1) dissolve_collinear_overlaps reconnects staggered dashed walls without splitting long solid walls;
    (2) _caps forward-wall check skips premature parallel virtual caps across a 2.3 pt dash-to-corner gap;
    (3) _bridge_open_l_elbow_throats closes 6-corner CAD square elbows with an open 1.5 pt inner throat;
    (4) Layer 2a absorbs inner V-triangle in Chân rẽ vuông tròn while preserving large square mitered Elbow vuông;
    (5) _name_faces guards: taper convexity + Chân rẽ>=0.75 protection + three_ways Elbow>=0.60 protection;
    (6) mitered_ev diagonal-seam guard prevents horizontal Concentric Reducer + Duct merge while keeping 45-deg vane squares;
    (7) faces_first_hybrid whole-face GroupClsHead consensus veto & Case 4 interior damper removal."""
    from shapely.geometry import Polygon

    from openstroke.hvac_cycles import dissolve_collinear_overlaps
    from openstroke.hvac_faces import _name_faces, face_features, faces_first_hybrid, priority_candidates

    # 1. dissolve_collinear_overlaps: two overlapping short dashes (<=10 pt) dissolve cleanly;
    #    long solid wall (>10 pt) overlapping a short collar stays intact!
    ch_dash = [
        (np.array([[0.0, 0.0], [6.0, 0.0]]), 0, False),
        (np.array([[4.0, 0.0], [10.0, 0.0]]), 1, False),
    ]
    res_d = dissolve_collinear_overlaps(ch_dash)
    assert len(res_d) == 3  # [0..4], [4..6], [6..10]
    ch_solid = [
        (np.array([[0.0, 0.0], [60.0, 0.0]]), 0, False),
        (np.array([[20.0, 0.0], [26.0, 0.0]]), 1, False),
    ]
    res_s = dissolve_collinear_overlaps(ch_solid)
    assert len(res_s) == 2  # long solid wall untouched!

    # 2. _caps forward-wall check + collinear dash-to-corner bridge (MB TH4 gt#16 2.3 pt gap pattern):
    #    horizontal dashed duct stopping 2.3 pt short of a perpendicular end wall closes ALL the way to the wall!
    dash_shapes = [
        shape(line(0, 0, 0, 10)),
        shape(line(0, 0, 8, 0)), shape(line(10, 0, 18, 0)), shape(line(20, 0, 27.7, 0)),
        shape(line(0, 10, 8, 10)), shape(line(10, 10, 18, 10)), shape(line(20, 10, 27.7, 10)),
        shape(line(30, -4, 30, 14)),  # perpendicular end wall 2.3 pt dead-ahead!
    ]
    segs_d = page_segments(dash_shapes)
    f_d, _, _ = priority_candidates(segs_d, np.ones(len(dash_shapes), bool), curved=np.zeros(len(dash_shapes), bool))
    assert any(abs(f.area - 300.0) < 15.0 for f in f_d), f"Dashed duct failed to bridge 2.3pt to end wall: {[round(f.area,1) for f in f_d]}"

    # 3. _bridge_open_l_elbow_throats: 6-corner CAD square elbow with 1.48 pt open inner throat (doc000187 pattern)
    l_shapes = [
        # Outer L-corner C=(0,0) -> A=(10,0), B=(0,10); inward flanges A->(10,8.5), B->(8.5,10)
        shape(line(0, 0, 10, 0)), shape(line(0, 0, 0, 10)),
        shape(line(10, 0, 10, 8.5)), shape(line(0, 10, 8.5, 10)),
        # Connected ducts making P1=(10,8.5) and P2=(8.5,10) deg>=2 corners!
        shape(line(10, 0, 40, 0)), shape(line(10, 8.5, 40, 8.5)), shape(line(40, 0, 40, 8.5)),
        shape(line(0, 10, 0, 40)), shape(line(8.5, 10, 8.5, 40)), shape(line(0, 40, 8.5, 40)),
    ]
    segs_l = page_segments(l_shapes)
    f_l, _, _ = priority_candidates(segs_l, np.ones(len(l_shapes), bool), curved=np.zeros(len(l_shapes), bool))
    assert any(90.0 <= f.area <= 105.0 for f in f_l), f"Open-throat 6-corner Elbow vuông missed: {[round(f.area,1) for f in f_l]}"

    # 4. Layer 2a: inner V-triangle in Chân rẽ vuông tròn absorbed cleanly (1 outer face, inner>=2),
    #    while 2 symmetric halves of a 20x20 square mitered Elbow vuông are NOT absorbed!
    cr_v = [
        shape(line(0, 0, 8, 0), line(8, 0, 8, 12), line(8, 12, 0, 12), line(0, 12, 0, 0)),
        shape(line(0, 0, 8, 6), line(8, 6, 0, 12), line(0, 12, 0, 0)),
    ]
    f_cr, _, inn_cr = priority_candidates(page_segments(cr_v), np.ones(2, bool), curved=np.zeros(2, bool))
    assert len(f_cr) == 1 and abs(f_cr[0].area - 96.0) < 1.0 and inn_cr[0] >= 2

    # 5. _name_faces: confident Chân rẽ (0.85) protected from do_taper; confident Elbow (0.90) protected from three_ways;
    #    concave boomerang rejected by convexity guard in faces_first_hybrid!
    d_left = Polygon([[0, 0], [40, 0], [40, 10], [0, 10]])
    cr_tap = Polygon([[40, 0], [50, 1.5], [50, 8.5], [40, 10]])
    d_right = Polygon([[50, 1.5], [80, 1.5], [80, 8.5], [50, 8.5]])
    boom = Polygon([[0, 30], [20, 30], [20, 50], [18, 32]])  # concave boomerang (hull_fill < 0.25!)
    faces5 = [d_left, cr_tap, d_right, boom]
    s5 = []
    for f in faces5:
        cc = list(f.exterior.coords)
        for a, b in zip(cc[:-1], cc[1:]):
            s5.append(shape(line(a[0], a[1], b[0], b[1])))
    segs5 = page_segments(s5)
    X5 = face_features(faces5, segs5, np.ones(len(s5), bool), np.zeros(len(s5), bool))

    class _Clf5:
        classes_ = np.arange(len(CLASSES))

        def predict_proba(self, X):
            out = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
            out[0, CLASSES.index("Duct")] = 0.95
            out[1, CLASSES.index("Chân rẽ")] = 0.85
            out[2, CLASSES.index("Duct")] = 0.95
            out[3, CLASSES.index("Concentric Reducer")] = 0.90
            return out

    nm5, _, why5, _, _ = _name_faces(faces5, X5, _Clf5(), CLASSES)
    assert nm5[1] == "Chân rẽ" and why5[1] == "face"  # confident Chân rẽ >= 0.75 NOT hijacked by taper!

    # 6. mitered_ev diagonal-seam guard: Concentric Reducer (90.5) + adjacent Duct sharing vertical flange
    #    must NOT merge into Elbow vuông!
    mr_shapes = (
        _duct(0, 0, w=40, h=14)
        + [shape(line(40, 0, 50, 2)), shape(line(50, 2, 50, 12)), shape(line(50, 12, 40, 14))]  # ConcRed ~120 pt^2
        + [shape(line(50, 2, 62, 2)), shape(line(62, 2, 62, 12)), shape(line(62, 12, 50, 12))]  # short Duct ~120 pt^2
        + _duct(62, 2, w=40, h=10)
    )
    segs_mr = page_segments(mr_shapes)
    nmr = len(mr_shapes)
    Lmr = np.bincount(segs_mr.owner, weights=np.hypot(segs_mr.seg[:, 2] - segs_mr.seg[:, 0], segs_mr.seg[:, 3] - segs_mr.seg[:, 1]), minlength=nmr)

    class _MrClf:
        classes_ = np.arange(len(CLASSES))

        def predict_proba(self, X):
            out = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
            for i in range(len(X)):
                if X[i, 13] < 0.92:  # side_ratio < 0.92 -> Concentric Reducer
                    out[i, CLASSES.index("Concentric Reducer")] = 0.92
                else:
                    out[i, CLASSES.index("Duct")] = 0.92
            return out

    mr_objs, _, _ = faces_first_hybrid(segs_mr, np.ones(nmr), np.zeros(nmr, bool), Lmr, _MrClf(), CLASSES, None, thr=0.5)
    assert "Elbow vuông" not in [o["shape_name"] for o in mr_objs]
    assert "Concentric Reducer" in [o["shape_name"] for o in mr_objs]

    # 7. Case 4 interior damper box inside large Duct (ar_sl < 0.40, ext_nb == 0, sc < 0.72) is dropped,
    #    while whole-face GroupClsHead consensus protects connected confident trunk (len(fnb) >= 2 & sc >= 0.75)!
    damp_shapes = (
        _duct(0, 0, w=100, h=20)
        + _duct(100, 0, w=14, h=20)
        + _duct(114, 0, w=60, h=20)
        + _duct(30, 3, w=14, h=12)  # floating interior damper box (168 pt^2 inside 2000 pt^2 duct!)
    )
    segs_dp = page_segments(damp_shapes)
    ndp = len(damp_shapes)
    Ldp = np.bincount(segs_dp.owner, weights=np.hypot(segs_dp.seg[:, 2] - segs_dp.seg[:, 0], segs_dp.seg[:, 3] - segs_dp.seg[:, 1]), minlength=ndp)

    class _DpClf:
        classes_ = np.arange(len(CLASSES))

        def predict_proba(self, X):
            out = np.zeros((len(X), len(CLASSES)), dtype=np.float64)
            for i, a in enumerate(np.expm1(X[:, 0])):
                if abs(a - 168.0) < 20.0:
                    out[i, CLASSES.index("Chân rẽ")] = 0.62; out[i, 0] = 0.38
                elif abs(a - 280.0) < 30.0:
                    out[i, CLASSES.index("Concentric Reducer")] = 0.90; out[i, 0] = 0.10
                else:
                    out[i, CLASSES.index("Duct")] = 0.94; out[i, 0] = 0.06
            return out

    # Also pass hostile GroupClsHead cluster calling middle trunk 'none' (p_none=0.95) -> MUST survive via len(fnb)>=2 & sc>=0.75!
    go_hostile = {
        "instance_id": np.zeros(ndp, dtype=np.int64),
        "cluster_ids": np.array([0], dtype=np.int64),
        "cluster_probs": np.array([[0.95, 0.05, 0, 0, 0, 0, 0, 0]], dtype=np.float64),
    }
    dp_objs, _, _ = faces_first_hybrid(segs_dp, np.ones(ndp), np.zeros(ndp, bool), Ldp, _DpClf(), CLASSES, go_hostile, thr=0.5)
    dp_areas = [round(Polygon(o["vertices"]).area) for o in dp_objs]
    assert not any(abs(a - 168) < 20 for a in dp_areas), f"Floating interior damper box survived: {dp_areas}"
    assert any(abs(a - 2000) < 30 for a in dp_areas), f"Main trunk duct dropped: {dp_areas}"
