import numpy as np

from openstroke.assemble import assign_members, dedupe_boxes, poly_area

SQ = lambda x0, y0, s: np.array([[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s]], float)


def test_score_order_is_the_old_rule():
    L = np.ones(6)
    boxes = [(1, SQ(0, 0, 10), 0.9, [0, 1, 2, 3]), (2, SQ(2, 2, 2), 0.5, [2, 3, 4])]
    pi = assign_members(boxes, L, order="score", dedupe=False)
    old = np.full(6, -1)
    for k in np.argsort([b[2] for b in boxes]):  # obb_score_probe before E84: the highest score is written last
        old[boxes[k][3]] = k
    assert pi.tolist() == old.tolist()


def test_small_box_wins_and_duplicates_drop():
    L = np.ones(6)
    boxes = [(1, SQ(0, 0, 10), 0.9, [0, 1, 2, 3]), (2, SQ(2, 2, 2), 0.5, [2, 3]), (1, SQ(0, 0, 9), 0.4, [0, 1, 2])]
    pi = assign_members(boxes, L, order="small", dedupe=True)
    assert pi.tolist() == [0, 0, 1, 1, -1, -1]  # box 2 (same class, strokes held by box 0) is gone; the small box 1 keeps 2, 3
    assert poly_area(SQ(0, 0, 3)) == 9.0


def test_dedupe_boxes_by_area():
    boxes = [(1, SQ(0, 0, 10), 0.9, []), (1, SQ(1, 1, 5), 0.3, []), (2, SQ(1, 1, 5), 0.3, [])]
    assert dedupe_boxes(boxes) == [0, 2]
