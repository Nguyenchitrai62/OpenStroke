"""Signed grouping invariants and a missing-page regression for object scoring."""
from types import SimpleNamespace

import numpy as np

from openstroke.object_eval import piece_counts
from openstroke.stroke_groups import stroke_instances


def test_stronger_repulsion_survives_component_merges_and_payload_gaps():
    prob = np.array([0.9, 0.1, 0.8, 1.0, 0.9, 0.9])
    pairs = np.array([[0, 2], [3, 4], [2, 3], [2, 4], [1, 5]])
    logits = np.array([4.0, 3.0, 1.0, -5.0, 10.0])
    ids = stroke_instances(prob, 0.75, pairs, logits)
    assert ids[0] == ids[2] and ids[3] == ids[4] and ids[2] != ids[3]
    assert ids[1] == -1 and ids[5] not in ids[:5]
    # Input order and orientation are immaterial.
    np.testing.assert_array_equal(ids, stroke_instances(prob, 0.75, pairs[::-1, ::-1], logits[::-1]))


def test_zero_and_equal_confidence_edges():
    pairs = np.array([[0, 1], [1, 2], [0, 2], [3, 4]])
    ids = stroke_instances(np.ones(5), 0.5, pairs, [1.0, 1.0, -1.0, 0.0])
    assert ids[0] == ids[1] and ids[0] != ids[2] and ids[3] == ids[4]


def test_unpredicted_gt_page_counts_as_misses():
    page = SimpleNamespace(stem="gt", label=np.ones(2), ocls=np.array([1, 2]), oinst=np.array([10, 13]),
                           pieces={"pid": np.array([0]), "pfg": np.array([1]), "pinst": np.array([10]),
                                   "pcls": np.array([1]), "plen": np.array([5.0])})
    q = {"pid": np.empty(0, int), "fg": np.empty(0), "cls": np.empty((0, 0)), "pe": np.empty((0, 2), int),
         "plink": np.empty(0)}
    by_class = {}
    np.testing.assert_array_equal(piece_counts(page, q, by_class=by_class), [0, 0, 2])
    np.testing.assert_array_equal(by_class[2], [0, 0, 1])
