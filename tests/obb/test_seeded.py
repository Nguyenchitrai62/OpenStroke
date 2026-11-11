"""OpenStroke13 / E92: detector boxes as seeds grown along the stroke graph inside the box (docs/OPENSTROKE13.md, H107)."""
import sys
from pathlib import Path

import numpy as np

from openstroke.obb.models.stroke_graph import GraphPage
from openstroke.obb.seeded import seeded_objects
from openstroke.obb.vector_validation import DatasetVectorGT

sys.path.insert(0, str(Path(__file__).parent))
from test_vector_validation import make_page  # noqa: E402


def _vec(rows):
    """rows: (x0, y0, x1, y1) per primitive (page px) -> vector npz dict with pid = 10 + index."""
    tok = np.zeros((len(rows), 24), np.float32)
    for i, (x0, y0, x1, y1) in enumerate(rows):
        tok[i, 1:17] = np.linspace([x0, y0], [x1, y1], 8).ravel()
    tok[:, 22] = -1
    tok[:, 23] = 10 + np.arange(len(rows))
    return {"tokens": tok}


def _page(edges, background=()):
    gp = GraphPage(2)
    pid = np.arange(10, 16)
    prob = np.tile([0.1, 0.8, 0.1], (6, 1))
    for b in background:
        prob[b - 10] = [0.9, 0.05, 0.05]
    e = np.asarray([[a - 10, b - 10] for a, b, _ in edges], np.int64).reshape(-1, 2)
    gp.add(pid, prob, e, np.asarray([q for _, _, q in edges], float))
    return gp


ROWS = [(0, 0, 10, 0), (10, 0, 20, 0), (20, 0, 30, 0), (30, 0, 40, 0), (200, 0, 210, 0), (12, 0, 18, 0)]
BOX = {"class": "Duct", "class_id": 0, "score": 0.9, "poly": [0, -5, 25, -5, 25, 5, 0, 5], "members": [10]}


def test_grows_inside_the_box_only():
    gp = _page([(10, 11, 0.95), (11, 12, 0.9), (12, 13, 0.95), (13, 14, 0.99)])
    out = seeded_objects(gp, [BOX], _vec(ROWS), ["Duct", "Elbow"], edge_thr=0.5, margin=2.0)
    assert len(out) == 1 and out[0]["source"] == "seeded" and out[0]["class_id"] == 0 and out[0]["score"] == 0.9
    assert out[0]["members"] == [10, 11, 12]          # 13 (middle at x 35) and 14 lie outside the box grown by 2 px
    assert out[0]["seed_members"] == 1


def test_edge_threshold_and_background_strokes_stop_growth():
    gp = _page([(10, 11, 0.6), (11, 12, 0.95), (10, 15, 0.99)], background=(15,))
    assert seeded_objects(gp, [BOX], _vec(ROWS), ["Duct", "Elbow"], edge_thr=0.7, margin=2.0)[0]["members"] == [10]
    assert seeded_objects(gp, [BOX], _vec(ROWS), ["Duct", "Elbow"], edge_thr=0.5, margin=2.0)[0]["members"] == [10, 11, 12]


def test_box_without_known_members_is_skipped():
    gp = _page([])
    assert seeded_objects(gp, [dict(BOX, members=[999])], _vec(ROWS), ["Duct", "Elbow"]) == []


def test_scorer_picks_rows_by_source(tmp_path):
    images, _, stem = make_page(tmp_path)
    gt = DatasetVectorGT(images, ["Duct", "Elbow"], comparison_pages=("doc000022",))
    rows = [dict(class_id=0, score=0.8, members=[10, 11], source="seeded@0.7"), dict(class_id=1, score=0.8, members=[30], source="seeded@0.7"),
            dict(class_id=0, score=0.8, members=[99], source="seeded@0.5"),
            dict(class_id=1, score=0.8, members=[30]), dict(class_id=0, score=0.8, members=[10, 11], source="graph")]
    assert gt.evaluate({stem: rows}, "seeded@0.7", ["Duct", "Elbow"])["R"] == 1.0
    assert gt.evaluate({stem: rows}, "seeded@0.5", ["Duct", "Elbow"])["R"] == 0.0   # a background stroke only
    assert gt.evaluate({stem: rows}, "members", ["Duct", "Elbow"])["R"] == 0.5      # only the source-less box row (the elbow)
    assert gt.evaluate({stem: rows}, "graph", ["Duct", "Elbow"])["R"] == 0.5        # only the graph row (the duct)


def test_validator_scores_every_path_and_keeps_the_best_seeded(tmp_path, monkeypatch):
    import functools

    import openstroke.obb.vector_validation as vv
    from openstroke.obb.object_validation import VectorEpochEvaluator

    monkeypatch.setattr(vv, "DatasetVectorGT", functools.partial(vv.DatasetVectorGT, comparison_pages=("doc000022",)))
    images, _, stem = make_page(tmp_path)
    ev = VectorEpochEvaluator(None, images, ["Duct", "Elbow"], {"stroke_graph": True, "graph_decode": "seeded", "batch": 8})
    try:
        assert ev.seeded_edges() == (0.5, 0.7, 0.9)
        rows = [dict(class_id=0, score=0.8, members=[10, 11], source="seeded@0.9"), dict(class_id=1, score=0.8, members=[30], source="seeded@0.9"),
                dict(class_id=0, score=0.8, members=[99], source="seeded@0.5"), dict(class_id=0, score=0.8, members=[99], source="seeded@0.7"),
                dict(class_id=1, score=0.8, members=[30]), dict(class_id=0, score=0.8, members=[10, 11], source="graph")]
        s = ev.paths({stem: rows})
        assert s["how"] == "seeded@0.9" and s["F2"] == 1.0
        assert set(s["paths"]) == {"members", "graph", "seeded@0.5", "seeded@0.7", "seeded@0.9"}
        assert s["paths"]["members"]["R"] == 0.5 and s["paths"]["seeded@0.5"]["R"] == 0.0
    finally:
        ev.close()
    off = VectorEpochEvaluator(None, images, ["Duct", "Elbow"], {"stroke_graph": True, "batch": 8})
    try:
        assert off.seeded_edges() is None                     # default decoding is unchanged
    finally:
        off.close()
