"""Tests for openstroke.line_decoder (FIRE & PLUMBING direct vector-to-pipe-line decoding, 0 layers)."""
from types import SimpleNamespace

import numpy as np

from openstroke.line_decoder import decode_lines, score_lines_length


def _segs(rows):
    arr = np.asarray(rows, np.float64).reshape(-1, 4)
    return SimpleNamespace(seg=arr, owner=np.arange(len(arr), dtype=np.int64))


def test_dashed_pipe_merges_and_t_junction_splits_cleanly():
    # Horizontal trunk dashed into 3 segments (gap=10pt each): (0,0)-(30,0), (40,0)-(70,0), (80,0)-(110,0)
    # Vertical branch meeting trunk at (55,0)-(55,40) -> must split trunk into 2 halves + produce 1 Tee!
    segs = _segs([
        [0, 0, 30, 0],
        [40, 0, 70, 0],
        [80, 0, 110, 0],
        [55, 0, 55, 40],
        [200, 200, 220, 200],  # background stroke (prob 0.1 -> ignored)
    ])
    prob = np.array([0.9, 0.9, 0.9, 0.9, 0.1])
    out = decode_lines(segs, prob, thr=0.5, mode="fire", gap=12.0)
    assert len(out["lines"]) == 3   # left trunk (0..55), right trunk (55..110), vertical branch (55,0..40)
    fittings = [f["shape_name"] for f in out["fittings"]]
    assert fittings == ["Tee"]

    gt = [{"vertices": [[0, 0], [55, 0]]}, {"vertices": [[55, 0], [110, 0]]}, {"vertices": [[55, 0], [55, 40]]}]
    sc = score_lines_length(out["lines"], gt)
    assert sc["f2"] == 1.0 and sc["precision"] == 1.0 and sc["recall"] == 1.0


def test_x_crossing_is_not_split_and_elbow_is_detected():
    # Two pipes crossing in X (no endpoint at intersection -> NOT split, NOT a Cross)
    # Plus an L-bend at (100, 0) -> 1 Elbow!
    segs = _segs([
        [0, 20, 40, 20],      # horizontal crossing
        [20, 0, 20, 40],      # vertical crossing
        [80, 0, 100, 0],      # elbow arm 1
        [100, 0, 100, 25],    # elbow arm 2
        [100, 25, 100, 26.2], # tiny 1.2pt dangling tick -> pruned (< MIN_STUB)
    ])
    prob = np.ones(5)
    out = decode_lines(segs, prob, thr=0.5, mode="plumbing")
    assert len(out["lines"]) == 4   # 2 crossing lines (uncut!) + 2 elbow arms; tiny tick pruned
    fittings = [f["shape_name"] for f in out["fittings"]]
    assert fittings == ["Elbow"]


def test_plumbing_wide_dash_gaps_up_to_20pt_bridge_cleanly():
    segs = _segs([
        [0, 0, 15, 0],
        [33, 0, 48, 0],   # 18 pt gap!
        [66, 0, 81, 0],   # 18 pt gap!
    ])
    prob = np.ones(3)
    out = decode_lines(segs, prob, thr=0.5, mode="plumbing")
    assert len(out["lines"]) == 1
    assert abs(out["lines"][0]["length"] - 81.0) < 0.1
