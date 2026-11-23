"""E139 / H158: augmented copies cut and join PDF items without changing labels, length or geometry (CPU, no training)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def _page():
    # shape 0: polyline of 3 segments; shape 1 + 2: two lines meeting end to start, same style; shape 3: a fill
    seg = np.array([[0, 0, 10, 0], [10, 0, 10, 10], [10, 10, 20, 10],
                    [30, 0, 40, 0], [40, 0, 50, 0], [0, 50, 5, 50]], np.float32)
    off = np.array([0, 3, 4, 5, 6], np.int64)
    n = 4
    L = np.array([30, 10, 10, 5], np.float32)
    return {"meta": np.array("m"), "segments": seg, "shape_offsets": off, "length": L,
            "label": np.array([1, 0, 0, 0], np.uint8), "type_code": np.array([0, 0, 0, 1], np.int8),
            "layer_id": np.zeros(n, np.int32), "layer_names": np.array([""]), "stroke_rgb": np.zeros((n, 3), np.float32),
            "fill_rgb": np.full((n, 3), np.nan, np.float32), "width": np.ones(n, np.float32), "close_path": np.zeros(n, np.uint8),
            "op_counts": np.tile(np.array([[1, 0, 0, 0]], np.int32), (n, 1)), "bbox": np.zeros((n, 4), np.float32),
            "oinst": np.array([0, -1, -1, -1], np.int32), "pc_shape": np.zeros(3, np.int32)}


def test_split_and_merge_keep_total_length_labels_and_segments():
    import make_split_aug as m

    m.SPLIT_SHARE, m.MERGE_SHARE, m.MIN_LEN = 1.0, 1.0, 1.0
    d = _page()
    out, c = m.augment_page(d, np.random.default_rng(0))
    assert "pc_shape" not in out
    off = out["shape_offsets"]
    assert off[-1] == len(out["segments"]) and c["after"] == len(off) - 1
    assert c["after"] == c["shapes"] + c["split"] - c["merged"]
    # total length per label is preserved
    for lab in (0, 1):
        assert np.isclose(out["length"][out["label"] == lab].sum(), d["length"][d["label"] == lab].sum(), atol=1e-3)
    # the fill is never cut
    assert (out["type_code"] == 1).sum() == 1
    # every output shape's bbox covers its segments
    for i in range(len(off) - 1):
        s = out["segments"][off[i]:off[i + 1]]
        assert out["bbox"][i, 0] <= s[:, [0, 2]].min() + 1e-5 and out["bbox"][i, 2] >= s[:, [0, 2]].max() - 1e-5


def test_merge_joins_meeting_strokes_of_one_style():
    import make_split_aug as m

    m.SPLIT_SHARE, m.MERGE_SHARE = 0.0, 1.0
    d = _page()
    out, c = m.augment_page(d, np.random.default_rng(0))
    assert c["merged"] == 1 and c["after"] == 3
    assert np.isclose(out["length"][1], 20.0)
    assert out["op_counts"][1, 0] == 2
