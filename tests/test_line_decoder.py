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


def test_short_1_8pt_dashes_near_corner_merge_without_collapsing():
    """F234: 1.8pt dashes (shorter than NODE_TOL=2.0) approaching a 90-deg corner must bridge cleanly and not collapse."""
    segs = _segs([
        [0.0, 0.0, 1.8, 0.0],
        [2.7, 0.0, 4.5, 0.0],
        [5.4, 0.0, 7.2, 0.0],
        [8.1, 0.0, 9.9, 0.0],
        [9.9, 0.0, 9.9, 1.8],
        [9.9, 2.7, 9.9, 4.5],
        [9.9, 5.4, 9.9, 7.2],
        [0.0, 0.0, 0.2, 0.1],   # tiny 0.22pt chamfer dot must NOT block dash merge
    ])
    out = decode_lines(segs, np.ones(8), thr=0.5, mode="fire")
    lens = sorted(round(l["length"], 1) for l in out["lines"])
    assert lens == [7.2, 9.9]
    assert [f["shape_name"] for f in out["fittings"]] == ["Elbow"]


def test_filled_double_wall_pipe_collapses_to_centerline():
    """F234: two opposite edges of a 1.8pt thick filled pipe average to exact centerline (not outer wall)."""
    segs = _segs([
        [0.0, 0.0, 100.0, 0.0],
        [0.0, 1.8, 100.0, 1.8],
    ])
    out = decode_lines(segs, np.ones(2), thr=0.5, mode="fire")
    assert len(out["lines"]) == 1
    v = out["lines"][0]["vertices"]
    assert abs(v[0][1] - 0.9) < 0.05 and abs(v[1][1] - 0.9) < 0.05


def test_thin_line_counts_and_summarize_lines(tmp_path):
    """F234: object_eval.thin_line_counts scores collinear pipe-length F2 on fire/plumbing npz without SQLite."""
    from types import SimpleNamespace

    from openstroke.object_eval import line_f2, summarize_lines, thin_line_counts

    npz = tmp_path / "doc1_p0001_gt1.npz"
    segs = np.array([[0.0, 0.0, 50.0, 0.0], [55.0, 0.0, 100.0, 0.0], [0.0, 50.0, 40.0, 50.0]], dtype=np.float32)
    np.savez(npz, segments=segs, shape_offsets=np.array([0, 1, 2, 3], np.int32), label=np.array([1, 1, 0], np.uint8))
    page = SimpleNamespace(stem="doc1_p0001_gt1", mode="fire", index=np.array([2, 0, 1], np.int64))
    # Permuted token order: shape 2 -> 0.1, shape 0 -> 0.9, shape 1 -> 0.9
    cnt = thin_line_counts(page, np.array([0.1, 0.9, 0.9]), 0.5, npz)
    assert cnt is not None
    f2_val, prec, rec = line_f2(cnt)
    assert f2_val > 0.999 and prec > 0.999 and rec > 0.999
    s = summarize_lines({"doc1_p0001_gt1": cnt}, {"doc1_p0001_gt1": "fire:gt"})
    assert s["fire:gt"]["f2"] == 1.0 and s["all"]["gt_len"] == 100.0


def test_decode_fire_network_3class_and_closed_perimeter_pruning():
    """FIRE 3-class post-processing preserves main/branch classes & fittings and prunes isolated closed 0-leaf frames."""
    from openstroke.line_decoder import decode_fire_network

    segs = _segs([
        # Real sprinkler network: horizontal main trunk (0..100, y=0) + vertical branch off Tee at x=50 + elbow at x=100
        [0.0, 0.0, 100.0, 0.0],       # 0: main trunk
        [50.0, 0.0, 50.0, 40.0],      # 1: branch off Tee
        [100.0, 0.0, 100.0, 30.0],    # 2: main elbow turn
        # Isolated closed 200x100 site-boundary frame far away (600 pt perimeter, pure main, 0 branches, 0 leaf nodes)
        [300.0, 300.0, 500.0, 300.0], # 3
        [500.0, 300.0, 500.0, 400.0], # 4
        [500.0, 400.0, 300.0, 400.0], # 5
        [300.0, 400.0, 300.0, 300.0], # 6
        # Background stroke
        [0.0, 200.0, 50.0, 200.0],    # 7
    ])
    cls_prob = np.array([
        [0.05, 0.90, 0.05],  # 0: main
        [0.05, 0.10, 0.85],  # 1: branch
        [0.05, 0.88, 0.07],  # 2: main
        [0.05, 0.92, 0.03],  # 3: frame main FP
        [0.05, 0.92, 0.03],  # 4: frame main FP
        [0.05, 0.92, 0.03],  # 5: frame main FP
        [0.05, 0.92, 0.03],  # 6: frame main FP
        [0.95, 0.03, 0.02],  # 7: other
    ], dtype=np.float32)
    prob = 1.0 - cls_prob[:, 0]
    length = np.array([100.0, 40.0, 30.0, 200.0, 100.0, 200.0, 100.0, 50.0], dtype=np.float64)

    out = decode_fire_network(segs, prob, cls_prob, length, thr=0.30, class_names=("other", "main", "branch"))
    assert out["dropped_loops"] == 1
    assert out["counts"]["main"] == 3   # split trunk (0..50, 50..100) + main elbow arm (100,0..30)
    assert out["counts"]["branch"] == 1
    assert out["counts"].get("Tee") == 1 and out["counts"].get("Elbow") == 1
    assert abs(out["lengths"]["main"] - 130.0) < 0.2
    assert abs(out["lengths"]["branch"] - 40.0) < 0.2
    # Frame strokes (3..6) and background (7) must have cls == 0 and inst == -1
    assert np.all(out["cls"][3:] == 0) and np.all(out["inst"][3:] == -1)
    assert out["cls"][0] == 1 and out["cls"][1] == 2 and out["cls"][2] == 1

    # Negative page (all below threshold) -> clean empty response
    empty = decode_fire_network(segs, np.zeros(8), cls_prob, length, thr=0.30)
    assert empty["lines"] == [] and empty["fittings"] == [] and empty["dropped_loops"] == 0


def test_decode_fire_network_v2_hysteresis_and_2core_ring_smoothing():
    """FIRE Post-Proc v2 recovers weak bridging strokes [0.10, thr) and closes majority-main 2-core rings without leaking to perpendicular walls."""
    from openstroke.line_decoder import decode_fire_network

    segs = _segs([
        # 120x80 main ring with 3 confident main sides + 1 weak-main/branch-flipped top gap + 1 branch drop + 1 perpendicular wall (weak)
        [0.0, 0.0, 120.0, 0.0],       # 0: bottom main (conf)
        [120.0, 0.0, 120.0, 80.0],    # 1: right main (conf)
        [0.0, 80.0, 0.0, 0.0],        # 2: left main (conf)
        [0.0, 80.0, 60.0, 80.0],      # 3: top-left weak bridge (p=0.18 < 0.30, bridges node (0,80) & (60,80)!)
        [60.0, 80.0, 120.0, 80.0],    # 4: top-right mislabeled branch (p_main=0.25, p_branch=0.65)
        [120.0, 40.0, 160.0, 40.0],   # 5: real sprinkler branch sticking out right (leaf -> MUST stay branch!)
        [60.0, 0.0, 60.0, -50.0],     # 6: perpendicular weak wall touching bottom main (p=0.18 -> MUST NOT be rescued!)
    ])
    cls_prob = np.array([
        [0.05, 0.90, 0.05],  # 0: main
        [0.05, 0.90, 0.05],  # 1: main
        [0.05, 0.90, 0.05],  # 2: main
        [0.82, 0.14, 0.04],  # 3: weak bridge p=0.18 (rescued by hysteresis!)
        [0.10, 0.25, 0.65],  # 4: ring gap mislabeled branch (repaired to main by 2-core / main-bridge!)
        [0.05, 0.05, 0.90],  # 5: leaf branch (stays branch!)
        [0.82, 0.09, 0.09],  # 6: perpendicular weak wall (stays dropped!)
    ], dtype=np.float32)
    prob = 1.0 - cls_prob[:, 0]
    length = np.array([120.0, 80.0, 80.0, 60.0, 60.0, 40.0, 50.0], dtype=np.float64)

    out = decode_fire_network(segs, prob, cls_prob, length, thr=0.30, class_names=("other", "main", "branch"))
    assert out["dropped_loops"] == 0   # has real branch -> NOT dropped!
    assert out["cls"][3] == 1          # weak bridge rescued AND classified main!
    assert out["cls"][4] == 1          # ring/bridge segment repaired from branch -> main!
    assert out["cls"][5] == 2          # leaf branch stays branch!
    assert out["cls"][6] == 0          # perpendicular weak wall NOT rescued!
    assert abs(out["lengths"]["main"] - 400.0) < 0.5
    assert abs(out["lengths"]["branch"] - 40.0) < 0.5


def test_hvac_app_fire_logic_schema():
    """Web viewer backend run_logic() supports selectable post-processing systems (fire / plumbing / none) for any class set."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("hvac_app_test", Path(__file__).resolve().parents[1] / "app" / "hvac_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        segs = _segs([[0.0, 0.0, 80.0, 0.0], [40.0, 0.0, 40.0, 30.0]])
        cls_prob = np.array([[0.05, 0.90, 0.05], [0.05, 0.08, 0.87]], dtype=np.float32)
        job = {
            "doc": "d0", "page": 1,
            "data": {
                "prob": (1.0 - cls_prob[:, 0]).astype(np.float32),
                "cls_prob": cls_prob,
                "class_names": ["other", "main", "branch"],
                "segs": segs,
                "length": np.array([80.0, 30.0], dtype=np.float64),
            },
        }
        mod.APP["info"] = {"is_multiclass": True, "is_object_model": False, "class_names": ["other", "main", "branch"]}
        # Default post_system is 'none' (user picks HVAC / FIRE / PLUMBING explicitly)
        res_def = mod.run_logic(job, 0.30)
        assert res_def["kind"] == "none" and res_def["objects"] == 0

        res = mod.run_logic(job, 0.30, system="fire")
        assert res["kind"] == "line" and res["system"] == "fire"
        assert res["objects"] == 3 and res["fittings_count"] == 1
        assert res["counts"]["main"] == 2 and res["counts"]["branch"] == 1 and res["counts"]["Tee"] == 1
        assert len(res["lines"]) == 3 and len(res["fittings"]) == 1
        assert res["classes"] == ["other", "main", "branch"]

        # Selectable plumbing mode works even on custom class names (model-agnostic)
        job["data"]["class_names"] = ["other", "cold", "hot"]
        res_p = mod.run_logic(job, 0.30, system="plumbing")
        assert res_p["kind"] == "line" and res_p["system"] == "plumbing"
        assert res_p["counts"]["cold"] == 2 and res_p["counts"]["hot"] == 1

        # Selectable none mode disables post-processing cleanly
        res_n = mod.run_logic(job, 0.30, system="none")
        assert res_n["kind"] == "none" and res_n["objects"] == 0
    finally:
        import shutil
        shutil.rmtree(mod.WORK, ignore_errors=True)


def test_v3_auto_accept_style_elbow_midspan_tee_dash_gap_and_guards():
    """FIRE Post-Proc v3 auto-accepts same-style dash-gap, 90-deg elbow, mid-span Tee & filled ribbon,
    while blocking different-style walls, 4th-harmonic circles, and isolated low-conf FP anchor growth."""
    import math
    from openstroke.line_decoder import decode_fire_network

    # Build a 16-segment circle of radius 15pt centered at (50, 15) touching (50, 0)
    circle_segs = []
    for k in range(16):
        a0 = 2.0 * math.pi * k / 16.0
        a1 = 2.0 * math.pi * (k + 1) / 16.0
        circle_segs.append([50.0 + 15.0 * math.cos(a0), 15.0 + 15.0 * math.sin(a0),
                            50.0 + 15.0 * math.cos(a1), 15.0 + 15.0 * math.sin(a1)])

    rows = [
        [0.0, 0.0, 100.0, 0.0],     # 0: confident red trunk anchor (main)
        [100.0, 0.0, 100.0, 40.0],  # 1: confident red elbow arm (makes anchor component n_segs >= 2!)
        [110.0, 0.0, 150.0, 0.0],   # 2: missing same-style collinear dash across 10pt gap (prob=0.05 -> RESCUE)
        [150.0, 0.0, 150.0, 35.0],  # 3: missing same-style 90-deg elbow at end of rescued dash (prob=0.05 -> RESCUE)
        [50.0, 0.0, 50.0, -40.0],   # 4: missing same-style mid-span Tee branch (prob=0.05 -> RESCUE)
        [30.0, 0.0, 30.0, -50.0],   # 5: thin black architectural wall touching mid-span (prob=0.08 -> BLOCK!)
        [0.0, 0.0, -40.0, 0.0],     # 6: filled red main ribbon (tc=2, w=0.0) touching trunk (prob=0.04 -> RESCUE!)
        [400.0, 400.0, 440.0, 400.0],  # 7: isolated weak FP seed (prob=0.35 -> NOT trusted anchor!)
        [440.0, 400.0, 480.0, 400.0],  # 8: same-style neighbor of isolated FP seed (prob=0.06 -> MUST NOT GROW!)
    ] + circle_segs                 # 9: 16-seg red coverage circle touching (50,0) (prob=0.06 -> BLOCK by coh4!)

    owners = [0, 1, 2, 3, 4, 5, 6, 7, 8] + [9] * 16
    segs = SimpleNamespace(seg=np.asarray(rows, np.float64), owner=np.asarray(owners, np.int64))
    n_sh = 10
    cls_prob = np.array([
        [0.05, 0.90, 0.05],  # 0: conf main
        [0.05, 0.85, 0.10],  # 1: conf main
        [0.95, 0.04, 0.01],  # 2: weak dash gap
        [0.95, 0.01, 0.04],  # 3: weak elbow
        [0.95, 0.01, 0.04],  # 4: weak midspan tee
        [0.92, 0.04, 0.04],  # 5: thin black wall
        [0.96, 0.035, 0.005],# 6: filled red ribbon
        [0.65, 0.10, 0.25],  # 7: isolated FP seed (p=0.35)
        [0.94, 0.02, 0.04],  # 8: neighbor of FP seed (p=0.06)
        [0.94, 0.03, 0.03],  # 9: coverage circle (p=0.06)
    ], dtype=np.float32)
    prob = 1.0 - cls_prob[:, 0]
    dxy = segs.seg[:, 2:4] - segs.seg[:, 0:2]
    sl = np.hypot(dxy[:, 0], dxy[:, 1])
    length = np.bincount(segs.owner, weights=sl, minlength=n_sh)
    attrs = {
        "type_code": np.array([0, 0, 0, 0, 0, 0, 2, 0, 0, 0], dtype=np.int8),
        "stroke_rgb": np.array([
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        ], dtype=np.float32),
        "fill_rgb": np.array([
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        ], dtype=np.float32),
        "width": np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.15, 0.0, 0.8, 0.8, 0.8], dtype=np.float32),
    }

    out = decode_fire_network(segs, prob, cls_prob, length, thr=0.30, class_names=("other", "main", "branch"), attrs=attrs)
    assert out["cls"][2] > 0, "Collinear dash across 10pt gap must be auto-accepted"
    assert out["cls"][3] > 0, "90-deg same-style elbow off rescued dash must be auto-accepted"
    assert out["cls"][4] > 0, "Mid-span same-style Tee branch must be auto-accepted"
    assert out["cls"][6] > 0, "Same-color filled ribbon touching stroked pipe must be auto-accepted"
    assert out["cls"][5] == 0, "Different-style perpendicular wall must NOT be rescued"
    assert out["cls"][8] == 0, "Isolated low-confidence FP must NOT seed auto-accept growth"
    assert out["cls"][9] == 0, "16-segment circle must be rejected by 4th-harmonic coherence guard"


def test_v3_class_barrier_and_contributed_length_weighting():
    """Giant multi-segment main shapes must NOT contaminate inline touching branches via shape-total length weighting."""
    from openstroke.line_decoder import decode_fire_network

    # Shape 0 is a giant 4-segment main path (1,000pt total) ending collinearly at (100, 0)
    # Shape 1 is a 40pt branch continuing collinearly from (100, 0) to (140, 0)
    rows = [
        [0.0, 0.0, 100.0, 0.0],       # shape 0 seg 0
        [0.0, 0.0, 0.0, 300.0],       # shape 0 seg 1
        [0.0, 300.0, 300.0, 300.0],   # shape 0 seg 2
        [300.0, 300.0, 300.0, 0.0],   # shape 0 seg 3
        [100.0, 0.0, 140.0, 0.0],     # shape 1: collinear inline branch continuation!
    ]
    segs = SimpleNamespace(seg=np.asarray(rows, np.float64), owner=np.array([0, 0, 0, 0, 1], np.int64))
    cls_prob = np.array([[0.01, 0.98, 0.01], [0.01, 0.01, 0.98]], dtype=np.float32)
    prob = 1.0 - cls_prob[:, 0]
    length = np.array([1000.0, 40.0], dtype=np.float64)

    out = decode_fire_network(segs, prob, cls_prob, length, thr=0.30, class_names=("other", "main", "branch"))
    assert out["cls"][0] == 1 and out["cls"][1] == 2
    assert abs(out["lengths"]["branch"] - 40.0) < 0.2
    assert abs(out["lengths"]["main"] - 1000.0) < 0.5
    # Straight 180-deg main->branch transition must NOT emit a spurious Elbow at (100, 0)
    elbow_pts = [f["point"] for f in out["fittings"] if f["shape_name"] == "Elbow"]
    assert not any(abs(p[0] - 100.0) < 1.0 and abs(p[1] - 0.0) < 1.0 for p in elbow_pts)


def test_v3_virtual_valve_gap_2core_ring_and_collinear_chain_bridge():
    """Virtual valve-gap (<= 8pt) closes 2-core hydraulic ring mains without synthesizing fake geometry,
    and collinear multi-edge main-to-main chain bridges repair split mains while preserving exact Oracle."""
    from openstroke.line_decoder import decode_fire_network

    # 120x80 ring main with a 5pt open valve gap at top-center (57.5..62.5, y=80)
    # Bottom + right are main; left + top-left + top-right are misclassified branch (pM=0.015, pB=0.85)
    # Plus 3 leaf sprinkler branches hanging off the ring -> MUST stay branch!
    # Plus a horizontal 2-edge branch chain split by a Tee between two main stubs on the right.
    segs = _segs([
        [0.0, 0.0, 120.0, 0.0],       # 0: bottom main (120)
        [120.0, 0.0, 120.0, 80.0],    # 1: right main (80)
        [0.0, 0.0, 0.0, 80.0],        # 2: left side misclassified branch (80)
        [0.0, 80.0, 57.5, 80.0],      # 3: top-left misclassified branch (57.5) -- 5pt valve gap to #4!
        [62.5, 80.0, 120.0, 80.0],    # 4: top-right misclassified branch (57.5)
        [40.0, 0.0, 40.0, -35.0],     # 5: leaf branch 1
        [80.0, 0.0, 80.0, -35.0],     # 6: leaf branch 2
        [0.0, 40.0, -35.0, 40.0],     # 7: leaf branch 3
        # Collinear 2-edge chain bridge from (120, 40) -> (180, 40) -> (240, 40) -> main stub (240, 40..90)
        [120.0, 40.0, 180.0, 40.0],   # 8: chain edge 1 (misclassified branch)
        [180.0, 40.0, 240.0, 40.0],   # 9: chain edge 2 (misclassified branch)
        [180.0, 40.0, 180.0, 10.0],   # 10: intermediate Tee leaf branch (MUST stay branch!)
        [240.0, 40.0, 240.0, 90.0],   # 11: far main anchor
    ])
    cls_prob = np.array([
        [0.05, 0.90, 0.05],  # 0: main
        [0.05, 0.90, 0.05],  # 1: main
        [0.10, 0.015, 0.885],# 2: ring branch -> main
        [0.10, 0.015, 0.885],# 3: ring branch -> main
        [0.10, 0.015, 0.885],# 4: ring branch -> main
        [0.05, 0.01, 0.94],  # 5: leaf branch
        [0.05, 0.01, 0.94],  # 6: leaf branch
        [0.05, 0.01, 0.94],  # 7: leaf branch
        [0.08, 0.06, 0.86],  # 8: chain edge 1 -> main
        [0.08, 0.06, 0.86],  # 9: chain edge 2 -> main
        [0.05, 0.01, 0.94],  # 10: Tee leaf branch
        [0.05, 0.90, 0.05],  # 11: far main anchor
    ], dtype=np.float32)
    prob = 1.0 - cls_prob[:, 0]
    length = np.hypot(segs.seg[:, 2] - segs.seg[:, 0], segs.seg[:, 3] - segs.seg[:, 1])

    out = decode_fire_network(segs, prob, cls_prob, length, thr=0.30, class_names=("other", "main", "branch"), hysteresis=False)
    for idx_m in (0, 1, 2, 3, 4, 8, 9, 11):
        assert out["cls"][idx_m] == 1, f"Edge {idx_m} should be promoted to main"
    for idx_b in (5, 6, 7, 10):
        assert out["cls"][idx_b] == 2, f"Leaf branch {idx_b} must stay branch"

    # Exact Oracle check: when GT branch has p_main == 0.0, ZERO GT branch edges may ever flip!
    gt_y = np.array([1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1], dtype=np.int64)
    oh = np.zeros((len(gt_y), 3), dtype=np.float32)
    oh[np.arange(len(gt_y)), gt_y] = 1.0
    ora = decode_fire_network(segs, np.ones(len(gt_y)), oh, length, thr=0.50, class_names=("other", "main", "branch"))
    assert np.array_equal(ora["cls"], gt_y)


def test_v3_postprocess_lines_extracts_visual_style_layer_free():
    """postprocess_lines automatically extracts visual style (0 PDF layers) from payload shapes for style-verified auto-accept."""
    from openstroke.line_decoder import postprocess_lines

    pay = {
        "shapes": [
            {"items": [["l", [0.0, 0.0], [100.0, 0.0]]], "type": "s", "width": 0.9, "color": [1.0, 0.0, 0.0]},
            {"items": [["l", [100.0, 0.0], [100.0, 40.0]]], "type": "s", "width": 0.9, "color": [1.0, 0.0, 0.0]},
            {"items": [["l", [50.0, 0.0], [50.0, -35.0]]], "type": "s", "width": 0.9, "color": [1.0, 0.0, 0.0]},
            {"items": [["l", [25.0, 0.0], [25.0, -45.0]]], "type": "s", "width": 0.15, "color": [0.0, 0.0, 0.0]},
        ]
    }
    raw = {
        "path": "test.json", "arch": "openstroke46", "n_shapes": 4, "kept": 4, "threshold": 0.30,
        "format": "dense",
        "prob": [0.95, 0.95, 0.05, 0.08],
        "valid": [True, True, True, True],
        "cls_prob": [[0.05, 0.90, 0.05], [0.05, 0.85, 0.10], [0.95, 0.01, 0.04], [0.92, 0.04, 0.04]],
        "class_names": ["other", "main", "branch"],
    }
    out = postprocess_lines(pay, raw, thr=0.30, mode="fire")
    assert out["cls"][2] > 0, "Same-style weak branch must be auto-accepted via payload visual style"
    assert out["cls"][3] == 0, "Different-style thin black wall must be rejected"
