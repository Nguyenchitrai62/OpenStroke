"""E0 converter: geometry, label rules, coordinate check, read-only sources, end-to-end build."""

import gzip
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely import STRtree

from openstroke.build import build_all
from openstroke.geometry import (IDENTITY, ROT90_CCW, calibrate, item_polylines, page_segments,
                                 sample_segments, transform_points)
from openstroke.labels import gt_coverage, gt_geometry, is_true_object, label_shapes, refine_hvac, segment_lines
from openstroke.sources import connect_ro, latest_payload_path


def line(x0, y0, x1, y1):
    return ["l", [x0, y0], [x1, y1]]


def shape(*items, layer="P", width=0.5):
    return {"items": list(items), "type": "s", "color": [1, 0, 0], "width": width, "layer": layer,
            "closePath": False, "dashes": "[] 0", "seqno": 0}


# --- geometry -------------------------------------------------------------------------------------------

def test_bezier_is_sampled_along_the_arc_not_the_chord():
    # Quarter circle of radius 100 as a cubic bezier: arc length 157.08, chord 141.42.
    k = 0.5522847498 * 100
    poly = item_polylines(["c", [100, 0], [100, k], [k, 100], [0, 100]])[0]
    length = np.hypot(*np.diff(poly, axis=0).T).sum()
    assert abs(length - math.pi * 50) < 0.5
    assert length > 150


def test_rect_and_quad_are_closed():
    (r,) = item_polylines(["re", [0, 0, 10, 5], -1])
    (q,) = item_polylines(["qu", [[0, 0], [10, 0], [10, 5], [0, 5]]])
    for p in (r, q):
        assert np.allclose(p[0], p[-1])
        assert abs(np.hypot(*np.diff(p, axis=0).T).sum() - 30) < 1e-9


def test_samples_carry_segment_length():
    seg = np.array([[0, 0, 10, 0], [0, 0, 0, 1000], [5, 5, 5, 5]], dtype=float)
    pts, idx, w = sample_segments(seg)
    assert np.allclose(np.bincount(idx, weights=w), [10, 1000, 0])
    assert (np.bincount(idx) <= 64).all()


def test_page_segments_owner_and_offsets():
    segs = page_segments([shape(line(0, 0, 1, 0), line(1, 0, 1, 1)), shape(), shape(["re", [0, 0, 2, 2], -1])])
    assert list(segs.shape_offsets) == [0, 2, 2, 6]
    assert np.allclose(segs.shape_length(), [2, 0, 8])


# --- label rules ----------------------------------------------------------------------------------------

def test_class_scope():
    assert is_true_object("fire", "Line")
    assert not is_true_object("fire", "Sprinkler") and not is_true_object("fire", "Tee")
    for name in ("Duct", "Elbow 90", "Elbow vuông", "Concentric Reducer vuông tròn", "Eccentric Reducer",
                 "Chân rẽ", "Chân rẽ vuông tròn"):
        assert is_true_object("hvac", name), name
    for name in ("Line", "Fan", "Ống nước ngưng", "Ống đồng"):
        assert not is_true_object("hvac", name), name


def test_plumbing_pipes_follow_the_fire_rule():
    assert is_true_object("plumbing", "Ống chưa rõ loại") and is_true_object("plumbing", "Ống cấp nước")
    for name in ("Lavabol", "Xi_bet", "Pheu_thu_nuoc_san", "Line"):
        assert not is_true_object("plumbing", name), name
    # a 3-vertex pipe polyline is a line along its vertices, not a polygon (HVAC would close it)
    gt = gt_geometry("plumbing", [{"shape_name": "Ống chưa rõ loại", "vertices": [[0, 0], [100, 0], [100, 100]]}])
    assert gt.closed == [False]
    shapes = [shape(line(10, 1, 90, 1)), shape(line(50, 50, 60, 60)), shape(line(101, 20, 101, 80))]
    assert list(label_shapes(page_segments(shapes), gt).label) == [1, 0, 1]


def test_fire_majority_of_length_within_tolerance():
    gt = gt_geometry("fire", [{"shape_name": "Line", "vertices": [[0, 0], [100, 0]], "group": "primary"}])
    shapes = [
        shape(line(10, 1.5, 50, 1.5)),            # on the pipe (1.5 pt off) -> true
        shape(line(10, 5, 50, 5)),                # 5 pt away -> other
        shape(line(80, 0, 120, 0)),               # half on (80..100), half beyond the end
        shape(line(0, 0, 30, 0), line(0, 50, 0, 100)),  # 30 on, 50 off -> other
        shape(line(-10, 0, 0, 0)),                # touches the end only -> other
    ]
    lab = label_shapes(page_segments(shapes), gt)
    assert list(lab.label) == [1, 0, 1, 0, 0]
    assert abs(lab.true_frac[2] - (22 / 40)) < 0.03  # 80..102 inside incl. the 2 pt end cap
    assert abs(lab.true_frac[3] - 30 / 80) < 0.02


def test_fire_dashes_each_short_shape_true():
    gt = gt_geometry("fire", [{"shape_name": "Line", "vertices": [[0, 0], [100, 0]]}])
    dashes = [shape(line(x, 0, x + 3, 0)) for x in range(0, 100, 6)]
    assert label_shapes(page_segments(dashes), gt).label.all()


def test_hvac_inside_polygon_and_walls_on_boundary():
    duct = {"shape_name": "Duct", "vertices": [[0, 0], [200, 0], [200, 40], [0, 40]]}
    gt = gt_geometry("hvac", [duct, {"shape_name": "Fan", "vertices": [[300, 0], [340, 0], [340, 40], [300, 40]]}])
    assert len(gt) == 1  # Fan is out of scope
    shapes = [
        shape(line(0, 0, 200, 0)),          # wall on the boundary -> true
        shape(line(20, 20, 180, 20)),       # centre line inside -> true
        shape(line(100, 20, 100, 140)),     # 20 of 120 inside -> other
        shape(line(310, 20, 330, 20)),      # inside the Fan only -> other
    ]
    assert list(label_shapes(page_segments(shapes), gt).label) == [1, 1, 0, 0]


def test_hvac_refinement_drops_other_disciplines_under_a_duct():
    duct = [[0, 0], [200, 0], [200, 20], [0, 20]]
    gt = gt_geometry("hvac", [{"shape_name": "Duct", "vertices": duct}])
    shapes = [
        shape(line(0, 0, 200, 0), layer="D"),       # duct outline -> true
        shape(line(10, 10, 190, 10), layer="D"),     # centre line on the duct layer (layer mostly inside) -> true
        shape(line(50, 5, 50, 15), layer="A"),       # wall stub under the duct, architecture layer -> other
        shape(line(300, 50, 900, 50), layer="A"),    # the rest of the architecture layer, far away -> other
        shape(line(100, 19, 150, 19), layer="A"),    # architecture line lying on the duct outline -> kept true
    ]
    segs = page_segments(shapes)
    lab = label_shapes(segs, gt)
    assert list(lab.label) == [1, 1, 1, 0, 1]  # the plain "inside" rule
    new, diag = refine_hvac(segs, gt, lab, np.array([0, 0, 1, 1, 1]))
    assert list(new.label) == [1, 1, 0, 0, 1]
    assert diag["outline_frac"][4] > 0.9 and diag["layer_inside_share"][2] < 0.5 < diag["layer_inside_share"][0]


def test_hvac_invalid_elbow_polygon_is_repaired():
    bow = {"shape_name": "Elbow 90", "vertices": [[0, 0], [10, 10], [10, 0], [0, 10]]}  # self-intersecting
    gt = gt_geometry("hvac", [bow])
    assert gt.geoms[0].is_valid
    assert label_shapes(page_segments([shape(line(2, 1, 8, 1))]), gt).label[0] == 1


def test_no_gt_means_all_other():
    gt = gt_geometry("fire", [])
    assert label_shapes(page_segments([shape(line(0, 0, 1, 1))]), gt).label.sum() == 0


def test_gt_coverage_sees_missing_strokes():
    gt = gt_geometry("fire", [{"shape_name": "Line", "vertices": [[0, 0], [100, 0]]},
                              {"shape_name": "Line", "vertices": [[0, 500], [100, 500]]}])
    true_seg = np.array([[0, 0, 60, 0]], dtype=float)
    cov = gt_coverage(gt, true_seg)
    assert abs(cov[0] - 0.62) < 0.02 and cov[1] == 0.0


# --- coordinate check -----------------------------------------------------------------------------------

def _grid_page(w=1000, h=600, seed=0):
    rng = np.random.default_rng(seed)
    # Asymmetric on purpose: no flip or turn maps the pipes onto themselves.
    pipes = [[[x, 60], [x, 380]] for x in (90, 170, 330, 610)] + [[[40, 180], [700, 180]]]
    noise = [[[a, b], [a + rng.uniform(-30, 30), b + rng.uniform(-30, 30)]]
             for a, b in rng.uniform([0, 0], [w, h], size=(400, 2))]
    return pipes, noise


def test_calibration_identity_is_conclusive():
    w, h = 1000, 600
    pipes, noise = _grid_page(w, h)
    seg = np.array([p[0] + p[1] for p in pipes + noise], dtype=float)
    cal = calibrate(STRtree(segment_lines(seg)), pipes, [False] * len(pipes), w, h)
    assert cal["chosen"] == IDENTITY and cal["conclusive"]


def test_calibration_finds_rotated_gt():
    w, h = 1000, 600
    pipes, noise = _grid_page(w, h)
    seg = np.array([p[0] + p[1] for p in pipes + noise], dtype=float)
    # GT stored in the unrotated (600 x 1000) frame: the inverse of rot90ccw (x, y) -> (y, h - x).
    gt = [[[h - y, x] for x, y in p] for p in pipes]
    cal = calibrate(STRtree(segment_lines(seg)), gt, [False] * len(gt), w, h)
    assert cal["chosen"] == ROT90_CCW and cal["conclusive"]
    back = transform_points(ROT90_CCW, np.array(gt[0], dtype=float), w, h)
    assert np.allclose(back, pipes[0])


def test_calibration_rejects_gt_that_matches_nothing():
    w, h = 1000, 600
    _, noise = _grid_page(w, h)
    seg = np.array([p[0] + p[1] for p in noise], dtype=float)
    rng = np.random.default_rng(7)
    gt = [[[a, b], [a + 40, b]] for a, b in rng.uniform([0, 0], [w - 40, h], size=(60, 2))]
    cal = calibrate(STRtree(segment_lines(seg)), gt, [False] * len(gt), w, h)
    assert not cal["conclusive"]


def test_calibration_offpage_points_count_as_misses():
    # A wide page where a quarter turn keeps only a small, ink-dense corner of the GT on the page.
    w, h = 2000, 600
    rng = np.random.default_rng(3)
    pipes = [[[x, 50], [x, 550]] for x in range(60, 1900, 37)]
    dense = [[[a, b], [a + 3, b + 3]] for a, b in rng.uniform([0, 0], [600, 600], size=(6000, 2))]
    seg = np.array([p[0] + p[1] for p in pipes + dense], dtype=float)
    cal = calibrate(STRtree(segment_lines(seg)), pipes, [False] * len(pipes), w, h)
    assert cal["chosen"] == IDENTITY


# --- sources (read-only) and end-to-end -------------------------------------------------------------------

def _fake_ai_takeoff(root: Path) -> Path:
    be = root / "BE"
    (be / "data" / "drawings" / "document-000007").mkdir(parents=True)
    (be / "data" / "drawings" / "document-000008").mkdir(parents=True)
    db = be / "data" / "ai_takeoff.sqlite3"
    con = sqlite3.connect(db)
    con.execute("create table documents (id integer, filename text, stored_name text, sha256 text)")
    con.execute("create table ground_truths (id integer, document_id integer, mode text, pages_json text, updated_at text)")
    con.executemany("insert into documents values (?,?,?,?)",
                    [(7, "a.pdf", "a", "sha-a"), (8, "a copy.pdf", "a2", "sha-a")])
    obj = lambda name, v, group="primary": {"shape_name": name, "vertices": v, "group": group, "uid": name + str(v[0])}
    pipe = [[100, 100], [900, 100]]
    pages7 = [{"page_number": 1, "page_width": 1000, "page_height": 600, "objects": [
        obj("Line", pipe), obj("Line", [[950, 150], [950, 550]], group="detected"), obj("Sprinkler", [[0, 0], [1, 0], [1, 1], [0, 1]])]}]
    pages8 = [{"page_number": 1, "page_width": 1000, "page_height": 600, "objects": [obj("Line", pipe)]}]
    con.executemany("insert into ground_truths values (?,?,?,?,?)",
                    [(1, 7, "fire", json.dumps(pages7), "t"), (2, 8, "fire", json.dumps(pages8), "t")])
    con.commit()
    con.close()
    _, noise = _grid_page()
    shapes = [shape(line(x, 100, x + 4, 100)) for x in range(100, 900, 8)]
    shapes += [shape(line(950, 150 + y, 950, 154 + y), layer="X") for y in range(0, 400, 8)]
    shapes += [shape(line(*p[0], *p[1]), layer="N") for p in noise]
    for doc in (7, 8):
        for v, payload in ((2, {"shapes": []}), (3, {"shapes": shapes, "document_metadata": {
                "page_width": 1000, "page_height": 600, "rotation": 0}})):
            with gzip.open(be / "data" / "drawings" / f"document-{doc:06d}" / f"page-0001.v{v}.json.gz", "wt") as fh:
                json.dump(payload, fh)
        with gzip.open(be / "data" / "drawings" / f"document-{doc:06d}" / "page-0001.v9.view.json.gz", "wt") as fh:
            json.dump({"shapes": []}, fh)
    return be


def test_latest_payload_skips_view(tmp_path):
    be = _fake_ai_takeoff(tmp_path)
    p = latest_payload_path(be / "data" / "drawings", 7, 1)
    assert p.name == "page-0001.v3.json.gz"


def test_db_opened_read_only(tmp_path):
    be = _fake_ai_takeoff(tmp_path)
    con = connect_ro(be / "data" / "ai_takeoff.sqlite3")
    with pytest.raises(sqlite3.OperationalError):
        con.execute("insert into documents values (9, 'x', 'x', 'x')")
    con.close()


def test_build_end_to_end(tmp_path):
    be = _fake_ai_takeoff(tmp_path)
    db = be / "data" / "ai_takeoff.sqlite3"
    before = db.stat().st_mtime_ns, db.read_bytes()
    out = tmp_path / "e0"
    m = build_all(be, out, log=lambda s: None)
    assert (db.stat().st_mtime_ns, db.read_bytes()) == before  # source untouched
    assert len(m["physical_pages"]) == 1 and m["physical_pages"][0]["decision"] == "pending_user_choice"
    assert {r["status"] for r in m["records"]} == {"ok"}
    r = next(r for r in m["records"] if r["gt_id"] == 1)
    z = np.load(out / r["npz"])
    # 100 dashes on the primary GT pipe are true; the vertical run sits only on a *detected* object.
    assert z["label"][:100].all() and not z["label"][100:150].any()
    assert r["gt_objects_in_scope"] == 1 and r["calibration"]["chosen"] == "identity"
    assert z["shape_offsets"][-1] == len(z["segments"])
    meta = json.loads(str(z["meta"]))
    assert meta["coordinate_frame"] == "display_top_left_pt" and meta["page_number"] == 1
    assert (out / "manifest.json").exists()
    (pair,) = m["duplicate_pairs"]
    assert pair["comparable"] and pair["length_iou"] == 1.0
    from openstroke.qa import write_report

    text = write_report(out / "manifest.json", tmp_path / "qa.md")
    assert "doc000007_p0001_gt1" in text and "IoU" in text


def test_package_does_not_import_pymupdf():
    import ast

    src = Path(__file__).resolve().parents[1] / "openstroke"
    banned = {"fitz", "pymupdf"}
    for f in src.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not {n.split(".")[0].lower() for n in names} & banned, f
