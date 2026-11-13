"""F137: the balanced re-split relinks files, keeps provenance in step and refuses a held-out document in train."""
import json

import pytest

from openstroke.obb import resplit as rs


def _page(root, split, stem):
    for kind, suf in rs.KINDS:
        p = root / kind / split / (stem + suf)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(stem)


@pytest.fixture
def src(tmp_path, monkeypatch):
    root = tmp_path / "src"
    rows = [dict(stem="doc000190_p0001_gt1", split="val", objects=5), dict(stem="doc000180_p0001_gt2", split="val", objects=3),
            dict(stem="doc000181_p0001_gt3", split="train", objects=2),
            dict(stem="vw_aaaaaa_p0001", split="train", document="HELD", objects=4),
            dict(stem="vw_aaaaaa_p0002", split="train", document="HELD", objects=1),
            dict(stem="vw_bbbbbb_p0001", split="train", document="KEEP", objects=7)]
    for r in rows:
        _page(root, r["split"], r["stem"])
    (root / "pages.json").write_text(json.dumps(rows))
    (root / "gt_provenance.json").write_text(json.dumps({"schema": "openstroke.remote_gt.v1",
                                                         "records": [{"stem": r["stem"], "split": r["split"]} for r in rows]}))
    (root / "data.yaml").write_text("names: {0: Duct}\n")
    monkeypatch.setitem(rs.PLANS, "t", {"gt_to_train": ["doc000190"], "viewer_val": ["vw_aaaaaa_p0001"],
                                        "viewer_drop": ["vw_aaaaaa_p0002"], "held_out_documents": ["HELD"]})
    return root


def test_resplit_moves_and_drops(src, tmp_path):
    out = tmp_path / "out"
    count = rs.resplit(src, out, "t", log=lambda _: None)
    assert count[("train", "GT")] == (2, 7) and count[("val", "GT")] == (1, 3)
    assert count[("val", "viewer")] == (1, 4) and count[("train", "viewer")] == (1, 7)
    assert (out / "images" / "train" / "doc000190_p0001_gt1.png").exists()
    assert not (out / "images" / "val" / "doc000190_p0001_gt1.png").exists()
    assert not any(out.rglob("vw_aaaaaa_p0002*"))
    proof = json.loads((out / "gt_provenance.json").read_text())
    assert {r["stem"]: r["split"] for r in proof["records"]} == {r["stem"]: r["split"] for r in json.loads((out / "pages.json").read_text())}
    assert proof["split_plan"]["name"] == "t"


def test_resplit_refuses_a_held_out_document_in_train(src, tmp_path):
    rs.PLANS["t"]["viewer_drop"] = []
    with pytest.raises(SystemExit, match="leak"):
        rs.resplit(src, tmp_path / "out", "t", log=lambda _: None)


def test_hvac10b_plan_is_disjoint():
    p = rs.PLANS["hvac10b"]
    assert not set(p["viewer_val"]) & set(p["viewer_drop"])
    assert len(p["viewer_val"]) == 9 and p["gt_to_train"] == ["doc000190"]


def test_trainer_knows_eval_init():
    from openstroke.obb import cli, train
    cli.check_keys({"eval_init": True})
    assert train.build_parser().parse_args(["--data", "x", "--out", "y", "--eval-init"]).eval_init
