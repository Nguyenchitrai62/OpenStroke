"""Object F2 after training (score_pack=): the val page folder from data.yaml, and a run without weights is skipped quietly."""
from openstroke.obb.cli import _score_after, _val_images


def test_val_images_from_yaml(tmp_path):
    (tmp_path / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: Duct\n", encoding="utf-8")
    assert _val_images(tmp_path / "data.yaml") == tmp_path / "images/val"
    other = tmp_path / "elsewhere"
    (tmp_path / "b.yaml").write_text(f"path: {other}\nval: pages/val\n", encoding="utf-8")
    assert _val_images(tmp_path / "b.yaml") == other / "pages/val"


def test_score_after_without_best_is_skipped(tmp_path, capsys):
    _score_after(tmp_path, tmp_path / "data.yaml", {"score_pack": "pack.zip"}, 1)
    assert "skipped" in capsys.readouterr().out
    _score_after(tmp_path, tmp_path / "data.yaml", {}, 1)            # no score_pack: nothing at all
    assert capsys.readouterr().out == ""


def test_graph_epoch_console_prioritises_object_evaluation():
    import io
    from openstroke.obb.console import val_rows

    stream = io.StringIO()
    val_rows({"mAP50": 0.42, "members": {"graph": {"decode_R_length": 0.629}}}, seconds=37.8, stream=stream)
    text = stream.getvalue()
    assert "object F2: evaluated after training" in text
    assert "62.9%" in text and "0.420" in text
    assert "edge" not in text and "Precision" not in text


def test_final_object_console_has_main_score_and_keeps_diagnostics_in_file(tmp_path, monkeypatch, capsys):
    import json
    import numpy as np
    from openstroke.obb import object_score
    from openstroke.objects import N_CLASSES

    (tmp_path / "pages.json").write_text("[]")
    detections = tmp_path / "dets.json"
    detections.write_text("{}")
    counts = np.zeros((N_CLASSES, 3), int)
    counts[1] = [8, 10, 20]
    values = {conf: {doc + "_page": counts for doc in object_score.PIPELINE_PAGES}
              for conf in object_score.CONFS}
    monkeypatch.setattr(object_score, "score", lambda *args, **kwargs: values)
    object_score.main([f"pack={tmp_path}", f"dets={detections}", "how=graph,members,rule", "require_graph=false"])
    # Empty detections normally omit the graph, so check the model-returned-strokes main row.
    text = capsys.readouterr().out
    assert "FINAL OBJECT EVALUATION" in text and "F2" in text and "Recall" in text and "Precision" in text
    assert "Model-returned strokes [MAIN]" in text and "0.444" in text and "0.400" in text and "0.800" in text
    assert "conf 0.10" not in text
    assert "conf 0.10" in (tmp_path / "object_score_details.log").read_text()
    assert json.loads((tmp_path / "object_score.json").read_text())["members"]["R"] == 0.4


def test_score_after_reuses_best_epoch_without_predicting_again(tmp_path, monkeypatch, capsys):
    import json
    from openstroke.obb import object_score

    (tmp_path / "weights").mkdir()
    (tmp_path / "weights/best.pt").write_bytes(b"already saved")
    summary = {"graph": {"F2": 0.61, "R": 0.68, "P": 0.44, "epoch": 4}}
    (tmp_path / "object_score.json").write_text(json.dumps(summary))
    calls = []
    monkeypatch.setattr(object_score, "main", lambda argv: calls.append(argv))
    _score_after(tmp_path, tmp_path / "data.yaml",
                 {"score_pack": "pack.zip", "score_how": "members,graph,rule"},
                 1, require_graph=True, object_selected=True)
    assert calls == []
    assert capsys.readouterr().out == ""  # trainer already printed the best score once

    _score_after(tmp_path, tmp_path / "data.yaml",
                 {"score_pack": "pack.zip", "score_how": "members,graph,rule", "score_final": True},
                 1, require_graph=True, object_selected=True)
    assert len(calls) == 1 and "how=members,graph,rule" in calls[0]


def test_missing_cached_score_runs_only_graph_once(tmp_path, monkeypatch):
    from openstroke.obb import object_score

    (tmp_path / "weights").mkdir()
    (tmp_path / "weights/best.pt").write_bytes(b"already saved")
    calls = []
    monkeypatch.setattr(object_score, "main", lambda argv: calls.append(argv))
    _score_after(tmp_path, tmp_path / "data.yaml", {"score_pack": "pack.zip"},
                 1, require_graph=True, object_selected=True)
    assert len(calls) == 1 and "how=graph" in calls[0]


def test_missing_score_pack_fails_before_dataset_model_or_gpu_setup(tmp_path, monkeypatch):
    import pytest
    from openstroke.obb import cli
    for name in ("_set_device", "prepare_data", "_run_trainer"):
        monkeypatch.setattr(cli, name, lambda *a, **k: pytest.fail("Missing pack must stop before setup/training"))
    with pytest.raises(FileNotFoundError, match="score_pack not found"):
        cli.train(data="data.yaml", model="checkpoint.pt", score_pack=tmp_path / "missing.zip", project=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_score_pack_preflight_accepts_nested_zip_and_rejects_bad_input(tmp_path):
    import pytest
    import zipfile
    from openstroke.obb.object_score import check_pack_path
    valid = tmp_path / "valid.zip"
    with zipfile.ZipFile(valid, "w") as z:
        z.writestr("hvac7/pages.json", "[]")
    check_pack_path(valid)
    assert list(tmp_path.iterdir()) == [valid]  # no extraction before training
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as z:
        z.writestr("data.yaml", "")
    with pytest.raises(ValueError, match="no pages.json"):
        check_pack_path(empty)
    bad = tmp_path / "bad.zip"
    bad.write_text("not a ZIP")
    with pytest.raises(zipfile.BadZipFile):
        check_pack_path(bad)
