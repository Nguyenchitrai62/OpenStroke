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
