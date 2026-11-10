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
