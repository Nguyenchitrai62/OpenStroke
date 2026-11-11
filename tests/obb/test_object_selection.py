"""Regression checks for selecting deployable EMA weights by full-page object F2."""
import json
from pathlib import Path

import torch

from openstroke.obb import train as trainer


def test_trainer_selects_object_f2_even_when_box_map_improves(tmp_path, monkeypatch):
    import sys
    from torch.utils.data import Dataset
    from openstroke.obb import object_validation
    from openstroke.obb.report import append_results
    sys.path.insert(0, str(Path(__file__).parent))
    from test_openobb25 import IMG, NC, _batch

    class TinyDataset(Dataset):
        def __init__(self, *args, **kwargs):
            self.mosaic_p = 0
            self.items = [{"name": "tile", "objs": []}] * 2
        def __len__(self):
            return 2
        def __getitem__(self, index):
            return torch.zeros(3, IMG, IMG, dtype=torch.uint8), _batch()[index]

    scores = iter([0.60, 0.55, 0.52])
    states = []
    class Evaluator:
        def __init__(self, *args):
            pass
        def evaluate(self, module, epoch):
            states.append({k: v.detach().cpu().clone() for k, v in module.state_dict().items()})
            return {"F2": next(scores), "P": 0.5, "R": 0.65, "best_conf": 0.2}
        def evaluate_checkpoint(self, path):
            return {"F2": 0.60, "P": 0.5, "R": 0.65, "best_conf": 0.2}
        def close(self):
            pass

    def box_eval(*args, **kwargs):
        step = len(states)
        return {"mAP50": 0.2 + step * 0.1, "mAP50_95": 0.1 + step * 0.1, "P": 0.5, "R": 0.5}, {}

    data = tmp_path / "data"
    (data / "gt/val").mkdir(parents=True)
    (data / "gt/val/a.txt").write_text("")
    out = tmp_path / "run"
    monkeypatch.setattr(trainer, "DotaPatches", TinyDataset)
    monkeypatch.setattr(trainer, "dataset_classes", lambda _: ["a", "b", "c"])
    monkeypatch.setattr(trainer, "_log_vector_tiles", lambda *args: None)
    monkeypatch.setattr(trainer, "eval_dota", box_eval)
    monkeypatch.setattr(trainer, "plot_labels", lambda *args: None)
    monkeypatch.setattr(trainer, "plot_results", lambda *args: None)
    monkeypatch.setattr(trainer, "plot_train_batch", lambda *args, **kwargs: None)
    monkeypatch.setattr(object_validation, "GraphEpochEvaluator", Evaluator)
    args = ["--data", str(data), "--out", str(out), "--arch", "v25", "--size", "s",
            "--img", str(IMG), "--groups", "0,1,1", "--vec-dim", "32", "--vec-layers", "1",
            "--stroke-graph", "--graph-foreground", "--no-pretrained", "--workers", "0", "--threads", "1",
            "--batch", "2", "--nbs", "2", "--epochs", "2", "--max-iters", "1", "--vec-ground", "1",
            "--score-pack", "pack.zip", "--score-images", "val-images", "--no-amp"]
    trainer.main(args)
    best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert best["epoch"] == 0 and last["epoch"] == 1
    assert best["val_object_F2"] == 0.60 and best["best_metric"] == "object_f2"
    assert last["best_score"] == 0.60 and last["best_epoch"] == 0
    assert all(torch.equal(best["model"][k], states[0][k]) for k in states[0])
    assert json.loads((out / "object_score.json").read_text())["graph"]["epoch"] == 1
    assert "object_F2" in (out / "results.csv").read_text().splitlines()[0]

    # Resume an older mAP-ranked run: re-score its best before ranking new epochs.
    last["best_metric"] = "map"
    last["best_score"] = 0.99
    torch.save(last, out / "last.pt")
    (out / "train_done").unlink()
    resumed = list(args)
    resumed[resumed.index("--epochs") + 1] = "3"
    trainer.main(resumed)
    best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert best["epoch"] == 0 and best["val_object_F2"] == 0.60
    assert last["epoch"] == 2 and last["best_score"] == 0.60


def test_results_csv_adds_object_columns_when_resuming_old_format(tmp_path):
    import csv
    from openstroke.obb.report import append_results

    (tmp_path / "results.csv").write_text("epoch,lr,mAP50\n1,0.001,0.4\n")
    append_results(tmp_path, {"epoch": 1, "object_F2": 0.6, "object_R": 0.65, "object_P": 0.5})
    with (tmp_path / "results.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["mAP50"] == "0.4" and rows[0]["object_F2"] == ""
    assert rows[1]["object_F2"] == "0.6"


def test_graph_scoring_cache_matches_the_uncached_product_score(tmp_path):
    import numpy as np
    from openstroke.obb.object_score import score

    stem = "doc000022_test"
    (tmp_path / "npz").mkdir()
    (tmp_path / "gt").mkdir()
    (tmp_path / "names.json").write_text(json.dumps(["Duct"]))
    (tmp_path / "pages.json").write_text(json.dumps([{"stem": stem, "scale": 1}]))
    segments = np.array([[0, 0, 10, 0], [0, 2, 10, 2]], float)
    np.savez(tmp_path / "npz" / f"{stem}.npz", segments=segments, shape_offsets=np.array([0, 1, 2]),
             label=np.ones(2, bool), length=np.array([10., 10.]), layer_id=np.zeros(2, int),
             stroke_rgb=np.zeros((2, 3)), width=np.ones(2), type_code=np.zeros(2, int), op_counts=np.ones((2, 4)))
    polygon = [[0, -1], [10, -1], [10, 3], [0, 3]]
    (tmp_path / "gt" / f"{stem}.json").write_text(json.dumps([{"shape_name": "Duct", "vertices": polygon}]))
    detections = {stem: [{"source": "graph", "class_id": 0, "score": 0.9,
                          "poly": np.asarray(polygon).reshape(-1).tolist(), "members": [0, 1]}]}
    original = score(tmp_path, detections, "graph", log=lambda _: None)
    cache = {}
    first = score(tmp_path, detections, "graph", log=lambda _: None, cache=cache)
    (tmp_path / "npz" / f"{stem}.npz").unlink()
    (tmp_path / "gt" / f"{stem}.json").unlink()
    cached = score(tmp_path, detections, "graph", log=lambda _: None, cache=cache)
    for conf in original:
        assert np.array_equal(first[conf][stem], original[conf][stem])
        assert np.array_equal(cached[conf][stem], original[conf][stem])
        assert original[conf][stem][1].tolist() == [1, 1, 1]
