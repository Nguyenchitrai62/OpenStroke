"""Dataset-native scoring: ground-truth IDs are already present, no score ZIP or polygon assignment."""
from pathlib import Path

import numpy as np
import pytest

from openstroke.obb.vector_validation import DatasetVectorGT


def make_page(tmp_path):
    images = tmp_path / "images/val"
    vectors = tmp_path / "vectors/val"
    labels = tmp_path / "labels/val"
    for folder in (images, vectors, labels):
        folder.mkdir(parents=True)
    stem = "doc000022_test"
    (images / (stem + ".png")).write_bytes(b"")
    tok = np.zeros((4,24),np.float32)
    for i,length in enumerate([10.,10.,10.,5.]):
        tok[i,1:17] = np.linspace([0,i*5],[length,i*5],8).flatten()
    tok[:,21] = [4,8,3,11]  # arbitrary layer IDs must never affect scoring
    tok[:,22] = [0,0,1,-1]
    tok[:,23] = [10,11,30,99]
    np.savez(vectors / (stem + ".npz"),tokens=tok)
    (labels / (stem + ".txt")).write_text("0 0 0 1 0 1 1 0 1\n1 0 0 1 0 1 1 0 1\n")
    return images, vectors / (stem + ".npz"), stem


def predictions(stem):
    return {stem: [dict(class_id=0,score=0.8,members=[10,11],source="graph"),
                   dict(class_id=1,score=0.8,members=[30],source="graph"),
                   dict(class_id=0,score=0.2,members=[99],source="graph")]}


def test_native_gt_matches_by_stroke_ids_and_class_without_external_pack(tmp_path):
    images, path, stem = make_page(tmp_path)
    gt = DatasetVectorGT(images,["Duct","Elbow"],comparison_pages=("doc000022",))
    summary=gt.evaluate(predictions(stem),"graph",["Duct","Elbow"])
    assert summary["F2"] == summary["P"] == summary["R"] == 1.
    assert summary["best_conf"] == 0.25 and summary["protocol"] == "dataset_vectors.sampled8.v1"
    assert summary["per_class"]["Duct"]["gt"] == 1 and summary["per_class"]["Elbow"]["gt"] == 1
    wrong=predictions(stem)
    wrong[stem][1]["class_id"]=0
    assert gt.evaluate(wrong,"graph",["Duct","Elbow"])["R"] == 0.5
    with np.load(path) as z: tok=z["tokens"]
    tok[:,21]=12345
    np.savez(path,tokens=tok)
    other=DatasetVectorGT(images,["Duct","Elbow"],comparison_pages=("doc000022",))
    assert other.evaluate(predictions(stem),"graph",["Duct","Elbow"]) == summary
    for row in wrong[stem]: row.pop("source")
    assert gt.evaluate(wrong,"members",["Duct","Elbow"])["R"] == 0.5


def test_native_gt_validates_inst_pid_predictions_and_optional_exact_length(tmp_path):
    images,path,stem=make_page(tmp_path)
    with np.load(path) as z: tok=z["tokens"]
    np.savez(path,tokens=tok,length=np.array([10.,10.,10.,5.]))
    gt=DatasetVectorGT(images,["Duct","Elbow"],comparison_pages=("doc000022",))
    assert gt.protocol == "dataset_vectors.exact_length.v1"
    bad=predictions(stem);bad[stem][0]["members"]=[1234]
    with pytest.raises(ValueError,match="unknown primitive"):
        gt.evaluate(bad,"graph",["Duct","Elbow"])
    tok[0,22]=-2
    np.savez(path,tokens=tok)
    with pytest.raises(ValueError,match="Invalid GT"):
        DatasetVectorGT(images,["Duct","Elbow"],comparison_pages=("doc000022",))


def test_epoch_evaluator_uses_dataset_without_opening_any_pack(tmp_path,monkeypatch):
    from openstroke.obb import object_validation, vector_validation, cli
    images,path,stem=make_page(tmp_path)
    gt=DatasetVectorGT(images,["Duct","Elbow"],comparison_pages=("doc000022",))
    monkeypatch.setattr(vector_validation,"DatasetVectorGT",lambda *a,**k:gt)
    monkeypatch.setattr(object_validation,"open_pack",lambda *a,**k:pytest.fail("native evaluation must not open a score pack"))
    monkeypatch.setattr(cli,"predict",lambda *a,**k:{stem+".png": predictions(stem)[stem]})
    evaluator=object_validation.VectorEpochEvaluator(None,images,["Duct","Elbow"],dict(batch=1,stroke_graph=True))
    try:
        assert evaluator.evaluate_checkpoint("weights.pt")["F2"] == 1.
        assert evaluator.protocol == gt.protocol
    finally:
        evaluator.close()


def test_viewer_geometry_matches_legacy_info_without_writing_npz(tmp_path):
    from app.obb_page import vector_page_info
    from openstroke.infer import payload_npz
    shapes=[dict(items=[["l",[0,0],[10,0]]],type="s",color=[0,0,0],width=1,layer=""),
            dict(items=[["c",[10,0],[10,5],[15,10],[20,10]]],type="s",color=[0,0,0],width=1,layer="")]
    payload=dict(shapes=shapes,document_metadata=dict(page_width=30,page_height=20))
    native=vector_page_info(payload)
    legacy=payload_npz(payload,tmp_path/"legacy.npz","hvac")
    assert native["shapes"] == legacy["shapes"] and native["width"] == legacy["width"]
    assert np.array_equal(native["length"],legacy["length"])
    assert np.array_equal(native["segments"].seg,legacy["segments"].seg)


def test_packless_real_dataset_metadata_is_valid():
    """Read-only metadata acceptance on the local uploaded dataset; no model inference/scoring."""
    import yaml
    root=Path(__file__).resolve().parents[2]/"datasets/obb_hvac7"
    if not root.exists():
        pytest.skip("local dataset is not shipped with the source")
    names=list(yaml.safe_load((root/"data.yaml").read_text(encoding="utf-8"))["names"].values())
    gt=DatasetVectorGT(root/"images/val",names)
    assert len(gt.pages) == 8 and gt.protocol == "dataset_vectors.sampled8.v1"
    assert all(value["label_objects"] > 0 for value in gt.coverage.values())


def test_vector_request_cannot_select_fire_or_unused_eight_pass_tta():
    from app.obb_page import page_settings
    assert page_settings(dict(kind="obb"),dict(mode="fire",tta=True)) == ("hvac",1)
    assert page_settings(dict(kind="obb"),dict(mode="plumbing",tta=False)) == ("hvac",1)
    assert page_settings({},dict(mode="fire",tta=True)) == ("fire",8)
