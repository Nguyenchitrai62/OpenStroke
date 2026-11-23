"""OpenStroke38 / H153: the image-first input set keeps only the geometry a picture shows; every CAD-encoding field and page statistic is
hidden (CPU, no training)."""


def test_image_first_hides_the_cad_encoding_and_keeps_geometry():
    from openstroke.features import FEATURE_GROUPS, FEATURE_NAMES, GEOMETRY, feature_mask

    hidden = {FEATURE_NAMES[i] for i in feature_mask(["image_first"])}
    kept = set(FEATURE_NAMES) - hidden
    assert kept == set(GEOMETRY)
    for group in ("strict", "colour", "layer_stats", "context", "attrs"):
        assert set(FEATURE_GROUPS[group]) <= hidden
    for name in ("log_width", "stroke_r", "type_fs", "log_nseg", "frac_c", "seqno_rank", "dashed"):
        assert name in hidden


def test_image_first_model_ignores_hidden_features():
    import pytest
    torch = pytest.importorskip("torch")
    from openstroke.features import FEATURE_NAMES, feature_mask
    from openstroke.models.openstroke2 import OpenStroke2

    m = OpenStroke2(len(FEATURE_NAMES), dims=(16, 32), enc_depths=(1, 1), dec_depths=(1,), patch=32, feature_drop=["image_first"])
    keep = m.feat_keep.numpy()
    assert keep.sum() == len(FEATURE_NAMES) - len(feature_mask(["image_first"]))


def test_image_branch_takes_an_openobb_detector_backbone(tmp_path):
    """OpenStroke38: an OpenOBB training checkpoint ({"model": state, "args": ...}) gives its backbone.* weights to the HGNetv2 image
    branch unchanged; a bare backbone state dict (E48) still loads."""
    import pytest
    torch = pytest.importorskip("torch")
    from openstroke.models.hgnetv2 import HGNetv2
    from openstroke.models.image_branch import ImageBranch, obb_backbone_state

    net = HGNetv2("B5", use_lab=False, return_idx=[0, 1, 2, 3], freeze_at=-1, freeze_norm=True, pretrained=False)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.01)
    full = {"model": {**{f"backbone.{k}": v for k, v in net.state_dict().items()}, "head.w": torch.zeros(3)}, "args": {"lr0": 1e-4}}
    torch.save(full, tmp_path / "obb.pt")
    torch.save(net.state_dict(), tmp_path / "bare.pt")
    a, b = obb_backbone_state(tmp_path / "obb.pt"), obb_backbone_state(tmp_path / "bare.pt")
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    br = ImageBranch(16, backbone="hgnetv2_b5", weights=str(tmp_path / "obb.pt"), fpn=True)
    k = "stem.stem1.conv.weight"
    assert torch.equal(br.hg.state_dict()[k], net.state_dict()[k])
