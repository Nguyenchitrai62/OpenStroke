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
