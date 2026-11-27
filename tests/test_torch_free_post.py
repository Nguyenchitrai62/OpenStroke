"""Pre- and post-processing run without torch (extract-duct worker: CPU image, model behind an endpoint)."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = textwrap.dedent('''
    import importlib.abc, sys

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in ("torch", "torchvision"):
                raise ImportError("torch is blocked in this test: " + name)
            return None

    sys.meta_path.insert(0, Block())
    import numpy as np
    from openstroke.build import shape_attributes
    from openstroke.geometry import page_segments
    from openstroke import hvac_faces
    from openstroke.objects import HVAC_TO_OBJ8, OBJ8_CLASSES

    pts = [((100, 100), (400, 100)), ((400, 100), (400, 140)), ((400, 140), (100, 140)), ((100, 140), (100, 100))]
    shapes = [{"items": [["l", list(a), list(b)]], "type": "s", "color": [0, 0, 0], "width": 0.5, "layer": ""} for a, b in pts]
    segs = page_segments(shapes)
    attrs = shape_attributes(shapes, segs)
    prob = np.full(4, 0.95)
    groups = {"instance_id": np.zeros(4, np.int64), "cluster_ids": np.array([0]),
              "cluster_probs": np.array([[0.02, 0.9, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01]])}
    objects, cls, inst = hvac_faces.hybrid_group_objects(segs, prob, segs.shape_length(), groups, thr=0.5)
    sobj = hvac_faces.segment_objects(objects, segs, prob >= 0.5)
    assert len(OBJ8_CLASSES) == 8 and len(HVAC_TO_OBJ8) == 10
    assert "torch" not in sys.modules
    print("objects", [o["shape_name"] for o in objects])
''')


def test_post_processing_needs_no_torch():
    r = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "objects ['Duct']" in r.stdout


def test_group_head_reexports_the_object_classes():
    import pytest

    pytest.importorskip("torch")
    from openstroke import objects
    from openstroke.models import group_head

    assert group_head.OBJ8_CLASSES is objects.OBJ8_CLASSES
    assert group_head.HVAC_TO_OBJ8 is objects.HVAC_TO_OBJ8
