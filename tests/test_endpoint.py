"""Reference endpoint of the HVAC duct vector model: the request / response contract the extract-duct module reads."""
from __future__ import annotations

import json

import numpy as np
import pytest

from openstroke import endpoint


class FakePredictor:
    path = "runs/e151/last_deploy.pt"
    threshold = 0.5

    def __init__(self):
        self.calls = []

    def predict_payload(self, payload, tta=1):
        self.calls.append((payload, tta))
        n = len(payload["shapes"])
        return {"prob": np.linspace(0, 1, n), "group_objects": {
            "instance_id": np.array([-1] + [0] * (n - 1)), "cluster_ids": np.array([0]),
            "cluster_probs": np.full((1, 8), 0.125), "stroke_cls": np.zeros((n, 8)), "names": ("none",)}}

    def describe(self):
        return {"device": "cpu", "precision": "fp32"}


def test_respond_contract():
    pred = FakePredictor()
    body = {"payload": {"shapes": [{}] * 3, "document_metadata": {"page_width": 10, "page_height": 10}}, "tta": 1}
    resp = endpoint.respond(pred, body)
    json.dumps(resp)                                     # plain JSON
    assert resp["prob"] == [0.0, 0.5, 1.0]
    assert resp["threshold"] == 0.5
    assert resp["group_objects"] == {"instance_id": [-1, 0, 0], "cluster_ids": [0], "cluster_probs": [[0.125] * 8]}
    assert resp["model"]["name"] == "last_deploy"
    assert pred.calls[0][1] == 1


def test_respond_needs_shapes():
    with pytest.raises(ValueError, match="payload.shapes"):
        endpoint.respond(FakePredictor(), {"payload": {}})
