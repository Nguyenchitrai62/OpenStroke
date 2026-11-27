"""OpenStroke: per-shape probabilities for the vector strokes of a PDF drawing page (raw output).

Deployment (OpenOBB-style interface — load model -> infer vector payload -> raw result; post-processing is separate):

    from openstroke import OpenStroke
    model = OpenStroke("best.pt")
    for r in model.predict(payload):          # dict payload, .json / .json.gz path, folder, glob, or list
        print(r.prob, r.valid, r.edges, r.groups, r.speed)

Input: the drawing commands of one page (JSON payload produced elsewhere, e.g. AI_Takeoff ``split_countfire``);
output: raw per-shape probabilities (and optional same-object edges / object-group logits when the checkpoint has
those heads). No discipline/mode flag is required at inference: what "a stroke to extract" means comes strictly
from the training data, while building polygons / lines is handled by a separate post-processing module.
The package never imports PyMuPDF (AGPL).
"""
from __future__ import annotations

__version__ = "0.48.7"   # 0.<latest research architecture>.<patch>, as OpenOBB; bumped by tools/push_all.py


def __getattr__(name: str):   # lazy: `python -m openstroke.cli` avoids circular imports and keeps CLI startup instant
    if name in ("OpenStroke", "Predictor", "load_payload"):
        from . import infer as _infer

        return getattr(_infer, name)
    if name in ("Results", "Strokes"):
        from . import results as _results

        return getattr(_results, name)
    if name == "strip_checkpoint":
        from .strip import strip as _strip

        return _strip
    raise AttributeError(f"module 'openstroke' has no attribute {name!r}")


__all__ = ["OpenStroke", "Predictor", "Results", "Strokes", "load_payload", "strip_checkpoint", "__version__"]
