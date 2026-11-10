"""On the Colab VM after an OpenOBB run: detections on every val page as JSON (pixel polygons of the page image), scored locally by
``research/probes/obb_score_probe.py``. ``scale``: the run's tile_scale (detections come back in the page image's own pixels).

    python tools/obb_predict_val.py <best.pt> <images/val> <out.json> [scale]
"""

import json
import sys
from pathlib import Path

from openstroke.obb.cli import predict

best, src, out = sys.argv[1:4]
scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
dets = predict(best, src, conf=0.05, tile_scale=scale, save=False)
Path(out).write_text(json.dumps({Path(k).stem: v for k, v in dets.items()}, default=float), encoding="utf-8")
print("wrote", out, {Path(k).stem: len(v) for k, v in dets.items()})
