"""Score saved OpenOBB val detections against a score pack (no AI_Takeoff DB). The scorer lives in ``openstroke.obb.object_score``;
this keeps the old command line.

    python tools/obb_score_pack.py PACK_DIR_OR_ZIP VAL_DETS_JSON [members,rule,graph,...]
"""

import json
import sys
from pathlib import Path

from openstroke.obb.object_score import CONFS, PIPELINE, PIPELINE_PAGES, REF, boxes_to_strokes, open_pack, report, score  # noqa: F401

if __name__ == "__main__":
    pack, dets = open_pack(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    for how in (sys.argv[3] if len(sys.argv) > 3 else "members,rule").split(","):
        report(score(pack, dets, how), how)
