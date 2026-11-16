"""Score pack of EVERY page of a split (train + val, F149), for tuning post-processing locally on a run's all_scores.npz
(``openstroke scores``). Stroke arrays and labels are the dataset's own (datasets/e8: what the model was trained / validated on); GT objects
come from AI_Takeoff (doc pages, read-only) or the Viewer "Labels V1" export (vw pages), moved by the page's calibration transform.
pages.json keeps each page's split, so a scorer reports train and val apart. Written as a folder (local use only, gitignored).

    py -3.12 tools/make_score_pack_all.py datasets/obb_hvac10b datasets/e8 datasets/viewer_hvac19/exports_20261113/labels.json \
        datasets/score_pack_e8_all
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from make_score_pack_split import viewer_objects  # noqa: E402
from openstroke.geometry import IDENTITY, transform_points  # noqa: E402

DB = r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3"


def main(split, built, labels_json, out):
    from openstroke.sources import load_gt_pages

    split, built, out = Path(split), Path(built), Path(out)
    rows = json.loads((split / "pages.json").read_text(encoding="utf-8"))
    names = [ln.split(": ", 1)[1].strip() for ln in (split / "data.yaml").read_text(encoding="utf-8").splitlines()
             if ln.startswith("  ") and ": " in ln]
    gt = {p.stem: p for p in load_gt_pages(DB)}
    labels = json.loads(Path(labels_json).read_text(encoding="utf-8"))
    (out / "gt").mkdir(parents=True, exist_ok=True)
    (out / "npz").mkdir(parents=True, exist_ok=True)
    (out / "pages.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    (out / "names.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    for r in rows:
        st = r["stem"]
        gobj = viewer_objects(labels, st) if st.startswith("vw_") else [o for o in gt[st].objects if len(o.get("vertices") or [])]
        if r["transform"] != IDENTITY:
            w, h = r["size_pt"]
            gobj = [{**o, "vertices": transform_points(r["transform"], np.asarray(o["vertices"], float), w, h).tolist()} for o in gobj]
        (out / "gt" / f"{st}.json").write_text(json.dumps(gobj, ensure_ascii=False, default=float), encoding="utf-8")
        shutil.copyfile(built / f"{st}.npz", out / "npz" / f"{st}.npz")
        print(f"{r['split']:5s} {st}: {len(gobj)} objects", flush=True)
    print(f"{out}: {len(rows)} pages ({sum(r['split'] == 'val' for r in rows)} val)")


if __name__ == "__main__":
    main(*sys.argv[1:])
