"""Build the self-contained val scoring pack for Colab (``tools/obb_score_pack.py`` runs on it; no AI_Takeoff DB on the VM).

GT objects are read once here (DB opened read-only by ``load_gt_pages``) and written already in page coordinates (the page transform
applied), next to the e6 stroke arrays of the val pages, the val page records and the class names. Data only; no weights.

    py -3.12 tools/make_score_pack.py datasets/obb_hvac7 datasets/e6 datasets/openstroke_score_pack_hvac7.zip
"""

import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openstroke.geometry import IDENTITY, transform_points  # noqa: E402
from openstroke.sources import load_gt_pages  # noqa: E402

DB = r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3"


def main(obb: str, built: str, out: str) -> None:
    obb, built, out = Path(obb), Path(built), Path(out)
    pages = [r for r in json.loads((obb / "pages.json").read_text(encoding="utf-8")) if r["split"] == "val"]
    names = [ln.split(": ", 1)[1].strip() for ln in (obb / "data.yaml").read_text(encoding="utf-8").splitlines()
             if ln.startswith("  ") and ": " in ln]
    gt = {p.stem: p for p in load_gt_pages(DB)}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("score_pack/pages.json", json.dumps(pages, ensure_ascii=False))
        z.writestr("score_pack/names.json", json.dumps(names, ensure_ascii=False))
        z.write(Path(__file__).with_name("obb_score_pack.py"), "score_pack/obb_score_pack.py")
        for rec in pages:
            stem = rec["stem"]
            w, h = rec["size_pt"]
            gobj = [o for o in gt[stem].objects if len(o.get("vertices") or [])]
            if rec["transform"] != IDENTITY:
                gobj = [{**o, "vertices": transform_points(rec["transform"], np.asarray(o["vertices"], float), w, h).tolist()} for o in gobj]
            z.writestr(f"score_pack/gt/{stem}.json", json.dumps(gobj, ensure_ascii=False, default=float))
            z.write(built / f"{stem}.npz", f"score_pack/npz/{stem}.npz")
            print(stem, len(gobj), "GT objects")
    print("wrote", out, out.stat().st_size)


if __name__ == "__main__":
    main(*sys.argv[1:])
