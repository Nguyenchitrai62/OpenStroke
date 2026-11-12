"""Score pack of the remote-GT dataset (F129): like tools/make_score_pack.py, but the stroke label of every val page is recomputed from the
current GT (the e6 arrays carry labels of an older GT: doc000190 was edited on the remote deploy). Stroke geometry / styles come from e6
(they depend on the PDF only). Data only; AI_Takeoff is only read.

    py -3.12 tools/make_score_pack_remote.py datasets/obb_hvac8r datasets/e6 datasets/openstroke_score_pack_hvac8r.zip
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openstroke.geometry import IDENTITY, transform_points  # noqa: E402
from openstroke.relabel import STYLE_KEYS, object_strokes, page_segments_of  # noqa: E402
from openstroke.sources import load_gt_pages  # noqa: E402

DB = r"F:\Source_code\AI_Takeoff\BE\data\ai_takeoff.sqlite3"


def main(obb: str, built: str, out: str) -> None:
    obb, built, out = Path(obb), Path(built), Path(out)
    pages = [r for r in json.loads((obb / "pages.json").read_text(encoding="utf-8")) if r["split"] == "val"]
    names = [ln.split(": ", 1)[1].strip() for ln in (obb / "data.yaml").read_text(encoding="utf-8").splitlines()
             if ln.startswith("  ") and ": " in ln]
    gt = {p.stem: p for p in load_gt_pages(DB)}
    here = Path(__file__).resolve().parent
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("score_pack/pages.json", json.dumps(pages, ensure_ascii=False))
        z.writestr("score_pack/names.json", json.dumps(names, ensure_ascii=False))
        if (here / "obb_score_pack.py").exists():
            z.write(here / "obb_score_pack.py", "score_pack/obb_score_pack.py")
        for rec in pages:
            stem = rec["stem"]
            w, h = rec["size_pt"]
            gobj = [o for o in gt[stem].objects if len(o.get("vertices") or [])]
            if rec["transform"] != IDENTITY:
                gobj = [{**o, "vertices": transform_points(rec["transform"], np.asarray(o["vertices"], float), w, h).tolist()} for o in gobj]
            z.writestr(f"score_pack/gt/{stem}.json", json.dumps(gobj, ensure_ascii=False, default=float))
            with np.load(built / f"{stem}.npz") as src:
                d = {k: src[k] for k in src.files}
            segs = page_segments_of(d)
            attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
            old = d["label"].astype(bool)
            d["label"] = object_strokes("hvac", gobj, segs, attrs, d["length"].astype(np.float64)).astype(np.uint8)
            buf = io.BytesIO()
            np.savez_compressed(buf, **d)
            z.writestr(f"score_pack/npz/{stem}.npz", buf.getvalue())
            print(f"{stem}: {len(gobj)} GT objects, label strokes {int(old.sum())} -> {int(d['label'].sum())}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
