"""Score pack of a re-split dataset (F137, openstroke.obb.resplit): the val pages of the new split, two groups scored with the same rule_nl
decode. GT val pages are copied from the GT score pack (their stroke labels from the current GT, tools/make_score_pack_remote.py); Viewer val
pages are built from their payload + Viewer "Labels V1" boxes with the same label rule (relabel.object_strokes). comparison.json names the
groups: "main" (the GT pages: MAIN F2) and "groups" (GT, viewer) reported beside it. Data only; AI_Takeoff and the Viewer are only read.

    py -3.12 tools/make_score_pack_split.py datasets/obb_hvac10b datasets/openstroke_score_pack_hvac8r.zip \
        datasets/viewer_hvac19/exports_20261113/labels.json datasets/openstroke_score_pack_hvac10b.zip
"""
from __future__ import annotations

import gzip
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from openstroke.build import shape_attributes  # noqa: E402
from openstroke.geometry import IDENTITY, page_segments, transform_points  # noqa: E402
from openstroke.relabel import object_strokes  # noqa: E402
from viewer_obb_dataset import CLASS_MAP, SET  # noqa: E402


def viewer_objects(labels, stem):
    key = stem[3:]                                   # vw_<sha256[:12]>_p<page>
    sha, page = key.rsplit("_p", 1)
    for p in labels["pdfs"]:
        if p["sha256"][:12] != sha:
            continue
        for pg in p["pages"]:
            if int(pg["page"]) == int(page):
                return [{"shape_name": CLASS_MAP[o["class"]], "vertices": o["points"]}
                        for o in (pg["labels"].get(SET) or {}).get("objects") or [] if o["class"] in CLASS_MAP]
    raise SystemExit(f"no Viewer labels for {stem}")


def main(data, gt_pack, labels_json, out):
    data, out = Path(data), Path(out)
    repo = data.resolve().parents[1]
    rows = [r for r in json.loads((data / "pages.json").read_text(encoding="utf-8")) if r["split"] == "val"]
    plan = json.loads((data / "gt_provenance.json").read_text(encoding="utf-8")).get("split_plan") or {}
    labels = json.loads(Path(labels_json).read_text(encoding="utf-8"))
    src = zipfile.ZipFile(gt_pack)
    have = {Path(n).stem for n in src.namelist() if n.endswith(".npz")}
    gt_rows = [r for r in rows if not r["stem"].startswith("vw_")]
    vw_rows = [r for r in rows if r["stem"].startswith("vw_")]
    missing = {r["stem"] for r in gt_rows} - have
    if missing:
        raise SystemExit(f"the GT score pack lacks val pages {sorted(missing)}")
    comparison = {"name": plan.get("name") or data.name, "main": sorted({r["stem"][:9] for r in gt_rows}),
                  "groups": {"GT": sorted({r["stem"][:9] for r in gt_rows}), "viewer": sorted({r["stem"][:9] for r in vw_rows})}}
    if set(comparison["groups"]["GT"]) & set(comparison["groups"]["viewer"]):
        raise SystemExit("page prefixes of the two groups overlap")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("score_pack/pages.json", json.dumps(rows, ensure_ascii=False))
        z.writestr("score_pack/names.json", src.read("score_pack/names.json"))
        z.writestr("score_pack/comparison.json", json.dumps(comparison, ensure_ascii=False, indent=1))
        if "score_pack/obb_score_pack.py" in src.namelist():
            z.writestr("score_pack/obb_score_pack.py", src.read("score_pack/obb_score_pack.py"))
        for r in gt_rows:
            for kind, suf in (("gt", ".json"), ("npz", ".npz")):
                z.writestr(f"score_pack/{kind}/{r['stem']}{suf}", src.read(f"score_pack/{kind}/{r['stem']}{suf}"))
            print(f"GT     {r['stem']}: copied", flush=True)
        for r in vw_rows:
            with gzip.open(repo / r["payload"].replace("\\", "/"), "rt", encoding="utf-8") as fh:
                shapes = json.load(fh).get("shapes") or []
            gobj = viewer_objects(labels, r["stem"])
            w, h = r["size_pt"]
            if r["transform"] != IDENTITY:
                gobj = [{**o, "vertices": transform_points(r["transform"], np.asarray(o["vertices"], float), w, h).tolist()} for o in gobj]
            segs = page_segments(shapes)
            attrs = shape_attributes(shapes, segs)
            length = np.bincount(segs.owner, weights=segs.length, minlength=len(shapes)).astype(np.float32)
            label = object_strokes("hvac", gobj, segs, attrs, length.astype(np.float64)).astype(np.uint8)
            buf = io.BytesIO()
            np.savez_compressed(buf, label=label, length=length, segments=segs.seg.astype(np.float32),
                                shape_offsets=segs.shape_offsets, **attrs)
            z.writestr(f"score_pack/npz/{r['stem']}.npz", buf.getvalue())
            z.writestr(f"score_pack/gt/{r['stem']}.json", json.dumps(gobj, ensure_ascii=False, default=float))
            print(f"viewer {r['stem']}: {len(gobj)} objects, label strokes {int(label.sum())} / {len(label)}", flush=True)
    print(f"{out}: {len(gt_rows)} GT + {len(vw_rows)} Viewer val pages; comparison {comparison}")


if __name__ == "__main__":
    main(*sys.argv[1:])
