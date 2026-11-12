"""Crops of the val regions where E69 (boxes + label rule) predicts a Duct that lies mostly on strokes the GT leaves unlabelled: candidates for
missing GT labels (AGENTS 3.2). 0 GPU, saved predictions. Red = the prediction's box, green = GT objects nearby. One PNG per candidate plus
index.md, written to <out>.

    py -3.12 research/probes/gt_review_export.py <obb_hvac7> <final_json_rule folder> <out folder> [max]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from openstroke.assemble import assign_members
from openstroke.obb.object_score import PIPELINE_PAGES
from openstroke.obb.vector_validation import DatasetVectorGT, HVAC_CLASSES


def main(folder, preds, out, limit="80", split="val"):
    folder, preds, out, limit = Path(folder), Path(preds), Path(out), int(limit)
    out.mkdir(parents=True, exist_ok=True)
    names = list(yaml.safe_load((folder / "data.yaml").read_text(encoding="utf-8"))["names"].values())
    stems = {f.stem for f in preds.glob("*.json")}
    gt = DatasetVectorGT(folder / "images" / split, names, comparison_pages=tuple(sorted({x[:9] for x in stems})))
    canon = {n.casefold(): i for i, n in enumerate(HVAC_CLASSES)}
    D = HVAC_CLASSES.index("Duct")
    scale = {r["stem"]: r.get("scale", 4.0) for r in json.loads((folder / "pages.json").read_text(encoding="utf-8")) if r.get("stem")}
    cands = []
    for stem, (L, inst, gcls, at) in gt.pages.items():
        if stem not in stems:
            continue
        objs = [o for o in json.loads((preds / f"{stem}.json").read_text(encoding="utf-8"))["objects"] if o["score"] >= 0.2]
        keep = [(canon[o["shape_name"].replace("_", " ").casefold()], np.empty((0, 2)), o["score"], [at[p] for p in o["shapes"] if p in at]) for o in objs]
        pred = assign_members(keep, L, order="score", dedupe=False)
        for i, (c, _, sc, _) in enumerate(keep):
            s = pred == i
            if c != D or not s.any():
                continue
            bg = L[s & (inst < 0)].sum() / max(L[s].sum(), 1e-9)
            if bg >= 0.8 and L[s].sum() > 40:
                cands.append((sc, stem, np.asarray(objs[i]["vertices"], float) * scale[stem], bg))
    cands.sort(key=lambda x: -x[0])
    lines = ["# Vùng nghi GT thiếu nhãn (E69 đoán Duct, ≥ 80 % chiều dài nét của nó không thuộc object GT nào)", "",
             "Đỏ: hộp dự đoán. Xanh lá: object GT gần đó. Nếu vùng đỏ đúng là ống gió, GT đang thiếu nhãn.", ""]
    imgs = {}
    for k, (sc, stem, box, bg) in enumerate(cands[:limit]):
        if stem not in imgs:
            imgs[stem] = cv2.imread(str(folder / "images" / split / f"{stem}.png"))
        img = imgs[stem]
        x0, y0 = np.floor(box.min(0)).astype(int) - 150
        x1, y1 = np.ceil(box.max(0)).astype(int) + 150
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
        crop = img[y0:y1, x0:x1].copy()
        cv2.polylines(crop, [np.round(box - [x0, y0]).astype(np.int32)], True, (0, 0, 255), 3)
        name = f"{k:03d}_{stem[:9]}_score{sc:.2f}.png"
        cv2.imwrite(str(out / name), crop)
        cx, cy = box.mean(0) / scale[stem]
        lines.append(f"- {k:03d} `{stem}` tâm ({cx:.0f}, {cy:.0f}) pt, điểm {sc:.2f}, {bg:.0%} nét nền: ![]({name})")
    (out / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(cands)} candidates, {min(limit, len(cands))} exported to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
