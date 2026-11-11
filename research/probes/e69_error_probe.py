"""Where does E69 lose F2 object? 0 GPU, no inference: reads E69's saved member predictions (final_json_members) and the
native GT (vectors inst/pid), assigns strokes exactly as the native scorer, and puts every GT object of the 7 comparison pages
in one bucket:
  matched     same class, stroke-length IoU >= 0.5 with one prediction
  wrong_class a prediction of another class has IoU >= 0.5
  merged      one prediction holds >= 50 % of the object's length but is much larger (IoU < 0.5): objects fused
  split       no prediction holds 50 %, but predictions together hold >= 50 %: object broken into pieces
  partial     predictions hold 20-50 % of its length
  missed      predictions hold < 20 % of its length
Also by class and by size (long side of the object in image px; a 1024 px tile with 200 px overlap cannot hold > ~717 px at 70 %).

    py -3.12 research/probes/e69_error_probe.py <obb_hvac7 folder> <final_json_members folder> [conf]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from openstroke.assemble import assign_members
from openstroke.obb.vector_validation import DatasetVectorGT, HVAC_CLASSES
from openstroke.obb.object_score import PIPELINE_PAGES


def main(folder, preds, conf="0.2"):
    folder, preds, conf = Path(folder), Path(preds), float(conf)
    names = list(yaml.safe_load((folder / "data.yaml").read_text(encoding="utf-8"))["names"].values())
    gt = DatasetVectorGT(folder / "images" / "val", names)
    canon = {n.casefold(): i for i, n in enumerate(HVAC_CLASSES)}
    cid = {n: canon[n.replace("_", " ").casefold()] for n in names}
    buckets, by_cls, by_size, fp = Counter(), defaultdict(Counter), defaultdict(Counter), Counter()
    by_n = defaultdict(Counter)
    tp_n = pred_n = gt_n = 0
    for stem, (length, inst, gt_class, at) in gt.pages.items():
        if stem[:9] not in PIPELINE_PAGES:
            continue
        with np.load(folder / "vectors" / "val" / f"{stem}.npz") as z:
            pts = np.asarray(z["tokens"])[:, 1:17].reshape(-1, 8, 2)
        objs = json.loads((preds / f"{stem}.json").read_text(encoding="utf-8"))["objects"]
        keep = []
        for o in objs:
            if o["score"] < conf:
                continue
            rows = [at[p] for p in o["shapes"] if p in at]
            keep.append((cid[o["shape_name"]] if o["shape_name"] in cid else canon[o["shape_name"].casefold()], np.empty((0, 2)), float(o["score"]), rows))
        pred = assign_members(keep, length, order="score", dedupe=False)
        pcls = np.array([k[0] for k in keep], np.int64)
        lp = np.bincount(pred[pred >= 0], length[pred >= 0], minlength=len(keep))
        used_p = set()
        gids = np.unique(inst[inst >= 0])
        for g in gids:
            sel = inst == g
            lg = length[sel].sum()
            c = gt_class[int(g)]
            share = np.bincount(pred[sel & (pred >= 0)], length[sel & (pred >= 0)], minlength=len(keep)) if len(keep) else np.zeros(0)
            covered = share.sum() / max(lg, 1e-9)
            iou = share / np.maximum(lp + lg - share, 1e-9) if len(keep) else np.zeros(0)
            best = int(np.argmax(iou)) if len(iou) else -1
            if best >= 0 and iou[best] >= 0.5 and pcls[best] == c:
                b = "matched"; used_p.add(best)
            elif best >= 0 and iou[best] >= 0.5:
                b = "wrong_class"
            elif len(share) and share.max() >= 0.5 * lg:
                b = "merged"
            elif covered >= 0.5:
                b = "split"
            elif covered >= 0.2:
                b = "partial"
            else:
                b = "missed"
            p = pts[sel].reshape(-1, 2)
            ext = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]))
            size = ("a <50 px" if ext < 50 else "b 50-100 px" if ext < 100 else "c 100-200 px" if ext < 200
                    else "d 200-717 px" if ext < 717 else "e >717 px (no tile holds it)")
            buckets[b] += 1; by_cls[HVAC_CLASSES[c]][b] += 1; by_size[size][b] += 1
            k = int(sel.sum()); by_n["a 1 stroke" if k == 1 else "b 2 strokes" if k == 2 else "c 3-4 strokes" if k <= 4 else "d 5-10 strokes" if k <= 10 else "e >10 strokes"][b] += 1
        gt_n += len(gids); pred_n += len(keep); tp_n += buckets["matched"] - sum(fp.values()) * 0
    total = sum(buckets.values())
    order = ["matched", "wrong_class", "merged", "split", "partial", "missed"]
    print(f"conf {conf}: {total} GT objects on {len(PIPELINE_PAGES)} pages, {pred_n} predictions")
    print("all      " + "  ".join(f"{b} {buckets[b]} ({buckets[b] / total:.1%})" for b in order))
    for title, table in (("class", by_cls), ("size", by_size), ("strokes", by_n)):
        print(f"-- by {title}")
        for k, cnt in (sorted(table.items()) if title in ("size", "strokes") else sorted(table.items(), key=lambda kv: -sum(kv[1].values()))):
            n = sum(cnt.values())
            print(f"{k[:28]:28s} n={n:5d}  " + "  ".join(f"{b} {cnt[b] / n:.0%}" for b in order))


if __name__ == "__main__":
    main(*sys.argv[1:])
