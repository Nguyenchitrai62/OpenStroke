"""Small GT defects in a built obb dataset (derived copy only; AI_Takeoff GT is never written): duplicate objects (same class, rotated-box
IoU >= 0.9), degenerate boxes (short side < 1 px), objects that own no stroke (no vector token has their inst). With --fix, duplicates and
degenerate boxes are removed from labels/ and vectors/ (inst re-indexed); objects without strokes are kept (a box is still a box) and listed.

    py -3.12 tools/gt_clean_check.py datasets/obb_hvac8r [--fix]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from openstroke.obb.ops.obb import poly_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args()
    report = {}
    for split in ("train", "val"):
        for lab in sorted((a.root / "labels" / split).glob("*.txt")):
            rows = [l.split() for l in lab.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not rows:
                continue
            cls = np.array([int(r[0]) for r in rows]); poly = np.array([[float(v) for v in r[1:9]] for r in rows])
            from PIL import Image
            w, h = Image.open(a.root / "images" / split / (lab.stem + ".png")).size
            px = poly * np.tile([w, h], 4)
            q = px.reshape(-1, 4, 2)
            side = np.minimum(np.linalg.norm(q[:, 1] - q[:, 0], axis=1), np.linalg.norm(q[:, 2] - q[:, 1], axis=1))
            bad = set(np.flatnonzero(side < 1.0).tolist())
            iou = poly_iou(px, px)
            dup = set()
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    if cls[i] == cls[j] and iou[i, j] >= 0.9 and j not in dup:
                        dup.add(j)
            vec = a.root / "vectors" / split / (lab.stem + ".npz")
            with np.load(vec) as z:
                tok = np.asarray(z["tokens"])
            inst = np.rint(tok[:, 22]).astype(int)
            nostroke = sorted(set(range(len(rows))) - set(inst[inst >= 0].tolist()))
            report[lab.stem] = dict(objects=len(rows), duplicates=len(dup), degenerate=len(bad), without_strokes=len(nostroke))
            drop = dup | bad
            if a.fix and drop:
                keep = [i for i in range(len(rows)) if i not in drop]
                remap = {old: new for new, old in enumerate(keep)}
                lab.write_text("".join(" ".join(rows[i]) + "\n" for i in keep), encoding="utf-8")
                tok[:, 22] = [remap.get(int(v), -1) if v >= 0 else v for v in inst]
                np.savez_compressed(vec, tokens=tok)
    tot = {k: sum(r[k] for r in report.values()) for k in ("objects", "duplicates", "degenerate", "without_strokes")}
    (a.root / "gt_clean.json").write_text(json.dumps(dict(total=tot, pages=report, fixed=a.fix), indent=1), encoding="utf-8")
    print("total", tot)
    for s, r in report.items():
        if r["duplicates"] or r["degenerate"] or r["without_strokes"]:
            print(s, r)


if __name__ == "__main__":
    main()
