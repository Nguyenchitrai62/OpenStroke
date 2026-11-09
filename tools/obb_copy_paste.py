"""E77: rare HVAC objects pasted into the white space of other train pages (copy-paste augmentation, image level).

The train split has 4-12 Elbow vuông / Reducer vuông tròn / Tee and ~60 Eccentric Reducer: too few to learn their look (F78: wrong class
0.27-0.60 on val). Every rare object of a train page is cut out along its oriented box (plus ``--margin`` px, background outside the box
set to white so no unlabelled stroke comes along) and pasted, randomly rotated by 0/90/180/270 deg, into blank areas of another train page;
the label moves with it. Writes ``images/train/<stem>_cp.png`` + labels next to the existing train pages. Val pages are never touched.

    py -3.12 tools/obb_copy_paste.py datasets/obb_hvac9 --copies 40
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

RARE = {2, 4, 5, 7, 8}  # Elbow vuông, Eccentric Reducer, Reducer vuông tròn, Chân rẽ vuông tròn, Tee (data.yaml ids)


def read_labels(f: Path) -> list[tuple[int, np.ndarray]]:
    out = []
    for ln in f.read_text(encoding="utf-8").splitlines():
        v = ln.split()
        if len(v) == 9:
            out.append((int(v[0]), np.array(list(map(float, v[1:]))).reshape(4, 2)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("obb")
    ap.add_argument("--copies", type=int, default=40, help="pasted objects per new page")
    ap.add_argument("--margin", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    root = Path(a.obb)
    pages = sorted(p for p in (root / "images" / "train").glob("*.png") if "_rs" not in p.stem and not p.stem.endswith("_cp"))
    bank = []  # (class, patch BGR, polygon relative to the patch)
    for p in pages:
        labels = read_labels(root / "labels" / "train" / f"{p.stem}.txt")
        rare = [(c, q) for c, q in labels if c in RARE]
        if not rare:
            continue
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        for c, q in rare:
            poly = q * [w, h]
            x0, y0 = np.floor(poly.min(0)).astype(int) - a.margin
            x1, y1 = np.ceil(poly.max(0)).astype(int) + a.margin
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            patch = img[y0:y1, x0:x1].copy()
            mask = np.zeros(patch.shape[:2], np.uint8)
            cv2.fillPoly(mask, [np.round(poly - [x0, y0]).astype(np.int32)], 255)
            mask = cv2.dilate(mask, np.ones((2 * a.margin + 1, 2 * a.margin + 1), np.uint8))
            patch[mask == 0] = 255
            bank.append((c, patch, poly - [x0, y0]))
    print("rare objects in the bank:", len(bank), {c: sum(b[0] == c for b in bank) for c in sorted(RARE)})
    if not bank:
        return 0
    for p in pages:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (w // 16, h // 16), interpolation=cv2.INTER_AREA)
        labels = read_labels(root / "labels" / "train" / f"{p.stem}.txt")
        added, tries = [], 0
        while len(added) < a.copies and tries < a.copies * 40:
            tries += 1
            c, patch, poly = bank[rng.randrange(len(bank))]
            k = rng.randrange(4)
            ph, pw = patch.shape[:2]
            pt = np.rot90(patch, k).copy()
            q = poly.copy()
            for _ in range(k):  # rotate the polygon with np.rot90 (counter-clockwise): (x, y) -> (y, W - x), W the current width
                q = np.c_[q[:, 1], pw - q[:, 0]]
                ph, pw = pw, ph
            if pw >= w or ph >= h:
                continue
            x, y = rng.randrange(0, w - pw), rng.randrange(0, h - ph)
            if small[y // 16:(y + ph) // 16 + 1, x // 16:(x + pw) // 16 + 1].min() < 250:  # only blank paper
                continue
            img[y:y + ph, x:x + pw] = np.minimum(img[y:y + ph, x:x + pw], pt)
            small[y // 16:(y + ph) // 16 + 1, x // 16:(x + pw) // 16 + 1] = 0
            added.append((c, (q + [x, y]) / [w, h]))
        cv2.imwrite(str(root / "images" / "train" / f"{p.stem}_cp.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        lines = [f"{c} " + " ".join(f"{v:.6f}" for v in np.clip(q, 0, 1).ravel()) for c, q in labels + added]
        (root / "labels" / "train" / f"{p.stem}_cp.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(p.stem, "pasted", len(added), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
