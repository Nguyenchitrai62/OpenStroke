"""OpenStroke39 / E139 (H158): augmented copies of the train pages where the way strokes are cut into PDF items changes, the labels do not.

A CAD export cuts the same drawing into items in its own way (F186: items span objects, polylines run on), and the token model reads that
encoding (E135a vs E135b, F215). Each train page of e12 gets one copy where
  - SPLIT_SHARE of its stroke-only shapes (>= MIN_LEN pt) are cut in two: at a random vertex of a polyline, or at t in [0.25, 0.75] of a
    single segment;
  - MERGE_SHARE of the pairs of consecutive stroke-only shapes with the same style, label, layer and object that meet end to start are
    joined into one shape.
Pieces keep their parent's label, object, colour, width and the other attributes; length and bbox are recomputed. The picture is drawn
from the vectors, so image and vectors stay in sync. Val pages are never touched; copies share their page's PDF, so they sit on the same
side of every split (AGENTS 0.3 / 0.4). The original GT is not modified.

Output: a folder holding only the copies + a manifest / folds that list e12 and the copies. On Colab:
    cp -al /content/os34/e12 /content/os34/e12s && unzip -q -o openstroke_e12s_addon.zip -d /content/os34
    py -3.12 tools/make_split_aug.py datasets/e12 datasets/e12s [datasets/openstroke_e12s_addon.zip]
"""
from __future__ import annotations

import json
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np

SPLIT_SHARE = 0.2
MERGE_SHARE = 0.3
MIN_LEN = 4.0
JOIN = 0.05        # pt: end of one shape to start of the next
SUFFIX = "_sa1"


def augment_page(d: dict, rng: np.random.Generator) -> tuple[dict, dict]:
    """-> (new npz arrays, counts). Per-shape arrays are every array whose first dimension is the number of shapes."""
    off = d["shape_offsets"].astype(np.int64)
    seg = d["segments"].astype(np.float32)
    n = len(off) - 1
    per_shape = [k for k, v in d.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n and k not in ("shape_offsets",)
                 and not k.startswith("pc_")]
    nseg = np.diff(off)
    stroke = (d["type_code"] == 0) & (nseg > 0)
    length = d["length"].astype(np.float64)
    cand = np.flatnonzero(stroke & (length >= MIN_LEN))
    split = np.zeros(n, bool)
    split[rng.choice(cand, size=int(round(SPLIT_SHARE * len(cand))), replace=False)] = True
    # merge candidates: consecutive stroke-only shapes, same style / label / layer / object, end meets start, neither split
    first = off[:-1]
    last = off[1:] - 1
    i = np.arange(n - 1)
    ok = stroke[:-1] & stroke[1:] & ~split[:-1] & ~split[1:]
    ok &= (d["label"][:-1] == d["label"][1:]) & (d["layer_id"][:-1] == d["layer_id"][1:])
    if "oinst" in d:
        ok &= d["oinst"][:-1] == d["oinst"][1:]
    ok &= np.all(np.isclose(d["stroke_rgb"][:-1], d["stroke_rgb"][1:], equal_nan=True), axis=1)
    ok &= np.isclose(d["width"][:-1], d["width"][1:], equal_nan=True)
    gap = np.hypot(seg[np.clip(last[:-1], 0, None), 2] - seg[np.clip(first[1:], 0, None), 0],
                   seg[np.clip(last[:-1], 0, None), 3] - seg[np.clip(first[1:], 0, None), 1]) if len(seg) else np.zeros(n - 1)
    ok &= gap <= JOIN
    merge = np.zeros(n - 1, bool)
    pairs = i[ok]
    merge[pairs[rng.random(len(pairs)) < MERGE_SHARE]] = True
    merge &= ~np.r_[False, merge[:-1]]          # a shape joins at most one neighbour (no chains)

    src, parts, close_keep, ops_sum = [], [], [], {}
    s = 0
    while s < n:
        a, b = off[s], off[s + 1]
        if s < n - 1 and merge[s]:
            ops_sum[len(src)] = s
            src.append(s)
            parts.append(np.r_[seg[a:b], seg[off[s + 1]:off[s + 2]]])
            close_keep.append(False)
            s += 2
            continue
        if split[s]:
            ss = seg[a:b]
            if len(ss) >= 2:
                c = int(rng.integers(1, len(ss)))
                halves = (ss[:c], ss[c:])
            else:
                t = float(rng.uniform(0.25, 0.75))
                p = ss[0, :2] + t * (ss[0, 2:] - ss[0, :2])
                halves = (np.array([[*ss[0, :2], *p]], np.float32), np.array([[*p, *ss[0, 2:]]], np.float32))
            for h in halves:
                src.append(s)
                parts.append(h)
                close_keep.append(False)
        else:
            src.append(s)
            parts.append(seg[a:b])
            close_keep.append(True)
        s += 1
    src = np.asarray(src, np.int64)
    out = {k: v for k, v in d.items() if k not in per_shape and k not in ("segments", "shape_offsets") and not k.startswith("pc_")}
    for k in per_shape:
        out[k] = d[k][src].copy()
    lens = [np.hypot(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]).sum() if len(p) else 0.0 for p in parts]
    out["segments"] = (np.concatenate(parts) if parts else np.zeros((0, 4))).astype(d["segments"].dtype)
    out["shape_offsets"] = np.r_[0, np.cumsum([len(p) for p in parts])].astype(np.int64)
    out["length"] = np.asarray(lens, d["length"].dtype)
    bb = np.array([[p[:, [0, 2]].min(), p[:, [1, 3]].min(), p[:, [0, 2]].max(), p[:, [1, 3]].max()] if len(p) else d["bbox"][j]
                   for p, j in zip(parts, src)], np.float32)
    out["bbox"] = bb.astype(d["bbox"].dtype)
    if "close_path" in out:
        out["close_path"] = np.where(np.asarray(close_keep), out["close_path"], 0).astype(out["close_path"].dtype)
    if "op_counts" in out:   # a joined shape counts both shapes' commands
        for row, s0 in ops_sum.items():
            out["op_counts"][row] = d["op_counts"][s0] + d["op_counts"][s0 + 1]
    return out, {"shapes": n, "split": int(split.sum()), "merged": int(merge.sum()), "after": len(src)}


def main(src, out, zip_out=None):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    train = [r for r in man["records"] if r.get("status") == "ok" and r.get("split") == "train"]
    tot = {"shapes": 0, "split": 0, "merged": 0, "after": 0}
    new_recs, new_phys = [], []
    for k, r in enumerate(train, 1):
        stem = r["stem"] + SUFFIX
        rng = np.random.default_rng(zlib.crc32(stem.encode()))
        with np.load(src / r["npz"]) as z:
            d = {kk: z[kk] for kk in z.files}
        arr, c = augment_page(d, rng)
        np.savez_compressed(out / f"{stem}.npz", **arr)
        for kk in tot:
            tot[kk] += c[kk]
        new_recs.append({**{kk: v for kk, v in r.items() if kk not in ("stem", "npz")}, "stem": stem, "npz": f"{stem}.npz",
                         "augmented_from": r["stem"], "augment": f"split {SPLIT_SHARE} / merge {MERGE_SHARE} (tools/make_split_aug.py)"})
        new_phys.append({"document_sha256": r["document_sha256"], "page_number": r["page_number"], "filename": r.get("filename", ""),
                         "mode": r.get("mode", "hvac"), "gt_variants": [stem], "duplicate_gt": False, "decision": "augmented copy"})
        print(f"[{k}/{len(train)}] {stem}: {c}", flush=True)
    man = dict(man)
    man["records"] = man["records"] + new_recs
    man["physical_pages"] = man.get("physical_pages", []) + new_phys
    man["e12s"] = {"tool": "tools/make_split_aug.py", "copies": len(new_recs), "split_share": SPLIT_SHARE, "merge_share": MERGE_SHARE, **tot}
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for f in src.glob("folds*.json"):
        (out / f.name).write_bytes(f.read_bytes())
    print("total", tot)
    if zip_out:
        with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_STORED) as zf:
            for f in sorted(out.iterdir()):
                zf.write(f, f"e12s/{f.name}")
        print(f"{zip_out}: {Path(zip_out).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:])
