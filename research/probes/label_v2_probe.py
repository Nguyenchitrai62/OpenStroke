"""Labels v1 (e1) -> v2 (e2, HVAC refinement): what changed, and what the same predictions score against each.

1. FIRE / PLUMBING records must be identical. 2. HVAC: true length removed per page and the layers it came from.
3. Predictions cached by ``error_probe.py`` (``runs/error_probe/<stem>.prob.npy``) scored against v1 and v2 labels.

    py -3.12 research/probes/label_v2_probe.py datasets/e1 datasets/e2 > research/probes/label_v2_probe.log
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.metrics import length_counts, prf  # noqa: E402


def main(a_dir, b_dir, thr=0.10):
    a_dir, b_dir = Path(a_dir), Path(b_dir)
    A = {r["stem"]: r for r in json.loads((a_dir / "manifest.json").read_text(encoding="utf-8"))["records"]}
    B = {r["stem"]: r for r in json.loads((b_dir / "manifest.json").read_text(encoding="utf-8"))["records"]}
    same = {"fire": 0, "plumbing": 0}
    diff = []
    tot = defaultdict(float)
    lay_removed = defaultdict(float)
    rows = []
    for stem, rb in B.items():
        ra = A.get(stem)
        if ra is None or rb.get("status") != "ok":
            print("not comparable:", stem)
            continue
        x, y = np.load(a_dir / ra["npz"]), np.load(b_dir / rb["npz"])
        la, lb, L = x["label"].astype(bool), y["label"].astype(bool), y["length"].astype(np.float64)
        if rb["mode"] != "hvac":
            if np.array_equal(la, lb):
                same[rb["mode"]] += 1
            else:
                diff.append(stem)
            continue
        assert not (lb & ~la).any(), "v2 only removes true labels"
        rem = la & ~lb
        tot["v1"] += L[la].sum()
        tot["removed"] += L[rem].sum()
        names = [str(n) for n in y["layer_names"]]
        for i in np.unique(y["layer_id"][rem]):
            lay_removed[names[i] if i < len(names) else "?"] += L[rem & (y["layer_id"] == i)].sum()
        rows.append((L[rem].sum() / max(L[la].sum(), 1e-9), stem, rb["filename"], L[la].sum()))
    print(f"FIRE / PLUMBING identical: {same}; different: {diff}")
    print(f"HVAC true length v1 {tot['v1']:.0f} pt, removed by v2 {tot['removed']:.0f} pt "
          f"({100 * tot['removed'] / max(tot['v1'], 1e-9):.1f}%)")
    print("\nHVAC pages, share of v1 true length that v2 makes other")
    for f, stem, fn, T in sorted(rows, reverse=True):
        print(f"  {f:6.1%}  {stem:24s} {fn[:44]:44s} v1 true {T:9.0f} pt")
    print("\nLayers that lost the most true length (all HVAC pages)")
    for k, v in sorted(lay_removed.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {v:9.0f} pt  {k[:90]}")
    print(f"\nSame predictions (runs/error_probe, threshold {thr}) against v1 and v2 labels")
    for f in sorted(Path("runs/error_probe").glob("*.prob.npy")):
        stem = f.name.replace(".prob.npy", "")
        if stem not in A or stem not in B:
            continue
        p = np.load(f) >= thr
        out = []
        for d, r in ((a_dir, A[stem]), (b_dir, B[stem])):
            z = np.load(d / r["npz"])
            out.append(prf(*length_counts(p, z["label"], z["length"])))
        print(f"  {stem} {B[stem]['mode']}: v1 R {out[0]['recall']:.3f} P {out[0]['precision']:.3f} F2 {out[0]['f2']:.3f}"
              f"  ->  v2 R {out[1]['recall']:.3f} P {out[1]['precision']:.3f} F2 {out[1]['f2']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
