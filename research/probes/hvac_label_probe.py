"""HVAC rule "true = most of the length inside a Duct / Elbow / Reducer polygon": how much of that true length is on layers
that are mostly *outside* every duct (architecture, structure, text drawn under the duct)?

Per page and layer: share of the layer's length that is true. True length on layers whose share is below ``LOW`` is
"under-duct" true: strokes of another discipline that the polygon rule calls pipe. Same for FIRE / PLUMBING (line rule)
for comparison. No model involved.

    py -3.12 research/probes/hvac_label_probe.py datasets/e1 > research/probes/hvac_label_probe.log
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LOW = (0.1, 0.3)


def main(data):
    data = Path(data)
    man = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    tot = defaultdict(lambda: np.zeros(1 + len(LOW)))
    rows = []
    for r in man["records"]:
        if r.get("status") != "ok":
            continue
        z = np.load(data / r["npz"])
        y = z["label"].astype(bool)
        L = z["length"].astype(np.float64)
        lid = z["layer_id"].astype(np.int64)
        n = int(lid.max()) + 1 if len(lid) else 1
        lay_all = np.bincount(lid, weights=L, minlength=n)
        lay_true = np.bincount(lid, weights=L * y, minlength=n)
        share = lay_true / np.maximum(lay_all, 1e-9)
        T = L[y].sum()
        under = [L[y & (share[lid] < lo)].sum() for lo in LOW]
        tot[r["mode"]] += np.array([T, *under])
        if r["mode"] == "hvac":
            rows.append((under[1] / max(T, 1e-9), r["stem"], r["filename"], T, under))
    for mode, v in sorted(tot.items()):
        print(f"{mode:8s} true {v[0]:12.0f} pt | on layers with true share < {LOW[0]}: {100 * v[1] / v[0]:5.1f}% "
              f"| < {LOW[1]}: {100 * v[2] / v[0]:5.1f}%")
    print("\nHVAC pages, share of true length on layers with true share < 0.3 (worst first)")
    for frac, stem, fn, T, under in sorted(rows, reverse=True):
        print(f"  {frac:6.1%}  {stem:24s} {fn[:44]:44s} true {T:9.0f} pt")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "datasets/e1")
