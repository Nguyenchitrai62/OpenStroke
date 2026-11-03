"""E53: on labels v3, what the FIRE seed and the chosen-layers step cost in missed pages. Saved scores only, no model run.

    py -3.12 research/probes/label_v3_seed_probe.py
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.metrics import length_counts, prf  # noqa: E402
from openstroke.postprocess import fire_mask, on_chosen_layers, tidy  # noqa: E402
from openstroke.relabel import STYLE_KEYS, page_segments_of  # noqa: E402

DATA = Path("datasets/e5")
CACHE = Path("runs_colab/chain_cache/e37")
SEEDS = (0.35, 0.45, 0.55, 0.65, 0.75)


def one(stem):
    z = np.load(DATA / f"{stem}.npz")
    d = {k: z[k] for k in z.files}
    g = np.load(CACHE / f"{stem}.npz")
    if str(g["mode"]) != "fire":
        return None
    segs = page_segments_of(d)
    y, L, group, p = d["label"].astype(bool), d["length"].astype(np.float64), g["group"], g["p"]
    attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
    names = [str(d["layer_names"][i]) for i in d["layer_id"]]
    out = {"stem": stem}
    for s in SEEDS:
        m = tidy(fire_mask(segs, p, L, group, {"seed": s, "group": 0.6}), segs, attrs, group, L)
        out[f"{s} all"] = length_counts(m, y, L)
        out[f"{s} chosen"] = length_counts(on_chosen_layers(m, p, L, names), y, L)
    return out


def main():
    with ProcessPoolExecutor(3) as ex:
        rows = [r for r in ex.map(one, [f.stem for f in sorted(CACHE.glob("*.npz"))]) if r]
    print("doc | " + " | ".join(str(r["stem"][:9]) for r in rows))
    for k in [f"{s} {c}" for s in SEEDS for c in ("all", "chosen")]:
        m = prf(*np.sum([r[k] for r in rows], axis=0))
        rec = sorted(prf(*r[k])["recall"] for r in rows)
        print(f"seed {k:12s} F2 {m['f2']:.3f} R {m['recall']:.3f} P {m['precision']:.3f}  recall by page {[round(prf(*r[k])['recall'], 2) for r in rows]}  pages under 0.8: {sum(x < 0.8 for x in rec)}")


if __name__ == "__main__":
    main()
