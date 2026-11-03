"""What the FIRE mask gets wrong, page by page and by layer, from saved scores: no model run.

    py -3.12 research/probes/fire_fp_probe.py e37
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.metrics import length_counts, prf  # noqa: E402
from openstroke.postprocess import fire_mask, load_params, on_chosen_layers, style_groups, tidy  # noqa: E402
from openstroke.relabel import STYLE_KEYS, page_segments_of  # noqa: E402

DATA = Path("datasets/e5")


def one(job):
    run, fold, stem = job
    z = np.load(f"runs_colab/{run}/fold{fold}/val_scores.npz")
    d = np.load(DATA / f"{stem}.npz")
    d = {k: d[k] for k in d.files}
    if json.loads(str(d["meta"]))["mode"] != "fire":
        return None
    y, L = d["label"].astype(bool), d["length"].astype(np.float64)
    p = np.zeros(len(y))
    p[z[f"{stem}__index"]] = z[stem]
    segs = page_segments_of(d)
    attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
    group = style_groups(attrs)
    par = load_params(f"runs_colab/{run}/fold{fold}/best.pt")[0]["fire"]
    names = np.array([str(x) for x in d["layer_names"]])
    final = on_chosen_layers(tidy(fire_mask(segs, p, L, group, par), segs, attrs, group, L), p, L, list(names[d["layer_id"]]))
    lay = d["layer_id"].astype(np.int64)
    n = len(names)
    tot, fp, tp, fn = (np.bincount(lay[m], L[m], n) for m in (np.ones(len(y), bool), final & ~y, final & y, ~final & y))
    rgb = d["stroke_rgb"]
    lum = np.nan_to_num(rgb, nan=0).mean(1)
    bins = np.array([5, 20, 100, 1e9])
    lb = np.digitize(L, bins)
    return {"stem": stem, "c": length_counts(final, y, L), "fp_by_len": np.bincount(lb[final & ~y], L[final & ~y], 4).round().tolist(), "fn_by_len": np.bincount(lb[~final & y], L[~final & y], 4).round().tolist(),
            "fp_grey": float(L[final & ~y & (lum > 0.3)].sum()),
            "fp": [(names[i][:34], round(fp[i]), round(tp[i]), round(tot[i]), round(float((p[lay == i] * L[lay == i]).sum() / max(tot[i], 1e-9)), 2)) for i in np.argsort(-fp)[:5] if fp[i] > 0],
            "fn": [(names[i][:34], round(fn[i]), round(tp[i]), round(float((p[lay == i] * L[lay == i]).sum() / max(tot[i], 1e-9)), 2)) for i in np.argsort(-fn)[:3] if fn[i] > 0]}


def main(run):
    jobs = [(run, f.name[4:], pg["stem"]) for f in sorted(Path(f"runs_colab/{run}").glob("fold*")) for pg in json.loads((f / "result.json").read_text(encoding="utf-8"))["pages"]]
    with ProcessPoolExecutor(3) as ex:
        rows = [r for r in ex.map(one, jobs) if r]
    m = prf(*np.sum([r["c"] for r in rows], 0))
    print(f"all F2 {m['f2']:.3f} R {m['recall']:.3f} P {m['precision']:.3f}")
    for r in rows:
        m = prf(*r["c"])
        print(f"\n{r['stem'][:9]} F2 {m['f2']:.3f} R {m['recall']:.2f} P {m['precision']:.2f} tp {r['c'][0]:.0f} fp {r['c'][1]:.0f} fn {r['c'][2]:.0f} | fp by len <5,<20,<100,long {r['fp_by_len']} fn {r['fn_by_len']} fp grey {r['fp_grey']:.0f}")
        print("  FP layer,fp,tp,total,meanp:", r["fp"])
        print("  FN layer,fn,tp,meanp:", r["fn"])


if __name__ == "__main__":
    main(sys.argv[1])
