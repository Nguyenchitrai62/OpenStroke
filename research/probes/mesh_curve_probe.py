"""E52, user 2026-11-02: honeycomb noise on MB TH4 and missing curves. Two label-free rules on saved E37 scores, no model run.

``mesh``: short strokes connected into a large cluster of junctions (insulation hatch), stroke by stroke. Dashes do not meet at junctions, so dashed pipes stay.
``bridge``: a chain of curve shapes with free ends all resting on the mask (an elbow between two pipes). Closed loops (symbols) have no free ends.

The current labels call hatch inside a duct true (F54), so results are also shown on labels without the mesh strokes, and curves on pipe layers are counted apart.

    py -3.12 research/probes/mesh_curve_probe.py e37
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.geometry import PageSegments  # noqa: E402
from openstroke.metrics import length_counts, prf  # noqa: E402
from openstroke.postprocess import bridge_curves, by_group, load_params, mesh_strokes, style_groups  # noqa: E402

OP_C = 1


def pages(run):
    params, _ = load_params(f"runs_colab/{run}/fold0/best.pt")
    for k in range(5):
        for pg in json.load(open(f"runs_colab/{run}/fold{k}/result.json"))["pages"]:
            s = pg["stem"]
            f = Path(f"runs_colab/chain_cache/{run}/{s}.npz")
            if not f.exists():
                continue
            g = np.load(f)
            d = np.load(f"datasets/e3/{s}.npz")
            mode = str(g["mode"])
            p, L = g["p"], g["L"]
            attrs = {x: d[x] for x in ("layer_id", "stroke_rgb", "width", "type_code")}
            group = style_groups(attrs)
            par = params[mode]
            base = g["obj"] if mode == "hvac" else g["t0.55"]
            rule = base | by_group(p, L, group, par["group"])
            if mode == "hvac":
                rule = rule | (p >= par["sure"])
            off = d["shape_offsets"].astype(np.int64)
            segs = PageSegments(seg=d["segments"].astype(np.float64), owner=np.repeat(np.arange(len(off) - 1), np.diff(off)), shape_offsets=off)
            curve = d["op_counts"][:, OP_C] > 0
            lay = g["lay"]
            n = lay.max() + 1
            share = np.bincount(lay[g["y"]], L[g["y"]], n) / np.maximum(np.bincount(lay, L, n), 1e-9)
            mesh = mesh_strokes(segs, group, L)
            bridged = bridge_curves(segs, rule, curve)
            yield dict(stem=s, mode=mode, y=g["y"], L=L, p=p, rule=rule, curve=curve, mesh=mesh, bridged=bridged, pipe=(share >= 0.5)[lay], group=group, names=[str(x) for x in d["layer_names"]], lay=lay)
            print(s[:21], mode, flush=True)


def show(name, ps, m, y):
    v = prf(*np.sum([length_counts(m(g), y(g), g["L"]) for g in ps], 0))
    print(f"  {name:30s} F2 {v['f2']:.3f} R {v['recall']:.3f} P {v['precision']:.3f}", flush=True)


def main(run):
    ps = list(pages(run))
    print("\ngroups the mesh rule takes (true % = current labels):")
    for g in ps:
        for i in np.unique(g["group"][g["mesh"]]):
            c = g["group"] == i
            j = np.flatnonzero(c)[0]
            print(f"  {g['stem'][:9]} {g['mode']} {g['L'][c].sum() / 1e3:7.1f}k pt {c.sum():7d} shapes mean {g['L'][c].mean():5.1f} true {g['L'][c & g['y']].sum() / max(g['L'][c].sum(), 1) * 100:5.1f} % {g['names'][g['lay'][j]][:30]}")
    for mode in ("fire", "hvac"):
        a = [g for g in ps if g["mode"] == mode]
        print(f"\n{mode}, {len(a)} pages; mesh length {sum(g['L'][g['mesh']].sum() for g in a) / 1e3:.0f}k pt, of it labelled true {sum(g['L'][g['mesh'] & g['y']].sum() for g in a) / 1e3:.0f}k")
        now = lambda g: g["y"]  # noqa: E731
        clean = lambda g: g["y"] & ~g["mesh"]  # noqa: E731
        for name, y in (("current labels", now), ("labels without mesh", clean)):
            print(f" {name}")
            show("rule", a, lambda g: g["rule"], y)
            show("rule - mesh", a, lambda g: g["rule"] & ~g["mesh"], y)
            show("rule + bridge", a, lambda g: g["rule"] | g["bridged"], y)
            show("rule + bridge - mesh", a, lambda g: (g["rule"] | g["bridged"]) & ~g["mesh"], y)
        cp = lambda g: g["curve"] & g["pipe"] & ~g["mesh"]  # noqa: E731
        tot = sum(g["L"][cp(g)].sum() for g in a)
        print(" pages with most curve length on pipe layers left out: page, left out pt, of curves there, mean p, labelled true")
        rows = []
        for g in a:
            c, miss = cp(g), cp(g) & ~(g["rule"] | g["bridged"])
            rows.append((g["L"][miss].sum(), g["stem"][:9], g["L"][c].sum(), float(g["p"][miss].mean()) if miss.any() else 0, g["L"][miss & g["y"]].sum() / max(g["L"][miss].sum(), 1)))
        for r in sorted(rows, reverse=True)[:5]:
            print(f"  {r[1]} {r[0]:7.0f} of {r[2]:7.0f}  p {r[3]:.2f}  true {r[4]:.2f}")
        print(f" curves on pipe layers {tot / 1e3:.0f}k pt: labelled true {sum(g['L'][cp(g) & g['y']].sum() for g in a) / max(tot, 1):.3f}, in mask {sum(g['L'][cp(g) & g['rule']].sum() for g in a) / max(tot, 1):.3f}, with bridge {sum(g['L'][cp(g) & (g['rule'] | g['bridged'])].sum() for g in a) / max(tot, 1):.3f}; bridge adds {sum(g['L'][g['bridged'] & ~g['rule']].sum() for g in a) / 1e3:.1f}k pt, labelled true {sum(g['L'][g['bridged'] & ~g['rule'] & g['y']].sum() for g in a) / 1e3:.1f}k")


if __name__ == "__main__":
    main(sys.argv[1])
