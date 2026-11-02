"""MB TH4.pdf, user 2026-11-02: honeycomb noise comes out as pipe. The page is unseen by every E37 fold (it has two GT versions, both left out of the folds). Scores saved by the test app,
no model run. Which groups are the extra strokes in, and could a rule on the group's shape tell them apart?

    py -3.12 research/probes/mbth4_probe.py runs/app_cache/a82fde3504662c38 runs_colab/viz
"""

import base64
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.metrics import length_counts, prf  # noqa: E402
from openstroke.geometry import PageSegments  # noqa: E402
from openstroke.postprocess import DEFAULT_PARAMS, by_group, mesh_strokes, style_groups  # noqa: E402

STEMS = ("doc000196_p0001_gt123", "doc000023_p0001_gt12")


def main(cache, out):
    cache, out = Path(cache), Path(out)
    best = None
    for f in sorted(cache.glob("*.result.json"), key=lambda x: x.stat().st_mtime):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("post", {}).get("objects") is not None:
            best = r
    assert best, "no HVAC result in this cache"
    prob = np.frombuffer(base64.b64decode(best["prob"]), np.uint8) / 255
    final = np.frombuffer(base64.b64decode(best["final"]), np.uint8) > 0
    print("saved result", best["post"].get("kind"), "shapes", best["n"], "objects", best["post"].get("objects"), "layers", best["post"].get("layers"))
    d = np.load(f"datasets/e3/{STEMS[0]}.npz")
    assert len(d["label"]) == len(prob), (len(d["label"]), len(prob))
    L = d["length"].astype(np.float64)
    names = [str(x) for x in d["layer_names"]]
    lay = d["layer_id"].astype(np.int64)
    group = style_groups({k: d[k] for k in ("layer_id", "stroke_rgb", "width", "type_code")})
    par = DEFAULT_PARAMS["hvac"]
    for s in STEMS:
        y = np.load(f"datasets/e3/{s}.npz")["label"].astype(bool)
        print(s, "true length", round(L[y].sum()), {k: round(v, 3) for k, v in prf(*length_counts(final, y, L)).items() if k in ("f2", "recall", "precision")})
    y = d["label"].astype(bool)
    rgb = np.nan_to_num(d["stroke_rgb"], nan=-1).round(2)
    w = np.round(np.nan_to_num(d["width"], nan=-1), 2)
    n = group.max() + 1
    tot = np.maximum(np.bincount(group, L, n), 1e-9)
    cnt = np.bincount(group, minlength=n)
    mean = np.bincount(group, prob * L, n) / tot
    fp = np.bincount(group[final & ~y], L[final & ~y], n)
    first = np.zeros(n, np.int64)
    first[group[::-1]] = np.arange(len(group))[::-1]
    print("extra length k | group length k | shapes | mean length | mean p | true % | in mask % | layer | rgb | width")
    for i in np.argsort(-fp)[:12]:
        j = first[i]
        print(f"{fp[i] / 1e3:7.1f} {tot[i] / 1e3:7.1f} {cnt[i]:7d} {tot[i] / max(cnt[i], 1):7.1f} {mean[i]:.3f} {L[(group == i) & y].sum() / tot[i] * 100:5.1f} {L[(group == i) & final].sum() / tot[i] * 100:5.1f} {names[lay[j]][:28]:28s} {rgb[j]} {w[j]}")
    print("p histogram of extra length:", np.round(np.histogram(prob[final & ~y], bins=[0, .1, .3, .5, .7, .8, .9, 1.01], weights=L[final & ~y])[0] / 1e3))
    print("p histogram of true length: ", np.round(np.histogram(prob[y], bins=[0, .1, .3, .5, .7, .8, .9, 1.01], weights=L[y])[0] / 1e3))
    off = d["shape_offsets"].astype(np.int64)
    seg = d["segments"].astype(np.float64)
    own = np.repeat(np.arange(len(off) - 1), np.diff(off))
    fig, ax = plt.subplots(figsize=(16, 11), dpi=95)
    cls = np.where((y & final)[own], 1, np.where((y & ~final)[own], 2, np.where((~y & final)[own], 3, 0)))
    for k, col, lw in ((0, "#d5d5d5", 0.3), (1, "#2a9d2a", 0.8), (3, "#2060ff", 0.8), (2, "#e01010", 1.0)):
        ax.add_collection(LineCollection(seg[cls == k].reshape(-1, 2, 2), colors=col, linewidths=lw))
    ax.autoscale()
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("MB TH4: green right, red missed, blue extra (GT version gt123)")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "mbth4.png")
    print("group rule alone would take:", round(L[by_group(prob, L, group, par['group'])].sum() / 1e3), "k pt")
    segs = PageSegments(seg=seg, owner=own, shape_offsets=off)
    mesh = mesh_strokes(segs, group, L)
    hc = np.array([names[i] == "CDC-ACMV-SE (EI)" for i in lay])
    print(f"mesh rule: {mesh.sum()} strokes {L[mesh].sum() / 1e3:.1f}k pt; of the honeycomb layer {L[hc & mesh].sum() / max(L[hc].sum(), 1):.3f}; mask length {L[final].sum() / 1e3:.1f}k -> {L[final & ~mesh].sum() / 1e3:.1f}k")
    for s, yy in ((STEMS[0], y), (STEMS[0] + " without mesh in labels", y & ~mesh)):
        print(s, "mask", {k: round(v, 3) for k, v in prf(*length_counts(final, yy, L)).items() if k in ("f2", "recall", "precision")}, "mask - mesh", {k: round(v, 3) for k, v in prf(*length_counts(final & ~mesh, yy, L)).items() if k in ("f2", "recall", "precision")})


if __name__ == "__main__":
    main(*sys.argv[1:3])
