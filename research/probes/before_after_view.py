"""Picture of post-processing on a run's val pages: the window with the most missed length, before and after. Green right, red missed, blue extra.
Needs the cache of ``chain_probe.py``. Saved scores only.

    py -3.12 research/probes/before_after_view.py e37 runs_colab/viz doc000191 doc000154 doc000195
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.postprocess import by_group, load_params, style_groups  # noqa: E402

WINDOW = 800


def main(run, out, *docs):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    params, src = load_params(f"runs_colab/{run}/fold0/best.pt")
    print(src, params)
    stems = [pg["stem"] for k in range(5) for pg in json.load(open(f"runs_colab/{run}/fold{k}/result.json"))["pages"]]
    for s in stems:
        if s[:9] not in docs:
            continue
        g = np.load(f"runs_colab/chain_cache/{run}/{s}.npz")
        d = np.load(f"datasets/e3/{s}.npz")
        mode = str(g["mode"])
        p, L, y = g["p"], g["L"], g["y"]
        group = style_groups({k: d[k] for k in ("layer_id", "stroke_rgb", "width", "type_code")})
        par = params[mode]
        if mode == "hvac":
            base = g["obj"]
            rule = base | (p >= par["sure"]) | by_group(p, L, group, par["group"])
        else:
            base = g["t0.55"]
            key = f"t{par['seed']}"
            rule = (g[key] if key in g.files else base) | by_group(p, L, group, par["group"])
        off = d["shape_offsets"].astype(np.int64)
        seg = d["segments"].astype(np.float64)
        own = np.repeat(np.arange(len(off) - 1), np.diff(off))
        mid = (seg[:, :2] + seg[:, 2:]) / 2
        ln = np.hypot(*(seg[:, 2:] - seg[:, :2]).T)
        c = (y & ~base)[own]
        q = np.floor(mid / WINDOW).astype(int)
        ids = q[:, 0] * 100000 + q[:, 1]
        u, inv = np.unique(ids[c], return_inverse=True)
        b = u[np.bincount(inv, ln[c]).argmax()]
        x0, y0 = (b // 100000) * WINDOW, (b % 100000) * WINDOW
        ins = (mid[:, 0] >= x0) & (mid[:, 0] < x0 + WINDOW) & (mid[:, 1] >= y0) & (mid[:, 1] < y0 + WINDOW)
        fig, axes = plt.subplots(1, 2, figsize=(20, 10.5), dpi=90)
        for ax, m, title in ((axes[0], base, "before"), (axes[1], rule, "after")):
            cls = np.where((y & m)[own], 1, np.where((y & ~m)[own], 2, np.where((~y & m)[own], 3, 0)))
            for k, col, lw in ((0, "#cccccc", 0.4), (1, "#2a9d2a", 1.0), (3, "#2060ff", 1.0), (2, "#e01010", 1.3)):
                i = ins & (cls == k)
                ax.add_collection(LineCollection(seg[i].reshape(-1, 2, 2), colors=col, linewidths=lw))
            ax.set_xlim(x0, x0 + WINDOW)
            ax.set_ylim(y0 + WINDOW, y0)
            ax.set_aspect("equal")
            ax.axis("off")
            tp, fn, fp = L[y & m].sum(), L[y & ~m].sum(), L[~y & m].sum()
            ax.set_title(f"{title}: page R {tp / max(tp + fn, 1):.3f} P {tp / max(tp + fp, 1):.3f}")
        fig.suptitle(f"{s[:24]} {mode}")
        fig.tight_layout()
        fig.savefig(out / f"{s[:9]}.png")
        plt.close(fig)
        print("saved", s[:9], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
