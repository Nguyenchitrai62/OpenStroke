"""Dashed runs on one page of the app's cache, step by step: drawn in the payload, kept for the model, scored by it, chosen by post-processing. Dashes are found from geometry alone, labels only report.

    py -3.12 research/probes/dash_trace_probe.py runs/app_cache/<sha16> <result tag> runs_colab/viz/dash [labels.npz]
"""

import base64
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.geometry import page_segments  # noqa: E402

MAX_DASH, MAX_GAP, MAX_TURN, MIN_RUN, MIN_LEN = 30.0, 15.0, 5.0, 4, 60.0  # pt, pt, degrees, dashes, pt


def dash_runs(seg, off, group):
    """Run id per shape, -1 outside runs: one-segment short shapes of one style and direction whose ends meet across a gap, in line."""

    n = len(off) - 1
    one = np.flatnonzero(np.diff(off) == 1)
    s = seg[off[one]]
    d = s[:, 2:] - s[:, :2]
    ln = np.hypot(*d.T)
    ok = (ln > 0.5) & (ln <= MAX_DASH)
    idx, s, d, ln = one[ok], s[ok], d[ok], ln[ok]
    ang = np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi)
    key = np.c_[group[idx], np.round(ang / np.radians(MAX_TURN)).astype(np.int64)]
    _, cell = np.unique(key, axis=0, return_inverse=True)
    cell = cell.ravel()
    rows, cols = [], []
    for c in np.flatnonzero(np.bincount(cell) >= MIN_RUN):
        i = np.flatnonzero(cell == c)
        ends = np.r_[s[i, :2], s[i, 2:]]
        own = np.r_[i, i]
        pairs = cKDTree(ends).query_pairs(MAX_GAP, output_type="ndarray")
        if not len(pairs):
            continue
        link = ends[pairs[:, 0]] - ends[pairs[:, 1]]
        gap = np.hypot(*link.T)
        a = own[pairs[:, 0]]
        across = np.degrees(np.abs(np.angle(np.exp(2j * (np.arctan2(link[:, 1], link[:, 0]) - ang[a]))))) / 2
        good = (a != own[pairs[:, 1]]) & (gap > 0.3) & (across <= MAX_TURN)
        rows.append(a[good])
        cols.append(own[pairs[:, 1]][good])
    a, b = np.concatenate(rows), np.concatenate(cols)
    label = connected_components(coo_matrix((np.ones(len(a), bool), (a, b)), shape=(len(s), len(s))), directed=False)[1]
    size, total = np.bincount(label), np.bincount(label, ln)
    out = np.full(n, -1, np.int64)
    good = (size[label] >= MIN_RUN) & (total[label] >= MIN_LEN)
    out[idx[good]] = label[good]
    return out


def main(cache, tag, out, labels=None):
    cache, out = Path(cache), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    payload = json.load(gzip.open(next(cache.glob("page-*.json.gz")), "rt", encoding="utf-8"))
    shapes = payload["shapes"]
    res = json.loads((cache / f"{tag}.result.json").read_text(encoding="utf-8"))
    print("job", {k: v for k, v in json.loads((cache / f"{tag}.job.json").read_text(encoding="utf-8")).items() if k in ("model", "mode", "tta")}, "| post", res["post"].get("kind"), res["post"].get("params"))
    segs = page_segments(shapes)
    off, seg = segs.shape_offsets, segs.seg
    L = np.bincount(segs.owner, segs.length, len(shapes))
    prob = np.frombuffer(base64.b64decode(res["prob"]), np.uint8) / 255
    final = np.frombuffer(base64.b64decode(res["final"]), np.uint8).astype(bool)
    names = np.array([str(s.get("layer") or "") for s in shapes])
    print(f"shapes in payload {len(shapes)}, with geometry {int((np.diff(off) > 0).sum())}, handed to the model {res['kept']}; with a PDF dash attribute {sum(str(s.get('dashes')) != '[] 0' for s in shapes)}")
    rgb = np.array([tuple(x.get("color") or (0, 0, 0))[:3] for x in shapes], float).round(2)
    style = np.unique(np.c_[np.unique(names, return_inverse=True)[1], rgb, np.round([float(x.get("width") or 0) for x in shapes], 2)], axis=0, return_inverse=True)[1].ravel()
    run = dash_runs(seg, off, style)
    m = run >= 0
    print(f"dashes found from geometry: {int(m.sum())} shapes, {L[m].sum():.0f} pt, {np.unique(run[m]).size} runs")
    print(f"  of those length: scored p>=0.5 {L[m & (prob >= 0.5)].sum() / L[m].sum():.3f}, p>=0.15 {L[m & (prob >= 0.15)].sum() / L[m].sum():.3f}, mean p {np.average(prob[m], weights=L[m]):.3f}; in final {L[m & final].sum() / L[m].sum():.3f}")
    y = None
    if labels:
        z = np.load(labels)
        y = z["label"].astype(bool)
        assert len(y) == len(shapes)
        print(f"  labels: dashes that are true {L[m & y].sum():.0f} pt of {L[y].sum():.0f} pt true; of dashes true: p>=0.5 {L[m & y & (prob >= 0.5)].sum() / max(L[m & y].sum(), 1):.3f}, in final {L[m & y & final].sum() / max(L[m & y].sum(), 1):.3f}")
        print(f"  all true: in final {L[y & final].sum() / L[y].sum():.3f}; true that is not a dash: in final {L[y & ~m & final].sum() / max(L[y & ~m].sum(), 1):.3f}")
    c = Counter()
    for k, v in zip(names[m], L[m]):
        c[k] += float(v)
    print("  by layer: name, dash pt, mean p, in final, label true")
    for k, v in c.most_common(10):
        q = m & (names == k)
        print(f"    {k[:44]:44s} {v:8.0f} {np.average(prob[q], weights=L[q]):.2f} {L[q & final].sum() / v:.2f} {'' if y is None else round(float(L[q & y].sum() / v), 2)}")
    w, h = res["w"], res["h"]
    fig, ax = plt.subplots(figsize=(w / h * 13, 13), dpi=110)
    own = segs.owner
    for mask, col, lw in ((~m[own] & ~final[own], "#dddddd", 0.25), (~m[own] & final[own], "#1a9a1a", 0.6), (m[own] & final[own], "#2060ff", 1.2), (m[own] & ~final[own], "#e01010", 1.2)):
        ax.add_collection(LineCollection(seg[mask].reshape(-1, 2, 2), colors=col, linewidths=lw))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("green final, blue dashes kept, red dashes lost, grey other")
    fig.tight_layout()
    fig.savefig(out / f"{cache.name[:8]}_{tag[:8]}.png")
    print(out / f"{cache.name[:8]}_{tag[:8]}.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
