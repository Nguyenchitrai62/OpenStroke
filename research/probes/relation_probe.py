"""Which relations between shapes carry the label? (evidence for OpenStroke3, no model involved)

1. Endpoint graph: two shapes are linked when an endpoint of a segment of one lies within ``TOL`` pt of an endpoint of a
   segment of the other (pipes are drawn as chains; arch lines also meet at corners). Purity = share of links whose two
   shapes have the same label; and for true shapes, the share of their links that go to true shapes.
2. ``category`` of the payload (pdf_etl ``linkConsecutiveSeqnos``: runs of consecutive seqno = one CAD object exploded):
   share of length in groups that are >= 90 % one label, group sizes.
Both are compared with the label prior (what a random link / group would give).

    py -3.12 research/probes/relation_probe.py datasets/e2 > research/probes/relation_probe.log
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.sources import load_payload  # noqa: E402

TOL = 0.5


def endpoint_links(seg, owner, hub=32):
    """Shape pairs sharing an endpoint cell of ``TOL`` pt (two grids offset by half a cell, so pairs across a cell
    border are not missed). Cells touched by more than ``hub`` shapes (hatch, dense text) are skipped: they would
    give k^2 pairs and say nothing about chains."""

    pts = np.concatenate([seg[:, :2], seg[:, 2:]])
    own = np.concatenate([owner, owner]).astype(np.int64)
    out = []
    for shift in (0.0, TOL / 2):
        q = np.floor((pts + shift) / TOL).astype(np.int64)
        cell = (q[:, 0] + 1_000_000) * 4_000_000 + (q[:, 1] + 1_000_000)
        co = np.unique(np.stack([cell, own], 1), axis=0)  # one row per (cell, shape)
        c, start, cnt = np.unique(co[:, 0], return_index=True, return_counts=True)
        sel = (cnt >= 2) & (cnt <= hub)
        for s0, k in zip(start[sel], cnt[sel]):
            o = co[s0:s0 + k, 1]
            i, j = np.triu_indices(k, 1)
            out.append(np.stack([o[i], o[j]], 1))
    if not out:
        return np.zeros((0, 2), np.int64)
    e = np.concatenate(out)
    return np.unique(np.sort(e, axis=1), axis=0)


def main(data):
    data = Path(data)
    recs = [r for r in json.loads((data / "manifest.json").read_text(encoding="utf-8"))["records"] if r["status"] == "ok"]
    acc = defaultdict(lambda: defaultdict(float))
    for i, r in enumerate(recs):
        if i % 20 == 0:
            print(f"  ... page {i}/{len(recs)}", file=sys.stderr, flush=True)
        z = np.load(data / r["npz"])
        y = z["label"].astype(bool)
        L = z["length"].astype(np.float64)
        off = z["shape_offsets"]
        owner = np.repeat(np.arange(len(y)), np.diff(off))
        e = endpoint_links(z["segments"].astype(np.float64), owner)
        A = acc[r["mode"]]
        if len(e):
            ya, yb = y[e[:, 0]], y[e[:, 1]]
            A["links"] += len(e)
            A["same"] += float((ya == yb).sum())
            A["tt"] += float((ya & yb).sum())
            A["t_any"] += float((ya | yb).sum())
            A["t_links"] += float(ya.sum() + yb.sum())  # link ends at true shapes
            A["t_to_t"] += float(2 * (ya & yb).sum())
            deg = np.bincount(e.ravel(), minlength=len(y))
            A["true_len_isolated"] += L[y & (deg == 0)].sum()
            A["true_len"] += L[y].sum()
            pi = L[y].sum() / max(L.sum(), 1e-9)
            n_true = y.mean()
            A["prior_same"] += len(e) * (n_true ** 2 + (1 - n_true) ** 2)
        cat = np.array([s.get("category", -1) if s.get("category") is not None else -1
                        for s in load_payload(r["payload"]["path"])["shapes"]])
        if len(cat) == len(y):
            ok = cat >= 0
            ids, inv = np.unique(cat[ok], return_inverse=True)
            tl = np.bincount(inv, weights=(L * y)[ok], minlength=len(ids))
            al = np.bincount(inv, weights=L[ok], minlength=len(ids))
            cnt = np.bincount(inv, minlength=len(ids))
            frac = tl / np.maximum(al, 1e-9)
            pure = (frac >= 0.9) | (frac <= 0.1)
            A["cat_len"] += al.sum()
            A["cat_pure_len"] += al[pure].sum()
            A["cat_true_len"] += tl.sum()
            A["cat_true_in_pure"] += tl[frac >= 0.9].sum()
            A["cat_groups"] += len(ids)
            A["cat_shapes"] += ok.sum()
            A["cat_multi"] += (cnt > 1).sum()
            A["nocat_len"] += L[~ok].sum()
    for mode, A in sorted(acc.items()):
        print(f"=== {mode}")
        print(f"  endpoint links: {A['links']:.0f}; same label {A['same'] / A['links']:.1%} (prior by count {A['prior_same'] / A['links']:.1%})")
        print(f"  link ends at a true shape that lead to a true shape: {A['t_to_t'] / max(A['t_links'], 1):.1%}")
        print(f"  true length with no endpoint link (isolated): {A['true_len_isolated'] / max(A['true_len'], 1):.1%}")
        if A["cat_len"]:
            print(f"  category groups: {A['cat_groups']:.0f} for {A['cat_shapes']:.0f} shapes ({A['cat_multi']:.0f} with > 1 shape); "
                  f"length in >= 90 % pure groups {A['cat_pure_len'] / A['cat_len']:.1%}; true length in >= 90 % true groups "
                  f"{A['cat_true_in_pure'] / max(A['cat_true_len'], 1):.1%}; length without category {A['nocat_len'] / (A['nocat_len'] + A['cat_len']):.1%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "datasets/e2")
