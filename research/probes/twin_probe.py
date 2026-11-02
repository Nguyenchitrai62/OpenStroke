"""E40: does a stroke have a parallel twin among the model's own confident strokes? (no model is run, saved val scores only)

Fire pipes are drawn as single lines and ducts as two parallel walls, so a parallel twin should mean opposite things in the two modes.
For the straight segments of the shapes with p >= t: sample a point every STEP pt, look for a sample of another shape within R pt
with the same direction (|cos| >= COS) that lies across (perpendicular offset in [D_MIN, R], offset along the stroke <= STEP). A shape's
twin fraction is the length-weighted share of its samples that have one.

    py -3.12 research/probes/twin_probe.py e30
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from postproc_oracle_probe import load_run  # noqa: E402
from openstroke.metrics import length_counts, prf  # noqa: E402

STEP = 10.0
COS = 0.995
D_MIN = 2.5  # a closer "twin" is the same stroke drawn twice (the 2 pt of the label rule)
R = 120.0
K = 48
GRID = np.round(np.arange(0.02, 0.96, 0.01), 2)


def twin_fraction(pg, off, seg, keep, step=STEP):
    """-> (fraction per shape, NaN when the shape has no sample, i.e. no straight segment long enough to judge)."""

    n = len(keep)
    owner = np.repeat(np.arange(n), np.diff(off))
    d = seg[:, 2:] - seg[:, :2]
    sl = np.hypot(d[:, 0], d[:, 1])
    m = keep[owner] & (sl >= step)
    frac = np.full(n, np.nan)
    if not m.any():
        return frac
    s0, u, sl_m, own = seg[m, :2], d[m] / sl[m, None], sl[m], owner[m]
    cnt = np.floor(sl_m / step).astype(np.int64) + 1
    idx = np.repeat(np.arange(len(sl_m)), cnt)
    off_t = np.concatenate([(np.arange(c) + 0.5) * (sl_m[i] / c) for i, c in enumerate(cnt)])
    pts, dirs, who = s0[idx] + u[idx] * off_t[:, None], u[idx], own[idx]
    dist, nb = cKDTree(pts).query(pts, k=min(K, len(pts)), distance_upper_bound=R, workers=1)
    if dist.ndim == 1:
        dist, nb = dist[:, None], nb[:, None]
    ok = np.isfinite(dist)
    nb_safe = np.where(ok, nb, 0)
    other = ok & (who[nb_safe] != who[:, None])
    parallel = np.abs((dirs[nb_safe] * dirs[:, None]).sum(-1)) >= COS
    rel = pts[nb_safe] - pts[:, None]
    along = np.abs((rel * dirs[:, None]).sum(-1))
    perp = np.abs(rel[..., 0] * dirs[:, None, 1] - rel[..., 1] * dirs[:, None, 0])
    has = (other & parallel & (along <= step) & (perp >= D_MIN) & (perp <= R)).any(1)
    tot = np.bincount(who, minlength=n).astype(np.float64)
    hit = np.bincount(who, has, minlength=n)
    frac[tot > 0] = hit[tot > 0] / tot[tot > 0]
    return frac


def page_geometry(stem):
    a = np.load(f"datasets/e3/{stem}.npz")
    return a["shape_offsets"].astype(np.int64), a["segments"].astype(np.float64)


def main(run):
    pages = load_run(run)
    bins = [("no long segment", None), ("0", (0, 1e-9)), ("(0,0.2)", (1e-9, 0.2)), ("[0.2,0.5)", (0.2, 0.5)), ("[0.5,0.8)", (0.5, 0.8)), ("[0.8,1]", (0.8, 1.01))]
    for mode in ("fire", "hvac"):
        pm = [pg for pg in pages if pg["mode"] == mode]
        t0 = max(GRID, key=lambda t: prf(*np.sum([length_counts(pg["p"] >= t, pg["y"], pg["L"]) for pg in pm], 0))["f2"])
        tp = np.zeros(len(bins))
        fp = np.zeros(len(bins))
        for pg in pm:
            off, seg = page_geometry(pg["stem"])
            keep = pg["p"] >= t0
            f = twin_fraction(pg, off, seg, keep)
            for i, (_, b) in enumerate(bins):
                m = keep & (np.isnan(f) if b is None else (f >= b[0]) & (f < b[1]))
                tp[i] += pg["L"][m & pg["y"]].sum()
                fp[i] += pg["L"][m & ~pg["y"]].sum()
        print(f"{mode} t {t0:.2f}  TP {tp.sum():.0f}  FP {fp.sum():.0f}")
        for i, (name, _) in enumerate(bins):
            print(f"  twin {name:16s} TP {tp[i] / tp.sum():.3f}  FP {fp[i] / fp.sum():.3f}  purity {tp[i] / max(tp[i] + fp[i], 1):.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
