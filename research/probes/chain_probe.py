"""E51: post-processing so a pipe is not left in pieces and as little as possible is missed (user 2026-11-02: "đừng nham nhở").

Saved val scores only: no model, no logic run. Base mask: FIRE strokes along p >= t, HVAC the logic's objects. Rules on top: the strokes the model is sure of
(the logic dropped whole ducts the model scored 0.999, doc191), a whole layer + style group whose mean p is high, and for FIRE a lower threshold on pages where
the model hesitates. Parameters are picked on the tune run.

Patchiness: ``frag`` = pieces >= 5 pt of the mask / of the true strokes, both under the same-style graph (1 = as connected as GT);
``run90`` = share of true length in runs (>= 50 pt) that are at least 90 % covered. Also shown: labels v3 (F51) with the mask kept to the layers the model chose.

    py -3.12 research/probes/chain_probe.py e30 e37
"""

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.geometry import PageSegments  # noqa: E402
from openstroke.metrics import length_counts, prf  # noqa: E402
from openstroke.postprocess import HVAC_LAYER_MEAN, along_confident, objects_mask  # noqa: E402

DATA = Path("datasets/e3")
CACHE = Path("runs_colab/chain_cache")
HVAC_OBJ = {"e30": Path("runs_colab/logic_hvac"), "e37": Path("runs_colab/logic_hvac2")}
SEEDS = (0.15, 0.25, 0.35, 0.45, 0.55)
HVAC_SEEDS = (0.55, 0.75, 0.9)
SURE = (0.5, 0.7, 0.8, 0.9, 0.95, 2.0)  # 2.0 = off
GROUP = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 2.0)
HESITATE = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6)  # 0.0 = off


def comps(n, e):
    return connected_components(coo_matrix((np.ones(len(e), bool), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)[1]


def prepare(job):
    run, k, s = job
    out = CACHE / run / f"{s}.npz"
    if out.exists():
        return s
    d = np.load(DATA / f"{s}.npz")
    mode = json.loads(str(d["meta"]))["mode"]
    z = np.load(f"runs_colab/{run}/fold{k}/val_scores.npz")
    p = np.zeros(len(d["label"]), np.float32)
    p[z[f"{s}__index"]] = z[s]
    off = d["shape_offsets"].astype(np.int64)
    segs = PageSegments(seg=d["segments"].astype(np.float64), owner=np.repeat(np.arange(len(off) - 1), np.diff(off)), shape_offsets=off)
    lay = d["layer_id"].astype(np.int64)
    base = {}
    if mode == "hvac":
        f = HVAC_OBJ[run] / s / f"{run}_union0.1.json"
        if not f.exists():
            return s
        base["obj"] = objects_mask("hvac", json.loads(f.read_text(encoding="utf-8"))["objects"], segs, lay)
    base.update({f"t{t}": along_confident(segs, p, t) for t in (SEEDS if mode == "fire" else HVAC_SEEDS)})
    e = np.load(DATA / f"{s}.graph.g3.npz")["edges"].astype(np.int64)
    rgb = np.nan_to_num(d["stroke_rgb"], nan=-1).round(2)
    style = np.unique(np.c_[rgb, np.round(d["width"], 2), d["type_code"]], axis=0, return_inverse=True)[1].ravel()
    same = style[e[:, 0]] == style[e[:, 1]]
    group = np.unique(np.c_[lay, style], axis=0, return_inverse=True)[1].ravel()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, mode=mode, p=p, y=d["label"].astype(bool), L=d["length"].astype(np.float64), lay=lay, run=comps(len(p), e[same]), group=group, edges=e[same], **base)
    return s


def load(run):
    res = []
    for k in range(5):
        for pg in json.load(open(f"runs_colab/{run}/fold{k}/result.json"))["pages"]:
            f = CACHE / run / f"{pg['stem']}.npz"
            if f.exists():
                z = np.load(f)
                res.append({**{n: z[n] for n in z.files}, "stem": pg["stem"], "mode": str(z["mode"])})
    return res


def by_group(g, a):
    n = g["group"].max() + 1
    mean = np.bincount(g["group"], g["p"] * g["L"], n) / np.maximum(np.bincount(g["group"], g["L"], n), 1e-9)
    return (mean >= a)[g["group"]]


def fire_mask(g, t, low, c, a):
    """Strokes along p >= t, along p >= low on a page where the model hesitates, plus sure groups."""

    L, p = g["L"], g["p"]
    hesitant = L[p >= 0.55].sum() < c * L[p >= 0.25].sum()
    return g[f"t{low if hesitant else t}"] | by_group(g, a)


def hvac_mask(g, t, a):
    return g["obj"] | (g["p"] >= t) | by_group(g, a)


def labels_v3(g):
    lay, L, y = g["lay"], g["L"], g["y"]
    n = lay.max() + 1
    if n < 2:
        return y
    return y & ((np.bincount(lay[y], L[y], n) / np.maximum(np.bincount(lay, L, n), 1e-9)) >= 0.1)[lay]


def chosen(g):
    lay, L, p = g["lay"], g["L"], g["p"]
    n = lay.max() + 1
    return np.ones(len(p), bool) if n < 2 else ((np.bincount(lay, p * L, n) / np.maximum(np.bincount(lay, L, n), 1e-9)) >= HVAC_LAYER_MEAN)[lay]


def pieces(g, m):
    e = g["edges"]
    c = comps(len(m), e[m[e[:, 0]] & m[e[:, 1]]])
    return int((np.bincount(c[m], g["L"][m]) >= 5).sum())


def patch(g, m):
    """(pieces of mask, pieces of true, true length in runs, of it in runs covered >= 90 %)."""

    r, L, y = g["run"], g["L"], g["y"]
    n = r.max() + 1
    t = np.bincount(r[y], L[y], n)
    c = np.bincount(r[y & m], L[y & m], n)
    big = t >= 50
    return np.array([pieces(g, m), pieces(g, y), t[big].sum(), t[big & (c >= 0.9 * t)].sum()], float)


def f2(ps, f):
    return prf(*np.sum([length_counts(f(g), g["y"], g["L"]) for g in ps], 0))["f2"]


def show(name, ps, f):
    m = prf(*np.sum([length_counts(f(g), g["y"], g["L"]) for g in ps], 0))
    v = prf(*np.sum([length_counts(f(g) & chosen(g), labels_v3(g), g["L"]) for g in ps], 0))
    pa = np.sum([patch(g, f(g)) for g in ps], 0)
    print(f"  {name:22s} v2 F2 {m['f2']:.3f} R {m['recall']:.3f} P {m['precision']:.3f} | frag {pa[0] / pa[1]:.2f} run90 {pa[3] / max(pa[2], 1):.3f} | v3+layers F2 {v['f2']:.3f} R {v['recall']:.3f} P {v['precision']:.3f}", flush=True)


def per_page(ps, f0, f1):
    rows = []
    for g in ps:
        a, b = prf(*length_counts(f0(g), g["y"], g["L"])), prf(*length_counts(f1(g), g["y"], g["L"]))
        rows.append((b["recall"], g["stem"][:9], a["recall"], a["precision"], a["f2"], b["precision"], b["f2"]))
    for r in sorted(rows):
        print(f"    {r[1]} base R {r[2]:.3f} P {r[3]:.3f} F2 {r[4]:.3f} -> R {r[0]:.3f} P {r[5]:.3f} F2 {r[6]:.3f}")


def main(tune, test):
    jobs = [(r, k, pg["stem"]) for r in (tune, test) for k in range(5) for pg in json.load(open(f"runs_colab/{r}/fold{k}/result.json"))["pages"]]
    with mp.Pool(6) as pool:
        for i, s in enumerate(pool.imap_unordered(prepare, jobs)):
            print(i + 1, "/", len(jobs), s[:21], flush=True)
    tu, te = load(tune), load(test)
    fa, fb = [g for g in tu if g["mode"] == "fire"], [g for g in te if g["mode"] == "fire"]
    ha, hb = [g for g in tu if g["mode"] == "hvac"], [g for g in te if g["mode"] == "hvac"]
    print(f"\nfire: tune {tune} {len(fa)} pages, test {test} {len(fb)} pages")
    grid = [(t, low, c, a) for t in (0.45, 0.55) for low in SEEDS for c in HESITATE for a in GROUP if low <= t]
    q = max(grid, key=lambda x: f2(fa, lambda g: fire_mask(g, *x)))
    print(f"  picked on {tune}: t {q[0]} low {q[1]} hesitate < {q[2]} group >= {q[3]}")
    base, rule = (lambda g: g["t0.55"]), (lambda g: fire_mask(g, *q))
    show(f"{tune} base", fa, base)
    show(f"{tune} rule", fa, rule)
    show(f"{test} base", fb, base)
    show(f"{test} rule", fb, rule)
    per_page(fb, base, rule)
    print(f"\nhvac: tune {tune} {len(ha)} pages, test {test} {len(hb)} pages")
    grid = [(t, a) for t in SURE for a in GROUP]
    q = max(grid, key=lambda x: f2(ha, lambda g: hvac_mask(g, *x)))
    print(f"  picked on {tune}: p >= {q[0]} group >= {q[1]}")
    base, rule = (lambda g: g["obj"]), (lambda g: hvac_mask(g, *q))
    show(f"{tune} base", ha, base)
    show(f"{tune} rule", ha, rule)
    show(f"{test} base", hb, base)
    show(f"{test} rule", hb, rule)
    for t in HVAC_SEEDS:
        show(f"{test} obj + along {t}", hb, lambda g, t=t: g["obj"] | g[f"t{t}"])
    per_page(hb, base, rule)


if __name__ == "__main__":
    main(*sys.argv[1:3])
