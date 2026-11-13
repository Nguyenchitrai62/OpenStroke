"""OpenStroke28 (F142): a stroke model's val probabilities through AI_Takeoff's HVAC logic, every way of handing the strokes over.
0 GPU, no model: reads <run>/fold0/val_scores.npz (written by `openstroke train`), the payloads of the split's val pages and, for GT
pages, the text the AI_Takeoff run stored. Arms (cached in runs_colab/logic_e8/<stem>/<tag>_<arm>.json):
  real<t>   strokes with p >= t on their own PDF layer (the user's design: the model replaces the layer selector)
  remap<t>  as real<t>, then a selected stroke on a layer holding < 20 % of the selected length takes the layer of the nearest stroke on a
            major selected layer (strokes drawn on a wrong layer rejoin their object)
  one<t>    strokes with p >= t on one layer (what a PDF without layers allows)
  style<t>  strokes with p >= t on layer_1: one layer per (colour, width), no PDF layer
  layers<a> / union<a>  the model as a layer selector (E37 arms: layers whose length-weighted mean p >= a, + strokes p >= 0.05 elsewhere)
  remapf<t> as real<t>, but only strokes on a FOREIGN layer (selected strokes < 5 % of that layer's length) move (F144)
--scatter=P --pack=<score_pack folder>: every stroke of a fraction P of the GT objects is first moved to one wrong layer (arms <tag>_osP_*)
Score: py -3.12 research/probes/logic_object_probe.py <score_pack folder> runs_colab/logic_e8 <tag>_real0.5 ...

    PYTHONDONTWRITEBYTECODE=1 <AI_Takeoff python> research/probes/logic_mask_probe.py <run dir> <obb split dir> <tag> [t ...]
"""
import gzip
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from logic_hvac_probe import page_text, run_arm  # noqa: E402

CACHE = Path("runs_colab/logic_e8")
MINOR, AGG, UNION_T, MASK = 0.2, (0.1,), 0.05, "__MASK__"
SCATTER = next((float(a.split('=', 1)[1]) for a in sys.argv if a.startswith('--scatter=')), 0.0)   # F144
PACK = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--pack=')), None)


def centers(shapes):
    out = np.zeros((len(shapes), 2))
    for i, s in enumerate(shapes):
        r = s.get("rect") or [0, 0, 0, 0]
        out[i] = ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
    return out


def style_layer(s):
    """layer_1 (F145): a layer named after the stroke colour and width only, what any PDF has without layers."""
    c = s.get("color") or [0, 0, 0]
    return "S|" + ",".join(f"{float(v):.2f}" for v in list(c)[:3]) + f"|w{float(s.get('width') or 0):.2f}"


def remap(sel, layer, length, ctr):
    use = layer.copy()
    idx = np.flatnonzero(sel)
    share = Counter()
    for i in idx:
        share[use[i]] += length[i]
    tot = sum(share.values()) or 1.0
    major = {n for n, v in share.items() if v >= MINOR * tot}
    keep = np.array([i for i in idx if use[i] in major])
    if len(keep):
        for i in idx:
            if use[i] not in major:
                use[i] = use[keep[np.argmin(((ctr[keep] - ctr[i]) ** 2).sum(1))]]
    return use


def remap_foreign(sel, layer, length, ctr, frac=0.05):
    """F144: only strokes on a FOREIGN layer move: a layer where the selected strokes are < frac of the layer's own length (an
    architectural / annotation layer the drafter put a few duct lines on). They take the layer of the nearest selected stroke on a
    non-foreign layer. HVAC layers (mostly selected), small or large, keep their strokes."""
    use = layer.copy()
    idx = np.flatnonzero(sel)
    tot, chosen = Counter(), Counter()
    for i in range(len(layer)):
        tot[layer[i]] += length[i]
    for i in idx:
        chosen[layer[i]] += length[i]
    foreign = {n for n, v in chosen.items() if v < frac * max(tot[n], 1e-9)}
    keep = np.array([i for i in idx if layer[i] not in foreign])
    if len(keep):
        for i in idx:
            if layer[i] in foreign:
                use[i] = layer[keep[np.argmin(((ctr[keep] - ctr[i]) ** 2).sum(1))]]
    return use


def scatter(stems_pack, stem, lab_layer, shapes, p, seed=0):
    """F144 robustness: every stroke of a fraction p of the GT objects is moved to one wrong layer (the page's non-HVAC layer with the
    most strokes, else '0'); returns the new layer array. The model's scores do not change: it does not read layers (no_layers)."""
    from openstroke.objects import assign_objects
    from openstroke.relabel import page_segments_of
    with np.load(Path(stems_pack) / "npz" / f"{stem}.npz") as z:
        segs = page_segments_of({k: z[k] for k in ("segments", "shape_offsets")})
        lab = z["label"].astype(bool)
    gobj = json.loads((Path(stems_pack) / "gt" / f"{stem}.json").read_text(encoding="utf-8"))
    inst = assign_objects(segs, gobj, lab)[1]
    rng = np.random.default_rng(seed)
    ids = np.unique(inst[inst >= 0])
    moved = np.flatnonzero(np.isin(inst, ids[rng.random(len(ids)) < p]))
    hv = set(lab_layer[lab])
    others = Counter(lab_layer[~lab])
    wrong = next((n for n, _ in others.most_common() if n and n not in hv), "0")
    out = lab_layer.copy()
    out[moved] = wrong
    return out, len(moved)


def one(run, row, repo, tag, ts):
    stem = row["stem"]
    out_dir = CACHE / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(repo / row["payload"].replace("\\", "/"), "rt", encoding="utf-8") as fh:
        pay = json.load(fh)
    shapes = pay["shapes"]
    meta = pay.get("document_metadata") or {}
    area = float(meta.get("page_width") or 0) * float(meta.get("page_height") or 0) or None
    text = page_text(int(stem[3:9]), int(stem[11:15])) if stem.startswith("doc") else {}
    sc = np.load(Path(run) / "fold0" / "val_scores.npz")
    p = np.zeros(len(shapes))
    p[sc[f"{stem}__index"]] = sc[stem]
    layer = np.array([str(s.get("layer") or "") for s in shapes], dtype=object)
    if SCATTER and PACK and (Path(PACK) / "npz" / f"{stem}.npz").exists():
        layer, _ = scatter(PACK, stem, layer, shapes, SCATTER)
        tag = f"{tag}_os{SCATTER:g}"
    from openstroke.geometry import page_segments
    segs = page_segments(shapes)
    length = np.bincount(segs.owner, weights=segs.length, minlength=len(shapes))
    ctr = centers(shapes)
    style = np.array([style_layer(s) for s in shapes], dtype=object)
    done = {}

    def go(name, lay_of, sel):
        d = [dict(s) for s in shapes]
        names = set()
        for i, s in enumerate(d):
            if sel[i]:
                s["layer"] = lay_of[i]
                names.add(lay_of[i])
            else:
                s["layer"] = "__OFF__"
        res = run_arm(out_dir / f"{tag}_{name}.json", d, sorted(n for n in names if n), text, area)
        done[name] = len(res["objects"])

    for t in ts:
        sel = p >= t
        go(f"real{t:g}", layer, sel)
        go(f"remap{t:g}", remap(sel, layer, length, ctr), sel)
        go(f"remapf{t:g}", remap_foreign(sel, layer, length, ctr), sel)
        go(f"one{t:g}", np.full(len(shapes), MASK, dtype=object), sel)
        go(f"style{t:g}", style, sel)   # layer_1: one layer per (stroke colour, width), no PDF layer (user 2026-11-13)
    uniq, inv = np.unique(layer.astype(str), return_inverse=True)
    agg = np.bincount(inv, p * length, len(uniq)) / np.maximum(np.bincount(inv, length, len(uniq)), 1e-9)
    for a in AGG:
        chosen = agg[inv] >= a
        lay = np.where(chosen, layer, MASK)
        go(f"layers{a:g}", layer, chosen)
        go(f"union{a:g}", lay, chosen | (p >= UNION_T))
    return stem, done


def main() -> int:
    run, split, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    ts = [float(x) for x in sys.argv[4:] if not x.startswith("--")]
    if not ts:
        r = json.loads((run / "fold0" / "result.json").read_text(encoding="utf-8"))
        ts = sorted({round(float(r.get("threshold") or 0.5), 2), 0.5})
    repo = split.resolve().parents[1]
    sc = np.load(run / "fold0" / "val_scores.npz")
    rows = [r for r in json.loads((split / "pages.json").read_text(encoding="utf-8")) if r["split"] == "val" and r["stem"] in sc.files]
    print(f"{len(rows)} val pages, thresholds {ts}", flush=True)
    with ProcessPoolExecutor(int(__import__("os").environ.get("LOGIC_WORKERS", "3"))) as ex:
        for f in as_completed([ex.submit(one, run, r, repo, tag, ts) for r in rows]):
            print(*f.result(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
