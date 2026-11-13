"""OpenStroke28 (F142): a stroke model's val probabilities through AI_Takeoff's HVAC logic, every way of handing the strokes over.
0 GPU, no model: reads <run>/fold0/val_scores.npz (written by `openstroke train`), the payloads of the split's val pages and, for GT
pages, the text the AI_Takeoff run stored. Arms (cached in runs_colab/logic_e8/<stem>/<tag>_<arm>.json):
  real<t>   strokes with p >= t on their own PDF layer (the user's design: the model replaces the layer selector)
  remap<t>  as real<t>, then a selected stroke on a layer holding < 20 % of the selected length takes the layer of the nearest stroke on a
            major selected layer (strokes drawn on a wrong layer rejoin their object)
  one<t>    strokes with p >= t on one layer (what a PDF without layers allows)
  layers<a> / union<a>  the model as a layer selector (E37 arms: layers whose length-weighted mean p >= a, + strokes p >= 0.05 elsewhere)
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


def centers(shapes):
    out = np.zeros((len(shapes), 2))
    for i, s in enumerate(shapes):
        r = s.get("rect") or [0, 0, 0, 0]
        out[i] = ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
    return out


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
    from openstroke.geometry import page_segments
    segs = page_segments(shapes)
    length = np.bincount(segs.owner, weights=segs.length, minlength=len(shapes))
    ctr = centers(shapes)
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
        go(f"one{t:g}", np.full(len(shapes), MASK, dtype=object), sel)
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
    ts = [float(x) for x in sys.argv[4:]]
    if not ts:
        r = json.loads((run / "fold0" / "result.json").read_text(encoding="utf-8"))
        ts = sorted({round(float(r.get("threshold") or 0.5), 2), 0.5})
    repo = split.resolve().parents[1]
    sc = np.load(run / "fold0" / "val_scores.npz")
    rows = [r for r in json.loads((split / "pages.json").read_text(encoding="utf-8")) if r["split"] == "val" and r["stem"] in sc.files]
    print(f"{len(rows)} val pages, thresholds {ts}", flush=True)
    with ProcessPoolExecutor(3) as ex:
        for f in as_completed([ex.submit(one, run, r, repo, tag, ts) for r in rows]):
            print(*f.result(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
