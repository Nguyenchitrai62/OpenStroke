"""Pick a run's post-processing parameters on its own val scores (``fold<k>/val_scores.npz``, which every run saves): no model and no GPU needed.

Same logic for every model, its thresholds fitted to how that model scores. ``<run>/postprocess.json`` is what ``postprocess.load_params`` finds from a checkpoint.
HVAC needs the logic's objects on those pages (``research/probes/logic_hvac_probe.py``); without them HVAC keeps the defaults and the file says so. The numbers in the file
are measured on the pages the parameters were picked on: slightly optimistic, like the trainer's own threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import PageSegments
from .metrics import length_counts, prf
from .postprocess import DEFAULT_PARAMS, PARAMS_FILE, along_confident, by_group, objects_mask, style_groups

SEEDS = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)
SURE = (0.5, 0.7, 0.8, 0.9, 0.95, 2.0)  # 2.0 = off
GROUP = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 2.0)


def _pages(run: Path, data: Path, objects: Path | None):
    for fold in sorted(run.glob("fold*")):
        if not (fold / "val_scores.npz").exists():
            continue
        z = np.load(fold / "val_scores.npz")
        for pg in json.loads((fold / "result.json").read_text(encoding="utf-8"))["pages"]:
            s = pg["stem"]
            d = np.load(data / f"{s}.npz")
            mode = json.loads(str(d["meta"]))["mode"]
            p = np.zeros(len(d["label"]))
            p[z[f"{s}__index"]] = z[s]
            off = d["shape_offsets"].astype(np.int64)
            segs = PageSegments(seg=d["segments"].astype(np.float64), owner=np.repeat(np.arange(len(off) - 1), np.diff(off)), shape_offsets=off)
            attrs = {k: d[k] for k in ("layer_id", "stroke_rgb", "width", "type_code")}
            g = dict(stem=s, mode=mode, p=p, y=d["label"].astype(bool), L=d["length"].astype(np.float64), group=style_groups(attrs))
            if mode == "hvac":
                f = None if objects is None else objects / s / f"{run.name}_union0.1.json"
                if f is None or not f.exists():
                    continue
                g["obj"] = objects_mask("hvac", json.loads(f.read_text(encoding="utf-8"))["objects"], segs, d["layer_id"].astype(np.int64))
            else:
                g["along"] = {t: along_confident(segs, p, t) for t in SEEDS}
            yield g


def _fit(pages: list[dict], grid: list[tuple], mask, names: tuple[str, str]) -> dict:
    count = lambda q: np.sum([length_counts(mask(g, q), g["y"], g["L"]) for g in pages], 0)  # noqa: E731
    best = max(grid, key=lambda q: prf(*count(q))["f2"])
    m = prf(*count(best))
    return {names[0]: best[0], names[1]: best[1], "pages": len(pages), "f2": round(m["f2"], 4), "recall": round(m["recall"], 4), "precision": round(m["precision"], 4)}


def calibrate(run: Path, data: Path, objects: Path | None = None, log=print) -> dict:
    pages = list(_pages(Path(run), Path(data), objects))
    out: dict = {"run": Path(run).name}
    for mode in ("fire", "plumbing"):
        ps = [g for g in pages if g["mode"] == mode]
        if ps:
            out[mode] = _fit(ps, [(t, a) for t in SEEDS for a in GROUP], lambda g, q: g["along"][q[0]] | by_group(g["p"], g["L"], g["group"], q[1]), ("seed", "group"))
    if "plumbing" not in out and "fire" in out:
        out["plumbing"] = {k: out["fire"][k] for k in ("seed", "group")}
    ps = [g for g in pages if g["mode"] == "hvac"]
    if ps:
        out["hvac"] = _fit(ps, [(t, a) for t in SURE for a in GROUP], lambda g, q: g["obj"] | (g["p"] >= q[0]) | by_group(g["p"], g["L"], g["group"], q[1]), ("sure", "group"))
    else:
        out["hvac"] = {**DEFAULT_PARAMS["hvac"], "source": "defaults: no logic objects for this run"}
    f = Path(run) / PARAMS_FILE
    f.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"wrote {f}: {json.dumps(out)}")
    return out
