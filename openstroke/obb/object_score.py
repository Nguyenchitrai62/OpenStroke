"""Object F2 of an HVAC OpenOBB model on the val pages, the product's own measure, in one command (Colab GPU or local).

The model predicts the val page images (with their vector tokens), then every predicted object is turned into strokes and matched to the
GT objects by shared stroke length (>= half of the union, same class), like research/probes/obb_score_probe.py and the pipeline's own
number. F2 = recall weighted twice (AI_Takeoff evaluate.fbeta, beta = 2). Three ways to give an object its strokes:

- ``members``: the strokes the model itself returned for each box (OpenOBB25 / OpenStroke9);
- ``graph``: the objects of the stroke graph (OpenStroke10, ``stroke_graph=True``), already made of strokes;
- ``rule``: the label rule picks the strokes of each box (light post-processing; slower).

The GT side comes from a score pack (``tools/make_score_pack.py``: GT objects in page coordinates + stroke arrays of the val pages), so no
AI_Takeoff database is needed. Reported on the 7 val pages that have a pipeline run.

    python -m openstroke.obb.object_score model=best.pt images=<obb_hvac7>/images/val pack=openstroke_score_pack_hvac7.zip
    python -m openstroke.obb.object_score dets=val_dets.json pack=<dir or zip> how=members,rule
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

CONFS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
# F129 (user 2026-11-12): the official GT is the one of the remote HVAC deploy. doc000022 and doc000030 exist only in the local DB
# (not remote mirrors) and may carry wrong GT: they leave training and scoring. The 6 remote val pages are scored (doc000187 added).
LOCAL_ONLY_DOCS = ("doc000022", "doc000030")
PIPELINE_PAGES = ("doc000180", "doc000187", "doc000189", "doc000190", "doc000200", "doc000201")
REF = {"members": "E69 members F2 0.513 (P 0.460 R 0.528)", "rule": "E69 rule F2 0.584 (P 0.571 R 0.588)",
       "graph": "best so far: E69 rule F2 0.584"}
PIPELINE = "pipeline with layers F2 0.888 (P 0.814 R 0.908)"


def boxes_to_strokes(how, keep, segs, attrs, L):
    """-> stroke -> index into ``keep`` (-1 none). how: members / graph / rule, and the E84 variants (members_small, members_dedupe,
    members_g1, members_clip, members_style, members_cs, rule_g1)."""
    from openstroke.assemble import assign_members, dedupe_boxes
    from openstroke.objects import HVAC_CLASSES, assign_objects
    from openstroke.relabel import object_strokes

    if how in ("members_clip", "members_style", "members_cs"):
        from openstroke.assemble import filter_members
        sty = None
        if how != "members_clip":
            rgb = np.asarray(attrs["stroke_rgb"]).reshape(len(L), -1)
            key = np.c_[rgb, np.round(np.asarray(attrs["width"], float), 2), np.asarray(attrs["type_code"])]
            sty = np.unique(key, axis=0, return_inverse=True)[1].ravel()
        keep = filter_members(keep, segs, L, style=sty, clip=how != "members_style")
        return assign_members(keep, L, order="score", dedupe=False)
    if how in ("members", "graph"):  # graph objects are already strokes: one component per stroke
        return assign_members(keep, L, order="score", dedupe=False)
    if how.startswith("members"):
        return assign_members(keep, L, order="score" if how == "members_dedupe" else "small", dedupe=how != "members_small")
    idx = dedupe_boxes(keep) if how == "rule_g1" else list(range(len(keep)))
    sub = [keep[i] for i in idx]
    objs = [{"shape_name": HVAC_CLASSES[c], "vertices": np.asarray(v).tolist(), "score": s} for c, v, s, _ in sub]
    pm = object_strokes("hvac", objs, segs, attrs, L) if objs else np.zeros(len(L), bool)
    pi = assign_objects(segs, objs, pm)[1].astype(np.int64)
    return np.where(pi >= 0, np.asarray(idx + [-1], np.int64)[pi], -1)


def check_pack_path(pack):
    """Cheap preflight before dataset preparation/model allocation; no extraction or scoring."""
    p = Path(pack)
    if not p.exists():
        raise FileNotFoundError(f"score_pack not found: {p}. Check the Drive path and upload the score ZIP before training.")
    if p.is_file() and p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as archive:
            if not any(Path(name).name == "pages.json" for name in archive.namelist()):
                raise ValueError(f"score_pack ZIP has no pages.json: {p}")
    elif p.is_dir():
        if not (p / "pages.json").is_file() and not next(p.rglob("pages.json"), None):
            raise ValueError(f"score_pack folder has no pages.json: {p}")
    else:
        raise ValueError(f"score_pack must be a .zip file or folder: {p}")


def open_pack(pack, temporary=None) -> Path:
    """A pack folder, or a .zip of one (extracted to a temporary folder); -> the folder holding pages.json."""
    p = Path(pack)
    if p.suffix.lower() == ".zip":
        tmp = Path(temporary) if temporary is not None else Path(tempfile.mkdtemp(prefix="score_pack_"))
        zipfile.ZipFile(p).extractall(tmp)
        p = tmp
    hits = [p] if (p / "pages.json").exists() else [q.parent for q in p.rglob("pages.json")]
    if not hits:
        raise FileNotFoundError(f"no pages.json in the score pack {pack}")
    return hits[0]


def score(pack: Path, dets: dict, how: str, log=print, cache=None, merge=False) -> dict:
    """-> {conf: {page: (N_CLASSES, 3) matched / predicted / GT objects}}."""
    from openstroke.objects import HVAC_CLASSES, assign_objects, object_counts
    from openstroke.relabel import STYLE_KEYS, page_segments_of

    names = json.loads((pack / "names.json").read_text(encoding="utf-8"))
    canon = {n.replace("_", " "): n for n in HVAC_CLASSES}  # data.yaml writes spaces as underscores
    res = {c: {} for c in CONFS}
    records = json.loads((pack / "pages.json").read_text(encoding="utf-8"))

    def one(rec):  # one page; pages are independent (F138: scored in threads, identical counts)
        stem = rec["stem"]
        if stem not in dets:
            return stem, None, 0
        cached = cache.get(stem) if cache is not None and how in ("graph", "members") and not merge else None
        if cached is not None:
            L, ginst, gcls_of = cached
            segs = attrs = None  # graph predictions already carry their source stroke IDs
        else:
            with np.load(pack / "npz" / f"{stem}.npz") as z:
                d = {k: z[k] for k in z.files}
            segs = page_segments_of(d)
            y, L = d["label"].astype(bool), d["length"].astype(np.float64)
            attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
            if how == "rule_nl":  # layer-free product: the rule's style groups use colour / width / type only (F109)
                attrs["layer_id"] = np.zeros_like(attrs["layer_id"])
            gobj = json.loads((pack / "gt" / f"{stem}.json").read_text(encoding="utf-8"))
            any_idx = set()
            if merge:  # hvac6: GT objects renamed / dropped before strokes are assigned to them
                from openstroke.objects import ANY_REDUCER, base6, class_of
                gobj = [o for o in gobj if not class_of(o.get("shape_name") or "") or base6(o["shape_name"]) is not None]
                any_idx = {i for i, o in enumerate(gobj) if base6(o.get("shape_name")) == ANY_REDUCER}   # base not recorded
                gobj = [dict(o, shape_name=HVAC_CLASSES[merge6_any(base6(o["shape_name"]))]) if class_of(o.get("shape_name") or "") else o
                        for o in gobj]
            gcls, ginst, _ = assign_objects(segs, gobj, y)
            gcls_of = {int(i): int(c) for i, c in zip(ginst[ginst >= 0], gcls[ginst >= 0])}
            if cache is not None and how in ("graph", "members"):
                cache[stem] = (L, ginst, gcls_of)
        allp = []
        for q in dets[stem]:
            if (q.get("source") or "members") != (how if how == "graph" or how.startswith(("seeded", "panoptic")) else "members"):  # one source per how
                continue
            name = names[int(q["class_id"])].replace("_", " ")
            if merge:
                from openstroke.objects import HVAC6_MERGE
                name = HVAC6_MERGE.get(name, name)
                if name is None:
                    continue
            allp.append((HVAC_CLASSES.index(canon.get(name, name)), np.asarray(q["poly"], float).reshape(4, 2) / rec["scale"],
                         float(q["score"]), q.get("members") or []))
        out = {}
        for conf in CONFS:
            keep = [p for p in allp if p[2] >= conf]
            pi = boxes_to_strokes(how, keep, segs, attrs, L)
            pcls = np.array([c for c, _, _, _ in keep] or [0], np.int64)
            cls_of = gcls_of
            if merge and any_idx:
                from openstroke.objects import resolve_any_reducer
                cls_of = resolve_any_reducer(pi.astype(np.int64), pcls, ginst, gcls_of, L, any_idx)
            out[conf] = object_counts(pi.astype(np.int64), pcls, ginst, cls_of, L)
        return stem, out, len(allp)

    workers = max(1, min(len(records), int(os.environ.get("OPENSTROKE_SCORE_WORKERS", min(8, os.cpu_count() or 1)))))
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers) as ex:
            done = list(ex.map(one, records))
    else:
        done = [one(rec) for rec in records]
    for stem, out, n in done:  # pages.json order, as before
        if out is None:
            log(f"  no detections for {stem}")
            continue
        for conf in CONFS:
            res[conf][stem] = out[conf]
        log(f"  {how}: {stem} scored ({n} objects)")
    return res


def merge6_any(c):
    from openstroke.objects import ANY_REDUCER, merge6
    return merge6(c) if c == ANY_REDUCER else c


def _f(c):
    tp, npred, ngt = np.asarray(c)[1:].sum(0)
    p, r = tp / max(npred, 1), tp / max(ngt, 1)
    return 5 * p * r / max(4 * p + r, 1e-9), 2 * tp / max(npred + ngt, 1), p, r


def report(res: dict, how: str, log=print, comparison_pages=PIPELINE_PAGES, reference=True) -> dict:
    """Prints F2 / F1 / P / R per confidence and per-class P / R at the best F2; -> summary dict."""
    from openstroke.objects import HVAC_CLASSES, N_CLASSES

    tot = {c: np.sum([v for s, v in res[c].items() if s[:9] in comparison_pages], 0) for c in CONFS}
    tot = {c: v if np.ndim(v) == 2 else np.zeros((N_CLASSES, 3), np.int64) for c, v in tot.items()}
    if reference:
        log(f"\n=== {how}: object F2 on the 7 pipeline pages | reference {REF.get(how, REF['members'])}; {PIPELINE}")
    else:
        log(f"\n=== {how}: vector object F2 on {len(comparison_pages)} fixed dataset pages")
    for c in CONFS:
        f2, f1, p, r = _f(tot[c])
        log(f"  conf {c:.2f}  F2 {f2:.3f}  F1 {f1:.3f}  P {p:.3f}  R {r:.3f}")
    best = max(CONFS, key=lambda c: _f(tot[c])[0])
    o = tot[best]
    f2, f1, p, r = _f(o)
    log(f"  best conf {best} (by F2): F2 {f2:.3f}  F1 {f1:.3f}  P {p:.3f}  R {r:.3f}")
    per = {}
    for k in range(1, N_CLASSES):
        tp, npred, ngt = (int(v) for v in o[k])
        if ngt or npred:
            pk, rk = tp / max(npred, 1), tp / max(ngt, 1)
            per[HVAC_CLASSES[k]] = {"P": pk, "R": rk, "F2": 5 * pk * rk / max(4 * pk + rk, 1e-9), "tp": tp, "pred": npred, "gt": ngt}
            log(f"    {HVAC_CLASSES[k]:20s} P {pk:.2f}  R {rk:.2f}  F2 {per[HVAC_CLASSES[k]]['F2']:.2f}  (tp {tp}, pred {npred}, gt {ngt})")
    per_page = {}
    for stem, counts in res[best].items():
        page_f2, _, page_p, page_r = _f(counts)
        per_page[stem] = {"F2": page_f2, "P": page_p, "R": page_r, "pipeline_page": stem[:9] in comparison_pages}
    return {"how": how, "best_conf": best, "F2": f2, "F1": f1, "P": p, "R": r,
            "per_class": per, "per_page": per_page}


def main(argv=None) -> int:
    kv = dict(a.split("=", 1) for a in (argv if argv is not None else sys.argv[1:]) if "=" in a)
    pack = open_pack(kv["pack"])
    if kv.get("dets"):
        dets = json.loads(Path(kv["dets"]).read_text(encoding="utf-8"))
        out = Path(kv.get("out") or Path(kv["dets"]).with_name("object_score.json"))
    else:
        from openstroke.obb.cli import predict

        model, images = kv["model"], kv["images"]
        print(f"predicting {images} with {model} ...", flush=True)
        scale = float(kv["tile_scale"]) if kv.get("tile_scale") else None
        raw = predict(model, images, conf=0.05, tile_scale=scale, save=False, batch=int(kv.get("batch", 8)))
        dets = {Path(k).stem: v for k, v in raw.items()}
        out = Path(kv.get("out") or Path(model).with_name("object_score.json"))
        out.with_name("val_dets.json").write_text(json.dumps(dets, default=float), encoding="utf-8")
        print("detections:", {k: len(v) for k, v in dets.items()}, flush=True)
    require_graph = kv.get("require_graph", "").lower() == "true"
    if require_graph:
        stems = {p["stem"] for p in json.loads((pack / "pages.json").read_text(encoding="utf-8"))}
        missing = set(PIPELINE_PAGES) - {stem[:9] for stem in stems & dets.keys()}
        if missing:
            raise ValueError(f"missing fixed comparison pages: {', '.join(sorted(missing))}")
    has_graph = require_graph or any(q.get("source") == "graph" for v in dets.values() for q in v)
    hows = [h for h in kv.get("how", "members,graph,rule").split(",") if h and (h != "graph" or has_graph)]
    details = out.with_name("object_score_details.log")
    with details.open("w", encoding="utf-8") as detail_file:
        def detail_log(message):
            print(message, file=detail_file, flush=True)
            if kv.get("verbose", "").lower() == "true":
                print(message, flush=True)

        summary = {h: report(score(pack, dets, h, log=detail_log), h, log=detail_log) for h in hows}
    print("\n=== FINAL OBJECT EVALUATION | 7 comparison pages ===")
    print("Main score: F2 (higher is better). Recall and precision are measured on full-page objects.")
    print(f"  {'Method':28s} {'F2':>7s} {'Recall':>9s} {'Precision':>10s}")
    primary = "graph" if "graph" in summary else "members"
    labels = {"graph": "Stroke graph [MAIN]", "members": "Model-returned strokes",
              "rule": "Model + stroke rule"}
    for h, s in summary.items():
        label = labels.get(h, h)
        if h == primary and h != "graph":
            label += " [MAIN]"
        print(f"  {label:28s} {s['F2']:7.3f} {s['R']:9.3f} {s['P']:10.3f}")
    print(f"  {'E69 + stroke rule':28s} {0.584:7.3f} {0.588:9.3f} {0.571:10.3f}")
    print(f"  {'Pipeline with layers':28s} {0.888:7.3f} {0.908:9.3f} {0.814:10.3f}")
    print(f"Detailed thresholds and per-class results: {details}")
    out.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
