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
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

CONFS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
PIPELINE_PAGES = ("doc000022", "doc000030", "doc000180", "doc000189", "doc000190", "doc000200", "doc000201")
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


def open_pack(pack) -> Path:
    """A pack folder, or a .zip of one (extracted to a temporary folder); -> the folder holding pages.json."""
    p = Path(pack)
    if p.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="score_pack_"))
        zipfile.ZipFile(p).extractall(tmp)
        p = tmp
    hits = [p] if (p / "pages.json").exists() else [q.parent for q in p.rglob("pages.json")]
    if not hits:
        raise FileNotFoundError(f"no pages.json in the score pack {pack}")
    return hits[0]


def score(pack: Path, dets: dict, how: str, log=print) -> dict:
    """-> {conf: {page: (N_CLASSES, 3) matched / predicted / GT objects}}."""
    from openstroke.objects import HVAC_CLASSES, assign_objects, object_counts
    from openstroke.relabel import STYLE_KEYS, page_segments_of

    names = json.loads((pack / "names.json").read_text(encoding="utf-8"))
    canon = {n.replace("_", " "): n for n in HVAC_CLASSES}  # data.yaml writes spaces as underscores
    res = {c: {} for c in CONFS}
    for rec in json.loads((pack / "pages.json").read_text(encoding="utf-8")):
        stem = rec["stem"]
        if stem not in dets:
            log(f"  no detections for {stem}")
            continue
        z = np.load(pack / "npz" / f"{stem}.npz")
        d = {k: z[k] for k in z.files}
        segs = page_segments_of(d)
        y, L = d["label"].astype(bool), d["length"].astype(np.float64)
        attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
        gobj = json.loads((pack / "gt" / f"{stem}.json").read_text(encoding="utf-8"))
        gcls, ginst, _ = assign_objects(segs, gobj, y)
        gcls_of = {int(i): int(c) for i, c in zip(ginst[ginst >= 0], gcls[ginst >= 0])}
        allp = []
        for q in dets[stem]:
            if (q.get("source") == "graph") != (how == "graph"):  # graph objects only with how=graph
                continue
            name = names[int(q["class_id"])].replace("_", " ")
            allp.append((HVAC_CLASSES.index(canon.get(name, name)), np.asarray(q["poly"], float).reshape(4, 2) / rec["scale"],
                         float(q["score"]), q.get("members") or []))
        for conf in CONFS:
            keep = [p for p in allp if p[2] >= conf]
            pi = boxes_to_strokes(how, keep, segs, attrs, L)
            pcls = np.array([c for c, _, _, _ in keep] or [0], np.int64)
            res[conf][stem] = object_counts(pi.astype(np.int64), pcls, ginst, gcls_of, L)
        log(f"  {how}: {stem} scored ({len(allp)} objects)")
    return res


def _f(c):
    tp, npred, ngt = np.asarray(c)[1:].sum(0)
    p, r = tp / max(npred, 1), tp / max(ngt, 1)
    return 5 * p * r / max(4 * p + r, 1e-9), 2 * tp / max(npred + ngt, 1), p, r


def report(res: dict, how: str, log=print) -> dict:
    """Prints F2 / F1 / P / R per confidence and per-class P / R at the best F2; -> summary dict."""
    from openstroke.objects import HVAC_CLASSES, N_CLASSES

    tot = {c: np.sum([v for s, v in res[c].items() if s[:9] in PIPELINE_PAGES], 0) for c in CONFS}
    tot = {c: v if np.ndim(v) == 2 else np.zeros((N_CLASSES, 3), np.int64) for c, v in tot.items()}
    log(f"\n=== {how}: object F2 on the 7 pipeline pages | reference {REF.get(how, REF['members'])}; {PIPELINE}")
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
    return {"how": how, "best_conf": best, "F2": f2, "F1": f1, "P": p, "R": r, "per_class": per}


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
        raw = predict(model, images, conf=0.05, tile_scale=float(kv.get("tile_scale", 1)), save=False, batch=int(kv.get("batch", 8)))
        dets = {Path(k).stem: v for k, v in raw.items()}
        out = Path(kv.get("out") or Path(model).with_name("object_score.json"))
        out.with_name("val_dets.json").write_text(json.dumps(dets, default=float), encoding="utf-8")
        print("detections:", {k: len(v) for k, v in dets.items()}, flush=True)
    has_graph = any(q.get("source") == "graph" for v in dets.values() for q in v)
    hows = [h for h in kv.get("how", "members,graph,rule").split(",") if h and (h != "graph" or has_graph)]
    summary = {h: report(score(pack, dets, h), h) for h in hows}
    print("\n=== summary (best conf by F2, 7 pipeline pages) | " + PIPELINE)
    for h, s in summary.items():
        print(f"  {h:8s} F2 {s['F2']:.3f}  P {s['P']:.3f}  R {s['R']:.3f}  (conf {s['best_conf']})")
    out.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
