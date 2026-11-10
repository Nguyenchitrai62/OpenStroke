"""Why GT objects are not matched (the recall ceiling of the OpenOBB HVAC objects, step 1 of OpenStroke10). 0 GPU.

For each GT object of the 7 pipeline pages, from the strokes of the predicted objects (rule or members, one confidence):
- matched: same class, shared length >= half of the union (the scorer's rule)
- wrong class: a predicted object of another class would match
- missed: predicted objects hold < 10 % of its length
- split: >= 50 % of its length is held, but by several predicted objects, none reaching the match
- partial: 10-50 % of its length is held
- merged: one predicted object holds >= 50 % of it but is too large (union too big)
Run on the pack of tools/make_score_pack.py.

    py -3.12 research/probes/obb_miss_probe.py PACK VAL_DETS rule|members CONF [CONF ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openstroke.objects import HVAC_CLASSES, assign_objects  # noqa: E402
from openstroke.relabel import STYLE_KEYS, object_strokes, page_segments_of  # noqa: E402

PIPE = ("doc000022", "doc000030", "doc000180", "doc000189", "doc000190", "doc000200", "doc000201")
CATS = ("matched", "wrong class", "missed", "split", "partial", "merged")


def boxes_to_strokes(how, keep, segs, attrs, L):
    """-> (stroke -> index into keep, -1 none). how: members / members_small / members_dedupe / members_g1 / rule / rule_g1 (E84)."""
    from openstroke.assemble import assign_members, dedupe_boxes
    if how in ("members_clip", "members_style", "members_cs"):
        from openstroke.assemble import filter_members
        sty = None
        if how != "members_clip":
            rgb = np.asarray(attrs["stroke_rgb"]).reshape(len(L), -1)
            key = np.c_[rgb, np.round(np.asarray(attrs["width"], float), 2), np.asarray(attrs["type_code"])]
            sty = np.unique(key, axis=0, return_inverse=True)[1].ravel()
        keep = filter_members(keep, segs, L, style=sty, clip=how != "members_style")
        return assign_members(keep, L, order="score", dedupe=False)
    if how.startswith("members"):
        if how == "members":
            return assign_members(keep, L, order="score", dedupe=False)
        return assign_members(keep, L, order="score" if how == "members_dedupe" else "small", dedupe=how != "members_small")
    idx = dedupe_boxes(keep) if how == "rule_g1" else list(range(len(keep)))
    sub = [keep[i] for i in idx]
    objs = [{"shape_name": HVAC_CLASSES[c], "vertices": np.asarray(v).tolist(), "score": s} for c, v, s, _ in sub]
    pm = object_strokes("hvac", objs, segs, attrs, L) if objs else np.zeros(len(L), bool)
    pi = assign_objects(segs, objs, pm)[1].astype(np.int64)
    return np.where(pi >= 0, np.asarray(idx + [-1], np.int64)[pi], -1)


def predicted(how, keep, segs, attrs, L):
    objs = [{"shape_name": HVAC_CLASSES[c], "vertices": v.tolist(), "score": s} for c, v, s, _ in keep]
    if how == "members":
        pi = np.full(len(L), -1, np.int64)
        for k in np.argsort([q[2] for q in keep]):
            m = np.asarray(keep[k][3], np.int64)
            pi[m[(m >= 0) & (m < len(L))]] = k
        return pi
    pm = object_strokes("hvac", objs, segs, attrs, L) if objs else np.zeros(len(L), bool)
    return assign_objects(segs, objs, pm)[1].astype(np.int64)


def classify(pi, pcls, ginst, gcls_of, L):
    out = []
    lp = np.bincount(pi[pi >= 0], L[pi >= 0], minlength=max(int(pi.max()) + 1, 1)) if (pi >= 0).any() else np.zeros(1)
    for g in np.unique(ginst[ginst >= 0]):
        m = ginst == g
        lg = L[m].sum()
        held = pi[m]
        share = np.bincount(held[held >= 0], L[m][held >= 0], minlength=len(lp)) if (held >= 0).any() else np.zeros(len(lp))
        cov = share.sum() / max(lg, 1e-9)
        iou = share / np.maximum(lp + lg - share, 1e-9)
        c = gcls_of[int(g)]
        ok = (iou >= 0.5)
        if (ok & (pcls[:len(ok)] == c)).any():
            cat = "matched"
        elif ok.any():
            cat = "wrong class"
        elif cov < 0.1:
            cat = "missed"
        elif share.max() >= 0.5 * lg:
            cat = "merged"
        elif cov >= 0.5:
            cat = "split"
        else:
            cat = "partial"
        out.append((c, cat, lg))
    return out


def main(pack, dets_file, how, *confs):
    pack = Path(pack)
    dets = json.loads(Path(dets_file).read_text(encoding="utf-8"))
    names = json.loads((pack / "names.json").read_text(encoding="utf-8"))
    pages = [r for r in json.loads((pack / "pages.json").read_text(encoding="utf-8")) if r["stem"][:9] in PIPE]
    cache = {}
    for rec in pages:
        z = np.load(pack / "npz" / f"{rec['stem']}.npz")
        d = {k: z[k] for k in z.files}
        segs = page_segments_of(d)
        y, L = d["label"].astype(bool), d["length"].astype(np.float64)
        attrs = {k: d[k] for k in STYLE_KEYS} | {"op_counts": d["op_counts"]}
        gobj = json.loads((pack / "gt" / f"{rec['stem']}.json").read_text(encoding="utf-8"))
        gcls, ginst, _ = assign_objects(segs, gobj, y)
        gcls_of = {int(i): int(c) for i, c in zip(ginst[ginst >= 0], gcls[ginst >= 0])}
        allp = [(HVAC_CLASSES.index(names[int(q["class_id"])]), np.asarray(q["poly"], float).reshape(4, 2) / rec["scale"], float(q["score"]),
                 q.get("members") or []) for q in dets.get(rec["stem"], [])]
        cache[rec["stem"]] = (segs, attrs, L, ginst, gcls_of, allp)
    for conf in map(float, confs):
        rows = []
        for stem, (segs, attrs, L, ginst, gcls_of, allp) in cache.items():
            keep = [p for p in allp if p[2] >= conf]
            pi = boxes_to_strokes(how, keep, segs, attrs, L)
            pcls = np.array([c for c, _, _, _ in keep] or [0], np.int64)
            rows += classify(pi, pcls, ginst, gcls_of, L)
        n = len(rows)
        print(f"\n{how} conf {conf}: {n} GT objects; predicted boxes kept {sum(len([p for p in v[5] if p[2] >= conf]) for v in cache.values())}")
        print("  all      " + "  ".join(f"{c} {sum(r[1] == c for r in rows) / n:.3f}" for c in CATS))
        for k in (1, 2, 4, 7, 8):
            rk = [r for r in rows if r[0] == k]
            if rk:
                print(f"  {HVAC_CLASSES[k][:12]:12s} " + "  ".join(f"{c} {sum(r[1] == c for r in rk) / len(rk):.2f}" for c in CATS) + f"  (n {len(rk)})")


if __name__ == "__main__":
    main(*sys.argv[1:])
