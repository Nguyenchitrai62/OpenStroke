"""Replay H142 on archived stroke/edge scores. Uses no model, fitting or label-derived inputs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openstroke.stroke_groups import stroke_instances  # noqa: E402
from openstroke.object_eval import MATCH_IOU, f2  # noqa: E402
from openstroke.objects import HVAC_CLASSES, class_of, general_class  # noqa: E402


def overlaps(labels, gt, length, gt_length):
    """Same greedy length-IoU match as object_eval.match, retaining matched GT IDs."""
    _, pred = np.unique(labels, return_inverse=True)
    pred = pred.ravel()
    size = np.bincount(pred, weights=length)
    keep = gt >= 0
    pairs, inverse = np.unique(pred[keep] * len(gt_length) + gt[keep], return_inverse=True)
    inter = np.bincount(inverse, weights=length[keep])
    p, g = np.divmod(pairs, max(1, len(gt_length)))
    iou = inter / np.maximum(size[p] + gt_length[g] - inter, 1e-9)
    ok = iou >= MATCH_IOU
    matched, used = set(), set()
    for _, pi, gi in sorted(zip(iou[ok], p[ok], g[ok]), reverse=True):
        if pi not in used and gi not in matched:
            used.add(pi)
            matched.add(gi)
    return len(size), sorted(matched)


def replay_page(rec, data, pack, sc, threshold):
    stem = rec["stem"]
    objects = json.loads((pack / "gt" / f"{stem}.json").read_text(encoding="utf-8"))
    raw_class = general_class(np.array([class_of(o.get("shape_name")) if o.get("group") in ("primary", "reference")
                                        else 0 for o in objects], dtype=int))
    with np.load(data / rec["npz"], allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if (meta["source"], meta["stem"], meta["document_sha256"], meta["page_number"]) != (
                "ai_takeoff_gt", stem, rec["document_sha256"], rec["page_number"]):
            raise ValueError(f"{stem}: unexpected provenance")
        idx = sc[f"{stem}__index"]
        length = z["length"].astype(float)
        gt = np.where((z["label"] > 0) & (z["ocls"] > 0), z["oinst"], -1)
        owned = gt >= 0
        if not np.array_equal(raw_class[gt[owned]], z["ocls"][owned]):
            raise ValueError(f"{stem}: GT pack/classes disagree with object IDs")
    prob = sc[stem]
    edges = sc[f"{stem}__eij"]
    logits = sc[f"{stem}__elog"].astype(float)
    if (len(idx) != len(prob) or len(np.unique(idx)) != len(idx) or np.any(idx < 0) or np.any(idx >= len(gt))
            or np.any(edges < 0) or np.any(edges >= len(gt)) or not np.isfinite(logits).all() or not np.isfinite(prob).all()):
        raise ValueError(f"{stem}: invalid cached scores")
    # Compact IDs only after selecting vertices by prediction. Isolated selected vertices are objects too.
    nodes = np.sort(idx[prob >= threshold])
    local = np.full(len(gt), -1, np.int64)
    local[nodes] = np.arange(len(nodes))
    score = np.zeros(len(gt), dtype=prob.dtype)
    score[idx] = prob
    e = local[edges]
    selected = (e >= 0).all(1)
    e = np.sort(e[selected], axis=1)
    a, b = e[logits[selected] >= 0].T
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(nodes), len(nodes)))
    cc = connected_components(graph, directed=False)[1]
    signed = stroke_instances(score, threshold, edges, logits)[nodes]
    gt_length = np.bincount(gt[owned], weights=length[owned], minlength=len(objects))
    covered = gt_length > 0
    has = gt[nodes] >= 0
    picked_length = np.bincount(gt[nodes][has], weights=length[nodes][has], minlength=len(objects))
    output = {}
    for arm, labels in (("components", cc), ("mutex", signed)):
        n_pred, hits = overlaps(labels, gt[nodes], length[nodes], gt_length)
        output[arm] = np.array([len(hits), n_pred, covered.sum()])
        output[arm + "_hits"] = np.bincount(raw_class[hits], minlength=len(HVAC_CLASSES))
    output["gt"] = np.bincount(raw_class[covered], minlength=len(HVAC_CLASSES))
    output["raw"] = np.bincount(raw_class[raw_class > 0], minlength=len(HVAC_CLASSES))
    output["insufficient"] = int(np.sum(covered & (picked_length < MATCH_IOU * gt_length)))
    return output


def describe(c):
    a, p, r = f2(c)
    return f"{a:.4f}/{p:.4f}/{r:.4f} ({int(c[0])}/{int(c[1])}/{int(c[2])})"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "pack", "scores"):
        ap.add_argument(arg, type=Path)
    ap.add_argument("--threshold", required=True, type=float)
    ap.add_argument("--split", choices=("train", "val"), default="val")
    args = ap.parse_args()
    records = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))["records"]
    provenance = json.loads((args.pack / "pages.json").read_text(encoding="utf-8"))
    source = {r["stem"]: r for r in provenance}
    selected = []
    for rec in records:
        if rec["status"] != "ok" or rec["split"] != args.split or rec["mode"] != "hvac":
            continue
        p = source[rec["stem"]]
        if rec["stem"].startswith("vw_") or p.get("pseudo") is not False:
            continue
        if p["split"] != args.split or p["page_number"] != rec["page_number"]:
            raise ValueError(f"score pack and dataset disagree for {rec['stem']}")
        selected.append(rec)
    if not selected:
        raise ValueError("No AI_Takeoff GT pages in split")
    totals = Counter()
    with np.load(args.scores, allow_pickle=False) as sc:
        for rec in selected:
            v = replay_page(rec, args.data, args.pack, sc, args.threshold)
            for k, value in v.items():
                totals[k] += value
            print(rec["stem"], "CC", describe(v["components"]), "mutex", describe(v["mutex"]),
                  "covered/raw", v["gt"].sum(), v["raw"].sum(), "insufficient", v["insufficient"])
    c = totals["components"]
    print("micro F2/P/R (matched/predicted/covered GT)", describe(c), "mutex", describe(totals["mutex"]))
    print("Oracle verification", describe([c[0], c[0], c[2]]), "raw GT", totals["raw"].sum(),
          "insufficient selected length", totals["insufficient"])
    print("Class-agnostic matches by true class: class CC_hits mutex_hits covered raw")
    for k, name in enumerate(HVAC_CLASSES):
        if totals["raw"][k]:
            print(name, *(int(totals[key][k]) for key in ("components_hits", "mutex_hits", "gt", "raw")))


if __name__ == "__main__":
    main()
