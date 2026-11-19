"""Frozen hvac10b GT subset and class-aware object reporting for OpenStroke35."""
import hashlib
import json

import numpy as np

from .object_eval import f2, gt_objects, match
from .objects import HVAC_CLASSES

GT_SPLIT = "54c0c303775333c06833d01baaa37f22ddc10ea05a2eca260d60593cf57af954"


def gt_entries(data, manifest, entries, *, require_pieces=True):
    records = {r["stem"]: r for r in manifest["records"] if r["status"] == "ok"}
    selected = [e for e in entries if e[0].startswith("doc") and e[2] == "hvac"]
    rows = []
    for stem, fold, _ in selected:
        r = records[stem]
        split = {-1: "train", 0: "val"}.get(fold)
        if split is None or r["split"] != split:
            raise ValueError(f"{stem}: expected frozen hvac10b fold")
        with np.load(data / r["npz"], allow_pickle=False) as z:
            m = json.loads(str(z["meta"]))
            if (m["source"], m["stem"], m["document_sha256"], m["page_number"]) != (
                    "ai_takeoff_gt", stem, r["document_sha256"], r["page_number"]):
                raise ValueError(f"{stem}: unexpected page provenance")
            if require_pieces:
                if not {"pc_inst", "pc_cls", "pc_fg", "pc_edges", "pc_ekind"} <= set(z.files):
                    raise ValueError(f"{stem}: OpenStroke35 needs the e12 piece dataset")
                if split == "val" and ("pc_pick_thr" not in z or not np.isclose(z["pc_pick_thr"], 0.75)):
                    raise ValueError(f"{stem}: expected validation geometry from E122 predicted strokes")
        rows.append((stem, r["document_sha256"], r["page_number"], split))
    rows.sort()
    digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    if digest != GT_SPLIT:
        raise ValueError("OpenStroke35 needs the frozen 19 train / 5 val AI_Takeoff pages of hvac10b")
    proof = {"sha256": digest, "rows": rows, "source": "ai_takeoff_gt"}
    if require_pieces:
        proof["validation_candidates"] = "E122@0.75"
    return selected, proof


def _metrics(c):
    score, precision, recall = f2(c)
    return dict(f2=float(score), precision=float(precision), recall=float(recall),
                matched=int(c[0]), predicted=int(c[1]), gt=int(c[2]))


def report_regions(pages, outputs, say, split, details=False):
    if len(outputs) != len(pages):
        raise ValueError("A scored page is missing its region output")
    per_page, by_class = {}, {}
    total = np.zeros(3)
    represented = 0
    for page, q in zip(pages, outputs):
        pc = page.pieces
        order = np.argsort(pc["pid"])
        at = order[np.searchsorted(pc["pid"][order], q["pid"])]
        pred = np.full(len(order), -1, np.int64)
        cls = np.zeros(len(order), np.int64)
        pred[at], cls[at] = q["region_id"], q["region_cls"]
        gc = pc["pcls"]
        gt = np.where((pc["pfg"] > 0) & (gc > 0), pc["pinst"], -1)
        item_gt, n_item = gt_objects(page)
        represented += len(np.unique(gt[gt >= 0]))
        active = (pred >= 0) | (gt >= 0)
        c = match(pred[active], cls[active], gt[active], gc[active], pc["plen"][active], n_item)
        total += c
        per_page[page.stem] = _metrics(c)
        if details:
            for k in range(1, len(HVAC_CLASSES)):
                mask = ((cls == k) & (pred >= 0)) | ((gc == k) & (gt >= 0))
                # Keep every element belonging to either side of the class-specific match.
                ids = np.union1d(gt[(gc == k) & (gt >= 0)], item_gt[(page.ocls == k) & (item_gt >= 0)])
                cnt = match(np.where(cls[mask] == k, pred[mask], -1), None, np.where(gc[mask] == k, gt[mask], -1),
                            None, pc["plen"][mask], len(ids))
                by_class[k] = by_class.get(k, np.zeros(3)) + cnt
    result = _metrics(total)
    say(f"OBJECTS regions {split}: F2 {result['f2']:.3f} P {result['precision']:.3f} R {result['recall']:.3f} "
        f"({result['matched']}/{result['predicted']}/{result['gt']}; GT with candidate pieces {represented})")
    if details:
        result["pages"] = per_page
        result["classes"] = {HVAC_CLASSES[k]: _metrics(v) for k, v in by_class.items() if v[1:].any()}
        for name, row in list(per_page.items()) + list(result["classes"].items()):
            say(f"  {name} F2 {row['f2']:.3f} P {row['precision']:.3f} R {row['recall']:.3f} "
                f"({row['matched']}/{row['predicted']}/{row['gt']})")
    return result


def aligned_memberships(ref_shape, ref_len, ref_id, ref_cls, live):
    """Intersect two partitions in arclength order within each original PDF shape."""
    shape = live["piece_shape"]
    pred = np.full(len(shape), -1, np.int64)
    pc = np.zeros(len(shape), np.int64)
    for k, obj in enumerate(live["objects"]):
        idx = obj["piece_ids"]
        pred[idx] = k
        pc[idx] = HVAC_CLASSES.index(obj["class"])
    parts = []
    for s in np.union1d(ref_shape[ref_id >= 0], shape[pred >= 0]):
        lo, hi = np.searchsorted(ref_shape, [s, s + 1])
        a, b = np.searchsorted(shape, [s, s + 1])
        ref_end = np.cumsum(ref_len[lo:hi], dtype=float)
        end = np.cumsum(live["piece_length"][a:b], dtype=float)
        cuts = np.unique(np.r_[0., ref_end, end])
        length = np.diff(cuts)
        mid = cuts[:-1] + length / 2
        ri = np.searchsorted(ref_end, mid, side="right")
        li = np.searchsorted(end, mid, side="right")
        # A shape present on only one side still contributes its whole unmatched length.
        parts.append((np.r_[pred[a:b], -1][li], np.r_[pc[a:b], 0][li], np.r_[ref_id[lo:hi], -1][ri],
                      np.r_[ref_cls[lo:hi], 0][ri], length))
    return tuple(np.concatenate(v) for v in zip(*parts)) if parts else (
        np.empty(0, int), np.empty(0, int), np.empty(0, int), np.empty(0, int), np.empty(0))


def live_report(model, pages, scores, data, window, batch, device, amp, say):
    """Run the deployed second pass on stage-1 picks. Reference membership stays frozen when the model changes."""
    import copy

    from .piece_candidates import predict_regions
    from .relabel import page_segments_of

    total, per_page, class_counts = np.zeros(3), {}, {}
    for page, score in zip(pages, scores):
        with np.load(data / (page.stem + ".npz"), allow_pickle=False) as raw:
            segs = page_segments_of(raw)
            prob = np.zeros(len(segs.shape_offsets) - 1, dtype=float)
            prob[page.index] = score
            attrs = {k: raw[k] for k in ("stroke_rgb", "type_code", "width")}
            live = predict_regions(model, copy.copy(page), {"segments": segs, "attrs": attrs}, prob, window, device, batch, amp)
            cls = raw["pc_cls"]
            gt = np.where((raw["pc_fg"] > 0) & (cls > 0), raw["pc_inst"], -1)
            p, c, g, gc, length = aligned_memberships(raw["pc_shape"], raw["pc_len"], gt, cls, live)
        ig, n_item = gt_objects(page)
        counts = match(p, c, g, gc, length, n_item)
        total += counts
        per_page[page.stem] = _metrics(counts)
        for k in range(1, len(HVAC_CLASSES)):
            ids = np.union1d(ig[(ig >= 0) & (page.ocls == k)], gt[(gt >= 0) & (cls == k)])
            sel = ((c == k) & (p >= 0)) | ((gc == k) & (g >= 0))
            cnt = match(np.where(c[sel] == k, p[sel], -1), None, np.where(gc[sel] == k, g[sel], -1), None,
                        length[sel], len(ids))
            class_counts[k] = class_counts.get(k, np.zeros(3)) + cnt
    result = _metrics(total)
    result["pages"] = per_page
    result["classes"] = {HVAC_CLASSES[k]: _metrics(v) for k, v in class_counts.items() if v[1:].any()}
    for name, row in [("GT", result), *per_page.items(), *result["classes"].items()]:
        say(f"OBJECTS live regions {name}: F2 {row['f2']:.3f} P {row['precision']:.3f} R {row['recall']:.3f} "
            f"({row['matched']}/{row['predicted']}/{row['gt']})")
    return result
