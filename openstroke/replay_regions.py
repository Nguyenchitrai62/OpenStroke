"""Diagnose object recall on frozen OpenStroke35 validation candidates using saved predictions."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from .folds import resolve
from .models.region_decoder import proposals
from .object_eval import MATCH_IOU, gt_objects, match
from .objects import HVAC_CLASSES
from .region_data import gt_entries
from .replay_pieces import counts_text


STAGES = ("parent@0.75", "piece@0.5", "component+oracle mask", "kept mask, class-free", "kept mask, typed")


def sufficient_length(gt, length, selected, full):
    valid = (gt >= 0) & selected
    length_found = np.bincount(gt[valid], weights=length[valid], minlength=len(full))
    return np.flatnonzero((full > 0) & (length_found >= MATCH_IOU * full))


def feasible_objects(rows, groups, gt, length, full):
    """Maximum one-to-one recall with a perfect class and a mask that can only remove pieces."""
    valid = gt[rows] >= 0
    if not valid.any():
        return np.empty(0, dtype=int)
    cover = csr_matrix((length[rows[valid]], (groups[valid], gt[rows[valid]])), shape=(int(groups.max()) + 1, len(full)))
    entries = cover.tocoo()
    enough = entries.data >= MATCH_IOU * full[entries.col]
    # This maximum is optimistic. When an ambiguous component can match two GT objects,
    # the selected GT for per-class coverage is arbitrary among maximum matchings.
    feasible = csr_matrix((np.ones(enough.sum()), (entries.row[enough], entries.col[enough])), shape=cover.shape)
    return np.flatnonzero(maximum_bipartite_matching(feasible, perm_type="row") >= 0)


def read_page(data, rec, sc):
    stem = rec["stem"]

    def score(name):
        return sc[f"{stem}__{name}"]

    with np.load(data / rec["npz"], allow_pickle=False) as z:
        cls, length = z["pc_cls"], z["pc_len"].astype(float)
        gt = np.where((z["pc_fg"] > 0) & (cls > 0), z["pc_inst"], -1)
        parent = z["pc_shape"]
        idx = score("index")
        page = SimpleNamespace(label=z["label"][idx], ocls=z["ocls"][idx], oinst=z["oinst"][idx])
        prob = np.zeros(len(z["label"]))
        prob[idx] = sc[stem]
    pid, pe, pfg, plink = score("pid"), score("pe"), score("pfg"), score("plink")
    rid, rcls = score("region_id"), score("region_cls")
    if (pid.shape != pfg.shape or pid.shape != rid.shape or pid.shape != rcls.shape
            or pe.shape != (len(plink), 2) or np.any(pid < 0) or np.any(pid >= len(gt))
            or len(np.unique(pid)) != len(pid) or not np.isin(pe, pid).all()):
        raise ValueError(f"{stem}: score piece IDs do not match the e12 graph")
    piece = np.zeros(len(gt), dtype=bool)
    piece[pid] = pfg >= 0.5
    rows, groups = proposals(piece, pe, plink >= 0)
    pred, pred_cls = np.full(len(gt), -1, int), np.zeros(len(gt), int)
    pred[pid], pred_cls[pid] = rid, rcls
    item_gt, n_item = gt_objects(page)
    full = np.bincount(gt[gt >= 0], weights=length[gt >= 0])
    reachable = [sufficient_length(gt, length, prob[parent] >= 0.75, full),
                 sufficient_length(gt, length, piece, full), feasible_objects(rows, groups, gt, length, full)]
    active = (pred >= 0) | (gt >= 0)
    free = match(pred[active], None, gt[active], None, length[active], n_item)
    typed = match(pred[active], pred_cls[active], gt[active], cls[active], length[active], n_item)

    by_class = {}
    for k in np.union1d(cls[gt >= 0], page.ocls[item_gt >= 0]):
        mask = (gt >= 0) & (cls == k)
        ids = np.union1d(gt[mask], item_gt[(item_gt >= 0) & (page.ocls == k)])
        counts = [np.isin(ids, a).sum() for a in reachable]
        for typed_match in (False, True):
            members = ((pred_cls == k) & (pred >= 0)) if typed_match else (pred >= 0)
            keep = members | mask
            c = match(np.where(members[keep], pred[keep], -1), None, np.where(mask[keep], gt[keep], -1),
                      None, length[keep], len(ids))
            counts.append(int(c[0]))
        by_class[int(k)] = np.array([*counts, len(ids)])
    return np.array([*(len(a) for a in reachable), int(free[0]), int(typed[0]), int(typed[2])]), typed, by_class


def print_row(name, row, say):
    gt = int(row[-1])
    cells = " ".join(f"{int(n)}/{gt} ({int(n) / max(1, gt):.1%})" for n in row[:-1])
    say(f"{name:32s} {cells}")


def report(data, scores, say=print):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    folds = json.loads((data / "folds.json").read_text(encoding="utf-8"))
    entries, _ = resolve(manifest, folds)
    verified, _ = gt_entries(data, manifest, entries)
    val = {stem for stem, fold, _ in verified if fold == 0}
    records = [r for r in manifest["records"] if r["stem"] in val and r["status"] == "ok"]
    say("Columns: " + " | ".join(STAGES))
    say("Parent threshold is a diagnostic on the frozen e12 pieces; it does not rerun live candidate construction.")
    say("Per-class class-free recall is matched within each class independently; its sum can differ from the aggregate.")
    total, typed, classes = np.zeros(6, int), np.zeros(3, int), {}
    with np.load(scores, allow_pickle=False) as sc:
        for rec in records:
            row, result, by_class = read_page(data, rec, sc)
            print_row(rec["stem"], row, say)
            total += row
            typed += result.astype(int)
            for k, c in by_class.items():
                classes[k] = classes.get(k, np.zeros(6, int)) + c
    print_row("GT5", total, say)
    for k, row in sorted(classes.items()):
        print_row(HVAC_CLASSES[k], row, say)
    say("Frozen candidates actual " + counts_text(typed))
    ceiling = np.array([total[2], total[2], total[-1]])
    say("Optimistic perfect mask/class in current components " + counts_text(ceiling))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("scores", type=Path)
    args = parser.parse_args()
    report(args.data, args.scores)


if __name__ == "__main__":
    main()
