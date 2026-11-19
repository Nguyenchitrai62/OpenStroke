"""Replay E127b class-aware object ablations from its archived Colab scores. No network forward pass."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .data import attach_pieces
from .object_eval import f2, piece_counts
from .objects import HVAC_CLASSES

ARMS = ("components", "mutex", "parent")


def counts_text(counts):
    score, precision, recall = f2(counts)
    return f"F2 {score:.4f} P {precision:.4f} R {recall:.4f} ({int(counts[0])}/{int(counts[1])}/{int(counts[2])})"


def edge_diagnosis(raw, q):
    """TP, predicted positive, true positive-class edges among links with two pipe-labelled endpoints."""
    a, b = np.asarray(q["pe"], dtype=np.int64).T
    inst = raw["pc_inst"]
    n = len(inst)
    fg = (raw["pc_fg"] > 0) & (inst >= 0)
    ends = np.sort(raw["pc_edges"].astype(np.int64), axis=1)
    keys = ends[:, 0] * n + ends[:, 1]
    order = np.argsort(keys)
    query = np.minimum(a, b) * n + np.maximum(a, b)
    offsets = np.searchsorted(keys[order], query)
    if np.any(offsets >= len(keys)) or not np.array_equal(keys[order[offsets]], query):
        raise ValueError("Score archive contains edges absent from this piece graph")
    kind = raw["pc_ekind"][order[offsets]]
    truth = inst[a] == inst[b]
    positive = q["plink"] >= 0
    result = {}
    for k in (1, 2):
        mask = fg[a] & fg[b] & (kind == k)
        result[k] = np.array([(mask & positive & truth).sum(), (mask & positive).sum(), (mask & truth).sum()])
    return result


def replay(rec, data, sc):
    st = rec["stem"]
    index = sc[st + "__index"].astype(np.int64)
    with np.load(data / rec["npz"], allow_pickle=False) as raw:
        meta = json.loads(str(raw["meta"]))
        if (meta["source"], meta["stem"], meta["document_sha256"], meta["page_number"]) != (
                "ai_takeoff_gt", st, rec["document_sha256"], rec["page_number"]):
            raise ValueError(f"{st}: GT source mismatch")
        if "pc_pick_thr" not in raw.files or not np.isclose(float(raw["pc_pick_thr"]), 0.75):
            raise ValueError(f"{st}: expected e12 validation candidates from E122 picks at 0.75")
        label = raw["label"]
        if len(np.unique(index)) != len(index) or np.any(index < 0) or np.any(index >= len(label)):
            raise ValueError(f"{st}: invalid payload row IDs")
        page = SimpleNamespace(index=index, stem=st, label=label[index], oinst=raw["oinst"][index], ocls=raw["ocls"][index])
        attach_pieces(page, raw)
        q = {k: sc[st + "__" + field] for k, field in (("pid", "pid"), ("fg", "pfg"), ("cls", "pcls"),
                                                         ("pe", "pe"), ("plink", "plink"))}
        pid = q["pid"].astype(np.int64)
        if (len(np.unique(pid)) != len(pid) or not np.isin(pid, page.pieces["pid"]).all()
                or not np.isin(q["pe"], pid).all()):
            raise ValueError(f"{st}: score and dataset piece IDs disagree")
        parent_prob = sc[st]
        row = np.full(len(label), -1, np.int64)
        row[index] = np.arange(len(index))
        parent = parent_prob[row[raw["pc_shape"][pid]]]
        edges = edge_diagnosis(raw, q)
    results = {}
    for arm in ARMS:
        by_class = {}
        out = {**q, "fg": parent} if arm == "parent" else q
        c = piece_counts(page, out, signed_links=arm == "mutex", by_class=by_class)
        results[arm] = c, by_class
    return results, edges


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path)
    ap.add_argument("scores", type=Path)
    args = ap.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    known = [r for r in manifest["records"] if r["status"] == "ok" and r["mode"] == "hvac"]
    pages = [r for r in known if r["split"] == "val" and not r["stem"].startswith("vw_")]
    train_sha = {r["document_sha256"] for r in known if r["split"] == "train"}
    if (manifest.get("split_plan") != "hvac10b" or len(pages) != 5
            or any(r["document_sha256"] in train_sha for r in pages)):
        raise ValueError("Expected the frozen five hvac10b validation GT pages")
    total = {a: np.zeros(3) for a in ARMS}
    class_total = {a: defaultdict(lambda: np.zeros(3)) for a in ARMS}
    edge_total = {1: np.zeros(3), 2: np.zeros(3)}
    recall_delta = {a: [] for a in ARMS}
    with np.load(args.scores, allow_pickle=False) as sc:
        for rec in pages:
            results, edges = replay(rec, args.data, sc)
            base_r = f2(results["components"][0])[2]
            for arm, (c, per_class) in results.items():
                print(rec["stem"], arm, counts_text(c))
                total[arm] += c
                recall_delta[arm].append(f2(c)[2] - base_r)
                for k, cnt in per_class.items():
                    class_total[arm][k] += cnt
            for k, cnt in edges.items():
                edge_total[k] += cnt
    for arm in ARMS:
        print("GT5", arm, counts_text(total[arm]))
        for k, c in sorted(class_total[arm].items()):
            print("  ", HVAC_CLASSES[k], counts_text(c))
    print("Per-class GT denominator is the union of item and piece object IDs; aggregate uses F194's legacy denominator.")
    for k, c in edge_total.items():
        _, p, r = f2(c)
        print(("touch" if k == 1 else "facing"), f"link P {p:.4f} R {r:.4f}", c.astype(int).tolist())
    base = total["components"]
    if (tuple(base[1:]) != (3025, 1368) or round(f2(base)[0], 3) != 0.563):
        raise SystemExit("Archive does not reproduce F194; withhold H143 verdict (compare score export precision and checkpoint).")
    for arm in ARMS[1:]:
        gain = f2(total[arm])[0] - f2(base)[0]
        worst = min(recall_delta[arm])
        verdict = "passes" if gain >= 0.02 and worst >= -0.02 else ("rejected" if gain < 0.005 else "inconclusive")
        print(f"H143 {arm}: {verdict}, delta F2 {gain:+.4f}, worst page delta recall {worst:+.4f}")


if __name__ == "__main__":
    main()
