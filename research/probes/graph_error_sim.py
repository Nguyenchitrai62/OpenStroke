"""Why the stroke-graph objects score ~0.1 (OpenStroke10-12, F94): the GT graph of the 7 val pages, corrupted to the edge / class
quality the training logs report, decoded the way GraphPage does (connected components, class = length-weighted stroke class). 0 GPU,
no model: only the GT strokes, their k-NN candidate edges and random corruption.

- edges: the k-NN candidates (k = 8 of the 3 key points, 40 pt, every stroke of the page, as the model's head). A predicted positive set
  with recall R_e and precision P_e: true edges kept with probability R_e, false edges added between strokes of two *different* GT objects
  (the gate keeps object strokes only here: the best case for the gate) until the precision is P_e.
- classes: each stroke keeps its GT class with probability ``acc``, else a random other class (the logs: 0.12-0.18 on object strokes).
- box-gated decoding (the v13 candidate): an edge counts only when its two strokes are members of one detector box (E69 at conf >= 0.2),
  and the object class is the class of the highest-scored box among its strokes.

    py -3.12 research/probes/graph_error_sim.py PACK E69_VAL_DETS
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from duct_graph_oracle import knn_edges  # noqa: E402
from openstroke.objects import HVAC_CLASSES, N_CLASSES, assign_objects, object_counts  # noqa: E402
from openstroke.obb.object_score import open_pack  # noqa: E402
from openstroke.relabel import page_segments_of  # noqa: E402

PIPE = ("doc000022", "doc000030", "doc000180", "doc000189", "doc000190", "doc000200", "doc000201")


def load(pack, dets_file, conf=0.2):
    pack = open_pack(pack)
    dets = json.loads(Path(dets_file).read_text(encoding="utf-8"))
    names = json.loads((pack / "names.json").read_text(encoding="utf-8"))
    pages = []
    for rec in json.loads((pack / "pages.json").read_text(encoding="utf-8")):
        if rec["stem"][:9] not in PIPE:
            continue
        z = np.load(pack / "npz" / f"{rec['stem']}.npz")
        d = {k: z[k] for k in z.files}
        segs = page_segments_of(d)
        y, L = d["label"].astype(bool), d["length"].astype(np.float64)
        gobj = json.loads((pack / "gt" / f"{rec['stem']}.json").read_text(encoding="utf-8"))
        gcls, ginst, _ = assign_objects(segs, gobj, y)
        e = knn_edges(segs, np.ones(len(L), bool), 8, 40.0)
        boxes = [(HVAC_CLASSES.index(names[int(q["class_id"])].replace("_", " ")), float(q["score"]), np.asarray(q.get("members") or [], np.int64))
                 for q in dets.get(rec["stem"], []) if q.get("source") != "graph" and float(q["score"]) >= conf]
        pages.append((L, gcls.astype(np.int64), ginst.astype(np.int64), e, boxes))
    return pages


def decode(n, L, edges, keep, scls, box_class=None):
    """Components of ``edges`` among ``keep`` strokes; class = length-weighted stroke class (or the given per-component class)."""
    g = coo_matrix((np.ones(len(edges), bool), (edges[:, 0], edges[:, 1])), shape=(n, n)) if len(edges) else coo_matrix((n, n), dtype=bool)
    _, lab = connected_components(g, directed=False)
    lab = np.where(keep, lab, -1)
    ids, inv = np.unique(lab[lab >= 0], return_inverse=True)
    pred = np.full(n, -1, np.int64)
    pred[lab >= 0] = inv
    pcls = np.zeros(max(len(ids), 1), np.int64)
    for k in range(len(ids)):
        m = pred == k
        pcls[k] = box_class(m) if box_class else np.bincount(scls[m], L[m], minlength=N_CLASSES).argmax()
    return pred, pcls


def run(pages, R_e, P_e, acc, gated=False, seed=0):
    rng = np.random.default_rng(seed)
    tot = np.zeros((N_CLASSES, 3), np.int64)
    for L, gcls, ginst, e, boxes in pages:
        n = len(L)
        obj = ginst >= 0
        same = obj[e[:, 0]] & (ginst[e[:, 0]] == ginst[e[:, 1]])
        cross = obj[e[:, 0]] & obj[e[:, 1]] & ~same            # hard negatives: neighbouring objects
        tp = e[same][rng.random(int(same.sum())) < R_e]
        n_fp = int(round(len(tp) * (1 - P_e) / max(P_e, 1e-9)))
        neg = e[cross]
        fp = neg[rng.choice(len(neg), size=min(n_fp, len(neg)), replace=False)] if len(neg) and n_fp else np.zeros((0, 2), np.int64)
        pe = np.r_[tp, fp]
        scls = gcls.copy()
        wrong = obj & (rng.random(n) >= acc)
        scls[wrong] = (gcls[wrong] - 1 + rng.integers(1, N_CLASSES - 1, int(wrong.sum()))) % (N_CLASSES - 1) + 1
        box_class = None
        if gated:
            owner = [set() for _ in range(n)]
            for b, (_, _, mem) in enumerate(boxes):
                for s in mem[(mem >= 0) & (mem < n)].tolist():
                    owner[s].add(b)
            pe = np.asarray([p for p in pe.tolist() if owner[p[0]] & owner[p[1]]], np.int64).reshape(-1, 2)
            score = np.asarray([b[1] for b in boxes] + [0.0])
            cls_of = np.asarray([b[0] for b in boxes] + [0])

            def box_class(m, owner=owner, score=score, cls_of=cls_of):
                bs = set().union(*(owner[s] for s in np.flatnonzero(m)))
                return int(cls_of[max(bs, key=lambda b: score[b])]) if bs else 0
            keep = obj & np.asarray([bool(o) for o in owner])
        else:
            keep = obj
        pred, pcls = decode(n, L, pe, keep, scls, box_class)
        gcls_of = {int(i): int(c) for i, c in zip(ginst[obj], gcls[obj])}
        tot += object_counts(pred, pcls, ginst, gcls_of, L)
    tp_, npred, ngt = tot[1:].sum(0)
    p, r = tp_ / max(npred, 1), tp_ / max(ngt, 1)
    return 5 * p * r / max(4 * p + r, 1e-9), p, r


def main(pack, dets):
    pages = load(pack, dets)
    ne = sum(len(p[3]) for p in pages)
    pos = sum(int(((p[2][p[3][:, 0]] >= 0) & (p[2][p[3][:, 0]] == p[2][p[3][:, 1]])).sum()) for p in pages)
    print(f"7 pages: {ne} candidate edges, {pos} same-object ({pos / max(ne, 1):.2%})")
    print("\nconnected components, gate = GT object strokes (best case), class from strokes")
    for R_e, P_e in ((0.6, 0.03), (0.8, 0.07), (0.7, 0.2), (0.8, 0.5), (0.85, 0.8), (0.9, 0.95), (1.0, 1.0)):
        row = []
        for acc in (0.18, 0.6, 1.0):
            f2, p, r = run(pages, R_e, P_e, acc)
            row.append(f"acc {acc:.2f}: F2 {f2:.3f} (P {p:.2f} R {r:.2f})")
        print(f"  edges R {R_e:.2f} P {P_e:.2f} | " + " | ".join(row))
    print("\nbox-gated (E69 boxes conf >= 0.2 as seeds, class from the box)")
    for R_e, P_e in ((0.6, 0.03), (0.8, 0.07), (0.8, 0.5), (1.0, 1.0)):
        f2, p, r = run(pages, R_e, P_e, 0.18, gated=True)
        print(f"  edges R {R_e:.2f} P {P_e:.2f} | F2 {f2:.3f} (P {p:.2f} R {r:.2f})")


if __name__ == "__main__":
    main(*sys.argv[1:])
