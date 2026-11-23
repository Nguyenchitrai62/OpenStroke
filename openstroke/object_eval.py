"""Object F2 printed at the end of training, from the model alone (user 2026-11-19: the log must carry the object numbers, no files sent).

Needs only the val pages' own arrays (``label``, ``length``, ``oinst`` / ``ocls`` of e11 / e12) and the model's outputs; no AI_Takeoff logic, no
score pack, no GT class used to build or name objects. Two protocols, both IoU >= 0.5 on stroke / piece length, one match per object:

- ``strokes``: objects = connected components of the strokes the model picks (p >= checkpoint threshold) along its same-object links
  (sigmoid >= LINK); matched without class (the stroke model has no class head). GT objects = ``oinst`` of the pipe strokes.
- ``pieces`` (OpenStroke34): objects = pipe pieces (p >= PIECE_THR) along same-object links (>= LINK) between pieces of the same predicted
  class, named by their length-majority class; matched WITH class on GT pages, without on Viewer pages (their class is not learned, F162).
  GT objects with no candidate piece count as missed (same as research/probes/piece_object_probe.py).
"""

import numpy as np

LINK = 0.5
PIECE_THR = 0.5
MATCH_IOU = 0.5


def _components(n: int, e: np.ndarray) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    e = np.asarray(e, np.int64).reshape(-1, 2)
    _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
    return lab


def match(pred: np.ndarray, pred_cls: np.ndarray | None, gt: np.ndarray, gt_cls: np.ndarray | None, length: np.ndarray,
          n_gt: int | None = None) -> np.ndarray:
    """-> [matched, predicted, gt]. ``pred`` / ``gt``: object id per element (-1 none); ``*_cls``: class per element (None = class-free;
    an object takes its length-majority class). ``n_gt``: GT objects on the page when some have no element here (counted as missed)."""
    L = np.asarray(length, float)
    pids, gids = np.unique(pred[pred >= 0]), np.unique(gt[gt >= 0])
    n_g = max(len(gids), n_gt or 0)
    if not len(pids) or not len(gids):
        return np.array([0, len(pids), n_g], float)

    def size_cls(lab, cls, ids):
        size = {int(i): float(L[lab == i].sum()) for i in ids}
        if cls is None:
            return size, {int(i): 0 for i in ids}
        c = np.where(cls >= 0, cls, 0)
        return size, {int(i): int(np.bincount(c[lab == i], weights=L[lab == i]).argmax()) for i in ids}

    lp, cp = size_cls(pred, pred_cls, pids)
    lg, cg = size_cls(gt, gt_cls, gids)
    both = (pred >= 0) & (gt >= 0)
    base = int(gt.max()) + 1
    u, inv = np.unique(pred[both].astype(np.int64) * base + gt[both], return_inverse=True)
    inter = np.bincount(inv, weights=L[both])
    cand = []
    for k, a in zip(u, inter):
        p, g = int(k // base), int(k % base)
        iou = a / max(lp[p] + lg[g] - a, 1e-9)
        if iou >= MATCH_IOU and cp[p] == cg[g]:
            cand.append((iou, p, g))
    used_p, used_g, tp = set(), set(), 0
    for _, p, g in sorted(cand, reverse=True):
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        tp += 1
    return np.array([tp, len(pids), n_g], float)


def gt_objects(page) -> tuple[np.ndarray, int]:
    """(GT object id per page shape or -1, number of GT objects): ``oinst`` of the pipe strokes whose object has a class. FIRE / PLUMBING
    pages (fire1, thin1, F213) have no HVAC class (``ocls`` 0): every true stroke with a GT line object counts."""
    oinst = getattr(page, "oinst", None)
    if oinst is None:
        return None, 0
    ocls = getattr(page, "ocls", None)
    if getattr(page, "mode", "hvac") != "hvac":
        ocls = None
    ok = (page.label > 0) & (oinst >= 0) & ((ocls > 0) if ocls is not None else True)
    g = np.where(ok, oinst, -1)
    return g, int(len(np.unique(g[g >= 0])))


def stroke_counts(page, prob: np.ndarray, thr: float, edges: tuple | None, link: float = LINK) -> np.ndarray | None:
    """Strokes protocol for one page; ``edges`` = (E, 2) page positions and their same-object logits (predict_page edge_out)."""
    g, _ = gt_objects(page)
    if g is None:
        return None
    n = len(prob)
    pick = prob >= thr
    if edges is not None and len(edges[0]):
        e, z = edges
        keep = (1.0 / (1.0 + np.exp(-np.asarray(z, float))) >= link) & pick[e[:, 0]] & pick[e[:, 1]]
        lab = _components(n, e[keep])
    else:   # no link head: each picked stroke is its own object
        lab = np.arange(n)
    pred = np.where(pick, lab, -1)
    return match(pred, None, g, None, page.length)


def piece_counts(page, q: dict, link: float = LINK, thr: float = PIECE_THR, gt_fg: bool = False, gt_link: bool = False,
                 gt_cls: bool = False, same_cls: bool = True, signed_links: bool = False,
                 by_class: dict | None = None) -> np.ndarray | None:
    """Pieces protocol for one page; ``q`` = merge_pieces output (rows = npz piece rows). The ``gt_*`` switches put the label in place of one
    model output (pipe pieces / links / classes) and ``same_cls=False`` joins across predicted classes: diagnosis of which step fails
    (labels only score here, they never reach training)."""
    pc = page.pieces
    if pc is None:
        return None
    _, n_gt = gt_objects(page)
    order = np.argsort(pc["pid"])
    rows = pc["pid"][order]
    at = order[np.searchsorted(rows, q["pid"])]                       # page piece index of each output row
    P = len(pc["pid"])
    fg = np.zeros(P)
    fg[at] = q["fg"]
    cls = np.full(P, -1, np.int64)
    pcl = np.asarray(q["cls"], float).copy()
    if pcl.shape[1] > 1:
        pcl[:, 0] = -1.0                                              # class 0 = background, never a name
    if len(at):
        cls[at] = pcl.argmax(1)
    gfg = pc["pfg"].astype(bool) & (pc["pinst"] >= 0)
    gcls = pc["pcls"].astype(np.int64)
    viewer = page.stem.startswith("vw_") or (gfg.any() and not (gcls[gfg] > 0).any())
    on = gfg.copy() if gt_fg else fg >= thr
    if gt_cls and not viewer:
        cls = np.where(gcls > 0, gcls, cls)
    lab = np.arange(P)
    if len(q["pe"]):
        a = order[np.searchsorted(rows, q["pe"][:, 0])]
        b = order[np.searchsorted(rows, q["pe"][:, 1])]
        eligible = on[a] & on[b]
        if same_cls:
            eligible &= cls[a] == cls[b]
        pairs = np.c_[a, b]
        if signed_links:
            from .stroke_groups import mutex_components

            if gt_link or link != LINK:
                raise ValueError("Signed decoding uses the model logits at a fixed decision boundary of zero")
            lab = mutex_components(P, pairs[eligible], np.asarray(q["plink"])[eligible])
        else:
            if gt_link:
                ok = gfg[a] & gfg[b] & (pc["pinst"][a] == pc["pinst"][b])
            else:
                ok = 1.0 / (1.0 + np.exp(-np.asarray(q["plink"], float))) >= link
            lab = _components(P, pairs[eligible & ok])
    pred = np.where(on, lab, -1)
    gt = np.where(gfg & ((gcls > 0) | viewer), pc["pinst"], -1)
    if by_class is not None:
        if viewer or not same_cls:
            raise ValueError("Per-class breakdown requires labelled GT and within-class components")
        item_gt, _ = gt_objects(page)
        for k in np.union1d(np.union1d(cls[on], gcls[gfg]), page.ocls[page.label > 0]):
            if k <= 0:
                continue
            # Item and piece assignment can cover different GT IDs. Keep their union per class.
            ids = np.union1d(item_gt[(item_gt >= 0) & (page.ocls == k)], gt[(gt >= 0) & (gcls == k)])
            by_class[int(k)] = match(np.where(cls == k, pred, -1), None, np.where(gcls == k, gt, -1), None, pc["plen"], len(ids))
    return match(pred, None if viewer else cls, gt, None if viewer else gcls, pc["plen"], n_gt)


def group_of(stem: str) -> str:
    return "Viewer" if stem.startswith("vw_") else "GT"


def f2(c) -> tuple[float, float, float]:
    p, r = c[0] / max(c[1], 1), c[0] / max(c[2], 1)
    return 5 * p * r / max(4 * p + r, 1e-9), p, r


def summarize(per_page: dict[str, np.ndarray], groups: dict[str, str] | None = None) -> dict:
    """{group: {f2, precision, recall, matched, predicted, gt, pages}} plus 'all', micro over pages. ``groups``: {stem: group name}
    instead of GT / Viewer (the cohorts of a FIRE / PLUMBING dataset, F213)."""
    out = {}
    if groups:
        named = [(g, lambda s, g=g: groups.get(s) == g) for g in sorted(set(groups.values()))]
    else:
        named = [("GT", lambda s: group_of(s) == "GT"), ("Viewer", lambda s: group_of(s) == "Viewer")]
    for name, keep in (*named, ("all", lambda s: True)):
        pages = [s for s in per_page if keep(s)]
        if not pages:
            continue
        c = np.sum([per_page[s] for s in pages], 0)
        a, p, r = f2(c)
        out[name] = {"f2": round(a, 4), "precision": round(p, 4), "recall": round(r, 4), "matched": int(c[0]), "predicted": int(c[1]),
                     "gt": int(c[2]), "pages": len(pages)}
    return out


#: diagnosis arms printed after the pieces table: (name, piece_counts keyword arguments)
PIECE_DIAG = (("model", {}),
              ("join across classes", {"same_cls": False}),
              ("links >= 0.3", {"link": 0.3}),
              ("links >= 0.7", {"link": 0.7}),
              ("pipe >= 0.7", {"thr": 0.7}),
              ("label pipe pieces", {"gt_fg": True}),
              ("label links", {"gt_link": True}),
              ("label classes", {"gt_cls": True}),
              ("label pipe + links", {"gt_fg": True, "gt_link": True}))
