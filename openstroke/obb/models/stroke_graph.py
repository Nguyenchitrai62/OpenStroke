"""OpenStroke10: HVAC objects as connected components of the stroke graph (docs/OPENSTROKE10.md, H100).

Two heads on a tile's primitive tokens (OpenOBB25's vector branch):
- class: every primitive -> 0 (not part of an HVAC object) or 1 + the object class;
- edge: candidate pairs (the k nearest of the 3 key points of each primitive: both ends and the middle, within ``radius`` px) -> "same
  object" logit from the two node features and their geometry.
Labels come from the token's ``inst`` column (index of its object in the tile's labels, -1 none): no new annotation. A page is decoded by
pooling the tiles per primitive id (``GraphPage``) and taking connected components of confident edges between object strokes.

The oracle on the 7 val pages (research/probes/duct_graph_oracle.py): perfect edges on this k = 8 candidate set give object F2 0.946.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .openobb8 import _sample

KEY = (0, 4, 7)  # key points of the 8 arc-length points: both ends and the middle
N_GEO = 6


def knn_pairs(pts: torch.Tensor, k: int = 8, radius: float = 160.0, chunk: int = 2048) -> torch.Tensor:
    """pts (M, 8, 2) px -> (E, 2) long, i < j: primitives whose key points are among each other's k nearest within ``radius``."""

    M = pts.shape[0]
    if M < 2:
        return pts.new_zeros((0, 2), dtype=torch.long)
    key = pts[:, KEY].reshape(-1, 2).float()
    owner = torch.arange(M, device=pts.device).repeat_interleave(len(KEY))
    kk = min(k + 1, len(key))
    out = []
    for s in range(0, len(key), chunk):
        d = torch.cdist(key[s:s + chunk], key)
        dist, nb = d.topk(kk, largest=False)
        a = owner[s:s + chunk, None].expand_as(nb)
        ok = dist <= radius
        out.append(torch.stack([a[ok], owner[nb[ok]]], 1))
    e = torch.cat(out)
    e = e[e[:, 0] != e[:, 1]]
    e = torch.sort(e, 1).values
    return torch.unique(e, dim=0)


def pair_geometry(pts: torch.Tensor, e: torch.Tensor, radius: float) -> torch.Tensor:
    """(E, 6): closest distance between the two strokes' 8 points, closest end to end distance (both / radius), |cos| of their
    directions, length ratio, lateral and axial offset of j's middle in i's frame (/ radius)."""

    p, q = pts[e[:, 0]].float(), pts[e[:, 1]].float()
    dmin = torch.cdist(p, q).flatten(1).amin(1)
    ends = torch.cdist(p[:, (0, -1)], q[:, (0, -1)]).flatten(1).amin(1)
    up, uq = p[:, -1] - p[:, 0], q[:, -1] - q[:, 0]
    lp, lq = up.norm(dim=1).clamp_min(1e-3), uq.norm(dim=1).clamp_min(1e-3)
    up, uq = up / lp[:, None], uq / lq[:, None]
    cos = (up * uq).sum(1).abs()
    ratio = torch.minimum(lp, lq) / torch.maximum(lp, lq)
    off = q[:, 4] - p[:, 4]
    axial = (off * up).sum(1)
    lateral = off[:, 0] * up[:, 1] - off[:, 1] * up[:, 0]
    return torch.stack([dmin / radius, ends / radius, cos, ratio, lateral.abs() / radius, axial.abs() / radius], 1).clamp(0, 4)


class StrokeGraphHead(nn.Module):
    """Class of every primitive + same-object logit of candidate primitive pairs (module docstring)."""

    def __init__(self, d_tok: int, num_classes: int, image_channels: int, d: int = 128, k: int = 8, radius: float = 160.0, foreground: bool = False, object_class: bool = False):
        super().__init__()
        self.k, self.radius = k, radius
        self.tok = nn.Linear(d_tok, d)
        self.image = nn.Linear(image_channels, d)
        self.cls = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, num_classes + 1))
        self.edge = nn.Sequential(nn.Linear(2 * d + N_GEO, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.cls[-1].weight)
        nn.init.zeros_(self.cls[-1].bias)
        nn.init.zeros_(self.edge[-1].weight)
        nn.init.constant_(self.edge[-1].bias, -4.0)  # ~0.2 % of candidate pairs are one object (oracle): start near that prior

        self.foreground = None
        if foreground:
            # Preserve the control run's RNG state and let all strokes pass at step zero.
            with torch.random.fork_rng(devices=[]):
                self.foreground = nn.Linear(d, 1)
            nn.init.zeros_(self.foreground.weight)
            nn.init.zeros_(self.foreground.bias)

        self.object_class = None
        if object_class:
            with torch.random.fork_rng(devices=[]):
                self.object_class = nn.Sequential(nn.Linear(num_classes + 3, 32), nn.GELU(), nn.Linear(32, num_classes))
            nn.init.zeros_(self.object_class[-1].weight)
            nn.init.zeros_(self.object_class[-1].bias)

    @staticmethod
    def page_logits(cls: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
        """Adjust the background logit without changing the conditional class evidence."""
        bg = cls[:, 0] - fg
        class_mass = torch.logsumexp(cls[:, 1:], dim=1)
        # A stroke rejected in every tile cannot turn into an object of its own.
        bg = torch.where(fg >= 0, bg, torch.maximum(bg, class_mass + 1e-4))
        return torch.cat((bg[:, None], cls[:, 1:]), dim=1)

    def classify_objects(self, prior: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        """Redistribute class probabilities while keeping the object's total HVAC probability."""
        mass = prior.sum(dim=1, keepdim=True)
        evidence = prior / mass.clamp_min(1e-9)
        inputs = torch.cat((evidence.float(), geometry.float()), dim=1)
        correction = self.object_class(inputs).to(prior.dtype)
        reweighted = prior * (correction - correction.amax(dim=1, keepdim=True)).exp()
        return reweighted * (mass / reweighted.sum(dim=1, keepdim=True).clamp_min(1e-9))

    def nodes(self, tok: torch.Tensor, pts: torch.Tensor, f4: torch.Tensor | None) -> torch.Tensor:
        """tok (M, d_tok), pts (M, 8, 2) px, f4 (C, H, W) stride-4 map of the detector (read only) -> (M, d)."""

        h = self.tok(tok.float())
        if f4 is not None and len(pts):
            with torch.no_grad():
                key = pts[:, KEY].float()
                H, W = f4.shape[-2:]
                vis = ((key >= 0) & (key < key.new_tensor([4 * W, 4 * H]))).all(-1)
                s = _sample(f4[None].float(), key.flatten(0, 1)[None], stride=4)[0].reshape(len(key), len(KEY), f4.shape[0])
                pooled = (s * vis[..., None]).sum(1) / vis.sum(1, keepdim=True).clamp_min(1)
            h = h + self.image(pooled)
        return h

    def forward(self, tok: torch.Tensor, pts: torch.Tensor, f4: torch.Tensor | None = None,
                train_foreground: torch.Tensor | None = None):
        """Tile class/edge logits, candidate pairs in tile row coordinates, and optional binary foreground logits."""

        h = self.nodes(tok, pts, f4)
        cls = self.cls(h).float()
        pts = pts.detach()
        fg = None
        if self.foreground is None:
            e = knn_pairs(pts, self.k, self.radius)
        else:
            fg = self.foreground(h).squeeze(1).float()
            keep = fg.detach() >= 0
            if train_foreground is not None:
                # A gate miss must not remove supervised positive examples from the edge head.
                candidates = (keep | train_foreground).nonzero(as_tuple=True)[0]
            else:
                candidates = keep.nonzero(as_tuple=True)[0]
                cls = self.page_logits(cls, fg)
            e = candidates[knn_pairs(pts[candidates], self.k, self.radius)]
        if len(e) == 0:
            return cls, e, cls.new_zeros(0), fg
        hi, hj = h[e[:, 0]], h[e[:, 1]]
        geo = pair_geometry(pts.detach(), e, self.radius).to(h.dtype)
        logit = self.edge(torch.cat([hi + hj, (hi - hj).abs(), geo], 1)).squeeze(1).float()
        return cls, e, logit, fg


def graph_targets(inst: torch.Tensor, labels: torch.Tensor, e: torch.Tensor):
    """inst (M,) object index in the tile's labels (-1 none), labels (N,) classes -> class target (M,), edge target (E,)."""

    inst = inst.round().long()
    has = (inst >= 0) & (inst < len(labels))
    cls = torch.zeros_like(inst)
    cls[has] = labels.to(inst.device).long()[inst[has]] + 1
    same = has[e[:, 0]] & has[e[:, 1]] & (inst[e[:, 0]] == inst[e[:, 1]]) if len(e) else inst.new_zeros(0, dtype=torch.bool)
    return cls, same.float()


def graph_loss(cls_logit, edge_logit, cls_t, edge_t, other_weight: float = 0.25, hard_negative=None, known_edge=None):
    """Class CE (the many non-object primitives down-weighted) + balanced edge BCE (mean over positives + mean over negatives)."""

    w = cls_logit.new_ones(cls_logit.shape[1])
    w[0] = other_weight
    lc = F.cross_entropy(cls_logit, cls_t, weight=w)
    if edge_logit.numel() == 0:
        return lc, edge_logit.sum() * 0.0
    lb = F.binary_cross_entropy_with_logits(edge_logit, edge_t, reduction="none")
    pos, neg = edge_t > 0.5, edge_t <= 0.5
    if hard_negative is None:
        le = (lb[pos].mean() if bool(pos.any()) else lb.sum() * 0.0) + (lb[neg].mean() if bool(neg.any()) else lb.sum() * 0.0)
    else:
        # Distinguish architectural/background clutter from neighboring HVAC objects.
        def mean(mask):
            return (lb * mask).sum() / mask.sum().clamp_min(1)

        hard, easy = neg & known_edge & hard_negative, neg & known_edge & ~hard_negative
        w_hard, w_easy = 0.75 * hard.any(), 0.25 * easy.any()
        negative = (w_hard * mean(hard) + w_easy * mean(easy)) / (w_hard + w_easy).clamp_min(0.25)
        le = mean(pos & known_edge) + negative
    return lc, le


def foreground_loss(logit: torch.Tensor, positive: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    """Binary gate loss with separate class means and a higher cost for missed HVAC strokes."""
    loss = F.binary_cross_entropy_with_logits(logit, positive.float(), reduction="none")
    pos, neg = known & positive, known & ~positive
    return (loss * pos).sum() / pos.sum().clamp_min(1) + 0.25 * (loss * neg).sum() / neg.sum().clamp_min(1)


def object_geometry(pts: torch.Tensor, group: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-object features shared by GT training groups and predicted page components."""
    length = pts.diff(dim=1).norm(dim=2).sum(dim=1)
    count = pts.new_zeros(n).index_add_(0, group, torch.ones_like(length))
    total = pts.new_zeros(n).index_add_(0, group, length)
    center = pts.new_zeros((n, 2)).index_add_(0, group, pts.mean(dim=1)) / count.clamp_min(1)[:, None]
    x, y = (pts - center[group, None]).unbind(dim=2)
    second = torch.stack((x.square().mean(1), y.square().mean(1), (x * y).mean(1)), dim=1)
    xx, yy, xy = (pts.new_zeros((n, 3)).index_add_(0, group, second) / count.clamp_min(1)[:, None]).unbind(dim=1)
    spread = ((xx - yy).square() + 4 * xy.square()).sqrt()
    major = (0.5 * (xx + yy + spread)).clamp_min(1e-12)
    minor = (0.5 * (xx + yy - spread)).clamp_min(0)
    features = torch.stack((count.log1p(), (minor / major).sqrt(), (total / (2 * major.sqrt())).log1p()), dim=1)
    return features, length


def object_class_loss(head, cls, pts, inst, labels):
    """Learn the residual on GT groups. The primitive class head still has its original loss."""
    valid = (inst >= 0) & (inst < len(labels))
    if not bool(valid.any()):
        return cls.sum() * 0.0
    group = inst[valid].long()
    geometry, lengths = object_geometry(pts[valid], group, len(labels))
    # Learn to correct the current class evidence without a shortcut through the primitive classifier.
    prob = cls[valid].detach().softmax(dim=1)[:, 1:]
    mass = prob.new_zeros(len(labels)).index_add_(0, group, lengths)
    prior = prob.new_zeros((len(labels), prob.shape[1])).index_add_(0, group, prob * lengths[:, None])
    used = mass > 0
    if not bool(used.any()):
        return cls.sum() * 0.0
    adjusted = head.classify_objects(prior[used] / mass[used, None], geometry[used])
    return F.cross_entropy(adjusted.clamp_min(1e-9).log(), labels[used])


class GraphPage:
    """Tiles of one page -> objects. Class probabilities are averaged per primitive id over the tiles that saw it; an edge keeps the
    highest probability any tile gave it."""

    def __init__(self, num_classes: int):
        self.nc = num_classes
        self.prob, self.count, self.edges = {}, {}, {}

    def add(self, pid: np.ndarray, cls_prob: np.ndarray, e: np.ndarray, edge_prob: np.ndarray) -> None:
        for p, c in zip(pid.tolist(), cls_prob):
            if p in self.prob:
                self.prob[p] += c
                self.count[p] += 1
            else:
                self.prob[p] = c.astype(np.float64).copy()
                self.count[p] = 1
        for (a, b), q in zip(pid[e].tolist() if len(e) else [], edge_prob.tolist()):
            key = (a, b) if a < b else (b, a)
            if q > self.edges.get(key, -1.0):
                self.edges[key] = q

    def objects(self, length: dict | None = None, edge_thr: float = 0.5, keep_thr: float = 0.5, same_class: bool = False, class_evidence: bool = False) -> list[dict]:
        """-> [{"class_id", "score", "members"}]: components of edges >= edge_thr between primitives with P(no object) < keep_thr."""

        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        if not self.prob:
            return []
        ids = np.fromiter(self.prob, np.int64)
        P = np.stack([self.prob[p] / self.count[p] for p in ids.tolist()])
        keep = P[:, 0] < keep_thr
        pos = {p: i for i, p in enumerate(ids.tolist())}
        best = P[:, 1:].argmax(1)
        rows, cols = [], []
        for (a, b), q in self.edges.items():
            i, j = pos.get(a), pos.get(b)
            if i is None or j is None or q < edge_thr or not (keep[i] and keep[j]):
                continue
            if same_class and best[i] != best[j]:
                continue
            rows.append(i)
            cols.append(j)
        n = len(ids)
        _, lab = connected_components(coo_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(n, n)), directed=False)
        w = np.asarray([float((length or {}).get(p, 1.0)) for p in ids.tolist()])
        out = []
        kept = np.flatnonzero(keep)
        order = kept[np.argsort(lab[kept], kind="stable")]
        cuts = np.flatnonzero(np.diff(lab[order])) + 1
        for rows in np.split(order, cuts) if len(order) else ():
            cp = (P[rows, 1:] * w[rows, None]).sum(0) / max(w[rows].sum(), 1e-9)
            k = int(cp.argmax())
            obj = {"class_id": k, "score": float(cp[k]), "members": sorted(ids[rows].tolist())}
            if class_evidence:
                obj["class_evidence"] = cp
            out.append(obj)
        return out
