"""OpenStroke19: panoptic labels on strokes beside the OpenOBB25 (E69) detector (docs/OPENSTROKE19.md).

Every stroke of the tile gets a class (9 HVAC classes + background) and an offset to the centre of its object; objects are
same-class strokes whose predicted centres are close (PointGroup-style grouping; per-primitive semantics as in SymPoint /
CADSpotting). Own code. The detector, its members head and the trunk are untouched: E69's `members` output is unchanged.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class StrokePanoptic(nn.Module):
    def __init__(self, d_tok, image_channels, num_classes, d=128):
        super().__init__()
        self.nc = int(num_classes)
        self.image = nn.Conv2d(image_channels, d, 1)
        self.mlp = nn.Sequential(nn.LayerNorm(d_tok + 2 * d + 4), nn.Linear(d_tok + 2 * d + 4, d), nn.GELU(),
                                 nn.Linear(d, d), nn.GELU())
        self.cls = nn.Linear(d, self.nc + 1)
        self.off = nn.Linear(d, 2)
        nn.init.zeros_(self.cls.bias)
        with torch.no_grad():
            self.cls.bias[-1] = 2.0                      # background first: most strokes of a tile are not HVAC
        nn.init.zeros_(self.off.weight)
        nn.init.zeros_(self.off.bias)

    def forward(self, tok, vec, mask, f4, size):
        """tok (B, M, d_tok), vec (B, M, >=17) points in tile units, f4 (B, C, H/4, W/4) -> logits (B, M, nc+1) float32,
        centres (B, M, 2) float32 in tile units."""
        b, m, _ = tok.shape
        pts = vec[..., 1:17].float().reshape(b, m, 8, 2)
        centre = pts.mean(2)
        if m:
            img = self.image(f4)
            extent = pts.new_tensor([f4.shape[-1] * 4, f4.shape[-2] * 4])
            grid = (pts * float(size) / extent * 2 - 1).reshape(b, m * 8, 1, 2).to(img.dtype)
            s = F.grid_sample(img, grid, align_corners=False).squeeze(-1).reshape(b, -1, m, 8)
            img_mean, img_max = s.mean(-1).transpose(1, 2), s.amax(-1).transpose(1, 2)
        else:
            img_mean = img_max = tok.new_zeros(b, 0, self.image.out_channels)
        ext = pts.amax(2) - pts.amin(2)
        h = self.mlp(torch.cat([tok.float(), img_mean.float(), img_max.float(), centre, ext], -1))
        logits = self.cls(h).float().masked_fill(~mask[..., None], 0.0)
        return logits, (centre + self.off(h).float() * 0.25)


def panoptic_loss(logits, centres, vec, mask, targets, bg_weight=0.25, off_weight=5.0):
    """CE over every stroke with a known label (inst != -2; inst >= 0 -> class of its object, -1 -> background) and L1 from
    each object stroke's predicted centre to its object's centre in the tile (length-weighted mean of its strokes' centres)."""
    nc = logits.shape[-1] - 1
    weight = logits.new_ones(nc + 1)
    weight[-1] = bg_weight
    ce = off = logits.sum() * 0.0
    n_ce = n_off = 0
    for b, t in enumerate(targets):
        inst = vec[b, :, 22].round().long() if vec.shape[-1] > 22 else None
        if inst is None:
            continue
        labels = t["labels"].to(logits.device).long()
        known = mask[b] & (inst != -2) & (inst < len(labels))
        if not bool(known.any()):
            continue
        target = torch.full_like(inst, nc)
        obj = known & (inst >= 0)
        target[obj] = labels[inst[obj]]
        ce = ce + F.cross_entropy(logits[b][known], target[known], weight=weight, reduction="sum")
        n_ce += int(known.sum())
        if bool(obj.any()):
            pts = vec[b, :, 1:17].float().reshape(-1, 8, 2)
            length = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1).sum(-1).clamp_min(1e-4)
            c = pts.mean(1)
            ids = inst[obj]
            k = int(ids.max()) + 1
            wsum = torch.zeros(k, device=c.device).index_add_(0, ids, length[obj])
            csum = torch.zeros(k, 2, device=c.device).index_add_(0, ids, c[obj] * length[obj, None])
            goal = csum[ids] / wsum[ids, None].clamp_min(1e-6)
            off = off + (centres[b][obj] - goal).abs().sum()
            n_off += int(obj.sum())
    return {"panoptic_cls": ce / max(n_ce, 1), "panoptic_off": off_weight * off / max(n_off, 1)}


class PanopticPage:
    """Accumulates a page's per-stroke class probabilities and predicted centres (page px) over its tiles."""

    def __init__(self, num_classes):
        self.nc = num_classes
        self.prob, self.centre, self.count = {}, {}, {}

    def add(self, pids, prob, centre):
        for p, pr, c in zip(pids.tolist(), prob, centre):
            if p in self.prob:
                self.prob[p] += pr
                self.centre[p] += c
                self.count[p] += 1
            else:
                self.prob[p], self.centre[p], self.count[p] = pr.copy(), c.copy(), 1


def panoptic_objects(page, names, radius=24.0, min_prob=0.3):
    """Objects of the page: strokes whose best non-background class has mean probability >= min_prob, grouped per class by
    connected components of predicted centres within `radius` page px. Score = mean class probability of the strokes."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    if not page.prob:
        return []
    pids = np.fromiter(page.prob.keys(), np.int64)
    cnt = np.array([page.count[p] for p in pids], np.float64)[:, None]
    prob = np.stack([page.prob[p] for p in pids]) / cnt
    centre = np.stack([page.centre[p] for p in pids]) / cnt
    cls = prob[:, :-1].argmax(1)
    conf = prob[np.arange(len(pids)), cls]
    out = []
    for c in np.unique(cls[conf >= min_prob]):
        rows = np.flatnonzero((cls == c) & (conf >= min_prob))
        pairs = cKDTree(centre[rows]).query_pairs(radius, output_type="ndarray")
        n, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs) else
                                                 (np.zeros(0), (np.zeros(0, int), np.zeros(0, int))),
                                                 shape=(len(rows), len(rows))), directed=False)
        for k in range(n):
            r = rows[lab == k]
            cx, cy = centre[r].mean(0)
            out.append({"class_id": int(c), "class": names[int(c)], "score": float(conf[r].mean()),
                        "poly": [float(cx) - 1, float(cy) - 1, float(cx) + 1, float(cy) - 1, float(cx) + 1, float(cy) + 1,
                                 float(cx) - 1, float(cy) + 1],
                        "members": sorted(int(p) for p in pids[r]), "source": "panoptic"})
    return out
