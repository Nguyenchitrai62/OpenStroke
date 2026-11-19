"""OpenStroke35: classify proposed instances and predict a member mask in each proposal."""
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch import nn
from torch.nn import functional as F


def proposals(selected, edges, positive):
    """Component id for every selected row, in ascending row order (including isolated rows)."""
    rows = np.flatnonzero(selected)
    local = np.full(len(selected), -1, np.int64)
    local[rows] = np.arange(len(rows))
    pairs = local[edges[positive]]
    a, b = pairs[(pairs >= 0).all(1)].T
    graph = coo_matrix((np.ones(len(a), np.uint8), (a, b)), shape=(len(rows), len(rows)))
    _, groups = connected_components(graph, directed=False)
    return rows, groups.astype(np.int64)


def targets(rows, groups, truth, classes, length):
    """Best visible GT by length IoU per proposal, class 0 below 0.5. IDs are local to one window."""
    count = int(groups.max()) + 1
    cls = np.zeros(count, np.int64)
    owner = np.full(count, -1, np.int64)
    valid = (truth >= 0) & (classes > 0)
    if valid.any() and valid[rows].any():
        ng = int(truth[valid].max()) + 1
        size = np.bincount(groups, weights=length[rows], minlength=count)
        gt_size = np.bincount(truth[valid], weights=length[valid], minlength=ng)
        k = valid[rows]
        pairs, inv = np.unique(groups[k] * ng + truth[rows[k]], return_inverse=True)
        area = np.bincount(inv.ravel(), weights=length[rows[k]])
        p, g = np.divmod(pairs, ng)
        iou = area / np.maximum(size[p] + gt_size[g] - area, 1e-8)
        ok = np.flatnonzero(iou >= 0.5)
        # Sort best last for each proposal, including deterministic ties at exactly half.
        for j in ok[np.argsort(iou[ok], kind="stable")]:
            owner[p[j]] = g[j]
        gt_cls = np.zeros(ng, np.int64)
        gt_cls[truth[valid]] = classes[valid]
        cls[owner >= 0] = gt_cls[owner[owner >= 0]]
    return cls, (owner[groups] >= 0) & (truth[rows] == owner[groups])


class RegionDecoder(nn.Module):
    width = 24

    def __init__(self, dim, classes):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.width))
        self.local = nn.Sequential(nn.Linear(self.width + 3, 48), nn.GELU(), nn.Linear(48, 48))
        self.context = nn.Sequential(nn.Linear(99, 64), nn.LayerNorm(64), nn.GELU())
        self.classifier = nn.Linear(64, classes)
        self.mask = nn.Sequential(nn.Linear(112, 48), nn.GELU(), nn.Linear(48, 1))

    def forward(self, features, centres, length, group, count):
        # Centre/extent per instance in units of 100 PDF points. Pooling in fp32, including under AMP.
        weight = length.clamp_min(0).add(1).sqrt()
        mass = weight.new_zeros((count,)).index_add_(0, group, weight)
        centre = centres.new_zeros((count, 2)).index_add_(0, group, centres * weight[:, None]) / mass[:, None]
        delta = centres - centre[group]
        extent = centres.new_zeros((count, 2)).scatter_reduce_(0, group[:, None].expand(-1, 2), delta.abs(), reduce="amax")
        local = self.local(torch.cat((features.float(), delta / extent[group].clamp_min(0.01),
                                      length.clamp_min(0).log1p()[:, None] / 5), dim=1)).float()
        mean = local.new_zeros((count, 48)).index_add_(0, group, local * weight[:, None]) / mass[:, None]
        peak = local.new_full((count, 48), -torch.inf).scatter_reduce_(0, group[:, None].expand_as(local), local, reduce="amax")
        ctx = self.context(torch.cat((mean, peak, torch.log1p(extent), mass.log1p()[:, None] / 5), dim=1))
        return self.classifier(ctx), self.mask(torch.cat((local, ctx[group]), dim=1)).squeeze(-1)

    def training_loss(self, out, batch):
        z = out["region_features"]
        B, P, _ = z.shape
        fg = ((out["fg"].detach() >= 0) & batch["pc_mask"]).cpu().numpy()
        edges = torch.stack((out["i"], out["j"]), dim=1).detach().cpu().numpy()
        positive = (out["link"].detach() >= 0).cpu().numpy()
        truth = batch["pc_inst"].detach().cpu().numpy()
        classes = batch["pc_cls"].detach().cpu().numpy()
        lengths = batch["pc_len"].detach().float().cpu().numpy()
        losses = []
        for i in range(B):
            in_window = edges[:, 0] // P == i
            rows, groups = proposals(fg[i], edges[in_window] - i * P, positive[in_window])
            # GT groups provide positive examples from epoch 1. Never used at validation or inference.
            teacher = np.flatnonzero((truth[i] >= 0) & (classes[i] > 0))
            examples = [(rows, groups)]
            if len(teacher):
                examples.append((teacher, np.unique(truth[i, teacher], return_inverse=True)[1].ravel()))
            for ids, group in examples:
                if not len(ids):
                    continue
                target, members = targets(ids, group, truth[i], classes[i], lengths[i])
                idx = torch.as_tensor(ids, device=z.device)
                inv = torch.as_tensor(group, device=z.device)
                c, m = self(z[i, idx], batch["pc_c"][i, idx].float(), batch["pc_len"][i, idx].float(), inv, len(target))
                y = torch.as_tensor(target, device=z.device)
                ce = F.cross_entropy(c.float(), y, reduction="none")
                # Balance proposal positives and negatives; all background still teaches rejection.
                loss = sum((ce * mask).sum() / mask.sum().clamp_min(1) for mask in (y > 0, y == 0))
                mask_loss = F.binary_cross_entropy_with_logits(m.float(), torch.as_tensor(members, device=z.device).float(), reduction="none")
                valid = y[inv] > 0
                loss = loss + (mask_loss * valid).sum() / valid.sum().clamp_min(1)
                losses.append(loss)
        return torch.stack(losses).mean() if losses else z.sum() * 0

    @torch.no_grad()
    def decode_page(self, page, q, features):
        pid = q["pid"]
        result = {"region_id": np.full(len(pid), -1, np.int64), "region_cls": np.zeros(len(pid), np.int64)}
        if not len(pid):
            return result
        # merge_pieces emits sorted original piece ids and links in that same id space.
        rows, groups = proposals(q["fg"] >= 0.5, np.searchsorted(pid, q["pe"]), q["plink"] >= 0)
        if not len(rows):
            return result
        pc = page.pieces
        order = np.argsort(pc["pid"])
        at = order[np.searchsorted(pc["pid"][order], pid[rows])]
        device = next(self.parameters()).device
        def tensor(a):
            return torch.as_tensor(a, device=device)
        c, m = self(tensor(features[rows]).float(), tensor(pc["ppts"][at].mean(1) / 100).float(),
                    tensor(pc["plen"][at]).float(), tensor(groups), int(groups.max()) + 1)
        names = c.argmax(1).cpu().numpy()[groups]
        keep = (names > 0) & (m.cpu().numpy() >= 0)
        result["region_id"][rows[keep]] = groups[keep]
        result["region_cls"][rows[keep]] = names[keep]
        return result
