"""OpenStroke41 (H162): End-to-end object-level 6-class + FP-rejection head (GroupClsHead) on top of OpenStroke31.

SOTA grounding (AGENTS.md §0.10):
  - PointGroup (Jiang et al., CVPR 2020) & SoftGroup (Vu et al., CVPR 2022): dual-stage bottom-up instance segmentation
    where foreground elements are grouped by learned affinity, then a group-level ScoreNet/Semantic head pools features
    (mean + max) + geometric descriptors over each cluster to jointly name the object class AND reject false-positive
    (over-merged / fragment / background) clusters.
  - GAT-CADNet (Zheng et al., CVPR 2022): primitive graph attention + edge affinity for CAD instance grouping.

Unlike OpenStroke29 (per-stroke classification before GraphAttention) and OpenStroke34/35 (3.85x piece tokens on 19 GT
pages), GroupClsHead adds ~50K parameters (~0.3% overhead) directly onto the proven OpenStroke31 backbone (17.2M params,
0.948 stroke F2 / 0.817-0.805 PureObj(mx)) and trains on ALL 188 clean 6-class pages of e13_v3 (75,000+ objects).
Works on BOTH closed ducts AND open-ended branches (no closed planar polygon required).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features import FEATURE_NAMES
from ..objects import HVAC_CLASSES

#: Canonical 8-class object taxonomy matching openstroke.hvac_faces.CLASSES
OBJ8_CLASSES = (
    "none",
    "Duct",
    "Elbow",
    "Elbow vuông",
    "Concentric Reducer",
    "Eccentric Reducer",
    "Chân rẽ",
    "Tee",
)
N_OBJ8 = len(OBJ8_CLASSES)

#: Map openstroke.objects.HVAC_CLASSES (0..9) -> OBJ8_CLASSES (0..7)
HVAC_TO_OBJ8 = np.array([
    0,  # 0: other -> none
    1,  # 1: Duct -> Duct
    2,  # 2: Elbow -> Elbow
    3,  # 3: Elbow vuông -> Elbow vuông
    4,  # 4: Concentric Reducer -> Concentric Reducer
    5,  # 5: Eccentric Reducer -> Eccentric Reducer
    4,  # 6: Reducer vuông tròn -> Concentric Reducer (resolved as either 4 or 5 during scoring)
    6,  # 7: Chân rẽ -> Chân rẽ
    6,  # 8: Chân rẽ vuông tròn -> Chân rẽ
    7,  # 9: Tee -> Tee
], dtype=np.int64)

_IDX_COS2 = FEATURE_NAMES.index("dir_cos2")
_IDX_SIN2 = FEATURE_NAMES.index("dir_sin2")
_IDX_FRAC_C = FEATURE_NAMES.index("frac_c")
_IDX_CLOSED = FEATURE_NAMES.index("closed_ends")
_IDX_LOG_W = FEATURE_NAMES.index("log_bbox_w")
_IDX_LOG_H = FEATURE_NAMES.index("log_bbox_h")

N_GEOM = 13


def hvac_to_obj8_tensor(ocls: torch.Tensor) -> torch.Tensor:
    lut = torch.as_tensor(HVAC_TO_OBJ8, device=ocls.device, dtype=torch.long)
    return lut[ocls.clamp(0, len(HVAC_TO_OBJ8) - 1)]


def cluster_descriptors(
    h: torch.Tensor,
    xy: torch.Tensor,
    length: torch.Tensor,
    x_raw: torch.Tensor,
    prob: torch.Tensor,
    cid: torch.Tensor,
    n_clusters: int,
    ext_touch: torch.Tensor | None = None,
    int_eprob: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized (mean_h, max_h, 13 rotation-invariant cluster geometry features) -> (K, 2*C + 13).

    ``cid``: (M,) cluster index in 0..K-1 for each active stroke.
    """
    dev, dt = h.device, torch.float32
    hf = h.float()
    K, C = n_clusters, hf.shape[1]
    if K == 0 or len(cid) == 0:
        return torch.zeros((0, 2 * C + N_GEOM), device=dev, dtype=h.dtype)

    L = length.float().clamp(min=1e-3)
    w_sum = torch.zeros(K, device=dev, dtype=dt).index_add_(0, cid, L).clamp(min=1e-3)
    cnt = torch.zeros(K, device=dev, dtype=dt).index_add_(0, cid, torch.ones_like(L)).clamp(min=1.0)

    # Length-weighted mean and element-wise max of token features h
    h_mean = torch.zeros((K, C), device=dev, dtype=dt).index_add_(0, cid, hf * L[:, None]) / w_sum[:, None]
    h_max = torch.full((K, C), -1e4, device=dev, dtype=dt).scatter_reduce(
        0, cid[:, None].expand(-1, C), hf, reduce="amax", include_self=True
    )

    # Rotation-invariant 2D covariance eigenvalues of stroke centres + per-stroke bbox size
    xyf = xy.float()
    c_xy = torch.zeros((K, 2), device=dev, dtype=dt).index_add_(0, cid, xyf * L[:, None]) / w_sum[:, None]
    dxy = xyf - c_xy[cid]
    # Add individual stroke half-span variance so 1-stroke clusters still have accurate size
    sw = torch.expm1(x_raw[:, _IDX_LOG_W].float().clamp(min=0, max=10))
    sh = torch.expm1(x_raw[:, _IDX_LOG_H].float().clamp(min=0, max=10))
    s_var = (sw * sw + sh * sh) / 24.0
    vxx = torch.zeros(K, device=dev, dtype=dt).index_add_(0, cid, (dxy[:, 0] ** 2 + s_var) * L) / w_sum
    vyy = torch.zeros(K, device=dev, dtype=dt).index_add_(0, cid, (dxy[:, 1] ** 2 + s_var) * L) / w_sum
    vxy = torch.zeros(K, device=dev, dtype=dt).index_add_(0, cid, (dxy[:, 0] * dxy[:, 1]) * L) / w_sum
    tr = vxx + vyy
    disc = torch.sqrt(((vxx - vyy) ** 2 + 4.0 * vxy * vxy).clamp(min=1e-8))
    lam1 = (0.5 * (tr + disc)).clamp(min=0.0)
    lam2 = (0.5 * (tr - disc)).clamp(min=0.0)
    long_span = 2.0 * torch.sqrt(lam1)
    short_span = 2.0 * torch.sqrt(lam2)
    aspect = ((long_span + 1.0) / (short_span + 1.0)).clamp(1.0, 30.0)

    # Rotation-invariant direction coherence, curve fraction, closed-ends fraction
    cos2 = torch.zeros(K, device=dev, dtype=dt).index_add_(0, cid, x_raw[:, _IDX_COS2].float() * L) / w_sum
    sin2 = torch.zeros(K, device=dev, dtype=dt).index_add_(0, cid, x_raw[:, _IDX_SIN2].float() * L) / w_sum
    dir_coh = torch.sqrt((cos2 * cos2 + sin2 * sin2).clamp(min=1e-8))
    curved_sh = torch.zeros(K, device=dev, dtype=dt).index_add_(0, cid, x_raw[:, _IDX_FRAC_C].float() * L) / w_sum
    closed_sh = torch.zeros(K, device=dev, dtype=dt).index_add_(0, cid, x_raw[:, _IDX_CLOSED].float() * L) / w_sum

    # Foreground confidence statistics (detached)
    pf = prob.detach().float()
    p_mean = torch.zeros(K, device=dev, dtype=dt).index_add_(0, cid, pf * L) / w_sum
    p_min = torch.full((K,), 1.0, device=dev, dtype=dt).scatter_reduce(0, cid, pf, reduce="amin", include_self=True)

    # Topology signals (external connections to other foreground clusters + internal link strength)
    ext_r = torch.log1p(ext_touch.float()) / 3.0 if ext_touch is not None else torch.zeros(K, device=dev, dtype=dt)
    int_p = int_eprob.float() if int_eprob is not None else torch.full((K,), 0.5, device=dev, dtype=dt)

    g = torch.stack([
        torch.log1p(w_sum) / 5.0,
        torch.log1p(cnt) / 3.0,
        torch.log1p(long_span) / 5.0,
        torch.log1p(short_span) / 5.0,
        torch.log1p(aspect) / 3.5,
        torch.log1p(w_sum / cnt) / 5.0,
        curved_sh.clamp(0.0, 1.0),
        closed_sh.clamp(0.0, 1.0),
        dir_coh.clamp(0.0, 1.0),
        p_mean,
        p_min,
        ext_r.clamp(0.0, 3.0),
        int_p.clamp(0.0, 1.0),
    ], dim=-1)
    return torch.cat([h_mean, h_max, g], dim=-1).to(h.dtype)


def _edge_cluster_stats(
    cid_full: torch.Tensor,
    n_clusters: int,
    ei: torch.Tensor,
    ej: torch.Tensor,
    ep: torch.Tensor,
    fg_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per cluster: (external foreground touch count, mean internal edge probability)."""
    dev = cid_full.device
    ext = torch.zeros(n_clusters, device=dev, dtype=torch.float32)
    int_s = torch.zeros(n_clusters, device=dev, dtype=torch.float32)
    int_c = torch.zeros(n_clusters, device=dev, dtype=torch.float32)
    if len(ei) and n_clusters > 0:
        ca, cb = cid_full[ei], cid_full[ej]
        both_fg = fg_mask[ei] & fg_mask[ej]
        same = both_fg & (ca >= 0) & (ca == cb)
        if bool(same.any()):
            int_s.index_add_(0, ca[same], ep[same].float())
            int_c.index_add_(0, ca[same], torch.ones_like(ep[same], dtype=torch.float32))
        diff = both_fg & (ca != cb)
        da = diff & (ca >= 0)
        db = diff & (cb >= 0)
        if bool(da.any()):
            ext.index_add_(0, ca[da], torch.ones(int(da.sum()), device=dev, dtype=torch.float32))
        if bool(db.any()):
            ext.index_add_(0, cb[db], torch.ones(int(db.sum()), device=dev, dtype=torch.float32))
    int_mean = torch.where(int_c > 0, int_s / int_c.clamp(min=1.0), torch.full_like(int_s, 0.5))
    return ext, int_mean


class GroupClsHead(nn.Module):
    """Object-level 8-class + FP-rejection head pooled over grouped stroke clusters, plus lightweight auxiliary per-stroke head."""

    def __init__(self, dim: int, n_classes: int = N_OBJ8, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_classes = n_classes
        in_dim = 2 * dim + N_GEOM
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        self.stroke_cls = nn.Linear(dim, n_classes)

    def forward_strokes(self, h: torch.Tensor) -> torch.Tensor:
        return self.stroke_cls(h)

    def forward_clusters(
        self,
        h: torch.Tensor,
        xy: torch.Tensor,
        length: torch.Tensor,
        x_raw: torch.Tensor,
        prob: torch.Tensor,
        cid: torch.Tensor,
        n_clusters: int,
        ext_touch: torch.Tensor | None = None,
        int_eprob: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feat = cluster_descriptors(h, xy, length, x_raw, prob, cid, n_clusters, ext_touch, int_eprob)
        return self.mlp(feat)

    @torch.no_grad()
    def classify_page_clusters(
        self,
        h_page: np.ndarray,
        page,
        prob: np.ndarray,
        thr: float,
        edges: tuple[np.ndarray, np.ndarray] | None,
        inst: np.ndarray,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Classify full-page H142 mutex clusters -> (cluster_ids, probs (K, 8)) on GPU/CPU without window cuts."""
        picked = prob >= thr
        active = np.flatnonzero(picked & (inst >= 0))
        if not len(active):
            return np.zeros(0, np.int64), np.zeros((0, self.n_classes), np.float32)
        u_ids, local_cid = np.unique(inst[active], return_inverse=True)
        K = len(u_ids)
        full_cid = np.full(len(prob), -1, np.int64)
        full_cid[active] = local_cid

        h_t = torch.as_tensor(h_page[active], device=device, dtype=torch.float32)
        xy_t = torch.as_tensor(page.xy[active], device=device, dtype=torch.float32)
        len_t = torch.as_tensor(page.length[active], device=device, dtype=torch.float32)
        xr_t = torch.as_tensor(page.x[active], device=device, dtype=torch.float32)
        pr_t = torch.as_tensor(prob[active], device=device, dtype=torch.float32)
        cid_t = torch.as_tensor(local_cid, device=device, dtype=torch.long)

        ext_t = int_t = None
        if edges is not None and len(edges[0]):
            e, lg = edges
            e = np.asarray(e, np.int64).reshape(-1, 2)
            ep = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(lg, np.float32), -20.0, 20.0)))
            fc_t = torch.as_tensor(full_cid, device=device, dtype=torch.long)
            fg_t = torch.as_tensor(picked, device=device, dtype=torch.bool)
            ext_t, int_t = _edge_cluster_stats(
                fc_t, K,
                torch.as_tensor(e[:, 0], device=device, dtype=torch.long),
                torch.as_tensor(e[:, 1], device=device, dtype=torch.long),
                torch.as_tensor(ep, device=device, dtype=torch.float32),
                fg_t,
            )
        logits = self.forward_clusters(h_t, xy_t, len_t, xr_t, pr_t, cid_t, K, ext_t, int_t)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float32)
        return u_ids.astype(np.int64), probs


def group_class_weights(pages) -> list[float]:
    """Inverse-sqrt class weights for OBJ8 (0..7) computed on TRAIN pages only, clamped [0.5, 4.0]."""
    tot = np.zeros(N_OBJ8, dtype=np.float64)
    for p in pages:
        if getattr(p, "ocls", None) is None or getattr(p, "oinst", None) is None:
            continue
        m = (p.label > 0) & (p.oinst >= 0) & (p.ocls > 0)
        if not m.any():
            continue
        c8 = HVAC_TO_OBJ8[np.clip(p.ocls[m], 0, len(HVAC_TO_OBJ8) - 1)]
        tot += np.bincount(c8, weights=p.length[m], minlength=N_OBJ8)
    w = np.ones(N_OBJ8, dtype=np.float64)
    fg = tot[1:] > 0
    if fg.any():
        inv = np.where(tot[1:] > 0, 1.0 / np.sqrt(np.maximum(tot[1:], 1.0)), 0.0)
        inv[fg] = inv[fg] / inv[fg].mean()
        w[1:] = np.clip(inv, 0.5, 4.0)
    w[0] = 1.0   # none (FP rejection class)
    return w.astype(np.float32).tolist()


def group_cls_loss(
    head: GroupClsHead,
    h: torch.Tensor,
    logit: torch.Tensor,
    last_edge: tuple | None,
    b: dict,
    class_weight: torch.Tensor,
    alpha: float = 0.5,
    max_pred_clusters: int = 384,
) -> torch.Tensor:
    """Joint object-level classification + online predicted-cluster FP rejection + auxiliary per-stroke CE loss."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if "oinst" not in b or "ocls" not in b:
        return logit.float().sum() * 0.0

    B, N, C = h.shape
    dev = h.device
    hf = h.reshape(B * N, C)
    xyf = b["xy0"].reshape(B * N, 2)
    lenf = b["length"].reshape(B * N)
    xrf = b["x"].reshape(B * N, -1)
    maskf = b["mask0"].reshape(B * N)
    y_fg = (b["label"].reshape(B * N) > 0) & maskf
    o_cls8 = hvac_to_obj8_tensor(b["ocls"].reshape(B * N))
    oinst = b["oinst"].reshape(B * N)
    prob = torch.sigmoid(logit.detach().float()).reshape(B * N)

    # Unique flat GT object id across batch
    b_off = (torch.arange(B, device=dev, dtype=torch.long) * (int(oinst.max().clamp(min=0)) + 2))[:, None].expand(B, N).reshape(-1)
    gt_valid = y_fg & (oinst >= 0) & (o_cls8 > 0)
    if not bool(gt_valid.any()):
        # Batch has no 6-class HVAC object labels (e.g. background-only window or FIRE/PLUMBING): skip cluster FP rejection too.
        return logit.float().sum() * 0.0
    gt_flat = torch.where(gt_valid, oinst.long() + 1 + b_off, torch.full_like(b_off, -1))

    ei = ej = ep = None
    if last_edge is not None:
        ez, ei, ej = last_edge
        ep = torch.sigmoid(ez.detach().float())

    losses = []

    # 1. Auxiliary per-stroke 8-class CE on foreground strokes (length-weighted)
    if bool(gt_valid.any()):
        s_logits = head.forward_strokes(hf[gt_valid]).float()
        w_s = (lenf[gt_valid].float().clamp(min=0) + 1.0) ** alpha
        ce_s = F.cross_entropy(s_logits, o_cls8[gt_valid], weight=class_weight, reduction="none")
        losses.append(0.3 * ((ce_s * w_s).sum() / w_s.sum().clamp(min=1e-6)))

    # 2. GT object cluster loss (all GT objects in batch -> classes 1..7)
    if bool(gt_valid.any()):
        idx = torch.nonzero(gt_valid, as_tuple=False).squeeze(1)
        u_gt, cid = torch.unique(gt_flat[idx], return_inverse=True)
        K_gt = len(u_gt)
        # Majority-length class of each GT cluster
        gt_y = torch.zeros(K_gt, device=dev, dtype=torch.long)
        gt_y.scatter_(0, cid, o_cls8[idx])   # all strokes of one GT object share one class
        ext_t = int_t = None
        if ei is not None:
            full_cid = torch.full((B * N,), -1, device=dev, dtype=torch.long)
            full_cid[idx] = cid
            ext_t, int_t = _edge_cluster_stats(full_cid, K_gt, ei, ej, ep, y_fg)
        g_logits = head.forward_clusters(hf[idx], xyf[idx], lenf[idx], xrf[idx], prob[idx], cid, K_gt, ext_t, int_t).float()
        losses.append(F.cross_entropy(g_logits, gt_y, weight=class_weight))

    # 3. Online predicted-cluster loss (ScoreNet / SoftGroup: teaches BOTH naming noisy clusters AND rejecting FP as class 0=none!)
    pred_fg = (prob >= 0.5) & maskf
    if bool(pred_fg.any()):
        p_idx_np = torch.nonzero(pred_fg, as_tuple=False).squeeze(1).cpu().numpy()
        local_map = np.full(B * N, -1, np.int64)
        local_map[p_idx_np] = np.arange(len(p_idx_np))
        if ei is not None and len(ei):
            ea = local_map[ei.cpu().numpy()]
            eb = local_map[ej.cpu().numpy()]
            ok = (ea >= 0) & (eb >= 0) & (ep.cpu().numpy() >= 0.5)
            ea, eb = ea[ok], eb[ok]
        else:
            ea = eb = np.zeros(0, np.int64)
        K_pr, pr_lab = connected_components(
            coo_matrix((np.ones(len(ea), np.float32), (ea, eb)), shape=(len(p_idx_np), len(p_idx_np))),
            directed=False,
        )
        # Subsample clusters if batch has very many tiny fragments (keeps step fast)
        if K_pr > max_pred_clusters:
            rng_idx = np.random.choice(K_pr, size=max_pred_clusters, replace=False)
            keep_c = np.full(K_pr, -1, np.int64)
            keep_c[rng_idx] = np.arange(max_pred_clusters)
            pr_lab = keep_c[pr_lab]
            keep_m = pr_lab >= 0
            p_idx_np = p_idx_np[keep_m]
            pr_lab = pr_lab[keep_m]
            K_pr = max_pred_clusters

        if K_pr > 0 and len(p_idx_np) > 0:
            # Compute length IoU of each predicted cluster vs GT objects in batch
            L_np = lenf.detach().cpu().numpy().astype(np.float64)
            gt_np = gt_flat.detach().cpu().numpy()
            oc_np = o_cls8.detach().cpu().numpy()
            # Total length of each GT object in batch
            gt_pos = gt_np >= 0
            if gt_pos.any():
                ug, g_inv = np.unique(gt_np[gt_pos], return_inverse=True)
                gt_tot = np.bincount(g_inv, weights=L_np[gt_pos])
                gt_cls_map = np.zeros(len(ug), np.int64)
                gt_cls_map[g_inv] = oc_np[gt_pos]
                gt_local = np.full(B * N, -1, np.int64)
                gt_local[gt_pos] = g_inv
            else:
                gt_tot = np.zeros(0)
                gt_cls_map = np.zeros(0, np.int64)
                gt_local = np.full(B * N, -1, np.int64)

            pr_tot = np.bincount(pr_lab, weights=L_np[p_idx_np], minlength=K_pr)
            p_gt = gt_local[p_idx_np]
            both = p_gt >= 0
            best_iou = np.zeros(K_pr, np.float64)
            best_cls = np.zeros(K_pr, np.int64)
            if both.any() and len(gt_tot) > 0:
                ng = len(gt_tot)
                pairs, inv = np.unique(pr_lab[both] * ng + p_gt[both], return_inverse=True)
                inter = np.bincount(inv, weights=L_np[p_idx_np[both]])
                pi, gi = np.divmod(pairs, ng)
                ious = inter / np.maximum(pr_tot[pi] + gt_tot[gi] - inter, 1e-9)
                for p_id, g_id, v in zip(pi, gi, ious):
                    if v > best_iou[p_id]:
                        best_iou[p_id] = v
                        best_cls[p_id] = gt_cls_map[g_id]
            target_np = np.full(K_pr, -100, np.int64)
            target_np[best_iou >= 0.50] = best_cls[best_iou >= 0.50]
            target_np[best_iou < 0.35] = 0   # none = false positive / over-merged / fragment
            if (target_np >= 0).any():
                p_idx_t = torch.as_tensor(p_idx_np, device=dev, dtype=torch.long)
                pr_cid_t = torch.as_tensor(pr_lab, device=dev, dtype=torch.long)
                ext_t = int_t = None
                if ei is not None:
                    full_cid = torch.full((B * N,), -1, device=dev, dtype=torch.long)
                    full_cid[p_idx_t] = pr_cid_t
                    ext_t, int_t = _edge_cluster_stats(full_cid, K_pr, ei, ej, ep, pred_fg)
                p_logits = head.forward_clusters(
                    hf[p_idx_t], xyf[p_idx_t], lenf[p_idx_t], xrf[p_idx_t], prob[p_idx_t], pr_cid_t, K_pr, ext_t, int_t
                ).float()
                tgt_t = torch.as_tensor(target_np, device=dev, dtype=torch.long)
                losses.append(0.5 * F.cross_entropy(p_logits, tgt_t, weight=class_weight, ignore_index=-100))

    if not losses:
        return logit.float().sum() * 0.0
    return sum(losses)
