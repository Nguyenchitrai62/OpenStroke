"""OpenOBB25: OpenOBB24 + CAD vector primitives as input, and for every detected box the primitives that make it.

Goal (user, HVAC take-off): the detector's output must turn straight into the final take-off JSON, which lists for
every object (Duct, Elbow, Reducer, Chan re, Tee...) the drawing commands it is made of. A box alone is not enough:
the pipeline then re-finds the strokes inside it by rules (layer, overlap), and that is where it fails on drawings it
has not seen (docs/OPENOBB25.md).

Mechanism (docs/OPENOBB25.md):
  * vector input: every primitive of the tile is a token (type, 8 points resampled along it, stroke rgb, width; the CAD
    layer is NOT used: the model must not depend on PDF layers). OpenOBB1's H7 VectorBranch (Fourier point features,
    2-layer transformer over the tile's tokens) splats each token's feature onto the P3 / P4 / P5 cells its stroke
    crosses and adds it to the neck maps through zero-initialised 1x1 convs: at init the boxes are OpenOBB24's;
  * membership head: for a box b (class c, rotated rectangle) and a primitive m,
        logit(b, m) = bias + w . g(b, m) + <W_q h_b + E_c, W_t t_m> / sqrt(d)
    h_b = the edge refiner's last hidden layer at b (stride-4 trunk + P3 samples on b's edges, detached), E_c a class
    embedding, t_m the token feature, g four geometric relations computed exactly (fraction of the 8 points inside b
    + 2 px, inside b + a loose margin, on b's outline, size of m against b's thickness). Primitives with no point in
    the loose region get no membership (logit -30). The prior w starts as "all points inside the box"
    (members_from_boxes); the content term learns which of those strokes belong to the object (outline, centre line,
    fitting marks) and which do not (text, hatch, other systems crossing it);
  * training: the refined candidates (top ref_k_train + jittered labelled boxes) are matched to the labelled objects by
    rotated IoU >= 0.5 (best object); at most member_k (256) per image, spread round-robin over the objects; target =
    the primitives whose `inst` (object id carried by the data) is that object (members_from_boxes when the data has
    no ids). Loss: BCE (mean over the gated primitives) + dice, x member_weight (`vec_ground`);
  * inference: the NMS keeps candidate indices (fast_nms return_index); the membership row of every kept box is
    computed on its final box (the same box kind as in training): out["member_logits"] (B, max_det, M).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from openstroke.obb.data.vectors import INST_UNKNOWN
from openstroke.obb.ops.obb_torch import rotated_iou

from .endpoint_members import EndpointMembers
from .stroke_graph import StrokeGraphHead, graph_loss, graph_targets
from .openobb18_loss import OpenOBB18Loss
from .openobb24 import OpenOBB24
from .vector import VectorBranch, members_from_boxes

MEMBER_K = 256           # matched candidates per image that get the membership loss
MEMBER_IOU = 0.5         # rotated IoU of a candidate to its labelled object for it to learn that object's primitives
TIGHT_PX = 2.0           # g0: a point is inside the box expanded by this (px)
LOOSE_PX, LOOSE_T = 8.0, 0.25   # g1 / gate: loose margin = LOOSE_PX + LOOSE_T x thickness (px)
EDGE_SIGMA = 3.0         # g2: Gaussian width (px) of "on the box outline"
N_PAIR = 4
PAIR_CHUNK = 1 << 22     # box x primitive x point elements per chunk of the pair geometry


@torch.no_grad()
def pair_geometry(boxes, pts):
    """boxes (K, 5) px (cx, cy, w, h, theta: w along theta), pts (M, 8, 2) px -> g (K, M, N_PAIR) float32 and
    gate (K, M) bool (at least one point inside the loose region)."""
    K, M = boxes.shape[0], pts.shape[0]
    g = boxes.new_zeros((K, M, N_PAIR), dtype=torch.float32)
    gate = torch.zeros((K, M), dtype=torch.bool, device=boxes.device)
    if K == 0 or M == 0:
        return g, gate
    b = boxes.float()
    p = pts.float()
    ext = (p.amax(1) - p.amin(1)).norm(dim=-1)                                  # (M,) size of each primitive
    step = max(1, PAIR_CHUNK // max(M * 8, 1))
    for k0 in range(0, K, step):
        bb = b[k0:k0 + step]
        c, s = bb[:, 4].cos(), bb[:, 4].sin()
        hw, hh = bb[:, 2].abs() / 2, bb[:, 3].abs() / 2
        dx = p[None, :, :, 0] - bb[:, 0, None, None]
        dy = p[None, :, :, 1] - bb[:, 1, None, None]
        u = (dx * c[:, None, None] + dy * s[:, None, None]).abs()
        v = (-dx * s[:, None, None] + dy * c[:, None, None]).abs()
        hw3, hh3 = hw[:, None, None], hh[:, None, None]
        tight = ((u <= hw3 + TIGHT_PX) & (v <= hh3 + TIGHT_PX)).float()
        m = (LOOSE_PX + LOOSE_T * torch.minimum(hw, hh) * 2)[:, None, None]
        loose = (u <= hw3 + m) & (v <= hh3 + m)
        ou, ov = (u - hw3).clamp(min=0), (v - hh3).clamp(min=0)
        inside = (ou == 0) & (ov == 0)
        d = torch.where(inside, torch.minimum(hw3 - u, hh3 - v), torch.sqrt(ou ** 2 + ov ** 2))
        edge = torch.exp(-(d ** 2) / (2 * EDGE_SIGMA ** 2))
        thick = (2 * torch.minimum(hw, hh)).clamp(min=1.0)
        rel = (torch.log(ext.clamp(min=0.5)[None, :] / thick[:, None]).clamp(-3, 3) / 3)
        g[k0:k0 + step] = torch.stack([tight.mean(-1), loose.float().mean(-1), edge.mean(-1), rel], -1)
        gate[k0:k0 + step] = loose.any(-1)
    return g, gate


class MemberHead25(nn.Module):
    """Membership logits of boxes against the primitives of their tile (module docstring)."""

    def __init__(self, d_box, d_tok, num_classes, d=128, endpoint_channels=None):
        super().__init__()
        self.d_box = d_box
        self.q = nn.Linear(d_box, d)
        self.cls = nn.Embedding(max(num_classes, 1), d)
        self.t = nn.Linear(d_tok, d)
        self.scale = d ** -0.5
        self.endpoints = None if endpoint_channels is None else EndpointMembers(endpoint_channels, d)
        nn.init.normal_(self.cls.weight, std=0.02)
        # prior = members_from_boxes: all 8 points inside the box -> logit +2, none -> -4
        self.geo_w = nn.Parameter(torch.tensor([6.0, 0.0, 0.0, 0.0]))
        self.bias = nn.Parameter(torch.tensor(-4.0))

    def forward(self, h, labels, boxes, tok, pts, f4=None):
        """h (K, d_box), labels (K,), boxes (K, 5) px, tok (M, d_tok), pts (M, 8, 2) px -> logits (K, M) float32,
        gate (K, M) bool."""
        g, gate = pair_geometry(boxes, pts)
        q = self.q(h.float()) + self.cls(labels.long().clamp(0, self.cls.num_embeddings - 1))
        content = q @ self.t(tok.float()).t() * self.scale
        logit = self.bias + (g * self.geo_w).sum(-1) + content
        if self.endpoints is not None and logit.numel():
            logit = self.endpoints(logit, q, boxes, pts, gate, f4)
        return logit.masked_fill(~gate, -30.0), gate


@torch.no_grad()
def near_iou(a, b):
    """Exact rotated IoU (K, N) of boxes a (K, 5) and b (N, 5) px, computed only for pairs whose centres are closer
    than the sum of their half-diagonals (others cannot overlap: 0)."""
    out = a.new_zeros((a.shape[0], b.shape[0]))
    if out.numel() == 0:
        return out
    ra = 0.5 * torch.sqrt(a[:, 2] ** 2 + a[:, 3] ** 2)
    rb = 0.5 * torch.sqrt(b[:, 2] ** 2 + b[:, 3] ** 2)
    near = torch.cdist(a[None, :, :2], b[None, :, :2])[0] <= ra[:, None] + rb[None, :]
    i, j = near.nonzero(as_tuple=True)
    if len(i):
        out[i, j] = rotated_iou(a[i], b[j]).to(out.dtype)
    return out


def balanced_pick(best, gt_idx, k):
    """Indices of at most k candidates among those with best IoU >= MEMBER_IOU, round-robin over the objects (each
    object's best-matching candidates first)."""
    ii = (best >= MEMBER_IOU).nonzero(as_tuple=True)[0]
    if len(ii) == 0:
        return ii
    ii = ii[torch.argsort(-best[ii])]
    jj = gt_idx[ii]
    o2 = torch.argsort(jj, stable=True)
    js = jj[o2]
    first = torch.searchsorted(js, js)
    rank = torch.empty_like(jj)
    rank[o2] = torch.arange(len(js), device=js.device) - first
    key = rank.float() * 2.0 - best[ii].float()
    return ii[torch.argsort(key)[:k]]


class OpenOBB25(OpenOBB24):
    """OpenOBB24 + vector tokens fused into P3-P5 + per-box primitive membership. docs/OPENOBB25.md."""
    arch = "v25"

    def __init__(self, *args, vec_dim=128, vec_layers=2, member_d=128, member_k=MEMBER_K,
                 endpoint_members=False, stroke_graph=False, **kw):
        super().__init__(*args, **kw)
        self.vec = VectorBranch([self.hid] * 3, d=vec_dim, layers=vec_layers, lfe=False)
        self.d_box = self.edge.mlp[-1].in_features if getattr(self, "edge", None) is not None else 1
        endpoint_channels = self.backbone._out_channels[0] if endpoint_members else None
        self.member = MemberHead25(self.d_box, vec_dim, self.nc, d=member_d,
                                   endpoint_channels=endpoint_channels)
        self.member_k = member_k
        # OpenStroke10: class of every primitive + same-object edges; objects = connected components (docs/OPENSTROKE10.md)
        self.graph = StrokeGraphHead(vec_dim, self.nc, self.backbone._out_channels[0]) if stroke_graph else None
        self.member_thr = 0.5            # probability for a primitive to be listed as a member at inference
        self.max_tokens = 8192           # tile cap used by the data / inference (longest primitives kept)

    # ---- vector input
    def encode_inputs(self, x, ctx=None):
        f4, feats = self.encode(x)
        if ctx is None or ctx.get("vec") is None or self.vec is None:
            return f4, feats, {}
        vec, mask = ctx["vec"].float(), ctx["vec_mask"]
        vin = vec[..., :22].clone()
        vin[..., 21] = 0                 # CAD layers are not used (unknown layer for every primitive)
        feats, tok = self.vec(list(feats), vin, mask, return_tokens=True)
        return f4, list(feats), {"tok": tok, "vec": vec, "mask": mask}

    # ---- box features for the membership head
    def box_hidden(self, f4, p3, boxes):
        """Refiner hidden layer at boxes (B, K, 5) px (detached; zeros without a refiner)."""
        if getattr(self, "edge", None) is None:
            return boxes.new_zeros((*boxes.shape[:2], 1))
        with torch.no_grad():
            return self.edge.refine_with_fit(f4, p3, boxes.float(), hidden=True)[2].float()

    def _members_of(self, h, labels, boxes, tok, vec, mask, size, f4):
        """One image: h (K, d), labels (K,), boxes (K, 5) px -> logits (K, M_valid), gate, token rows used."""
        rows = mask.nonzero(as_tuple=True)[0]
        pts = vec[rows, 1:17].float().view(-1, 8, 2) * size
        lg, gate = self.member(h, labels, boxes, tok[rows], pts, f4)
        return lg, gate, rows

    # ---- training
    def member_targets(self, out, targets, size):
        """-> list per image of (selected final boxes px, matched object index) for the membership loss."""
        final = out["ref_stages"][-1][1] if out.get("ref_stages") else out["ref_boxes"]
        valid = out["ref_valid"]
        picks = []
        for b, t in enumerate(targets):
            gt = t["boxes"].float().to(final.device).clone()
            if len(gt) == 0:
                picks.append(None)
                continue
            gt[:, :4] *= size
            fb = final[b].detach().float()
            ok = valid[b].nonzero(as_tuple=True)[0]
            iou = near_iou(fb[ok], gt)                                # (K, N)
            best, j = iou.max(1)
            sel = balanced_pick(best, j, self.member_k)
            picks.append((fb[ok][sel], j[sel]) if len(sel) else None)
        return picks

    def member_loss(self, out, targets, size):
        side = out["side"]
        tok, vec, mask = side["tok"], side["vec"], side["mask"]
        picks = self.member_targets(out, targets, size)
        bce = dice = tok.sum() * 0.0
        n = 0
        cand = [(b, p) for b, p in enumerate(picks) if p is not None and bool(mask[b].any())]
        if not cand:
            return {"member_bce": bce, "member_dice": dice}
        K = max(len(p[0]) for _, p in cand)
        boxes = vec.new_zeros((len(cand), K, 5))
        boxes[..., 2:4] = 1.0
        for i, (b, (fb, _)) in enumerate(cand):
            boxes[i, :len(fb)] = fb
        bi = torch.tensor([b for b, _ in cand], device=vec.device)
        h = self.box_hidden(out["f4"][bi], out["feats"][0][bi], boxes)
        for i, (b, (fb, j)) in enumerate(cand):
            labels = targets[b]["labels"].to(vec.device)[j]
            lg, gate, rows = self._members_of(h[i, :len(fb)], labels, fb, tok[b], vec[b], mask[b], size,
                                               out["f4"][b])
            inst = vec[b, rows, 22] if vec.shape[-1] > 22 else torch.full_like(rows, INST_UNKNOWN, dtype=vec.dtype)
            if bool((inst == INST_UNKNOWN).all()):                   # no object ids: geometric membership
                gt = targets[b]["boxes"].float().to(vec.device)[j]
                tgt = members_from_boxes(vec[b, rows, 1:17].float().view(-1, 8, 2), gt).float()
            else:
                tgt = (inst.round().long()[None, :] == j[:, None]).float()
            g = gate.float()
            cnt = g.sum(1)
            keep = cnt > 0
            if not bool(keep.any()):
                continue
            lb = F.binary_cross_entropy_with_logits(lg, tgt, reduction="none") * g
            bce = bce + (lb.sum(1)[keep] / cnt[keep]).sum()
            pr = lg.sigmoid() * g
            tg = tgt * g
            dice = dice + (1 - (2 * (pr * tg).sum(1) + 1) / (pr.sum(1) + tg.sum(1) + 1))[keep].sum()
            n += int(keep.sum())
        n = max(n, 1)
        return {"member_bce": bce / n, "member_dice": dice / n}

    # ---- OpenStroke10: stroke graph
    def _graph_of(self, tok, vec, mask, size, f4):
        rows = mask.nonzero(as_tuple=True)[0]
        pts = vec[rows, 1:17].float().view(-1, 8, 2) * size
        cls, e, lg = self.graph(tok[rows], pts, f4)
        return cls, e, lg, rows

    def graph_loss_of(self, out, targets, size):
        side = out["side"]
        tok, vec, mask = side["tok"], side["vec"], side["mask"]
        lc_sum = le_sum = tok.sum() * 0.0
        n = 0
        for b, t in enumerate(targets):
            if vec.shape[-1] <= 22 or not bool(mask[b].any()):
                continue
            rows = mask[b].nonzero(as_tuple=True)[0]
            inst = vec[b, rows, 22]
            if bool((inst.round() == INST_UNKNOWN).all()):
                continue
            cls, e, lg, _ = self._graph_of(tok[b], vec[b], mask[b], size, out["f4"][b])
            cls_t, edge_t = graph_targets(inst, t["labels"].to(vec.device), e)
            lc, le = graph_loss(cls, lg, cls_t, edge_t)
            lc_sum, le_sum, n = lc_sum + lc, le_sum + le, n + 1
        n = max(n, 1)
        return {"graph_cls": lc_sum / n, "graph_edge": le_sum / n}

    @torch.no_grad()
    def graph_outputs(self, st):
        """Per image: (class logits (M_valid, C + 1), edges (E, 2) into the valid rows, edge logits (E,), valid rows)."""
        side = st["side"]
        tok, vec, mask = side["tok"], side["vec"], side["mask"]
        return [self._graph_of(tok[b], vec[b], mask[b], self.img_size, st["f4"][b]) for b in range(vec.shape[0])]

    # ---- forward
    def forward(self, x, targets=None, ctx=None):
        if self.training:
            out = self._forward_train(x, targets, ctx)
            if "side" in out and targets is not None:
                out.update(self.member_loss(out, targets, x.shape[-1]))
                if self.graph is not None:
                    out.update(self.graph_loss_of(out, targets, x.shape[-1]))
            return out
        rank, out_l, boxes, nms, st = self._scored(x, ctx)
        out = self.select(rank, out_l, boxes, nms, with_index=True)
        if st["side"]:
            out["member_logits"] = self.member_logits(out, st)
            if self.graph is not None:
                out["graph"] = self.graph_outputs(st)
        return out

    @torch.no_grad()
    def member_logits(self, out, st):
        """Membership logits (B, max_det, M) of every output row (padding rows and non-members: -30)."""
        side = st["side"]
        tok, vec, mask = side["tok"], side["vec"], side["mask"]
        logits = out["pred_logits"].float()
        B, R, _ = logits.shape
        M = vec.shape[1]
        res = logits.new_full((B, R, M), -30.0)
        conf = getattr(self, "nms_conf", 0.001)
        sc, lab = logits.sigmoid().max(-1)
        live = sc > conf
        K = int(live.sum(1).max()) if B else 0
        if K == 0 or M == 0:
            return res
        boxes = out["pred_boxes"].float().clone()
        boxes[..., :4] *= self.img_size
        rows_k = [live[b].nonzero(as_tuple=True)[0] for b in range(B)]
        pad = boxes.new_zeros((B, K, 5))
        pad[..., 2:4] = 1.0
        for b, r in enumerate(rows_k):
            pad[b, :len(r)] = boxes[b, r]
        h = self.box_hidden(st["f4"], st["feats"][0], pad)
        for b, r in enumerate(rows_k):
            if len(r) == 0 or not bool(mask[b].any()):
                continue
            lg, _, cols = self._members_of(h[b, :len(r)], lab[b, r], pad[b, :len(r)], tok[b], vec[b], mask[b],
                                           self.img_size, st["f4"][b])
            res[b, r[:, None], cols[None, :]] = lg
        return res


class OpenOBB25Loss(OpenOBB18Loss):
    """OpenOBB24's losses + the membership loss (BCE + dice) x member_weight."""

    def __init__(self, *args, member_weight=1.0, graph_weight=1.0, **kw):
        super().__init__(*args, **kw)
        self.member_weight = float(member_weight)
        self.graph_weight = float(graph_weight)

    def forward(self, outputs, targets):
        losses = super().forward(outputs, targets)
        if "member_bce" in outputs and self.member_weight > 0:
            losses["member_bce"] = outputs["member_bce"] * self.member_weight
            losses["member_dice"] = outputs["member_dice"] * self.member_weight
        if "graph_cls" in outputs and self.graph_weight > 0:          # OpenStroke10
            losses["graph_cls"] = outputs["graph_cls"] * self.graph_weight
            losses["graph_edge"] = outputs["graph_edge"] * self.graph_weight
        return losses


@torch.no_grad()
def member_stats(out, targets, acc, img_size, conf=0.25, thr=0.5):
    """Validation of the memberships on tiles with object ids: every labelled object is matched to the kept box (score
    >= conf, any class) of highest rotated IoU >= MEMBER_IOU; its predicted primitives (p >= thr) are compared with the
    labelled ones. acc (dict) accumulates tp / fp / fn primitives, objects, matched objects, objects with F1 >= 0.9."""
    lt = math.log(thr / (1 - thr))
    logits, boxes, ml = out["pred_logits"].float(), out["pred_boxes"].float(), out["member_logits"]
    for i, t in enumerate(targets):
        vec = t.get("vec")
        if vec is None or vec.shape[-1] < 24 or len(t["boxes"]) == 0:
            continue
        inst = vec[:, 22].round().long()
        if bool((inst == INST_UNKNOWN).all()):
            continue
        gt = t["boxes"].float().clone()
        gt[:, :4] *= img_size
        live = (logits[i].sigmoid().amax(-1) >= conf).nonzero(as_tuple=True)[0]
        acc["objects"] = acc.get("objects", 0) + len(gt)
        if len(live) == 0:
            continue
        b = boxes[i, live].clone()
        b[:, :4] *= img_size
        iou = near_iou(b.cpu(), gt)                                     # (K, N)
        best, k = iou.max(0)
        pred = (ml[i, live][:, :len(inst)] >= lt).cpu()
        for j in (best >= MEMBER_IOU).nonzero(as_tuple=True)[0].tolist():
            p, g = pred[k[j]], inst == j
            tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
            acc["tp"], acc["fp"], acc["fn"] = acc.get("tp", 0) + tp, acc.get("fp", 0) + fp, acc.get("fn", 0) + fn
            acc["matched"] = acc.get("matched", 0) + 1
            acc["good"] = acc.get("good", 0) + int(2 * tp >= 0.9 * (2 * tp + fp + fn) and (tp + fn) > 0)


@torch.no_grad()
def graph_stats(out, targets, acc, thr=0.5):
    """OpenStroke10 validation on tiles with object ids: stroke class accuracy (object strokes / others) and same-object edges
    (P / R at probability thr) of the candidate pairs. Written into member_stats' accumulator (keys g_*)."""
    for i, t in enumerate(targets):
        vec = t.get("vec")
        if vec is None or vec.shape[-1] < 24 or i >= len(out["graph"]):
            continue
        inst = vec[:, 22]
        if bool((inst.round() == INST_UNKNOWN).all()):
            continue
        cls, e, lg, rows = out["graph"][i]
        cls, e, lg = cls[:len(inst)].cpu(), e.cpu(), lg.cpu()
        keep = (e < len(inst)).all(1) if len(e) else torch.zeros(0, dtype=torch.bool)
        e, lg = e[keep], lg[keep]
        ct, et = graph_targets(inst.cpu(), t["labels"].cpu(), e)
        pc = cls.argmax(1)
        obj = ct > 0
        acc["g_obj"] = acc.get("g_obj", 0) + int(obj.sum())
        acc["g_obj_ok"] = acc.get("g_obj_ok", 0) + int((pc == ct)[obj].sum())
        acc["g_oth"] = acc.get("g_oth", 0) + int((~obj).sum())
        acc["g_oth_ok"] = acc.get("g_oth_ok", 0) + int((pc == 0)[~obj].sum())
        pe, ge = lg >= math.log(thr / (1 - thr)), et > 0.5
        acc["g_tp"] = acc.get("g_tp", 0) + int((pe & ge).sum())
        acc["g_fp"] = acc.get("g_fp", 0) + int((pe & ~ge).sum())
        acc["g_fn"] = acc.get("g_fn", 0) + int((~pe & ge).sum())


def member_summary(acc):
    """-> (dict, log line) from member_stats' accumulator."""
    tp, fp, fn = acc.get("tp", 0), acc.get("fp", 0), acc.get("fn", 0)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    n, m, g = acc.get("objects", 0), acc.get("matched", 0), acc.get("good", 0)
    d = {"P": p, "R": r, "F1": f1, "objects": n, "matched": m, "good": g}
    line = (f"Members (primitives of the labelled objects matched at IoU {MEMBER_IOU}, tiles): P {p:.3f} R {r:.3f} "
            f"F1 {f1:.3f}; {m}/{n} objects matched, {g} with member F1 >= 0.9")
    if "g_obj" in acc:                                               # OpenStroke10
        tp, fp, fn = acc["g_tp"], acc["g_fp"], acc["g_fn"]
        ep, er = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        d["graph"] = {"obj_acc": acc["g_obj_ok"] / max(acc["g_obj"], 1), "other_acc": acc["g_oth_ok"] / max(acc["g_oth"], 1),
                      "edge_P": ep, "edge_R": er, "edge_F1": 2 * ep * er / max(ep + er, 1e-9)}
        g_ = d["graph"]
        line += (f"\nGraph (OpenStroke10, tiles): stroke class acc object {g_['obj_acc']:.3f} / other {g_['other_acc']:.3f}; "
                 f"same-object edges P {ep:.3f} R {er:.3f} F1 {g_['edge_F1']:.3f} ({tp + fn} positive)")
    return d, line


def member_lists(logits, pid, mask, thr=0.5):
    """One tile: logits (R, M), pid (M,), mask (M,) -> list per output row of primitive ids with p >= thr."""
    lt = math.log(thr / (1 - thr)) if 0 < thr < 1 else 0.0
    sel = (logits >= lt) & mask[None, :]
    return [pid[s].long().tolist() for s in sel]
