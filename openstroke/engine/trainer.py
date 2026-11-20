"""Cross-validation trainer with YOLO / OpenOBB conventions.

Per validation fold: AdamW (no weight decay on norms / biases), linear warmup then cosine (or linear) decay
from ``lr0`` to ``lr0 * lrf``, AMP (bf16 when the GPU has it, else fp16 + GradScaler), gradient clipping,
EMA weights for validation and checkpoints, non-finite steps skipped, ``best.pt`` (EMA, by val F2),
``last.pt`` (everything, for ``resume``), ``results.csv`` / ``results.png``, early stop after
``patience`` epochs without a better F2. The run ends with a per-page table and a summary over folds.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..data import Augment, Page, WindowDataset, coincident_weights, collate, describe, dihedral, eval_windows, raw_items, window
from ..features import FEATURE_NAMES, N_FEATURES
from ..gpu_batch import build_batch
from ..metrics import length_counts, per_page, prf, sweep
from .console import VAL_HEAD, EpochBar, dur, table, val_row
from .ema import ModelEMA
from .report import append_results, plot_results


def param_groups(model, weight_decay, slow_image=True):
    """AdamW groups; the pretrained image backbone (E28 ``img.``, not its new ``img.proj``) learns at 0.1 x lr."""

    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nd = p.ndim < 2 or "norm" in n or n.endswith(".bias")
        slow = slow_image and n.startswith("img.") and not n.startswith(("img.proj", "img.stroke_context."))
        groups.setdefault((nd, slow), []).append(p)
    return [{"params": ps, "weight_decay": 0.0 if nd else weight_decay, "lr_scale": 0.1 if slow else 1.0}
            for (nd, slow), ps in groups.items()]


def lr_factor(it: int, total: int, warmup: int, lrf: float, cos: bool) -> float:
    warm = min(1.0, (it + 1) / max(warmup, 1))
    prog = min(max(it - warmup, 0) / max(total - warmup, 1), 1.0)
    f = lrf + (1 - lrf) * (0.5 * (1 + math.cos(math.pi * prog)) if cos else (1 - prog))
    return warm * f


def use_gpu_batch(cfg, device) -> bool:
    v = str(getattr(cfg, "gpu_batch", "auto")).lower()
    return device.type == "cuda" if v == "auto" else v in ("1", "true", "yes")


def to_device(b, device):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


def score_items(model, page: Page, items: list, window_size: int, device, amp_dtype=None, use_layer: bool = False, use_obj: bool = False,
                edge_acc: list | None = None, piece_acc: list | None = None):
    """One batch of (window start, dihedral view) -> token positions, their probabilities, their layer probabilities (or None)."""

    import torch

    with torch.no_grad():
        ws = [dihedral(window(page, s, window_size), k) if k else window(page, s, window_size) for s, k in items]
        b = build_batch(ws, device) if device.type == "cuda" else to_device(collate(ws), device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float16,
                            enabled=device.type == "cuda"):
            prob = torch.sigmoid(model(b).float())
        prob = prob.cpu().numpy()
        pos = b["pos"].cpu().numpy()
        m = pos >= 0
        tok = None
        if use_layer:
            lp = torch.sigmoid(model.last_layer.float())
            tok = torch.gather(lp, 1, b["layer"].clamp(max=lp.shape[1] - 1)).cpu().numpy()[m]
        cls = None
        if use_obj:   # the class head ran in the same forward pass: read it here instead of a second pass (speed)
            from ..objects import N_CLASSES, OFFSET_SCALE
            o = model.last_obj.float()
            cls = torch.cat([torch.softmax(o[..., :N_CLASSES], -1), o[..., N_CLASSES:] * OFFSET_SCALE], -1).cpu().numpy()[m]   # probs + offset (pt)
        if edge_acc is not None and getattr(model, "last_edge", None) is not None:   # OpenStroke30: same-object logit per link, as page positions
            z, i, j = model.last_edge
            pf = pos.reshape(-1)
            a, c = pf[i.cpu().numpy()], pf[j.cpu().numpy()]
            ok = (a >= 0) & (c >= 0)
            edge_acc.append((np.minimum(a, c)[ok], np.maximum(a, c)[ok], z.float().cpu().numpy()[ok]))
        lp = getattr(model, "last_piece", None)
        if piece_acc is not None and lp is not None:   # OpenStroke34: piece outputs of the plain views (pieces are not turned by TTA here)
            plain = np.array([k == 0 for _, k in items])
            pid = b["pc_id"].cpu().numpy()
            pm = b["pc_mask"].cpu().numpy() & plain[:, None]
            fg = torch.sigmoid(lp["fg"].float()).cpu().numpy()
            pc = torch.softmax(lp["cls"].float(), -1).cpu().numpy()
            if "region_features" in lp:
                pc = np.concatenate((pc, lp["region_features"].float().cpu().numpy()), axis=-1)
            P = pid.shape[1]
            li, lj = lp["i"].cpu().numpy(), lp["j"].cpu().numpy()
            ok = plain[li // P] & plain[lj // P] if len(li) else np.zeros(0, bool)
            a, c = pid.reshape(-1)[li[ok]], pid.reshape(-1)[lj[ok]]
            piece_acc.append((pid[pm], fg[pm], pc[pm], np.minimum(a, c), np.maximum(a, c), lp["link"].float().cpu().numpy()[ok]))
    return pos[m], prob[m], tok, cls


def predict_page(model, page: Page, window_size: int, device, batch: int = 4, tta: int = 1, amp_dtype=None,
                 layer_out: list | None = None, on_batch=None, obj_out: list | None = None, edge_out: list | None = None,
                 piece_out: list | None = None):
    """Probabilities for every token of ``page``: windows with stride window/2 and ``tta`` dihedral views,
    all averaged. With a layer head and ``layer_out`` given, also appends each token's layer probability (its layer's
    logit in each window it was in, averaged the same way). ``on_batch(done, total)`` reports progress in windows."""

    total = np.zeros(len(page), np.float64)
    count = np.zeros(len(page), np.float64)
    ltotal = np.zeros(len(page), np.float64)
    use_layer = layer_out is not None and getattr(model, "layer_mlp", None) is not None
    use_obj = obj_out is not None and getattr(model, "obj_mlp", None) is not None
    ctotal = None
    items = [(s, k) for s in eval_windows(page, window_size) for k in range(tta)]
    edge_acc = [] if edge_out is not None and getattr(model, "edge_mlp", None) is not None else None
    piece_acc = [] if piece_out is not None and getattr(model, "piece", None) is not None and page.pieces is not None else None
    model.eval()
    for a in range(0, len(items), batch):
        pos, prob, tok, cls = score_items(model, page, items[a:a + batch], window_size, device, amp_dtype, use_layer, use_obj, edge_acc,
                                          piece_acc)
        np.add.at(total, pos, prob)
        np.add.at(count, pos, 1.0)
        if use_layer:
            np.add.at(ltotal, pos, tok)
        if use_obj:   # per-shape class probabilities: averaged over windows and views like the stroke probability
            if ctotal is None:
                ctotal = np.zeros((len(page), cls.shape[1]), np.float64)
            np.add.at(ctotal, pos, cls)
        if on_batch is not None:
            on_batch(min(a + batch, len(items)), len(items))
    if use_layer:
        layer_out.append(ltotal / np.maximum(count, 1))
    if use_obj:   # (n, N_CLASSES + 2): class probabilities, then the offset to the object centre in pt
        obj_out.append(ctotal / np.maximum(count, 1)[:, None] if ctotal is not None else np.zeros((len(page), 0)))
    if edge_acc is not None:   # (E, 2) page positions (a < b) and the mean same-object logit over windows and views
        edge_out.append(merge_edges(edge_acc, len(page)))
    if piece_acc is not None:   # OpenStroke34: per piece (npz row) its mean pipe probability / class probabilities, per link its mean logit
        q = merge_pieces(piece_acc)
        if model.piece.region is not None:
            from ..objects import N_CLASSES
            embedding = q["cls"][:, N_CLASSES:]
            q["cls"] = q["cls"][:, :N_CLASSES].copy()
            q.update(model.piece.region.decode_page(page, q, embedding))
        piece_out.append(q)
    return total / np.maximum(count, 1)


def merge_pieces(acc: list) -> dict:
    """Piece outputs seen in several windows: means per piece id and per link. -> {pid, fg, cls, pe (E, 2), plink}."""
    if not acc:
        return {"pid": np.zeros(0, np.int64), "fg": np.zeros(0, np.float32), "cls": np.zeros((0, 0), np.float32),
                "pe": np.zeros((0, 2), np.int64), "plink": np.zeros(0, np.float32)}
    ids = np.concatenate([x[0] for x in acc]).astype(np.int64)
    u, inv = np.unique(ids, return_inverse=True)
    cnt = np.bincount(inv)
    fg = np.bincount(inv, weights=np.concatenate([x[1] for x in acc])) / cnt
    cl = np.concatenate([x[2] for x in acc])
    cls = np.stack([np.bincount(inv, weights=cl[:, k], minlength=len(u)) for k in range(cl.shape[1])], 1) / cnt[:, None]
    n = int(max(u.max() if len(u) else 0, max((x[4].max() for x in acc if len(x[4])), default=0))) + 1
    pe, plink = merge_edges([(x[3], x[4], x[5]) for x in acc], n)
    return {"pid": u, "fg": fg.astype(np.float32), "cls": cls.astype(np.float32), "pe": pe, "plink": plink}


def merge_edges(acc: list, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per link seen in several windows / views: its mean logit. -> ((E, 2) int64, (E,) float32)."""
    if not acc:
        return np.zeros((0, 2), np.int64), np.zeros(0, np.float32)
    a = np.concatenate([x[0] for x in acc]).astype(np.int64)
    b = np.concatenate([x[1] for x in acc]).astype(np.int64)
    z = np.concatenate([x[2] for x in acc]).astype(np.float64)
    key, inv = np.unique(a * n + b, return_inverse=True)
    mean = np.bincount(inv, weights=z) / np.bincount(inv)
    return np.stack([key // n, key % n], 1), mean.astype(np.float32)


def class_accuracy(pages, obj_out) -> tuple[float, dict]:
    """OpenStroke29: length-weighted accuracy of the class head on the labelled strokes that have a class (ocls > 0), overall and per
    class (6 benchmark classes: square-to-round variants merged into their base; Tee kept). -> (overall, {class: accuracy})."""
    from ..objects import HVAC6_MERGE_IDX, HVAC_CLASSES, N_CLASSES

    def m6(k):
        k = np.asarray(k)
        out = k.copy()
        for a, b in HVAC6_MERGE_IDX.items():
            out[k == a] = b if b is not None else a
        return out

    hit = np.zeros(N_CLASSES)
    tot = np.zeros(N_CLASSES)
    for p, pc in zip(pages, obj_out):
        y = (p.label > 0) & (p.ocls > 0)
        if not y.any():
            continue
        g = m6(p.ocls[y])
        pr = pc[y, :N_CLASSES].copy()
        pr[:, 0] = -1.0
        q = m6(pr.argmax(1))
        w = p.length[y]
        np.add.at(tot, g, w)
        np.add.at(hit, g, w * (q == g))
    per = {HVAC_CLASSES[k]: float(hit[k] / tot[k]) for k in range(1, N_CLASSES) if tot[k] > 0}
    return float(hit.sum() / max(tot.sum(), 1e-9)), per


def evaluate(model, pages, cfg, device, amp_dtype, tta=1, layer_out: list | None = None, obj_out: list | None = None,
             edge_out: list | None = None, piece_out: list | None = None):
    scores = [predict_page(model, p, cfg.window, device, cfg.val_batch, tta, amp_dtype, layer_out, obj_out=obj_out, edge_out=edge_out,
                           piece_out=piece_out if p.pieces is not None else None)
              for p in pages]
    equal_pages = getattr(cfg, "gt_only", False) and not any(
        getattr(model, key, None) is not None for key in ("obj_mlp", "edge_mlp", "piece", "layer_mlp"))
    labels, lengths = [p.label for p in pages], [p.length for p in pages]
    if getattr(cfg, "hard_pairs", False):
        from ..hard_pairs import THRESHOLD

        fixed, at05 = sweep(scores, labels, lengths, thresholds=np.array([THRESHOLD, 0.5]), equal_pages=True)["curve"]
        return scores, fixed, at05
    sw = sweep(scores, labels, lengths, equal_pages=equal_pages)
    at05 = next(r for r in sw["curve"] if abs(r["threshold"] - 0.5) < 1e-9)
    return scores, sw["best"], at05


def object_class_weights(pages) -> list[float]:
    """E62: class weights of the object loss, the inverse square root of each class's true length in the train pages (0 for class 0)."""

    from ..objects import N_CLASSES

    tot = np.zeros(N_CLASSES)
    for p in pages:
        if p.ocls is not None:
            tot += np.bincount(p.ocls, weights=p.length, minlength=N_CLASSES)[:N_CLASSES]
    w = np.where(tot > 0, 1.0 / np.sqrt(np.maximum(tot, 1.0)), 0.0)
    w[0] = 0.0
    seen = w > 0
    if seen.any():
        w[seen] = w[seen] / w[seen].mean()
    return w.astype(np.float32).tolist()


def object_loss(out, b, class_weight, alpha: float = 0.5):
    """E62: cross-entropy of the object class and smooth L1 of the centre offset, on shapes that draw an object; weighted by (length + 1) ** alpha."""

    import torch
    import torch.nn.functional as F

    from ..objects import N_CLASSES

    m = (b["ocls"] > 0) & b["mask0"]
    if not bool(m.any()):
        return out.float().sum() * 0.0
    w = (b["length"][m].float().clamp(min=0) + 1.0) ** alpha
    o = out.float()[m]
    ce = F.cross_entropy(o[:, :N_CLASSES], b["ocls"][m], weight=class_weight, reduction="none")
    reg = F.smooth_l1_loss(o[:, N_CLASSES:], b["ooff"][m].float(), reduction="none").sum(-1)
    return ((ce + reg) * w).sum() / w.sum().clamp(min=1e-6)


def predict_objects(model, page, window_size: int, device, batch: int = 4, amp_dtype=None):
    """E62: per shape of ``page``, object class probabilities (n, classes) and offset to the object centre (n, 2) in pt; one view, windows averaged."""

    import torch

    from ..objects import N_CLASSES, OFFSET_SCALE

    n = len(page)
    prob = np.zeros((n, N_CLASSES), np.float64)
    off = np.zeros((n, 2), np.float64)
    count = np.zeros(n, np.float64)
    starts = eval_windows(page, window_size)
    model.eval()
    with torch.no_grad():
        for a in range(0, len(starts), batch):
            ws = [window(page, s, window_size) for s in starts[a:a + batch]]
            b = build_batch(ws, device) if device.type == "cuda" else to_device(collate(ws), device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float16, enabled=device.type == "cuda"):
                model(b)
            o = model.last_obj.float()
            pr = torch.softmax(o[..., :N_CLASSES], -1).cpu().numpy()
            of = (o[..., N_CLASSES:] * OFFSET_SCALE).cpu().numpy()
            pos = b["pos"].cpu().numpy()
            k = pos >= 0
            np.add.at(prob, pos[k], pr[k])
            np.add.at(off, pos[k], of[k])
            np.add.at(count, pos[k], 1.0)
    c = np.maximum(count, 1.0)[:, None]
    return prob / c, off / c


def layer_loss(logits, b):
    """BCE of the layer head: one target per (window, layer) = whether that layer is a pipe layer of its page
    (``data.layer_targets``), weighted by sqrt of the layer's length in the window."""

    import torch

    B, n_l = logits.shape
    m = b["mask0"] > 0
    key = (b["layer"] + torch.arange(B, device=logits.device)[:, None] * n_l)[m]
    z = lambda: torch.zeros(B * n_l, device=logits.device, dtype=torch.float32)  # noqa: E731
    cnt = z().index_add_(0, key, torch.ones_like(key, dtype=torch.float32))
    y = z().index_add_(0, key, b["ylayer"][m].float()) / cnt.clamp(min=1)
    w = z().index_add_(0, key, b["length"][m].float()).sqrt()
    ok = cnt > 0
    w = w[ok] / w[ok].mean().clamp(min=1e-6)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits.reshape(-1)[ok].float(), y[ok], weight=w, pos_weight=torch.tensor(3.0, device=logits.device))


class Trainer:
    def __init__(self, cfg, log, say, pdf_by_stem=None):
        self.cfg, self.log, self.say = cfg, log, say
        self.pdf_by_stem = pdf_by_stem

    def _device(self):
        import torch

        c = self.cfg
        dev = torch.device(c.device if c.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        if dev.type == "cuda":   # TF32 for the matmuls / convolutions not under bf16 autocast (speed; negligible numeric change)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if bf16 else torch.float16
        return dev, amp_dtype, dev.type == "cuda" and c.amp

    def build_model(self, train_pages, init=None):
        import torch

        from ..models import build, from_config, load_weights

        c = self.cfg
        if init is not None:
            ck = torch.load(init, map_location="cpu", weights_only=False)
            cfg = dict(ck["config"])
            # heads asked for on the command line but absent from the checkpoint start fresh (E26: pretrained -> two heads)
            extra = {k: True for k in ("layer_head", "layer_head_names", "image_branch", "img_lum", "obj_head", "edge_head")
                     if getattr(c, k, False) and not cfg.get(k)}
            if getattr(c, "edge_pairs", False) and not cfg.get("edge_pairs"):   # OpenStroke31: same network, another link loss
                extra["edge_pairs"] = True
            if getattr(c, "piece_head", False) and not cfg.get("piece_head"):   # OpenStroke34: new piece head on a trained model
                extra["piece_head"] = True
            if getattr(c, "region_head", False) and not cfg.get("region_head"):
                extra.update(piece_head=True, region_head=True)
            if getattr(c, "stroke_context", False) and not cfg.get("stroke_context"):
                extra["stroke_context"] = True
            if getattr(c, "continuation", False) and not cfg.get("continuation"):
                extra["continuation"] = True
            if int(getattr(c, "rel_attn", 0) or 0) and not cfg.get("rel_attn"):   # OpenStroke30: zero-init graph attention on a trained model
                extra["rel_attn"] = int(c.rel_attn)
            if str(getattr(c, "pool_by", "layer")) != cfg.get("pool_by", "layer"):  # E31: re-group LayerPool (same weights)
                extra["pool_by"] = str(getattr(c, "pool_by", "layer"))
            if extra.get("image_branch"):
                extra["img_pretrained"] = bool(getattr(c, "img_pretrained", True))
                extra["img_ppt"] = float(getattr(c, "img_ppt", 1.0))  # E34: resolution of the new image branch
                extra["img_max_px"] = int(getattr(c, "img_max_px", 1536))
            new_bb = str(getattr(c, "img_backbone", "resnet18"))
            if new_bb != cfg.get("img_backbone", "resnet18"):  # E48: another image backbone; old image weights do not apply, the projections restart at 0
                extra["img_backbone"] = new_bb
                extra["img_weights"] = getattr(c, "img_weights", None)
                extra["img_pretrained"] = False
                ck["model"] = {k: v for k, v in ck["model"].items() if not k.startswith("img.")}
            fd = sorted(getattr(c, "feature_drop", None) or [])
            if fd != sorted(cfg.get("feature_drop") or []):   # F185: the command's feature mask wins over the checkpoint's (it was ignored)
                extra["feature_drop"] = fd or None
                ck["model"] = {k: v for k, v in ck["model"].items() if k != "feat_keep"}   # the new mask, not the checkpoint's
            if getattr(c, "mode_input", False) and not cfg.get("mode_input"):  # E47: the page's system, known before inference; zero-init
                extra["mode_input"] = True
            if getattr(c, "img_fpn", False) and not cfg.get("img_fpn"):  # E37: new layer4 starts from ImageNet, not random
                extra["img_fpn"] = True
            if float(getattr(c, "img_aux", 0) or 0) > 0 and not cfg.get("img_aux"):
                extra["img_aux"] = True
            model = from_config({**cfg, **extra}, weights_follow=not (extra.get("image_branch") or extra.get("img_fpn") or extra.get("img_backbone")))
            load_weights(model, ck["model"], new_heads=bool(extra))
            self.log(f"init from {init}")
            return self._tag_layers(model)
        model = build(c.arch, N_FEATURES, dims=c.dims, enc_depths=c.enc_depths, dec_depths=c.dec_depths,
                      patch=c.patch, drop_path=c.drop_path, layer_pool=c.layer_pool, dim=c.dim, depth=c.depth, k=c.k,
                      mode_input=getattr(c, "mode_input", False), layer_names=getattr(c, "layer_names", False),
                      group_context=getattr(c, "group_context", False), relations=getattr(c, "relations", 0),
                      feature_drop=getattr(c, "feature_drop", None) or None,
                      pretrain=float(getattr(c, "pretrain", 0) or 0) > 0,
                      layer_head=bool(getattr(c, "layer_head", False)),
                      layer_head_names=bool(getattr(c, "layer_head_names", False)),
                      image_branch=bool(getattr(c, "image_branch", False)),
                      img_pretrained=bool(getattr(c, "img_pretrained", True)),
                      img_lum=bool(getattr(c, "img_lum", False)), pool_by=str(getattr(c, "pool_by", "layer")),
                      img_ppt=float(getattr(c, "img_ppt", 1.0)), img_max_px=int(getattr(c, "img_max_px", 1536)),
                      img_fpn=bool(getattr(c, "img_fpn", False)), img_aux=float(getattr(c, "img_aux", 0) or 0) > 0,
                      obj_head=bool(getattr(c, "obj_head", False)), rel_attn=int(getattr(c, "rel_attn", 0) or 0),
                      edge_head=bool(getattr(c, "edge_head", False)), edge_pairs=bool(getattr(c, "edge_pairs", False)),
                      piece_head=bool(getattr(c, "piece_head", False)), region_head=bool(getattr(c, "region_head", False)),
                      continuation=bool(getattr(c, "continuation", False)))
        rng = np.random.default_rng(0)
        xs = np.concatenate([p.x[rng.choice(len(p), min(len(p), 100_000), replace=False)] for p in train_pages])
        model.set_normalization(xs.mean(axis=0), xs.std(axis=0))
        return self._tag_layers(model)

    def _tag_layers(self, model):
        """OpenStroke28: the checkpoint remembers that its pages were read without layers (inference does the same)."""
        if bool(getattr(self.cfg, "no_layers", False)):
            model.config = {**model.config, "no_layers": True}
        return model

    def run_fold(self, fold, n_folds, train_pages, val_pages, out: Path, init=None):
        import torch

        from ..models import loss as loss_fn

        c = self.cfg
        out.mkdir(parents=True, exist_ok=True)
        device, amp_dtype, amp = self._device()
        torch.manual_seed(c.seed + fold)
        last = out / "last.pt"
        hard_pairs = getattr(c, "hard_pairs", False)
        if hard_pairs and device.type != "cuda":
            raise ValueError("hard_pairs requires a CUDA runtime for teacher scoring and training")
        model = self.build_model(train_pages, init).to(device)
        if hard_pairs:
            from ..hard_pairs import LOSS_WEIGHT, MARGIN, MAX_RECALL_DROP, MIN_F2_GAIN, THRESHOLD, rank_loss

            self.say(f"Scoring the input checkpoint on {len(train_pages)} GT train pages for hard pairs (TTA x1)")
            pages_with_pairs = 0
            # Run before restoring last.pt, so a resume always uses the original teacher.
            for i, page in enumerate(train_pages, 1):
                scores = predict_page(model, page, c.window, device, c.val_batch, 1, amp_dtype).astype(np.float32)
                if not np.isfinite(scores).all():
                    raise ValueError(f"Non-finite input checkpoint scores on {page.stem}")
                page.pair_teacher = scores
                valid = page.length > 0
                missed = valid & (page.label > 0) & (scores < THRESHOLD)
                extra = valid & (page.label == 0) & (scores >= THRESHOLD)
                shared = np.intersect1d(page.style[missed], page.style[extra])
                pages_with_pairs += bool(len(shared))
                self.say(f"  [{i}/{len(train_pages)}] {page.stem}: {len(shared)} hard-pair styles")
            if not pages_with_pairs:
                raise ValueError("No same-style errors found on GT train")
            self.say(f"Hard pairs: {pages_with_pairs}/{len(train_pages)} pages eligible, "
                     f"rank weight {LOSS_WEIGHT:g}, margin {MARGIN:g}; fixed GT threshold {THRESHOLD:.2f}")

        continuation = getattr(model, "continuation", None)
        if continuation is not None:
            model.requires_grad_(False)
            continuation.requires_grad_(True)
        ema = ModelEMA(model, decay=c.ema, warmups=max(c.steps, 100))
        region_pilot = getattr(c, "model", None) == "openstroke35"
        stroke_context = getattr(getattr(model, "img", None), "stroke_context", None)
        fixed_schedule = region_pilot or continuation is not None or stroke_context is not None or hard_pairs
        opt = torch.optim.AdamW(param_groups(model, c.weight_decay, slow_image=not region_pilot), lr=c.lr0, betas=(0.9, 0.999))
        scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
        start_epoch, it, best, best_epoch = 0, 0, -1.0, -1
        if c.resume and last.exists():
            ck = torch.load(last, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            ema.load_state_dict(ck["ema"])
            opt.load_state_dict(ck["opt"])
            if ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            if fixed_schedule and "rng" in ck:
                torch.set_rng_state(ck["rng"].cpu())
                if device.type == "cuda" and ck.get("cuda_rng") is not None:
                    torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device)
            start_epoch, it, best, best_epoch = ck["epoch"] + 1, ck["iter"], ck["best"], ck["best_epoch"]
            self.say(f"resume  fold {fold} from epoch {start_epoch + 1} (best F2 {best:.4f} at epoch {best_epoch + 1})")
        trained_key = (False, -1.0)
        trained_epoch = None
        hard_baseline = None
        if hard_pairs and start_epoch:
            hard_baseline = json.loads((out / "before_finetune.json").read_text(encoding="utf-8"))["validation"]
            updated = out / "best_finetuned.pt"
            if updated.exists():
                trained = torch.load(updated, map_location="cpu", weights_only=False)
                if trained["epoch"] < start_epoch:
                    trained_key = (trained["recall_ok"], trained["fitness"])
                    trained_epoch = trained["epoch"] + 1
                del trained
        nparams = sum(p.numel() for p in model.parameters())
        names = ", ".join(sorted({p.stem.split("_gt")[0] for p in val_pages}))
        self.say(f"\nFold {fold + 1}/{n_folds}  val: {describe(val_pages)} [{names}]  |  train: {describe(train_pages)}")
        if val_pages:
            tl = sum(float(p.length[p.label > 0].sum()) for p in val_pages)
            pi = tl / max(sum(float(p.length.sum()) for p in val_pages), 1e-9)
            self.say(f"baseline val F2 if every shape is called true (no learning): {5 * pi / (4 * pi + 1):.3f} "
                     f"(true = {pi:.1%} of length)")
        from ..train import arch_name, describe_arch
        model.config["arch"] = arch_name(model.config)   # the research version (openstroke28 / 29 / 30) saved in the checkpoint
        if fold == 0 or start_epoch == 0:
            self.say(f"model   {describe_arch(model.config)} {nparams / 1e6:.2f}M params, " + json.dumps(model.config))
            if continuation is not None:
                trainable = sum(p.numel() for p in continuation.parameters())
                self.say(f"Frozen E114 backbone; {trainable:,} trainable continuation parameters, cosine schedule 50 epochs")
        aug = Augment(fliplr=c.fliplr, rot90=c.rot90, scale=c.scale, color=c.color, gray=c.gray, restyle=c.restyle,
                      restyle_mode=str(getattr(c, "restyle_mode", "free")),
                      name_drop=getattr(c, "name_drop", 0.0)
                      if getattr(c, "layer_names", False) or getattr(c, "layer_head_names", False) else 0.0,
                      drop_tokens=c.drop_tokens, paste=float(getattr(c, "paste", 0.0) or 0.0),
                      paste_mode=str(getattr(c, "paste_mode", "region") or "region"))
        gpu_batch = use_gpu_batch(c, device) and getattr(model, "piece", None) is None   # pieces are augmented in numpy (data.augment)
        co = float(getattr(c, "coincide", 0) or 0)
        for p in train_pages:  # E36: no loss on true shapes that only lie under a pipe (their group is mostly background)
            p.lw = coincident_weights(p, co) if co > 0 else None
        if co > 0:
            n0 = sum(float(p.length[p.lw == 0].sum()) for p in train_pages)
            n1 = sum(float(p.length[p.label > 0].sum()) for p in train_pages)
            self.say(f"coincide {co}: {n0 / max(n1, 1e-9):.1%} of the true length of the train pages carries no loss")
        ds = WindowDataset(train_pages, c.steps * c.batch, c.window, aug, seed=c.seed + 1000 * fold,
                           plumbing_share=getattr(c, "plumbing_share", None), raw=gpu_batch,
                           viewer_share=getattr(c, "viewer_share", None),
                           pdf_by_stem=self.pdf_by_stem if getattr(c, "pdf_balance", False) else None)
        if self.pdf_by_stem is not None:
            mass = {}
            for p, weight in zip(train_pages, ds.prob):
                key = self.pdf_by_stem[p.stem]
                mass[key] = mass.get(key, 0.0) + float(weight)
            self.say(f"PDF sampling: {len(mass)} train PDFs, min {min(mass.values()):.1%}, max {max(mass.values()):.1%}")
        dl = torch.utils.data.DataLoader(ds, batch_size=c.batch, shuffle=False, num_workers=c.workers,
                                         collate_fn=raw_items if gpu_batch else collate,
                                         pin_memory=device.type == "cuda" and not gpu_batch,
                                         persistent_workers=False, prefetch_factor=4 if c.workers else None)
        if fold == 0 or start_epoch == 0:
            self.say(f"batches built on {'the GPU (gpu_batch)' if gpu_batch else 'CPU workers'}")
        total_iters = (50 if fixed_schedule else c.epochs) * c.steps
        warmup = int(c.warmup_epochs * c.steps)
        mae_ratio = float(getattr(c, "pretrain", 0) or 0)  # E19: masked-shape pretraining instead of the label loss
        cols = ["mse"] if mae_ratio > 0 else (["bce", "lovasz"] if c.lovasz > 0 else ["bce"])
        if hard_pairs:
            cols = cols + ["rank"]
        if mae_ratio <= 0 and getattr(c, "layer_head", False):
            cols = cols + ["layer"]
        aux_w = float(getattr(c, "img_aux", 0) or 0)  # E38: dense pipe-ink loss of the image branch
        from ..models.image_branch import aux_loss
        if mae_ratio <= 0 and aux_w > 0:
            cols = cols + ["aux"]
        obj_w = float(getattr(c, "obj_loss", 1.0) or 0.0) if getattr(model, "obj_mlp", None) is not None else 0.0
        if mae_ratio <= 0 and obj_w > 0:
            cols = cols + ["obj"]
            cw = torch.tensor(object_class_weights(train_pages), device=device)
        closure_w = float(getattr(c, "closure", 0) or 0) if mae_ratio <= 0 else 0.0
        edge_w = float(getattr(c, "edge_loss", 0.5) or 0) if mae_ratio <= 0 and getattr(model, "edge_mlp", None) is not None else 0.0
        if closure_w > 0:
            cols = cols + ["close"]
        if edge_w > 0:
            cols = cols + ["edge"]
        piece_w = float(getattr(c, "piece_loss", 1.0) or 0) if mae_ratio <= 0 and getattr(model, "piece", None) is not None else 0.0
        if piece_w > 0:
            cols = cols + ["piece"]
        region = getattr(getattr(model, "piece", None), "region", None)
        if region is not None:
            cols = cols + ["region"]
        if mae_ratio > 0:
            self.say(f"pretrain masked shapes: {mae_ratio:.0%} of tokens hidden, recon of their features (labels unused)")
        if (getattr(c, "gt_only", False) and init is not None and val_pages and mae_ratio == 0
                and (not c.resume or (hard_pairs and start_epoch == 0))):
            inputs, initial, _ = evaluate(ema.module, val_pages, c, device, amp_dtype)
            if hard_pairs:
                hard_baseline = initial
            labels, lengths, stems = [p.label for p in val_pages], [p.length for p in val_pages], [p.stem for p in val_pages]
            by_page = per_page(inputs, labels, lengths, stems, initial["threshold"])
            before = {"validation": initial, "pages": by_page}
            (out / "before_finetune.json").write_text(json.dumps(before, indent=1), encoding="utf-8")
            self.say(f"Before fine-tune: GT validation stroke F2 {initial['f2']:.3f} "
                     f"P {initial['precision']:.3f} R {initial['recall']:.3f} @ {initial['threshold']:.2f}")
            if "macro_f2" in initial:
                self.say(f"  Mean PDF F2 {initial['macro_f2']:.3f}, worst PDF {initial['worst_f2']:.3f}")
            for row in by_page:
                self.say(f"  {row['stem']} F2 {row['f2']:.3f} P {row['precision']:.3f} R {row['recall']:.3f}")
            # A fine-tune that never improves over its input should keep the input.
            if not any(getattr(model, key, None) is not None for key in ("obj_mlp", "edge_mlp", "piece", "layer_mlp")):
                best = initial.get("macro_f2", initial["f2"])
                torch.save({"model": ema.module.state_dict(), "config": ema.module.config, "epoch": -1,
                            "threshold": initial["threshold"], "val_f2": initial["f2"], "fitness": best,
                            "feature_names": FEATURE_NAMES, "args": vars(c)}, out / "best.pt")
        epochs_trained = start_epoch
        history = []
        for epoch in range(start_epoch, c.epochs):
            ds.epoch = epoch
            if continuation is None:
                model.train()
            else:
                model.eval()
                continuation.train()
            t0 = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            sums, n, shapes, bad = {}, 0, 0, 0
            epoch_pairs = 0
            bar = EpochBar(epoch, c.epochs, c.steps, cols, c.window, header=True)
            for bi, b in enumerate(dl):
                step_pairs = 0
                for g in opt.param_groups:
                    g["lr"] = c.lr0 * lr_factor(it, total_iters, warmup, c.lrf, c.cos_lr) * g.get("lr_scale", 1.0)
                if gpu_batch:
                    b = build_batch([w for w, _ in b], device, aug, [s for _, s in b])
                else:
                    b = to_device(b, device)
                if mae_ratio > 0:
                    b["mae"] = (torch.rand(b["mask0"].shape, device=device) < mae_ratio) & (b["mask0"] > 0)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                    logits = model(b)
                if mae_ratio > 0:
                    err = ((logits.float() - model.target_features(b)) ** 2).mean(-1)
                    total = err[b["mae"]].mean()
                    parts = {"mse": total}
                else:
                    total, parts = loss_fn(c.arch, logits, b, c.pos_weight, c.alpha, c.lovasz, float(getattr(c, "poly", 0.0) or 0.0))
                    if hard_pairs:
                        rank, step_pairs = rank_loss(logits, b)
                        total = total + LOSS_WEIGHT * rank
                        parts["rank"] = rank
                    if getattr(model, "layer_mlp", None) is not None and "ylayer" in b:
                        lb = layer_loss(model.last_layer, b)
                        total = total + float(getattr(c, "layer_loss", 0.5)) * lb
                        parts = {**parts, "layer": lb}
                    if closure_w > 0 or edge_w > 0:   # OpenStroke30 (H132 / H133)
                        from ..models.openstroke2 import closure_loss, edge_loss
                        if closure_w > 0:
                            lc = closure_loss(logits, b)
                            total = total + closure_w * lc
                            parts = {**parts, "close": lc}
                        if edge_w > 0:
                            le = edge_loss(model.last_edge, b, pairs=bool(getattr(model, "edge_pairs", False)))
                            total = total + edge_w * le
                            parts = {**parts, "edge": le}
                    if piece_w > 0 and getattr(model, "last_piece", None) is not None:   # OpenStroke34 (H141)
                        from ..models.openstroke2 import piece_loss
                        lpc = piece_loss(model.last_piece, b)
                        total = total + piece_w * lpc
                        parts = {**parts, "piece": lpc}
                        if region is not None:
                            lrgn = region.training_loss(model.last_piece, b)
                            total = total + lrgn
                            parts["region"] = lrgn
                    if obj_w > 0 and "ocls" in b and model.last_obj is not None:
                        lo = object_loss(model.last_obj, b, cw, c.alpha)
                        total = total + obj_w * lo
                        parts = {**parts, "obj": lo}
                    img = getattr(model, "img", None)
                    if aux_w > 0 and img is not None and img.last_aux is not None:
                        la = aux_loss(*img.last_aux)
                        img.last_aux = None
                        total = total + aux_w * la
                        parts = {**parts, "aux": la}
                if not torch.isfinite(total):
                    bad += 1
                    opt.zero_grad(set_to_none=True)
                    it += 1
                    continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip)
                if not torch.isfinite(gn):
                    bad += 1
                    scaler.update()
                    it += 1
                    continue
                scaler.step(opt)
                scaler.update()
                ema.update(model)
                epoch_pairs += step_pairs
                for k, v in parts.items():
                    sums[k] = sums.get(k, 0.0) + float(v.detach())
                n += 1
                it += 1
                shapes += int(b["mask0"].sum())
                bar.update(bi + 1, {k: sums[k] / n for k in sums}, int(b["mask0"].sum()),
                           torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else None,
                           force=bi + 1 == c.steps)
            bar.close()
            dt = time.time() - t0
            rec = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], "epoch_s": round(dt, 1),
                   "shapes_s": round(shapes / max(dt, 1e-9)),
                   "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if device.type == "cuda" else ""}
            rec.update({f"{k}_loss": round(v / max(n, 1), 5) for k, v in sums.items()})
            if bad:
                self.say(f"WARNING: {bad} non-finite steps skipped in epoch {epoch + 1}")
            if hard_pairs:
                rec["hard_pairs"] = epoch_pairs
                self.say(f"Hard pairs epoch {epoch + 1}: {epoch_pairs:,} used in successful steps")
                if not epoch_pairs:
                    self.say("WARNING: no hard pairs in successful batches this epoch")
            if mae_ratio > 0:  # no labels to score: keep the latest EMA weights as the start for fine-tuning
                best, best_epoch = -1.0, epoch
                torch.save({"model": ema.module.state_dict(), "config": ema.module.config, "epoch": epoch,
                            "threshold": 0.5, "val_f2": -1.0, "feature_names": FEATURE_NAMES, "args": vars(c)},
                           out / "best.pt")
            elif val_pages and ((epoch + 1) % c.val_every == 0 or epoch + 1 == c.epochs):
                tv = time.time()
                ema.repair(model, self.log)
                oval = [] if getattr(ema.module, "obj_mlp", None) is not None and all(q.ocls is not None for q in val_pages) else None
                ev = [] if getattr(ema.module, "edge_mlp", None) is not None else None
                pv = [] if getattr(ema.module, "piece", None) is not None else None
                vscores, bst, at05 = evaluate(ema.module, val_pages, c, device, amp_dtype, obj_out=oval, edge_out=ev, piece_out=pv)
                rec.update({"P": round(bst["precision"], 4), "R": round(bst["recall"], 4), "F2": round(bst["f2"], 4),
                            "F2@0.5": round(at05["f2"], 4), "thr": bst["threshold"]})
                print(VAL_HEAD + f"  ({dur(time.time() - tv)})")
                print(val_row("all", len(val_pages), sum(len(p) for p in val_pages),
                              sum(float(p.length[p.label > 0].sum()) for p in val_pages),
                              bst["precision"], bst["recall"], bst["f2"], at05["f2"], bst["threshold"]), flush=True)
                fitness = bst.get("macro_f2", bst["f2"])
                if "macro_f2" in bst:
                    rec.update(macro_f2=fitness, worst_f2=bst["worst_f2"])
                    self.say(f"GT validation mean PDF F2 {fitness:.3f}, worst PDF {bst['worst_f2']:.3f}")
                if not getattr(c, "gt_only", False) or ev is not None or pv is not None:
                    self.epoch_objects(val_pages, vscores, bst["threshold"], ev, pv, rec)
                if pv is not None and "obj_pieces_all" in rec:
                    # OpenStroke34: best.pt must not keep an untrained piece head (E127 kept epoch 1 on a stroke F2 tie): F2 x piece object F2
                    fitness = bst["f2"] * rec["obj_pieces_all"]
                    rec["fitness"] = round(fitness, 4)
                if oval:   # OpenStroke29: what the class head adds, visible every epoch (user 2026-11-16)
                    acc, per = class_accuracy(val_pages, oval)
                    macro = float(np.mean(list(per.values()))) if per else 0.0
                    # best.pt must not keep an untrained class head (E116 kept epoch 1 by stroke F2 alone): F2 x class accuracy
                    fitness = bst["f2"] * (0.5 * acc + 0.5 * macro)
                    rec.update({"cls_acc": round(acc, 4), "cls_macro": round(macro, 4), "fitness": round(fitness, 4)})
                    print("       class  acc %.3f  macro %.3f | " % (acc, macro)
                          + "  ".join(f"{k} {v:.2f}" for k, v in per.items()) + "  (true strokes, by length)"
                          + "   fitness F2 x class %.4f" % fitness, flush=True)
                if region is not None:
                    from ..region_data import report_regions
                    diag = sorted(train_pages, key=lambda p: p.stem)[:3]
                    tq = []
                    evaluate(ema.module, diag, c, device, amp_dtype, piece_out=tq)
                    tr = report_regions(diag, tq, self.say, "train3")
                    rec["train_region_f2"] = tr["f2"]
                    rs = report_regions(val_pages, pv, self.say, "val")
                    fitness = rs["f2"]
                    rec.update(region_f2=fitness, region_precision=rs["precision"], region_recall=rs["recall"], fitness=fitness)
                if continuation is not None or stroke_context is not None or hard_pairs:
                    diag = sorted(train_pages, key=lambda p: p.stem)[:3]
                    train_scores = [predict_page(ema.module, p, c.window, device, c.val_batch, 1, amp_dtype) for p in diag]
                    train_rows = per_page(train_scores, [p.label for p in diag], [p.length for p in diag],
                                          [p.stem for p in diag], bst["threshold"])
                    rec["train3_f2"] = round(float(np.mean([r["f2"] for r in train_rows])), 4)
                    self.say(f"Stroke F2 train3 {rec['train3_f2']:.3f} / val {fitness:.3f} @ {bst['threshold']:.2f}")
                if len(val_pages) <= 12:  # small val (dev split): every page, to see which one collapses
                    for pg, sc, row in zip(val_pages, vscores, per_page(vscores, [q.label for q in val_pages],
                                                                         [q.length for q in val_pages],
                                                                         [q.stem for q in val_pages], bst["threshold"])):
                        r05 = prf(*length_counts(sc >= 0.5, pg.label, pg.length))
                        print(val_row(pg.stem.split("_gt")[0][:12], 1, len(pg), float(pg.length[pg.label > 0].sum()),
                                      row["precision"], row["recall"], row["f2"], r05["f2"], bst["threshold"]), flush=True)
                        rec[f"F2:{pg.stem}"] = round(row["f2"], 4)
                save_best = fitness > best
                save_finetuned = False
                if hard_pairs:
                    delta_f2 = fitness - hard_baseline["macro_f2"]
                    delta_recall = bst["recall"] - hard_baseline["recall"]
                    recall_ok = delta_recall >= -MAX_RECALL_DROP
                    passes = bool(epoch_pairs) and delta_f2 >= MIN_F2_GAIN and recall_ok
                    rec.update(delta_macro_f2=delta_f2, delta_recall=delta_recall, hard_pair_goal=passes)
                    self.say(f"GT fixed @{THRESHOLD:.2f}: mean PDF F2 {fitness:.4f} ({delta_f2:+.4f} vs E114), "
                             f"micro R {bst['recall']:.4f} ({delta_recall:+.4f}); "
                             f"pilot gate {'PASS' if passes else 'not met'}")
                    key = (recall_ok, fitness)
                    save_finetuned = bool(epoch_pairs) and key > trained_key
                    save_best = save_best and passes
                if save_best or save_finetuned:
                    checkpoint = {"model": ema.module.state_dict(), "config": ema.module.config, "epoch": epoch,
                                  "threshold": bst["threshold"], "val_f2": bst["f2"], "fitness": fitness,
                                  "feature_names": FEATURE_NAMES, "args": vars(c)}
                    if save_finetuned:
                        checkpoint["recall_ok"] = recall_ok
                        pending = out / "best_finetuned.pt.tmp"
                        torch.save(checkpoint, pending)
                        pending.replace(out / "best_finetuned.pt")
                        trained_key, trained_epoch = key, epoch + 1
                        self.say(f"best_finetuned.pt <- epoch {trained_epoch}")
                    if save_best:
                        best, best_epoch = fitness, epoch
                        torch.save(checkpoint, out / "best.pt")
                        self.log(f"best.pt <- epoch {epoch + 1} (val F2 {bst['f2']:.4f} @ {bst['threshold']:.2f}, fitness {best:.4f})")
            epochs_trained = epoch + 1
            rng = {"rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None} if fixed_schedule else {}
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), **rng, "epoch": epoch,
                        "iter": it, "best": best, "best_epoch": best_epoch, "config": model.config, "args": vars(c)}, last)
            history.append(rec)
            append_results(out, rec)
            plot_results(out)
            self.log(f"epoch {epoch + 1}: " + json.dumps(rec))
            if mae_ratio <= 0 and c.patience and best_epoch >= 0 and epoch - best_epoch >= c.patience:
                self.say(f"Stopping early: no better val score (F2, x class accuracy with a class head) for {c.patience} epochs (best {best:.4f} at epoch "
                         f"{best_epoch + 1}).")
                break
        if hard_pairs:
            chosen = f"epoch {trained_epoch}" if trained_epoch is not None else "none (no successful paired steps)"
            self.say(f"Hard-pair pilot GT gate {'PASS' if best_epoch >= 0 else 'not met'}; "
                     f"best trained checkpoint {chosen}, best.pt epoch {best_epoch + 1}")
        return self.finish_fold(fold, model, [] if mae_ratio > 0 else val_pages, out, device, amp_dtype, best,
                                best_epoch, nparams, epochs_trained, best_finetuned_epoch=trained_epoch)

    def finish_fold(self, fold, model, val_pages, out, device, amp_dtype, best, best_epoch, nparams, epochs_trained=None,
                    best_finetuned_epoch=None):
        import torch

        from ..models import from_config, load_weights

        c = self.cfg
        result = {"fold": fold, "val": [p.stem for p in val_pages], "params": nparams, "best_f2": best,
                  "best_epoch": best_epoch + 1, "epochs_trained": epochs_trained}
        if getattr(c, "hard_pairs", False):
            result.update(best_finetuned_epoch=best_finetuned_epoch, hard_pair_goal=best_epoch >= 0)
        if not val_pages or not (out / "best.pt").exists():
            (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
            return result
        ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        m = from_config(ck["config"], weights_follow=True)
        load_weights(m, ck["model"])
        m.to(device)
        tta = 8 if c.tta else 1
        lscores = [] if getattr(m, "layer_mlp", None) is not None else None
        oscores = [] if getattr(m, "obj_mlp", None) is not None else None
        escores = [] if getattr(m, "edge_mlp", None) is not None else None
        pscores = [] if getattr(m, "piece", None) is not None else None
        scores, bst, at05 = evaluate(m, val_pages, c, device, amp_dtype, tta=tta, layer_out=lscores, obj_out=oscores, edge_out=escores,
                                     piece_out=pscores)
        thr = bst["threshold"]
        ck["threshold"] = thr
        ck["tta"] = tta
        torch.save(ck, out / "best.pt")
        labels = [p.label for p in val_pages]
        lengths = [p.length for p in val_pages]
        stems = [p.stem for p in val_pages]
        result.update({"tta": tta, "threshold": thr, "val_at_threshold": bst, "val_at_0.5": at05,
                       "pages": per_page(scores, labels, lengths, stems, thr),
                       "pages_at_0.5": per_page(scores, labels, lengths, stems, 0.5)})
        objs = {}
        if oscores is not None:  # E62 / OpenStroke29: per-shape class probabilities, from the same TTA pass as the stroke scores
            from ..objects import N_CLASSES
            for p, pc in zip(val_pages, oscores):
                objs[f"{p.stem}__oprob"] = pc[:, :N_CLASSES].astype(np.float16)
                objs[f"{p.stem}__ooff"] = pc[:, N_CLASSES:].astype(np.float32)
        if escores:   # OpenStroke30 / F169: same-object logit of every linked stroke pair, as payload shape ids
            for p, (e, z) in zip(val_pages, escores):
                objs[f"{p.stem}__eij"] = p.index[e].astype(np.int32)
                objs[f"{p.stem}__elog"] = z.astype(np.float16)
        if pscores:   # OpenStroke34: piece outputs (npz piece rows; research/probes/piece_object_probe.py builds objects from them)
            for p, q in zip([p for p in val_pages if p.pieces is not None], pscores):
                if "region_id" in q:
                    objs[f"{p.stem}__region_id"] = q["region_id"].astype(np.int32)
                    objs[f"{p.stem}__region_cls"] = q["region_cls"].astype(np.int16)
                objs[f"{p.stem}__pid"] = q["pid"].astype(np.int32)
                objs[f"{p.stem}__pfg"] = q["fg"].astype(np.float16)
                objs[f"{p.stem}__pcls"] = q["cls"].astype(np.float16)
                objs[f"{p.stem}__pe"] = q["pe"].astype(np.int32)
                objs[f"{p.stem}__plink"] = q["plink"].astype(np.float16)
        np.savez_compressed(out / "val_scores.npz", **{s: sc.astype(np.float32) for s, sc in zip(stems, scores)},
                            **{f"{p.stem}__index": p.index for p in val_pages}, **objs,
                            **({f"{s}__layer": ls.astype(np.float32) for s, ls in zip(stems, lscores)} if lscores else {}))
        if getattr(getattr(m, "piece", None), "region", None) is None:
            (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        self.say(f"\nFold {fold + 1} best.pt (epoch {best_epoch + 1}){' + TTA x8' if tta > 1 else ''}, threshold {thr:.2f}")
        rows = [[p["stem"], f"{p['recall']:.3f}", f"{p['precision']:.3f}", f"{p['f2']:.3f}",
                 f"{p['true_length_share']:.1%}", f"{p['pred_length_share']:.1%}"] for p in result["pages"]]
        rows.append(["all", f"{bst['recall']:.3f}", f"{bst['precision']:.3f}", f"{bst['f2']:.3f}", "", ""])
        self.say(table(rows, ["page", "R", "P", "F2", "true", "pred"]))
        if "macro_f2" in bst:
            self.say(f"GT validation mean PDF F2 {bst['macro_f2']:.3f}, worst PDF {bst['worst_f2']:.3f}")
        obj = {}
        if not getattr(c, "gt_only", False) or escores is not None or pscores is not None:
            obj = self.object_report(val_pages, scores, thr, escores, pscores)
        if getattr(getattr(m, "piece", None), "region", None) is not None:
            from ..region_data import report_regions
            obj["regions"] = report_regions(val_pages, pscores, self.say, "best val", details=True)
            from ..region_data import live_report
            obj["live_regions"] = live_report(m, val_pages, scores, Path(c.data), c.window, c.val_batch, device, amp_dtype, self.say)
        if obj:
            result["objects"] = obj
            (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        return result

    def epoch_objects(self, val_pages, scores, thr, escores, pscores, rec: dict):
        """One line per epoch: object F2 (GT / Viewer) of the stroke and piece protocols (openstroke.object_eval); best.pt still follows
        stroke F2 (the pre-registered selection)."""
        from .. import object_eval as oe

        parts = []
        for name, per in (("strokes", {p.stem: oe.stroke_counts(p, s, thr, escores[i] if escores else None)
                                        for i, (p, s) in enumerate(zip(val_pages, scores))}),
                          ("pieces", {p.stem: oe.piece_counts(p, q) for p, q in
                                      zip([p for p in val_pages if p.pieces is not None], pscores or [])})):
            per = {k: v for k, v in per.items() if v is not None}
            if not per:
                continue
            s = oe.summarize(per)
            parts.append(f"{name} " + "  ".join(f"{g} {v['f2']:.3f} (P {v['precision']:.2f} R {v['recall']:.2f})"
                                               for g, v in s.items() if g != "all"))
            rec.update({f"obj_{name}_{g}": v["f2"] for g, v in s.items()})
        if parts:
            print("     objects  " + " | ".join(parts), flush=True)

    def object_report(self, val_pages, scores, thr, escores, pscores) -> dict:
        """Object F2 from the model alone (openstroke.object_eval), printed in the log so no file has to leave Colab (user 2026-11-19)."""
        from .. import object_eval as oe

        res, by_stroke, by_piece = {}, {}, {}
        for i, p in enumerate(val_pages):
            c = oe.stroke_counts(p, scores[i], thr, escores[i] if escores else None)
            if c is not None:
                by_stroke[p.stem] = c
        if pscores:
            for p, q in zip([p for p in val_pages if p.pieces is not None], pscores):
                c = oe.piece_counts(p, q)
                if c is not None:
                    by_piece[p.stem] = c
        for name, per_page, what in (
                ("strokes", by_stroke, f"strokes p >= {thr:.2f} joined by same-object links >= {oe.LINK}, matched without class"),
                ("pieces", by_piece, f"pieces p >= {oe.PIECE_THR} joined by links >= {oe.LINK} within one predicted class, matched with class "
                                     "on GT pages")):
            if not per_page:
                continue
            s = oe.summarize(per_page)
            res[name] = {"groups": s, "pages": {k: [int(x) for x in v] for k, v in per_page.items()}}
            self.say(f"\nObject F2 ({name}): {what}; IoU >= {oe.MATCH_IOU} on length, model only (no AI_Takeoff logic)")
            rows = [[g, v["pages"], f"{v['recall']:.3f}", f"{v['precision']:.3f}", f"{v['f2']:.3f}", v["matched"], v["predicted"], v["gt"]]
                    for g, v in s.items()]
            rows += [[k, "", f"{oe.f2(v)[2]:.3f}", f"{oe.f2(v)[1]:.3f}", f"{oe.f2(v)[0]:.3f}", int(v[0]), int(v[1]), int(v[2])]
                     for k, v in sorted(per_page.items(), key=lambda kv: oe.f2(kv[1])[0])]
            self.say(table(rows, ["group / page", "pages", "R", "P", "F2", "match", "pred", "gt"]))
            for g, v in s.items():
                self.say(f"OBJECTS {name} {g}: F2 {v['f2']:.3f} P {v['precision']:.3f} R {v['recall']:.3f}")
        if by_piece:   # which step loses objects: one model output at a time replaced by its label (scoring only)
            pp = [(p, q) for p, q in zip([p for p in val_pages if p.pieces is not None], pscores)]
            rows, diag = [], {}
            for arm, kw in oe.PIECE_DIAG:
                per = {p.stem: c for p, q in pp if (c := oe.piece_counts(p, q, **kw)) is not None}
                s = oe.summarize(per)
                diag[arm] = s
                rows.append([arm] + [f"{s[g]['f2']:.3f} (P {s[g]['precision']:.2f} R {s[g]['recall']:.2f}, {s[g]['predicted']}/{s[g]['gt']})"
                                     if g in s else "" for g in ("GT", "Viewer")])
            res["pieces"]["diagnosis"] = diag
            self.say("\nPieces diagnosis (labels replace one model output at a time; for reading only, not a result)")
            self.say(table(rows, ["arm", "GT F2 (P R, pred/gt)", "Viewer F2 (P R, pred/gt)"]))
        return res


def summarize(results: dict[int, dict], pages: dict[str, Page], out: Path, say) -> dict:
    """Micro over every fold's validation pages (each at its fold's threshold, and at 0.5)."""

    summary = {}
    for key, thr_of in (("at_fold_threshold", lambda r: r["threshold"]), ("at_0.5", lambda r: 0.5)):
        acc: dict[str, list[float]] = {}
        for f, r in results.items():
            fo = out / f"fold{f}"
            if "threshold" not in r or not (fo / "val_scores.npz").exists():
                continue
            z = np.load(fo / "val_scores.npz")
            for s in r["val"]:
                p = pages[s]
                tp, fp, fn = length_counts(z[s] >= thr_of(r), p.label, p.length)
                for k in ("all", p.mode):
                    a = acc.setdefault(k, [0.0, 0.0, 0.0])
                    a[0] += tp
                    a[1] += fp
                    a[2] += fn
        summary[key] = {k: prf(*v) for k, v in acc.items()}
    rows = []
    for f, r in sorted(results.items()):
        if "val_at_threshold" in r:
            v = r["val_at_threshold"]
            rows.append([f"fold {f + 1}", len(r["val"]), r["best_epoch"], f"{r['threshold']:.2f}", f"{v['recall']:.3f}",
                         f"{v['precision']:.3f}", f"{v['f2']:.3f}"])
    for k, v in summary.get("at_fold_threshold", {}).items():
        rows.append([k, "", "", "", f"{v['recall']:.3f}", f"{v['precision']:.3f}", f"{v['f2']:.3f}"])
    if not rows:  # a pretraining run (E19) scores nothing
        say("\nno scored validation (pretraining run: fine-tune it with model=<this run folder>)")
        return summary
    say("\nCross-validation summary (micro over every fold's val pages, length-weighted)")
    say(table(rows, ["", "pages", "best_ep", "thr", "R", "P", "F2"]))
    for k, v in summary.get("at_0.5", {}).items():
        say(f"SUMMARY at_0.5 {k}: R {v['recall']:.3f} P {v['precision']:.3f} F2 {v['f2']:.3f}")
    for k, v in summary.get("at_fold_threshold", {}).items():
        say(f"SUMMARY {k}: R {v['recall']:.3f} P {v['precision']:.3f} F2 {v['f2']:.3f}")
    worst = sorted((p for r in results.values() for p in r.get("pages", [])), key=lambda p: p["f2"])
    if worst:
        say("Pages by F2 (worst first):")
        say(table([[p["stem"], pages[p["stem"]].mode, f"{p['recall']:.3f}", f"{p['precision']:.3f}", f"{p['f2']:.3f}",
                    f"{p['true_length_share']:.1%}", f"{p['pred_length_share']:.1%}"] for p in worst],
                  ["page", "mode", "R", "P", "F2", "true", "pred"]))
    return summary
