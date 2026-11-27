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

from ..data import (Augment, Page, WindowDataset, coincident_weights, collate, describe, dihedral, eval_windows, page_cohort, raw_items,
                    viewer_isolated_weights, window)
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
        slow = slow_image and n.startswith("img.") and not n.startswith(("img.proj", "img.stroke_context.", "img.cad_stem."))
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


def _worker_init_fn(_worker_id: int) -> None:
    """Prevent DataLoader CPU workers from spawning nested OpenMP/MKL thread pools that thrash the host CPU."""
    import torch

    torch.set_num_threads(1)


def to_device(b, device):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


def _effective_dtype(model, amp_dtype):
    """Autocast dtype of an inference forward: None = fp32. A model whose fp16 outputs overflowed once stays in fp32."""
    import torch

    if amp_dtype == torch.float16 and getattr(model, "_fp16_unsafe", False):
        return None
    return amp_dtype


def _check_finite(model, raw_z) -> None:
    """fp16 inference: inf / NaN in the outputs (bf16-trained weights can overflow fp16) -> NonFiniteOutput (redo in fp32)."""
    import torch

    from ..gpu import NonFiniteOutput

    outs = [raw_z]
    h = getattr(model, "last_h", None)
    if h is not None:
        outs.append(h)
    e = getattr(model, "last_edge", None)
    if e is not None:
        outs.append(e[0])
    o = getattr(model, "last_obj", None)
    if o is not None:
        outs.append(o)
    if not all(bool(torch.isfinite(t).all()) for t in outs if torch.is_tensor(t) and t.is_floating_point()):
        raise NonFiniteOutput("fp16 forward produced inf / NaN")


def score_items(model, page: Page, items: list, window_size: int, device, amp_dtype=None, use_layer: bool = False, use_obj: bool = False,
                edge_acc: list | None = None, piece_acc: list | None = None, group_acc: list | None = None,
                prebuilt_b: dict | None = None):
    """One batch of (window start, dihedral view) -> token positions, their probabilities, their layer probabilities (or None)."""

    import torch

    from ..models.openstroke2 import topo_mode_of

    with torch.inference_mode():
        if prebuilt_b is not None:
            b = to_device(prebuilt_b, device) if device.type == "cuda" else prebuilt_b
        else:
            ws = [dihedral(window(page, s, window_size), k) if k else window(page, s, window_size) for s, k in items]
            t_mode = topo_mode_of(model)
            if device.type == "cuda" and not t_mode:
                b = build_batch(ws, device)
            else:
                b = to_device(collate(ws, topo_mode=t_mode), device)
        dtype = _effective_dtype(model, amp_dtype)
        with torch.autocast(device_type=device.type, dtype=dtype or torch.float32,
                            enabled=device.type == "cuda" and dtype is not None):
            raw_z = model(b).float()
        if dtype == torch.float16:
            _check_finite(model, raw_z)
        if raw_z.ndim == 3 and raw_z.shape[-1] > 1:
            scls_p = torch.softmax(raw_z, dim=-1)
            prob = (1.0 - scls_p[..., 0]).cpu().numpy()
            scls_np = scls_p.cpu().numpy()
        else:
            prob = torch.sigmoid(raw_z).cpu().numpy()
            scls_np = None
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
        if group_acc is not None and getattr(model, "group_head", None) is not None and getattr(model, "last_h", None) is not None:
            hf = model.last_h.float()
            scls = torch.softmax(model.group_head.forward_strokes(hf), -1).half().cpu().numpy()
            hf_cpu = hf.half().cpu().numpy()
            # Keep per-window slices so predict_page uses C-vectorized direct indexing instead of slow np.add.at (~3x faster!)
            group_acc.append([(pos[bi][m[bi]], hf_cpu[bi][m[bi]], scls[bi][m[bi]]) for bi in range(len(items))])
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
    return pos[m], prob[m], tok, cls, (scls_np[m] if scls_np is not None else None)


def predict_page(model, page: Page, window_size: int, device, batch: int = 4, tta: int = 1, amp_dtype=None,
                 layer_out: list | None = None, on_batch=None, obj_out: list | None = None, edge_out: list | None = None,
                 piece_out: list | None = None, group_out: list | None = None, cls_out: list | None = None):
    """Probabilities for every token of ``page``: windows with stride window/2 and ``tta`` dihedral views,
    all averaged. With a layer head and ``layer_out`` given, also appends each token's layer probability (its layer's
    logit in each window it was in, averaged the same way). For multi-class stroke models (``stroke_classes > 1``),
    returns foreground probability ``1 - p_other`` and appends full ``(N, C)`` class probabilities into ``cls_out``.

    ``amp_dtype``: autocast dtype on CUDA, None = fp32 (see ``openstroke.gpu.amp_dtype``). fp16 outputs with inf / NaN
    (bf16-trained weights overflowing on a GPU without bf16, e.g. Turing) redo the page in fp32, and the model stays in
    fp32 from then on (the output lists are only appended once the whole page is done)."""
    import torch

    args = (model, page, window_size, device, batch, tta)
    outs = dict(layer_out=layer_out, on_batch=on_batch, obj_out=obj_out, edge_out=edge_out, piece_out=piece_out,
                group_out=group_out, cls_out=cls_out)
    from ..gpu import NonFiniteOutput

    try:
        return _predict_page(*args, _effective_dtype(model, amp_dtype), **outs)
    except NonFiniteOutput:
        if amp_dtype != torch.float16:
            raise
        model._fp16_unsafe = True
        import warnings

        warnings.warn("fp16 inference overflowed (inf / NaN): this model now runs in fp32 on this GPU", stacklevel=2)
        return _predict_page(*args, None, **outs)


def _predict_page(model, page: Page, window_size: int, device, batch: int = 4, tta: int = 1, amp_dtype=None,
                  layer_out: list | None = None, on_batch=None, obj_out: list | None = None, edge_out: list | None = None,
                  piece_out: list | None = None, group_out: list | None = None, cls_out: list | None = None):

    total = np.zeros(len(page), np.float64)
    count = np.zeros(len(page), np.float64)
    ltotal = np.zeros(len(page), np.float64)
    use_layer = layer_out is not None and getattr(model, "layer_mlp", None) is not None
    use_obj = obj_out is not None and getattr(model, "obj_mlp", None) is not None
    use_group = group_out is not None and getattr(model, "group_head", None) is not None
    use_cls = cls_out is not None and int(getattr(model, "stroke_classes", 1)) > 1
    ctotal = None
    sctotal = None
    htotal = None
    stotal = None
    items = [(s, k) for s in eval_windows(page, window_size) for k in range(tta)]
    edge_acc = [] if edge_out is not None and getattr(model, "edge_mlp", None) is not None else None
    piece_acc = [] if piece_out is not None and getattr(model, "piece", None) is not None and page.pieces is not None else None
    model.eval()
    n_page = len(page)
    batches = [items[a:a + batch] for a in range(0, len(items), batch)]
    # On CUDA multi-window pages (e.g. val / TTA x8 / large sheets), overlap NEXT batch's CPU collate+topo_feat
    # on a background thread while the GPU executes the current batch forward (exact parity, hides CPU latency!).
    use_prefetch = (device.type == "cuda") and len(batches) > 1
    if use_prefetch:
        from concurrent.futures import ThreadPoolExecutor
        from ..models.openstroke2 import topo_mode_of

        t_mode = topo_mode_of(model)

        def _prep(sub_items):
            ws = [dihedral(window(page, s, window_size), k) if k else window(page, s, window_size) for s, k in sub_items]
            b_cpu = collate(ws, topo_mode=t_mode)
            return {k: (v.pin_memory() if hasattr(v, "pin_memory") else v) for k, v in b_cpu.items()}

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_prep, batches[0])
            done = 0
            for bi, sub in enumerate(batches):
                prebuilt = fut.result()
                if bi + 1 < len(batches):
                    fut = ex.submit(_prep, batches[bi + 1])
                g_batch = [] if use_group else None
                pos, prob, tok, cls, scls = score_items(
                    model, page, sub, window_size, device, amp_dtype, use_layer, use_obj, edge_acc,
                    piece_acc, group_acc=g_batch, prebuilt_b=prebuilt,
                )
                total += np.bincount(pos, weights=prob, minlength=n_page)
                count += np.bincount(pos, minlength=n_page)
                if use_cls and scls is not None:
                    if sctotal is None:
                        sctotal = np.zeros((n_page, scls.shape[1]), np.float64)
                    np.add.at(sctotal, pos, scls)
                if use_layer:
                    ltotal += np.bincount(pos, weights=tok, minlength=n_page)
                if use_obj:
                    if ctotal is None:
                        ctotal = np.zeros((len(page), cls.shape[1]), np.float64)
                    np.add.at(ctotal, pos, cls)
                if use_group and g_batch:
                    for p_w, h_w, s_w in g_batch[0]:
                        if not len(p_w):
                            continue
                        if htotal is None:
                            htotal = np.zeros((n_page, h_w.shape[1]), np.float32)
                            stotal = np.zeros((n_page, s_w.shape[1]), np.float32)
                        htotal[p_w] += h_w
                        stotal[p_w] += s_w
                done += len(sub)
                if on_batch is not None:
                    on_batch(done, len(items))
    else:
        for a in range(0, len(items), batch):
            g_batch = [] if use_group else None
            pos, prob, tok, cls, scls = score_items(model, page, items[a:a + batch], window_size, device, amp_dtype, use_layer, use_obj, edge_acc,
                                                    piece_acc, group_acc=g_batch)
            total += np.bincount(pos, weights=prob, minlength=n_page)
            count += np.bincount(pos, minlength=n_page)
            if use_cls and scls is not None:
                if sctotal is None:
                    sctotal = np.zeros((n_page, scls.shape[1]), np.float64)
                np.add.at(sctotal, pos, scls)
            if use_layer:
                ltotal += np.bincount(pos, weights=tok, minlength=n_page)
            if use_obj:   # per-shape class probabilities: averaged over windows and views like the stroke probability
                if ctotal is None:
                    ctotal = np.zeros((len(page), cls.shape[1]), np.float64)
                np.add.at(ctotal, pos, cls)
            if use_group and g_batch:
                for p_w, h_w, s_w in g_batch[0]:
                    if not len(p_w):
                        continue
                    if htotal is None:
                        htotal = np.zeros((n_page, h_w.shape[1]), np.float32)
                        stotal = np.zeros((n_page, s_w.shape[1]), np.float32)
                    htotal[p_w] += h_w
                    stotal[p_w] += s_w
            if on_batch is not None:
                on_batch(min(a + batch, len(items)), len(items))
    if use_cls:
        nc = int(getattr(model, "stroke_classes", 3))
        if sctotal is not None:
            cls_out.append((sctotal / np.maximum(count, 1)[:, None]).astype(np.float32))
        else:
            z0 = np.zeros((n_page, nc), np.float32)
            z0[:, 0] = 1.0
            cls_out.append(z0)
    if use_layer:
        layer_out.append(ltotal / np.maximum(count, 1))
    if use_obj:   # (n, N_CLASSES + 2): class probabilities, then the offset to the object centre in pt
        obj_out.append(ctotal / np.maximum(count, 1)[:, None] if ctotal is not None else np.zeros((len(page), 0)))
    if use_group:
        denom = np.maximum(count, 1.0).astype(np.float32)[:, None]
        if htotal is not None:
            group_out.append(((htotal / denom).astype(np.float16), (stotal / denom).astype(np.float16)))
        else:
            C = getattr(model.group_head, "dim", 64)
            K = getattr(model.group_head, "n_classes", 8)
            group_out.append((np.zeros((len(page), C), np.float16), np.zeros((len(page), K), np.float16)))
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
    if not len(a):
        return np.zeros((0, 2), np.int64), np.zeros(0, np.float32)
    b = np.concatenate([x[1] for x in acc]).astype(np.int64)
    z = np.concatenate([x[2] for x in acc]).astype(np.float64)
    raw_k = a * n + b
    order = np.argsort(raw_k, kind="quicksort")
    sk = raw_k[order]
    chg = np.empty(len(sk), dtype=bool)
    chg[0] = True
    if len(sk) > 1:
        chg[1:] = sk[1:] != sk[:-1]
    starts = np.flatnonzero(chg)
    key = sk[starts]
    cnt = np.empty(len(starts), dtype=np.float64)
    cnt[:-1] = starts[1:] - starts[:-1]
    cnt[-1] = len(sk) - starts[-1]
    mean = np.add.reduceat(z[order], starts) / cnt
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
             edge_out: list | None = None, piece_out: list | None = None, group_out: list | None = None,
             cls_out: list | None = None):
    scores = [predict_page(model, p, cfg.window, device, cfg.val_batch, tta, amp_dtype, layer_out, obj_out=obj_out, edge_out=edge_out,
                           piece_out=piece_out if p.pieces is not None else None, group_out=group_out, cls_out=cls_out)
              for p in pages]
    equal_pages = getattr(cfg, "gt_only", False) and not any(
        getattr(model, key, None) is not None for key in ("obj_mlp", "edge_mlp", "piece", "layer_mlp", "group_head"))
    labels, lengths = [p.label for p in pages], [p.length for p in pages]
    if getattr(cfg, "hard_pairs", False):
        from ..hard_pairs import THRESHOLD

        fixed, at05 = sweep(scores, labels, lengths, thresholds=np.array([THRESHOLD, 0.5]), equal_pages=True)["curve"]
        return scores, fixed, at05
    select = (getattr(cfg, "protocol", None) or {}).get("select")
    if select:   # a dataset protocol (F213): threshold and best.pt from the mean F2 of the selected cohorts
        best, at05 = cohort_select(scores, pages, select)
        return scores, best, at05
    sw = sweep(scores, labels, lengths, equal_pages=equal_pages)
    at05 = next(r for r in sw["curve"] if abs(r["threshold"] - 0.5) < 1e-9)
    best_row = sw["best"]
    # F249: When select_by=object on a flat F2 plateau (e.g. E151 F2=0.943..0.945 across thr=0.45..0.75), break ties
    # within 0.003 of peak F2 by maximizing Stroke F1 (higher Precision without sacrificing F2) so late epochs land at
    # their true operating threshold (~0.70) instead of randomly picking thr=0.45 and getting penalized vs early epochs.
    if str(getattr(cfg, "select_by", "stroke")) == "object" and sw["curve"]:
        peak_f2 = float(best_row["f2"])
        def _f1_key(r):
            p_r, r_r = float(r["precision"]), float(r["recall"])
            f1_r = (2.0 * p_r * r_r / (p_r + r_r)) if (p_r + r_r) > 0.0 else 0.0
            return (f1_r, float(r["f2"]), float(r["threshold"]))
        plateau = [r for r in sw["curve"] if float(r["f2"]) >= peak_f2 - 0.003]
        if plateau:
            best_row = max(plateau, key=_f1_key)
    return scores, best_row, at05


def compute_object_fitness(bst: dict, rec: dict) -> float:
    """F245 / F249: Precision- & object-aware composite fitness for ``select_by=object`` (fixes E150/E151 early-epoch freeze).

    Formula: ``0.25 * Stroke_F1 + 0.15 * Stroke_F2 + 0.25 * PureObj_mx + 0.35 * PureObj_6cls``
    (gracefully falls back when ``mx`` or ``6cls`` is absent for older architectures).
    """
    p = float(bst.get("precision", 0.0))
    r = float(bst.get("recall", 0.0))
    f2 = float(bst.get("f2", 0.0))
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0.0 else f2

    mx = None
    if "obj_mx_GT" in rec and "obj_mx_Viewer" in rec:
        mx = 0.5 * (float(rec["obj_mx_GT"]) + float(rec["obj_mx_Viewer"]))
    elif "obj_mx_all" in rec:
        mx = float(rec["obj_mx_all"])

    c6 = None
    if "obj_6cls_GT" in rec and "obj_6cls_Viewer" in rec:
        c6 = 0.5 * (float(rec["obj_6cls_GT"]) + float(rec["obj_6cls_Viewer"]))
    elif "obj_6cls_all" in rec:
        c6 = float(rec["obj_6cls_all"])

    if mx is not None and c6 is not None:
        return 0.25 * f1 + 0.15 * f2 + 0.25 * mx + 0.35 * c6
    if mx is not None:
        return 0.35 * f1 + 0.25 * f2 + 0.40 * mx
    if c6 is not None:
        return 0.35 * f1 + 0.25 * f2 + 0.40 * c6
    return 0.55 * f1 + 0.45 * f2


def cohort_groups(pages) -> dict[str, str] | None:
    """{stem: cohort} when the val pages are not all HVAC (fire1 / thin1, F213), else None (the GT / Viewer groups of HVAC logs)."""
    if all(p.mode == "hvac" for p in pages):
        return None
    return {p.stem: page_cohort(p.mode, p.stem) for p in pages}


def cohort_select(scores, pages, select) -> tuple[dict, dict]:
    """F213: length-micro stroke F2 per cohort (data.page_cohort) on one threshold grid; the threshold maximises the mean F2 of the
    ``select`` cohorts present (each cohort weighs the same, however many pages or strokes it has). -> (best, at 0.5), each with
    precision / recall / f2 = the means over the selected cohorts and ``cohorts`` = {cohort: its row, pages, strokes, true length}."""
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(pages):
        groups.setdefault(page_cohort(p.mode, p.stem), []).append(i)
    curves = {g: sweep([scores[i] for i in idx], [pages[i].label for i in idx], [pages[i].length for i in idx])["curve"]
              for g, idx in groups.items()}
    sel = [g for g in select if g in curves] or sorted(curves)
    n = len(next(iter(curves.values())))

    def row(j):
        mean = {k: float(np.mean([curves[g][j][k] for g in sel])) for k in ("precision", "recall", "f2")}
        return {"threshold": curves[sel[0]][j]["threshold"], **mean, "selected": sel,
                "cohorts": {g: {**curves[g][j], "pages": len(groups[g]), "shapes": sum(len(pages[i]) for i in groups[g]),
                                "true_length": float(sum(pages[i].length[pages[i].label > 0].sum() for i in groups[g]))}
                            for g in sorted(curves)}}

    j = max(range(n), key=lambda j: np.mean([curves[g][j]["f2"] for g in sel]))
    j05 = next(k for k in range(n) if abs(curves[sel[0]][k]["threshold"] - 0.5) < 1e-9)
    return row(j), row(j05)


def stroke_class_weights(pages, n_classes: int, pos_weight: float = 2.0) -> list[float]:
    """Dynamic length-balanced foreground class weights for K-class stroke CE (0=other -> 1.0, 1..K-1 balanced
    by inverse sqrt of true length share, scaled around 1.5*pos_weight and clamped to [max(pos_weight, 1.5),
    max(pos_weight*3, 4.5)] so rare backbone classes like FIRE main never drown under frequent branch strokes)."""
    k = max(int(n_classes), 2)
    tot = np.zeros(k, dtype=np.float64)
    for p in pages:
        y = np.clip(p.label.astype(np.int64), 0, k - 1)
        tot += np.bincount(y, weights=np.maximum(p.length.astype(np.float64), 0.0), minlength=k)[:k]
    w = np.ones(k, dtype=np.float64)
    fg = tot[1:]
    seen = fg > 0
    if seen.any():
        inv = np.where(seen, 1.0 / np.sqrt(np.maximum(fg, 1.0)), 0.0)
        inv[seen] = inv[seen] / inv[seen].mean()
        base = max(float(pos_weight) * 1.5, 1.5)
        lo, hi = max(float(pos_weight), 1.25), max(float(pos_weight) * 3.0, 4.5)
        w[1:] = np.where(seen, np.clip(inv * base, lo, hi), base)
    else:
        w[1:] = max(float(pos_weight) * 1.5, 1.5)
    return w.astype(np.float32).tolist()


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
    with torch.inference_mode():
        for a in range(0, len(starts), batch):
            ws = [window(page, s, window_size) for s in starts[a:a + batch]]
            b = build_batch(ws, device) if device.type == "cuda" else to_device(collate(ws), device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32,
                                enabled=device.type == "cuda" and amp_dtype is not None):
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
        from .. import gpu

        c = self.cfg
        # any GPU (openstroke.gpu): a build without kernels for it -> CPU; bf16 where native (sm_80+), fp16 + GradScaler
        # on Volta / Turing (T4, RTX 20xx), fp32 below; TF32 only where it exists; cuDNN autotuning on for training
        dev = gpu.resolve_device(c.device)
        if dev.type == "cuda":
            gpu.configure(dev, train=True)
        else:
            from ..infer import _configure_cpu_runtime

            _configure_cpu_runtime()
        amp_dtype = gpu.amp_dtype(dev, train=True)
        return dev, amp_dtype, dev.type == "cuda" and c.amp and amp_dtype is not None

    def build_model(self, train_pages, init=None):
        import torch

        from ..models import build, from_config, load_weights

        c = self.cfg
        if init is not None:
            ck = torch.load(init, map_location="cpu", weights_only=False)
            cfg = dict(ck["config"])
            # heads asked for on the command line but absent from the checkpoint start fresh (E26: pretrained -> two heads)
            extra = {k: True for k in ("layer_head", "layer_head_names", "image_branch", "img_lum", "obj_head", "edge_head", "group_head", "facing_edges")
                     if getattr(c, k, False) and not cfg.get(k)}
            if getattr(c, "edge_pairs", False) and not cfg.get("edge_pairs"):   # OpenStroke31: same network, another link loss
                extra["edge_pairs"] = True
            if getattr(c, "piece_head", False) and not cfg.get("piece_head"):   # OpenStroke34: new piece head on a trained model
                extra["piece_head"] = True
            if getattr(c, "region_head", False) and not cfg.get("region_head"):
                extra.update(piece_head=True, region_head=True)
            if getattr(c, "stroke_context", False) and not cfg.get("stroke_context"):
                extra["stroke_context"] = True
            if getattr(c, "img_cad", False) and not cfg.get("img_cad") and not getattr(c, "img_cad_v2", False):
                extra["img_cad"] = True
            if getattr(c, "img_cad_v2", False) and not cfg.get("img_cad_v2"):
                extra["img_cad_v2"] = True
                extra["img_cad"] = False
                ck["model"] = {k: v for k, v in ck["model"].items() if not k.startswith("img.cad_stem.")}
            if getattr(c, "graph_topo", False) and not cfg.get("graph_topo") and not any(getattr(c, k, False) for k in ("graph_topo_v2", "graph_topo_v3", "graph_topo_fire", "graph_topo_uni")):
                extra["graph_topo"] = True
            if getattr(c, "graph_topo_v2", False) and not cfg.get("graph_topo_v2") and not any(getattr(c, k, False) for k in ("graph_topo_v3", "graph_topo_fire", "graph_topo_uni")):
                extra["graph_topo_v2"] = True
                extra["graph_topo"] = False
                ck["model"] = {k: v for k, v in ck["model"].items() if not k.startswith("topo_mlp.")}
            if getattr(c, "graph_topo_v3", False) and not cfg.get("graph_topo_v3") and not any(getattr(c, k, False) for k in ("graph_topo_fire", "graph_topo_uni")):
                extra["graph_topo_v3"] = True
                extra["graph_topo_v2"] = False
                extra["graph_topo"] = False
                ck["model"] = {k: v for k, v in ck["model"].items() if not k.startswith("topo_mlp.")}
            if getattr(c, "graph_topo_fire", False) and not cfg.get("graph_topo_fire") and not getattr(c, "graph_topo_uni", False):
                extra["graph_topo_fire"] = True
                extra["graph_topo_v3"] = False
                extra["graph_topo_v2"] = False
                extra["graph_topo"] = False
                ck["model"] = {k: v for k, v in ck["model"].items() if not k.startswith("topo_mlp.")}
            if getattr(c, "graph_topo_uni", False) and not cfg.get("graph_topo_uni"):
                extra["graph_topo_uni"] = True
                extra["graph_topo_fire"] = False
                extra["graph_topo_v3"] = False
                extra["graph_topo_v2"] = False
                extra["graph_topo"] = False
                ck["model"] = {k: v for k, v in ck["model"].items() if not k.startswith("topo_mlp.")}
            if getattr(c, "edge_style", False) and not cfg.get("edge_style"):
                extra["edge_style"] = True
            for hk in ("edge_head", "edge_pairs", "group_head", "obj_head", "piece_head", "region_head"):
                if cfg.get(hk) and not getattr(c, hk, False):
                    extra[hk] = False
            sc_req = max(int(getattr(c, "stroke_classes", 1) or 1), 1)
            if sc_req != int(cfg.get("stroke_classes", 1) or 1):
                extra["stroke_classes"] = sc_req
                ck["model"] = {k: v for k, v in ck["model"].items() if not k.startswith("head.3.")}
            cn_req = getattr(c, "class_names", None)
            if sc_req > 1 and cn_req and list(cn_req) != list(cfg.get("class_names") or []):
                extra["class_names"] = list(cn_req)
            elif sc_req == 1 and cfg.get("class_names"):
                extra["class_names"] = None
            if int(getattr(c, "img_max_px", 1536) or 1536) != int(cfg.get("img_max_px", 1536) or 1536):
                extra["img_max_px"] = int(c.img_max_px)
            if getattr(c, "continuation", False) and not cfg.get("continuation"):
                extra["continuation"] = True
            if int(getattr(c, "rel_attn", 0) or 0) > int(cfg.get("rel_attn", 0) or 0):   # OpenStroke30/44: zero-init extra graph attention blocks
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
                      continuation=bool(getattr(c, "continuation", False)),
                      stroke_context=bool(getattr(c, "stroke_context", False)),
                      group_head=bool(getattr(c, "group_head", False)),
                      facing_edges=bool(getattr(c, "facing_edges", False)),
                      img_cad=bool(getattr(c, "img_cad", False)),
                      img_cad_v2=bool(getattr(c, "img_cad_v2", False)),
                      graph_topo=bool(getattr(c, "graph_topo", False)),
                      graph_topo_v2=bool(getattr(c, "graph_topo_v2", False)),
                      graph_topo_v3=bool(getattr(c, "graph_topo_v3", False)),
                      graph_topo_fire=bool(getattr(c, "graph_topo_fire", False)),
                      graph_topo_uni=bool(getattr(c, "graph_topo_uni", False)),
                      edge_style=bool(getattr(c, "edge_style", False)),
                      stroke_classes=max(int(getattr(c, "stroke_classes", 1) or 1), 1),
                      class_names=getattr(c, "class_names", None),
                      # OpenStroke38: a fresh model may read the image with OpenOBB's backbone (img_weights = its weights)
                      img_backbone=str(getattr(c, "img_backbone", "resnet18") or "resnet18"),
                      img_weights=getattr(c, "img_weights", None))
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
        recoverable = getattr(c, "region_recoverable", False)
        if hard_pairs and device.type != "cuda":
            raise ValueError("hard_pairs requires a CUDA runtime for teacher scoring and training")
        model = self.build_model(train_pages, init).to(device)
        if device.type == "cuda":
            from ..gpu import channels_last_ok, describe as gpu_describe
            from ..models.image_branch import enable_image_channels_last

            if channels_last_ok(device, amp_dtype if amp else None):   # NHWC pays with half precision on tensor cores only
                enable_image_channels_last(model)
            self.say("device  " + gpu_describe(device, amp_dtype if amp else None))
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
        os37_pilot = stroke_context is not None and getattr(model, "group_head", None) is None and not model.config.get("facing_edges", False)
        fixed_schedule = region_pilot or continuation is not None or os37_pilot or hard_pairs
        pgs = param_groups(model, c.weight_decay, slow_image=not region_pilot)
        if device.type == "cuda":
            try:
                opt = torch.optim.AdamW(pgs, lr=c.lr0, betas=(0.9, 0.999), fused=True)
            except (TypeError, RuntimeError):
                opt = torch.optim.AdamW(pgs, lr=c.lr0, betas=(0.9, 0.999))
        else:
            opt = torch.optim.AdamW(pgs, lr=c.lr0, betas=(0.9, 0.999))
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
        model.config["arch"] = arch_name(model.config)   # the research version (openstroke28 / 29 / 30 / 41) saved in the checkpoint
        ema.module.config["arch"] = model.config["arch"]
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
                      paste_mode=str(getattr(c, "paste_mode", "region") or "region"),
                      sparse_bg=float(getattr(c, "sparse_bg", 0.0) or 0.0),
                      neg_boxes=float(getattr(c, "neg_boxes", 0.0) or 0.0))
        gpu_batch = (use_gpu_batch(c, device) and getattr(model, "piece", None) is None
                     and aug.sparse_bg <= 0 and aug.neg_boxes <= 0)   # pieces / sparse_bg / neg_boxes are augmented in numpy (data.augment)
        co = float(getattr(c, "coincide", 0) or 0)
        clean_iso = bool(getattr(c, "viewer_clean_iso", False))
        for p in train_pages:  # E36 coincide + OpenStroke42 Viewer V1 isolated-object hygiene (train pages only; val untouched)
            w_p = coincident_weights(p, co) if co > 0 else None
            if clean_iso:
                wi = viewer_isolated_weights(p)
                w_p = wi if w_p is None else (w_p * wi)
            p.lw = w_p
        if co > 0:
            n0 = sum(float(p.length[p.lw == 0].sum()) for p in train_pages if p.lw is not None)
            n1 = sum(float(p.length[p.label > 0].sum()) for p in train_pages)
            self.say(f"coincide {co}: {n0 / max(n1, 1e-9):.1%} of the true length of the train pages carries no loss")
        if clean_iso:
            vw_tr = [p for p in train_pages if str(p.stem).startswith("vw_")]
            n0 = sum(float(p.length[(p.label > 0) & (p.lw == 0)].sum()) for p in vw_tr if p.lw is not None)
            n1 = sum(float(p.length[p.label > 0].sum()) for p in vw_tr)
            self.say(f"viewer_clean_iso: {n0 / max(n1, 1e-9):.2%} of Viewer train true length (isolated fittings / small boxes) ignored (lw=0)")
        ds = WindowDataset(train_pages, c.steps * c.batch, c.window, aug, seed=c.seed + 1000 * fold,
                           plumbing_share=getattr(c, "plumbing_share", None), raw=gpu_batch,
                           viewer_share=getattr(c, "viewer_share", None),
                           pdf_by_stem=self.pdf_by_stem if getattr(c, "pdf_balance", False) else None,
                           cohort_share=(getattr(c, "protocol", None) or {}).get("sampling"))
        if (getattr(c, "protocol", None) or {}).get("sampling"):
            mass = {}
            for p, weight in zip(train_pages, ds.prob):
                g = page_cohort(p.mode, p.stem)
                mass[g] = [mass.get(g, [0, 0.0])[0] + 1, mass.get(g, [0, 0.0])[1] + float(weight)]
            self.say("sampling " + "  ".join(f"{g} {v[1]:.0%} ({v[0]} pages)" for g, v in sorted(mass.items())) + "  (dataset protocol)")
        if self.pdf_by_stem is not None:
            mass = {}
            for p, weight in zip(train_pages, ds.prob):
                key = self.pdf_by_stem[p.stem]
                mass[key] = mass.get(key, 0.0) + float(weight)
            self.say(f"PDF sampling: {len(mass)} train PDFs, min {min(mass.values()):.1%}, max {max(mass.values()):.1%}")
        from functools import partial
        from ..models.openstroke2 import topo_mode_of

        t_mode = topo_mode_of(model)
        col_fn = raw_items if gpu_batch else (partial(collate, topo_mode=t_mode) if t_mode else collate)
        dl = torch.utils.data.DataLoader(ds, batch_size=c.batch, shuffle=False, num_workers=c.workers,
                                         collate_fn=col_fn,
                                         pin_memory=device.type == "cuda" and not gpu_batch,
                                         persistent_workers=c.workers > 0,
                                         worker_init_fn=_worker_init_fn if c.workers > 0 else None,
                                         prefetch_factor=4 if c.workers else None)
        if fold == 0 or start_epoch == 0:
            t_note = f" + topo({t_mode}) in workers" if (t_mode and not gpu_batch) else ""
            self.say(f"batches built on {'the GPU (gpu_batch)' if gpu_batch else 'CPU workers' + t_note}")
        total_iters = (50 if fixed_schedule else c.epochs) * c.steps
        warmup = int(c.warmup_epochs * c.steps)
        mae_ratio = float(getattr(c, "pretrain", 0) or 0)  # E19: masked-shape pretraining instead of the label loss
        cols = ["mse"] if mae_ratio > 0 else (["bce", "lovasz"] if c.lovasz > 0 else ["bce"])
        scw = None
        if mae_ratio <= 0 and int(getattr(model, "stroke_classes", 1)) > 1:
            from ..metrics import resolve_class_names
            n_sc = int(model.stroke_classes)
            w_sc = stroke_class_weights(train_pages, n_sc, c.pos_weight)
            scw = torch.tensor(w_sc, device=device)
            cn = resolve_class_names(n_sc, model.config.get("class_names"))
            self.say("class_weights  " + "  ".join(f"{nm}={wt:.2f}" for nm, wt in zip(cn, w_sc)))
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
        group_w = float(getattr(c, "group_loss", 0.5) or 0) if mae_ratio <= 0 and getattr(model, "group_head", None) is not None else 0.0
        gcw = None
        if group_w > 0:
            from ..models.group_head import group_class_weights, group_cls_loss
            cols = cols + ["grp"]
            gcw = torch.tensor(group_class_weights(train_pages), device=device)
        if mae_ratio > 0:
            self.say(f"pretrain masked shapes: {mae_ratio:.0%} of tokens hidden, recon of their features (labels unused)")
        if ((getattr(c, "gt_only", False) or getattr(c, "protocol", None)) and init is not None and val_pages and mae_ratio == 0
                and (not c.resume or (hard_pairs and start_epoch == 0))):   # F213: also the start checkpoint on a protocol dataset
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
            for g, v in (initial.get("cohorts") or {}).items():
                self.say(f"  {g}: F2 {v['f2']:.3f} P {v['precision']:.3f} R {v['recall']:.3f} ({v['pages']} pages)")
            for row in by_page if len(by_page) <= 12 else []:   # every page is in before_finetune.json
                self.say(f"  {row['stem']} F2 {row['f2']:.3f} P {row['precision']:.3f} R {row['recall']:.3f}")
            # A fine-tune that never improves over its input should keep the input.
            if not any(getattr(model, key, None) is not None for key in ("obj_mlp", "edge_mlp", "piece", "layer_mlp", "group_head")):
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
            sums_t, n, shapes_t, bad = {}, 0, torch.zeros((), device=device, dtype=torch.int64), 0
            epoch_pairs = 0
            epoch_recovered, epoch_regions = 0, 0
            bar = EpochBar(epoch, c.epochs, c.steps, cols, c.window, header=True)
            for bi, b in enumerate(dl):
                step_pairs = 0
                recovered, regions = 0, 0
                for g in opt.param_groups:
                    g["lr"] = c.lr0 * lr_factor(it, total_iters, warmup, c.lrf, c.cos_lr) * g.get("lr_scale", 1.0)
                if gpu_batch:
                    b = build_batch([w for w, _ in b], device, aug, [s for _, s in b])
                else:
                    b = to_device(b, device)
                if mae_ratio > 0:
                    b["mae"] = (torch.rand(b["mask0"].shape, device=device) < mae_ratio) & (b["mask0"] > 0)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                    logits = model(b)
                if mae_ratio > 0:
                    err = ((logits.float() - model.target_features(b)) ** 2).mean(-1)
                    total = err[b["mae"]].mean()
                    parts = {"mse": total}
                else:
                    total, parts = loss_fn(c.arch, logits, b, c.pos_weight, c.alpha, c.lovasz, float(getattr(c, "poly", 0.0) or 0.0), class_weights=scw)
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
                            lrgn, recovered, regions = region.training_loss(model.last_piece, b, recoverable=recoverable)
                            total = total + lrgn
                            parts["region"] = lrgn
                    if obj_w > 0 and "ocls" in b and model.last_obj is not None:
                        lo = object_loss(model.last_obj, b, cw, c.alpha)
                        total = total + obj_w * lo
                        parts = {**parts, "obj": lo}
                    if group_w > 0 and getattr(model, "last_h", None) is not None:
                        lg = group_cls_loss(model.group_head, model.last_h, logits, model.last_edge, b, gcw, c.alpha)
                        total = total + group_w * lg
                        parts = {**parts, "grp": lg}
                    img = getattr(model, "img", None)
                    if aux_w > 0 and img is not None and img.last_aux is not None:
                        la = aux_loss(*img.last_aux)
                        img.last_aux = None
                        total = total + aux_w * la
                        parts = {**parts, "aux": la}
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip)
                if not bool(torch.isfinite(total.detach()) & torch.isfinite(gn)):
                    bad += 1
                    opt.zero_grad(set_to_none=True)
                    scaler.update()
                    it += 1
                    continue
                scaler.step(opt)
                scaler.update()
                ema.update(model)
                epoch_pairs += step_pairs
                epoch_recovered += recovered
                epoch_regions += regions
                # Accumulate losses & shape counts on device (zero GPU->CPU syncs on non-display steps!)
                for k, v in parts.items():
                    vd = v.detach().float()
                    sums_t[k] = vd if k not in sums_t else (sums_t[k] + vd)
                step_m = b["mask0"].sum()
                shapes_t = shapes_t + step_m
                n += 1
                it += 1
                is_last = (bi + 1 == c.steps)
                if is_last or (time.time() - bar.last >= bar.every):
                    bar.update(bi + 1, {k: float(sums_t[k]) / n for k in sums_t}, int(step_m),
                               torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else None,
                               force=is_last)
            bar.close()
            sums = {k: float(v) for k, v in sums_t.items()}
            shapes = int(shapes_t)
            dt = time.time() - t0
            rec = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], "epoch_s": round(dt, 1),
                   "shapes_s": round(shapes / max(dt, 1e-9)),
                   "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if device.type == "cuda" else ""}
            rec.update({f"{k}_loss": round(v / max(n, 1), 5) for k, v in sums.items()})
            if bad:
                self.say(f"WARNING: {bad} non-finite steps skipped in epoch {epoch + 1}")
                if amp and amp_dtype == torch.float16 and bad >= max(5, 0.25 * (n + bad)):
                    # GPU without bf16 (Volta / Turing): fp16 keeps overflowing (often bf16-trained init weights) -> fp32
                    amp, amp_dtype = False, None
                    self.say("WARNING: fp16 keeps overflowing on this GPU: training and validation continue in fp32 "
                             "(set OPENSTROKE_AMP=fp32 to start in fp32)")
            if hard_pairs:
                rec["hard_pairs"] = epoch_pairs
                self.say(f"Hard pairs epoch {epoch + 1}: {epoch_pairs:,} used in successful steps")
                if not epoch_pairs:
                    self.say("WARNING: no hard pairs in successful batches this epoch")
            if recoverable:
                rec.update(region_recovered=epoch_recovered, region_proposals=epoch_regions)
                self.say(f"Region targets epoch {epoch + 1}: {epoch_recovered:,}/{epoch_regions:,} predicted proposals promoted "
                         "in successful training steps")
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
                gv = [] if getattr(ema.module, "group_head", None) is not None else None
                cv = [] if int(getattr(ema.module, "stroke_classes", 1)) > 1 else None
                vscores, bst, at05 = evaluate(ema.module, val_pages, c, device, amp_dtype, obj_out=oval, edge_out=ev, piece_out=pv, group_out=gv, cls_out=cv)
                p_v, r_v = float(bst["precision"]), float(bst["recall"])
                f1_v = (2.0 * p_v * r_v / (p_v + r_v)) if (p_v + r_v) > 0.0 else float(bst["f2"])
                rec.update({"P": round(p_v, 4), "R": round(r_v, 4), "F1": round(f1_v, 4), "F2": round(bst["f2"], 4),
                            "F2@0.5": round(at05["f2"], 4), "thr": bst["threshold"]})
                print(VAL_HEAD + f"  ({dur(time.time() - tv)})")
                print(val_row("all" if "cohorts" not in bst else "select", len(val_pages), sum(len(p) for p in val_pages),
                              sum(float(p.length[p.label > 0].sum()) for p in val_pages),
                              bst["precision"], bst["recall"], bst["f2"], at05["f2"], bst["threshold"]), flush=True)
                for g, v in (bst.get("cohorts") or {}).items():   # F213: every cohort at the selected threshold
                    print(val_row(g + (" *" if g in bst["selected"] else ""), v["pages"], v["shapes"], v["true_length"],
                                  v["precision"], v["recall"], v["f2"], at05["cohorts"][g]["f2"], bst["threshold"]), flush=True)
                    rec[f"F2:{g}"] = round(v["f2"], 4)
                fitness = bst.get("macro_f2", bst["f2"])
                if cv is not None:
                    from ..metrics import multiclass_metrics, resolve_class_names

                    mc_names = resolve_class_names(int(ema.module.stroke_classes), ema.module.config.get("class_names"))
                    mc = multiclass_metrics(cv, [p.label for p in val_pages], [p.length for p in val_pages], names=mc_names)
                    fitness = float(mc["fitness"])
                    rec["fitness"] = round(fitness, 4)
                    rec["macro_cls_f2"] = round(float(mc["macro_fg_f2"]), 4)
                    c_parts = []
                    for cname in mc_names[1:]:
                        cm = mc["per_class"][cname]
                        rec[f"F2_{cname}"] = round(float(cm["f2"]), 4)
                        c_parts.append(f"{cname} {cm['f2']:.3f} (P {cm['precision']:.2f} R {cm['recall']:.2f})")
                    print(f"     classes  " + " | ".join(c_parts) + f" | macro {mc['macro_fg_f2']:.3f} | fitness {fitness:.4f}", flush=True)
                if "macro_f2" in bst:
                    rec.update(macro_f2=fitness, worst_f2=bst["worst_f2"])
                    self.say(f"GT validation mean PDF F2 {fitness:.3f}, worst PDF {bst['worst_f2']:.3f}")
                if not getattr(c, "gt_only", False) or ev is not None or pv is not None or gv is not None:
                    self.epoch_objects(val_pages, vscores, bst["threshold"], ev, pv, rec, gscores=gv, gmodel=ema.module, device=device)
                if str(getattr(c, "select_by", "stroke")) == "object":
                    fitness = compute_object_fitness(bst, rec)
                    rec["fitness"] = round(float(fitness), 4)
                    print(f"     fitness  {fitness:.4f} (0.25*F1 {f1_v:.4f} + 0.15*F2 {bst['f2']:.4f} + 0.25*mx + 0.35*6cls)", flush=True)
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
                if continuation is not None or stroke_context is not None or hard_pairs or (getattr(c, "gt_only", False) and region is None):
                    # train3 next to val on every GT-only stroke run: the train / val gap is what OpenStroke38 measures (AGENTS 0.12)
                    diag = sorted(train_pages, key=lambda p: p.stem)[:3]
                    train_scores = [predict_page(ema.module, p, c.window, device, c.val_batch, 1, amp_dtype) for p in diag]
                    train_rows = per_page(train_scores, [p.label for p in diag], [p.length for p in diag],
                                          [p.stem for p in diag], bst["threshold"])
                    rec["train3_f2"] = round(float(np.mean([r["f2"] for r in train_rows])), 4)
                    val_s = bst["f2"] if getattr(c, "select_by", "stroke") == "object" else fitness
                    self.say(f"Stroke F2 train3 {rec['train3_f2']:.3f} / val {val_s:.3f} @ {bst['threshold']:.2f}")
                if len(val_pages) <= 12:  # small val (dev split): every page, to see which one collapses
                    for pg, sc, row in zip(val_pages, vscores, per_page(vscores, [q.label for q in val_pages],
                                                                         [q.length for q in val_pages],
                                                                         [q.stem for q in val_pages], bst["threshold"])):
                        r05 = prf(*length_counts(sc >= 0.5, pg.label, pg.length))
                        print(val_row(pg.stem.split("_gt")[0][:12], 1, len(pg), float(pg.length[pg.label > 0].sum()),
                                      row["precision"], row["recall"], row["f2"], r05["f2"], bst["threshold"]), flush=True)
                        rec[f"F2:{pg.stem}"] = round(row["f2"], 4)
                # F245 / F249: Fine-tune convergence guard for select_by=object — when fine-tuning a multi-epoch run
                # (>= 10 epochs) with new zero-init branches, wait until half the cosine schedule (max(warmup+2, epochs//2))
                # before locking best.pt so early high-LR epochs (like E150 ep5 / E151 ep6) never freeze ahead of
                # converged EMA branches (crash-safe: epoch 1 still writes best.pt).
                min_best_ep = int(round(float(getattr(c, "warmup_epochs", 1.0)))) + 2
                if int(c.epochs) >= 10:
                    min_best_ep = max(min_best_ep, int(c.epochs) // 2)
                warmup_guard = (
                    init is not None
                    and str(getattr(c, "select_by", "stroke")) == "object"
                    and int(c.epochs) >= min_best_ep + 3
                    and not hard_pairs
                )
                if warmup_guard and (epoch + 1) == min_best_ep and best_epoch + 1 < min_best_ep:
                    best = -1.0
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
        m.to(device).eval()
        from ..models.image_branch import fuse_image_branch

        fuse_image_branch(m, channels_last=True)
        tta = 8 if c.tta else 1
        lscores = [] if getattr(m, "layer_mlp", None) is not None else None
        oscores = [] if getattr(m, "obj_mlp", None) is not None else None
        escores = [] if getattr(m, "edge_mlp", None) is not None else None
        pscores = [] if getattr(m, "piece", None) is not None else None
        gscores = [] if getattr(m, "group_head", None) is not None else None
        cscores = [] if int(getattr(m, "stroke_classes", 1)) > 1 else None
        scores, bst, at05 = evaluate(m, val_pages, c, device, amp_dtype, tta=tta, layer_out=lscores, obj_out=oscores, edge_out=escores,
                                     piece_out=pscores, group_out=gscores, cls_out=cscores)
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
        if cscores:
            from ..metrics import multiclass_metrics, resolve_class_names

            mc_names = resolve_class_names(int(m.stroke_classes), m.config.get("class_names"))
            mc_all = multiclass_metrics(cscores, labels, lengths, names=mc_names)
            result["multiclass"] = mc_all
            for p, cp in zip(val_pages, cscores):
                objs[f"{p.stem}__cls_prob"] = cp.astype(np.float16)
            self.say(f"\nMulti-class stroke breakdown (argmax, length-weighted, classes={mc_names}):")
            mrows = [[cname, f"{mc_all['per_class'][cname]['recall']:.3f}", f"{mc_all['per_class'][cname]['precision']:.3f}",
                      f"{mc_all['per_class'][cname]['f2']:.3f}"] for cname in mc_names]
            fg_lbl = f"fg(1..{len(mc_names) - 1})" if len(mc_names) != 3 else "pipe(1+2)"
            mrows.append([fg_lbl, f"{mc_all['pipe']['recall']:.3f}", f"{mc_all['pipe']['precision']:.3f}", f"{mc_all['pipe']['f2']:.3f}"])
            mrows.append(["macro_fg", "", "", f"{mc_all['macro_fg_f2']:.3f}"])
            self.say(table(mrows, ["class", "R", "P", "F2"]))
        if oscores is not None:  # E62 / OpenStroke29: per-shape class probabilities, from the same TTA pass as the stroke scores
            from ..objects import N_CLASSES
            for p, pc in zip(val_pages, oscores):
                objs[f"{p.stem}__oprob"] = pc[:, :N_CLASSES].astype(np.float16)
                objs[f"{p.stem}__ooff"] = pc[:, N_CLASSES:].astype(np.float32)
        if escores:   # OpenStroke30 / F169: same-object logit of every linked stroke pair, as payload shape ids
            for p, (e, z) in zip(val_pages, escores):
                objs[f"{p.stem}__eij"] = p.index[e].astype(np.int32)
                objs[f"{p.stem}__elog"] = z.astype(np.float16)
        gdecoded = self._decode_groups(m, val_pages, scores, thr, escores, gscores, device) if gscores else None
        if gscores and gdecoded is not None:   # OpenStroke41: per-stroke 8-class probs + full-page cluster ids & 8-class probs
            for p, (_, scls), (inst, cids, cprobs) in zip(val_pages, gscores, gdecoded):
                objs[f"{p.stem}__scls"] = scls.astype(np.float16)
                objs[f"{p.stem}__ginst"] = inst.astype(np.int32)
                objs[f"{p.stem}__gcids"] = cids.astype(np.int32)
                objs[f"{p.stem}__gcprobs"] = cprobs.astype(np.float16)
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
        if not getattr(c, "gt_only", False) or escores is not None or pscores is not None or gscores is not None:
            obj = self.object_report(val_pages, scores, thr, escores, pscores, gdecoded=gdecoded)
        if getattr(getattr(m, "piece", None), "region", None) is not None:
            from ..region_data import report_regions
            obj["regions"] = report_regions(val_pages, pscores, self.say, "best val", details=True)
            from ..region_data import live_report
            obj["live_regions"] = live_report(m, val_pages, scores, Path(c.data), c.window, c.val_batch, device, amp_dtype, self.say)
        if obj:
            result["objects"] = obj
            (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        # F249: Auto-evaluate final EMA (last.pt) with TTA when best_epoch != epochs_trained so best vs last is always
        # directly comparable without a manual post-run cell, AND export compact FP16 deploy checkpoints (~35 MB).
        last_pt = out / "last.pt"
        if (last_pt.exists() and epochs_trained and best_epoch + 1 != epochs_trained
                and not getattr(c, "hard_pairs", False)
                and getattr(getattr(m, "piece", None), "region", None) is None):
            ck_l = torch.load(last_pt, map_location="cpu", weights_only=False)
            ema_sd = (ck_l.get("ema") or {}).get("module") or ck_l.get("model")
            if ema_sd is not None:
                ml = from_config(ck_l["config"], weights_follow=True)
                load_weights(ml, ema_sd)
                ml.to(device).eval()
                fuse_image_branch(ml, channels_last=True)
                l_es = [] if getattr(ml, "edge_mlp", None) is not None else None
                l_gs = [] if getattr(ml, "group_head", None) is not None else None
                l_cs = [] if int(getattr(ml, "stroke_classes", 1)) > 1 else None
                l_scores, l_bst, l_at05 = evaluate(ml, val_pages, c, device, amp_dtype, tta=tta,
                                                   edge_out=l_es, group_out=l_gs, cls_out=l_cs)
                l_thr = float(l_bst["threshold"])
                ck_l["threshold"] = l_thr
                ck_l["tta"] = tta
                torch.save(ck_l, last_pt)
                l_objs = {}
                if l_cs:
                    for p, cp in zip(val_pages, l_cs):
                        l_objs[f"{p.stem}__cls_prob"] = cp.astype(np.float16)
                if l_es:
                    for p, (e, z) in zip(val_pages, l_es):
                        l_objs[f"{p.stem}__eij"] = p.index[e].astype(np.int32)
                        l_objs[f"{p.stem}__elog"] = z.astype(np.float16)
                l_gdec = self._decode_groups(ml, val_pages, l_scores, l_thr, l_es, l_gs, device) if l_gs else None
                if l_gs and l_gdec is not None:
                    for p, (_, scls), (inst, cids, cprobs) in zip(val_pages, l_gs, l_gdec):
                        l_objs[f"{p.stem}__scls"] = scls.astype(np.float16)
                        l_objs[f"{p.stem}__ginst"] = inst.astype(np.int32)
                        l_objs[f"{p.stem}__gcids"] = cids.astype(np.int32)
                        l_objs[f"{p.stem}__gcprobs"] = cprobs.astype(np.float16)
                np.savez_compressed(out / "val_scores_last.npz",
                                    **{s: sc.astype(np.float32) for s, sc in zip(stems, l_scores)},
                                    **{f"{p.stem}__index": p.index for p in val_pages}, **l_objs)
                result["last_val"] = {"epoch": epochs_trained, "threshold": l_thr, "val_at_threshold": l_bst, "val_at_0.5": l_at05}
                self.say(f"Fold {fold + 1} last.pt (epoch {epochs_trained}, EMA){' + TTA x8' if tta > 1 else ''}: "
                         f"F2 {l_bst['f2']:.4f} (P {l_bst['precision']:.3f} R {l_bst['recall']:.3f} @ {l_thr:.2f}) -> val_scores_last.npz")
                (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        try:
            from ..strip import strip as _strip_ckpt
            _strip_ckpt(out / "best.pt", out / "best_deploy.pt", half=True)
            if last_pt.exists():
                _strip_ckpt(last_pt, out / "last_deploy.pt", half=True)
        except Exception as exc:
            self.log(f"auto-strip skipped: {exc}")
        if getattr(c, "region_recoverable", False):
            from ..replay_regions import report

            report(Path(c.data), out / "val_scores.npz", say=self.say, folds=Path(c.folds) if c.folds else None)
        return result

    @staticmethod
    def _decode_groups(model, val_pages, scores, thr, escores, gscores, device):
        """OpenStroke41: full-page H142 mutex instances + GroupClsHead 8-class probabilities per val page."""
        from ..stroke_groups import stroke_instances

        out = []
        for i, p in enumerate(val_pages):
            if escores is not None and len(escores[i][0]):
                inst = stroke_instances(scores[i], thr, escores[i][0], escores[i][1])
            else:
                inst = np.where(scores[i] >= thr, np.arange(len(scores[i]), dtype=np.int64), -1)
            cids, cprobs = model.group_head.classify_page_clusters(
                gscores[i][0], p, scores[i], thr, escores[i] if escores else None, inst, device
            )
            out.append((inst, cids, cprobs))
        return out

    def epoch_objects(self, val_pages, scores, thr, escores, pscores, rec: dict,
                      gscores=None, gmodel=None, device=None):
        """One line per epoch: object F2 (GT / Viewer) of strokes (CC), H142 mutex (mx), OpenStroke41 6-class (6cls), and pieces."""
        from .. import object_eval as oe

        suites = [
            ("strokes", {p.stem: oe.stroke_counts(p, s, thr, escores[i] if escores else None)
                         for i, (p, s) in enumerate(zip(val_pages, scores))}),
        ]
        if escores is not None:
            suites.append(
                ("mx", {p.stem: oe.stroke_counts(p, s, thr, escores[i], signed=True)
                        for i, (p, s) in enumerate(zip(val_pages, scores))})
            )
        if gscores is not None and gmodel is not None and getattr(gmodel, "group_head", None) is not None:
            dec = self._decode_groups(gmodel, val_pages, scores, thr, escores, gscores, device)
            suites.append(
                ("6cls", {p.stem: oe.group_counts_6class(p, scores[i], thr, *dec[i])
                          for i, p in enumerate(val_pages)})
            )
        suites.append(
            ("pieces", {p.stem: oe.piece_counts(p, q) for p, q in
                        zip([p for p in val_pages if p.pieces is not None], pscores or [])})
        )
        parts = []
        for name, per in suites:
            per = {k: v for k, v in per.items() if v is not None}
            if not per:
                continue
            s = oe.summarize(per, cohort_groups(val_pages))
            sub = [f"{g} {v['f2']:.3f} (P {v['precision']:.2f} R {v['recall']:.2f})"
                   for g, v in s.items() if g != "all"]
            if not sub and "all" in s:
                v = s["all"]
                sub = [f"all {v['f2']:.3f} (P {v['precision']:.2f} R {v['recall']:.2f})"]
            parts.append(f"{name} " + "  ".join(sub))
            rec.update({f"obj_{name}_{g}": v["f2"] for g, v in s.items()})
        if parts:
            print("     objects  " + " | ".join(parts), flush=True)

    def object_report(self, val_pages, scores, thr, escores, pscores, gdecoded=None) -> dict:
        """Object F2 from the model alone (openstroke.object_eval), printed in the log so no file has to leave Colab (user 2026-11-19)."""
        from .. import object_eval as oe

        res, by_stroke, by_mx, by_6cls, by_piece, by_lines = {}, {}, {}, {}, {}, {}
        cls_gt, cls_vw = {}, {}
        npz_by_stem: dict[str, Path] = {}
        data_dir = Path(self.cfg.data) if getattr(self.cfg, "data", None) else None
        if data_dir is not None and any(getattr(p, "mode", "hvac") in ("fire", "plumbing") for p in val_pages):
            mf = data_dir / "manifest.json"
            if mf.exists():
                try:
                    for rec in json.loads(mf.read_text(encoding="utf-8")).get("records", []):
                        if rec.get("stem") and rec.get("npz"):
                            npz_by_stem[rec["stem"]] = data_dir / rec["npz"]
                except Exception:
                    pass
        for i, p in enumerate(val_pages):
            c = oe.stroke_counts(p, scores[i], thr, escores[i] if escores else None)
            if c is not None:
                by_stroke[p.stem] = c
            if escores is not None:
                cm = oe.stroke_counts(p, scores[i], thr, escores[i], signed=True)
                if cm is not None:
                    by_mx[p.stem] = cm
            if gdecoded is not None:
                bc = cls_vw if oe.group_of(p.stem) == "Viewer" else cls_gt
                c6 = oe.group_counts_6class(p, scores[i], thr, *gdecoded[i], by_class=bc)
                if c6 is not None:
                    by_6cls[p.stem] = c6
            if getattr(p, "mode", "hvac") in ("fire", "plumbing"):
                npzp = npz_by_stem.get(p.stem) or (data_dir / f"{p.stem}.npz" if data_dir is not None else None)
                if npzp is not None:
                    cl = oe.thin_line_counts(p, scores[i], thr, npzp)
                    if cl is not None:
                        by_lines[p.stem] = cl
        if pscores:
            for p, q in zip([p for p in val_pages if p.pieces is not None], pscores):
                c = oe.piece_counts(p, q)
                if c is not None:
                    by_piece[p.stem] = c
        for name, per_page, what in (
                ("strokes", by_stroke, f"strokes p >= {thr:.2f} joined by same-object links >= {oe.LINK}, matched without class"),
                ("mx", by_mx, f"strokes p >= {thr:.2f} grouped by H142 signed mutex partition, matched without class"),
                ("6cls", by_6cls, f"OpenStroke41 end-to-end 6-class objects (H142 mutex + GroupClsHead, FP rejected), matched WITH class"),
                ("pieces", by_piece, f"pieces p >= {oe.PIECE_THR} joined by links >= {oe.LINK} within one predicted class, matched with class "
                                     "on GT pages")):
            if not per_page:
                continue
            s = oe.summarize(per_page, cohort_groups(val_pages))
            res[name] = {"groups": s, "pages": {k: [int(x) for x in v] for k, v in per_page.items()}}
            self.say(f"\nObject F2 ({name}): {what}; IoU >= {oe.MATCH_IOU} on length, model only (no AI_Takeoff logic)")
            rows = [[g, v["pages"], f"{v['recall']:.3f}", f"{v['precision']:.3f}", f"{v['f2']:.3f}", v["matched"], v["predicted"], v["gt"]]
                    for g, v in s.items()]
            rows += [[k, "", f"{oe.f2(v)[2]:.3f}", f"{oe.f2(v)[1]:.3f}", f"{oe.f2(v)[0]:.3f}", int(v[0]), int(v[1]), int(v[2])]
                     for k, v in sorted(per_page.items(), key=lambda kv: oe.f2(kv[1])[0])]
            self.say(table(rows, ["group / page", "pages", "R", "P", "F2", "match", "pred", "gt"]))
            for g, v in s.items():
                self.say(f"OBJECTS {name} {g}: F2 {v['f2']:.3f} P {v['precision']:.3f} R {v['recall']:.3f}")
        if by_6cls:
            res["6cls"]["per_class"] = {}
            for gname, bc in (("GT", cls_gt), ("Viewer", cls_vw)):
                if not bc:
                    continue
                crows = []
                res["6cls"]["per_class"][gname] = {}
                for cname, cnt in bc.items():
                    f, pr, rc = oe.f2(cnt)
                    res["6cls"]["per_class"][gname][cname] = {
                        "f2": round(f, 4), "precision": round(pr, 4), "recall": round(rc, 4),
                        "matched": int(cnt[0]), "predicted": int(cnt[1]), "gt": int(cnt[2]),
                    }
                    crows.append([cname, f"{rc:.3f}", f"{pr:.3f}", f"{f:.3f}", int(cnt[0]), int(cnt[1]), int(cnt[2])])
                self.say(f"\nOpenStroke41 6-class breakdown ({gname}, model only):")
                self.say(table(crows, ["class", "R", "P", "F2", "match", "pred", "gt"]))
        if by_lines:
            s = oe.summarize_lines(by_lines, cohort_groups(val_pages))
            res["lines"] = {"groups": s, "pages": {k: [round(float(x), 1) for x in v] for k, v in by_lines.items()}}
            self.say(f"\nObject F2 (lines): line_decoder.decode_lines on strokes p >= {thr:.2f} vs GT decoded lines "
                     "(collinear pipe-length F2, 0 layers, no AI_Takeoff logic)")
            lrows = [[g, v["pages"], f"{v['recall']:.3f}", f"{v['precision']:.3f}", f"{v['f2']:.3f}",
                      f"{v['pred_len']:.0f}", f"{v['gt_len']:.0f}"] for g, v in s.items()]
            lrows += [[k, "", f"{oe.line_f2(v)[2]:.3f}", f"{oe.line_f2(v)[1]:.3f}", f"{oe.line_f2(v)[0]:.3f}",
                       f"{v[2]:.0f}", f"{v[3]:.0f}"]
                      for k, v in sorted(by_lines.items(), key=lambda kv: oe.line_f2(kv[1])[0])]
            self.say(table(lrows, ["group / page", "pages", "R", "P", "F2", "pred_pt", "gt_pt"]))
            for g, v in s.items():
                self.say(f"OBJECTS lines {g}: F2 {v['f2']:.3f} P {v['precision']:.3f} R {v['recall']:.3f}")
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
                for k in dict.fromkeys(("all", p.mode, page_cohort(p.mode, p.stem))):
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
