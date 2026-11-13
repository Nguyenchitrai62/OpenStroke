"""Dataset over the patch split produced by colab/data/split_dota.py.

Targets per image: dict(labels=int64 (N,), boxes=float32 (N, 5) = (cx, cy, w, h, theta) with
cx, cy, w, h normalised by the (square) patch size and theta in le90 radians).
Images are returned as uint8 CHW RGB tensors; normalisation happens on the GPU.
With vectors=True (CAD data with {root}/vectors/{split}/{name}.npz) the target also carries "vec" (M, 22):
type, 8 points / image size, rgb, width / image size, CAD layer id (0 = unknown), transformed together with the
image by every augmentation.
With members=True (OpenOBB25) every primitive also carries the index of its object in the sample's final target list
("inst", -1 = none, INST_UNKNOWN when the tile has no object ids) and its source id ("pid"): vec is (M, 24). Object
identity is tracked through flips, rotations, mosaics and crops (an object dropped by a crop frees its primitives),
and the token cap keeps the longest primitives (deterministic, the rule used at inference).
"""
import json
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import shapely
import torch
from torch.utils.data import Dataset

from openstroke.obb.data.imgio import imread
from openstroke.obb.data.vectors import INST_UNKNOWN, cap_order

cv2.setNumThreads(0)
PAD_BGR = (104, 116, 124)
SQUARE_NAMES = ("storage-tank", "roundabout")    # DOTA: only 90-degree rotations (RTMDet-R / O2 practice)
OID_STEP = 1 << 20      # mosaic: object / primitive ids of the 4 sources are kept apart by this offset


def polys_to_obb(polys):
    """(N, 8) pixel polygons -> (N, 5) le90 OBBs in pixels via min-area rectangles."""
    out = np.zeros((len(polys), 5), dtype=np.float32)
    for i, p in enumerate(polys):
        (cx, cy), (w, h), a = cv2.minAreaRect(p.reshape(4, 2).astype(np.float32))
        t = math.radians(a)
        if w < h:
            w, h, t = h, w, t + math.pi / 2
        t = (t + math.pi / 2) % math.pi - math.pi / 2
        out[i] = cx, cy, w, h, t
    return out


def clip_to_window(polys, labels, S, iof_thr=0.7):
    """Keep objects with area(obj & window)/area(obj) >= iof_thr; truncated ones become the min-area rectangle
    of their visible part (same rule as colab/data/split_dota.py)."""
    if len(polys) == 0:
        return polys, labels
    inside = (polys >= 0).all(1) & (polys <= S).all(1)
    if inside.all():
        return polys.astype(np.float32), labels
    g = shapely.polygons(polys.reshape(-1, 4, 2).astype(np.float64))
    bad = ~shapely.is_valid(g)
    if bad.any():
        g[bad] = shapely.convex_hull(g[bad])
    area = shapely.area(g)
    inter = shapely.intersection(g, shapely.box(0, 0, S, S))
    iof = np.where(area > 0, shapely.area(inter) / np.maximum(area, 1e-9), 0)
    iof[inside] = 1.0
    out_p, out_l = [], []
    for k in np.nonzero(iof >= iof_thr)[0]:
        if iof[k] > 0.999:
            q = polys[k]
        else:
            pts = np.asarray(shapely.get_coordinates(inter[k]), dtype=np.float32)
            q = cv2.boxPoints(cv2.minAreaRect(pts)).reshape(-1)
        out_p.append(np.clip(q, 0, S))
        out_l.append(labels[k])
    if not out_p:
        return np.zeros((0, 8), np.float32), np.zeros((0,) + labels.shape[1:], np.int64)
    return np.array(out_p, np.float32), np.array(out_l, np.int64)


def clip_to_convex(a, b, poly):
    """Segment a-b clipped to a convex polygon (Cyrus-Beck) -> (a', b') or None."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    poly = np.asarray(poly, float).reshape(-1, 2)
    c = poly.mean(0)
    d = b - a
    t0, t1 = 0.0, 1.0
    for k in range(len(poly)):
        p, q = poly[k], poly[(k + 1) % len(poly)]
        e = q - p
        n = np.array([-e[1], e[0]])
        if np.dot(n, c - p) < 0:
            n = -n                                         # inward normal
        den = float(np.dot(n, d))
        num = float(np.dot(n, a - p))
        if abs(den) < 1e-12:
            if num < 0:
                return None
            continue
        t = -num / den
        if den > 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return a + t0 * d, a + t1 * d


def to_gray(img, vec):
    """OpenStroke21 colour-invariance augment: the tile image (BGR uint8) becomes grey and every stroke token's colour becomes its
    luminance, so image and vectors stay consistent (colours differ between projects: F97, the train palette has 6 colours)."""
    g = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if vec is not None and len(vec["attr"]):
        rgb = vec["attr"][:, 1:4]
        lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        vec["attr"][:, 1:4] = lum[:, None]
    return img, vec


class DotaPatches(Dataset):
    def __init__(self, root, split, size=1024, augment=False, filter_empty=False, min_size=2.0,
                 hsv=(0.015, 0.5, 0.3), rot90=True, flip=True, limit=None, keep_difficult=True, rotate_p=0.0,
                 mosaic_p=0.0, context=False, thumb=512, ctx_dropout=0.0, vectors=False, max_tokens=4096,
                 scale_jitter=0.0, translate=0.0, mosaic_mode="half", layer_drop=0.0, aug_iof=0.7, cache=False, gray_p=0.0, copy_paste=0.0, min_extent=0.0, hatch_p=0.0,
                 fliplr=0.5, flipud=0.5, rect=False, members=False):
        self.root, self.split, self.size = Path(root), split, size
        # OpenOBB25: labels carry (class, object id) through every augmentation, vec gets inst / pid columns
        self.members = bool(members and vectors)
        # rect (evaluation only, YOLO rect): a tile whose valid area (meta w x h) is smaller than the stored square is
        # cropped to that area padded up to multiples of 32 (rect_hw); collate pads a batch to its largest shape
        self.rect = rect and not augment
        # H11 (YOLO recipe idea): random scale in [1 - scale_jitter, 1 + scale_jitter] and translation of
        # +-translate x size on every training sample; mosaic_mode "yolo" = 4 full-resolution patches around a
        # random centre on a 2S canvas, then the same scale/translate warp to S (objects keep their size)
        self.scale_jitter, self.translate, self.mosaic_mode = scale_jitter, translate, mosaic_mode
        # visible fraction an object cut by a mosaic/zoom crop needs to keep its label (a visible but unlabelled
        # part is trained as background; the one-stage recipe keeps even small pieces)
        self.aug_iof = aug_iof
        self.layer_drop = layer_drop    # H16: chance to merge all CAD layers into one (do not over-rely on layers)
        self.gray_p = float(gray_p)     # OpenStroke21: chance to drop colour from the tile image AND its stroke tokens
        self.copy_paste = float(copy_paste)  # OpenStroke22: mean number of GT objects pasted (image + strokes) per tile
        self.min_extent = float(min_extent)  # OpenStroke23: strokes smaller than this (input px) are not model inputs
        self.hatch_p = float(hatch_p)        # OpenStroke26: chance per Duct object to get a drawn insulation hatch (image + strokes)
        self._duct_id = None
        self._cp_accept = None
        self.vectors, self.max_tokens = vectors, max_tokens
        self.context, self.thumb, self.ctx_dropout = context, thumb, ctx_dropout
        self.rotate_p, self.mosaic_p = rotate_p, mosaic_p
        self.augment, self.min_size, self.hsv, self.rot90, self.flip = augment, min_size, hsv, rot90, flip
        self.fliplr, self.flipud = (fliplr, flipud) if flip else (0.0, 0.0)
        items = [json.loads(l) for l in open(self.root / "meta" / f"{split}.jsonl")]
        items.sort(key=lambda m: m["name"])
        if filter_empty:
            items = [m for m in items if m["objs"]]
        if limit:
            items = items[:limit]
        self.items = items
        try:                                     # by name: class ids of another dataset must not match DOTA's
            self.square_ids = [i for i, n in enumerate(dataset_classes(root)) if n in SQUARE_NAMES]
        except Exception:  # noqa: BLE001
            self.square_ids = []
        self.cache = {}
        from openstroke.obb.runtime import ByteLRU
        self.vector_cache = ByteLRU(int(os.environ.get("OPENSTROKE_TRAIN_VECTOR_CACHE_MB", "64")) * 1024**2)
        if cache:                               # decode once in the main process; forked workers share the pages
            from concurrent.futures import ThreadPoolExecutor
            files = [self.root / "images" / self.split / m.get("file", f"{m['name']}.jpg") for m in self.items]
            with ThreadPoolExecutor(8) as ex:
                imgs = list(ex.map(imread, files))
            self.cache = {m["name"]: im for m, im in zip(self.items, imgs)}
        self.keep_difficult = keep_difficult

    def __len__(self):
        return len(self.items)

    def _load(self, m):
        img = self.cache.get(m["name"])
        if img is None:
            img = imread(self.root / "images" / self.split / m.get("file", f"{m['name']}.jpg"))
        objs = [(k, o) for k, o in enumerate(m["objs"]) if self.keep_difficult or o[1] == 0]
        polys = np.array([o[3:] for _, o in objs], dtype=np.float32).reshape(-1, 8)
        labels = np.array([o[0] for _, o in objs], dtype=np.int64)
        if self.members:                        # (class, index of the object in the tile's list = its vector inst)
            labels = np.stack([labels, np.array([k for k, _ in objs], dtype=np.int64)], 1).reshape(-1, 2)
        return img, polys, labels

    def _load_vec(self, m):
        """Primitive tokens in pixel units: pts (M, 16) and attr (M, 6) = type, r, g, b, width, layer id."""
        key = (m["name"], self.size, self.members)
        cached = self.vector_cache.get(key)
        if cached is not None:
            return {k: v.copy() for k, v in cached.items()}
        with np.load(self.root / "vectors" / self.split / f"{m['name']}.npz") as source:
            z = {k: source[k] for k in source.files}
        t = z["tokens"].astype(np.float32)
        view = float(z["view"]) if "view" in z else 140.0     # FloorPlanCAD: widths in viewBox units
        S = self.size
        layer = z["layer"].astype(np.float32)[:, None] + 1 if "layer" in z else np.zeros((len(t), 1), np.float32)
        attr = np.concatenate([t[:, :1], t[:, 17:20], t[:, 20:21] / view * S, np.maximum(layer, 0)], 1)
        if self.members:                        # inst (tile object index, -1 none), pid (source primitive id)
            inst = z["inst"].astype(np.float32) if "inst" in z else np.full(len(t), INST_UNKNOWN, np.float32)
            pid = z["pid"].astype(np.float32) if "pid" in z else np.full(len(t), -1, np.float32)
            attr = np.concatenate([attr, inst[:, None], pid[:, None]], 1)
        result = {"pts": t[:, 1:17] * S, "attr": attr}
        if getattr(self, "min_extent", 0.0):
            from openstroke.obb.data.vectors import small_strokes
            big = ~small_strokes(result["pts"], self.min_extent)
            result = {"pts": result["pts"][big], "attr": attr[big]}
        self.vector_cache.put(key, result)
        return {k: v.copy() for k, v in result.items()}

    def _vec_out(self, vec):
        """Drop primitives that left the image, cap the count, normalise -> (M, 22) float32 tensor ((M, 24) with
        members: + inst, pid). members: the cap keeps the longest primitives (cap_order, as at inference)."""
        S = self.size
        pts, attr = vec["pts"], vec["attr"]
        x, y = pts[:, 0::2], pts[:, 1::2]
        inside = ((x >= 0) & (x <= S) & (y >= 0) & (y <= S)).any(1)
        pts, attr = pts[inside], attr[inside]
        if len(pts) > self.max_tokens:
            if self.members:
                sel = cap_order(pts, self.max_tokens)
            else:
                sel = (np.sort(np.random.choice(len(pts), self.max_tokens, replace=False)) if self.augment
                       else np.arange(self.max_tokens))
            pts, attr = pts[sel], attr[sel]
        cols = [attr[:, :1], pts / S, attr[:, 1:4], attr[:, 4:5] / S, attr[:, 5:6]] + (
            [attr[:, 6:8]] if self.members else [])
        width = 24 if self.members else 22
        return torch.from_numpy(np.concatenate(cols, 1).astype(np.float32).reshape(-1, width))

    def _load_ctx(self, m):
        """Whole-image thumbnail on a THUMB x THUMB canvas + the tile's box on that canvas (global context, H6)."""
        T = self.thumb
        canvas = np.empty((T, T, 3), np.uint8)
        canvas[:] = PAD_BGR
        th = imread(self.root / "thumbs" / self.split / f"{m['src']}.jpg")
        if th is None or "img_w" not in m:
            return {"thumb": canvas, "tile": np.array([0, 0, T, T], np.float32), "valid": False}
        h, w = th.shape[:2]
        canvas[:min(h, T), :min(w, T)] = th[:T, :T]
        ts = T / max(m["img_w"], m["img_h"])
        r = m["rate"]
        tile = np.array([m["x0"] / r, m["y0"] / r, (m["x0"] + self.size) / r, (m["y0"] + self.size) / r]) * ts
        return {"thumb": canvas, "tile": tile.astype(np.float32), "valid": True}

    @staticmethod
    def _ctx_rot90(ctx):
        T = ctx["thumb"].shape[0]
        ctx["thumb"] = np.rot90(ctx["thumb"])
        x0, y0, x1, y1 = ctx["tile"]
        ctx["tile"] = np.array([y0, T - x1, y1, T - x0], np.float32)     # (x, y) -> (y, T - x)

    def _rotate(self, img, polys, labels, ctx=None, vec=None):
        """Arbitrary-angle rotation about the centre (images with square-like classes: 90-degree steps only)."""
        S = img.shape[0]
        if self.square_ids and np.isin(labels[:, 0] if labels.ndim == 2 else labels, self.square_ids).any():
            ang = 90.0 * random.randint(0, 3)
        else:
            ang = random.uniform(-180, 180)
        M = cv2.getRotationMatrix2D((S / 2, S / 2), ang, 1.0)
        img = cv2.warpAffine(np.ascontiguousarray(img), M, (S, S), flags=cv2.INTER_LINEAR, borderValue=PAD_BGR)
        if ctx is not None:                     # rotate the thumbnail about the tile centre: the tile stays aligned
            T = ctx["thumb"].shape[0]
            c = ((ctx["tile"][0] + ctx["tile"][2]) / 2, (ctx["tile"][1] + ctx["tile"][3]) / 2)
            Mt = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang, 1.0)
            ctx["thumb"] = cv2.warpAffine(np.ascontiguousarray(ctx["thumb"]), Mt, (T, T), flags=cv2.INTER_LINEAR,
                                          borderValue=PAD_BGR)
        if vec is not None and len(vec["pts"]):
            vec["pts"] = (vec["pts"].reshape(-1, 2) @ M[:, :2].T + M[:, 2]).reshape(-1, 16).astype(np.float32)
        if len(polys):
            pts = polys.reshape(-1, 2) @ M[:, :2].T + M[:, 2]
            polys, labels = clip_to_window(pts.reshape(-1, 8).astype(np.float32), labels, S)
        return img, polys, labels

    def _warp(self, img, polys, labels, vec, src_centre):
        """Random scale/translate warp of `img` (S x S or a 2S x 2S mosaic canvas) to an S x S output; polygons are
        re-clipped with the iof rule, vector points and line widths follow."""
        S = self.size
        s = random.uniform(1 - self.scale_jitter, 1 + self.scale_jitter)
        tx, ty = (random.uniform(-self.translate, self.translate) * S for _ in range(2))
        M = np.array([[s, 0, S / 2 + tx - s * src_centre], [0, s, S / 2 + ty - s * src_centre]], np.float32)
        if s < 1:                               # zoom-out: area resize first so 1-2 px lines are not broken up
            h, w = img.shape[:2]
            small = cv2.resize(np.ascontiguousarray(img), (max(1, round(w * s)), max(1, round(h * s))),
                               interpolation=cv2.INTER_AREA)
            T = np.array([[1, 0, M[0, 2]], [0, 1, M[1, 2]]], np.float32)
            img = cv2.warpAffine(small, T, (S, S), flags=cv2.INTER_LINEAR, borderValue=PAD_BGR)
        else:
            img = cv2.warpAffine(np.ascontiguousarray(img), M, (S, S), flags=cv2.INTER_LINEAR, borderValue=PAD_BGR)
        if vec is not None and len(vec["pts"]):
            vec["pts"] = (vec["pts"].reshape(-1, 2) @ M[:, :2].T + M[:, 2]).reshape(-1, 16).astype(np.float32)
            vec["attr"][:, 4] *= s
        if len(polys):
            pts = polys.reshape(-1, 2) @ M[:, :2].T + M[:, 2]
            polys, labels = clip_to_window(pts.reshape(-1, 8).astype(np.float32), labels, S, self.aug_iof)
        return img, polys, labels

    def _hatch(self, img, polys, labels, vec):
        """OpenStroke26: insulated ducts are drawn with a hatch inside the duct outline and the model misses them on unseen pages
        (F113). Each Duct object of the tile, with probability hatch_p, gets parallel hatch lines at +-45 deg to its axis, 5-16 px apart,
        clipped to its box: dark lines in the image (min blending) and the same segments as background stroke tokens (inst -1, pid -1).
        The duct's own strokes and label are untouched."""
        if self._duct_id is None:
            try:
                names = [n.replace("_", " ") for n in dataset_classes(self.root)]
                self._duct_id = names.index("Duct")
            except Exception:  # noqa: BLE001
                self._duct_id = 0
        img = np.ascontiguousarray(img).copy()
        lab = np.asarray(labels).reshape(len(polys), -1)
        segs = []
        for poly, row in zip(np.asarray(polys, np.float64).reshape(-1, 4, 2), lab):
            if int(row[0]) != self._duct_id or random.random() > self.hatch_p:
                continue
            e0, e1 = poly[1] - poly[0], poly[2] - poly[1]
            axis = e0 if np.linalg.norm(e0) >= np.linalg.norm(e1) else e1
            if min(np.linalg.norm(e0), np.linalg.norm(e1)) < 6:
                continue
            ang = math.atan2(axis[1], axis[0]) + random.choice((math.pi / 4, -math.pi / 4))
            d, nrm = np.array([math.cos(ang), math.sin(ang)]), np.array([-math.sin(ang), math.cos(ang)])
            c0 = poly.mean(0)
            R = float(np.linalg.norm(poly - c0, axis=1).max())
            spacing = random.uniform(5.0, 16.0)
            for t in np.arange(-R + spacing / 2, R, spacing):
                seg = clip_to_convex(c0 + nrm * t - d * R, c0 + nrm * t + d * R, poly)
                if seg is not None and np.linalg.norm(seg[1] - seg[0]) >= 3:
                    segs.append(seg)
        if not segs:
            return img, vec
        layer = np.full(img.shape[:2], 255, np.uint8)
        for a, b in segs:
            cv2.line(layer, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b), 80, 1, cv2.LINE_AA)
        img = np.minimum(img, layer[..., None])
        pts = np.stack([np.linspace(a, b, 8).reshape(-1) for a, b in segs]).astype(np.float32)
        attr = np.zeros((len(segs), vec["attr"].shape[1]), np.float32)
        attr[:, 1:4] = 80 / 255.0                     # grey, like the drawn line
        attr[:, 4] = 1.0                              # 1 px wide
        if attr.shape[1] > 7:
            attr[:, 6] = -1                           # background: not part of any object
            attr[:, 7] = -1
        return img, {"pts": np.concatenate([vec["pts"], pts]), "attr": np.concatenate([vec["attr"], attr])}

    def _accept_prob(self):
        """Class-balanced object sampling for copy-paste: accept an object of class c with sqrt(median freq / freq c), <= 1."""
        if self._cp_accept is None:
            freq = {}
            for m in self.items:
                for o in m["objs"]:
                    freq[int(o[0])] = freq.get(int(o[0]), 0) + 1
            med = float(np.median(list(freq.values()))) if freq else 1.0
            self._cp_accept = {c: min(1.0, math.sqrt(med / n)) for c, n in freq.items()}
        return self._cp_accept

    def _paste_objects(self, img, polys, labels, vec, n, max_side=0.5, tries=10):
        """OpenStroke22 copy-paste: n GT objects from random training tiles, each with its own strokes (inst == its id) and
        the image pixels under those strokes only, are pasted on blank areas of this tile. Image and vectors move together;
        the object keeps its class, gets a new id (q x OID_STEP, q >= 8) and unknown pids. Small objects only (side <=
        max_side x tile): they are what the detector misses (F100, F109)."""
        if n <= 0 or not len(self.items):
            return img, polys, labels, vec
        S = self.size
        img = np.ascontiguousarray(img).copy()
        accept = self._accept_prob()
        new_p, new_l, new_pts, new_attr = [], [], [], []
        for k in range(int(n)):
            for _ in range(tries):
                m = self.items[random.randrange(len(self.items))]
                if not m["objs"]:
                    continue
                simg, spolys, slabels = self._load(m)
                if not len(slabels):
                    continue
                r = random.randrange(len(slabels))
                cls, oid = int(slabels[r, 0]), int(slabels[r, 1])
                if random.random() > accept.get(cls, 1.0):
                    continue
                svec = self._load_vec(m)
                rows = np.flatnonzero(svec["attr"][:, 6] == oid)
                if not len(rows):
                    continue
                pts = svec["pts"][rows]
                p = pts.reshape(-1, 2)
                x0, y0 = np.floor(p.min(0)).astype(int) - 3
                x1, y1 = np.ceil(p.max(0)).astype(int) + 3
                x0, y0 = max(x0, 0), max(y0, 0)
                x1, y1 = min(x1, simg.shape[1]), min(y1, simg.shape[0])
                w, h = x1 - x0, y1 - y0
                if w < 4 or h < 4 or w > max_side * S or h > max_side * S:
                    continue
                spot = None
                for _ in range(tries):
                    tx, ty = random.randrange(0, S - w + 1), random.randrange(0, S - h + 1)
                    region = img[ty:ty + h, tx:tx + w]
                    if region.shape[:2] == (h, w) and (region.min(2) < 200).mean() < 0.01:
                        spot = (tx, ty)
                        break
                if spot is None:
                    continue
                tx, ty = spot
                mask = np.zeros((h, w), np.uint8)
                for s in pts:
                    q = np.round(s.reshape(8, 2) - [x0, y0]).astype(np.int32)
                    cv2.polylines(mask, [q], False, 255, thickness=3)
                sel = mask > 0
                region = img[ty:ty + h, tx:tx + w]
                region[sel] = np.minimum(region[sel], simg[y0:y1, x0:x1][sel])
                shift = np.array([tx - x0, ty - y0], np.float32)
                moved = pts + np.tile(shift, 8)
                attr = svec["attr"][rows].copy()
                nid = oid + (8 + k) * OID_STEP
                attr[:, 6] = nid
                attr[:, 7] = -1
                poly = spolys[r].reshape(4, 2) + shift
                new_p.append(poly.reshape(8).astype(np.float32))
                new_l.append([cls, nid])
                new_pts.append(moved.astype(np.float32))
                new_attr.append(attr)
                break
        if not new_p:
            return img, polys, labels, vec
        polys = np.concatenate([np.asarray(polys, np.float32).reshape(-1, 8), np.stack(new_p)])
        labels = np.concatenate([np.asarray(labels, np.int64).reshape(-1, 2), np.asarray(new_l, np.int64)])
        vec = {"pts": np.concatenate([vec["pts"]] + new_pts), "attr": np.concatenate([vec["attr"]] + new_attr)}
        return img, polys, labels, vec

    def _offset_ids(self, vec, labels, q):
        """Mosaic source q: its object ids (labels column 1) and its primitives' inst move by q x OID_STEP (in place)."""
        if not self.members or q == 0:
            return
        if len(labels):
            labels[:, 1] += q * OID_STEP
        inst = vec["attr"][:, 6]
        vec["attr"][:, 6] = np.where(inst >= 0, inst + q * OID_STEP, inst)

    def _mosaic_yolo(self, i):
        """Four full-resolution patches meeting at a random centre of a 2S canvas, then one scale/translate warp."""
        S = self.size
        canvas = np.empty((2 * S, 2 * S, 3), np.uint8)
        canvas[:] = PAD_BGR
        xc, yc = (int(random.uniform(0.5 * S, 1.5 * S)) for _ in range(2))
        P, L, V = [], [], []
        for q, j in enumerate([i] + [random.randrange(len(self.items)) for _ in range(3)]):
            img, polys, labels = self._load(self.items[j])
            vec = self._load_vec(self.items[j]) if self.vectors else None
            img, polys = self._augment(img, polys, color=False, vec=vec)
            h, w = img.shape[:2]
            if q == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
            elif q == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * S), yc
                x1b, y1b = 0, h - (y2a - y1a)
            elif q == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, 2 * S)
                x1b, y1b = w - (x2a - x1a), 0
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * S), min(yc + h, 2 * S)
                x1b, y1b = 0, 0
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y1b + (y2a - y1a), x1b:x1b + (x2a - x1a)]
            ox, oy = x1a - x1b, y1a - y1b
            if vec is not None:
                vec["pts"][:, 0::2] += ox
                vec["pts"][:, 1::2] += oy
                vec["attr"][:, 5] += q * 10000 * (vec["attr"][:, 5] > 0)     # keep the 4 sources' layers apart
                self._offset_ids(vec, labels, q)
                V.append(vec)
            if len(polys):
                pp = polys.copy()
                pp[:, 0::2] += ox
                pp[:, 1::2] += oy
                P.append(pp)
                L.append(labels)
        polys = np.concatenate(P) if P else np.zeros((0, 8), np.float32)
        labels = np.concatenate(L) if L else np.zeros((0, 2) if self.members else 0, np.int64)
        vec = ({"pts": np.concatenate([v["pts"] for v in V]), "attr": np.concatenate([v["attr"] for v in V])}
               if V else None)
        if len(polys):                          # objects cut by the patch borders inside the canvas
            polys, labels = clip_to_window(polys, labels, 2 * S, self.aug_iof)
        img, polys, labels = self._warp(canvas, polys, labels, vec, src_centre=S)
        return img, polys, labels, vec

    def _mosaic(self, i):
        """Four patches at half scale, each with its own random flip/rot90 (oriented dense one-to-one, RiO-DETR)."""
        S, h = self.size, self.size // 2
        canvas = np.empty((S, S, 3), np.uint8)
        canvas[:] = PAD_BGR
        P, L, V = [], [], []
        for q, j in enumerate([i] + [random.randrange(len(self.items)) for _ in range(3)]):
            img, polys, labels = self._load(self.items[j])
            vec = self._load_vec(self.items[j]) if self.vectors else None
            img, polys = self._augment(img, polys, color=False, vec=vec)
            ox, oy = (q % 2) * h, (q // 2) * h
            if vec is not None:
                vp = vec["pts"] * 0.5
                vp[:, 0::2] += ox
                vp[:, 1::2] += oy
                vec["attr"][:, 4] *= 0.5
                vec["attr"][:, 5] += q * 10000 * (vec["attr"][:, 5] > 0)
                self._offset_ids(vec, labels, q)
                V.append({"pts": vp, "attr": vec["attr"]})
            canvas[oy:oy + h, ox:ox + h] = cv2.resize(np.ascontiguousarray(img), (h, h), interpolation=cv2.INTER_AREA)
            if len(polys):
                pp = polys * 0.5
                pp[:, 0::2] += ox
                pp[:, 1::2] += oy
                P.append(pp)
                L.append(labels)
        polys = np.concatenate(P) if P else np.zeros((0, 8), np.float32)
        labels = np.concatenate(L) if L else np.zeros((0, 2) if self.members else 0, np.int64)
        vec = ({"pts": np.concatenate([v["pts"] for v in V]), "attr": np.concatenate([v["attr"] for v in V])}
               if V else None)
        return canvas, polys, labels, vec

    def _augment(self, img, polys, color=True, ctx=None, vec=None):
        """Flips / rot90 applied identically to the patch, the vector points and the context thumbnail + tile box."""
        vp = vec["pts"] if vec is not None else np.zeros((0, 16), np.float32)
        S = img.shape[0]
        T = ctx["thumb"].shape[0] if ctx is not None else 0
        if random.random() < self.fliplr:
            img = img[:, ::-1]
            polys[:, 0::2] = S - polys[:, 0::2]
            vp[:, 0::2] = S - vp[:, 0::2]
            if ctx is not None:
                ctx["thumb"] = ctx["thumb"][:, ::-1]
                x0, y0, x1, y1 = ctx["tile"]
                ctx["tile"] = np.array([T - x1, y0, T - x0, y1], np.float32)
        if random.random() < self.flipud:
            img = img[::-1]
            polys[:, 1::2] = S - polys[:, 1::2]
            vp[:, 1::2] = S - vp[:, 1::2]
            if ctx is not None:
                ctx["thumb"] = ctx["thumb"][::-1]
                x0, y0, x1, y1 = ctx["tile"]
                ctx["tile"] = np.array([x0, T - y1, x1, T - y0], np.float32)
        if self.rot90:
            k = random.randint(0, 3)
            for _ in range(k):                      # 90 deg counter-clockwise: (x, y) -> (y, S - x)
                img = np.rot90(img)
                x = polys[:, 0::2].copy()
                polys[:, 0::2] = polys[:, 1::2]
                polys[:, 1::2] = S - x
                x = vp[:, 0::2].copy()
                vp[:, 0::2] = vp[:, 1::2]
                vp[:, 1::2] = S - x
                if ctx is not None:
                    self._ctx_rot90(ctx)
        if color:
            img = self._hsv(img)
        return img, polys

    def _hsv(self, img):
        """Colour jitter in HSV space with the gains of the YOLO recipe (hsv_h / hsv_s / hsv_v): the hue is shifted by
        up to hsv_h of a full turn (OpenCV hue is 0..179), saturation and value are scaled by 1 +- hsv_s / hsv_v.
        Computed directly on the pixel values (float, then truncated back to uint8). Original OpenOBB code."""
        if not (self.hsv and any(self.hsv)):
            return img
        gain = 1.0 + np.random.uniform(-1.0, 1.0, 3) * np.asarray(self.hsv, dtype=np.float64)
        hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2HSV).astype(np.float64)
        hue = np.mod(hsv[..., 0] + (gain[0] - 1.0) * 180.0, 180.0)
        sat = np.minimum(hsv[..., 1] * gain[1], 255.0)
        val = np.minimum(hsv[..., 2] * gain[2], 255.0)
        jittered = np.stack((hue, sat, val), axis=-1).astype(np.uint8)
        return cv2.cvtColor(jittered, cv2.COLOR_HSV2BGR)

    def __getitem__(self, i):
        m = self.items[i]
        ctx = None
        vec = None
        if self.augment and self.mosaic_p and random.random() < self.mosaic_p:
            img, polys, labels, vec = self._mosaic_yolo(i) if self.mosaic_mode == "yolo" else self._mosaic(i)
            img = self._hsv(img)
            if self.rotate_p and random.random() < self.rotate_p:      # arbitrary angle after mosaic (YOLO order)
                img, polys, labels = self._rotate(img, np.asarray(polys, np.float32).reshape(-1, 8), labels, vec=vec)
            if self.context:                    # a mosaic has no single parent image: no context
                ctx = self._load_ctx(m)
                ctx["valid"] = False
        else:
            img, polys, labels = self._load(m)
            vec = self._load_vec(m) if self.vectors else None
            if self.context:
                ctx = self._load_ctx(m)
                if self.augment and self.ctx_dropout and random.random() < self.ctx_dropout:
                    ctx["valid"] = False         # context dropout: do not let scene priors dominate rare classes
            if self.augment:
                img, polys = self._augment(img, polys, ctx=ctx, vec=vec)
                if self.rotate_p and random.random() < self.rotate_p:
                    img, polys, labels = self._rotate(img, polys, labels, ctx=ctx, vec=vec)
                if self.scale_jitter or self.translate:
                    img, polys, labels = self._warp(img, polys, labels, vec, src_centre=self.size / 2)
                    if ctx is not None:
                        ctx["valid"] = False     # the thumbnail tile box does not follow the warp
        if self.augment and self.copy_paste and self.members and vec is not None:
            img, polys, labels, vec = self._paste_objects(img, polys, labels, vec, np.random.poisson(self.copy_paste))
        if self.augment and getattr(self, "hatch_p", 0.0) and self.members and vec is not None and len(polys):
            img, vec = self._hatch(img, polys, labels, vec)
        if self.augment and self.gray_p and random.random() < self.gray_p:
            img, vec = to_gray(img, vec)
        oid = None
        if self.members:                        # (class, object id) -> class; the ids follow the kept objects below
            labels = np.asarray(labels, np.int64).reshape(-1, 2)
            labels, oid = labels[:, 0].copy(), labels[:, 1].copy()
        obb = polys_to_obb(polys)
        # thin lines (1-2 px walls/pipes) keep their label with the short side clamped to min_size; only boxes
        # that vanished in both directions are dropped (deleting them made them permanent misses at eval)
        keep = np.maximum(obb[:, 2], obb[:, 3]) >= self.min_size if len(obb) else np.zeros(0, bool)
        obb, labels = obb[keep], labels[keep]
        if oid is not None:
            oid = oid[keep]
        obb[:, 2:4] = np.maximum(obb[:, 2:4], self.min_size)
        obb[:, :4] /= self.size
        if self.rect:
            rh, rw = self.rect_hw(m, img.shape[:2])
            img = img[:rh, :rw]
        img = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))      # BGR HWC -> RGB CHW
        tgt = {"labels": torch.from_numpy(labels), "boxes": torch.from_numpy(obb), "name": m["name"]}
        if ctx is not None:
            tgt["thumb"] = torch.from_numpy(np.ascontiguousarray(ctx["thumb"][..., ::-1].transpose(2, 0, 1)))
            tgt["tile"] = torch.from_numpy(np.asarray(ctx["tile"], np.float32))
            tgt["ctx_valid"] = bool(ctx["valid"])
        if self.vectors:
            if self.augment and self.layer_drop and random.random() < self.layer_drop:
                vec["attr"][:, 5] = np.minimum(vec["attr"][:, 5], 1.0)
            if oid is not None:                 # inst -> index of the object in this sample's final target list
                inst = vec["attr"][:, 6]
                known = inst >= 0
                if known.any():
                    lut = {int(o): j for j, o in enumerate(oid)}
                    vec["attr"][known, 6] = [lut.get(int(v), INST_UNKNOWN) for v in inst[known]]
            tgt["vec"] = self._vec_out(vec)
        return torch.from_numpy(img), tgt

    @staticmethod
    def rect_hw(m, shape, stride=32):
        """Network input (H, W) of a stored tile in rect mode: its valid area rounded up to `stride` (the rest is
        the split's padding), never larger than the stored tile."""
        H, W = shape
        if "h" not in m or "w" not in m:
            return H, W
        return (min(math.ceil(m["h"] / stride) * stride, H), min(math.ceil(m["w"] / stride) * stride, W))

    def rect_batches(self, batch, stride=32):
        """Index batches for rect evaluation (YOLO rect): tiles sorted by their input shape so a batch shares one
        shape and pads little. Tiles of unknown size count as full squares."""
        big = 1 << 30
        key = [self.rect_hw(m, (big, big), stride) for m in self.items]
        order = sorted(range(len(self.items)), key=lambda i: (key[i][0] / key[i][1], key[i], i))
        return [order[i:i + batch] for i in range(0, len(order), batch)]


def repeat_factors(items, num_classes, t):
    """Repeat-factor sampling (LVIS, Gupta et al. 2019): r_c = max(1, sqrt(t / f_c)), f_c = fraction of patches
    containing class c; a patch is drawn with weight max_c r_c over its classes (1 for empty patches)."""
    import numpy as np
    n = max(len(items), 1)
    has = [set(o[0] for o in m["objs"]) for m in items]
    f = np.array([sum(c in h for h in has) / n for c in range(num_classes)])
    r = np.where(f > 0, np.maximum(1.0, np.sqrt(t / np.maximum(f, 1e-12))), 1.0)
    w = np.array([max([r[c] for c in h], default=1.0) for h in has])
    return w, f, r


def mix_weights(items, w, prefix, share):
    """OpenStroke27: sampling weights giving tiles of pages whose name starts with `prefix` a total probability `share`
    and the other tiles 1 - share, each group keeping its relative weights `w` (1, or repeat factors). -> (weights,
    number of prefixed tiles)."""
    import numpy as np
    if not 0.0 < float(share) < 1.0:
        raise ValueError(f"mix_share must be in (0, 1), got {share}")
    w = np.asarray(w, dtype=np.float64)
    extra = np.array([str(m.get("src") or m["name"]).startswith(prefix) for m in items], dtype=bool)
    if extra.all() or not extra.any():
        raise SystemExit(f"mix_share needs tiles both from '{prefix}*' pages and from other pages "
                         f"({int(extra.sum())} of {len(items)} train tiles match)")
    out = np.empty_like(w)
    out[extra] = float(share) * w[extra] / w[extra].sum()
    out[~extra] = (1.0 - float(share)) * w[~extra] / w[~extra].sum()
    return out, int(extra.sum())


def dataset_classes(root):
    """Class names of a prepared split ({root}/classes.json), DOTA-v1.0 by default."""
    from openstroke.obb.eval.dota import DOTA1_CLASSES
    p = Path(root) / "classes.json"
    return tuple(json.loads(p.read_text())) if p.exists() else DOTA1_CLASSES


def collate(batch):
    shapes = {tuple(b[0].shape[-2:]) for b in batch}
    if len(shapes) == 1:
        imgs = torch.stack([b[0] for b in batch])
    else:                                       # rect evaluation: pad to the batch's largest H and W (top-left)
        H, W = max(s[0] for s in shapes), max(s[1] for s in shapes)
        imgs = torch.empty((len(batch), 3, H, W), dtype=batch[0][0].dtype)
        imgs[:] = torch.tensor(PAD_BGR[::-1], dtype=imgs.dtype).view(1, 3, 1, 1)     # RGB
        for i, b in enumerate(batch):
            imgs[i, :, :b[0].shape[1], :b[0].shape[2]] = b[0]
    return imgs, [b[1] for b in batch]
