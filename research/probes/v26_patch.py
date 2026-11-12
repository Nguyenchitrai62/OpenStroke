"""One-off patch script for OpenStroke26 (hatch augmentation); kept with the probes for the record."""
from pathlib import Path


def edit(path, pairs):
    p = Path(path); s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert old in s, (path, old[:100])
        s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8")


edit("openstroke/obb/data/dota.py", [
    ('''aug_iof=0.7, cache=False, gray_p=0.0, copy_paste=0.0, min_extent=0.0,''',
     '''aug_iof=0.7, cache=False, gray_p=0.0, copy_paste=0.0, min_extent=0.0, hatch_p=0.0,'''),
    ('''        self.min_extent = float(min_extent)  # OpenStroke23: strokes smaller than this (input px) are not model inputs''',
     '''        self.min_extent = float(min_extent)  # OpenStroke23: strokes smaller than this (input px) are not model inputs
        self.hatch_p = float(hatch_p)        # OpenStroke26: chance per Duct object to get a drawn insulation hatch (image + strokes)
        self._duct_id = None'''),
    ('''        if self.augment and self.gray_p and random.random() < self.gray_p:''',
     '''        if self.augment and getattr(self, "hatch_p", 0.0) and self.members and vec is not None and len(polys):
            img, vec = self._hatch(img, polys, labels, vec)
        if self.augment and self.gray_p and random.random() < self.gray_p:'''),
    ('''    def _accept_prob(self):''', '''    def _hatch(self, img, polys, labels, vec):
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

    def _accept_prob(self):'''),
])

p = Path("openstroke/obb/data/dota.py"); s = p.read_text(encoding="utf-8")
i = s.index("def to_gray(img, vec):")
s = s[:i] + '''def clip_to_convex(a, b, poly):
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


''' + s[i:]
p.write_text(s, encoding="utf-8")

edit("openstroke/obb/train.py", [
    ('''                     copy_paste=a.copy_paste, min_extent=a.min_extent,''',
     '''                     copy_paste=a.copy_paste, min_extent=a.min_extent, hatch_p=a.hatch_p,'''),
    ('''    ap.add_argument("--min-extent", type=float, default=0.0,''', '''    ap.add_argument("--hatch-p", type=float, default=0.0,
                    help="OpenStroke26: chance per Duct object (training) to get an insulation hatch drawn inside it, image and strokes together")
    ap.add_argument("--min-extent", type=float, default=0.0,'''),
])
print("ok")
