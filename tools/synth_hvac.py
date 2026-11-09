"""E78 (version B, breakthrough): procedural HVAC duct drawings with exact object labels, in the OpenOBB dataset format.

Why (F78): with 18 GT pages the detector learns each project's drawing style (fittings 0.96 on train, 0.48-0.79 on new projects) and
sees 4-12 examples of the rare classes. A generator gives unlimited pages where only the geometry identifies an object: random duct trees
on a grid (straight Duct pieces, split at random points with a damper mark like the GT splits at VCD / FD), 90° round Elbow and square
Elbow vuông with turning vanes, Concentric / Eccentric Reducer, Reducer vuông tròn, rectangular Chân rẽ (45° shoe) and round Chân rẽ vuông
tròn (collar + round duct with a centre line), Tee; under random architecture (grid axes, walls, columns, doors, hatches, text-like
clutter, equipment boxes). Every style (pen width, colour, grey level, insulation outline, centre lines) is random per page. Labels are
the polygons of the objects, turned into oriented boxes exactly like the real data (minimum rotated rectangle).

    python tools/synth_hvac.py <out_dir> --pages 300 --size 900x650 --scale 4
"""

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import shapely

NAMES = ["Duct", "Elbow", "Elbow vuông", "Concentric Reducer", "Eccentric Reducer", "Reducer vuông tròn", "Chân rẽ", "Chân rẽ vuông tròn", "Tee"]
DUCT, ELBOW, ELBOW_SQ, CONC, ECC, RED_RT, BRANCH, BRANCH_RT, TEE = range(9)


class Page:
    def __init__(self, w_pt, h_pt, scale, rng):
        self.s, self.rng = scale, rng
        self.img = np.full((int(h_pt * scale), int(w_pt * scale), 3), 255, np.uint8)
        self.w, self.h = w_pt, h_pt
        self.objects = []  # (class, polygon pt)
        self.occupied = []  # duct footprints, to keep runs apart
        dark = rng.random() < 0.5
        self.duct_col = self.colour(0.0, 0.35 if dark else 0.6)
        self.duct_px = rng.choice([1, 2, 2, 3, 4])
        self.arch_col = self.colour(0.35, 0.8, grey=rng.random() < 0.7)
        self.arch_px = rng.choice([1, 1, 2])
        self.centre = rng.random() < 0.6  # centre lines on round ducts
        self.insul = rng.random() < 0.25  # an outer insulation outline on rectangular ducts

    def colour(self, lo, hi, grey=None):
        r = self.rng
        if grey or (grey is None and r.random() < 0.5):
            v = int(255 * r.uniform(lo, hi))
            return (v, v, v)
        c = [int(255 * r.uniform(lo, hi)) for _ in range(3)]
        c[r.randrange(3)] = int(255 * r.uniform(0, lo))
        return tuple(c)

    def P(self, p):
        return (int(round(p[0] * self.s)), int(round(p[1] * self.s)))

    def line(self, a, b, col, px, dash=None):
        if dash:
            L = math.dist(a, b)
            if L < 1e-6:
                return
            u = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
            t, on, i = 0.0, True, 0
            while t < L:
                d = dash[i % len(dash)]
                if on:
                    e = min(t + d, L)
                    cv2.line(self.img, self.P((a[0] + u[0] * t, a[1] + u[1] * t)), self.P((a[0] + u[0] * e, a[1] + u[1] * e)), col, px, cv2.LINE_AA)
                t += d
                on, i = not on, i + 1
        else:
            cv2.line(self.img, self.P(a), self.P(b), col, px, cv2.LINE_AA)

    def poly(self, pts, col, px, closed=True):
        cv2.polylines(self.img, [np.array([self.P(p) for p in pts], np.int32)], closed, col, px, cv2.LINE_AA)

    def arc(self, c, r, a0, a1, col, px, n=24):
        pts = [(c[0] + r * math.cos(a0 + (a1 - a0) * i / n), c[1] + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]
        self.poly(pts, col, px, closed=False)
        return pts

    def label(self, cls, pts):
        self.objects.append((cls, [tuple(map(float, p)) for p in pts]))


def frame(p0, d, n, along, across):
    """Point at distance ``along`` from p0 in direction d and ``across`` to its left normal n."""
    return (p0[0] + d[0] * along + n[0] * across, p0[1] + d[1] * along + n[1] * across)


def draw_duct(pg, p0, d, L, w, round_duct=False, split=True):
    """A straight duct of width w from p0 along d (unit, axis-aligned), cut into labelled pieces; returns the end point."""
    n = (-d[1], d[0])
    r = pg.rng
    cuts = [0.0]
    if split and L > 40:
        for _ in range(r.choice([0, 0, 1, 1, 2, 3])):
            cuts.append(r.uniform(0.1, 0.9) * L)
    cuts = sorted(set(cuts)) + [L]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 2:
            continue
        q = [frame(p0, d, n, a, -w / 2), frame(p0, d, n, b, -w / 2), frame(p0, d, n, b, w / 2), frame(p0, d, n, a, w / 2)]
        pg.label(DUCT, q)
    pg.line(frame(p0, d, n, 0, -w / 2), frame(p0, d, n, L, -w / 2), pg.duct_col, pg.duct_px)
    pg.line(frame(p0, d, n, 0, w / 2), frame(p0, d, n, L, w / 2), pg.duct_col, pg.duct_px)
    if pg.insul and not round_duct:
        g = w / 2 + max(1.5, w * 0.12)
        pg.line(frame(p0, d, n, 0, -g), frame(p0, d, n, L, -g), pg.duct_col, 1, dash=(3, 2))
        pg.line(frame(p0, d, n, 0, g), frame(p0, d, n, L, g), pg.duct_col, 1, dash=(3, 2))
    if round_duct and pg.centre:
        pg.line(p0, frame(p0, d, n, L, 0), pg.duct_col, 1, dash=(6, 2, 1, 2))
    for c in cuts[1:-1]:  # damper / flange mark at a split, as the GT cuts ducts at VCD / FD
        kind = r.random()
        pg.line(frame(p0, d, n, c, -w / 2), frame(p0, d, n, c, w / 2), pg.duct_col, pg.duct_px)
        if kind < 0.5:
            pg.line(frame(p0, d, n, c - w * 0.3, -w / 2), frame(p0, d, n, c + w * 0.3, w / 2), pg.duct_col, 1)
        elif kind < 0.8:
            pg.poly([frame(p0, d, n, c - 1.5, -w / 2 - 2), frame(p0, d, n, c + 1.5, -w / 2 - 2), frame(p0, d, n, c + 1.5, w / 2 + 2), frame(p0, d, n, c - 1.5, w / 2 + 2)],
                    pg.duct_col, 1)
    pg.occupied.append(shapely.Polygon([frame(p0, d, n, 0, -w), frame(p0, d, n, L, -w), frame(p0, d, n, L, w), frame(p0, d, n, 0, w)]))
    return frame(p0, d, n, L, 0)


def elbow(pg, p, d, w, turn):
    """90° turn at p (end of a duct going d) to the left (turn=1) or right (-1); returns (new start, new direction)."""
    n = (-d[1], d[0])
    nd = (n[0] * turn, n[1] * turn)
    if pg.rng.random() < 0.3:  # square elbow with turning vanes
        a = frame(p, d, n, 0, -w / 2 * turn)
        b = frame(p, d, n, w, -w / 2 * turn)
        c = frame(p, d, n, w, w / 2 * turn)
        e = frame(p, d, n, 0, w / 2 * turn)
        pg.line(a, b, pg.duct_col, pg.duct_px)
        pg.line(b, c, pg.duct_col, pg.duct_px)
        for k in range(1, 4):
            m = frame(p, d, n, w * k / 4, w / 2 * turn - w * k / 4 * turn)
            pg.arc(m, w * 0.12, 0, math.pi, pg.duct_col, 1, n=6)
        pg.label(ELBOW_SQ, [a, b, c, e])
        return frame(p, d, n, w / 2, w / 2 * turn), nd
    ri = w * pg.rng.uniform(0.4, 1.0)
    centre = frame(p, d, n, 0, (ri + w / 2) * turn)
    base = math.atan2(p[1] - centre[1], p[0] - centre[0])
    sweep = (math.pi / 2) * (1 if (d[0] * nd[1] - d[1] * nd[0]) > 0 else -1)
    inner = pg.arc(centre, ri, base, base + sweep, pg.duct_col, pg.duct_px)
    outer = pg.arc(centre, ri + w, base, base + sweep, pg.duct_col, pg.duct_px)
    pg.label(ELBOW, inner + outer[::-1])
    end = (centre[0] + (ri + w / 2) * math.cos(base + sweep), centre[1] + (ri + w / 2) * math.sin(base + sweep))
    return end, nd


def reducer(pg, p, d, w1, w2):
    n = (-d[1], d[0])
    L = max(6.0, abs(w1 - w2) * pg.rng.uniform(1.0, 2.0))
    k = pg.rng.random()
    if k < 0.5:
        cls, q = CONC, [frame(p, d, n, 0, -w1 / 2), frame(p, d, n, L, -w2 / 2), frame(p, d, n, L, w2 / 2), frame(p, d, n, 0, w1 / 2)]
    elif k < 0.85:
        cls, q = ECC, [frame(p, d, n, 0, -w1 / 2), frame(p, d, n, L, -w1 / 2), frame(p, d, n, L, -w1 / 2 + w2), frame(p, d, n, 0, w1 / 2)]
    else:
        cls, q = RED_RT, [frame(p, d, n, 0, -w1 / 2), frame(p, d, n, L, -w2 / 2), frame(p, d, n, L, w2 / 2), frame(p, d, n, 0, w1 / 2)]
    pg.line(q[0], q[1], pg.duct_col, pg.duct_px)
    pg.line(q[3], q[2], pg.duct_col, pg.duct_px)
    pg.line(q[0], q[3], pg.duct_col, 1)
    pg.line(q[1], q[2], pg.duct_col, 1)
    if cls == RED_RT:
        pg.line(q[0], q[2], pg.duct_col, 1)
        pg.line(q[3], q[1], pg.duct_col, 1)
    pg.label(cls, q)
    mid = ((q[1][0] + q[2][0]) / 2, (q[1][1] + q[2][1]) / 2)
    return mid, cls


def branch(pg, p0, d, w, at, side, wb, depth):
    """A takeoff from the side of a duct (start p0, direction d) at distance ``at``: rectangular shoe or round collar."""
    n = (-d[1], d[0])
    s = side
    bd = (n[0] * s, n[1] * s)
    root = frame(p0, d, n, at, w / 2 * s)
    r = pg.rng.random()
    if r < 0.45:  # rectangular 45° shoe
        L = wb * 0.8
        q = [frame(p0, d, n, at - wb / 2 - L, w / 2 * s), frame(p0, d, n, at + wb / 2, w / 2 * s),
             frame(p0, d, n, at + wb / 2, w / 2 * s + wb * 0.6 * s), frame(p0, d, n, at - wb / 2, w / 2 * s + wb * 0.6 * s)]
        pg.line(q[0], q[3], pg.duct_col, pg.duct_px)
        pg.line(q[1], q[2], pg.duct_col, pg.duct_px)
        pg.label(BRANCH, q)
        start = frame(p0, d, n, at, w / 2 * s + wb * 0.6 * s)
        rd = False
    elif r < 0.8:  # round collar
        c = wb * 0.35
        q = [frame(p0, d, n, at - wb / 2, w / 2 * s), frame(p0, d, n, at + wb / 2, w / 2 * s),
             frame(p0, d, n, at + wb / 2, w / 2 * s + c * s), frame(p0, d, n, at - wb / 2, w / 2 * s + c * s)]
        pg.poly(q, pg.duct_col, pg.duct_px)
        pg.label(BRANCH_RT, q)
        start = frame(p0, d, n, at, w / 2 * s + c * s)
        rd = True
    else:  # tee: square takeoff with rounded corners
        q = [frame(p0, d, n, at - wb, w / 2 * s), frame(p0, d, n, at + wb, w / 2 * s),
             frame(p0, d, n, at + wb / 2, w / 2 * s + wb * 0.5 * s), frame(p0, d, n, at - wb / 2, w / 2 * s + wb * 0.5 * s)]
        pg.line(q[0], q[3], pg.duct_col, pg.duct_px)
        pg.line(q[1], q[2], pg.duct_col, pg.duct_px)
        pg.label(TEE, q)
        start = frame(p0, d, n, at, w / 2 * s + wb * 0.5 * s)
        rd = False
    run(pg, start, bd, wb, depth + 1, round_duct=rd)
    return root


def terminal(pg, p, d, w):
    """Flexible duct (wavy double line) to a square diffuser with a cross or a round one."""
    n = (-d[1], d[0])
    r = pg.rng
    L, amp, k = r.uniform(15, 40), r.uniform(0.6, 1.5), r.uniform(2.5, 4)
    for side in (-1, 1):
        pts = [frame(p, d, n, t, side * w * 0.25 + amp * math.sin(t / k * math.pi)) for t in np.linspace(0, L, 40)]
        pg.poly(pts, pg.duct_col, 1, closed=False)
    c = frame(p, d, n, L + w * 0.6, 0)
    s = w * r.uniform(0.6, 1.0)
    if r.random() < 0.6:
        q = [(c[0] - s, c[1] - s), (c[0] + s, c[1] - s), (c[0] + s, c[1] + s), (c[0] - s, c[1] + s)]
        pg.poly(q, pg.duct_col, pg.duct_px)
        pg.line(q[0], q[2], pg.duct_col, 1)
        pg.line(q[1], q[3], pg.duct_col, 1)
        if r.random() < 0.5:
            pg.poly([(c[0] - s * 0.6, c[1] - s * 0.6), (c[0] + s * 0.6, c[1] - s * 0.6), (c[0] + s * 0.6, c[1] + s * 0.6), (c[0] - s * 0.6, c[1] + s * 0.6)], pg.duct_col, 1)
    else:
        cv2.circle(pg.img, pg.P(c), int(s * pg.s), pg.duct_col, pg.duct_px, cv2.LINE_AA)
        cv2.circle(pg.img, pg.P(c), int(s * 0.5 * pg.s), pg.duct_col, 1, cv2.LINE_AA)


def run(pg, p, d, w, depth, round_duct=False):
    r = pg.rng
    for _ in range(r.randint(1, 4)):
        L = r.uniform(25, 160)
        end = frame(p, d, (-d[1], d[0]), L, 0)
        if not (20 < end[0] < pg.w - 20 and 20 < end[1] < pg.h - 20):
            L = max(0.0, min(L, (pg.w - 20 - p[0]) / d[0] if d[0] > 0 else (p[0] - 20) / -d[0] if d[0] < 0 else L,
                             (pg.h - 20 - p[1]) / d[1] if d[1] > 0 else (p[1] - 20) / -d[1] if d[1] < 0 else L))
        if L < 12:
            return
        foot = shapely.Polygon([frame(p, d, (-d[1], d[0]), 2, -w), frame(p, d, (-d[1], d[0]), L, -w), frame(p, d, (-d[1], d[0]), L, w), frame(p, d, (-d[1], d[0]), 2, w)])
        if any(foot.intersects(o) for o in pg.occupied[:-1]):
            return
        start = p
        p = draw_duct(pg, p, d, L, w, round_duct)
        if depth < 3 and L > 3 * w and r.random() < 0.55:
            branch(pg, start, d, w, r.uniform(1.5 * w, L - 1.5 * w), r.choice([-1, 1]), max(5.0, w * r.uniform(0.4, 0.8)), depth)
        k = r.random()
        if k > 0.85:  # run ends at a diffuser through a flexible duct: frequent false positives on real pages, never labelled
            terminal(pg, p, d, w)
            return
        if k < 0.45:
            p, d = elbow(pg, p, d, w, r.choice([-1, 1]))
        elif k < 0.75 and w > 9:
            w2 = max(5.0, w * r.uniform(0.5, 0.8))
            p, cls = reducer(pg, p, d, w, w2)
            w = w2
            round_duct = round_duct or cls == RED_RT
        else:
            return


def architecture(pg):
    r = pg.rng
    col, px = pg.arch_col, pg.arch_px
    sx, sy = r.uniform(80, 200), r.uniform(80, 200)
    xs = np.arange(r.uniform(10, 60), pg.w, sx)
    ys = np.arange(r.uniform(10, 60), pg.h, sy)
    for x in xs:
        pg.line((x, 0), (x, pg.h), col, 1, dash=(14, 3, 2, 3))
    for y in ys:
        pg.line((0, y), (pg.w, y), col, 1, dash=(14, 3, 2, 3))
    c = r.uniform(4, 10)
    for x in xs:
        for y in ys:
            if r.random() < 0.7:
                q = [(x - c / 2, y - c / 2), (x + c / 2, y - c / 2), (x + c / 2, y + c / 2), (x - c / 2, y + c / 2)]
                cv2.fillPoly(pg.img, [np.array([pg.P(p) for p in q], np.int32)], col)
    for _ in range(r.randint(4, 14)):  # walls: double lines
        x0, y0 = r.uniform(0, pg.w), r.uniform(0, pg.h)
        horiz = r.random() < 0.5
        L, t = r.uniform(60, 400), r.uniform(3, 8)
        a, b = ((x0, y0), (x0 + L, y0)) if horiz else ((x0, y0), (x0, y0 + L))
        off = (0, t) if horiz else (t, 0)
        pg.line(a, b, col, px + 1)
        pg.line((a[0] + off[0], a[1] + off[1]), (b[0] + off[0], b[1] + off[1]), col, px + 1)
        if r.random() < 0.4:  # door swing
            pg.arc(a, r.uniform(15, 30), 0, math.pi / 2, col, 1)
    for _ in range(r.randint(0, 6)):  # hatched areas
        x0, y0, ww, hh = r.uniform(0, pg.w), r.uniform(0, pg.h), r.uniform(20, 120), r.uniform(20, 120)
        for k in np.arange(0, ww + hh, r.uniform(3, 8)):
            pg.line((x0 + max(0, k - hh), y0 + min(k, hh)), (x0 + min(k, ww), y0 + max(0, k - ww)), col, 1)
    for _ in range(r.randint(10, 60)):  # text-like clutter and size labels
        x0, y0 = r.uniform(0, pg.w), r.uniform(0, pg.h)
        hgt = r.uniform(2, 6)
        tc = pg.duct_col if r.random() < 0.4 else col
        x = x0
        for _ in range(r.randint(3, 10)):
            gx = [(x + r.uniform(0, hgt * 0.6), y0 + r.uniform(0, hgt)) for _ in range(r.randint(2, 4))]
            pg.poly(gx, tc, 1, closed=False)
            x += hgt * 0.7
    for _ in range(r.randint(0, 6)):  # equipment boxes with a cross (FCU, fans): not HVAC objects in scope
        x0, y0, ww, hh = r.uniform(0, pg.w), r.uniform(0, pg.h), r.uniform(10, 50), r.uniform(10, 40)
        q = [(x0, y0), (x0 + ww, y0), (x0 + ww, y0 + hh), (x0, y0 + hh)]
        pg.poly(q, pg.duct_col if r.random() < 0.5 else col, pg.duct_px)
        pg.line(q[0], q[2], pg.duct_col, 1)
        pg.line(q[1], q[3], pg.duct_col, 1)


def make_page(w, h, scale, seed):
    rng = random.Random(seed)
    pg = Page(w, h, scale, rng)
    architecture(pg)
    for _ in range(rng.randint(2, 6)):
        d = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        p = (rng.uniform(60, w - 60), rng.uniform(60, h - 60))
        run(pg, p, d, rng.uniform(8, 32), 0, round_duct=rng.random() < 0.15)
    return pg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pages", type=int, default=300)
    ap.add_argument("--size", default="900x650")
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    w, h = map(float, a.size.split("x"))
    out = Path(a.out)
    (out / "images" / a.split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / a.split).mkdir(parents=True, exist_ok=True)
    counts = [0] * len(NAMES)
    for i in range(a.pages):
        pg = make_page(w, h, a.scale, a.seed * 100000 + i)
        H, W = pg.img.shape[:2]
        lines = []
        for c, q in pg.objects:
            r = shapely.get_coordinates(shapely.minimum_rotated_rectangle(shapely.multipoints(np.array(q))))[:4]
            if len(r) < 4:
                continue
            xy = np.clip(r * a.scale / [W, H], 0, 1)
            lines.append(f"{c} " + " ".join(f"{v:.6f}" for v in xy.ravel()))
            counts[c] += 1
        stem = f"syn_{a.seed}_{i:05d}"
        cv2.imwrite(str(out / "images" / a.split / f"{stem}.png"), pg.img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        (out / "labels" / a.split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("pages", a.pages, "objects", dict(zip(NAMES, counts)))
    (out / "synth.json").write_text(json.dumps({"pages": a.pages, "size_pt": [w, h], "scale": a.scale, "counts": dict(zip(NAMES, counts))}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
