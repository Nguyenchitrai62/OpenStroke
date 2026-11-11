"""OpenStroke13 / E92: objects seeded by the detector's boxes, grown along the stroke graph (docs/OPENSTROKE13.md, H107).

v10-v12 decode objects as connected components of the edge graph with the class voted by the strokes. With edge precision 0.03-0.07
(F83-F86) and stroke classes right 12-18 % of the time that gives object F2 ~0.05-0.1 (F95, research/probes/graph_error_sim.py). Here
every detector box is an object (its class and score), its strokes start as the box's own members, and the graph only adds strokes
reached through confident edges *inside the box grown by ``margin`` px*: an edge can extend a duct cut between two tiles, never chain
across the page. A stroke claimed by several boxes goes to the highest score when scored (``assemble.assign_members``).

The simulation gives F2 0.524 with the same poor edges (connected components: 0.052) and 0.624 with perfect ones.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

SEEDED_EDGES = (0.5, 0.7, 0.9)  # edge thresholds tried in validation (the edge head is not calibrated to the 0.16 % base rate)


def seeded_objects(gp, boxes, vec, names, edge_thr=0.5, keep_thr=0.5, margin=16.0, source=None):
    """``gp``: the page's GraphPage; ``boxes``: detector detections ({"class_id", "score", "poly" (8 page px), "members" (primitive
    ids)}); ``vec``: the page's vector npz dict. -> one detection per box with source "seeded" (or ``source``), its members grown."""
    import shapely

    from openstroke.obb.data.vectors import page_columns

    tok, _, _, pid = page_columns(vec)
    mid = tok[:, 1:17].reshape(-1, 8, 2).mean(1)
    at = {int(p): i for i, p in enumerate(pid.tolist())}
    kept = set()
    for p, s in gp.prob.items():  # strokes the graph sees as part of an object (P(no object) < keep_thr)
        if s[0] / gp.count[p] < keep_thr:
            kept.add(p)
    adj = defaultdict(list)
    for (a, b), q in gp.edges.items():
        if q >= edge_thr and a in kept and b in kept:
            adj[a].append(b)
            adj[b].append(a)
    out = []
    for box in boxes:
        members = {int(m) for m in box.get("members") or [] if int(m) in at}
        if not members:
            continue
        region = shapely.Polygon(np.asarray(box["poly"], float).reshape(4, 2)).buffer(margin)
        shapely.prepare(region)
        inside = {}

        def ok(t):
            if t not in inside:
                x, y = mid[at[t]] if t in at else (np.nan, np.nan)
                inside[t] = bool(np.isfinite(x) and shapely.contains_xy(region, x, y))
            return inside[t]

        grown, frontier = set(members), list(members)
        while frontier:
            s = frontier.pop()
            for t in adj.get(s, ()):
                if t not in grown and ok(t):
                    grown.add(t)
                    frontier.append(t)
        out.append({**{k: v for k, v in box.items() if k != "members"}, "members": sorted(grown),
                    "source": source or "seeded", "seed_members": len(members)})
    return out
