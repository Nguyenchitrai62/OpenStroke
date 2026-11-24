"""Instance membership from signed same-object logits (H142). No fitting or layer inputs."""

import numpy as np


def mutex_components(n, edges, logits):
    """Partition a compact graph; positive logits attract, negative logits repel.

    Inputs are unique undirected links. Isolated vertices each form one component.
    Strong constraints act first, then repulsion before attraction on equal confidence.
    """
    edges = np.sort(np.asarray(edges, dtype=np.int64).reshape(-1, 2), axis=1)
    logits = np.asarray(logits, dtype=float)
    if logits.shape != (len(edges),) or not np.isfinite(logits).all():
        raise ValueError("Expected one finite logit per edge")
    if len(edges) and (int(edges.min()) < 0 or int(edges.max()) >= n):
        raise ValueError("Edge endpoint outside graph")
    if not len(edges):
        return np.arange(n, dtype=np.int64)
    parent = list(range(n))
    sizes = [1] * n
    exclude: list[set[int]] = [set() for _ in range(n)]

    # Highest confidence first. Float16 archives often tie; use canonical endpoint order.
    order = np.lexsort((edges[:, 1], edges[:, 0], logits >= 0, -np.abs(logits)))
    ea = edges[order, 0].tolist()
    eb = edges[order, 1].tolist()
    lg = logits[order].tolist()
    for a, b, w in zip(ea, eb, lg):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a == b:
            continue
        if w < 0.0:
            exclude[a].add(b)
            exclude[b].add(a)
        elif b not in exclude[a]:
            if sizes[a] < sizes[b]:
                a, b = b, a
            eb_set = exclude[b]
            if eb_set:
                for other in eb_set:
                    eo = exclude[other]
                    eo.remove(b)
                    eo.add(a)
                exclude[a].update(eb_set)
                eb_set.clear()
            parent[b] = a
            sizes[a] += sizes[b]
    for i in range(n):
        r = i
        while parent[r] != r:
            r = parent[r]
        parent[i] = r
    return np.asarray(parent, dtype=np.int64)


def stroke_instances(prob, threshold, edges, logits):
    """Return an object id per original shape (-1 for unselected shapes). No GT required."""
    prob = np.asarray(prob)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    logits = np.asarray(logits)
    n_prob = len(prob)
    if logits.shape != (len(edges),) or (len(edges) and (int(edges.min()) < 0 or int(edges.max()) >= n_prob)):
        raise ValueError("Malformed stroke edges or logits")
    picked = prob >= threshold
    nodes = np.flatnonzero(picked)
    membership = np.full(n_prob, -1, dtype=np.int64)
    if not len(nodes):
        return membership
    membership[nodes] = np.arange(len(nodes), dtype=np.int64)
    if len(edges):
        selected = picked[edges[:, 0]] & picked[edges[:, 1]]
        local = membership[edges[selected]]
        labels = mutex_components(len(nodes), local, logits[selected])
    else:
        labels = np.arange(len(nodes), dtype=np.int64)
    membership[nodes] = np.unique(labels, return_inverse=True)[1].ravel()
    return membership
