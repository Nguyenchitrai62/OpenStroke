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
    if np.any(edges < 0) or np.any(edges >= n):
        raise ValueError("Edge endpoint outside graph")
    parent = list(range(n))
    sizes = [1] * n
    exclude = [set() for _ in range(n)]

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Highest confidence first. Float16 archives often tie; use canonical endpoint order.
    order = np.lexsort((edges[:, 1], edges[:, 0], logits >= 0, -np.abs(logits)))
    for k in order:
        a, b = (root(int(i)) for i in edges[k])
        if a == b:
            continue
        if logits[k] < 0:
            exclude[a].add(b)
            exclude[b].add(a)
        elif b not in exclude[a]:
            if sizes[a] < sizes[b]:
                a, b = b, a
            for other in exclude[b]:
                exclude[other].remove(b)
                exclude[other].add(a)
            exclude[a].update(exclude[b])
            exclude[b].clear()
            parent[b] = a
            sizes[a] += sizes[b]
    return np.fromiter((root(i) for i in range(n)), dtype=np.int64, count=n)


def stroke_instances(prob, threshold, edges, logits):
    """Return an object id per original shape (-1 for unselected shapes). No GT required."""
    prob = np.asarray(prob)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    logits = np.asarray(logits)
    if logits.shape != (len(edges),) or np.any(edges < 0) or np.any(edges >= len(prob)):
        raise ValueError("Malformed stroke edges or logits")
    nodes = np.flatnonzero(prob >= threshold)
    membership = np.full(len(prob), -1, dtype=np.int64)
    membership[nodes] = np.arange(len(nodes))
    local = membership[edges]
    selected = (local >= 0).all(1)
    labels = mutex_components(len(nodes), local[selected], logits[selected])
    membership[nodes] = np.unique(labels, return_inverse=True)[1].ravel()
    return membership
