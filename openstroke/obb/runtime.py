"""Runtime-only batching/cache helpers. No trainable state or checkpoint schema changes."""
from collections import OrderedDict, defaultdict

import numpy as np
import torch


def token_groups(lengths):
    groups = defaultdict(list)
    for i, length in enumerate(lengths):
        if length:
            groups[int(length).bit_length()].append(i)
    return tuple((tuple(ids), max(lengths[i] for i in ids)) for _, ids in sorted(groups.items()))


def nbytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(nbytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(nbytes(v) for v in value)
    return 0


class ByteLRU:
    def __init__(self, limit):
        self.limit, self.bytes = int(limit), 0
        self.data = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key):
        if key not in self.data:
            self.misses += 1
            return None
        self.hits += 1
        self.data.move_to_end(key)
        return self.data[key][0]

    def put(self, key, value):
        size = nbytes(value)
        if not self.limit or size > self.limit:
            return
        if key in self.data:
            self.bytes -= self.data.pop(key)[1]
        while self.data and self.bytes + size > self.limit:
            self.bytes -= self.data.popitem(last=False)[1][1]
        self.data[key] = (value, size)
        self.bytes += size
