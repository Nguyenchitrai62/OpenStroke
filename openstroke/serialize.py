"""Space-filling-curve serialization (Point Transformer V3): z-order and Hilbert codes of 2-D points.

Codes are int64 over a ``2**BITS`` grid. A token set sorted by z-order keeps every grid cell of size
``2**s`` contiguous, which is what makes pooling by ``code >> 2*s`` a contiguous-segment reduction.
"""

from __future__ import annotations

import numpy as np

BITS = 16  # 65536 cells per axis: 1 pt quantization covers any drawing sheet


def quantize(xy: np.ndarray, grid: float = 1.0) -> np.ndarray:
    q = np.floor((xy - xy.min(axis=0, keepdims=True)) / grid).astype(np.int64) if len(xy) else np.zeros((0, 2), np.int64)
    return np.clip(q, 0, (1 << BITS) - 1)


def _spread(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint64)
    v = (v | (v << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
    v = (v | (v << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
    v = (v | (v << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    v = (v | (v << np.uint64(2))) & np.uint64(0x3333333333333333)
    v = (v | (v << np.uint64(1))) & np.uint64(0x5555555555555555)
    return v


def zorder(q: np.ndarray) -> np.ndarray:
    """(n, 2) int grid coords -> (n,) int64 Morton code (x on even bits)."""

    return (_spread(q[:, 0]) | (_spread(q[:, 1]) << np.uint64(1))).astype(np.int64)


def hilbert(q: np.ndarray, bits: int = BITS) -> np.ndarray:
    """(n, 2) int grid coords -> (n,) int64 Hilbert index (classic xy2d, vectorized)."""

    n = 1 << bits
    x = q[:, 0].astype(np.int64).copy()
    y = q[:, 1].astype(np.int64).copy()
    d = np.zeros(len(q), np.int64)
    s = n >> 1
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        flip = (ry == 0) & (rx == 1)
        x = np.where(flip, n - 1 - x, x)
        y = np.where(flip, n - 1 - y, y)
        swap = ry == 0
        x, y = np.where(swap, y, x), np.where(swap, x, y)
        s >>= 1
    return d


ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")


def codes(q: np.ndarray, order: str) -> np.ndarray:
    """Code of ``order`` for grid coords ``q``. Hilbert uses just enough bits for the extent (an order-k curve
    is itself a Hilbert curve; 12 bits instead of 16 on a sheet saves a quarter of the work)."""

    if order.startswith("hilbert"):
        bits = max(1, int(q.max()).bit_length()) if len(q) else 1
        return hilbert(q if order == "hilbert" else q[:, ::-1], bits=bits)
    if order == "z":
        return zorder(q)
    if order == "z-trans":
        return zorder(q[:, ::-1])
    if order == "hilbert":
        return hilbert(q)
    if order == "hilbert-trans":
        return hilbert(q[:, ::-1])
    raise ValueError(order)
