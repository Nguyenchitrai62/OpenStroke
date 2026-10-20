"""Layer names as model input (H15, LEDGER F19): words and character 3-grams of a PDF layer name, hashed to ids.

Names are free text and differ between offices, but discipline words recur ("ACMV", "FF", "THOAT NUOC", "ARC", "STR",
"DUCT"...): a token-count probe beat the no-learning baseline by +0.29 / +0.39 F2 on two folds and lost on two others,
so the model gets them as a zero-init input with name dropout during training, never as a rule.
"""

from __future__ import annotations

import re
import unicodedata
import zlib

import numpy as np

N_BUCKETS = 4096  # id 0 = padding / no name
MAX_TOKENS = 48  # per layer name


def tokens(name: str) -> list[str]:
    s = unicodedata.normalize("NFKD", str(name).lower()).encode("ascii", "ignore").decode()
    words = [w for w in re.split(r"[^a-z0-9]+", s) if w and not w.isdigit()]
    out = [f"w:{w}" for w in words]
    for w in words:
        p = f"^{w}$"
        out += [f"c:{p[i:i + 3]}" for i in range(len(p) - 2)]
    return list(dict.fromkeys(out))  # unique, in order


def token_ids(name: str) -> np.ndarray:
    ids = [1 + zlib.crc32(t.encode()) % (N_BUCKETS - 1) for t in tokens(name)]
    return np.array(sorted(set(ids))[:MAX_TOKENS], np.int64)


def layer_table(names) -> np.ndarray:
    """(n_layers, MAX_TOKENS) token ids per page-local layer id, 0-padded."""

    tab = np.zeros((max(len(names), 1), MAX_TOKENS), np.int64)
    for i, n in enumerate(names):
        ids = token_ids(n)
        tab[i, : len(ids)] = ids
    return tab
