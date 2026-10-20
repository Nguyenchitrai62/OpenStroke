"""Build the Colab upload: one zip with the E0 dataset (npz + manifest + folds).

The notebook installs the code from GitHub (https://github.com/Nguyenchitrai62/OpenStroke), so push before
asking the user to train.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path


def pack(data: Path, folds: Path, out: Path) -> Path:
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:  # npz are already compressed
        zf.write(data / "manifest.json", "e0/manifest.json")
        zf.write(folds, "e0/folds.json")
        for r in manifest["records"]:
            if r.get("status") == "ok":
                zf.write(data / r["npz"], f"e0/{r['npz']}")
    return out
