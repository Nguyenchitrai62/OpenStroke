"""Build the Colab upload: one zip with the E0 dataset (npz + manifest + folds).

The notebook installs the code from GitHub (https://github.com/Nguyenchitrai62/OpenStroke), so push before
asking the user to train.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path


def pack(data: Path, folds: Path, out: Path, extra_folds=()) -> Path:
    """Zip ``<data.name>/`` (e0, e1...): manifest, ``folds.json`` (= ``folds``), other fold files by name, npz."""

    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    root = Path(data).name
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:  # npz are already compressed
        zf.write(data / "manifest.json", f"{root}/manifest.json")
        zf.write(folds, f"{root}/folds.json")
        for f in extra_folds:
            zf.write(f, f"{root}/{Path(f).name}")
        for r in manifest["records"]:
            if r.get("status") == "ok":
                zf.write(data / r["npz"], f"{root}/{r['npz']}")
    return out
