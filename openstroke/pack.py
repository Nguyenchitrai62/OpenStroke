"""Build the Colab upload: one zip holding the package source and the E0 dataset (npz + manifest + folds).

No GitHub remote exists yet, so the notebook installs OpenStroke from this zip.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path


def pack(repo: Path, data: Path, folds: Path, out: Path) -> Path:
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:  # npz are already compressed
        for f in sorted((repo / "openstroke").rglob("*.py")):
            zf.write(f, f"openstroke_src/{f.relative_to(repo).as_posix()}")
        zf.write(repo / "pyproject.toml", "openstroke_src/pyproject.toml")
        zf.write(data / "manifest.json", "e0/manifest.json")
        zf.write(folds, "e0/folds.json")
        for r in manifest["records"]:
            if r.get("status") == "ok":
                zf.write(data / r["npz"], f"e0/{r['npz']}")
    return out
