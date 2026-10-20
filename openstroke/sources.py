"""Read-only access to AI_Takeoff ground truth (SQLite) and ``split_countfire`` payloads.

Nothing here writes: the database is opened with ``mode=ro`` and payloads are only read.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_AI_TAKEOFF_BE = Path(os.environ.get("OPENSTROKE_AI_TAKEOFF_BE", r"F:\Source_code\AI_Takeoff\BE"))

#: Groups a human has reviewed. ``detected``, ``water`` and ``detection`` are debug output of the old
#: pipeline (AI_Takeoff ``evaluate.DEFAULT_GROUPS``); learning from them would copy its mistakes.
GT_GROUPS = ("primary", "reference")

_PAYLOAD_RE = re.compile(r"^page-(\d{4})\.v(\d+)\.json\.gz$")


@dataclass
class GTPage:
    gt_id: int
    document_id: int
    mode: str  # "fire" | "hvac"
    filename: str
    sha256: str
    page_number: int  # 1-based
    page_width: float | None
    page_height: float | None
    updated_at: str
    objects: list[dict] = field(default_factory=list)  # primary/reference only

    @property
    def page_key(self) -> tuple[str, int]:
        return (self.sha256, self.page_number)

    @property
    def stem(self) -> str:
        return f"doc{self.document_id:06d}_p{self.page_number:04d}_gt{self.gt_id}"


def connect_ro(db_path: str | Path) -> sqlite3.Connection:
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def load_gt_pages(db_path: str | Path, groups: tuple[str, ...] = GT_GROUPS) -> list[GTPage]:
    """Every ground-truth page that has at least one object of ``groups``."""

    con = connect_ro(db_path)
    try:
        rows = con.execute(
            "select g.id, g.document_id, g.mode, d.filename, d.sha256, g.pages_json, g.updated_at "
            "from ground_truths g join documents d on d.id = g.document_id order by g.document_id, g.id"
        ).fetchall()
    finally:
        con.close()
    out: list[GTPage] = []
    for gt_id, doc, mode, filename, sha, pages_json, updated in rows:
        for page in json.loads(pages_json or "[]"):
            objs = [o for o in page.get("objects") or [] if o.get("group") in groups]
            if not objs:
                continue
            out.append(GTPage(gt_id=int(gt_id), document_id=int(doc), mode=str(mode).casefold(),
                              filename=filename, sha256=sha, page_number=int(page["page_number"]),
                              page_width=page.get("page_width"), page_height=page.get("page_height"),
                              updated_at=str(updated), objects=objs))
    return out


def group_by_page(pages: list[GTPage]) -> dict[tuple[str, int], list[GTPage]]:
    """Group GT pages by (document sha256, page number): one key per physical page."""

    groups: dict[tuple[str, int], list[GTPage]] = {}
    for p in pages:
        groups.setdefault(p.page_key, []).append(p)
    return groups


def latest_payload_path(drawings_dir: str | Path, document_id: int, page_number: int) -> Path | None:
    """Highest ``vK`` payload of a page, ignoring ``.view.`` files."""

    folder = Path(drawings_dir) / f"document-{document_id:06d}"
    best: tuple[int, Path] | None = None
    if not folder.is_dir():
        return None
    for f in folder.iterdir():
        m = _PAYLOAD_RE.match(f.name)
        if not m or int(m.group(1)) != page_number:
            continue
        v = int(m.group(2))
        if best is None or v > best[0]:
            best = (v, f)
    return best[1] if best else None


def payload_version(path: Path) -> int:
    m = _PAYLOAD_RE.match(path.name)
    return int(m.group(2)) if m else -1


def load_payload(path: str | Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
