"""Write ``split_countfire`` payloads for GT pages that AI_Takeoff never opened (no ``page-*.json.gz`` yet).

Run with AI_Takeoff's own venv, which has ``pdf_etl`` (and PyMuPDF, AGPL): the OpenStroke package itself never
imports either. Reads the AI_Takeoff database (``mode=ro``) and uploaded PDFs; writes **only** under ``--out``
(default ``datasets/payloads``), as ``document-NNNNNN/page-PPPP.v0.json.gz``. Same payload as AI_Takeoff
``drawing_cache._build_page_json``: ``run_pipeline`` + ``bbox_all`` + ``document_metadata`` (page size, rotation,
source layers). ``--check`` rebuilds pages that AI_Takeoff already has and compares their shapes.

    F:\\Source_code\\AI_Takeoff\\BE\\.venv\\Scripts\\python.exe tools/make_payloads.py --workers 4
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BE = Path(r"F:\Source_code\AI_Takeoff\BE")
_PAYLOAD_RE = re.compile(r"^page-(\d{4})\.v(\d+)\.json\.gz$")


def existing(drawings: Path, doc: int, page: int) -> Path | None:
    folder = drawings / f"document-{doc:06d}"
    if not folder.is_dir():
        return None
    hits = [(int(m.group(2)), f) for f in folder.iterdir() if (m := _PAYLOAD_RE.match(f.name)) and int(m.group(1)) == page]
    return max(hits)[1] if hits else None


def gt_pages(db: Path, modes: set[str]) -> list[tuple[int, int, str, str]]:
    con = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = con.execute("select g.document_id, g.mode, g.pages_json, d.stored_name from ground_truths g "
                           "join documents d on d.id = g.document_id").fetchall()
    finally:
        con.close()
    out = set()
    for doc, mode, pages_json, stored in rows:
        if modes and str(mode).casefold() not in modes:
            continue
        for p in json.loads(pages_json or "[]"):
            if any(o.get("group") in ("primary", "reference") for o in p.get("objects") or []):
                out.add((int(doc), int(p["page_number"]), str(mode).casefold(), stored))
    return sorted(out)


def build(pdf: str, page: int) -> dict:
    import fitz
    from pdf_etl import run_pipeline

    payload = dict(run_pipeline(pdf_path=pdf, page_input=page))
    with fitz.open(pdf) as d:
        pg = d.load_page(page - 1)
        meta = {"page_number": page, "page_count": d.page_count, "page_width": float(pg.rect.width),
                "page_height": float(pg.rect.height), "rotation": int(pg.rotation)}
    payload.setdefault("bbox_all", [0.0, 0.0, meta["page_width"], meta["page_height"]])
    dm = dict(payload.get("document_metadata") or {})
    dm.setdefault("bbox_all", payload["bbox_all"])
    dm.update(meta)
    try:
        from pdf_etl import get_layers

        dm["source_layers"] = list(get_layers(pdf_path=Path(pdf), page_index=page) or [])
    except Exception:  # no optional content groups
        dm["source_layers"] = []
    dm["openstroke_generator"] = "tools/make_payloads.py (pdf_etl.run_pipeline)"
    payload["document_metadata"] = dm
    return payload


def job(pdf: str, doc: int, page: int, out: str) -> tuple[int, int, int, float]:
    t0 = time.time()
    p = build(pdf, page)
    dst = Path(out) / f"document-{doc:06d}" / f"page-{page:04d}.v0.json.gz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=1) as fh:
        json.dump(p, fh, ensure_ascii=False)
    tmp.replace(dst)
    return doc, page, len(p.get("shapes") or []), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--be", type=Path, default=BE)
    ap.add_argument("--out", type=Path, default=Path("datasets/payloads"))
    ap.add_argument("--modes", default="", help="comma list, e.g. plumbing (default: every mode)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check", type=int, default=0, help="rebuild N pages AI_Takeoff already has and compare")
    ap.add_argument("--pages-json", type=Path, default=None,
                    help="extra pages without GT: JSON list of {document_id, page_number} (pseudo labels, E8)")
    a = ap.parse_args()
    db, drawings, uploads = a.be / "data" / "ai_takeoff.sqlite3", a.be / "data" / "drawings", a.be / "data" / "uploads"
    modes = {m.strip() for m in a.modes.split(",") if m.strip()}
    pages = gt_pages(db, modes)
    if a.pages_json:
        import sqlite3 as _sq

        con = _sq.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
        stored = dict(con.execute("select id, stored_name from documents").fetchall())
        con.close()
        want = json.loads(a.pages_json.read_text(encoding="utf-8"))
        pages = sorted({(int(r["document_id"]), int(r["page_number"]), r.get("mode", ""), stored[int(r["document_id"])])
                        for r in want})
    if a.check:
        have = [(d, p, s) for d, p, _, s in pages if existing(drawings, d, p)][: a.check]
        for d, p, s in have:
            with gzip.open(existing(drawings, d, p), "rt", encoding="utf-8") as fh:
                ref = json.load(fh)["shapes"]
            new = json.loads(json.dumps(build(str(uploads / s), p)["shapes"]))
            same = len(ref) == len(new) and all(x == y for x, y in zip(ref, new))
            print(f"check doc {d} page {p}: AI_Takeoff {len(ref)} shapes, rebuilt {len(new)}, identical={same}", flush=True)
        return 0
    todo = [(str(uploads / s), d, p) for d, p, _, s in pages
            if not existing(drawings, d, p) and not (a.out / f"document-{d:06d}" / f"page-{p:04d}.v0.json.gz").exists()]
    print(f"{len(pages)} GT pages, {len(todo)} without a payload -> {a.out}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(job, pdf, d, p, str(a.out)) for pdf, d, p in todo]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                d, p, n, dt = f.result()
                print(f"[{i}/{len(todo)}] doc {d} page {p}: {n} shapes, {dt:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
            except Exception as e:  # keep going; the build marks the page missing_payload
                print(f"[{i}/{len(todo)}] FAILED: {e!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
