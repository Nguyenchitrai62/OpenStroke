"""Payloads for UNLABELED drawing pages (self-supervised pretraining, E19): every PDF under ``--src``, plan-like pages only.

Run with AI_Takeoff's venv (``pdf_etl`` / PyMuPDF; the OpenStroke package never imports them). Reads PDFs, writes only
``--out``. A PDF whose sha256 is a document of the AI_Takeoff DB with GT, or of the dev split, is skipped (no dev page may be
seen even unlabeled). Plan-like = at least ``--min-drawings`` vector drawings (spec sheets and text pages have few). At most
``--per-pdf`` pages per PDF, evenly spread, so one 60-page set does not dominate.

    F:\\Source_code\\AI_Takeoff\\BE\\.venv\\Scripts\\python.exe tools/make_unlabeled.py --src F:\\Source_code\\_TONG_HOP_BAN_VE_TRAINING\\FIRE
"""

import argparse
import gzip
import hashlib
import json
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_payloads import BE, build  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def plan_pages(pdf: str, min_drawings: int, per_pdf: int) -> list[tuple[int, int]]:
    import fitz

    out = []
    with fitz.open(pdf) as d:
        for i in range(d.page_count):
            try:
                n = len(d.load_page(i).get_cdrawings())
            except Exception:
                continue
            if n >= min_drawings:
                out.append((i + 1, n))
    if len(out) > per_pdf:
        step = len(out) / per_pdf
        out = [out[int(k * step)] for k in range(per_pdf)]
    return out


def job(pdf: str, page: int, dst: str) -> tuple[str, int, int, float]:
    t0 = time.time()
    p = build(pdf, page)
    tmp = Path(dst + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=1) as fh:
        json.dump(p, fh, ensure_ascii=False)
    tmp.replace(dst)
    return dst, page, len(p.get("shapes") or []), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("datasets/unlabeled/payloads"))
    ap.add_argument("--dev-folds", type=Path, default=Path("datasets/e2/folds_dev.json"))
    ap.add_argument("--min-drawings", type=int, default=5000)
    ap.add_argument("--per-pdf", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    con = sqlite3.connect((BE / "data" / "ai_takeoff.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    gt_sha = {s for (s,) in con.execute("select d.sha256 from documents d join ground_truths g on g.document_id = d.id")}
    con.close()
    dev = {g["sha256"] for g in json.loads(a.dev_folds.read_text(encoding="utf-8"))["groups"] if g["fold"] == 0}
    a.out.mkdir(parents=True, exist_ok=True)
    todo, index = [], []
    for pdf in sorted(a.src.rglob("*.pdf")):
        s = sha256(pdf)
        if s in gt_sha or s in dev:
            print(f"skip (GT / dev) {pdf.name}", flush=True)
            continue
        try:
            pages = plan_pages(str(pdf), a.min_drawings, a.per_pdf)
        except Exception as e:
            print(f"skip (unreadable: {e!r}) {pdf.name}", flush=True)
            continue
        for page, n in pages:
            dst = a.out / f"{s[:12]}_p{page:04d}.json.gz"
            index.append({"file": str(pdf), "sha256": s, "page_number": page, "drawings": n, "payload": dst.name})
            if not dst.exists():
                todo.append((str(pdf), page, str(dst)))
        print(f"{pdf.name}: {len(pages)} plan pages", flush=True)
    (a.out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(index)} pages, {len(todo)} to build -> {a.out}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(job, *t) for t in todo]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                dst, page, n, dt = f.result()
                print(f"[{i}/{len(todo)}] {Path(dst).name}: {n} shapes, {dt:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"[{i}/{len(todo)}] FAILED: {e!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
