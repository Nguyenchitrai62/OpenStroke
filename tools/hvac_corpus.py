"""E65 teacher data, step 1-2: pick HVAC plan pages from the local drawing corpus and write their payloads. Run with AI_Takeoff's Python
(``pdf_etl`` / PyMuPDF, never imported by the package). Reads PDFs only; writes only under ``--out``.

    <python> tools/hvac_corpus.py select --pdfs datasets/hvac_corpus/pdfs.json --out datasets/hvac_corpus
    <python> tools/hvac_corpus.py extract --out datasets/hvac_corpus --workers 3

``select``: drops symbol / schedule / detail / diagram / piping sheets and every PDF of a project that has a val page (M-MH / M-SH, 9062,
DC4, the ``_test_HVAC`` copies already in the GT or pseudo sets), one copy per sha256, then up to ``--per-pdf`` pages per PDF, plan pages
first (page text says plan / mặt bằng and not diagram / detail). ``extract``: payload per page (``pages/<key>.json.gz``), pages with more
than ``--max-shapes`` shapes are skipped (memory).
"""

import argparse
import gzip
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SKIP_NAME = re.compile(r"ký hiệu|symbol|schedule|detail|flow-diagram|site-plan|piping|refrigerant|gas nc|\bFP\b|M-MH|M-SH|ACMV LAYOUT|9062|DC4-MEP|"
                       r"_test_HVAC|Y1_|H1-3-HH03|C3-CH01|Không layer|Ống đồng", re.I)
PLAN = re.compile(r"MẶT BẰNG|MAT BANG|PLAN|LAYOUT", re.I)
NOT_PLAN = re.compile(r"SƠ ĐỒ|SO DO|DIAGRAM|CHI TIẾT|CHI TIET|DETAIL|RISER|SECTION|MẶT CẮT|MAT CAT|SCHEDULE|THỐNG KÊ|THONG KE|GHI CHÚ|LEGEND|KÝ HIỆU", re.I)


def page_kind(pdf: str, n: int) -> tuple[int, float]:
    """(2 plan, 1 unknown, 0 not plan; page area): from the page's own text, title block included."""

    import fitz

    with fitz.open(pdf) as d:
        pg = d[n - 1]
        t = pg.get_text("text")[:200000]
        area = pg.rect.width * pg.rect.height
    if NOT_PLAN.search(t) and not PLAN.search(t):
        return 0, area
    return (2 if PLAN.search(t) else 1), area


def select(a) -> int:
    pdfs = json.loads(Path(a.pdfs).read_text(encoding="utf-8"))
    seen, rows = set(), []
    for r in pdfs:
        if SKIP_NAME.search(r["path"]) or r["sha256"] in seen:
            continue
        seen.add(r["sha256"])
        kinds = []
        for n in range(1, r["pages"] + 1):
            try:
                k, area = page_kind(r["path"], n)
            except Exception:
                continue
            if k and area >= 0.5 * 1190 * 842:  # A1-ish sheets and larger; A3 / A4 notes are not plans
                kinds.append((k, n))
        plan = [n for k, n in kinds if k == 2] or [n for k, n in kinds]
        step = max(1, len(plan) / a.per_pdf)
        pick = sorted({plan[int(i * step)] for i in range(min(a.per_pdf, len(plan)))})
        stem = re.sub(r"[^0-9A-Za-z]+", "_", Path(r["path"]).stem)[:40]
        for n in pick:
            rows.append({"key": f"{r['sha256'][:10]}_{stem}_p{n:04d}", "pdf": r["path"], "page": n, "sha256": r["sha256"]})
        print(len(pick), "/", r["pages"], Path(r["path"]).name, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pages.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    print("pages", len(rows))
    return 0


def one(pdf: str, page: int, dst: str, max_shapes: int) -> tuple[str, int, float]:
    from make_payloads import build

    t0 = time.time()
    p = build(pdf, page)
    n = len(p.get("shapes") or [])
    if n > max_shapes or n < 2000:
        return dst, -n, time.time() - t0
    tmp = Path(dst + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=3) as fh:
        json.dump(p, fh, ensure_ascii=False)
    tmp.replace(dst)
    return dst, n, time.time() - t0


def extract(a) -> int:
    out = Path(a.out)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    rows = json.loads((out / "pages.json").read_text(encoding="utf-8"))
    skipped = out / "skipped.json"
    skip = json.loads(skipped.read_text(encoding="utf-8")) if skipped.exists() else {}
    todo = [r for r in rows if not (out / "pages" / f"{r['key']}.json.gz").exists() and r["key"] not in skip]
    print("todo", len(todo), flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        fut = {ex.submit(one, r["pdf"], r["page"], str(out / "pages" / f"{r['key']}.json.gz"), a.max_shapes): r for r in todo}
        for f in as_completed(fut):
            r = fut[f]
            try:
                _, n, s = f.result()
            except Exception as e:
                n, s = 0, 0.0
                print("fail", r["key"], type(e).__name__, str(e)[:120], flush=True)
            if n <= 0:
                skip[r["key"]] = n
                skipped.write_text(json.dumps(skip), encoding="utf-8")
            print(r["key"], n, f"{s:.0f}s", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("select", "extract"))
    ap.add_argument("--pdfs", default="datasets/hvac_corpus/pdfs.json")
    ap.add_argument("--out", default="datasets/hvac_corpus")
    ap.add_argument("--per-pdf", type=int, default=14)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-shapes", type=int, default=700000)
    a = ap.parse_args()
    return select(a) if a.cmd == "select" else extract(a)


if __name__ == "__main__":
    sys.exit(main())
