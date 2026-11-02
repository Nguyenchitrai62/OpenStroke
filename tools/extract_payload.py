"""PDF -> ``split_countfire`` payload, for the test app (``app/streamlit_app.py``). Run with a Python that has AI_Takeoff's
``pdf_etl`` (its venv): the OpenStroke package never imports ``pdf_etl`` / PyMuPDF (AGPL).

    <python> tools/extract_payload.py info <file.pdf>                 -> {"pages": n, "sizes": [[w, h], ...]}
    <python> tools/extract_payload.py page <file.pdf> <page> <out.json.gz>
    <python> tools/extract_payload.py thumbs <file.pdf> <out_dir>     -> {"done": n}; page-PPPP.jpg, 240 px wide, stops after 40 s
"""

import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_payloads import build  # noqa: E402  (same payload as AI_Takeoff drawing_cache)

THUMB_WIDTH = 240
THUMB_SECONDS = 40


def main(argv):
    if argv[0] == "info":
        import fitz

        with fitz.open(argv[1]) as d:
            print(json.dumps({"pages": d.page_count, "sizes": [[p.rect.width, p.rect.height] for p in d]}))
        return 0
    if argv[0] == "thumbs":
        import fitz

        out = Path(argv[2])
        out.mkdir(parents=True, exist_ok=True)
        t0, done = time.time(), 0
        with fitz.open(argv[1]) as d:
            for i, page in enumerate(d, 1):
                f = out / f"page-{i:04d}.jpg"
                if not f.exists():
                    if time.time() - t0 > THUMB_SECONDS:  # very heavy pages: the picker still lists them, without a picture
                        break
                    z = THUMB_WIDTH / max(page.rect.width, 1.0)
                    page.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False, annots=False).save(f, jpg_quality=70)
                done += 1
        print(json.dumps({"done": done}))
        return 0
    if argv[0] == "page":
        payload = build(argv[1], int(argv[2]))
        with gzip.open(argv[3], "wt", encoding="utf-8", compresslevel=1) as fh:
            json.dump(payload, fh, ensure_ascii=False)
        print(json.dumps({"shapes": len(payload.get("shapes") or [])}))
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
