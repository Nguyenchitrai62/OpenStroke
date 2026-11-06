"""E65 teacher, step 4: AI_Takeoff's HVAC geometry logic on each corpus page, reading the layers the stroke model chose
(``tools/teacher_hvac.py``). Run with AI_Takeoff's Python (``run_hvac_logic.py``: logic and PDF text use PyMuPDF). Writes only
``<corpus>/objects/<key>.json``.

    <python> tools/teacher_logic.py datasets/hvac_corpus --workers 3
"""

import argparse
import contextlib
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def one(payload: str, choice: str, pdf: str, page: int, out: str) -> tuple[str, int, float]:
    from run_hvac_logic import main as logic

    t0 = time.time()
    tmp = out + ".tmp"
    with contextlib.redirect_stdout(io.StringIO()):
        logic(payload, choice, pdf, page, tmp)
    Path(tmp).replace(out)
    return out, len(json.loads(Path(out).read_text(encoding="utf-8"))["objects"]), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    c = Path(a.corpus)
    (c / "objects").mkdir(exist_ok=True)
    rows = {r["key"]: r for r in json.loads((c / "pages.json").read_text(encoding="utf-8"))}
    todo = [k for k in rows if (c / "choice" / f"{k}.choice.json").exists() and not (c / "objects" / f"{k}.json").exists()]
    print("todo", len(todo), flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        fut = {ex.submit(one, str(c / "pages" / f"{k}.json.gz"), str(c / "choice" / f"{k}.choice.json"), rows[k]["pdf"], rows[k]["page"],
                         str(c / "objects" / f"{k}.json")): k for k in todo}
        for f in as_completed(fut):
            try:
                _, n, s = f.result()
                print(fut[f], n, "objects", f"{s:.0f}s", flush=True)
            except Exception as e:
                print("fail", fut[f], type(e).__name__, str(e)[:160], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
