"""OpenStroke quick test: drop a PDF, see which strokes the model and its post-processing call pipe.

    pip install streamlit
    streamlit run app/streamlit_app.py      (or double-click run_app.bat)

Flow (as in AI_Takeoff): PDF page -> ``split_countfire`` payload (``tools/extract_payload.py``, run with AI_Takeoff's venv
because ``pdf_etl`` / PyMuPDF stay out of this package) -> the same arrays and features as a training page
(``openstroke.infer``) -> the model -> one probability per shape -> post-processing per system (``openstroke.postprocess``).
The whole window is the viewer (``app/viewer``): drop or pick a PDF, click a page in the page list (a one-page PDF runs at once), choose
the mode, model path and threshold in its left panel; post-processing uses the parameters calibrated for that model (``openstroke calibrate``); zoom / pan, colours, threshold and the JSON export run in the browser.
One long-lived process (``app/page_worker.py``) loads the model once and runs the pages; the viewer polls its status files, so it can show
each step and its time without slowing the model down.

Change ``MODEL`` at the top of ``run_app.bat``, or ``MODEL_PATH`` below, or type another path in the panel. ``EXTRACTOR_PYTHON`` must have ``pdf_etl``.
"""

import base64
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openstroke.postprocess import params_file  # noqa: E402


MODEL_PATH = str(ROOT / "runs_colab" / "e54" / "fold3" / "best.pt")  # best so far (F63): E54 fold 3, trained on labels v3; val pages: DC4-MEP-MV [40], 0875-FP-CT2-C2Z1-303, L30M-LUMI, 04. MECHANICAL [58], 26 M-MH-2007.1
MODEL_PATH = (os.environ.get("OPENSTROKE_MODEL") or "").strip().strip('"') or MODEL_PATH  # run_app.bat sets it (MODEL=...); quotes from the .bat are not part of the path
EXTRACTOR_PYTHON = os.environ.get("OPENSTROKE_EXTRACTOR_PYTHON", r"F:\Source_code\AI_Takeoff\BE\.venv\Scripts\python.exe")
CACHE = ROOT / "runs" / "app_cache"  # payloads and results of dropped PDFs (gitignored)
THREADS = max(1, (os.cpu_count() or 2) // 2)  # physical cores: 2.7x faster than 1 thread on this machine, measured
RESULT_VERSION = 8  # bump when post-processing changes: saved results of earlier versions no longer apply

viewer = components.declare_component("stroke_viewer", path=str(Path(__file__).with_name("viewer")))
HIDE_PAGE = """<style>
header, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none !important}
[data-testid="stMainBlockContainer"], .block-container {padding: 0 !important; max-width: none !important}
</style>"""


def _run_tool(*args) -> dict:
    cmd = [EXTRACTOR_PYTHON, str(ROOT / "tools" / "extract_payload.py"), *map(str, args)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"Trích PDF lỗi ({EXTRACTOR_PYTHON}):\n" + (r.stderr or r.stdout)[-1500:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def save_pdf(data: bytes) -> tuple[str, Path]:
    sha = hashlib.sha256(data).hexdigest()[:16]
    d = CACHE / sha
    d.mkdir(parents=True, exist_ok=True)
    pdf = d / "file.pdf"
    if not pdf.exists():
        pdf.write_bytes(data)
    return sha, pdf


@st.cache_data(show_spinner=False, max_entries=64)
def pdf_info(sha: str, pdf: str) -> dict:
    return _run_tool("info", pdf)


@st.cache_data(show_spinner=False, max_entries=16)
def page_list(sha: str, pdf: str, pages: int, sizes: tuple) -> list[dict]:
    """What the page list shows: number, size in pt and a small picture when there was time to render it."""

    out = CACHE / sha / "thumbs"
    _run_tool("thumbs", pdf, out)
    rows = []
    for i in range(1, pages + 1):
        f = out / f"page-{i:04d}.jpg"
        w, h = sizes[i - 1]
        rows.append({"page": i, "w": round(w), "h": round(h), "img": base64.b64encode(f.read_bytes()).decode() if f.exists() else None})
    return rows


@st.cache_resource(show_spinner=False)
def workers() -> dict:
    """The model process of this app; one lives at a time. Browser sessions run the script concurrently, hence the lock."""

    return {"lock": threading.Lock()}


def get_worker(model: str, mtime: float) -> dict:
    """The process holding this model, started on first use: the model loads once. A new model path replaces it."""

    store = workers()
    key = f"{model}|{mtime}"
    with store["lock"]:
        w = store.get("w")
        if w and w["key"] == key:
            if w["proc"].poll() is None:
                return w
            store.pop("w")
            try:  # the worker writes why it could not load the model before it exits: show that, not a generic message
                why = json.loads(w["info"].read_text(encoding="utf-8")).get("error")
            except (OSError, ValueError):
                why = None
            raise RuntimeError(why or f"Tiến trình model dừng đột ngột (mã {w['proc'].returncode}); thao tác lại để nạp lại model.")
        if w:
            w["proc"].stdin.close()
            store.pop("w")
        CACHE.mkdir(parents=True, exist_ok=True)
        info = CACHE / f"worker-{hashlib.sha256(key.encode()).hexdigest()[:12]}.info.json"
        info.unlink(missing_ok=True)
        proc = subprocess.Popen([sys.executable, str(ROOT / "app" / "page_worker.py"), model, str(THREADS), str(info)], stdin=subprocess.PIPE, text=True,
                                env=dict(os.environ, OMP_NUM_THREADS=str(THREADS)))
        store["w"] = {"key": key, "proc": proc, "info": info, "sent": set(), "send": threading.Lock()}
        return store["w"]


@st.cache_data(show_spinner=False, max_entries=4)
def read_result(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def page_job(worker: dict, model: str, mtime: float, pdf: Path, sha: str, page: int, mode: str, tta: int) -> tuple[Path, Path]:
    """-> (status file, result file); sends the page to the model process unless it is done or already sent."""

    d = CACHE / sha
    tag = hashlib.sha256(f"{RESULT_VERSION}|{model}|{mtime}|{page}|{mode}|{tta}".encode()).hexdigest()[:16]
    status, result = d / f"{tag}.status.json", d / f"{tag}.result.json"
    with worker["send"]:
        if not result.exists() and str(status) not in worker["sent"]:
            job = d / f"{tag}.job.json"
            job.write_text(json.dumps({"model": model, "pdf": str(pdf), "page": page, "mode": mode, "tta": tta, "threads": THREADS, "python": EXTRACTOR_PYTHON,
                                       "payload": str(d / f"page-{page:04d}.v0.json.gz"), "status": str(status), "result": str(result)}), encoding="utf-8")
            status.unlink(missing_ok=True)
            worker["proc"].stdin.write(str(job) + "\n")
            worker["proc"].stdin.flush()
            worker["sent"].add(str(status))
    return status, result


def main():
    st.set_page_config(page_title="OpenStroke test", page_icon="🧵", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(HIDE_PAGE, unsafe_allow_html=True)
    req = st.session_state.get("viewer") or {}  # what the viewer asked for last
    path = Path(req.get("model") or MODEL_PATH)
    path = path if path.is_absolute() else ROOT / path
    args = {"id": req.get("id", 0), "model": str(path)}
    try:
        if not path.exists():
            raise FileNotFoundError(f"Không thấy model: {path}\n(sửa MODEL= ở đầu run_app.bat, hoặc MODEL_PATH trong app/streamlit_app.py, hoặc gõ đường dẫn rồi Enter)")
        pf = params_file(path)
        mt = path.stat().st_mtime + (pf.stat().st_mtime if pf else 0)  # a new calibration counts as a new model: new process, new results
        worker = get_worker(str(path), mt)  # loads the model when the app opens, then only when another model is chosen
        if worker["info"].exists():
            loaded = json.loads(worker["info"].read_text(encoding="utf-8"))
            if loaded.get("error"):
                raise RuntimeError(loaded["error"])
            args["modelinfo"] = loaded["info"]
        else:
            args["loading"] = True
        if req.get("data") or req.get("sha"):
            if req.get("data"):
                sha, pdf = save_pdf(base64.b64decode(req["data"]))
            else:
                sha, pdf = req["sha"], CACHE / req["sha"] / "file.pdf"
            info = pdf_info(sha, str(pdf))
            file = req.get("file") or "file.pdf"
            args["sha"] = sha
            if not req.get("page") and info["pages"] > 1:  # the user picks the page first: no model run yet
                args["picker"] = {"sha": sha, "file": file, "pages": info["pages"],
                                  "rows": page_list(sha, str(pdf), info["pages"], tuple(map(tuple, info["sizes"])))}
            else:
                page = max(1, min(int(req.get("page") or 1), info["pages"]))
                mode = req.get("mode") or "fire"
                tta = 8 if req.get("tta") else 1
                status, result = page_job(worker, str(path), mt, pdf, sha, page, mode, tta)
                base = {"file": file, "page": page, "mode": mode}
                if result.exists():
                    args["result"] = {**read_result(str(result), result.stat().st_mtime), **base, "pages": info["pages"], "sha": sha, "tta": tta}
                else:
                    s = json.loads(status.read_text(encoding="utf-8")) if status.exists() else {"steps": [], "t0": None}
                    if s.get("error"):  # shown once; the next request for this page tries again
                        args["error"] = s["error"]
                        status.unlink(missing_ok=True)
                        worker["sent"].discard(str(status))
                    args["progress"] = {**base, "steps": s["steps"], "t0": s.get("t0")}
    except Exception as e:  # shown in the viewer's panel
        args["error"] = f"{type(e).__name__}: {e}"
    viewer(**args, key="viewer", default=None)


if __name__ == "__main__":
    main()
