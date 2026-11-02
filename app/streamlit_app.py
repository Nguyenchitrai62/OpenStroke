"""OpenStroke quick test: drop a PDF, see which strokes the model and its post-processing call pipe.

    pip install streamlit
    streamlit run app/streamlit_app.py      (or double-click run_app.bat)

Flow (as in AI_Takeoff): PDF page -> ``split_countfire`` payload (``tools/extract_payload.py``, run with AI_Takeoff's venv
because ``pdf_etl`` / PyMuPDF stay out of this package) -> the same arrays and features as a training page
(``openstroke.infer``) -> the model -> one probability per shape -> post-processing per system (``openstroke.postprocess``).
The whole window is the viewer (``app/viewer``): drop or pick a PDF, click a page in the page list (a one-page PDF runs at once), choose
the mode, model path and threshold in its left panel; zoom / pan, colours, threshold and the JSON export run in the browser.
A page runs in its own process (``app/page_worker.py``) and the viewer polls a status file, so it can show each step and its time
without slowing the model down.

Change ``MODEL`` at the top of ``run_app.bat``, or ``MODEL_PATH`` below, or type another path in the panel. ``EXTRACTOR_PYTHON`` must have ``pdf_etl``.
"""

import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_PATH = str(ROOT / "runs_colab" / "e37" / "fold0" / "best.pt")  # E37 fold 0 (val pages: H1_KS.pdf, 02.DN-TT-FP-BLOCK A_R1-Model.pdf)
MODEL_PATH = os.environ.get("OPENSTROKE_MODEL") or MODEL_PATH  # run_app.bat sets it (MODEL=...)
EXTRACTOR_PYTHON = os.environ.get("OPENSTROKE_EXTRACTOR_PYTHON", r"F:\Source_code\AI_Takeoff\BE\.venv\Scripts\python.exe")
CACHE = ROOT / "runs" / "app_cache"  # payloads and results of dropped PDFs (gitignored)
THREADS = max(1, (os.cpu_count() or 2) // 2)  # physical cores: 2.7x faster than 1 thread on this machine, measured

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


@st.cache_data(show_spinner=False, max_entries=8)
def model_info(path: str, mtime: float) -> dict:
    """Read in a short-lived process: the app itself stays free of torch."""

    code = ("import json,sys;sys.path.insert(0,sys.argv[2]);from openstroke.infer import Predictor;"
            "print(json.dumps(Predictor(sys.argv[1],device='cpu',threads=1).describe()))")
    r = subprocess.run([sys.executable, "-c", code, path, str(ROOT)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ, OMP_NUM_THREADS="1"))
    if r.returncode != 0:
        raise RuntimeError(f"Không nạp được model {path}:\n" + (r.stderr or r.stdout)[-1500:])
    return json.loads(r.stdout.strip().splitlines()[-1])


@st.cache_resource(show_spinner=False)
def running() -> dict:
    """Worker processes by status file."""

    return {}


@st.cache_data(show_spinner=False, max_entries=4)
def read_result(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def page_job(model: str, mtime: float, pdf: Path, sha: str, page: int, mode: str, tta: int) -> tuple[Path, Path]:
    """-> (status file, result file); starts the worker unless the page is done or running."""

    d = CACHE / sha
    tag = hashlib.sha256(f"{model}|{mtime}|{page}|{mode}|{tta}".encode()).hexdigest()[:16]
    status, result = d / f"{tag}.status.json", d / f"{tag}.result.json"
    procs = running()
    proc = procs.get(str(status))
    if proc is not None and proc.poll() is not None and not result.exists() and status.exists():  # ended without a result: report it once
        s = json.loads(status.read_text(encoding="utf-8"))
        if not s.get("error"):
            status.write_text(json.dumps({**s, "error": f"Tiến trình xử lý trang dừng đột ngột (mã {proc.returncode})."}), encoding="utf-8")
        procs.pop(str(status))
    elif not result.exists() and proc is None and not (status.exists() and json.loads(status.read_text(encoding="utf-8")).get("error")):
        job = d / f"{tag}.job.json"
        job.write_text(json.dumps({"model": model, "pdf": str(pdf), "page": page, "mode": mode, "tta": tta, "threads": THREADS, "python": EXTRACTOR_PYTHON,
                                   "payload": str(d / f"page-{page:04d}.v0.json.gz"), "status": str(status), "result": str(result)}), encoding="utf-8")
        status.unlink(missing_ok=True)
        procs[str(status)] = subprocess.Popen([sys.executable, str(ROOT / "app" / "page_worker.py"), str(job)], env=dict(os.environ, OMP_NUM_THREADS=str(THREADS)))
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
        mt = path.stat().st_mtime
        args["modelinfo"] = model_info(str(path), mt)
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
                status, result = page_job(str(path), mt, pdf, sha, page, mode, tta)
                base = {"file": file, "page": page, "mode": mode}
                if result.exists():
                    args["result"] = {**read_result(str(result), result.stat().st_mtime), **base, "pages": info["pages"], "sha": sha, "tta": tta}
                else:
                    s = json.loads(status.read_text(encoding="utf-8")) if status.exists() else {"steps": [], "t0": None}
                    if s.get("error"):  # shown once; the next request for this page tries again
                        args["error"] = s["error"]
                        status.unlink(missing_ok=True)
                        running().pop(str(status), None)
                    args["progress"] = {**base, "steps": s["steps"], "t0": s.get("t0")}
    except Exception as e:  # shown in the viewer's panel
        args["error"] = f"{type(e).__name__}: {e}"
    viewer(**args, key="viewer", default=None)


if __name__ == "__main__":
    main()
