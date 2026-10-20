"""OpenStroke quick test: drop a PDF, see which strokes the model calls pipe.

    pip install streamlit
    streamlit run app/streamlit_app.py

Flow (as in AI_Takeoff): PDF page -> ``split_countfire`` payload (``tools/extract_payload.py``, run with AI_Takeoff's venv
because ``pdf_etl`` / PyMuPDF stay out of this package) -> the same arrays and features as a training page
(``openstroke.infer``) -> the model -> one probability per shape. The whole window is the viewer (``app/viewer``):
drop or pick a PDF, choose the page, mode, model path and threshold in its left panel; zoom / pan, colours, threshold
and the JSON export run in the browser without re-running the model.

Change ``MODEL_PATH`` below (or type another path in the panel). ``EXTRACTOR_PYTHON`` must have ``pdf_etl``.
"""

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # CPU torch on this machine runs single-threaded (AGENTS.md section 7)

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openstroke.infer import Predictor  # noqa: E402
from openstroke.sources import load_payload  # noqa: E402

MODEL_PATH = r"F:\Downloads\openstroke_v2.pt"  # run A fold 0 (val pages: H1_KS.pdf, 02.DN-TT-FP-BLOCK A_R1-Model.pdf)
EXTRACTOR_PYTHON = os.environ.get("OPENSTROKE_EXTRACTOR_PYTHON", r"F:\Source_code\AI_Takeoff\BE\.venv\Scripts\python.exe")
CACHE = ROOT / "runs" / "app_cache"  # payloads of dropped PDFs (gitignored)

viewer = components.declare_component("stroke_viewer", path=str(Path(__file__).with_name("viewer")))
HIDE_PAGE = """<style>
header, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none !important}
[data-testid="stMainBlockContainer"], .block-container {padding: 0 !important; max-width: none !important}
</style>"""


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _run_extractor(*args) -> dict:
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
    return _run_extractor("info", pdf)


def payload_path(sha: str, pdf: Path, page: int) -> tuple[Path, float]:
    out = CACHE / sha / f"page-{page:04d}.v0.json.gz"
    t = time.time()
    if not out.exists():
        _run_extractor("page", pdf, page, out)
    return out, time.time() - t


@st.cache_resource(show_spinner=False, max_entries=4)
def get_model(path: str, mtime: float) -> Predictor:
    return Predictor(path)


@st.cache_data(show_spinner=False, max_entries=16)
def predict(model_path: str, mtime: float, payload_file: str, mode: str, tta: int) -> dict:
    pred = get_model(model_path, mtime)
    out = pred.predict_payload(load_payload(payload_file), mode=mode, tta=tta)
    info = out["info"]
    seg = info["segments"]
    xy = seg.seg.reshape(-1, 2)
    lo = xy.min(axis=0) if len(xy) else np.zeros(2)
    span = np.maximum((xy.max(axis=0) - lo) if len(xy) else np.ones(2), 1e-6)
    q = np.round((seg.seg.reshape(-1, 2) - lo) / span * 65535).astype(np.uint16)  # 2 bytes per coordinate
    attrs = info["attrs"]
    names = [str(n) for n in attrs["layer_names"]]
    return {
        "w": info["width"], "h": info["height"], "n": int(info["shapes"]), "m": int(len(seg.seg)),
        "lo": lo.tolist(), "span": span.tolist(), "seg": _b64(q), "off": _b64(seg.shape_offsets.astype(np.int32)),
        "prob": _b64(np.round(out["prob"] * 255).astype(np.uint8)), "len": _b64(info["length"].astype(np.float32)),
        "layer": _b64(attrs["layer_id"].astype(np.int32)), "layers": names, "kept": out["kept"],
        "seconds": out["seconds"],
    }


def main():
    st.set_page_config(page_title="OpenStroke test", page_icon="🧵", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(HIDE_PAGE, unsafe_allow_html=True)
    req = st.session_state.get("viewer") or {}  # what the viewer asked for last
    path = Path(req.get("model") or MODEL_PATH)
    path = path if path.is_absolute() else ROOT / path
    args = {"id": req.get("id", 0), "model": str(path)}
    try:
        if not path.exists():
            raise FileNotFoundError(f"Không thấy model: {path}\n(sửa MODEL_PATH trong app/streamlit_app.py hoặc gõ đường dẫn rồi Enter)")
        mt = path.stat().st_mtime
        pred = get_model(str(path), mt)
        args["modelinfo"] = pred.describe()
        if req.get("data") or req.get("sha"):
            if req.get("data"):
                sha, pdf = save_pdf(base64.b64decode(req["data"]))
            else:
                sha, pdf = req["sha"], CACHE / req["sha"] / "file.pdf"
            info = pdf_info(sha, str(pdf))
            page = max(1, min(int(req.get("page") or 1), info["pages"]))
            pfile, t_ex = payload_path(sha, pdf, page)
            mode = req.get("mode") or "fire"
            tta = 8 if req.get("tta") else 1
            res = predict(str(path), mt, str(pfile), mode, tta)
            args["result"] = {**res, "file": req.get("file") or "file.pdf", "sha": sha, "page": page,
                              "pages": info["pages"], "mode": mode, "tta": tta, "extract_s": round(t_ex, 1)}
    except Exception as e:  # shown in the viewer's panel
        args["error"] = f"{type(e).__name__}: {e}"
    viewer(**args, key="viewer", default=None)


if __name__ == "__main__":
    main()
