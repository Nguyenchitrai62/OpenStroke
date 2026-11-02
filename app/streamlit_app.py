"""OpenStroke quick test: drop a PDF, see which strokes the model and its post-processing call pipe.

    pip install streamlit
    streamlit run app/streamlit_app.py      (or double-click run_app.bat)

Flow (as in AI_Takeoff): PDF page -> ``split_countfire`` payload (``tools/extract_payload.py``, run with AI_Takeoff's venv
because ``pdf_etl`` / PyMuPDF stay out of this package) -> the same arrays and features as a training page
(``openstroke.infer``) -> the model -> one probability per shape -> post-processing per system (``openstroke.postprocess``).
The whole window is the viewer (``app/viewer``): drop or pick a PDF, click a page in the page list (a one-page PDF runs at once), choose
the mode, model path and threshold in its left panel; zoom / pan, colours, threshold and the JSON export run in the browser.
A page runs in a background thread and the viewer polls it, so it can show each step and its time.

Change ``MODEL`` at the top of ``run_app.bat``, or ``MODEL_PATH`` below, or type another path in the panel. ``EXTRACTOR_PYTHON`` must have ``pdf_etl``.
"""

import base64
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # CPU torch on this machine runs single-threaded (AGENTS.md section 7)

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openstroke.infer import Predictor  # noqa: E402
from openstroke.postprocess import ALONG_THRESHOLD, along_confident, hvac_input, objects_mask  # noqa: E402
from openstroke.sources import load_payload  # noqa: E402

MODEL_PATH = str(ROOT / "runs_colab" / "e37" / "fold0" / "best.pt")  # E37 fold 0 (val pages: H1_KS.pdf, 02.DN-TT-FP-BLOCK A_R1-Model.pdf)
MODEL_PATH = os.environ.get("OPENSTROKE_MODEL") or MODEL_PATH  # run_app.bat sets it (MODEL=...)
EXTRACTOR_PYTHON = os.environ.get("OPENSTROKE_EXTRACTOR_PYTHON", r"F:\Source_code\AI_Takeoff\BE\.venv\Scripts\python.exe")
CACHE = ROOT / "runs" / "app_cache"  # payloads of dropped PDFs (gitignored)
STEPS = (("extract", "Trích PDF"), ("features", "Tính đặc trưng"), ("model", "Chạy model"), ("post", "Hậu kỳ"))
MAX_JOBS = 16

viewer = components.declare_component("stroke_viewer", path=str(Path(__file__).with_name("viewer")))
HIDE_PAGE = """<style>
header, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none !important}
[data-testid="stMainBlockContainer"], .block-container {padding: 0 !important; max-width: none !important}
</style>"""


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _run_tool(script: str, *args) -> dict:
    cmd = [EXTRACTOR_PYTHON, str(ROOT / "tools" / script), *map(str, args)]
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{script} lỗi ({EXTRACTOR_PYTHON}):\n" + (r.stderr or r.stdout)[-1500:])
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
    return _run_tool("extract_payload.py", "info", pdf)


@st.cache_data(show_spinner=False, max_entries=16)
def page_list(sha: str, pdf: str, pages: int, sizes: tuple) -> list[dict]:
    """What the page list shows: number, size in pt and a small picture when there was time to render it."""

    out = CACHE / sha / "thumbs"
    _run_tool("extract_payload.py", "thumbs", pdf, out)
    rows = []
    for i in range(1, pages + 1):
        f = out / f"page-{i:04d}.jpg"
        w, h = sizes[i - 1]
        rows.append({"page": i, "w": round(w), "h": round(h), "img": base64.b64encode(f.read_bytes()).decode() if f.exists() else None})
    return rows


@st.cache_resource(show_spinner=False, max_entries=4)
def get_model(path: str, mtime: float) -> Predictor:
    return Predictor(path)


@st.cache_resource(show_spinner=False)
def jobs() -> dict:
    return {"lock": threading.Lock(), "run": threading.Lock(), "all": {}}


class Job:
    """One page through every step, in a background thread; the page polls ``snapshot``."""

    def __init__(self, pred, pdf: Path, sha: str, page: int, mode: str, tta: int, meta: dict, one_at_a_time: threading.Lock):
        self.pred, self.pdf, self.sha, self.page, self.mode, self.tta, self.meta = pred, pdf, sha, page, mode, tta, meta
        self.one_at_a_time = one_at_a_time
        self.t0 = time.time()
        self.steps = {k: {"name": n, "state": "wait", "seconds": None, "detail": ""} for k, n in STEPS}
        self.current, self.started = None, None
        self.result, self.error = None, None
        threading.Thread(target=self._run, daemon=True).start()

    def _step(self, key: str | None, detail: str = "") -> None:
        now = time.time()
        if self.current and key != self.current:
            s = self.steps[self.current]
            s["state"], s["seconds"] = "done", round(now - self.started, 1)
        if key and key != self.current:
            self.current, self.started = key, now
            self.steps[key]["state"] = "run"
        if not key:
            self.current = None
        if key:
            self.steps[key]["detail"] = detail

    def _run(self) -> None:
        try:
            with self.one_at_a_time:  # one page at a time on this machine
                self.t0 = time.time()
                self._work()
        except Exception as e:  # shown in the viewer's panel
            self.error = f"{type(e).__name__}: {e}"
            if self.current:
                self.steps[self.current]["state"] = "fail"

    def _work(self) -> None:
        self._step("extract")
        pfile = CACHE / self.sha / f"page-{self.page:04d}.v0.json.gz"
        if not pfile.exists():
            _run_tool("extract_payload.py", "page", self.pdf, self.page, pfile)
        payload = load_payload(pfile)

        def on_stage(name: str, done: int, total: int) -> None:
            self._step(name, f"{done} / {total} cửa sổ" if total else "")

        out = self.pred.predict_payload(payload, mode=self.mode, tta=self.tta, on_stage=on_stage)
        info = out["info"]
        self._step("post")
        final, post = self._post(payload, out, pfile)
        self._step(None)
        seg = info["segments"]
        xy = seg.seg.reshape(-1, 2)
        lo = xy.min(axis=0) if len(xy) else np.zeros(2)
        span = np.maximum((xy.max(axis=0) - lo) if len(xy) else np.ones(2), 1e-6)
        q = np.round((seg.seg.reshape(-1, 2) - lo) / span * 65535).astype(np.uint16)  # 2 bytes per coordinate
        attrs = info["attrs"]
        names = [str(n) for n in attrs["layer_names"]]
        self.result = {
            "w": info["width"], "h": info["height"], "n": int(info["shapes"]), "m": int(len(seg.seg)),
            "lo": lo.tolist(), "span": span.tolist(), "seg": _b64(q), "off": _b64(seg.shape_offsets.astype(np.int32)),
            "prob": _b64(np.round(out["prob"] * 255).astype(np.uint8)), "final": _b64(final.astype(np.uint8)), "post": post,
            "len": _b64(info["length"].astype(np.float32)), "layer": _b64(attrs["layer_id"].astype(np.int32)), "layers": names,
            "kept": out["kept"], "seconds": {k: s["seconds"] for k, s in self.steps.items()}, "total_s": round(time.time() - self.t0, 1),
            "mode": self.mode, "tta": self.tta, "page": self.page, "sha": self.sha, **self.meta,
        }

    def _post(self, payload: dict, out: dict, pfile: Path) -> tuple[np.ndarray, dict]:
        info, p = out["info"], out["prob"]
        segs = info["segments"]
        if self.mode != "hvac":
            t = ALONG_THRESHOLD[self.mode]
            self._step("post", f"nét dọc nét p ≥ {t}")
            return along_confident(segs, p, t), {"kind": "Nét dọc nét", "threshold": t, "note": "PLUMBING dùng ngưỡng của FIRE, chưa đo." if self.mode == "plumbing" else ""}
        layers, stray = hvac_input(p, info["length"].astype(np.float64), [str(s.get("layer") or "") for s in payload.get("shapes") or []])
        self._step("post", f"logic hình học, {len(layers)} layer")
        tag = f"{pfile.stem}.{self.mode}.{hashlib.sha256(np.round(p * 255).astype(np.uint8).tobytes()).hexdigest()[:12]}"
        choice, res = pfile.with_name(tag + ".choice.json"), pfile.with_name(tag + ".objects.json")
        if not res.exists():
            choice.write_text(json.dumps({"layers": layers, "stray": np.flatnonzero(stray).tolist()}), encoding="utf-8")
            _run_tool("run_hvac_logic.py", pfile, choice, self.pdf, self.page, res)
        logic = json.loads(res.read_text(encoding="utf-8"))
        objects = logic["objects"]
        return objects_mask("hvac", objects, segs, info["attrs"]["layer_id"]), {
            "kind": "Chọn layer + logic hình học", "layers": len(layers), "stray": int(stray.sum()), "objects": len(objects), "texts": logic["texts"],
            "note": "Chỉ dùng chữ trong PDF, chưa có OCR, detector và symbol như pipeline đầy đủ.",
        }

    def snapshot(self) -> dict:
        return {"steps": [{"key": k, **s} for k, s in self.steps.items()], "elapsed": round(time.time() - self.t0, 1), "file": self.meta.get("file"), "page": self.page, "mode": self.mode}


def get_job(pred, model_key: str, pdf: Path, sha: str, page: int, mode: str, tta: int, meta: dict) -> Job:
    store = jobs()
    key = (model_key, sha, page, mode, tta)
    with store["lock"]:
        job = store["all"].get(key)
        if job is None or job.error:
            while len(store["all"]) >= MAX_JOBS:
                store["all"].pop(next(iter(store["all"])))
            job = store["all"][key] = Job(pred, pdf, sha, page, mode, tta, meta, store["run"])
    return job


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
        pred = get_model(str(path), mt)
        args["modelinfo"] = pred.describe()
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
                job = get_job(pred, f"{path}|{mt}", pdf, sha, page, mode, tta, {"file": file, "pages": info["pages"]})
                if job.error:
                    args["error"] = job.error
                    args["progress"] = job.snapshot()
                elif job.result is None:
                    args["progress"] = job.snapshot()
                else:
                    args["result"] = job.result
    except Exception as e:  # shown in the viewer's panel
        args["error"] = f"{type(e).__name__}: {e}"
    viewer(**args, key="viewer", default=None)


if __name__ == "__main__":
    main()
