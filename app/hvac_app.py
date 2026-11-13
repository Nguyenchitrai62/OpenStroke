"""OpenStroke HVAC test app (OpenStroke28 stroke model): drop a PDF, pick a page, see which strokes the model picks, move the threshold and
watch the selection change at once, then run AI_Takeoff's HVAC logic on the strokes above the threshold and see the objects it builds.

    py -3.12 app/hvac_app.py --model F:\\Downloads\\v28.pt [--port 8765]      (run_app.bat does this and opens the browser)

Steps for a page, the same as training (no augmentation, no layers for the model):
  1. Trích PDF          tools/extract_payload.py with AI_Takeoff's Python (pdf_etl / PyMuPDF stay out of this package)
  2. Đọc nét vẽ         every stroke -> shape, length, direction, colour, width, dash / fill, neighbours (openstroke.features)
  3. Chạy model         one probability per stroke (openstroke.infer.Predictor; the checkpoint says it reads no layers)
The threshold, the colours and the selection are computed in the browser. "Chạy logic HVAC" sends the strokes with p >= threshold to
extract_hvac.detection.run_detection (tools/run_hvac_logic.py), on their own PDF layer or all on one layer, and shows its objects.
Working files live in a temporary folder removed when the app stops. Only standard-library HTTP; no Streamlit.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXTRACTOR = os.environ.get("OPENSTROKE_EXTRACTOR_PYTHON", r"F:\Source_code\AI_Takeoff\BE\.venv\Scripts\python.exe")
HTML = Path(__file__).with_name("hvac_viewer.html")
WORK = Path(tempfile.mkdtemp(prefix="openstroke_hvac_"))
APP = {"pred": None, "info": None, "error": None, "model": None}
DOCS: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
MODEL_LOCK = threading.Lock()  # one page through the model at a time (CPU)


def tool(script: str, *args, timeout: int = 1800) -> dict:
    """A tool run with AI_Takeoff's Python; -> the JSON of its last output line."""
    r = subprocess.run([EXTRACTOR, str(ROOT / "tools" / script), *map(str, args)], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"{script} lỗi ({EXTRACTOR}):\n" + (r.stderr or r.stdout)[-1500:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def load_model(path: str, threads: int) -> None:
    try:
        import psutil
        import torch

        from openstroke.infer import Predictor
        torch.set_flush_denormal(True)
        workers = 2 if psutil.virtual_memory().available >= 12 * 2**30 else 1
        pred = Predictor(path, threads=threads, workers=workers)
        info = pred.describe()
        info["no_layers"] = bool(pred.config.get("no_layers"))
        APP["warming"] = True
        t0 = time.time()
        rng = np.random.default_rng(0)  # warm-up: the first page through the model pays for allocations / thread pools, not the user's
        xy = rng.uniform(0, 500, (400, 4))
        pred.predict_payload({"shapes": [{"items": [["l", [a, b], [c, d]]], "type": "s", "color": [0, 0, 0], "width": 0.3, "layer": "0",
                                          "rect": [min(a, c), min(b, d), max(a, c), max(b, d)]} for a, b, c, d in xy],
                              "document_metadata": {"page_width": 500.0, "page_height": 500.0}}, mode="hvac", tta=1)
        info["warmup_s"] = round(time.time() - t0, 1)
        APP.update(pred=pred, info=info, model=path, warming=False)
    except Exception as e:  # shown in the page
        APP["error"] = f"Không nạp được model {path}: {type(e).__name__}: {e}"


# ------------------------------------------------------------------ page runs
STEP_NAMES = (("pdf", "Trích PDF"), ("features", "Tiền xử lý (giống lúc train)"), ("model", "Chạy model"),
              ("post", "Hậu kỳ: logic HVAC → object (Duct, Elbow, Reducer, Chân rẽ…)"))
# Post-processing uses no PDF layer (user 2026-11-13): "one" every picked stroke on one layer, or "style" = layer_1, one layer per stroke
# colour + width (what any PDF has). E114 on the 5 GT val pages, 6 classes (F144, F145): layer_1 at p >= 0.05 0.841 (the best of every way,
# PDF layers 0.836, the model as a layer selector 0.840), one layer 0.820-0.823.
POST_DEFAULT = {"how": "style", "thr": 0.05}
LAYER_MEAN = 0.1


def new_job(doc: dict, page: int) -> dict:
    """The job of a page (made once per page; queued, not started: the PDF's queue thread runs the pages one after another)."""
    with doc["lock"]:
        if page in doc["jobs"]:
            return JOBS[doc["jobs"][page]]
        jid = uuid.uuid4().hex[:10]
        job = {"id": jid, "doc": doc["id"], "page": page, "t0": None, "error": None, "done": False, "started": False,
               "steps": [{"key": k, "name": n, "state": "wait", "seconds": None, "detail": ""} for k, n in STEP_NAMES]}
        JOBS[jid] = job
        doc["jobs"][page] = jid
        return job


def want(doc: dict, page: int) -> dict:
    """The user opened this page: it runs next (after the page now in the model)."""
    job = new_job(doc, page)
    with doc["lock"]:
        if not job["started"]:
            doc["order"] = [page] + [x for x in doc["order"] if x != page]
    return job


def run_queue(doc: dict) -> None:
    """Every page of the PDF in turn (page 1 first, a page the user opens jumps the line), then its preview image."""
    while True:
        with doc["lock"]:
            if not doc["order"]:
                return
            page = doc["order"].pop(0)
        job = new_job(doc, page)
        if job["started"]:
            continue
        job["started"], job["t0"] = True, time.time()
        run_page(job, doc)
        if job["done"]:
            try:
                job["preview"] = tool("extract_payload.py", "preview", doc["pdf"], page, timeout=300)["png"]
            except Exception:  # noqa: BLE001  (the page still shows, without the picture underneath)
                pass


def step(job: dict, key: str, detail: str = "") -> None:
    now = time.time()
    for s in job["steps"]:
        if s["state"] == "run" and s["key"] != key:
            s["state"], s["seconds"] = "done", round(now - s["t"], 1)
        if s["key"] == key:
            if s["state"] != "run":
                s["state"], s["t"] = "run", now
            s["detail"] = detail


def run_page(job: dict, doc: dict) -> None:
    try:
        while APP.get("pred") is None or APP.get("warming"):   # a PDF dropped while the model still loads waits for it
            if APP.get("error"):
                raise RuntimeError(APP["error"])
            step(job, "pdf", "chờ model nạp xong")
            time.sleep(0.5)
        pfile = Path(doc["dir"]) / f"page-{job['page']:04d}.json.gz"
        step(job, "pdf")
        if not pfile.exists():
            tool("extract_payload.py", "page", doc["pdf"], job["page"], pfile)
        from openstroke.sources import load_payload
        payload = load_payload(pfile)
        step(job, "features", "chờ model rảnh" if MODEL_LOCK.locked() else "")
        with MODEL_LOCK:
            def on_stage(name, done, total):
                step(job, name, f"{done} / {total} cửa sổ" if total else "")
            out = APP["pred"].predict_payload(payload, mode="hvac", tta=1, on_stage=on_stage)
        info = out["info"]
        seg = info["segments"]
        xy = seg.seg.reshape(-1, 2)
        lo = xy.min(axis=0) if len(xy) else np.zeros(2)
        span = np.maximum((xy.max(axis=0) - lo) if len(xy) else np.ones(2), 1e-6)
        q = np.round((xy - lo) / span * 65535).astype(np.uint16)
        attrs = info["attrs"]
        job["data"] = {"payload": str(pfile), "prob": out["prob"], "segs": seg, "attrs": attrs, "length": info["length"].astype(np.float64)}
        job["result"] = {
            "id": job["id"], "file": doc["name"], "page": job["page"], "pages": doc["pages"], "w": info["width"], "h": info["height"],
            "n": int(info["shapes"]), "m": int(len(seg.seg)), "kept": out["kept"], "lo": lo.tolist(), "span": span.tolist(),
            "seg": b64(q), "off": b64(seg.shape_offsets.astype(np.int32)), "prob": b64(np.round(out["prob"] * 255).astype(np.uint8)),
            "len": b64(info["length"].astype(np.float32)), "layer": b64(attrs["layer_id"].astype(np.int32)),
            "layers": [str(n) for n in attrs["layer_names"]], "seconds": {s["key"]: s["seconds"] for s in job["steps"]},
            "total_s": None}
        step(job, "post")
        try:
            job["result"]["logic"] = run_logic(job, POST_DEFAULT["thr"], POST_DEFAULT["how"])
        except Exception as e:  # noqa: BLE001  (the model result still shows; the logic error is reported with it)
            job["result"]["logic"] = {"error": f"{type(e).__name__}: {e}"}
        step(job, "_end")
        job["result"]["seconds"] = {s["key"]: s["seconds"] for s in job["steps"]}
        job["result"]["total_s"] = round(time.time() - job["t0"], 1)
        job["done"] = True
    except Exception as e:  # noqa: BLE001
        job["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        for s in job["steps"]:
            if s["state"] == "run":
                s["state"] = "fail"


def run_logic(job: dict, thr: float, how: str) -> dict:
    """AI_Takeoff's HVAC logic on the strokes with p >= thr: "one" all on one layer (no layers at all), "style" on layer_1 (one layer per
    stroke colour + width). "real" (PDF layers) and "layers" (the model as a layer selector) stay for research probes, not in the app."""
    from openstroke.objects import HVAC_CLASSES, assign_objects
    from openstroke.relabel import object_strokes

    d = job["data"]
    doc = DOCS[job["doc"]]
    if how == "layers":
        names = np.array([str(x) for x in d["attrs"]["layer_names"]])[d["attrs"]["layer_id"]]
        uniq, inv = np.unique(names, return_inverse=True)
        mean = np.bincount(inv, d["prob"] * d["length"], len(uniq)) / np.maximum(np.bincount(inv, d["length"], len(uniq)), 1e-9)
        chosen = [str(x) for x in uniq[(mean >= LAYER_MEAN) & (uniq != "")]]
        sel = np.flatnonzero(np.isin(names, chosen))
        spec = {"layers": chosen, "stray": []}
        thr = LAYER_MEAN
    else:
        sel = np.flatnonzero(d["prob"] >= thr)
        spec = {"selected": sel.tolist(), "how": how}
    key = f"{job['page']}_{how}_{thr:.2f}"
    choice = Path(doc["dir"]) / f"choice_{key}.json"
    out = Path(doc["dir"]) / f"logic_{key}.json"
    t0 = time.time()
    if not out.exists():
        choice.write_text(json.dumps(spec), encoding="utf-8")
        tool("run_hvac_logic.py", d["payload"], choice, doc["pdf"], job["page"], out)
    res = json.loads(out.read_text(encoding="utf-8"))
    objects = res["objects"]
    mask = object_strokes("hvac", objects, d["segs"], d["attrs"], d["length"]) if objects else np.zeros(len(d["prob"]), bool)
    cls, inst, _ = assign_objects(d["segs"], objects, mask)
    from openstroke.objects import class_of
    counts, polys = {}, []
    for o in objects:
        k = class_of(o.get("shape_name"))
        n = HVAC_CLASSES[k] if k else f"khác: {o.get('shape_name')}"
        counts[n] = counts.get(n, 0) + 1
        v = o.get("vertices") or []   # one entry per object, same index as inst (an object without a polygon keeps an empty one)
        polys.append({"c": HVAC_CLASSES[k] if k else "khác", "name": o.get("shape_name"),
                      "v": np.round(np.asarray(v, float), 1).tolist() if len(v) >= 3 else []})
    return {"thr": thr, "how": how, "selected": int(len(sel)), "objects": len(objects), "texts": res.get("texts", 0), "counts": counts,
            "classes": list(HVAC_CLASSES), "cls": b64(cls.astype(np.int8)), "inst": b64(inst.astype(np.int32)), "polys": polys,
            "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet console
        return

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                body = HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/model":
                ready = APP.get("pred") is not None and not APP.get("warming")
                self.send_json({"model": APP["model"], "info": APP["info"] if ready else None, "error": APP["error"], "post": POST_DEFAULT,
                                "loading": not ready and APP["error"] is None})
            elif u.path == "/api/thumbs":
                doc = DOCS[q["doc"]]
                if "thumbs" not in doc:
                    out = Path(doc["dir"]) / "thumbs"
                    tool("extract_payload.py", "thumbs", doc["pdf"], out, timeout=120)
                    doc["thumbs"] = [{"page": i, "w": round(w), "h": round(h),
                                      "img": base64.b64encode((out / f"page-{i:04d}.jpg").read_bytes()).decode() if (out / f"page-{i:04d}.jpg").exists() else None}
                                     for i, (w, h) in enumerate(doc["sizes"], 1)]
                self.send_json(doc["thumbs"])
            elif u.path == "/api/doc":
                doc = DOCS[q["doc"]]
                st = {}
                for page in range(1, doc["pages"] + 1):
                    j = JOBS.get(doc["jobs"].get(page, ""))
                    st[page] = "done" if j and j["done"] else "error" if j and j["error"] else "run" if j and j["started"] else "wait"
                self.send_json({"pages": st})
            elif u.path == "/api/job":
                job = JOBS[q["id"]]
                el = round(time.time() - job["t0"], 1) if job["t0"] else 0.0
                self.send_json({"steps": [{k: s[k] for k in ("name", "state", "seconds", "detail")} for s in job["steps"]], "elapsed": el,
                                "done": job["done"], "error": job["error"], "queued": not job["started"]})
            elif u.path == "/api/result":
                self.send_json(JOBS[q["id"]]["result"])
            elif u.path == "/api/preview":
                job = JOBS[q["id"]]
                if "preview" not in job:
                    job["preview"] = tool("extract_payload.py", "preview", DOCS[job["doc"]]["pdf"], job["page"], timeout=300)["png"]
                self.send_json({"png": job["preview"]})
            else:
                self.send_json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/pdf":
                data = self.body()
                did = uuid.uuid4().hex[:10]
                d = WORK / did
                d.mkdir(parents=True)
                pdf = d / "file.pdf"
                pdf.write_bytes(data)
                info = tool("extract_payload.py", "info", pdf, timeout=300)
                doc = DOCS[did] = {"id": did, "dir": str(d), "pdf": str(pdf), "name": q.get("name") or "file.pdf", "pages": info["pages"],
                                   "sizes": info["sizes"], "jobs": {}, "order": list(range(1, info["pages"] + 1)), "lock": threading.Lock()}
                threading.Thread(target=run_queue, args=(doc,), daemon=True).start()   # all pages, in the background, right away
                self.send_json({"doc": did, "name": doc["name"], "pages": info["pages"]})
            elif u.path == "/api/run":
                a = json.loads(self.body() or b"{}")
                if APP["error"]:
                    raise RuntimeError(APP["error"])
                doc = DOCS[a["doc"]]
                job = want(doc, max(1, min(int(a["page"]), doc["pages"])))
                self.send_json({"job": job["id"]})
            elif u.path == "/api/logic":
                a = json.loads(self.body() or b"{}")
                job = JOBS[a["job"]]
                how = a.get("how") if a.get("how") in ("one", "style") else POST_DEFAULT["how"]
                job["result"]["logic"] = run_logic(job, round(float(a["thr"]), 2), how)
                self.send_json(job["result"]["logic"])
            else:
                self.send_json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # a cmd window's code page cannot print Vietnamese: never crash on a message
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("OPENSTROKE_MODEL", ""))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    model = a.model.strip().strip('"')
    if not model or not Path(model).exists():
        print(f"Không thấy model: {model!r}. Sửa MODEL= trong run_app.bat hoặc kéo thả file .pt vào run_app.bat.")
        return 1
    if not Path(EXTRACTOR).exists():
        print(f"Không thấy Python của AI_Takeoff ({EXTRACTOR}); đặt OPENSTROKE_EXTRACTOR_PYTHON trỏ tới python có pdf_etl.")
        return 1
    threads = max(1, (os.cpu_count() or 2) // 2)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    threading.Thread(target=load_model, args=(model, threads), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"OpenStroke HVAC: {url}  (model {model}; Ctrl+C để dừng)")
    if not a.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        shutil.rmtree(WORK, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
