"""The test app's model process: loads the model once, then runs each page it is sent through every step.

Its own process because polling for progress re-runs the Streamlit script: in the same process those re-runs fight the model for the GIL
(a page's model time went from 5.3 s to 15 s, measured). It reads one job file path per line from stdin and exits when stdin closes, so it
ends with the app. Progress goes to a small status file, the result to a JSON file.

    python app/page_worker.py <model.pt> <threads> <info.json>
"""

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openstroke.infer import Predictor  # noqa: E402
from openstroke.postprocess import ALONG_THRESHOLD, along_confident, hvac_input, objects_mask  # noqa: E402
from openstroke.sources import load_payload  # noqa: E402

STEPS = (("extract", "Trích PDF"), ("features", "Tính đặc trưng"), ("model", "Chạy model"), ("post", "Hậu kỳ"))


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def run_tool(python: str, script: str, *args) -> dict:
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([python, str(ROOT / "tools" / script), *map(str, args)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{script} lỗi ({python}):\n" + (r.stderr or r.stdout)[-1500:])
    return json.loads(r.stdout.strip().splitlines()[-1])


class Status:
    def __init__(self, path: Path):
        self.path, self.t0, self.current, self.started = path, time.time(), None, None
        self.steps = {k: {"name": n, "state": "wait", "seconds": None, "detail": ""} for k, n in STEPS}
        self.error = None
        self.write()

    def write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"steps": [{"key": k, **s} for k, s in self.steps.items()], "t0": self.t0, "error": self.error}), encoding="utf-8")
        tmp.replace(self.path)

    def step(self, key: str | None, detail: str = "") -> None:
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
        self.write()

    def fail(self, message: str) -> None:
        self.error = message
        if self.current:
            self.steps[self.current]["state"] = "fail"
        self.write()


def post_process(job: dict, st: Status, payload: dict, out: dict, pfile: Path) -> tuple[np.ndarray, dict]:
    info, p, mode = out["info"], out["prob"], job["mode"]
    segs = info["segments"]
    if mode != "hvac":
        t = ALONG_THRESHOLD[mode]
        st.step("post", f"nét dọc nét p ≥ {t}")
        return along_confident(segs, p, t), {"kind": "Nét dọc nét", "threshold": t, "note": "PLUMBING dùng ngưỡng của FIRE, chưa đo." if mode == "plumbing" else ""}
    layers, stray = hvac_input(p, info["length"].astype(np.float64), [str(s.get("layer") or "") for s in payload.get("shapes") or []])
    st.step("post", f"logic hình học, {len(layers)} layer")
    tag = f"{pfile.stem}.{mode}.{hashlib.sha256(np.round(p * 255).astype(np.uint8).tobytes()).hexdigest()[:12]}"
    choice, res = pfile.with_name(tag + ".choice.json"), pfile.with_name(tag + ".objects.json")
    if not res.exists():
        choice.write_text(json.dumps({"layers": layers, "stray": np.flatnonzero(stray).tolist()}), encoding="utf-8")
        run_tool(job["python"], "run_hvac_logic.py", pfile, choice, job["pdf"], job["page"], res)
    logic = json.loads(res.read_text(encoding="utf-8"))
    objects = logic["objects"]
    return objects_mask("hvac", objects, segs, info["attrs"]["layer_id"]), {
        "kind": "Chọn layer + logic hình học", "layers": len(layers), "stray": int(stray.sum()), "objects": len(objects), "texts": logic["texts"],
        "note": "Chỉ dùng chữ trong PDF, chưa có OCR, detector và symbol như pipeline đầy đủ.",
    }


def work(job: dict, st: Status, pred: Predictor) -> None:
    st.step("extract")
    pfile = Path(job["payload"])
    if not pfile.exists():
        run_tool(job["python"], "extract_payload.py", "page", job["pdf"], job["page"], pfile)
    payload = load_payload(pfile)

    def on_stage(name: str, done: int, total: int) -> None:
        st.step(name, f"{done} / {total} cửa sổ" if total else "")

    out = pred.predict_payload(payload, mode=job["mode"], tta=job["tta"], on_stage=on_stage)
    info = out["info"]
    st.step("post")
    final, post = post_process(job, st, payload, out, pfile)
    st.step(None)
    seg = info["segments"]
    xy = seg.seg.reshape(-1, 2)
    lo = xy.min(axis=0) if len(xy) else np.zeros(2)
    span = np.maximum((xy.max(axis=0) - lo) if len(xy) else np.ones(2), 1e-6)
    q = np.round((seg.seg.reshape(-1, 2) - lo) / span * 65535).astype(np.uint16)  # 2 bytes per coordinate
    attrs = info["attrs"]
    result = {
        "w": info["width"], "h": info["height"], "n": int(info["shapes"]), "m": int(len(seg.seg)),
        "lo": lo.tolist(), "span": span.tolist(), "seg": _b64(q), "off": _b64(seg.shape_offsets.astype(np.int32)),
        "prob": _b64(np.round(out["prob"] * 255).astype(np.uint8)), "final": _b64(final.astype(np.uint8)), "post": post,
        "len": _b64(info["length"].astype(np.float32)), "layer": _b64(attrs["layer_id"].astype(np.int32)), "layers": [str(n) for n in attrs["layer_names"]],
        "kept": out["kept"], "seconds": {k: s["seconds"] for k, s in st.steps.items()}, "total_s": round(time.time() - st.t0, 1), "threads": job["threads"],
    }
    out_file = Path(job["result"])
    tmp = out_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(result), encoding="utf-8")
    tmp.replace(out_file)


def main(model: str, threads: str, info_file: str) -> int:
    torch.set_flush_denormal(True)
    info = Path(info_file)
    try:
        pred = Predictor(model, threads=int(threads))
        tmp = info.with_suffix(".tmp")
        tmp.write_text(json.dumps({"info": pred.describe()}), encoding="utf-8")
    except Exception as e:  # shown in the viewer's panel
        tmp = info.with_suffix(".tmp")
        tmp.write_text(json.dumps({"error": f"Không nạp được model {model}: {type(e).__name__}: {e}"}), encoding="utf-8")
        tmp.replace(info)
        return 1
    tmp.replace(info)
    for line in sys.stdin:
        job = json.loads(Path(line.strip()).read_text(encoding="utf-8"))
        st = Status(Path(job["status"]))
        try:
            work(job, st, pred)
        except Exception as e:  # one bad page must not stop the app; shown in the viewer's panel
            st.fail(f"{type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
