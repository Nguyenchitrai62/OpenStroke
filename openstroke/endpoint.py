"""Reference HTTP endpoint of the HVAC duct vector model (the contract the extract-duct module calls).

ai-torch-service hosts the model behind ``POST <path>`` (default ``/hvac/duct-vector/predict``); this module is both the
reference implementation of that contract and a local server for tests (AI_Takeoff, the extract-duct worker):

    python -m openstroke.endpoint e151_os46_unified_hvac/fold0/last_deploy.pt --port 8090
    # then: API_TORCH_URL=http://127.0.0.1:8090 for the extract-duct module

Request JSON: ``{"payload": {"shapes": [...], "document_metadata": {"page_width": w, "page_height": h}}, "tta": 1}``.
Response JSON: ``{"prob": [n floats], "threshold": t, "group_objects": {"instance_id": [n ints], "cluster_ids": [K ints],
"cluster_probs": [[8 floats] x K]}, "model": {"name", "version", "device", "precision"}}`` with one entry per shape, in
the order of ``shapes``. Only the raw model output: the post-processing runs in the extract-duct module.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_PATH = "/hvac/duct-vector/predict"


def respond(predictor, body: dict[str, Any]) -> dict[str, Any]:
    """Raw model output for one request body (see the module docstring); ``predictor`` is an ``openstroke.Predictor``."""
    from . import __version__

    payload = body.get("payload") if isinstance(body.get("payload"), dict) else body
    if not isinstance(payload.get("shapes"), list):
        raise ValueError("request needs payload.shapes (the page JSON drawing)")
    out = predictor.predict_payload(payload, tta=int(body.get("tta", 1) or 1))
    groups = out.get("group_objects")
    info = predictor.describe()
    resp: dict[str, Any] = {
        "prob": np.round(np.asarray(out["prob"], np.float64), 5).tolist(),
        "threshold": float(predictor.threshold),
        "model": {"name": Path(str(predictor.path)).stem, "version": __version__, "device": info.get("device"),
                  "precision": info.get("precision")},
    }
    if groups is not None:
        resp["group_objects"] = {
            "instance_id": np.asarray(groups["instance_id"], np.int64).tolist(),
            "cluster_ids": np.asarray(groups["cluster_ids"], np.int64).tolist(),
            "cluster_probs": np.round(np.asarray(groups["cluster_probs"], np.float64), 5).tolist(),
        }
    return resp


def serve(model: str, host: str = "127.0.0.1", port: int = 8090, path: str = DEFAULT_PATH, device: str | None = None) -> None:
    from .infer import Predictor

    predictor = Predictor(model, device=device or "auto")
    if predictor.device.type == "cuda":
        predictor.warmup()
    lock = threading.Lock()   # one page at a time on the model

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):   # health check
            self._send(200, {"ok": True, "model": Path(model).stem, "runtime": predictor.runtime})

        def do_POST(self):
            if self.path.rstrip("/") != path.rstrip("/"):
                return self._send(404, {"error": f"unknown path {self.path}, use {path}"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                t0 = time.perf_counter()
                with lock:
                    resp = respond(predictor, body)
                resp["model"]["seconds"] = round(time.perf_counter() - t0, 3)
                self._send(200, resp)
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt, *args):
            print(f"[endpoint] {self.address_string()} {fmt % args}", flush=True)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[endpoint] {Path(model).name} on http://{host}:{port}{path} | {predictor.runtime}", flush=True)
    server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Reference HTTP endpoint of the HVAC duct vector model")
    ap.add_argument("model", help="checkpoint .pt (deploy or training)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--device", default=None, help="auto | cpu | cuda:0")
    a = ap.parse_args(argv)
    serve(a.model, a.host, a.port, a.path, a.device)


if __name__ == "__main__":
    main()
