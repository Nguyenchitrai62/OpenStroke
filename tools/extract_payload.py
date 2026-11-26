"""PDF -> ``split_countfire`` payload, for the test app (``app/hvac_app.py``). Run with a Python that has AI_Takeoff's
``pdf_etl`` (its venv): the OpenStroke package never imports ``pdf_etl`` / PyMuPDF (AGPL).

    <python> tools/extract_payload.py info <file.pdf>                 -> {"pages": n, "sizes": [[w, h], ...]}
    <python> tools/extract_payload.py page <file.pdf> <page> <out.json[.gz]> [--no-layers]
    <python> tools/extract_payload.py preview <file.pdf> <page>      -> {"png": base64 grayscale PDF preview}
    <python> tools/extract_payload.py thumbs <file.pdf> <out_dir>     -> {"done": n}; page-PPPP.jpg, 240 px wide, stops after 40 s
    <python> tools/extract_payload.py worker                         -> persistent JSON-line RPC worker (0 cold-start, EcoQoS OFF)
"""
from __future__ import annotations

import base64
import ctypes
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PREVIEW_EDGE = 2560
THUMB_WIDTH = 240
THUMB_SECONDS = 40

try:
    import orjson as _orjson
except ImportError:
    _orjson = None


def _disable_ecoqos() -> None:
    """Opt out of Windows 11 EcoQoS E-core throttling so headless PDF extraction runs at full P-core speed."""
    if sys.platform != "win32":
        return
    try:
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        proc = k32.GetCurrentProcess()
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL

        class _PPTS(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        st = _PPTS(1, 1, 0)  # ProcessPowerThrottling (4): EXECUTION_SPEED (1), State=0 (EcoQoS OFF)
        k32.SetProcessInformation(proc, 4, ctypes.byref(st), ctypes.sizeof(st))
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass.restype = wintypes.BOOL
        k32.SetPriorityClass(proc, 0x00000020)  # NORMAL_PRIORITY_CLASS
    except Exception:
        pass


def _dump_json_bytes(obj) -> bytes:
    if _orjson is not None:
        return _orjson.dumps(obj)
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _import_fitz():
    try:
        import pymupdf as fitz
        return fitz
    except ImportError:
        import fitz
        return fitz


def _build_fast(pdf: str, page: int, *, skip_layers: bool = False) -> dict:
    """Build exact ``split_countfire`` payload; skips redundant 2nd PDF parse in ``get_layers`` when ``skip_layers=True``."""
    fitz = _import_fitz()
    from pdf_etl import run_pipeline

    payload = dict(run_pipeline(pdf_path=pdf, page_input=page))
    with fitz.open(pdf) as d:
        pg = d.load_page(page - 1)
        meta = {
            "page_number": page,
            "page_count": d.page_count,
            "page_width": float(pg.rect.width),
            "page_height": float(pg.rect.height),
            "rotation": int(pg.rotation),
        }
    payload.setdefault("bbox_all", [0.0, 0.0, meta["page_width"], meta["page_height"]])
    dm = dict(payload.get("document_metadata") or {})
    dm.setdefault("bbox_all", payload["bbox_all"])
    dm.update(meta)
    if skip_layers:
        dm.setdefault("source_layers", [])
    else:
        try:
            from pdf_etl import get_layers

            dm["source_layers"] = list(get_layers(pdf_path=Path(pdf), page_index=page) or [])
        except Exception:
            dm["source_layers"] = []
    dm["openstroke_generator"] = "tools/extract_payload.py (pdf_etl.run_pipeline)"
    payload["document_metadata"] = dm
    return payload


def _handle(argv: list[str]) -> dict:
    cmd = argv[0]
    if cmd == "info":
        fitz = _import_fitz()

        with fitz.open(argv[1]) as d:
            return {"pages": d.page_count, "sizes": [[p.rect.width, p.rect.height] for p in d]}
    if cmd == "preview":
        fitz = _import_fitz()

        with fitz.open(argv[1]) as d:
            page = d[int(argv[2]) - 1]
            z = PREVIEW_EDGE / max(page.rect.width, page.rect.height, 1.0)
            pix = page.get_pixmap(matrix=fitz.Matrix(z, z), colorspace=fitz.csGRAY, alpha=False, annots=False)
            return {"png": base64.b64encode(pix.tobytes("png")).decode("ascii")}
    if cmd == "thumbs":
        fitz = _import_fitz()

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
        return {"done": done}
    if cmd == "page":
        skip_layers = "--no-layers" in argv[4:]
        payload = _build_fast(argv[1], int(argv[2]), skip_layers=skip_layers)
        dst = Path(argv[3])
        raw = _dump_json_bytes(payload)
        if dst.name.lower().endswith(".gz"):
            dst.write_bytes(gzip.compress(raw, compresslevel=1))
        else:
            dst.write_bytes(raw)
        return {"shapes": len(payload.get("shapes") or [])}
    raise ValueError(f"Unknown command: {cmd!r}")


def _run_worker() -> int:
    """Persistent stdin/stdout JSON-line worker: keeps fitz, pdf_etl and orjson hot in RAM with EcoQoS OFF."""
    # Redirect stdout -> stderr FIRST so no deprecation/library print can ever corrupt the JSON-line RPC channel.
    proto_out = sys.stdout
    sys.stdout = sys.stderr
    # Pre-warm imports once at startup so the first user PDF pays 0s import cost
    try:
        _import_fitz()
        from pdf_etl import run_pipeline  # noqa: F401
    except Exception:
        pass
    proto_out.write('{"ready":true}\n')
    proto_out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                proto_out.write('{"ok":true}\n')
                proto_out.flush()
                break
            res = _handle(req["args"])
            proto_out.write(json.dumps({"ok": True, "res": res}) + "\n")
        except Exception as exc:  # noqa: BLE001
            proto_out.write(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        proto_out.flush()
    return 0


def main(argv):
    _disable_ecoqos()
    if not argv:
        raise SystemExit(__doc__)
    if argv[0] == "worker":
        return _run_worker()
    print(json.dumps(_handle(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
