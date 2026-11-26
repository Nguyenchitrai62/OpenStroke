"""Inference and deployment interface (OpenOBB-style): vector payload -> raw per-shape predictions.

    from openstroke import OpenStroke
    model = OpenStroke("best.pt")                         # auto GPU (bf16 + fused Conv/BN + channels_last) or CPU
    results = model.predict("page-0001.v0.json.gz")       # or model(payload_dict), folder, glob, or list of sources
    for r in results:
        r.prob, r.valid, r.edges, r.groups, r.speed       # strictly RAW outputs; post-processing lives in a separate module

No ``mode`` (discipline) is required: modern OpenStroke checkpoints are purely vector-driven (``mode_input=False``),
so any model trained on HVAC / FIRE / PLUMBING / custom vector datasets runs through the exact same interface.
Legacy ``Predictor`` is retained for full backward compatibility (``mode`` is now optional).
"""
from __future__ import annotations

import ctypes
import glob as _glob
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .results import Results, Strokes
from .sources import load_payload  # noqa: F401  (re-exported for callers)

_CPU_RUNTIME_CONFIGURED = False


def _configure_cpu_runtime() -> None:
    """Eliminate Windows 11 EcoQoS E-core throttling & OpenMP spin-wait stalls on CPU (exact parity, 3.8x-5.5x on multi-window)."""
    global _CPU_RUNTIME_CONFIGURED
    import torch

    torch.set_flush_denormal(True)
    if _CPU_RUNTIME_CONFIGURED:
        return
    _CPU_RUNTIME_CONFIGURED = True
    if sys.platform == "win32":
        try:
            from ctypes import wintypes

            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            k32.SetProcessInformation.restype = wintypes.BOOL

            class _PPTS(ctypes.Structure):
                _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

            # ProcessPowerThrottling (4): PROCESS_POWER_THROTTLING_EXECUTION_SPEED (1), State=0 (opt out of EcoQoS)
            proc = k32.GetCurrentProcess()
            st = _PPTS(1, 1, 0)
            k32.SetProcessInformation(proc, 4, ctypes.byref(st), ctypes.sizeof(st))
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.SetPriorityClass.restype = wintypes.BOOL
            k32.SetPriorityClass(proc, 0x00000020)  # NORMAL_PRIORITY_CLASS
        except Exception:
            pass
        try:
            from ctypes import wintypes

            k32 = ctypes.windll.kernel32
            k32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
            k32.GetModuleHandleW.restype = wintypes.HMODULE
            if k32.GetModuleHandleW("libiomp5md.dll"):
                iomp = ctypes.CDLL("libiomp5md.dll")
                iomp.kmp_set_blocktime.argtypes = [ctypes.c_int]
                iomp.kmp_set_blocktime(0)
        except Exception:
            pass

#: window used when a checkpoint does not record its training arguments (first v1 trainer)
DEFAULT_WINDOW = {"openstroke1": 4096, "openstroke2": 65536, "openstroke3": 65536, "openstroke4": 65536}
GPU_BATCH = 4
PAYLOAD_EXT = (".json", ".gz")
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")


def _device(device=None):
    """'auto' | None | 'cpu' | 0 | '0' | 'cuda' | 'cuda:1' -> torch.device (matches OpenOBB)."""
    import torch

    if device in (None, "", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, int) or (isinstance(device, str) and device.strip().isdigit()):
        device = f"cuda:{int(device)}"
    if isinstance(device, str) and "," in device:
        device = f"cuda:{int(device.split(',')[0])}"
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def compile_mode(compile=None) -> str | None:
    """``compile=`` of OpenStroke / predict (as OpenOBB): False (off), True ('default') or a torch.compile mode name.
    When None, reads ``OPENSTROKE_COMPILE`` (default off)."""
    if compile is None:
        compile = os.environ.get("OPENSTROKE_COMPILE", "0")
    if isinstance(compile, str):
        v = compile.strip().lower()
        if v in ("", "0", "false", "no", "off", "none"):
            return None
        if v in ("1", "true", "yes", "on"):
            return "default"
        if v not in COMPILE_MODES:
            raise ValueError(f"compile={compile!r}: use True / False or one of {', '.join(COMPILE_MODES)}")
        return v
    return "default" if compile else None


def validate_payload(payload: Any) -> dict:
    """Validate an incoming page payload dict (AI_Takeoff ``split_countfire`` format)."""
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be a dict with a 'shapes' list, got {type(payload).__name__}")
    shapes = payload.get("shapes")
    if shapes is None:
        raise ValueError("payload is missing required key 'shapes'")
    if not isinstance(shapes, (list, tuple)):
        raise TypeError(f"payload['shapes'] must be a list of shape dicts, got {type(shapes).__name__}")
    return payload


def _is_payload_file(p: Path) -> bool:
    name = p.name.lower()
    if ".view." in name:
        return False
    return name.endswith(".json") or name.endswith(".json.gz")


def _payload_files(s: str) -> list[Path]:
    p = Path(s)
    if any(ch in s for ch in "*?["):
        files = sorted(Path(f) for f in _glob.glob(s, recursive=True) if Path(f).is_file() and _is_payload_file(Path(f)))
    elif p.is_dir():
        files = sorted(f for f in p.iterdir() if f.is_file() and _is_payload_file(f))
    elif not p.exists():
        raise FileNotFoundError(f"{s} does not exist")
    elif not _is_payload_file(p):
        raise ValueError(f"unsupported payload source {s!r} (expected .json or .json.gz)")
    else:
        files = [p]
    return files


def list_sources(source: Any) -> list[tuple[str, Any]]:
    """Normalize ``source`` -> list of ``(name, dict | loader)`` (lazy file loading, as OpenOBB).

    Accepts: a payload dict, a ``.json`` / ``.json.gz`` path, a directory of payload files, a glob pattern,
    or a list/tuple mixing any of the above.
    """
    if source is None:
        raise ValueError("source is required (payload dict, .json/.json.gz path, folder, glob, or list)")
    if isinstance(source, dict):
        return [("payload0.json", validate_payload(source))]
    if isinstance(source, (list, tuple)):
        out: list[tuple[str, Any]] = []
        for i, item in enumerate(source):
            for name, src in list_sources(item):
                out.append((f"payload{i}.json" if name == "payload0.json" else name, src))
        return out
    if isinstance(source, (str, Path)):
        files = _payload_files(str(source))
        if not files:
            raise FileNotFoundError(f"no .json / .json.gz payload files found in {source}")
        return [(str(f), lambda f=f: validate_payload(load_payload(f))) for f in files]
    raise TypeError(f"unsupported source type {type(source).__name__}: expected dict, path, folder, glob, or list")


def payload_arrays(payload: dict, mode: str | None = None, stem: str = "page") -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build the E0 npz arrays of a payload page in memory (every label ``other``, no GT); returns (arrays, page info)."""
    from .build import COORDINATE_FRAME, SCHEMA_VERSION, shape_attributes
    from .geometry import page_segments
    from .labels import LABEL_SCHEMA

    validate_payload(payload)
    shapes = payload.get("shapes") or []
    meta = payload.get("document_metadata") or {}
    n = len(shapes)
    segs = page_segments(shapes)
    length = segs.shape_length().astype(np.float32)
    attrs = shape_attributes(shapes, segs)
    meta_dict = {
        "stem": stem,
        "mode": mode or "raw",
        "schema_version": SCHEMA_VERSION,
        "label_schema": LABEL_SCHEMA,
        "coordinate_frame": COORDINATE_FRAME,
        "source": "payload (inference, no GT)",
    }
    arrays = {
        "meta": np.array(json.dumps(meta_dict)),
        "label": np.zeros(n, np.uint8),
        "true_frac": np.zeros(n, np.float32),
        "length": length,
        "segments": segs.seg.astype(np.float32),
        "shape_offsets": segs.shape_offsets,
        "seg_inside": np.zeros(len(segs.seg), np.float32),
        "qa_suspicious_other": np.zeros(n, np.uint8),
        "gt_class": np.array([], dtype=str),
        "gt_coverage": np.zeros(0, np.float32),
        **attrs,
    }
    info = {
        "shapes": n,
        "width": float(meta.get("page_width") or 0),
        "height": float(meta.get("page_height") or 0),
        "segments": segs,
        "length": length,
        "attrs": attrs,
    }
    return arrays, info


def payload_npz(payload: dict, path: Path, mode: str | None = None, stem: str = "page", compress: bool = False) -> dict[str, Any]:
    """Write the E0 npz fields of a payload page with every label ``other`` (no GT); returns page info."""
    arrays, info = payload_arrays(payload, mode=mode, stem=stem)
    (np.savez_compressed if compress else np.savez)(path, **arrays)
    return info


def _compile_submodules(model, device, mode: str | None) -> bool:
    """Optionally compile the heavy CNN trunk of ImageBranch on GPU (exact parity, graceful fallback as OpenOBB)."""
    import torch

    if mode is None or not hasattr(torch, "compile"):
        return False
    if torch.device(device).type == "cpu":
        warnings.warn("compile: CPU inductor compilation is skipped; running eager", stacklevel=3)
        return False
    img = getattr(model, "img", None)
    if img is None:
        return False
    try:
        if getattr(img, "stem", None) is not None:
            img.stem = torch.compile(img.stem, mode=mode, dynamic=True)
            img.layer2 = torch.compile(img.layer2, mode=mode, dynamic=True)
            img.layer3 = torch.compile(img.layer3, mode=mode, dynamic=True)
            if getattr(img, "layer4", None) is not None:
                img.layer4 = torch.compile(img.layer4, mode=mode, dynamic=True)
        if getattr(img, "cad_stem", None) is not None:
            img.cad_stem = torch.compile(img.cad_stem, mode=mode, dynamic=True)
        return True
    except Exception as exc:
        warnings.warn(f"compile={mode!r} failed ({type(exc).__name__}: {exc}); running eager", stacklevel=3)
        return False


def _load_checkpoint_into_model(ck_path: Path, dev, *, fuse: bool = True):
    """Load any OpenStroke training or stripped deploy checkpoint -> (model, meta dict)."""
    import torch

    from .models import canonical_arch, from_config, load_weights
    from .models.image_branch import fuse_image_branch

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    if "config" not in ck:
        kind = f"OpenOBB {ck.get('version', '')}" if "classes" in ck else "không phải model nét OpenStroke"
        raise ValueError(f"{ck_path.name}: checkpoint {kind}. Chỉ nạp được checkpoint model nét OpenStroke (.pt).")

    config = dict(ck["config"])
    arch = canonical_arch(config.get("arch"))
    model = from_config(config, weights_follow=True)
    raw_sd = (ck.get("ema") or {}).get("module") or ck["model"]
    # Stripped half=True deploy checkpoints store float16 weights -> promote cleanly back to float32 before loading
    sd = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float16 else v) for k, v in raw_sd.items()}
    load_weights(model, sd)

    if config.get("no_layers"):
        model.config["no_layers"] = True

    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        _configure_cpu_runtime()

    model.to(dev).eval()
    if fuse:
        fuse_image_branch(model, channels_last=True)

    args = ck.get("args") or {}
    default_mode = ck.get("default_mode")
    if default_mode is None:
        m_arg = str(args.get("modes") or "").strip().lower()
        if m_arg in ("fire", "hvac", "plumbing"):
            default_mode = m_arg

    meta = {
        "config": config,
        "arch": arch,
        "window": int(ck.get("window") or args.get("window") or DEFAULT_WINDOW.get(arch, 65536)),
        "threshold": float(ck.get("threshold", 0.5)),
        "tta": int(ck.get("tta", 1)),
        "val_f2": ck.get("val_f2"),
        "epoch": ck.get("epoch"),
        "mode_input": bool(config.get("mode_input", False)),
        "default_mode": default_mode,
        "feature_names": ck.get("feature_names"),
        "deploy": bool(ck.get("deploy", False)),
    }
    return model, meta


class OpenStroke:
    """OpenOBB-style deployment & training entry point for OpenStroke vector models.

    Deployment (raw inference only — post-processing is kept in a separate module, no ``mode`` needed):

        from openstroke import OpenStroke
        model = OpenStroke("best.pt")
        for r in model.predict("page-0001.v0.json.gz"):      # or model(payload_dict)
            print(r.prob, r.valid, r.edges, r.groups, r.speed)

    Training from a named architecture or checkpoint:

        OpenStroke("openstroke45").train(data="datasets/e13_v4", epochs=15)
    """

    task = "stroke"

    def __init__(
        self,
        model: str | Path = "best.pt",
        *,
        device: str | int | None = None,
        batch: int | None = None,
        threads: int | None = None,
        workers: int = 1,
        compile: bool | str | None = None,
        fuse: bool = True,
    ):
        self.ckpt_path = str(model)
        self.path = Path(model)
        self._threads = threads
        self._workers = int(workers)
        self._batch_arg = batch
        self._fuse = bool(fuse)
        self.compile = compile_mode(compile)
        self.model = None
        self.device = None
        self.config: dict[str, Any] = {}
        self.arch: str = ""
        self.window: int = 65536
        self.threshold: float = 0.5
        self.tta: int = 1
        self.val_f2 = None
        self.epoch = None
        self.mode_input: bool = False
        self.default_mode: str | None = None
        self.feature_names = None
        self.deploy: bool = False
        self.batch: int = 1
        self.pool = None
        self.pool_error = None

        if self.ckpt_path.lower().endswith(".pt") or self.path.is_file():
            if not self.path.exists():
                raise FileNotFoundError(self.ckpt_path)
            self._load(device=device, compile=self.compile)

    def _load(self, device=None, compile=None) -> None:
        import torch

        if self._threads:
            torch.set_num_threads(int(self._threads))
        dev = _device(device)
        model, meta = _load_checkpoint_into_model(self.path, dev, fuse=self._fuse)
        self.model = model
        self.device = dev
        self.config = meta["config"]
        self.arch = meta["arch"]
        self.window = meta["window"]
        self.threshold = meta["threshold"]
        self.tta = meta["tta"]
        self.val_f2 = meta["val_f2"]
        self.epoch = meta["epoch"]
        self.mode_input = meta["mode_input"]
        self.default_mode = meta["default_mode"]
        self.feature_names = meta["feature_names"]
        self.deploy = meta["deploy"]
        self.batch = int(self._batch_arg) if self._batch_arg is not None else (1 if dev.type == "cpu" else GPU_BATCH)

        if self.pool is not None:
            self.pool.close()
            self.pool = None
        needs_inproc = (
            getattr(self.model, "obj_mlp", None) is not None
            or getattr(self.model, "edge_mlp", None) is not None
            or getattr(self.model, "group_head", None) is not None
            or int(getattr(self.model, "stroke_classes", 1)) > 1
        )
        if dev.type == "cpu" and self._workers > 1 and not needs_inproc:
            from .parallel import WindowPool

            self.pool = WindowPool(self.path, self._workers, self._threads or torch.get_num_threads())

        mode = compile_mode(compile)
        self.compile = mode if _compile_submodules(self.model, dev, mode) else None

    def _resolve_mode(self, mode: str | None = None) -> str:
        if not self.mode_input:
            return mode or "raw"
        if mode:
            return str(mode)
        if self.default_mode:
            return self.default_mode
        return "hvac"

    def to(self, device) -> "OpenStroke":
        self._load(device=device, compile=self.compile)
        return self

    def eval(self) -> "OpenStroke":
        if self.model is not None:
            self.model.eval()
        return self

    def fuse(self) -> "OpenStroke":
        if self.model is not None:
            from .models.image_branch import fuse_image_branch

            fuse_image_branch(self.model, channels_last=True)
        return self

    def describe(self) -> dict[str, Any]:
        from .train import describe_arch

        return {
            "checkpoint": str(self.path),
            "arch": describe_arch(self.config) if self.config else self.ckpt_path,
            "device": str(self.device) if self.device is not None else None,
            "window": self.window,
            "threshold": self.threshold,
            "stroke_classes": int(self.config.get("stroke_classes", 1) or 1),
            "class_names": list(self.config["class_names"]) if self.config.get("class_names") else None,
            "tta_used_in_val": self.tta,
            "val_f2": self.val_f2,
            "epoch": None if self.epoch is None else int(self.epoch) + 1,
            "mode_input": self.mode_input,
            "deploy": self.deploy,
            "compiled": bool(self.compile),
        }

    def info(self, verbose: bool = True) -> str:
        n = sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0
        d = self.describe()
        line = (
            f"OpenStroke ({d['arch']}): {n / 1e6:.2f}M params, device={d['device']}, "
            f"window={self.window}, threshold={self.threshold:.2f}, deploy={self.deploy}"
        )
        if verbose:
            print(line)
        return line

    def warmup(self, n_shapes: int = 256) -> float:
        """Warm up CUDA kernels / thread pools on a tiny synthetic payload (recommended once on server startup)."""
        rng = np.random.default_rng(0)
        xy = rng.uniform(0.0, 500.0, (int(n_shapes), 4))
        synth = {
            "shapes": [
                {"items": [["l", [float(a), float(b)], [float(c), float(d)]]], "type": "s",
                 "color": [0.0, 0.0, 0.0], "width": 0.35, "layer": "0"}
                for a, b, c, d in xy
            ],
            "document_metadata": {"page_width": 500.0, "page_height": 500.0},
        }
        t0 = time.perf_counter()
        self.predict(synth, tta=1, verbose=False)
        return round((time.perf_counter() - t0) * 1e3, 1)

    def strip(self, out: str | Path | None = None, half: bool = False) -> str:
        """Export a stripped deployment checkpoint (no training paths/args; optional fp16 weights)."""
        from .strip import strip as _strip

        return _strip(self.path, out=out, half=half)

    def __call__(self, source: Any, **kw):
        return self.predict(source, **kw)

    def _predict_one(
        self,
        payload: dict,
        path_name: str,
        *,
        mode: str | None = None,
        tta: int = 1,
        batch: int | None = None,
        include_groups: bool = True,
        include_edges: bool = True,
        include_info: bool = False,
        group_strokes: bool = False,
        on_stage=None,
        save_dir: Path | None = None,
    ) -> Results:
        import torch

        from .data import page_from_arrays
        from .engine.trainer import predict_page
        from .features import FEATURE_NAMES
        from .train import describe_arch

        if self.feature_names is not None and tuple(self.feature_names) != tuple(FEATURE_NAMES):
            raise ValueError("checkpoint was trained on another feature set than this code version")
        if group_strokes and getattr(self.model, "edge_mlp", None) is None:
            raise ValueError("group_strokes needs a checkpoint with a same-object edge head")

        eff_mode = self._resolve_mode(mode)
        eff_batch = max(int(batch), 1) if batch is not None else self.batch
        eff_tta = max(int(tta), 1)

        t0 = time.perf_counter()
        if on_stage:
            on_stage("features", 0, 0)
        arrays, info = payload_arrays(payload, mode=eff_mode)
        graph: bool | str = (
            bool(self.config.get("relations", 0))
            or "graph" in (self.config.get("mixers") or [])
            or bool(self.config.get("rel_attn", 0))
            or bool(self.config.get("edge_head"))
            or bool(self.config.get("group_head"))
        )
        if self.config.get("continuation"):
            graph = "continuation"
        elif self.config.get("facing_edges"):
            graph = "facing"
        page = page_from_arrays(arrays, "page", mode=eff_mode, graph=graph, no_layers=bool(self.config.get("no_layers")))
        t1 = time.perf_counter()

        n_shapes = info["shapes"]
        prob = np.zeros(n_shapes, np.float32)
        valid = np.zeros(n_shapes, bool)
        if len(page):
            valid[page.index] = True

        if on_stage:
            on_stage("model", 0, 0)

        pc_out = e_out = g_out = c_out = None
        n_scls = int(getattr(self.model, "stroke_classes", 1))
        if len(page):
            on_batch = (lambda done, total: on_stage("model", done, total)) if on_stage else None
            has_cls = getattr(self.model, "obj_mlp", None) is not None
            has_edge = (getattr(self.model, "edge_mlp", None) is not None) and (include_edges or include_groups or group_strokes)
            has_group = (getattr(self.model, "group_head", None) is not None) and include_groups
            has_scls = n_scls > 1
            p = None
            if self.pool is not None and not (has_cls or has_edge or has_group or has_scls):
                with tempfile.TemporaryDirectory() as td:
                    npz = Path(td) / "page.npz"
                    np.savez(npz, **arrays)
                    p = self.pool.predict(page, npz, eff_mode, graph, self.window, eff_tta, on_batch)
                if p is None and self.pool:
                    self.pool_error = self.pool.error
                    self.pool.close()
                    self.pool = None
            if p is None:
                amp = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else None
                pc_out = [] if has_cls else None
                e_out = [] if has_edge else None
                g_out = [] if has_group else None
                c_out = [] if has_scls else None
                try:
                    p = predict_page(
                        self.model, page, self.window, self.device, batch=eff_batch, tta=eff_tta, amp_dtype=amp,
                        on_batch=on_batch, obj_out=pc_out, edge_out=e_out, group_out=g_out, cls_out=c_out,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    self.batch = 1
                    pc_out = [] if has_cls else None
                    e_out = [] if has_edge else None
                    g_out = [] if has_group else None
                    c_out = [] if has_scls else None
                    p = predict_page(
                        self.model, page, self.window, self.device, batch=1, tta=eff_tta, amp_dtype=amp,
                        on_batch=on_batch, obj_out=pc_out, edge_out=e_out, group_out=g_out, cls_out=c_out,
                    )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            prob[page.index] = p
        t2 = time.perf_counter()

        cls_prob = None
        class_names = None
        if n_scls > 1:
            from .metrics import resolve_class_names

            class_names = resolve_class_names(n_scls, self.config.get("class_names"))
            cls_prob = np.zeros((n_shapes, n_scls), np.float32)
            cls_prob[:, 0] = 1.0
            if c_out and len(page):
                cls_prob[page.index] = c_out[0]

        oprob = None
        if pc_out:
            from .objects import N_CLASSES

            oprob = np.zeros((n_shapes, N_CLASSES), np.float32)
            oprob[page.index] = pc_out[0][:, :N_CLASSES]

        edges = None
        if e_out and (include_edges or group_strokes):
            ep_idx = page.index[e_out[0][0]] if len(e_out[0][0]) else np.zeros((0, 2), np.int64)
            ep_pr = (1.0 / (1.0 + np.exp(-np.clip(e_out[0][1].astype(np.float64), -20.0, 20.0)))).astype(np.float32)
            edges = (ep_idx, ep_pr)

        groups = None
        extra: dict[str, Any] = {}
        if g_out and len(page):
            from .models.group_head import OBJ8_CLASSES, supplement_fg_edges
            from .stroke_groups import stroke_instances

            p_page = prob[page.index]
            e_pairs, e_logits = e_out[0] if e_out else (np.zeros((0, 2), np.int64), np.zeros(0, np.float32))
            e_pairs, e_logits = supplement_fg_edges(
                self.model, g_out[0][0], page, p_page, self.threshold, e_pairs, e_logits,
                info.get("segments"), n_shapes, self.device,
            )
            if include_edges and len(e_pairs):
                edges = (
                    page.index[e_pairs],
                    (1.0 / (1.0 + np.exp(-np.clip(e_logits.astype(np.float64), -20.0, 20.0)))).astype(np.float32),
                )
            inst_p = (
                stroke_instances(p_page, self.threshold, e_pairs, e_logits)
                if len(e_pairs)
                else np.where(p_page >= self.threshold, np.arange(len(p_page), dtype=np.int64), -1)
            )
            cids, cprobs = self.model.group_head.classify_page_clusters(
                g_out[0][0], page, p_page, self.threshold, (e_pairs, e_logits) if len(e_pairs) else None, inst_p, self.device
            )
            full_inst = np.full(n_shapes, -1, np.int64)
            full_inst[page.index] = inst_p
            scls = np.zeros((n_shapes, g_out[0][1].shape[1]), np.float32)
            scls[page.index] = g_out[0][1]
            groups = {
                "instance_id": full_inst,
                "cluster_ids": cids,
                "cluster_probs": cprobs,
                "stroke_cls": scls,
                "names": OBJ8_CLASSES,
            }

        if self.config.get("region_head"):
            from .piece_candidates import predict_regions

            amp = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else None
            try:
                extra = predict_regions(self.model, page, info, prob, self.window, self.device, eff_batch, amp)
            except torch.cuda.OutOfMemoryError:
                if eff_batch == 1:
                    raise
                self.batch = 1
                torch.cuda.empty_cache()
                extra = predict_regions(self.model, page, info, prob, self.window, self.device, 1, amp)

        if group_strokes:
            from .stroke_groups import stroke_instances

            pairs, logits = (edges[0], e_out[0][1]) if e_out else (np.empty((0, 2), np.int64), np.empty(0))
            extra["instance_id"] = stroke_instances(prob, self.threshold, pairs, logits)
        t3 = time.perf_counter()

        speed = {
            "preprocess": (t1 - t0) * 1e3,
            "inference": (t2 - t1) * 1e3,
            "postprocess": (t3 - t2) * 1e3,
        }
        return Results(
            Strokes(prob, valid, edges=edges if include_edges else None, groups=groups, oprob=oprob,
                    cls_prob=cls_prob, class_names=class_names),
            path=path_name,
            threshold=self.threshold,
            arch=describe_arch(self.config),
            speed=speed,
            info=info if include_info else None,
            save_dir=str(save_dir) if save_dir is not None else None,
            extra=extra,
        )

    def predict(
        self,
        source: Any,
        *,
        tta: int = 1,
        batch: int | None = None,
        device: str | int | None = None,
        stream: bool = False,
        include_groups: bool = True,
        include_edges: bool = True,
        include_info: bool = False,
        save: bool = False,
        save_json: bool = False,
        project: str | Path | None = None,
        name: str | None = None,
        exist_ok: bool = False,
        verbose: bool = False,
        compile: bool | str | None = None,
        on_stage=None,
        mode: str | None = None,
    ):
        """Run raw inference on one or more vector page payloads -> ``list[Results]`` (or generator if ``stream=True``).

        No ``mode`` is required: ``mode`` is only accepted for backward compatibility with legacy ``mode_input=True``
        checkpoints. Post-processing is intentionally NOT performed here.
        """
        if compile is not None:
            new_c = compile_mode(compile)
            if self.model is None:
                self.compile = new_c
            elif new_c != self.compile:
                self._load(device=self.device if device in (None, "") else device, compile=new_c)
        if self.model is None:
            self._load(device=device, compile=self.compile)
        elif device not in (None, "", "auto") and _device(device) != self.device:
            self._load(device=device, compile=self.compile)

        save_dir = None
        if save or save_json:
            from .train import next_run_dir

            base = Path(project or "runs/predict") / (name or "predict")
            save_dir = base if exist_ok else next_run_dir(base)
            save_dir.mkdir(parents=True, exist_ok=True)

        srcs = list_sources(source)

        def _gen():
            t_sum = np.zeros(3, dtype=np.float64)
            n_done = 0
            for k, (path_name, item) in enumerate(srcs):
                pay = item() if callable(item) else item
                r = self._predict_one(
                    pay,
                    path_name,
                    mode=mode,
                    tta=tta,
                    batch=batch,
                    include_groups=include_groups,
                    include_edges=include_edges,
                    include_info=include_info,
                    on_stage=on_stage,
                    save_dir=save_dir,
                )
                t_sum += [r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]]
                n_done += 1
                if verbose:
                    print(
                        f"page {k + 1}/{len(srcs)} {path_name}: {r.verbose()}"
                        f"pre {r.speed['preprocess']:.1f}ms, infer {r.speed['inference']:.1f}ms, raw-head {r.speed['postprocess']:.1f}ms"
                    )
                if save_dir is not None:
                    stem = Path(path_name).name
                    if stem.endswith(".json.gz"):
                        stem = stem[:-8]
                    elif stem.endswith(".json"):
                        stem = stem[:-5]
                    if save:
                        r.save_npz(save_dir / f"{stem}.npz")
                    if save_json:
                        r.save_json(save_dir / f"{stem}.json")
                yield r
            if verbose and n_done:
                avg = t_sum / n_done
                print(
                    f"Speed: {avg[0]:.1f}ms preprocess, {avg[1]:.1f}ms inference, {avg[2]:.1f}ms raw-head per page"
                )
            if verbose and save_dir is not None:
                print(f"Results saved to {save_dir}")

        return _gen() if stream else list(_gen())

    def train(self, data: str | Path, **kw) -> int:
        """Train or fine-tune using the same CLI options as ``openstroke train`` (as ``OpenOBB(...).train(...)``)."""
        from .cli import main as cli_main

        argv = ["train", f"model={self.ckpt_path}", f"data={data}"]
        for k, v in kw.items():
            if v is not None:
                argv.append(f"{k}={str(v).lower() if isinstance(v, bool) else v}")
        return cli_main(argv)


class Predictor(OpenStroke):
    """Backward-compatible wrapper around ``OpenStroke`` (``predict_payload`` returns legacy dict; ``mode`` optional)."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
        threads: int | None = None,
        workers: int = 1,
        batch: int | None = None,
        compile: bool | str | None = None,
    ):
        super().__init__(checkpoint, device=device, batch=batch, threads=threads, workers=workers, compile=compile)

    def predict_payload(
        self,
        payload: dict,
        mode: str | None = None,
        tta: int = 1,
        on_stage=None,
        *,
        group_strokes: bool = False,
        include_groups: bool = True,
        include_edges: bool = True,
    ) -> dict[str, Any]:
        """Legacy dict output: ``{"prob", "valid", "oprob", "edges", "group_objects", "info", "kept", "seconds"}``.

        ``mode`` is optional and ignored by all modern ``no_layers`` / ``mode_input=False`` checkpoints.
        """
        r = self._predict_one(
            validate_payload(payload),
            "payload0.json",
            mode=mode,
            tta=tta,
            include_groups=include_groups,
            include_edges=include_edges,
            include_info=True,
            group_strokes=group_strokes,
            on_stage=on_stage,
        )
        return r.to_legacy_dict()
