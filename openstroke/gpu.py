"""Runtime capability detection: one place that decides device, mixed precision and CUDA backend flags for any GPU.

Why (user 2026-11-27): an RTX 2070 SUPER (Turing, sm_75) failed with "sm75 not supported / no bf16". Since torch 2.x
``torch.cuda.is_bf16_supported()`` defaults to ``including_emulation=True`` and answers True on Turing, so the code
autocast to bf16 that Turing only emulates (slow) and some kernels (SDPA, cuBLAS / cuDNN bf16, Triton) refuse.
The first page on the GPU was also slower than on the CPU: ``cudnn.benchmark`` re-tuned every conv for each new
image size (the image branch renders windows at up to 48 sizes), so inference paid autotuning again and again.

Rules (each GPU picks its own path, nothing to configure):

- device: a CUDA device whose architecture this PyTorch build has no kernels for (``torch.cuda.get_arch_list()``)
  or that fails a tiny smoke test falls back to the CPU with a clear warning instead of crashing mid-page;
- precision ``auto``: native bf16 (sm_80+: Ampere, Ada, Hopper, Blackwell) -> bf16; tensor-core fp16 (sm_70 / sm_75:
  Volta, Turing incl. T4 / RTX 20xx) -> fp16 (inference checks every output for inf / NaN and redoes the page in fp32,
  training falls back to fp32 when fp16 keeps overflowing); older -> fp32. ``OPENSTROKE_AMP=bf16|fp16|fp32`` forces one;
- backend flags: TF32 only where it exists (sm_80+); ``cudnn.benchmark`` on for training (shapes repeat across
  epochs), off for inference unless ``OPENSTROKE_CUDNN_BENCHMARK=1``;
- channels_last only where it pays: CPU (oneDNN) and half precision on tensor cores (sm_70+), not fp32 on the GPU;
- torch.compile only on sm_70+ (Triton's minimum).
"""
from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache

AMP_ENV = "OPENSTROKE_AMP"
BENCHMARK_ENV = "OPENSTROKE_CUDNN_BENCHMARK"
_OFF = ("fp32", "float32", "off", "none", "0", "false", "no")

#: messages of kernels / dtypes a GPU or a PyTorch build does not support (retry in fp32, or on the CPU)
_PRECISION_ERRORS = re.compile(
    r"bfloat16|bf16|\bsm_?\d{2,3}\b|not (?:be )?supported|no kernel image|CUBLAS_STATUS_(?:NOT_SUPPORTED|ARCH_MISMATCH)|"
    r"CUDNN_STATUS_(?:NOT_SUPPORTED|ARCH_MISMATCH)|FlashAttention|Ampere|compute capability|cuda capability",
    re.IGNORECASE,
)
_KERNEL_ERRORS = re.compile(r"no kernel image|ARCH_MISMATCH|not compatible with the current PyTorch|cuda capability",
                            re.IGNORECASE)


class NonFiniteOutput(RuntimeError):
    """fp16 inference produced inf / NaN (weights trained in bf16 can overflow fp16): the caller redoes it in fp32."""


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    capability: tuple[int, int]
    memory_gb: float
    kernels: bool                  # this PyTorch build ships SASS or PTX that runs on the device
    arch_list: tuple[str, ...]

    @property
    def sm(self) -> str:
        return f"sm_{self.capability[0]}{self.capability[1]}"

    @property
    def bf16(self) -> bool:        # native bf16 tensor cores (emulated bf16 on Turing is slow and partly unsupported)
        return self.capability >= (8, 0)

    @property
    def fp16(self) -> bool:        # fp16 tensor cores
        return self.capability >= (7, 0)

    @property
    def tf32(self) -> bool:
        return self.capability >= (8, 0)

    @property
    def compile_ok(self) -> bool:  # Triton needs sm_70+
        return self.capability >= (7, 0)


def _arch_num(tag: str) -> tuple[str, tuple[int, int], bool] | None:
    """'sm_86' -> ('sm', (8, 6), False); 'compute_120' -> ('compute', (12, 0), False); 'sm_90a' -> ('sm', (9, 0), True)."""
    m = re.fullmatch(r"(sm|compute)_(\d+)([a-z]?)", tag.strip())
    if not m:
        return None
    n = int(m.group(2))
    return m.group(1), (n // 10, n % 10), bool(m.group(3))


def build_supports(capability: tuple[int, int], arch_list) -> bool:
    """Whether a PyTorch build compiled for ``arch_list`` runs on a GPU of ``capability``.

    SASS (``sm_XY``) runs on the same major version with an equal or higher minor (sm_80 code runs on sm_86);
    arch-specific variants (``sm_90a``) only on that exact GPU; PTX (``compute_XY``) is JIT-compiled for any newer GPU.
    An empty list (unknown build) is trusted."""
    tags = [t for t in (arch_list or []) if t]
    if not tags:
        return True
    cap = tuple(capability)
    for t in tags:
        parsed = _arch_num(t)
        if parsed is None:
            continue
        kind, (maj, mnr), specific = parsed
        if kind == "sm" and maj == cap[0] and (mnr == cap[1] if specific else mnr <= cap[1]):
            return True
        if kind == "compute" and not specific and (maj, mnr) <= cap:
            return True
    return False


def _index(device) -> int:
    import torch

    dev = torch.device(device)
    return dev.index if dev.index is not None else torch.cuda.current_device()


@lru_cache(maxsize=None)
def _gpu_info(index: int) -> GpuInfo:
    import torch

    p = torch.cuda.get_device_properties(index)
    cap = (int(p.major), int(p.minor))
    try:
        arch = tuple(torch.cuda.get_arch_list())
    except Exception:  # noqa: BLE001
        arch = ()
    return GpuInfo(index, str(p.name), cap, round(p.total_memory / 2**30, 1), build_supports(cap, arch), arch)


def gpu_info(device) -> GpuInfo | None:
    """Capabilities of a CUDA device (None for the CPU)."""
    import torch

    if torch.device(device).type != "cuda":
        return None
    return _gpu_info(_index(device))


@lru_cache(maxsize=None)
def _smoke(index: int) -> str | None:
    """Run a matmul and a conv on the device: None when they work, else the error text."""
    import torch

    try:
        dev = torch.device("cuda", index)
        a = torch.randn(64, 64, device=dev)
        (a @ a).sum().item()
        x = torch.randn(1, 3, 32, 32, device=dev)
        torch.nn.functional.conv2d(x, torch.randn(8, 3, 3, 3, device=dev)).sum().item()
        return None
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def parse_device(device=None):
    """'auto' | None | 'cpu' | 0 | '0' | 'cuda' | 'cuda:1' | '0,1' -> torch.device (no capability check)."""
    import torch

    if device in (None, "", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, int) or (isinstance(device, str) and device.strip().isdigit()):
        device = f"cuda:{int(device)}"
    if isinstance(device, str) and "," in device:
        device = f"cuda:{int(device.split(',')[0])}"
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            warnings.warn(f"device={device!r}: CUDA is not available in this PyTorch build, running on the CPU", stacklevel=3)
            return torch.device("cpu")
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def resolve_device(device=None):
    """``parse_device`` + a check that this PyTorch build actually runs on the GPU; falls back to the CPU otherwise."""
    import torch

    dev = parse_device(device)
    if dev.type != "cuda":
        return dev
    info = gpu_info(dev)
    err = None if info.kernels else (f"this PyTorch build has kernels for {', '.join(info.arch_list) or '?'}, "
                                     f"not for {info.sm}")
    err = err or _smoke(info.index)
    if err:
        warnings.warn(f"GPU {info.name} ({info.sm}) cannot run this PyTorch build ({err}); running on the CPU. "
                      f"Install a PyTorch wheel built for {info.sm} (e.g. the cu126 / cu128 builds) to use the GPU.",
                      stacklevel=3)
        return torch.device("cpu")
    return dev


def amp_dtype(device, *, train: bool = False, override: str | None = None):
    """Autocast dtype for ``device`` (None = fp32 / autocast off). See the module docstring for the rules."""
    import torch

    info = gpu_info(device)
    if info is None:
        return None
    mode = str(override if override is not None else os.environ.get(AMP_ENV, "auto")).strip().lower()
    if mode in _OFF:
        return None
    if mode in ("bf16", "bfloat16"):
        if info.bf16:
            return torch.bfloat16
        warnings.warn(f"{AMP_ENV}=bf16: {info.name} ({info.sm}) has no native bf16, using the automatic choice", stacklevel=2)
    elif mode in ("fp16", "float16", "half"):
        return torch.float16
    elif mode not in ("auto", ""):
        raise ValueError(f"{AMP_ENV}={mode!r}: use auto, bf16, fp16 or fp32")
    if info.bf16:
        return torch.bfloat16
    if info.fp16:
        return torch.float16
    return None


def configure(device, *, train: bool) -> None:
    """CUDA backend flags for the device: TF32 where it exists, cuDNN autotuning for training only (see module doc)."""
    import torch

    info = gpu_info(device)
    if info is None:
        return
    torch.backends.cuda.matmul.allow_tf32 = info.tf32
    torch.backends.cudnn.allow_tf32 = info.tf32
    env = os.environ.get(BENCHMARK_ENV)
    torch.backends.cudnn.benchmark = (env.strip().lower() not in _OFF) if env is not None else bool(train)


def channels_last_ok(device, dtype) -> bool:
    """NHWC conv weights pay off on the CPU (oneDNN) and in half precision on tensor cores; fp32 on a GPU stays NCHW."""
    import torch

    info = gpu_info(device)
    if info is None:
        return True
    return info.fp16 and dtype in (torch.float16, torch.bfloat16)


def is_precision_error(exc: BaseException) -> bool:
    """A dtype / kernel the GPU or the build does not support (worth one retry in fp32)."""
    return isinstance(exc, NonFiniteOutput) or bool(_PRECISION_ERRORS.search(str(exc)))


def is_kernel_error(exc: BaseException) -> bool:
    """No kernel of this build runs on the GPU at all (retry on the CPU)."""
    return bool(_KERNEL_ERRORS.search(str(exc)))


def dtype_name(dtype) -> str:
    import torch

    return {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(dtype, "fp32")


def describe(device, dtype=None) -> str:
    """One log line: 'cuda:0 NVIDIA GeForce RTX 2070 SUPER (sm_75, 8.0 GB) | fp16 (no native bf16) | cudnn.benchmark off'."""
    import torch

    info = gpu_info(device)
    if info is None:
        return f"cpu ({torch.get_num_threads()} threads) | fp32"
    why = " (no native bf16)" if dtype == torch.float16 and not info.bf16 else ""
    return (f"cuda:{info.index} {info.name} ({info.sm}, {info.memory_gb} GB) | {dtype_name(dtype)}{why} | "
            f"tf32 {'on' if torch.backends.cuda.matmul.allow_tf32 else 'off'} | "
            f"cudnn.benchmark {'on' if torch.backends.cudnn.benchmark else 'off'}")
