"""openstroke.gpu: device / precision / backend choice per GPU architecture, and the runtime fallbacks (CPU-only checks)."""
from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")

from openstroke import gpu  # noqa: E402
from openstroke.engine import trainer  # noqa: E402

CUDA0 = torch.device("cuda", 0)


def info(cap, name="GPU", arch=("sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120")):
    return gpu.GpuInfo(0, name, cap, 8.0, gpu.build_supports(cap, arch), tuple(arch))


@pytest.fixture
def fake_gpu(monkeypatch):
    def use(cap, **kw):
        g = info(cap, **kw)
        monkeypatch.setattr(gpu, "gpu_info", lambda device: None if torch.device(device).type == "cpu" else g)
        return g
    monkeypatch.delenv(gpu.AMP_ENV, raising=False)
    monkeypatch.delenv(gpu.BENCHMARK_ENV, raising=False)
    return use


def test_build_supports_sass_ptx_and_arch_specific():
    assert gpu.build_supports((7, 5), ["sm_75", "sm_86"])
    assert gpu.build_supports((8, 6), ["sm_80"])                    # sm_80 SASS runs on sm_86
    assert not gpu.build_supports((7, 5), ["sm_80", "sm_90"])       # Turing without sm_75 code: not runnable
    assert gpu.build_supports((7, 5), ["sm_80", "compute_70"])      # PTX JIT for a newer GPU
    assert not gpu.build_supports((9, 1), ["sm_90a"])               # arch-specific only on that exact GPU
    assert gpu.build_supports((12, 0), ["sm_100", "sm_120"])        # 3-digit Blackwell tags
    assert gpu.build_supports((7, 5), [])                           # unknown build: trusted


def test_turing_gets_fp16_not_emulated_bf16(fake_gpu):
    g = fake_gpu((7, 5), name="NVIDIA GeForce RTX 2070 SUPER")
    assert g.kernels and not g.bf16 and g.fp16 and not g.tf32
    assert gpu.amp_dtype(CUDA0) == torch.float16
    assert gpu.amp_dtype(CUDA0, train=True) == torch.float16


@pytest.mark.parametrize("cap", [(8, 0), (8, 6), (8, 9), (9, 0), (12, 0)])
def test_ampere_and_newer_use_bf16(fake_gpu, cap):
    fake_gpu(cap)
    assert gpu.amp_dtype(CUDA0) == torch.bfloat16


def test_pascal_runs_fp32_and_cpu_has_no_autocast(fake_gpu):
    fake_gpu((6, 1), arch=("sm_61",))
    assert gpu.amp_dtype(CUDA0) is None
    assert gpu.amp_dtype(torch.device("cpu")) is None


def test_env_override(fake_gpu, monkeypatch):
    fake_gpu((7, 5))
    monkeypatch.setenv(gpu.AMP_ENV, "fp32")
    assert gpu.amp_dtype(CUDA0) is None
    monkeypatch.setenv(gpu.AMP_ENV, "bf16")                          # not native on Turing -> automatic choice
    with pytest.warns(UserWarning, match="no native bf16"):
        assert gpu.amp_dtype(CUDA0) == torch.float16
    monkeypatch.setenv(gpu.AMP_ENV, "nonsense")
    with pytest.raises(ValueError):
        gpu.amp_dtype(CUDA0)


def test_configure_benchmark_only_for_training(fake_gpu, monkeypatch):
    fake_gpu((7, 5))
    gpu.configure(CUDA0, train=False)
    assert torch.backends.cudnn.benchmark is False                   # inference: no autotuning per image size
    assert torch.backends.cuda.matmul.allow_tf32 is False            # Turing has no TF32
    gpu.configure(CUDA0, train=True)
    assert torch.backends.cudnn.benchmark is True
    monkeypatch.setenv(gpu.BENCHMARK_ENV, "1")
    gpu.configure(CUDA0, train=False)
    assert torch.backends.cudnn.benchmark is True
    torch.backends.cudnn.benchmark = False


def test_channels_last_only_where_it_pays(fake_gpu):
    fake_gpu((7, 5))
    assert gpu.channels_last_ok(torch.device("cpu"), None)
    assert gpu.channels_last_ok(CUDA0, torch.float16)
    assert not gpu.channels_last_ok(CUDA0, None)                     # fp32 on the GPU stays NCHW


def test_resolve_device_falls_back_when_build_has_no_kernels(fake_gpu, monkeypatch):
    fake_gpu((7, 5), arch=("sm_80", "sm_90", "sm_120"))
    monkeypatch.setattr(gpu, "parse_device", lambda device=None: CUDA0)
    with pytest.warns(UserWarning, match="sm_75"):
        assert gpu.resolve_device("cuda").type == "cpu"


def test_error_classification():
    assert gpu.is_precision_error(RuntimeError("expected scalar type BFloat16 but sm_75 is not supported"))
    assert gpu.is_precision_error(gpu.NonFiniteOutput("x"))
    assert gpu.is_kernel_error(RuntimeError("CUDA error: no kernel image is available for execution on the device"))
    assert not gpu.is_precision_error(RuntimeError("shape mismatch: [3, 4] vs [4, 5]"))


def test_predict_page_redoes_fp16_overflow_in_fp32(monkeypatch):
    calls = []

    def fake(model, page, window_size, device, batch, tta, amp_dtype, **outs):
        calls.append(amp_dtype)
        if amp_dtype == torch.float16:
            raise gpu.NonFiniteOutput("inf")
        return "fp32-result"

    monkeypatch.setattr(trainer, "_predict_page", fake)
    model = torch.nn.Linear(1, 1)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert trainer.predict_page(model, None, 1, torch.device("cpu"), amp_dtype=torch.float16) == "fp32-result"
    assert calls == [torch.float16, None] and model._fp16_unsafe
    assert trainer.predict_page(model, None, 1, torch.device("cpu"), amp_dtype=torch.float16) == "fp32-result"
    assert calls[-1] is None                                         # the model stays in fp32 afterwards


def test_check_finite_raises_on_nan():
    model = torch.nn.Linear(1, 1)
    trainer._check_finite(model, torch.zeros(3))
    with pytest.raises(gpu.NonFiniteOutput):
        trainer._check_finite(model, torch.tensor([0.0, float("nan")]))


def _guarded_predictor(monkeypatch, amp):
    from openstroke.infer import OpenStroke

    p = OpenStroke.__new__(OpenStroke)
    p.model, p.device, p.amp_dtype, p.batch, p._batch_arg, p.runtime = torch.nn.Linear(1, 1), CUDA0, amp, 4, None, ""
    monkeypatch.setattr(gpu, "describe", lambda device, dtype=None: f"{device} {dtype}")
    return p


def test_run_guarded_unsupported_bf16_retries_in_fp32(monkeypatch):
    p = _guarded_predictor(monkeypatch, torch.bfloat16)
    seen = []

    def fn(batch):
        seen.append(p.amp_dtype)
        if p.amp_dtype is not None:
            raise RuntimeError("FlashAttention only supports Ampere GPUs or newer (bf16 on sm_75)")
        return "ok"

    with pytest.warns(UserWarning, match="fp32"):
        assert p._run_guarded(fn, 4) == "ok"
    assert seen == [torch.bfloat16, None] and p.amp_dtype is None


def test_run_guarded_missing_kernels_moves_to_cpu(monkeypatch):
    p = _guarded_predictor(monkeypatch, None)

    def fn(batch):
        if p.device.type == "cuda":
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
        return batch

    with pytest.warns(UserWarning, match="CPU"):
        assert p._run_guarded(fn, 4) == 1
    assert p.device.type == "cpu"
