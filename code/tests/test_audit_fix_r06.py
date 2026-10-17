"""Regression tests for audit work unit r06 (W2-122, W2-129, W2-137 residuals)."""

from __future__ import annotations

import contextlib
import importlib
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

import labs.top_k_kernel.top_k_kernel_common as topk_common
import labs.top_k_kernel.top_k_kernel_extension as topk_extension
from labs.kv_optimization.optimized_kv_standard import OptimizedKVFP8Compressed
from labs.kv_optimization.verification import FP8_KV_OUTPUT_TOLERANCE
from labs.top_k_kernel.top_k_kernel_common import TopKKernelBenchmark, build_verification_tensor

_AUTOTUNE_MODULE = "labs.custom_vs_cublas.autotune"
_TCGEN05_KERNEL_NAMES = (
    "matmul_tcgen05",
    "matmul_tcgen05_pipelined",
    "matmul_tcgen05_3stage",
    "matmul_tcgen05_swizzled",
    "matmul_tcgen05_cluster",
    "matmul_tcgen05_warp_spec",
)


@pytest.fixture
def autotune_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    # autotune.py imports the JIT-built tcgen05 extension at module scope. The
    # caching logic under test never calls those kernels, so stub the loader.
    loader = types.ModuleType("tcgen05_loader")
    for name in _TCGEN05_KERNEL_NAMES:
        setattr(loader, name, lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "tcgen05_loader", loader)
    previous = sys.modules.pop(_AUTOTUNE_MODULE, None)
    try:
        yield importlib.import_module(_AUTOTUNE_MODULE)
    finally:
        sys.modules.pop(_AUTOTUNE_MODULE, None)
        if previous is not None:
            sys.modules[_AUTOTUNE_MODULE] = previous


_SWEEP_TIMINGS_MS = {"basic": 3.0, "swizzled": 2.0, "cluster": 1.0}
_SWEEP_KEY = "B200_10.0_256x512x128"


class _FakeSweep:
    """Replaces only the device, allocation, and timing seams of autotune()."""

    def __init__(self, autotune: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        self.cache_file = tmp_path / "autotune_cache.json"
        self.failing: set[str] = set()
        self.benchmarked: list[str] = []
        monkeypatch.setattr(autotune, "_CACHE_FILE", str(self.cache_file))
        monkeypatch.setattr(
            autotune, "_resolve_cuda_device", lambda _device: torch.device("cuda", 0)
        )
        monkeypatch.setattr(autotune, "_get_device_key", lambda _device: "B200_10.0")
        monkeypatch.setattr(autotune.torch, "randn", lambda *_args, **_kwargs: object())
        monkeypatch.setattr(
            autotune.torch.cuda, "device", lambda _device: contextlib.nullcontext()
        )
        monkeypatch.setattr(autotune.torch.cuda, "empty_cache", lambda: None)
        kernels = {name: object() for name in _SWEEP_TIMINGS_MS}
        monkeypatch.setattr(autotune, "KERNELS", kernels)

        def benchmark_kernel(fn, _a, _b) -> float:
            name = next(kernel_name for kernel_name, kernel in kernels.items() if kernel is fn)
            self.benchmarked.append(name)
            if name in self.failing:
                raise RuntimeError("transient launch failure")
            return _SWEEP_TIMINGS_MS[name]

        monkeypatch.setattr(autotune, "_benchmark_kernel", benchmark_kernel)

    def read_cache(self) -> dict:
        return json.loads(self.cache_file.read_text(encoding="utf-8"))

    def write_cache(self, payload: object) -> None:
        self.cache_file.write_text(json.dumps(payload), encoding="utf-8")


def _complete_sweep_cache(winner: str = "cluster", timings: dict | None = None) -> dict:
    return {
        "format_version": 2,
        "entries": {
            _SWEEP_KEY: {
                "winner": winner,
                "timings_ms": dict(_SWEEP_TIMINGS_MS if timings is None else timings),
            }
        },
    }


def test_w2_122_partial_sweep_winner_is_not_written_to_disk_cache(
    autotune_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    autotune = autotune_module
    sweep = _FakeSweep(autotune, monkeypatch, tmp_path)
    cache_file = sweep.cache_file
    failing = sweep.failing
    benchmarked = sweep.benchmarked
    failing.add("cluster")

    # The fastest kernel fails once. Its loss must not become the stored winner.
    with pytest.warns(
        RuntimeWarning,
        match=r"1 of 3 kernels failed \(cluster=RuntimeError: transient launch failure\)",
    ):
        assert autotune.autotune(256, 512, 128, verbose=False) == "swizzled"
    assert benchmarked == ["basic", "swizzled", "cluster"]
    assert not cache_file.exists()

    # The same process keeps the partial-sweep choice instead of re-tuning per call.
    benchmarked.clear()
    assert autotune.autotune(256, 512, 128, verbose=False) == "swizzled"
    assert benchmarked == []

    # clear_cache() forces a new sweep. A complete sweep is persisted.
    failing.clear()
    autotune.clear_cache()
    assert autotune.autotune(256, 512, 128, verbose=False) == "cluster"
    assert benchmarked == ["basic", "swizzled", "cluster"]
    assert sweep.read_cache() == _complete_sweep_cache()

    benchmarked.clear()
    assert autotune.autotune(256, 512, 128, verbose=False) == "cluster"
    assert benchmarked == []


def test_w2_122_pre_version_cache_entries_are_ignored_and_retuned(
    autotune_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    autotune = autotune_module
    sweep = _FakeSweep(autotune, monkeypatch, tmp_path)
    # Same layout as the entries committed in f2646de5e, which the pre-fix code
    # wrote even when a kernel failed during the sweep.
    sweep.write_cache({_SWEEP_KEY: "swizzled", "B200_10.0_16384x16384x16384": "4stage"})

    with pytest.warns(RuntimeWarning, match=r"Ignoring 2 autotune cache entries"):
        assert autotune.autotune(256, 512, 128, verbose=False) == "cluster"
    assert sweep.benchmarked == ["basic", "swizzled", "cluster"]
    # The complete sweep replaces the old file, so its unverified entries are gone.
    assert sweep.read_cache() == _complete_sweep_cache()


def test_w2_122_repo_cache_file_is_not_served_unverified(
    autotune_module: types.ModuleType,
) -> None:
    autotune = autotune_module
    cache_path = Path(autotune._CACHE_FILE)
    if not cache_path.exists():
        pytest.skip(f"SKIPPED: no autotune cache file at {cache_path}")
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    if "format_version" in data:
        # A regenerated file must pass the complete-sweep validation as written.
        assert autotune._load_cache() == data["entries"]
    else:
        with pytest.warns(RuntimeWarning, match=rf"Ignoring {len(data)} autotune cache entries"):
            assert autotune._load_cache() == {}


def test_w2_122_entry_from_a_different_kernel_set_is_retuned(
    autotune_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    autotune = autotune_module
    sweep = _FakeSweep(autotune, monkeypatch, tmp_path)
    # A sweep that never timed "cluster" did not cover the current kernels.
    sweep.write_cache(_complete_sweep_cache("swizzled", {"basic": 3.0, "swizzled": 2.0}))

    assert autotune.autotune(256, 512, 128, verbose=False) == "cluster"
    assert sweep.benchmarked == ["basic", "swizzled", "cluster"]
    assert sweep.read_cache() == _complete_sweep_cache()


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("{not json", r"is not valid JSON"),
        ({"format_version": 3, "entries": {}}, r"must have format_version 2"),
        (_complete_sweep_cache("basic"), r"fastest kernel as 'winner'"),
        (
            {"format_version": 2, "entries": {_SWEEP_KEY: {"winner": "cluster"}}},
            r"fastest kernel as 'winner'",
        ),
    ],
)
def test_w2_122_malformed_cache_fails_fast(
    autotune_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    payload: object,
    match: str,
) -> None:
    autotune = autotune_module
    sweep = _FakeSweep(autotune, monkeypatch, tmp_path)
    if isinstance(payload, str):
        sweep.cache_file.write_text(payload, encoding="utf-8")
    else:
        sweep.write_cache(payload)

    with pytest.raises(RuntimeError, match=match):
        autotune.autotune(256, 512, 128, verbose=False)
    assert sweep.benchmarked == []


def test_w2_122_cache_write_failure_is_raised(
    autotune_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    autotune = autotune_module
    _FakeSweep(autotune, monkeypatch, tmp_path)
    unwritable = tmp_path / "missing_dir" / "autotune_cache.json"
    monkeypatch.setattr(autotune, "_CACHE_FILE", str(unwritable))

    with pytest.raises(FileNotFoundError):
        autotune.autotune(256, 512, 128, verbose=False)
    assert not unwritable.exists()


class _CpuOptimizedKV(OptimizedKVFP8Compressed):
    allow_cpu = True


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch.float8_e4m3fn required")
def test_w2_129_optimized_fp8_quantize_path_meets_shared_tolerance() -> None:
    bench = _CpuOptimizedKV(
        batch_size=8,
        num_layers=2,
        num_heads=32,
        head_dim=128,
        max_seq_length=4,
        active_layers=1,
        num_decode_steps=2,
    )
    generator = torch.Generator().manual_seed(42)
    step = torch.randn(8, 32, 128, generator=generator, dtype=torch.bfloat16)
    quantized = torch.empty(step.shape, dtype=bench.cache_dtype)

    scale = bench._compute_scale(step)
    bench._quantize_step_into(step, scale, quantized)
    assert scale.dtype == torch.float32

    # One rounding from the FP32 product, not a BF16 product rounded again.
    single_rounding = (step.float() * scale).to(bench.cache_dtype)
    assert torch.equal(quantized.view(torch.uint8), single_rounding.view(torch.uint8))

    # Dequantize the way _build_verification_output does and apply the shared bound.
    restored = quantized.float() / scale
    reference = step.float()
    rtol, atol = FP8_KV_OUTPUT_TOLERANCE
    torch.testing.assert_close(restored, reference, rtol=rtol, atol=atol)

    # Some values scale into the E4M3 subnormal range. Their dequantized error
    # stays far below atol, so atol is not hiding normal-range error.
    scaled_magnitude = (reference * scale).abs()
    subnormal = (scaled_magnitude > 0) & (scaled_magnitude < 2.0**-6)
    assert bool(subnormal.any())
    assert float((restored - reference).abs()[subnormal].max()) < atol / 10

    # A never-written cache or a dropped scale fails the same bound.
    assert not torch.allclose(torch.zeros_like(reference), reference, rtol=rtol, atol=atol)
    assert not torch.allclose(quantized.float(), reference, rtol=rtol, atol=atol)


def _harness_topk_tolerance(backend: str, mode: str) -> tuple[float, float]:
    # Mirrors TopKKernelBenchmark.capture_verification_payload.
    return (2e-2, 2e-2) if backend == "cuda" and mode == "fwd_bwd" else (1e-2, 1e-2)


def _topk_overrides(mode: str) -> list[str]:
    return [
        "--batch-size", "1",
        "--heads", "4",
        "--kv-heads", "2",
        "--q-len", "8",
        "--compressed-k-len", "64",
        "--head-dim", "16",
        "--top-k", "2",
        "--selection-block-size", "8",
        "--mode", mode,
        "--dtype", "fp16",
    ]


def _run_topk_benchmark(backend: str, mode: str) -> topk_common.TopKKernelOutputs:
    bench = TopKKernelBenchmark(backend=backend, label=f"r06_{backend}_{mode}")
    bench.apply_target_overrides(_topk_overrides(mode))
    bench.setup()
    try:
        bench.benchmark_fn()
        assert bench.validate_result() is None
        assert bench.outputs is not None
        return bench.outputs
    finally:
        bench.teardown()


@pytest.mark.parametrize("mode", ["forward", "fwd_bwd"])
def test_w2_137_cuda_topk_preloads_extension_and_matches_baseline(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    events: list[str] = []

    class ReferenceGemmExtension:
        # CPU stand-in for the CUTLASS GEMM, which cannot build on this host.
        # The B200 run of the CUDA-gated test below covers the real kernel.
        @staticmethod
        def matmul_cutlass_topk(a: torch.Tensor, b_rows: torch.Tensor) -> torch.Tensor:
            events.append("gemm")
            assert a.dtype == torch.float16 and b_rows.dtype == torch.float16
            assert a.is_contiguous() and b_rows.is_contiguous()
            return a.float() @ b_rows.float().T

    def load_extension() -> type[ReferenceGemmExtension]:
        events.append("load")
        return ReferenceGemmExtension

    real_select = topk_common.run_optimized_top_k_select

    def recording_select(*args, **kwargs):
        events.append("select")
        return real_select(*args, **kwargs)

    monkeypatch.setattr(topk_common, "resolve_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(topk_extension, "load_top_k_kernel_extension", load_extension)
    monkeypatch.setattr(topk_common, "run_optimized_top_k_select", recording_select)

    cuda_outputs = _run_topk_benchmark("cuda", mode)
    # setup() builds the extension before its warmup selection in both modes.
    assert events[:2] == ["load", "select"]
    assert "gemm" in events
    assert not hasattr(topk_common, "_cutlass_group_backward")

    baseline_outputs = _run_topk_benchmark("baseline", mode)
    rtol, atol = _harness_topk_tolerance("cuda", mode)
    torch.testing.assert_close(
        build_verification_tensor(cuda_outputs),
        build_verification_tensor(baseline_outputs),
        rtol=rtol,
        atol=atol,
    )
    if mode == "fwd_bwd":
        assert cuda_outputs.q_grad is not None and baseline_outputs.q_grad is not None
        assert cuda_outputs.k_grad is not None and baseline_outputs.k_grad is not None
        torch.testing.assert_close(
            cuda_outputs.q_grad.float(), baseline_outputs.q_grad.float(), rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            cuda_outputs.k_grad.float(), baseline_outputs.k_grad.float(), rtol=2e-2, atol=2e-2
        )


def _cuda_cutlass_topk_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)


@pytest.mark.skipif(
    not _cuda_cutlass_topk_supported(),
    reason="SKIPPED: CUDA device with sm_80 or newer required for the CUTLASS Top-K GEMM",
)
@pytest.mark.parametrize("mode", ["forward", "fwd_bwd"])
def test_w2_137_cutlass_topk_matches_baseline_on_gpu(mode: str) -> None:
    # q_len 1100 spans one full 1024-row CUTLASS query tile plus a 76-row tail.
    overrides = [
        "--batch-size", "1",
        "--heads", "8",
        "--kv-heads", "2",
        "--q-len", "1100",
        "--compressed-k-len", "1024",
        "--head-dim", "128",
        "--top-k", "4",
        "--selection-block-size", "64",
        "--mode", mode,
        "--dtype", "fp16",
    ]
    device = torch.device("cuda")
    workload = topk_common.apply_workload_overrides(topk_common.TopKKernelWorkload(), overrides)
    topk_common._validate_cuda_backend_workload(workload)
    inputs = topk_common.build_inputs(workload, device)

    # Every score element, including the tail tile, matches an FP64 GEMM of the
    # same FP16-rounded operands that the CUTLASS kernel reads.
    with torch.inference_mode():
        scores, q_sum, block_k = topk_common._run_cuda_reduced_group_block_scores(
            inputs.q, inputs.k, workload, use_cutlass=True
        )
        reference_scores = (
            q_sum.half().double() @ block_k.half().double().transpose(-1, -2)
        ) * workload.scale
    torch.testing.assert_close(scores.double(), reference_scores, rtol=1e-3, atol=1e-2)

    outputs: dict[str, topk_common.TopKKernelOutputs] = {}
    for backend in ("baseline", "cuda"):
        bench = TopKKernelBenchmark(backend=backend, label=f"r06_gpu_{backend}_{mode}")
        bench.apply_target_overrides(overrides)
        bench.setup()
        try:
            bench.benchmark_fn()
            torch.cuda.synchronize()
            assert bench.validate_result() is None
            assert bench.outputs is not None
            outputs[backend] = bench.outputs
        finally:
            bench.teardown()

    rtol, atol = _harness_topk_tolerance("cuda", mode)
    torch.testing.assert_close(
        build_verification_tensor(outputs["cuda"]),
        build_verification_tensor(outputs["baseline"]),
        rtol=rtol,
        atol=atol,
    )


def test_w2_137_cutlass_helper_comments_match_the_kernel_and_backward_path() -> None:
    import inspect

    cuda_source = (
        Path(topk_extension.__file__).with_name("top_k_kernel_cuda.cu").read_text(encoding="utf-8")
    )
    header = cuda_source.split("*/", 1)[0]
    header_flat = " ".join(line.strip(" */") for line in header.splitlines()).split()
    header_text = " ".join(header_flat)
    assert "score reduced grouped query rows against block-compressed K tiles" in header_text
    assert "The grouped backward GEMMs run through torch.matmul." in header_text
    assert "to run the grouped backward GEMMs" not in header_text

    # The backward pass really uses torch.matmul and never loads the CUTLASS helper.
    backward_source = inspect.getsource(topk_common._cuda_reduced_group_backward)
    assert "torch.matmul" in backward_source
    assert "load_top_k_kernel_extension" not in backward_source

    doc = topk_extension.load_top_k_kernel_extension.__doc__ or ""
    assert doc == "Load the CUTLASS SM80 tensor-op GEMM helper that scores the forward CUDA Top-K path."
    assert "SM100" not in doc
    assert "cutlass::arch::OpClassTensorOp" in cuda_source
    assert "cutlass::arch::Sm80" in cuda_source
    assert "Sm100" not in cuda_source
