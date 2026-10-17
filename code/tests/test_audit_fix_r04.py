"""Regression tests for audit work unit r04 (W2-057, W2-058, W2-070, W2-083)."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch

from ch15 import expert_parallelism
from ch15.expert_parallelism import _throughput_summary
from ch16.awq_gptq_smoothquant_benchmarks import (
    PTQ_OUTPUT_TOLERANCE,
    PTQMLP,
    PTQWorkload,
    ReferenceMLP,
)
from ch18.eos_early_exit_common import EosEarlyExitBenchmark, EosEarlyExitConfig
from core.benchmark.quarantine import QuarantineManager
from core.benchmark.verification import ToleranceSpec
from core.benchmark.verify_runner import VerifyRunner
from labs.cutlass_profiler_kernel_selector.run_triton_matmul import (
    MAX_ABS_ERROR_PER_SQRT_K,
    TRITON_AVAILABLE,
    benchmark_triton_matmul,
    verify_triton_matmul_output,
)
from labs.cutlass_profiler_kernel_selector.shapes import GemmShape

# W2-057: distributed mode must not present a per-rank rate as the whole run.


def test_w2_057_distributed_throughput_reports_per_rank_and_aggregate() -> None:
    summary = _throughput_summary(
        mode="distributed",
        tokens_per_rank=1024,
        per_iter_ms=2.0,
        world_size=4,
    )

    assert summary == "512,000 tok/s per rank, 2,048,000 tok/s across 4 ranks"


def test_w2_057_local_throughput_is_single_gpu_rate() -> None:
    summary = _throughput_summary(
        mode="local",
        tokens_per_rank=1024,
        per_iter_ms=2.0,
        world_size=1,
    )

    assert summary == "512,000 tok/s"


@pytest.mark.parametrize("per_iter_ms", (0.0, -1.0, float("nan")))
def test_w2_057_throughput_rejects_invalid_timing(per_iter_ms: float) -> None:
    with pytest.raises(RuntimeError, match="must be positive"):
        _throughput_summary(
            mode="distributed",
            tokens_per_rank=1024,
            per_iter_ms=per_iter_ms,
            world_size=2,
        )


@pytest.mark.parametrize(
    ("mode", "world_size"),
    (("local", 2), ("distributed", 1), ("hybrid", 2)),
)
def test_w2_057_throughput_rejects_inconsistent_mode(mode: str, world_size: int) -> None:
    with pytest.raises(ValueError):
        _throughput_summary(
            mode=mode,
            tokens_per_rank=1024,
            per_iter_ms=2.0,
            world_size=world_size,
        )


def test_w2_057_main_reports_throughput_through_the_summary() -> None:
    source = Path(expert_parallelism.__file__).read_text(encoding="utf-8")
    main_def = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [node for node in ast.walk(main_def) if isinstance(node, ast.Call)]
    summary_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "_throughput_summary"
    ]
    all_reduce_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "all_reduce"
    ]
    formatted = [ast.unparse(node) for node in ast.walk(main_def) if isinstance(node, ast.JoinedStr)]

    assert len(summary_calls) == 1
    assert {keyword.arg: ast.unparse(keyword.value) for keyword in summary_calls[0].keywords} == {
        "mode": "args.mode",
        "tokens_per_rank": "args.batch * args.seq",
        "per_iter_ms": "per_iter_ms",
        "world_size": "ctx.world_size if ctx is not None else 1",
    }
    assert len(all_reduce_calls) == 1
    assert all_reduce_calls[0].lineno < summary_calls[0].lineno
    assert any("{throughput}" in text for text in formatted)
    assert not any("tok/s" in text for text in formatted)


# W2-058: the shared PTQ tolerance must pass real INT4/INT8 outputs and reject
# a uniform output scale error. The fixture uses the benchmark's full workload
# shape because the worst element error grows with the number of rows.


@pytest.fixture(scope="module")
def ptq_outputs() -> dict[str, torch.Tensor]:
    workload = PTQWorkload(dtype=torch.bfloat16)
    device = torch.device("cpu")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        reference = ReferenceMLP(workload, device).eval()
        inputs = torch.randn(workload.batch_size, workload.in_features, dtype=workload.dtype)
        calibration = torch.randn(
            workload.calibration_samples,
            workload.in_features,
            dtype=workload.dtype,
        )
    outputs: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        outputs["baseline"] = reference(inputs).float()
        for scheme in ("awq", "gptq", "smoothquant"):
            quantized = PTQMLP(reference, calibration, scheme=scheme).eval()
            quantized.prepare_buffers(workload.batch_size, device)
            outputs[scheme] = quantized(inputs).clone()
    return outputs


def _harness_accepts(tmp_path: Path, expected: torch.Tensor, actual: torch.Tensor) -> bool:
    runner = VerifyRunner(
        cache_dir=tmp_path / "golden",
        quarantine_manager=QuarantineManager(cache_dir=tmp_path / "quarantine"),
    )
    rtol, atol = PTQ_OUTPUT_TOLERANCE
    details = runner._compare_outputs(
        {"output": expected},
        {"output": actual},
        ToleranceSpec(rtol=rtol, atol=atol),
    )
    return bool(details.passed)


@pytest.mark.parametrize("scheme", ("awq", "gptq", "smoothquant"))
def test_w2_058_real_ptq_outputs_pass_the_shared_tolerance(
    tmp_path: Path,
    ptq_outputs: dict[str, torch.Tensor],
    scheme: str,
) -> None:
    assert _harness_accepts(tmp_path, ptq_outputs["baseline"], ptq_outputs[scheme])


@pytest.mark.parametrize("scale", (0.75, 1.25))
def test_w2_058_tolerance_rejects_uniform_output_scale_error(
    tmp_path: Path,
    ptq_outputs: dict[str, torch.Tensor],
    scale: float,
) -> None:
    baseline = ptq_outputs["baseline"]

    assert not _harness_accepts(tmp_path, baseline, baseline * scale)


def test_w2_058_tolerance_rejects_zero_and_sign_flipped_outputs(
    tmp_path: Path,
    ptq_outputs: dict[str, torch.Tensor],
) -> None:
    baseline = ptq_outputs["baseline"]

    assert not _harness_accepts(tmp_path, baseline, torch.zeros_like(baseline))
    assert not _harness_accepts(tmp_path, baseline, -baseline)


# W2-070: a row that reached EOS records EOS at every later step, so the
# optimized arm's early exit cannot change the verified output.


def _run_eos_decode(**overrides: object) -> EosEarlyExitBenchmark:
    bench = EosEarlyExitBenchmark(
        EosEarlyExitConfig(
            batch_size=1,
            decode_tokens=8,
            force_eos_after_tokens=8,
            **overrides,
        )
    )
    bench.device = torch.device("cpu")
    bench.setup()
    bench.benchmark_fn()
    return bench


def test_w2_070_natural_eos_output_matches_between_arms() -> None:
    probe_tokens = _run_eos_decode().output[0].clone()
    eos_token_id = int(probe_tokens[1])
    first_eos_step = int((probe_tokens == eos_token_id).nonzero()[0])

    baseline = _run_eos_decode(eos_token_id=eos_token_id, stop_on_all_done=False)
    optimized = _run_eos_decode(eos_token_id=eos_token_id, stop_on_all_done=True)

    assert optimized._decoded_steps == first_eos_step + 1
    assert optimized._decoded_steps < optimized.cfg.force_eos_after_tokens
    assert baseline._decoded_steps == baseline.cfg.decode_tokens
    assert torch.equal(baseline.output[0, : first_eos_step + 1], probe_tokens[: first_eos_step + 1])
    assert torch.all(baseline.output[0, first_eos_step:] == eos_token_id)
    assert torch.equal(optimized.output, baseline.output)


# W2-083: the Triton runner must check its timed output before reporting TFLOP/s.


def _fp16_operands(m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(83)
    a = torch.randn((m, k), generator=generator).half()
    b = torch.randn((k, n), generator=generator).half()
    return a, b


def test_w2_083_verify_accepts_a_correct_product() -> None:
    a, b = _fp16_operands(64, 48, 100)
    c = torch.matmul(a.float(), b.float())

    max_abs_error = verify_triton_matmul_output(c, a, b)

    assert max_abs_error <= MAX_ABS_ERROR_PER_SQRT_K * math.sqrt(100)


def test_w2_083_verify_rejects_a_product_missing_the_k_tail() -> None:
    a, b = _fp16_operands(64, 48, 100)
    full_k_tiles = (100 // 32) * 32
    c = torch.matmul(a[:, :full_k_tiles].float(), b[:full_k_tiles].float())

    with pytest.raises(RuntimeError, match="does not match torch.matmul"):
        verify_triton_matmul_output(c, a, b)


def test_w2_083_verify_rejects_non_finite_and_misshaped_outputs() -> None:
    a, b = _fp16_operands(64, 48, 100)
    c = torch.matmul(a.float(), b.float())
    c[3, 5] = float("nan")

    with pytest.raises(RuntimeError, match="non-finite"):
        verify_triton_matmul_output(c, a, b)
    with pytest.raises(RuntimeError, match="shape"):
        verify_triton_matmul_output(torch.zeros(64, 47), a, b)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_AVAILABLE),
    reason="SKIPPED: requires CUDA and Triton to run the custom Triton matmul kernel",
)
def test_w2_083_real_triton_kernel_reports_verified_tail_shape() -> None:
    shape = GemmShape(name="k_tail_m96_n80_k100", m=96, n=80, k=100)

    metrics = benchmark_triton_matmul(shape, warmup=1, iters=2, dtype=torch.float16)

    assert metrics is not None
    assert metrics["tflops"] > 0.0
    assert 0.0 <= metrics["max_abs_error"] <= MAX_ABS_ERROR_PER_SQRT_K * math.sqrt(shape.k)
