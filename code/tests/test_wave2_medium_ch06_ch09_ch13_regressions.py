from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]


def test_expectation_keys_have_one_canonical_record_per_benchmark() -> None:
    from core.harness.run_benchmarks import expectation_example_key

    expectation_paths = (
        CODE_ROOT / "ch06" / "expectations_b200.json",
        CODE_ROOT / "ch06" / "expectations_4x_gb200.json",
        CODE_ROOT / "ch08" / "expectations_b200.json",
        CODE_ROOT / "ch08" / "expectations_4x_gb200.json",
        CODE_ROOT / "ch10" / "expectations_b200.json",
        CODE_ROOT / "ch10" / "expectations_4x_gb200.json",
    )

    for path in expectation_paths:
        examples = json.loads(path.read_text(encoding="utf-8"))["examples"]
        identities: set[tuple[str, str]] = set()
        for key, entry in examples.items():
            identity = (entry["example"], entry["type"])
            assert identity not in identities, f"duplicate expectation identity in {path}: {identity}"
            identities.add(identity)
            assert key == expectation_example_key(*identity), f"noncanonical key in {path}: {key}"

    assert "add_cuda" in json.loads(
        (CODE_ROOT / "ch06" / "expectations_4x_gb200.json").read_text(encoding="utf-8")
    )["examples"]
    assert "hbm_cuda" in json.loads(
        (CODE_ROOT / "ch08" / "expectations_4x_gb200.json").read_text(encoding="utf-8")
    )["examples"]


def test_cublaslt_per_channel_scaling_uses_column_major_output_offsets() -> None:
    for filename in (
        "baseline_cublas_gemm_fp4_perchannel.cu",
        "optimized_cublas_gemm_fp4_perchannel.cu",
    ):
        source = (CODE_ROOT / "ch09" / filename).read_text(encoding="utf-8")
        kernel = source.split("__global__ void apply_per_channel_scale", 1)[1].split(
            "void quantize_to_nvfp4_kmajor", 1
        )[0]

        assert "static_cast<size_t>(col) * rows + row" in kernel
        assert "row * cols + col" not in kernel
        assert "scales[col]" in kernel

    for filename in (
        "expectations_b200.json",
        "expectations_2x_b200.json",
        "expectations_4x_gb200.json",
    ):
        examples = json.loads((CODE_ROOT / "ch09" / filename).read_text(encoding="utf-8"))[
            "examples"
        ]
        assert "cublas_gemm_fp4_perchannel_cuda" not in examples


def test_memory_bound_roofline_metrics_include_every_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ch09.baseline_memory_bound import BaselineMemoryBoundBenchmark
    from ch09.optimized_memory_bound import OptimizedMemoryBoundBenchmark
    from core.benchmark import metrics as benchmark_metrics

    def capture_metrics(**kwargs: float | str | None) -> dict[str, float | str | None]:
        return kwargs

    monkeypatch.setattr(benchmark_metrics, "compute_roofline_metrics", capture_metrics)

    for benchmark_type in (BaselineMemoryBoundBenchmark, OptimizedMemoryBoundBenchmark):
        benchmark = object.__new__(benchmark_type)
        benchmark.N = 17
        benchmark.repeats = 64
        benchmark._last_elapsed_ms = 2.5

        metrics = benchmark.get_custom_metrics()
        assert metrics is not None
        assert metrics["total_flops"] == 17 * 2 * 64
        assert metrics["total_bytes"] == 17 * torch.float32.itemsize * 2 * 64
        assert metrics["elapsed_ms"] == 2.5

    b200_expectations = json.loads(
        (CODE_ROOT / "ch09" / "expectations_b200.json").read_text(encoding="utf-8")
    )["examples"]
    assert "custom_metrics" not in b200_expectations["memory_bound"]


def test_all_concepts_cutlass_kernel_uses_claimed_four_cta_cluster() -> None:
    source = (CODE_ROOT / "ch09" / "optimized_cutlass_gemm_fp4_all_concepts.cu").read_text(
        encoding="utf-8"
    )

    assert "using ClusterShape = Shape<_1, _4, _1>;" in source
    assert "using ClusterShape = Shape<_1, _1, _1>;" not in source

    for filename in ("expectations_b200.json", "expectations_4x_gb200.json"):
        examples = json.loads((CODE_ROOT / "ch09" / filename).read_text(encoding="utf-8"))[
            "examples"
        ]
        assert "cutlass_gemm_fp4_all_concepts_cuda" not in examples


def test_tcgen05_bias_silu_partitions_bias_by_global_output_column() -> None:
    source = (CODE_ROOT / "ch09" / "tcgen05_basic.cu").read_text(encoding="utf-8")
    kernel = source.split("__global__ void gemm_device", 1)[1].split(
        "torch::Tensor run_tcgen05_matmul", 1
    )[0]

    assert "make_layout(shape(mD), make_stride(Int<0>{}, Int<1>{}))" in kernel
    assert "Tensor tDgBias = thr_t2r_copy.partition_D(tCgBias);" in kernel
    assert "copy(tDgBias, tDrBias);" in kernel
    assert (
        "static_cast<float>(tDrAcc(i)) + static_cast<float>(tDrBias(i))"
        in kernel
    )
    assert "bias application would need proper coordinate mapping" not in kernel


def test_transformer_engine_precision_pair_keeps_optimized_path_eager() -> None:
    source = (CODE_ROOT / "ch13" / "optimized_precisionfp8_te.py").read_text(
        encoding="utf-8"
    )

    assert "torch.cuda.CUDAGraph" not in source
    assert "torch.cuda.graph(" not in source
    assert ".graph.replay()" not in source
    assert "capture_stream" not in source
    assert "foreach=False" not in source
    assert '"fp8": True' in source

    for filename in ("expectations_b200.json", "expectations_2x_b200.json"):
        examples = json.loads((CODE_ROOT / "ch13" / filename).read_text(encoding="utf-8"))[
            "examples"
        ]
        assert "precisionfp8_te" not in examples

    readme = (CODE_ROOT / "ch13" / "README.md").read_text(encoding="utf-8")
    assert "compared eager FP16 with CUDA-graph-replayed FP8" in readme
    assert "publish a new speed result only after a fresh B200" in readme


def test_transformer_engine_eager_benchmark_runs_one_fp8_training_step() -> None:
    from ch13.optimized_precisionfp8_te import OptimizedTEFP8Benchmark

    benchmark = object.__new__(OptimizedTEFP8Benchmark)
    benchmark.input_pool = [torch.tensor([1.0])]
    benchmark.target_pool = [torch.tensor([2.0])]
    benchmark.output_buffer = torch.empty(1)
    benchmark.output = None
    benchmark._verify_input = benchmark.input_pool[0].clone()
    benchmark._nvtx_range = lambda _label: nullcontext()
    calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def train_step(batch: torch.Tensor, target: torch.Tensor, capture_output: bool = False) -> None:
        calls.append((batch, target, capture_output))
        benchmark.output_buffer.copy_(batch + target)

    benchmark._train_step_impl = train_step
    benchmark.benchmark_fn()

    assert calls == [(benchmark.input_pool[0], benchmark.target_pool[0], True)]
    assert benchmark.output is benchmark.output_buffer
    torch.testing.assert_close(benchmark.output, torch.tensor([3.0]))
