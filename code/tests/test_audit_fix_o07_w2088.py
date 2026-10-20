"""Regressions for audit findings W2-088 and W2-134 in the GPT-4 lab."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import labs.real_world_models.gpt4_architecture_optimization as gpt4_module
from labs.real_world_models.gpt4_architecture_optimization import (
    GPT4ArchitectureOptimization,
    GPT4ArchitectureOptimizationBenchmark,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
README = CODE_ROOT / "labs" / "real_world_models" / "README.md"
REPORTED_FLAGS = ("use_moe", "use_fp8", "use_context_parallel")


def _run_short_proxy() -> dict[str, object]:
    proxy = GPT4ArchitectureOptimization(
        batch_size=1,
        seq_length=4,
        use_moe=True,
        use_fp8=True,
        use_context_parallel=True,
    )
    proxy.setup()
    try:
        proxy.run()
        proxy.finalize_timing()
        assert proxy.output is not None
        return {
            "module_types": {type(module).__name__ for module in proxy.layers.modules()},
            "dtypes": {param.dtype for param in proxy.layers.parameters()} | {proxy.output.dtype},
            "num_layers": len(proxy.layers),
            "hidden_size": proxy.input.shape[-1],
        }
    finally:
        proxy.cleanup()


def _run_wrapper(monkeypatch: pytest.MonkeyPatch) -> GPT4ArchitectureOptimizationBenchmark:
    class _ShortProxy(GPT4ArchitectureOptimization):
        def __init__(self) -> None:
            super().__init__(batch_size=1, seq_length=4)

    monkeypatch.setattr(gpt4_module, "GPT4ArchitectureOptimization", _ShortProxy)
    benchmark = GPT4ArchitectureOptimizationBenchmark()
    if not torch.cuda.is_available():
        # setup() rejects non-CUDA devices, and the proxy places itself on CPU.
        device_tag = SimpleNamespace(type="cuda")
        benchmark.device = device_tag
        real_empty = torch.empty

        def empty_on_cpu(*args, **kwargs):
            if kwargs.get("device") is device_tag:
                kwargs = {**kwargs, "device": torch.device("cpu")}
            return real_empty(*args, **kwargs)

        monkeypatch.setattr(gpt4_module.torch, "empty", empty_on_cpu)
    benchmark.setup()
    benchmark.benchmark_fn()
    benchmark.capture_verification_payload()
    return benchmark


def test_w2_088_module_docstring_describes_only_the_executed_proxy() -> None:
    doc = " ".join((gpt4_module.__doc__ or "").split())

    for unexecuted_claim in (
        "Expert parallelism for MoE layers",
        "Context parallelism for 128K context",
        "FP8 quantization",
        "Disaggregated prefill/decode",
    ):
        assert unexecuted_claim not in doc
    assert "four-layer LayerNorm plus Linear residual stack in BF16 at hidden size 4096" in doc
    assert "reports MoE, FP8, and context parallelism as not executed" in doc

    executed = _run_short_proxy()
    assert executed["module_types"] == {"ModuleList", "SimplifiedGPT4Layer", "LayerNorm", "Linear"}
    assert executed["dtypes"] == {torch.bfloat16}
    assert executed["num_layers"] == 4
    assert executed["hidden_size"] == 4096


def test_w2_088_readme_matches_the_reported_benchmark_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readme = README.read_text(encoding="utf-8")
    contents_row = next(
        line for line in readme.splitlines()
        if line.startswith("| `gpt4_architecture_optimization.py` |")
    )
    validation_line = next(
        line for line in readme.splitlines()
        if line.startswith("- `gpt4_architecture_optimization.py`")
    )

    assert "MoE + context-parallel sketch with FP8 support" not in readme
    assert "runs the context-parallel path" not in readme
    assert "24+ GPUs" not in readme
    assert "four-layer dense BF16 timing proxy" in contents_row
    assert "does not run MoE, FP8, or context parallelism" in contents_row
    assert "`use_moe`, `use_fp8`, and `use_context_parallel` as `0.0`" in validation_line
    assert "memory-only estimate" in validation_line

    benchmark = _run_wrapper(monkeypatch)
    try:
        metrics = benchmark.get_custom_metrics()
        for flag in REPORTED_FLAGS:
            assert f"`{flag}`" in validation_line
            assert metrics[f"gpt4_architecture.{flag}"] == 0.0
        assert metrics["gpt4_architecture.mean_time_ms"] > 0.0
        precision = benchmark._require_payload().precision_flags
        assert precision.bf16 is True
        assert precision.fp8 is False
    finally:
        benchmark.teardown()


def test_w2_134_fp8_estimate_stores_weights_and_kv_at_one_byte() -> None:
    batch_size = 2
    seq_length = 4096
    bf16 = GPT4ArchitectureOptimization(
        batch_size=batch_size,
        seq_length=seq_length,
        use_moe=True,
        use_fp8=False,
    )
    fp8 = GPT4ArchitectureOptimization(
        batch_size=batch_size,
        seq_length=seq_length,
        use_moe=True,
        use_fp8=True,
    )
    kv_values = batch_size * seq_length * bf16.NUM_LAYERS * 2 * bf16.HIDDEN_SIZE

    assert bf16.estimated_bytes_per_value == 2
    assert fp8.estimated_bytes_per_value == 1
    assert fp8.estimated_parameter_count == bf16.estimated_parameter_count
    assert bf16.estimated_parameter_memory_gb == pytest.approx(
        bf16.estimated_parameter_count * 2 / 1024**3
    )
    assert fp8.estimated_parameter_memory_gb == pytest.approx(
        fp8.estimated_parameter_count / 1024**3
    )
    assert bf16.estimated_kv_memory_gb == pytest.approx(kv_values * 2 / 1024**3)
    assert fp8.estimated_kv_memory_gb == pytest.approx(kv_values / 1024**3)
    for estimate in (bf16, fp8):
        assert estimate.estimated_min_b200_gpus == math.ceil(
            estimate.estimated_total_memory_gb / 192
        )
    assert fp8.estimated_min_b200_gpus < bf16.estimated_min_b200_gpus
