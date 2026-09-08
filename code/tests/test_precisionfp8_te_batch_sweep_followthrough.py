from __future__ import annotations

import pytest

from ch13.baseline_precisionfp8_te import BaselineTEFP8Benchmark
from ch13.optimized_precisionfp8_te import OptimizedTEFP8Benchmark
from ch13.te_runtime_common import (
    TE_PRECISION_DEFAULT_BATCH_SIZE,
)
from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness

BENCHMARK_TYPES = (BaselineTEFP8Benchmark, OptimizedTEFP8Benchmark)
FROZEN_TOLERANCES = {
    "prediction": (0.4, 1.0),
    "parameter.fc1.weight": (0.001, 0.00075),
    "parameter.fc1.bias": (0.001, 0.00005),
    "parameter.fc2.weight": (0.001, 0.00075),
    "parameter.fc2.bias": (0.001, 0.00005),
}


@pytest.mark.parametrize("benchmark_type", BENCHMARK_TYPES)
def test_te_precision_batch_override_keeps_256_default_and_frozen_contract(
    benchmark_type,
) -> None:
    benchmark = benchmark_type()

    assert benchmark.batch_size == TE_PRECISION_DEFAULT_BATCH_SIZE == 256
    assert benchmark._workload.tokens_per_iteration == 256 * 4096
    assert benchmark.get_workload_metadata().tokens_per_iteration == 256 * 4096
    assert benchmark.get_output_tolerances() == FROZEN_TOLERANCES
    config = benchmark.get_config()
    assert (config.iterations, config.warmup) == (50, 10)


@pytest.mark.parametrize("batch_size", (256, 1024, 4096))
def test_te_precision_batch_override_is_symmetric_and_updates_workload_metadata(
    batch_size: int,
) -> None:
    benchmarks = [benchmark_type() for benchmark_type in BENCHMARK_TYPES]

    for benchmark in benchmarks:
        benchmark.apply_target_overrides(["--batch-size", str(batch_size)])
        assert benchmark.batch_size == batch_size
        expected_tokens = float(batch_size * benchmark.hidden_dim)
        assert benchmark._workload.tokens_per_iteration == expected_tokens
        assert benchmark.get_workload_metadata().tokens_per_iteration == expected_tokens

    assert benchmarks[0].signature_equivalence_group == benchmarks[1].signature_equivalence_group
    assert benchmarks[0].signature_equivalence_ignore_fields == ("precision_flags",)
    assert benchmarks[1].signature_equivalence_ignore_fields == ("precision_flags",)


@pytest.mark.parametrize("benchmark_type", BENCHMARK_TYPES)
def test_harness_routes_target_extra_batch_argument_to_te_precision_pair(
    benchmark_type,
) -> None:
    config = BenchmarkConfig(
        target_label="ch13:precisionfp8_te",
        target_extra_args={
            "ch13:precisionfp8_te": ["--batch-size", "1024"],
        },
    )
    benchmark = benchmark_type()

    BenchmarkHarness(config=config)._apply_target_overrides(benchmark, config)

    assert benchmark.batch_size == 1024
    assert benchmark.get_workload_metadata().tokens_per_iteration == 1024 * 4096


@pytest.mark.parametrize("benchmark_type", BENCHMARK_TYPES)
@pytest.mark.parametrize("value", ("0", "-1", "not-an-integer"))
def test_te_precision_batch_override_rejects_invalid_values_without_fallback(
    benchmark_type,
    value: str,
) -> None:
    benchmark = benchmark_type()

    with pytest.raises(ValueError, match="--batch-size must be a positive integer"):
        benchmark.apply_target_overrides(["--batch-size", value])

    assert benchmark.batch_size == 256
    with pytest.raises(ValueError, match="Invalid target override"):
        benchmark.setup()
