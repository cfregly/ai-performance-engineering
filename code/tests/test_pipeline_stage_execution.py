from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

import ch04.baseline_pipeline_parallel as baseline
import ch04.optimized_pipeline_parallel_1f1b as optimized
from core.harness.benchmark_harness import BenchmarkConfig


def _scaled_stage(scale: float) -> nn.ModuleList:
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(2) * scale)
    return nn.ModuleList([layer])


@pytest.mark.parametrize(
    "run_rank_stage",
    (baseline._run_rank_stage, optimized._run_rank_stage_inplace),
)
def test_worker_rank_executes_its_own_nested_stage(
    run_rank_stage: Callable[[nn.ModuleList, int, torch.Tensor], torch.Tensor],
) -> None:
    stages = nn.ModuleList([_scaled_stage(2.0), _scaled_stage(3.0)])
    inputs = torch.tensor([[1.0, 2.0]])

    with torch.inference_mode():
        output = run_rank_stage(stages, 1, inputs)

    torch.testing.assert_close(output, inputs * 3.0)


@pytest.mark.parametrize(
    "run_rank_stage",
    (baseline._run_rank_stage, optimized._run_rank_stage_inplace),
)
def test_worker_rank_rejects_a_stage_outside_the_pipeline(
    run_rank_stage: Callable[[nn.ModuleList, int, torch.Tensor], torch.Tensor],
) -> None:
    stages = nn.ModuleList([_scaled_stage(1.0), _scaled_stage(1.0)])

    with pytest.raises(IndexError, match="outside 2 stages"):
        run_rank_stage(stages, 2, torch.ones((1, 2)))


def test_optimized_virtual_path_matches_baseline_stage_order_on_cpu() -> None:
    device = torch.device("cpu")
    torch.manual_seed(42)
    baseline_fwd, baseline_bwd = baseline._build_stage_layers(
        hidden=4,
        layers_per_stage=2,
        stage_count=2,
        device=device,
    )
    torch.manual_seed(42)
    optimized_fwd, optimized_bwd = optimized._build_stage_layers(
        hidden=4,
        layers_per_stage=2,
        stage_count=2,
        device=device,
    )
    torch.manual_seed(9)
    inputs = torch.randn((4, 3, 4), dtype=torch.bfloat16)

    with torch.inference_mode():
        expected = baseline._run_virtual_pipeline_baseline(
            inputs,
            baseline_fwd,
            baseline_bwd,
            num_micro_batches=2,
        )
        benchmark = optimized.OptimizedPipelineParallelBenchmark()
        benchmark._input = inputs
        benchmark._micro_batch = inputs.narrow(0, 0, 2)
        benchmark._fwd_layers = optimized_fwd
        benchmark._bwd_layers = optimized_bwd
        benchmark._world_size = 2
        benchmark._world_size_range = range(2)
        benchmark.benchmark_fn()

    assert benchmark._output is not None
    torch.testing.assert_close(benchmark._output, expected, rtol=0.0, atol=0.0)
    assert benchmark._output.shape == (2, 3, 4)


@pytest.mark.parametrize(
    "benchmark_factory",
    (
        baseline.BaselinePipelineParallelBenchmark,
        optimized.OptimizedPipelineParallelBenchmark,
    ),
)
def test_pipeline_specs_declare_their_worker_iteration_mean(
    benchmark_factory: type,
) -> None:
    benchmark = benchmark_factory()
    benchmark._subprocess_verify_output = torch.ones(1)
    config = BenchmarkConfig(iterations=7, warmup=5, nproc_per_node=2)

    spec = benchmark.get_torchrun_spec(config)

    assert spec.timing_source == "rank0_time_per_iter_ms"
    assert spec.timing_iterations_per_sample == 7
