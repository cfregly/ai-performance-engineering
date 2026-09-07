from __future__ import annotations

import copy

import torch

from ch13.baseline_sequence_parallel_multigpu import (
    BaselineSequenceParallelMultigpuBenchmark,
)
from ch13.optimized_sequence_parallel_multigpu import (
    OptimizedSequenceParallelMultigpuBenchmark,
)
from ch13.sequence_parallel_benchmark_common import (
    SequenceParallelConfig,
    build_layers,
)
from core.harness.benchmark_harness import BaseBenchmark


def _build_cpu_surrogate(
    benchmark_type: type[
        BaselineSequenceParallelMultigpuBenchmark
        | OptimizedSequenceParallelMultigpuBenchmark
    ],
    *,
    config: SequenceParallelConfig,
    input_tensor: torch.Tensor,
    up_proj: torch.nn.ModuleList,
    down_proj: torch.nn.ModuleList,
    norms: torch.nn.ModuleList,
) -> BaselineSequenceParallelMultigpuBenchmark | OptimizedSequenceParallelMultigpuBenchmark:
    benchmark = object.__new__(benchmark_type)
    BaseBenchmark.__init__(benchmark)
    benchmark._sp_config = config
    benchmark._world_size = 2
    benchmark._seq_len = config.seq_len
    benchmark._up_proj = copy.deepcopy(up_proj)
    benchmark._down_proj = copy.deepcopy(down_proj)
    benchmark._norms = copy.deepcopy(norms)
    benchmark._layer_triples = tuple(
        zip(benchmark._up_proj, benchmark._down_proj, benchmark._norms, strict=True)
    )
    benchmark._input = input_tensor.clone()
    benchmark._output = None
    if isinstance(benchmark, BaselineSequenceParallelMultigpuBenchmark):
        benchmark._full_sequence = torch.empty(
            config.batch_size,
            config.seq_len,
            config.hidden_size,
            dtype=config.dtype,
        )
    return benchmark


def test_sequence_parallel_pair_declares_shared_all_reduce_workload_contract() -> None:
    config = SequenceParallelConfig(
        batch_size=2,
        seq_len=4,
        hidden_size=4,
        ffn_hidden_size=8,
        num_layers=2,
        dtype=torch.float32,
    )
    torch.manual_seed(42)
    up_proj, down_proj, norms = build_layers(config, 2, torch.device("cpu"))
    input_tensor = torch.randn(
        config.batch_size,
        config.seq_len // 2,
        config.hidden_size,
        dtype=config.dtype,
    )
    baseline = _build_cpu_surrogate(
        BaselineSequenceParallelMultigpuBenchmark,
        config=config,
        input_tensor=input_tensor,
        up_proj=up_proj,
        down_proj=down_proj,
        norms=norms,
    )
    optimized = _build_cpu_surrogate(
        OptimizedSequenceParallelMultigpuBenchmark,
        config=config,
        input_tensor=input_tensor,
        up_proj=up_proj,
        down_proj=down_proj,
        norms=norms,
    )

    baseline.benchmark_fn()
    optimized.benchmark_fn()
    torch.testing.assert_close(baseline._output, optimized._output, rtol=0.0, atol=0.0)

    baseline.capture_verification_payload()
    optimized.capture_verification_payload()
    baseline_signature = baseline.get_input_signature()
    optimized_signature = optimized.get_input_signature()

    assert baseline_signature.collective_type == "all_reduce"
    assert baseline_signature.to_dict() == optimized_signature.to_dict()
