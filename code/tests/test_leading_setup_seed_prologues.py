"""Regression controls for benchmarks that consume harness-owned RNG state."""

from __future__ import annotations

import random

import torch

from ch09.baseline_memory_bound import BaselineMemoryBoundBenchmark
from ch17.baseline_dynamic_routing import BaselineDynamicRoutingBenchmark
from core.harness.benchmark_harness import BenchmarkConfig
from tests.protection_test_utils import preserve_rng_state


def _memory_bound_input(seed: int) -> tuple[torch.Tensor, int]:
    random.seed(seed)
    torch.manual_seed(seed)
    benchmark = BaselineMemoryBoundBenchmark()
    benchmark.device = torch.device("cpu")
    benchmark.N = 64
    try:
        benchmark.setup()
        assert benchmark.tensor is not None
        return benchmark.tensor.detach().clone(), int(torch.initial_seed())
    finally:
        benchmark.teardown()


def _routing_inputs(seed: int) -> tuple[list[int], torch.Tensor, int]:
    random.seed(seed)
    torch.manual_seed(seed)
    benchmark = BaselineDynamicRoutingBenchmark()
    benchmark.batch_size = 8
    try:
        benchmark.setup()
        assert benchmark._queue_length_table is not None
        return (
            list(benchmark._cached_prompt_lengths),
            benchmark._queue_length_table.detach().clone(),
            int(torch.initial_seed()),
        )
    finally:
        benchmark.teardown()


def test_tensor_setup_preserves_active_seed_and_changes_fresh_inputs() -> None:
    with preserve_rng_state():
        first_42, observed_42 = _memory_bound_input(42)
        second_42, repeated_42 = _memory_bound_input(42)
        fresh_1042, observed_1042 = _memory_bound_input(1042)

    assert BenchmarkConfig().seed == 42
    assert observed_42 == repeated_42 == 42
    assert observed_1042 == 1042
    assert torch.equal(first_42, second_42)
    assert not torch.equal(first_42, fresh_1042)


def test_python_and_tensor_setup_inputs_follow_the_active_seed() -> None:
    with preserve_rng_state():
        prompts_42, queues_42, observed_42 = _routing_inputs(42)
        repeated_prompts, repeated_queues, repeated_seed = _routing_inputs(42)
        prompts_1042, queues_1042, observed_1042 = _routing_inputs(1042)

    assert observed_42 == repeated_seed == 42
    assert observed_1042 == 1042
    assert prompts_42 == repeated_prompts
    assert torch.equal(queues_42, repeated_queues)
    assert prompts_42 != prompts_1042
    assert not torch.equal(queues_42, queues_1042)
