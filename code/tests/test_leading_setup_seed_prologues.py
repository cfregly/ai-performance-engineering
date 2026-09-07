"""Regression controls for benchmarks that consume harness-owned RNG state."""

from __future__ import annotations

import random
from typing import TypeAlias

import torch

from ch09.baseline_memory_bound import BaselineMemoryBoundBenchmark
from ch13.baseline_dataloader_default import (
    BaselineDataloaderDefaultBenchmark,
    SimpleModel,
    SyntheticDataset,
)
from ch17.baseline_dynamic_routing import BaselineDynamicRoutingBenchmark
from ch17.baseline_routing_static import (
    BaselineRoutingStaticBenchmark,
)
from ch17.baseline_routing_static import (
    LargeModel as BaselineStaticModel,
)
from ch17.optimized_routing_static import (
    LargeModel as OptimizedStaticModel,
)
from ch17.optimized_routing_static import (
    OptimizedRoutingStaticBenchmark,
)
from core.harness.benchmark_harness import BenchmarkConfig
from tests.protection_test_utils import preserve_rng_state

TensorTuple: TypeAlias = tuple[torch.Tensor, ...]


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


def _dataloader_artifacts(seed: int) -> tuple[TensorTuple, torch.Tensor, torch.Tensor, int]:
    random.seed(seed)
    torch.manual_seed(seed)
    benchmark = BaselineDataloaderDefaultBenchmark()
    benchmark.device = torch.device("cpu")
    benchmark.dataset_size = 8
    benchmark.batch_size = 4
    benchmark.feature_dim = 8
    benchmark.preprocess_steps = 1
    previous_threads = torch.get_num_threads()
    try:
        benchmark.setup()
        assert benchmark.model is not None
        assert benchmark.dataloader is not None
        dataset = benchmark.dataloader.dataset
        assert isinstance(dataset, SyntheticDataset)
        parameters = tuple(parameter.detach().clone() for parameter in benchmark.model.parameters())
        inputs = dataset.data.detach().clone()
        with torch.inference_mode():
            output = benchmark.model(inputs[: benchmark.batch_size]).detach().clone()
        return parameters, inputs, output, int(torch.initial_seed())
    finally:
        benchmark.teardown()
        torch.set_num_threads(previous_threads)


def _legacy_dataloader_artifacts(seed: int) -> tuple[TensorTuple, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    model = SimpleModel(input_dim=8)
    dataset = SyntheticDataset(num_samples=8, feature_dim=8, preprocess_steps=1)
    with torch.inference_mode():
        output = model(dataset.data[:4]).detach().clone()
    return (
        tuple(parameter.detach().clone() for parameter in model.parameters()),
        dataset.data.detach().clone(),
        output,
    )


def _static_routing_artifacts(
    benchmark_type: type[BaselineRoutingStaticBenchmark] | type[OptimizedRoutingStaticBenchmark],
    seed: int,
) -> tuple[TensorTuple, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    random.seed(seed)
    torch.manual_seed(seed)
    benchmark = benchmark_type()
    benchmark.device = torch.device("cpu")
    benchmark.batch_size = 4
    benchmark.hidden_dim = 8
    benchmark.num_layers = 2
    benchmark.requests_per_iteration = 3
    benchmark.num_routes = 4
    try:
        benchmark.setup()
        assert benchmark.model is not None
        assert benchmark.inputs is not None
        assert benchmark._verify_input is not None
        parameters = tuple(parameter.detach().clone() for parameter in benchmark.model.parameters())
        with torch.inference_mode():
            output = benchmark.model(benchmark._verify_input).detach().clone()
        return (
            parameters,
            benchmark.inputs.detach().clone(),
            benchmark._verify_input.detach().clone(),
            output,
            int(torch.initial_seed()),
        )
    finally:
        benchmark.teardown()


def _legacy_static_routing_artifacts(
    model_type: type[BaselineStaticModel] | type[OptimizedStaticModel],
) -> tuple[TensorTuple, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    model = model_type(hidden_dim=8, num_layers=2).eval()
    inputs = torch.randn(4, 8)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    verify_input = torch.randn(4, 8)
    with torch.inference_mode():
        output = model(verify_input).detach().clone()
    return (
        tuple(parameter.detach().clone() for parameter in model.parameters()),
        inputs,
        verify_input,
        output,
    )


def _assert_tensors_equal(actual: TensorTuple, expected: TensorTuple) -> None:
    assert len(actual) == len(expected)
    assert all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))


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


def test_post_check_setup_keeps_default_artifacts_and_honors_fresh_seed() -> None:
    with preserve_rng_state():
        expected_parameters, expected_inputs, expected_output = _legacy_dataloader_artifacts(42)
        parameters_42, inputs_42, output_42, observed_42 = _dataloader_artifacts(42)
        repeated_parameters, repeated_inputs, repeated_output, repeated_seed = (
            _dataloader_artifacts(42)
        )
        parameters_1042, inputs_1042, output_1042, observed_1042 = _dataloader_artifacts(1042)

    _assert_tensors_equal(parameters_42, expected_parameters)
    _assert_tensors_equal(parameters_42, repeated_parameters)
    assert torch.equal(inputs_42, expected_inputs)
    assert torch.equal(inputs_42, repeated_inputs)
    assert torch.equal(output_42, expected_output)
    assert torch.equal(output_42, repeated_output)
    assert observed_42 == repeated_seed == 42
    assert observed_1042 == 1042
    assert not torch.equal(parameters_42[0], parameters_1042[0])
    assert not torch.equal(inputs_42, inputs_1042)
    assert not torch.equal(output_42, output_1042)


def test_alignment_restart_uses_active_seed_and_preserves_default_artifacts() -> None:
    cases = (
        (BaselineRoutingStaticBenchmark, BaselineStaticModel),
        (OptimizedRoutingStaticBenchmark, OptimizedStaticModel),
    )
    with preserve_rng_state():
        for benchmark_type, model_type in cases:
            expected = _legacy_static_routing_artifacts(model_type)
            actual_42 = _static_routing_artifacts(benchmark_type, 42)
            repeated_42 = _static_routing_artifacts(benchmark_type, 42)
            fresh_1042 = _static_routing_artifacts(benchmark_type, 1042)

            _assert_tensors_equal(actual_42[0], expected[0])
            _assert_tensors_equal(actual_42[0], repeated_42[0])
            for actual, legacy, repeated in zip(actual_42[1:4], expected[1:], repeated_42[1:4], strict=True):
                assert torch.equal(actual, legacy)
                assert torch.equal(actual, repeated)
            assert actual_42[4] == repeated_42[4] == 42
            assert fresh_1042[4] == 1042
            assert not torch.equal(actual_42[0][0], fresh_1042[0][0])
            assert not torch.equal(actual_42[1], fresh_1042[1])
            assert not torch.equal(actual_42[2], fresh_1042[2])
            assert not torch.equal(actual_42[3], fresh_1042[3])
