"""Seed-ownership controls for the two formerly excluded lab factories."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

import labs.real_world_models.gpt4_architecture_optimization as gpt4_module
from core.harness.benchmark_harness import BenchmarkConfig
from labs.dynamic_router.topology_probe import TopologyProbeBenchmark
from labs.real_world_models.gpt4_architecture_optimization import (
    GPT4ArchitectureOptimizationBenchmark,
)


@pytest.mark.parametrize(
    "setup",
    (TopologyProbeBenchmark.setup, GPT4ArchitectureOptimizationBenchmark.setup),
)
def test_lab_setup_does_not_replace_harness_seed(setup) -> None:
    source = inspect.getsource(setup)

    assert "torch.manual_seed" not in source
    assert "torch.cuda.manual_seed_all" not in source


def _gpt4_setup_snapshot(seed: int, cuda_tag: SimpleNamespace) -> tuple[torch.Tensor, torch.Tensor, int]:
    benchmark = GPT4ArchitectureOptimizationBenchmark()
    benchmark.device = cuda_tag
    torch.manual_seed(seed)
    try:
        benchmark.setup()
        assert benchmark.model_wrapper is not None
        parameter = next(benchmark.model_wrapper.layers.parameters())
        return (
            parameter.detach().flatten()[:64].clone(),
            benchmark.model_wrapper.input.detach().flatten()[:64].clone(),
            int(torch.initial_seed()),
        )
    finally:
        benchmark.teardown()


def test_lab_setups_preserve_default_and_fresh_verification_seeds(monkeypatch) -> None:
    cuda_tag = SimpleNamespace(type="cuda")
    real_empty = torch.empty

    def empty_on_cpu_for_tagged_device(*args, **kwargs):
        if kwargs.get("device") is cuda_tag:
            kwargs = {**kwargs, "device": torch.device("cpu")}
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(gpt4_module.torch, "empty", empty_on_cpu_for_tagged_device)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        topology_default = TopologyProbeBenchmark()
        topology_default.setup()
        observed_topology_default = int(torch.initial_seed())

        torch.manual_seed(1042)
        topology_fresh = TopologyProbeBenchmark()
        topology_fresh.setup()
        observed_topology_fresh = int(torch.initial_seed())

        default = _gpt4_setup_snapshot(42, cuda_tag)
        repeated = _gpt4_setup_snapshot(42, cuda_tag)
        fresh = _gpt4_setup_snapshot(1042, cuda_tag)

    assert BenchmarkConfig().seed == 42
    assert observed_topology_default == 42
    assert observed_topology_fresh == 1042
    assert default[2] == repeated[2] == 42
    assert fresh[2] == 1042
    torch.testing.assert_close(default[0], repeated[0], rtol=0, atol=0)
    torch.testing.assert_close(default[1], repeated[1], rtol=0, atol=0)
    assert not torch.equal(default[0], fresh[0])
    assert not torch.equal(default[1], fresh[1])
