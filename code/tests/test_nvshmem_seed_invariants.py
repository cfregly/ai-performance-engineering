from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
import torch

_MODULES = (
    "ch04.nvshmem_pipeline_parallel_multigpu",
    "ch04.nvshmem_training_patterns",
    "ch04.nvshmem_vs_nccl_benchmark",
    "ch04.symmetric_memory_example",
)


@pytest.mark.parametrize("module_name", _MODULES)
def test_rank_local_generators_preserve_global_cpu_seed_and_state(module_name: str) -> None:
    module = importlib.import_module(module_name)
    torch.manual_seed(987_654)
    expected_seed = torch.initial_seed()
    expected_state = torch.random.get_rng_state().clone()

    first = module._make_rank_generator("cpu", rank=1, base_seed=42)
    first_values = torch.randn(32, generator=first)
    repeated = module._make_rank_generator("cpu", rank=1, base_seed=42)
    other_rank = module._make_rank_generator("cpu", rank=0, base_seed=42)

    assert first.initial_seed() == 43
    torch.testing.assert_close(first_values, torch.randn(32, generator=repeated))
    assert not torch.equal(first_values, torch.randn(32, generator=other_rank))
    assert torch.initial_seed() == expected_seed
    assert torch.equal(torch.random.get_rng_state(), expected_state)


@pytest.mark.parametrize("module_name", _MODULES)
def test_nvshmem_workers_do_not_reseed_global_torch_generators(module_name: str) -> None:
    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    forbidden = {
        "torch.manual_seed",
        "torch.cuda.manual_seed",
        "torch.cuda.manual_seed_all",
        "torch.random.manual_seed",
    }

    calls = {
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert calls.isdisjoint(forbidden)

