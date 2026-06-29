"""Smoke tests for shared Chapter 15 benchmark wrapper factories."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from ch15.moe_routing_benchmark_common import (
    active_expert_ids_for_static_route,
    pseudo_uniform_expert_ids,
    topology_aware_expert_ids,
)
from core.hot_path_checks import (
    check_benchmark_fn_antipatterns,
    check_benchmark_fn_sync_calls,
)
from core.harness.benchmark_harness import BaseBenchmark


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "relative_path",
    [
        "ch15/baseline_guided_decoding.py",
        "ch15/optimized_guided_decoding.py",
        "ch15/baseline_speculative_decoding.py",
        "ch15/optimized_speculative_decoding.py",
        "ch15/baseline_medusa_eagle_speculative.py",
        "ch15/optimized_medusa_eagle_speculative_medusa.py",
        "ch15/optimized_medusa_eagle_speculative_eagle.py",
        "ch15/baseline_moe_comm_exchange.py",
        "ch15/optimized_moe_comm_exchange_overlap.py",
        "ch15/optimized_moe_comm_exchange_hierarchical.py",
    ],
)
def test_shared_ch15_wrappers_attach_metadata(relative_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    module = _load_module(module_path)

    bench = module.get_benchmark()

    assert isinstance(bench, BaseBenchmark)
    assert getattr(bench, "_module_file_override", None) == str(module_path)
    assert getattr(bench, "_factory_name_override", None) == "get_benchmark"


@pytest.mark.parametrize(
    "relative_path",
    [
        "ch15/baseline_guided_decoding.py",
        "ch15/optimized_guided_decoding.py",
        "ch15/baseline_speculative_decoding.py",
        "ch15/optimized_speculative_decoding.py",
        "ch15/baseline_medusa_eagle_speculative.py",
        "ch15/optimized_medusa_eagle_speculative_medusa.py",
        "ch15/optimized_medusa_eagle_speculative_eagle.py",
        "ch15/baseline_moe_comm_exchange.py",
        "ch15/optimized_moe_comm_exchange_overlap.py",
        "ch15/optimized_moe_comm_exchange_hierarchical.py",
    ],
)
def test_shared_ch15_benchmark_fns_stay_hot_path_clean(relative_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / relative_path)
    bench = module.get_benchmark()

    sync_ok, sync_warnings = check_benchmark_fn_sync_calls(bench.benchmark_fn)
    antipattern_ok, antipattern_warnings = check_benchmark_fn_antipatterns(
        bench.benchmark_fn,
        allowed_codes=getattr(bench, "allowed_benchmark_fn_antipatterns", ()),
    )

    assert sync_ok, sync_warnings
    assert antipattern_ok, antipattern_warnings


@pytest.mark.parametrize(
    "relative_path,expected_goal",
    [
        ("ch15/baseline_medusa_eagle_speculative.py", "throughput"),
        ("ch15/optimized_medusa_eagle_speculative_medusa.py", "throughput"),
        ("ch15/optimized_medusa_eagle_speculative_eagle.py", "throughput"),
        ("ch15/baseline_moe_comm_exchange.py", "speed"),
        ("ch15/optimized_moe_comm_exchange_overlap.py", "speed"),
        ("ch15/optimized_moe_comm_exchange_hierarchical.py", "speed"),
    ],
)
def test_shared_ch15_family_goals_match_benchmark_contract(relative_path: str, expected_goal: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / relative_path)
    bench = module.get_benchmark()

    assert bench.get_optimization_goal() == expected_goal


def test_static_moe_route_active_experts_match_tensor_routes() -> None:
    token_ids = torch.arange(37, dtype=torch.int64)
    uniform_ids = pseudo_uniform_expert_ids(token_ids, num_experts=16)
    topology_ids = topology_aware_expert_ids(token_ids, local_experts=6)

    assert active_expert_ids_for_static_route(
        route_mode="uniform",
        num_tokens=token_ids.numel(),
        num_experts=16,
        local_experts=6,
    ) == sorted(int(expert_id) for expert_id in torch.unique(uniform_ids))
    assert active_expert_ids_for_static_route(
        route_mode="topology_aware",
        num_tokens=token_ids.numel(),
        num_experts=16,
        local_experts=6,
    ) == sorted(int(expert_id) for expert_id in torch.unique(topology_ids))
