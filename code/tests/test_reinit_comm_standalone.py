from __future__ import annotations

import inspect

import pytest
import torch
import torch.distributed as dist

from ch04.baseline_reinit_comm import get_benchmark as get_baseline_benchmark
from ch04.optimized_reinit_comm import get_benchmark as get_optimized_benchmark
from ch04.reinit_comm_common import (
    ReinitCommLaunchContext,
    initialize_reinit_process_group,
    resolve_reinit_comm_launch,
)


def _destroy_default_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_standalone_hash_store_recreates_real_one_rank_process_groups() -> None:
    """Exercise actual Store, process-group, collective, and destroy lifecycles."""
    context = ReinitCommLaunchContext(
        rank=0,
        world_size=1,
        local_rank=0,
        standalone=True,
    )
    stores: list[dist.Store] = []
    try:
        for value in (3.0, 7.0):
            store = initialize_reinit_process_group(
                context,
                backend="gloo",
                device_id=None,
            )
            assert isinstance(store, dist.HashStore)
            stores.append(store)
            scalar = torch.tensor([[value]], dtype=torch.float32)
            dist.all_reduce(scalar)
            torch.testing.assert_close(scalar, torch.tensor([[value]]))
            assert dist.get_rank() == 0
            assert dist.get_world_size() == 1
            _destroy_default_group()

        assert stores[0] is not stores[1]
    finally:
        _destroy_default_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_standalone_hash_store_supports_reusing_one_real_group() -> None:
    context = ReinitCommLaunchContext(0, 1, 0, standalone=True)
    try:
        store = initialize_reinit_process_group(
            context,
            backend="gloo",
            device_id=None,
        )
        assert isinstance(store, dist.HashStore)
        process_group = dist.group.WORLD
        for value in (2.0, 5.0):
            scalar = torch.tensor([[value]], dtype=torch.float32)
            dist.all_reduce(scalar)
            torch.testing.assert_close(scalar, torch.tensor([[value]]))
            assert dist.group.WORLD is process_group
    finally:
        _destroy_default_group()


def test_reinit_pair_declares_one_rank_local_communicator_overhead() -> None:
    baseline = get_baseline_benchmark()
    optimized = get_optimized_benchmark()

    for benchmark in (baseline, optimized):
        config = benchmark.get_config()
        assert benchmark.multi_gpu_required is False
        assert config.multi_gpu_required is False
        assert config.iterations == 5
        assert config.warmup == 5
        assert benchmark._workload.bytes_per_iteration == 4.0
        assert "local communicator lifecycle overhead" in inspect.getmodule(benchmark).__doc__
        assert "does not measure network communication performance" in inspect.getmodule(
            benchmark
        ).__doc__

    baseline_hot_path = inspect.getsource(baseline.benchmark_fn)
    optimized_setup = inspect.getsource(optimized.setup)
    optimized_hot_path = inspect.getsource(optimized.benchmark_fn)
    assert "initialize_reinit_process_group(" in baseline_hot_path
    assert "dist.destroy_process_group()" in baseline_hot_path
    assert "initialize_reinit_process_group(" in optimized_setup
    assert "self._launch_context.standalone and dist.is_initialized()" in optimized_setup
    assert "initialize_reinit_process_group(" not in optimized_hot_path
    assert "dist.destroy_process_group()" not in optimized_hot_path


def test_launch_resolution_keeps_strict_torchrun_branch_and_never_seeds_env() -> None:
    source = inspect.getsource(resolve_reinit_comm_launch)
    assert 'if "RANK" in os.environ or "WORLD_SIZE" in os.environ:' in source
    assert "setup_single_gpu_env(" in source
    assert "min_world_size=1" in source
    assert "os.environ.setdefault" not in source
    assert "os.environ[" not in source


def test_invalid_standalone_identity_fails_before_process_group_creation() -> None:
    context = ReinitCommLaunchContext(0, 2, 0, standalone=True)
    with pytest.raises(ValueError, match="rank 0 of world size 1"):
        initialize_reinit_process_group(
            context,
            backend="gloo",
            device_id=None,
        )
    assert not dist.is_initialized()
