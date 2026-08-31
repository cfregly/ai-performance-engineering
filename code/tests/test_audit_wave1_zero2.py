"""Real distributed optimizer checks for W1-004, without GPU emulation.

The Gloo case verifies optimizer updates/state with real two-rank CPU DDP.
The separate NCCL case exercises the production reduce-scatter/all-gather hook;
it requires two real GPUs and must not be credited when skipped.
"""

from __future__ import annotations

import copy
import json
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def _train_worker(rank: int, rendezvous: str, backend: str, output_dir: str) -> None:
    from labs.train_distributed.optimized_zero2_multigpu import (
        _build_optimizer,
        _reduce_scatter_allgather_hook,
    )
    from labs.train_distributed.training_utils.memory import get_optimizer_memory

    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend, init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(20260830)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3)
        ).to(device)
        reference = copy.deepcopy(model)
        ddp = DistributedDataParallel(
            model, device_ids=[rank] if backend == "nccl" else None,
            static_graph=True, gradient_as_bucket_view=True,
        )
        if backend == "nccl":
            ddp.register_comm_hook(dist.group.WORLD, _reduce_scatter_allgather_hook)
        optimizer = _build_optimizer(ddp.parameters(), 0.01)
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=0.01, betas=(0.9, 0.95), weight_decay=0.05,
            fused=True,
        )
        rng = torch.Generator().manual_seed(91)
        deltas = []
        for _step in range(3):
            before = [p.detach().clone() for p in model.parameters()]
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            for _micro in range(2):
                all_x = torch.randn(4, 4, generator=rng).to(device)
                all_y = torch.randn(4, 3, generator=rng).to(device)
                rows = slice(rank * 2, (rank + 1) * 2)
                loss = torch.nn.functional.mse_loss(ddp(all_x[rows]), all_y[rows]) / 2
                reference_loss = torch.nn.functional.mse_loss(reference(all_x), all_y) / 2
                loss.backward()
                reference_loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            optimizer.step()
            reference_optimizer.step()
            delta = sum(float((p.detach() - old).abs().sum()) for p, old in zip(model.parameters(), before, strict=True))
            assert delta > 0, "Explicit ZeRO step did not update any parameters"
            deltas.append(delta)
            for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                replicas = [torch.empty_like(actual) for _ in range(2)]
                dist.all_gather(replicas, actual.detach())
                torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
            assert get_optimizer_memory(optimizer) > 0, "Local optimizer state was not reported"
        local_optimizer = optimizer.optim
        state_count = len(local_optimizer.state)
        assert 0 < state_count < len(list(model.parameters()))
        for state in local_optimizer.state.values():
            assert int(state["step"]) == 3
            assert state["exp_avg"].abs().sum() > 0
        Path(output_dir, f"{backend}-rank-{rank}.json").write_text(json.dumps({
            "backend": backend, "device": str(device), "rank": rank,
            "world_size": 2, "parameter_l1_deltas": deltas,
            "local_optimizer_state_entries": state_count,
            "optimizer_memory_mb": get_optimizer_memory(optimizer),
            "reference": "full-batch AdamW, three steps, two microbatches per step",
        }, indent=2))
    finally:
        dist.destroy_process_group()


def _run_distributed(tmp_path: Path, backend: str) -> None:
    context = torch.multiprocessing.spawn(
        _train_worker,
        args=((tmp_path / "rendezvous").as_uri(), backend, str(tmp_path)),
        nprocs=2, join=False,
    )
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail(f"Two-rank {backend} optimizer validation timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    receipts = [json.loads((tmp_path / f"{backend}-rank-{rank}.json").read_text()) for rank in range(2)]
    assert all(len(receipt["parameter_l1_deltas"]) == 3 for receipt in receipts)
    assert sum(receipt["local_optimizer_state_entries"] for receipt in receipts) == 4


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_zero2_explicit_steps_match_full_batch_reference_on_two_cpu_ranks(tmp_path):
    _run_distributed(tmp_path, "gloo")


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_nccl_available() or torch.cuda.device_count() < 2,
    reason="Production reduce-scatter hook requires NCCL and two real GPUs",
)
def test_zero2_production_hook_matches_reference_on_two_cuda_ranks(tmp_path):
    _run_distributed(tmp_path, "nccl")
