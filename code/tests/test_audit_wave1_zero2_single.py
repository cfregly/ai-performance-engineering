"""Real one-rank DDP/AdamW regression for the single-rank ZeRO entrypoint.

Gloo exercises the production single-rank hook branch and optimizer, not CUDA.
The shared multi-rank hook has a separate actual-NCCL acceptance gate.
"""

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(autouse=True)
def isolated_rng():
    with torch.random.fork_rng(devices=[]):
        yield


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_single_rank_zero2_updates_match_dense_adamw(tmp_path):
    from labs.train_distributed.optimized_zero2 import _build_training_components

    assert not dist.is_initialized(), "Test requires its own isolated process group"
    dist.init_process_group(
        "gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1,
        timeout=timedelta(seconds=15),
    )
    try:
        torch.manual_seed(731)
        model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.GELU(), torch.nn.Linear(5, 2))
        reference = copy.deepcopy(model)
        ddp, optimizer = _build_training_components(model, 0.01)
        dense = torch.optim.AdamW(reference.parameters(), lr=0.01, betas=(0.9, 0.95), weight_decay=0.05, fused=True)
        inputs, target = torch.randn(4, 3), torch.randn(4, 2)
        for _ in range(3):
            before = [param.detach().clone() for param in model.parameters()]
            optimizer.zero_grad(set_to_none=True)
            dense.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(ddp(inputs), target).backward()
            torch.nn.functional.mse_loss(reference(inputs), target).backward()
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            optimizer.step()
            dense.step()
            assert sum(float((param.detach() - old).abs().sum()) for param, old in zip(model.parameters(), before)) > 0
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert len(optimizer.optim.state) == len(list(model.parameters()))
        assert all(int(state["step"]) == 3 for state in optimizer.optim.state.values())
    finally:
        dist.destroy_process_group()
