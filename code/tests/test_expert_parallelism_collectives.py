from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ch15.expert_parallelism import DistributedContext, Top2MoE


def _reference_forward(model: Top2MoE, tokens: torch.Tensor) -> torch.Tensor:
    batch, seq, hidden = tokens.shape
    flat_tokens = tokens.reshape(batch * seq, hidden)
    logits = model.gate(tokens)
    top2_logits, top2_idx = torch.topk(logits, k=2, dim=-1)
    top2_weights = torch.exp(top2_logits - torch.logsumexp(logits, dim=-1, keepdim=True))
    flat_idx = top2_idx.reshape(batch * seq, 2)
    flat_weights = top2_weights.reshape(batch * seq, 2)
    expected = torch.zeros_like(flat_tokens)
    for position in range(flat_tokens.size(0)):
        token = flat_tokens[position : position + 1]
        for slot in range(2):
            expert_id = int(flat_idx[position, slot])
            expected[position] += (
                model.experts[expert_id](token)[0] * flat_weights[position, slot]
            )
    return expected.reshape(batch, seq, hidden)


def _gloo_worker(rank: int, world_size: int, init_method: str, result_dir: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(1234)
        model = Top2MoE(hidden_dim=2, num_experts=4, capacity_factor=4.0).eval()
        with torch.no_grad():
            model.gate.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [-1.0, 0.0],
                        [0.0, -1.0],
                    ]
                )
            )
            rank_tokens = (
                torch.tensor([[[3.0, 1.0], [-3.0, 1.0], [1.0, -3.0]]])
                if rank == 0
                else torch.tensor([[[-3.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]])
            )
            expected = _reference_forward(model, rank_tokens)
            actual = model.forward_distributed(
                rank_tokens,
                ctx=DistributedContext(
                    rank=rank,
                    world_size=world_size,
                    local_rank=rank,
                    device=torch.device("cpu"),
                ),
            )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        (Path(result_dir) / f"rank-{rank}.passed").write_text("passed\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="the real CPU distributed path requires the Gloo backend",
)
def test_expert_parallelism_uses_supported_split_keywords_on_real_collectives(
    tmp_path: Path,
) -> None:
    world_size = 2
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    mp.spawn(
        _gloo_worker,
        args=(world_size, (tmp_path / "rendezvous").as_uri(), str(result_dir)),
        nprocs=world_size,
        join=True,
        daemon=False,
    )
    assert sorted(path.name for path in result_dir.iterdir()) == [
        "rank-0.passed",
        "rank-1.passed",
    ]
