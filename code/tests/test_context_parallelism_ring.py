from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ch13.context_parallelism import RingAttention


def _dense_causal_reference(
    module: RingAttention,
    local_x: torch.Tensor,
    global_x: torch.Tensor,
    *,
    rank: int,
) -> torch.Tensor:
    batch, seq_shard, _ = local_x.shape
    global_seq = global_x.size(1)
    q = module.q_proj(local_x).view(batch, seq_shard, module.num_heads, module.head_dim)
    k = module.k_proj(global_x).view(batch, global_seq, module.num_heads, module.head_dim)
    v = module.v_proj(global_x).view(batch, global_seq, module.num_heads, module.head_dim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) * module.scale
    query_positions = torch.arange(
        rank * seq_shard,
        (rank + 1) * seq_shard,
    ).view(1, 1, seq_shard, 1)
    key_positions = torch.arange(global_seq).view(1, 1, 1, global_seq)
    scores.masked_fill_(key_positions > query_positions, float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention, v)
    output = output.transpose(1, 2).contiguous().view(batch, seq_shard, module.hidden_size)
    return module.o_proj(output)


def _gloo_ring_worker(rank: int, world_size: int, init_method: str, result_dir: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(2026)
        hidden_size = 8
        num_heads = 2
        seq_shard = 3
        module = RingAttention(
            hidden_size,
            num_heads,
            process_group=dist.group.WORLD,
            rank=rank,
            world_size=world_size,
        ).eval()
        global_x = torch.linspace(
            -1.5,
            2.0,
            steps=world_size * seq_shard * hidden_size,
            dtype=torch.float32,
        ).view(1, world_size * seq_shard, hidden_size)
        local_x = global_x[:, rank * seq_shard : (rank + 1) * seq_shard]
        with torch.no_grad():
            expected = _dense_causal_reference(module, local_x, global_x, rank=rank)
            actual = module(local_x, causal=True)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert torch.isfinite(actual).all()
        (Path(result_dir) / f"rank-{rank}.passed").write_text("passed\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="the real CPU ring test requires the Gloo backend",
)
def test_two_rank_ring_matches_dense_causal_attention(tmp_path: Path) -> None:
    world_size = 2
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    mp.spawn(
        _gloo_ring_worker,
        args=(world_size, (tmp_path / "rendezvous").as_uri(), str(result_dir)),
        nprocs=world_size,
        join=True,
        daemon=False,
    )
    assert sorted(path.name for path in result_dir.iterdir()) == [
        "rank-0.passed",
        "rank-1.passed",
    ]


def test_ring_uses_owned_send_buffers_and_batched_p2p() -> None:
    source = (Path(__file__).resolve().parents[1] / "ch13/context_parallelism.py").read_text(
        encoding="utf-8"
    )
    init = source.split("def _init_distributed", 1)[1].split("class RingAttention", 1)[0]
    ring = source.split("def _ring_pass", 1)[1].split("def forward", 1)[0]

    assert init.index("torch.cuda.set_device(device)") < init.index("dist.init_process_group(")
    assert 'dist.init_process_group(backend="nccl", device_id=device)' in init
    assert "k_current = k_local.contiguous()" in ring
    assert "v_current = v_local.contiguous()" in ring
    assert "dist.batch_isend_irecv(ops)" in ring
    assert "dist.isend(k_current.contiguous()" not in ring
