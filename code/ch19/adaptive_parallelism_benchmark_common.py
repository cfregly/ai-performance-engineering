"""Shared workload and logic for Chapter 19 adaptive parallelism benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from ch19.adaptive_parallelism_strategy import (
    ParallelismStrategy,
    choose_worker_pool,
)


STRATEGY_TO_ID = {
    ParallelismStrategy.TENSOR: 0,
    ParallelismStrategy.PIPELINE: 1,
    ParallelismStrategy.HYBRID: 2,
    ParallelismStrategy.DATA: 3,
}


@dataclass(frozen=True)
class AdaptiveParallelismBenchmarkConfig:
    num_requests: int = 16384


def build_workload(
    cfg: AdaptiveParallelismBenchmarkConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Create a deterministic workload that covers every routing branch."""
    slots = torch.arange(cfg.num_requests, device=device, dtype=torch.int64) % 4
    slot1 = slots == 1
    slot2 = slots == 2
    slot3 = slots == 3

    seq_len = torch.full((cfg.num_requests,), 512, device=device, dtype=torch.int64)
    seq_len.masked_fill_(slot1, 8192)
    seq_len.masked_fill_(slot2, 2048)

    batch_size = torch.full((cfg.num_requests,), 4, device=device, dtype=torch.int64)
    batch_size.masked_fill_(slot3, 8)

    concurrent_reqs = torch.full((cfg.num_requests,), 8, device=device, dtype=torch.int64)
    concurrent_reqs.masked_fill_(slot3, 48)

    prefill_tokens = seq_len.clone()
    prefill_tokens.masked_fill_(slot3, 128)

    decode_tokens = torch.full((cfg.num_requests,), 512, device=device, dtype=torch.int64)
    decode_tokens.masked_fill_(slot1, 1024)
    decode_tokens.masked_fill_(slot3, 128)

    gpu_mem_util = torch.full((cfg.num_requests,), 0.40, device=device, dtype=torch.float32)
    gpu_mem_util.masked_fill_(slot2, 0.88)
    gpu_mem_util.masked_fill_(slot1, 0.50)

    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "gpu_mem_util": gpu_mem_util,
        "concurrent_reqs": concurrent_reqs,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
    }


def classify_baseline(workload: Dict[str, torch.Tensor], *, device: torch.device) -> torch.Tensor:
    """Reference implementation using the chapter's existing Python helper.

    Materialize routing features to CPU once, then run per-request ``choose_worker_pool``
    in Python. Calling ``.item()`` on CUDA tensors inside the loop would force a device
    sync per scalar read (~6× per request), which dominates timing and dwarfs the
    actual routing logic—this path keeps the same semantics without that artifact.
    """
    feature_rows = torch.stack(
        (
            workload["seq_len"].to(dtype=torch.float64),
            workload["gpu_mem_util"].to(dtype=torch.float64),
            workload["concurrent_reqs"].to(dtype=torch.float64),
            workload["batch_size"].to(dtype=torch.float64),
            workload["prefill_tokens"].to(dtype=torch.float64),
            workload["decode_tokens"].to(dtype=torch.float64),
        ),
        dim=1,
    ).detach().cpu().tolist()

    strategy_ids: list[int] = []
    for seq_len, gpu_mem_util, concurrent_reqs, batch_size, prefill_tokens, decode_tokens in feature_rows:
        config = choose_worker_pool(
            seq_len=int(seq_len),
            gpu_mem_util=float(gpu_mem_util),
            concurrent_reqs=int(concurrent_reqs),
            batch_size=int(batch_size),
            prefill_tokens=int(prefill_tokens),
            decode_tokens=int(decode_tokens),
        )
        strategy_ids.append(STRATEGY_TO_ID[config.strategy])

    return torch.tensor(strategy_ids, device=device, dtype=torch.int64)


def classify_vectorized(workload: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Vectorized implementation of the same routing rules."""
    seq_len = workload["seq_len"]
    gpu_mem_util = workload["gpu_mem_util"]
    concurrent_reqs = workload["concurrent_reqs"]
    prefill_tokens = workload["prefill_tokens"]
    decode_tokens = workload["decode_tokens"]

    result = torch.full_like(seq_len, STRATEGY_TO_ID[ParallelismStrategy.TENSOR])

    steady_decode = (decode_tokens > 0) & (decode_tokens <= 256)
    data_mask = (concurrent_reqs > 32) & steady_decode

    long_prefill = (prefill_tokens > 0) & (prefill_tokens >= decode_tokens * 2)
    heavy_context = (seq_len > 1024) | (gpu_mem_util > 0.85) | long_prefill
    pipeline_mask = heavy_context & ((seq_len > 4096) | (gpu_mem_util > 0.92)) & ~data_mask
    hybrid_mask = heavy_context & ~pipeline_mask & ~data_mask

    result[hybrid_mask] = STRATEGY_TO_ID[ParallelismStrategy.HYBRID]
    result[pipeline_mask] = STRATEGY_TO_ID[ParallelismStrategy.PIPELINE]
    result[data_mask] = STRATEGY_TO_ID[ParallelismStrategy.DATA]
    return result


def classify_vectorized_out(
    workload: Dict[str, torch.Tensor],
    *,
    result: torch.Tensor,
    steady_decode: torch.Tensor,
    data_mask: torch.Tensor,
    long_prefill: torch.Tensor,
    heavy_context: torch.Tensor,
    pipeline_mask: torch.Tensor,
    hybrid_mask: torch.Tensor,
    doubled_decode_tokens: torch.Tensor,
) -> torch.Tensor:
    """Vectorized routing using caller-owned output and mask buffers."""
    seq_len = workload["seq_len"]
    gpu_mem_util = workload["gpu_mem_util"]
    concurrent_reqs = workload["concurrent_reqs"]
    prefill_tokens = workload["prefill_tokens"]
    decode_tokens = workload["decode_tokens"]

    result.fill_(STRATEGY_TO_ID[ParallelismStrategy.TENSOR])

    torch.gt(decode_tokens, 0, out=steady_decode)
    torch.le(decode_tokens, 256, out=data_mask)
    torch.logical_and(steady_decode, data_mask, out=steady_decode)
    torch.gt(concurrent_reqs, 32, out=data_mask)
    torch.logical_and(data_mask, steady_decode, out=data_mask)

    torch.gt(prefill_tokens, 0, out=long_prefill)
    torch.mul(decode_tokens, 2, out=doubled_decode_tokens)
    torch.ge(prefill_tokens, doubled_decode_tokens, out=hybrid_mask)
    torch.logical_and(long_prefill, hybrid_mask, out=long_prefill)

    torch.gt(seq_len, 1024, out=heavy_context)
    torch.gt(gpu_mem_util, 0.85, out=hybrid_mask)
    torch.logical_or(heavy_context, hybrid_mask, out=heavy_context)
    torch.logical_or(heavy_context, long_prefill, out=heavy_context)

    torch.gt(seq_len, 4096, out=pipeline_mask)
    torch.gt(gpu_mem_util, 0.92, out=hybrid_mask)
    torch.logical_or(pipeline_mask, hybrid_mask, out=pipeline_mask)
    torch.logical_and(heavy_context, pipeline_mask, out=pipeline_mask)
    torch.logical_not(data_mask, out=hybrid_mask)
    torch.logical_and(pipeline_mask, hybrid_mask, out=pipeline_mask)

    torch.logical_not(pipeline_mask, out=hybrid_mask)
    torch.logical_and(hybrid_mask, heavy_context, out=hybrid_mask)
    torch.logical_not(data_mask, out=steady_decode)
    torch.logical_and(hybrid_mask, steady_decode, out=hybrid_mask)

    result[hybrid_mask] = STRATEGY_TO_ID[ParallelismStrategy.HYBRID]
    result[pipeline_mask] = STRATEGY_TO_ID[ParallelismStrategy.PIPELINE]
    result[data_mask] = STRATEGY_TO_ID[ParallelismStrategy.DATA]
    return result
