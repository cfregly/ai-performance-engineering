"""Shared multi-GPU cache-aware disaggregated inference helpers."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist

from ch17.prefill_decode_disagg_multigpu_common import TinyPrefillDecode
from core.benchmark.metrics import compute_inference_metrics
from core.benchmark.verification import PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
    WorkloadMetadata,
)

METRICS_ENV_VAR = "AISP_CACHE_AWARE_DISAGG_METRICS_PATH"


class DecodeAffinityMode(str, Enum):
    ROUND_ROBIN = "round_robin"
    STICKY = "sticky"


@dataclass(frozen=True)
class CacheAwareDisaggMultiGPUConfig:
    hidden_size: int = 256
    num_layers: int = 2
    batch_size: int = 1
    requests_per_rank: int = 8
    context_window: int = 2048
    chunk_size: int = 256
    decode_tokens: int = 24
    warm_request_ratio: float = 0.75
    warm_prefix_ratio: float = 0.75
    prefill_ranks: Optional[int] = None
    dtype: torch.dtype = torch.bfloat16

    @property
    def num_chunks(self) -> int:
        return max(1, math.ceil(self.context_window / self.chunk_size))

    @property
    def tokens_per_request(self) -> int:
        return self.context_window + self.decode_tokens


@dataclass(frozen=True)
class DistributedRequestPlan:
    prefill_rank: int
    local_request_idx: int
    global_request_idx: int
    warm_chunks: int
    total_chunks: int

    @property
    def is_warm(self) -> bool:
        return self.warm_chunks > 0


def _split_prompt(prompt: torch.Tensor, chunk_size: int) -> Sequence[torch.Tensor]:
    return tuple(prompt.split(chunk_size, dim=1))


def _tensor_nbytes(tensor: torch.Tensor) -> float:
    return float(tensor.numel() * tensor.element_size())


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _empty_kv(cfg: CacheAwareDisaggMultiGPUConfig, device: torch.device) -> torch.Tensor:
    return torch.empty(
        cfg.batch_size,
        0,
        cfg.hidden_size,
        device=device,
        dtype=cfg.dtype,
    )


def _extend_cache_buffer(
    cfg: CacheAwareDisaggMultiGPUConfig,
    *,
    request_id: int,
    cache: torch.Tensor,
    chunk_kv: torch.Tensor,
    kv_buffers: Dict[int, torch.Tensor],
) -> torch.Tensor:
    current_kv_len = cache.size(1)
    next_kv_len = current_kv_len + chunk_kv.size(1)
    target_device = cache.device
    kv_buffer = kv_buffers.get(request_id)
    if (
        kv_buffer is None
        or kv_buffer.device != target_device
        or kv_buffer.dtype != chunk_kv.dtype
        or kv_buffer.size(1) < next_kv_len
    ):
        buffer_tokens = max(cfg.context_window, next_kv_len)
        kv_buffer = torch.empty(
            cfg.batch_size,
            buffer_tokens,
            cfg.hidden_size,
            device=target_device,
            dtype=chunk_kv.dtype,
        )
        kv_buffers[request_id] = kv_buffer

    if current_kv_len and cache.data_ptr() != kv_buffer.data_ptr():
        kv_buffer[:, :current_kv_len].copy_(cache)
    kv_buffer[:, current_kv_len:next_kv_len].copy_(chunk_kv, non_blocking=True)
    return kv_buffer[:, :next_kv_len]


def _world_size_hint() -> int:
    if not torch.cuda.is_available():
        return 2
    return max(int(torch.cuda.device_count()), 2)


def _resolve_runtime_world_size() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("SKIPPED: CUDA required for cache-aware disaggregated inference multi-GPU lab")
    world_size = int(torch.cuda.device_count())
    if world_size < 2:
        raise RuntimeError(f"SKIPPED: Requires >= 2 GPUs (found {world_size} GPU)")
    return world_size


def _hint_prefill_ranks(world_size: int, requested: Optional[int]) -> int:
    if requested is not None:
        return max(1, min(int(requested), max(world_size - 1, 1)))
    if world_size <= 2:
        return 1
    return world_size // 2


def _resolve_prefill_ranks(world_size: int, requested: Optional[int]) -> int:
    if requested is None:
        if world_size % 2 != 0:
            raise RuntimeError(
                "--prefill-ranks must be set when world_size is odd for cache-aware disaggregation"
            )
        return world_size // 2
    prefill_ranks = int(requested)
    if prefill_ranks < 1:
        raise RuntimeError(f"--prefill-ranks must be >= 1 (got {prefill_ranks})")
    if prefill_ranks >= world_size:
        raise RuntimeError(
            f"--prefill-ranks={prefill_ranks} must be < world_size={world_size}"
        )
    return prefill_ranks


def _emit_split_advice(prefill_ranks: int, decode_ranks: int) -> None:
    split_label = f"{prefill_ranks}P{decode_ranks}D"
    if prefill_ranks == 2 and decode_ranks == 1:
        print(f"rank0 cache-aware split {split_label}: recommended article-faithful 2P1D layout")
    elif prefill_ranks == decode_ranks:
        print(f"rank0 cache-aware split {split_label}: balanced split")
    else:
        print(f"rank0 cache-aware split {split_label}: custom split")


def _chunk_length(cfg: CacheAwareDisaggMultiGPUConfig, chunk_idx: int) -> int:
    remaining = max(cfg.context_window - (chunk_idx * cfg.chunk_size), 0)
    return min(cfg.chunk_size, remaining)


def _prefix_length(cfg: CacheAwareDisaggMultiGPUConfig, chunk_count: int) -> int:
    return min(cfg.context_window, max(0, chunk_count) * cfg.chunk_size)


def _home_decode_rank(global_request_idx: int, prefill_ranks: int, decode_ranks: int) -> int:
    return prefill_ranks + (global_request_idx % decode_ranks)


def _choose_decode_rank(
    plan: DistributedRequestPlan,
    stage_idx: int,
    *,
    affinity_mode: DecodeAffinityMode,
    prefill_ranks: int,
    decode_ranks: int,
) -> int:
    home_rank = _home_decode_rank(plan.global_request_idx, prefill_ranks, decode_ranks)
    if affinity_mode == DecodeAffinityMode.STICKY:
        return home_rank
    return prefill_ranks + ((plan.global_request_idx + stage_idx) % decode_ranks)


def _build_request_plans(
    cfg: CacheAwareDisaggMultiGPUConfig,
    *,
    prefill_ranks: int,
) -> List[DistributedRequestPlan]:
    warm_requests = int(round(cfg.requests_per_rank * cfg.warm_request_ratio))
    warm_chunks = min(
        cfg.num_chunks - 1,
        max(0, int(round(cfg.num_chunks * cfg.warm_prefix_ratio))),
    )
    plans: List[DistributedRequestPlan] = []
    for prefill_rank in range(prefill_ranks):
        for local_request_idx in range(cfg.requests_per_rank):
            plans.append(
                DistributedRequestPlan(
                    prefill_rank=prefill_rank,
                    local_request_idx=local_request_idx,
                    global_request_idx=(prefill_rank * cfg.requests_per_rank) + local_request_idx,
                    warm_chunks=warm_chunks if local_request_idx < warm_requests else 0,
                    total_chunks=cfg.num_chunks,
                )
            )
    return plans


def _build_reference_state(cfg: CacheAwareDisaggMultiGPUConfig) -> Dict[str, torch.Tensor]:
    reference = TinyPrefillDecode(
        cfg.hidden_size,
        cfg.num_layers,
        torch.device("cpu"),
        torch.float32,
    ).eval()
    return {
        name: tensor.detach().cpu()
        for name, tensor in reference.state_dict().items()
    }


def _shared_args_to_cli(cfg: CacheAwareDisaggMultiGPUConfig) -> List[str]:
    args = [
        "--hidden-size",
        str(cfg.hidden_size),
        "--num-layers",
        str(cfg.num_layers),
        "--batch-size",
        str(cfg.batch_size),
        "--requests-per-rank",
        str(cfg.requests_per_rank),
        "--context-window",
        str(cfg.context_window),
        "--chunk-size",
        str(cfg.chunk_size),
        "--decode-tokens",
        str(cfg.decode_tokens),
        "--warm-request-ratio",
        str(cfg.warm_request_ratio),
        "--warm-prefix-ratio",
        str(cfg.warm_prefix_ratio),
    ]
    if cfg.prefill_ranks is not None:
        args.extend(["--prefill-ranks", str(cfg.prefill_ranks)])
    return args


def _sync_and_barrier(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    device_index = 0 if device.index is None else int(device.index)
    dist.barrier(device_ids=[device_index])


def _write_metrics_sidecar(
    *,
    label: str,
    optimized: bool,
    custom_metrics: Dict[str, float],
) -> None:
    payload = {
        "benchmark": label,
        "optimized": optimized,
        "custom_metrics": custom_metrics,
    }
    metrics_path = os.environ.get(METRICS_ENV_VAR)
    if metrics_path:
        Path(metrics_path).write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _init_distributed() -> tuple[int, int, torch.device]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required for cache-aware disaggregation")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Run the multi-GPU cache-aware lab under torchrun")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("torchrun LOCAL_RANK is required")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def _stage_runtime_cache(
    *,
    cfg: CacheAwareDisaggMultiGPUConfig,
    rank: int,
    device: torch.device,
    request_id: int,
    target_rank: int,
    current_owner: Optional[int],
    current_cache_len: int,
    warm_cache_store: Dict[int, torch.Tensor],
    active_caches: Dict[int, torch.Tensor],
    metrics: Dict[str, float],
) -> None:
    if current_owner == target_rank:
        if rank != target_rank:
            return
        if request_id in active_caches:
            metrics["cache_hits"] += 1.0
            return
        warm_cache = warm_cache_store.get(request_id)
        if warm_cache is not None:
            active_caches[request_id] = warm_cache
            metrics["cache_misses"] += 1.0
            metrics["shared_reload_bytes"] += _tensor_nbytes(warm_cache)
            return
        active_caches[request_id] = _empty_kv(cfg, device)
        metrics["cache_misses"] += 1.0
        return

    if current_owner is not None:
        if rank == current_owner:
            source = active_caches.pop(request_id, None)
            if source is None:
                source = warm_cache_store.get(request_id)
            if source is None:
                raise RuntimeError(f"Missing cache for peer handoff request {request_id}")
            dist.send(source, dst=target_rank)
            metrics["cache_misses"] += 1.0
            metrics["peer_handoffs"] += 1.0
            metrics["worker_switches"] += 1.0
            metrics["kv_transfer_bytes"] += _tensor_nbytes(source)
            return
        if rank == target_rank:
            recv_cache = torch.empty(
                (cfg.batch_size, current_cache_len, cfg.hidden_size),
                device=device,
                dtype=cfg.dtype,
            )
            dist.recv(recv_cache, src=current_owner)
            active_caches[request_id] = recv_cache
            return
        return

    if rank != target_rank:
        return
    warm_cache = warm_cache_store.get(request_id)
    if warm_cache is not None:
        active_caches[request_id] = warm_cache
        metrics["cache_misses"] += 1.0
        metrics["shared_reload_bytes"] += _tensor_nbytes(warm_cache)
        return
    active_caches[request_id] = _empty_kv(cfg, device)
    metrics["cache_misses"] += 1.0


def _run_torchrun_worker(
    cfg: CacheAwareDisaggMultiGPUConfig,
    *,
    affinity_mode: DecodeAffinityMode,
    label: str,
    iters: int,
    warmup: int,
) -> None:
    rank, world_size, device = _init_distributed()
    if world_size < 2:
        raise RuntimeError("cache-aware disaggregated inference requires >=2 torchrun ranks")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"torchrun world_size={world_size} exceeds visible GPUs ({torch.cuda.device_count()})"
        )

    prefill_ranks = _resolve_prefill_ranks(world_size, cfg.prefill_ranks)
    decode_ranks = world_size - prefill_ranks
    if decode_ranks < 1:
        raise RuntimeError("decode ranks must be >= 1")
    if rank == 0:
        _emit_split_advice(prefill_ranks, decode_ranks)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = TinyPrefillDecode(cfg.hidden_size, cfg.num_layers, device, cfg.dtype).eval()

    prompts: Optional[torch.Tensor] = None
    prompt_chunks: Dict[int, Sequence[torch.Tensor]] = {}
    if rank < prefill_ranks:
        prompts = torch.randn(
            cfg.requests_per_rank,
            cfg.batch_size,
            cfg.context_window,
            cfg.hidden_size,
            device=device,
            dtype=cfg.dtype,
        )
        prompt_chunks = {
            request_idx: _split_prompt(prompts[request_idx], cfg.chunk_size)
            for request_idx in range(cfg.requests_per_rank)
        }

    plans = _build_request_plans(cfg, prefill_ranks=prefill_ranks)
    warm_cache_store: Dict[int, torch.Tensor] = {}
    prefill_seed_store: Dict[int, torch.Tensor] = {}

    for plan in plans:
        if plan.warm_chunks <= 0:
            continue
        home_rank = _home_decode_rank(plan.global_request_idx, prefill_ranks, decode_ranks)
        if rank == plan.prefill_rank:
            assert prompts is not None
            chunks = prompt_chunks[plan.local_request_idx]
            prefix_cache = torch.empty(
                (cfg.batch_size, _prefix_length(cfg, plan.warm_chunks), cfg.hidden_size),
                device=device,
                dtype=cfg.dtype,
            )
            seed: Optional[torch.Tensor] = None
            offset = 0
            for chunk in chunks[: plan.warm_chunks]:
                next_offset = offset + int(chunk.size(1))
                _, seed = model.prefill_into(
                    chunk,
                    prefix_cache[:, offset:next_offset],
                )
                offset = next_offset
            if seed is None:
                raise RuntimeError(f"Warm request {plan.global_request_idx} did not produce a seed")
            prefill_seed_store[plan.global_request_idx] = seed
            dist.send(prefix_cache, dst=home_rank)
        elif rank == home_rank:
            prefix_cache = torch.empty(
                (cfg.batch_size, _prefix_length(cfg, plan.warm_chunks), cfg.hidden_size),
                device=device,
                dtype=cfg.dtype,
            )
            dist.recv(prefix_cache, src=plan.prefill_rank)
            warm_cache_store[plan.global_request_idx] = prefix_cache
        _sync_and_barrier(device)

    recv_chunk_buffers: Dict[int, torch.Tensor] = {}
    recv_seed_buffer = torch.empty(0, device=device, dtype=cfg.dtype)
    if rank >= prefill_ranks:
        recv_chunk_buffers = {
            chunk_idx: torch.empty(
                (cfg.batch_size, _chunk_length(cfg, chunk_idx), cfg.hidden_size),
                device=device,
                dtype=cfg.dtype,
            )
            for chunk_idx in range(cfg.num_chunks)
        }
        recv_seed_buffer = torch.empty(
            (cfg.batch_size, cfg.hidden_size),
            device=device,
            dtype=cfg.dtype,
        )

    active_caches: Dict[int, torch.Tensor] = {}
    kv_buffers: Dict[int, torch.Tensor] = {}
    local_metrics = {
        "cache_hits": 0.0,
        "cache_misses": 0.0,
        "worker_switches": 0.0,
        "peer_handoffs": 0.0,
        "kv_transfer_bytes": 0.0,
        "shared_reload_bytes": 0.0,
    }

    def run_iteration() -> tuple[Dict[str, float], float, float, int]:
        active_caches.clear()
        local_metrics["cache_hits"] = 0.0
        local_metrics["cache_misses"] = 0.0
        local_metrics["worker_switches"] = 0.0
        local_metrics["peer_handoffs"] = 0.0
        local_metrics["kv_transfer_bytes"] = 0.0
        local_metrics["shared_reload_bytes"] = 0.0
        ttft_sum_ms = 0.0
        tpot_sum_ms = 0.0
        request_count = 0

        for plan in plans:
            current_owner = (
                _home_decode_rank(plan.global_request_idx, prefill_ranks, decode_ranks)
                if plan.is_warm
                else None
            )
            current_cache_len = _prefix_length(cfg, plan.warm_chunks)
            seed = prefill_seed_store.get(plan.global_request_idx)
            chunks: Sequence[torch.Tensor] = ()
            if rank == plan.prefill_rank:
                assert prompts is not None
                chunks = prompt_chunks[plan.local_request_idx]
                torch.cuda.synchronize(device)
                request_start = time.perf_counter()
            else:
                request_start = 0.0

            for chunk_idx in range(plan.warm_chunks, plan.total_chunks):
                target_rank = _choose_decode_rank(
                    plan,
                    chunk_idx,
                    affinity_mode=affinity_mode,
                    prefill_ranks=prefill_ranks,
                    decode_ranks=decode_ranks,
                )
                _stage_runtime_cache(
                    cfg=cfg,
                    rank=rank,
                    device=device,
                    request_id=plan.global_request_idx,
                    target_rank=target_rank,
                    current_owner=current_owner,
                    current_cache_len=current_cache_len,
                    warm_cache_store=warm_cache_store,
                    active_caches=active_caches,
                    metrics=local_metrics,
                )
                _sync_and_barrier(device)

                if rank == plan.prefill_rank:
                    chunk_kv, seed = model.prefill(chunks[chunk_idx])
                    dist.send(chunk_kv.contiguous(), dst=target_rank)
                    local_metrics["kv_transfer_bytes"] += _tensor_nbytes(chunk_kv)
                elif rank == target_rank:
                    recv_chunk = recv_chunk_buffers[chunk_idx]
                    dist.recv(recv_chunk, src=plan.prefill_rank)
                    base = active_caches.get(plan.global_request_idx)
                    if base is None:
                        base = _empty_kv(cfg, device)
                    active_caches[plan.global_request_idx] = _extend_cache_buffer(
                        cfg,
                        request_id=plan.global_request_idx,
                        cache=base,
                        chunk_kv=recv_chunk,
                        kv_buffers=kv_buffers,
                    )
                _sync_and_barrier(device)

                current_owner = target_rank
                current_cache_len += _chunk_length(cfg, chunk_idx)

            if rank == plan.prefill_rank:
                prefill_end = time.perf_counter()
            else:
                prefill_end = 0.0

            decode_rank = _choose_decode_rank(
                plan,
                plan.total_chunks,
                affinity_mode=affinity_mode,
                prefill_ranks=prefill_ranks,
                decode_ranks=decode_ranks,
            )
            _stage_runtime_cache(
                cfg=cfg,
                rank=rank,
                device=device,
                request_id=plan.global_request_idx,
                target_rank=decode_rank,
                current_owner=current_owner,
                current_cache_len=current_cache_len,
                warm_cache_store=warm_cache_store,
                active_caches=active_caches,
                metrics=local_metrics,
            )
            _sync_and_barrier(device)

            if rank == plan.prefill_rank:
                if seed is None:
                    raise RuntimeError(f"Request {plan.global_request_idx} has no decode seed")
                dist.send(seed.contiguous(), dst=decode_rank)
            elif rank == decode_rank:
                recv_seed = recv_seed_buffer
                dist.recv(recv_seed, src=plan.prefill_rank)
                cache = active_caches[plan.global_request_idx]
                _ = model.decode(recv_seed, cache, cfg.decode_tokens)
                active_caches.pop(plan.global_request_idx, None)
            _sync_and_barrier(device)

            if rank == plan.prefill_rank:
                total_ms = (time.perf_counter() - request_start) * 1000.0
                ttft_ms = (prefill_end - request_start) * 1000.0
                ttft_sum_ms += ttft_ms
                tpot_sum_ms += max(total_ms - ttft_ms, 0.0) / max(cfg.decode_tokens, 1)
                request_count += 1

        return local_metrics, ttft_sum_ms, tpot_sum_ms, request_count

    _sync_and_barrier(device)
    for _ in range(max(int(warmup), 0)):
        run_iteration()
        _sync_and_barrier(device)

    elapsed_start = time.perf_counter()
    aggregate = {
        "cache_hits": 0.0,
        "cache_misses": 0.0,
        "worker_switches": 0.0,
        "peer_handoffs": 0.0,
        "kv_transfer_bytes": 0.0,
        "shared_reload_bytes": 0.0,
    }
    ttft_total_ms = 0.0
    tpot_total_ms = 0.0
    request_total = 0
    for _ in range(max(int(iters), 1)):
        iteration_metrics, iteration_ttft, iteration_tpot, iteration_requests = run_iteration()
        for key, value in iteration_metrics.items():
            aggregate[key] += float(value)
        ttft_total_ms += float(iteration_ttft)
        tpot_total_ms += float(iteration_tpot)
        request_total += int(iteration_requests)
        _sync_and_barrier(device)
    elapsed_s = max(time.perf_counter() - elapsed_start, 1e-9)

    reduced = torch.tensor(
        [
            aggregate["cache_hits"],
            aggregate["cache_misses"],
            aggregate["worker_switches"],
            aggregate["peer_handoffs"],
            aggregate["kv_transfer_bytes"],
            aggregate["shared_reload_bytes"],
            ttft_total_ms,
            tpot_total_ms,
            float(request_total),
        ],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)

    if rank == 0:
        reduced_host = reduced.detach().cpu()
        cache_hits = float(reduced_host[0])
        cache_misses = float(reduced_host[1])
        worker_switches = float(reduced_host[2])
        peer_handoffs = float(reduced_host[3])
        kv_transfer_bytes = float(reduced_host[4])
        shared_reload_bytes = float(reduced_host[5])
        ttft_reduced_ms = float(reduced_host[6])
        tpot_reduced_ms = float(reduced_host[7])
        request_count = float(reduced_host[8])
        total_requests = max(float(request_count), 1.0)
        cache_decisions = max(float(cache_hits + cache_misses), 1.0)
        total_generated_tokens = (
            cfg.requests_per_rank * prefill_ranks * cfg.batch_size * cfg.decode_tokens
        )
        custom_metrics = {
            **compute_inference_metrics(
                ttft_ms=float(ttft_reduced_ms) / total_requests,
                tpot_ms=float(tpot_reduced_ms) / total_requests,
                total_tokens=total_generated_tokens,
                total_requests=int(cfg.requests_per_rank * prefill_ranks * cfg.batch_size),
                batch_size=cfg.batch_size,
                max_batch_size=max(cfg.batch_size * decode_ranks, cfg.batch_size),
            ),
            "cache_aware.cache_hit_rate": float(cache_hits) / cache_decisions,
            "cache_aware.kv_transfer_mb": float(kv_transfer_bytes) / 1e6,
            "cache_aware.worker_switches_per_request": float(worker_switches) / total_requests,
            "cache_aware.peer_handoffs": float(peer_handoffs),
            "cache_aware.shared_reload_mb": float(shared_reload_bytes) / 1e6,
            "cache_aware.time_per_iter_ms": (elapsed_s / max(int(iters), 1)) * 1000.0,
            "cache_aware.wall_tokens_per_second": (
                total_generated_tokens * (max(int(iters), 1) / elapsed_s)
            ),
        }
        _write_metrics_sidecar(
            label=label,
            optimized=affinity_mode == DecodeAffinityMode.STICKY,
            custom_metrics=custom_metrics,
        )

    dist.destroy_process_group()


class CacheAwareDisaggMultiGPUBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark object for the torchrun-backed cache-aware multi-GPU lab."""

    multi_gpu_required = True
    allowed_benchmark_fn_antipatterns = ("sync",)
    story_metadata = {
        "pair_role": "exemplar",
        "chapter_alignment": "supplementary",
        "chapter_native_exemplar": True,
        "optimization_mechanism": "preserve decode-rank KV locality under real cross-rank handoff",
        "benchmark_story": "cache-aware disaggregated inference multigpu",
    }

    def __init__(
        self,
        *,
        optimized: bool,
        label: str,
        cfg: Optional[CacheAwareDisaggMultiGPUConfig] = None,
        script_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.optimized = optimized
        self.affinity_mode = (
            DecodeAffinityMode.STICKY if optimized else DecodeAffinityMode.ROUND_ROBIN
        )
        self.label = label
        self.cfg = cfg or CacheAwareDisaggMultiGPUConfig()
        self.script_path = str(Path(script_path).resolve()) if script_path else ""
        self._metrics_sidecar_path = Path(tempfile.gettempdir()) / f"aisp_{label}_metrics.json"
        self._resolved_world_size: Optional[int] = None
        self._resolved_prefill_ranks: Optional[int] = None
        self._resolved_decode_ranks: Optional[int] = None
        self._verify_prompt: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_parts: List[torch.Tensor] = []
        self._output_part_count = 0
        self._output_stack: Optional[torch.Tensor] = None
        self._outputs_ready = False
        self.parameter_count: int = 0
        self._custom_metrics: Dict[str, float] = {}
        self._pending_metrics: Dict[str, float] = {}
        self._request_plans: List[DistributedRequestPlan] = []
        self._request_plan_count = 0
        self._decode_chunk_ranges: Dict[int, range] = {}
        self._decode_chunk_range_count = 0
        self._prefill_models: Dict[int, TinyPrefillDecode] = {}
        self._decode_models: Dict[int, TinyPrefillDecode] = {}
        self._decode_model_count = 0
        self._prompts: Dict[int, torch.Tensor] = {}
        self._prompt_chunks: Dict[tuple[int, int], Sequence[torch.Tensor]] = {}
        self._warm_cache_store: Dict[int, Dict[int, torch.Tensor]] = {}
        self._prefill_seed_store: Dict[int, torch.Tensor] = {}
        self._empty_kv_by_device: Dict[str, torch.Tensor] = {}
        self._decode_devices: Dict[int, torch.device] = {}
        self._decode_seed_buffers: Dict[int, torch.Tensor] = {}
        self._active_caches: Dict[int, Dict[int, torch.Tensor]] = {}
        self._active_cache_count = 0
        self._kv_buffer_pools: Dict[int, Dict[int, torch.Tensor]] = {}
        self._kv_buffer_pool_count = 0
        self._sync_devices: List[torch.device] = []
        self._verify_output = torch.zeros(1, dtype=torch.float32)
        self._metric_request_count = 1.0
        self._metric_total_tokens = 0
        self._metric_total_batch_requests = 0
        self._metric_max_batch_size = self.cfg.batch_size
        self._decode_token_divisor = float(max(self.cfg.decode_tokens, 1))
        self._register_workload_metadata(
            world_size=_world_size_hint(),
            prefill_ranks=_hint_prefill_ranks(_world_size_hint(), self.cfg.prefill_ranks),
        )

    def _register_workload_metadata(self, *, world_size: int, prefill_ranks: int) -> None:
        total_requests = self.cfg.requests_per_rank * prefill_ranks * self.cfg.batch_size
        self.register_workload_metadata(
            requests_per_iteration=float(total_requests),
            tokens_per_iteration=float(total_requests * self.cfg.tokens_per_request),
        )

    def _resolve_runtime_layout(self) -> tuple[int, int, int]:
        world_size = _resolve_runtime_world_size()
        prefill_ranks = _resolve_prefill_ranks(world_size, self.cfg.prefill_ranks)
        decode_ranks = world_size - prefill_ranks
        if decode_ranks < 1:
            raise RuntimeError("decode ranks must be >= 1")
        return world_size, prefill_ranks, decode_ranks

    def _sync_local_devices(self) -> None:
        for device in self._sync_devices:
            torch.cuda.synchronize(device)

    def _decode_device_for_rank(self, rank: int) -> torch.device:
        device = self._decode_devices.get(rank)
        if device is None:
            raise RuntimeError(f"Decode device missing for rank {rank}")
        return device

    def _empty_kv_for_device(self, device: torch.device) -> torch.Tensor:
        cached = self._empty_kv_by_device.get(str(device))
        if cached is None:
            raise RuntimeError(f"Empty KV template missing for {device}")
        return cached

    def _allocate_host_tensor(self, shape: torch.Size | tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        try:
            return torch.empty(shape, device="cpu", dtype=dtype, pin_memory=True)
        except RuntimeError:
            return torch.empty(shape, device="cpu", dtype=dtype)

    def _allocate_host_output_stack(self) -> torch.Tensor:
        shape = (
            len(self._request_plans),
            self.cfg.batch_size,
            self.cfg.hidden_size,
        )
        return self._allocate_host_tensor(shape, self.cfg.dtype)

    def _ensure_local_cache(
        self,
        *,
        request_id: int,
        target_rank: int,
        current_owner: Optional[int],
        active_caches: Dict[int, Dict[int, torch.Tensor]],
        metrics: Dict[str, float],
    ) -> torch.Tensor:
        target_device = self._decode_device_for_rank(target_rank)
        target_active = active_caches[target_rank]
        if current_owner == target_rank:
            cache = target_active.get(request_id)
            if cache is not None:
                metrics["cache_hits"] += 1.0
                return cache
            warm_cache = self._warm_cache_store[target_rank].get(request_id)
            if warm_cache is not None:
                target_active[request_id] = warm_cache
                metrics["cache_misses"] += 1.0
                metrics["shared_reload_bytes"] += _tensor_nbytes(warm_cache)
                return warm_cache
            cache = self._empty_kv_for_device(target_device)
            target_active[request_id] = cache
            metrics["cache_misses"] += 1.0
            return cache

        if current_owner is not None:
            source = active_caches[current_owner].pop(request_id, None)
            if source is None:
                source = self._warm_cache_store[current_owner].get(request_id)
            if source is None:
                raise RuntimeError(f"Missing cache for peer handoff request {request_id}")
            transferred = source.to(target_device)
            target_active[request_id] = transferred
            metrics["cache_misses"] += 1.0
            metrics["peer_handoffs"] += 1.0
            metrics["worker_switches"] += 1.0
            metrics["kv_transfer_bytes"] += _tensor_nbytes(source)
            return transferred

        warm_cache = self._warm_cache_store[target_rank].get(request_id)
        if warm_cache is not None:
            target_active[request_id] = warm_cache
            metrics["cache_misses"] += 1.0
            metrics["shared_reload_bytes"] += _tensor_nbytes(warm_cache)
            return warm_cache
        cache = self._empty_kv_for_device(target_device)
        target_active[request_id] = cache
        metrics["cache_misses"] += 1.0
        return cache

    def setup(self) -> None:
        world_size, prefill_ranks, decode_ranks = self._resolve_runtime_layout()
        self._resolved_world_size = world_size
        self._resolved_prefill_ranks = prefill_ranks
        self._resolved_decode_ranks = decode_ranks
        self._register_workload_metadata(world_size=world_size, prefill_ranks=prefill_ranks)
        self._metric_request_count = max(
            float(self.cfg.requests_per_rank * prefill_ranks),
            1.0,
        )
        self._metric_total_tokens = int(
            self.cfg.requests_per_rank
            * prefill_ranks
            * self.cfg.batch_size
            * self.cfg.decode_tokens
        )
        self._metric_total_batch_requests = int(
            self.cfg.requests_per_rank * prefill_ranks * self.cfg.batch_size
        )
        self._metric_max_batch_size = max(
            self.cfg.batch_size * decode_ranks,
            self.cfg.batch_size,
        )
        self._decode_token_divisor = float(max(self.cfg.decode_tokens, 1))

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        reference_state = _build_reference_state(self.cfg)
        self._prefill_models = {}
        self._decode_models = {}
        self._prompts = {}
        self._prompt_chunks = {}
        self._request_plans = _build_request_plans(self.cfg, prefill_ranks=prefill_ranks)
        self._request_plan_count = len(self._request_plans)
        self._decode_chunk_ranges = {
            plan.global_request_idx: range(plan.warm_chunks, plan.total_chunks)
            for plan in self._request_plans
        }
        self._decode_chunk_range_count = len(self._decode_chunk_ranges)
        self._warm_cache_store = {
            rank: {} for rank in range(prefill_ranks, world_size)
        }
        self._prefill_seed_store = {}
        self.output = None
        self._output_parts = [torch.empty(0) for _ in self._request_plans]
        self._output_part_count = len(self._output_parts)
        self._output_stack = self._allocate_host_output_stack()
        self._outputs_ready = False
        self._custom_metrics.clear()
        self._empty_kv_by_device = {}
        self._decode_devices = {}
        self._decode_seed_buffers = {}
        self._active_caches = {}
        self._active_cache_count = 0
        self._kv_buffer_pools = {}
        self._kv_buffer_pool_count = 0
        self._sync_devices = []
        total_params = 0

        for rank in range(prefill_ranks, world_size):
            device = torch.device(f"cuda:{rank}")
            model = TinyPrefillDecode(
                self.cfg.hidden_size,
                self.cfg.num_layers,
                device,
                self.cfg.dtype,
            ).eval()
            model.load_state_dict(reference_state)
            self._decode_models[rank] = model
            self._decode_devices[rank] = device
            self._empty_kv_by_device[str(device)] = torch.empty(
                self.cfg.batch_size,
                0,
                self.cfg.hidden_size,
                device=device,
                dtype=self.cfg.dtype,
            )
            self._decode_seed_buffers[rank] = torch.empty(
                self.cfg.batch_size,
                self.cfg.hidden_size,
                device=device,
                dtype=self.cfg.dtype,
            )
            total_params += sum(p.numel() for p in model.parameters())
        self._decode_model_count = len(self._decode_models)

        for rank in range(prefill_ranks):
            device = torch.device(f"cuda:{rank}")
            model = TinyPrefillDecode(
                self.cfg.hidden_size,
                self.cfg.num_layers,
                device,
                self.cfg.dtype,
            ).eval()
            model.load_state_dict(reference_state)
            self._prefill_models[rank] = model
            prompts = torch.randn(
                self.cfg.requests_per_rank,
                self.cfg.batch_size,
                self.cfg.context_window,
                self.cfg.hidden_size,
                device=device,
                dtype=self.cfg.dtype,
            )
            self._prompts[rank] = prompts
            for request_idx in range(self.cfg.requests_per_rank):
                self._prompt_chunks[(rank, request_idx)] = _split_prompt(
                    prompts[request_idx],
                    self.cfg.chunk_size,
                )
            total_params += sum(p.numel() for p in model.parameters())

        self._sync_devices = [
            torch.device(f"cuda:{rank}") for rank in range(world_size)
        ]
        self._active_caches = {rank: {} for rank in self._decode_models}
        self._active_cache_count = len(self._active_caches)
        self._kv_buffer_pools = {rank: {} for rank in self._decode_models}
        self._kv_buffer_pool_count = len(self._kv_buffer_pools)

        with torch.inference_mode():
            for plan in self._request_plans:
                if plan.warm_chunks <= 0:
                    continue
                chunks = self._prompt_chunks[(plan.prefill_rank, plan.local_request_idx)]
                warm_chunks = chunks[: plan.warm_chunks]
                prefix_tokens = sum(int(chunk.size(1)) for chunk in warm_chunks)
                prefix_buffer = torch.empty(
                    self.cfg.batch_size,
                    prefix_tokens,
                    self.cfg.hidden_size,
                    device=chunks[0].device,
                    dtype=self.cfg.dtype,
                )
                seed: Optional[torch.Tensor] = None
                prefill_model = self._prefill_models[plan.prefill_rank]
                offset = 0
                for chunk in warm_chunks:
                    next_offset = offset + int(chunk.size(1))
                    _, seed = prefill_model.prefill_into(
                        chunk,
                        prefix_buffer[:, offset:next_offset],
                    )
                    offset = next_offset
                if seed is None:
                    raise RuntimeError(f"Warm request {plan.global_request_idx} did not produce a seed")
                home_rank = _home_decode_rank(plan.global_request_idx, prefill_ranks, decode_ranks)
                home_device = self._decode_device_for_rank(home_rank)
                self._warm_cache_store[home_rank][plan.global_request_idx] = prefix_buffer.to(home_device)
                self._prefill_seed_store[plan.global_request_idx] = seed

        self.parameter_count = total_params
        self._verify_prompt = self._allocate_host_tensor(
            self._prompts[0][0].shape,
            self._prompts[0][0].dtype,
        )
        self._verify_prompt.copy_(self._prompts[0][0], non_blocking=False)
        self._pending_metrics = {
            "cache_hits": 0.0,
            "cache_misses": 0.0,
            "worker_switches": 0.0,
            "peer_handoffs": 0.0,
            "kv_transfer_bytes": 0.0,
            "shared_reload_bytes": 0.0,
        }
        self._sync_local_devices()

    def benchmark_fn(self) -> None:
        if not self._prefill_models or not self._decode_models:
            raise RuntimeError("setup() must run before benchmark_fn()")
        assert self._resolved_prefill_ranks is not None
        assert self._resolved_decode_ranks is not None

        active_caches = self._active_caches
        kv_buffers = self._kv_buffer_pools
        decode_model_count = self._decode_model_count
        if (
            self._active_cache_count != decode_model_count
            or self._kv_buffer_pool_count != decode_model_count
        ):
            raise RuntimeError("Cache control-plane slots not initialized")
        for rank in self._decode_models:
            if rank not in active_caches or rank not in kv_buffers:
                raise RuntimeError("Cache control-plane slots not initialized")
        for cache in active_caches.values():
            cache.clear()
        outputs = self._output_parts
        request_plan_count = self._request_plan_count
        if self._output_part_count != request_plan_count:
            raise RuntimeError("Decode output slots not initialized")
        if self._decode_chunk_range_count != request_plan_count:
            raise RuntimeError("Decode chunk ranges not initialized")
        output_idx = 0
        ttft_total_ms = 0.0
        tpot_total_ms = 0.0
        timing_count = 0
        metrics = self._pending_metrics
        metrics["cache_hits"] = 0.0
        metrics["cache_misses"] = 0.0
        metrics["worker_switches"] = 0.0
        metrics["peer_handoffs"] = 0.0
        metrics["kv_transfer_bytes"] = 0.0
        metrics["shared_reload_bytes"] = 0.0
        decode_chunk_ranges = self._decode_chunk_ranges

        with torch.inference_mode():
            for plan in self._request_plans:
                chunks = self._prompt_chunks[(plan.prefill_rank, plan.local_request_idx)]
                current_owner = (
                    _home_decode_rank(
                        plan.global_request_idx,
                        self._resolved_prefill_ranks,
                        self._resolved_decode_ranks,
                    )
                    if plan.is_warm
                    else None
                )
                seed = self._prefill_seed_store.get(plan.global_request_idx)

                self._sync_local_devices()
                request_start = time.perf_counter()

                for chunk_idx in decode_chunk_ranges[plan.global_request_idx]:
                    target_rank = _choose_decode_rank(
                        plan,
                        chunk_idx,
                        affinity_mode=self.affinity_mode,
                        prefill_ranks=self._resolved_prefill_ranks,
                        decode_ranks=self._resolved_decode_ranks,
                    )
                    cache = self._ensure_local_cache(
                        request_id=plan.global_request_idx,
                        target_rank=target_rank,
                        current_owner=current_owner,
                        active_caches=active_caches,
                        metrics=metrics,
                    )
                    chunk_kv, seed = self._prefill_models[plan.prefill_rank].prefill(chunks[chunk_idx])
                    active_caches[target_rank][plan.global_request_idx] = _extend_cache_buffer(
                        self.cfg,
                        request_id=plan.global_request_idx,
                        cache=cache,
                        chunk_kv=chunk_kv,
                        kv_buffers=kv_buffers[target_rank],
                    )
                    metrics["kv_transfer_bytes"] += _tensor_nbytes(chunk_kv)
                    current_owner = target_rank

                self._sync_local_devices()
                prefill_end = time.perf_counter()

                decode_rank = _choose_decode_rank(
                    plan,
                    plan.total_chunks,
                    affinity_mode=self.affinity_mode,
                    prefill_ranks=self._resolved_prefill_ranks,
                    decode_ranks=self._resolved_decode_ranks,
                )
                cache = self._ensure_local_cache(
                    request_id=plan.global_request_idx,
                    target_rank=decode_rank,
                    current_owner=current_owner,
                    active_caches=active_caches,
                    metrics=metrics,
                )
                if seed is None:
                    raise RuntimeError(f"Request {plan.global_request_idx} has no decode seed")
                decode_device = self._decode_device_for_rank(decode_rank)
                seed_buffer = self._decode_seed_buffers.get(decode_rank)
                if seed_buffer is None or seed_buffer.device != decode_device:
                    raise RuntimeError(f"Decode seed buffer missing for rank {decode_rank}")
                seed_buffer.copy_(seed, non_blocking=True)
                output = self._decode_models[decode_rank].decode(
                    seed_buffer,
                    cache,
                    self.cfg.decode_tokens,
                )
                active_caches[decode_rank].pop(plan.global_request_idx, None)
                self._sync_local_devices()
                total_ms = (time.perf_counter() - request_start) * 1000.0
                ttft_ms = (prefill_end - request_start) * 1000.0
                ttft_total_ms += ttft_ms
                tpot_total_ms += max(total_ms - ttft_ms, 0.0) / self._decode_token_divisor
                timing_count += 1
                outputs[output_idx] = output
                output_idx += 1

        self.output = None
        self._output_parts = outputs
        self._outputs_ready = True
        total_requests = self._metric_request_count
        cache_decisions = max(metrics["cache_hits"] + metrics["cache_misses"], 1.0)
        timing_divisor = max(timing_count, 1)
        custom_metrics = self._custom_metrics
        custom_metrics.clear()
        custom_metrics.update(
            compute_inference_metrics(
                ttft_ms=ttft_total_ms / timing_divisor,
                tpot_ms=tpot_total_ms / timing_divisor,
                total_tokens=self._metric_total_tokens,
                total_requests=self._metric_total_batch_requests,
                batch_size=self.cfg.batch_size,
                max_batch_size=self._metric_max_batch_size,
            )
        )
        custom_metrics["cache_aware.cache_hit_rate"] = metrics["cache_hits"] / cache_decisions
        custom_metrics["cache_aware.kv_transfer_mb"] = metrics["kv_transfer_bytes"] / 1e6
        custom_metrics["cache_aware.worker_switches_per_request"] = (
            metrics["worker_switches"] / total_requests
        )
        custom_metrics["cache_aware.peer_handoffs"] = metrics["peer_handoffs"]
        custom_metrics["cache_aware.shared_reload_mb"] = metrics["shared_reload_bytes"] / 1e6

    def capture_verification_payload(self) -> None:
        if not self._outputs_ready or self._verify_prompt is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        if self._output_stack is None:
            raise RuntimeError("Output stack buffer not initialized")
        for output_idx, part in enumerate(self._output_parts):
            self._output_stack[output_idx].copy_(part, non_blocking=False)
        self.output = self._output_stack
        world_size = self._resolved_world_size or _world_size_hint()
        prefill_ranks = self._resolved_prefill_ranks or _hint_prefill_ranks(world_size, self.cfg.prefill_ranks)
        decode_ranks = max(world_size - prefill_ranks, 1)
        self._set_verification_payload(
            inputs={"prompt": self._verify_prompt},
            output=self.output,
            batch_size=int(self.cfg.batch_size),
            parameter_count=int(self.parameter_count),
            precision_flags=PrecisionFlags(
                bf16=self.cfg.dtype == torch.bfloat16,
                tf32=torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            ),
            output_tolerance=(1e-2, 1e-2),
            signature_overrides={
                "world_size": world_size,
                "pipeline_stages": 2,
                "pipeline_stage_boundaries": [
                    (0, prefill_ranks - 1),
                    (prefill_ranks, prefill_ranks + decode_ranks - 1),
                ],
                "per_rank_batch_size": self.cfg.batch_size,
                "collective_type": "send_recv",
            },
        )

    def _prepare_verification_payload(self) -> None:
        if hasattr(self, "_subprocess_verify_output"):
            return
        self.setup()
        try:
            self.benchmark_fn()
            self.capture_verification_payload()
            self._subprocess_verify_output = self.get_verify_output()
            self._subprocess_output_tolerance = self.get_output_tolerance()
            self._subprocess_input_signature = self.get_input_signature()
        finally:
            self.teardown()

    def teardown(self) -> None:
        self._prefill_models = {}
        self._decode_models = {}
        self._decode_model_count = 0
        self._prompts = {}
        self._prompt_chunks = {}
        self._warm_cache_store = {}
        self._prefill_seed_store = {}
        self._request_plans = []
        self._request_plan_count = 0
        self._decode_chunk_ranges = {}
        self._decode_chunk_range_count = 0
        self.output = None
        self._output_parts = []
        self._output_part_count = 0
        self._output_stack = None
        self._outputs_ready = False
        self._empty_kv_by_device = {}
        self._decode_devices = {}
        self._decode_seed_buffers = {}
        self._active_caches = {}
        self._active_cache_count = 0
        self._kv_buffer_pools = {}
        self._kv_buffer_pool_count = 0
        self._sync_devices = []
        self._verify_prompt = None
        self._metric_request_count = 1.0
        self._metric_total_tokens = 0
        self._metric_total_batch_requests = 0
        self._metric_max_batch_size = self.cfg.batch_size
        self._decode_token_divisor = float(max(self.cfg.decode_tokens, 1))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        world_size = self._resolved_world_size or _world_size_hint()
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=world_size,
            iterations=4,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=900,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload_metadata

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        if (
            torch.cuda.is_available()
            and torch.cuda.device_count() >= 2
            and (
                self.cfg.prefill_ranks is not None
                or (torch.cuda.device_count() % 2 == 0)
            )
        ):
            self._prepare_verification_payload()
        self._metrics_sidecar_path.unlink(missing_ok=True)
        return TorchrunLaunchSpec(
            script_path=Path(self.script_path) if self.script_path else None,
            script_args=_shared_args_to_cli(self.cfg),
            env={METRICS_ENV_VAR: str(self._metrics_sidecar_path)},
            parse_rank0_only=True,
            multi_gpu_required=True,
            name=self.label,
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
            },
        )

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if self._metrics_sidecar_path.exists():
            try:
                payload = json.loads(self._metrics_sidecar_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            metrics = payload.get("custom_metrics")
            if isinstance(metrics, dict):
                return {
                    key: float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                }
        return dict(self._custom_metrics) if self._custom_metrics else None

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "No output captured"
        if not self._custom_metrics:
            return "No custom metrics captured"
        return None


def build_config_from_args(args: argparse.Namespace) -> CacheAwareDisaggMultiGPUConfig:
    return CacheAwareDisaggMultiGPUConfig(
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        batch_size=int(args.batch_size),
        requests_per_rank=int(args.requests_per_rank),
        context_window=int(args.context_window),
        chunk_size=int(args.chunk_size),
        decode_tokens=int(args.decode_tokens),
        warm_request_ratio=float(args.warm_request_ratio),
        warm_prefix_ratio=float(args.warm_prefix_ratio),
        prefill_ranks=int(args.prefill_ranks) if args.prefill_ranks is not None else None,
    )


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-size", type=int, default=CacheAwareDisaggMultiGPUConfig.hidden_size)
    parser.add_argument("--num-layers", type=int, default=CacheAwareDisaggMultiGPUConfig.num_layers)
    parser.add_argument("--batch-size", type=int, default=CacheAwareDisaggMultiGPUConfig.batch_size)
    parser.add_argument(
        "--requests-per-rank",
        type=int,
        default=CacheAwareDisaggMultiGPUConfig.requests_per_rank,
    )
    parser.add_argument("--context-window", type=int, default=CacheAwareDisaggMultiGPUConfig.context_window)
    parser.add_argument("--chunk-size", type=int, default=CacheAwareDisaggMultiGPUConfig.chunk_size)
    parser.add_argument("--decode-tokens", type=int, default=CacheAwareDisaggMultiGPUConfig.decode_tokens)
    parser.add_argument(
        "--warm-request-ratio",
        type=float,
        default=CacheAwareDisaggMultiGPUConfig.warm_request_ratio,
    )
    parser.add_argument(
        "--warm-prefix-ratio",
        type=float,
        default=CacheAwareDisaggMultiGPUConfig.warm_prefix_ratio,
    )
    parser.add_argument(
        "--prefill-ranks",
        type=int,
        default=CacheAwareDisaggMultiGPUConfig.prefill_ranks,
        help="Number of prefill ranks (required when world size is odd).",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multi-GPU cache-aware disaggregated inference lab under torchrun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_shared_args(parser)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def run_cli(*, optimized: bool) -> None:
    args = _parse_args()
    cfg = build_config_from_args(args)
    _run_torchrun_worker(
        cfg,
        affinity_mode=DecodeAffinityMode.STICKY if optimized else DecodeAffinityMode.ROUND_ROBIN,
        label=(
            "optimized_cache_aware_disagg_multigpu"
            if optimized
            else "baseline_cache_aware_disagg_multigpu"
        ),
        iters=int(args.iters),
        warmup=int(args.warmup),
    )
