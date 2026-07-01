"""Shared multi-GPU disaggregated prefill/decode helpers (Chapter 17)."""

from __future__ import annotations

import inspect
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from core.benchmark.verification import PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)


class HandoffMode(str, Enum):
    SERIAL = "serial"
    OVERLAP = "overlap"
    BATCHED = "batched"


@dataclass(frozen=True)
class PrefillDecodeConfig:
    hidden_size: int = 2048
    num_layers: int = 4
    batch_size: int = 2
    requests_per_rank: int = 24
    context_window: int = 1536
    decode_tokens: int = 512
    transfer_group: int = 4
    sync_per_request: bool = False
    barrier_per_request: bool = False
    dtype: torch.dtype = torch.bfloat16

    @property
    def tokens_per_request(self) -> int:
        return self.context_window + self.decode_tokens


@dataclass
class _LocalPair:
    prefill_device: torch.device
    decode_device: torch.device
    prefill_model: "TinyPrefillDecode"
    decode_model: "TinyPrefillDecode"
    prompts: torch.Tensor
    prefill_kv_chunks: List[torch.Tensor]
    prefill_seed_chunks: List[torch.Tensor]
    transfer_kv_chunks: List[torch.Tensor]
    transfer_seed_chunks: List[torch.Tensor]
    transfer_slots: Tuple[Tuple[int, torch.Tensor, torch.Tensor], ...]
    transfer_slot_counts: Tuple[int, int, int]
    expected_transfer_slot_counts: Tuple[int, int, int]
    decode_output_chunks: List[torch.Tensor]


class TinyPrefillDecode(nn.Module):
    """Simple prefill/decode model to emulate KV cache traffic."""

    def __init__(self, hidden_size: int, num_layers: int, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)
        ])
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to(device=device, dtype=dtype)

    def prefill(self, prompts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = prompts
        for layer in self.layers:
            x = torch.relu_(layer(x))
        logits = self.proj(x)
        kv_cache = x.contiguous()
        seed = logits[:, -1, :].contiguous()
        return kv_cache, seed

    def prefill_into(
        self,
        prompts: torch.Tensor,
        kv_out: torch.Tensor,
        seed_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if kv_out.shape != prompts.shape:
            raise RuntimeError(
                f"prefill_into destination shape {tuple(kv_out.shape)} does not match prompt shape {tuple(prompts.shape)}"
            )
        if kv_out.device != prompts.device or kv_out.dtype != prompts.dtype:
            raise RuntimeError("prefill_into destination must match prompt device and dtype")

        with torch.no_grad():
            x = prompts
            if self.layers:
                for layer in self.layers[:-1]:
                    x = torch.relu_(layer(x))
                final_layer = self.layers[-1]
                weight_t = final_layer.weight.t().expand(x.shape[0], -1, -1)
                torch.bmm(x, weight_t, out=kv_out)
                if final_layer.bias is not None:
                    kv_out.add_(final_layer.bias)
                torch.relu_(kv_out)
            else:
                kv_out.copy_(x)

            if seed_out is None:
                seed_out = torch.empty(
                    prompts.shape[0],
                    prompts.shape[2],
                    device=prompts.device,
                    dtype=prompts.dtype,
                )
            elif (
                seed_out.shape != (prompts.shape[0], prompts.shape[2])
                or seed_out.device != prompts.device
                or seed_out.dtype != prompts.dtype
            ):
                raise RuntimeError(
                    "prefill_into seed destination must match prompt batch, hidden size, device, and dtype"
                )

            torch.matmul(kv_out[:, -1, :], self.proj.weight.t(), out=seed_out)
            if self.proj.bias is not None:
                seed_out.add_(self.proj.bias)
        return kv_out, seed_out

    def decode(self, seed: torch.Tensor, kv_cache: torch.Tensor, decode_tokens: int) -> torch.Tensor:
        x = seed
        context = kv_cache.shape[1]
        for step in range(decode_tokens):
            kv = kv_cache[:, step % context, :]
            x = x + kv
            for layer in self.layers:
                x = torch.relu_(layer(x))
            x = self.proj(x)
        return x


def _resolve_world_size() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("SKIPPED: CUDA required for disaggregated prefill/decode")
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError(f"SKIPPED: Requires >= 2 GPUs (found {world_size} GPU)")
    return world_size


def _resolve_prefill_ranks(world_size: int, prefill_ranks: Optional[int]) -> int:
    if prefill_ranks is None:
        if world_size % 2 != 0:
            raise RuntimeError(
                "--prefill-ranks must be set when world_size is odd."
            )
        return world_size // 2
    requested = int(prefill_ranks)
    if requested < 1:
        raise RuntimeError(f"--prefill-ranks must be >= 1 (got {requested}).")
    if requested >= world_size:
        raise RuntimeError(
            f"--prefill-ranks={requested} must be < world_size={world_size}."
        )
    return requested


def _prefill_to_decode_rank(prefill_rank: int, prefill_ranks: int, decode_ranks: int) -> int:
    return prefill_ranks + (prefill_rank % decode_ranks)


def _decode_assigned_prefills(decode_rank: int, prefill_ranks: int, decode_ranks: int) -> List[int]:
    decode_idx = decode_rank - prefill_ranks
    return [rank for rank in range(prefill_ranks) if (rank % decode_ranks) == decode_idx]


def _emit_split_advice(prefill_ranks: int, decode_ranks: int) -> None:
    ratio = prefill_ranks / max(decode_ranks, 1)
    split_label = f"P{prefill_ranks}:D{decode_ranks}"
    if prefill_ranks == decode_ranks:
        print(f"Split {split_label} is balanced (recommended default).")
    if decode_ranks == 1 and prefill_ranks > 1:
        print(
            f"WARNING: split {split_label} may bottleneck TPOT/long outputs with a single decode rank."
        )
    if prefill_ranks == 1 and decode_ranks > 1:
        print(
            f"WARNING: split {split_label} may bottleneck TTFT for long prompts with a single prefill rank."
        )
    if ratio >= 3:
        print(
            f"WARNING: prefill-heavy split {split_label} may under-provision decode for TPOT."
        )
    elif ratio <= (1 / 3):
        print(
            f"WARNING: decode-heavy split {split_label} may under-provision prefill for TTFT."
        )
    print(
        "Recommendation: use a balanced split when unsure; prefer 2P1D for TTFT-focused runs "
        "and 1P2D+ for TPOT/long outputs."
    )


def _init_distributed() -> Tuple[int, int, torch.device]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required for disaggregated prefill/decode")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Run with torchrun (missing RANK/WORLD_SIZE env vars).")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Run with torchrun (missing LOCAL_RANK env var).")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def _run_prefill(
    cfg: PrefillDecodeConfig,
    model: TinyPrefillDecode,
    prompts: torch.Tensor,
    kv_chunks: Optional[List[torch.Tensor]] = None,
    seed_chunks: Optional[List[torch.Tensor]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if kv_chunks is None or len(kv_chunks) != cfg.requests_per_rank:
        kv_chunks = [torch.empty(0) for _ in range(cfg.requests_per_rank)]
    if seed_chunks is None or len(seed_chunks) != cfg.requests_per_rank:
        seed_chunks = [torch.empty(0) for _ in range(cfg.requests_per_rank)]
    with torch.inference_mode():
        for req_idx in range(cfg.requests_per_rank):
            request_prompt = prompts[req_idx]
            kv_cache, seed = model.prefill(request_prompt)
            kv_chunks[req_idx] = kv_cache
            seed_chunks[req_idx] = seed
    return kv_chunks, seed_chunks


def _run_decode(
    cfg: PrefillDecodeConfig,
    model: TinyPrefillDecode,
    kv_chunks: List[torch.Tensor],
    seed_chunks: List[torch.Tensor],
    outputs: Optional[List[torch.Tensor]] = None,
) -> List[torch.Tensor]:
    if outputs is None or len(outputs) != len(kv_chunks):
        outputs = [torch.empty(0) for _ in range(len(kv_chunks))]
    with torch.inference_mode():
        for output_idx, (kv_cache, seed) in enumerate(zip(kv_chunks, seed_chunks)):
            outputs[output_idx] = model.decode(seed, kv_cache, cfg.decode_tokens)
    return outputs


def _run_torchrun_worker(
    cfg: PrefillDecodeConfig,
    *,
    handoff_mode: HandoffMode,
    label: str,
    iters: int,
    warmup: int,
    prefill_ranks: Optional[int],
) -> None:
    rank, world_size, device = _init_distributed()
    if world_size < 2:
        raise RuntimeError(f"SKIPPED: Requires >= 2 GPUs (found {world_size} GPU)")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"torchrun world_size={world_size} exceeds visible GPUs ({torch.cuda.device_count()})."
        )

    prefill_ranks = _resolve_prefill_ranks(world_size, prefill_ranks)
    decode_ranks = world_size - prefill_ranks
    if decode_ranks < 1:
        raise RuntimeError("decode_ranks must be >= 1 for disaggregated prefill/decode")
    if rank == 0:
        _emit_split_advice(prefill_ranks, decode_ranks)

    pair_groups: dict[int, dist.ProcessGroup] = {}
    for prefill_rank in range(prefill_ranks):
        decode_rank = _prefill_to_decode_rank(prefill_rank, prefill_ranks, decode_ranks)
        pair_groups[prefill_rank] = dist.new_group(ranks=[prefill_rank, decode_rank])

    is_prefill = rank < prefill_ranks
    peer_rank = (
        _prefill_to_decode_rank(rank, prefill_ranks, decode_ranks)
        if is_prefill
        else -1
    )
    device_index = 0 if device.index is None else int(device.index)
    use_overlap = handoff_mode == HandoffMode.OVERLAP
    use_batched = handoff_mode == HandoffMode.BATCHED
    comm_stream = torch.cuda.Stream(device=device, priority=1) if use_overlap else None

    def _barrier() -> None:
        dist.barrier(device_ids=[device_index])

    def _pair_barrier(group: dist.ProcessGroup) -> None:
        dist.barrier(group=group, device_ids=[device_index])

    def _batch_isend(
        kv_cache: torch.Tensor,
        seed: torch.Tensor,
        dst: int,
        group: dist.ProcessGroup,
        *,
        ready_event: Optional[torch.cuda.Event] = None,
    ) -> List[dist.Work]:
        if comm_stream is None:
            raise RuntimeError("comm_stream unavailable for overlap sends")
        with torch.cuda.stream(comm_stream):
            if ready_event is not None:
                comm_stream.wait_event(ready_event)
            ops = [
                dist.P2POp(dist.isend, kv_cache, dst, group=group),
                dist.P2POp(dist.isend, seed, dst, group=group),
            ]
            return dist.batch_isend_irecv(ops)

    def _batch_irecv(
        kv_buf: torch.Tensor,
        seed_buf: torch.Tensor,
        src: int,
        group: dist.ProcessGroup,
    ) -> List[dist.Work]:
        if comm_stream is None:
            raise RuntimeError("comm_stream unavailable for overlap receives")
        with torch.cuda.stream(comm_stream):
            ops = [
                dist.P2POp(dist.irecv, kv_buf, src, group=group),
                dist.P2POp(dist.irecv, seed_buf, src, group=group),
            ]
            return dist.batch_isend_irecv(ops)

    def _send_blocking(
        kv_cache: torch.Tensor,
        seed: torch.Tensor,
        dst: int,
        group: dist.ProcessGroup,
    ) -> None:
        dist.send(kv_cache, dst, group=group)
        dist.send(seed, dst, group=group)

    def _recv_blocking(
        kv_buf: torch.Tensor,
        seed_buf: torch.Tensor,
        src: int,
        group: dist.ProcessGroup,
    ) -> None:
        dist.recv(kv_buf, src, group=group)
        dist.recv(seed_buf, src, group=group)

    def _wait_handles(handles: List[dist.Work]) -> None:
        for req in handles:
            req.wait()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = TinyPrefillDecode(cfg.hidden_size, cfg.num_layers, device, cfg.dtype).eval()

    prompts: Optional[torch.Tensor] = None
    if is_prefill:
        prompts = torch.randn(
            cfg.requests_per_rank,
            cfg.batch_size,
            cfg.context_window,
            cfg.hidden_size,
            device=device,
            dtype=cfg.dtype,
        )

    assigned_prefills: List[int] = []

    recv_kv_batches: dict[int, torch.Tensor] = {}
    recv_seed_batches: dict[int, torch.Tensor] = {}
    recv_kv_chunks: dict[int, List[torch.Tensor]] = {}
    recv_seed_chunks: dict[int, List[torch.Tensor]] = {}
    decode_batch_buffers: dict[int, List[Tuple[torch.Tensor, torch.Tensor, int]]] = {}
    decode_outputs: List[torch.Tensor] = []
    prefill_send_handles: List[Optional[List[dist.Work]]] = []
    prefill_inflight_groups: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
    decode_recv_entries: List[Optional[Tuple[List[dist.Work], torch.Tensor, torch.Tensor, int]]] = []
    serial_kv_chunks: List[torch.Tensor] = []
    serial_seed_chunks: List[torch.Tensor] = []
    serial_prefill_kv_chunks: List[torch.Tensor] = []
    serial_prefill_seed_chunks: List[torch.Tensor] = []
    group_size = max(1, min(cfg.transfer_group, cfg.requests_per_rank))
    group_slices = [
        (start, min(group_size, cfg.requests_per_rank - start))
        for start in range(0, cfg.requests_per_rank, group_size)
    ]
    ready_events = (
        [torch.cuda.Event(blocking=False) for _ in group_slices]
        if is_prefill and use_overlap
        else []
    )

    if not is_prefill:
        assigned_prefills = _decode_assigned_prefills(rank, prefill_ranks, decode_ranks)
        decode_outputs = [
            torch.empty(0)
            for _ in range(len(assigned_prefills) * cfg.requests_per_rank)
        ]
        if use_overlap:
            decode_recv_entries = [None] * (len(assigned_prefills) * len(group_slices))
            for src_rank in assigned_prefills:
                decode_batch_buffers[src_rank] = []
                for _, group_len in group_slices:
                    kv_group = torch.empty(
                        (group_len, cfg.batch_size, cfg.context_window, cfg.hidden_size),
                        device=device,
                        dtype=cfg.dtype,
                    )
                    seed_group = torch.empty(
                        (group_len, cfg.batch_size, cfg.hidden_size),
                        device=device,
                        dtype=cfg.dtype,
                    )
                    decode_batch_buffers[src_rank].append((kv_group, seed_group, group_len))
        elif use_batched:
            for src_rank in assigned_prefills:
                recv_kv_batches[src_rank] = torch.empty(
                    (cfg.requests_per_rank, cfg.batch_size, cfg.context_window, cfg.hidden_size),
                    device=device,
                    dtype=cfg.dtype,
                )
                recv_seed_batches[src_rank] = torch.empty(
                    (cfg.requests_per_rank, cfg.batch_size, cfg.hidden_size),
                    device=device,
                    dtype=cfg.dtype,
                )
        else:
            for src_rank in assigned_prefills:
                recv_kv_chunks[src_rank] = [
                    torch.empty(
                        (cfg.batch_size, cfg.context_window, cfg.hidden_size),
                        device=device,
                        dtype=cfg.dtype,
                    )
                    for _ in range(cfg.requests_per_rank)
                ]
                recv_seed_chunks[src_rank] = [
                    torch.empty(
                        (cfg.batch_size, cfg.hidden_size),
                        device=device,
                        dtype=cfg.dtype,
                    )
                    for _ in range(cfg.requests_per_rank)
                ]
            serial_kv_chunks = [torch.empty(0) for _ in range(len(decode_outputs))]
            serial_seed_chunks = [torch.empty(0) for _ in range(len(decode_outputs))]
    elif use_overlap:
        prefill_send_handles = [None] * len(group_slices)
        prefill_inflight_groups = [None] * len(group_slices)
    elif not use_batched:
        serial_prefill_kv_chunks = [torch.empty(0) for _ in range(cfg.requests_per_rank)]
        serial_prefill_seed_chunks = [torch.empty(0) for _ in range(cfg.requests_per_rank)]

    def run_iteration() -> List[torch.Tensor]:
        if is_prefill:
            if use_overlap:
                prefill_stream = torch.cuda.current_stream(device)
                with torch.inference_mode():
                    for group_idx, (start, group_len) in enumerate(group_slices):
                        group_prompts = prompts[start:start + group_len].reshape(
                            group_len * cfg.batch_size,
                            cfg.context_window,
                            cfg.hidden_size,
                        )
                        kv_cache, seed = model.prefill(group_prompts)
                        kv_group = kv_cache.view(
                            group_len,
                            cfg.batch_size,
                            cfg.context_window,
                            cfg.hidden_size,
                        )
                        seed_group = seed.view(group_len, cfg.batch_size, cfg.hidden_size)
                        prefill_inflight_groups[group_idx] = (kv_group, seed_group)
                        ready = ready_events[group_idx]
                        ready.record(prefill_stream)
                        prefill_send_handles[group_idx] = _batch_isend(
                            kv_group,
                            seed_group,
                            peer_rank,
                            pair_groups[rank],
                            ready_event=ready,
                        )
                for group_idx in range(len(group_slices)):
                    handles = prefill_send_handles[group_idx]
                    if handles is None:
                        raise RuntimeError("Missing pending prefill send handles")
                    _wait_handles(handles)
                    prefill_send_handles[group_idx] = None
                    prefill_inflight_groups[group_idx] = None
            elif use_batched:
                with torch.inference_mode():
                    batch_prompts = prompts.reshape(
                        cfg.requests_per_rank * cfg.batch_size,
                        cfg.context_window,
                        cfg.hidden_size,
                    )
                    kv_cache, seed = model.prefill(batch_prompts)
                    kv_batch = kv_cache.view(
                        cfg.requests_per_rank,
                        cfg.batch_size,
                        cfg.context_window,
                        cfg.hidden_size,
                    )
                    seed_batch = seed.view(
                        cfg.requests_per_rank,
                        cfg.batch_size,
                        cfg.hidden_size,
                    )
                handles = dist.batch_isend_irecv([
                    dist.P2POp(dist.isend, kv_batch, peer_rank, group=pair_groups[rank]),
                    dist.P2POp(dist.isend, seed_batch, peer_rank, group=pair_groups[rank]),
                ])
                _wait_handles(handles)
            else:
                kv_chunks, seed_chunks = _run_prefill(
                    cfg,
                    model,
                    prompts,
                    serial_prefill_kv_chunks,
                    serial_prefill_seed_chunks,
                )
                for kv_cache, seed in zip(kv_chunks, seed_chunks):
                    _send_blocking(kv_cache, seed, peer_rank, pair_groups[rank])
                    if cfg.sync_per_request:
                        torch.cuda.synchronize(device)
                    if cfg.barrier_per_request:
                        _pair_barrier(pair_groups[rank])
            return []

        outputs = decode_outputs
        if use_overlap:
            entry_count = 0
            for src_rank in assigned_prefills:
                for kv_group, seed_group, group_len in decode_batch_buffers[src_rank]:
                    handles = _batch_irecv(
                        kv_group,
                        seed_group,
                        src_rank,
                        pair_groups[src_rank],
                    )
                    decode_recv_entries[entry_count] = (handles, kv_group, seed_group, group_len)
                    entry_count += 1
            output_idx = 0
            for entry_idx in range(entry_count):
                entry = decode_recv_entries[entry_idx]
                if entry is None:
                    raise RuntimeError("Missing pending decode receive entry")
                handles, kv_group, seed_group, group_len = entry
                _wait_handles(handles)
                flat_seed = seed_group.reshape(group_len * cfg.batch_size, cfg.hidden_size)
                flat_kv = kv_group.reshape(group_len * cfg.batch_size, cfg.context_window, cfg.hidden_size)
                decoded = model.decode(flat_seed, flat_kv, cfg.decode_tokens)
                for decoded_output in decoded.view(group_len, cfg.batch_size, cfg.hidden_size).unbind(0):
                    outputs[output_idx] = decoded_output
                    output_idx += 1
                decode_recv_entries[entry_idx] = None
            return outputs

        if use_batched:
            output_idx = 0
            for src_rank in assigned_prefills:
                kv_batch = recv_kv_batches[src_rank]
                seed_batch = recv_seed_batches[src_rank]
                handles = dist.batch_isend_irecv([
                    dist.P2POp(dist.irecv, kv_batch, src_rank, group=pair_groups[src_rank]),
                    dist.P2POp(dist.irecv, seed_batch, src_rank, group=pair_groups[src_rank]),
                ])
                _wait_handles(handles)
                flat_seed = seed_batch.reshape(
                    cfg.requests_per_rank * cfg.batch_size,
                    cfg.hidden_size,
                )
                flat_kv = kv_batch.reshape(
                    cfg.requests_per_rank * cfg.batch_size,
                    cfg.context_window,
                    cfg.hidden_size,
                )
                decoded = model.decode(flat_seed, flat_kv, cfg.decode_tokens)
                for decoded_output in decoded.view(
                    cfg.requests_per_rank,
                    cfg.batch_size,
                    cfg.hidden_size,
                ).unbind(0):
                    outputs[output_idx] = decoded_output
                    output_idx += 1
            return outputs

        kv_chunks = serial_kv_chunks
        seed_chunks = serial_seed_chunks
        chunk_idx = 0
        for src_rank in assigned_prefills:
            for req_idx in range(cfg.requests_per_rank):
                kv_buf = recv_kv_chunks[src_rank][req_idx]
                seed_buf = recv_seed_chunks[src_rank][req_idx]
                _recv_blocking(
                    kv_buf,
                    seed_buf,
                    src_rank,
                    pair_groups[src_rank],
                )
                if cfg.sync_per_request:
                    torch.cuda.synchronize(device)
                if cfg.barrier_per_request:
                    _pair_barrier(pair_groups[src_rank])
                kv_chunks[chunk_idx] = kv_buf
                seed_chunks[chunk_idx] = seed_buf
                chunk_idx += 1
        return _run_decode(cfg, model, kv_chunks, seed_chunks, outputs)

    _barrier()
    torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(max(warmup, 0)):
            run_iteration()
        torch.cuda.synchronize(device)
        _barrier()

        start = time.perf_counter()
        for _ in range(max(iters, 1)):
            run_iteration()
        torch.cuda.synchronize(device)
    _barrier()
    elapsed = time.perf_counter() - start

    if rank == 0:
        total_requests = cfg.requests_per_rank * prefill_ranks * cfg.batch_size
        tokens_per_iter = total_requests * cfg.tokens_per_request
        tokens_per_s = tokens_per_iter * (max(iters, 1) / max(elapsed, 1e-9))
        time_per_iter_ms = (elapsed / max(iters, 1)) * 1000.0
        print(f"rank0 {label} tokens/s: {tokens_per_s:.2f} tokens/s")
        print(f"rank0 {label} time_per_iter_ms: {time_per_iter_ms:.3f}")

    dist.destroy_process_group()


class _PrefillDecodeMultiGPUBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Shared multi-GPU disaggregated prefill/decode harness."""

    multi_gpu_required = True

    def __init__(
        self,
        *,
        handoff_mode: HandoffMode,
        label: str,
        cfg: Optional[PrefillDecodeConfig] = None,
        prefill_ranks: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or PrefillDecodeConfig()
        self.world_size = _resolve_world_size()
        self.prefill_ranks = _resolve_prefill_ranks(self.world_size, prefill_ranks)
        self.decode_ranks = self.world_size - self.prefill_ranks
        if self.decode_ranks < 1:
            raise RuntimeError("decode_ranks must be >= 1 for disaggregated prefill/decode")
        self.handoff_mode = handoff_mode
        self.overlap = handoff_mode == HandoffMode.OVERLAP
        self.label = label
        self._pairs: List[_LocalPair] = []
        self._output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._pending_outputs: List[torch.Tensor] = []
        self._pending_output_count = 0
        self._expected_output_count = 0
        self._verify_prompt: Optional[torch.Tensor] = None
        self._metadata_inputs: Dict[str, torch.Tensor] = {}
        self._param_count: int = 0

        total_requests = self.cfg.requests_per_rank * self.prefill_ranks * self.cfg.batch_size
        tokens_per_iter = total_requests * self.cfg.tokens_per_request
        self.register_workload_metadata(
            requests_per_iteration=float(total_requests),
            tokens_per_iteration=float(tokens_per_iter),
        )

    def _allocate_output_buffer(self) -> torch.Tensor:
        shape = (
            len(self._pairs) * self.cfg.requests_per_rank,
            self.cfg.batch_size,
            self.cfg.hidden_size,
        )
        try:
            return torch.empty(shape, device="cpu", dtype=self.cfg.dtype, pin_memory=True)
        except RuntimeError:
            return torch.empty(shape, device="cpu", dtype=self.cfg.dtype)

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for disaggregated prefill/decode")
        if torch.cuda.device_count() < self.world_size:
            raise RuntimeError(
                f"SKIPPED: requires >= {self.world_size} GPUs (found {torch.cuda.device_count()})"
            )

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self._pairs = []
        total_params = 0
        reference = TinyPrefillDecode(
            self.cfg.hidden_size,
            self.cfg.num_layers,
            torch.device("cpu"),
            torch.float32,
        ).eval()
        reference_state = {k: v.detach().cpu() for k, v in reference.state_dict().items()}

        decode_devices = [
            torch.device(f"cuda:{idx}")
            for idx in range(self.prefill_ranks, self.prefill_ranks + self.decode_ranks)
        ]
        decode_models: dict[torch.device, TinyPrefillDecode] = {}
        for device in decode_devices:
            model = TinyPrefillDecode(
                self.cfg.hidden_size,
                self.cfg.num_layers,
                device,
                self.cfg.dtype,
            ).eval()
            model.load_state_dict(reference_state)
            decode_models[device] = model
            total_params += sum(p.numel() for p in model.parameters())

        for prefill_rank in range(self.prefill_ranks):
            prefill_device = torch.device(f"cuda:{prefill_rank}")
            decode_device = decode_devices[prefill_rank % len(decode_devices)]
            prefill_model = TinyPrefillDecode(
                self.cfg.hidden_size,
                self.cfg.num_layers,
                prefill_device,
                self.cfg.dtype,
            ).eval()
            prefill_model.load_state_dict(reference_state)
            prompts = torch.randn(
                self.cfg.requests_per_rank,
                self.cfg.batch_size,
                self.cfg.context_window,
                self.cfg.hidden_size,
                device=prefill_device,
                dtype=self.cfg.dtype,
            )
            total_params += sum(p.numel() for p in prefill_model.parameters())
            prefill_kv_chunks = [torch.empty(0) for _ in range(self.cfg.requests_per_rank)]
            prefill_seed_chunks = [torch.empty(0) for _ in range(self.cfg.requests_per_rank)]
            transfer_kv_chunks = [
                torch.empty(
                    (
                        self.cfg.batch_size,
                        self.cfg.context_window,
                        self.cfg.hidden_size,
                    ),
                    device=decode_device,
                    dtype=self.cfg.dtype,
                )
                for _ in range(self.cfg.requests_per_rank)
            ]
            transfer_seed_chunks = [
                torch.empty(
                    (self.cfg.batch_size, self.cfg.hidden_size),
                    device=decode_device,
                    dtype=self.cfg.dtype,
                )
                for _ in range(self.cfg.requests_per_rank)
            ]
            transfer_slots = tuple(
                (req_idx, transfer_kv_chunks[req_idx], transfer_seed_chunks[req_idx])
                for req_idx in range(self.cfg.requests_per_rank)
            )
            transfer_slot_counts = (
                len(transfer_kv_chunks),
                len(transfer_seed_chunks),
                len(transfer_slots),
            )
            expected_transfer_slot_counts = (
                self.cfg.requests_per_rank,
                self.cfg.requests_per_rank,
                self.cfg.requests_per_rank,
            )
            self._pairs.append(
                _LocalPair(
                    prefill_device=prefill_device,
                    decode_device=decode_device,
                    prefill_model=prefill_model,
                    decode_model=decode_models[decode_device],
                    prompts=prompts,
                    prefill_kv_chunks=prefill_kv_chunks,
                    prefill_seed_chunks=prefill_seed_chunks,
                    transfer_kv_chunks=transfer_kv_chunks,
                    transfer_seed_chunks=transfer_seed_chunks,
                    transfer_slots=transfer_slots,
                    transfer_slot_counts=transfer_slot_counts,
                    expected_transfer_slot_counts=expected_transfer_slot_counts,
                    decode_output_chunks=[
                        torch.empty(0) for _ in range(self.cfg.requests_per_rank)
                    ],
                )
            )

        self._param_count = total_params
        self._output = None
        if not self._pairs:
            raise RuntimeError("Failed to initialize prompts for verification")
        expected_outputs = len(self._pairs) * self.cfg.requests_per_rank
        self._pending_outputs = [torch.empty(0) for _ in range(expected_outputs)]
        self._pending_output_count = len(self._pending_outputs)
        self._expected_output_count = expected_outputs
        self._verify_prompt = self._pairs[0].prompts[0]
        self._output_buffer = self._allocate_output_buffer()
        meta_dtype = torch.float32
        self._metadata_inputs = {
            "decode_tokens": torch.zeros((self.cfg.decode_tokens,), dtype=meta_dtype),
            "hidden_size": torch.zeros((self.cfg.hidden_size,), dtype=meta_dtype),
            "num_layers": torch.zeros((self.cfg.num_layers,), dtype=meta_dtype),
        }
        for pair in self._pairs:
            torch.cuda.synchronize(pair.prefill_device)
            torch.cuda.synchronize(pair.decode_device)

    def benchmark_fn(self) -> None:
        if not self._pairs:
            raise RuntimeError("setup() must run before benchmark_fn()")

        outputs = self._pending_outputs
        if self._pending_output_count != self._expected_output_count:
            raise RuntimeError("Decode output slots not initialized")
        output_idx = 0
        with torch.inference_mode():
            for pair in self._pairs:
                same_device_handoff = pair.prefill_device == pair.decode_device
                if self.overlap:
                    for req_idx, transfer_kv, transfer_seed in pair.transfer_slots:
                        if same_device_handoff:
                            pair.prefill_model.prefill_into(
                                pair.prompts[req_idx],
                                transfer_kv,
                                transfer_seed,
                            )
                        else:
                            kv_cache, seed = pair.prefill_model.prefill(pair.prompts[req_idx])
                            with torch.cuda.device(pair.decode_device):
                                transfer_kv.copy_(kv_cache, non_blocking=True)
                                transfer_seed.copy_(seed, non_blocking=True)
                        outputs[output_idx] = pair.decode_model.decode(
                            transfer_seed,
                            transfer_kv,
                            self.cfg.decode_tokens,
                        )
                        output_idx += 1
                else:
                    if pair.transfer_slot_counts != pair.expected_transfer_slot_counts:
                        raise RuntimeError("Transfer chunk slots not initialized")
                    if same_device_handoff:
                        for req_idx, transfer_kv, transfer_seed in pair.transfer_slots:
                            pair.prefill_model.prefill_into(
                                pair.prompts[req_idx],
                                transfer_kv,
                                transfer_seed,
                            )
                            pair.prefill_kv_chunks[req_idx] = transfer_kv
                            pair.prefill_seed_chunks[req_idx] = transfer_seed
                    else:
                        kv_chunks, seed_chunks = _run_prefill(
                            self.cfg,
                            pair.prefill_model,
                            pair.prompts,
                            pair.prefill_kv_chunks,
                            pair.prefill_seed_chunks,
                        )
                        for req_idx, transfer_kv, transfer_seed in pair.transfer_slots:
                            with torch.cuda.device(pair.decode_device):
                                transfer_kv.copy_(
                                    kv_chunks[req_idx],
                                    non_blocking=True,
                                )
                                transfer_seed.copy_(
                                    seed_chunks[req_idx],
                                    non_blocking=True,
                                )
                    decoded = _run_decode(
                        self.cfg,
                        pair.decode_model,
                        pair.transfer_kv_chunks,
                        pair.transfer_seed_chunks,
                        pair.decode_output_chunks,
                    )
                    for decoded_output in decoded:
                        outputs[output_idx] = decoded_output
                        output_idx += 1

        self._pending_outputs = outputs
        self._output = None

    def capture_verification_payload(self) -> None:
        if self._verify_prompt is None or (self._output is None and not self._pending_outputs):
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._output is None:
            if self._output_buffer is None:
                raise RuntimeError("Output buffer not initialized")
            for output_idx, output in enumerate(self._pending_outputs):
                self._output_buffer[output_idx].copy_(output, non_blocking=False)
            self._output = self._output_buffer
        tf32_enabled = torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32)
        if not self._metadata_inputs:
            raise RuntimeError("setup() must initialize verification metadata tensors")
        self._set_verification_payload(
            inputs={
                "prompt": self._verify_prompt,
                "decode_tokens": self._metadata_inputs["decode_tokens"],
                "hidden_size": self._metadata_inputs["hidden_size"],
                "num_layers": self._metadata_inputs["num_layers"],
            },
            output=self._output,
            batch_size=int(self._output.shape[0]),
            parameter_count=int(self._param_count),
            precision_flags=PrecisionFlags(bf16=True, tf32=tf32_enabled),
            output_tolerance=(0.0, 0.0),
            signature_overrides={
                "world_size": self.world_size,
                "pipeline_stages": 2,
                "pipeline_stage_boundaries": [
                    (0, self.prefill_ranks - 1),
                    (self.prefill_ranks, self.prefill_ranks + self.decode_ranks - 1),
                ],
                "per_rank_batch_size": self.cfg.requests_per_rank,
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
        self._pairs = []
        self._output = None
        self._output_buffer = None
        self._pending_outputs = []
        self._pending_output_count = 0
        self._expected_output_count = 0
        self._verify_prompt = None
        self._metadata_inputs = {}
        torch.cuda.empty_cache()

    def validate_result(self) -> Optional[str]:
        if self._output is None and not self._pending_outputs:
            return "No output captured"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=self.world_size,
            iterations=4,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=900,
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        self._prepare_verification_payload()
        module = inspect.getmodule(self.__class__)
        module_name = getattr(module, "__name__", None) if module else None
        master_port = os.environ.get("MASTER_PORT", "29517")
        script_args = ["--prefill-ranks", str(self.prefill_ranks)]
        return TorchrunLaunchSpec(
            module_name=module_name,
            script_args=script_args,
            env={
                "NCCL_DEBUG": "WARN",
                "OMP_NUM_THREADS": "1",
                "MASTER_PORT": master_port,
            },
            parse_rank0_only=True,
            multi_gpu_required=True,
            name=self.label,
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
            },
        )
