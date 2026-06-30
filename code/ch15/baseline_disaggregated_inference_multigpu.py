"""Baseline disaggregated inference benchmark (multi-GPU torchrun pipeline).

Chapter 15: Disaggregated Inference

This benchmark models a disaggregated prefill/decode pipeline across multiple GPUs.
Baseline behavior is serialized: prefill completes for all requests before decode starts.
The optimized pair overlaps prefill and decode via pipelined transfers.
"""

from __future__ import annotations

from pathlib import Path

import argparse
import inspect
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from core.benchmark.verification_mixin import VerificationPayloadMixin  # noqa: E402
from core.benchmark.verification import PrecisionFlags  # noqa: E402
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from core.optimization.moe_inference import (  # noqa: E402
    MoeInferenceConfig,
    SimpleMoEGPT,
    allocate_kv_cache,
    env_override_int,
)

@dataclass(frozen=True)
class DisaggConfig:
    vocab_size: int = 16384
    hidden_size: int = 768
    ffn_size: int = 512
    num_layers: int = 1
    num_moe_layers: int = 1
    num_experts: int = 4
    top_k: int = 2
    batch_size: int = 2
    requests_per_rank: int = 96
    context_window: int = 1024
    decode_tokens: int = 32
    dtype: torch.dtype = torch.bfloat16

    @property
    def tokens_per_request(self) -> int:
        return self.context_window + self.decode_tokens


@dataclass
class _LocalPair:
    prefill_device: torch.device
    decode_device: torch.device
    prefill_model: SimpleMoEGPT
    decode_model: SimpleMoEGPT
    prompts: torch.Tensor
    decode_kv_cache: torch.Tensor
    decode_outputs: List[torch.Tensor]
    prefill_kv_chunks: List[torch.Tensor]
    prefill_seed_chunks: List[torch.Tensor]
    transfer_kv_chunks: List[torch.Tensor]
    transfer_seed_chunks: List[torch.Tensor]
    transfer_slots: Tuple[Tuple[int, torch.Tensor, torch.Tensor], ...]
    transfer_slot_counts: Tuple[int, int, int]
    expected_transfer_slot_counts: Tuple[int, int, int]


def _build_moe_config(cfg: DisaggConfig) -> MoeInferenceConfig:
    return MoeInferenceConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        ffn_size=cfg.ffn_size,
        num_layers=cfg.num_layers,
        num_moe_layers=cfg.num_moe_layers,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        moe_layer_frequency=1,
        batch_size=cfg.batch_size,
        context_window=cfg.context_window,
        decode_tokens=cfg.decode_tokens,
        router_noise=0.0,
        dtype=cfg.dtype,
    )


def _apply_profile_overrides(cfg: DisaggConfig) -> DisaggConfig:
    return DisaggConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        ffn_size=cfg.ffn_size,
        num_layers=cfg.num_layers,
        num_moe_layers=cfg.num_moe_layers,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        batch_size=env_override_int("AISP_NCU_PROFILE_BATCH", cfg.batch_size),
        requests_per_rank=env_override_int("AISP_NCU_PROFILE_REQUESTS", cfg.requests_per_rank),
        context_window=env_override_int("AISP_NCU_PROFILE_CONTEXT", cfg.context_window),
        decode_tokens=env_override_int("AISP_NCU_PROFILE_DECODE", cfg.decode_tokens),
        dtype=cfg.dtype,
    )


def _resolve_world_size() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("SKIPPED: CUDA required for multi-GPU disaggregation")
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError(f"SKIPPED: Requires >= 2 GPUs (found {world_size} GPU)")
    return world_size


def _init_distributed() -> Tuple[int, int, torch.device]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required for multi-GPU disaggregation")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Run with torchrun (missing LOCAL_RANK env var).")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def _run_prefill(
    cfg: DisaggConfig,
    model: SimpleMoEGPT,
    prompts: torch.Tensor,
    kv_chunks: Optional[List[torch.Tensor]] = None,
    seed_chunks: Optional[List[torch.Tensor]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    reuse_kv_chunks = kv_chunks is not None and len(kv_chunks) == cfg.requests_per_rank
    reuse_seed_chunks = seed_chunks is not None and len(seed_chunks) == cfg.requests_per_rank
    if not reuse_kv_chunks:
        kv_chunks = [torch.empty(0) for _ in range(cfg.requests_per_rank)]
    if not reuse_seed_chunks:
        seed_chunks = [torch.empty(0) for _ in range(cfg.requests_per_rank)]
    assert kv_chunks is not None and seed_chunks is not None
    with torch.inference_mode():
        for req_idx in range(cfg.requests_per_rank):
            request_prompt = prompts[req_idx]
            hidden, logits = model.prefill(request_prompt)
            seed_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            kv_out = kv_chunks[req_idx]
            seed_out = seed_chunks[req_idx]
            if kv_out.shape == hidden.shape and kv_out.device == hidden.device and kv_out.dtype == hidden.dtype:
                kv_out.copy_(hidden)
            else:
                kv_chunks[req_idx] = hidden.contiguous()
            if (
                seed_out.shape == seed_tokens.shape
                and seed_out.device == seed_tokens.device
                and seed_out.dtype == seed_tokens.dtype
            ):
                seed_out.copy_(seed_tokens)
            else:
                seed_chunks[req_idx] = seed_tokens.contiguous()
    return kv_chunks, seed_chunks


def _run_decode(
    cfg: DisaggConfig,
    model: SimpleMoEGPT,
    kv_chunks: List[torch.Tensor],
    seed_chunks: List[torch.Tensor],
    device: torch.device,
    *,
    kv_cache: Optional[torch.Tensor] = None,
    outputs: Optional[List[torch.Tensor]] = None,
) -> List[torch.Tensor]:
    if outputs is None or len(outputs) != len(kv_chunks):
        outputs = [torch.empty(0) for _ in range(len(kv_chunks))]
    with torch.inference_mode():
        for output_idx, (kv_prompt, seed_tokens) in enumerate(zip(kv_chunks, seed_chunks)):
            request_kv_cache = kv_cache
            if request_kv_cache is None:
                request_kv_cache = allocate_kv_cache(
                    cfg.batch_size,
                    cfg.tokens_per_request,
                    cfg.hidden_size,
                    cfg.dtype,
                    device,
                )
            request_kv_cache[:, : cfg.context_window].copy_(kv_prompt)
            tokens = seed_tokens
            for step in range(cfg.decode_tokens):
                _, decode_logits = model.decode(
                    tokens,
                    kv_cache=request_kv_cache,
                    position=cfg.context_window + step,
                )
                tokens = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)
            outputs[output_idx] = tokens
    return outputs


def _run_torchrun_worker(
    cfg: DisaggConfig,
    *,
    overlap: bool,
    label: str,
    iters: int,
    warmup: int,
) -> None:
    cfg = _apply_profile_overrides(cfg)
    rank, world_size, device = _init_distributed()
    if world_size < 2:
        raise RuntimeError(f"SKIPPED: Requires >= 2 GPUs (found {world_size} GPU)")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"torchrun world_size={world_size} exceeds visible GPUs ({torch.cuda.device_count()})."
        )
    if world_size % 2 != 0:
        raise RuntimeError("world_size must be even (prefill ranks + decode ranks)")

    num_pairs = world_size // 2
    is_prefill = rank < num_pairs
    pair_id = rank if is_prefill else rank - num_pairs
    peer_rank = pair_id + num_pairs if is_prefill else pair_id
    pair_groups = [
        dist.new_group(ranks=[idx, idx + num_pairs]) for idx in range(num_pairs)
    ]
    device_index = 0 if device.index is None else int(device.index)
    comm_stream = torch.cuda.Stream(device=device, priority=1)

    def _barrier() -> None:
        dist.barrier(device_ids=[device_index])

    def _batch_isend(
        kv_cache: torch.Tensor,
        seed: torch.Tensor,
        *,
        ready_event: Optional[torch.cuda.Event] = None,
    ) -> List[dist.Work]:
        with torch.cuda.stream(comm_stream):
            if ready_event is not None:
                comm_stream.wait_event(ready_event)
            ops = [
                dist.P2POp(dist.isend, kv_cache, peer_rank, group=pair_groups[pair_id]),
                dist.P2POp(dist.isend, seed, peer_rank, group=pair_groups[pair_id]),
            ]
            return dist.batch_isend_irecv(ops)

    def _send_blocking(kv_cache: torch.Tensor, seed: torch.Tensor) -> None:
        dist.send(kv_cache, peer_rank, group=pair_groups[pair_id])
        dist.send(seed, peer_rank, group=pair_groups[pair_id])

    def _batch_irecv(kv_buf: torch.Tensor, seed_buf: torch.Tensor) -> List[dist.Work]:
        with torch.cuda.stream(comm_stream):
            ops = [
                dist.P2POp(dist.irecv, kv_buf, peer_rank, group=pair_groups[pair_id]),
                dist.P2POp(dist.irecv, seed_buf, peer_rank, group=pair_groups[pair_id]),
            ]
            return dist.batch_isend_irecv(ops)

    def _recv_blocking(kv_buf: torch.Tensor, seed_buf: torch.Tensor) -> None:
        dist.recv(kv_buf, peer_rank, group=pair_groups[pair_id])
        dist.recv(seed_buf, peer_rank, group=pair_groups[pair_id])

    def _wait_handles(handles: List[dist.Work]) -> None:
        for req in handles:
            req.wait()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    moe_cfg = _build_moe_config(cfg)
    model = SimpleMoEGPT(moe_cfg, device=device).eval()

    prompts: Optional[torch.Tensor] = None
    send_kv_bufs: List[torch.Tensor] = []
    send_seed_bufs: List[torch.Tensor] = []
    if is_prefill:
        prompts = torch.randint(
            0,
            cfg.vocab_size,
            (cfg.requests_per_rank, cfg.batch_size, cfg.context_window),
            device=device,
            dtype=torch.long,
        )
        if overlap:
            send_kv_bufs = [
                torch.empty(
                    (cfg.batch_size, cfg.context_window, cfg.hidden_size),
                    device=device,
                    dtype=cfg.dtype,
                )
                for _ in range(cfg.requests_per_rank)
            ]
            send_seed_bufs = [
                torch.empty(
                    (cfg.batch_size, 1),
                    device=device,
                    dtype=torch.long,
                )
                for _ in range(cfg.requests_per_rank)
            ]

    recv_kv_bufs: List[torch.Tensor] = []
    recv_seed_bufs: List[torch.Tensor] = []
    ready_events = (
        [torch.cuda.Event() for _ in range(cfg.requests_per_rank)]
        if is_prefill and overlap
        else []
    )
    max_inflight = max(1, min(8, cfg.requests_per_rank))
    prefill_pending_slots: List[Optional[List[dist.Work]]] = (
        [None] * max_inflight if is_prefill and overlap else []
    )
    recv_pending_slots: List[Optional[List[dist.Work]]] = (
        [None] * cfg.requests_per_rank if (not is_prefill and overlap) else []
    )
    decode_kv_cache: Optional[torch.Tensor] = None
    decode_outputs: List[torch.Tensor] = []
    if not is_prefill:
        recv_kv_bufs = [
            torch.empty(
                (cfg.batch_size, cfg.context_window, cfg.hidden_size),
                device=device,
                dtype=cfg.dtype,
            )
            for _ in range(cfg.requests_per_rank)
        ]
        recv_seed_bufs = [
            torch.empty(
                (cfg.batch_size, 1),
                device=device,
                dtype=torch.long,
            )
            for _ in range(cfg.requests_per_rank)
        ]
        decode_kv_cache = allocate_kv_cache(
            cfg.batch_size,
            cfg.tokens_per_request,
            cfg.hidden_size,
            cfg.dtype,
            device,
        )
        decode_outputs = [torch.empty(0) for _ in range(cfg.requests_per_rank)]

    def run_iteration() -> List[torch.Tensor]:
        if is_prefill:
            if overlap:
                pending = prefill_pending_slots
                pending_count = 0
                pending_read_idx = 0
                pending_write_idx = 0
                prefill_stream = torch.cuda.current_stream(device)
                with torch.inference_mode():
                    for req_idx in range(cfg.requests_per_rank):
                        request_prompt = prompts[req_idx]
                        hidden, logits = model.prefill(request_prompt)
                        seed_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                        send_kv_bufs[req_idx].copy_(hidden)
                        send_seed_bufs[req_idx].copy_(seed_tokens)
                        ready = ready_events[req_idx]
                        ready.record(prefill_stream)
                        handles = _batch_isend(
                            send_kv_bufs[req_idx],
                            send_seed_bufs[req_idx],
                            ready_event=ready,
                        )
                        pending[pending_write_idx] = handles
                        pending_write_idx = (pending_write_idx + 1) % max_inflight
                        pending_count += 1
                        if pending_count >= max_inflight:
                            oldest = pending[pending_read_idx]
                            if oldest is None:
                                raise RuntimeError("Missing pending send handle")
                            _wait_handles(oldest)
                            pending[pending_read_idx] = None
                            pending_read_idx = (pending_read_idx + 1) % max_inflight
                            pending_count -= 1
                for _ in range(pending_count):
                    handles = pending[pending_read_idx]
                    if handles is None:
                        raise RuntimeError("Missing pending send handle")
                    _wait_handles(handles)
                    pending[pending_read_idx] = None
                    pending_read_idx = (pending_read_idx + 1) % max_inflight
            else:
                kv_chunks, seed_chunks = _run_prefill(cfg, model, prompts)
                for kv_prompt, seed_tokens in zip(kv_chunks, seed_chunks):
                    _send_blocking(kv_prompt, seed_tokens)
                    torch.cuda.synchronize(device)
                    # Naive handoff: sync per request to keep baseline fully serialized.
                    _barrier()
            return []

        if overlap:
            if not recv_kv_bufs or not recv_seed_bufs or decode_kv_cache is None:
                raise RuntimeError("Overlap buffers not initialized")
            outputs = decode_outputs
            if len(outputs) != cfg.requests_per_rank:
                raise RuntimeError("Decode output slots not initialized")
            pending = recv_pending_slots
            pending[0] = _batch_irecv(recv_kv_bufs[0], recv_seed_bufs[0])
            with torch.inference_mode():
                for req_idx in range(cfg.requests_per_rank):
                    next_idx = req_idx + 1
                    if next_idx < cfg.requests_per_rank:
                        pending[next_idx] = _batch_irecv(
                            recv_kv_bufs[next_idx],
                            recv_seed_bufs[next_idx],
                        )
                    handles = pending[req_idx]
                    if handles is None:
                        raise RuntimeError("Missing receive handle in overlap pipeline")
                    _wait_handles(handles)
                    pending[req_idx] = None
                    decode_kv_cache[:, : cfg.context_window].copy_(recv_kv_bufs[req_idx])
                    tokens = recv_seed_bufs[req_idx]
                    for step in range(cfg.decode_tokens):
                        _, decode_logits = model.decode(
                            tokens,
                            kv_cache=decode_kv_cache,
                            position=cfg.context_window + step,
                        )
                        tokens = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)
                    outputs[req_idx] = tokens
            return outputs

        if not recv_kv_bufs or not recv_seed_bufs or decode_kv_cache is None:
            raise RuntimeError("Decode receive buffers not initialized")
        for req_idx in range(cfg.requests_per_rank):
            kv_buf = recv_kv_bufs[req_idx]
            seed_buf = recv_seed_bufs[req_idx]
            _recv_blocking(kv_buf, seed_buf)
            torch.cuda.synchronize(device)
            # Naive handoff: sync per request to keep baseline fully serialized.
            _barrier()
        decoded = _run_decode(
            cfg,
            model,
            recv_kv_bufs,
            recv_seed_bufs,
            device,
            kv_cache=decode_kv_cache,
            outputs=decode_outputs,
        )
        return decoded

    _barrier()
    torch.cuda.synchronize(device)

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
        total_requests = cfg.requests_per_rank * num_pairs * cfg.batch_size
        tokens_per_iter = total_requests * cfg.tokens_per_request
        tokens_per_s = tokens_per_iter * (max(iters, 1) / max(elapsed, 1e-9))
        time_per_iter_ms = (elapsed / max(iters, 1)) * 1000.0
        print(f"rank0 {label} tokens/s: {tokens_per_s:.2f} tokens/s")
        print(f"rank0 {label} time_per_iter_ms: {time_per_iter_ms:.3f}")

    dist.destroy_process_group()


class _DisaggregatedInferenceMultiGPUBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Shared multi-GPU disaggregated inference harness."""

    multi_gpu_required = True
    ncu_env_overrides = {
        "AISP_NCU_PROFILE_REQUESTS": "4",
        "AISP_NCU_PROFILE_CONTEXT": "256",
        "AISP_NCU_PROFILE_DECODE": "8",
        "AISP_NCU_PROFILE_BATCH": "1",
    }

    def __init__(self, *, overlap: bool, label: str) -> None:
        super().__init__()
        self.cfg = _apply_profile_overrides(DisaggConfig())
        self.world_size = _resolve_world_size()
        if self.world_size % 2 != 0:
            raise RuntimeError("world_size must be even for disaggregated inference")
        self.num_pairs = self.world_size // 2
        self.overlap = bool(overlap)
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

        total_requests = self.cfg.requests_per_rank * self.num_pairs * self.cfg.batch_size
        tokens_per_iter = total_requests * self.cfg.tokens_per_request
        self.register_workload_metadata(
            requests_per_iteration=float(total_requests),
            tokens_per_iteration=float(tokens_per_iter),
        )

    def _allocate_output_buffer(self) -> torch.Tensor:
        shape = (
            self.num_pairs * self.cfg.requests_per_rank * self.cfg.batch_size,
            1,
        )
        try:
            return torch.empty(shape, device="cpu", dtype=torch.long, pin_memory=True)
        except RuntimeError:
            return torch.empty(shape, device="cpu", dtype=torch.long)

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for multi-GPU disaggregation")
        if torch.cuda.device_count() < self.world_size:
            raise RuntimeError(
                f"SKIPPED: requires >= {self.world_size} GPUs (found {torch.cuda.device_count()})"
            )

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        moe_cfg = _build_moe_config(self.cfg)
        self._pairs = []
        total_params = 0
        for pair_id in range(self.num_pairs):
            prefill_device = torch.device(f"cuda:{pair_id}")
            decode_device = torch.device(f"cuda:{pair_id + self.num_pairs}")
            prefill_model = SimpleMoEGPT(moe_cfg, device=prefill_device).eval()
            decode_model = SimpleMoEGPT(moe_cfg, device=decode_device).eval()
            decode_model.load_state_dict(prefill_model.state_dict())
            decode_kv_cache = allocate_kv_cache(
                self.cfg.batch_size,
                self.cfg.tokens_per_request,
                self.cfg.hidden_size,
                self.cfg.dtype,
                decode_device,
            )
            prefill_kv_chunks = [
                torch.empty(
                    self.cfg.batch_size,
                    self.cfg.context_window,
                    self.cfg.hidden_size,
                    device=prefill_device,
                    dtype=self.cfg.dtype,
                )
                for _ in range(self.cfg.requests_per_rank)
            ]
            prefill_seed_chunks = [
                torch.empty(
                    self.cfg.batch_size,
                    1,
                    device=prefill_device,
                    dtype=torch.long,
                )
                for _ in range(self.cfg.requests_per_rank)
            ]
            transfer_kv_chunks = [
                torch.empty(
                    self.cfg.batch_size,
                    self.cfg.context_window,
                    self.cfg.hidden_size,
                    device=decode_device,
                    dtype=self.cfg.dtype,
                )
                for _ in range(self.cfg.requests_per_rank)
            ]
            transfer_seed_chunks = [
                torch.empty(
                    self.cfg.batch_size,
                    1,
                    device=decode_device,
                    dtype=torch.long,
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
            prompts = torch.randint(
                0,
                self.cfg.vocab_size,
                (self.cfg.requests_per_rank, self.cfg.batch_size, self.cfg.context_window),
                device=prefill_device,
                dtype=torch.long,
            )
            total_params += sum(p.numel() for p in prefill_model.parameters())
            total_params += sum(p.numel() for p in decode_model.parameters())
            self._pairs.append(
                _LocalPair(
                    prefill_device=prefill_device,
                    decode_device=decode_device,
                    prefill_model=prefill_model,
                    decode_model=decode_model,
                    prompts=prompts,
                    decode_kv_cache=decode_kv_cache,
                    decode_outputs=[torch.empty(0) for _ in range(self.cfg.requests_per_rank)],
                    prefill_kv_chunks=prefill_kv_chunks,
                    prefill_seed_chunks=prefill_seed_chunks,
                    transfer_kv_chunks=transfer_kv_chunks,
                    transfer_seed_chunks=transfer_seed_chunks,
                    transfer_slots=transfer_slots,
                    transfer_slot_counts=transfer_slot_counts,
                    expected_transfer_slot_counts=expected_transfer_slot_counts,
                )
            )

        self._param_count = total_params
        if not self._pairs:
            raise RuntimeError("Failed to initialize prompts for verification")
        self._verify_prompt = self._pairs[0].prompts
        self._pending_outputs = [
            torch.empty(0) for _ in range(self.num_pairs * self.cfg.requests_per_rank)
        ]
        self._pending_output_count = len(self._pending_outputs)
        self._expected_output_count = self.num_pairs * self.cfg.requests_per_rank
        self._output_buffer = self._allocate_output_buffer()
        meta_dtype = torch.float32
        self._metadata_inputs = {
            "decode_tokens": torch.zeros((self.cfg.decode_tokens,), dtype=meta_dtype),
            "hidden_size": torch.zeros((self.cfg.hidden_size,), dtype=meta_dtype),
            "num_layers": torch.zeros((self.cfg.num_layers,), dtype=meta_dtype),
            "num_experts": torch.zeros((self.cfg.num_experts,), dtype=meta_dtype),
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
                kv_chunks, seed_chunks = _run_prefill(
                    self.cfg,
                    pair.prefill_model,
                    pair.prompts,
                    pair.prefill_kv_chunks,
                    pair.prefill_seed_chunks,
                )
                if pair.transfer_slot_counts != pair.expected_transfer_slot_counts:
                    raise RuntimeError("Transfer chunk slots not initialized")
                for req_idx, transfer_kv, transfer_seed in pair.transfer_slots:
                    transfer_kv.copy_(
                        kv_chunks[req_idx],
                        non_blocking=self.overlap,
                    )
                    transfer_seed.copy_(
                        seed_chunks[req_idx],
                        non_blocking=self.overlap,
                    )
                decoded = _run_decode(
                    self.cfg,
                    pair.decode_model,
                    pair.transfer_kv_chunks,
                    pair.transfer_seed_chunks,
                    pair.decode_device,
                    kv_cache=pair.decode_kv_cache,
                    outputs=pair.decode_outputs,
                )
                for decoded_tokens in decoded:
                    outputs[output_idx] = decoded_tokens
                    output_idx += 1

        self._pending_outputs = outputs
        self._output = None

    def capture_verification_payload(self) -> None:
        if self._verify_prompt is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._output is None:
            if not self._pending_outputs:
                raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
            if self._output_buffer is None:
                raise RuntimeError("Output buffer not initialized")
            output_offset = 0
            for output in self._pending_outputs:
                output_rows = int(output.shape[0])
                self._output_buffer[output_offset : output_offset + output_rows].copy_(
                    output,
                    non_blocking=False,
                )
                output_offset += output_rows
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
                "num_experts": self._metadata_inputs["num_experts"],
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
                    (0, self.num_pairs - 1),
                    (self.num_pairs, self.world_size - 1),
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
        if self._output is None:
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
            # NCU application replay can hang on this workload; default to kernel replay.
            ncu_replay_mode="kernel",
            ncu_replay_mode_override=True,
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        self._prepare_verification_payload()
        master_port = os.environ.get("MASTER_PORT", "29515")
        module = inspect.getmodule(self.__class__)
        script_path = Path(module.__file__).resolve() if module and module.__file__ else Path(__file__).resolve()
        return TorchrunLaunchSpec(
            script_path=script_path,
            script_args=[],
            env={
                "OMP_NUM_THREADS": "1",
                "MASTER_PORT": master_port,
                "TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING": "0",
            },
            parse_rank0_only=True,
            multi_gpu_required=True,
            name=self.label,
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
            },
        )



class BaselineDisaggregatedInferenceMultiGPUBenchmark(_DisaggregatedInferenceMultiGPUBenchmark):
    """Serialized prefill then decode across multi-GPU ranks."""

    story_metadata = {
        "pair_role": "canonical",
        "chapter_alignment": "native",
        "chapter_native_exemplar": True,
        "timed_launch_mode": "torchrun_multi_gpu",
        "verification_mode": "local_multi_device_surrogate",
        "shared_harness_layout": "baseline_owned_shared_base",
        "shared_harness_owner": "ch15/baseline_disaggregated_inference_multigpu.py",
        "execution_pattern": "serialized_prefill_then_decode",
    }

    def __init__(self) -> None:
        super().__init__(overlap=False, label="baseline_disaggregated_inference_multigpu")


def get_benchmark() -> BaseBenchmark:
    return BaselineDisaggregatedInferenceMultiGPUBenchmark()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _run_torchrun_worker(
        DisaggConfig(),
        overlap=False,
        label="baseline_disaggregated_inference_multigpu",
        iters=int(args.iters),
        warmup=int(args.warmup),
    )
