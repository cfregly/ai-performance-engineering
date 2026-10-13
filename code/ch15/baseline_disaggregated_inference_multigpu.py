"""Baseline disaggregated inference benchmark (multi-GPU torchrun pipeline).

Chapter 15: Disaggregated Inference

This benchmark models a disaggregated prefill/decode pipeline across multiple GPUs.
Baseline behavior is serialized: prefill completes for all requests before decode starts.
The optimized pair overlaps prefill and decode via pipelined transfers.

Torchrun launches verify the timed ranks. Each decode rank starts from its
prefill rank's weights and publishes its final timed outputs. Its prefill rank
recomputes prefill and decode for the same prompts after timing ends. The parent
requires the two to match exactly, then compares baseline and optimized decode
outputs. In-process runs (setup() and benchmark_fn()) verify the local
multi-device path instead.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.distributed as dist

from core.benchmark.verification import InputSignature, PrecisionFlags  # noqa: E402
from core.benchmark.verification_mixin import VerificationPayloadMixin  # noqa: E402
from core.harness import benchmark_worker
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
from core.profiling.nvtx_helper import nvtx_range
from core.utils.worker_seed import apply_worker_seed

BASELINE_PROFILE_NVTX_RANGE = "compute_kernel:disaggregated_inference_serialized_pipeline"
OPTIMIZED_PROFILE_NVTX_RANGE = "compute_kernel:disaggregated_inference_overlapped_pipeline"

DECODE_OUTPUT_TOLERANCE = (0.0, 0.0)
TORCHRUN_RESULT_CALLBACK = "consume_torchrun_rank_results"
TORCHRUN_RESULT_DIR_ENV = "AISP_CH15_DISAGG_RESULT_DIR"
TORCHRUN_RESULT_TOKEN_ENV = "AISP_CH15_DISAGG_RESULT_TOKEN"
TORCHRUN_RESULT_LABEL_ENV = "AISP_CH15_DISAGG_RESULT_LABEL"
# The harness sets this only for launches that bind a result callback.
TORCHRUN_LAUNCH_WALL_NS_ENV = "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS"
TORCHRUN_RESULT_SCHEMA = "aisp.ch15-disaggregated-inference-multigpu.rank-result.v1"
_RESULT_DTYPES = {
    str(torch.bfloat16): torch.bfloat16,
    str(torch.float16): torch.float16,
    str(torch.float32): torch.float32,
}


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


def _build_pair_groups(num_pairs: int) -> List[Any]:
    """Create one process group per (prefill rank, decode rank) pair."""
    if num_pairs <= 0:
        raise ValueError(f"num_pairs must be positive, got {num_pairs}")
    return [dist.new_group(ranks=[idx, idx + num_pairs]) for idx in range(num_pairs)]


def _share_prefill_weights(model: SimpleMoEGPT, *, prefill_rank: int, group: Any) -> None:
    """Overwrite the decode rank's weights with its prefill rank's weights.

    Both ranks of a pair call this with the pair's group. prefill_rank is the
    global rank that sends. The local path does the same with load_state_dict().
    The post-timing check compares a decode rank's output with its prefill
    rank's recompute, so both ranks must hold the same weights. Matching seeds
    alone gives equal weights only when both GPUs run the same initialization
    kernels.
    """
    with torch.no_grad():
        for tensor in model.state_dict().values():
            dist.broadcast(tensor, src=prefill_rank, group=group)


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


def _kv_position_weights(context_window: int, device: torch.device) -> torch.Tensor:
    """Return fp32 pooling weights proportional to t + 1 that sum to one."""
    if context_window <= 0:
        raise ValueError(f"context_window must be positive, got {context_window}")
    return torch.arange(
        1,
        context_window + 1,
        device=device,
        dtype=torch.float32,
    ).mul_(2.0 / (context_window * (context_window + 1)))


def _transferred_kv_context(
    kv_cache: torch.Tensor,
    context_window: int,
    *,
    position_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool the transferred prompt state into decode context and a verification signal.

    Position t of the window gets weight proportional to t + 1, so later prompt
    tokens count more and the pooled context depends on token order. A plain
    mean would let a transfer that permutes positions inside the window pass
    verification unchanged. Per-request loops pass weights from
    _kv_position_weights() so they are built once per call, not per request.
    """
    if kv_cache.ndim != 3:
        raise ValueError("kv_cache must have shape (batch, sequence, hidden)")
    if context_window <= 0 or context_window > kv_cache.size(1):
        raise ValueError(
            f"context_window must be in [1, {kv_cache.size(1)}], got {context_window}"
        )
    if position_weights is None:
        position_weights = _kv_position_weights(context_window, kv_cache.device)
    elif (
        position_weights.shape != (context_window,)
        or position_weights.dtype != torch.float32
        or position_weights.device != kv_cache.device
    ):
        raise ValueError(
            "position_weights must be a float32 tensor of shape "
            f"({context_window},) on {kv_cache.device}"
        )
    # Mixed-dtype multiply promotes to fp32 in one kernel and one temporary.
    context_fp32 = (
        kv_cache[:, :context_window] * position_weights.view(1, context_window, 1)
    ).sum(dim=1, keepdim=True)
    return context_fp32.to(dtype=kv_cache.dtype), context_fp32


def _write_decode_verification_output(
    output: torch.Tensor,
    final_tokens: torch.Tensor,
    context_fp32: torch.Tensor,
) -> torch.Tensor:
    """Record final tokens plus a transfer-sensitive prompt-context fingerprint."""
    if final_tokens.ndim != 2 or final_tokens.size(1) != 1:
        raise ValueError("final_tokens must have shape (batch, 1)")
    batch_size = final_tokens.size(0)
    if context_fp32.ndim != 3 or context_fp32.size(0) != batch_size or context_fp32.size(1) != 1:
        raise ValueError("context_fp32 must have shape (batch, 1, hidden)")
    if context_fp32.device != final_tokens.device:
        raise ValueError("final_tokens and context_fp32 must be on the same device")
    hidden_size = context_fp32.size(-1)
    expected_shape = (batch_size, hidden_size + 1)
    if (
        output.shape != expected_shape
        or output.device != final_tokens.device
        or output.dtype != torch.float32
    ):
        output = torch.empty(expected_shape, device=final_tokens.device, dtype=torch.float32)
    output[:, :hidden_size].copy_(context_fp32.squeeze(1))
    output[:, hidden_size:].copy_(final_tokens)
    return output


def _allocate_decode_outputs(cfg: DisaggConfig, device: torch.device) -> List[torch.Tensor]:
    """Preallocate one transfer-sensitive verification result per request."""
    shape = (cfg.batch_size, cfg.hidden_size + 1)
    return [
        torch.empty(shape, device=device, dtype=torch.float32)
        for _ in range(cfg.requests_per_rank)
    ]


def _collect_decode_verification_outputs(
    output_buffer: torch.Tensor,
    outputs: List[torch.Tensor],
    *,
    batch_size: int,
    hidden_size: int,
) -> torch.Tensor:
    """Copy complete per-request verification results into the host payload."""
    expected_output_shape = (batch_size, hidden_size + 1)
    expected_buffer_shape = (len(outputs) * batch_size, hidden_size + 1)
    if output_buffer.shape != expected_buffer_shape or output_buffer.dtype != torch.float32:
        raise RuntimeError(
            "Decode verification buffer has shape/dtype "
            f"{tuple(output_buffer.shape)}/{output_buffer.dtype}; expected "
            f"{expected_buffer_shape}/{torch.float32}"
        )

    output_offset = 0
    for output_idx, output in enumerate(outputs):
        if output.shape != expected_output_shape or output.dtype != torch.float32:
            raise RuntimeError(
                f"Decode output {output_idx} has shape/dtype "
                f"{tuple(output.shape)}/{output.dtype}; expected "
                f"{expected_output_shape}/{torch.float32}"
            )
        output_end = output_offset + batch_size
        output_buffer[output_offset:output_end].copy_(output, non_blocking=False)
        output_offset = output_end
    if output_offset != output_buffer.size(0):
        raise RuntimeError("Decode verification output collection was incomplete")
    return output_buffer


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
        outputs = _allocate_decode_outputs(cfg, device)
    with torch.inference_mode():
        position_weights = _kv_position_weights(cfg.context_window, device)
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
            kv_context, context_fp32 = _transferred_kv_context(
                request_kv_cache,
                cfg.context_window,
                position_weights=position_weights,
            )
            tokens = seed_tokens
            for step in range(cfg.decode_tokens):
                _, decode_logits = model.decode(
                    tokens,
                    kv_cache=request_kv_cache,
                    position=cfg.context_window + step,
                    kv_context=kv_context,
                )
                tokens = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)
            tokens = _write_decode_verification_output(
                outputs[output_idx],
                tokens,
                context_fp32,
            )
            outputs[output_idx] = tokens
    return outputs


def _build_prefill_reference_output(
    cfg: DisaggConfig,
    model: SimpleMoEGPT,
    prompts: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Recompute prefill and decode for every request on the sending rank, after timing."""
    expected_shape = (cfg.requests_per_rank, cfg.batch_size, cfg.context_window)
    if tuple(prompts.shape) != expected_shape:
        raise RuntimeError(
            f"Reference prompts have shape {tuple(prompts.shape)}, expected {expected_shape}"
        )
    kv_chunks, seed_chunks = _run_prefill(cfg, model, prompts)
    return torch.cat(_run_decode(cfg, model, kv_chunks, seed_chunks, device), dim=0)


@dataclass(frozen=True)
class _TorchrunResultContract:
    """Workload identity that every torchrun rank result must repeat."""

    label: str
    world_size: int
    vocab_size: int
    hidden_size: int
    ffn_size: int
    num_layers: int
    num_moe_layers: int
    num_experts: int
    top_k: int
    batch_size: int
    requests_per_rank: int
    context_window: int
    decode_tokens: int
    dtype: str
    iterations: int
    warmup: int

    @property
    def num_pairs(self) -> int:
        return self.world_size // 2

    def validate(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("Disaggregated result label must be a non-empty string")
        for name in (
            "world_size",
            "vocab_size",
            "hidden_size",
            "ffn_size",
            "num_layers",
            "num_experts",
            "top_k",
            "batch_size",
            "requests_per_rank",
            "context_window",
            "decode_tokens",
            "iterations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Disaggregated result {name} must be a positive integer")
        for name in ("num_moe_layers", "warmup"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Disaggregated result {name} must be a non-negative integer")
        if self.world_size % 2 != 0:
            raise ValueError("Disaggregated result world_size must be even")
        if self.dtype not in _RESULT_DTYPES:
            raise ValueError(f"Unsupported disaggregated result dtype: {self.dtype!r}")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Any) -> _TorchrunResultContract:
        if not isinstance(payload, dict):
            raise TypeError("Disaggregated result contract must be a dictionary")
        try:
            contract = cls(**payload)
        except TypeError as exc:
            raise ValueError("Invalid disaggregated result contract fields") from exc
        contract.validate()
        return contract

    def disagg_config(self) -> DisaggConfig:
        return DisaggConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            ffn_size=self.ffn_size,
            num_layers=self.num_layers,
            num_moe_layers=self.num_moe_layers,
            num_experts=self.num_experts,
            top_k=self.top_k,
            batch_size=self.batch_size,
            requests_per_rank=self.requests_per_rank,
            context_window=self.context_window,
            decode_tokens=self.decode_tokens,
            dtype=_RESULT_DTYPES[self.dtype],
        )


def _make_torchrun_result_contract(
    cfg: DisaggConfig,
    *,
    label: str,
    world_size: int,
    iterations: int,
    warmup: int,
) -> _TorchrunResultContract:
    contract = _TorchrunResultContract(
        label=label,
        world_size=int(world_size),
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        ffn_size=cfg.ffn_size,
        num_layers=cfg.num_layers,
        num_moe_layers=cfg.num_moe_layers,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        batch_size=cfg.batch_size,
        requests_per_rank=cfg.requests_per_rank,
        context_window=cfg.context_window,
        decode_tokens=cfg.decode_tokens,
        dtype=str(cfg.dtype),
        iterations=int(iterations),
        warmup=int(warmup),
    )
    contract.validate()
    return contract


def _torchrun_input_signature(contract: _TorchrunResultContract, *, tf32: bool) -> InputSignature:
    """Describe the same workload fields as the in-process verification payload."""
    rows = contract.num_pairs * contract.requests_per_rank * contract.batch_size
    dtype = _RESULT_DTYPES[contract.dtype]
    meta_model = SimpleMoEGPT(
        _build_moe_config(contract.disagg_config()),
        device=torch.device("meta"),
    )
    per_model_parameters = sum(p.numel() for p in meta_model.parameters())
    return InputSignature(
        shapes={
            "prompt": (contract.requests_per_rank, contract.batch_size, contract.context_window),
            "decode_tokens": (contract.decode_tokens,),
            "hidden_size": (contract.hidden_size,),
            "num_layers": (contract.num_layers,),
            "num_experts": (contract.num_experts,),
            "output": (rows, contract.hidden_size + 1),
        },
        dtypes={
            "prompt": str(torch.int64),
            "decode_tokens": str(torch.float32),
            "hidden_size": str(torch.float32),
            "num_layers": str(torch.float32),
            "num_experts": str(torch.float32),
            "output": str(torch.float32),
        },
        batch_size=rows,
        parameter_count=contract.world_size * per_model_parameters,
        precision_flags=PrecisionFlags(
            fp16=dtype == torch.float16,
            bf16=dtype == torch.bfloat16,
            tf32=tf32,
        ),
        world_size=contract.world_size,
        pipeline_stages=2,
        pipeline_stage_boundaries=[
            (0, contract.num_pairs - 1),
            (contract.num_pairs, contract.world_size - 1),
        ],
        per_rank_batch_size=contract.requests_per_rank,
        collective_type="send_recv",
    )


def _check_rank_result_tensors(
    contract: _TorchrunResultContract,
    *,
    rank: int,
    output: Any,
    prompts: Any,
) -> None:
    output_shape = (contract.requests_per_rank * contract.batch_size, contract.hidden_size + 1)
    if (
        not isinstance(output, torch.Tensor)
        or output.dtype != torch.float32
        or tuple(output.shape) != output_shape
        or not bool(torch.isfinite(output).all())
    ):
        raise RuntimeError(
            f"Rank {rank} verification output must be finite float32 with shape {output_shape}"
        )
    if rank != 0:
        if prompts is not None:
            raise RuntimeError(f"Only rank 0 publishes prompts, but rank {rank} did")
        return
    prompt_shape = (contract.requests_per_rank, contract.batch_size, contract.context_window)
    if (
        not isinstance(prompts, torch.Tensor)
        or prompts.dtype != torch.int64
        or tuple(prompts.shape) != prompt_shape
        or bool((prompts < 0).any())
        or bool((prompts >= contract.vocab_size).any())
    ):
        raise RuntimeError(
            f"Rank 0 must publish int64 prompts with shape {prompt_shape} "
            f"and token ids below {contract.vocab_size}"
        )


def _write_torchrun_rank_result(
    contract: _TorchrunResultContract,
    *,
    rank: int,
    output: torch.Tensor,
    prompts: Optional[torch.Tensor],
) -> bool:
    """Atomically publish one rank's verification result for a callback-bound launch."""
    launch_wall_ns = os.environ.get(TORCHRUN_LAUNCH_WALL_NS_ENV)
    if not launch_wall_ns:
        return False
    result_dir_value = os.environ.get(TORCHRUN_RESULT_DIR_ENV)
    token = os.environ.get(TORCHRUN_RESULT_TOKEN_ENV)
    expected_label = os.environ.get(TORCHRUN_RESULT_LABEL_ENV)
    if not all((result_dir_value, token, expected_label)):
        raise RuntimeError("Disaggregated torchrun result environment is incomplete")
    contract.validate()
    if expected_label != contract.label:
        raise RuntimeError(
            f"Result label {expected_label!r} does not match worker label {contract.label!r}"
        )
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < contract.world_size:
        raise RuntimeError(f"Invalid disaggregated result rank: {rank!r}")
    _check_rank_result_tensors(contract, rank=rank, output=output, prompts=prompts)
    result_dir = Path(cast(str, result_dir_value))
    if result_dir.is_symlink() or not result_dir.is_dir():
        raise RuntimeError(f"Disaggregated result directory must be a regular directory: {result_dir}")
    payload = {
        "schema": TORCHRUN_RESULT_SCHEMA,
        "token": token,
        "label": contract.label,
        "rank": rank,
        "role": "prefill" if rank < contract.num_pairs else "decode",
        "contract": contract.to_dict(),
        "launch_wall_ns": int(launch_wall_ns),
        "created_wall_ns": time.time_ns(),
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "output": output.detach().to(device="cpu").contiguous(),
        "prompts": None if prompts is None else prompts.detach().to(device="cpu").contiguous(),
    }
    temporary = result_dir / f".rank-{rank}-{os.getpid()}.tmp"
    destination = result_dir / f"rank-{rank}.pt"
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return True


def _run_torchrun_worker(
    cfg: DisaggConfig,
    *,
    overlap: bool,
    label: str,
    iters: int,
    warmup: int,
    profile_nvtx_range: str,
    seed: int,
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
    pair_groups = _build_pair_groups(num_pairs)
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

    apply_worker_seed(seed)

    moe_cfg = _build_moe_config(cfg)
    model = SimpleMoEGPT(moe_cfg, device=device).eval()
    _share_prefill_weights(model, prefill_rank=pair_id, group=pair_groups[pair_id])

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
    decode_position_weights: Optional[torch.Tensor] = None
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
        decode_position_weights = _kv_position_weights(cfg.context_window, device)
        decode_outputs = _allocate_decode_outputs(cfg, device)

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
            if (
                not recv_kv_bufs
                or not recv_seed_bufs
                or decode_kv_cache is None
                or decode_position_weights is None
            ):
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
                    kv_context, context_fp32 = _transferred_kv_context(
                        decode_kv_cache,
                        cfg.context_window,
                        position_weights=decode_position_weights,
                    )
                    tokens = recv_seed_bufs[req_idx]
                    for step in range(cfg.decode_tokens):
                        _, decode_logits = model.decode(
                            tokens,
                            kv_cache=decode_kv_cache,
                            position=cfg.context_window + step,
                            kv_context=kv_context,
                        )
                        tokens = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)
                    tokens = _write_decode_verification_output(
                        outputs[req_idx],
                        tokens,
                        context_fp32,
                    )
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

    last_outputs: List[torch.Tensor] = []
    with nvtx_range(profile_nvtx_range, enable=True):
        start = time.perf_counter()
        for _ in range(max(iters, 1)):
            last_outputs = run_iteration()
        torch.cuda.synchronize(device)
        _barrier()
        elapsed = time.perf_counter() - start

    if os.environ.get(TORCHRUN_LAUNCH_WALL_NS_ENV):
        contract = _make_torchrun_result_contract(
            cfg,
            label=label,
            world_size=world_size,
            iterations=iters,
            warmup=warmup,
        )
        if is_prefill:
            if prompts is None:
                raise RuntimeError("Prefill prompts are unavailable for verification")
            rank_output = _build_prefill_reference_output(cfg, model, prompts, device)
        else:
            if len(last_outputs) != cfg.requests_per_rank:
                raise RuntimeError("Timed decode outputs do not cover every request")
            rank_output = torch.cat(last_outputs, dim=0)
        _write_torchrun_rank_result(
            contract,
            rank=rank,
            output=rank_output,
            prompts=prompts if rank == 0 else None,
        )
        _barrier()

    if rank == 0:
        total_requests = cfg.requests_per_rank * num_pairs * cfg.batch_size
        tokens_per_iter = total_requests * cfg.tokens_per_request
        tokens_per_s = tokens_per_iter * (max(iters, 1) / max(elapsed, 1e-9))
        time_per_iter_ms = (elapsed / max(iters, 1)) * 1000.0
        print(f"rank0 {label} tokens/s: {tokens_per_s:.2f} tokens/s")
        print(f"rank0 time_per_iter_ms: {time_per_iter_ms:.9f}")

    dist.destroy_process_group()


class _DisaggregatedInferenceMultiGPUBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Shared multi-GPU disaggregated inference harness."""

    multi_gpu_required = True
    preferred_ncu_replay_mode = "app-range"
    ncu_env_overrides = {
        "AISP_NCU_PROFILE_REQUESTS": "4",
        "AISP_NCU_PROFILE_CONTEXT": "256",
        "AISP_NCU_PROFILE_DECODE": "8",
        "AISP_NCU_PROFILE_BATCH": "1",
    }
    _torchrun_result_context: Optional[Dict[str, Any]] = None
    _torchrun_result_bundle: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        *,
        overlap: bool,
        label: str,
        profile_nvtx_range: str,
        worker_module: str,
    ) -> None:
        super().__init__()
        self.cfg = _apply_profile_overrides(DisaggConfig())
        self.world_size = _resolve_world_size()
        if self.world_size % 2 != 0:
            raise RuntimeError("world_size must be even for disaggregated inference")
        self.num_pairs = self.world_size // 2
        self.overlap = bool(overlap)
        self.label = label
        self.profile_nvtx_range = profile_nvtx_range
        self.worker_module = worker_module
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
            self.cfg.hidden_size + 1,
        )
        try:
            return torch.empty(shape, device="cpu", dtype=torch.float32, pin_memory=True)
        except RuntimeError:
            return torch.empty(shape, device="cpu", dtype=torch.float32)

    def setup(self) -> None:
        # An in-process run verifies its own local outputs. It must never
        # report results that an earlier torchrun launch left on this instance.
        self._clear_torchrun_rank_results()
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for multi-GPU disaggregation")
        if torch.cuda.device_count() < self.world_size:
            raise RuntimeError(
                f"SKIPPED: requires >= {self.world_size} GPUs (found {torch.cuda.device_count()})"
            )


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
                    decode_outputs=_allocate_decode_outputs(self.cfg, decode_device),
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
            self._output = _collect_decode_verification_outputs(
                self._output_buffer,
                self._pending_outputs,
                batch_size=self.cfg.batch_size,
                hidden_size=self.cfg.hidden_size,
            )
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
            output_tolerance=DECODE_OUTPUT_TOLERANCE,
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
        if self._torchrun_result_context is not None:
            self.require_torchrun_rank_results()
            return
        self.capture_verification_payload()

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
            # Capture the complete selected range without replaying stateful collectives per kernel.
            ncu_replay_mode="app-range",
            ncu_replay_mode_override=True,
            nsys_nvtx_include=[self.profile_nvtx_range],
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        effective_config = config or self.get_config()
        if int(effective_config.nnodes or 1) != 1:
            raise RuntimeError("Disaggregated rank-result transport requires nnodes == 1")
        nproc_per_node = int(effective_config.nproc_per_node or self.world_size)
        if nproc_per_node != self.world_size:
            raise RuntimeError(
                f"Disaggregated rank-result transport expects {self.world_size} ranks, "
                f"got nproc_per_node={nproc_per_node}"
            )
        contract = _make_torchrun_result_contract(
            _apply_profile_overrides(DisaggConfig()),
            label=self.label,
            world_size=nproc_per_node,
            iterations=int(effective_config.iterations or 0),
            warmup=int(effective_config.warmup or 0),
        )
        result_env = self._prepare_torchrun_rank_results(contract)
        master_port = os.environ.get("MASTER_PORT", "29515")
        return TorchrunLaunchSpec(
            timing_source="rank0_time_per_iter_ms",
            timing_iterations_per_sample=contract.iterations,
            script_path=Path(benchmark_worker.__file__).resolve(),
            script_args=["--module", self.worker_module, "--callable", "main", "--"],
            env={
                "OMP_NUM_THREADS": "1",
                "MASTER_PORT": master_port,
                "TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING": "0",
                **result_env,
            },
            parse_rank0_only=True,
            multi_gpu_required=True,
            name=self.label,
            result_callback=TORCHRUN_RESULT_CALLBACK,
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
                "seed": "--seed",
            },
        )

    def _clear_torchrun_rank_results(self) -> None:
        """Drop the launch context and every verification result from a torchrun launch."""
        self._torchrun_result_context = None
        self._torchrun_result_bundle = None
        for attribute in (
            "_subprocess_verify_inputs",
            "_subprocess_verify_output",
            "_subprocess_output_tolerance",
            "_subprocess_input_signature",
        ):
            if attribute in vars(self):
                delattr(self, attribute)

    def _prepare_torchrun_rank_results(self, contract: _TorchrunResultContract) -> Dict[str, str]:
        """Create a fresh result directory and token for one torchrun launch."""
        contract.validate()
        self._clear_torchrun_rank_results()
        result_dir = Path(tempfile.mkdtemp(prefix="aisp-ch15-disagg-result-"))
        token = uuid.uuid4().hex
        self._torchrun_result_context = {
            "result_dir": result_dir,
            "token": token,
            "contract": contract,
            "retention": "pending-rank-results",
        }
        return {
            TORCHRUN_RESULT_DIR_ENV: str(result_dir),
            TORCHRUN_RESULT_TOKEN_ENV: token,
            TORCHRUN_RESULT_LABEL_ENV: contract.label,
        }

    def consume_torchrun_rank_results(
        self,
        *,
        launch_wall_ns: int,
        finish_wall_ns: int,
        returncode: int,
        **_: Any,
    ) -> None:
        """Validate every rank result from this launch and expose the timed decode outputs."""
        context = self._torchrun_result_context
        if context is None:
            raise RuntimeError("Disaggregated rank-result callback has no launch context")
        result_dir = cast(Path, context["result_dir"])
        context["retention"] = "retained-invalid-rank-result"
        if returncode != 0:
            context["retention"] = "retained-child-failure"
            raise RuntimeError(
                "Disaggregated rank results require a clean torchrun exit. "
                f"Artifacts retained at {result_dir}"
            )
        expected = cast(_TorchrunResultContract, context["contract"])
        paths = sorted(result_dir.glob("rank-*.pt"))
        if len(paths) != expected.world_size:
            context["retention"] = "retained-incomplete-rank-quorum"
            raise RuntimeError(
                "Disaggregated rank-result quorum is incomplete: "
                f"expected {expected.world_size}, found {len(paths)}. "
                f"Artifacts retained at {result_dir}"
            )

        payloads: Dict[int, Dict[str, Any]] = {}
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Disaggregated rank result must be a regular file: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or payload.get("schema") != TORCHRUN_RESULT_SCHEMA:
                raise RuntimeError(f"Invalid disaggregated rank-result payload at {path}")
            if payload.get("token") != context["token"]:
                raise RuntimeError(f"Disaggregated rank-result token mismatch at {path}")
            if payload.get("label") != expected.label:
                raise RuntimeError(f"Disaggregated rank-result label mismatch at {path}")
            if _TorchrunResultContract.from_dict(payload.get("contract")) != expected:
                raise RuntimeError(f"Disaggregated rank-result workload mismatch at {path}")
            rank = payload.get("rank")
            if (
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or not 0 <= rank < expected.world_size
                or rank in payloads
            ):
                raise RuntimeError(f"Invalid or duplicate disaggregated rank at {path}")
            if path.name != f"rank-{rank}.pt":
                raise RuntimeError(f"Disaggregated rank-result filename does not match rank at {path}")
            created_wall_ns = payload.get("created_wall_ns")
            if not isinstance(created_wall_ns, int) or not (
                launch_wall_ns <= created_wall_ns <= finish_wall_ns
            ):
                raise RuntimeError(f"Stale disaggregated rank result at {path}")
            if payload.get("launch_wall_ns") != launch_wall_ns:
                raise RuntimeError(f"Disaggregated rank-result launch identity mismatch at {path}")
            expected_role = "prefill" if rank < expected.num_pairs else "decode"
            if payload.get("role") != expected_role:
                raise RuntimeError(f"Disaggregated rank-result role mismatch at {path}")
            if not isinstance(payload.get("tf32"), bool):
                raise RuntimeError(f"Disaggregated rank-result TF32 state is missing at {path}")
            _check_rank_result_tensors(
                expected,
                rank=rank,
                output=payload.get("output"),
                prompts=payload.get("prompts"),
            )
            payloads[rank] = payload

        tf32_states = {bool(payload["tf32"]) for payload in payloads.values()}
        if len(tf32_states) != 1:
            raise RuntimeError("Disaggregated ranks ran with different TF32 settings")
        timed_outputs: List[torch.Tensor] = []
        for pair_id in range(expected.num_pairs):
            decode_rank = pair_id + expected.num_pairs
            timed = cast(torch.Tensor, payloads[decode_rank]["output"])
            try:
                torch.testing.assert_close(
                    timed,
                    cast(torch.Tensor, payloads[pair_id]["output"]),
                    rtol=DECODE_OUTPUT_TOLERANCE[0],
                    atol=DECODE_OUTPUT_TOLERANCE[1],
                )
            except AssertionError as exc:
                raise RuntimeError(
                    f"Timed decode output on rank {decode_rank} differs from the reference "
                    f"that prefill rank {pair_id} recomputed, so the KV handoff or the "
                    f"decode path changed the result. Artifacts retained at {result_dir}"
                ) from exc
            timed_outputs.append(timed)

        signature = _torchrun_input_signature(expected, tf32=tf32_states.pop())
        signature_errors = signature.validate(strict=True)
        if signature_errors:
            raise RuntimeError(
                f"Invalid disaggregated rank-result input signature: {signature_errors[0]}"
            )
        verify_output = torch.cat(timed_outputs, dim=0)
        self._subprocess_verify_inputs = {
            "prompt": cast(torch.Tensor, payloads[0]["prompts"]),
            "decode_tokens": torch.zeros((expected.decode_tokens,), dtype=torch.float32),
            "hidden_size": torch.zeros((expected.hidden_size,), dtype=torch.float32),
            "num_layers": torch.zeros((expected.num_layers,), dtype=torch.float32),
            "num_experts": torch.zeros((expected.num_experts,), dtype=torch.float32),
        }
        self._subprocess_verify_output = verify_output
        self._subprocess_output_tolerance = DECODE_OUTPUT_TOLERANCE
        self._subprocess_input_signature = signature
        self._torchrun_result_bundle = {
            "contract": expected,
            "timed_output": verify_output,
        }
        try:
            shutil.rmtree(result_dir)
        except OSError:
            context["retention"] = "retained-cleanup-failure"
        else:
            context["retention"] = "cleaned-after-success"

    def require_torchrun_rank_results(self) -> None:
        if self._torchrun_result_bundle is None:
            context = self._torchrun_result_context
            retained = context.get("result_dir") if context else "unavailable"
            raise RuntimeError(
                "Disaggregated verification requires fresh results from every timed "
                f"torchrun rank. Retained path: {retained}"
            )

    def get_verify_inputs(self) -> Dict[str, torch.Tensor]:
        if self._torchrun_result_bundle is not None:
            return dict(self._subprocess_verify_inputs)
        return super().get_verify_inputs()

    def get_verify_output(self) -> torch.Tensor:
        if self._torchrun_result_bundle is not None:
            return self._subprocess_verify_output.detach().clone()
        return super().get_verify_output()

    def get_input_signature(self) -> InputSignature:
        if self._torchrun_result_bundle is not None:
            return self._subprocess_input_signature
        return super().get_input_signature()

    def get_output_tolerance(self) -> Tuple[float, float]:
        if self._torchrun_result_bundle is not None:
            return self._subprocess_output_tolerance
        return super().get_output_tolerance()



class BaselineDisaggregatedInferenceMultiGPUBenchmark(_DisaggregatedInferenceMultiGPUBenchmark):
    """Serialized prefill then decode across multi-GPU ranks."""

    profile_nvtx_range = BASELINE_PROFILE_NVTX_RANGE
    story_metadata = {
        "pair_role": "canonical",
        "chapter_alignment": "native",
        "chapter_native_exemplar": True,
        "timed_launch_mode": "torchrun_multi_gpu",
        "verification_mode": "timed_torchrun_rank_outputs",
        "shared_harness_layout": "baseline_owned_shared_base",
        "shared_harness_owner": "ch15/baseline_disaggregated_inference_multigpu.py",
        "execution_pattern": "serialized_prefill_then_decode",
    }

    def __init__(self) -> None:
        super().__init__(
            overlap=False,
            label="baseline_disaggregated_inference_multigpu",
            profile_nvtx_range=BASELINE_PROFILE_NVTX_RANGE,
            worker_module="ch15.baseline_disaggregated_inference_multigpu",
        )


def get_benchmark() -> BaseBenchmark:
    return BaselineDisaggregatedInferenceMultiGPUBenchmark()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _run_torchrun_worker(
        DisaggConfig(),
        overlap=False,
        label="baseline_disaggregated_inference_multigpu",
        iters=int(args.iters),
        warmup=int(args.warmup),
        profile_nvtx_range=BASELINE_PROFILE_NVTX_RANGE,
        seed=args.seed,
    )
