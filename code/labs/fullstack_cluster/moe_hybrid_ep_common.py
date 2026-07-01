"""Shared DeepSeek-style hybrid expert-parallel training benchmark helpers."""

from __future__ import annotations

import argparse
import json
import os

from core.common.device_utils import resolve_local_rank
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F

from core.benchmark.metrics import compute_moe_metrics
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from core.optimization.moe_inference import resolve_dtype

MOE_HYBRID_EP_ENTRYPOINT_MODULE = "labs.fullstack_cluster.moe_hybrid_ep_entrypoint"


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TopologyInfo:
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    node_rank: int
    num_nodes: int
    initialized: bool
    local_group: Optional[dist.ProcessGroup]

    @property
    def intra_node_only(self) -> bool:
        return self.num_nodes <= 1

    @property
    def hybrid_enabled(self) -> bool:
        return self.num_nodes > 1

    @property
    def local_ranks(self) -> List[int]:
        start = self.node_rank * self.local_world_size
        stop = min(self.world_size, start + self.local_world_size)
        return list(range(start, stop))


@dataclass
class PhaseEvents:
    start: torch.cuda.Event
    mid: torch.cuda.Event
    mid2: torch.cuda.Event
    end: torch.cuda.Event

    def to_metrics(self) -> Tuple[float, float, float]:
        # elapsed_time() requires BOTH events complete. The baseline caller invokes
        # this before forward_loss's torch.cuda.synchronize(), so on a rank whose
        # terminal event has not finished yet it raises "Both events must be
        # completed before calculating elapsed time" -- timing-dependent, so only
        # SOME ranks raise, bail to the shutdown `finally` barrier, and desync from
        # the ranks that proceed to the next collective (the moe_hybrid_ep hang).
        # Synchronizing the terminal event (recorded last on the stream) guarantees
        # all four events are complete before any elapsed_time call.
        self.end.synchronize()
        return (
            float(self.start.elapsed_time(self.mid)),
            float(self.mid.elapsed_time(self.mid2)),
            float(self.mid2.elapsed_time(self.end)),
        )


@dataclass
class StepArtifacts:
    metrics: Dict[str, float]
    loss: float


_AVERAGED_REDUCED_METRIC_KEYS = frozenset(
    {
        "moe.load_imbalance",
        "moe.expert_utilization_pct",
        "moe.routing_time_ms",
        "moe.expert_compute_time_ms",
        "moe.routing_overhead_pct",
        "moe.load_balance_loss",
        "moe.hybrid_enabled",
        "moe.intra_node_only",
        "moe.local_world_size",
        "moe.inter_node_world_size",
        "moe.num_experts",
        "moe.active_experts",
        "moe.total_tokens",
        "moe.max_tokens_per_expert",
        "moe.min_tokens_per_expert",
        "moe.router_entropy",
        "moe.gini_coefficient",
        "moe.expert_usage_variance",
        "moe.routing_uniform",
        "moe.routing_topology_aware",
    }
)


def _metric_should_average(key: str) -> bool:
    return key.endswith("_pct") or key.endswith("_ms") or key in _AVERAGED_REDUCED_METRIC_KEYS


def init_topology(backend: str = "nccl") -> TopologyInfo:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for moe_hybrid_ep benchmark")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = resolve_local_rank()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if local_world_size <= 0:
        local_world_size = max(world_size, 1)
    if world_size % local_world_size != 0:
        raise RuntimeError(
            f"WORLD_SIZE={world_size} must be divisible by LOCAL_WORLD_SIZE={local_world_size}"
        )
    node_rank = rank // local_world_size
    num_nodes = max(1, world_size // local_world_size)

    torch.cuda.set_device(local_rank)
    initialized = False
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        initialized = True
    elif dist.is_initialized():
        initialized = True

    local_group = None
    if world_size > 1 and local_world_size > 1:
        local_group = dist.new_group(
            ranks=list(range(node_rank * local_world_size, (node_rank + 1) * local_world_size))
        )

    return TopologyInfo(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        node_rank=node_rank,
        num_nodes=num_nodes,
        initialized=initialized,
        local_group=local_group,
    )


def shutdown_topology(topology: TopologyInfo) -> None:
    if topology.world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iters", type=int, default=6, help="Measured optimizer steps.")
    parser.add_argument("--tokens-per-rank", type=int, default=128, help="Synthetic token count per rank.")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Hidden dimension.")
    parser.add_argument("--num-experts", type=int, default=0, help="Global expert count. Default derives from local_experts * world_size.")
    parser.add_argument("--local-experts", type=int, default=4, help="Experts owned by each rank.")
    parser.add_argument("--top-k", type=int, default=2, help="Top-k experts per token.")
    parser.add_argument("--route-mode", choices=("uniform", "topology_aware"), default="uniform", help="Routing strategy.")
    parser.add_argument("--overlap-mode", choices=("disabled", "local_remote"), default="disabled", help="Overlap same-rank compute with remote traffic.")
    parser.add_argument("--dtype", default="bf16", help="Model dtype.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--aux-loss-scale", type=float, default=1e-2, help="Scale for routing load-balance loss.")
    parser.add_argument("--output-dir", default="artifacts/moe_hybrid_ep", help="Rank-0 report directory.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip topology and fabric discovery output.")
    parser.add_argument("--torch-profile-output", default="", help=argparse.SUPPRESS)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def maybe_write_sidecar(summary: Dict[str, float]) -> None:
    sidecar = os.environ.get("AISP_MOE_HYBRID_EP_METRICS_PATH")
    if not sidecar:
        return
    write_json(Path(sidecar), {"custom_metrics": summary})


def _steady_state_warmup_steps(topology: TopologyInfo) -> int:
    # Match the harness warmup so standalone report.json artifacts describe the
    # same steady-state step that the benchmark contract gates on.
    return 5 if topology.world_size <= 1 else 2


def _run_preflight(rank: int) -> None:
    if rank != 0:
        return
    commands = [
        ("nvidia_topo", ["nvidia-smi", "topo", "-m"]),
        ("nic_caps", ["bash", "-lc", "ibv_devinfo -v 2>/dev/null || true"]),
        ("rdma_link", ["bash", "-lc", "rdma link 2>/dev/null || true"]),
    ]
    print("\n[Preflight]")
    for name, command in commands:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            text = (result.stdout or result.stderr or "").strip()
        except Exception as exc:  # pragma: no cover - defensive
            text = f"(failed: {exc})"
        print(f"{name}: {text if text else '(no output)'}")


class LoadBalancedRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.register_buffer(
            "_gini_index",
            torch.arange(1, num_experts + 1, dtype=torch.float32),
            persistent=False,
        )

    def _gini_index_for(self, usage: torch.Tensor) -> torch.Tensor:
        index = self._gini_index
        if index.device != usage.device or index.dtype != usage.dtype:
            self._gini_index = index.to(device=usage.device, dtype=usage.dtype)
            index = self._gini_index
        return index

    def forward(self, x: torch.Tensor, *, expert_bias: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        logits = self.gate(x)
        if expert_bias is not None:
            logits = logits + expert_bias
        log_route_probs = F.log_softmax(logits, dim=-1)
        route_probs = log_route_probs.exp()
        top_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)
        top_weights = F.softmax(top_logits, dim=-1)
        expert_usage = route_probs.mean(dim=0)
        balance_loss = torch.var(expert_usage) * float(self.num_experts)
        sorted_usage = torch.sort(expert_usage)[0]
        n = len(sorted_usage)
        index = self._gini_index_for(sorted_usage)
        gini = (2 * (index * sorted_usage).sum()) / (n * sorted_usage.sum().clamp_min(1e-9)) - (n + 1) / n
        return top_weights, top_indices, {
            "balance_loss": balance_loss,
            "expert_usage_variance": torch.var(expert_usage),
            "gini_coefficient": gini,
            "router_entropy": -(route_probs * log_route_probs).sum(dim=-1).mean(),
        }


class ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.down_proj = nn.Linear(ffn_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        F.silu(gate, inplace=True)
        up = self.up_proj(x)
        gate.mul_(up)
        return self.down_proj(gate)


class DeepSeekHybridEPModule(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        local_experts: int,
        top_k: int,
        topology: TopologyInfo,
        *,
        route_mode: str,
        optimized: bool,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_size = hidden_size * 4
        self.num_experts = num_experts
        self.local_experts = local_experts
        self.top_k = top_k
        self.topology = topology
        self.route_mode = route_mode
        self.optimized = optimized
        self.input_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.router = LoadBalancedRouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList([ExpertMLP(hidden_size, self.ffn_size) for _ in range(local_experts)])
        self._cached_bias: Optional[torch.Tensor] = None
        self._buffer_cache: Dict[Tuple[str, torch.device, torch.dtype], torch.Tensor] = {}
        self._token_index_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        self._range_index_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        self._event_pair_cache: Dict[str, Tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._phase_event_cache: Dict[str, PhaseEvents] = {}
        self._route_count_reduce_buffer: Optional[torch.Tensor] = None
        self._route_count_reduce_host_buffer: Optional[torch.Tensor] = None
        self._exchange_count_send_buffer: Optional[torch.Tensor] = None
        self._exchange_count_recv_buffer: Optional[torch.Tensor] = None
        self._exchange_count_host_buffer: Optional[torch.Tensor] = None
        self._destination_count_host_buffer: Optional[torch.Tensor] = None
        self._local_expert_count_host_buffer: Optional[torch.Tensor] = None
        self._local_expert_count_list_buffer = [0] * local_experts
        self._route_type_count_buffer: Optional[torch.Tensor] = None
        self._route_type_count_host_buffer: Optional[torch.Tensor] = None
        self._route_type_count_list_buffer = [0] * 3
        self._route_counts_host_buffer: Optional[torch.Tensor] = None
        self._route_counts_list_buffer: List[int] = []
        self._aux_metric_buffer: Optional[torch.Tensor] = None
        self._aux_metric_host_buffer: Optional[torch.Tensor] = None
        self._aux_metric_list_buffer = [0.0] * 4
        self._comm_stream = torch.cuda.Stream() if optimized else None

    @property
    def cuda_device(self) -> torch.device:
        return torch.device("cuda", torch.cuda.current_device())

    def replicated_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.input_proj.parameters()
        yield from self.output_proj.parameters()
        yield from self.router.parameters()

    def _expert_bias(self, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.route_mode != "topology_aware":
            return None
        if self.topology.world_size <= 1:
            return None
        if self._cached_bias is not None and self._cached_bias.device == device and self._cached_bias.dtype == dtype:
            return self._cached_bias
        owner = torch.arange(self.num_experts, device=device, dtype=torch.int64) // self.local_experts
        same_node = (owner // self.topology.local_world_size) == self.topology.node_rank
        same_rank = owner == self.topology.rank
        bias = torch.zeros(self.num_experts, device=device, dtype=dtype)
        bias = bias + same_node.to(dtype) * 0.35
        bias = bias + same_rank.to(dtype) * 0.25
        self._cached_bias = bias
        return bias

    def _buffer(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        *,
        reuse: bool,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        target_device = device if device is not None else self.cuda_device
        shape = tuple(int(dim) for dim in shape)
        if not reuse:
            return torch.empty(shape, device=target_device, dtype=dtype)
        numel = 1
        for dim in shape:
            numel *= dim
        key = (name, target_device, dtype)
        cached = self._buffer_cache.get(key)
        if cached is None or cached.numel() < numel:
            cached = torch.empty(numel, device=target_device, dtype=dtype)
            self._buffer_cache[key] = cached
        return cached[:numel].view(shape)

    def _event_pair(self, name: str) -> Tuple[torch.cuda.Event, torch.cuda.Event]:
        cached = self._event_pair_cache.get(name)
        if cached is None:
            cached = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            self._event_pair_cache[name] = cached
        return cached

    def _phase_events(self, name: str) -> PhaseEvents:
        cached = self._phase_event_cache.get(name)
        if cached is None:
            cached = PhaseEvents(
                start=torch.cuda.Event(enable_timing=True),
                mid=torch.cuda.Event(enable_timing=True),
                mid2=torch.cuda.Event(enable_timing=True),
                end=torch.cuda.Event(enable_timing=True),
            )
            self._phase_event_cache[name] = cached
        return cached

    def _reduce_route_counts(
        self,
        same_rank_count: float,
        same_node_count: float,
        remote_count: float,
        *,
        device: torch.device,
    ) -> Tuple[float, float, float]:
        if self._route_count_reduce_buffer is None or self._route_count_reduce_buffer.device != device:
            self._route_count_reduce_buffer = torch.empty(3, device=device, dtype=torch.float32)
            self._route_count_reduce_host_buffer = torch.empty(3, dtype=torch.float32)
        assert self._route_count_reduce_host_buffer is not None
        reduce_buffer = self._route_count_reduce_buffer
        host_buffer = self._route_count_reduce_host_buffer
        host_buffer[0] = same_rank_count
        host_buffer[1] = same_node_count
        host_buffer[2] = remote_count
        reduce_buffer.copy_(host_buffer, non_blocking=False)
        dist.all_reduce(reduce_buffer, op=dist.ReduceOp.SUM)
        host_buffer.copy_(reduce_buffer, non_blocking=False)
        return float(host_buffer[0]), float(host_buffer[1]), float(host_buffer[2])

    def _token_indices(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        key = (num_tokens, device)
        cached = self._token_index_cache.get(key)
        if cached is None:
            cached = torch.arange(num_tokens * self.top_k, device=device, dtype=torch.int64)
            if self.top_k != 1:
                cached.div_(self.top_k, rounding_mode="floor")
            self._token_index_cache[key] = cached
        return cached

    def _range_indices(self, length: int, device: torch.device) -> torch.Tensor:
        key = (length, device)
        cached = self._range_index_cache.get(key)
        if cached is None:
            cached = torch.arange(length, device=device, dtype=torch.int64)
            self._range_index_cache[key] = cached
        return cached

    def _local_expert_count_list(self, expert_ids: torch.Tensor) -> List[int]:
        counts = torch.bincount(expert_ids, minlength=self.local_experts)
        needs_pinned = counts.device.type == "cuda"
        if (
            self._local_expert_count_host_buffer is None
            or (
                needs_pinned
                and not self._local_expert_count_host_buffer.is_pinned()
            )
        ):
            self._local_expert_count_host_buffer = torch.empty(
                self.local_experts,
                dtype=torch.long,
                device="cpu",
                pin_memory=needs_pinned,
            )
        host_counts = self._local_expert_count_host_buffer
        host_counts.copy_(counts)
        count_list = self._local_expert_count_list_buffer
        for expert_idx in range(self.local_experts):
            count_list[expert_idx] = int(host_counts[expert_idx])
        return count_list

    def _route_type_count_list(
        self,
        same_rank_mask: torch.Tensor,
        same_node_mask: torch.Tensor,
        remote_node_mask: torch.Tensor,
    ) -> List[int]:
        device = same_rank_mask.device
        if self._route_type_count_buffer is None or self._route_type_count_buffer.device != device:
            self._route_type_count_buffer = torch.empty(3, device=device, dtype=torch.long)
            self._route_type_count_host_buffer = torch.empty(
                3,
                dtype=torch.long,
                device="cpu",
                pin_memory=device.type == "cuda",
            )
        assert self._route_type_count_host_buffer is not None
        count_buffer = self._route_type_count_buffer
        host_buffer = self._route_type_count_host_buffer
        torch.sum(same_rank_mask, dim=None, out=count_buffer[0])
        torch.sum(same_node_mask, dim=None, out=count_buffer[1])
        torch.sum(remote_node_mask, dim=None, out=count_buffer[2])
        host_buffer.copy_(count_buffer)
        count_list = self._route_type_count_list_buffer
        for route_idx in range(3):
            count_list[route_idx] = int(host_buffer[route_idx])
        return count_list

    def _route_counts_list(self, route_counts: torch.Tensor) -> List[int]:
        needs_pinned = route_counts.device.type == "cuda"
        if (
            self._route_counts_host_buffer is None
            or self._route_counts_host_buffer.numel() != route_counts.numel()
            or (needs_pinned and not self._route_counts_host_buffer.is_pinned())
        ):
            self._route_counts_host_buffer = torch.empty(
                route_counts.numel(),
                dtype=route_counts.dtype,
                device="cpu",
                pin_memory=needs_pinned,
            )
        host_buffer = self._route_counts_host_buffer
        host_buffer.copy_(route_counts)
        count = route_counts.numel()
        if len(self._route_counts_list_buffer) != count:
            self._route_counts_list_buffer = [0] * count
        count_list = self._route_counts_list_buffer
        for expert_idx in range(count):
            count_list[expert_idx] = int(host_buffer[expert_idx])
        return count_list

    def _aux_metric_values(self, aux: Dict[str, torch.Tensor]) -> List[float]:
        first = aux["balance_loss"]
        if (
            self._aux_metric_buffer is None
            or self._aux_metric_buffer.device != first.device
            or self._aux_metric_buffer.dtype != first.dtype
        ):
            self._aux_metric_buffer = torch.empty(4, device=first.device, dtype=first.dtype)
            self._aux_metric_host_buffer = torch.empty(
                4,
                dtype=first.dtype,
                device="cpu",
                pin_memory=first.device.type == "cuda",
            )
        assert self._aux_metric_host_buffer is not None
        metric_buffer = self._aux_metric_buffer
        host_buffer = self._aux_metric_host_buffer
        metric_buffer[0].copy_(aux["balance_loss"])
        metric_buffer[1].copy_(aux["router_entropy"])
        metric_buffer[2].copy_(aux["gini_coefficient"])
        metric_buffer[3].copy_(aux["expert_usage_variance"])
        host_buffer.copy_(metric_buffer)
        metric_list = self._aux_metric_list_buffer
        for metric_idx in range(4):
            metric_list[metric_idx] = float(host_buffer[metric_idx])
        return metric_list

    def _apply_local_experts(self, tokens: torch.Tensor, expert_ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if tokens.numel() == 0:
            return tokens
        outputs = self._buffer(
            "local_outputs",
            tuple(tokens.shape),
            tokens.dtype,
            reuse=self.optimized,
            device=tokens.device,
        )
        sort_idx = torch.argsort(expert_ids)
        sorted_tokens = tokens.index_select(0, sort_idx)
        sorted_weights = weights.index_select(0, sort_idx)
        sorted_outputs = self._buffer(
            "local_sorted_outputs",
            tuple(sorted_tokens.shape),
            sorted_tokens.dtype,
            reuse=self.optimized,
            device=sorted_tokens.device,
        )
        expert_count_list = self._local_expert_count_list(expert_ids)
        offset = 0
        for local_id, expert in enumerate(self.experts):
            next_offset = offset + int(expert_count_list[local_id])
            if next_offset > offset:
                expert_out = expert(sorted_tokens[offset:next_offset])
                out_slice = sorted_outputs[offset:next_offset]
                torch.mul(expert_out, sorted_weights[offset:next_offset], out=out_slice)
            offset = next_offset
        outputs.index_copy_(0, sort_idx, sorted_outputs)
        return outputs

    def _exchange_counts(
        self,
        send_counts: Sequence[int],
        *,
        group: Optional[dist.ProcessGroup],
        group_size: int,
        group_rank: int,
    ) -> List[int]:
        if group_size == 1:
            return [int(send_counts[0] if send_counts else 0)]
        device = self.cuda_device
        if (
            self._exchange_count_send_buffer is None
            or self._exchange_count_send_buffer.device != device
            or self._exchange_count_send_buffer.numel() < group_size
        ):
            self._exchange_count_send_buffer = torch.empty(group_size, device=device, dtype=torch.int64)
            self._exchange_count_recv_buffer = torch.empty(group_size * group_size, device=device, dtype=torch.int64)
            self._exchange_count_host_buffer = torch.empty(group_size, dtype=torch.int64)
        assert self._exchange_count_recv_buffer is not None
        assert self._exchange_count_host_buffer is not None
        send_tensor = self._exchange_count_send_buffer[:group_size]
        recv_tensor = self._exchange_count_recv_buffer[: group_size * group_size]
        host_buffer = self._exchange_count_host_buffer[:group_size]
        for idx, count in enumerate(send_counts):
            host_buffer[idx] = int(count)
        send_tensor.copy_(host_buffer, non_blocking=False)
        if hasattr(dist, "all_gather_into_tensor"):
            dist.all_gather_into_tensor(recv_tensor, send_tensor, group=group)
            gathered_counts = recv_tensor.view(group_size, group_size)[:, group_rank]
        else:  # pragma: no cover - compatibility with older PyTorch builds
            gathered = [recv_tensor.narrow(0, rank * group_size, group_size) for rank in range(group_size)]
            dist.all_gather(gathered, send_tensor, group=group)
            gathered_counts = recv_tensor.view(group_size, group_size)[:, group_rank]
        host_buffer.copy_(gathered_counts, non_blocking=False)
        return [int(host_buffer[idx]) for idx in range(group_size)]

    def _destination_count_list(self, dest_ranks: torch.Tensor, group_size: int) -> List[int]:
        counts = torch.bincount(dest_ranks, minlength=group_size)
        needs_pinned = counts.device.type == "cuda"
        if (
            self._destination_count_host_buffer is None
            or self._destination_count_host_buffer.numel() < group_size
            or (needs_pinned and not self._destination_count_host_buffer.is_pinned())
        ):
            self._destination_count_host_buffer = torch.empty(
                group_size,
                dtype=counts.dtype,
                device="cpu",
                pin_memory=needs_pinned,
            )
        host_buffer = self._destination_count_host_buffer[:group_size]
        host_buffer.copy_(counts[:group_size])
        return [int(host_buffer[idx]) for idx in range(group_size)]

    def _split_list(self, tensor: torch.Tensor, counts: Sequence[int]) -> List[torch.Tensor]:
        parts: List[torch.Tensor] = []
        offset = 0
        trailing = tensor.shape[1:]
        for count in counts:
            if count:
                parts.append(tensor.narrow(0, offset, int(count)))
            else:
                parts.append(tensor.new_empty((0, *trailing)))
            offset += int(count)
        return parts

    def _all_to_all_list(self, tensor: torch.Tensor, send_counts: Sequence[int], recv_counts: Sequence[int], *, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
        if len(send_counts) == 1:
            return tensor
        recv_shape = (sum(int(x) for x in recv_counts), *tensor.shape[1:])
        recv = tensor.new_empty(recv_shape)
        recv_parts = self._split_list(recv, recv_counts)
        send_parts = self._split_list(tensor, send_counts)
        dist_nn.all_to_all(recv_parts, send_parts, group=group)
        return recv

    def _all_to_all_single(
        self,
        tensor: torch.Tensor,
        send_counts: Sequence[int],
        recv_counts: Sequence[int],
        *,
        group: Optional[dist.ProcessGroup],
        label: str,
        reuse: bool,
    ) -> torch.Tensor:
        if len(send_counts) == 1:
            return tensor
        recv_shape = (sum(int(x) for x in recv_counts), *tensor.shape[1:])
        output = self._buffer(label, recv_shape, tensor.dtype, reuse=reuse, device=tensor.device)
        return dist_nn.all_to_all_single(
            output,
            tensor,
            output_split_sizes=list(int(x) for x in recv_counts),
            input_split_sizes=list(int(x) for x in send_counts),
            group=group,
        )

    def _roundtrip_routes(
        self,
        *,
        tokens: torch.Tensor,
        weights: torch.Tensor,
        dest_ranks: torch.Tensor,
        token_indices: torch.Tensor,
        local_expert_ids: torch.Tensor,
        group: Optional[dist.ProcessGroup],
        group_size: int,
        group_rank: int,
        use_single: bool,
        reuse: bool,
        event_label: str,
    ) -> Tuple[torch.Tensor, Optional[PhaseEvents]]:
        if tokens.numel() == 0:
            return tokens, None
        sort_idx = torch.argsort(dest_ranks)
        inverse_sort = torch.empty_like(sort_idx)
        inverse_sort[sort_idx] = self._range_indices(sort_idx.numel(), sort_idx.device)
        sorted_tokens = tokens.index_select(0, sort_idx)
        sorted_weights = weights.index_select(0, sort_idx)
        sorted_token_indices = token_indices.index_select(0, sort_idx)
        sorted_local_ids = local_expert_ids.index_select(0, sort_idx)
        send_counts = self._destination_count_list(dest_ranks, group_size)
        recv_counts = self._exchange_counts(send_counts, group=group, group_size=group_size, group_rank=group_rank)

        events = self._phase_events(event_label)
        current_stream = torch.cuda.current_stream()
        events.start.record(current_stream)

        meta = self._buffer(
            "route_meta",
            (sort_idx.numel(), 2),
            sorted_token_indices.dtype,
            reuse=reuse,
            device=sorted_token_indices.device,
        )
        meta[:, 0].copy_(sorted_token_indices)
        meta[:, 1].copy_(sorted_local_ids)
        if use_single:
            recv_tokens = self._all_to_all_single(sorted_tokens, send_counts, recv_counts, group=group, label="recv_tokens", reuse=reuse)
            recv_weights = self._all_to_all_single(sorted_weights, send_counts, recv_counts, group=group, label="recv_weights", reuse=reuse)
            recv_meta = self._all_to_all_single(meta, send_counts, recv_counts, group=group, label="recv_meta", reuse=reuse)
        else:
            recv_tokens = self._all_to_all_list(sorted_tokens, send_counts, recv_counts, group=group)
            recv_weights = self._all_to_all_list(sorted_weights, send_counts, recv_counts, group=group)
            recv_meta = self._all_to_all_list(meta, send_counts, recv_counts, group=group)
        events.mid.record(current_stream)

        expert_outputs = self._apply_local_experts(
            recv_tokens,
            recv_meta[:, 1].to(torch.int64),
            recv_weights,
        )
        events.mid2.record(current_stream)

        if use_single:
            returned = self._all_to_all_single(expert_outputs, recv_counts, send_counts, group=group, label="return_tokens", reuse=reuse)
        else:
            returned = self._all_to_all_list(expert_outputs, recv_counts, send_counts, group=group)
        events.end.record(current_stream)

        return returned.index_select(0, inverse_sort), events

    def _route_tokens(
        self,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Optional[torch.Tensor]]]:
        expert_bias = self._expert_bias(hidden.device, hidden.dtype)
        top_weights, top_indices, aux = self.router(hidden, expert_bias=expert_bias)
        route_counts = torch.bincount(top_indices.reshape(-1), minlength=self.num_experts)
        token_indices = self._token_indices(hidden.size(0), hidden.device)
        expanded_tokens = hidden.index_select(0, token_indices)
        expanded_weights = top_weights.reshape(-1, 1).to(hidden.dtype)
        expanded_experts = top_indices.reshape(-1).to(torch.int64)
        owner_ranks = torch.div(expanded_experts, self.local_experts, rounding_mode="floor")
        local_expert_ids = torch.remainder(expanded_experts, self.local_experts)
        route_payload = {
            "token_indices": token_indices,
            "expanded_weights": expanded_weights,
            "expanded_experts": expanded_experts,
            "owner_ranks": owner_ranks,
            "local_expert_ids": local_expert_ids,
            "owner_nodes": None
            if self.topology.intra_node_only
            else torch.div(owner_ranks, self.topology.local_world_size, rounding_mode="floor"),
        }
        return expanded_tokens, top_indices, route_counts, aux, route_payload

    def forward_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        overlap_mode: str,
        aux_loss_scale: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        hidden = self.input_proj(inputs)
        routing_start, routing_end = self._event_pair("routing")
        current_stream = torch.cuda.current_stream()
        routing_start.record(current_stream)
        expanded_tokens, top_indices, route_counts, aux, route_payload = self._route_tokens(hidden)
        routing_end.record(current_stream)

        owner_ranks = route_payload["owner_ranks"]
        owner_nodes = route_payload["owner_nodes"]
        local_expert_ids = route_payload["local_expert_ids"]
        token_indices = route_payload["token_indices"]
        expanded_weights = route_payload["expanded_weights"]

        same_rank_metrics = (0.0, 0.0, 0.0)
        same_node_metrics = (0.0, 0.0, 0.0)
        remote_metrics = (0.0, 0.0, 0.0)
        overlap_pct = 0.0

        combined = self._buffer(
            "combined_outputs",
            tuple(hidden.shape),
            hidden.dtype,
            reuse=self.optimized,
            device=hidden.device,
        )
        combined.zero_()
        same_rank_count = 0.0
        same_node_count = 0.0
        remote_count = 0.0
        same_rank_events: Optional[Tuple[torch.cuda.Event, torch.cuda.Event]] = None
        same_node_events: Optional[PhaseEvents] = None
        remote_events: Optional[PhaseEvents] = None
        local_branch_events: Optional[Tuple[torch.cuda.Event, torch.cuda.Event]] = None
        remote_branch_events: Optional[Tuple[torch.cuda.Event, torch.cuda.Event]] = None
        overlap_window_events: Optional[Tuple[torch.cuda.Event, torch.cuda.Event]] = None

        if self.optimized and self.topology.world_size == 1:
            same_rank_start, same_rank_end = self._event_pair("same_rank_single")
            same_rank_start.record(current_stream)
            local_outputs = self._apply_local_experts(
                expanded_tokens,
                local_expert_ids,
                expanded_weights,
            )
            combined.index_add_(0, token_indices, local_outputs)
            same_rank_end.record(current_stream)
            same_rank_events = (same_rank_start, same_rank_end)
            same_rank_count = float(token_indices.numel())
        elif self.optimized:
            same_rank_mask = owner_ranks == self.topology.rank
            if owner_nodes is None:
                same_node_mask = ~same_rank_mask
                remote_node_mask = self._buffer(
                    "remote_node_mask",
                    tuple(same_rank_mask.shape),
                    same_rank_mask.dtype,
                    reuse=True,
                    device=same_rank_mask.device,
                )
                remote_node_mask.zero_()
            else:
                same_node_mask = (owner_nodes == self.topology.node_rank) & ~same_rank_mask
                remote_node_mask = owner_nodes != self.topology.node_rank

            remote_outputs: Optional[torch.Tensor] = None
            route_type_counts = self._route_type_count_list(
                same_rank_mask,
                same_node_mask,
                remote_node_mask,
            )
            same_rank_count_int, same_node_count_int, remote_count_int = (
                int(count) for count in route_type_counts
            )
            overlap_active = (
                remote_count_int > 0 and self.topology.world_size > 1 and overlap_mode == "local_remote"
            )
            if overlap_active:
                local_branch_start, local_branch_end = self._event_pair("local_branch")
                remote_branch_start, remote_branch_end = self._event_pair("remote_branch")
                overlap_window_start, overlap_window_end = self._event_pair("overlap_window")
                local_branch_events = (local_branch_start, local_branch_end)
                remote_branch_events = (remote_branch_start, remote_branch_end)
                overlap_window_events = (overlap_window_start, overlap_window_end)
                overlap_window_start.record(current_stream)
            if overlap_active:
                assert self._comm_stream is not None
                self._comm_stream.wait_stream(current_stream)
                with torch.cuda.stream(self._comm_stream):
                    remote_branch_start.record(self._comm_stream)
                    remote_outputs, remote_events = self._roundtrip_routes(
                        tokens=expanded_tokens[remote_node_mask],
                        weights=expanded_weights[remote_node_mask],
                        dest_ranks=owner_ranks[remote_node_mask],
                        token_indices=token_indices[remote_node_mask],
                        local_expert_ids=local_expert_ids[remote_node_mask],
                        group=None,
                        group_size=self.topology.world_size,
                        group_rank=self.topology.rank,
                        use_single=True,
                        reuse=True,
                        event_label="remote",
                    )
                    remote_branch_end.record(self._comm_stream)

            if overlap_active and local_branch_events is not None:
                local_branch_events[0].record(current_stream)
            if same_rank_count_int > 0:
                same_rank_start, same_rank_end = self._event_pair("same_rank")
                same_rank_start.record(current_stream)
                local_outputs = self._apply_local_experts(
                    expanded_tokens[same_rank_mask],
                    local_expert_ids[same_rank_mask],
                    expanded_weights[same_rank_mask],
                )
                combined.index_add_(0, token_indices[same_rank_mask], local_outputs)
                same_rank_end.record(current_stream)
                same_rank_events = (same_rank_start, same_rank_end)
            if same_node_count_int > 0 and self.topology.local_group is not None:
                local_group_ranks = owner_ranks[same_node_mask] % self.topology.local_world_size
                same_node_outputs, same_node_events = self._roundtrip_routes(
                    tokens=expanded_tokens[same_node_mask],
                    weights=expanded_weights[same_node_mask],
                    dest_ranks=local_group_ranks,
                    token_indices=token_indices[same_node_mask],
                    local_expert_ids=local_expert_ids[same_node_mask],
                    group=self.topology.local_group,
                    group_size=self.topology.local_world_size,
                    group_rank=self.topology.local_rank,
                    use_single=True,
                    reuse=True,
                    event_label="same_node",
                )
                combined.index_add_(0, token_indices[same_node_mask], same_node_outputs)
            else:
                same_node_events = None
            if overlap_active and local_branch_events is not None:
                local_branch_events[1].record(current_stream)

            if remote_outputs is None and remote_count_int > 0 and self.topology.world_size > 1:
                if remote_branch_events is not None:
                    remote_branch_events[0].record(current_stream)
                remote_outputs, remote_events = self._roundtrip_routes(
                    tokens=expanded_tokens[remote_node_mask],
                    weights=expanded_weights[remote_node_mask],
                    dest_ranks=owner_ranks[remote_node_mask],
                    token_indices=token_indices[remote_node_mask],
                    local_expert_ids=local_expert_ids[remote_node_mask],
                    group=None,
                    group_size=self.topology.world_size,
                    group_rank=self.topology.rank,
                    use_single=True,
                    reuse=True,
                    event_label="remote",
                )
                if remote_branch_events is not None:
                    remote_branch_events[1].record(current_stream)

            if remote_outputs is not None:
                if self._comm_stream is not None and overlap_mode == "local_remote":
                    current_stream.wait_stream(self._comm_stream)
                combined.index_add_(0, token_indices[remote_node_mask], remote_outputs)
            if overlap_active and overlap_window_events is not None:
                overlap_window_events[1].record(current_stream)
            same_rank_count = float(same_rank_count_int)
            same_node_count = float(same_node_count_int)
            remote_count = float(remote_count_int)
        else:
            baseline_outputs, baseline_events = self._roundtrip_routes(
                tokens=expanded_tokens,
                weights=expanded_weights,
                dest_ranks=owner_ranks,
                token_indices=token_indices,
                local_expert_ids=local_expert_ids,
                group=None,
                group_size=self.topology.world_size,
                group_rank=self.topology.rank,
                use_single=False,
                reuse=False,
                event_label="baseline",
            )
            combined.index_add_(0, token_indices, baseline_outputs)
            same_node_metrics = baseline_events.to_metrics() if baseline_events is not None else (0.0, 0.0, 0.0)
            same_rank_count = 0.0
            same_node_count = float(int(token_indices.numel()))
            remote_count = 0.0

        outputs = self.output_proj(hidden + combined)
        loss = F.mse_loss(outputs, targets) + aux["balance_loss"] * float(aux_loss_scale)
        torch.cuda.synchronize()
        routing_ms = float(routing_start.elapsed_time(routing_end))

        if same_rank_events is not None:
            same_rank_metrics = (
                0.0,
                float(same_rank_events[0].elapsed_time(same_rank_events[1])),
                0.0,
            )
        if self.optimized:
            if same_node_events is not None:
                same_node_metrics = same_node_events.to_metrics()
            if remote_events is not None:
                remote_metrics = remote_events.to_metrics()
            if (
                local_branch_events is not None
                and remote_branch_events is not None
                and overlap_window_events is not None
            ):
                local_branch_ms = float(local_branch_events[0].elapsed_time(local_branch_events[1]))
                remote_branch_ms = float(remote_branch_events[0].elapsed_time(remote_branch_events[1]))
                overlap_window_ms = float(overlap_window_events[0].elapsed_time(overlap_window_events[1]))
                overlap_saved_ms = max(0.0, (local_branch_ms + remote_branch_ms) - overlap_window_ms)
                overlap_pct = 100.0 * overlap_saved_ms / max(local_branch_ms + remote_branch_ms, 1e-6)

        route_counts_global = route_counts
        if self.topology.world_size > 1 and dist.is_initialized():
            dist.all_reduce(route_counts_global)
            same_rank_count, same_node_count, remote_count = self._reduce_route_counts(
                same_rank_count,
                same_node_count,
                remote_count,
                device=hidden.device,
            )
        route_counts_cpu = self._route_counts_list(route_counts_global)
        total_routes = max(float(sum(route_counts_cpu)), 1.0)
        (
            load_balance_loss,
            router_entropy,
            gini_coefficient,
            expert_usage_variance,
        ) = self._aux_metric_values(aux)
        metrics = compute_moe_metrics(
            num_experts=self.num_experts,
            active_experts=self.top_k,
            tokens_per_expert=[int(x) for x in route_counts_cpu],
            routing_time_ms=routing_ms,
            expert_compute_time_ms=same_rank_metrics[1] + same_node_metrics[1] + remote_metrics[1],
            load_balance_loss=float(load_balance_loss),
        )
        metrics.update(
            {
                "moe.routing_uniform": 1.0 if self.route_mode == "uniform" else 0.0,
                "moe.routing_topology_aware": 1.0 if self.route_mode == "topology_aware" else 0.0,
                "moe.hybrid_enabled": 1.0 if self.topology.hybrid_enabled else 0.0,
                "moe.intra_node_only": 1.0 if self.topology.intra_node_only else 0.0,
                "moe.local_world_size": float(self.topology.local_world_size),
                "moe.inter_node_world_size": float(max(self.topology.world_size - self.topology.local_world_size, 0)),
                "moe.same_rank_token_pct": 100.0 * same_rank_count / total_routes,
                "moe.same_node_token_pct": 100.0 * same_node_count / total_routes,
                "moe.remote_token_pct": 100.0 * remote_count / total_routes,
                "moe.step.dispatch_ms": same_node_metrics[0] + remote_metrics[0],
                "moe.step.combine_ms": same_node_metrics[2] + remote_metrics[2],
                "moe.step.expert_compute_ms": same_rank_metrics[1] + same_node_metrics[1] + remote_metrics[1],
                "moe.step.dispatch_intra_node_ms": same_node_metrics[0],
                "moe.step.dispatch_inter_node_ms": remote_metrics[0],
                "moe.step.combine_intra_node_ms": same_node_metrics[2],
                "moe.step.combine_inter_node_ms": remote_metrics[2],
                "moe.step.expert_compute_local_ms": same_rank_metrics[1],
                "moe.step.expert_compute_intra_node_ms": same_node_metrics[1],
                "moe.step.expert_compute_inter_node_ms": remote_metrics[1],
                "moe.step.overlap_pct": overlap_pct,
                "moe.router_entropy": float(router_entropy),
                "moe.gini_coefficient": float(gini_coefficient),
                "moe.expert_usage_variance": float(expert_usage_variance),
            }
        )
        return loss, metrics


class HybridEPTrainer:
    def __init__(self, args: argparse.Namespace, topology: TopologyInfo, *, optimized: bool) -> None:
        self.args = args
        self.topology = topology
        self.optimized = optimized
        self.device = torch.device("cuda", topology.local_rank)
        self.dtype = resolve_dtype(args.dtype)
        if args.top_k <= 0:
            raise ValueError("--top-k must be >= 1")
        self.local_experts = max(1, args.local_experts)
        derived_experts = self.local_experts * max(topology.world_size, 1)
        self.num_experts = args.num_experts or derived_experts
        if self.num_experts != derived_experts:
            raise ValueError(
                f"--num-experts ({self.num_experts}) must equal local_experts * world_size ({derived_experts})"
            )
        self.model = DeepSeekHybridEPModule(
            hidden_size=args.hidden_size,
            num_experts=self.num_experts,
            local_experts=self.local_experts,
            top_k=args.top_k,
            topology=topology,
            route_mode=args.route_mode,
            optimized=optimized,
        ).to(device=self.device, dtype=self.dtype)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        data_seed = 4242 + topology.rank
        generator = torch.Generator(device=self.device)
        generator.manual_seed(data_seed)
        self.inputs = torch.randn(
            args.tokens_per_rank,
            args.hidden_size,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        self.targets = torch.randn(
            args.tokens_per_rank,
            args.hidden_size,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        self._step_events: Dict[str, torch.cuda.Event] = {
            "start": torch.cuda.Event(enable_timing=True),
            "after_forward": torch.cuda.Event(enable_timing=True),
            "after_backward": torch.cuda.Event(enable_timing=True),
            "after_sync": torch.cuda.Event(enable_timing=True),
            "end": torch.cuda.Event(enable_timing=True),
        }
        self._metric_reduce_buffer: Optional[torch.Tensor] = None
        self._metric_reduce_host_buffer: Optional[torch.Tensor] = None
        self._loss_host_buffer = torch.empty((), dtype=torch.float32, pin_memory=True)

    def _sync_replicated_grads(self) -> None:
        if self.topology.world_size <= 1:
            return
        for param in self.model.replicated_parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(float(self.topology.world_size))

    def run_step(self) -> StepArtifacts:
        self.optimizer.zero_grad(set_to_none=True)
        total_start = self._step_events["start"]
        total_after_forward = self._step_events["after_forward"]
        total_after_backward = self._step_events["after_backward"]
        total_after_sync = self._step_events["after_sync"]
        total_end = self._step_events["end"]
        current_stream = torch.cuda.current_stream()

        total_start.record(current_stream)
        loss, metrics = self.model.forward_loss(
            self.inputs,
            self.targets,
            overlap_mode=self.args.overlap_mode,
            aux_loss_scale=self.args.aux_loss_scale,
        )
        total_after_forward.record(current_stream)
        loss.backward()
        total_after_backward.record(current_stream)
        self._sync_replicated_grads()
        total_after_sync.record(current_stream)
        self.optimizer.step()
        total_end.record(current_stream)
        torch.cuda.synchronize()

        metrics.update(
            {
                "moe.step.backward_ms": float(total_after_forward.elapsed_time(total_after_backward)),
                "moe.step.grad_sync_ms": float(total_after_backward.elapsed_time(total_after_sync)),
                "moe.step.optimizer_ms": float(total_after_sync.elapsed_time(total_end)),
                "moe.step.total_ms": float(total_start.elapsed_time(total_end)),
            }
        )
        metrics = self._reduce_metrics(metrics)
        self._loss_host_buffer.copy_(loss.detach(), non_blocking=False)
        return StepArtifacts(metrics=metrics, loss=float(self._loss_host_buffer))

    def _reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        if self.topology.world_size <= 1:
            return metrics
        if not metrics:
            return {}

        keys = tuple(metrics.keys())
        num_metrics = len(keys)
        if self._metric_reduce_buffer is None or self._metric_reduce_buffer.numel() != num_metrics:
            self._metric_reduce_buffer = torch.empty(num_metrics, device=self.device, dtype=torch.float64)
            self._metric_reduce_host_buffer = torch.empty(num_metrics, dtype=torch.float64)
        assert self._metric_reduce_host_buffer is not None
        device_buffer = self._metric_reduce_buffer
        host_buffer = self._metric_reduce_host_buffer
        for index, key in enumerate(keys):
            host_buffer[index] = float(metrics[key])
        device_buffer.copy_(host_buffer, non_blocking=False)
        dist.all_reduce(device_buffer, op=dist.ReduceOp.SUM)
        host_buffer.copy_(device_buffer, non_blocking=False)

        world_size = float(self.topology.world_size)
        reduced: Dict[str, float] = {}
        for index, key in enumerate(keys):
            value = float(host_buffer[index])
            if _metric_should_average(key):
                value /= world_size
            reduced[key] = value
        return reduced


def summarize_and_write_report(
    *,
    args: argparse.Namespace,
    topology: TopologyInfo,
    step_history: List[StepArtifacts],
    optimized: bool,
) -> Dict[str, float]:
    mean_metrics: Dict[str, float] = {}
    if step_history:
        metric_keys = sorted(step_history[0].metrics.keys())
        metric_totals = {key: 0.0 for key in metric_keys}
        loss_total = 0.0
        for step in step_history:
            loss_total += step.loss
            for key in metric_keys:
                metric_totals[key] += step.metrics[key]
        step_count = len(step_history)
        for key, total in metric_totals.items():
            mean_metrics[key] = float(total / step_count)
        mean_metrics["moe.step.loss"] = float(loss_total / step_count)

    payload = {
        "benchmark": "optimized_moe_hybrid_ep" if optimized else "baseline_moe_hybrid_ep",
        "config": {
            "iters": args.iters,
            "tokens_per_rank": args.tokens_per_rank,
            "hidden_size": args.hidden_size,
            "num_experts": args.num_experts or (args.local_experts * topology.world_size),
            "local_experts": args.local_experts,
            "top_k": args.top_k,
            "route_mode": args.route_mode,
            "overlap_mode": args.overlap_mode,
            "dtype": args.dtype,
        },
        "topology": {
            "rank": topology.rank,
            "world_size": topology.world_size,
            "local_rank": topology.local_rank,
            "local_world_size": topology.local_world_size,
            "node_rank": topology.node_rank,
            "num_nodes": topology.num_nodes,
            "hybrid_enabled": topology.hybrid_enabled,
            "intra_node_only": topology.intra_node_only,
        },
        "summary": mean_metrics,
        "iterations": [{"loss": step.loss, "metrics": step.metrics} for step in step_history],
    }
    if topology.rank == 0:
        out_dir = Path(args.output_dir)
        write_json(out_dir / "report.json", payload)
        maybe_write_sidecar(mean_metrics)
        print(
            f"{'Optimized' if optimized else 'Baseline'} DeepSeek-style hybrid EP mean step: "
            f"{mean_metrics.get('moe.step.total_ms', 0.0):.3f} ms",
            flush=True,
        )
        print(
            "Hybrid enabled: "
            f"{int(round(mean_metrics.get('moe.hybrid_enabled', 0.0)))} | "
            f"intra-node-only: {int(round(mean_metrics.get('moe.intra_node_only', 0.0)))} | "
            f"remote token pct: {mean_metrics.get('moe.remote_token_pct', 0.0):.2f}",
            flush=True,
        )
    return mean_metrics


def build_parser(*, optimized: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Optimized DeepSeek-style hybrid EP step" if optimized else "Baseline DeepSeek-style hybrid EP step"
    )
    add_shared_args(parser)
    parser.set_defaults(
        route_mode="uniform",
        overlap_mode="local_remote" if optimized else "disabled",
        output_dir="artifacts/moe_hybrid_ep_optimized" if optimized else "artifacts/moe_hybrid_ep_baseline",
    )
    return parser


def run_cli(*, optimized: bool, argv: Optional[Sequence[str]] = None) -> Dict[str, float]:
    parser = build_parser(optimized=optimized)
    args = parser.parse_args(list(argv) if argv is not None else None)
    topology = init_topology()
    try:
        if not args.skip_preflight:
            _run_preflight(topology.rank)
        trainer = HybridEPTrainer(args, topology, optimized=optimized)
        warmup_steps = _steady_state_warmup_steps(topology)
        if args.torch_profile_output:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            for _ in range(warmup_steps):
                trainer.run_step()
            with torch.profiler.profile(activities=activities) as prof:
                trainer.run_step()
            if topology.rank == 0:
                trace_path = Path(args.torch_profile_output)
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(str(trace_path))
            return {}
        for _ in range(warmup_steps):
            trainer.run_step()
        step_history = [trainer.run_step() for _ in range(args.iters)]
        return summarize_and_write_report(
            args=args,
            topology=topology,
            step_history=step_history,
            optimized=optimized,
        )
    finally:
        shutdown_topology(topology)


class MoEHybridEPBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(
        self,
        *,
        optimized: bool,
        multigpu: bool,
        script_path: str,
        label: str,
    ) -> None:
        super().__init__()
        self.optimized = optimized
        self.multigpu = multigpu
        self.script_path = str(Path(script_path).resolve())
        self.name = label
        self.parameter_count = 0
        self.workload_size = 1
        self.tokens_per_rank = 128
        self.register_workload_metadata(requests_per_iteration=1.0, tokens_per_iteration=float(self.tokens_per_rank))
        self._workload_registered = True
        # Keep constructor-side verification state on CPU so benchmark discovery/load
        # does not create a parent-process CUDA context before subprocess isolation.
        self._verify_output = torch.zeros(1, dtype=torch.float32)
        self._verify_probe = torch.zeros(1, dtype=torch.float32)
        self._metrics_sidecar_path = Path(tempfile.gettempdir()) / (
            f"aisp_{label}_metrics.json"
        )
        self._single_gpu_args: Optional[argparse.Namespace] = None
        self._single_gpu_topology: Optional[TopologyInfo] = None
        self._single_gpu_trainer: Optional[HybridEPTrainer] = None
        self._latest_metrics: Dict[str, float] = {}
        self._has_latest_metrics = False

    def benchmark_fn(self) -> None:
        if self._single_gpu_trainer is None:
            self._verify_output.zero_()
            self._has_latest_metrics = False
            return
        artifacts = self._single_gpu_trainer.run_step()
        latest_metrics = self._latest_metrics
        latest_metrics.clear()
        latest_metrics.update(artifacts.metrics)
        latest_metrics["moe_hybrid_ep.workload_size"] = float(self.workload_size)
        self._has_latest_metrics = True
        self._verify_output.fill_(artifacts.loss)

    def setup(self) -> None:
        self._latest_metrics.clear()
        self._has_latest_metrics = False
        self._verify_output.zero_()
        self._metrics_sidecar_path.unlink(missing_ok=True)
        if not self._use_local_single_gpu_runner():
            return
        parser = build_parser(optimized=self.optimized)
        args = parser.parse_args(["--skip-preflight", "--iters", "1"])
        topology = init_topology()
        trainer = HybridEPTrainer(args, topology, optimized=self.optimized)
        self._single_gpu_args = args
        self._single_gpu_topology = topology
        self._single_gpu_trainer = trainer
        self.parameter_count = sum(p.numel() for p in trainer.model.parameters())

    def teardown(self) -> None:
        topology = self._single_gpu_topology
        self._single_gpu_args = None
        self._single_gpu_topology = None
        self._single_gpu_trainer = None
        self._latest_metrics.clear()
        self._has_latest_metrics = False
        self._metrics_sidecar_path.unlink(missing_ok=True)
        if topology is not None:
            shutdown_topology(topology)
        super().teardown()

    def _use_local_single_gpu_runner(self) -> bool:
        return (not self.multigpu) and torch.cuda.is_available() and torch.cuda.device_count() == 1

    def get_config(self) -> BenchmarkConfig:
        visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if not self.multigpu and visible_gpus == 1:
            return BenchmarkConfig(
                launch_via=LaunchVia.PYTHON,
                iterations=1,
                warmup=5,
                use_subprocess=False,
                single_gpu=True,
                measurement_timeout_seconds=1800,
                timing_method="wall_clock",
            )
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=max(1, visible_gpus),
            nnodes=None if self.multigpu else "1",
            iterations=1,
            warmup=2,
            multi_gpu_required=self.multigpu,
            measurement_timeout_seconds=1800,
            timing_method="wall_clock",
        )

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if self._has_latest_metrics:
            return dict(self._latest_metrics)
        if not self._metrics_sidecar_path.exists():
            return {
                "moe_hybrid_ep.workload_size": float(self.workload_size),
            }
        try:
            payload = json.loads(self._metrics_sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "moe_hybrid_ep.workload_size": float(self.workload_size),
            }
        metrics = payload.get("custom_metrics")
        if isinstance(metrics, dict):
            metrics["moe_hybrid_ep.workload_size"] = float(self.workload_size)
            return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        return {
            "moe_hybrid_ep.workload_size": float(self.workload_size),
        }

    def get_torchrun_spec(self, config: BenchmarkConfig | None = None) -> TorchrunLaunchSpec:
        self._prepare_verification_payload()
        self._metrics_sidecar_path.unlink(missing_ok=True)
        script_args = ["--skip-preflight"]
        if self.optimized:
            script_args = ["--optimized", *script_args]
        return TorchrunLaunchSpec(
            module_name=MOE_HYBRID_EP_ENTRYPOINT_MODULE,
            script_args=script_args,
            env={
                "AISP_MOE_HYBRID_EP_METRICS_PATH": str(self._metrics_sidecar_path),
            },
            parse_rank0_only=True,
            multi_gpu_required=self.multigpu,
            name=self.name,
            config_arg_map={
                "iterations": "--iters",
            },
        )

    def get_profile_torchrun_spec(
        self,
        *,
        profiler: str,
        config: BenchmarkConfig | None = None,
        output_path: Optional[Path] = None,
    ) -> Optional[TorchrunLaunchSpec]:
        spec = self.get_torchrun_spec(config)
        if profiler != "torch":
            return spec
        if output_path is None:
            raise ValueError("torch profiler launch requires an output_path")
        return TorchrunLaunchSpec(
            script_path=spec.script_path,
            module_name=spec.module_name,
            script_args=[*spec.script_args, "--torch-profile-output", str(output_path)],
            env=dict(spec.env),
            parse_rank0_only=spec.parse_rank0_only,
            multi_gpu_required=spec.multi_gpu_required,
            name=spec.name,
            config_arg_map=dict(spec.config_arg_map),
        )

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"probe": self._verify_probe},
            output=self._verify_output,
            batch_size=1,
            parameter_count=int(self.parameter_count),
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
            signature_overrides={
                "world_size": max(torch.cuda.device_count(), 1),
                "collective_type": "hybrid_ep_optimizer_step",
            },
        )

    def _prepare_verification_payload(self) -> None:
        if hasattr(self, "_subprocess_verify_output"):
            return
        self.benchmark_fn()
        self.capture_verification_payload()
        self._subprocess_verify_output = self.get_verify_output()
        self._subprocess_output_tolerance = self.get_output_tolerance()
        self._subprocess_input_signature = self.get_input_signature()
