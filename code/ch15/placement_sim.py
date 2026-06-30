"""Lightweight simulator for prefill/decode placement strategies.

The goal is to express the heuristics from Ch15 (multinode inference) in code:
  - Keep MoE expert parallelism inside an NVLink island.
  - Use tensor parallelism for prefill but collapse it to 1 for decode when memory allows.
  - Route sessions to replicas (CP) across nodes while keeping decode sticky to the node
    that holds the KV cache.
  - Penalize cross-node collectives and KV moves to surface tail-latency risks.

The simulator leans on PyTorch PRNG utilities (targeting torch>=2.10 / CUDA 13) so
benchmarks remain deterministic and GPU-friendly on B200/B300 and GB200/GB300.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import torch


def _torch_version_tuple() -> Tuple[int, int]:
    try:
        major, minor, *_ = torch.__version__.split(".")
        return int(major), int(minor)
    except Exception:
        return (0, 0)


_TORCH_MIN = (2, 10)
if _torch_version_tuple() < _TORCH_MIN:
    # Soft guard only; we do not raise to keep CI runs alive on older stacks.
    warnings.warn(
        "[ch15] placement_sim expects torch>=2.10 with CUDA 13 for Blackwell (B200/B300, GB200/GB300); "
        "falling back to compatibility mode.",
        RuntimeWarning,
        stacklevel=1,
    )


@dataclass
class PlacementConfig:
    """Tunable knobs for the placement simulation."""

    prefill_tp_size: int
    prefill_span_nodes: bool
    decode_tp_size: int
    decode_span_nodes: bool
    decode_microbatch: int
    remote_expert_fraction: float
    router_sticky_decode: bool
    kv_transfer_policy: str  # "allow_cross_node" | "local_only"
    prompt_tokens: Tuple[int, int] = (2048, 8192)
    decode_tokens: Tuple[int, int] = (64, 256)
    batch_size: int = 2
    hidden_size: int = 4096
    dtype: torch.dtype = torch.bfloat16
    moe_top_k: int = 2
    notes: str = ""


@dataclass
class PlacementMetrics:
    """Aggregated metrics per simulation run."""

    ttft_ms: List[float]
    decode_ms: List[float]
    ttft_total_ms: float
    decode_total_ms: float
    prefill_collective_ms: float
    decode_collective_ms: float
    kv_transfer_ms: float
    remote_expert_ms: float
    cross_node_kv_moves: int
    cross_node_collectives: int
    sessions: int
    tokens_processed: int


class PlacementSimulator:
    """Compute-level simulator that bakes in NVLink vs cross-node penalties."""

    def __init__(
        self,
        *,
        nvlink_gbps: float = 900.0,
        cross_node_gbps: float = 400.0,
        prefill_tps: float = 22000.0,
        decode_tps: float = 4200.0,
        nodes: Tuple[str, str] = ("node0", "node1"),
    ) -> None:
        """
        Args:
            nvlink_gbps: Aggregate in-island bandwidth (B200/B300 NVSwitch ballpark).
            cross_node_gbps: HDR/NDR-ish IB bandwidth used for collectives/KV moves.
            prefill_tps: Prompt tokens/s a single B200-class GPU can sustain.
            decode_tps: Decode tokens/s per GPU when TP=1.
            nodes: Synthetic node identifiers; two islands by default.
        """

        self.nvlink_gbps = nvlink_gbps
        self.cross_node_gbps = cross_node_gbps
        self.prefill_tps = prefill_tps
        self.decode_tps = decode_tps
        self.nodes = list(nodes)

    def simulate(self, cfg: PlacementConfig, *, sessions: int = 64, seed: int = 0) -> PlacementMetrics:
        """Run a deterministic batch of synthetic sessions."""
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        ttft_ms: List[float] = []
        decode_ms: List[float] = []

        prefill_collective_ms = 0.0
        decode_collective_ms = 0.0
        ttft_total_ms = 0.0
        decode_total_ms = 0.0
        kv_transfer_ms = 0.0
        remote_expert_ms = 0.0
        cross_node_kv_moves = 0
        cross_node_collectives = 0
        tokens_processed = 0

        prompt_token_samples = torch.randint(
            cfg.prompt_tokens[0],
            cfg.prompt_tokens[1] + 1,
            (sessions,),
            generator=g,
        ).tolist()
        decode_token_samples = torch.randint(
            cfg.decode_tokens[0],
            cfg.decode_tokens[1] + 1,
            (sessions,),
            generator=g,
        ).tolist()

        for sess_idx, (prompt_tokens, decode_tokens) in enumerate(
            zip(prompt_token_samples, decode_token_samples)
        ):
            tokens_processed += prompt_tokens + decode_tokens

            prefill_node = self.nodes[sess_idx % len(self.nodes)]
            decode_node = (
                prefill_node if cfg.router_sticky_decode else self.nodes[(sess_idx + 1) % len(self.nodes)]
            )
            kv_move_needed = decode_node != prefill_node
            if kv_move_needed and cfg.kv_transfer_policy == "local_only":
                decode_node = prefill_node
                kv_move_needed = False

            ttft, pref_collective = self._prefill_latency_ms(
                prompt_tokens=prompt_tokens,
                batch_size=cfg.batch_size,
                hidden_size=cfg.hidden_size,
                dtype=cfg.dtype,
                tp_size=cfg.prefill_tp_size,
                span_nodes=cfg.prefill_span_nodes,
            )
            ttft_ms.append(ttft)
            ttft_total_ms += ttft
            prefill_collective_ms += pref_collective
            if cfg.prefill_tp_size > 1 and cfg.prefill_span_nodes:
                cross_node_collectives += 1

            kv_ms = (
                self._kv_transfer_ms(
                    prompt_tokens=prompt_tokens,
                    batch_size=cfg.batch_size,
                    hidden_size=cfg.hidden_size,
                    dtype=cfg.dtype,
                    local=not kv_move_needed,
                )
                if kv_move_needed
                else 0.0
            )
            if kv_move_needed:
                kv_transfer_ms += kv_ms
                cross_node_kv_moves += 1

            decode_time, decode_collective, expert_penalty = self._decode_latency_ms(
                decode_tokens=decode_tokens,
                batch_size=cfg.batch_size,
                hidden_size=cfg.hidden_size,
                dtype=cfg.dtype,
                tp_size=cfg.decode_tp_size,
                span_nodes=cfg.decode_span_nodes,
                microbatch=cfg.decode_microbatch,
                remote_expert_fraction=cfg.remote_expert_fraction,
                moe_top_k=cfg.moe_top_k,
            )
            total_decode_ms = decode_time + kv_ms
            decode_ms.append(total_decode_ms)
            decode_total_ms += total_decode_ms
            decode_collective_ms += decode_collective
            remote_expert_ms += expert_penalty
            if cfg.decode_tp_size > 1 and cfg.decode_span_nodes:
                cross_node_collectives += 1

        return PlacementMetrics(
            ttft_ms=ttft_ms,
            decode_ms=decode_ms,
            ttft_total_ms=ttft_total_ms,
            decode_total_ms=decode_total_ms,
            prefill_collective_ms=prefill_collective_ms,
            decode_collective_ms=decode_collective_ms,
            kv_transfer_ms=kv_transfer_ms,
            remote_expert_ms=remote_expert_ms,
            cross_node_kv_moves=cross_node_kv_moves,
            cross_node_collectives=cross_node_collectives,
            sessions=sessions,
            tokens_processed=tokens_processed,
        )

    def _prefill_latency_ms(
        self,
        *,
        prompt_tokens: int,
        batch_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        tp_size: int,
        span_nodes: bool,
    ) -> Tuple[float, float]:
        """Return (latency_ms, collective_ms)."""
        tp = max(1, tp_size)
        bytes_per_token = batch_size * hidden_size * self._dtype_bytes(dtype)
        compute_ms = (prompt_tokens / (self.prefill_tps * tp)) * 1000.0
        collective_ms = 0.0
        if tp > 1:
            bw = self.cross_node_gbps if span_nodes else self.nvlink_gbps
            collective_ms = self._allreduce_ms(bytes_per_token * prompt_tokens, bw, shards=tp)
        return compute_ms, collective_ms

    def _decode_latency_ms(
        self,
        *,
        decode_tokens: int,
        batch_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        tp_size: int,
        span_nodes: bool,
        microbatch: int,
        remote_expert_fraction: float,
        moe_top_k: int,
    ) -> Tuple[float, float, float]:
        """Return (latency_ms, collective_ms, expert_penalty_ms)."""
        tp = max(1, tp_size)
        mb = max(1, microbatch)
        steps = math.ceil(decode_tokens / mb)
        bytes_per_token = batch_size * hidden_size * self._dtype_bytes(dtype)

        total_ms = 0.0
        collective_ms = 0.0
        expert_ms = 0.0
        for step in range(steps):
            tokens_step = min(mb, decode_tokens - step * mb)
            compute_ms = (tokens_step / (self.decode_tps * tp)) * 1000.0
            total_ms += compute_ms

            if tp > 1:
                bw = self.cross_node_gbps if span_nodes else self.nvlink_gbps
                collective = self._allreduce_ms(bytes_per_token * tokens_step, bw, shards=tp)
                collective_ms += collective
                total_ms += collective

            # Remote expert all-to-all penalty (fraction of tokens per step).
            remote_tokens = tokens_step * max(0.0, min(1.0, remote_expert_fraction))
            if remote_tokens > 0:
                bw = self.cross_node_gbps if span_nodes else self.nvlink_gbps
                expert_cost = self._alltoall_ms(
                    bytes_per_token * remote_tokens * moe_top_k,
                    bw,
                    shards=tp,
                )
                expert_ms += expert_cost
                total_ms += expert_cost

        return total_ms, collective_ms, expert_ms

    def _kv_transfer_ms(
        self,
        *,
        prompt_tokens: int,
        batch_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        local: bool,
    ) -> float:
        """KV move between pools: local=NVLink vs cross-node."""
        bw = self.nvlink_gbps if local else self.cross_node_gbps
        bytes_total = batch_size * prompt_tokens * hidden_size * self._dtype_bytes(dtype) * 2  # K + V
        return (bytes_total * 8.0 / 1e9) / bw * 1000.0

    @staticmethod
    def _dtype_bytes(dtype: torch.dtype) -> int:
        return torch.finfo(dtype).bits // 8

    @staticmethod
    def _allreduce_ms(bytes_total: float, bw_gbps: float, *, shards: int) -> float:
        # Approximation: 2 * (shards - 1) / shards * message size / bandwidth
        if shards <= 1:
            return 0.0
        factor = 2.0 * (shards - 1) / shards
        return factor * (bytes_total * 8.0 / 1e9) / bw_gbps * 1000.0

    @staticmethod
    def _alltoall_ms(bytes_total: float, bw_gbps: float, *, shards: int) -> float:
        if shards <= 1:
            return 0.0
        return (bytes_total * 8.0 / 1e9) / bw_gbps * 1000.0


def _percentile_from_ordered(xs: List[float], pct: float) -> float:
    assert 0.0 <= pct <= 100.0
    k = (len(xs) - 1) * (pct / 100.0)
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return xs[int(k)]
    return xs[lower] * (upper - k) + xs[upper] * (k - lower)


def percentile(data: List[float], pct: float) -> float:
    """Lightweight percentile helper that tolerates empty lists."""
    if not data:
        return 0.0
    data.sort()
    return _percentile_from_ordered(data, pct)


def percentiles(data: List[float], pcts: Tuple[float, ...]) -> Tuple[float, ...]:
    """Compute several percentiles after sorting the input once."""
    if not data:
        return tuple(0.0 for _ in pcts)
    data.sort()
    return tuple(_percentile_from_ordered(data, pct) for pct in pcts)
