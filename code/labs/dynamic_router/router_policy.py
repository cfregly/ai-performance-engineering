"""
Telemetry-driven router policy scaffold with prefill/decode separation.

Key features:
  - EWMA-smoothed TTFT/TPOT metrics
  - Scoring-based routing for prefill and decode pools
  - KV locality hinting for decode placement
  - Optional migration planner with a sliding-window budget

This file is engine-agnostic: fill the INTEGRATION POINTs in your lab driver or
serving harness (vLLM, SGLang, TensorRT-LLM).
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple

# -------------------------
# EWMA helper
# -------------------------


class EWMA:
    """Simple exponential weighted moving average."""

    def __init__(self, alpha: float, initial: Optional[float] = None) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.value = initial
        self._initialized = initial is not None

    def update(self, x: float) -> float:
        if not self._initialized:
            self.value = x
            self._initialized = True
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value

    def get(self, default: float = 0.0) -> float:
        return self.value if self._initialized else default


# -------------------------
# Metric and state structs
# -------------------------


@dataclass
class GPUMetrics:
    ttft_ms: EWMA
    tpot: EWMA
    queue_depth: EWMA
    mem_free_gb: EWMA
    kv_hit_rate: EWMA


@dataclass
class GPUState:
    gpu_id: str
    is_prefill: bool
    is_decode: bool
    hourly_cost: float
    metrics: GPUMetrics
    numa_node: Optional[int] = None
    last_raw_metrics: Dict[str, float] = field(default_factory=dict)

    def update_metrics(self, raw: Dict[str, float]) -> None:
        """Ingest raw metrics into the EWMAs."""
        self.last_raw_metrics = raw
        if "ttft_ms" in raw:
            self.metrics.ttft_ms.update(raw["ttft_ms"])
        if "tpot" in raw:
            self.metrics.tpot.update(raw["tpot"])
        if "queue_depth" in raw:
            self.metrics.queue_depth.update(raw["queue_depth"])
        if "mem_free_gb" in raw:
            self.metrics.mem_free_gb.update(raw["mem_free_gb"])
        if "kv_hit_rate" in raw:
            self.metrics.kv_hit_rate.update(raw["kv_hit_rate"])

    def snapshot(self) -> Dict[str, float]:
        """Return the smoothed metrics for scoring."""
        return {
            "ttft_ms": self.metrics.ttft_ms.get(),
            "tpot": self.metrics.tpot.get(),
            "queue_depth": self.metrics.queue_depth.get(),
            "mem_free_gb": self.metrics.mem_free_gb.get(),
            "kv_hit_rate": self.metrics.kv_hit_rate.get(),
        }


@dataclass
class SequenceInfo:
    """
    Minimal sequence metadata for decode routing and migration.

    kv_gpus: GPUs that already hold KV for this sequence.
    priority: lower is more important.
    """

    seq_id: str
    current_gpu: str
    kv_gpus: Set[str]
    expected_tokens_remaining: Optional[int] = None
    priority: int = 0
    numa_node: Optional[int] = None


# -------------------------
# Scoring
# -------------------------


def default_scoring_fn(
    gpu: GPUState,
    snap: Dict[str, float],
    *,
    role: str,
    kv_locality_boost: float = 0.0,
    queue_weight: float = 1.0,
) -> float:
    """Higher is better."""
    ttft_ms = snap["ttft_ms"]
    tpot = snap["tpot"]
    qd = snap["queue_depth"]
    mem_free = snap["mem_free_gb"]

    inv_ttft = 1.0 / (1.0 + ttft_ms)
    inv_qd = queue_weight / (1.0 + qd)

    if role == "prefill":
        score = (
            1.5 * inv_ttft
            + 0.5 * tpot
            + 1.0 * mem_free
            + 0.5 * inv_qd
        )
    else:
        score = (
            0.5 * inv_ttft
            + 1.5 * tpot
            + 0.5 * mem_free
            + 1.0 * inv_qd
        )

    score += kv_locality_boost
    return score


ScoringFn = Callable[[GPUState, Dict[str, float]], float]


# -------------------------
# Router
# -------------------------


class Router:
    """
    Dynamic router with pool-aware routing and optional migration.
    """

    def __init__(
        self,
        ewma_alpha: float = 0.3,
        scoring_fn_prefill: Optional[ScoringFn] = None,
        scoring_fn_decode: Optional[ScoringFn] = None,
        migration_budget_per_window: int = 32,
        migration_window_seconds: float = 1.0,
        migration_min_score_gap: float = 0.5,
        kv_locality_boost: float = 0.3,
        queue_urgency: float = 1.0,
        decode_cost_penalty: float = 0.0,
    ) -> None:
        self._gpus: Dict[str, GPUState] = {}
        self._scoring_fn_prefill = scoring_fn_prefill
        self._scoring_fn_decode = scoring_fn_decode

        # Migration budget state
        self._budget = migration_budget_per_window
        self._window_s = migration_window_seconds
        self._min_gap = migration_min_score_gap
        self._migration_times: Deque[float] = deque()

        self._ewma_alpha = ewma_alpha
        self._kv_locality_boost = kv_locality_boost
        self._queue_urgency = queue_urgency
        self._decode_cost_penalty = decode_cost_penalty

    # Registration
    def register_gpu(
        self,
        gpu_id: str,
        *,
        is_prefill: bool,
        is_decode: bool,
        hourly_cost: float = 1.0,
        numa_node: Optional[int] = None,
    ) -> None:
        """Register a GPU in one or both pools."""
        if gpu_id in self._gpus:
            st = self._gpus[gpu_id]
            st.is_prefill = is_prefill
            st.is_decode = is_decode
            st.hourly_cost = hourly_cost
            st.numa_node = numa_node
            return

        metrics = GPUMetrics(
            ttft_ms=EWMA(self._ewma_alpha),
            tpot=EWMA(self._ewma_alpha),
            queue_depth=EWMA(self._ewma_alpha),
            mem_free_gb=EWMA(self._ewma_alpha),
            kv_hit_rate=EWMA(self._ewma_alpha),
        )
        self._gpus[gpu_id] = GPUState(
            gpu_id=gpu_id,
            is_prefill=is_prefill,
            is_decode=is_decode,
            hourly_cost=hourly_cost,
            metrics=metrics,
            numa_node=numa_node,
        )

    # Metrics ingestion
    def update_metrics(self, gpu_id: str, raw_metrics: Dict[str, float]) -> None:
        """Called from telemetry path."""
        gpu = self._gpus.get(gpu_id)
        if not gpu:
            return
        gpu.update_metrics(raw_metrics)

    # Scoring helpers
    def _score_prefill_gpu(self, gpu: GPUState) -> float:
        snap = gpu.snapshot()
        if self._scoring_fn_prefill:
            return self._scoring_fn_prefill(gpu, snap)
        return default_scoring_fn(gpu, snap, role="prefill", queue_weight=self._queue_urgency)

    def _score_decode_gpu(self, gpu: GPUState, kv_local: bool = False) -> float:
        snap = gpu.snapshot()
        if self._scoring_fn_decode:
            base = self._scoring_fn_decode(gpu, snap)
        else:
            base = default_scoring_fn(gpu, snap, role="decode", kv_locality_boost=0.0, queue_weight=self._queue_urgency)
        kv_boost = self._kv_locality_boost if kv_local else 0.0
        score = base + kv_boost
        if self._decode_cost_penalty > 0.0:
            score /= max(1e-6, gpu.hourly_cost ** self._decode_cost_penalty)
        return score

    # Routing
    def choose_prefill_gpu(self) -> Optional[str]:
        """Pick a GPU for a new prefill request."""
        best_score = float("-inf")
        best_gpu = None
        for gpu in self._gpus.values():
            if not gpu.is_prefill:
                continue
            score = self._score_prefill_gpu(gpu)
            if score > best_score:
                best_score = score
                best_gpu = gpu.gpu_id
        return best_gpu

    def choose_decode_gpu(self, seq: SequenceInfo) -> Optional[str]:
        """Pick a GPU for decode, preferring KV-local placements."""
        best_score = float("-inf")
        best_gpu = None
        for gpu in self._gpus.values():
            if not gpu.is_decode:
                continue
            kv_local = gpu.gpu_id in seq.kv_gpus
            score = self._score_decode_gpu(gpu, kv_local=kv_local)
            if score > best_score:
                best_score = score
                best_gpu = gpu.gpu_id
        return best_gpu

    # Migration helpers
    def _refresh_budget(self) -> None:
        now = time.time()
        window_start = now - self._window_s
        while self._migration_times and self._migration_times[0] < window_start:
            self._migration_times.popleft()

    def remaining_budget(self) -> int:
        self._refresh_budget()
        return max(self._budget - len(self._migration_times), 0)

    def _consume_budget(self, n: int) -> None:
        now = time.time()
        for _ in range(n):
            self._migration_times.append(now)

    def plan_migrations(
        self, active_sequences: Iterable[SequenceInfo], max_per_call: Optional[int] = None
    ) -> List[Tuple[str, str, str]]:
        """
        Suggest (seq_id, src_gpu, dst_gpu) moves when decode scores are imbalanced.
        """
        scores = {
            gpu.gpu_id: self._score_decode_gpu(gpu)
            for gpu in self._gpus.values()
            if gpu.is_decode
        }
        if len(scores) < 2:
            return []

        best = max(scores, key=scores.get)
        worst = min(scores, key=scores.get)
        if scores[best] - scores[worst] < self._min_gap:
            return []

        budget = self.remaining_budget()
        if max_per_call is not None:
            budget = min(budget, max_per_call)
        if budget <= 0:
            return []

        by_gpu: Dict[str, List[SequenceInfo]] = {}
        for s in active_sequences:
            by_gpu.setdefault(s.current_gpu, []).append(s)

        # Move low-priority, long sequences first from lowest-scoring GPUs.
        migrations: List[Tuple[str, str, str]] = []
        active_source_ids = [gpu_id for gpu_id in by_gpu if gpu_id in scores]
        for src in heapq.nsmallest(len(active_source_ids), active_source_ids, key=scores.get):
            if budget <= 0:
                break
            seq_heap = [
                (s.priority, -(s.expected_tokens_remaining or 0), order, s)
                for order, s in enumerate(by_gpu[src])
            ]
            heapq.heapify(seq_heap)
            while seq_heap and budget > 0:
                _, _, _, seq = heapq.heappop(seq_heap)
                dst = self._choose_migration_target(seq, scores)
                if dst is None or dst == src:
                    continue
                migrations.append((seq.seq_id, src, dst))
                budget -= 1

        if migrations:
            self._consume_budget(len(migrations))
        return migrations

    def _choose_migration_target(
        self, seq: SequenceInfo, scores: Dict[str, float], kv_boost: float = 0.3
    ) -> Optional[str]:
        """Pick best destination incorporating KV locality."""
        best_score = float("-inf")
        best_gpu = None
        for gid, base_score in scores.items():
            score = base_score + (kv_boost if gid in seq.kv_gpus else 0.0)
            if score > best_score:
                best_score = score
                best_gpu = gid
        return best_gpu
