"""Shared cache-aware disaggregated inference benchmark helpers.

This lab is a reproducible logical simulation of the Together AI
"Cache-Aware Disaggregated Inference" article. It keeps the key contrast:

- baseline: cache-unaware round-robin handoff across logical decode workers
- optimized: cache-affine placement that preserves warm KV locality

The implementation intentionally uses logical workers on one GPU so the lab can
be exercised locally without requiring a full cluster.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from ch17.prefill_decode_disagg_multigpu_common import TinyPrefillDecode
from core.benchmark.cuda_event_timing import elapsed_ms
from core.benchmark.metrics import compute_inference_metrics
from core.benchmark.verification import PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


@dataclass(frozen=True)
class CacheAwareDisaggConfig:
    hidden_size: int = 512
    num_layers: int = 4
    batch_size: int = 2
    requests_per_iteration: int = 12
    context_window: int = 1536
    chunk_size: int = 256
    decode_tokens: int = 96
    logical_decode_workers: int = 3
    warm_request_ratio: float = 0.5
    warm_prefix_ratio: float = 0.5
    dtype: torch.dtype = torch.bfloat16

    @property
    def num_chunks(self) -> int:
        return max(1, math.ceil(self.context_window / self.chunk_size))

    @property
    def total_requests(self) -> int:
        return self.requests_per_iteration * self.batch_size


@dataclass(frozen=True)
class RequestPlan:
    request_idx: int
    warm_chunks: int
    total_chunks: int

    @property
    def is_warm(self) -> bool:
        return self.warm_chunks > 0


def _split_prompt(prompt: torch.Tensor, chunk_size: int) -> Sequence[torch.Tensor]:
    return tuple(prompt.split(chunk_size, dim=1))


def _empty_kv(cfg: CacheAwareDisaggConfig, device: torch.device) -> torch.Tensor:
    return torch.empty(
        cfg.batch_size,
        0,
        cfg.hidden_size,
        device=device,
        dtype=cfg.dtype,
    )


def _extend_cache_buffer(
    cfg: CacheAwareDisaggConfig,
    *,
    request_id: int,
    cache: torch.Tensor,
    chunk_kv: torch.Tensor,
    kv_buffers: Dict[int, torch.Tensor],
) -> torch.Tensor:
    prefix, append_view = _reserve_cache_append_buffer(
        cfg,
        request_id=request_id,
        cache=cache,
        append_tokens=int(chunk_kv.size(1)),
        append_device=chunk_kv.device,
        append_dtype=chunk_kv.dtype,
        kv_buffers=kv_buffers,
    )
    append_view.copy_(chunk_kv)
    return prefix


def _reserve_cache_append_buffer(
    cfg: CacheAwareDisaggConfig,
    *,
    request_id: int,
    cache: torch.Tensor,
    append_tokens: int,
    append_device: torch.device,
    append_dtype: torch.dtype,
    kv_buffers: Dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    current_kv_len = cache.size(1)
    next_kv_len = current_kv_len + append_tokens
    kv_buffer = kv_buffers.get(request_id)
    if (
        kv_buffer is None
        or kv_buffer.device != append_device
        or kv_buffer.dtype != append_dtype
        or kv_buffer.size(1) < next_kv_len
    ):
        buffer_tokens = max(cfg.context_window, next_kv_len)
        kv_buffer = torch.empty(
            cfg.batch_size,
            buffer_tokens,
            cfg.hidden_size,
            device=append_device,
            dtype=append_dtype,
        )
        kv_buffers[request_id] = kv_buffer

    if current_kv_len and cache.data_ptr() != kv_buffer.data_ptr():
        kv_buffer[:, :current_kv_len].copy_(cache)
    return kv_buffer[:, :next_kv_len], kv_buffer[:, current_kv_len:next_kv_len]


class CacheAwareDisaggBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Single-GPU logical-worker benchmark for cache-aware disaggregation."""

    story_metadata = {
        "pair_role": "exemplar",
        "chapter_alignment": "supplementary",
        "chapter_native_exemplar": True,
        "optimization_mechanism": "preserve decode-worker KV locality instead of round-robin chunk handoff",
        "benchmark_story": "cache-aware disaggregated inference",
    }

    def __init__(
        self,
        *,
        optimized: bool,
        label: str,
        cfg: Optional[CacheAwareDisaggConfig] = None,
        script_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.optimized = optimized
        self.label = label
        self.cfg = cfg or CacheAwareDisaggConfig()
        self.script_path = script_path or ""
        self._logical_decode_worker_count = self.cfg.logical_decode_workers
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.cfg.total_requests),
            tokens_per_iteration=float(
                self.cfg.total_requests * (self.cfg.context_window + self.cfg.decode_tokens)
            ),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.cfg.total_requests),
            tokens_per_iteration=float(
                self.cfg.total_requests * (self.cfg.context_window + self.cfg.decode_tokens)
            ),
        )
        self.prefill_model: Optional[TinyPrefillDecode] = None
        self.decode_model: Optional[TinyPrefillDecode] = None
        self.prompts: Optional[torch.Tensor] = None
        self.request_plans: List[RequestPlan] = []
        self.shared_prefix_store: Dict[int, torch.Tensor] = {}
        self.shared_seed_store: Dict[int, torch.Tensor] = {}
        self.output: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self.reload_materialization_passes = 2
        self._timing_history: Dict[str, List[float]] = {"ttft": [], "tpot": []}
        self._custom_metrics: Dict[str, float] = {}
        self._empty_kv_template: Optional[torch.Tensor] = None
        self._last_outputs: List[torch.Tensor] = []
        self._output_stack: Optional[torch.Tensor] = None
        self._outputs_ready = False
        self._request_event_pool: List[tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]] = []
        self._request_event_triplets: List[tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]] = []
        self._request_event_count = 0
        self._pending_metrics: Dict[str, float] = {}
        self._kv_buffers: Dict[int, torch.Tensor] = {}
        self._reload_buffers: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._worker_caches: List[Dict[int, torch.Tensor]] = []
        self._worker_cache_count = 0
        self._owners: Dict[int, int] = {}
        self._prompt_chunks: List[Sequence[torch.Tensor]] = []
        self._prompt_chunk_count = 0
        self._request_plan_count = 0
        self._warm_request_count = 0
        self._request_event_groups: List[tuple[int, RequestPlan]] = []
        self._request_event_group_count = 0
        self._last_output_count = 0
        self._decode_chunk_ranges: Dict[int, range] = {}
        self._decode_chunk_range_count = 0
        self._decode_token_divisor = float(max(self.cfg.decode_tokens, 1))
        self._prefill_seed_buffer: Optional[torch.Tensor] = None

    def _build_request_plans(self) -> List[RequestPlan]:
        warm_requests = int(round(self.cfg.requests_per_iteration * self.cfg.warm_request_ratio))
        warm_chunks = min(
            self.cfg.num_chunks - 1,
            max(0, int(round(self.cfg.num_chunks * self.cfg.warm_prefix_ratio))),
        )
        plans: List[RequestPlan] = []
        for request_idx in range(self.cfg.requests_per_iteration):
            plans.append(
                RequestPlan(
                    request_idx=request_idx,
                    warm_chunks=warm_chunks if request_idx < warm_requests else 0,
                    total_chunks=self.cfg.num_chunks,
                )
            )
        return plans

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for cache-aware disaggregated inference lab")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.request_plans = self._build_request_plans()
        self._request_plan_count = len(self.request_plans)
        self._decode_chunk_ranges = {
            plan.request_idx: range(plan.warm_chunks, plan.total_chunks)
            for plan in self.request_plans
        }
        self._decode_chunk_range_count = len(self._decode_chunk_ranges)
        self._request_event_groups = list(enumerate(self.request_plans))
        self._request_event_group_count = len(self._request_event_groups)
        self.prompts = torch.randn(
            self.cfg.requests_per_iteration,
            self.cfg.batch_size,
            self.cfg.context_window,
            self.cfg.hidden_size,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        self._prompt_chunks = [
            _split_prompt(self.prompts[request_idx], self.cfg.chunk_size)
            for request_idx in range(self.cfg.requests_per_iteration)
        ]
        self._prompt_chunk_count = len(self._prompt_chunks)
        self._warm_request_count = sum(1 for plan in self.request_plans if plan.is_warm)

        reference = TinyPrefillDecode(
            self.cfg.hidden_size,
            self.cfg.num_layers,
            torch.device("cpu"),
            torch.float32,
        ).eval()
        reference_state = {name: tensor.detach().cpu() for name, tensor in reference.state_dict().items()}
        self.prefill_model = TinyPrefillDecode(
            self.cfg.hidden_size,
            self.cfg.num_layers,
            self.device,
            self.cfg.dtype,
        ).eval()
        self.prefill_model.load_state_dict(reference_state)
        self.decode_model = TinyPrefillDecode(
            self.cfg.hidden_size,
            self.cfg.num_layers,
            self.device,
            self.cfg.dtype,
        ).eval()
        self.decode_model.load_state_dict(reference_state)
        self.parameter_count = sum(p.numel() for p in self.prefill_model.parameters()) + sum(
            p.numel() for p in self.decode_model.parameters()
        )

        self.shared_prefix_store = {}
        self.shared_seed_store = {}
        self._empty_kv_template = torch.empty(
            self.cfg.batch_size,
            0,
            self.cfg.hidden_size,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        self._kv_buffers = {
            plan.request_idx: torch.empty(
                self.cfg.batch_size,
                self.cfg.context_window,
                self.cfg.hidden_size,
                device=self.device,
                dtype=self.cfg.dtype,
            )
            for plan in self.request_plans
        }
        self._reload_buffers = {
            plan.request_idx: (
                torch.empty(
                    self.cfg.batch_size,
                    self.cfg.context_window,
                    self.cfg.hidden_size,
                    device=self.device,
                    dtype=self.cfg.dtype,
                ),
                torch.empty(
                    self.cfg.batch_size,
                    self.cfg.context_window,
                    self.cfg.hidden_size,
                    device=self.device,
                    dtype=self.cfg.dtype,
                ),
            )
            for plan in self.request_plans
        }
        self._prefill_seed_buffer = torch.empty(
            self.cfg.batch_size,
            self.cfg.hidden_size,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        self._worker_caches = [{} for _ in range(self._logical_decode_worker_count)]
        self._worker_cache_count = len(self._worker_caches)
        self._owners = {}
        with torch.inference_mode():
            assert self.prompts is not None
            for plan in self.request_plans:
                chunks = self._prompt_chunks[plan.request_idx]
                if plan.warm_chunks <= 0:
                    self.shared_prefix_store[plan.request_idx] = self._empty_kv()
                    continue
                warm_chunks = chunks[: plan.warm_chunks]
                prefix_tokens = sum(int(chunk.size(1)) for chunk in warm_chunks)
                prefix_buffer = torch.empty(
                    self.cfg.batch_size,
                    prefix_tokens,
                    self.cfg.hidden_size,
                    device=self.device,
                    dtype=self.cfg.dtype,
                )
                seed: Optional[torch.Tensor] = None
                offset = 0
                for chunk in warm_chunks:
                    next_offset = offset + int(chunk.size(1))
                    _, seed = self.prefill_model.prefill_into(
                        chunk,
                        prefix_buffer[:, offset:next_offset],
                    )
                    offset = next_offset
                self.shared_prefix_store[plan.request_idx] = prefix_buffer
                if seed is not None:
                    self.shared_seed_store[plan.request_idx] = seed

        self.output = None
        self._output_stack = torch.empty(
            self._request_plan_count,
            self.cfg.batch_size,
            self.cfg.hidden_size,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        self._last_outputs = list(self._output_stack.unbind(0))
        self._last_output_count = self._output_stack.size(0)
        self._outputs_ready = False
        self._timing_history = {"ttft": [], "tpot": []}
        self._custom_metrics.clear()
        self._request_event_pool = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in self.request_plans
        ]
        self._request_event_count = len(self._request_event_pool)
        self._request_event_triplets = []
        self._pending_metrics = {
            "cache_hits": 0.0,
            "cache_misses": 0.0,
            "shared_reloads": 0.0,
            "peer_reloads": 0.0,
            "worker_switches": 0.0,
            "kv_transfer_bytes": 0.0,
            "warm_requests": float(self._warm_request_count),
            "warm_requests_served_local": 0.0,
        }
        torch.cuda.synchronize(self.device)

    def _empty_kv(self) -> torch.Tensor:
        if self._empty_kv_template is None:
            raise RuntimeError("Empty KV template not initialized")
        return self._empty_kv_template

    def _reload_cache_buffer_pair(self, request_idx: int, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reload_buffers = self._reload_buffers.get(request_idx)
        required_tokens = int(source.size(1))
        if (
            reload_buffers is None
            or reload_buffers[0].device != source.device
            or reload_buffers[0].dtype != source.dtype
            or reload_buffers[0].size(1) < required_tokens
        ):
            buffer_tokens = max(self.cfg.context_window, required_tokens)
            reload_buffers = (
                torch.empty(
                    self.cfg.batch_size,
                    buffer_tokens,
                    self.cfg.hidden_size,
                    device=source.device,
                    dtype=source.dtype,
                ),
                torch.empty(
                    self.cfg.batch_size,
                    buffer_tokens,
                    self.cfg.hidden_size,
                    device=source.device,
                    dtype=source.dtype,
                ),
            )
            self._reload_buffers[request_idx] = reload_buffers
        return reload_buffers

    def _materialize_reload_cache(self, request_idx: int, source: torch.Tensor) -> torch.Tensor:
        primary, secondary = self._reload_cache_buffer_pair(request_idx, source)
        source_tokens = int(source.size(1))
        views = (primary[:, :source_tokens], secondary[:, :source_tokens])
        cache = views[0]
        cache.copy_(source)
        for pass_idx in range(self.reload_materialization_passes):
            next_cache = views[(pass_idx + 1) & 1]
            next_cache.copy_(cache)
            cache = next_cache
        return cache

    def _choose_worker(
        self,
        request_idx: int,
        stage_idx: int,
        owner: Optional[int],
    ) -> int:
        if self.optimized and owner is not None:
            return owner
        if self.optimized:
            return request_idx % self._logical_decode_worker_count
        return (request_idx + stage_idx) % self._logical_decode_worker_count

    def _load_request_cache(
        self,
        *,
        request_idx: int,
        worker_id: int,
        worker_caches: List[Dict[int, torch.Tensor]],
        owners: Dict[int, int],
        metrics: Dict[str, float],
    ) -> torch.Tensor:
        owner = owners.get(request_idx)
        if owner == worker_id and request_idx in worker_caches[worker_id]:
            metrics["cache_hits"] += 1.0
            return worker_caches[worker_id][request_idx]

        source: Optional[torch.Tensor] = None
        if owner is not None and request_idx in worker_caches[owner]:
            source = worker_caches[owner].pop(request_idx)
            metrics["peer_reloads"] += 1.0
            metrics["worker_switches"] += 1.0
        else:
            source = self.shared_prefix_store.get(request_idx)
            if source is not None and source.numel() > 0:
                metrics["shared_reloads"] += 1.0
            elif owner is not None and owner != worker_id:
                metrics["worker_switches"] += 1.0

        if source is None:
            source = self._empty_kv()

        if source.numel() > 0:
            cache = self._materialize_reload_cache(request_idx, source)
            metrics["kv_transfer_bytes"] += float(
                cache.numel()
                * cache.element_size()
                * (1 + self.reload_materialization_passes)
            )
            metrics["cache_misses"] += 1.0
        else:
            cache = source
            if owner is not None:
                metrics["cache_misses"] += 1.0

        worker_caches[worker_id][request_idx] = cache
        owners[request_idx] = worker_id
        return cache

    def benchmark_fn(self) -> None:
        if self.prefill_model is None or self.decode_model is None or self.prompts is None:
            raise RuntimeError("setup() must run before benchmark_fn()")

        worker_caches = self._worker_caches
        if self._worker_cache_count != self._logical_decode_worker_count:
            raise RuntimeError("Worker cache slots not initialized")
        for cache in worker_caches:
            cache.clear()
        owners = self._owners
        owners.clear()
        kv_buffers = self._kv_buffers
        outputs = self._last_outputs
        request_plan_count = self._request_plan_count
        if self._last_output_count != request_plan_count:
            raise RuntimeError("Decode output slots not initialized")
        if self._decode_chunk_range_count != request_plan_count:
            raise RuntimeError("Decode chunk ranges not initialized")
        prompt_chunks = self._prompt_chunks
        if self._prompt_chunk_count != request_plan_count:
            raise RuntimeError("Prompt chunk views not initialized")
        prefill_seed_buffer = self._prefill_seed_buffer
        if prefill_seed_buffer is None:
            raise RuntimeError("Prefill seed buffer not initialized")
        request_event_groups = self._request_event_groups
        if self._request_event_group_count != request_plan_count:
            raise RuntimeError("Request event groups not initialized")
        metrics = self._pending_metrics
        metrics["cache_hits"] = 0.0
        metrics["cache_misses"] = 0.0
        metrics["shared_reloads"] = 0.0
        metrics["peer_reloads"] = 0.0
        metrics["worker_switches"] = 0.0
        metrics["kv_transfer_bytes"] = 0.0
        metrics["warm_requests"] = float(self._warm_request_count)
        metrics["warm_requests_served_local"] = 0.0
        request_events = self._request_event_pool
        if self._request_event_count != request_plan_count:
            raise RuntimeError("Request timing events not initialized")
        decode_chunk_ranges = self._decode_chunk_ranges

        current_stream = torch.cuda.current_stream()
        with torch.inference_mode():
            for event_idx, plan in request_event_groups:
                request_start, prefill_end, decode_end = request_events[event_idx]
                request_start.record(current_stream)

                chunks = prompt_chunks[plan.request_idx]
                seed = self.shared_seed_store.get(plan.request_idx)
                owner = owners.get(plan.request_idx)
                current_worker = self._choose_worker(plan.request_idx, 0, owner)
                accumulated_kv = self._load_request_cache(
                    request_idx=plan.request_idx,
                    worker_id=current_worker,
                    worker_caches=worker_caches,
                    owners=owners,
                    metrics=metrics,
                )
                if plan.is_warm and owners.get(plan.request_idx) == current_worker:
                    metrics["warm_requests_served_local"] += 1.0

                for chunk_idx in decode_chunk_ranges[plan.request_idx]:
                    chunk = chunks[chunk_idx]
                    current_worker = self._choose_worker(
                        plan.request_idx,
                        chunk_idx,
                        owners.get(plan.request_idx),
                    )
                    accumulated_kv = self._load_request_cache(
                        request_idx=plan.request_idx,
                        worker_id=current_worker,
                        worker_caches=worker_caches,
                        owners=owners,
                        metrics=metrics,
                    )
                    accumulated_kv, append_kv = _reserve_cache_append_buffer(
                        self.cfg,
                        request_id=plan.request_idx,
                        cache=accumulated_kv,
                        append_tokens=int(chunk.size(1)),
                        append_device=chunk.device,
                        append_dtype=chunk.dtype,
                        kv_buffers=kv_buffers,
                    )
                    _, seed = self.prefill_model.prefill_into(
                        chunk,
                        append_kv,
                        prefill_seed_buffer,
                    )
                    worker_caches[current_worker][plan.request_idx] = accumulated_kv
                    owners[plan.request_idx] = current_worker

                decode_worker = self._choose_worker(
                    plan.request_idx,
                    plan.total_chunks,
                    owners.get(plan.request_idx),
                )
                accumulated_kv = self._load_request_cache(
                    request_idx=plan.request_idx,
                    worker_id=decode_worker,
                    worker_caches=worker_caches,
                    owners=owners,
                    metrics=metrics,
                )
                prefill_end.record(current_stream)
                if seed is None:
                    raise RuntimeError("Request finished without a decode seed")
                outputs[event_idx].copy_(
                    self.decode_model.decode(seed, accumulated_kv, self.cfg.decode_tokens)
                )
                decode_end.record(current_stream)

        self._last_outputs = outputs
        self._outputs_ready = True
        self.output = self._output_stack
        self._request_event_triplets = request_events
        self._pending_metrics = metrics

    def capture_verification_payload(self) -> None:
        if self.prompts is None or not self._outputs_ready:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        if self.output is None:
            if self._output_stack is None:
                raise RuntimeError("Output stack buffer not initialized")
            self.output = self._output_stack
        request_ttft: List[float] = []
        request_tpot: List[float] = []
        ttft_total_ms = 0.0
        tpot_total_ms = 0.0
        for start, prefill_end, decode_end in self._request_event_triplets:
            ttft_ms = elapsed_ms((start, prefill_end))
            total_ms = elapsed_ms((start, decode_end))
            tpot_ms = max(total_ms - ttft_ms, 0.0) / self._decode_token_divisor
            request_ttft.append(ttft_ms)
            request_tpot.append(tpot_ms)
            ttft_total_ms += ttft_ms
            tpot_total_ms += tpot_ms
        self._timing_history = {"ttft": request_ttft, "tpot": request_tpot}
        timing_count = max(len(request_ttft), 1)

        metrics = self._pending_metrics
        cache_decisions = metrics.get("cache_hits", 0.0) + metrics.get("cache_misses", 0.0)
        cache_hit_rate = metrics.get("cache_hits", 0.0) / cache_decisions if cache_decisions else 0.0
        warm_requests = metrics.get("warm_requests", 0.0)
        warm_local_rate = (
            metrics.get("warm_requests_served_local", 0.0) / warm_requests
            if warm_requests
            else 0.0
        )
        max_transitions = float(max(1, self.cfg.requests_per_iteration * self.cfg.num_chunks))
        temporal_locality = 1.0 - min(metrics.get("worker_switches", 0.0) / max_transitions, 1.0)
        custom_metrics = self._custom_metrics
        custom_metrics.clear()
        custom_metrics.update(
            compute_inference_metrics(
                ttft_ms=ttft_total_ms / timing_count,
                tpot_ms=tpot_total_ms / timing_count,
                total_tokens=self.cfg.total_requests * self.cfg.decode_tokens,
                total_requests=self.cfg.total_requests,
                batch_size=self.cfg.batch_size,
                max_batch_size=self.cfg.batch_size * self.cfg.logical_decode_workers,
            )
        )
        custom_metrics["cache_aware.cache_hit_rate"] = cache_hit_rate
        custom_metrics["cache_aware.cache_miss_rate"] = 1.0 - cache_hit_rate
        custom_metrics["cache_aware.warm_request_local_rate"] = warm_local_rate
        custom_metrics["cache_aware.worker_switches_per_request"] = metrics.get(
            "worker_switches",
            0.0,
        ) / max(float(self.cfg.requests_per_iteration), 1.0)
        custom_metrics["cache_aware.temporal_locality_score"] = temporal_locality
        custom_metrics["cache_aware.shared_reloads"] = metrics.get("shared_reloads", 0.0)
        custom_metrics["cache_aware.peer_reloads"] = metrics.get("peer_reloads", 0.0)
        custom_metrics["cache_aware.kv_transfer_mb"] = (
            metrics.get("kv_transfer_bytes", 0.0) / 1e6
        )
        custom_metrics["cache_aware.logical_decode_workers"] = float(
            self.cfg.logical_decode_workers
        )
        custom_metrics["cache_aware.prefill_chunks_per_request"] = float(self.cfg.num_chunks)
        custom_metrics["cache_aware.warm_request_ratio"] = float(self.cfg.warm_request_ratio)
        self._set_verification_payload(
            inputs={"prompt": self.prompts[0]},
            output=self.output,
            batch_size=int(self.cfg.batch_size),
            parameter_count=int(self.parameter_count),
            precision_flags=PrecisionFlags(bf16=self.cfg.dtype == torch.bfloat16, tf32=torch.backends.cuda.matmul.allow_tf32),
            output_tolerance=(1e-2, 1e-2),
            signature_overrides={
                "world_size": 1,
                "pipeline_stages": 2,
                "pipeline_stage_boundaries": [(0, 0), (1, 1)],
                "per_rank_batch_size": self.cfg.batch_size,
                "collective_type": "logical_send_recv",
            },
        )

    def teardown(self) -> None:
        self.prefill_model = None
        self.decode_model = None
        self.prompts = None
        self.request_plans = []
        self.shared_prefix_store = {}
        self.shared_seed_store = {}
        self.output = None
        self._last_outputs = []
        self._output_stack = None
        self._outputs_ready = False
        self._empty_kv_template = None
        self._request_event_pool = []
        self._request_event_triplets = []
        self._request_event_count = 0
        self._pending_metrics = {}
        self._kv_buffers = {}
        self._reload_buffers = {}
        self._worker_caches = []
        self._worker_cache_count = 0
        self._owners = {}
        self._prompt_chunks = []
        self._prompt_chunk_count = 0
        self._request_plan_count = 0
        self._warm_request_count = 0
        self._request_event_groups = []
        self._request_event_group_count = 0
        self._last_output_count = 0
        self._decode_chunk_ranges = {}
        self._decode_chunk_range_count = 0
        self._decode_token_divisor = float(max(self.cfg.decode_tokens, 1))
        self._prefill_seed_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=4, warmup=5, measurement_timeout_seconds=900)

    def get_optimization_goal(self) -> str:
        """Single-GPU cache-aware disaggregation is judged here as a locality comparison surface."""
        return "comparison"

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return dict(self._custom_metrics) if self._custom_metrics else None

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "No output captured"
        if not self._timing_history["ttft"]:
            return "No TTFT samples recorded"
        if not self._timing_history["tpot"]:
            return "No TPOT samples recorded"
        return None


def build_config_from_args(args: argparse.Namespace) -> CacheAwareDisaggConfig:
    return CacheAwareDisaggConfig(
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        batch_size=int(args.batch_size),
        requests_per_iteration=int(args.requests_per_iteration),
        context_window=int(args.context_window),
        chunk_size=int(args.chunk_size),
        decode_tokens=int(args.decode_tokens),
        logical_decode_workers=int(args.logical_decode_workers),
        warm_request_ratio=float(args.warm_request_ratio),
        warm_prefix_ratio=float(args.warm_prefix_ratio),
    )


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-size", type=int, default=CacheAwareDisaggConfig.hidden_size)
    parser.add_argument("--num-layers", type=int, default=CacheAwareDisaggConfig.num_layers)
    parser.add_argument("--batch-size", type=int, default=CacheAwareDisaggConfig.batch_size)
    parser.add_argument(
        "--requests-per-iteration",
        type=int,
        default=CacheAwareDisaggConfig.requests_per_iteration,
    )
    parser.add_argument("--context-window", type=int, default=CacheAwareDisaggConfig.context_window)
    parser.add_argument("--chunk-size", type=int, default=CacheAwareDisaggConfig.chunk_size)
    parser.add_argument("--decode-tokens", type=int, default=CacheAwareDisaggConfig.decode_tokens)
    parser.add_argument(
        "--logical-decode-workers",
        type=int,
        default=CacheAwareDisaggConfig.logical_decode_workers,
    )
    parser.add_argument(
        "--warm-request-ratio",
        type=float,
        default=CacheAwareDisaggConfig.warm_request_ratio,
    )
    parser.add_argument(
        "--warm-prefix-ratio",
        type=float,
        default=CacheAwareDisaggConfig.warm_prefix_ratio,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the cache-aware disaggregated inference lab directly.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_shared_args(parser)
    return parser.parse_args()


def run_cli(*, optimized: bool) -> None:
    args = _parse_args()
    cfg = build_config_from_args(args)
    bench = CacheAwareDisaggBenchmark(
        optimized=optimized,
        label="optimized_cache_aware_disagg" if optimized else "baseline_cache_aware_disagg",
        cfg=cfg,
    )
    bench.setup()
    try:
        bench.benchmark_fn()
        bench.capture_verification_payload()
        payload = {
            "benchmark": bench.label,
            "optimized": optimized,
            "custom_metrics": bench.get_custom_metrics(),
            "validation_error": bench.validate_result(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        bench.teardown()
