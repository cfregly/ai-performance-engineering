"""Baseline harness for Chapter 17 dynamic routing."""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

import torch

from ch17.dynamic_routing import DisaggregatedRouter, Priority, Request, WorkerMetrics  # noqa: E402
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class _DynamicRoutingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Shared logic for baseline/optimized routing harnesses."""

    def __init__(self, *, batch_size: int, vectorized: bool):
        super().__init__()
        self.batch_size = batch_size
        self.vectorized = vectorized
        self.router = DisaggregatedRouter()
        self._latency_total_ms = 0.0
        self._latency_count = 0
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(batch_size),
            tokens_per_iteration=float(batch_size * 128),
        )
        self.output = None
        self._output_values: list[float] = [0.0, 0.0, 0.0]
        self._output_values_ready = False
        self._result_metrics: Dict[str, float] = {
            "requests": 0.0,
            "served": 0.0,
            "rejected": 0.0,
            "offloaded": 0.0,
        }
        self._verification_payload = None
        self._output_tensor: Optional[torch.Tensor] = None
        self._iteration = 0
        self._queue_length_table: Optional[torch.Tensor] = None
        self._queue_length_rows: Optional[list[list[int]]] = None
        self.register_workload_metadata(
            requests_per_iteration=float(batch_size),
            tokens_per_iteration=float(batch_size * 128),
        )
        # Pre-generated requests (created once in setup, reused in benchmark)
        self._cached_requests: List[Request] = []
        self._cached_prompt_lengths: List[int] = []
        self._request_count = 0
        self._request_count_float = 0.0
        # Pre-allocated tensors for vectorized path (reused each iteration)
        self._prompt_lengths: Optional[torch.Tensor] = None
        self._cached_lengths: Optional[torch.Tensor] = None
        self._queue_lengths: Optional[torch.Tensor] = None
        self._priorities: Optional[torch.Tensor] = None
        self._remaining_lengths: Optional[torch.Tensor] = None
        self._long_prefill: Optional[torch.Tensor] = None
        self._capacity_mask: Optional[torch.Tensor] = None
        self._offload_mask: Optional[torch.Tensor] = None
        self._admit_mask: Optional[torch.Tensor] = None
        self._served_offload_mask: Optional[torch.Tensor] = None
        self._count_values: Optional[torch.Tensor] = None
        self._cached_request_groups: List[tuple[int, Request]] = []

    def setup(self) -> None:
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        now = time.time()
        for idx in range(4):
            self.router.prefill_workers[f"prefill-{idx}"] = self._make_metrics(queue=idx, now=now)
            self.router.decode_workers[f"decode-{idx}"] = self._make_metrics(queue=idx // 2, now=now)
        
        # Pre-generate requests once (not part of benchmark timing)
        self._cached_requests = self._generate_requests()
        self._cached_prompt_lengths = [len(r.prompt_tokens) for r in self._cached_requests]
        self._request_count = len(self._cached_requests)
        self._request_count_float = float(self._request_count)
        self._cached_request_groups = list(enumerate(self._cached_requests))
        
        # Pre-allocate tensors for vectorized routing (avoids allocation in hot path)
        if self.vectorized:
            self._prompt_lengths = torch.tensor(
                self._cached_prompt_lengths, dtype=torch.int32
            )
            self._cached_lengths = torch.tensor(
                [r.prefix_cached_length for r in self._cached_requests], dtype=torch.int32
            )
            self._queue_lengths = torch.empty(self.batch_size, dtype=torch.int32)
            self._priorities = torch.tensor(
                [0 if r.priority is Priority.LOW else (2 if r.priority is Priority.HIGH else 1) 
                 for r in self._cached_requests], dtype=torch.int32
            )
            self._remaining_lengths = torch.empty_like(self._prompt_lengths)
            self._long_prefill = torch.empty_like(self._priorities, dtype=torch.bool)
            self._capacity_mask = torch.empty_like(self._long_prefill)
            self._offload_mask = torch.empty_like(self._long_prefill)
            self._admit_mask = torch.empty_like(self._long_prefill)
            self._served_offload_mask = torch.empty_like(self._long_prefill)
            self._count_values = torch.empty(2, dtype=torch.int64, device=self._priorities.device)
        cfg = self.get_config()
        num_iters = (cfg.warmup or 0) + (cfg.iterations or 0) + 5
        high = self.router.PREFILL_QUEUE_MAX + 6  # match baseline randint upper bound
        self._queue_length_table = torch.randint(
            0,
            high,
            (num_iters, self.batch_size),
            dtype=torch.int32,
        )
        self._queue_length_rows = self._queue_length_table.tolist()
        self._iteration = 0
        self._output_values[0] = 0.0
        self._output_values[1] = 0.0
        self._output_values[2] = 0.0
        self._output_tensor = torch.empty(len(self._output_values), dtype=torch.float32)
        self._output_values_ready = False

    def _make_metrics(self, queue: int, now: float):
        return WorkerMetrics(
            queue_length=queue,
            gpu_utilization=random.uniform(0.4, 0.8),
            memory_usage=random.uniform(30.0, 70.0),
            kv_cache_usage=random.uniform(10.0, 50.0),
            active_requests=random.randint(1, 4),
            last_updated=now,
        )

    def _generate_requests(self) -> List[Request]:
        reqs: List[Request] = []
        priority_choices = tuple(Priority)
        for idx in range(self.batch_size):
            prompt_len = random.randint(64, 2048)
            cached = random.randint(0, min(prompt_len // 2, 512))
            reqs.append(
                Request(
                    id=f"req-{idx}",
                    prompt_tokens=range(prompt_len),
                    priority=random.choice(priority_choices),
                    timestamp=time.time(),
                    prefix_cached_length=cached,
                    expected_output_length=random.randint(16, 128),
                )
            )
        return reqs

    def benchmark_fn(self) -> Dict[str, float]:
        # Use pre-generated requests (generation time excluded from benchmark)
        prompt_lengths = self._cached_prompt_lengths
        rejects = 0
        offloaded = 0
        count_values_ready = False
        start = self._record_start()
        queue_lengths: Optional[torch.Tensor] = None
        queue_lengths_host: Optional[list[int]] = None
        if self._queue_length_table is not None:
            table_idx = self._iteration % self._queue_length_table.shape[0]
            queue_lengths = self._queue_length_table[table_idx]
            if self._queue_length_rows is not None:
                queue_lengths_host = self._queue_length_rows[table_idx]
        self._iteration += 1

        if self.vectorized and self._prompt_lengths is not None:
            # Vectorized routing decisions using pre-allocated tensors
            if (
                queue_lengths is None
                or self._queue_lengths is None
                or self._cached_lengths is None
                or self._priorities is None
                or self._remaining_lengths is None
                or self._long_prefill is None
                or self._capacity_mask is None
                or self._offload_mask is None
                or self._admit_mask is None
                or self._served_offload_mask is None
                or self._count_values is None
            ):
                raise RuntimeError("Vectorized routing buffers not initialized")
            with torch.inference_mode():
                self._queue_lengths.copy_(queue_lengths)

                # Vectorized boolean operations reuse buffers to avoid hot-path allocation.
                torch.sub(self._prompt_lengths, self._cached_lengths, out=self._remaining_lengths)
                torch.gt(
                    self._remaining_lengths,
                    self.router.PREFILL_LENGTH_THRESHOLD,
                    out=self._long_prefill,
                )
                torch.lt(
                    self._queue_lengths,
                    self.router.PREFILL_QUEUE_MAX,
                    out=self._capacity_mask,
                )
                torch.logical_and(self._long_prefill, self._capacity_mask, out=self._offload_mask)

                est_ttft = (
                    self.router.get_current_prefill_queue_length() * self.router.avg_prefill_time_per_req
                    + self.router.get_current_decode_queue_length() * self.router.avg_decode_time_per_req
                )
                if est_ttft > self.router.TTFT_SLO_MAX:
                    torch.ne(self._priorities, 0, out=self._admit_mask)
                else:
                    self._admit_mask.fill_(True)

                torch.sum(self._admit_mask, dim=(), dtype=torch.int64, out=self._count_values[0])
                self._count_values[0].neg_().add_(self.batch_size)
                torch.logical_and(self._admit_mask, self._offload_mask, out=self._served_offload_mask)
                torch.sum(self._served_offload_mask, dim=(), dtype=torch.int64, out=self._count_values[1])
            count_values_ready = True
        else:
            # Python loop-based routing (sequential, one-at-a-time)
            if not prompt_lengths:
                raise RuntimeError("Cached prompt lengths not initialized")
            est_ttft = (
                self.router.get_current_prefill_queue_length()
                * self.router.avg_prefill_time_per_req
                + self.router.get_current_decode_queue_length()
                * self.router.avg_decode_time_per_req
            )
            reject_low_priority = est_ttft > self.router.TTFT_SLO_MAX
            for idx, req in self._cached_request_groups:
                if reject_low_priority and req.priority is Priority.LOW:
                    rejects += 1
                    continue
                if queue_lengths is None:
                    raise RuntimeError("Queue length inputs not initialized")
                if queue_lengths_host is None:
                    raise RuntimeError("Queue length host inputs not initialized")
                queue_depth = queue_lengths_host[idx % self.batch_size]
                prompt_length = prompt_lengths[idx]
                if self.router.should_offload_prefill(prompt_length, req.prefix_cached_length, queue_depth):
                    offloaded += 1

        elapsed_ms = self._record_stop(start)
        if count_values_ready:
            if self._count_values is None:
                raise RuntimeError("Vectorized count buffer not initialized")
            count_values = self._count_values
            rejects = int(count_values[0])
            offloaded = int(count_values[1])
        self._latency_total_ms += elapsed_ms
        self._latency_count += 1
        served = self._request_count - rejects

        self._output_values[0] = float(served)
        self._output_values[1] = float(rejects)
        self._output_values[2] = float(offloaded)
        self._output_values_ready = True
        if queue_lengths is None:
            raise RuntimeError("Queue length inputs not initialized")
        self._payload_input_snapshot = queue_lengths
        self._result_metrics["requests"] = self._request_count_float
        self._result_metrics["served"] = float(served)
        self._result_metrics["rejected"] = float(rejects)
        self._result_metrics["offloaded"] = float(offloaded)
        return self._result_metrics

    def capture_verification_payload(self) -> None:
        input_snapshot = self._payload_input_snapshot
        if not self._output_values_ready:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._output_tensor is None:
            raise RuntimeError("setup() must initialize verification tensors")
        for idx, value in enumerate(self._output_values):
            self._output_tensor[idx] = value
        self.output = self._output_tensor
        self._set_verification_payload(
            inputs={"queue_lengths": input_snapshot},
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.1, 10.0),
        )

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=8, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if self._latency_count == 0:
            return None
        return {
            "routing.latency_ms": float(self._latency_total_ms / self._latency_count),
        }

    def validate_result(self) -> Optional[str]:
        return None

    def teardown(self) -> None:
        self._cached_requests = []
        self._cached_prompt_lengths = []
        self._request_count = 0
        self._request_count_float = 0.0
        self._cached_request_groups = []
        self._prompt_lengths = None
        self._cached_lengths = None
        self._queue_lengths = None
        self._priorities = None
        self._remaining_lengths = None
        self._long_prefill = None
        self._capacity_mask = None
        self._offload_mask = None
        self._admit_mask = None
        self._served_offload_mask = None
        self._count_values = None
        self._output_tensor = None
        self._queue_length_table = None
        self._queue_length_rows = None
        self._latency_total_ms = 0.0
        self._latency_count = 0
        self._output_values[0] = 0.0
        self._output_values[1] = 0.0
        self._output_values[2] = 0.0
        self._output_values_ready = False
        self._result_metrics["requests"] = 0.0
        self._result_metrics["served"] = 0.0
        self._result_metrics["rejected"] = 0.0
        self._result_metrics["offloaded"] = 0.0
        super().teardown()


class BaselineDynamicRoutingBenchmark(_DynamicRoutingBenchmark):
    """Python loop-based routing decisions."""
    def __init__(self) -> None:
        # Match optimized batch size for fair comparison
        super().__init__(batch_size=1024, vectorized=False)


def get_benchmark():
    return BaselineDynamicRoutingBenchmark()
