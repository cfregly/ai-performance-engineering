"""labs.moe_cuda/baseline_decode_attention.py - Naive decode attention."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch

from core.benchmark.cuda_event_timing import elapsed_ms
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range


class BaselineDecodeAttentionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Naive decode attention with explicit matmul + softmax."""

    def __init__(self) -> None:
        super().__init__()
        # Realistic decode workload where BF16 optimization shows benefit
        self.batch = 32
        self.num_heads = 12
        self.kv_seq = 512
        self.head_dim = 64
        self.q: Optional[torch.Tensor] = None  # [B, H, 1, D]
        self.k: Optional[torch.Tensor] = None  # [B, H, S, D]
        self.v: Optional[torch.Tensor] = None  # [B, H, S, D]
        self._k_t: Optional[torch.Tensor] = None
        self._scale = 0.0
        tokens = self.batch * self.kv_seq
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._history: Dict[str, List[float]] = {"latency_ms": []}
        self.output: Optional[torch.Tensor] = None
        self._payload_meta: Optional[torch.Tensor] = None
        self._timing_pair: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._pending_timing_pair: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._enable_nvtx = False

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("labs.moe_cuda decode attention requires CUDA")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.q = torch.randn(self.batch, self.num_heads, 1, self.head_dim, device=self.device, dtype=torch.float32)
        self.k = torch.randn(
            self.batch,
            self.num_heads,
            self.kv_seq,
            self.head_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.v = torch.randn_like(self.k)
        self._k_t = self.k.transpose(-2, -1)
        self._scale = 1.0 / math.sqrt(self.head_dim)
        torch.cuda.synchronize(self.device)
        self.output = None
        self._payload_meta = torch.tensor(
            [self.batch, self.kv_seq, self.num_heads, self.head_dim],
            dtype=torch.int64,
            device="cpu",
        )
        self._timing_pair = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    def _get_timing_pair(self) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        if self._timing_pair is None:
            self._timing_pair = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        return self._timing_pair

    def benchmark_fn(self) -> Dict[str, List[float]]:
        if any(t is None for t in (self.q, self.k, self.v, self._k_t)):
            raise RuntimeError("Decode tensors missing")

        with nvtx_range("moe_cuda_decode_naive", enable=self._enable_nvtx):
            with torch.inference_mode():
                timing_pair = self._get_timing_pair()
                start_event, end_event = timing_pair
                current_stream = torch.cuda.current_stream(self.device)
                start_event.record(current_stream)
                q = self.q
                v = self.v
                scores = torch.matmul(q, self._k_t)
                scores.mul_(self._scale)
                probs = torch.softmax(scores, dim=-1)
                attn = torch.matmul(probs, v)
                attn_out = attn.transpose(1, 2).reshape(self.batch, 1, self.num_heads * self.head_dim)
                end_event.record(current_stream)
                self._pending_timing_pair = timing_pair
                self.output = attn_out.detach()
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")
        return None

    def finalize_iteration_metrics(self) -> Optional[Dict[str, List[float]]]:
        if self._pending_timing_pair is None:
            return None
        latency_ms = elapsed_ms(self._pending_timing_pair)
        self._pending_timing_pair = None
        self._history["latency_ms"].append(latency_ms)
        return {"decode_ms": [latency_ms]}

    def capture_verification_payload(self) -> None:
        self.finalize_iteration_metrics()
        meta = self._payload_meta
        self._set_verification_payload(
            inputs={"meta": meta, "q": self.q, "k": self.k, "v": self.v},
            output=self.output.float().clone(),
            batch_size=self.batch,
            parameter_count=0,
            precision_flags={"tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.q = None
        self.k = None
        self.v = None
        self._k_t = None
        self.output = None
        self._payload_meta = None
        self._timing_pair = None
        self._pending_timing_pair = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=8, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        self.finalize_iteration_metrics()
        if not self._history["latency_ms"]:
            return None
        return {
            "decode.mean_ms": float(sum(self._history["latency_ms"]) / len(self._history["latency_ms"]))
        }

    def validate_result(self) -> Optional[str]:
        if any(t is None for t in (self.q, self.k, self.v)):
            return "Decode tensors missing"
        return None

def get_benchmark() -> BaseBenchmark:
    return BaselineDecodeAttentionBenchmark()
