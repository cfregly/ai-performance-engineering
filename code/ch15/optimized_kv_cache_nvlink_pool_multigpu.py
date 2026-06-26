"""optimized_kv_cache_nvlink_pool_multigpu.py

Tiered KV cache: local HBM, peer HBM over NVLink, then host.
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from core.benchmark.gpu_requirements import skip_if_insufficient_gpus, require_peer_access
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin


class OptimizedKVCacheNvlinkPoolBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Tiered KV cache with NVLink peer pooling."""

    multi_gpu_required = True
    allowed_benchmark_fn_antipatterns = ("host_transfer",)

    def __init__(self):
        super().__init__()
        self.output = None
        self.model: Optional[nn.MultiheadAttention] = None
        self.local_cache_limit = 16
        self.peer_cache_limit = 256
        self.hidden = 1024
        self.heads = 16
        self.batch = 8
        self.seq_len = 512
        self.device_ids: List[int] = []
        self.peer_devices: List[torch.device] = []
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._verify_q: Optional[torch.Tensor] = None
        self._query_steps: Optional[torch.Tensor] = None
        self._key_steps: Optional[torch.Tensor] = None
        self._value_steps: Optional[torch.Tensor] = None
        self._k_gather_buffer: Optional[torch.Tensor] = None
        self._v_gather_buffer: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        skip_if_insufficient_gpus(2)

        self.device_ids = list(range(torch.cuda.device_count()))
        self.peer_devices = [torch.device(f"cuda:{idx}") for idx in self.device_ids[1:]]
        for idx in self.device_ids[1:]:
            require_peer_access(0, idx)

        self.model = nn.MultiheadAttention(self.hidden, self.heads, batch_first=True).to(self.device).eval()
        self._query_steps = torch.randn(self.seq_len, self.batch, 1, self.hidden, device=self.device)
        self._key_steps = torch.randn(self.seq_len, self.batch, 1, self.hidden, device=self.device)
        self._value_steps = torch.randn(self.seq_len, self.batch, 1, self.hidden, device=self.device)
        self._k_gather_buffer = torch.empty(self.batch, self.seq_len, self.hidden, device=self.device)
        self._v_gather_buffer = torch.empty_like(self._k_gather_buffer)
        self._verify_q = self._query_steps[0, :1].detach().clone()
        self._synchronize()

    def _place_kv(self, k: torch.Tensor, v: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor, str, Optional[torch.device]]:
        if step < self.local_cache_limit:
            return k, v, "local", None
        if self.peer_devices and step < self.local_cache_limit + self.peer_cache_limit:
            peer = self.peer_devices[(step - self.local_cache_limit) % len(self.peer_devices)]
            return k.to(peer, non_blocking=True), v.to(peer, non_blocking=True), "peer", peer
        return k.cpu(), v.cpu(), "host", None

    def _gather_kv_into_buffers(
        self,
        cache_k: list[torch.Tensor],
        cache_v: list[torch.Tensor],
        tiers: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._k_gather_buffer is None or self._v_gather_buffer is None:
            raise RuntimeError("KV gather buffers not initialized")
        for idx, (tk, tv, tier) in enumerate(zip(cache_k, cache_v, tiers)):
            non_blocking = tier != "local"
            self._k_gather_buffer[:, idx : idx + 1, :].copy_(tk, non_blocking=non_blocking)
            self._v_gather_buffer[:, idx : idx + 1, :].copy_(tv, non_blocking=non_blocking)
        gathered_len = len(cache_k)
        return (
            self._k_gather_buffer[:, :gathered_len, :],
            self._v_gather_buffer[:, :gathered_len, :],
        )

    def benchmark_fn(self) -> None:
        assert self.model is not None
        assert self._query_steps is not None and self._key_steps is not None and self._value_steps is not None
        assert self._k_gather_buffer is not None and self._v_gather_buffer is not None
        with self._nvtx_range("optimized_kv_cache_nvlink_pool_multigpu"):
            cache_k: list[torch.Tensor] = []
            cache_v: list[torch.Tensor] = []
            tiers: list[str] = []
            for step in range(self.seq_len):
                q = self._query_steps[step]
                k = self._key_steps[step]
                v = self._value_steps[step]
                placed_k, placed_v, tier, _peer = self._place_kv(k, v, step)
                cache_k.append(placed_k)
                cache_v.append(placed_v)
                tiers.append(tier)

                k_all, v_all = self._gather_kv_into_buffers(cache_k, cache_v, tiers)
                out, _ = self.model(q, k_all, v_all)
                self.output = out

    def capture_verification_payload(self) -> None:
        if self.model is None or self.output is None or self._verify_q is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._synchronize()
        self._set_verification_payload(
            inputs={"q": self._verify_q},
            output=self.output,
            batch_size=int(self.batch),
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(5e-1, 5e-1),
        )

    def teardown(self) -> None:
        self.model = None
        self._query_steps = None
        self._key_steps = None
        self._value_steps = None
        self._k_gather_buffer = None
        self._v_gather_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=3, warmup=5, multi_gpu_required=True)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=None,
            tpot_ms=None,
            total_tokens=getattr(self, "total_tokens", 256),
            total_requests=getattr(self, "total_requests", 1),
            batch_size=getattr(self, "batch_size", 1),
            max_batch_size=getattr(self, "max_batch_size", 32),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedKVCacheNvlinkPoolBenchmark()

