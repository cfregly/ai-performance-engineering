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
        self._decode_step_inputs: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._k_gather_buffer: Optional[torch.Tensor] = None
        self._v_gather_buffer: Optional[torch.Tensor] = None
        self._k_gather_step_views: List[torch.Tensor] = []
        self._v_gather_step_views: List[torch.Tensor] = []
        self._k_gather_prefix_views: List[torch.Tensor] = []
        self._v_gather_prefix_views: List[torch.Tensor] = []
        self._cache_key_slots: List[torch.Tensor] = []
        self._cache_value_slots: List[torch.Tensor] = []
        self._host_key_slots: List[torch.Tensor] = []
        self._host_value_slots: List[torch.Tensor] = []
        self._tier_slots: List[str] = []
        self._cache_slot_count = 0
        self._host_slot_count = 0
        self._gather_view_count = 0
        self._slot_counts: Tuple[int, ...] = ()
        self._expected_slot_counts: Tuple[int, ...] = ()
        self._gather_view_counts: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._expected_gather_view_counts: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._payload_parameter_count = 0

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        skip_if_insufficient_gpus(2)

        self.device_ids = list(range(torch.cuda.device_count()))
        self.peer_devices = [torch.device(f"cuda:{idx}") for idx in self.device_ids[1:]]
        for idx in self.device_ids[1:]:
            require_peer_access(0, idx)

        self.model = nn.MultiheadAttention(self.hidden, self.heads, batch_first=True).to(self.device).eval()
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        self._query_steps = torch.randn(self.seq_len, self.batch, 1, self.hidden, device=self.device)
        self._key_steps = torch.randn(self.seq_len, self.batch, 1, self.hidden, device=self.device)
        self._value_steps = torch.randn(self.seq_len, self.batch, 1, self.hidden, device=self.device)
        self._decode_step_inputs = list(
            zip(range(self.seq_len), self._query_steps, self._key_steps, self._value_steps, strict=True)
        )
        self._k_gather_buffer = torch.empty(self.batch, self.seq_len, self.hidden, device=self.device)
        self._v_gather_buffer = torch.empty_like(self._k_gather_buffer)
        self._k_gather_step_views = [
            self._k_gather_buffer[:, idx : idx + 1, :] for idx in range(self.seq_len)
        ]
        self._v_gather_step_views = [
            self._v_gather_buffer[:, idx : idx + 1, :] for idx in range(self.seq_len)
        ]
        self._k_gather_prefix_views = [
            self._k_gather_buffer[:, : idx + 1, :] for idx in range(self.seq_len)
        ]
        self._v_gather_prefix_views = [
            self._v_gather_buffer[:, : idx + 1, :] for idx in range(self.seq_len)
        ]
        self._cache_key_slots = [
            torch.empty(0, device=self.device)
            for _ in range(self.seq_len)
        ]
        self._cache_value_slots = [
            torch.empty(0, device=self.device)
            for _ in range(self.seq_len)
        ]
        host_capacity = max(self.seq_len - self.local_cache_limit - self.peer_cache_limit, 0)
        self._host_key_slots = [
            torch.empty(
                self.batch,
                1,
                self.hidden,
                dtype=self._key_steps.dtype,
                pin_memory=True,
            )
            for _ in range(host_capacity)
        ]
        self._host_value_slots = [
            torch.empty(
                self.batch,
                1,
                self.hidden,
                dtype=self._value_steps.dtype,
                pin_memory=True,
            )
            for _ in range(host_capacity)
        ]
        self._tier_slots = [""] * self.seq_len
        self._cache_slot_count = self.seq_len
        self._host_slot_count = host_capacity
        self._gather_view_count = self.seq_len
        self._slot_counts = (
            len(self._cache_key_slots),
            len(self._cache_value_slots),
            len(self._tier_slots),
            len(self._decode_step_inputs),
        )
        self._expected_slot_counts = (
            self.seq_len,
            self.seq_len,
            self.seq_len,
            self.seq_len,
        )
        self._gather_view_counts = (
            len(self._k_gather_step_views),
            len(self._v_gather_step_views),
            len(self._k_gather_prefix_views),
            len(self._v_gather_prefix_views),
        )
        self._expected_gather_view_counts = (
            self.seq_len,
            self.seq_len,
            self.seq_len,
            self.seq_len,
        )
        self._verify_q = self._query_steps[0, :1].detach().clone()
        self._synchronize()

    def _place_kv(self, k: torch.Tensor, v: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor, str, Optional[torch.device]]:
        if step < self.local_cache_limit:
            return k, v, "local", None
        if self.peer_devices and step < self.local_cache_limit + self.peer_cache_limit:
            peer = self.peer_devices[(step - self.local_cache_limit) % len(self.peer_devices)]
            return k.to(peer, non_blocking=True), v.to(peer, non_blocking=True), "peer", peer
        host_idx = step - self.local_cache_limit - self.peer_cache_limit
        if host_idx >= self._host_slot_count:
            raise RuntimeError("Host KV cache slots not initialized")
        host_k = self._host_key_slots[host_idx]
        host_v = self._host_value_slots[host_idx]
        host_k.copy_(k, non_blocking=True)
        host_v.copy_(v, non_blocking=True)
        return host_k, host_v, "host", None

    def _gather_kv_into_buffers(
        self,
        cache_k: list[torch.Tensor],
        cache_v: list[torch.Tensor],
        tiers: list[str],
        cache_len: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._k_gather_buffer is None or self._v_gather_buffer is None:
            raise RuntimeError("KV gather buffers not initialized")
        gathered_len = self._cache_slot_count if cache_len is None else cache_len
        if (
            gathered_len > self._gather_view_count
            or self._gather_view_counts != self._expected_gather_view_counts
        ):
            raise RuntimeError("KV gather views not initialized")
        for idx in range(gathered_len):
            tk = cache_k[idx]
            tv = cache_v[idx]
            tier = tiers[idx]
            non_blocking = tier != "local"
            self._k_gather_step_views[idx].copy_(tk, non_blocking=non_blocking)
            self._v_gather_step_views[idx].copy_(tv, non_blocking=non_blocking)
        if gathered_len <= 0:
            return self._k_gather_buffer[:, :0, :], self._v_gather_buffer[:, :0, :]
        return (
            self._k_gather_prefix_views[gathered_len - 1],
            self._v_gather_prefix_views[gathered_len - 1],
        )

    def _append_kv_into_buffers(
        self,
        cache_k: list[torch.Tensor],
        cache_v: list[torch.Tensor],
        tiers: list[str],
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._k_gather_buffer is None or self._v_gather_buffer is None:
            raise RuntimeError("KV gather buffers not initialized")
        if (
            step >= self._gather_view_count
            or self._gather_view_counts != self._expected_gather_view_counts
        ):
            raise RuntimeError("KV gather views not initialized")
        tier = tiers[step]
        non_blocking = tier != "local"
        self._k_gather_step_views[step].copy_(cache_k[step], non_blocking=non_blocking)
        self._v_gather_step_views[step].copy_(cache_v[step], non_blocking=non_blocking)
        return (
            self._k_gather_prefix_views[step],
            self._v_gather_prefix_views[step],
        )

    def benchmark_fn(self) -> None:
        assert self.model is not None
        assert self._query_steps is not None and self._key_steps is not None and self._value_steps is not None
        assert self._k_gather_buffer is not None and self._v_gather_buffer is not None
        with torch.inference_mode(), self._nvtx_range("optimized_kv_cache_nvlink_pool_multigpu"):
            if self._slot_counts != self._expected_slot_counts:
                raise RuntimeError("KV cache slots not initialized")
            cache_k = self._cache_key_slots
            cache_v = self._cache_value_slots
            tiers = self._tier_slots
            for step, q, k, v in self._decode_step_inputs:
                placed_k, placed_v, tier, _peer = self._place_kv(k, v, step)
                cache_k[step] = placed_k
                cache_v[step] = placed_v
                tiers[step] = tier

                k_all, v_all = self._append_kv_into_buffers(cache_k, cache_v, tiers, step)
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
            parameter_count=self._payload_parameter_count,
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
        self._decode_step_inputs = []
        self._k_gather_buffer = None
        self._v_gather_buffer = None
        self._k_gather_step_views = []
        self._v_gather_step_views = []
        self._k_gather_prefix_views = []
        self._v_gather_prefix_views = []
        self._cache_key_slots = []
        self._cache_value_slots = []
        self._host_key_slots = []
        self._host_value_slots = []
        self._tier_slots = []
        self._cache_slot_count = 0
        self._host_slot_count = 0
        self._gather_view_count = 0
        self._slot_counts = ()
        self._expected_slot_counts = ()
        self._gather_view_counts = (0, 0, 0, 0)
        self._expected_gather_view_counts = (0, 0, 0, 0)
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
