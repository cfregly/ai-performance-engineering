"""baseline_kv_cache_nvlink_pool.py

Baseline KV-cache strategy: keep everything in local HBM and evict to host when full.
No NVLink pooling or peer placement is used.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin


class BaselineKVCacheLocalOnlyBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Local-only KV cache with host spill."""

    allowed_benchmark_fn_antipatterns = ("host_transfer",)

    def __init__(self):
        super().__init__()
        self.output = None
        self.model: Optional[nn.MultiheadAttention] = None
        self.hidden = 512
        self.heads = 8
        self.batch = 4
        self.seq_len = 256
        self.local_cache_limit = 32  # tokens before spill
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._verify_q: Optional[torch.Tensor] = None
        self._query_steps: Optional[torch.Tensor] = None
        self._key_steps: Optional[torch.Tensor] = None
        self._value_steps: Optional[torch.Tensor] = None
        self._decode_step_inputs: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._k_gather_buffer: Optional[torch.Tensor] = None
        self._v_gather_buffer: Optional[torch.Tensor] = None
        self._k_gather_step_views: list[torch.Tensor] = []
        self._v_gather_step_views: list[torch.Tensor] = []
        self._k_gather_prefix_views: list[torch.Tensor] = []
        self._v_gather_prefix_views: list[torch.Tensor] = []
        self._local_key_slots: list[torch.Tensor] = []
        self._local_value_slots: list[torch.Tensor] = []
        self._host_key_slots: list[torch.Tensor] = []
        self._host_value_slots: list[torch.Tensor] = []
        self._slot_counts: tuple[int, ...] = ()
        self._expected_slot_counts: tuple[int, ...] = ()
        self._host_gather_ranges: list[range] = []
        self._local_gather_ranges: list[range] = []
        self._payload_parameter_count = 0

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: requires CUDA")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
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
        self._local_key_slots = [
            torch.empty(0, device=self.device)
            for _ in range(self.local_cache_limit)
        ]
        self._local_value_slots = [
            torch.empty(0, device=self.device)
            for _ in range(self.local_cache_limit)
        ]
        host_capacity = max(self.seq_len - self.local_cache_limit, 0)
        self._host_key_slots = [
            torch.empty(self.batch, 1, self.hidden, dtype=self._key_steps.dtype)
            for _ in range(host_capacity)
        ]
        self._host_value_slots = [
            torch.empty(self.batch, 1, self.hidden, dtype=self._value_steps.dtype)
            for _ in range(host_capacity)
        ]
        self._host_gather_ranges = [range(count) for count in range(host_capacity + 1)]
        self._local_gather_ranges = [
            range(count) for count in range(self.local_cache_limit + 1)
        ]
        self._slot_counts = (
            len(self._local_key_slots),
            len(self._local_value_slots),
            len(self._host_key_slots),
            len(self._host_value_slots),
            len(self._decode_step_inputs),
            len(self._k_gather_step_views),
            len(self._v_gather_step_views),
            len(self._k_gather_prefix_views),
            len(self._v_gather_prefix_views),
            len(self._host_gather_ranges),
            len(self._local_gather_ranges),
        )
        self._expected_slot_counts = (
            self.local_cache_limit,
            self.local_cache_limit,
            host_capacity,
            host_capacity,
            self.seq_len,
            self.seq_len,
            self.seq_len,
            self.seq_len,
            self.seq_len,
            host_capacity + 1,
            self.local_cache_limit + 1,
        )
        self._verify_q = self._query_steps[0, :1].detach().clone()
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.model is not None
        assert self._query_steps is not None and self._key_steps is not None and self._value_steps is not None
        assert self._k_gather_buffer is not None and self._v_gather_buffer is not None
        with torch.inference_mode(), self._nvtx_range("baseline_kv_cache_local_only"):
            if self._slot_counts != self._expected_slot_counts:
                raise RuntimeError("KV cache slots not initialized")
            local_keys = self._local_key_slots
            local_values = self._local_value_slots
            host_keys = self._host_key_slots
            host_values = self._host_value_slots
            k_gather_steps = self._k_gather_step_views
            v_gather_steps = self._v_gather_step_views
            k_gather_prefixes = self._k_gather_prefix_views
            v_gather_prefixes = self._v_gather_prefix_views
            host_gather_ranges = self._host_gather_ranges
            local_gather_ranges = self._local_gather_ranges
            local_start = 0
            local_count = 0
            host_count = 0
            for _step, q, k, v in self._decode_step_inputs:
                if local_count < self.local_cache_limit:
                    local_slot = (local_start + local_count) % self.local_cache_limit
                    local_count += 1
                else:
                    # Spill oldest to host (slow, pageable)
                    local_slot = local_start
                    host_keys[host_count].copy_(local_keys[local_slot])
                    host_values[host_count].copy_(local_values[local_slot])
                    host_count += 1
                    local_start = (local_start + 1) % self.local_cache_limit
                local_keys[local_slot] = k
                local_values[local_slot] = v

                gather_idx = 0
                for host_idx in host_gather_ranges[host_count]:
                    hk = host_keys[host_idx]
                    hv = host_values[host_idx]
                    k_gather_steps[gather_idx].copy_(hk)
                    v_gather_steps[gather_idx].copy_(hv)
                    gather_idx += 1
                for local_offset in local_gather_ranges[local_count]:
                    slot_idx = (local_start + local_offset) % self.local_cache_limit
                    lk = local_keys[slot_idx]
                    lv = local_values[slot_idx]
                    k_gather_steps[gather_idx].copy_(lk)
                    v_gather_steps[gather_idx].copy_(lv)
                    gather_idx += 1

                k_all = k_gather_prefixes[gather_idx - 1]
                v_all = v_gather_prefixes[gather_idx - 1]
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
        self._local_key_slots = []
        self._local_value_slots = []
        self._host_key_slots = []
        self._host_value_slots = []
        self._slot_counts = ()
        self._expected_slot_counts = ()
        self._host_gather_ranges = []
        self._local_gather_ranges = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=None,
            tpot_ms=None,
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineKVCacheLocalOnlyBenchmark()
