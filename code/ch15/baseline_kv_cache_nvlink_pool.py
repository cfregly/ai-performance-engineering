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
        self._k_gather_buffer: Optional[torch.Tensor] = None
        self._v_gather_buffer: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: requires CUDA")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model = nn.MultiheadAttention(self.hidden, self.heads, batch_first=True).to(self.device).eval()
        self._query_steps = torch.randn(self.seq_len, self.batch, 1, self.hidden, device=self.device)
        self._key_steps = torch.randn(self.seq_len, self.batch, 1, self.hidden, device=self.device)
        self._value_steps = torch.randn(self.seq_len, self.batch, 1, self.hidden, device=self.device)
        self._k_gather_buffer = torch.empty(self.batch, self.seq_len, self.hidden, device=self.device)
        self._v_gather_buffer = torch.empty_like(self._k_gather_buffer)
        self._verify_q = self._query_steps[0, :1].detach().clone()
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.model is not None
        assert self._query_steps is not None and self._key_steps is not None and self._value_steps is not None
        assert self._k_gather_buffer is not None and self._v_gather_buffer is not None
        with self._nvtx_range("baseline_kv_cache_local_only"):
            local_keys: list[torch.Tensor] = []
            local_values: list[torch.Tensor] = []
            host_keys: list[torch.Tensor] = []
            host_values: list[torch.Tensor] = []
            for step in range(self.seq_len):
                q = self._query_steps[step]
                k = self._key_steps[step]
                v = self._value_steps[step]
                local_keys.append(k)
                local_values.append(v)

                if len(local_keys) > self.local_cache_limit:
                    # Spill oldest to host (slow, pageable)
                    host_keys.append(local_keys.pop(0).cpu())
                    host_values.append(local_values.pop(0).cpu())

                gather_idx = 0
                for hk, hv in zip(host_keys, host_values):
                    self._k_gather_buffer[:, gather_idx : gather_idx + 1, :].copy_(hk)
                    self._v_gather_buffer[:, gather_idx : gather_idx + 1, :].copy_(hv)
                    gather_idx += 1
                for lk, lv in zip(local_keys, local_values):
                    self._k_gather_buffer[:, gather_idx : gather_idx + 1, :].copy_(lk)
                    self._v_gather_buffer[:, gather_idx : gather_idx + 1, :].copy_(lv)
                    gather_idx += 1

                k_all = self._k_gather_buffer[:, :gather_idx, :]
                v_all = self._v_gather_buffer[:, :gather_idx, :]
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

