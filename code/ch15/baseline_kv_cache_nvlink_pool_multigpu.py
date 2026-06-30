"""baseline_kv_cache_nvlink_pool_multigpu.py

Baseline KV-cache strategy: local HBM, then host-staged remote pool, then host spill.
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from core.benchmark.gpu_requirements import skip_if_insufficient_gpus
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin


class BaselineKVCacheLocalOnlyBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Local-first KV cache with host-staged remote pool (no peer copies)."""

    multi_gpu_required = True
    allowed_benchmark_fn_antipatterns = ("host_transfer",)

    def __init__(self):
        super().__init__()
        self.output = None
        self.model: Optional[nn.MultiheadAttention] = None
        self.hidden = 1024
        self.heads = 16
        self.batch = 8
        self.seq_len = 512
        self.local_cache_limit = 16
        self.peer_cache_limit = 256
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
        self._peer_host_k_stage: Optional[torch.Tensor] = None
        self._peer_host_v_stage: Optional[torch.Tensor] = None
        self._cache_key_slots: List[torch.Tensor] = []
        self._cache_value_slots: List[torch.Tensor] = []
        self._tier_slots: List[str] = []
        self._peer_target_slots: List[Optional[torch.device]] = []
        self._slot_counts: Tuple[int, ...] = ()
        self._expected_slot_counts: Tuple[int, ...] = ()
        self._cache_gather_ranges: List[range] = []
        self._payload_parameter_count = 0

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        skip_if_insufficient_gpus(2)

        self.device_ids = list(range(torch.cuda.device_count()))
        self.peer_devices = [torch.device(f"cuda:{idx}") for idx in self.device_ids[1:]]
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
        self._peer_host_k_stage = torch.empty(
            self.batch,
            1,
            self.hidden,
            dtype=self._key_steps.dtype,
            pin_memory=True,
        )
        self._peer_host_v_stage = torch.empty(
            self.batch,
            1,
            self.hidden,
            dtype=self._value_steps.dtype,
            pin_memory=True,
        )
        self._cache_key_slots = [
            torch.empty(0, device=self.device)
            for _ in range(self.seq_len)
        ]
        self._cache_value_slots = [
            torch.empty(0, device=self.device)
            for _ in range(self.seq_len)
        ]
        self._tier_slots = [""] * self.seq_len
        self._peer_target_slots = [None] * self.seq_len
        self._cache_gather_ranges = [range(step + 1) for step in range(self.seq_len)]
        self._slot_counts = (
            len(self._cache_key_slots),
            len(self._cache_value_slots),
            len(self._tier_slots),
            len(self._peer_target_slots),
            len(self._decode_step_inputs),
            len(self._k_gather_step_views),
            len(self._v_gather_step_views),
            len(self._k_gather_prefix_views),
            len(self._v_gather_prefix_views),
            len(self._cache_gather_ranges),
        )
        self._expected_slot_counts = (
            self.seq_len,
            self.seq_len,
            self.seq_len,
            self.seq_len,
            self.seq_len,
            self.seq_len,
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
            return k.cpu().to(peer, non_blocking=False), v.cpu().to(peer, non_blocking=False), "peer", peer
        return k.cpu(), v.cpu(), "host", None

    def benchmark_fn(self) -> None:
        assert self.model is not None
        assert self._query_steps is not None and self._key_steps is not None and self._value_steps is not None
        assert self._k_gather_buffer is not None and self._v_gather_buffer is not None
        with torch.inference_mode(), self._nvtx_range("baseline_kv_cache_local_only"):
            if self._slot_counts != self._expected_slot_counts:
                raise RuntimeError("KV cache slots not initialized")
            cache_k = self._cache_key_slots
            cache_v = self._cache_value_slots
            tiers = self._tier_slots
            peer_targets = self._peer_target_slots
            k_gather_steps = self._k_gather_step_views
            v_gather_steps = self._v_gather_step_views
            k_gather_prefixes = self._k_gather_prefix_views
            v_gather_prefixes = self._v_gather_prefix_views
            cache_gather_ranges = self._cache_gather_ranges
            for step, q, k, v in self._decode_step_inputs:
                placed_k, placed_v, tier, peer = self._place_kv(k, v, step)
                cache_k[step] = placed_k
                cache_v[step] = placed_v
                tiers[step] = tier
                peer_targets[step] = peer

                gather_idx = 0
                for cache_idx in cache_gather_ranges[step]:
                    tk = cache_k[cache_idx]
                    tv = cache_v[cache_idx]
                    t = tiers[cache_idx]
                    peer_dev = peer_targets[cache_idx]
                    if t == "local":
                        k_gather_steps[gather_idx].copy_(tk)
                        v_gather_steps[gather_idx].copy_(tv)
                    elif t == "peer" and peer_dev is not None:
                        if self._peer_host_k_stage is None or self._peer_host_v_stage is None:
                            raise RuntimeError("Peer host staging buffers not initialized")
                        self._peer_host_k_stage.copy_(tk, non_blocking=False)
                        self._peer_host_v_stage.copy_(tv, non_blocking=False)
                        k_gather_steps[gather_idx].copy_(self._peer_host_k_stage)
                        v_gather_steps[gather_idx].copy_(self._peer_host_v_stage)
                    else:
                        k_gather_steps[gather_idx].copy_(tk)
                        v_gather_steps[gather_idx].copy_(tv)
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
        self._peer_host_k_stage = None
        self._peer_host_v_stage = None
        self._cache_key_slots = []
        self._cache_value_slots = []
        self._tier_slots = []
        self._peer_target_slots = []
        self._slot_counts = ()
        self._expected_slot_counts = ()
        self._cache_gather_ranges = []
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
    return BaselineKVCacheLocalOnlyBenchmark()
