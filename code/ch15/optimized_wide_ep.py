#!/usr/bin/env python3
"""optimized_wide_ep.py - Wide expert-parallel all-to-all (fused pack/unpack) (Ch15).

Pairs with: baseline_wide_ep.py

Semantic contract:
- Both variants apply the same shared expert to the same token activations.
- Routing/placement changes do NOT change the final output tensor because the
  expert weights are shared across expert ids.

Optimization behavior:
- Precomputes a single GPU permutation to pack tokens by destination rank.
- Uses single-shot gather/scatter ops instead of a Python loop over ranks.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.optimization.moe_inference import ExpertMLP


def _pseudo_uniform_expert_ids(token_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    if token_ids.dtype != torch.int64:
        token_ids = token_ids.to(torch.int64)
    return ((token_ids * 1103515245 + 12345) % int(num_experts)).to(torch.int64)


class OptimizedWideEPBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: fused pack/unpack for wide expert-parallel all-to-all."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 1024
        self.ffn_size = 4096
        self.world_size = 64
        self.experts_per_rank = 1
        self.num_experts = self.world_size * self.experts_per_rank
        self.batch = 128
        self.seq = 32
        self.dtype = torch.bfloat16

        tokens = self.batch * self.seq
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

        self.expert: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.expert_ids: Optional[torch.Tensor] = None
        self._dest_ranks: Optional[torch.Tensor] = None
        self._perm: Optional[torch.Tensor] = None
        self._recv_buf: Optional[torch.Tensor] = None
        self._out_flat: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_probe: Optional[torch.Tensor] = None
        self._verify_meta: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for wide-EP benchmark")

        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.experts_per_rank <= 0:
            raise ValueError("experts_per_rank must be positive")
        if self.num_experts % self.world_size != 0:
            raise ValueError("num_experts must be divisible by world_size")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.expert = ExpertMLP(self.hidden_size, self.ffn_size, device=self.device, dtype=self.dtype).eval()
        self.inputs = torch.randn(self.batch, self.seq, self.hidden_size, device=self.device, dtype=self.dtype)

        token_ids = torch.arange(self.batch * self.seq, device=self.device, dtype=torch.int64)
        self.expert_ids = _pseudo_uniform_expert_ids(token_ids, self.num_experts).view(self.batch, self.seq)
        expert_ids_flat = self.expert_ids.reshape(-1)
        self._dest_ranks = torch.div(expert_ids_flat, self.experts_per_rank, rounding_mode="floor")
        self._perm = torch.argsort(self._dest_ranks)
        flat = self.inputs.view(-1, self.hidden_size)
        self._recv_buf = torch.empty_like(flat)
        self._out_flat = torch.empty_like(flat)

        self._verify_probe = self.inputs[:1, :1, :256].detach().cpu()
        self._verify_meta = torch.tensor(
            [int(self.world_size), int(self.experts_per_rank), int(self.num_experts)],
            dtype=torch.int64,
        )

        for _ in range(3):
            with torch.no_grad():
                _ = self.expert(self.inputs.view(-1, self.hidden_size))

    def benchmark_fn(self) -> None:
        if (
            self.expert is None
            or self.inputs is None
            or self._perm is None
            or self._recv_buf is None
            or self._out_flat is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")

        flat = self.inputs.view(-1, self.hidden_size)

        with self._nvtx_range("optimized_wide_ep"):
            with torch.no_grad():
                perm = self._perm
                recv_buf = self._recv_buf
                torch.index_select(flat, 0, perm, out=recv_buf)

                recv_out = self.expert(recv_buf)

                out_flat = self._out_flat
                out_flat.index_copy_(0, perm, recv_out)
                self.output = out_flat.view(self.batch, self.seq, self.hidden_size)


    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_probe is None or self._verify_meta is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[:2, :2, :256].detach().cpu().float().clone()
        param_count = sum(p.numel() for p in self.expert.parameters()) if self.expert is not None else 0
        self._set_verification_payload(
            inputs={"probe": self._verify_probe, "routing": self._verify_meta},
            output=output_slice,
            batch_size=int(self.batch),
            parameter_count=int(param_count),
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.0, 0.0),
            signature_overrides={
                "world_size": int(self.world_size),
                "collective_type": "all_to_all",
            },
        )

    def teardown(self) -> None:
        self.expert = None
        self.inputs = None
        self.expert_ids = None
        self._dest_ranks = None
        self._perm = None
        self._recv_buf = None
        self._out_flat = None
        self.output = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=10)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedWideEPBenchmark()
