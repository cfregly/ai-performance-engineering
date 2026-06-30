#!/usr/bin/env python3
"""Shared helpers for the split Chapter 15 MoE routing/dispatch benchmarks."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.optimization.moe_inference import ExpertMLP
from core.optimization.shared_expert_dispatch import (
    dispatch_shared_expert_active_experts,
    dispatch_shared_expert_mask_scan,
    dispatch_shared_expert_precomputed_indices,
)


def pseudo_uniform_expert_ids(token_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Stable pseudo-uniform routing used by the Chapter 15 baselines."""
    if token_ids.dtype != torch.int64:
        token_ids = token_ids.to(torch.int64)
    return ((token_ids * 1103515245 + 12345) % int(num_experts)).to(torch.int64)


def topology_aware_expert_ids(token_ids: torch.Tensor, *, local_experts: int) -> torch.Tensor:
    """Simple locality-biased routing used by the topology-aware variant."""
    if token_ids.dtype != torch.int64:
        token_ids = token_ids.to(torch.int64)
    return (token_ids % int(local_experts)).to(torch.int64)


def active_expert_ids_for_static_route(
    *,
    route_mode: str,
    num_tokens: int,
    num_experts: int,
    local_experts: int,
) -> list[int]:
    """Return active experts for deterministic routes without a GPU readback."""
    if num_tokens <= 0:
        return []
    if route_mode == "uniform":
        return sorted(
            {
                int((token_id * 1103515245 + 12345) % int(num_experts))
                for token_id in range(int(num_tokens))
            }
        )
    if route_mode == "topology_aware":
        return list(range(min(int(local_experts), int(num_tokens))))
    raise ValueError(f"Unknown route mode: {route_mode}")


class SharedExpertMoEBenchmarkBase(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark base that keeps expert weights fixed across routing studies."""

    route_mode = "uniform"
    dispatch_mode = "mask_scan"
    nvtx_label = "moe"

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 1024
        self.ffn_size = 4096
        self.num_experts = 64
        self.local_experts = 8
        self.batch = 64
        self.seq = 16
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
        self._flat_inputs: Optional[torch.Tensor] = None
        self.expert_ids: Optional[torch.Tensor] = None
        self._expert_ids_flat: Optional[torch.Tensor] = None
        self._active_dispatch_indices: Optional[list[torch.Tensor]] = None
        self._out_flat: Optional[torch.Tensor] = None
        self._output_view: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_probe: Optional[torch.Tensor] = None
        self._verify_meta: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0

    def _build_expert_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.route_mode == "uniform":
            return pseudo_uniform_expert_ids(token_ids, self.num_experts)
        if self.route_mode == "topology_aware":
            return topology_aware_expert_ids(token_ids, local_experts=self.local_experts)
        raise ValueError(f"Unknown route mode: {self.route_mode}")

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for MoE routing benchmark")
        if self.local_experts <= 0 or self.local_experts > self.num_experts:
            raise ValueError("local_experts must be in [1, num_experts]")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.expert = ExpertMLP(self.hidden_size, self.ffn_size, device=self.device, dtype=self.dtype).eval()
        self._payload_parameter_count = sum(p.numel() for p in self.expert.parameters())
        self.inputs = torch.randn(self.batch, self.seq, self.hidden_size, device=self.device, dtype=self.dtype)
        self._flat_inputs = self.inputs.view(-1, self.hidden_size)

        token_ids = torch.arange(self.batch * self.seq, device=self.device, dtype=torch.int64)
        self.expert_ids = self._build_expert_ids(token_ids).view(self.batch, self.seq)
        self._expert_ids_flat = self.expert_ids.reshape(-1)
        expert_ids_flat = self._expert_ids_flat
        if self.dispatch_mode == "active_experts":
            active_experts = active_expert_ids_for_static_route(
                route_mode=self.route_mode,
                num_tokens=expert_ids_flat.numel(),
                num_experts=self.num_experts,
                local_experts=self.local_experts,
            )
            self._active_dispatch_indices = [
                (expert_ids_flat == int(expert_id)).nonzero(as_tuple=False).squeeze(-1)
                for expert_id in active_experts
            ]
        else:
            self._active_dispatch_indices = None
        self._out_flat = torch.empty(self.batch * self.seq, self.hidden_size, device=self.device, dtype=self.dtype)
        self._output_view = self._out_flat.view(self.batch, self.seq, self.hidden_size)
        probe_cols = min(256, self.hidden_size)
        self._verify_probe = torch.empty((1, 1, probe_cols), dtype=self.inputs.dtype, pin_memory=True)
        self._verify_probe.copy_(
            self.inputs[:1, :1, :probe_cols],
            non_blocking=False,
        )
        self._verify_meta = torch.zeros(self.num_experts, dtype=torch.int8)
        self._verify_output_buffer = torch.empty((2, 2, 256), dtype=torch.float32)

        for _ in range(3):
            with torch.inference_mode():
                _ = self.expert(self._flat_inputs)

    def benchmark_fn(self) -> None:
        if (
            self.expert is None
            or self.inputs is None
            or self._flat_inputs is None
            or self.expert_ids is None
            or self._expert_ids_flat is None
            or self._out_flat is None
            or self._output_view is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")

        flat = self._flat_inputs
        expert_ids_flat = self._expert_ids_flat

        with self._nvtx_range(self.nvtx_label):
            with torch.inference_mode():
                if self.dispatch_mode == "mask_scan":
                    dispatch_shared_expert_mask_scan(
                        flat,
                        expert_ids_flat,
                        self.expert,
                        num_experts=self.num_experts,
                        out=self._out_flat,
                    )
                elif self.dispatch_mode == "active_experts":
                    if self._active_dispatch_indices is not None:
                        dispatch_shared_expert_precomputed_indices(
                            flat,
                            self.expert,
                            out=self._out_flat,
                            index_groups=self._active_dispatch_indices,
                        )
                    else:
                        dispatch_shared_expert_active_experts(
                            flat,
                            expert_ids_flat,
                            self.expert,
                            out=self._out_flat,
                        )
                else:
                    raise ValueError(f"Unknown dispatch mode: {self.dispatch_mode}")
                self.output = self._output_view

    def capture_verification_payload(self) -> None:
        if (
            self.output is None
            or self._verify_probe is None
            or self._verify_meta is None
            or self._verify_output_buffer is None
        ):
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
            : self._verify_output_buffer.shape[2],
        ].detach()
        self._verify_output_buffer.copy_(output_slice, non_blocking=False)
        self._set_verification_payload(
            inputs={"probe": self._verify_probe, "expert_meta": self._verify_meta},
            output=self._verify_output_buffer,
            batch_size=int(self.batch),
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.expert = None
        self.inputs = None
        self._flat_inputs = None
        self.expert_ids = None
        self._expert_ids_flat = None
        self._active_dispatch_indices = None
        self._out_flat = None
        self._output_view = None
        self.output = None
        self._verify_probe = None
        self._verify_meta = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=30, warmup=10)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return {
            "moe.num_experts": float(self.num_experts),
            "moe.local_experts": float(
                self.local_experts if self.route_mode == "topology_aware" else self.num_experts
            ),
            "moe.routing_uniform": 1.0 if self.route_mode == "uniform" else 0.0,
            "moe.routing_topology_aware": 1.0 if self.route_mode == "topology_aware" else 0.0,
            "moe.dispatch_mask_scan": 1.0 if self.dispatch_mode == "mask_scan" else 0.0,
            "moe.dispatch_active_experts": 1.0 if self.dispatch_mode == "active_experts" else 0.0,
        }

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        return None
