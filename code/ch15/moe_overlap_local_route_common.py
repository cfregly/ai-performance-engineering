"""Shared implementation for the Ch15 local-route MoE overlap benchmark.

The pair compares two serving placements with identical tensor math:

- remote routed placement: route-ordered activations are transferred back from a
  simulated remote expert-parallel shard, then scattered into token order.
- local routed placement: the routed expert is colocated with the shared expert,
  so routed output is produced in token order and accumulated directly.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.optimization.moe_inference import ExpertMLP
from core.optimization.shared_expert_dispatch import dispatch_shared_expert_packed_scatter


Variant = Literal["remote_overlap", "local_direct"]


def _pseudo_uniform_expert_ids(token_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    if token_ids.dtype != torch.int64:
        token_ids = token_ids.to(torch.int64)
    return ((token_ids * 1103515245 + 12345) % int(num_experts)).to(torch.int64)


class MoeOverlapLocalRouteBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """MoE overlap benchmark parameterized by routed expert placement."""

    def __init__(self, *, variant: Variant, label: str) -> None:
        super().__init__()
        if variant not in ("remote_overlap", "local_direct"):
            raise ValueError(f"Unsupported local route variant: {variant}")
        self.variant = variant
        self.label = label
        self.hidden_size = 2048
        self.shared_ffn_size = 8192
        self.routed_ffn_size = 128
        self.num_experts = 4
        self.batch = 128
        self.seq = 64
        self.dtype = torch.bfloat16
        self.comm_chunks = 8
        self.comm_round_trips = 12

        tokens = self.batch * self.seq
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

        self.shared_expert: Optional[nn.Module] = None
        self.routed_expert: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.expert_ids: Optional[torch.Tensor] = None
        self._flat_inputs: Optional[torch.Tensor] = None
        self._dispatch_order: Optional[torch.Tensor] = None
        self._remote_cpu_flat: Optional[torch.Tensor] = None
        self._packed_comm_flat: Optional[torch.Tensor] = None
        self._routed_out_flat: Optional[torch.Tensor] = None
        self._comm_stream: Optional[torch.cuda.Stream] = None
        self._comm_copy_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._comm_round_range = range(self.comm_round_trips)
        self._comm_round_count = 0
        self.output: Optional[torch.Tensor] = None
        self._verify_probe: Optional[torch.Tensor] = None
        self._verify_meta: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for MoE local-route benchmark")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.shared_expert = ExpertMLP(
            self.hidden_size,
            self.shared_ffn_size,
            device=self.device,
            dtype=self.dtype,
        ).eval()
        self.routed_expert = ExpertMLP(
            self.hidden_size,
            self.routed_ffn_size,
            device=self.device,
            dtype=self.dtype,
        ).eval()
        self._payload_parameter_count = sum(p.numel() for p in self.shared_expert.parameters()) + sum(
            p.numel() for p in self.routed_expert.parameters()
        )
        self.inputs = torch.randn(self.batch, self.seq, self.hidden_size, device=self.device, dtype=self.dtype)
        self._flat_inputs = self.inputs.view(-1, self.hidden_size)

        token_ids = torch.arange(self.batch * self.seq, device=self.device, dtype=torch.int64)
        self.expert_ids = _pseudo_uniform_expert_ids(token_ids, self.num_experts).view(self.batch, self.seq)
        if self.variant == "remote_overlap":
            self._setup_remote_overlap()
        else:
            self._comm_round_count = len(self._comm_round_range)

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
                _ = self.shared_expert(self._flat_inputs)
                _ = self.routed_expert(self._flat_inputs)
        self._synchronize()

    def _setup_remote_overlap(self) -> None:
        if self._flat_inputs is None or self.expert_ids is None:
            raise RuntimeError("shared setup must initialize inputs before remote overlap setup")
        self._dispatch_order = torch.argsort(self.expert_ids.reshape(-1))
        self._packed_comm_flat = torch.empty(
            self.batch * self.seq,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        self._routed_out_flat = torch.empty_like(self._packed_comm_flat)
        self._comm_stream = torch.cuda.Stream(device=self.device)
        self._remote_cpu_flat = self._flat_inputs.index_select(0, self._dispatch_order).detach().cpu().pin_memory()
        total_tokens = self._flat_inputs.shape[0]
        chunk_tokens = max(1, (total_tokens + self.comm_chunks - 1) // self.comm_chunks)
        self._comm_copy_pairs = [
            (self._packed_comm_flat[start:end], self._remote_cpu_flat[start:end])
            for start in range(0, total_tokens, chunk_tokens)
            for end in (min(start + chunk_tokens, total_tokens),)
        ]
        self._comm_round_range = range(self.comm_round_trips)
        self._comm_round_count = len(self._comm_round_range)

    def get_custom_streams(self) -> list[torch.cuda.Stream]:
        if self._comm_stream is None:
            return []
        return [self._comm_stream]

    def benchmark_fn(self) -> None:
        if self.variant == "remote_overlap":
            self._benchmark_remote_overlap()
        else:
            self._benchmark_local_direct()

    def _benchmark_remote_overlap(self) -> None:
        if (
            self.shared_expert is None
            or self.routed_expert is None
            or self.inputs is None
            or self._flat_inputs is None
            or self._dispatch_order is None
            or self._remote_cpu_flat is None
            or self._packed_comm_flat is None
            or self._routed_out_flat is None
            or self._comm_stream is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")
        if not self._comm_copy_pairs or self._comm_round_count != self.comm_round_trips:
            raise RuntimeError("setup() must initialize communication chunk views")

        flat = self._flat_inputs
        with self._nvtx_range(self.label):
            with torch.inference_mode():
                with torch.cuda.stream(self._comm_stream):
                    for _ in self._comm_round_range:
                        for comm_chunk, remote_chunk in self._comm_copy_pairs:
                            comm_chunk.copy_(remote_chunk, non_blocking=True)
                shared_out = self.shared_expert(flat)
                torch.cuda.current_stream(self.device).wait_stream(self._comm_stream)
                dispatch_shared_expert_packed_scatter(
                    self._packed_comm_flat,
                    self.routed_expert,
                    out=self._routed_out_flat,
                    sort_idx=self._dispatch_order,
                )
                shared_out.add_(self._routed_out_flat)
                self.output = shared_out.view(self.batch, self.seq, self.hidden_size)

    def _benchmark_local_direct(self) -> None:
        if (
            self.shared_expert is None
            or self.routed_expert is None
            or self.inputs is None
            or self._flat_inputs is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")

        flat = self._flat_inputs
        with self._nvtx_range(self.label):
            with torch.inference_mode():
                shared_out = self.shared_expert(flat)
                shared_out.add_(self.routed_expert(flat))
                self.output = shared_out.view(self.batch, self.seq, self.hidden_size)

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
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.shared_expert = None
        self.routed_expert = None
        self.inputs = None
        self.expert_ids = None
        self._flat_inputs = None
        self._dispatch_order = None
        self._remote_cpu_flat = None
        self._packed_comm_flat = None
        self._routed_out_flat = None
        self._comm_stream = None
        self._comm_copy_pairs = []
        self._comm_round_range = range(self.comm_round_trips)
        self._comm_round_count = 0
        self.output = None
        self._verify_probe = None
        self._verify_meta = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=10)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        return None
