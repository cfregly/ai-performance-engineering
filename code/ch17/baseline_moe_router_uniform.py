"""baseline_moe_router_uniform.py - Uniform MoE routing with remote-transfer cost (Ch17).

Pairs with: optimized_moe_router_uniform_topology.py

Design choice (for benchmarkability):
- All experts share the same weights (a single shared expert module).
- Routing decisions therefore do NOT change the final output tensor semantics.
- The measured speedup comes from improved locality (less remote transfer), not
  changing the math.

Baseline behavior:
- Routes tokens uniformly across all islands (topology-agnostic).
- Simulates cross-island expert-parallel communication by gathering "remote"
  tokens and performing device-to-device copies.
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


class BaselineMoERouterUniformBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: topology-agnostic routing with high remote-transfer fraction."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 1024
        # Keep expert compute modest so the (simulated) remote-transfer cost is measurable.
        self.ffn_size = 128
        self.num_islands = 4
        self.experts_per_island = 16
        self.num_experts = self.num_islands * self.experts_per_island
        self.batch = 256
        self.seq = 64
        self.dtype = torch.bfloat16
        # Model remote EP transfers by scaling the payload size (fewer kernel launches,
        # more bytes moved) so results aren't dominated by copy launch overhead.
        self.remote_round_trips = 256

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
        self.local_island: Optional[torch.Tensor] = None
        self._remote_idx: Optional[torch.Tensor] = None
        self._remote_buf_a: Optional[torch.Tensor] = None
        self._remote_buf_b: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_probe: Optional[torch.Tensor] = None
        self._verify_meta: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for MoE router benchmark")

        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.experts_per_island <= 0:
            raise ValueError("experts_per_island must be positive")
        if self.num_experts != self.num_islands * self.experts_per_island:
            raise ValueError("num_experts must equal num_islands * experts_per_island")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.expert = ExpertMLP(self.hidden_size, self.ffn_size, device=self.device, dtype=self.dtype).eval()
        self._payload_parameter_count = sum(p.numel() for p in self.expert.parameters())
        self.inputs = torch.randn(self.batch, self.seq, self.hidden_size, device=self.device, dtype=self.dtype)
        self._flat_inputs = self.inputs.view(-1, self.hidden_size)

        token_ids = torch.arange(self.batch * self.seq, device=self.device, dtype=torch.int64)
        self.local_island = (token_ids % int(self.num_islands)).to(torch.int64).view(self.batch, self.seq)
        self.expert_ids = _pseudo_uniform_expert_ids(token_ids, self.num_experts).view(self.batch, self.seq)

        expert_ids_flat = self.expert_ids.reshape(-1)
        local_island_flat = self.local_island.reshape(-1)
        dest_island = torch.div(expert_ids_flat, self.experts_per_island, rounding_mode="floor")
        remote_mask = dest_island != local_island_flat
        self._remote_idx = remote_mask.nonzero(as_tuple=False).squeeze(-1)
        if self._remote_idx.numel() > 0:
            remote_tokens = int(self._remote_idx.numel())
            payload_dim = int(self.hidden_size) * int(self.remote_round_trips)
            self._remote_buf_a = torch.empty((remote_tokens, payload_dim), device=self.device, dtype=self.dtype)
            self._remote_buf_b = torch.empty((remote_tokens, payload_dim), device=self.device, dtype=self.dtype)

        probe_cols = min(256, self.hidden_size)
        self._verify_probe = torch.empty((1, 1, probe_cols), dtype=self.inputs.dtype, pin_memory=True)
        self._verify_probe.copy_(
            self.inputs[:1, :1, :probe_cols],
            non_blocking=False,
        )
        self._verify_meta = torch.tensor(
            [int(self.num_islands), int(self.experts_per_island), int(self.num_experts)],
            dtype=torch.int64,
        )
        self._verify_output_buffer = torch.empty(
            (min(2, self.batch), min(2, self.seq), min(256, self.hidden_size)),
            dtype=torch.float32,
            device="cpu",
        )

        for _ in range(3):
            with torch.inference_mode():
                _ = self.expert(self._flat_inputs)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if (
            self.expert is None
            or self.inputs is None
            or self._flat_inputs is None
            or self.expert_ids is None
            or self.local_island is None
            or self._remote_idx is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")

        flat = self._flat_inputs

        with self._nvtx_range("baseline_moe_router_uniform"):
            with torch.inference_mode():
                if self._remote_idx.numel() > 0:
                    if self._remote_buf_a is None or self._remote_buf_b is None:
                        raise RuntimeError("Remote buffers not initialized")
                    torch.index_select(flat, 0, self._remote_idx, out=self._remote_buf_a[:, : self.hidden_size])
                    self._remote_buf_b.copy_(self._remote_buf_a)
                    self._remote_buf_a.copy_(self._remote_buf_b)

                out_flat = self.expert(flat)
                self.output = out_flat.view(self.batch, self.seq, self.hidden_size)


    def capture_verification_payload(self) -> None:
        if (
            self.output is None
            or self._verify_probe is None
            or self._verify_meta is None
            or self._verify_output_buffer is None
        ):
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        verify_output = self._verify_output_buffer
        output_slice = self.output[: verify_output.shape[0], : verify_output.shape[1], : verify_output.shape[2]]
        verify_output.copy_(output_slice)
        self._set_verification_payload(
            inputs={"probe": self._verify_probe, "topology": self._verify_meta},
            output=verify_output,
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
        self.expert = None
        self.inputs = None
        self._flat_inputs = None
        self.expert_ids = None
        self.local_island = None
        self._remote_idx = None
        self._remote_buf_a = None
        self._remote_buf_b = None
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


def get_benchmark() -> BaseBenchmark:
    return BaselineMoERouterUniformBenchmark()
