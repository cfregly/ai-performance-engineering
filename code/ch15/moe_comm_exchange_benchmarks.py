"""Shared MoE communication benchmarks with explicit overlap and hierarchy variants."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.wrapper_utils import attach_benchmark_metadata as attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.optimization.moe_inference import ExpertMLP


def _pseudo_uniform_expert_ids(token_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    if token_ids.dtype != torch.int64:
        token_ids = token_ids.to(torch.int64)
    return ((token_ids * 1103515245 + 12345) % int(num_experts)).to(torch.int64)


class MoeCommExchangeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Communication-focused MoE benchmark with flat, overlap, and hierarchical paths."""

    def __init__(self, *, variant: str, label: str) -> None:
        super().__init__()
        self.variant = str(variant).strip().lower()
        if self.variant not in {"baseline", "overlap", "hierarchical"}:
            raise ValueError(f"Unsupported MoE communication variant '{variant}'")
        self.label = label

        self.hidden_size = 1024
        self.ffn_size = 4096
        self.logical_world_size = 32
        self.ranks_per_group = 4
        self.experts_per_rank = 1
        self.num_experts = self.logical_world_size * self.experts_per_rank
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
        self._flat_inputs: Optional[torch.Tensor] = None
        self.expert_ids: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._dest_ranks: Optional[torch.Tensor] = None
        self._dest_groups: Optional[torch.Tensor] = None
        self._out_flat: Optional[torch.Tensor] = None
        self._output_view: Optional[torch.Tensor] = None
        self._baseline_perm: Optional[torch.Tensor] = None
        self._baseline_packed: Optional[torch.Tensor] = None
        self._local_perm: Optional[torch.Tensor] = None
        self._local_packed: Optional[torch.Tensor] = None
        self._remote_perm: Optional[torch.Tensor] = None
        self._remote_cpu_sorted: Optional[torch.Tensor] = None
        self._remote_packed: Optional[torch.Tensor] = None
        self._hierarchical_perm: Optional[torch.Tensor] = None
        self._hierarchical_cpu_sorted: Optional[torch.Tensor] = None
        self._hierarchical_packed: Optional[torch.Tensor] = None
        self._group_offsets: Optional[torch.Tensor] = None
        self._group_ranges: Optional[list[tuple[int, int]]] = None
        self._comm_stream: Optional[torch.cuda.Stream] = None
        self._verify_probe: Optional[torch.Tensor] = None
        self._verify_meta: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for MoE communication benchmark")
        if self.logical_world_size % self.ranks_per_group != 0:
            raise ValueError("logical_world_size must be divisible by ranks_per_group")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.expert = ExpertMLP(self.hidden_size, self.ffn_size, device=self.device, dtype=self.dtype).eval()
        self._payload_parameter_count = sum(p.numel() for p in self.expert.parameters())
        self.inputs = torch.randn(self.batch, self.seq, self.hidden_size, device=self.device, dtype=self.dtype)
        self._flat_inputs = self.inputs.view(-1, self.hidden_size)
        flat = self._flat_inputs
        token_ids = torch.arange(flat.shape[0], device=self.device, dtype=torch.int64)
        self.expert_ids = _pseudo_uniform_expert_ids(token_ids, self.num_experts).view(self.batch, self.seq)
        self._dest_ranks = torch.div(self.expert_ids.reshape(-1), self.experts_per_rank, rounding_mode="floor")
        self._dest_groups = torch.div(self._dest_ranks, self.ranks_per_group, rounding_mode="floor")
        self._out_flat = torch.empty_like(flat)
        self._output_view = self._out_flat.view(self.batch, self.seq, self.hidden_size)

        baseline_perm_parts: list[torch.Tensor] = []
        for rank in range(self.logical_world_size):
            indices = (self._dest_ranks == rank).nonzero(as_tuple=False).squeeze(-1)
            if indices.numel() > 0:
                baseline_perm_parts.append(indices)
        if not baseline_perm_parts:
            raise RuntimeError("Routing produced no tokens for any logical rank")
        self._baseline_perm = torch.cat(baseline_perm_parts, dim=0)
        self._baseline_packed = torch.empty_like(flat)

        local_mask = self._dest_groups == 0
        remote_mask = ~local_mask
        self._local_perm = local_mask.nonzero(as_tuple=False).squeeze(-1)
        self._remote_perm = remote_mask.nonzero(as_tuple=False).squeeze(-1)
        if self._local_perm.numel() > 0:
            self._local_packed = torch.empty(
                self._local_perm.numel(),
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self._local_packed = None

        if self._remote_perm.numel() > 0:
            remote_sort = torch.argsort(
                (self._dest_groups.index_select(0, self._remote_perm) * self.logical_world_size)
                + self._dest_ranks.index_select(0, self._remote_perm)
            )
            self._remote_perm = self._remote_perm.index_select(0, remote_sort)
            self._remote_cpu_sorted = flat.index_select(0, self._remote_perm).detach().cpu().pin_memory()
            self._remote_packed = torch.empty(
                self._remote_perm.numel(),
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self._remote_cpu_sorted = None
            self._remote_packed = None

        hierarchical_key = (self._dest_groups * self.logical_world_size) + self._dest_ranks
        self._hierarchical_perm = torch.argsort(hierarchical_key)
        self._hierarchical_cpu_sorted = flat.index_select(0, self._hierarchical_perm).detach().cpu().pin_memory()
        self._hierarchical_packed = torch.empty_like(flat)
        group_counts = torch.bincount(self._dest_groups, minlength=self.logical_world_size // self.ranks_per_group)
        self._group_offsets = torch.empty(group_counts.numel() + 1, device=self.device, dtype=torch.int64)
        self._group_offsets[0] = 0
        torch.cumsum(group_counts, dim=0, out=self._group_offsets[1:])
        group_offsets_host = self._group_offsets.detach().cpu()
        self._group_ranges = [
            (int(group_offsets_host[idx]), int(group_offsets_host[idx + 1]))
            for idx in range(group_offsets_host.numel() - 1)
        ]
        self._comm_stream = torch.cuda.Stream(device=self.device)

        probe_cols = min(256, self.hidden_size)
        self._verify_probe = torch.empty((1, 1, probe_cols), dtype=self.inputs.dtype, pin_memory=True)
        self._verify_probe.copy_(
            self.inputs[:1, :1, :probe_cols],
            non_blocking=False,
        )
        self._verify_meta = torch.tensor(
            [int(self.logical_world_size), int(self.ranks_per_group), int(self.num_experts)],
            dtype=torch.int64,
        )
        self._verify_output_buffer = torch.empty((2, 2, 256), dtype=torch.float32)

        for _ in range(3):
            with torch.inference_mode():
                _ = self.expert(flat)
        self._synchronize()

    def get_custom_streams(self) -> list[torch.cuda.Stream]:
        if self._comm_stream is None or self.variant != "overlap":
            return []
        return [self._comm_stream]

    def benchmark_fn(self) -> None:
        if (
            self.expert is None
            or self.inputs is None
            or self._flat_inputs is None
            or self._out_flat is None
            or self._output_view is None
            or self._dest_ranks is None
            or self._dest_groups is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")

        if self.variant == "baseline":
            self._run_baseline()
        elif self.variant == "overlap":
            self._run_overlap()
        else:
            self._run_hierarchical()

    def _run_baseline(self) -> None:
        if (
            self.expert is None
            or self.inputs is None
            or self._flat_inputs is None
            or self._out_flat is None
            or self._output_view is None
            or self._baseline_perm is None
            or self._baseline_packed is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")

        flat = self._flat_inputs

        with self._nvtx_range(self.label):
            with torch.inference_mode():
                torch.index_select(flat, 0, self._baseline_perm, out=self._baseline_packed)
                baseline_out = self.expert(self._baseline_packed)
                self._out_flat.index_copy_(0, self._baseline_perm, baseline_out)
                self.output = self._output_view

    def _run_overlap(self) -> None:
        if (
            self.expert is None
            or self.inputs is None
            or self._flat_inputs is None
            or self._out_flat is None
            or self._output_view is None
            or self._local_perm is None
            or (self._local_perm.numel() > 0 and self._local_packed is None)
            or self._remote_perm is None
            or self._remote_cpu_sorted is None
            or self._remote_packed is None
            or self._comm_stream is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")

        flat = self._flat_inputs

        with self._nvtx_range(self.label):
            with torch.inference_mode():
                if self._remote_perm.numel() > 0:
                    with torch.cuda.stream(self._comm_stream):
                        self._remote_packed.copy_(self._remote_cpu_sorted, non_blocking=True)
                if self._local_perm.numel() > 0:
                    torch.index_select(flat, 0, self._local_perm, out=self._local_packed)
                    local_out = self.expert(self._local_packed)
                    self._out_flat.index_copy_(0, self._local_perm, local_out)
                if self._remote_perm.numel() > 0:
                    torch.cuda.current_stream(self.device).wait_stream(self._comm_stream)
                    remote_out = self.expert(self._remote_packed)
                    self._out_flat.index_copy_(0, self._remote_perm, remote_out)
                self.output = self._output_view

    def _run_hierarchical(self) -> None:
        if (
            self.expert is None
            or self._out_flat is None
            or self._output_view is None
            or self._hierarchical_perm is None
            or self._hierarchical_cpu_sorted is None
            or self._hierarchical_packed is None
            or self._group_ranges is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")

        with self._nvtx_range(self.label):
            with torch.inference_mode():
                self._hierarchical_packed.copy_(self._hierarchical_cpu_sorted, non_blocking=True)
                for start, end in self._group_ranges:
                    if end <= start:
                        continue
                    group_out = self.expert(self._hierarchical_packed[start:end])
                    self._out_flat.index_copy_(0, self._hierarchical_perm[start:end], group_out)
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
            inputs={"probe": self._verify_probe, "routing": self._verify_meta},
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
            signature_overrides={
                "world_size": int(self.logical_world_size),
                "shards": int(self.logical_world_size // self.ranks_per_group),
                "collective_type": "all_to_all",
            },
        )

    def teardown(self) -> None:
        self.expert = None
        self.inputs = None
        self._flat_inputs = None
        self.expert_ids = None
        self.output = None
        self._dest_ranks = None
        self._dest_groups = None
        self._out_flat = None
        self._output_view = None
        self._baseline_perm = None
        self._baseline_packed = None
        self._local_perm = None
        self._local_packed = None
        self._remote_perm = None
        self._remote_cpu_sorted = None
        self._remote_packed = None
        self._hierarchical_perm = None
        self._hierarchical_cpu_sorted = None
        self._hierarchical_packed = None
        self._group_offsets = None
        self._group_ranges = None
        self._comm_stream = None
        self._verify_probe = None
        self._verify_meta = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=10)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        total_tokens = float(self.batch * self.seq)
        remote_tokens = 0.0 if self._remote_perm is None else float(self._remote_perm.numel())
        return {
            "moe_comm.logical_world_size": float(self.logical_world_size),
            "moe_comm.ranks_per_group": float(self.ranks_per_group),
            "moe_comm.logical_groups": float(self.logical_world_size // self.ranks_per_group),
            "moe_comm.remote_token_pct": (remote_tokens / max(total_tokens, 1.0)) * 100.0,
            "moe_comm.variant_baseline": 1.0 if self.variant == "baseline" else 0.0,
            "moe_comm.variant_overlap": 1.0 if self.variant == "overlap" else 0.0,
            "moe_comm.variant_hierarchical": 1.0 if self.variant == "hierarchical" else 0.0,
        }

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        return None
