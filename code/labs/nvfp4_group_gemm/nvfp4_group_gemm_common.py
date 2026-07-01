"""Harness-compliant benchmarks for the NVFP4 grouped GEMM shape suite.

This lab intentionally uses the same benchmark shapes as:
`gpu-mode/reference-kernels/problems/nvidia/nvfp4_group_gemm/task.yml`.

We keep all non-timed work (input generation, kernel compilation, metadata allocation)
in setup() so the harness measures steady-state kernel runtime. The public benchmark
surface now uses shape-based target names instead of the older competition case IDs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from core.benchmark.verification import PrecisionFlags, simple_signature
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    ExecutionMode,
    WorkloadMetadata,
)

from labs.nvfp4_group_gemm.nvfp4_group_gemm_inputs import generate_input

input_t = Tuple[
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],  # (a, b, c) per group
    List[Tuple[torch.Tensor, torch.Tensor]],  # (sfa, sfb) per group (may be CPU)
    List[Tuple[torch.Tensor, torch.Tensor]],  # (sfa_reordered, sfb_reordered) per group (GPU)
    List[Tuple[int, int, int, int]],  # problem_sizes per group
]

output_t = List[torch.Tensor]


def _resolve_inputs_per_iteration(default_value: int) -> int:
    """Resolve inputs-per-iteration with an explicit env override.

    This keeps the default benchmark behavior unchanged while allowing
    competition-equivalent per-call measurements via:
      AISP_NVFP4_GROUP_GEMM_INPUTS_PER_ITERATION=1
    """
    override = os.environ.get("AISP_NVFP4_GROUP_GEMM_INPUTS_PER_ITERATION")
    if override is None or override.strip() == "":
        return int(default_value)
    try:
        value = int(override)
    except ValueError as exc:  # pragma: no cover - defensive validation
        raise ValueError(
            "AISP_NVFP4_GROUP_GEMM_INPUTS_PER_ITERATION must be an integer"
        ) from exc
    if value <= 0:
        raise ValueError("AISP_NVFP4_GROUP_GEMM_INPUTS_PER_ITERATION must be > 0")
    return value


def _resolve_capture_iter_graph(default_value: bool) -> bool:
    """Resolve iter-graph capture mode with an explicit env override.

    This allows apples-to-apples matrix runs where baseline and optimized wrappers
    are forced to the same graph setting:
      AISP_NVFP4_GROUP_GEMM_CAPTURE_ITER_GRAPH=0|1
    """
    override = os.environ.get("AISP_NVFP4_GROUP_GEMM_CAPTURE_ITER_GRAPH")
    if override is None or override.strip() == "":
        return bool(default_value)

    value = override.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "AISP_NVFP4_GROUP_GEMM_CAPTURE_ITER_GRAPH must be one of "
        "{0,1,true,false,yes,no,on,off}"
    )


def _resolve_optional_bool_env(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of {{0,1,true,false,yes,no,on,off}}")


def _resolve_timing_method(default_value: str = "cuda_event") -> str:
    raw = os.environ.get("AISP_NVFP4_GROUP_GEMM_TIMING_METHOD")
    if raw is None or raw.strip() == "":
        return str(default_value)
    value = raw.strip().lower()
    if value not in {"cuda_event", "wall_clock"}:
        raise ValueError(
            "AISP_NVFP4_GROUP_GEMM_TIMING_METHOD must be one of {cuda_event,wall_clock}"
        )
    return value


@dataclass(frozen=True)
class GroupGemmCase:
    name: str
    m: Tuple[int, ...]
    n: Tuple[int, ...]
    k: Tuple[int, ...]
    g: int
    seed: int

    def problem_sizes(self) -> List[Tuple[int, int, int, int]]:
        l = 1
        return [(int(self.m[i]), int(self.n[i]), int(self.k[i]), l) for i in range(self.g)]


# Benchmark shapes (order still matches task.yml even though the public targets are
# now shape-named instead of case-named).
COMPETITION_CASES: Tuple[GroupGemmCase, ...] = (
    GroupGemmCase(
        name="g8_n4096_k7168",
        m=(80, 176, 128, 72, 64, 248, 96, 160),
        n=(4096,) * 8,
        k=(7168,) * 8,
        g=8,
        seed=1111,
    ),
    GroupGemmCase(
        name="g8_n7168_k2048",
        m=(40, 76, 168, 72, 164, 148, 196, 160),
        n=(7168,) * 8,
        k=(2048,) * 8,
        g=8,
        seed=1111,
    ),
    GroupGemmCase(
        name="g2_n3072_k4096",
        m=(192, 320),
        n=(3072,) * 2,
        k=(4096,) * 2,
        g=2,
        seed=1111,
    ),
    GroupGemmCase(
        name="g2_n4096_k1536",
        m=(128, 384),
        n=(4096,) * 2,
        k=(1536,) * 2,
        g=2,
        seed=1111,
    ),
)


class NVFP4GroupGemmBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark wrapper that executes a provided NVFP4 grouped GEMM implementation."""

    def __init__(
        self,
        *,
        case: GroupGemmCase,
        custom_kernel: Callable[[input_t], output_t],
        prepare: Optional[Callable[[Sequence[input_t]], None]] = None,
        inputs_per_iteration: int = 15,
        capture_iter_graph: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.case = case
        self._name = name or case.name
        self._custom_kernel = custom_kernel
        self._prepare = prepare
        self.inputs_per_iteration = _resolve_inputs_per_iteration(int(inputs_per_iteration))

        self.data_list: List[input_t] = []
        self._last_output: Optional[output_t] = None
        self._verify_output: Optional[torch.Tensor] = None
        self._iter_graph: Optional[torch.cuda.CUDAGraph] = None
        self._iter_graph_last_output: Optional[output_t] = None
        self._capture_iter_graph = _resolve_capture_iter_graph(capture_iter_graph)

        # Workload invariant: competition eval averages 15 independent inputs per timing run.
        self._workload = WorkloadMetadata(requests_per_iteration=float(self.inputs_per_iteration))

        # This benchmark is not deterministic with respect to seed (different inputs -> different outputs).
        self._is_deterministic = False

    def setup(self) -> None:
        # Rebuild inputs on every setup() call because verify_runner reuses benchmark instances.
        self.data_list = []
        self._last_output = None
        self._verify_output = None
        self._iter_graph = None
        self._iter_graph_last_output = None

        # IMPORTANT: incorporate the harness-configured seed so fresh-input checks can validate
        # that different seeds produce different outputs (and to prevent accidental "constant
        # output" behavior). Do NOT reseed here; the harness enforces seed immutability.
        harness_seed = int(torch.initial_seed())
        base_seed = int(self.case.seed) + harness_seed
        for i in range(self.inputs_per_iteration):
            seed = base_seed + 42 * i
            self.data_list.append(
                generate_input(
                    m=self.case.m,
                    n=self.case.n,
                    k=self.case.k,
                    g=self.case.g,
                    seed=seed,
                )
            )

        if self._prepare is not None:
            maybe_data = self._prepare(self.data_list)
            if maybe_data is not None:
                self.data_list = list(maybe_data)

        total_output_elements = sum(
            int(m) * int(n) * int(l)
            for m, n, _k, l in self.case.problem_sizes()
        )
        self._verify_output = torch.empty(
            total_output_elements,
            device=self.device,
            dtype=torch.float16,
        )

        # Avoid async work leaking into first measured iteration.
        self._synchronize()

        # Optional: capture one CUDA graph for the full 15-request loop to reduce launch overhead.
        # Workload remains identical (same inputs and same per-request kernels), but replay uses
        # a single graph launch in benchmark_fn().
        if self._capture_iter_graph:
            try:
                with torch.inference_mode():
                    out = None
                    for data in self.data_list:
                        out = self._custom_kernel(data)
                self._synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.inference_mode(), torch.cuda.graph(graph):
                    out = None
                    for data in self.data_list:
                        out = self._custom_kernel(data)

                self._iter_graph = graph
                self._iter_graph_last_output = out
                self._synchronize()
            except RuntimeError:
                # Some tuned kernel schedules are not graph-capture safe on all shapes.
                # Fall back to steady-state non-graph execution while keeping correctness.
                self._iter_graph = None
                self._iter_graph_last_output = None

    def benchmark_fn(self) -> None:
        if not self.data_list:
            raise RuntimeError("setup() did not create inputs")

        if self._iter_graph is not None:
            with torch.inference_mode():
                self._iter_graph.replay()
            self._last_output = self._iter_graph_last_output
            if self._last_output is None:
                raise RuntimeError("iter-graph capture did not produce output")
        else:
            with torch.inference_mode():
                out: Optional[output_t] = None
                for data in self.data_list:
                    out = self._custom_kernel(data)
            self._last_output = out
            if self._last_output is None:
                raise RuntimeError("custom_kernel did not produce output")
        # BenchmarkHarness performs full-device synchronization around timed iterations.
        # Avoiding an extra in-function sync reduces fixed per-iteration overhead.

    def capture_verification_payload(self) -> None:
        if not self.data_list or self._last_output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")

        # Use FP4 A from the first group/input as the jitter probe input. Jitter check will likely
        # skip for Float4/Float8 because torch.randn_like is not implemented, but output verification
        # still provides strong protection.
        a0 = self.data_list[0][0][0][0]

        # Combine per-group outputs into a single tensor to avoid input-output aliasing on C buffers.
        if self._verify_output is None:
            raise RuntimeError("Verification output buffer not initialized")
        offset = 0
        for group_output in self._last_output:
            flat_output = group_output.reshape(-1)
            next_offset = offset + int(flat_output.numel())
            self._verify_output[offset:next_offset].copy_(flat_output)
            offset = next_offset
        combined = self._verify_output[:offset]

        self._set_verification_payload(
            inputs={"a0": a0},
            output=combined,
            batch_size=self.case.g,
            parameter_count=0,
            output_tolerance=(1e-3, 1e-3),
            precision_flags={"fp16": True, "fp8": True, "tf32": False},
        )

    def teardown(self) -> None:
        self.data_list = []
        self._last_output = None
        self._verify_output = None
        self._iter_graph = None
        self._iter_graph_last_output = None
        torch.cuda.empty_cache()

    def validate_result(self) -> Optional[str]:
        if self._last_output is None:
            return "Output not produced"
        if len(self._last_output) != self.case.g:
            return f"Expected {self.case.g} group outputs, got {len(self._last_output)}"
        return None

    def get_input_signature(self) -> dict:
        tf32_enabled = torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32)
        return simple_signature(
            batch_size=self.case.g,
            dtype="float16",
            groups=self.case.g,
            total_m=sum(int(v) for v in self.case.m),
            total_n=sum(int(v) for v in self.case.n),
            total_k=sum(int(v) for v in self.case.k),
            inputs_per_iteration=self.inputs_per_iteration,
            seed=self.case.seed,
            precision_flags=PrecisionFlags(fp16=True, fp8=True, tf32=tf32_enabled),
        ).to_dict()

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_config(self) -> BenchmarkConfig:
        # Compilation happens in setup(); keep measured loop focused on steady-state execution.
        config = BenchmarkConfig(
            iterations=25,
            warmup=5,
            setup_timeout_seconds=900,
            measurement_timeout_seconds=300,
            execution_mode=ExecutionMode.THREAD,
            clear_l2_cache=True,
            enable_nvtx=True,
            timing_method=_resolve_timing_method("cuda_event"),
            timing_cross_validation_threshold=0.40,
            # Nsight Compute wrapper scripts profile benchmark_fn() directly (not via BenchmarkHarness),
            # so we rely on an explicit NVTX include filter and a matching NVTX range around the
            # measured call. See core/harness/run_benchmarks.py profiling wrappers.
            nsys_nvtx_include=["compute_kernel:benchmark_fn"],
        )
        cross_validate_override = _resolve_optional_bool_env(
            "AISP_NVFP4_GROUP_GEMM_CROSS_VALIDATE_TIMING"
        )
        if cross_validate_override is not None:
            config.cross_validate_timing = bool(cross_validate_override)
        return config

    def get_custom_metrics(self) -> Optional[dict]:
        # Per-call microseconds, matching competition-style reporting.
        cfg = self.get_config()
        # The harness records latency_ms for each iteration; we expose the conversion
        # factor so downstream analysis can compute per-call numbers deterministically.
        return {
            "requests_per_iteration": float(self.inputs_per_iteration),
            "case_groups": float(self.case.g),
        }


__all__ = [
    "COMPETITION_CASES",
    "GroupGemmCase",
    "NVFP4GroupGemmBenchmark",
    "attach_benchmark_metadata",
    "input_t",
    "output_t",
]
