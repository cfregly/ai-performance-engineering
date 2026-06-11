"""Harness-friendly Triton matmul benchmark with Proton defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.occupancy_tuning import triton_matmul


def _tcgen05_codegen_broken() -> bool:
    """Previously True on Blackwell Ultra (sm_103, CC 10.3) with Triton >= 3.6: the
    matmul kernel JITted to a tcgen05.wait.st intrinsic the LLVM backend could not
    select ("LLVM ERROR: Cannot select"), an uncatchable abort that forced this lab to
    skip on GB300.

    ROOT CAUSE (mechanism-proven 2026-06-11, Front P2): core/benchmark/triton_compat.py
    (imported via `import arch_config`) de-suffixed sm_103a -> sm_103, and Triton's
    arch-conditional tcgen05 intrinsics are only selectable for the 'a' target — probe D
    in code/upstream/triton-tcgen05-wait-st/STATUS.md reproduces the abort from exactly
    that transform on vanilla Triton 3.7. (The dead `num_warps: tl.constexpr` kernel
    param suspected by B18 was a red herring: probe B shows it JITs clean without the
    de-suffix.) triton_compat.py now preserves the 'a' suffix for all arches, so the
    arch_config import is unconditional again and the lab does not skip."""
    return False


@dataclass(frozen=True)
class MatmulSchedule:
    name: str
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    notes: str


BASELINE_SCHEDULE = MatmulSchedule(
    name="bm64_bn64_bk32",
    block_m=64,
    block_n=64,
    block_k=32,
    num_warps=4,
    notes="Small tile that minimizes per-block resources (baseline for comparison).",
)

OPTIMIZED_SCHEDULE = MatmulSchedule(
    name="bm128_bn128_bk64",
    block_m=128,
    block_n=128,
    block_k=64,
    num_warps=4,
    notes="Larger tile that improves compute density on Blackwell's high-bandwidth SM.",
)

EXTRA_SCHEDULE = MatmulSchedule(
    name="bm128_bn256_bk64",
    block_m=128,
    block_n=256,
    block_k=64,
    num_warps=8,
    notes="Wide-N tile with more warps for higher throughput on large matrices.",
)

WIDE_N_LATENCY_SCHEDULE = MatmulSchedule(
    name="bm64_bn256_bk32",
    block_m=64,
    block_n=256,
    block_k=32,
    num_warps=8,
    notes="Wide-N tile that keeps smaller M/K tiles while increasing work per launch.",
)

WARP_HEAVY_SCHEDULE = MatmulSchedule(
    name="bm128_bn128_bk32_nw8",
    block_m=128,
    block_n=128,
    block_k=32,
    num_warps=8,
    notes="More warps for better latency hiding; good for memory-bound scenarios.",
)

LATENCY_FRIENDLY_SCHEDULE = MatmulSchedule(
    name="bm64_bn64_bk32_nw2",
    block_m=64,
    block_n=64,
    block_k=32,
    num_warps=2,
    notes="Small tile for high occupancy; useful for verifying Proton vs Nsight agreement.",
)

SCHEDULES = [
    BASELINE_SCHEDULE,
    OPTIMIZED_SCHEDULE,
    EXTRA_SCHEDULE,
    WIDE_N_LATENCY_SCHEDULE,
    WARP_HEAVY_SCHEDULE,
    LATENCY_FRIENDLY_SCHEDULE,
]


def _ensure_inductor_env() -> None:
    os.environ.setdefault("TORCHINDUCTOR_PROTON_LOGDIR", ".proton")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM", "1")


class TritonMatmulProtonBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Wrap a Triton matmul run so the harness can drive Proton automatically."""

    allow_cpu = True

    def __init__(
        self,
        schedule: MatmulSchedule,
        *,
        size: int = 4096,
        size_k: int = 256,
        iterations: int = 2,
        warmup: int = 10,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.schedule = schedule
        self._size_m = size
        self._size_n = size
        # Use a moderate K so the matmul kernel is large enough that Proton/NVTX
        # overhead does not dominate the measurement, while still leaving room for
        # occupancy/warp-scheduling differences between schedules.
        self._size_k = size_k
        self._dtype = dtype
        self._runner: Optional[Callable[[], torch.Tensor]] = None
        self._output: Optional[torch.Tensor] = None
        self._reference: Optional[torch.Tensor] = None
        self._a: Optional[torch.Tensor] = None
        self._b: Optional[torch.Tensor] = None
        self._scratch: Optional[torch.Tensor] = None
        # Keep harness runs fast; enable profiling/proton explicitly for analysis.
        self._config = BenchmarkConfig(
            iterations=iterations,
            warmup=warmup,
            enable_nvtx=True,
            enable_profiling=False,
            enable_nsys=False,
            enable_ncu=False,
            enable_proton=False,
            profile_type="minimal",
            full_device_sync=True,
            target_label=f"labs/occupancy_tuning:{schedule.name}",
            use_subprocess=True,
        )
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: Triton Proton lab requires a CUDA device.")
        if torch.cuda.device_count() < 1:
            raise RuntimeError("SKIPPED: CUDA device unavailable for Triton Proton lab.")
        if _tcgen05_codegen_broken():
            raise RuntimeError(
                "SKIPPED: Triton >=3.6 on sm_103 (Blackwell Ultra) emits an "
                "unloadable tcgen05 kernel (LLVM ERROR: Cannot select); use the "
                "repo-pinned Triton 3.5.0."
            )

        device = torch.device("cuda")
        _ensure_inductor_env()

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self._a = torch.randn((self._size_m, self._size_k), dtype=self._dtype, device=device)
        self._b = torch.randn((self._size_k, self._size_n), dtype=self._dtype, device=device)
        self._scratch = torch.empty((self._size_m, self._size_n), dtype=self._dtype, device=device)
        with torch.no_grad():
            self._reference = torch.matmul(self._a, self._b)

        def _run_once() -> torch.Tensor:
            assert self._a is not None and self._b is not None and self._scratch is not None
            return triton_matmul.run_one(
                M=self._size_m,
                N=self._size_n,
                K=self._size_k,
                bm=self.schedule.block_m,
                bn=self.schedule.block_n,
                bk=self.schedule.block_k,
                nw=self.schedule.num_warps,
                dtype=self._dtype,
                device=device,
                a=self._a,
                b=self._b,
                c=self._scratch,
            )

        # DO NOT use torch.compile on Triton kernel calls
        # torch.compile with fullgraph=True causes SymNodeVariable AttributeError 
        # when tracing through Triton kernel launches with symbolic shapes.
        # The Triton JIT compiler already handles kernel compilation/caching.
        self._runner = _run_once

    def benchmark_fn(self) -> None:
        assert self._runner is not None
        try:
            with self._nvtx_range(self.schedule.name):
                self._output = self._runner()
                # Expose output for harness verification (harness looks for self.output)
                self.output = self._output
        except AttributeError as exc:
            if "SymNodeVariable" in str(exc):
                raise RuntimeError("SKIPPED: Triton/Proton SymNode inference is incompatible on this build.") from exc
            raise

    def capture_verification_payload(self) -> None:
        if self.output is None or self._a is None or self._b is None:
            raise RuntimeError("benchmark_fn() did not produce output or inputs")
        self._set_verification_payload(
            inputs={"a": self._a.detach(), "b": self._b.detach()},
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={
                "fp16": self._dtype == torch.float16,
                "bf16": self._dtype == torch.bfloat16,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def validate_result(self) -> Optional[str]:
        if self._reference is None or self._output is None:
            return None
        diff = (self._output - self._reference).abs().max().item()
        if torch.isnan(self._output).any():
            return "NaNs detected in Triton output"
        if diff > 2.0:
            return f"Max diff {diff:.4f} exceeds tolerance"
        return None

    def teardown(self) -> None:
        self._runner = None
        self._output = None
        self._reference = None
        self._a = None
        self._b = None
        self._scratch = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return self._config

    def get_custom_metrics(self) -> Optional[dict]:
        """Return Triton matmul schedule and roofline metrics."""
        M, N, K = self._size_m, self._size_n, self._size_k
        flops = 2.0 * M * N * K  # MAD operations
        bytes_transferred = (M * K + K * N + M * N) * 2.0  # fp16
        arithmetic_intensity = flops / bytes_transferred if bytes_transferred > 0 else 0.0
        return {
            f"triton.{self.schedule.name}.block_m": float(self.schedule.block_m),
            f"triton.{self.schedule.name}.block_n": float(self.schedule.block_n),
            f"triton.{self.schedule.name}.block_k": float(self.schedule.block_k),
            f"triton.{self.schedule.name}.num_warps": float(self.schedule.num_warps),
            f"triton.{self.schedule.name}.matrix_size": float(M),
            f"triton.{self.schedule.name}.flops": flops,
            f"triton.{self.schedule.name}.arithmetic_intensity": arithmetic_intensity,
        }

__all__ = [
    "MatmulSchedule",
    "TritonMatmulProtonBenchmark",
    "BASELINE_SCHEDULE",
    "OPTIMIZED_SCHEDULE",
    "EXTRA_SCHEDULE",
    "WIDE_N_LATENCY_SCHEDULE",
    "WARP_HEAVY_SCHEDULE",
    "LATENCY_FRIENDLY_SCHEDULE",
    "SCHEDULES",
]
