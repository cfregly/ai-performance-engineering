"""Harness-comparable B200 BF16 GEMM and native copy benchmark pairs."""

from __future__ import annotations

from typing import Any

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.arithmetic_intensity_b200.ceiling_extension import load_ceiling_extension
from labs.arithmetic_intensity_b200.optimization_extension import (
    load_optimization_extension,
)
from labs.arithmetic_intensity_b200.optimization_protocol import (
    select_algorithm_index,
)
from labs.arithmetic_intensity_b200.verification_tool import (
    _bf16_fp32_accumulation_policy,
)

GEMM_SIZE = 16_384
GEMM_SEED = 20_261_014
GEMM_WORKSPACE_BYTES = 512 * 1024 * 1024
GEMM_MAX_ALGORITHMS = 32
SELECTED_GEMM_ALGORITHM_SIGNATURE = (
    "id=66,tile=513,stages=35,split_k=1,reduction=0,swizzle=0,custom=1,cluster=3"
)

COPY_PAYLOAD_BYTES = 8 * 1024 * 1024 * 1024
COPY_SEED = 20_271_014
BASELINE_COPY_THREADS = 512
BASELINE_COPY_BLOCKS_PER_SM = 16
SELECTED_COPY_THREADS = 512
SELECTED_COPY_BLOCKS_PER_SM = 16


def _require_b200(device: torch.device) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The arithmetic-intensity B200 optimization pair requires CUDA")
    major, minor = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    if (major, minor) != (10, 0) or "B200" not in name:
        raise RuntimeError(
            "The arithmetic-intensity B200 optimization pair requires an NVIDIA B200 "
            f"with compute capability 10.0, got {name} at {major}.{minor}"
        )


class Bf16GemmBenchmarkBase(VerificationPayloadMixin, BaseBenchmark):
    """Equivalent dense BF16 beta-zero GEMM workload for both implementations."""

    implementation_id = ""
    preferred_ncu_replay_mode = "kernel"

    def __init__(self) -> None:
        super().__init__()
        self.a: torch.Tensor | None = None
        self.b: torch.Tensor | None = None
        self._output_buffer: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self._extension: Any = None
        self._plan: Any = None
        self._algorithm_index: int | None = None
        self._algorithm_metadata: dict[str, Any] = {}
        self._benchmark_has_run = False
        self._verification_reference: torch.Tensor | None = None
        self._primary_timed_output_verification_passed = False
        self._post_timing_verification_passed = False
        self._bf16_policy_context: Any = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            samples_per_iteration=float(GEMM_SIZE * GEMM_SIZE),
            bytes_per_iteration=float((2 * GEMM_SIZE**2 + GEMM_SIZE**2) * 2),
            custom_units_per_iteration=float(2 * GEMM_SIZE**3),
            custom_unit_name="FLOP",
        )

    def _prepare_implementation(self) -> None:
        return None

    def _run_implementation(self) -> None:
        raise NotImplementedError

    def setup(self) -> None:
        try:
            self._bf16_policy_context = _bf16_fp32_accumulation_policy(torch)
            self._bf16_policy_context.__enter__()
            _require_b200(self.device)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(GEMM_SEED)
            scale = GEMM_SIZE**-0.25
            self.a = torch.randn(
                (GEMM_SIZE, GEMM_SIZE),
                dtype=torch.bfloat16,
                device=self.device,
                generator=generator,
            ).mul_(scale)
            self.b = torch.randn(
                (GEMM_SIZE, GEMM_SIZE),
                dtype=torch.bfloat16,
                device=self.device,
                generator=generator,
            ).mul_(scale)
            self._output_buffer = torch.empty_like(self.a)
            self._prepare_implementation()
            previous = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("highest")
            try:
                self._verification_reference = torch.mm(self.a.float(), self.b.float())
            finally:
                torch.set_float32_matmul_precision(previous)
            self._output_buffer.fill_(float("nan"))
            torch.cuda.synchronize(self.device)
            self._run_implementation()
            torch.cuda.synchronize(self.device)
            torch.testing.assert_close(
                self._output_buffer.float(),
                self._verification_reference,
                rtol=0.03,
                atol=0.03,
            )
            self._output_buffer.fill_(float("nan"))
            torch.cuda.synchronize(self.device)
            self.output = None
            self._benchmark_has_run = False
            self._primary_timed_output_verification_passed = False
            self._post_timing_verification_passed = False
        except Exception:
            if self._bf16_policy_context is not None:
                self._bf16_policy_context.__exit__(None, None, None)
                self._bf16_policy_context = None
            raise

    def benchmark_fn(self) -> None:
        with self._nvtx_range(f"ai_b200_optimization_{self.implementation_id}"):
            self._run_implementation()
        self.output = self._output_buffer
        self._benchmark_has_run = True

    def capture_verification_payload(self) -> None:
        if not self._benchmark_has_run:
            raise RuntimeError("benchmark_fn() must run before verification capture")
        if (
            self.a is None
            or self.b is None
            or self._output_buffer is None
            or self._verification_reference is None
        ):
            raise RuntimeError("benchmark tensors are unavailable")
        torch.cuda.synchronize(self.device)
        torch.testing.assert_close(
            self._output_buffer.float(),
            self._verification_reference,
            rtol=0.03,
            atol=0.03,
        )
        self._primary_timed_output_verification_passed = True
        self.output = None
        self._output_buffer.fill_(float("nan"))
        torch.cuda.synchronize(self.device)
        self._run_implementation()
        torch.cuda.synchronize(self.device)
        torch.testing.assert_close(
            self._output_buffer.float(),
            self._verification_reference,
            rtol=0.03,
            atol=0.03,
        )
        self.output = self._output_buffer
        self._post_timing_verification_passed = True
        self._set_verification_payload(
            inputs={"a": self.a, "b": self.b},
            output=self.output,
            batch_size=GEMM_SIZE,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": True, "fp8": False, "tf32": False},
            output_tolerance=(0.03, 0.03),
        )

    def validate_result(self) -> str | None:
        if self.output is None:
            return "benchmark_fn() did not produce output"
        if tuple(self.output.shape) != (GEMM_SIZE, GEMM_SIZE):
            return f"unexpected GEMM output shape {tuple(self.output.shape)}"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=210,
            warmup=30,
            profiling_iterations=1,
            setup_timeout_seconds=900,
            measurement_timeout_seconds=600,
            profiling_timeout_seconds=1_200,
            ncu_timeout_seconds=1_200,
            nsys_timeout_seconds=1_200,
            enable_nvtx=True,
        )

    def get_custom_metrics(self) -> dict[str, float]:
        metrics = {
            "b200_optimization.gemm_size": float(GEMM_SIZE),
            "b200_optimization.beta_zero": 1.0,
            "b200_optimization.dense_bf16": 1.0,
            "b200_optimization.allow_bf16_reduced_precision_reduction": 0.0,
            "b200_optimization.primary_timed_output_verification": float(
                self._primary_timed_output_verification_passed
            ),
            "b200_optimization.fresh_post_timing_verification": float(
                self._post_timing_verification_passed
            ),
        }
        if self._algorithm_index is not None:
            metrics["b200_optimization.cublaslt_heuristic_index"] = float(self._algorithm_index)
        return metrics

    def teardown(self) -> None:
        self._plan = None
        self._extension = None
        self.a = None
        self.b = None
        self._output_buffer = None
        self.output = None
        self._benchmark_has_run = False
        self._verification_reference = None
        self._primary_timed_output_verification_passed = False
        self._post_timing_verification_passed = False
        try:
            super().teardown()
        finally:
            if self._bf16_policy_context is not None:
                self._bf16_policy_context.__exit__(None, None, None)
                self._bf16_policy_context = None


class BaselineBf16GemmBenchmark(Bf16GemmBenchmarkBase):
    """Torch matrix multiplication dispatched through its default cuBLAS path."""

    implementation_id = "torch_mm_out_default_cublas"

    def _run_implementation(self) -> None:
        if self.a is None or self.b is None or self._output_buffer is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        torch.mm(self.a, self.b, out=self._output_buffer)


class OptimizedBf16GemmBenchmark(Bf16GemmBenchmarkBase):
    """Persistent direct-cuBLASLt plan using the frozen supported algorithm."""

    implementation_id = "direct_cublaslt_frozen_algorithm"

    def _prepare_implementation(self) -> None:
        if self.a is None or self.b is None or self._output_buffer is None:
            raise RuntimeError("GEMM tensors must exist before plan creation")
        self._extension = load_optimization_extension()
        self._plan = self._extension.Bf16LtMatmulPlan(
            self.a,
            self.b,
            self._output_buffer,
            GEMM_WORKSPACE_BYTES,
            GEMM_MAX_ALGORITHMS,
        )
        algorithms = list(self._plan.algorithms())
        self._algorithm_index = select_algorithm_index(
            algorithms,
            SELECTED_GEMM_ALGORITHM_SIGNATURE,
        )
        self._algorithm_metadata = dict(algorithms[self._algorithm_index])

    def _run_implementation(self) -> None:
        if (
            self._plan is None
            or self._algorithm_index is None
            or self.a is None
            or self.b is None
            or self._output_buffer is None
        ):
            raise RuntimeError("setup() must create the cuBLASLt plan before benchmark_fn()")
        self._plan.run(self.a, self.b, self._output_buffer, self._algorithm_index)


class NativeCopyBenchmarkBase(VerificationPayloadMixin, BaseBenchmark):
    """Equivalent random 8 GiB uint8 native-copy workload for both kernels."""

    implementation_id = ""
    preferred_ncu_replay_mode = "kernel"

    def __init__(self) -> None:
        super().__init__()
        self.source: torch.Tensor | None = None
        self._output_buffer: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self._extension: Any = None
        self._blocks = 0
        self._benchmark_has_run = False
        self._primary_timed_output_verification_passed = False
        self._post_timing_verification_passed = False
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            samples_per_iteration=float(COPY_PAYLOAD_BYTES),
            bytes_per_iteration=float(2 * COPY_PAYLOAD_BYTES),
        )

    def _prepare_implementation(self) -> None:
        raise NotImplementedError

    def _run_implementation(self) -> None:
        raise NotImplementedError

    def setup(self) -> None:
        _require_b200(self.device)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(COPY_SEED)
        self.source = torch.randint(
            0,
            256,
            (COPY_PAYLOAD_BYTES,),
            dtype=torch.uint8,
            device=self.device,
            generator=generator,
        )
        self._output_buffer = torch.empty_like(self.source)
        self._prepare_implementation()
        torch.bitwise_not(self.source, out=self._output_buffer)
        torch.cuda.synchronize(self.device)
        self._run_implementation()
        torch.cuda.synchronize(self.device)
        if not torch.equal(self._output_buffer, self.source):
            raise RuntimeError(f"{self.implementation_id} failed full-tensor equality")
        torch.bitwise_not(self.source, out=self._output_buffer)
        torch.cuda.synchronize(self.device)
        self.output = None
        self._benchmark_has_run = False
        self._primary_timed_output_verification_passed = False
        self._post_timing_verification_passed = False

    def benchmark_fn(self) -> None:
        with self._nvtx_range(f"ai_b200_optimization_{self.implementation_id}"):
            self._run_implementation()
        self.output = self._output_buffer
        self._benchmark_has_run = True

    def capture_verification_payload(self) -> None:
        if not self._benchmark_has_run:
            raise RuntimeError("benchmark_fn() must run before verification capture")
        if self.source is None or self._output_buffer is None:
            raise RuntimeError("benchmark tensors are unavailable")
        torch.cuda.synchronize(self.device)
        if not torch.equal(self._output_buffer, self.source):
            raise RuntimeError(f"{self.implementation_id} failed full timed-output equality")
        self._primary_timed_output_verification_passed = True
        self.output = None
        torch.bitwise_not(self.source, out=self._output_buffer)
        torch.cuda.synchronize(self.device)
        self._run_implementation()
        torch.cuda.synchronize(self.device)
        if not torch.equal(self._output_buffer, self.source):
            raise RuntimeError(
                f"{self.implementation_id} failed the fresh post-timing full-tensor audit"
            )
        self.output = self._output_buffer
        self._post_timing_verification_passed = True
        self._set_verification_payload(
            inputs={"source": self.source},
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.0, 0.0),
        )

    def validate_result(self) -> str | None:
        if self.output is None:
            return "benchmark_fn() did not produce output"
        if self.output.numel() != COPY_PAYLOAD_BYTES:
            return f"unexpected copy output byte count {self.output.numel()}"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=405,
            warmup=27,
            profiling_iterations=1,
            setup_timeout_seconds=900,
            measurement_timeout_seconds=600,
            profiling_timeout_seconds=1_200,
            ncu_timeout_seconds=1_200,
            nsys_timeout_seconds=1_200,
            enable_nvtx=True,
        )

    def get_custom_metrics(self) -> dict[str, float]:
        return {
            "b200_optimization.copy_payload_bytes": float(COPY_PAYLOAD_BYTES),
            "b200_optimization.copy_requested_bytes": float(2 * COPY_PAYLOAD_BYTES),
            "b200_optimization.copy_blocks": float(self._blocks),
            "b200_optimization.primary_timed_output_verification": float(
                self._primary_timed_output_verification_passed
            ),
            "b200_optimization.fresh_post_timing_verification": float(
                self._post_timing_verification_passed
            ),
        }

    def teardown(self) -> None:
        self._extension = None
        self.source = None
        self._output_buffer = None
        self.output = None
        self._benchmark_has_run = False
        self._primary_timed_output_verification_passed = False
        self._post_timing_verification_passed = False
        super().teardown()


class BaselineNativeCopyBenchmark(NativeCopyBenchmarkBase):
    """Existing grid-stride uint4 kernel at its retained launch configuration."""

    implementation_id = "native_uint4_grid_stride"

    def _prepare_implementation(self) -> None:
        self._extension = load_ceiling_extension()
        self._blocks = torch.cuda.get_device_properties(self.device).multi_processor_count * (
            BASELINE_COPY_BLOCKS_PER_SM
        )

    def _run_implementation(self) -> None:
        if self._extension is None or self.source is None or self._output_buffer is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        self._extension.copy_uint4(
            self.source,
            self._output_buffer,
            BASELINE_COPY_THREADS,
            self._blocks,
        )


class OptimizedNativeCopyBenchmark(NativeCopyBenchmarkBase):
    """Cache-global, four-way-unrolled uint4 kernel at the frozen launch."""

    implementation_id = "native_uint4_cg_unroll4"

    def _prepare_implementation(self) -> None:
        self._extension = load_optimization_extension()
        self._blocks = torch.cuda.get_device_properties(self.device).multi_processor_count * (
            SELECTED_COPY_BLOCKS_PER_SM
        )

    def _run_implementation(self) -> None:
        if self._extension is None or self.source is None or self._output_buffer is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        self._extension.copy_uint4_cg_unroll4(
            self.source,
            self._output_buffer,
            SELECTED_COPY_THREADS,
            self._blocks,
        )
