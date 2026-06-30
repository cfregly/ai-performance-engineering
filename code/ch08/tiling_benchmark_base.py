"""Shared utilities for ch08 tiling benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.utils import scalar_tensor_to_float
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.utils.extension_loader_template import load_cuda_extension


class TilingBenchmarkBase(VerificationPayloadMixin, BaseBenchmark):
    """Base class for the Chapter 8 matmul bridge-comparison tiling pairs."""

    extension_name = "ch08_tiling_kernels"
    kernel_source = Path(__file__).with_name("tiling_kernels.cu")
    extra_cuda_cflags = ["-O3", "--use_fast_math", "-lineinfo"]
    extra_ldflags = ["-lcublas"]
    tensor_dtype = torch.float32
    output_tolerance = (0.1, 1.0)

    nvtx_label: str = "tiling"
    matrix_rows: int = 4096
    matrix_cols: int = 4096
    shared_dim: int = 4096
    # Keep enough repeated kernel work inside one timed call that the bridge
    # comparison pair reflects the kernel delta instead of Python launch overhead.
    inner_iterations: int = 12

    def __init__(self) -> None:
        super().__init__()
        self.extension = None
        self.matrix_a: Optional[torch.Tensor] = None
        self.matrix_b: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._inner_iteration_range = range(self.inner_iterations)
        self.register_workload_metadata(requests_per_iteration=float(self.inner_iterations))

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA required for Chapter 8 tiling benchmarks")

    # --------------------------------------------------------------------- #
    # Benchmark lifecycle
    # --------------------------------------------------------------------- #
    def setup(self) -> None:
        """Compile/load the CUDA extension, allocate tensors, and warm up."""
        self._load_extension()

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.matrix_a = torch.randn(
            self.matrix_rows,
            self.shared_dim,
            device=self.device,
            dtype=self.tensor_dtype,
        )
        self.matrix_b = torch.randn(
            self.shared_dim,
            self.matrix_cols,
            device=self.device,
            dtype=self.tensor_dtype,
        )
        self.output = None
        self._output_buffer = torch.empty(
            self.matrix_rows,
            self.matrix_cols,
            device=self.device,
            dtype=self.tensor_dtype,
        )
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Run the core kernel with NVTX labeling."""
        if self._output_buffer is None:
            raise RuntimeError("setup() must initialize the output buffer")
        self.output = self._output_buffer

        with torch.inference_mode(), self._nvtx_range(self.nvtx_label):
            for _ in self._inner_iteration_range:
                self._invoke_kernel()

    def capture_verification_payload(self) -> None:
        """Register verification payload once after timing."""
        if self.matrix_a is None or self.matrix_b is None or self.output is None:
            raise RuntimeError("capture_verification_payload() requires initialized tensors")
        self._set_verification_payload(
            inputs={"matrix_a": self.matrix_a, "matrix_b": self.matrix_b},
            output=self.output.detach(),
            batch_size=self.matrix_rows,
            parameter_count=int(self.shared_dim * self.matrix_cols),
            precision_flags={
                "fp16": self.tensor_dtype == torch.float16,
                "bf16": self.tensor_dtype == torch.bfloat16,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=self.output_tolerance,
        )

    def teardown(self) -> None:
        """Release GPU memory."""
        self.matrix_a = None
        self.matrix_b = None
        self.output = None
        self._output_buffer = None
        torch.cuda.empty_cache()

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _invoke_kernel(self) -> None:
        """Call into the CUDA extension (implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _invoke_kernel()")

    def _validate_correctness(self) -> None:
        """Ensure the CUDA kernel matches torch.matmul for the same inputs."""
        assert self.matrix_a is not None
        assert self.matrix_b is not None
        assert self.output is not None

        with torch.no_grad():
            reference = torch.matmul(self.matrix_a, self.matrix_b)
        torch.cuda.synchronize()

        max_error = scalar_tensor_to_float(torch.max(torch.abs(self.output - reference)))
        # Large GEMMs accumulate floating-point error quickly; tolerate small
        # absolute differences that stem from reordering in the tiled kernel.
        if max_error > 1e-1:
            raise RuntimeError(
                f"Tiling kernel validation failed (max error={max_error:.4f})"
            )

    # --------------------------------------------------------------------- #
    # Benchmark configuration
    # --------------------------------------------------------------------- #
    def get_config(self) -> BenchmarkConfig:
        """Use fewer iterations because each kernel is compute-heavy."""
        return BenchmarkConfig(
            iterations=12,
            warmup=5,
        )

    def validate_result(self) -> Optional[str]:
        """Verify tensors are present before the harness records results."""
        if self.extension is None:
            return "CUDA extension not loaded"
        if self.matrix_a is None or self.matrix_b is None:
            return "Input matrices not initialized"
        if self.output is None:
            return "Output buffer not initialized"
        return None

    # ------------------------------------------------------------------ #
    # Extension loading (allow subclasses to override)
    # ------------------------------------------------------------------ #
    def _load_extension(self) -> None:
        """Compile/load the default CUDA extension."""
        self.extension = load_cuda_extension(
            extension_name=self.extension_name,
            cuda_source_file=str(self.kernel_source),
            extra_cuda_cflags=self.extra_cuda_cflags,
            extra_ldflags=self.extra_ldflags,
        )

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return super().get_output_tolerance()

    def get_custom_metrics(self) -> Optional[dict]:
        """Return tiling optimization metrics for roofline analysis."""
        M, K, N = self.matrix_rows, self.shared_dim, self.matrix_cols
        flops = 2.0 * M * K * N * self.inner_iterations  # MAD operations
        bytes_transferred = (M * K + K * N + M * N) * 4.0 * self.inner_iterations  # float32
        return {
            f"{self.nvtx_label}.matrix_m": float(M),
            f"{self.nvtx_label}.matrix_k": float(K),
            f"{self.nvtx_label}.matrix_n": float(N),
            f"{self.nvtx_label}.flops": flops,
            f"{self.nvtx_label}.bytes_transferred": bytes_transferred,
            f"{self.nvtx_label}.arithmetic_intensity": flops / bytes_transferred if bytes_transferred > 0 else 0.0,
            "story.comparison_pair": 1.0,
            "story.chapter_native_exemplar": 0.0,
            "story.bridge_to_ch09": 1.0,
        }

    def get_optimization_goal(self) -> str:
        """Keep the Chapter 8 bridge-comparison tiling pairs out of speed gating."""
        return "comparison"
