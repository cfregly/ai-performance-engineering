"""Shared benchmark bases for the block scaling lab."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.block_scaling.block_scaling_common import (
    BLOCK_SCALING_SOURCE_URL,
    BlockScalingConfig,
    BlockScalingProblem,
    build_problem,
    load_lab_config_from_env,
    theoretical_flops,
    verification_inputs,
    verification_output_slice,
)


class BlockScalingBenchmarkBase(VerificationPayloadMixin, BaseBenchmark):
    """Common harness integration for the software and hardware block-scaling paths."""

    benchmark_path = ""
    nvtx_label = ""
    compile_hardware = False
    def __init__(self) -> None:
        super().__init__()
        self.config: BlockScalingConfig = load_lab_config_from_env()
        self.problem: Optional[BlockScalingProblem] = None
        self.last_run_completed = False
        self.verification_summary: Optional[dict[str, float]] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.config.l),
            custom_units_per_iteration=theoretical_flops(self.config),
            custom_unit_name="FLOPs",
        )

    def setup(self) -> None:
        self.problem = build_problem(self.config, compile_hardware=self.compile_hardware)
        self.last_run_completed = False
        self.verification_summary = None
        self._post_setup()

    def _post_setup(self) -> None:
        """Hook for subclasses that need additional setup-time work."""

    def _require_problem(self) -> BlockScalingProblem:
        if self.problem is None:
            raise RuntimeError("Block scaling problem is not initialized")
        return self.problem

    def benchmark_fn(self) -> None:
        problem = self._require_problem()
        with torch.inference_mode():
            with self._nvtx_range(self.nvtx_label):
                self._run_problem(problem)
        self.last_run_completed = True

    def _run_problem(self, problem: BlockScalingProblem) -> None:
        raise NotImplementedError

    def _build_verification_output(self, problem: BlockScalingProblem) -> torch.Tensor:
        raise NotImplementedError

    def teardown(self) -> None:
        self.problem = None
        self.last_run_completed = False
        self.verification_summary = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            measurement_timeout_seconds=120,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        problem_metrics = {
            "block_scaling.path": self.benchmark_path,
            "block_scaling.mnkl": ",".join(str(x) for x in self.config.mnkl),
            "block_scaling.mma_tiler_mn": ",".join(str(x) for x in self.config.mma_tiler_mn),
            "block_scaling.cluster_shape_mn": ",".join(
                str(x) for x in self.config.cluster_shape_mn
            ),
            "block_scaling.sf_vec_size": float(self.config.sf_vec_size),
            "block_scaling.source": BLOCK_SCALING_SOURCE_URL,
        }
        if self.verification_summary is not None:
            problem_metrics["block_scaling.max_abs_error"] = self.verification_summary["max_abs_error"]
            problem_metrics["block_scaling.mean_abs_error"] = self.verification_summary["mean_abs_error"]
        return problem_metrics

    def capture_verification_payload(self) -> None:
        problem = self._require_problem()
        output = self._build_verification_output(problem)
        self._set_verification_payload(
            inputs=verification_inputs(self.config),
            output=output,
            batch_size=self.config.l,
            parameter_count=0,
            precision_flags={
                **self._verification_precision_flags(),
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1e-2, self.config.tolerance),
            signature_overrides={"quantization_mode": "nvfp4_block_scaling"},
        )

    def validate_result(self) -> Optional[str]:
        if self.problem is None:
            return "Block scaling problem is not initialized"
        if not self.last_run_completed:
            return f"{self.benchmark_path} benchmark did not complete"
        return None

    def _verification_precision_flags(self) -> dict[str, bool]:
        return {}


class BaselineBlockScalingBenchmarkBase(BlockScalingBenchmarkBase):
    """Materialize scale factors in BF16 and then call matmul."""

    benchmark_path = "software_dequant_bf16"
    nvtx_label = "block_scaling_software_dequant"
    def _run_problem(self, problem: BlockScalingProblem) -> None:
        problem.run_software()

    def _build_verification_output(self, problem: BlockScalingProblem) -> torch.Tensor:
        return verification_output_slice(problem.run_software())

    def _verification_precision_flags(self) -> dict[str, bool]:
        return {
            "fp16": self.config.software_dtype == torch.float16,
            "bf16": self.config.software_dtype == torch.bfloat16,
        }


class OptimizedBlockScalingBenchmarkBase(BlockScalingBenchmarkBase):
    """Compile once and measure only Blackwell's hardware blockscaled path."""

    benchmark_path = "hardware_blockscaled_cutlass"
    nvtx_label = "block_scaling_hardware_blockscaled"
    compile_hardware = True

    def __init__(self) -> None:
        super().__init__()
        self._replay_graph: Optional[torch.cuda.CUDAGraph] = None

    def _post_setup(self) -> None:
        """Capture the compiled blockscaled launch in a CUDA graph for replay.

        The CuTeDSL host invoke costs ~8us per call; inside the harness's
        per-iteration timing frame (event bracket + full-device sync) that
        host time is exposed on the GPU timeline. Replaying a pre-captured
        graph (~2us launch) removes it. Inputs are static device buffers
        (identical to the eager path), so every replay re-executes the full
        GEMM on the same data; the verification path still calls
        problem.run_hardware() eagerly and is unaffected. Set
        AISP_BLOCK_SCALING_DISABLE_GRAPH=1 to fall back to eager launches.
        """
        import os as _os

        if _os.getenv("AISP_BLOCK_SCALING_DISABLE_GRAPH", "").strip().lower() in {
            "1", "true", "yes", "on",
        }:
            self._replay_graph = None
            return
        problem = self._require_problem()
        if problem.compiled_gemm is None:
            self._replay_graph = None
            return
        try:
            import cuda.bindings.driver as _cuda_drv

            capture_stream = torch.cuda.Stream()
            with torch.cuda.stream(capture_stream):
                problem.compiled_gemm(
                    problem.a_tensor,
                    problem.b_tensor,
                    problem.sfa_tensor,
                    problem.sfb_tensor,
                    problem.c_tensor,
                    _cuda_drv.CUstream(capture_stream.cuda_stream),
                )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                current = torch.cuda.current_stream()
                problem.compiled_gemm(
                    problem.a_tensor,
                    problem.b_tensor,
                    problem.sfa_tensor,
                    problem.sfb_tensor,
                    problem.c_tensor,
                    _cuda_drv.CUstream(current.cuda_stream),
                )
            torch.cuda.synchronize()
            self._replay_graph = graph
        except Exception:
            self._replay_graph = None

    def _run_problem(self, problem: BlockScalingProblem) -> None:
        if self._replay_graph is not None:
            self._replay_graph.replay()
        else:
            problem.run_hardware()

    def teardown(self) -> None:
        self._replay_graph = None
        super().teardown()

    def get_custom_metrics(self) -> Optional[dict]:
        metrics = super().get_custom_metrics() or {}
        metrics["block_scaling.graph_replay"] = float(self._replay_graph is not None)
        return metrics

    def _build_verification_output(self, problem: BlockScalingProblem) -> torch.Tensor:
        problem.run_hardware()
        self._synchronize()
        return verification_output_slice(problem.extract_hardware_output())

    def _verification_precision_flags(self) -> dict[str, bool]:
        return {"fp16": False, "bf16": True}
