"""optimized_nvfp4_trtllm.py - NVFP4/TRT-LLM integration path.

If TensorRT-LLM is present and an engine path is provided via TRT_LLM_ENGINE,
run a small inference; otherwise fall back to a Transformer Engine NVFP4 demo
or report SKIPPED with a clear message.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class NVFP4TRTLLMBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.linear: Optional[nn.Linear] = None
        self.inputs: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(tokens_per_iteration=0.0)
        self._stack_available = False
        self.output: Optional[torch.Tensor] = None
        self.graph = None
        self._trt_runner = None
        self._verification_payload = None
        self._enable_nvtx = False
        self._empty_iteration_result = {}
        self._payload_parameter_count = 0
        self._fp8_autocast = None

    def setup(self) -> None:
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False

        # TensorRT-LLM path first, with optional CUDA Graph capture.
        engine_path = os.getenv("TRT_LLM_ENGINE")
        trt_exc: Optional[Exception] = None
        try:
            from tensorrt_llm.runtime import ModelRunner  # type: ignore
            if engine_path is not None:
                self._stack_available = True
                self._trt_runner = ModelRunner.from_engine(engine_path)
                self.inputs = torch.randint(0, 1000, (1, 32), device=self.device, dtype=torch.int32)
                try:
                    stream = torch.cuda.Stream()
                    torch.cuda.synchronize()
                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph, stream=stream):
                        self._trt_runner.generate(self.inputs)  # type: ignore[attr-defined]
                except Exception:
                    self.graph = None
                return
        except Exception as exc:  # pragma: no cover - optional dependency
            trt_exc = exc
        trt_msg = f"{trt_exc}" if trt_exc is not None else "TensorRT-LLM unavailable"

        try:
            import transformer_engine.pytorch as te  # type: ignore
            from transformer_engine.pytorch import fp8_autocast  # type: ignore
            self._stack_available = True
            self._te = te
            self._fp8_autocast = fp8_autocast
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(f"SKIPPED: NVFP4 stack not available ({trt_msg})") from exc

        self.linear = nn.Linear(1024, 1024, bias=False).to(self.device).to(torch.float16)
        self._payload_parameter_count = sum(p.numel() for p in self.linear.parameters())
        self.inputs = torch.randn(32, 1024, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[dict]:
        if not self._stack_available:
            raise RuntimeError("SKIPPED: NVFP4 stack not available")

        # TensorRT-LLM path if runner exists.
        if self._trt_runner is not None and self.inputs is not None:
            with nvtx_range("nvfp4_trtllm_engine", enable=self._enable_nvtx):
                outputs = self._trt_runner.generate(self.inputs)  # type: ignore[attr-defined]
                try:
                    if isinstance(outputs, dict):
                        first = outputs.get("output_ids")
                    else:
                        first = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                    if not isinstance(first, torch.Tensor):
                        raise TypeError(f"expected Tensor output, got {type(first).__name__}")
                    self.output = first
                except Exception as exc:
                    raise RuntimeError(
                        f"FAIL FAST: TRT-LLM generate returned an unsupported output payload "
                        f"({type(exc).__name__}: {exc})"
                    ) from exc
            if self.output is None:
                raise RuntimeError("TRT-LLM generate did not produce output")
            return self._empty_iteration_result

        if self.linear is None or self.inputs is None:
            raise RuntimeError("SKIPPED: NVFP4 linear model not initialized")

        with nvtx_range("nvfp4_te_fp8", enable=self._enable_nvtx):
            try:
                if self._fp8_autocast is None:
                    raise RuntimeError("Transformer Engine FP8 autocast not initialized")
                with self._fp8_autocast():
                    self.output = self.linear(self.inputs)
            except Exception as exc:
                raise RuntimeError(
                    f"FAIL FAST: Transformer Engine FP8 path failed in nvfp4_trtllm_tool "
                    f"({type(exc).__name__}: {exc})"
                ) from exc
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output")
        return self._empty_iteration_result

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"inputs": self.inputs},
            output=self.output,
            batch_size=self.inputs.shape[0],
            parameter_count=self._payload_parameter_count,
            precision_flags={"fp16": self.output.dtype == torch.float16, "bf16": self.output.dtype == torch.bfloat16, "fp8": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

def get_benchmark() -> BaseBenchmark:
    return NVFP4TRTLLMBenchmark()
