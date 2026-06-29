"""optimized_dtensor_mesh.py - DTensor mesh setup placeholder.

Creates a small DTensor mesh when the feature is available; otherwise reports
SKIPPED so the docs target stays runnable without exotic dependencies.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class DTensorMeshBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self._workload = WorkloadMetadata(bytes_per_iteration=0.0)
        self.mesh = None
        self.tensor: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._enable_nvtx = False
        self._empty_iteration_result = {}

    def setup(self) -> None:
        try:
            from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"SKIPPED: DTensor not available ({exc})") from exc

        if torch.cuda.device_count() < 1:
            raise RuntimeError("SKIPPED: CUDA device required for DTensor mesh demo")

        devices = list(range(min(2, torch.cuda.device_count())))
        self.mesh = DeviceMesh("cuda", devices)
        local = torch.randn(4, 4, device=f"cuda:{devices[0]}")
        self.tensor = distribute_tensor(local, placements=[Replicate()], device_mesh=self.mesh)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False

    def benchmark_fn(self) -> Optional[dict]:
        if self.mesh is None or self.tensor is None:
            raise RuntimeError("SKIPPED: DTensor mesh not initialized")

        with nvtx_range("dtensor_mesh", enable=self._enable_nvtx):
            self.output = (self.tensor * 2).redistribute(self.mesh, placements=self.tensor.placements)
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        input_local = self.tensor.to_local() if hasattr(self.tensor, "to_local") else self.tensor
        output_local = self.output.to_local() if hasattr(self.output, "to_local") else self.output
        self.output = output_local
        self._payload_input_local = input_local
        return self._empty_iteration_result

    def capture_verification_payload(self) -> None:
        input_local = self._payload_input_local
        self._set_verification_payload(
            inputs={"input": input_local},
            output=self.output.detach().float().clone(),
            batch_size=int(input_local.shape[0]) if input_local is not None else 1,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=None,
            reduced_precision_time_ms=getattr(self, '_last_elapsed_ms', None),
            precision_type="fp8",
        )

def get_benchmark() -> BaseBenchmark:
    return DTensorMeshBenchmark()
