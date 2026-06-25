"""baseline_distributed_multigpu.py - Baseline CPU-staged reduction across GPUs."""

from __future__ import annotations

from typing import List, Optional

import torch

from core.benchmark.gpu_requirements import skip_if_insufficient_gpus
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineDistributedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: CPU-staged reduction across all visible GPUs."""

    multi_gpu_required = True
    allowed_benchmark_fn_antipatterns = ("host_transfer",)

    def __init__(self):
        super().__init__()
        self.data: List[torch.Tensor] = []
        self.device_ids: List[int] = []
        self.output: Optional[torch.Tensor] = None
        self._cpu_total: Optional[float] = None
        self.num_elements = 200_000_000
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.num_elements),
        )

    def setup(self) -> None:
        skip_if_insufficient_gpus(2)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.device_ids = list(range(torch.cuda.device_count()))
        self.data = [
            torch.randn(self.num_elements, device=f"cuda:{device_id}")
            for device_id in self.device_ids
        ]
        total_tokens = self.num_elements * len(self.device_ids)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(len(self.device_ids)),
            tokens_per_iteration=float(total_tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(len(self.device_ids)),
            tokens_per_iteration=float(total_tokens),
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        with self._nvtx_range("baseline_distributed_multigpu"):
            cpu_total = 0.0
            for tensor in self.data:
                cpu_total += float(tensor.sum().cpu())
            self._cpu_total = cpu_total

    def capture_verification_payload(self) -> None:
        if self._cpu_total is None or not self.data:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self.output = torch.tensor(
            [self._cpu_total],
            device=f"cuda:{self.device_ids[0]}",
            dtype=torch.float32,
        )
        probe = self.data[0][:256].detach().cpu()
        self._set_verification_payload(
            inputs={"data_probe": probe},
            output=self.output.detach().clone(),
            batch_size=1,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.data = []
        self.output = None
        self._cpu_total = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            multi_gpu_required=True,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        from ch05.metrics_common import compute_multi_gpu_reduction_metrics

        return compute_multi_gpu_reduction_metrics(
            num_elements_per_gpu=self.num_elements,
            device_count=len(self.device_ids) if self.device_ids else 0,
            uses_cpu_staging=True,
        )

    def validate_result(self) -> Optional[str]:
        if not self.data:
            return "Data not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineDistributedBenchmark()
