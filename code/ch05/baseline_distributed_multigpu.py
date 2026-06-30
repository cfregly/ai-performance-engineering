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
        self.local_sums: List[torch.Tensor] = []
        self.host_sums: List[torch.Tensor] = []
        self._sum_groups: List[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.device_ids: List[int] = []
        self.output: Optional[torch.Tensor] = None
        self._cpu_total: Optional[float] = None
        self._host_total: Optional[torch.Tensor] = None
        self._output_tensor: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._verify_probe_buffer: Optional[torch.Tensor] = None
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
        self.local_sums = [torch.empty(1, device=t.device, dtype=torch.float32) for t in self.data]
        self.host_sums = [
            torch.empty(1, dtype=torch.float32, pin_memory=True)
            for _ in self.data
        ]
        self._sum_groups = list(zip(self.data, self.local_sums, self.host_sums, strict=True))
        self._host_total = torch.empty(1, dtype=torch.float32, pin_memory=True)
        self._output_tensor = torch.empty(1, device=f"cuda:{self.device_ids[0]}", dtype=torch.float32)
        self._verify_output_buffer = torch.empty_like(self._output_tensor)
        self._verify_probe_buffer = torch.empty(256, dtype=self.data[0].dtype, pin_memory=True)
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
        with torch.inference_mode(), self._nvtx_range("baseline_distributed_multigpu"):
            if not self.data:
                raise RuntimeError("setup() must be called before benchmark_fn()")
            if not self.local_sums or not self.host_sums or self._host_total is None:
                raise RuntimeError("Reduction staging buffers not initialized")
            self._host_total.zero_()
            for tensor, local_sum, host_sum in self._sum_groups:
                torch.sum(tensor, dim=0, keepdim=True, out=local_sum)
                host_sum.copy_(local_sum, non_blocking=False)
                self._host_total.add_(host_sum)
            self._cpu_total = float(self._host_total[0])

    def capture_verification_payload(self) -> None:
        if self._cpu_total is None or not self.data:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        if self._output_tensor is None:
            raise RuntimeError("setup() must initialize verification output tensor")
        self._output_tensor[0] = self._cpu_total
        self.output = self._output_tensor
        if self._verify_output_buffer is None:
            raise RuntimeError("setup() must initialize verification output buffer")
        if self._verify_probe_buffer is None:
            raise RuntimeError("setup() must initialize verification input buffer")
        self._verify_output_buffer.copy_(self.output)
        self._verify_probe_buffer.copy_(
            self.data[0][: self._verify_probe_buffer.numel()],
            non_blocking=False,
        )
        self._set_verification_payload(
            inputs={"data_probe": self._verify_probe_buffer},
            output=self._verify_output_buffer,
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
        self.local_sums = []
        self.host_sums = []
        self._sum_groups = []
        self.output = None
        self._cpu_total = None
        self._host_total = None
        self._output_tensor = None
        self._verify_output_buffer = None
        self._verify_probe_buffer = None
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
