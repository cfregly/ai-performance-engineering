"""baseline_nvlink_multigpu.py - Host-staged copies across all visible GPUs."""

from __future__ import annotations

from typing import Optional, List, Tuple

import torch

from core.benchmark.gpu_requirements import require_min_gpus
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin


class BaselineNVLinkBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: PCIe-style host staging for inter-GPU copies."""

    multi_gpu_required = True

    def __init__(self):
        super().__init__()
        self.device_ids: List[int] = []
        self.pairs: List[Tuple[int, int]] = []
        self.src: List[torch.Tensor] = []
        self.dst: List[torch.Tensor] = []
        self.host_buffers: List[torch.Tensor] = []
        self._copy_groups: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.output: Optional[torch.Tensor] = None
        self.numel = 50_000_000
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.numel),
        )

    def _sync_all(self) -> None:
        for device_id in self.device_ids:
            torch.cuda.synchronize(device_id)

    def setup(self) -> None:
        require_min_gpus(2, "baseline_nvlink_multigpu.py")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.device_ids = list(range(torch.cuda.device_count()))
        self.pairs = [(idx, (idx + 1) % len(self.device_ids)) for idx in self.device_ids]

        self.src = [
            torch.randn(self.numel, device=f"cuda:{src}", dtype=torch.float32)
            for src, _ in self.pairs
        ]
        self.dst = [
            torch.empty(self.numel, device=f"cuda:{dst}", dtype=torch.float32)
            for _, dst in self.pairs
        ]
        self.host_buffers = [
            torch.empty(self.numel, device="cpu", dtype=torch.float32)
            for _ in self.pairs
        ]
        self._copy_groups = list(zip(self.host_buffers, self.src, self.dst, strict=True))
        total_tokens = self.numel * len(self.pairs)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(len(self.pairs)),
            tokens_per_iteration=float(total_tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(len(self.pairs)),
            tokens_per_iteration=float(total_tokens),
        )
        self._sync_all()

    def benchmark_fn(self) -> None:
        with torch.inference_mode(), self._nvtx_range("baseline_nvlink_multigpu"):
            for host_buffer, src, dst in self._copy_groups:
                host_buffer.copy_(src, non_blocking=False)
                dst.copy_(host_buffer, non_blocking=False)
        self.output = self.dst[0]

    def capture_verification_payload(self) -> None:
        if not self.src or not self.dst:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        probe = self.src[0][: 256 * 256].view(256, 256)
        output = self.dst[0][: 256 * 256].view(256, 256)
        self._set_verification_payload(
            inputs={"src": probe},
            output=output,
            batch_size=int(probe.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.src = []
        self.dst = []
        self.host_buffers = []
        self._copy_groups = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5, multi_gpu_required=True)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_memory_transfer_metrics
        bytes_transferred = float(self.numel * 4 * 2 * len(self.pairs))
        return compute_memory_transfer_metrics(
            bytes_transferred=bytes_transferred,
            elapsed_ms=getattr(self, "_last_elapsed_ms", None),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if not self.src:
            return "Buffers not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineNVLinkBenchmark()
