"""optimized_nvlink_topology_aware.py - single-GPU copy with async launch."""

from __future__ import annotations

from typing import Optional

import torch
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin


def _resolve_device_index(device: torch.device) -> int:
    return 0 if device.index is None else int(device.index)


class OptimizedNvlinkTopologyAwareBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Single-GPU copy using a non-blocking path."""

    def __init__(self):
        super().__init__()
        self.src: Optional[torch.Tensor] = None
        self.dst: Optional[torch.Tensor] = None
        self.host_buffer: Optional[torch.Tensor] = None
        self._src_chunks: list[torch.Tensor] = []
        self._dst_chunks: list[torch.Tensor] = []
        self._host_chunks: list[torch.Tensor] = []
        self._chunk_groups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        # Match baseline: 16M float16 = 32 MB to show topology benefit on same workload
        self.numel = 16 * 1024 * 1024
        self.chunk_elems = self.numel // 8
        self.dtype = torch.float16  # Match baseline dtype
        self.src_id = 0
        self.dst_id = 0
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.numel),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: requires CUDA")
        self.src_id = _resolve_device_index(self.device)
        self.dst_id = self.src_id
        n = self.numel

        self.src = torch.randn(n, device=self.device, dtype=self.dtype)
        self.dst = torch.empty_like(self.src)
        self.host_buffer = torch.empty(n, device="cpu", dtype=self.dtype, pin_memory=True)
        self._src_chunks = list(self.src.split(self.chunk_elems))
        self._dst_chunks = list(self.dst.split(self.chunk_elems))
        self._host_chunks = list(self.host_buffer.split(self.chunk_elems))
        self._chunk_groups = list(zip(self._host_chunks, self._src_chunks, self._dst_chunks, strict=True))
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.src is not None and self.dst is not None and self.host_buffer is not None
        if not self._chunk_groups:
            raise RuntimeError("setup() must initialize chunk views")
        with torch.inference_mode(), self._nvtx_range("optimized_nvlink_topology_aware"):
            for host_chunk, src_chunk, dst_chunk in self._chunk_groups:
                host_chunk.copy_(src_chunk, non_blocking=True)
                dst_chunk.copy_(host_chunk, non_blocking=True)

    def capture_verification_payload(self) -> None:
        if self.src is None or self.dst is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        probe = self.src[: 256 * 256].view(256, 256)
        output = self.dst[: 256 * 256].view(256, 256)
        self._set_verification_payload(
            inputs={"src": probe},
            output=output,
            batch_size=int(probe.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.src = None
        self.dst = None
        self.host_buffer = None
        self._src_chunks = []
        self._dst_chunks = []
        self._host_chunks = []
        self._chunk_groups = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if self.src is None or self.dst is None:
            return "Buffers not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.0, 0.0)


def get_benchmark() -> BaseBenchmark:
    return OptimizedNvlinkTopologyAwareBenchmark()
