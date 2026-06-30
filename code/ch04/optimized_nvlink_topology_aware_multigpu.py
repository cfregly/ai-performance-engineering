"""optimized_nvlink_topology_aware_multigpu.py

Optimized NVLink benchmark that selects peer-accessible pairs and uses P2P copies.
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import torch

from core.benchmark.gpu_requirements import require_min_gpus
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin


class OptimizedNvlinkTopologyAwareBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Topology-aware P2P copies across all visible GPUs."""

    multi_gpu_required = True

    def __init__(self):
        super().__init__()
        self.device_ids: List[int] = []
        self.pairs: List[Tuple[int, int]] = []
        self.src: List[torch.Tensor] = []
        self.dst: List[torch.Tensor] = []
        self.streams: List[torch.cuda.Stream] = []
        self._copy_groups: List[Tuple[torch.cuda.Stream, int, torch.Tensor, torch.Tensor]] = []
        self._wait_groups: List[Tuple[torch.cuda.Stream, int]] = []
        self.output: Optional[torch.Tensor] = None
        self.numel = 32 * 1024 * 1024  # 64 MB
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.numel),
        )

    def _sync_all(self) -> None:
        for device_id in self.device_ids:
            torch.cuda.synchronize(device_id)

    @staticmethod
    def _build_peer_pairs(device_ids: List[int]) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        for src_id in device_ids:
            dst_id = None
            for candidate in device_ids:
                if candidate == src_id:
                    continue
                if torch.cuda.can_device_access_peer(src_id, candidate):
                    dst_id = candidate
                    break
            if dst_id is None:
                dst_id = device_ids[(device_ids.index(src_id) + 1) % len(device_ids)]
            pairs.append((src_id, dst_id))
        return pairs

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        require_min_gpus(2, "optimized_nvlink_topology_aware_multigpu.py")

        self.device_ids = list(range(torch.cuda.device_count()))
        self.pairs = self._build_peer_pairs(self.device_ids)

        self.src = [
            torch.randn(self.numel, device=f"cuda:{src}", dtype=torch.float16)
            for src, _ in self.pairs
        ]
        self.dst = [
            torch.empty(self.numel, device=f"cuda:{dst}", dtype=torch.float16)
            for _, dst in self.pairs
        ]
        self.streams = [torch.cuda.Stream(device=dst) for _, dst in self.pairs]
        self._copy_groups = [
            (stream, dst_id, src, dst)
            for stream, (_, dst_id), src, dst in zip(self.streams, self.pairs, self.src, self.dst, strict=True)
        ]
        self._wait_groups = [
            (stream, dst_id)
            for stream, (_, dst_id) in zip(self.streams, self.pairs, strict=True)
        ]

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
        with torch.inference_mode(), self._nvtx_range("optimized_nvlink_topology_aware_multigpu"):
            for stream, dst_id, src, dst in self._copy_groups:
                with torch.cuda.device(dst_id), torch.cuda.stream(stream):
                    dst.copy_(src, non_blocking=True)
            for stream, dst_id in self._wait_groups:
                with torch.cuda.device(dst_id):
                    torch.cuda.current_stream(dst_id).wait_stream(stream)
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
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.src = []
        self.dst = []
        self.streams = []
        self._copy_groups = []
        self._wait_groups = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5, multi_gpu_required=True)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_memory_transfer_metrics
        bytes_transferred = float(self.numel * 2 * len(self.pairs))
        return compute_memory_transfer_metrics(
            bytes_transferred=bytes_transferred,
            elapsed_ms=getattr(self, "_last_elapsed_ms", None),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if not self.src:
            return "Buffers not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        return (0.0, 0.0)

    def get_custom_streams(self) -> List[torch.cuda.Stream]:
        return self.streams


def get_benchmark() -> BaseBenchmark:
    return OptimizedNvlinkTopologyAwareBenchmark()
