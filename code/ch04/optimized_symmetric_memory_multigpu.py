"""Optimized CUDA-backend symmetric-memory ring with an NCCL control plane."""

from __future__ import annotations

import torch

from ch04.nccl_blackwell_config import (
    configure_nccl_for_blackwell,
    configure_nccl_for_gb200_gb300,
    configure_nccl_for_multigpu,
    detect_b200_multigpu_topology,
)
from ch04.nvshmem_child_result import (
    NVSHMEM_CHILD_RESULT_CALLBACK,
    NVSHMEMChildResultMixin,
)
from ch04.symmetric_memory_example import (
    SYMMETRIC_RING_NVTX_RANGE,
    SYMMETRIC_RING_TRANSPORT_BACKEND,
    TRADITIONAL_RING_TRANSPORT_BACKEND,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from core.optimization.symmetric_memory_patch import symmetric_memory_backend_supported


def _configure_blackwell_nccl() -> dict[str, str]:
    try:
        topo = detect_b200_multigpu_topology()
    except Exception:
        return configure_nccl_for_blackwell(verbose=False)

    if topo.get("has_grace_cpu"):
        return configure_nccl_for_gb200_gb300(verbose=False)
    if topo.get("num_gpus", 0) >= 2 and topo.get("is_b200_multigpu"):
        return configure_nccl_for_multigpu(
            num_gpus=topo.get("num_gpus", 2),
            verbose=False,
        )
    return configure_nccl_for_blackwell(verbose=False)


class OptimizedSymmetricMemoryMultiGPU(
    NVSHMEMChildResultMixin,
    VerificationPayloadMixin,
    BaseBenchmark,
):
    multi_gpu_required = True
    preferred_ncu_replay_mode = "app-range"

    def __init__(self) -> None:
        super().__init__()
        self.register_workload_metadata(requests_per_iteration=1.0)
        self._benchmark_ready = False

    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: symmetric_memory requires >=2 GPUs")
        if not symmetric_memory_backend_supported(SYMMETRIC_RING_TRANSPORT_BACKEND):
            raise RuntimeError(
                "SKIPPED: symmetric_memory requires the CUDA symmetric-memory backend API"
            )
        _configure_blackwell_nccl()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._benchmark_ready = True

    def benchmark_fn(self) -> None:
        if not self._benchmark_ready:
            raise RuntimeError("setup() must run before benchmark_fn()")

    def capture_verification_payload(self) -> None:
        self.require_nvshmem_child_result()

    def teardown(self) -> None:
        self._benchmark_ready = False
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=torch.cuda.device_count(),
            iterations=1,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=300,
            nsys_nvtx_include=[SYMMETRIC_RING_NVTX_RANGE],
            ncu_replay_mode="app-range",
            ncu_replay_mode_override=True,
        )

    def get_torchrun_spec(
        self,
        config: BenchmarkConfig | None = None,
    ) -> TorchrunLaunchSpec:
        effective_config = config or self.get_config()
        if int(effective_config.nnodes or 1) != 1:
            raise RuntimeError(
                "Symmetric-memory ring child-result transport requires nnodes == 1"
            )
        world_size = int(
            effective_config.nproc_per_node or max(2, torch.cuda.device_count())
        )
        result_env = self.prepare_nvshmem_child_result(
            variant="optimized",
            workload="symmetric-ring",
            world_size=world_size,
            iterations=400,
            configuration={
                "benchmark_mode": "symmetric",
                "tensor_bytes": 2097152,
                "iterations": 400,
                "process_group_backend": TRADITIONAL_RING_TRANSPORT_BACKEND,
                "requested_transport_backend": SYMMETRIC_RING_TRANSPORT_BACKEND,
                "observed_transport_backend": SYMMETRIC_RING_TRANSPORT_BACKEND,
            },
        )
        return TorchrunLaunchSpec(
            module_name="core.harness.benchmark_worker",
            script_args=[
                "--module",
                "ch04.nvshmem_worker",
                "--callable",
                "main",
                "--",
                "--workload",
                "symmetric-ring",
                "--variant",
                "optimized",
                "--benchmark-mode",
                "symmetric",
                "--tensor-bytes",
                "2097152",
                "--iterations",
                "400",
            ],
            env={**result_env, **_configure_blackwell_nccl()},
            multi_gpu_required=True,
            name="optimized_symmetric_memory_multigpu",
            result_callback=NVSHMEM_CHILD_RESULT_CALLBACK,
            timing_source="rank0_time_per_iter_ms",
            timing_iterations_per_sample=400,
        )

    def get_custom_metrics(self) -> dict | None:
        """Return domain-specific metrics using the standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics

        return compute_memory_transfer_metrics(
            bytes_transferred=(
                self._bytes_transferred
                if hasattr(self, "_bytes_transferred")
                else float(getattr(self, "N", 1024) * 4)
            ),
            elapsed_ms=getattr(self, "_last_elapsed_ms", None),
            transfer_type="hbm",
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedSymmetricMemoryMultiGPU()
