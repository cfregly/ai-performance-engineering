"""nvls_collectives.py - NVLS-style NCCL collectives placeholder.

This benchmark checks for multi-GPU availability and, when possible, exercises a
small all-reduce with NCCL. If the environment lacks multiple GPUs or torchrun,
it cleanly reports SKIPPED so the harness stays green.
"""

from __future__ import annotations

import os

from core.common.device_utils import resolve_local_rank
from typing import Optional

import torch
import torch.distributed as dist

from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402
from core.benchmark.verification_mixin import VerificationPayloadMixin


class NVLSCollectivesBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Tiny NCCL all-reduce meant to mirror the doc's NVLS target."""

    def __init__(self) -> None:
        super().__init__()
        self.tensor: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._initialized = False
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(32 * 32 * 4),
        )
        self._enable_nvtx = False
        self._empty_iteration_result = {}

    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: NVLS collectives require >=2 GPUs")
        if not dist.is_available():
            raise RuntimeError("SKIPPED: torch.distributed not available")
        if not dist.is_initialized():
            if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
                raise RuntimeError("SKIPPED: launch with torchrun to enable NCCL NVLS demo")
            local_rank = resolve_local_rank()
            torch.cuda.set_device(local_rank)
            dist.init_process_group("nccl", device_id=local_rank)

        os.environ.setdefault("NCCL_NVLS_ENABLE", "1")
        os.environ.setdefault("NCCL_ALGO", "Tree,Ring,NVLS")
        os.environ.setdefault("NCCL_COLLNET_ENABLE", "1")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.tensor = torch.randn(32, 32, device=self.device, dtype=torch.float32)
        self._verify_output_buffer = torch.empty_like(self.tensor)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self._initialized = True

    def benchmark_fn(self) -> Optional[dict]:
        if not self._initialized or self.tensor is None:
            raise RuntimeError("SKIPPED: NVLS benchmark not initialized")

        with nvtx_range("nvls_allreduce", enable=self._enable_nvtx):
            dist.all_reduce(self.tensor)
        return self._empty_iteration_result

    def capture_verification_payload(self) -> None:
        if (
            not self._initialized
            or self.tensor is None
            or self._verify_output_buffer is None
        ):
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.tensor)
        self._set_verification_payload(
            inputs={"tensor": self.tensor},
            output=self._verify_output_buffer,
            batch_size=int(self.tensor.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-5, 1e-5),
        )

    def teardown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        self.tensor = None
        self._verify_output_buffer = None
        self._initialized = False
        super().teardown()

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
    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (1e-5, 1e-5)


def get_benchmark() -> BaseBenchmark:
    return NVLSCollectivesBenchmark()
