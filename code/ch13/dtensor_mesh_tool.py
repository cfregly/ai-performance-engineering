"""Small one- or two-rank CUDA DTensor mesh tool.

DTensor meshes contain process ranks, so this tool must run under ``torchrun``
even for a one-GPU smoke test. For a two-GPU run from the ``code/`` directory::

    python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 \
      --rdzv_backend static --rdzv_endpoint 127.0.0.1:29403 \
      -m ch13.dtensor_mesh_tool
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

import torch
import torch.distributed as dist

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


_TORCHRUN_ENV_KEYS = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
_TORCHRUN_EXAMPLE = (
    "python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 "
    "--rdzv_backend static --rdzv_endpoint 127.0.0.1:29403 "
    "-m ch13.dtensor_mesh_tool"
)


def _resolve_torchrun_context(
    environ: Optional[Mapping[str, str]] = None,
    *,
    cuda_device_count: Optional[int] = None,
) -> tuple[int, int, int]:
    """Return ``(rank, world_size, local_rank)`` for the supported mesh."""
    env = os.environ if environ is None else environ
    missing = [key for key in _TORCHRUN_ENV_KEYS if not str(env.get(key, "")).strip()]
    if missing:
        raise RuntimeError(
            "DTensor mesh tool requires torchrun so every mesh rank has a process; "
            f"missing {', '.join(missing)}. Run: {_TORCHRUN_EXAMPLE}"
        )

    values: dict[str, int] = {}
    for key in _TORCHRUN_ENV_KEYS:
        raw = str(env[key]).strip()
        try:
            values[key] = int(raw)
        except ValueError as exc:
            raise RuntimeError(f"Invalid {key}={raw!r}; expected an integer from torchrun.") from exc

    rank = values["RANK"]
    world_size = values["WORLD_SIZE"]
    local_rank = values["LOCAL_RANK"]
    if world_size not in (1, 2):
        raise RuntimeError(
            f"DTensor mesh tool supports one or two ranks, got WORLD_SIZE={world_size}."
        )
    if rank < 0 or rank >= world_size:
        raise RuntimeError(f"RANK={rank} must satisfy 0 <= RANK < WORLD_SIZE={world_size}.")

    device_count = torch.cuda.device_count() if cuda_device_count is None else int(cuda_device_count)
    if local_rank < 0 or local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is invalid for cuda.device_count()={device_count}."
        )
    return rank, world_size, local_rank


class DTensorMeshBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self._workload = WorkloadMetadata(bytes_per_iteration=0.0)
        self.mesh = None
        self.tensor: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._payload_input_local: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._enable_nvtx = False
        self._empty_iteration_result = {}
        self._to_local = None
        self._owns_process_group = False

    def setup(self) -> None:
        try:
            from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"SKIPPED: DTensor not available ({exc})") from exc

        if torch.cuda.device_count() < 1:
            raise RuntimeError("SKIPPED: CUDA device required for DTensor mesh demo")

        rank, world_size, local_rank = _resolve_torchrun_context()
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        self.device = device
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", device_id=device)
            self._owns_process_group = True
        if dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise RuntimeError(
                "Initialized process group does not match torchrun RANK/WORLD_SIZE "
                f"({dist.get_rank()}/{dist.get_world_size()} versus {rank}/{world_size})."
            )

        mesh_ranks = list(range(world_size))
        self.mesh = DeviceMesh("cuda", mesh_ranks)
        local = torch.randn(4, 4, device=device)
        self._verify_output_buffer = torch.empty_like(local, dtype=torch.float32)
        self.tensor = distribute_tensor(local, placements=[Replicate()], device_mesh=self.mesh)
        self._to_local = getattr(type(self.tensor), "to_local", None)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False

    def benchmark_fn(self) -> Optional[dict]:
        if self.mesh is None or self.tensor is None:
            raise RuntimeError("SKIPPED: DTensor mesh not initialized")

        with nvtx_range("dtensor_mesh", enable=self._enable_nvtx):
            self.output = (self.tensor * 2).redistribute(self.mesh, placements=self.tensor.placements)
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        to_local = self._to_local
        input_local = to_local(self.tensor) if to_local is not None else self.tensor
        output_local = to_local(self.output) if to_local is not None else self.output
        self.output = output_local
        self._payload_input_local = input_local
        return self._empty_iteration_result

    def capture_verification_payload(self) -> None:
        input_local = self._payload_input_local
        if input_local is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": input_local},
            output=self._verify_output_buffer,
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

    def teardown(self) -> None:
        self.mesh = None
        self.tensor = None
        self.output = None
        self._payload_input_local = None
        self._verify_output_buffer = None
        self._to_local = None
        super().teardown()
        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
        self._owns_process_group = False


def get_benchmark() -> BaseBenchmark:
    return DTensorMeshBenchmark()


def main() -> None:
    """Run the chapter tool through the shared standalone benchmark helper."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13 DTensor mesh tool")
    _resolve_torchrun_context()

    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(
        get_benchmark,
        iterations=10,
        warmup=5,
        name="ch13_dtensor_mesh_tool",
    )


if __name__ == "__main__":
    main()
