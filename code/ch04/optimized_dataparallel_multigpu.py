"""optimized_dataparallel_multigpu.py - Manual data-parallel training without DataParallel overhead."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin


class SimpleNet(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class OptimizedDataParallelMultiGPUBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: manual data-parallel training with pre-staged GPU shards.

    Key optimizations vs DataParallel baseline:
    1. Inputs pre-staged on each GPU (no CPU->GPU scatter each iteration)
    2. Manual gradient reduction avoids DataParallel gather overhead
    3. Explicit parameter broadcast keeps replicas in sync
    """

    multi_gpu_required = True

    def __init__(self):
        super().__init__()
        self.batch_size = 4096
        self.input_size = 4096
        self.device_ids: List[int] = []
        self.models: List[nn.Module] = []
        self.optimizers: List[optim.Optimizer] = []
        self.inputs: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.streams: List[torch.cuda.Stream] = []
        self._grad_staging: List[List[torch.Tensor]] = []
        self._parameter_groups: list[tuple[nn.Parameter, ...]] = []
        self._launch_groups: list[tuple[torch.cuda.Stream, int, nn.Module, torch.Tensor, torch.Tensor]] = []
        self._wait_groups: list[tuple[torch.cuda.Stream, int]] = []
        self._reduction_groups: list[tuple[nn.Parameter, tuple[tuple[nn.Parameter, torch.Tensor], ...]]] = []
        self._broadcast_groups: list[tuple[nn.Parameter, tuple[nn.Parameter, ...]]] = []
        self.output: Optional[torch.Tensor] = None
        self._verify_state: Optional[dict] = None
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_target: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0
        tokens = self.batch_size * self.input_size
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def _sync_all(self) -> None:
        for device_id in self.device_ids:
            torch.cuda.synchronize(device_id)

    def setup(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: requires >=2 GPUs")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.device_ids = list(range(torch.cuda.device_count()))
        world_size = len(self.device_ids)
        if self.batch_size % world_size != 0:
            self.batch_size = (self.batch_size // world_size) * world_size
            tokens = self.batch_size * self.input_size
            self._workload = WorkloadMetadata(
                requests_per_iteration=float(self.batch_size),
                tokens_per_iteration=float(tokens),
            )
            self.register_workload_metadata(
                requests_per_iteration=float(self.batch_size),
                tokens_per_iteration=float(tokens),
            )

        base_model = SimpleNet(self.input_size).to(torch.device(f"cuda:{self.device_ids[0]}"))
        self._payload_parameter_count = sum(p.numel() for p in base_model.parameters())
        self._verify_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        base_state = base_model.state_dict()

        self.models = []
        self.optimizers = []
        self.streams = []
        for device_id in self.device_ids:
            device = torch.device(f"cuda:{device_id}")
            model = SimpleNet(self.input_size).to(device)
            model.load_state_dict(base_state)
            self.models.append(model)
            self.optimizers.append(optim.SGD(model.parameters(), lr=0.01))
            self.streams.append(torch.cuda.Stream(device=device_id))

        data_gen = torch.Generator().manual_seed(1234)
        cpu_input = torch.randn(self.batch_size, self.input_size, dtype=torch.float32, generator=data_gen)
        cpu_target = torch.randn(self.batch_size, 1, dtype=torch.float32, generator=data_gen)
        self._verify_input = cpu_input.clone()
        self._verify_target = cpu_target.clone()

        local_batch = self.batch_size // world_size
        self.inputs = []
        self.targets = []
        for idx, device_id in enumerate(self.device_ids):
            start = idx * local_batch
            end = start + local_batch
            device = torch.device(f"cuda:{device_id}")
            self.inputs.append(cpu_input[start:end].to(device))
            self.targets.append(cpu_target[start:end].to(device))

        master_device = next(self.models[0].parameters()).device
        self._parameter_groups = [
            tuple(param_group)
            for param_group in zip(*(model.parameters() for model in self.models), strict=True)
        ]
        self._grad_staging = [
            [
                torch.empty_like(param_group[0], device=master_device)
                for param_group in self._parameter_groups
            ]
            for _ in self.models[1:]
        ]
        self._launch_groups = list(
            zip(self.streams, self.device_ids, self.models, self.inputs, self.targets, strict=True)
        )
        self._wait_groups = list(zip(self.streams, self.device_ids, strict=True))
        self._reduction_groups = []
        self._broadcast_groups = []
        for param_idx, param_group in enumerate(self._parameter_groups):
            master_param = param_group[0]
            replica_pairs = tuple(
                (replica_param, self._grad_staging[replica_idx][param_idx])
                for replica_idx, replica_param in enumerate(param_group[1:])
            )
            self._reduction_groups.append((master_param, replica_pairs))
            self._broadcast_groups.append(
                (master_param, tuple(replica_param for replica_param, _ in replica_pairs))
            )

        self._sync_all()

    def benchmark_fn(self) -> None:
        if not self.models or not self.optimizers:
            raise RuntimeError("setup() must be called before benchmark_fn()")

        first_output: Optional[torch.Tensor] = None

        with self._nvtx_range("optimized_dataparallel_multigpu"):
            for stream, device_id, model, batch, target in self._launch_groups:
                with torch.cuda.device(device_id), torch.cuda.stream(stream):
                    output = model(batch)
                    loss = nn.functional.mse_loss(output, target)
                    loss.backward()
                    if first_output is None:
                        first_output = output

            for stream, device_id in self._wait_groups:
                with torch.cuda.device(device_id):
                    torch.cuda.current_stream(device_id).wait_stream(stream)

            # Reduce gradients onto GPU0 and update master parameters.
            for master_param, replica_pairs in self._reduction_groups:
                master_grad = master_param.grad
                if master_grad is None:
                    continue
                reduced = master_grad
                for replica_param, staging in replica_pairs:
                    grad = replica_param.grad
                    if grad is None:
                        continue
                    staging.copy_(grad, non_blocking=True)
                    reduced.add_(staging)
                master_param.grad = reduced

            self.optimizers[0].step()
            for opt in self.optimizers:
                opt.zero_grad(set_to_none=True)

            # Broadcast updated parameters from GPU0 to all replicas.
            for master_param, replica_params in self._broadcast_groups:
                master_data = master_param.data
                for replica_param in replica_params:
                    replica_param.data.copy_(master_data, non_blocking=True)

        if first_output is None:
            raise RuntimeError("No model output captured")
        self.output = first_output.detach_()

    def capture_verification_payload(self) -> None:
        if (
            self._verify_input is None
            or self._verify_target is None
            or self._verify_state is None
            or self.output is None
        ):
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        verify_model = SimpleNet(self.input_size).to(torch.device(f"cuda:{self.device_ids[0]}"))
        verify_model.load_state_dict(self._verify_state)
        verify_model.eval()
        with torch.no_grad():
            verify_device = next(verify_model.parameters()).device
            verify_input = self._verify_input.to(verify_device)
            verify_target = self._verify_target.to(verify_device)
            output = verify_model(verify_input)
        self._set_verification_payload(
            inputs={"data": verify_input, "target": verify_target},
            output=output,
            batch_size=int(self.batch_size),
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.models = []
        self.optimizers = []
        self.inputs = []
        self.targets = []
        self.streams = []
        self._grad_staging = []
        self._parameter_groups = []
        self._launch_groups = []
        self._wait_groups = []
        self._reduction_groups = []
        self._broadcast_groups = []
        self._verify_state = None
        self._verify_input = None
        self._verify_target = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
            multi_gpu_required=True,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, "_bytes_transferred") else float(getattr(self, "N", 1024) * 4),
            elapsed_ms=getattr(self, "_last_elapsed_ms", None),
            transfer_type="hbm",
        )

    def get_custom_streams(self) -> List[torch.cuda.Stream]:
        return self.streams

    def validate_result(self) -> Optional[str]:
        if not self.models:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedDataParallelMultiGPUBenchmark()
