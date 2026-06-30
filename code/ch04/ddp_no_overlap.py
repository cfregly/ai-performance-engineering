"""ddp_no_overlap.py - Baseline DDP step without communication overlap.

This benchmark now requires a real multi-GPU torchrun launch. It no longer
publishes substitute one-rank communication under the DDP overlap name.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from core.benchmark.gpu_requirements import require_min_gpus
from ch04.distributed_helper import run_main_with_skip_status, setup_single_gpu_env

from typing import Optional

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range


class MultiLayerNet(nn.Module):
    """Multi-layer network for benchmarking."""
    
    def __init__(self, size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu_(self.fc1(x))
        x = torch.relu_(self.fc2(x))
        return self.fc3(x)


class BaselineNoOverlapBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """DDP baseline with explicit post-backward synchronization."""
    multi_gpu_required = True

    def __init__(self):
        super().__init__()
        self.model = None
        self.optimizer = None
        self.data = None
        self.target = None
        self.output = None
        self.rank = 0
        self.world_size = 1
        self.initialized = False
        self.batch_size = 128
        self.hidden_size = 1024
        self._enable_nvtx = False
        self._payload_parameter_count = 0
        self._model_parameters: tuple[nn.Parameter, ...] = ()
        tokens = self.batch_size * self.hidden_size
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup the distributed baseline without any overlap optimization."""
        require_min_gpus(2, "ddp_no_overlap.py")
        self.rank, self.world_size, self.local_rank = setup_single_gpu_env(
            "ddp_no_overlap",
            min_world_size=2,
        )
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", device_id=self.local_rank)
            self.initialized = True

        torch.manual_seed(42)
        model = MultiLayerNet(self.hidden_size).to(self.device)
        self.model = model
        self._model_parameters = tuple(self.model.parameters())
        self.optimizer = optim.SGD(self._model_parameters, lr=0.01)
        self._payload_parameter_count = sum(p.numel() for p in self._model_parameters)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False

        self.data = torch.randn(self.batch_size, self.hidden_size, device=self.device)
        self.target = torch.randn(self.batch_size, 1, device=self.device)
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark a no-overlap step with synchronous gradient all-reduce."""
        with nvtx_range("no_overlap", enable=self._enable_nvtx):
            output = self.model(self.data)
            loss = nn.functional.mse_loss(output, self.target)
            loss.backward()
            for param in self._model_parameters:
                if param.grad is None:
                    continue
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.mul_(1.0 / self.world_size)
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.output = output.detach_()

    def capture_verification_payload(self) -> None:
        if self.data is None or self.target is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"data": self.data, "target": self.target},
            output=self.output,
            batch_size=int(self.batch_size),
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
        )

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        if dist.is_initialized() and self.initialized:
            dist.destroy_process_group()
        self.model = None
        self.optimizer = None
        self.data = None
        self.target = None
        self._model_parameters = ()
        torch.cuda.empty_cache()
        self._config = None
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=max(torch.cuda.device_count(), 1),
            iterations=10,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
            multi_gpu_required=True,
        )
    
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
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.data is None:
            return "Data tensor not initialized"
        if self.target is None:
            return "Target tensor not initialized"
        # Check that model can produce valid output
        try:
            with torch.no_grad():
                test_output = self.model(self.data)
                if test_output.shape[0] != 128:
                    return f"Output batch size mismatch: expected 128, got {test_output.shape[0]}"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)

    def _prepare_verification_payload(self) -> None:
        if hasattr(self, "_subprocess_verify_output"):
            return
        self.setup()
        try:
            self.benchmark_fn()
            self.capture_verification_payload()
            self._subprocess_verify_output = self.get_verify_output()
            self._subprocess_output_tolerance = self.get_output_tolerance()
            self._subprocess_input_signature = self.get_input_signature()
        finally:
            self.teardown()

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        return TorchrunLaunchSpec(
            script_path=Path(__file__).resolve(),
            script_args=[],
            multi_gpu_required=True,
            name="baseline_no_overlap",
            config_arg_map={
                "iterations": "--iterations",
                "warmup": "--warmup",
            },
        )


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineNoOverlapBenchmark()


def _parse_args():
    parser = argparse.ArgumentParser(description="DDP baseline without communication overlap.")
    parser.add_argument("--iterations", type=int, default=20, help="Number of measurement iterations.")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations.")
    parser.add_argument("--enable-profiling", action="store_true", help="Enable profiling for the run.")
    parser.add_argument("--enable-memory-tracking", action="store_true", help="Track GPU memory usage.")
    return parser.parse_args()


def _run_worker(iterations: int, warmup: int) -> None:
    bench = BaselineNoOverlapBenchmark()
    bench.setup()
    try:
        for _ in range(max(warmup, 0)):
            bench.benchmark_fn()

        if torch.cuda.is_available():
            torch.cuda.synchronize(bench.device)
        if dist.is_initialized():
            dist.barrier()

        start = time.perf_counter()
        for _ in range(max(iterations, 0)):
            bench.benchmark_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize(bench.device)
        if dist.is_initialized():
            dist.barrier()
        elapsed = time.perf_counter() - start

        if bench.rank == 0 and iterations > 0:
            tokens_per_iter = float(bench.batch_size * bench.hidden_size)
            tokens_per_s = tokens_per_iter * (iterations / max(elapsed, 1e-9))
            print(f"rank0 tokens/s: {tokens_per_s:.2f} tokens/s")
            print(f"rank0 time_per_iter_ms: {(elapsed / iterations) * 1000.0:.3f}")
    finally:
        bench.teardown()


def main() -> None:
    args = _parse_args()
    _run_worker(args.iterations, args.warmup)


if __name__ == "__main__":
    raise SystemExit(run_main_with_skip_status(main))
