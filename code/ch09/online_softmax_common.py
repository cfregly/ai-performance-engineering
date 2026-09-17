"""Isolate the online-normalizer recurrence used by attention kernels."""

import importlib

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


def materialized_softmax(x: torch.Tensor) -> torch.Tensor:
    """Stable baseline exposing the intermediate arrays removed by fusion."""
    if x.ndim != 2 or x.dtype != torch.float32 or x.shape[1] == 0:
        raise ValueError("softmax requires a float32 matrix with nonempty rows")
    shifted = x - x.amax(dim=1, keepdim=True)
    numerator = shifted.exp()
    return numerator / numerator.sum(dim=1, keepdim=True)


def online_softmax(x: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    """Executable CPU/CUDA explanation of the running (maximum, mass) recurrence.

    This is a correctness study, not the optimized GPU benchmark backend.
    A second pass emits the normalized output after the final normalizer is known.
    """
    if x.ndim != 2 or not x.is_floating_point() or x.shape[1] == 0:
        raise ValueError("online softmax requires a floating matrix with nonempty rows")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    maximum = torch.full((x.shape[0], 1), -torch.inf, dtype=x.dtype, device=x.device)
    mass = torch.zeros_like(maximum)
    for start in range(0, x.shape[1], block_size):
        block = x[:, start : start + block_size]
        new_maximum = torch.maximum(maximum, block.amax(dim=1, keepdim=True))
        old_weight = torch.where(torch.isneginf(maximum), 0.0, (maximum - new_maximum).exp())
        block_mass = torch.where(
            torch.isneginf(new_maximum),
            0.0,
            (block - new_maximum).exp().sum(dim=1, keepdim=True),
        )
        mass = mass * old_weight + block_mass
        maximum = new_maximum
    return (x - maximum).exp() / mass


def load_kernels():
    if not torch.cuda.is_available():
        raise RuntimeError("SKIPPED: Triton online softmax require an NVIDIA CUDA GPU")
    try:
        return importlib.import_module("ch09.online_softmax_triton")
    except ModuleNotFoundError as exc:
        if exc.name not in {"triton", "triton.language"}:
            raise
        raise RuntimeError("SKIPPED: Online softmax require Triton") from exc


class OnlineSoftmaxBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self, optimized, rows=128, cols=8193):
        super().__init__()
        if any(type(v) is not int or v <= 0 for v in (rows, cols)):
            raise ValueError("rows and cols must be positive integers")
        self.optimized, self.rows, self.cols = optimized, rows, cols
        self.x = self.output = self.kernels = None
        self.executed = False
        self.register_workload_metadata(
            custom_units_per_iteration=rows * cols, custom_unit_name="normalized_values"
        )

    def setup(self):
        self.executed = False
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: online softmax benchmark pairs require CUDA")
        self.x = torch.randn((self.rows, self.cols), device=self.device, dtype=torch.float32) * 8
        self.output = torch.empty_like(self.x)
        if self.optimized:
            self.kernels = load_kernels()

    def benchmark_fn(self):
        self.executed = False
        if self.x is None:
            raise RuntimeError("setup() must run first")
        if self.optimized:
            self.kernels.online_softmax_rows[(self.rows,)](self.x, self.output, self.cols, 256)
        else:
            self.output = materialized_softmax(self.x)
        self.executed = True

    def capture_verification_payload(self):
        if not self.executed or self.output is None:
            raise RuntimeError("benchmark_fn() must run before verification")
        expected = torch.softmax(self.x.to(torch.float64), dim=1).to(torch.float32)
        torch.testing.assert_close(self.output, expected, rtol=2e-5, atol=2e-7)
        self._set_verification_payload(
            inputs={"x": self.x},
            output=self.output,
            batch_size=self.rows,
            parameter_count=0,
            output_tolerance=(2e-5, 2e-7),
        )

    def get_config(self):
        return BenchmarkConfig(iterations=30, warmup=10)

    def teardown(self):
        self.x = self.output = self.kernels = None
        self.executed = False
