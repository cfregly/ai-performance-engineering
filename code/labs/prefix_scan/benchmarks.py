"""Compare native scan, hierarchical Triton scan and CUDA look-back."""

import importlib

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.prefix_scan.reference import inclusive_scan


def load_kernels():
    if not torch.cuda.is_available():
        raise RuntimeError("SKIPPED: Triton prefix scan require an NVIDIA CUDA GPU")
    try:
        return importlib.import_module("labs.prefix_scan.kernels")
    except ModuleNotFoundError as exc:
        if exc.name not in {"triton", "triton.language"}:
            raise
        raise RuntimeError("SKIPPED: Prefix scan require Triton") from exc


class ScanPlan:
    """Hierarchical scan: local prefixes, recursive totals, then carry propagation.

    Separate launches provide the global dependency ordering. This deliberately
    does not claim to implement the paper's single-pass decoupled look-back.
    """

    def __init__(self, x: torch.Tensor, block_size: int = 1024):
        if x.device.type != "cuda":
            raise RuntimeError("SKIPPED: ScanPlan requires CUDA")
        if x.ndim != 1 or x.dtype != torch.int32 or not x.is_contiguous() or not x.numel():
            raise ValueError("scan requires a nonempty contiguous int32 vector")
        if block_size < 2 or block_size > 4096 or block_size & (block_size - 1):
            raise ValueError("block_size must be a power of two in [2, 4096]")
        self.kernels = load_kernels()
        self.block_size = block_size
        self.levels = []
        current = x
        while True:
            blocks = (current.numel() + block_size - 1) // block_size
            output = torch.empty_like(current)
            totals = torch.empty(blocks, dtype=x.dtype, device=x.device)
            self.levels.append((current, output, totals))
            if blocks == 1:
                break
            current = totals
        self.output = self.levels[0][1]

    def run(self) -> torch.Tensor:
        for source, output, totals in self.levels:
            self.kernels.scan_tiles[(totals.numel(),)](
                source,
                output,
                totals,
                source.numel(),
                self.block_size,
            )
        for index in range(len(self.levels) - 2, -1, -1):
            source, output, totals = self.levels[index]
            self.kernels.add_tile_carries[(totals.numel(),)](
                output,
                self.levels[index + 1][1],
                source.numel(),
                self.block_size,
            )
        return self.output


class PrefixScanBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self, optimized, numel=1_048_579, *, lookback=False):
        super().__init__()
        if type(numel) is not int or numel <= 0:
            raise ValueError("numel must be a positive integer")
        if lookback and not optimized:
            raise ValueError("lookback is an optimized scan backend")
        self.optimized, self.numel, self.lookback = optimized, numel, lookback
        self.x = self.output = self.plan = None
        self.executed = False
        self.register_workload_metadata(
            custom_units_per_iteration=numel, custom_unit_name="scanned_values"
        )

    def setup(self):
        self.executed = False
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: prefix scan benchmark pairs require CUDA")
        self.x = torch.randint(-3, 4, (self.numel,), dtype=torch.int32, device=self.device)
        self.output = torch.empty_like(self.x)
        if self.optimized:
            if self.lookback:
                from labs.prefix_scan.lookback import LookbackPlan

                self.plan = LookbackPlan(self.x)
            else:
                self.plan = ScanPlan(self.x)

    def benchmark_fn(self):
        self.executed = False
        if self.x is None:
            raise RuntimeError("setup() must run first")
        self.output = self.plan.run() if self.optimized else inclusive_scan(self.x, self.output)
        self.executed = True

    def capture_verification_payload(self):
        if not self.executed or self.output is None:
            raise RuntimeError("benchmark_fn() must run before verification")
        expected = torch.cumsum(self.x.to(torch.int64), dim=0).to(torch.int32)
        torch.testing.assert_close(self.output, expected, rtol=0, atol=0)
        self._set_verification_payload(
            inputs={"x": self.x},
            output=self.output,
            batch_size=1,
            parameter_count=0,
            output_tolerance=(0.0, 0.0),
        )

    def get_config(self):
        return BenchmarkConfig(iterations=30, warmup=10)

    def teardown(self):
        self.x = self.output = self.plan = None
        self.executed = False
