"""Example benchmark patterns showing different approaches.

This file demonstrates:
1. Simple benchmark with @export_benchmark (RECOMMENDED)
2. Deep class hierarchy with @export_benchmark
3. Manual get_benchmark() with @register_benchmark
4. Multiple benchmark classes in one module
5. Using create_benchmark_factory() for parameterized benchmarks

Contract:
- `get_benchmark()` is the required interface for discovery, `compare.py`, and
  `cli.aisp bench run`.
- Discoverable benchmark modules should not define module-level `__main__`
  entrypoints; run them through chapter `compare.py` scripts or the shared CLI.
- If a file needs a standalone `__main__`, keep it as a non-benchmark tool or
  utility script instead of a harness-discoverable benchmark module.

Run validation:
    python -m core.benchmark.registry --validate

Run discoverable benchmarks through chapter `compare.py` helpers or
``python -m cli.aisp bench run``.
Avoid local import-path mutation bootstrapping in benchmark/example files.
"""

from __future__ import annotations

import torch
from typing import Optional

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.registry import (
    export_benchmark,
    register_benchmark,
    create_benchmark_factory,
    require_get_benchmark,
)


# =============================================================================
# Pattern 1: Simple benchmark with @export_benchmark (RECOMMENDED)
# =============================================================================
# Use this when you have ONE benchmark class per file.
# The decorator auto-generates get_benchmark().

# @export_benchmark  # Uncomment to auto-generate get_benchmark()
class SimpleBenchmark(BaseBenchmark):
    """Simple benchmark example - most common case."""
    
    def __init__(self):
        super().__init__()
        self.data = None
    
    def setup(self) -> None:
        self.data = torch.randn(1024, 1024, device=self.device)
    
    def benchmark_fn(self) -> None:
        _ = self.data @ self.data
    
    def teardown(self) -> None:
        self.data = None
        torch.cuda.empty_cache()


# =============================================================================
# Pattern 2: Deep class hierarchy with @export_benchmark
# =============================================================================
# Use when you have a category of benchmarks sharing common functionality.

class InferenceBenchmarkBase(BaseBenchmark):
    """Base class for inference benchmarks with common setup patterns.
    
    This intermediate class provides:
    - Common initialization logic
    - Shared utility methods
    - Default configurations
    
    Subclasses only need to implement benchmark_fn().
    """
    
    def __init__(self, batch_size: int = 32, seq_len: int = 512):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.model = None
        self.data = None
    
    def _create_model(self) -> torch.nn.Module:
        """Override in subclasses to create specific model."""
        return torch.nn.Linear(512, 512)
    
    def setup(self) -> None:
        """Common setup: create model and data."""
        self.model = self._create_model().to(self.device)
        self.data = torch.randn(
            self.batch_size, self.seq_len, 512,
            device=self.device
        )
        # Warmup
        with torch.no_grad():
            _ = self.model(self.data)
        torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Common teardown: clean up resources."""
        self.model = None
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Default config for inference benchmarks."""
        return BenchmarkConfig(iterations=50, warmup=10)
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.seq_len),
        )


# @export_benchmark  # Uncomment to auto-generate get_benchmark()
class MLPInferenceBenchmark(InferenceBenchmarkBase):
    """Concrete benchmark inheriting from intermediate base.
    
    Only needs to implement _create_model() and benchmark_fn().
    Everything else is inherited!
    """
    
    def _create_model(self) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 512),
        )
    
    def benchmark_fn(self) -> None:
        with torch.no_grad():
            _ = self.model(self.data)


class TransformerInferenceBenchmark(InferenceBenchmarkBase):
    """Another concrete benchmark using the same base."""
    
    def _create_model(self) -> torch.nn.Module:
        return torch.nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048, batch_first=True
        )
    
    def benchmark_fn(self) -> None:
        with torch.no_grad():
            _ = self.model(self.data)


# =============================================================================
# Pattern 3: Multiple benchmark classes in one module
# =============================================================================
# Use @register_benchmark and manually define get_benchmark() to choose
# which class to export.

@register_benchmark
class VariantA(BaseBenchmark):
    """First variant."""
    
    def setup(self) -> None:
        pass
    
    def benchmark_fn(self) -> None:
        pass


@register_benchmark
class VariantB(BaseBenchmark):
    """Second variant."""
    
    def setup(self) -> None:
        pass
    
    def benchmark_fn(self) -> None:
        pass


# When you have multiple benchmarks, manually choose which to export:
# def get_benchmark() -> BaseBenchmark:
#     return VariantA()  # or VariantB()


# =============================================================================
# Pattern 4: Parameterized benchmark with factory
# =============================================================================
# Use when you need different configurations of the same benchmark.

class ParameterizedBenchmark(BaseBenchmark):
    """Benchmark that accepts configuration parameters."""
    
    def __init__(self, size: int = 1024, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.size = size
        self.dtype = dtype
        self.data = None
    
    def setup(self) -> None:
        self.data = torch.randn(self.size, self.size, device=self.device, dtype=self.dtype)
    
    def benchmark_fn(self) -> None:
        _ = self.data @ self.data


# Create factory with custom parameters:
# get_benchmark = create_benchmark_factory(ParameterizedBenchmark, size=2048, dtype=torch.float16)


# =============================================================================
# Pattern 5: Using require_get_benchmark() for validation
# =============================================================================
# Add this at the end of your file to fail-fast if get_benchmark() is missing.

# For this example file, we manually define get_benchmark():
def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return SimpleBenchmark()


# This validates that get_benchmark() exists at import time:
require_get_benchmark(__name__)


if __name__ == "__main__":
    print("Benchmark Examples")
    print("=" * 60)
    print()
    print("Available patterns:")
    print("  1. @export_benchmark - Auto-generates get_benchmark()")
    print("  2. Deep hierarchy - Intermediate base classes")
    print("  3. @register_benchmark - Manual get_benchmark()")
    print("  4. create_benchmark_factory() - Parameterized benchmarks")
    print("  5. require_get_benchmark() - Validation at import")
    print()
    
    # Test the benchmark
    bm = get_benchmark()
    print(f"Benchmark class: {type(bm).__name__}")
    print(f"Device: {bm.device}")
