"""
Shared PerformanceCore facade used internally by the engine layer.

This wraps PerformanceCoreBase without starting an HTTP server,
providing low-level access to GPU/system collection and profiling artifacts.

IMPORTANT: For most use cases, prefer the unified engine API:

    from core.engine import get_engine
    
    engine = get_engine()
    engine.gpu.info()          # Instead of get_core().get_gpu_info()
    engine.analyze.bottlenecks()  # Clean domain-organized API
    engine.optimize.recommend()

The engine provides a clean 10-domain API (gpu, system, profile, analyze,
optimize, distributed, inference, benchmark, ai, export) that is consistent
across CLI and MCP interfaces.

This module is primarily for internal use by engine.py and advanced users
who need direct access to the underlying PerformanceCoreBase methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.perf_core_base import PerformanceCoreBase


class PerfCore(PerformanceCoreBase):
    """Non-HTTP performance core used by CLI/MCP."""

    def __init__(self, data_file: Optional[Path] = None, bench_root: Optional[Path] = None):
        super().__init__(data_file=data_file, bench_root=bench_root)

    def set_bench_root(self, bench_root: Path) -> dict:
        return super().set_bench_root(bench_root)


_CORE_SINGLETON: Optional[PerfCore] = None


def get_core(data_file: Optional[Path] = None, bench_root: Optional[Path] = None, refresh: bool = False) -> PerfCore:
    """
    Get a singleton PerfCore instance.

    Args:
        data_file: Optional path to benchmark results.
        bench_root: Optional override for benchmark discovery root.
        refresh: If True, force a new instance (e.g., after changing data_file).
    """
    global _CORE_SINGLETON
    if refresh or _CORE_SINGLETON is None:
        _CORE_SINGLETON = PerfCore(data_file=data_file, bench_root=bench_root)
    return _CORE_SINGLETON
