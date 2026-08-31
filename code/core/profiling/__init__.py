"""
Profiling Module

Comprehensive GPU profiling with:
- torch.profiler integration
- Memory snapshots and timelines
- Flame graph generation
- CPU/GPU timeline visualization
- HTA (Holistic Trace Analysis) integration
- torch.compile diagnostics
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Keep the convenient package-level API without importing PyTorch (or any of
# the optional profiling stacks) when callers only need a lightweight sibling
# such as ``core.profiling.nsight_systems``.
_LAZY_EXPORTS = {
    "UnifiedProfiler": (".profiler", "UnifiedProfiler"),
    "ProfileSession": (".profiler", "ProfileSession"),
    "MemoryProfiler": (".memory", "MemoryProfiler"),
    "MemorySnapshot": (".memory", "MemorySnapshot"),
    "FlameGraphGenerator": (".flame_graph", "FlameGraphGenerator"),
    "TimelineGenerator": (".timeline", "TimelineGenerator"),
    "HTAAnalyzer": (".hta_integration", "HTAAnalyzer"),
    "TorchProfilerAutomation": (".torch_profiler", "TorchProfilerAutomation"),
    "HTACaptureAutomation": (".hta_capture", "HTACaptureAutomation"),
}

__all__ = [
    "UnifiedProfiler",
    "ProfileSession",
    "MemoryProfiler",
    "MemorySnapshot",
    "FlameGraphGenerator",
    "TimelineGenerator",
    "HTAAnalyzer",
    "TorchProfilerAutomation",
    "HTACaptureAutomation",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
