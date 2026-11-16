"""
Lightweight wrappers around analysis.advanced_analysis with safe fallbacks.

These helpers keep the CLI/MCP from duplicating import/try/except logic.
"""

from __future__ import annotations

from typing import Any, Dict


def _wrap(func_name: str, *args, **kwargs) -> Dict[str, Any]:
    try:
        import core.analysis.advanced_analysis as aa

        func = getattr(aa, func_name)
        return func(*args, **kwargs)
    except ImportError as e:
        return {"error": f"advanced_analysis module not available: {e}"}
    except Exception as e:
        return {"error": str(e)}


def cpu_memory_analysis() -> Dict[str, Any]:
    return _wrap("get_cpu_memory_analysis")


def system_parameters() -> Dict[str, Any]:
    return _wrap("get_system_parameters")


def container_limits() -> Dict[str, Any]:
    return _wrap("get_container_limits")


def warp_divergence(code: str = "") -> Dict[str, Any]:
    if not code:
        code = "// Provide code via query param or body"
    return _wrap("analyze_warp_divergence", code)


def bank_conflicts(stride: int = 1, element_size: int = 4) -> Dict[str, Any]:
    return _wrap("analyze_bank_conflicts", stride, element_size)


def memory_access(stride: int = 1, element_size: int = 4) -> Dict[str, Any]:
    return _wrap("analyze_memory_access", stride, element_size)


def auto_tuning(kernel_type: str = "matmul", max_configs: int = 50) -> Dict[str, Any]:
    return _wrap("run_auto_tuning", kernel_type, max_configs)


def full_system_analysis() -> Dict[str, Any]:
    return _wrap("get_full_system_analysis")


def energy_efficiency(gpu: str, power_limit: int | None = None) -> Dict[str, Any]:
    return _wrap("analyze_energy_efficiency", gpu, power_limit)


def multi_gpu_scaling(gpus: int, nvlink: bool, workload: str) -> Dict[str, Any]:
    return _wrap("estimate_multi_gpu_efficiency", gpus, nvlink, workload)


def predict_hardware_scaling(from_gpu: str, to_gpu: str, workload: str) -> Dict[str, Any]:
    return _wrap("predict_hardware_scaling", from_gpu, to_gpu, workload)
