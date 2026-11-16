"""
Shared wrappers around advanced optimization stacking/playbook helpers.

These wrappers centralize imports and fallbacks so CLI/MCP can share
the same implementation without duplicating logic.
"""

from __future__ import annotations

from typing import Any, Dict, List


def get_optimization_stacking(analyzer) -> Dict[str, Any]:
    """Delegate to analyzer for optimization stacking if available."""
    try:
        return analyzer.get_optimization_stacking()
    except Exception as e:
        return {"error": str(e)}


def get_all_optimizations() -> Dict[str, Any]:
    """Return all optimizations and categories from advanced_analysis."""
    try:
        from core.analysis.advanced_analysis import get_all_optimizations, OPTIMIZATION_DATABASE

        return {
            "optimizations": get_all_optimizations(),
            "categories": list(set(opt.category.value for opt in OPTIMIZATION_DATABASE.values())),
            "count": len(OPTIMIZATION_DATABASE),
        }
    except Exception as e:
        return {"error": f"advanced_analysis module not available: {e}"}


def get_optimization_playbooks() -> Dict[str, Any]:
    """Return predefined playbooks from advanced_analysis."""
    try:
        from core.analysis.advanced_analysis import get_all_playbooks

        pbs = get_all_playbooks()
        return {"playbooks": pbs, "count": len(pbs)}
    except Exception as e:
        return {"error": f"advanced_analysis module not available: {e}"}


def calculate_compound_optimization(optimizations: List[str], software_info: Dict[str, Any]) -> Dict[str, Any]:
    """Compute compound effects of multiple optimizations."""
    try:
        from core.analysis.advanced_analysis import CompoundOptimizationCalculator

        hardware = {"features": software_info.get("features", [])}
        calc = CompoundOptimizationCalculator(hardware)
        result = calc.calculate_compound(optimizations)

        return {
            "success": True,
            "optimizations": result.optimizations,
            "combined_speedup": result.combined_speedup,
            "combined_memory_reduction": result.combined_memory_reduction,
            "incremental_gains": [
                {"name": name, "cumulative_speedup": speedup, "cumulative_memory": mem}
                for name, speedup, mem in result.incremental_gains
            ],
            "conflicts": result.conflicts,
            "warnings": result.warnings,
            "code_changes": result.code_changes,
            "difficulty": result.total_difficulty,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_optimal_optimization_stack(target_speedup: float, max_difficulty: str, software_info: Dict[str, Any]) -> Dict[str, Any]:
    """Find optimal optimization stack for target speedup."""
    try:
        from core.analysis.advanced_analysis import CompoundOptimizationCalculator

        features = software_info.get("features", [])
        gpu_cap = software_info.get("compute_capability", "")
        hardware = {"features": features, "compute_capability": gpu_cap}
        calc = CompoundOptimizationCalculator(hardware)
        result = calc.find_optimal_stack(target_speedup, max_difficulty)
        return {
            "success": True,
            "stack": result.stack,
            "expected_speedup": result.expected_speedup,
            "difficulty": result.total_difficulty,
            "warnings": result.warnings,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
