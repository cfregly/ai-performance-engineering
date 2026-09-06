"""Inference CLI commands wired to the unified PerformanceEngine."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from core.engine import get_engine


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _print_result(result: Dict[str, Any], json_output: bool) -> None:
    if json_output:
        print(json.dumps(result, indent=2, default=_json_default))
        return
    print(json.dumps(result, indent=2, default=_json_default))


def _get_single_launch_argv(result: Dict[str, Any]) -> list[str]:
    """Return the producer-provided argv for a directly executable launch."""
    engine = result.get("engine")
    if not isinstance(engine, dict):
        raise ValueError("deployment result is missing structured engine data")
    launch_argv = engine.get("launch_argv")
    if launch_argv is None:
        raise ValueError(
            "generated launch plan contains multiple commands; run the displayed steps "
            "explicitly after review"
        )
    if (
        not isinstance(launch_argv, list)
        or not launch_argv
        or not all(isinstance(argument, str) for argument in launch_argv)
        or not launch_argv[0]
    ):
        raise ValueError("deployment result contains an invalid launch argv")
    return list(launch_argv)


def vllm_config(args: Any) -> int:
    """Generate vLLM configuration."""
    model_size = getattr(args, "model_size", None)
    if model_size is None:
        print("model_size is required (billions of parameters).")
        return 1
    model = getattr(args, "model", None) or "model"
    result = get_engine().inference.vllm_config(
        model=model,
        model_params_b=float(model_size),
        num_gpus=int(getattr(args, "gpus", 1)),
        gpu_memory_gb=float(getattr(args, "gpu_memory_gb", 80.0)),
        target=getattr(args, "target", "throughput"),
        max_seq_length=int(getattr(args, "max_seq_length", 8192)),
        quantization=getattr(args, "quantization", None),
        compare=bool(getattr(args, "compare", False)),
    )
    _print_result(result, getattr(args, "json", False))
    return 0 if result.get("success", True) else 1


def quantize(args: Any) -> int:
    """Quantization recommendations."""
    model_size = getattr(args, "model_size", None)
    result = get_engine().inference.quantization(model_size=model_size)
    _print_result(result, getattr(args, "json", False))
    return 0 if result.get("success", True) else 1


def deploy_config(args: Any) -> int:
    """Generate deployment configuration."""
    model_size = getattr(args, "model_size", None)
    if model_size is None:
        print("model_size is required (billions of parameters).")
        return 1
    model = getattr(args, "model", None) or "model"
    params = {
        "model": model,
        "model_params_b": float(model_size),
        "num_gpus": int(getattr(args, "gpus", 1)),
        "gpu_memory_gb": float(getattr(args, "gpu_memory_gb", 80.0)),
        "goal": getattr(args, "goal", "throughput"),
        "max_seq_length": int(getattr(args, "max_seq_length", 8192)),
    }
    result = get_engine().inference.deploy(params)
    _print_result(result, getattr(args, "json", False))
    return 0 if result.get("success", True) else 1


def estimate(args: Any) -> int:
    """Estimate inference performance."""
    model_size = getattr(args, "model_size", None)
    if model_size is None:
        print("model_size is required (billions of parameters).")
        return 1
    model = getattr(args, "model", None) or "model"
    params = {
        "model": model,
        "model_params_b": float(model_size),
        "num_gpus": int(getattr(args, "gpus", 1)),
        "gpu_memory_gb": float(getattr(args, "gpu_memory_gb", 80.0)),
        "goal": getattr(args, "goal", "throughput"),
        "max_seq_length": int(getattr(args, "max_seq_length", 8192)),
    }
    result = get_engine().inference.estimate(params)
    _print_result(result, getattr(args, "json", False))
    return 0 if result.get("success", True) else 1


def serve(args: Any) -> int:
    """Generate (and optionally run) an inference server command."""
    model_size = getattr(args, "model_size", None)
    if model_size is None:
        print("model_size is required (billions of parameters).")
        return 1
    model = getattr(args, "model", None) or "model"
    params = {
        "model": model,
        "model_params_b": float(model_size),
        "num_gpus": int(getattr(args, "gpus", 1)),
        "gpu_memory_gb": float(getattr(args, "gpu_memory_gb", 80.0)),
        "goal": getattr(args, "goal", "throughput"),
        "max_seq_length": int(getattr(args, "max_seq_length", 8192)),
    }
    result = get_engine().inference.deploy(params)
    if not result.get("success", True):
        _print_result(result, getattr(args, "json", False))
        return 1
    launch_cmd = result.get("launch_command")
    if not launch_cmd:
        print("No launch command available.")
        _print_result(result, getattr(args, "json", False))
        return 1
    if getattr(args, "run", False):
        try:
            launch_argv = _get_single_launch_argv(result)
        except ValueError as exc:
            print(f"Cannot execute launch command safely: {exc}.")
            return 1
        return subprocess.run(launch_argv).returncode
    print(launch_cmd)
    return 0
