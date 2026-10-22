#!/usr/bin/env python3
"""Controlled CUDA microbenchmark for Dynamo nested graph-break handling.

This worker intentionally runs one case, flag arm, and seed per fresh process.
It is a compiler microbenchmark.  It does not measure serving, a model, or
model quality.  A separate controller is responsible for process timeouts,
GPU ownership, physical-GPU guards, cold cache directories, and stderr capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import struct
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from counter_capture import prepare_counter_capture
from profile_capture import capture_untimed_profile


SCHEMA = "nested-graph-breaks-b200-benchmark/v1"
BENCHMARK_ID = "torch-dynamo-nested-graph-breaks-b200-20261020"
SCOPE = (
    "controlled compiler microbenchmark; no serving, model-throughput, or "
    "model-quality claim"
)
INPUT_ELEMENTS = 1 << 20
INPUT_DTYPE = "torch.float32"
PRE_OPS_PER_FRAME = 4
POST_OPS_PER_FRAME = 4
COLD_CALLS = 10
WARMUP_CALLS = 25
LATENCY_TRIALS = 100
THROUGHPUT_BLOCKS = 10
THROUGHPUT_CALLS_PER_BLOCK = 50
COMPILE_BACKEND = "inductor"
COMPILE_MODE = "reduce-overhead"
COMPILE_FULLGRAPH = False
RTOL = 1e-4
ATOL = 1e-5

CASE_SPECS: Dict[str, Dict[str, Any]] = {
    "no_break_depth4": {"depth": 4, "explicit_graph_breaks": 0},
    "nested_break_depth4": {"depth": 4, "explicit_graph_breaks": 1},
    "nested_break_depth8": {"depth": 8, "explicit_graph_breaks": 1},
}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=tuple(CASE_SPECS))
    parser.add_argument("--enabled", required=True, type=int, choices=(0, 1))
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2, 3))
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Fresh output directory; result.json is written inside it.",
    )
    parser.add_argument(
        "--expected-gpu-uuid",
        required=True,
        help="Exact UUID also used as the sole CUDA_VISIBLE_DEVICES entry.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture an untimed CPU+CUDA Chrome trace (seed 0, depth-8 only).",
    )
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    try:
        with path.open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return repr(value)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _normalize_uuid(value: str) -> str:
    return value.strip().upper()


def _validate_controller_environment(args: argparse.Namespace) -> Dict[str, Any]:
    """Validate non-CUDA preconditions before any CUDA API is used."""

    expected = args.expected_gpu_uuid.strip()
    if not expected:
        raise RuntimeError("--expected-gpu-uuid cannot be empty")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be set by the controller")
    visible_entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if len(visible_entries) != 1:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain exactly one controller-selected UUID"
        )
    if _normalize_uuid(visible_entries[0]) != _normalize_uuid(expected):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES does not match --expected-gpu-uuid: "
            f"{visible_entries[0]!r} != {expected!r}"
        )
    if not _normalize_uuid(expected).startswith("GPU-"):
        raise RuntimeError("--expected-gpu-uuid must be a full GPU-... UUID")

    cache_dirs: Dict[str, Dict[str, Any]] = {}
    resolved: List[str] = []
    for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        raw = os.environ.get(name)
        if not raw:
            raise RuntimeError(f"{name} must be set to a fresh directory")
        path = Path(raw).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        entries = sorted(child.name for child in path.iterdir())
        if entries:
            raise RuntimeError(f"{name} is not cold/empty: {path}")
        cache_dirs[name] = {
            "path": str(path),
            "initially_empty": True,
            "initial_entries": entries,
        }
        resolved.append(str(path))
    if resolved[0] == resolved[1]:
        raise RuntimeError("TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR must differ")

    if args.profile and (args.case != "nested_break_depth8" or args.seed != 0):
        raise RuntimeError("--profile is restricted to seed 0 nested_break_depth8")

    return {
        "expected_gpu_uuid": expected,
        "cuda_visible_devices": visible,
        "cache_dirs": cache_dirs,
    }


def _build_workload(torch: Any, case_name: str) -> Tuple[Any, str, str]:
    """Create uniquely named, depth-specific Python frames for one workload."""

    spec = CASE_SPECS[case_name]
    depth = int(spec["depth"])
    has_break = bool(spec["explicit_graph_breaks"])
    prefix = f"ngb_{case_name}"
    lines: List[str] = []

    # Emit the innermost frame first so every outer global reference resolves.
    for level in reversed(range(depth)):
        function_name = f"{prefix}_frame_{level}"
        lines.append(f"def {function_name}(y, residual):")
        lines.append("    y = torch.sin(y)")
        lines.append("    y = y * 0.5")
        lines.append("    y = y + residual")
        lines.append("    y = torch.cos(y)")
        if level + 1 < depth:
            child_name = f"{prefix}_frame_{level + 1}"
            lines.append(f"    y = {child_name}(y, residual)")
        elif has_break:
            # This is the only explicit graph break in a break-bearing case.
            lines.append("    torch._dynamo.graph_break()")
        lines.append("    y = torch.sin(y)")
        lines.append("    y = y * 0.25")
        lines.append("    y = y + residual")
        lines.append("    y = torch.cos(y)")
        lines.append("    return y")
        lines.append("")

    entry_name = f"{prefix}_entry"
    lines.append(f"def {entry_name}(x):")
    lines.append(f"    return {prefix}_frame_0(x, x)")
    lines.append("")
    source = "\n".join(lines)
    namespace: Dict[str, Any] = {"torch": torch}
    code = compile(source, f"<nested_graph_benchmark:{case_name}>", "exec")
    exec(code, namespace, namespace)
    return namespace[entry_name], source, _sha256_bytes(source.encode("utf-8"))


def _sample_output(torch: Any, output: Any) -> Dict[str, Any]:
    # One gathered transfer avoids one synchronization per selected element.
    indices = [
        0,
        1,
        17,
        1024,
        INPUT_ELEMENTS // 3,
        INPUT_ELEMENTS // 2,
        INPUT_ELEMENTS - 1,
    ]
    index_tensor = torch.tensor(indices, device=output.device, dtype=torch.long)
    values = output.detach().flatten().index_select(0, index_tensor).cpu().tolist()
    packed = b"".join(struct.pack("!d", float(value)) for value in values)
    return {
        "indices": indices,
        "values": [float(value) for value in values],
        "sha256": _sha256_bytes(packed),
        "hash_encoding": "network-order float64 values at selected indices",
    }


def _correctness(torch: Any, reference: Any, actual: Any) -> Dict[str, Any]:
    shape_ok = tuple(reference.shape) == tuple(actual.shape)
    dtype_ok = reference.dtype == actual.dtype
    finite = bool(torch.isfinite(actual).all().item())
    difference = (actual - reference).abs()
    max_abs = float(difference.max().item())
    denominator = reference.abs().clamp_min(ATOL)
    max_rel = float((difference / denominator).max().item())
    error: Optional[str] = None
    try:
        torch.testing.assert_close(actual, reference, rtol=RTOL, atol=ATOL)
        close = True
    except AssertionError as exc:
        close = False
        error = str(exc)
    passed = bool(shape_ok and dtype_ok and finite and close)
    return {
        "pass": passed,
        "assert_close_pass": close,
        "assert_close_error": error,
        "rtol": RTOL,
        "atol": ATOL,
        "finite": finite,
        "shape_match": shape_ok,
        "dtype_match": dtype_ok,
        "reference_shape": list(reference.shape),
        "compiled_shape": list(actual.shape),
        "reference_dtype": str(reference.dtype),
        "compiled_dtype": str(actual.dtype),
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "sample_output": _sample_output(torch, actual),
    }


def _timed_invocation(torch: Any, compiled: Any, x: Any) -> Tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    output = compiled(x)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return output, elapsed_ms


def _compiler_counter_snapshot(torch: Any) -> Dict[str, int]:
    """Flatten numeric compiler counters without invoking or retracing the model."""

    try:
        counters = torch._dynamo.utils.counters
    except AttributeError:
        return {}
    flattened: Dict[str, int] = {}
    for category, values in counters.items():
        if not isinstance(values, Mapping):
            continue
        for name, value in values.items():
            if isinstance(value, int) and not isinstance(value, bool):
                flattened[f"{category}.{name}"] = int(value)
    return flattened


def _counter_delta(
    before: Mapping[str, int], after: Mapping[str, int]
) -> Dict[str, int]:
    keys = set(before) | set(after)
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(keys)
        if int(after.get(key, 0)) != int(before.get(key, 0))
    }


def _run_timing(
    torch: Any, eager: Any, compiled: Any, x: Any, counter_capture: Any
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with torch.no_grad():
        reference = eager(x)
        torch.cuda.synchronize()

        first_output, first_ms = _timed_invocation(torch, compiled, x)
        counter_capture.observe("after_first_compiled_call")
        first_correctness = _correctness(torch, reference, first_output)
        del first_output
        if not first_correctness["pass"]:
            raise RuntimeError("compiled output failed eager-reference correctness")

        cold_raw_ms = [first_ms]
        for _ in range(COLD_CALLS - 1):
            cold_output, elapsed_ms = _timed_invocation(torch, compiled, x)
            cold_raw_ms.append(elapsed_ms)
            del cold_output

        for _ in range(WARMUP_CALLS):
            warmup_output = compiled(x)
            del warmup_output
        torch.cuda.synchronize()
        before_latency_counters = _compiler_counter_snapshot(torch)

        latency_raw_ms: List[float] = []
        for _ in range(LATENCY_TRIALS):
            latency_output, elapsed_ms = _timed_invocation(torch, compiled, x)
            latency_raw_ms.append(elapsed_ms)
            del latency_output
        after_latency_counters = _compiler_counter_snapshot(torch)

        throughput_blocks: List[Dict[str, float]] = []
        for _ in range(THROUGHPUT_BLOCKS):
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            for _ in range(THROUGHPUT_CALLS_PER_BLOCK):
                throughput_output = compiled(x)
                del throughput_output
            torch.cuda.synchronize()
            total_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            throughput_blocks.append(
                {
                    "total_ms": total_ms,
                    "per_call_ms": total_ms / THROUGHPUT_CALLS_PER_BLOCK,
                }
            )
        after_throughput_counters = _compiler_counter_snapshot(torch)

        # This verification is intentionally outside both steady timing and
        # compiler-counter stability windows.  Release the output immediately
        # after collecting the retained correctness evidence.
        final_output = compiled(x)
        torch.cuda.synchronize()
        post_steady_correctness = _correctness(torch, reference, final_output)
        del final_output
        if not post_steady_correctness["pass"]:
            raise RuntimeError("post-steady compiled output failed eager correctness")
        del reference

    latency_counter_delta = _counter_delta(
        before_latency_counters, after_latency_counters
    )
    throughput_counter_delta = _counter_delta(
        after_latency_counters, after_throughput_counters
    )

    metrics = {
        "first_compiled_call_host_ms": first_ms,
        "cold_first_10": {
            "raw_host_ms": cold_raw_ms,
            "total_host_ms": sum(cold_raw_ms),
            "mean_host_ms": statistics.fmean(cold_raw_ms),
            "median_host_ms": statistics.median(cold_raw_ms),
        },
        "warmup_calls": WARMUP_CALLS,
        "steady_invocation_latency": {
            "raw_host_ms": latency_raw_ms,
            "median_host_ms": statistics.median(latency_raw_ms),
            "mean_host_ms": statistics.fmean(latency_raw_ms),
            "p95_host_ms": _percentile(latency_raw_ms, 95.0),
        },
        "batched_throughput": {
            "calls_per_block": THROUGHPUT_CALLS_PER_BLOCK,
            "blocks": throughput_blocks,
            "median_per_call_ms": statistics.median(
                block["per_call_ms"] for block in throughput_blocks
            ),
        },
        "compiler_counter_stability": {
            "before_steady_latency": before_latency_counters,
            "after_steady_latency": after_latency_counters,
            "after_batched_throughput": after_throughput_counters,
            "steady_latency_delta": latency_counter_delta,
            "batched_throughput_delta": throughput_counter_delta,
            "stable": not latency_counter_delta and not throughput_counter_delta,
            "validation_rule": (
                "any compiler-counter delta during either timed steady window "
                "invalidates warmed measurements"
            ),
        },
        "acceptance_roles": {
            "steady_invocation_latency": "predeclared steady acceptance metric",
            "batched_throughput": "reported descriptive corroboration only",
            "cold_first_10": "predeclared startup-only acceptance metric",
        },
        "timing_semantics": (
            "host wall time; CUDA synchronize immediately before and after every "
            "individual latency invocation and each throughput block"
        ),
    }
    correctness = {
        "pass": bool(first_correctness["pass"] and post_steady_correctness["pass"]),
        "first_compiled_call": first_correctness,
        "post_steady_compiled_call": post_steady_correctness,
    }
    return correctness, metrics


def _dynamo_counters(torch: Any) -> Dict[str, Any]:
    counters = getattr(getattr(torch, "_dynamo", None), "utils", None)
    counters = getattr(counters, "counters", {})
    converted = _jsonable(counters)
    graph_breaks = (
        converted.get("graph_break", {}) if isinstance(converted, dict) else {}
    )
    stats = converted.get("stats", {}) if isinstance(converted, dict) else {}
    graph_break_values = (
        list(graph_breaks.values()) if isinstance(graph_breaks, dict) else []
    )
    event_sum = sum(
        int(value)
        for value in graph_break_values
        if isinstance(value, int) and not isinstance(value, bool)
    )
    unique_graphs = stats.get("unique_graphs") if isinstance(stats, dict) else None
    return {
        "all": converted,
        "stats": stats,
        "graph_break_reason_counts": graph_breaks,
        "graph_break_event_sum": event_sum,
        "graph_break_distinct_reason_count": len(graph_breaks)
        if isinstance(graph_breaks, dict)
        else None,
        "unique_graphs": unique_graphs,
    }


def _inductor_metrics(torch: Any) -> Dict[str, Any]:
    try:
        metrics = torch._inductor.metrics
    except (AttributeError, ImportError):
        return {"available": False}
    names = (
        "generated_kernel_count",
        "generated_cpp_vec_kernel_count",
        "num_bytes_accessed",
        "ir_nodes_pre_fusion",
        "cpp_to_dtype_count",
        "num_comprehensive_padding",
        "num_loop_reordering",
    )
    values: Dict[str, Any] = {"available": True}
    for name in names:
        if hasattr(metrics, name):
            values[name] = _jsonable(getattr(metrics, name))
    return values


def _safe_config_value(module: Any, dotted_name: str) -> Any:
    current = module
    try:
        for part in dotted_name.split("."):
            current = getattr(current, part)
        return _jsonable(current)
    except (AttributeError, RuntimeError):
        return None


def _set_required_config(module: Any, dotted_name: str, value: Any) -> None:
    """Set and verify a compiler config without silently accepting drift."""

    parts = dotted_name.split(".")
    current = module
    for part in parts[:-1]:
        if not hasattr(current, part):
            raise RuntimeError(f"required compiler config is missing: {dotted_name}")
        current = getattr(current, part)
    leaf = parts[-1]
    if not hasattr(current, leaf):
        raise RuntimeError(f"required compiler config is missing: {dotted_name}")
    setattr(current, leaf, value)
    actual = getattr(current, leaf)
    if actual != value:
        raise RuntimeError(
            f"compiler config {dotted_name} did not retain {value!r}; got {actual!r}"
        )


def _profile(torch: Any, compiled: Any, x: Any, out_dir: Path) -> Dict[str, Any]:
    return capture_untimed_profile(
        torch,
        compiled,
        x,
        out_dir,
        counter_snapshot_fn=lambda: _compiler_counter_snapshot(torch),
    )


def _run(args: argparse.Namespace) -> Tuple[Dict[str, Any], int]:
    started_at = _utc_now()
    worker_source = Path(__file__).resolve()
    config_source = worker_source.with_name("pytorch-main-dynamo-config.py")

    # Set environment-backed cache controls before torch imports their config
    # modules.  Direct config assignments below are also verified.  These
    # settings are fixed across arms and prevent global cache services from
    # bypassing the controller's fresh per-process cache directories.
    fixed_environment = {
        "CUDA_CACHE_DISABLE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
        "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE": "0",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
        "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE": "0",
        "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE": "0",
        "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE": "0",
    }
    os.environ.update(fixed_environment)

    # Importing torch is allowed here, but no torch.cuda call may precede the
    # feature-flag existence check below.
    import torch
    import torch._dynamo.config as dynamo_config

    if not hasattr(dynamo_config, "nested_graph_breaks"):
        raise RuntimeError(
            "installed torch lacks torch._dynamo.config.nested_graph_breaks; "
            "aborting before CUDA work"
        )
    dynamo_config.nested_graph_breaks = bool(args.enabled)
    actual_flag = bool(dynamo_config.nested_graph_breaks)
    if actual_flag != bool(args.enabled):
        raise RuntimeError("nested_graph_breaks did not retain the requested value")

    controller = _validate_controller_environment(args)

    import torch._functorch.config as functorch_config
    import torch._inductor as inductor
    import torch._inductor.config as inductor_config

    fixed_inductor_config = {
        "compile_threads": 1,
        "fx_graph_cache": False,
        "fx_graph_remote_cache": False,
        "autotune_remote_cache": False,
        "bundled_autotune_remote_cache": False,
    }
    fixed_functorch_config = {
        "enable_autograd_cache": False,
        "enable_remote_autograd_cache": False,
        "bundled_autograd_cache": False,
    }
    for name, value in fixed_inductor_config.items():
        _set_required_config(inductor_config, name, value)
    for name, value in fixed_functorch_config.items():
        _set_required_config(functorch_config, name, value)

    # These entry settings are identical in both arms.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("worker must see exactly one logical CUDA device")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    gpu_name = str(properties.name)
    if "B200" not in gpu_name.upper():
        raise RuntimeError(f"logical CUDA device 0 is not a B200: {gpu_name!r}")

    optional_torch_uuid = _jsonable(getattr(properties, "uuid", None))
    expected_uuid = controller["expected_gpu_uuid"]
    visible_uuid = controller["cuda_visible_devices"].strip()

    try:
        torch._dynamo.reset()
    except AttributeError:
        pass
    torch._dynamo.utils.counters.clear()
    counter_capture = prepare_counter_capture(torch._dynamo.utils.counters)
    try:
        import torch._inductor.metrics as inductor_metrics

        inductor_metrics.reset()
    except (AttributeError, ImportError):
        pass

    eager, generated_source, generated_source_sha = _build_workload(torch, args.case)
    compiled = torch.compile(
        eager,
        backend=COMPILE_BACKEND,
        mode=COMPILE_MODE,
        fullgraph=COMPILE_FULLGRAPH,
    )
    mode_options = _jsonable(inductor.list_mode_options(COMPILE_MODE))
    if mode_options.get("triton.cudagraphs") is not True:
        raise RuntimeError(
            "reduce-overhead mode does not report triton.cudagraphs=True in this build"
        )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    x = torch.randn(
        INPUT_ELEMENTS,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )

    recorded_warnings: List[Dict[str, Any]] = []
    original_showwarning = warnings.showwarning

    def showwarning(
        message: Warning,
        category: type,
        filename: str,
        lineno: int,
        file: Any = None,
        line: Optional[str] = None,
    ) -> None:
        recorded_warnings.append(
            {
                "category": getattr(category, "__name__", str(category)),
                "message": str(message),
                "filename": filename,
                "lineno": lineno,
            }
        )
        original_showwarning(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = showwarning
    try:
        correctness, timing = _run_timing(torch, eager, compiled, x, counter_capture)
        counter_capture.observe("after_run_timing")
        counter_lifecycle = counter_capture.finish()
        if not counter_lifecycle["invariants"]["lifecycle_valid"]:
            raise RuntimeError("Native compiler counter lifecycle was not preserved")
        counters = _dynamo_counters(torch)
        inductor_metrics = _inductor_metrics(torch)
        profile_result = (
            _profile(torch, compiled, x, args.out)
            if args.profile
            else {"requested": False, "status": "not_requested"}
        )
    finally:
        warnings.showwarning = original_showwarning

    spec = CASE_SPECS[args.case]
    math_signature = {
        "depth": int(spec["depth"]),
        "pre": ["sin", "multiply:0.5", "residual_add", "cos"],
        "post": ["sin", "multiply:0.25", "residual_add", "cos"],
    }
    result: Dict[str, Any] = {
        "schema": SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "scope": SCOPE,
        "status": "pass",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "case": args.case,
        "enabled": int(args.enabled),
        "seed": int(args.seed),
        "profile_requested": bool(args.profile),
        "treatment": {
            "name": "torch._dynamo.config.nested_graph_breaks",
            "requested": bool(args.enabled),
            "actual": actual_flag,
            "only_intended_arm_difference": True,
        },
        "workload": {
            "depth": int(spec["depth"]),
            "explicit_graph_breaks": int(spec["explicit_graph_breaks"]),
            "pre_pointwise_ops_per_frame": PRE_OPS_PER_FRAME,
            "post_pointwise_ops_per_frame": POST_OPS_PER_FRAME,
            "pointwise_math": ["sin", "multiply", "residual_add", "cos"],
            "input_elements": INPUT_ELEMENTS,
            "input_bytes": INPUT_ELEMENTS * 4,
            "input_dtype": INPUT_DTYPE,
            "input_distribution": "torch.randn with per-process CUDA generator",
            "generated_source": generated_source,
            "generated_source_sha256": generated_source_sha,
            "math_signature": math_signature,
            "math_signature_sha256": _sha256_bytes(
                json.dumps(
                    math_signature, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ),
        },
        "compile": {
            "api": "torch.compile",
            "backend": COMPILE_BACKEND,
            "mode": COMPILE_MODE,
            "fullgraph": COMPILE_FULLGRAPH,
            "mode_options": mode_options,
            "cuda_graph_eligibility": (
                "mode requests CUDA graphs; actual capture requires profiler markers "
                "or retained perf-hint evidence"
            ),
        },
        "device": {
            "type": "cuda",
            "logical_index": 0,
            "torch_name": gpu_name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(properties.total_memory),
            "expected_gpu_uuid": expected_uuid,
            "cuda_visible_devices": visible_uuid,
            "observed_uuid": visible_uuid,
            "observed_uuid_source": "controller-bound CUDA_VISIBLE_DEVICES",
            "optional_torch_device_uuid": optional_torch_uuid,
        },
        "correctness": correctness,
        "timing": timing,
        "dynamo": counters,
        "counter_lifecycle": counter_lifecycle,
        "inductor_metrics": inductor_metrics,
        "profile": profile_result,
        "runtime_warnings": recorded_warnings,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "torch_version": str(torch.__version__),
            "torch_git_version": _jsonable(getattr(torch.version, "git_version", None)),
            "torch_cuda_version": _jsonable(getattr(torch.version, "cuda", None)),
            "cudnn_version": _jsonable(torch.backends.cudnn.version()),
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "inductor_options": {
                "compile_threads": _safe_config_value(
                    inductor_config, "compile_threads"
                ),
                "fx_graph_cache": _safe_config_value(inductor_config, "fx_graph_cache"),
                "fx_graph_remote_cache": _safe_config_value(
                    inductor_config, "fx_graph_remote_cache"
                ),
                "autotune_remote_cache": _safe_config_value(
                    inductor_config, "autotune_remote_cache"
                ),
                "bundled_autotune_remote_cache": _safe_config_value(
                    inductor_config, "bundled_autotune_remote_cache"
                ),
                "triton.cudagraphs": _safe_config_value(
                    inductor_config, "triton.cudagraphs"
                ),
                "triton.cudagraph_skip_dynamic_graphs": _safe_config_value(
                    inductor_config, "triton.cudagraph_skip_dynamic_graphs"
                ),
                "max_autotune": _safe_config_value(inductor_config, "max_autotune"),
            },
            "functorch_options": {
                "enable_autograd_cache": _safe_config_value(
                    functorch_config, "enable_autograd_cache"
                ),
                "enable_remote_autograd_cache": _safe_config_value(
                    functorch_config, "enable_remote_autograd_cache"
                ),
                "bundled_autograd_cache": _safe_config_value(
                    functorch_config, "bundled_autograd_cache"
                ),
            },
            "fixed_environment": fixed_environment,
        },
        "controller_contract": controller,
        "provenance": {
            "argv": sys.argv,
            "worker_source_path": str(worker_source),
            "worker_source_sha256": _sha256_file(worker_source),
            "flag_source_snapshot_path": str(config_source),
            "flag_source_snapshot_sha256": _sha256_file(config_source),
            "flag_source_snapshot_line": 587,
            "timing_clock": "time.perf_counter_ns",
            "stderr_capture_owner": "controller",
        },
    }

    if args.profile and not bool(profile_result.get("evidence_complete")):
        # Preserve the valid benchmark result while making missing profile
        # evidence impossible to mistake for a successful capture.  The worker
        # still exits successfully so the controller completes the full matrix;
        # the analyzer gates any qualified win on both required traces.
        result["status"] = "benchmark_pass_profile_failed"
    return result, 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result_path = args.out / "result.json"
    try:
        result, exit_code = _run(args)
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "benchmark_id": BENCHMARK_ID,
            "scope": SCOPE,
            "status": "error",
            "finished_at": _utc_now(),
            "case": args.case,
            "enabled": int(args.enabled),
            "seed": int(args.seed),
            "profile_requested": bool(args.profile),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json_write(result_path, failure)
        print(
            f"benchmark failed; retained evidence at {result_path}: {exc}",
            file=sys.stderr,
        )
        return 1
    _atomic_json_write(result_path, result)
    print(str(result_path))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
