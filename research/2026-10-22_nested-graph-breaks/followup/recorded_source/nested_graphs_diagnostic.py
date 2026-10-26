#!/usr/bin/env python3
"""Bounded follow-up diagnostics for the retained nested-graph-break workload.

The controller launches every measurement cell in a fresh Python process with
fresh Inductor and Triton caches.  Torch is imported only by the private worker
subcommand.  ``--help`` and ``plan`` are standard-library-only operations.

This is diagnostic tooling, not a replacement for the frozen benchmark or its
qualification.  It never edits the imported source package.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "nested-graph-breaks-follow-up-diagnostic/v1"
CONTROLLER_SCHEMA = "nested-graph-breaks-follow-up-controller/v1"
EXPECTED_SOURCES = {
    "benchmark.py": "4c7a5a8dcef474405be19a87d22cfdeb7ea564c76f19acc67ddcd1125d247689",
    "counter_capture.py": "ef6c4dcb00491f1a5034ef14d55b9b505fbd9b6e744b383fd6d8c2868a938583",
    "profile_capture.py": "ab2bf71e11dab69f0e35f615a6810a15d883a6f26b9bb64430ff24f7d8e9f02e",
}
CASES = ("nested_break_depth8", "nested_break_depth4", "no_break_depth4")
VARIANTS = ("original", "no-cudagraph")
FIXED_ENVIRONMENT = {
    "CUDA_CACHE_DISABLE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
    "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE": "0",
    "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE": "0",
    "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE": "0",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
    "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE": "0",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
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
    return repr(value)


def _rss_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.system(),
        "current_rss_bytes": None,
        "high_water_rss_bytes": None,
        "source": None,
    }
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        values: dict[str, int] = {}
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, raw = line.split(":", 1)
                fields = raw.strip().split()
                if len(fields) >= 2 and fields[1].lower() == "kb":
                    values[key] = int(fields[0]) * 1024
        result.update(
            {
                "current_rss_bytes": values.get("VmRSS"),
                "high_water_rss_bytes": values.get("VmHWM"),
                "source": "/proc/self/status",
            }
        )
        return result
    usage = resource.getrusage(resource.RUSAGE_SELF)
    scale = 1 if platform.system() == "Darwin" else 1024
    result.update(
        {
            "high_water_rss_bytes": int(usage.ru_maxrss) * scale,
            "source": "resource.getrusage; current RSS unavailable",
        }
    )
    return result


def _validate_source_root(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"original source root is not a directory: {root}")
    manifest: dict[str, Any] = {}
    for name, expected in EXPECTED_SOURCES.items():
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"required regular source file is missing: {path}")
        observed = _sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"frozen source hash mismatch for {name}: {observed} != {expected}"
            )
        stat_result = path.stat()
        manifest[name] = {
            "path": str(path),
            "bytes": stat_result.st_size,
            "sha256": observed,
            "mode": stat_result.st_mode & 0o7777,
        }
    return {"root": str(root), "files": manifest}


def _load_original(root: Path) -> Any:
    """Load the frozen worker without importing Torch in controller paths."""

    _validate_source_root(root)
    root_text = str(root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    path = root / "benchmark.py"
    spec = importlib.util.spec_from_file_location("retained_nested_graph_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load retained worker from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = {
        "INPUT_ELEMENTS": 1 << 20,
        "COMPILE_BACKEND": "inductor",
        "COMPILE_MODE": "reduce-overhead",
        "COMPILE_FULLGRAPH": False,
        "WARMUP_CALLS": 25,
    }
    for name, expected in required.items():
        actual = getattr(module, name, None)
        if actual != expected:
            raise RuntimeError(
                f"retained worker constant drifted: {name}={actual!r}, expected {expected!r}"
            )
    return module


def _require_worker_environment(expected_gpu_uuid: str) -> dict[str, Any]:
    if not expected_gpu_uuid.upper().startswith("GPU-"):
        raise RuntimeError("--expected-gpu-uuid must be a full GPU-... UUID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    entries = [item.strip() for item in visible.split(",") if item.strip()]
    if len(entries) != 1 or entries[0].upper() != expected_gpu_uuid.upper():
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain exactly the expected full GPU UUID"
        )
    caches: dict[str, Any] = {}
    resolved: list[Path] = []
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        raw = os.environ.get(key)
        if not raw:
            raise RuntimeError(f"{key} must be set by the controller")
        path = Path(raw).resolve()
        if not path.is_dir() or path.is_symlink():
            raise RuntimeError(f"{key} must be a regular directory: {path}")
        entries_now = list(path.iterdir())
        if entries_now:
            raise RuntimeError(f"{key} must start empty: {path}")
        caches[key] = {"path": str(path), "initially_empty": True}
        resolved.append(path)
    if resolved[0] == resolved[1]:
        raise RuntimeError("Inductor and Triton cache directories must differ")
    for key, expected in FIXED_ENVIRONMENT.items():
        actual = os.environ.get(key)
        if actual != expected:
            raise RuntimeError(f"fixed environment mismatch: {key}={actual!r}")
    return {
        "expected_gpu_uuid": expected_gpu_uuid,
        "cuda_visible_devices": visible,
        "caches": caches,
        "fixed_environment": dict(FIXED_ENVIRONMENT),
    }


def _hardware_identity(torch: Any, expected_gpu_uuid: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("worker must see exactly one logical CUDA device")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    observed_uuid = _jsonable(getattr(properties, "uuid", None))
    if isinstance(observed_uuid, str) and observed_uuid.upper().startswith("GPU-"):
        if observed_uuid.upper() != expected_gpu_uuid.upper():
            raise RuntimeError(
                f"observed GPU UUID {observed_uuid} != expected {expected_gpu_uuid}"
            )
    driver_version = None
    driver_version_fn = getattr(torch._C, "_cuda_getDriverVersion", None)
    if callable(driver_version_fn):
        try:
            driver_version = _jsonable(driver_version_fn())
        except (RuntimeError, TypeError):
            driver_version = None
    return {
        "logical_device": 0,
        "name": str(properties.name),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "multiprocessor_count": int(properties.multi_processor_count),
        "total_memory_bytes": int(properties.total_memory),
        "observed_uuid": observed_uuid,
        "expected_gpu_uuid": expected_gpu_uuid,
        "torch_version": str(torch.__version__),
        "torch_git_version": _jsonable(getattr(torch.version, "git_version", None)),
        "torch_cuda_version": _jsonable(getattr(torch.version, "cuda", None)),
        "cuda_driver_version": driver_version,
        "cudnn_version": _jsonable(torch.backends.cudnn.version()),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }


def _prepare_compiled(
    original: Any,
    *,
    case: str,
    seed: int,
    enabled: int,
    variant: str,
    expected_gpu_uuid: str,
) -> dict[str, Any]:
    # Torch imports are intentionally confined to a fresh worker process.
    import torch
    import torch._dynamo.config as dynamo_config
    import torch._functorch.config as functorch_config
    import torch._inductor as inductor
    import torch._inductor.config as inductor_config

    if not hasattr(dynamo_config, "nested_graph_breaks"):
        raise RuntimeError("runtime lacks torch._dynamo.config.nested_graph_breaks")
    default_flag = bool(dynamo_config.nested_graph_breaks)
    if default_flag is not False:
        raise RuntimeError(
            "runtime default is not False; version portability must be reviewed before running"
        )
    dynamo_config.nested_graph_breaks = bool(enabled)
    if bool(dynamo_config.nested_graph_breaks) != bool(enabled):
        raise RuntimeError("nested_graph_breaks did not retain the requested value")

    fixed_inductor = {
        "compile_threads": 1,
        "fx_graph_cache": False,
        "fx_graph_remote_cache": False,
        "autotune_remote_cache": False,
        "bundled_autotune_remote_cache": False,
    }
    fixed_functorch = {
        "enable_autograd_cache": False,
        "enable_remote_autograd_cache": False,
        "bundled_autograd_cache": False,
    }
    for name, value in fixed_inductor.items():
        original._set_required_config(inductor_config, name, value)
    for name, value in fixed_functorch.items():
        original._set_required_config(functorch_config, name, value)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.manual_seed_all(seed)
    hardware = _hardware_identity(torch, expected_gpu_uuid)

    try:
        torch._dynamo.reset()
    except AttributeError:
        pass
    torch._dynamo.utils.counters.clear()
    try:
        import torch._inductor.metrics as inductor_metrics

        inductor_metrics.reset()
    except (AttributeError, ImportError):
        pass

    eager, generated_source, generated_source_sha256 = original._build_workload(
        torch, case
    )
    original_mode_options = _jsonable(inductor.list_mode_options("reduce-overhead"))
    if original_mode_options.get("triton.cudagraphs") is not True:
        raise RuntimeError("reduce-overhead no longer requests triton.cudagraphs=True")
    if variant == "original":
        compiled = torch.compile(
            eager,
            backend=original.COMPILE_BACKEND,
            mode=original.COMPILE_MODE,
            fullgraph=original.COMPILE_FULLGRAPH,
        )
        compile_configuration = {
            "mode": original.COMPILE_MODE,
            "options": original_mode_options,
            "isolation_status": "exact retained compile mode",
        }
    elif variant == "no-cudagraph":
        extra_options = sorted(
            key for key in original_mode_options if key != "triton.cudagraphs"
        )
        if extra_options:
            raise RuntimeError(
                "no-cudagraph is not a one-option ablation in this build; "
                f"unexpected reduce-overhead options: {extra_options}"
            )
        compiled = torch.compile(
            eager,
            backend=original.COMPILE_BACKEND,
            options={"triton.cudagraphs": False},
            fullgraph=original.COMPILE_FULLGRAPH,
        )
        compile_configuration = {
            "mode": None,
            "options": {"triton.cudagraphs": False},
            "isolation_status": (
                "candidate one-option ablation; valid only if an untimed profile "
                "records zero cudaGraphLaunch events and positive kernels"
            ),
        }
    else:
        raise RuntimeError(f"unsupported variant: {variant}")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    x = torch.randn(
        original.INPUT_ELEMENTS,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    with torch.no_grad():
        reference = eager(x)
        torch.cuda.synchronize()

    return {
        "torch": torch,
        "eager": eager,
        "compiled": compiled,
        "x": x,
        "reference": reference,
        "hardware": hardware,
        "treatment": {
            "name": "torch._dynamo.config.nested_graph_breaks",
            "default_before_set": default_flag,
            "requested": bool(enabled),
            "actual": bool(dynamo_config.nested_graph_breaks),
        },
        "compile": compile_configuration,
        "workload": {
            "case": case,
            "seed": seed,
            "depth": int(original.CASE_SPECS[case]["depth"]),
            "explicit_graph_breaks": int(
                original.CASE_SPECS[case]["explicit_graph_breaks"]
            ),
            "input_elements": original.INPUT_ELEMENTS,
            "input_dtype": original.INPUT_DTYPE,
            "generated_source_sha256": generated_source_sha256,
            "generated_source": generated_source,
        },
    }


def _cuda_memory_snapshot(torch: Any, phase: str, window: str) -> dict[str, Any]:
    torch.cuda.synchronize()
    return {
        "phase": phase,
        "peak_window": window,
        "cuda": {
            "memory_allocated_bytes": int(torch.cuda.memory_allocated()),
            "memory_reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "host": _rss_snapshot(),
    }


def _run_timeline(original: Any, state: Mapping[str, Any], horizon: int) -> dict[str, Any]:
    torch = state["torch"]
    compiled = state["compiled"]
    x = state["x"]
    reference = state["reference"]
    intervals_ns: list[int] = []
    active_cumulative_ns: list[int] = []
    continuous_elapsed_ns: list[int] = []
    active_total = 0
    final_output = None

    torch.cuda.synchronize()
    continuous_start = time.perf_counter_ns()
    with torch.no_grad():
        for index in range(horizon):
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            output = compiled(x)
            torch.cuda.synchronize()
            finished = time.perf_counter_ns()
            interval = finished - started
            if interval <= 0:
                raise RuntimeError(f"nonpositive interval at call {index + 1}")
            intervals_ns.append(interval)
            active_total += interval
            active_cumulative_ns.append(active_total)
            continuous_elapsed_ns.append(finished - continuous_start)
            if index + 1 == horizon:
                final_output = output
            else:
                del output

    if final_output is None:
        raise RuntimeError("timeline produced no final output")
    correctness = original._correctness(torch, reference, final_output)
    del final_output
    if not correctness.get("pass"):
        raise RuntimeError("eager correctness failed after the continuous timeline")

    return {
        "kind": "timeline",
        "horizon_calls": horizon,
        "interval_definition": (
            "perf_counter_ns around one compiled call with CUDA synchronization "
            "immediately before and after"
        ),
        "continuous_definition": (
            "wall time from immediately before call 1 through each synchronized "
            "call; includes bounded Python loop bookkeeping between calls but "
            "excludes process startup, imports, input creation, and eager reference"
        ),
        "per_call_interval_ns": intervals_ns,
        "active_call_cumulative_ns": active_cumulative_ns,
        "continuous_elapsed_ns": continuous_elapsed_ns,
        "correctness": correctness,
        "causal_limit": (
            "Observed crossover is diagnostic for this process and horizon. It is "
            "not a population estimate or a host-versus-GPU causal decomposition."
        ),
    }


def _run_memory(original: Any, state: Mapping[str, Any], horizon: int) -> dict[str, Any]:
    torch = state["torch"]
    compiled = state["compiled"]
    x = state["x"]
    reference = state["reference"]
    records: list[dict[str, Any]] = []

    records.append(
        _cuda_memory_snapshot(
            torch, "input_and_eager_reference_ready", "process start through eager"
        )
    )

    def reset_peak() -> None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    reset_peak()
    with torch.no_grad():
        first = compiled(x)
        torch.cuda.synchronize()
    records.append(
        _cuda_memory_snapshot(torch, "first_compiled_call", "first compiled call")
    )
    first_correctness = original._correctness(torch, reference, first)
    del first
    if not first_correctness.get("pass"):
        raise RuntimeError("eager correctness failed after first compiled call")

    boundaries = sorted({min(25, horizon), min(100, horizon), horizon})
    completed = 1
    final_output = None
    for boundary in boundaries:
        if boundary <= completed:
            continue
        reset_peak()
        with torch.no_grad():
            for call_index in range(completed + 1, boundary + 1):
                output = compiled(x)
                torch.cuda.synchronize()
                if call_index == horizon:
                    final_output = output
                else:
                    del output
        records.append(
            _cuda_memory_snapshot(
                torch,
                f"after_call_{boundary}",
                f"calls {completed + 1} through {boundary}",
            )
        )
        completed = boundary

    if final_output is None:
        with torch.no_grad():
            final_output = compiled(x)
            torch.cuda.synchronize()
    final_correctness = original._correctness(torch, reference, final_output)
    del final_output
    if not final_correctness.get("pass"):
        raise RuntimeError("eager correctness failed after memory horizon")

    return {
        "kind": "memory",
        "horizon_calls": horizon,
        "records": records,
        "first_correctness": first_correctness,
        "final_correctness": final_correctness,
        "measurement_limit": (
            "PyTorch allocated/reserved counters and process RSS are phase-level "
            "diagnostics. They do not identify every driver allocation or assign "
            "CUDA Graph private-pool bytes to individual graph nodes."
        ),
    }


def _interval_union_us(events: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((start, end) for start, end in events if end >= start)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _profile_decomposition(trace_path: Path) -> dict[str, Any]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    events = [event for event in trace.get("traceEvents", []) if isinstance(event, dict)]
    markers = [
        event
        for event in events
        if event.get("name") == "nested_graphs_compiled_invocation"
        and isinstance(event.get("ts"), (int, float))
        and isinstance(event.get("dur"), (int, float))
    ]
    invocations: list[dict[str, Any]] = []
    for index, marker in enumerate(sorted(markers, key=lambda item: item["ts"]), 1):
        start = float(marker["ts"])
        end = start + float(marker["dur"])
        members = [
            event
            for event in events
            if isinstance(event.get("ts"), (int, float))
            and start <= float(event["ts"]) <= end
        ]
        kernels = [
            event
            for event in members
            if "kernel" in str(event.get("cat", "")).lower()
            and isinstance(event.get("dur"), (int, float))
        ]
        cuda_apis = [
            event
            for event in members
            if any(
                token in str(event.get("cat", "")).lower().replace(" ", "_")
                for token in ("cuda_runtime", "cuda_driver")
            )
            and isinstance(event.get("dur"), (int, float))
        ]
        graph_launches = [
            event
            for event in cuda_apis
            if "cudagraphlaunch" in "".join(str(event.get("name", "")).lower().split())
        ]
        kernel_intervals = [
            (float(event["ts"]), float(event["ts"]) + float(event["dur"]))
            for event in kernels
        ]
        api_intervals = [
            (float(event["ts"]), float(event["ts"]) + float(event["dur"]))
            for event in cuda_apis
        ]
        first_kernel = min((item[0] for item in kernel_intervals), default=None)
        last_kernel = max((item[1] for item in kernel_intervals), default=None)
        kernel_union = _interval_union_us(kernel_intervals)
        gpu_span = (
            None
            if first_kernel is None or last_kernel is None
            else last_kernel - first_kernel
        )
        invocations.append(
            {
                "index": index,
                "cpu_marker_duration_us": float(marker["dur"]),
                "cuda_api_count": len(cuda_apis),
                "cuda_graph_launch_count": len(graph_launches),
                "kernel_count": len(kernels),
                "cuda_api_union_us": _interval_union_us(api_intervals),
                "kernel_active_union_us": kernel_union,
                "first_kernel_start_from_marker_us": (
                    None if first_kernel is None else first_kernel - start
                ),
                "gpu_kernel_span_us": gpu_span,
                "idle_inside_gpu_kernel_span_us": (
                    None if gpu_span is None else max(0.0, gpu_span - kernel_union)
                ),
                "marker_tail_after_last_kernel_us": (
                    None if last_kernel is None else end - last_kernel
                ),
            }
        )
    return {
        "trace_path": str(trace_path),
        "trace_sha256": _sha256_file(trace_path),
        "invocations": invocations,
        "interpretation": (
            "Intervals are descriptive trace geometry. API union, kernel-active "
            "union, and idle spans may include profiler perturbation and do not "
            "form a causal percentage decomposition."
        ),
    }


def _run_profile(
    original: Any,
    state: Mapping[str, Any],
    out_dir: Path,
    variant: str,
    invocations: int,
) -> dict[str, Any]:
    torch = state["torch"]
    compiled = state["compiled"]
    x = state["x"]
    reference = state["reference"]
    output = None
    with torch.no_grad():
        for _ in range(original.WARMUP_CALLS):
            output = compiled(x)
            torch.cuda.synchronize()
            del output
        output = compiled(x)
        torch.cuda.synchronize()
    correctness = original._correctness(torch, reference, output)
    del output
    if not correctness.get("pass"):
        raise RuntimeError("eager correctness failed before untimed profile")

    profile = original.capture_untimed_profile(
        torch,
        compiled,
        x,
        out_dir,
        invocations=invocations,
        counter_snapshot_fn=lambda: original._compiler_counter_snapshot(torch),
    )
    trace_path = out_dir / "profile.chrome.json"
    if not trace_path.is_file() or trace_path.stat().st_size <= 0:
        raise RuntimeError("untimed profile did not retain a nonempty Chrome trace")
    graph_launches = int(profile.get("cuda_graph_launch_event_count", 0))
    kernels = int(profile.get("cuda_kernel_event_count", 0))
    invocation_complete = bool(profile.get("invocation_evidence_complete"))
    counter_stable = bool(profile.get("compiler_counter_audit", {}).get("stable"))
    if variant == "original":
        variant_valid = bool(
            profile.get("evidence_complete")
            and graph_launches > 0
            and kernels > 0
            and invocation_complete
            and counter_stable
        )
        expected = "positive cudaGraphLaunch and kernel events"
    else:
        variant_valid = bool(
            graph_launches == 0
            and kernels > 0
            and invocation_complete
            and counter_stable
        )
        expected = "zero cudaGraphLaunch events, positive kernels, stable counters"
    if not variant_valid:
        raise RuntimeError(
            f"profile did not validate variant {variant!r}; expected {expected}"
        )
    return {
        "kind": "profile",
        "correctness": correctness,
        "profile": profile,
        "trace_decomposition": _profile_decomposition(trace_path),
        "variant_validation": {
            "variant": variant,
            "expected": expected,
            "passed": variant_valid,
        },
        "causal_limit": (
            "This profile identifies launch counts and time-correlated host/GPU "
            "intervals. It does not assign a causal percentage of the end-to-end "
            "speedup to host gaps versus device work."
        ),
    }


def _worker(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise RuntimeError(f"worker output must not already exist: {out_dir}")
    if not out_dir.parent.is_dir():
        raise RuntimeError(f"worker output parent is missing: {out_dir.parent}")
    out_dir.mkdir(mode=0o700)
    started = _utc_now()
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "started_at_utc": started,
        "kind": args.kind,
        "case": args.case,
        "seed": args.seed,
        "enabled": args.enabled,
        "variant": args.variant,
        "horizon_calls": args.horizon,
        "pre_import_host_memory": _rss_snapshot(),
        "scope": (
            "bounded synthetic compiler diagnostic; no serving, training, model "
            "quality, or cross-hardware portability claim"
        ),
    }
    exit_code = 1
    try:
        environment = _require_worker_environment(args.expected_gpu_uuid)
        source = _validate_source_root(Path(args.original_root))
        original = _load_original(Path(args.original_root))
        state = _prepare_compiled(
            original,
            case=args.case,
            seed=args.seed,
            enabled=args.enabled,
            variant=args.variant,
            expected_gpu_uuid=args.expected_gpu_uuid,
        )
        if args.kind == "timeline":
            measurement = _run_timeline(original, state, args.horizon)
        elif args.kind == "memory":
            measurement = _run_memory(original, state, args.horizon)
        elif args.kind == "profile":
            if args.case != "nested_break_depth8":
                raise RuntimeError("profile cells are restricted to nested_break_depth8")
            measurement = _run_profile(
                original,
                state,
                out_dir,
                args.variant,
                args.profile_invocations,
            )
        else:
            raise RuntimeError(f"unsupported worker kind: {args.kind}")
        result.update(
            {
                "status": "pass",
                "environment": environment,
                "source": source,
                "hardware": state["hardware"],
                "treatment": state["treatment"],
                "compile": state["compile"],
                "workload": state["workload"],
                "measurement": measurement,
            }
        )
        exit_code = 0
    except Exception as exc:
        result.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    result["finished_at_utc"] = _utc_now()
    _atomic_json(out_dir / "result.json", result)
    print(out_dir / "result.json")
    return exit_code


def _arm_order(seed: int) -> tuple[int, int]:
    return (0, 1) if seed % 2 == 0 else (1, 0)


def _cell_id(
    kind: str, case: str, seed: int, enabled: int, variant: str
) -> str:
    return f"{kind}-{case}-seed{seed}-flag{enabled}-{variant}"


def _build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for case in args.cases:
        for seed in args.seeds:
            for variant in args.variants:
                for enabled in _arm_order(seed):
                    cells.append(
                        {
                            "kind": "timeline",
                            "case": case,
                            "seed": seed,
                            "enabled": enabled,
                            "variant": variant,
                        }
                    )
    diagnostic_seed = args.diagnostic_seed
    for case in args.cases:
        for variant in args.variants:
            for enabled in _arm_order(diagnostic_seed):
                cells.append(
                    {
                        "kind": "memory",
                        "case": case,
                        "seed": diagnostic_seed,
                        "enabled": enabled,
                        "variant": variant,
                    }
                )
    if args.profile and "nested_break_depth8" in args.cases:
        for variant in args.variants:
            for enabled in _arm_order(diagnostic_seed):
                cells.append(
                    {
                        "kind": "profile",
                        "case": "nested_break_depth8",
                        "seed": diagnostic_seed,
                        "enabled": enabled,
                        "variant": variant,
                    }
                )
    for cell in cells:
        cell["id"] = _cell_id(
            cell["kind"],
            cell["case"],
            cell["seed"],
            cell["enabled"],
            cell["variant"],
        )
    return cells


def _validate_controller_args(args: argparse.Namespace) -> None:
    if not 3000 <= args.horizon <= 4000:
        raise RuntimeError("--horizon must be between 3000 and 4000 calls")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise RuntimeError("--seeds must be a nonempty unique list")
    if any(seed not in (0, 1, 2, 3) for seed in args.seeds):
        raise RuntimeError("seeds are restricted to 0, 1, 2, and 3")
    if args.diagnostic_seed not in args.seeds:
        raise RuntimeError("--diagnostic-seed must be included in --seeds")
    if len(set(args.cases)) != len(args.cases):
        raise RuntimeError("--cases must not contain duplicates")
    if len(set(args.variants)) != len(args.variants):
        raise RuntimeError("--variants must not contain duplicates")
    if not 10 <= args.child_timeout_seconds <= 120:
        raise RuntimeError("--child-timeout-seconds must be within [10, 120]")
    if not 30 <= args.total_budget_seconds <= 180:
        raise RuntimeError("--total-budget-seconds must be within [30, 180]")
    if not 1 <= args.profile_invocations <= 10:
        raise RuntimeError("--profile-invocations must be within [1, 10]")


def _run_child(
    args: argparse.Namespace,
    run_root: Path,
    cell: Mapping[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    cell_id = str(cell["id"])
    cell_out = run_root / "cells" / cell_id
    stdout_path = run_root / "control" / f"{cell_id}.stdout.txt"
    stderr_path = run_root / "control" / f"{cell_id}.stderr.txt"
    command = [
        sys.executable,
        "-I",
        "-B",
        str(Path(__file__).resolve()),
        "_worker",
        "--kind",
        str(cell["kind"]),
        "--case",
        str(cell["case"]),
        "--seed",
        str(cell["seed"]),
        "--enabled",
        str(cell["enabled"]),
        "--variant",
        str(cell["variant"]),
        "--horizon",
        str(args.horizon),
        "--profile-invocations",
        str(args.profile_invocations),
        "--original-root",
        str(Path(args.original_root).resolve()),
        "--expected-gpu-uuid",
        args.expected_gpu_uuid,
        "--out-dir",
        str(cell_out),
    ]
    environment = os.environ.copy()
    environment.update(FIXED_ENVIRONMENT)
    environment["CUDA_VISIBLE_DEVICES"] = args.expected_gpu_uuid
    started_wall = time.monotonic()
    timed_out = False
    return_code: int | None = None
    stdout = ""
    stderr = ""
    cache_parent = Path(tempfile.mkdtemp(prefix=f"ngbdiag-{cell_id}-"))
    try:
        inductor_cache = cache_parent / "inductor"
        triton_cache = cache_parent / "triton"
        inductor_cache.mkdir()
        triton_cache.mkdir()
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
        environment["TRITON_CACHE_DIR"] = str(triton_cache)
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
        return_code = process.returncode
    finally:
        cache_initial_root = cache_parent.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        if cache_initial_root.parent == temp_root and cache_initial_root.name.startswith(
            "ngbdiag-"
        ):
            shutil.rmtree(cache_initial_root)
        else:
            raise RuntimeError(f"refusing unsafe cache cleanup: {cache_initial_root}")
    _atomic_text(stdout_path, stdout)
    _atomic_text(stderr_path, stderr)
    receipt = {
        "cell": dict(cell),
        "command": command,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "return_code": return_code,
        "elapsed_seconds": time.monotonic() - started_wall,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "result": str(cell_out / "result.json"),
        "fresh_caches_removed": True,
    }
    return receipt


def _load_passed_results(run_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted((run_root / "cells").glob("*/result.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        value["_result_path"] = str(path)
        if value.get("status") == "pass":
            results.append(value)
    return results


def _first_crossing(
    baseline: Sequence[int], candidate: Sequence[int], *, sustained: bool
) -> int | None:
    if len(baseline) != len(candidate):
        return None
    if not sustained:
        for index, (base_value, candidate_value) in enumerate(
            zip(baseline, candidate), 1
        ):
            if candidate_value <= base_value:
                return index
        return None
    remains_below = True
    earliest: int | None = None
    for index in range(len(baseline) - 1, -1, -1):
        remains_below = remains_below and candidate[index] <= baseline[index]
        if remains_below:
            earliest = index + 1
    return earliest


def _summarize(run_root: Path, receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    results = _load_passed_results(run_root)
    by_key: dict[tuple[str, str, int, str, int], dict[str, Any]] = {}
    for result in results:
        key = (
            str(result["kind"]),
            str(result["case"]),
            int(result["seed"]),
            str(result["variant"]),
            int(result["enabled"]),
        )
        by_key[key] = result

    timeline_pairs: list[dict[str, Any]] = []
    memory_pairs: list[dict[str, Any]] = []
    profile_pairs: list[dict[str, Any]] = []
    group_keys = sorted({key[:4] for key in by_key})
    for kind, case, seed, variant in group_keys:
        baseline = by_key.get((kind, case, seed, variant, 0))
        candidate = by_key.get((kind, case, seed, variant, 1))
        if baseline is None or candidate is None:
            continue
        if kind == "timeline":
            base_measurement = baseline["measurement"]
            candidate_measurement = candidate["measurement"]
            base_continuous = base_measurement["continuous_elapsed_ns"]
            candidate_continuous = candidate_measurement["continuous_elapsed_ns"]
            base_active = base_measurement["active_call_cumulative_ns"]
            candidate_active = candidate_measurement["active_call_cumulative_ns"]
            timeline_pairs.append(
                {
                    "case": case,
                    "seed": seed,
                    "variant": variant,
                    "horizon_calls": len(base_continuous),
                    "continuous_wall": {
                        "first_candidate_le_baseline_call": _first_crossing(
                            base_continuous, candidate_continuous, sustained=False
                        ),
                        "first_sustained_candidate_le_baseline_call": _first_crossing(
                            base_continuous, candidate_continuous, sustained=True
                        ),
                        "baseline_final_ns": base_continuous[-1],
                        "candidate_final_ns": candidate_continuous[-1],
                    },
                    "active_synchronized_calls": {
                        "first_candidate_le_baseline_call": _first_crossing(
                            base_active, candidate_active, sustained=False
                        ),
                        "first_sustained_candidate_le_baseline_call": _first_crossing(
                            base_active, candidate_active, sustained=True
                        ),
                        "baseline_final_ns": base_active[-1],
                        "candidate_final_ns": candidate_active[-1],
                    },
                    "claim_limit": (
                        "Observed crossover within this paired process horizon only; "
                        "do not pool seeds or infer a population break-even."
                    ),
                }
            )
        elif kind == "memory":
            base_records = {
                row["phase"]: row for row in baseline["measurement"]["records"]
            }
            candidate_records = {
                row["phase"]: row for row in candidate["measurement"]["records"]
            }
            comparisons: list[dict[str, Any]] = []
            for phase in sorted(set(base_records) & set(candidate_records)):
                base_row = base_records[phase]
                candidate_row = candidate_records[phase]
                deltas: dict[str, Any] = {}
                for field in (
                    "memory_allocated_bytes",
                    "memory_reserved_bytes",
                    "max_memory_allocated_bytes",
                    "max_memory_reserved_bytes",
                ):
                    base_value = int(base_row["cuda"][field])
                    candidate_value = int(candidate_row["cuda"][field])
                    deltas[field] = {
                        "baseline": base_value,
                        "candidate": candidate_value,
                        "candidate_minus_baseline": candidate_value - base_value,
                    }
                comparisons.append({"phase": phase, "cuda": deltas})
            memory_pairs.append(
                {
                    "case": case,
                    "seed": seed,
                    "variant": variant,
                    "phase_comparisons": comparisons,
                    "claim_limit": (
                        "Arithmetic comparison only. No equivalence tolerance or "
                        "memory-parity gate was predeclared."
                    ),
                }
            )
        elif kind == "profile":
            profile_pairs.append(
                {
                    "case": case,
                    "seed": seed,
                    "variant": variant,
                    "baseline": {
                        "graph_launches": baseline["measurement"]["profile"].get(
                            "cuda_graph_launch_event_count"
                        ),
                        "kernels": baseline["measurement"]["profile"].get(
                            "cuda_kernel_event_count"
                        ),
                        "trace_decomposition": baseline["measurement"][
                            "trace_decomposition"
                        ],
                    },
                    "candidate": {
                        "graph_launches": candidate["measurement"]["profile"].get(
                            "cuda_graph_launch_event_count"
                        ),
                        "kernels": candidate["measurement"]["profile"].get(
                            "cuda_kernel_event_count"
                        ),
                        "trace_decomposition": candidate["measurement"][
                            "trace_decomposition"
                        ],
                    },
                    "claim_limit": (
                        "Descriptive profile evidence only. No causal percentage "
                        "split between host gaps and device execution."
                    ),
                }
            )

    return {
        "schema": "nested-graph-breaks-follow-up-summary/v1",
        "generated_at_utc": _utc_now(),
        "completed_results": len(results),
        "receipts": list(receipts),
        "timeline_pairs": timeline_pairs,
        "memory_pairs": memory_pairs,
        "profile_pairs": profile_pairs,
        "global_limits": [
            "Diagnostic evidence is not the frozen publication cohort.",
            "No result establishes serving, training, model quality, or portability.",
            "The no-cudagraph variant is interpretable only after its profile validates zero graph launches and positive kernel activity.",
            "No host-versus-GPU causal percentage is computed.",
        ],
    }


def _controller(args: argparse.Namespace) -> int:
    _validate_controller_args(args)
    original_root = Path(args.original_root).expanduser()
    requested_output = Path(args.output_root).expanduser()
    if not original_root.is_absolute():
        raise RuntimeError("--original-root must be absolute")
    if not requested_output.is_absolute():
        raise RuntimeError("--output-root must be absolute")
    source = _validate_source_root(original_root)
    run_root = requested_output.resolve()
    if run_root.exists():
        raise RuntimeError(f"output root already exists: {run_root}")
    if not run_root.parent.is_dir() or run_root.parent.is_symlink():
        raise RuntimeError(f"output parent must be an existing directory: {run_root.parent}")
    run_root.mkdir(mode=0o700)
    (run_root / "cells").mkdir()
    (run_root / "control").mkdir()
    plan = _build_plan(args)
    config = {
        "schema": CONTROLLER_SCHEMA,
        "started_at_utc": _utc_now(),
        "source": source,
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "horizon_calls": args.horizon,
        "cases": args.cases,
        "seeds": args.seeds,
        "variants": args.variants,
        "diagnostic_seed": args.diagnostic_seed,
        "profile": args.profile,
        "profile_invocations": args.profile_invocations,
        "child_timeout_seconds": args.child_timeout_seconds,
        "total_budget_seconds": args.total_budget_seconds,
        "plan": plan,
    }
    _atomic_json(run_root / "run-config.json", config)
    receipts: list[dict[str, Any]] = []
    started = time.monotonic()
    failure: str | None = None
    for cell in plan:
        remaining = args.total_budget_seconds - (time.monotonic() - started)
        if remaining < 10:
            failure = "overall diagnostic budget exhausted before next cell"
            break
        timeout = min(float(args.child_timeout_seconds), remaining)
        receipt = _run_child(args, run_root, cell, timeout)
        receipts.append(receipt)
        _atomic_json(run_root / "control" / f"{cell['id']}.receipt.json", receipt)
        if receipt["timed_out"] or receipt["return_code"] != 0:
            failure = f"cell failed closed: {cell['id']}"
            break
    summary = _summarize(run_root, receipts)
    completed = len(receipts) == len(plan) and failure is None
    summary.update(
        {
            "status": "COMPLETE_DIAGNOSTIC" if completed else "PARTIAL_DIAGNOSTIC",
            "failure": failure,
            "planned_cells": len(plan),
            "executed_cells": len(receipts),
            "elapsed_seconds": time.monotonic() - started,
            "finished_at_utc": _utc_now(),
        }
    )
    _atomic_json(run_root / "summary.json", summary)
    terminal_name = "COMPLETE.json" if completed else "PARTIAL.json"
    _atomic_json(
        run_root / terminal_name,
        {
            "status": summary["status"],
            "summary": str(run_root / "summary.json"),
            "failure": failure,
            "finished_at_utc": summary["finished_at_utc"],
        },
    )
    print(run_root / "summary.json")
    return 0 if completed else 1


def _add_plan_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=CASES,
        default=["nested_break_depth8"],
        help="Cases to run. The default is the minimum primary diagnostic.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1], help="Paired repeat seeds."
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=["original"],
        help="Original path, or an optional separately validated no-CUDA-Graph ablation.",
    )
    parser.add_argument("--diagnostic-seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=3500)
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture two untimed primary-case profiles per variant.",
    )
    parser.add_argument("--profile-invocations", type=int, default=5)
    parser.add_argument("--child-timeout-seconds", type=int, default=45)
    parser.add_argument("--total-budget-seconds", type=int, default=170)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan", help="Print the bounded process plan without importing Torch."
    )
    _add_plan_options(plan_parser)

    run_parser = subparsers.add_parser(
        "run", help="Execute the bounded diagnostic under an existing GPU owner."
    )
    _add_plan_options(run_parser)
    run_parser.add_argument("--original-root", type=Path, required=True)
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--expected-gpu-uuid", required=True)

    worker_parser = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--kind", choices=("timeline", "memory", "profile"), required=True)
    worker_parser.add_argument("--case", choices=CASES, required=True)
    worker_parser.add_argument("--seed", type=int, required=True)
    worker_parser.add_argument("--enabled", type=int, choices=(0, 1), required=True)
    worker_parser.add_argument("--variant", choices=VARIANTS, required=True)
    worker_parser.add_argument("--horizon", type=int, required=True)
    worker_parser.add_argument("--profile-invocations", type=int, required=True)
    worker_parser.add_argument("--original-root", type=Path, required=True)
    worker_parser.add_argument("--expected-gpu-uuid", required=True)
    worker_parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        _validate_controller_args(args)
        print(
            json.dumps(
                {
                    "schema": CONTROLLER_SCHEMA,
                    "torch_imported": "torch" in sys.modules,
                    "plan": _build_plan(args),
                    "limits": {
                        "horizon_calls": args.horizon,
                        "total_budget_seconds": args.total_budget_seconds,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run":
        return _controller(args)
    if args.command == "_worker":
        return _worker(args)
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
