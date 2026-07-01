#!/usr/bin/env python3
"""Isolated subprocess runner for benchmarks.

This script receives benchmark configuration via stdin JSON and runs the benchmark
in complete isolation from the parent process. This prevents CUDA context corruption
that can occur when forking after CUDA initialization.

Protocol:
- Input (stdin JSON):
  {
    "benchmark_module_path": "/path/to/benchmark.py",
    "benchmark_class_name": "MyBenchmark" | "get_benchmark",
    "config_dict": {...},
    "device": "cuda:0" | null,
    "initial_state": {...} | null
  }
  
- Output (stdout JSON):
  {
    "success": true/false,
    "result_json": "<serialized PydanticBenchmarkResult>",
    "errors": [...]
  }
"""

from __future__ import annotations

import copy
import gc
import io
import json
import os
import signal
import statistics
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from core.harness.triton_cache_utils import (
    get_triton_runtime_env_overrides,
    reset_triton_runtime_cache,
)


def _emit_runner_warning(message: str, **details: Any) -> None:
    payload: Dict[str, Any] = {"event": "isolated_runner_warning", "message": message}
    payload.update(details)
    try:
        print(json.dumps(payload, default=str), file=sys.stderr)
    except Exception:
        try:
            print(f"[isolated_runner_warning] {message}", file=sys.stderr)
        except Exception:
            pass


def _apply_owner_markers_from_argv(argv: List[str]) -> None:
    """Mirror owner markers from argv into the process environment.

    `/proc/<pid>/environ` can be transiently unreadable while a process is still
    visible to NVML. Passing the owner markers in argv gives the validator a
    second, exec-time-readable ownership channel for same-run worker detection.
    """
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--aisp-owner-run-id" and i + 1 < len(argv):
            os.environ["AISP_BENCHMARK_OWNER_RUN_ID"] = argv[i + 1]
            i += 2
            continue
        if token == "--aisp-owner-pid" and i + 1 < len(argv):
            os.environ["AISP_BENCHMARK_OWNER_PID"] = argv[i + 1]
            i += 2
            continue
        i += 1


def _device_may_use_cuda(device: Optional[str]) -> bool:
    if device is None:
        return True
    return str(device).split(":", maxsplit=1)[0].lower() == "cuda"


def _cuda_runtime_initialized(torch_module: Any) -> bool:
    is_initialized = getattr(getattr(torch_module, "cuda", None), "is_initialized", None)
    if not callable(is_initialized):
        # Older/fake torch modules used by tests lack this probe; keep legacy cleanup.
        return True
    try:
        return bool(is_initialized())
    except Exception as exc:
        _emit_runner_warning("Failed to query CUDA initialization state", error=str(exc))
        return True


def reset_cuda_state(device: Optional[str] = None) -> None:
    """Reset CUDA state before benchmark to ensure clean environment."""
    if not _device_may_use_cuda(device):
        gc.collect()
        return

    try:
        import torch
        if _cuda_runtime_initialized(torch) and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            # Reset CUDA graph pool
            if hasattr(torch.cuda, 'graph_pool_trim'):
                try:
                    torch.cuda.graph_pool_trim()
                except Exception as exc:
                    _emit_runner_warning("Failed to reset CUDA graph pool", error=str(exc))
            
            # Reset CUDA RNG state
            try:
                device_idx = torch.cuda.current_device()
                gen = torch.cuda.default_generators[device_idx]
                gen.set_offset(0)
                gen.manual_seed(0)
            except Exception as exc:
                _emit_runner_warning("Failed to reset CUDA RNG state", error=str(exc))
            
            # Reset dynamo/inductor state
            try:
                torch._dynamo.reset()
            except Exception as exc:
                _emit_runner_warning("Failed to reset torch._dynamo state", error=str(exc))
            
            try:
                torch._inductor.cudagraph_trees.reset_cudagraph_trees()
            except Exception as exc:
                _emit_runner_warning("Failed to reset torch._inductor cudagraph trees", error=str(exc))
        reset_triton_runtime_cache(
            lambda message, exc: _emit_runner_warning(message, error=str(exc))
        )
    except ImportError as exc:
        _emit_runner_warning(
            "PyTorch import unavailable during isolated runner CUDA reset",
            error=str(exc),
        )
    
    gc.collect()


def _safe_read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _collect_descendant_pids(root_pid: int) -> List[int]:
    parent_to_children: Dict[int, Set[int]] = {}
    try:
        proc_entries = list(Path("/proc").iterdir())
    except Exception:
        return []

    for proc_dir in proc_entries:
        if not proc_dir.name.isdigit():
            continue
        stat_text = _safe_read_text(proc_dir / "stat")
        if not stat_text:
            continue
        rparen = stat_text.rfind(")")
        if rparen < 0:
            continue
        tail = stat_text[rparen + 1 :].strip().split()
        if len(tail) < 2:
            continue
        try:
            pid = int(proc_dir.name)
            ppid = int(tail[1])
        except ValueError:
            continue
        parent_to_children.setdefault(ppid, set()).add(pid)

    descendants: List[int] = []
    stack: List[int] = [int(root_pid)]
    seen: Set[int] = {int(root_pid)}
    while stack:
        parent = stack.pop()
        for child in parent_to_children.get(parent, set()):
            if child in seen:
                continue
            seen.add(child)
            descendants.append(child)
            stack.append(child)
    descendants.sort()
    return descendants


def _signal_pids(pids: Iterable[int], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(int(pid), sig)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def _wait_for_exit(pids: Iterable[int], timeout_seconds: float) -> Set[int]:
    pending: Set[int] = {int(pid) for pid in pids}
    deadline = time.time() + max(timeout_seconds, 0.0)
    while pending and time.time() < deadline:
        exited: Set[int] = set()
        for pid in pending:
            if not Path(f"/proc/{pid}").exists():
                exited.add(pid)
        pending -= exited
        if pending:
            time.sleep(0.05)
    return pending


def _reap_descendant_processes(grace_seconds: float = 5.0) -> None:
    """Aggressively reap child processes before exiting this isolated runner."""
    root_pid = os.getpid()
    descendants = _collect_descendant_pids(root_pid)
    if not descendants:
        return

    _signal_pids(descendants, signal.SIGTERM)
    remaining = _wait_for_exit(descendants, timeout_seconds=grace_seconds)
    if remaining:
        _signal_pids(sorted(remaining), signal.SIGKILL)
        _wait_for_exit(sorted(remaining), timeout_seconds=2.0)


def run_benchmark(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run benchmark and return results in harness-expected format."""
    import importlib.util
    from core.harness.verify_output_codec import serialize_verify_tensor
    
    def _execute() -> Dict[str, Any]:
        """Execute a single benchmark inside an isolated subprocess.

        CRITICAL: This subprocess must run the *same* BenchmarkHarness timing path
        as in-process execution to keep all validity protections identical.
        """
        errors: List[str] = []

        # Extract input
        module_path = Path(input_data["benchmark_module_path"])
        class_name = input_data["benchmark_class_name"]
        config_dict = dict(input_data.get("config_dict", {}) or {})
        device_str = input_data.get("device")
        initial_state = input_data.get("initial_state")
        mode_str = input_data.get("mode") or config_dict.pop("mode", None)
        verify_output_path = input_data.get("verify_output_path")
        verify_output_max_bytes = int(input_data.get("verify_output_max_bytes", 0) or 0)

        benchmark_name = class_name

        try:
            # Reset CUDA state BEFORE loading the module
            reset_cuda_state(device_str)

            # Load module
            spec = importlib.util.spec_from_file_location("benchmark_module", str(module_path))
            if spec is None or spec.loader is None:
                errors.append(f"Failed to load module spec from {module_path}")
                return _make_error_response(errors)

            module = importlib.util.module_from_spec(spec)
            sys.modules["benchmark_module"] = module
            spec.loader.exec_module(module)

            # Get benchmark instance
            if class_name == "get_benchmark":
                if not hasattr(module, "get_benchmark"):
                    errors.append(f"Module {module_path} has no get_benchmark() function")
                    return _make_error_response(errors)
                benchmark = module.get_benchmark()
                benchmark_name = getattr(benchmark, "name", None) or benchmark.__class__.__name__
            else:
                if not hasattr(module, class_name):
                    errors.append(f"Module {module_path} has no class {class_name}")
                    return _make_error_response(errors)
                benchmark_class = getattr(module, class_name)
                benchmark = benchmark_class()
                benchmark_name = class_name

            # Apply initial state if provided
            if initial_state:
                for key, value in initial_state.items():
                    if hasattr(benchmark, key):
                        setattr(benchmark, key, value)

            # Build harness config from parent dict, but never recurse into subprocess/torchrun
            from core.harness.benchmark_harness import (
                BenchmarkConfig,
                BenchmarkHarness,
                BenchmarkMode,
                ExecutionMode,
                LaunchVia,
            )

            config = BenchmarkConfig(**config_dict)
            config.use_subprocess = False
            config.execution_mode = ExecutionMode.THREAD
            config.launch_via = LaunchVia.PYTHON
            config._sync_execution_mode()
            config._sync_launch_via()

            mode = BenchmarkMode(mode_str) if mode_str else BenchmarkMode.CUSTOM
            harness = BenchmarkHarness(mode=mode, config=config)

            # Capture verification artifacts at teardown-time to preserve the exact
            # timing-run state before benchmark.teardown() mutates benchmark fields.
            capture_state: Dict[str, Any] = {
                "captured": False,
                "error": None,
                "verify_output": None,
                "output_tolerance": None,
                "input_signature": None,
            }

            original_teardown = benchmark.teardown

            def _capture_verification_artifacts() -> None:
                if capture_state["captured"]:
                    return
                try:
                    verify_output_obj = benchmark.get_verify_output()
                    output_tol_obj = benchmark.get_output_tolerance()
                    signature_obj = benchmark.get_input_signature()

                    import torch  # local import after module load

                    if isinstance(verify_output_obj, torch.Tensor):
                        capture_state["verify_output"] = verify_output_obj.detach().cpu().clone()
                    elif isinstance(verify_output_obj, dict):
                        tensor_map: Dict[str, torch.Tensor] = {}
                        for name, tensor in verify_output_obj.items():
                            if not isinstance(tensor, torch.Tensor):
                                raise TypeError(
                                    f"verify_output['{name}'] must be a torch.Tensor, got {type(tensor)}"
                                )
                            tensor_map[name] = tensor.detach().cpu().clone()
                        capture_state["verify_output"] = tensor_map
                    else:
                        raise TypeError(
                            "get_verify_output() must return torch.Tensor or Dict[str, torch.Tensor], "
                            f"got {type(verify_output_obj)}"
                        )

                    capture_state["output_tolerance"] = (
                        float(output_tol_obj[0]),
                        float(output_tol_obj[1]),
                    )
                    capture_state["input_signature"] = (
                        signature_obj.to_dict()
                        if hasattr(signature_obj, "to_dict")
                        else copy.deepcopy(signature_obj)
                    )
                    capture_state["captured"] = True
                except Exception as exc:
                    capture_state["error"] = f"Failed to capture verification artifacts pre-teardown: {exc}"

            def _wrapped_teardown() -> None:
                _capture_verification_artifacts()
                original_teardown()

            benchmark.teardown = _wrapped_teardown  # type: ignore[method-assign]

            # Run through the real harness (includes all protections)
            bench_result = harness.benchmark(benchmark, name=benchmark_name)
            bench_result.runtime_env.update(get_triton_runtime_env_overrides())

            # If the benchmark already failed, propagate its errors without
            # attempting verification extraction (avoid masking root cause).
            if bench_result.errors:
                return {
                    "success": False,
                    "result_json": bench_result.model_dump_json(),
                    "errors": bench_result.errors,
                }

            if not capture_state["captured"]:
                _capture_verification_artifacts()
            if capture_state["error"]:
                raise RuntimeError(capture_state["error"])

            # Strictly extract verification artifacts captured from the timing run
            verify_output = capture_state["verify_output"]
            output_tol = capture_state["output_tolerance"]
            signature = capture_state["input_signature"]
            import torch  # local import after module load

            def _tensor_nbytes(t: "torch.Tensor") -> int:
                return int(t.numel() * t.element_size())

            def _dict_nbytes(tensors: Dict[str, "torch.Tensor"]) -> int:
                return int(sum(_tensor_nbytes(t) for t in tensors.values()))

            if isinstance(verify_output, torch.Tensor):
                if verify_output_max_bytes and _tensor_nbytes(verify_output) > verify_output_max_bytes:
                    if not verify_output_path:
                        raise RuntimeError(
                            "verify_output_path required when verify_output exceeds max bytes"
                        )
                    torch.save(verify_output.detach().cpu(), verify_output_path)
                    verify_output_data = {"kind": "tensor_file", "path": verify_output_path}
                else:
                    verify_output_data = {"kind": "tensor", **serialize_verify_tensor(verify_output)}
            elif isinstance(verify_output, dict):
                tensor_map: Dict[str, torch.Tensor] = {}
                for name, tensor in verify_output.items():
                    if not isinstance(tensor, torch.Tensor):
                        raise TypeError(f"verify_output['{name}'] must be a torch.Tensor, got {type(tensor)}")
                    tensor_map[name] = tensor
                if verify_output_max_bytes and _dict_nbytes(tensor_map) > verify_output_max_bytes:
                    if not verify_output_path:
                        raise RuntimeError(
                            "verify_output_path required when verify_output exceeds max bytes"
                        )
                    torch.save({name: t.detach().cpu() for name, t in tensor_map.items()}, verify_output_path)
                    verify_output_data = {"kind": "dict_file", "path": verify_output_path}
                else:
                    tensors: Dict[str, Any] = {}
                    for name, tensor in tensor_map.items():
                        tensors[name] = serialize_verify_tensor(tensor)
                    verify_output_data = {"kind": "dict", "tensors": tensors}
            else:
                raise TypeError(
                    f"get_verify_output() must return torch.Tensor or Dict[str, torch.Tensor], got {type(verify_output)}"
                )

            result_payload: Dict[str, Any] = {
                "success": True,
                "result_json": bench_result.model_dump_json(),
                "verify_output": verify_output_data,
                "output_tolerance": {"rtol": float(output_tol[0]), "atol": float(output_tol[1])},
                "input_signature": signature,
                "errors": bench_result.errors or [],
            }
            return result_payload

        except Exception as e:
            tb = traceback.format_exc()
            errors.append(f"Benchmark execution failed: {e}")
            errors.append(tb)
            return _make_error_response(errors)
    
    stdout_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer):
        result = _execute()
    captured = stdout_buffer.getvalue().strip()
    if captured:
        try:
            lines = [ln for ln in captured.splitlines() if ln]
            print(json.dumps({"event": "benchmark_stdout", "lines": lines}), file=sys.stderr)
        except Exception:
            print(captured, file=sys.stderr)
    return result


def _make_error_response(errors: List[str], seed_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create error response in harness-expected format."""
    # Build a minimal BenchmarkResult with errors
    from core.benchmark.models import BenchmarkResult, TimingStats
    
    result = BenchmarkResult(
        timing=TimingStats(
            mean_ms=0.0,
            median_ms=0.0,
            std_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            iterations=0,
            warmup_iterations=0,
            raw_times_ms=[],
        ),
        errors=errors,
        seeds=seed_info,
    )
    
    return {
        "success": False,
        "result_json": result.model_dump_json(),
        "errors": errors,
    }


def _make_success_response(
    times_ms: List[float],
    iterations: int,
    warmup: int,
    memory_peak_mb: Optional[float],
    memory_allocated_mb: Optional[float],
    benchmark_name: str,
    device_str: Optional[str],
    inference_timing_data: Optional[Dict[str, List[float]]],
    verify_output_data: Optional[Dict[str, Any]],
    errors: List[str],
    seed_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create success response in harness-expected format."""
    from core.benchmark.models import BenchmarkResult, TimingStats, MemoryStats, InferenceTimingStats
    
    # Calculate timing statistics
    if times_ms:
        mean_ms = statistics.mean(times_ms)
        median_ms = statistics.median(times_ms)
        std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        min_ms = min(times_ms)
        max_ms = max(times_ms)
    else:
        mean_ms = median_ms = std_ms = min_ms = max_ms = 0.0
    
    # Build timing stats
    timing = TimingStats(
        mean_ms=mean_ms,
        median_ms=median_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        iterations=iterations,
        warmup_iterations=warmup,
        raw_times_ms=times_ms,
    )
    
    # Build memory stats
    memory = None
    if memory_peak_mb is not None:
        memory = MemoryStats(
            peak_mb=memory_peak_mb,
            allocated_mb=memory_allocated_mb,
        )
    
    # Build inference timing stats
    inference_timing = None
    if inference_timing_data:
        inference_timing = InferenceTimingStats(**inference_timing_data)
    
    # Build full result
    result = BenchmarkResult(
        timing=timing,
        memory=memory,
        inference_timing=inference_timing,
        benchmark_name=benchmark_name,
        device=device_str,
        errors=errors,
        seeds=seed_info,
    )
    
    return {
        "success": True,
        "result_json": result.model_dump_json(),
        "verify_output": verify_output_data,
        "errors": errors,
    }


def main() -> None:
    """Main entry point - read JSON from stdin, run benchmark, write JSON to stdout."""
    try:
        _apply_owner_markers_from_argv(sys.argv[1:])
        # Read input JSON from stdin
        input_json = sys.stdin.read()
        input_data = json.loads(input_json)
        
        # Run benchmark
        result = run_benchmark(input_data)
        
        # Write result JSON to stdout
        print(json.dumps(result))
        
    except json.JSONDecodeError as e:
        error_result = _make_error_response([f"Failed to parse input JSON: {e}"])
        print(json.dumps(error_result))
        sys.exit(1)
    except Exception as e:
        error_result = _make_error_response([f"Runner failed: {e}", traceback.format_exc()])
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == "__main__":
    main()
