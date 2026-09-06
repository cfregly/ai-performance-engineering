"""Benchmark Validity Checks.

This module provides utilities to detect and prevent benchmark validity issues
that could lead to misleading performance measurements.

Categories covered:
- Memory manipulation (aliasing, pre-allocation)
- Work relocation (setup pre-computation, lazy evaluation)
- Environment issues (GPU state, GC interference)
- Compilation issues (cache, recompilation)

"""

from __future__ import annotations

import gc
import hashlib
import os
import re
import statistics
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from core.hot_path_checks import (
    check_benchmark_fn_antipatterns,
    check_benchmark_fn_sync_calls,
)
from core.utils.logger import get_logger

try:
    import torch
except ImportError:
    torch = None  # type: ignore


logger = get_logger(__name__)
_EMITTED_VALIDITY_LIMITATIONS: Set[str] = set()
_VALIDITY_LIMITATION_RECORDS: Dict[str, "RuntimeCapabilityLimitationRecord"] = {}


def _emit_validity_warning(message: str, *, exc: Optional[BaseException] = None) -> None:
    """Surface degraded validity checks through logs and runtime warnings."""
    detail = f"{message}: {exc}" if exc is not None else message
    logger.warning(detail)
    warnings.warn(detail, RuntimeWarning, stacklevel=2)


def _emit_validity_limitation_once(
    key: str,
    message: str,
    *,
    exc: Optional[BaseException] = None,
    component: str = "validity_checks",
    category: str = "runtime_capability_gap",
) -> None:
    """Surface known runtime capability gaps once per process."""
    if key in _EMITTED_VALIDITY_LIMITATIONS:
        return
    _EMITTED_VALIDITY_LIMITATIONS.add(key)
    _VALIDITY_LIMITATION_RECORDS[key] = RuntimeCapabilityLimitationRecord(
        key=key,
        category=category,
        component=component,
        summary=message,
        detail=str(exc) if exc is not None else None,
        first_observed_at=datetime.now(timezone.utc).isoformat(),
    )
    _emit_validity_warning(message, exc=exc)


@dataclass(frozen=True)
class RuntimeCapabilityLimitationRecord:
    """Structured record of a known runtime capability limitation."""

    key: str
    category: str
    component: str
    summary: str
    detail: Optional[str]
    first_observed_at: str

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "key": self.key,
            "category": self.category,
            "component": self.component,
            "summary": self.summary,
            "detail": self.detail,
            "first_observed_at": self.first_observed_at,
        }


def get_runtime_capability_limitations() -> List[Dict[str, Optional[str]]]:
    """Return structured runtime capability limitations observed in this process."""
    return [
        _VALIDITY_LIMITATION_RECORDS[key].to_dict()
        for key in sorted(_VALIDITY_LIMITATION_RECORDS)
    ]


def _is_cuda_release_pool_unsupported(exc: BaseException) -> bool:
    detail = str(exc)
    return (
        "_cuda_releasePool" in detail
        and "incompatible function arguments" in detail
    )


def _is_allocator_reset_setting_unsupported(exc: BaseException) -> bool:
    detail = str(exc)
    return "Unrecognized CachingAllocator option: reset_allocator" in detail


# =============================================================================
# GPU State Monitoring
# =============================================================================

@dataclass
class GPUState:
    """Snapshot of GPU state for validity checking."""
    device_index: int
    device_name: str
    temperature_c: Optional[float] = None
    clock_mhz: Optional[int] = None
    memory_clock_mhz: Optional[int] = None
    power_draw_w: Optional[float] = None
    memory_used_mb: Optional[float] = None
    memory_total_mb: Optional[float] = None
    throttle_reason: Optional[str] = None
    power_limit_w: Optional[float] = None
    device_uuid: Optional[str] = None


def capture_gpu_state(device_index: int = 0) -> GPUState:
    """Capture current GPU state for comparison.
    """
    if torch is None or not torch.cuda.is_available():
        return GPUState(device_index=device_index, device_name="N/A")
    
    props = torch.cuda.get_device_properties(device_index)
    state = GPUState(
        device_index=device_index,
        device_name=props.name,
    )

    state.memory_used_mb = torch.cuda.memory_allocated(device_index) / (1024 * 1024)
    state.memory_total_mb = props.total_memory / (1024 * 1024)

    try:
        import pynvml
    except ImportError as exc:
        raise RuntimeError(
            "capture_gpu_state requires pynvml (nvidia-ml-py) when CUDA is available."
        ) from exc

    # Collect detailed state via NVML (fail-fast on any NVML error).
    from core.profiling.gpu_telemetry import (
        normalize_gpu_uuid,
        resolve_nvml_device_handle,
    )

    pynvml.nvmlInit()
    try:
        cuda_device_uuid = normalize_gpu_uuid(getattr(props, "uuid", None))
        handle = resolve_nvml_device_handle(
            pynvml,
            device_index,
            cuda_device_uuid=cuda_device_uuid,
        )
        state.device_uuid = normalize_gpu_uuid(pynvml.nvmlDeviceGetUUID(handle))
        if cuda_device_uuid is not None and state.device_uuid != cuda_device_uuid:
            raise RuntimeError(
                "NVML device identity does not match the requested logical CUDA device: "
                f"cuda={cuda_device_uuid!r}, nvml={state.device_uuid!r}"
            )

        state.temperature_c = float(
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        )

        state.clock_mhz = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
        state.memory_clock_mhz = int(
            pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        )

        state.power_draw_w = float(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        state.power_limit_w = float(
            pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        )

        throttle = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        if throttle != 0:
            reasons = []
            if throttle & 0x1:
                reasons.append("GpuIdle")
            if throttle & 0x2:
                reasons.append("ApplicationsClocks")
            if throttle & 0x4:
                reasons.append("SwPowerCap")
            if throttle & 0x8:
                reasons.append("HwSlowdown")
            if throttle & 0x10:
                reasons.append("SyncBoost")
            if throttle & 0x20:
                reasons.append("SwThermalSlowdown")
            if throttle & 0x40:
                reasons.append("HwThermalSlowdown")
            if throttle & 0x80:
                reasons.append("HwPowerBrakeSlowdown")
            state.throttle_reason = ",".join(reasons) if reasons else None
    finally:
        pynvml.nvmlShutdown()
    
    return state


def check_gpu_state_consistency(before: GPUState, after: GPUState, 
                                  temp_threshold: float = 10.0,
                                  clock_threshold_pct: float = 10.0) -> Tuple[bool, List[str]]:
    """Classify significant changes between two GPU diagnostic snapshots.

    BenchmarkHarness currently emits returned reasons as ``RuntimeWarning`` and
    does not reject the benchmark. Its before snapshot precedes warmup, so an
    upward current-clock change can represent normal boost behavior.
    
    Returns:
        Tuple of (is_consistent, list_of_warnings)
    """
    warnings_list: List[str] = []
    
    # Temperature increase
    if before.temperature_c is not None and after.temperature_c is not None:
        temp_delta = after.temperature_c - before.temperature_c
        if temp_delta > temp_threshold:
            warnings_list.append(
                f"GPU temperature increased {temp_delta:.1f}°C during benchmark "
                f"({before.temperature_c}°C → {after.temperature_c}°C)"
            )
    
    # Clock speed change
    if before.clock_mhz is not None and after.clock_mhz is not None:
        if before.clock_mhz > 0:
            clock_change_pct = (
                (after.clock_mhz - before.clock_mhz) / before.clock_mhz * 100
            )
            if abs(clock_change_pct) > clock_threshold_pct:
                if clock_change_pct < 0:
                    warnings_list.append(
                        f"GPU clock dropped {abs(clock_change_pct):.1f}% during benchmark "
                        f"({before.clock_mhz}MHz → {after.clock_mhz}MHz) - possible thermal throttling"
                    )
                else:
                    warnings_list.append(
                        f"GPU clock increased {clock_change_pct:.1f}% during benchmark "
                        f"({before.clock_mhz}MHz → {after.clock_mhz}MHz) - "
                        "current-clock diagnostic; the pre-warmup snapshot may normally boost"
                    )

    # Configured power-limit change
    if before.power_limit_w is not None and after.power_limit_w is not None:
        if before.power_limit_w != after.power_limit_w:
            warnings_list.append(
                f"GPU power limit changed during benchmark "
                f"({before.power_limit_w:.1f}W → {after.power_limit_w:.1f}W)"
            )
    
    # Throttling detected
    if after.throttle_reason and "Thermal" in after.throttle_reason:
        warnings_list.append(f"GPU throttling detected: {after.throttle_reason}")
    
    return len(warnings_list) == 0, warnings_list


# =============================================================================
# Backend Precision Policy Guard
# =============================================================================

@dataclass(frozen=True)
class PrecisionPolicyState:
    """Snapshot of backend precision policy flags."""
    matmul_allow_tf32: Optional[bool] = None
    cudnn_allow_tf32: Optional[bool] = None
    float32_matmul_precision: Optional[str] = None
    allow_fp16_reduced_precision_reduction: Optional[bool] = None
    allow_bf16_reduced_precision_reduction: Optional[bool] = None
    cudnn_benchmark: Optional[bool] = None
    cudnn_deterministic: Optional[bool] = None
    deterministic_algorithms: Optional[bool] = None


def capture_precision_policy_state() -> PrecisionPolicyState:
    """Capture backend precision policy settings for consistency checks."""
    if torch is None:
        return PrecisionPolicyState()

    matmul_allow_tf32 = None
    allow_fp16_reduced_precision_reduction = None
    allow_bf16_reduced_precision_reduction = None
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        matmul = torch.backends.cuda.matmul
        if hasattr(matmul, "allow_tf32"):
            matmul_allow_tf32 = bool(matmul.allow_tf32)
        if hasattr(matmul, "allow_fp16_reduced_precision_reduction"):
            allow_fp16_reduced_precision_reduction = bool(matmul.allow_fp16_reduced_precision_reduction)
        if hasattr(matmul, "allow_bf16_reduced_precision_reduction"):
            allow_bf16_reduced_precision_reduction = bool(matmul.allow_bf16_reduced_precision_reduction)

    cudnn_allow_tf32 = None
    cudnn_benchmark = None
    cudnn_deterministic = None
    if hasattr(torch.backends, "cudnn"):
        cudnn = torch.backends.cudnn
        if hasattr(cudnn, "allow_tf32"):
            cudnn_allow_tf32 = bool(cudnn.allow_tf32)
        cudnn_benchmark = bool(cudnn.benchmark)
        cudnn_deterministic = bool(cudnn.deterministic)

    float32_matmul_precision = None
    if hasattr(torch, "get_float32_matmul_precision"):
        float32_matmul_precision = torch.get_float32_matmul_precision()

    deterministic_algorithms = None
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        deterministic_algorithms = bool(torch.are_deterministic_algorithms_enabled())

    return PrecisionPolicyState(
        matmul_allow_tf32=matmul_allow_tf32,
        cudnn_allow_tf32=cudnn_allow_tf32,
        float32_matmul_precision=float32_matmul_precision,
        allow_fp16_reduced_precision_reduction=allow_fp16_reduced_precision_reduction,
        allow_bf16_reduced_precision_reduction=allow_bf16_reduced_precision_reduction,
        cudnn_benchmark=cudnn_benchmark,
        cudnn_deterministic=cudnn_deterministic,
        deterministic_algorithms=deterministic_algorithms,
    )


def check_precision_policy_consistency(
    before: PrecisionPolicyState,
    after: PrecisionPolicyState,
) -> Tuple[bool, List[str]]:
    """Detect changes in backend precision policy settings."""
    warnings_list: List[str] = []
    for entry in fields(PrecisionPolicyState):
        name = entry.name
        before_val = getattr(before, name)
        after_val = getattr(after, name)
        if before_val is None or after_val is None:
            continue
        if before_val != after_val:
            warnings_list.append(f"{name} changed from {before_val} to {after_val}")
    return len(warnings_list) == 0, warnings_list


# =============================================================================
# Memory Validity Checks
# =============================================================================

def get_tensor_addresses(tensors: Dict[str, Any]) -> Dict[str, int]:
    """Get memory addresses of tensors for aliasing detection.
    
    Args:
        tensors: Dict of tensor name -> tensor
        
    Returns:
        Dict of tensor name -> data_ptr address
    """
    addresses = {}
    for name, tensor in tensors.items():
        if torch is not None and isinstance(tensor, torch.Tensor):
            addresses[name] = tensor.data_ptr()
    return addresses


def check_input_output_aliasing(input_tensors: Dict[str, Any],
                                  output_tensors: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Check if any output tensor aliases an input tensor.
    
    This detects the "pre-filled output" cheat where the output buffer
    already contains the result before benchmark_fn() runs.
    
    Returns:
        Tuple of (no_aliasing, error_message_if_aliasing)
    """
    input_addrs = get_tensor_addresses(input_tensors)
    output_addrs = get_tensor_addresses(output_tensors)
    
    for out_name, out_addr in output_addrs.items():
        for in_name, in_addr in input_addrs.items():
            if out_addr == in_addr:
                return False, (
                    f"OUTPUT ALIASING DETECTED: Output '{out_name}' has same memory address "
                    f"as input '{in_name}' (0x{out_addr:x}). This may indicate pre-filled results."
                )
    
    return True, None


def reset_cuda_memory_pool(device: Optional[Any] = None) -> None:
    """Reset CUDA memory pool to prevent memory reuse gaming.
    
    This ensures each benchmark run starts with a clean memory state,
    preventing cached allocations from skewing timing.
    """
    if torch is None or not torch.cuda.is_available():
        return
    
    # Full reset: sync, drop Python refs, flush allocator caches/stats
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception as exc:
        _emit_validity_warning(
            "Failed to collect CUDA IPC allocations while resetting the memory pool; allocator state may leak across benchmark runs",
            exc=exc,
        )
    
    # Best-effort allocator cache reset across PyTorch APIs
    try:
        if hasattr(torch, "_C") and hasattr(torch._C, "_cuda_releasePool"):
            torch._C._cuda_releasePool()
    except Exception as exc:
        if _is_cuda_release_pool_unsupported(exc):
            _emit_validity_limitation_once(
                "cuda_release_pool_signature",
                "This PyTorch runtime exposes torch._C._cuda_releasePool without a zero-argument reset entrypoint; memory-pool reset will fall back to cache/IPC/stat cleanup and allocator reuse checks are partially degraded",
                exc=exc,
                component="cuda_memory_pool_reset",
                category="allocator_cleanup",
            )
        else:
            _emit_validity_warning(
                "Failed to release the CUDA allocator pool during memory-pool reset; cached allocations may remain visible to later runs",
                exc=exc,
            )
    try:
        if hasattr(torch, "_C") and hasattr(torch._C, "_accelerator_setAllocatorSettings"):
            torch._C._accelerator_setAllocatorSettings("reset_allocator:True")
        elif hasattr(torch, "_C") and hasattr(torch._C, "_cuda_cudaCachingAllocator_set_allocator_settings"):
            torch._C._cuda_cudaCachingAllocator_set_allocator_settings("reset_allocator:True")
        elif hasattr(torch.cuda, "memory") and hasattr(torch.cuda.memory, "_set_allocator_settings"):
            torch.cuda.memory._set_allocator_settings("reset_allocator:True")
    except Exception as exc:
        if _is_allocator_reset_setting_unsupported(exc):
            _emit_validity_limitation_once(
                "cuda_allocator_reset_setting",
                "This PyTorch runtime does not support the reset_allocator allocator setting; memory-pool cleanup will proceed without forcing allocator-setting resets and allocator reuse checks are partially degraded",
                exc=exc,
                component="cuda_memory_pool_reset",
                category="allocator_cleanup",
            )
        else:
            _emit_validity_warning(
                "Failed to reset CUDA allocator settings during memory-pool cleanup; allocator reuse checks may be degraded",
                exc=exc,
            )
    
    # Reset allocator statistics
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats(device)
    if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
        torch.cuda.reset_accumulated_memory_stats(device)
    try:
        if hasattr(torch, "_C") and hasattr(torch._C, "_cuda_resetPeakHostMemoryStats"):
            torch._C._cuda_resetPeakHostMemoryStats()
    except Exception as exc:
        _emit_validity_warning(
            "Failed to reset CUDA host-memory statistics during memory-pool cleanup; post-run memory telemetry may be stale",
            exc=exc,
        )
    torch.cuda.synchronize(device)
    if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
        torch.cuda.reset_accumulated_memory_stats(device)
    gc.collect()
    torch.cuda.synchronize(device)


# =============================================================================
# Setup Pre-computation Detection
# =============================================================================

def hash_tensors(tensors: Dict[str, Any]) -> str:
    """Compute a hash of tensor contents for change detection.
    
    Used to detect if setup() pre-computes results by modifying tensors.
    """
    if torch is None:
        return "no_torch"
    
    hasher = hashlib.sha256()
    for name in sorted(tensors.keys()):
        tensor = tensors[name]
        if isinstance(tensor, torch.Tensor):
            # Hash tensor metadata and content
            hasher.update(name.encode())
            hasher.update(str(tensor.shape).encode())
            hasher.update(str(tensor.dtype).encode())
            hasher.update(str(tensor.device).encode())
            # Sample content (full hash would be too slow)
            if tensor.numel() > 0:
                flat = tensor.flatten()
                # Hash first, last, and middle elements
                samples = [flat[0].item(), flat[-1].item()]
                if tensor.numel() > 2:
                    samples.append(flat[tensor.numel() // 2].item())
                hasher.update(str(samples).encode())
    
    return hasher.hexdigest()[:16]


def check_setup_precomputation(get_outputs_fn, setup_fn) -> Tuple[bool, Optional[str]]:
    """Check if setup() pre-computes results.
    
    Args:
        get_outputs_fn: Function that returns dict of output tensors
        setup_fn: The benchmark's setup() function
        
    Returns:
        Tuple of (no_precomputation, error_message_if_detected)
    """
    # Get outputs before setup
    try:
        outputs_before = get_outputs_fn()
        hash_before = hash_tensors(outputs_before)
    except Exception:
        # Can't check if outputs not available
        return True, None
    
    # Run setup
    setup_fn()
    
    # Get outputs after setup
    try:
        outputs_after = get_outputs_fn()
        hash_after = hash_tensors(outputs_after)
    except Exception:
        return True, None
    
    if hash_before != hash_after:
        return False, (
            "SETUP PRE-COMPUTATION DETECTED: Output tensors changed during setup(). "
            f"Hash before: {hash_before}, after: {hash_after}. "
            "setup() should only initialize inputs (and may prepare steady-state state like CUDA graph capture), "
            "but must not compute and stash the final outputs used for timing/verification."
        )
    
    return True, None


# =============================================================================
# Garbage Collection Control
# =============================================================================

@contextmanager
def gc_disabled():
    """Context manager to disable garbage collection during timing.
    
    Prevents GC from interfering with timing measurements.
    
    Example:
        with gc_disabled():
            # GC won't run during this block
            results = benchmark_fn()
    """
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if gc_was_enabled:
            gc.enable()


# =============================================================================
# torch.compile Cache Management
# =============================================================================

def clear_compile_cache() -> bool:
    """Clear torch.compile/dynamo cache for consistent compilation.
    
    Returns:
        True if cache was cleared, False if not available
    """
    if torch is None:
        return False
    
    try:
        # Reset dynamo
        if hasattr(torch, '_dynamo'):
            torch._dynamo.reset()
        
        # Reset inductor cache if available
        if hasattr(torch, '_inductor') and hasattr(torch._inductor, 'codecache'):
            if hasattr(torch._inductor.codecache, 'clear'):
                torch._inductor.codecache.clear()
        
        return True
    except Exception as exc:
        _emit_validity_warning(
            "Failed to clear torch.compile caches; subsequent runs may inherit prior Dynamo or Inductor state",
            exc=exc,
        )
        return False


def get_compile_state() -> Dict[str, Any]:
    """Report observed Dynamo compilation counters, not resident cache size.

    Modern Dynamo records compiled graphs in ``stats.unique_graphs``; older
    versions may expose ``compile.calls``. Counters are process cumulative and
    need before/after comparison. Zero with ``compile_count_available=False``
    means no supported counter was observed, not proof that nothing compiled.
    The legacy cache_entries value remains for compatibility but is unavailable.
    """
    state = {
        "dynamo_available": False,
        "compile_count": 0,
        "compile_count_available": False,
        "compile_count_source": None,
        "cache_entries": 0,
        "cache_entries_available": False,
    }
    
    if torch is None:
        return state
    
    try:
        if hasattr(torch, '_dynamo'):
            state["dynamo_available"] = True
            if hasattr(torch._dynamo, 'utils') and hasattr(torch._dynamo.utils, 'counters'):
                counters = torch._dynamo.utils.counters
                graph_count = counters.get("stats", {}).get("unique_graphs")
                legacy_count = counters.get("compile", {}).get("calls")
                if graph_count is not None:
                    state["compile_count"] = int(graph_count)
                    state["compile_count_source"] = "stats.unique_graphs"
                    state["compile_count_available"] = True
                elif legacy_count is not None:
                    state["compile_count"] = int(legacy_count)
                    state["compile_count_source"] = "compile.calls"
                    state["compile_count_available"] = True
    except Exception as exc:
        _emit_validity_warning(
            "Failed to inspect torch.compile state; compile-cache consistency checks may be incomplete",
            exc=exc,
        )
    
    return state


# =============================================================================
# Lazy Evaluation Detection
# =============================================================================

def force_tensor_evaluation(tensors: Dict[str, Any]) -> None:
    """Force evaluation of potentially lazy tensors.
    
    This prevents the "lazy evaluation skip" cheat where unevaluated
    tensors are returned without doing actual computation.
    """
    if torch is None:
        return

    materialization_failures: List[str] = []
    for name, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            # Sync ensures any pending operations complete
            if tensor.is_cuda:
                torch.cuda.synchronize(tensor.device)
            # .item() on a small tensor forces full evaluation
            # (we use a single element to minimize overhead)
            if tensor.numel() > 0:
                try:
                    _ = tensor.flatten()[0].item()
                except Exception as exc:
                    materialization_failures.append(f"{name}: {exc}")

    if materialization_failures:
        sample = ", ".join(materialization_failures[:3])
        extra = "" if len(materialization_failures) <= 3 else f" (+{len(materialization_failures) - 3} more)"
        _emit_validity_limitation_once(
            "force_tensor_evaluation_item_failure",
            (
                "force_tensor_evaluation could not materialize one or more tensors via .item(); "
                f"lazy-evaluation protection may be incomplete. Affected tensors: {sample}{extra}"
            ),
        )


# =============================================================================
# Environment Validation
# =============================================================================

@dataclass(frozen=True)
class EnvironmentValidationResult:
    """Structured result for environment validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]
    notices: List[str] = field(default_factory=list)


@dataclass
class EnvironmentProbe:
    """Probe host environment via a filesystem root.

    Tests can provide a synthetic root containing /proc and /sys snapshots.
    """

    root: Path = Path("/")
    env: Dict[str, str] = field(default_factory=lambda: dict(os.environ))
    cpu_affinity: Optional[Set[int]] = None
    probe_errors: List[str] = field(default_factory=list)

    def effective_env(self) -> Dict[str, str]:
        """Return probe env with dynamic ownership/runtime markers refreshed from the live process."""
        merged = dict(self.env)
        for key in (
            "AISP_BENCHMARK_OWNER_RUN_ID",
            "AISP_BENCHMARK_OWNER_PID",
            "AISP_FOREIGN_GPU_PROCESS_MIN_MB",
        ):
            current = os.environ.get(key)
            if current is not None and key not in merged:
                merged[key] = current
        return merged

    def resolve(self, path: str | Path) -> Path:
        rel = str(path)
        rel = rel.lstrip("/")
        return self.root / rel

    def exists(self, path: str | Path) -> bool:
        return self.resolve(path).exists()

    def read_text(self, path: str | Path) -> str:
        return self.resolve(path).read_text(encoding="utf-8").strip()

    def record_probe_error(self, context: str, exc: Exception) -> None:
        message = f"{context}: {exc}"
        if message not in self.probe_errors:
            self.probe_errors.append(message)

    def get_cpu_affinity(self) -> Optional[Set[int]]:
        if self.cpu_affinity is not None:
            return set(self.cpu_affinity)
        try:
            return set(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        except Exception as exc:
            self.record_probe_error("os.sched_getaffinity(0)", exc)
            return None


@dataclass(frozen=True)
class ExecutionEnvironment:
    """High-level host execution context for benchmark provenance."""

    kind: str
    virtualized: Optional[bool]
    dmi_product_name: Optional[str] = None


def _parse_int_set(spec: str) -> Set[int]:
    """Parse Linux list formats like '0-3,8,10-12' into a set of ints."""
    spec = spec.strip()
    if not spec:
        return set()
    result: Set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid range '{part}'")
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return result


def _read_optional(probe: EnvironmentProbe, path: str | Path) -> Optional[str]:
    resolved = probe.resolve(path)
    if not resolved.exists():
        return None
    try:
        return resolved.read_text(encoding="utf-8").strip()
    except Exception as exc:
        probe.record_probe_error(str(resolved), exc)
        return None


def detect_execution_environment(probe: Optional[EnvironmentProbe] = None) -> ExecutionEnvironment:
    """Detect whether the current host context is bare metal or virtualized."""
    probe = probe or EnvironmentProbe()
    cpuinfo = _read_optional(probe, "/proc/cpuinfo")
    product_name = _read_optional(probe, "/sys/devices/virtual/dmi/id/product_name")

    if cpuinfo is None and product_name is None:
        return ExecutionEnvironment(kind="unknown", virtualized=None, dmi_product_name=None)

    is_virtualized = False
    if cpuinfo is not None and "hypervisor" in cpuinfo.lower():
        is_virtualized = True
    if product_name is not None and any(
        tag in product_name.lower() for tag in ("qemu", "kvm", "vmware", "virtualbox", "hyper-v")
    ):
        is_virtualized = True

    return ExecutionEnvironment(
        kind="virtualized" if is_virtualized else "bare_metal",
        virtualized=is_virtualized,
        dmi_product_name=product_name,
    )


def _parse_cgroup_v2_path(cgroup_text: str) -> Optional[str]:
    for line in cgroup_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # v2 unified format: "0::/some/path"
        if line.startswith("0::"):
            parts = line.split(":", 2)
            if len(parts) == 3:
                return parts[2] or "/"
    return None


def _list_foreign_cuda_compute_processes(
    *,
    device_index: int,
    current_pid: int,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Return compute-process records on this GPU excluding current_pid."""

    try:
        import pynvml
    except ImportError as exc:  # pragma: no cover - environment dependent
        return [], f"pynvml unavailable: {exc}"

    def _query_processes(handle: Any) -> List[Any]:
        for fn_name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses_v1",
            "nvmlDeviceGetComputeRunningProcesses",
        ):
            fn = getattr(pynvml, fn_name, None)
            if fn is None:
                continue
            return list(fn(handle))
        return []

    unknown_mem = getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", None)
    foreign: List[Dict[str, Any]] = []
    try:
        pynvml.nvmlInit()
        from core.profiling.gpu_telemetry import resolve_nvml_device_handle

        properties = torch.cuda.get_device_properties(device_index)
        handle = resolve_nvml_device_handle(
            pynvml,
            device_index,
            cuda_device_uuid=getattr(properties, "uuid", None),
        )
        for proc in _query_processes(handle):
            pid = int(getattr(proc, "pid", -1))
            if pid <= 0 or pid == int(current_pid):
                continue
            used_raw = getattr(proc, "usedGpuMemory", None)
            used_mb: Optional[float] = None
            try:
                if used_raw is not None:
                    used_int = int(used_raw)
                    if unknown_mem is None or used_int != int(unknown_mem):
                        used_mb = used_int / (1024.0 * 1024.0)
            except Exception:
                used_mb = None
            proc_name: Optional[str] = None
            try:
                raw_name = pynvml.nvmlSystemGetProcessName(pid)
                if isinstance(raw_name, (bytes, bytearray)):
                    proc_name = raw_name.decode("utf-8", errors="ignore")
                elif raw_name is not None:
                    proc_name = str(raw_name)
            except Exception:
                proc_name = None
            foreign.append(
                {
                    "pid": pid,
                    "process_name": proc_name,
                    "used_memory_mb": used_mb,
                }
            )
    except Exception as exc:  # pragma: no cover - environment dependent
        return [], f"NVML query failed: {exc}"
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception as exc:
            _emit_validity_warning(
                "Failed to shut down NVML after foreign CUDA process inspection",
                exc=exc,
            )
    return foreign, None


def _collect_process_tree_pids(root_pid: int, proc_root: Path = Path("/proc")) -> Set[int]:
    """Collect `root_pid` plus all descendant PIDs visible under /proc."""
    owned: Set[int] = {int(root_pid)}
    ppid_to_children: Dict[int, Set[int]] = {}

    try:
        proc_entries = list(proc_root.iterdir())
    except Exception:
        return owned

    for entry in proc_entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        stat_path = entry / "stat"
        try:
            stat_text = stat_path.read_text(encoding="utf-8")
        except Exception:
            continue
        # /proc/<pid>/stat format: "<pid> (<comm>) <state> <ppid> ..."
        close_paren = stat_text.rfind(")")
        if close_paren < 0:
            continue
        tail = stat_text[close_paren + 1 :].strip().split()
        if len(tail) < 2:
            continue
        try:
            ppid = int(tail[1])
        except ValueError:
            continue
        ppid_to_children.setdefault(ppid, set()).add(pid)

    stack = [int(root_pid)]
    while stack:
        parent = stack.pop()
        for child in ppid_to_children.get(parent, set()):
            if child in owned:
                continue
            owned.add(child)
            stack.append(child)
    return owned


def _collect_process_lineage_pids(root_pid: int, proc_root: Path = Path("/proc")) -> Set[int]:
    """Collect `root_pid` plus all ancestor PIDs visible under /proc."""
    lineage: Set[int] = {int(root_pid)}
    current_pid = int(root_pid)

    while True:
        stat_path = proc_root / str(current_pid) / "stat"
        try:
            stat_text = stat_path.read_text(encoding="utf-8")
        except Exception:
            break
        close_paren = stat_text.rfind(")")
        if close_paren < 0:
            break
        tail = stat_text[close_paren + 1 :].strip().split()
        if len(tail) < 2:
            break
        try:
            parent_pid = int(tail[1])
        except ValueError:
            break
        if parent_pid <= 0 or parent_pid in lineage:
            break
        lineage.add(parent_pid)
        current_pid = parent_pid

    return lineage


def _pid_is_live_process(pid: int, proc_root: Path = Path("/proc")) -> bool:
    """Return True when `/proc/<pid>` exists and the process is not a zombie."""
    stat_path = proc_root / str(int(pid)) / "stat"
    try:
        stat_text = stat_path.read_text(encoding="utf-8")
    except Exception:
        return False
    close_paren = stat_text.rfind(")")
    if close_paren < 0:
        return False
    tail = stat_text[close_paren + 1 :].strip().split()
    if not tail:
        return False
    state = tail[0]
    return state not in {"Z", "X", "x"}


def _read_process_environ_value(
    pid: int,
    key: str,
    proc_root: Path = Path("/proc"),
) -> Optional[str]:
    """Return a single environment variable from `/proc/<pid>/environ` when readable."""
    env_path = proc_root / str(int(pid)) / "environ"
    try:
        raw = env_path.read_bytes()
    except Exception:
        return None

    prefix = f"{key}=".encode("utf-8")
    for entry in raw.split(b"\0"):
        if entry.startswith(prefix):
            try:
                return entry[len(prefix) :].decode("utf-8", errors="ignore")
            except Exception:
                return None
    return None


def _read_process_cmdline_arg_value(
    pid: int,
    flag: str,
    proc_root: Path = Path("/proc"),
) -> Optional[str]:
    """Return a flag value from `/proc/<pid>/cmdline` when present."""
    cmdline_path = proc_root / str(int(pid)) / "cmdline"
    try:
        raw = cmdline_path.read_bytes()
    except Exception:
        return None

    tokens = [
        token.decode("utf-8", errors="ignore")
        for token in raw.split(b"\0")
        if token
    ]
    prefix = f"{flag}="
    for idx, token in enumerate(tokens):
        if token == flag and idx + 1 < len(tokens):
            return tokens[idx + 1]
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _filter_foreign_cuda_compute_processes(
    *,
    foreign_procs: List[Dict[str, Any]],
    current_pid: int,
    probe_env: Dict[str, str],
    proc_root: Path = Path("/proc"),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Drop owned, same-run, and dead CUDA processes from an NVML sample."""
    owned_related_pids = _collect_process_tree_pids(current_pid, proc_root=proc_root)
    owned_related_pids.update(_collect_process_lineage_pids(current_pid, proc_root=proc_root))

    owned_run_marker = str(probe_env.get("AISP_BENCHMARK_OWNER_RUN_ID", "")).strip()
    owned_owner_pid = str(probe_env.get("AISP_BENCHMARK_OWNER_PID", "")).strip()

    owned_owner_related_pids: Set[int] = set()
    if owned_owner_pid:
        try:
            owner_pid_int = int(owned_owner_pid)
        except ValueError:
            owner_pid_int = None
        if owner_pid_int is not None and owner_pid_int > 0:
            owned_owner_related_pids = _collect_process_tree_pids(owner_pid_int, proc_root=proc_root)
            owned_owner_related_pids.update(
                _collect_process_lineage_pids(owner_pid_int, proc_root=proc_root)
            )

    filtered_foreign: List[Dict[str, Any]] = []
    for proc in foreign_procs:
        pid = int(proc.get("pid", -1))
        if pid <= 0:
            continue
        if pid in owned_related_pids or pid in owned_owner_related_pids:
            continue
        if owned_run_marker:
            same_run_marker = _read_process_environ_value(
                pid,
                "AISP_BENCHMARK_OWNER_RUN_ID",
                proc_root=proc_root,
            )
            if same_run_marker != owned_run_marker:
                same_run_marker = _read_process_cmdline_arg_value(
                    pid,
                    "--aisp-owner-run-id",
                    proc_root=proc_root,
                )
            if same_run_marker == owned_run_marker:
                continue
        if owned_owner_pid:
            same_owner_pid = _read_process_environ_value(
                pid,
                "AISP_BENCHMARK_OWNER_PID",
                proc_root=proc_root,
            )
            if same_owner_pid != owned_owner_pid:
                same_owner_pid = _read_process_cmdline_arg_value(
                    pid,
                    "--aisp-owner-pid",
                    proc_root=proc_root,
                )
            if same_owner_pid == owned_owner_pid:
                continue
        if not _pid_is_live_process(pid, proc_root=proc_root):
            continue
        filtered_foreign.append(proc)

    details: Dict[str, Any] = {}
    if owned_run_marker:
        details["owned_benchmark_run_id"] = owned_run_marker
    if owned_owner_pid:
        details["owned_benchmark_owner_pid"] = owned_owner_pid
    return filtered_foreign, details


def validate_environment(
    *,
    device: Optional["torch.device"] = None,
    probe: Optional[EnvironmentProbe] = None,
    allow_virtualization: bool = False,
    allow_foreign_gpu_processes: bool = False,
) -> EnvironmentValidationResult:
    """Validate benchmark environment is suitable (fail-fast on known invalid states)."""
    probe = probe or EnvironmentProbe()
    probe_env = probe.effective_env()
    errors: List[str] = []
    warnings_list: List[str] = []
    notices_list: List[str] = []
    details: Dict[str, Any] = {
        "platform": sys.platform,
    }

    if not sys.platform.startswith("linux"):
        errors.append(f"Non-Linux platform '{sys.platform}' is not supported for benchmark validity checks.")
        return EnvironmentValidationResult(False, errors, warnings_list, details, notices_list)

    # Resolve device expectation
    if device is None:
        if torch is None:
            errors.append("PyTorch is not available; cannot validate benchmark environment.")
            return EnvironmentValidationResult(False, errors, warnings_list, details, notices_list)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    details["device_type"] = device.type

    # Virtualization detection (used by multiple validity checks)
    execution_environment = detect_execution_environment(probe)
    product_name = execution_environment.dmi_product_name
    is_virtualized = bool(execution_environment.virtualized)
    details["execution_environment"] = execution_environment.kind
    details["dmi_product_name"] = product_name
    details["virtualized"] = is_virtualized
    # CUDA availability (required if running on CUDA device)
    if device.type == "cuda":
        if torch is None or not torch.cuda.is_available():
            errors.append("CUDA device requested but CUDA is not available (missing /dev/nvidia* or driver issue).")
        else:
            gpu_count = torch.cuda.device_count()
            details["gpu_count"] = gpu_count
            if gpu_count > 1:
                warnings_list.append(
                    f"Multiple GPUs detected ({gpu_count}). Ensure benchmarks use consistent device placement."
                )
            device_index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
            details["gpu_device_index"] = device_index
            # Only perform live process checks against the real host filesystem.
            if probe.root.resolve() == Path("/"):
                details["allow_foreign_gpu_processes"] = bool(allow_foreign_gpu_processes)
                # Strict mode should reject moderate concurrent workloads too:
                # a few hundred MiB of active compute can still distort timings.
                min_foreign_mb = 512.0
                raw_min_foreign_mb = str(probe_env.get("AISP_FOREIGN_GPU_PROCESS_MIN_MB", "")).strip()
                if raw_min_foreign_mb:
                    try:
                        min_foreign_mb = float(raw_min_foreign_mb)
                    except ValueError:
                        warnings_list.append(
                            f"Invalid AISP_FOREIGN_GPU_PROCESS_MIN_MB='{raw_min_foreign_mb}'; using default {min_foreign_mb:.0f}MiB."
                        )
                foreign_procs, foreign_err = _list_foreign_cuda_compute_processes(
                    device_index=device_index,
                    current_pid=os.getpid(),
                )
                if foreign_err:
                    warnings_list.append(f"Could not validate foreign CUDA compute processes: {foreign_err}")
                else:
                    foreign_procs, owner_details = _filter_foreign_cuda_compute_processes(
                        foreign_procs=foreign_procs,
                        current_pid=os.getpid(),
                        probe_env=probe_env,
                    )
                    details.update(owner_details)
                    details["foreign_cuda_compute_process_min_mb"] = min_foreign_mb
                    significant_foreign = []
                    for proc in foreign_procs:
                        mem = proc.get("used_memory_mb")
                        if mem is None or float(mem) >= min_foreign_mb:
                            significant_foreign.append(proc)
                    if significant_foreign:
                        # NVML can briefly report just-exited worker processes between phases.
                        # Re-confirm only the would-be blocking sample before failing strict mode.
                        time.sleep(0.15)
                        confirm_procs, confirm_err = _list_foreign_cuda_compute_processes(
                            device_index=device_index,
                            current_pid=os.getpid(),
                        )
                        if confirm_err:
                            warnings_list.append(
                                "Could not re-confirm foreign CUDA compute processes after initial detection: "
                                f"{confirm_err}"
                            )
                        else:
                            confirm_procs, _ = _filter_foreign_cuda_compute_processes(
                                foreign_procs=confirm_procs,
                                current_pid=os.getpid(),
                                probe_env=probe_env,
                            )
                            foreign_procs = confirm_procs
                            details["foreign_cuda_compute_processes_reconfirmed"] = True
                            transient_pids = sorted(
                                {
                                    int(proc.get("pid", -1))
                                    for proc in significant_foreign
                                    if int(proc.get("pid", -1)) > 0
                                }
                                - {
                                    int(proc.get("pid", -1))
                                    for proc in foreign_procs
                                    if int(proc.get("pid", -1)) > 0
                                }
                            )
                            if transient_pids:
                                details["transient_foreign_cuda_compute_process_pids"] = transient_pids
                            significant_foreign = []
                            for proc in foreign_procs:
                                mem = proc.get("used_memory_mb")
                                if mem is None or float(mem) >= min_foreign_mb:
                                    significant_foreign.append(proc)
                    details["foreign_cuda_compute_processes"] = foreign_procs
                    if significant_foreign:
                        preview: List[str] = []
                        for proc in significant_foreign[:5]:
                            mem = proc.get("used_memory_mb")
                            mem_txt = f"{mem:.1f}MiB" if isinstance(mem, (int, float)) else "unknown"
                            name = proc.get("process_name") or "unknown"
                            preview.append(f"pid={proc['pid']} name={name} mem={mem_txt}")
                        if len(significant_foreign) > 5:
                            preview.append(f"+{len(significant_foreign) - 5} more")
                        msg = (
                            "Foreign CUDA compute process(es) detected on benchmark GPU before run: "
                            + "; ".join(preview)
                            + f". Threshold={min_foreign_mb:.0f}MiB. "
                            + "Stop concurrent GPU workloads (for example vLLM serve/bench) before strict benchmarking."
                        )
                        if allow_foreign_gpu_processes:
                            warnings_list.append(
                                msg + " Override acknowledged via allow_foreign_gpu_processes=True."
                            )
                        else:
                            errors.append(msg)
            else:
                details["foreign_cuda_compute_processes"] = []

    # Swap interference (Memory category)
    swaps = _read_optional(probe, "/proc/swaps")
    if swaps is None:
        warnings_list.append("Cannot read /proc/swaps; swap state cannot be validated.")
    else:
        swap_lines = [ln for ln in swaps.splitlines() if ln.strip()]
        if len(swap_lines) > 1:
            errors.append(f"Swap is enabled ({len(swap_lines) - 1} swap device(s) active). Disable swap for benchmarking.")
    swappiness = _read_optional(probe, "/proc/sys/vm/swappiness")
    if swappiness is not None:
        try:
            details["vm_swappiness"] = int(swappiness.strip())
        except ValueError:
            warnings_list.append(f"Could not parse /proc/sys/vm/swappiness value: '{swappiness}'.")

    # CPU governor (Environment category)
    governors: Dict[str, str] = {}
    cpufreq_root = probe.resolve("/sys/devices/system/cpu/cpufreq")
    if cpufreq_root.exists():
        for policy_dir in sorted(cpufreq_root.glob("policy*")):
            gov_path = policy_dir / "scaling_governor"
            if gov_path.exists():
                try:
                    governors[policy_dir.name] = gov_path.read_text(encoding="utf-8").strip()
                except Exception:
                    warnings_list.append(f"Failed to read CPU governor at {gov_path}.")
    if not governors:
        if is_virtualized:
            details["cpu_governor_validation"] = "skipped_virtualized"
        else:
            warnings_list.append("CPU governor not exposed via cpufreq sysfs; cannot validate governor lock.")
    else:
        details["cpu_governors"] = dict(governors)
        non_perf = {k: v for k, v in governors.items() if v != "performance"}
        if non_perf:
            errors.append(f"CPU governor mismatch: expected 'performance', got {non_perf}.")

    # Cgroup resource limits (Container Resource Limits)
    cgroup_mount = probe.resolve("/sys/fs/cgroup")
    if not cgroup_mount.exists():
        warnings_list.append("Cgroup mount (/sys/fs/cgroup) not found; cannot validate cgroup resource limits.")
    else:
        cgroup_text = _read_optional(probe, "/proc/self/cgroup")
        cgroup_path = _parse_cgroup_v2_path(cgroup_text or "")
        if cgroup_path is None:
            warnings_list.append("Could not parse cgroup v2 path from /proc/self/cgroup; cgroup limits not validated.")
        else:
            details["cgroup_path"] = cgroup_path
            # cpu.max (quota/period)
            cpu_max = _read_optional(probe, str(Path("/sys/fs/cgroup") / cgroup_path.lstrip("/") / "cpu.max"))
            if cpu_max is not None:
                parts = cpu_max.split()
                if parts:
                    details["cgroup_cpu_max"] = cpu_max
                    if parts[0] != "max":
                        errors.append(f"CPU quota is set via cgroup cpu.max='{cpu_max}'. Remove CPU quotas for benchmarking.")
            else:
                warnings_list.append("Could not read cgroup cpu.max; CPU quota limits not validated.")

            mem_max = _read_optional(probe, str(Path("/sys/fs/cgroup") / cgroup_path.lstrip("/") / "memory.max"))
            if mem_max is not None:
                details["cgroup_memory_max"] = mem_max
                if mem_max != "max":
                    errors.append(
                        f"Memory limit is set via cgroup memory.max='{mem_max}'. Remove memory limits for benchmarking."
                    )
            else:
                warnings_list.append("Could not read cgroup memory.max; memory limits not validated.")

            # cpuset effective files may not be present in delegated cgroups; fall back to root for visibility.
            cpuset_cpu = _read_optional(probe, str(Path("/sys/fs/cgroup") / cgroup_path.lstrip("/") / "cpuset.cpus.effective"))
            if cpuset_cpu is None:
                cpuset_cpu = _read_optional(probe, "/sys/fs/cgroup/cpuset.cpus.effective")
            if cpuset_cpu is not None:
                details["cpuset_cpus_effective"] = cpuset_cpu
            else:
                warnings_list.append("Could not read cpuset.cpus.effective; CPU pinning not validated.")

            cpuset_mems = _read_optional(probe, str(Path("/sys/fs/cgroup") / cgroup_path.lstrip("/") / "cpuset.mems.effective"))
            if cpuset_mems is None:
                cpuset_mems = _read_optional(probe, "/sys/fs/cgroup/cpuset.mems.effective")
            if cpuset_mems is not None:
                details["cpuset_mems_effective"] = cpuset_mems
            else:
                warnings_list.append("Could not read cpuset.mems.effective; NUMA pinning not validated.")

    # NUMA checks (NUMA Inconsistency)
    node_root = probe.resolve("/sys/devices/system/node")
    numa_nodes: List[int] = []
    if node_root.exists():
        for entry in sorted(node_root.glob("node*")):
            match = re.match(r"node(\d+)$", entry.name)
            if match:
                numa_nodes.append(int(match.group(1)))
    details["numa_nodes"] = list(numa_nodes)
    if len(numa_nodes) > 1:
        affinity = probe.get_cpu_affinity()
        if affinity is None:
            warnings_list.append("CPU affinity unavailable; cannot validate NUMA pinning.")
        else:
            # Map CPUs to nodes via node*/cpulist when present.
            cpu_to_nodes: Dict[int, int] = {}
            missing_cpulist = False
            for node_id in numa_nodes:
                cpulist_text = _read_optional(probe, f"/sys/devices/system/node/node{node_id}/cpulist")
                if cpulist_text is None:
                    missing_cpulist = True
                    continue
                try:
                    for cpu in _parse_int_set(cpulist_text):
                        cpu_to_nodes[cpu] = node_id
                except Exception:
                    missing_cpulist = True
            if missing_cpulist or not cpu_to_nodes:
                warnings_list.append("NUMA cpulist unavailable; cannot validate NUMA pinning.")
            else:
                nodes_used = {cpu_to_nodes[cpu] for cpu in affinity if cpu in cpu_to_nodes}
                details["numa_nodes_in_affinity"] = sorted(nodes_used)
                if len(nodes_used) > 1:
                    warnings_list.append(
                        f"CPU affinity spans multiple NUMA nodes ({sorted(nodes_used)}). "
                        "Pin to a single NUMA node for tighter benchmark variance."
                    )

    # Virtualization warning policy (Virtualization Overhead)
    if is_virtualized:
        product_hint = f" (dmi_product_name={product_name!r})" if product_name else ""
        message = (
            "Virtualization detected (hypervisor present). "
            "Bare metal is recommended for final benchmark numbers "
            "(Nsight tools like nsys/ncu may be unavailable or misleading)."
            f"{product_hint}"
        )
        if allow_virtualization:
            warnings_list.append(
                f"{message} Compatibility mode active (validity_profile=portable)."
            )
        else:
            notices_list.append(
                "!!! STRICT VALIDITY NOTICE [VIRTUALIZATION] !!! "
                f"{message} Strict mode will continue because virtualization alone is non-fatal. "
                "Treat these numbers as non-canonical and re-run on bare metal before publishing."
            )

    # torch.compile backend sanity (Compile category)
    try:
        if torch is not None and hasattr(torch, "_dynamo"):
            config = getattr(torch._dynamo, "config", None)
            if config and hasattr(config, "suppress_errors") and config.suppress_errors:
                warnings_list.append("torch._dynamo.config.suppress_errors=True may hide compilation issues.")
    except Exception:
        warnings_list.append("Failed to inspect torch._dynamo config for suppress_errors.")

    if probe.probe_errors:
        details["probe_errors"] = list(probe.probe_errors)
        summary = (
            f"Environment probe encountered {len(probe.probe_errors)} read/access error(s); "
            "see details['probe_errors']."
        )
        if summary not in warnings_list:
            warnings_list.append(summary)

    is_valid = len(errors) == 0
    return EnvironmentValidationResult(is_valid, errors, warnings_list, details, notices_list)


# =============================================================================
# Memory Allocation Tracking
# =============================================================================

@dataclass
class MemoryAllocationSnapshot:
    """Snapshot of CUDA memory allocations for tracking patterns."""
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    num_allocs: int
    num_frees: int
    
    @classmethod
    def capture(cls, device: Optional[Any] = None) -> "MemoryAllocationSnapshot":
        """Capture current memory allocation state."""
        if torch is None or not torch.cuda.is_available():
            return cls(0.0, 0.0, 0.0, 0, 0)
        
        stats = torch.cuda.memory_stats(device)
        return cls(
            allocated_mb=torch.cuda.memory_allocated(device) / (1024 * 1024),
            reserved_mb=torch.cuda.memory_reserved(device) / (1024 * 1024),
            max_allocated_mb=torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            num_allocs=stats.get("num_alloc_retries", 0) + stats.get("allocation.all.current", 0),
            num_frees=stats.get("num_ooms", 0),
        )


class MemoryAllocationTracker:
    """Track memory allocations during benchmark execution.
    
    Uses PyTorch's memory hooks to track allocations and detect
    suspicious patterns like pre-allocated outputs or memory leaks.
    
    Usage:
        tracker = MemoryAllocationTracker(device)
        tracker.start()
        # ... run benchmark ...
        tracker.stop()
        issues = tracker.check_patterns()
    """
    
    def __init__(self, device: Optional[Any] = None):
        self.device = device
        self.allocations: List[Dict[str, Any]] = []
        self.frees: List[Dict[str, Any]] = []
        self._hook_handle = None
        self.start_snapshot: Optional[MemoryAllocationSnapshot] = None
        self.end_snapshot: Optional[MemoryAllocationSnapshot] = None
    
    def start(self) -> None:
        """Start tracking memory allocations."""
        if torch is None or not torch.cuda.is_available():
            return
        
        self.allocations = []
        self.frees = []
        self.start_snapshot = MemoryAllocationSnapshot.capture(self.device)
        
        # Use memory snapshot for tracking (simpler than hooks)
        torch.cuda.reset_peak_memory_stats(self.device)
    
    def stop(self) -> None:
        """Stop tracking and capture final state."""
        if torch is None or not torch.cuda.is_available():
            return
        
        self.end_snapshot = MemoryAllocationSnapshot.capture(self.device)
    
    def check_patterns(
        self,
        max_memory_increase_mb: float = 100.0,
        max_peak_vs_end_ratio: float = 2.0,
    ) -> Tuple[bool, List[str]]:
        """Check for suspicious memory allocation patterns.
        
        Args:
            max_memory_increase_mb: Maximum allowed memory increase during benchmark
            max_peak_vs_end_ratio: Maximum allowed ratio of peak to final memory
            
        Returns:
            Tuple of (no_issues, list_of_warnings)
        """
        warnings_list: List[str] = []
        
        if self.start_snapshot is None or self.end_snapshot is None:
            return True, warnings_list
        
        # Check for memory increase (potential leak)
        memory_increase = self.end_snapshot.allocated_mb - self.start_snapshot.allocated_mb
        if memory_increase > max_memory_increase_mb:
            warnings_list.append(
                f"MEMORY INCREASE: Allocated memory increased by {memory_increase:.1f}MB "
                f"during benchmark (threshold: {max_memory_increase_mb}MB). "
                "This may indicate a memory leak or pre-allocated outputs."
            )
        
        # Check peak vs end ratio (could indicate temporary large allocations)
        if self.end_snapshot.allocated_mb > 0:
            peak_ratio = self.end_snapshot.max_allocated_mb / self.end_snapshot.allocated_mb
            if peak_ratio > max_peak_vs_end_ratio:
                warnings_list.append(
                    f"MEMORY PEAK SPIKE: Peak memory ({self.end_snapshot.max_allocated_mb:.1f}MB) "
                    f"was {peak_ratio:.1f}x final memory ({self.end_snapshot.allocated_mb:.1f}MB). "
                    "This may indicate inefficient memory usage."
                )
        
        # Check for potential pre-allocation (high starting memory that doesn't grow)
        if self.start_snapshot.allocated_mb > 100 and memory_increase < 10:
            warnings_list.append(
                f"POTENTIAL PRE-ALLOCATION: High initial memory ({self.start_snapshot.allocated_mb:.1f}MB) "
                f"with minimal growth ({memory_increase:.1f}MB). "
                "Verify outputs aren't pre-computed in setup()."
            )
        
        return len(warnings_list) == 0, warnings_list
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics summary."""
        if self.start_snapshot is None or self.end_snapshot is None:
            return {}
        
        return {
            "start_allocated_mb": self.start_snapshot.allocated_mb,
            "end_allocated_mb": self.end_snapshot.allocated_mb,
            "peak_allocated_mb": self.end_snapshot.max_allocated_mb,
            "memory_increase_mb": self.end_snapshot.allocated_mb - self.start_snapshot.allocated_mb,
        }


@contextmanager
def track_memory_allocations(device: Optional[Any] = None):
    """Context manager for memory allocation tracking.
    
    Example:
        with track_memory_allocations() as tracker:
            # ... run benchmark ...
        no_issues, warnings = tracker.check_patterns()
    """
    tracker = MemoryAllocationTracker(device)
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.stop()


# =============================================================================
# CUDA Stream Auditing
# =============================================================================


def _resolve_stream_id(stream: Any) -> Optional[int]:
    """Return a stable identifier for a CUDA stream."""
    if stream is None:
        return None
    try:
        if hasattr(stream, "cuda_stream"):
            return int(stream.cuda_stream)
    except Exception:
        return None
    try:
        return int(id(stream))
    except Exception:
        return None

@dataclass
class StreamUsageInfo:
    """Information about CUDA stream usage during benchmark."""
    default_stream_ops: int = 0
    custom_streams_detected: int = 0
    default_stream_id: Optional[int] = None
    stream_ids: Set[int] = None
    sync_operations: int = 0
    unsync_warning: bool = False
    
    def __post_init__(self):
        if self.stream_ids is None:
            self.stream_ids = set()


class StreamAuditor:
    """Audits CUDA stream usage during benchmark execution.
    
    This helps detect multi-stream timing exploits (Locus/KernelBench 2025)
    where work is launched on non-default streams while the timer only
    measures the default stream.
    
    Usage:
        auditor = StreamAuditor()
        auditor.start()
        # ... run benchmark ...
        auditor.stop()
        info = auditor.get_info()
        warnings = auditor.check_issues()
    """
    
    def __init__(self, device: Optional[Any] = None):
        self.device = device
        self._start_time: Optional[float] = None
        self._stream_events: List[Dict[str, Any]] = []
        self._default_stream_id: Optional[int] = None
        self._observed_streams: Set[int] = set()
        self._sync_count: int = 0
        self._orig_stream_cls = None
        self._orig_stream_synchronize = None
        self._orig_stream_wait_stream = None
        self._orig_synchronize = None
        self._orig_stream_fn = None
        self._orig_set_stream_fn = None
    
    def start(self) -> None:
        """Start stream auditing."""
        if torch is None or not torch.cuda.is_available():
            return
        
        import time
        self._start_time = time.perf_counter()
        self._stream_events = []
        self._observed_streams = set()
        self._sync_count = 0
        
        # Record default stream ID
        default_stream = torch.cuda.current_stream(self.device)
        self._default_stream_id = default_stream.cuda_stream
        self._observed_streams.add(self._default_stream_id)
        
        # Keep torch.cuda.Stream as a type so libraries doing isinstance(..., torch.cuda.Stream)
        # continue to work (e.g., TensorRT-LLM runtime). We intentionally avoid
        # monkeypatching Stream construction on this runtime.
        try:
            self._orig_stream_cls = torch.cuda.Stream
        except Exception as exc:
            _emit_validity_warning(
                "Failed to capture torch.cuda.Stream before starting stream auditing; custom-stream instrumentation will be incomplete",
                exc=exc,
            )
            self._orig_stream_cls = None

        # Monkeypatch Stream methods to capture stream sync / dependencies
        if self._orig_stream_cls is not None:
            try:
                self._orig_stream_synchronize = self._orig_stream_cls.synchronize
                auditor = self

                def _audited_stream_synchronize(stream_self, *args, **kwargs):
                    auditor.record_sync("stream")
                    return auditor._orig_stream_synchronize(stream_self, *args, **kwargs)

                self._orig_stream_cls.synchronize = _audited_stream_synchronize  # type: ignore[assignment]
            except Exception as exc:
                _emit_validity_warning(
                    "Failed to instrument torch.cuda.Stream.synchronize; stream-auditing sync coverage is degraded",
                    exc=exc,
                )
                self._orig_stream_synchronize = None

            try:
                self._orig_stream_wait_stream = self._orig_stream_cls.wait_stream
                auditor = self

                def _audited_stream_wait_stream(stream_self, stream, *args, **kwargs):
                    auditor.record_sync("wait_stream")
                    auditor.record_stream_event(stream, operation="wait_stream")
                    return auditor._orig_stream_wait_stream(stream_self, stream, *args, **kwargs)

                self._orig_stream_cls.wait_stream = _audited_stream_wait_stream  # type: ignore[assignment]
            except Exception as exc:
                _emit_validity_warning(
                    "Failed to instrument torch.cuda.Stream.wait_stream; dependency tracking between streams is degraded",
                    exc=exc,
                )
                self._orig_stream_wait_stream = None
        
        # Monkeypatch synchronize to capture syncs
        try:
            self._orig_synchronize = torch.cuda.synchronize
            auditor = self
            
            def _audited_synchronize(*args, **kwargs):
                auditor.record_sync("device")
                return auditor._orig_synchronize(*args, **kwargs)
            
            torch.cuda.synchronize = _audited_synchronize  # type: ignore[assignment]
        except Exception as exc:
            _emit_validity_warning(
                "Failed to instrument torch.cuda.synchronize; device-wide sync tracking is degraded",
                exc=exc,
            )
            self._orig_synchronize = None

        # Monkeypatch torch.cuda.stream to record usage of existing streams
        try:
            self._orig_stream_fn = torch.cuda.stream
            auditor = self

            def _audited_stream_fn(stream, *args, **kwargs):
                auditor.record_stream_event(stream, operation="stream_ctx")
                return auditor._orig_stream_fn(stream, *args, **kwargs)

            torch.cuda.stream = _audited_stream_fn  # type: ignore[assignment]
        except Exception as exc:
            _emit_validity_warning(
                "Failed to instrument torch.cuda.stream; custom-stream context tracking is degraded",
                exc=exc,
            )
            self._orig_stream_fn = None

        # Monkeypatch torch.cuda.set_stream to capture explicit stream switches
        try:
            self._orig_set_stream_fn = torch.cuda.set_stream
            auditor = self

            def _audited_set_stream(stream, *args, **kwargs):
                auditor.record_stream_event(stream, operation="set_stream")
                return auditor._orig_set_stream_fn(stream, *args, **kwargs)

            torch.cuda.set_stream = _audited_set_stream  # type: ignore[assignment]
        except Exception as exc:
            _emit_validity_warning(
                "Failed to instrument torch.cuda.set_stream; explicit stream-switch tracking is degraded",
                exc=exc,
            )
            self._orig_set_stream_fn = None
    
    def record_stream_event(self, stream: Any, operation: str = "kernel") -> None:
        """Record a stream event for auditing.
        
        Call this manually when a custom stream is used, or use hooks.
        """
        if torch is None or not torch.cuda.is_available():
            return
        
        import time
        stream_id = _resolve_stream_id(stream)
        if stream_id is None:
            return
        self._observed_streams.add(stream_id)
        
        self._stream_events.append({
            "time": time.perf_counter() - (self._start_time or 0),
            "stream_id": stream_id,
            "operation": operation,
            "is_default": stream_id == self._default_stream_id,
        })
    
    def record_sync(self, sync_type: str = "device") -> None:
        """Record a synchronization event."""
        self._sync_count += 1
        
        import time
        self._stream_events.append({
            "time": time.perf_counter() - (self._start_time or 0),
            "stream_id": None,
            "operation": f"sync_{sync_type}",
            "is_default": None,
        })
    
    def stop(self) -> None:
        """Stop stream auditing."""
        if torch is None or not torch.cuda.is_available():
            return
        if self._orig_stream_cls is not None:
            if self._orig_stream_synchronize is not None:
                try:
                    self._orig_stream_cls.synchronize = self._orig_stream_synchronize  # type: ignore[assignment]
                except Exception as exc:
                    _emit_validity_warning(
                        "Failed to restore torch.cuda.Stream.synchronize after stream auditing; CUDA stream state may remain patched",
                        exc=exc,
                    )
            if self._orig_stream_wait_stream is not None:
                try:
                    self._orig_stream_cls.wait_stream = self._orig_stream_wait_stream  # type: ignore[assignment]
                except Exception as exc:
                    _emit_validity_warning(
                        "Failed to restore torch.cuda.Stream.wait_stream after stream auditing; CUDA stream state may remain patched",
                        exc=exc,
                    )
        if self._orig_synchronize is not None:
            try:
                torch.cuda.synchronize = self._orig_synchronize  # type: ignore[assignment]
            except Exception as exc:
                _emit_validity_warning(
                    "Failed to restore torch.cuda.synchronize after stream auditing; device-sync behavior may remain patched",
                    exc=exc,
                )
        if self._orig_stream_fn is not None:
            try:
                torch.cuda.stream = self._orig_stream_fn  # type: ignore[assignment]
            except Exception as exc:
                _emit_validity_warning(
                    "Failed to restore torch.cuda.stream after stream auditing; stream context behavior may remain patched",
                    exc=exc,
                )
        if self._orig_set_stream_fn is not None:
            try:
                torch.cuda.set_stream = self._orig_set_stream_fn  # type: ignore[assignment]
            except Exception as exc:
                _emit_validity_warning(
                    "Failed to restore torch.cuda.set_stream after stream auditing; stream-switch behavior may remain patched",
                    exc=exc,
                )
    
    def get_info(self) -> StreamUsageInfo:
        """Get stream usage information."""
        info = StreamUsageInfo()
        
        if self._default_stream_id is not None:
            info.default_stream_id = self._default_stream_id
            info.default_stream_ops = sum(
                1 for e in self._stream_events 
                if e["is_default"] is True and e["operation"] != "sync_device"
            )
            
        info.custom_streams_detected = len(self._observed_streams) - 1  # Exclude default
        info.stream_ids = self._observed_streams.copy()
        info.sync_operations = self._sync_count
        
        # Check if custom streams were used without sync
        if info.custom_streams_detected > 0 and self._sync_count == 0:
            info.unsync_warning = True
        
        return info
    
    def check_issues(self) -> Tuple[bool, List[str]]:
        """Check for stream usage issues.
        
        Returns:
            Tuple of (no_issues, list_of_warnings)
        """
        warnings_list: List[str] = []
        info = self.get_info()
        
        # Warning: custom streams without synchronization
        if info.unsync_warning:
            warnings_list.append(
                f"STREAM SYNC WARNING: {info.custom_streams_detected} custom stream(s) "
                "detected but no synchronization was recorded. "
                "This could allow work to escape timing measurement. "
                "Use torch.cuda.synchronize(), stream.synchronize(), or current_stream.wait_stream(...) "
                "for accurate multi-stream timing."
            )
        
        # Non-fatal notice: some real workloads legitimately use multiple streams (e.g., NCCL).
        # Do not fail the harness solely on stream count; rely on the stronger checks above.
        if info.custom_streams_detected > 2:
            warnings.warn(
                f"MULTI-STREAM WARNING: {info.custom_streams_detected} custom streams detected. "
                "Unusual for standard benchmarks - verify all streams are properly synchronized.",
                RuntimeWarning,
                stacklevel=2,
            )
        
        return len(warnings_list) == 0, warnings_list


@contextmanager
def audit_streams(device: Optional[Any] = None):
    """Context manager for CUDA stream auditing.
    
    Example:
        with audit_streams() as auditor:
            # ... run benchmark ...
        no_issues, warnings = auditor.check_issues()
    """
    auditor = StreamAuditor(device)
    auditor.start()
    try:
        yield auditor
    finally:
        auditor.stop()


def get_active_streams(device: Optional[Any] = None, declared_streams: Optional[List[Any]] = None) -> List[int]:
    """Get list of active CUDA streams on a device.
    
    Note: This uses internal PyTorch APIs and may not be comprehensive.
    """
    if torch is None or not torch.cuda.is_available():
        return []
    
    streams: Set[int] = set()
    
    # Get default stream
    default_stream = torch.cuda.current_stream(device)
    streams.add(int(default_stream.cuda_stream))
    
    if declared_streams:
        for stream in declared_streams:
            stream_id = _resolve_stream_id(stream)
            if stream_id is not None:
                streams.add(stream_id)
    
    return list(streams)


def check_stream_sync_completeness(
    pre_streams: List[int],
    post_streams: List[int],
) -> Tuple[bool, Optional[str]]:
    """Check if stream synchronization was complete.
    
    Compares streams before and after an operation to detect
    potential timing issues.
    
    Args:
        pre_streams: Stream IDs before operation
        post_streams: Stream IDs after operation
        
    Returns:
        Tuple of (sync_complete, warning_message)
    """
    if not pre_streams or not post_streams:
        return True, None
    
    # If new streams appeared, that's suspicious
    new_streams = set(post_streams) - set(pre_streams)
    if new_streams:
        return False, (
            f"NEW STREAMS WARNING: {len(new_streams)} new stream(s) created during benchmark. "
            "Verify all streams are synchronized before timing ends."
        )
    
    return True, None


# =============================================================================
# Distributed Verification
# =============================================================================

@dataclass
class DistributedVerifyResult:
    """Result of distributed verification across ranks."""
    all_ranks_executed: bool
    outputs_consistent: bool
    rank_outputs: Dict[int, Any]  # rank -> output hash
    inconsistent_ranks: List[int]
    error_message: Optional[str] = None


def gather_rank_outputs(
    local_output: Any,
    world_size: int = 1,
    rank: int = 0,
) -> Dict[int, str]:
    """Gather output hashes from all ranks for consistency verification.
    
    In distributed settings, this uses torch.distributed to gather output
    checksums from all ranks to ensure they computed consistent results.
    
    Args:
        local_output: The local rank's output tensor(s)
        world_size: Total number of ranks
        rank: Current rank
        
    Returns:
        Dict mapping rank -> output hash (only complete on rank 0)
    """
    if torch is None:
        return {0: "no_torch"}
    
    # Compute local hash
    if isinstance(local_output, torch.Tensor):
        local_hash = hashlib.sha256(local_output.cpu().numpy().tobytes()).hexdigest()[:16]
    elif isinstance(local_output, dict):
        hasher = hashlib.sha256()
        for k in sorted(local_output.keys()):
            v = local_output[k]
            if isinstance(v, torch.Tensor):
                hasher.update(k.encode())
                hasher.update(v.cpu().numpy().tobytes())
        local_hash = hasher.hexdigest()[:16]
    else:
        local_hash = hashlib.sha256(str(local_output).encode()).hexdigest()[:16]
    
    # Single-GPU case
    if world_size == 1:
        return {0: local_hash}
    
    # Multi-GPU: gather hashes from all ranks
    try:
        import torch.distributed as dist
        
        if not dist.is_initialized():
            return {rank: local_hash}
        
        # Gather all hashes to rank 0
        all_hashes: List[str] = [None] * world_size  # type: ignore
        
        # Use all_gather_object for string data
        if hasattr(dist, 'all_gather_object'):
            dist.all_gather_object(all_hashes, local_hash)
            return {i: h for i, h in enumerate(all_hashes) if h is not None}
        else:
            # Fallback: just return local
            return {rank: local_hash}
            
    except Exception:
        # Distributed not available or failed
        return {rank: local_hash}


def verify_distributed_outputs(
    rank_outputs: Dict[int, str],
    expected_world_size: int,
) -> DistributedVerifyResult:
    """Verify that all ranks executed and produced consistent outputs.
    
    This detects:
    - Rank skipping: Some ranks didn't execute work
    - Inconsistent outputs: Ranks produced different results (when they shouldn't)
    
    Args:
        rank_outputs: Dict mapping rank -> output hash
        expected_world_size: Expected number of ranks
        
    Returns:
        DistributedVerifyResult with verification outcome
    """
    result = DistributedVerifyResult(
        all_ranks_executed=True,
        outputs_consistent=True,
        rank_outputs=rank_outputs,
        inconsistent_ranks=[],
    )
    
    # Check all ranks reported
    if len(rank_outputs) != expected_world_size:
        result.all_ranks_executed = False
        missing_ranks = set(range(expected_world_size)) - set(rank_outputs.keys())
        result.error_message = f"RANK SKIPPING: Missing outputs from ranks {sorted(missing_ranks)}"
        return result
    
    # Check outputs are consistent
    unique_hashes = set(rank_outputs.values())
    if len(unique_hashes) > 1:
        result.outputs_consistent = False
        
        # Find the majority hash (assumed correct)
        hash_counts: Dict[str, int] = {}
        for h in rank_outputs.values():
            hash_counts[h] = hash_counts.get(h, 0) + 1
        majority_hash = max(hash_counts, key=lambda x: hash_counts[x])
        
        # Find inconsistent ranks
        result.inconsistent_ranks = [
            r for r, h in rank_outputs.items() if h != majority_hash
        ]
        result.error_message = (
            f"OUTPUT INCONSISTENCY: Ranks {result.inconsistent_ranks} produced different outputs. "
            f"Majority hash: {majority_hash}, inconsistent: "
            f"{[rank_outputs[r] for r in result.inconsistent_ranks]}"
        )
    
    return result


def check_rank_execution(
    benchmark: Any,
    world_size: int,
    rank: int,
) -> Tuple[bool, Optional[str]]:
    """Check if a benchmark properly executed on the current rank.
    
    This is a lightweight check to detect rank skipping by verifying
    that the benchmark actually performed computation on this rank.
    
    Args:
        benchmark: The benchmark instance
        world_size: Total number of ranks
        rank: Current rank
        
    Returns:
        Tuple of (executed, error_message)
    """
    # Check for explicit rank skip flags
    if hasattr(benchmark, '_skip_rank') and benchmark._skip_rank:
        return False, f"Rank {rank} has _skip_rank=True"
    
    marker = getattr(benchmark, "_execution_marker", None)
    if marker is not None:
        try:
            if torch is not None and isinstance(marker, torch.Tensor):
                if marker.numel() == 0 or not bool(marker.detach().cpu().sum().item()):
                    return False, f"Rank {rank} execution marker empty"
                return True, None
            return False, f"Rank {rank} execution marker invalid type ({type(marker)})"
        except Exception as exc:
            return False, f"Rank {rank} execution marker check failed: {exc}"
    
    # Check if benchmark has output (indicates execution)
    if hasattr(benchmark, 'get_verify_output'):
        try:
            output = benchmark.get_verify_output()
            if output is None:
                return False, f"Rank {rank} produced no output"
            return True, None
        except NotImplementedError:
            # Benchmark doesn't implement output, can't verify
            return True, None
        except Exception as e:
            return False, f"Rank {rank} failed to get output: {e}"
    
    return True, None


# =============================================================================
# CUDA Graph Capture Cheat Detection
# =============================================================================

@dataclass
class GraphCaptureState:
    """State tracking for CUDA graph capture detection."""
    capturing: bool = False
    capture_start_time: Optional[float] = None
    capture_end_time: Optional[float] = None
    kernels_during_capture: int = 0
    memory_allocated_during_capture: float = 0.0
    work_detected_during_capture: bool = False


class GraphCaptureCheatDetector:
    """Detects if work is being done during CUDA graph capture instead of replay.
    
    The "graph capture cheat" is when a benchmark does actual computation
    during the graph capture phase (which happens once) rather than during
    graph replay (which is what's being timed). This makes the benchmark
    appear faster than it actually is.
    
    Detection Strategy:
    1. Track memory allocations during capture vs replay
    2. Track GPU utilization during capture
    3. Compare capture time to replay time (capture should be similar or longer)
    
    Usage:
        detector = GraphCaptureCheatDetector()
        
        # During capture
        detector.start_capture()
        with torch.cuda.graph(g):
            benchmark_fn()  # Capture the work
        detector.end_capture()
        
        # During replay (timed)
        detector.start_replay()
        g.replay()
        detector.end_replay()
        
        # Check for cheating
        is_cheating, reason = detector.check_for_cheat()
    """
    
    def __init__(self, device: Optional[Any] = None):
        self.device = device
        self.capture_state: Optional[GraphCaptureState] = None
        self.replay_times: List[float] = []
        self._capture_start_memory: float = 0.0
        self._replay_start_memory: float = 0.0
    
    def start_capture(self) -> None:
        """Mark the start of CUDA graph capture."""
        import time
        
        self.capture_state = GraphCaptureState(capturing=True)
        self.capture_state.capture_start_time = time.perf_counter()
        
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            self._capture_start_memory = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
    
    def end_capture(self) -> None:
        """Mark the end of CUDA graph capture."""
        import time
        
        if self.capture_state is None:
            return
        
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        
        self.capture_state.capture_end_time = time.perf_counter()
        self.capture_state.capturing = False
        
        if torch is not None and torch.cuda.is_available():
            capture_end_memory = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            self.capture_state.memory_allocated_during_capture = (
                capture_end_memory - self._capture_start_memory
            )
    
    def start_replay(self) -> None:
        """Mark the start of a CUDA graph replay iteration."""
        if torch is not None and torch.cuda.is_available():
            self._replay_start_memory = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
    
    def end_replay(self, replay_time_ms: float) -> None:
        """Record a replay iteration time.
        
        Args:
            replay_time_ms: Time for this replay iteration in milliseconds
        """
        self.replay_times.append(replay_time_ms)
    
    def check_for_cheat(
        self,
        capture_replay_ratio_threshold: float = 10.0,
        memory_threshold_mb: float = 100.0,
    ) -> Tuple[bool, Optional[str]]:
        """Check if the benchmark is cheating via graph capture.
        
        Signs of cheating:
        1. Capture time >> replay time (work done during capture, not replay)
        2. Large memory allocation during capture
        3. Replay times are suspiciously fast (near-zero)
        
        Args:
            capture_replay_ratio_threshold: Max allowed ratio of capture/replay time
            memory_threshold_mb: Memory allocation during capture that triggers warning
            
        Returns:
            Tuple of (is_cheating, reason_if_cheating)
        """
        if self.capture_state is None:
            return False, None
        
        warnings_list: List[str] = []
        
        # Check 1: Capture/replay time ratio
        if (self.capture_state.capture_start_time is not None and 
            self.capture_state.capture_end_time is not None and
            self.replay_times):
            
            capture_time_ms = (
                self.capture_state.capture_end_time - 
                self.capture_state.capture_start_time
            ) * 1000
            
            avg_replay_time_ms = sum(self.replay_times) / len(self.replay_times)
            
            if avg_replay_time_ms > 0:
                ratio = capture_time_ms / avg_replay_time_ms
                if ratio > capture_replay_ratio_threshold:
                    warnings_list.append(
                        f"GRAPH CAPTURE CHEAT SUSPECTED: Capture time ({capture_time_ms:.2f}ms) "
                        f"is {ratio:.1f}x longer than replay time ({avg_replay_time_ms:.2f}ms). "
                        "This suggests work is being done during capture, not replay."
                    )
        
        # Check 2: Memory allocation during capture
        if self.capture_state.memory_allocated_during_capture > memory_threshold_mb:
            warnings_list.append(
                f"SUSPICIOUS MEMORY ALLOCATION: {self.capture_state.memory_allocated_during_capture:.1f}MB "
                f"allocated during graph capture. This may indicate computation during capture."
            )
        
        # Check 3: Near-zero replay times
        if self.replay_times:
            min_replay = min(self.replay_times)
            if min_replay < 0.001:  # Less than 1 microsecond
                warnings_list.append(
                    f"SUSPICIOUS REPLAY TIME: Minimum replay time ({min_replay*1000:.3f}μs) "
                    "is near-zero. Graph may be empty or doing no actual work."
                )
        
        # Check 4: High variance across replays (should be stable for real graphs)
        if len(self.replay_times) >= 3:
            mean_replay = statistics.mean(self.replay_times)
            std_replay = statistics.pstdev(self.replay_times)
            if mean_replay > 0:
                cv = std_replay / mean_replay
                if cv > 0.5:
                    warnings_list.append(
                        f"REPLAY VARIANCE: Replay times vary too much (CV={cv:.2f}). "
                        "Real graph replay should be consistent."
                    )
        
        if warnings_list:
            return True, " | ".join(warnings_list)
        
        return False, None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get capture/replay statistics."""
        if self.capture_state is None:
            return {}
        
        capture_time_ms = 0.0
        if (self.capture_state.capture_start_time is not None and 
            self.capture_state.capture_end_time is not None):
            capture_time_ms = (
                self.capture_state.capture_end_time - 
                self.capture_state.capture_start_time
            ) * 1000
        
        return {
            "capture_time_ms": capture_time_ms,
            "capture_memory_mb": self.capture_state.memory_allocated_during_capture,
            "replay_count": len(self.replay_times),
            "avg_replay_time_ms": sum(self.replay_times) / len(self.replay_times) if self.replay_times else 0,
            "min_replay_time_ms": min(self.replay_times) if self.replay_times else 0,
            "max_replay_time_ms": max(self.replay_times) if self.replay_times else 0,
        }


@contextmanager
def detect_graph_capture_cheat(device: Optional[Any] = None):
    """Context manager for graph capture cheat detection.
    
    Example:
        with detect_graph_capture_cheat() as detector:
            detector.start_capture()
            with torch.cuda.graph(g):
                fn()
            detector.end_capture()
            
            for _ in range(10):
                detector.start_replay()
                g.replay()
                # Record replay time manually
                detector.end_replay(elapsed_ms)
        
        is_cheating, reason = detector.check_for_cheat()
    """
    detector = GraphCaptureCheatDetector(device)
    try:
        yield detector
    finally:
        pass  # No cleanup needed


def check_graph_capture_integrity(
    capture_time_ms: float,
    replay_times_ms: List[float],
    memory_during_capture_mb: float = 0.0,
) -> Tuple[bool, Optional[str]]:
    """Quick check for graph capture integrity without full detector.
    
    Args:
        capture_time_ms: Time to capture the graph
        replay_times_ms: List of replay iteration times
        memory_during_capture_mb: Memory allocated during capture
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not replay_times_ms:
        return True, None
    
    avg_replay = sum(replay_times_ms) / len(replay_times_ms)
    
    # Check for suspicious capture/replay ratio
    if avg_replay > 0:
        ratio = capture_time_ms / avg_replay
        if ratio > 10.0:
            return False, (
                f"Graph capture time ({capture_time_ms:.2f}ms) is {ratio:.1f}x "
                f"replay time ({avg_replay:.2f}ms). Suspected work during capture."
            )
    
    # Check for near-zero replays
    if min(replay_times_ms) < 0.001:
        return False, "Replay times are near-zero, graph may be empty."
    
    return True, None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # GPU State
    "check_benchmark_fn_sync_calls",
    "check_benchmark_fn_antipatterns",
    "GPUState",
    "capture_gpu_state", 
    "check_gpu_state_consistency",
    "PrecisionPolicyState",
    "capture_precision_policy_state",
    "check_precision_policy_consistency",
    
    # Memory Checks
    "get_tensor_addresses",
    "check_input_output_aliasing",
    "reset_cuda_memory_pool",
    
    # Memory Allocation Tracking
    "MemoryAllocationSnapshot",
    "MemoryAllocationTracker",
    "track_memory_allocations",
    
    # Setup Checks
    "hash_tensors",
    "check_setup_precomputation",
    
    # GC Control
    "gc_disabled",
    
    # Compile Checks
    "clear_compile_cache",
    "get_compile_state",
    
    # Lazy Evaluation
    "force_tensor_evaluation",
    
    # Environment
    "validate_environment",
    "EnvironmentProbe",
    "ExecutionEnvironment",
    "EnvironmentValidationResult",
    "detect_execution_environment",
    
    # Stream Auditing
    "StreamUsageInfo",
    "StreamAuditor",
    "audit_streams",
    "get_active_streams",
    "check_stream_sync_completeness",
    
    # Distributed Verification
    "DistributedVerifyResult",
    "gather_rank_outputs",
    "verify_distributed_outputs",
    "check_rank_execution",
    
    # Graph Capture Cheat Detection
    "GraphCaptureState",
    "GraphCaptureCheatDetector",
    "detect_graph_capture_cheat",
    "check_graph_capture_integrity",
]
