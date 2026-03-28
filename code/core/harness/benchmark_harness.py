"""Production-grade benchmarking harness with profiling integration.

Provides industry-standard benchmarking using Triton do_bench, PyTorch Timer,
and custom CUDA Events. Supports nsys, ncu, and PyTorch profiler integration.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import importlib
import inspect
import os
import random
import re
import shutil
import signal
import statistics
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeoutError
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch

from core.utils.compile_utils import enable_tf32
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.backend_policy import apply_backend_policy, normalize_backend_policy, restore_backend_policy
from core.harness.verify_output_codec import deserialize_verify_tensor

# Pydantic is required - fail fast if not available
from core.benchmark.models import (
    BenchmarkResult as PydanticBenchmarkResult,
    BenchmarkRun,
    MemoryStats,
    TimingStats,
    InferenceTimingStats,
    ProfilerArtifacts,
    ProfilerMetrics,
    NsysMetrics,
    NcuMetrics,
    ProtonMetrics,
    TorchMetrics,
    ThroughputStats,
)
from core.harness.validity_checks import (
    EnvironmentProbe,
    GraphCaptureCheatDetector,
    audit_streams,
    check_rank_execution,
    check_stream_sync_completeness,
)
from core.utils.python_entrypoints import build_repo_python_env

# Import unified profiling runner
try:
    from core.profiling.profiling_runner import (
        run_nsys_profiling,
        run_ncu_profiling,
        check_nsys_available,
        check_ncu_available,
        run_proton_profiling,
        check_proton_available,
    )
    PROFILING_RUNNER_AVAILABLE = True
except ImportError:
    PROFILING_RUNNER_AVAILABLE = False
    def run_nsys_profiling(
        benchmark: Any,
        benchmark_module: Any,
        benchmark_class: str,
        output_dir: Path,
        config: Any,
        profiler_config: Optional[Any] = None,
        timeout_seconds: Optional[int] = None,
        metrics: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        raise ImportError("profiling_runner not available")
    
    def run_ncu_profiling(
        benchmark: Any,
        benchmark_module: Any,
        benchmark_class: str,
        output_dir: Path,
        config: Any,
        profiler_config: Optional[Any] = None,
        timeout_seconds: Optional[int] = None,
        metrics: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        raise ImportError("profiling_runner not available")
    
    def check_nsys_available() -> bool:
        return False
    
    def check_ncu_available() -> bool:
        return False
    
    def run_proton_profiling(
        benchmark: Any,
        benchmark_module: Any,
        benchmark_class: str,
        output_dir: Path,
        config: Any,
        profiler_config: Optional[Any] = None,
        timeout_seconds: Optional[int] = None,
        metrics: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        raise ImportError("profiling_runner not available")
    
    def check_proton_available() -> bool:
        return False

# Import profiler wrapper
try:
    from core.profiling.profiler_wrapper import create_benchmark_wrapper
    PROFILER_WRAPPER_AVAILABLE = True
except ImportError:
    PROFILER_WRAPPER_AVAILABLE = False

# Import logger
try:
    from core.utils.logger import get_logger
    logger = get_logger(__name__)
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False


def _cleanup_process_group(pgid: Optional[int], *, grace_seconds: float = 2.0) -> None:
    """Terminate any descendants left behind by a subprocess benchmark worker.

    Some runtimes (notably TRT-LLM/OpenMPI helpers) react badly when they are reaped
    from inside the worker process itself. The isolated worker is started in its own
    process group, so the parent harness can clean up that group after stdout/stderr
    have been collected without risking loss of the serialized result payload.
    """
    if pgid is None:
        return

    try:
        os.killpg(int(pgid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return

    deadline = time.time() + max(float(grace_seconds), 0.0)
    while time.time() < deadline:
        try:
            os.killpg(int(pgid), 0)
        except (ProcessLookupError, PermissionError, OSError):
            return
        time.sleep(0.05)

    try:
        os.killpg(int(pgid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        return
    import logging
    logger = logging.getLogger(__name__)


def _benchmark_child_preexec() -> None:
    """Start benchmark workers in a fresh session without inherited suite alarms.

    The top-level bench CLI installs a suite-wide SIGALRM timeout. Benchmark
    workers launched via subprocess/torchrun must clear that inherited timer or
    a long-running parent suite can kill a later child immediately with the
    parent's timeout message.
    """
    os.setsid()
    sigalrm = getattr(signal, "SIGALRM", None)
    if sigalrm is None:
        return
    try:
        signal.alarm(0)
    except Exception:
        pass
    try:
        signal.signal(sigalrm, signal.SIG_DFL)
    except Exception:
        pass


_QUICK_WINS_CONFIGURED = False
_SDPA_KERNEL_CONTEXT = None


def _emit_harness_warning(message: str, *, exc: Optional[BaseException] = None) -> None:
    """Surface degraded harness behavior through logs and runtime warnings."""
    detail = f"{message}: {exc}" if exc is not None else message
    if LOGGER_AVAILABLE:
        logger.warning(detail)
    warnings.warn(detail, RuntimeWarning, stacklevel=2)


def _normalize_target_override_chapter(label: str) -> str:
    return label.replace("\\", "/").replace("/", "_")


def _lookup_target_extra_args(
    overrides_map: Dict[str, Union[str, Sequence[str]]],
    target_label: Optional[str],
) -> Optional[Union[str, Sequence[str]]]:
    """Resolve per-target overrides across slash/underscore chapter labels."""
    if not target_label:
        return None
    if target_label in overrides_map:
        return overrides_map[target_label]

    target_chapter, target_sep, target_example = target_label.partition(":")
    normalized_target_chapter = _normalize_target_override_chapter(target_chapter)
    for candidate_label, overrides in overrides_map.items():
        candidate_chapter, candidate_sep, candidate_example = str(candidate_label).partition(":")
        if bool(candidate_sep) != bool(target_sep):
            continue
        if target_sep and candidate_example != target_example:
            continue
        if _normalize_target_override_chapter(candidate_chapter) == normalized_target_chapter:
            return overrides
    return None


def _format_environment_invalid_message(errors: Sequence[str], config: "BenchmarkConfig") -> str:
    """Build a consistent environment failure message with recovery guidance.

    Strict mode is intentionally fail-fast. When environment capability checks
    fail there, always surface the explicit portability recovery flags.
    """
    message = "ENVIRONMENT INVALID: " + " | ".join(errors)
    validity_profile = str(getattr(config, "validity_profile", "strict")).strip().lower()
    if validity_profile == "strict":
        message += (
            " Recovery: rerun with --validity-profile portable to use the portable"
            " validity profile (compatibility mode)."
            " Consequence: strict environment protections are relaxed and expectation"
            " writes remain disabled unless --allow-portable-expectations-update is set."
        )
    return message


_FOREIGN_GPU_PROCESS_ERROR_PREFIX = (
    "Foreign CUDA compute process(es) detected on benchmark GPU before run: "
)


def _contains_only_foreign_gpu_process_errors(errors: Sequence[str]) -> bool:
    """Return True when every environment error is the strict foreign-process gate."""
    return bool(errors) and all(
        str(error).startswith(_FOREIGN_GPU_PROCESS_ERROR_PREFIX) for error in errors
    )


def _is_nvidia_smi_permission_error(exc: subprocess.CalledProcessError) -> bool:
    # Common failure mode when not root: `nvidia-smi` returns exit status 4 and
    # prints "Insufficient Permissions" in stdout/stderr (we redirect stderr).
    try:
        out = exc.output or b""
        out_s = out.decode(errors="ignore")
    except Exception:
        out_s = ""
    return exc.returncode == 4 or ("Insufficient Permissions" in out_s)


def _nvidia_smi(args: Sequence[str]) -> bytes:
    """Run `nvidia-smi` and retry once with `sudo -n` on permission failure."""
    cmd = ["nvidia-smi", *args]
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        if not _is_nvidia_smi_permission_error(exc):
            raise
        try:
            sudo_cmd = ["sudo", "-n", "nvidia-smi", *args]
            out = subprocess.check_output(sudo_cmd, stderr=subprocess.STDOUT)
            if LOGGER_AVAILABLE:
                logger.info("nvidia-smi required sudo for: %s", " ".join(shlex.quote(x) for x in cmd))
            return out
        except (FileNotFoundError, subprocess.CalledProcessError):
            # Keep the original error so callers see the true underlying failure.
            raise


def _resolve_physical_device_index(device_index: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return device_index
    tokens = [tok.strip() for tok in visible.split(",") if tok.strip()]
    if not tokens:
        return device_index
    if device_index < 0 or device_index >= len(tokens):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r} does not include index {device_index}"
        )
    token = tokens[device_index]
    if token.isdigit():
        return int(token)
    if token.startswith("GPU-"):
        try:
            import pynvml
        except ImportError as exc:
            raise RuntimeError("CUDA_VISIBLE_DEVICES uses UUIDs but pynvml is unavailable") from exc
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            for idx in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode()
                if uuid == token:
                    return idx
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception as exc:
                _emit_harness_warning(
                    "Failed to shut down NVML after resolving physical device index",
                    exc=exc,
                )
        raise RuntimeError(f"Unable to map CUDA_VISIBLE_DEVICES entry {token!r} to a GPU index")
    raise RuntimeError(f"Unsupported CUDA_VISIBLE_DEVICES entry {token!r}")


_SUDO_NONINTERACTIVE_OK: Optional[bool] = None


def _sudo_noninteractive_ok() -> bool:
    """Return True if `sudo -n` is available for this process.

    We use this to optionally run `nvidia-smi` as root for clock locking on
    managed systems where unprivileged users can't change clocks.
    """
    global _SUDO_NONINTERACTIVE_OK
    if _SUDO_NONINTERACTIVE_OK is not None:
        return _SUDO_NONINTERACTIVE_OK
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        _SUDO_NONINTERACTIVE_OK = False
        return False
    if shutil.which("sudo") is None:
        _SUDO_NONINTERACTIVE_OK = False
        return False
    try:
        subprocess.check_output(["sudo", "-n", "true"], stderr=subprocess.STDOUT)
        _SUDO_NONINTERACTIVE_OK = True
    except Exception:
        _SUDO_NONINTERACTIVE_OK = False
    return _SUDO_NONINTERACTIVE_OK


def ramp_gpu_clocks(device: int = 0, duration_ms: float = 50.0, max_iters: int = 200) -> None:
    """Run a short GPU workload to ramp clocks before measurement."""
    if not torch.cuda.is_available():
        raise RuntimeError("ramp_gpu_clocks requires CUDA")
    torch.cuda.set_device(device)
    dev = torch.device("cuda", device)
    size = 2048
    dtype = torch.float16
    a = torch.ones((size, size), device=dev, dtype=dtype)
    b = torch.ones((size, size), device=dev, dtype=dtype)
    c = torch.empty((size, size), device=dev, dtype=dtype)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    iters = 0
    while iters < max_iters:
        torch.matmul(a, b, out=c)
        iters += 1
        if iters % 4 == 0:
            end.record()
            torch.cuda.synchronize(device)
            if start.elapsed_time(end) >= duration_ms:
                break
    torch.cuda.synchronize(device)


@contextmanager
def lock_gpu_clocks(device: int = 0, sm_clock_mhz: Optional[int] = None, mem_clock_mhz: Optional[int] = None):
    """Lock GPU clocks for consistent benchmarking.
    
    Based on Triton's set_gpu_clock context manager:
    https://github.com/triton-lang/triton/blob/main/python/triton/testing.py
    
    Args:
        device: GPU device index (default 0)
        sm_clock_mhz: Target SM clock rate in MHz (None = auto-detect max)
        mem_clock_mhz: Target memory clock rate in MHz (None = auto-detect max)
    
    Yields:
        Tuple of (theoretical_tflops, theoretical_gbps) at the locked clocks
    
    Example:
        with lock_gpu_clocks(device=0, sm_clock_mhz=1350, mem_clock_mhz=1215):
            benchmark.run()
    
    Note:
        - Requires nvidia-smi and sudo/root permissions
        - Clocks are automatically reset when context exits
        - Raises RuntimeError if nvidia-smi fails
    """
    if not torch.cuda.is_available():
        raise RuntimeError("lock_gpu_clocks requires CUDA")
    props = torch.cuda.get_device_properties(device)
    physical_index = _resolve_physical_device_index(device)
    def _query_nvml_clocks() -> tuple[int, int, int, int]:
        """Return (app_sm, app_mem, cur_sm, cur_mem) for the physical GPU index."""
        try:
            import pynvml
        except ImportError as exc:
            raise RuntimeError("lock_gpu_clocks requires pynvml for clock verification") from exc
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
            app_sm = int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM))
            app_mem = int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_MEM))
            cur_sm = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
            cur_mem = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
            return app_sm, app_mem, cur_sm, cur_mem
        except Exception as exc:
            raise RuntimeError(f"Failed to query NVML clocks: {exc}") from exc
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception as exc:
                _emit_harness_warning(
                    "Failed to shut down NVML after querying application clocks",
                    exc=exc,
                )

    def _apply_clock_lock(target_sm: int, target_mem: int, *, retry_attempts: int = 1) -> tuple[int, int, int, int]:
        """Best-effort lock using application clocks, then hard locks as fallback.

        Returns the NVML-observed clocks after locking (app_sm, app_mem, cur_sm, cur_mem).
        Raises RuntimeError if the application clocks do not match the requested target.
        """
        lock_error: Optional[Exception] = None
        last_exc: Optional[BaseException] = None
        for _attempt in range(max(1, int(retry_attempts))):
            lock_error = None
            try:
                _nvidia_smi(
                    [
                        "-i",
                        str(physical_index),
                        f"--applications-clocks={target_mem},{target_sm}",
                    ]
                )
            except subprocess.CalledProcessError as exc:
                # Fallback to hard locks for platforms that do not support applications clocks.
                lock_error = exc
                try:
                    _nvidia_smi(
                        [
                            "-i",
                            str(physical_index),
                            f"--lock-gpu-clocks={target_sm},{target_sm}",
                        ]
                    )
                    _nvidia_smi(
                        [
                            "-i",
                            str(physical_index),
                            f"--lock-memory-clocks={target_mem},{target_mem}",
                        ]
                    )
                    lock_error = None
                except subprocess.CalledProcessError as exc2:
                    lock_error = exc2

            app_sm, app_mem, cur_sm, cur_mem = _query_nvml_clocks()
            if abs(app_sm - target_sm) <= 50 and abs(app_mem - target_mem) <= 50:
                if lock_error is not None and LOGGER_AVAILABLE:
                    logger.warning(
                        "GPU clock lock command failed (%s) but application clocks already match target; proceeding.",
                        lock_error,
                    )
                if LOGGER_AVAILABLE and (abs(cur_sm - target_sm) > 50 or abs(cur_mem - target_mem) > 50):
                    logger.info(
                        "GPU current clocks below locked target (idle/throttled): "
                        "current SM=%dMHz Mem=%dMHz, target SM=%dMHz Mem=%dMHz",
                        cur_sm,
                        cur_mem,
                        target_sm,
                        target_mem,
                    )
                return app_sm, app_mem, cur_sm, cur_mem

            last_exc = RuntimeError(
                f"GPU clock lock mismatch: requested SM={target_sm}MHz Mem={target_mem}MHz, "
                f"applications SM={app_sm}MHz Mem={app_mem}MHz"
                + (f" (lock error: {lock_error})" if lock_error is not None else "")
            )
            # Brief pause and retry. This helps when another process flips app clocks
            # while the harness is between targets.
            time.sleep(0.15)

        if last_exc is None:
            raise RuntimeError("GPU clock lock failed without a captured verification error")
        raise last_exc

    def _reassert_clock_lock(config_sm: Optional[int], config_mem: Optional[int]) -> tuple[int, int, int, int]:
        """Re-apply requested application clocks and verify they stick.

        This is intentionally called right before measurement. We have observed rare cases
        where application clocks drift (or are reset) between targets, which invalidates
        baseline-vs-optimized comparisons. Reasserting here keeps the manifest + telemetry honest.
        """
        if config_sm is None or config_mem is None:
            return _query_nvml_clocks()
        return _apply_clock_lock(int(config_sm), int(config_mem), retry_attempts=3)
    try:
        # Enable persistence mode
        _nvidia_smi(["-i", str(physical_index), "-pm", "1"])
        
        # Get max clocks if not specified
        if sm_clock_mhz is None or mem_clock_mhz is None:
            out = _nvidia_smi(
                [
                    "-i",
                    str(physical_index),
                    "--query-gpu=clocks.max.sm,clocks.max.memory",
                    "--format=csv,noheader,nounits",
                ]
            )
            max_sm, max_mem = [int(x.strip()) for x in out.decode().split(',')]
            sm_clock_mhz = sm_clock_mhz or max_sm
            mem_clock_mhz = mem_clock_mhz or max_mem
        
        if sm_clock_mhz is None or mem_clock_mhz is None:
            raise RuntimeError("Unable to determine target SM/memory clocks for lock_gpu_clocks")
        
        # Apply and verify. Use a couple retries to guard against rare clock drift between targets.
        app_sm, app_mem, cur_sm, cur_mem = _reassert_clock_lock(sm_clock_mhz, mem_clock_mhz)
        
        # Calculate theoretical performance at these clocks using device properties
        sm_count = props.multi_processor_count
        # memory_clock_rate is reported in kHz, memory_bus_width in bits
        mem_clock_hz = getattr(props, "memory_clock_rate", 0) * 1000
        bus_width_bits = getattr(props, "memory_bus_width", 0)
        theoretical_tflops = 1e-6 * 2 * sm_count * 4 * 256 * sm_clock_mhz
        theoretical_gbps = 0.0
        if mem_clock_hz > 0 and bus_width_bits > 0:
            # DDR: multiply by 2
            theoretical_gbps = mem_clock_hz * (bus_width_bits / 8) * 2 / 1e9
        
        logger.info(f"GPU clocks locked: SM={int(sm_clock_mhz)}MHz, Mem={int(mem_clock_mhz)}MHz")
        if theoretical_tflops > 0:
            logger.info(
                "Theoretical peak: %.1f TFLOPS (FP16), %.0f GB/s (bus=%dbit)",
                theoretical_tflops,
                theoretical_gbps,
                bus_width_bits,
            )
        
        yield theoretical_tflops, theoretical_gbps
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to lock GPU clocks: {e}")
    finally:
        # Reset clocks
        try:
            _nvidia_smi(["-i", str(physical_index), "-rgc"])
            _nvidia_smi(["-i", str(physical_index), "-rmc"])
            _nvidia_smi(["-i", str(physical_index), "-rac"])
            _nvidia_smi(["-i", str(physical_index), "-pm", "0"])
            logger.info("GPU clocks reset to default")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # Best effort cleanup


def _configure_quick_wins() -> None:
    """Apply global TF32 configuration once per process."""
    global _QUICK_WINS_CONFIGURED
    if _QUICK_WINS_CONFIGURED:
        return
    if torch.cuda.is_available():
        try:
            enable_tf32()
        except Exception as exc:  # pragma: no cover - defensive
            if LOGGER_AVAILABLE:
                logger.warning("TF32 enablement failed: %s", exc)
        _configure_attention_kernels()
        _configure_matmul_reduction()
    _QUICK_WINS_CONFIGURED = True


def _configure_attention_kernels() -> None:
    """Prefer flash/efficient SDPA backends when available."""
    global _SDPA_KERNEL_CONTEXT
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:  # pragma: no cover - API availability differs by PyTorch
        sdpa_kernel = None  # type: ignore[assignment]
        SDPBackend = None  # type: ignore[assignment]

    if sdpa_kernel is not None and SDPBackend is not None:
        preferred = [
            getattr(SDPBackend, name)
            for name in ("TRANSFORMER_ENGINE", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN")
            if hasattr(SDPBackend, name)
        ]
        if preferred:
            try:
                ctx = sdpa_kernel(preferred)
                ctx.__enter__()  # keep context active for process lifetime
                _SDPA_KERNEL_CONTEXT = ctx
                return
            except Exception as exc:  # pragma: no cover - fallback below
                if LOGGER_AVAILABLE:
                    logger.debug("SDPA kernel preference failed: %s", exc)

    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)  # type: ignore[attr-defined]
        torch.backends.cuda.enable_mem_efficient_sdp(True)  # type: ignore[attr-defined]
        torch.backends.cuda.enable_math_sdp(False)  # type: ignore[attr-defined]


def _configure_matmul_reduction() -> None:
    """Enable reduced-precision reductions to hit fused matmul fast paths."""
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # type: ignore[attr-defined]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True  # type: ignore[attr-defined]
    except Exception:
        return


def _is_chapter_or_labs_benchmark(benchmark: Any) -> bool:
    """Return True if the benchmark originates from any chapter directory (chXX/) or labs/."""
    try:
        file_path = inspect.getfile(benchmark.__class__)
    except Exception:
        return False
    normalized = str(Path(file_path)).replace("\\", "/")
    return "/labs/" in normalized or "/ch" in normalized


_configure_quick_wins()

_VERIFY_OUTPUT_MAX_BYTES = 64 * 1024 * 1024
"""Maximum verify_output payload size (bytes) before spilling to file."""


def _extract_skip_reason_from_messages(messages: Sequence[str]) -> Optional[str]:
    """Return the SKIPPED reason embedded in harness error messages."""
    for msg in messages:
        upper = msg.upper()
        if "SKIPPED" not in upper:
            continue
        if "SKIPPED:" in msg:
            return msg.split("SKIPPED:", 1)[1].strip()
        idx = upper.find("SKIPPED")
        if idx != -1:
            return msg[idx:].strip()
    return None

from core.profiling.gpu_memory_logger import (
    GpuMemoryLogger,
    resolve_gpu_log_interval,
    resolve_gpu_log_path,
    should_enable_gpu_memory_logging,
)
from core.profiling.gpu_telemetry import query_gpu_telemetry


class BenchmarkMode(Enum):
    """Benchmarking mode selection."""
    TRITON = "triton"  # Use triton.testing.do_bench
    PYTORCH = "pytorch"  # Use torch.utils.benchmark.Timer
    CUSTOM = "custom"  # Use CUDA Events / time.perf_counter


class ExecutionMode(str, Enum):
    """How the harness executes benchmarks."""
    SUBPROCESS = "subprocess"
    THREAD = "thread"


class LaunchVia(str, Enum):
    """Which launcher to use for running a benchmark."""
    PYTHON = "python"
    TORCHRUN = "torchrun"


# Import BenchmarkDefaults to use as source of truth for defaults
try:
    from core.benchmark.defaults import BenchmarkDefaults, get_defaults
    _get_defaults_fn: Optional[Callable[[], BenchmarkDefaults]] = get_defaults
except ImportError:
    # Fallback if benchmark defaults are unavailable (docs builds)
    _get_defaults_fn = None


def _get_default_value(attr_name: str, fallback):
    """Get default value from BenchmarkDefaults, with fallback."""
    if _get_defaults_fn is None:
        return fallback
    defaults = _get_defaults_fn()
    return getattr(defaults, attr_name, fallback)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.
    
    Default values are loaded from BenchmarkDefaults, which provides hardcoded defaults.
    This ensures a single source of truth for all configuration defaults.
    Configuration can be overridden via CLI flags or by passing values directly.
    
    Reproducibility Notes:
        - deterministic=True: Enables deterministic algorithms (torch.use_deterministic_algorithms).
          This ensures bitwise reproducibility but may impact performance:
          * cuDNN algorithms may be slower (no autotuning)
          * Some operations may use less efficient deterministic implementations
          * Operations without deterministic implementations may raise errors at runtime
          * Performance impact typically 5-20% depending on workload
          * Use only when reproducibility is more important than performance
        
        - seed: Sets random seeds for Python random, numpy, torch, and CUDA.
          When combined with deterministic=True, ensures reproducible results.
          Without deterministic=True, results may still vary due to non-deterministic algorithms.
    """
    # Use field(default_factory=...) to get defaults from BenchmarkDefaults at instance creation time
    # Each lambda calls _get_default_value() which calls get_defaults() to get current defaults
    iterations: int = field(default_factory=lambda: _get_default_value("iterations", 100))
    warmup: int = field(default_factory=lambda: _get_default_value("warmup", 10))
    min_run_time_ms: float = field(default_factory=lambda: _get_default_value("min_run_time_ms", 100.0))
    percentiles: List[float] = field(default_factory=lambda: (_get_default_value("percentiles", [25, 50, 75, 90, 99]) or [25, 50, 75, 90, 99]).copy())
    enable_memory_tracking: bool = field(default_factory=lambda: _get_default_value("enable_memory_tracking", False))
    deterministic: bool = field(default_factory=lambda: _get_default_value("deterministic", False))
    seed: Optional[int] = field(default_factory=lambda: _get_default_value("seed", None))
    device: Optional[torch.device] = None  # Not in BenchmarkDefaults (runtime-specific)
    enable_profiling: bool = field(default_factory=lambda: _get_default_value("enable_profiling", False))
    enable_nsys: bool = field(default_factory=lambda: _get_default_value("enable_nsys", False))
    enable_ncu: bool = field(default_factory=lambda: _get_default_value("enable_ncu", False))
    enable_proton: bool = field(default_factory=lambda: _get_default_value("enable_proton", False))
    profiling_output_dir: Optional[str] = field(default_factory=lambda: _get_default_value("profiling_output_dir", None))
    subprocess_stderr_dir: Optional[str] = field(default_factory=lambda: _get_default_value("subprocess_stderr_dir", None))
    profile_mode: Optional[str] = field(default_factory=lambda: _get_default_value("profile_mode", "none"))
    profile_type: str = field(default_factory=lambda: _get_default_value("profile_type", "minimal"))
    nsys_preset_override: Optional[str] = field(default_factory=lambda: _get_default_value("nsys_preset_override", None))
    nsys_nvtx_include: Optional[List[str]] = field(default_factory=lambda: _get_default_value("nsys_nvtx_include", None))
    backend_policy: str = field(default_factory=lambda: _get_default_value("backend_policy", "performance"))
    enable_nvtx: Optional[bool] = field(default_factory=lambda: _get_default_value("enable_nvtx", None))
    enable_cleanup: bool = field(default_factory=lambda: _get_default_value("enable_cleanup", False))
    use_subprocess: bool = field(default_factory=lambda: _get_default_value("use_subprocess", True))
    execution_mode: Union[ExecutionMode, str, None] = field(default_factory=lambda: _get_default_value("execution_mode", None))
    launch_via: Union[LaunchVia, str] = field(default_factory=lambda: _get_default_value("launch_via", "python"))
    nproc_per_node: Optional[int] = field(default_factory=lambda: _get_default_value("nproc_per_node", None))
    nnodes: Optional[str] = field(default_factory=lambda: _get_default_value("nnodes", None))
    rdzv_backend: Optional[str] = field(default_factory=lambda: _get_default_value("rdzv_backend", None))
    rdzv_endpoint: Optional[str] = field(default_factory=lambda: _get_default_value("rdzv_endpoint", None))
    env_passthrough: List[str] = field(default_factory=lambda: (_get_default_value("env_passthrough", ["CUDA_VISIBLE_DEVICES"]) or []).copy())
    target_extra_args: Dict[str, List[str]] = field(default_factory=lambda: (_get_default_value("target_extra_args", {}) or {}).copy())
    multi_gpu_required: bool = field(default_factory=lambda: bool(_get_default_value("multi_gpu_required", False)))
    required_world_size: Optional[int] = field(default_factory=lambda: _get_default_value("required_world_size", None))
    single_gpu: bool = field(default_factory=lambda: bool(_get_default_value("single_gpu", False)))
    target_label: Optional[str] = None
    _execution_mode_overridden: bool = field(init=False, repr=False, default=False)
    
    # Per-stage timeouts (in seconds)
    setup_timeout_seconds: Optional[int] = field(default_factory=lambda: _get_default_value("setup_timeout_seconds", 60))
    warmup_timeout_seconds: Optional[int] = field(default_factory=lambda: _get_default_value("warmup_timeout_seconds", None))
    measurement_timeout_seconds: int = field(default_factory=lambda: _get_default_value("measurement_timeout_seconds", 180))
    enable_gpu_memory_logging: bool = field(default_factory=lambda: _get_default_value("enable_gpu_memory_logging", False))
    gpu_memory_log_interval_seconds: float = field(default_factory=lambda: _get_default_value("gpu_memory_log_interval_seconds", 5.0))
    gpu_memory_log_path: Optional[str] = field(default_factory=lambda: _get_default_value("gpu_memory_log_path", None))
    profiling_timeout_seconds: Optional[int] = field(default_factory=lambda: _get_default_value("profiling_timeout_seconds", None))
    ncu_metric_set: str = field(default_factory=lambda: _get_default_value("ncu_metric_set", "auto"))
    pm_sampling_interval: Optional[int] = field(default_factory=lambda: _get_default_value("pm_sampling_interval", None))
    ncu_replay_mode: str = field(default_factory=lambda: _get_default_value("ncu_replay_mode", "application"))
    ncu_replay_mode_override: bool = field(default_factory=lambda: _get_default_value("ncu_replay_mode_override", False))
    profiling_warmup: Optional[int] = field(default_factory=lambda: _get_default_value("profiling_warmup", None))
    """Warmup iterations used by wrapper-based profiler runs (nsys/ncu/proton)."""
    profiling_iterations: Optional[int] = field(default_factory=lambda: _get_default_value("profiling_iterations", None))
    """Iteration count used by wrapper-based profiler runs (nsys/ncu/proton)."""
    profile_env_overrides: Optional[Dict[str, str]] = field(default_factory=lambda: _get_default_value("profile_env_overrides", None))
    """Optional environment overrides applied only to profiler subprocesses."""

    # Triton-style best practices (based on triton/testing.py)
    # See: https://github.com/triton-lang/triton/blob/main/python/triton/testing.py
    clear_l2_cache: bool = field(default_factory=lambda: _get_default_value("clear_l2_cache", True))
    """Clear L2 cache before each iteration to ensure fair memory-bound comparisons."""
    
    full_device_sync: bool = field(default_factory=lambda: _get_default_value("full_device_sync", True))
    """Use torch.cuda.synchronize() instead of event.synchronize() for stream-safe timing.
    Critical for protecting against multi-stream timing exploits (see Locus/KernelBench 2025)."""

    force_synchronize: bool = field(default_factory=lambda: _get_default_value("force_synchronize", False))
    """Force a device-wide synchronize immediately after benchmark_fn() (opt-in safeguard)."""

    timing_method: str = field(default_factory=lambda: _get_default_value("timing_method", "cuda_event"))
    """Timing method for CUDA benchmarks.

    - "cuda_event" (default): Use CUDA events for device-side timing.
    - "wall_clock": Use host wall clock timing + full device sync. Prefer this for
      workloads that launch work on non-default streams (e.g., framework runtimes).
    """
    
    grad_to_none: Optional[List[str]] = field(default_factory=lambda: _get_default_value("grad_to_none", None))
    """List of tensor attribute names to clear gradients for between iterations."""
    
    lock_gpu_clocks: bool = field(default_factory=lambda: _get_default_value("lock_gpu_clocks", False))
    """Lock GPU clocks for consistent benchmarking (requires nvidia-smi permissions)."""
    
    gpu_sm_clock_mhz: Optional[int] = field(default_factory=lambda: _get_default_value("gpu_sm_clock_mhz", None))
    """Target SM clock rate in MHz when lock_gpu_clocks=True."""
    
    gpu_mem_clock_mhz: Optional[int] = field(default_factory=lambda: _get_default_value("gpu_mem_clock_mhz", None))
    """Target memory clock rate in MHz when lock_gpu_clocks=True."""
    
    isolate_warmup_cache: bool = field(default_factory=lambda: _get_default_value("isolate_warmup_cache", True))
    """Clear L2 cache after warmup to prevent warmup from pre-populating caches for measurement.
    This ensures measurement iterations start with a cold L2 cache (Triton best practice)."""
    
    cross_validate_timing: bool = field(default_factory=lambda: _get_default_value("cross_validate_timing", True))
    """Cross-validate CUDA event timing against wall clock timing.
    Warns if CUDA events report significantly less time than wall clock,
    which could indicate timing manipulation or missing stream sync."""
    
    timing_cross_validation_threshold: float = field(default_factory=lambda: _get_default_value("timing_cross_validation_threshold", 0.5))
    """Threshold for timing cross-validation. If CUDA timing / wall clock < this value,
    a warning is issued. Default 0.5 means warn if CUDA reports < 50% of wall time."""
    
    enforce_config_immutability: bool = field(default_factory=lambda: _get_default_value("enforce_config_immutability", True))
    """Enforce that critical timing config (warmup, iterations) cannot be modified by benchmarks.
    If enabled, raises an error if these values change during benchmark execution."""
    
    # Adaptive iterations (Triton-style best practice)
    adaptive_iterations: bool = field(default_factory=lambda: _get_default_value("adaptive_iterations", True))
    """Enable adaptive iteration count to ensure statistically significant measurements.
    When enabled, iterations will be dynamically adjusted to achieve min_total_duration_ms.
    This is a Triton best practice for reliable benchmarking."""
    
    min_total_duration_ms: float = field(default_factory=lambda: _get_default_value("min_total_duration_ms", 100.0))
    """Minimum total duration for all measurement iterations (in ms).
    Only used when adaptive_iterations=True. Iterations continue until this duration is reached.
    Default 100ms provides a good balance of speed and statistical significance."""
    
    max_adaptive_iterations: int = field(default_factory=lambda: _get_default_value("max_adaptive_iterations", 10000))
    """Maximum iterations when using adaptive mode. Prevents runaway if kernels are very fast.
    Default 10000 should handle even microsecond-level kernels."""
    
    # CUDA Graph mode (Triton-style best practice)
    enable_cuda_graph: bool = field(default_factory=lambda: _get_default_value("enable_cuda_graph", False))
    """Enable CUDA Graph capture and replay for benchmarking.
    This can significantly reduce CPU overhead for repetitive kernel launches.
    The benchmark function is captured once, then replayed for all iterations.
    
    Requirements:
    - Benchmark must be deterministic (no data-dependent control flow)
    - All allocations must happen outside the graph
    - No CPU work inside benchmark_fn()
    
    Not compatible with:
    - Dynamic shapes
    - CPU-GPU synchronization inside benchmark_fn()
    - Host-side callbacks
    """
    
    cuda_graph_warmup_iters: int = field(default_factory=lambda: _get_default_value("cuda_graph_warmup_iters", 3))
    """Number of warmup iterations before CUDA graph capture.
    Graph capture requires a warm state for optimal performance."""
    
    # Additional Validity Protections (addressing remaining 49 issues)
    reset_memory_pool: bool = field(default_factory=lambda: _get_default_value("reset_memory_pool", True))
    """Reset CUDA memory pool before each benchmark to prevent memory reuse gaming."""
    
    disable_gc_during_timing: bool = field(default_factory=lambda: _get_default_value("disable_gc_during_timing", True))
    """Disable Python garbage collection during timing to prevent GC interference."""
    
    check_input_output_aliasing: bool = field(default_factory=lambda: _get_default_value("check_input_output_aliasing", True))
    """Verify output tensors don't alias input tensors (prevent pre-filled results)."""
    
    clear_compile_cache: bool = field(default_factory=lambda: _get_default_value("clear_compile_cache", True))
    """Clear torch.compile cache before benchmark to ensure consistent compilation state."""

    audit_stream_sync: bool = field(default_factory=lambda: _get_default_value("audit_stream_sync", True))
    """Enable CUDA stream auditing to catch missing synchronizations by default."""
    
    detect_setup_precomputation: bool = field(default_factory=lambda: _get_default_value("detect_setup_precomputation", True))
    """Hash inputs before/after setup() to detect pre-computation during setup."""

    detect_benchmark_fn_sync: bool = field(
        default_factory=lambda: _get_default_value("detect_benchmark_fn_sync", True)
    )
    """Statically inspect benchmark_fn() for explicit CUDA synchronization calls."""

    benchmark_fn_sync_policy: str = field(
        default_factory=lambda: _get_default_value("benchmark_fn_sync_policy", "warn")
    )
    """Policy for explicit synchronization inside benchmark_fn(): ignore, warn, or error."""

    detect_benchmark_fn_antipatterns: bool = field(
        default_factory=lambda: _get_default_value("detect_benchmark_fn_antipatterns", True)
    )
    """Statically inspect benchmark_fn() for hot-path performance anti-patterns."""

    benchmark_fn_antipattern_policy: str = field(
        default_factory=lambda: _get_default_value("benchmark_fn_antipattern_policy", "warn")
    )
    """Policy for hot-path anti-patterns inside benchmark_fn(): ignore, warn, or error."""
    
    monitor_gpu_state: bool = field(default_factory=lambda: _get_default_value("monitor_gpu_state", True))
    """Monitor GPU temperature, frequency, and power during benchmark."""

    monitor_backend_policy: bool = field(default_factory=lambda: _get_default_value("monitor_backend_policy", True))
    """Monitor backend precision policy flags for mutation during timing."""

    enforce_backend_policy_immutability: bool = field(
        default_factory=lambda: _get_default_value("enforce_backend_policy_immutability", True)
    )
    """Fail-fast when backend precision policy flags change during timing."""
    
    track_memory_allocations: bool = field(default_factory=lambda: _get_default_value("track_memory_allocations", True))
    """Track memory allocations during benchmark to detect suspicious patterns."""
    
    force_tensor_evaluation: bool = field(default_factory=lambda: _get_default_value("force_tensor_evaluation", True))
    """Force evaluation of lazy tensors by calling sync after operations."""

    enforce_environment_validation: bool = field(
        default_factory=lambda: _get_default_value("enforce_environment_validation", True)
    )
    """Fail-fast when validate_environment() reports errors.

    This should remain enabled for performance benchmarking. Virtualization is
    warning-only; disabling this gate is still useful only for correctness/unit
    test contexts where other host policy checks are intentionally bypassed.
    """

    validity_profile: str = field(default_factory=lambda: _get_default_value("validity_profile", "strict"))
    """Benchmark validity profile: strict (default) or portable.

    strict: fail-fast on environment and hardware capability requirements.
    portable: explicit opt-in mode for broader hardware support with loud warnings.
    """

    allow_virtualization: bool = field(default_factory=lambda: _get_default_value("allow_virtualization", False))
    """Compatibility flag retained for profile behavior wiring.

    Virtualization is warning-only in all profiles; this flag remains for
    profile-level compatibility semantics (for example telemetry relaxations).
    """

    allow_foreign_gpu_processes: bool = field(
        default_factory=lambda: _get_default_value("allow_foreign_gpu_processes", False)
    )
    """Warn instead of fail when unrelated CUDA compute processes are active on the benchmark GPU."""

    # Legacy timeout field (deprecated, use measurement_timeout_seconds)
    timeout_seconds: int = field(default_factory=lambda: _get_default_value("timeout_seconds", 180))

    # Graph capture cheat detection thresholds
    graph_capture_cheat_ratio_threshold: float = field(default_factory=lambda: _get_default_value("graph_capture_cheat_ratio_threshold", 10.0))
    """Max allowed capture/replay ratio before flagging a cheat (higher is more permissive)."""
    
    graph_capture_memory_threshold_mb: float = field(default_factory=lambda: _get_default_value("graph_capture_memory_threshold_mb", 100.0))
    """Memory allocated during capture above this threshold is considered suspicious."""
    
    # Profiler-specific timeouts
    nsys_timeout_seconds: int = field(default_factory=lambda: _get_default_value("nsys_timeout_seconds", 120))
    ncu_timeout_seconds: int = field(default_factory=lambda: _get_default_value("ncu_timeout_seconds", 3600))
    proton_timeout_seconds: int = field(default_factory=lambda: _get_default_value("proton_timeout_seconds", 120))
    
    # Timeout multiplier
    timeout_multiplier: float = field(default_factory=lambda: _get_default_value("timeout_multiplier", 1.0))
    
    def __post_init__(self):
        """Set enable_nvtx based on profiling if not explicitly set, apply timeout multiplier, and ensure percentiles is never None."""
        # CRITICAL: Ensure percentiles is always a list, never None (fixes comparison errors)
        if self.percentiles is None:
            self.percentiles = [25, 50, 75, 99]
        if self.env_passthrough is None:
            self.env_passthrough = []
        if self.target_extra_args is None:
            self.target_extra_args = {}

        validity_profile = str(getattr(self, "validity_profile", "strict")).strip().lower()
        if validity_profile not in {"strict", "portable"}:
            raise ValueError(
                f"Invalid validity_profile={validity_profile!r}. Expected 'strict' or 'portable'."
            )
        self.validity_profile = validity_profile
        self.allow_virtualization = validity_profile == "portable"
        
        # CRITICAL: Validate and enforce minimum warmup iterations
        # This ensures JIT/compile overhead is NEVER included in measurements.
        # See benchmark/defaults.py for rationale.
        from core.benchmark.defaults import validate_warmup, MINIMUM_WARMUP_ITERATIONS
        self.warmup = validate_warmup(self.warmup, context="BenchmarkConfig")
        
        if self.enable_nvtx is None:
            # Auto-enable NVTX whenever profiling is requested (for nsys traces)
            self.enable_nvtx = self.enable_profiling

        if self.nsys_preset_override is not None:
            preset = str(self.nsys_preset_override).strip().lower()
            if preset not in {"light", "full"}:
                raise ValueError(
                    f"Invalid nsys_preset_override={self.nsys_preset_override!r}. Expected 'light' or 'full'."
                )
            self.nsys_preset_override = preset
        
        # Apply timeout multiplier to all timeout fields
        if self.timeout_multiplier != 1.0:
            def _apply_timeout_multiplier(field_name: str) -> None:
                current_value = getattr(self, field_name)
                if current_value is None:
                    return
                default_value = _get_default_value(field_name, current_value)
                if current_value == default_value:
                    setattr(
                        self,
                        field_name,
                        int(current_value * self.timeout_multiplier),
                    )
            
            _apply_timeout_multiplier("setup_timeout_seconds")
            _apply_timeout_multiplier("warmup_timeout_seconds")
            _apply_timeout_multiplier("measurement_timeout_seconds")
            _apply_timeout_multiplier("profiling_timeout_seconds")
            _apply_timeout_multiplier("nsys_timeout_seconds")
            _apply_timeout_multiplier("ncu_timeout_seconds")
            _apply_timeout_multiplier("proton_timeout_seconds")
            _apply_timeout_multiplier("timeout_seconds")  # Legacy field
        
        # Set defaults for None timeouts
        if self.warmup_timeout_seconds is None:
            # Warmup timeout defaults to measurement timeout
            self.warmup_timeout_seconds = self.measurement_timeout_seconds
        if self.profiling_timeout_seconds is None:
            # Profiling timeout defaults to max of nsys/ncu timeouts
            self.profiling_timeout_seconds = max(self.nsys_timeout_seconds, self.ncu_timeout_seconds, self.proton_timeout_seconds)

        self._execution_mode_overridden = self.execution_mode is not None
        self._sync_execution_mode()
        self._sync_launch_via()

    def _sync_execution_mode(self) -> None:
        """Ensure execution_mode and use_subprocess remain consistent."""
        override = getattr(self, "_execution_mode_overridden", False)
        mode = self.execution_mode
        if not override:
            mode = ExecutionMode.SUBPROCESS if self.use_subprocess else ExecutionMode.THREAD
        else:
            if isinstance(mode, str):
                try:
                    mode = ExecutionMode(mode.lower())
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid execution_mode '{mode}'. Expected 'subprocess' or 'thread'."
                    ) from exc
            elif not isinstance(mode, ExecutionMode):
                raise TypeError(
                    f"execution_mode must be a string or ExecutionMode, got {type(mode).__name__}"
                )
        self.use_subprocess = (mode == ExecutionMode.SUBPROCESS)
        self.execution_mode = mode

    def _sync_launch_via(self) -> None:
        """Normalize launch_via to LaunchVia enum."""
        mode = self.launch_via
        if isinstance(mode, str):
            try:
                mode = LaunchVia(mode.lower())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid launch_via '{mode}'. Expected 'python' or 'torchrun'."
                ) from exc
        elif not isinstance(mode, LaunchVia):
            raise TypeError(
                f"launch_via must be a string or LaunchVia, got {type(mode).__name__}"
            )
        self.launch_via = mode

    def set_execution_mode(self, mode: Union[ExecutionMode, str]) -> None:
        """Explicitly set execution mode and mark it as user-overridden."""
        self.execution_mode = mode
        self._execution_mode_overridden = True
        self._sync_execution_mode()
        
        # Backward compatibility: sync timeout_seconds with measurement_timeout_seconds
        # (This ensures legacy code using timeout_seconds still works)
        if self.timeout_seconds == 15 and self.measurement_timeout_seconds != 15:
            # If timeout_seconds wasn't explicitly changed, sync it
            self.timeout_seconds = self.measurement_timeout_seconds
    
    def get_effective_timeout(self, stage: str) -> Optional[int]:
        """Get effective timeout for a given stage, accounting for multiplier.
        
        Args:
            stage: One of 'setup', 'warmup', 'measurement', 'profiling', 'torch', 'nsys', 'ncu', 'proton'
            
        Returns:
            Timeout in seconds, or None if no timeout for this stage
        """
        return self.timing.timeout_for(stage)
    
    def capture_timing_snapshot(self) -> Dict[str, Any]:
        """Capture a snapshot of timing-critical configuration values.
        
        Used to detect if a benchmark modifies timing config during execution,
        which would be a form of timing manipulation.
        
        Returns:
            Dict of timing-critical field names to their values
        """
        return self.timing.snapshot()

    @property
    def timing(self) -> "BenchmarkTimingView":
        return _build_timing_view_from_mapping(self.__dict__)

    @property
    def profiling(self) -> "BenchmarkProfilingView":
        return _build_profiling_view_from_mapping(self.__dict__)

    @property
    def validity(self) -> "BenchmarkValidityView":
        return _build_validity_view_from_mapping(self.__dict__)

    @property
    def launch(self) -> "BenchmarkLaunchView":
        return _build_launch_view_from_mapping(self.__dict__)

    @property
    def output(self) -> "BenchmarkOutputView":
        return _build_output_view_from_mapping(self.__dict__)

    def capture_config_snapshot(self) -> Dict[str, Any]:
        """Capture a snapshot of all public configuration values.

        Used to detect runtime mutation of BenchmarkConfig during execution.
        Benchmarks MUST treat harness configuration as immutable once a run starts.
        """
        snapshot: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if str(key).startswith("_"):
                continue
            snapshot[str(key)] = _freeze_benchmark_config_value(value)
        return snapshot

    def verify_config_unchanged(self, snapshot: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Verify configuration values haven't changed since snapshot.

        Args:
            snapshot: Dict from capture_config_snapshot()

        Returns:
            Tuple of (unchanged, error_message)
        """
        current = self.capture_config_snapshot()
        changes: List[str] = []
        all_keys = set(snapshot.keys()) | set(current.keys())
        for key in sorted(all_keys):
            if snapshot.get(key) != current.get(key):
                changes.append(f"{key}: {snapshot.get(key)!r} -> {current.get(key)!r}")
        if changes:
            preview = "; ".join(changes[:10])
            if len(changes) > 10:
                preview = f"{preview}; ... (+{len(changes) - 10} more)"
            return False, f"BenchmarkConfig modified during execution: {preview}"
        return True, None
    
    def verify_timing_unchanged(self, snapshot: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Verify timing-critical fields haven't changed since snapshot.
        
        Args:
            snapshot: Dict from capture_timing_snapshot()
            
        Returns:
            Tuple of (unchanged, error_message)
        """
        current = self.capture_timing_snapshot()
        changes = []
        for key, original_value in snapshot.items():
            current_value = current.get(key)
            if current_value != original_value:
                changes.append(f"{key}: {original_value} -> {current_value}")
        
        if changes:
            return False, f"Timing config modified during execution: {'; '.join(changes)}"
        return True, None

@dataclass(frozen=True)
class BenchmarkTimingView:
    iterations: int
    warmup: int
    min_run_time_ms: float
    percentiles: Tuple[float, ...]
    timing_method: str
    full_device_sync: bool
    force_synchronize: bool
    adaptive_iterations: bool
    min_total_duration_ms: float
    max_adaptive_iterations: int
    enable_cuda_graph: bool
    cuda_graph_warmup_iters: int
    setup_timeout_seconds: Optional[int]
    warmup_timeout_seconds: Optional[int]
    measurement_timeout_seconds: int
    profiling_timeout_seconds: Optional[int]
    nsys_timeout_seconds: int
    ncu_timeout_seconds: int
    proton_timeout_seconds: int
    timeout_multiplier: float

    def timeout_for(self, stage: str) -> Optional[int]:
        return {
            "setup": self.setup_timeout_seconds,
            "warmup": self.warmup_timeout_seconds,
            "measurement": self.measurement_timeout_seconds,
            "profiling": self.profiling_timeout_seconds,
            "torch": self.profiling_timeout_seconds,
            "nsys": self.nsys_timeout_seconds,
            "ncu": self.ncu_timeout_seconds,
            "proton": self.proton_timeout_seconds,
        }.get(stage)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "warmup": self.warmup,
            "iterations": self.iterations,
            "min_run_time_ms": self.min_run_time_ms,
        }


@dataclass(frozen=True)
class BenchmarkProfilingView:
    enable_profiling: bool
    enable_nsys: bool
    enable_ncu: bool
    enable_proton: bool
    enable_nvtx: Optional[bool]
    profile_mode: Optional[str]
    profile_type: str
    nsys_preset_override: Optional[str]
    nsys_nvtx_include: Optional[Tuple[str, ...]]
    ncu_metric_set: str
    pm_sampling_interval: Optional[int]
    ncu_replay_mode: str
    ncu_replay_mode_override: bool
    profiling_warmup: Optional[int]
    profiling_iterations: Optional[int]
    profile_env_overrides: Optional[MappingProxyType]


@dataclass(frozen=True)
class BenchmarkValidityView:
    validity_profile: str
    enforce_environment_validation: bool
    allow_virtualization: bool
    allow_foreign_gpu_processes: bool
    clear_l2_cache: bool
    isolate_warmup_cache: bool
    cross_validate_timing: bool
    timing_cross_validation_threshold: float
    enforce_config_immutability: bool
    reset_memory_pool: bool
    disable_gc_during_timing: bool
    check_input_output_aliasing: bool
    clear_compile_cache: bool
    audit_stream_sync: bool
    detect_setup_precomputation: bool
    detect_benchmark_fn_sync: bool
    benchmark_fn_sync_policy: str
    detect_benchmark_fn_antipatterns: bool
    benchmark_fn_antipattern_policy: str
    monitor_gpu_state: bool
    monitor_backend_policy: bool
    enforce_backend_policy_immutability: bool
    track_memory_allocations: bool
    force_tensor_evaluation: bool
    lock_gpu_clocks: bool
    gpu_sm_clock_mhz: Optional[int]
    gpu_mem_clock_mhz: Optional[int]
    graph_capture_cheat_ratio_threshold: float
    graph_capture_memory_threshold_mb: float


@dataclass(frozen=True)
class BenchmarkLaunchView:
    execution_mode: Union["ExecutionMode", str, None]
    use_subprocess: bool
    launch_via: Union["LaunchVia", str]
    nproc_per_node: Optional[int]
    nnodes: Optional[str]
    rdzv_backend: Optional[str]
    rdzv_endpoint: Optional[str]
    env_passthrough: Tuple[str, ...]
    target_extra_args: MappingProxyType
    multi_gpu_required: bool
    required_world_size: Optional[int]
    single_gpu: bool
    backend_policy: str
    deterministic: bool
    seed: Optional[int]
    device: Optional[torch.device]


@dataclass(frozen=True)
class BenchmarkOutputView:
    enable_memory_tracking: bool
    profiling_output_dir: Optional[str]
    subprocess_stderr_dir: Optional[str]
    enable_cleanup: bool
    enable_gpu_memory_logging: bool
    gpu_memory_log_interval_seconds: float
    gpu_memory_log_path: Optional[str]
    target_label: Optional[str]


def _build_timing_view_from_mapping(values: Dict[str, Any]) -> BenchmarkTimingView:
    return BenchmarkTimingView(
        iterations=int(values["iterations"]),
        warmup=int(values["warmup"]),
        min_run_time_ms=float(values["min_run_time_ms"]),
        percentiles=tuple(float(v) for v in values.get("percentiles", ()) or ()),
        timing_method=str(values["timing_method"]),
        full_device_sync=bool(values["full_device_sync"]),
        force_synchronize=bool(values["force_synchronize"]),
        adaptive_iterations=bool(values["adaptive_iterations"]),
        min_total_duration_ms=float(values["min_total_duration_ms"]),
        max_adaptive_iterations=int(values["max_adaptive_iterations"]),
        enable_cuda_graph=bool(values["enable_cuda_graph"]),
        cuda_graph_warmup_iters=int(values["cuda_graph_warmup_iters"]),
        setup_timeout_seconds=cast(Optional[int], values.get("setup_timeout_seconds")),
        warmup_timeout_seconds=cast(Optional[int], values.get("warmup_timeout_seconds")),
        measurement_timeout_seconds=int(values["measurement_timeout_seconds"]),
        profiling_timeout_seconds=cast(Optional[int], values.get("profiling_timeout_seconds")),
        nsys_timeout_seconds=int(values["nsys_timeout_seconds"]),
        ncu_timeout_seconds=int(values["ncu_timeout_seconds"]),
        proton_timeout_seconds=int(values["proton_timeout_seconds"]),
        timeout_multiplier=float(values["timeout_multiplier"]),
    )


def _build_profiling_view_from_mapping(values: Dict[str, Any]) -> BenchmarkProfilingView:
    nsys_nvtx_include = values.get("nsys_nvtx_include")
    return BenchmarkProfilingView(
        enable_profiling=bool(values["enable_profiling"]),
        enable_nsys=bool(values["enable_nsys"]),
        enable_ncu=bool(values["enable_ncu"]),
        enable_proton=bool(values["enable_proton"]),
        enable_nvtx=cast(Optional[bool], values.get("enable_nvtx")),
        profile_mode=cast(Optional[str], values.get("profile_mode")),
        profile_type=str(values["profile_type"]),
        nsys_preset_override=cast(Optional[str], values.get("nsys_preset_override")),
        nsys_nvtx_include=tuple(str(v) for v in nsys_nvtx_include) if nsys_nvtx_include else None,
        ncu_metric_set=str(values["ncu_metric_set"]),
        pm_sampling_interval=cast(Optional[int], values.get("pm_sampling_interval")),
        ncu_replay_mode=str(values["ncu_replay_mode"]),
        ncu_replay_mode_override=bool(values["ncu_replay_mode_override"]),
        profiling_warmup=cast(Optional[int], values.get("profiling_warmup")),
        profiling_iterations=cast(Optional[int], values.get("profiling_iterations")),
        profile_env_overrides=cast(
            Optional[MappingProxyType],
            _freeze_benchmark_config_value(values.get("profile_env_overrides")),
        ),
    )


def _build_validity_view_from_mapping(values: Dict[str, Any]) -> BenchmarkValidityView:
    return BenchmarkValidityView(
        validity_profile=str(values["validity_profile"]),
        enforce_environment_validation=bool(values["enforce_environment_validation"]),
        allow_virtualization=bool(values["allow_virtualization"]),
        allow_foreign_gpu_processes=bool(values["allow_foreign_gpu_processes"]),
        clear_l2_cache=bool(values["clear_l2_cache"]),
        isolate_warmup_cache=bool(values["isolate_warmup_cache"]),
        cross_validate_timing=bool(values["cross_validate_timing"]),
        timing_cross_validation_threshold=float(values["timing_cross_validation_threshold"]),
        enforce_config_immutability=bool(values["enforce_config_immutability"]),
        reset_memory_pool=bool(values["reset_memory_pool"]),
        disable_gc_during_timing=bool(values["disable_gc_during_timing"]),
        check_input_output_aliasing=bool(values["check_input_output_aliasing"]),
        clear_compile_cache=bool(values["clear_compile_cache"]),
        audit_stream_sync=bool(values["audit_stream_sync"]),
        detect_setup_precomputation=bool(values["detect_setup_precomputation"]),
        detect_benchmark_fn_sync=bool(values["detect_benchmark_fn_sync"]),
        benchmark_fn_sync_policy=str(values["benchmark_fn_sync_policy"]),
        detect_benchmark_fn_antipatterns=bool(values["detect_benchmark_fn_antipatterns"]),
        benchmark_fn_antipattern_policy=str(values["benchmark_fn_antipattern_policy"]),
        monitor_gpu_state=bool(values["monitor_gpu_state"]),
        monitor_backend_policy=bool(values["monitor_backend_policy"]),
        enforce_backend_policy_immutability=bool(values["enforce_backend_policy_immutability"]),
        track_memory_allocations=bool(values["track_memory_allocations"]),
        force_tensor_evaluation=bool(values["force_tensor_evaluation"]),
        lock_gpu_clocks=bool(values["lock_gpu_clocks"]),
        gpu_sm_clock_mhz=cast(Optional[int], values.get("gpu_sm_clock_mhz")),
        gpu_mem_clock_mhz=cast(Optional[int], values.get("gpu_mem_clock_mhz")),
        graph_capture_cheat_ratio_threshold=float(values["graph_capture_cheat_ratio_threshold"]),
        graph_capture_memory_threshold_mb=float(values["graph_capture_memory_threshold_mb"]),
    )


def _build_launch_view_from_mapping(values: Dict[str, Any]) -> BenchmarkLaunchView:
    env_passthrough = values.get("env_passthrough") or []
    return BenchmarkLaunchView(
        execution_mode=cast(Union["ExecutionMode", str, None], values.get("execution_mode")),
        use_subprocess=bool(values["use_subprocess"]),
        launch_via=cast(Union["LaunchVia", str], values["launch_via"]),
        nproc_per_node=cast(Optional[int], values.get("nproc_per_node")),
        nnodes=cast(Optional[str], values.get("nnodes")),
        rdzv_backend=cast(Optional[str], values.get("rdzv_backend")),
        rdzv_endpoint=cast(Optional[str], values.get("rdzv_endpoint")),
        env_passthrough=tuple(str(v) for v in env_passthrough),
        target_extra_args=cast(
            MappingProxyType,
            _freeze_benchmark_config_value(values.get("target_extra_args") or {}),
        ),
        multi_gpu_required=bool(values["multi_gpu_required"]),
        required_world_size=cast(Optional[int], values.get("required_world_size")),
        single_gpu=bool(values["single_gpu"]),
        backend_policy=str(values["backend_policy"]),
        deterministic=bool(values["deterministic"]),
        seed=cast(Optional[int], values.get("seed")),
        device=cast(Optional[torch.device], values.get("device")),
    )


def _build_output_view_from_mapping(values: Dict[str, Any]) -> BenchmarkOutputView:
    return BenchmarkOutputView(
        enable_memory_tracking=bool(values["enable_memory_tracking"]),
        profiling_output_dir=cast(Optional[str], values.get("profiling_output_dir")),
        subprocess_stderr_dir=cast(Optional[str], values.get("subprocess_stderr_dir")),
        enable_cleanup=bool(values["enable_cleanup"]),
        enable_gpu_memory_logging=bool(values["enable_gpu_memory_logging"]),
        gpu_memory_log_interval_seconds=float(values["gpu_memory_log_interval_seconds"]),
        gpu_memory_log_path=cast(Optional[str], values.get("gpu_memory_log_path")),
        target_label=cast(Optional[str], values.get("target_label")),
    )


def _freeze_benchmark_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze_benchmark_config_value(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_benchmark_config_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_benchmark_config_value(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze_benchmark_config_value(v) for v in value)
    return value


class ReadOnlyBenchmarkConfigView:
    """Read-only view of BenchmarkConfig exposed to benchmark code.

    Benchmarks MUST NOT mutate harness configuration at runtime. The harness
    passes this view via ``benchmark._config`` so that accidental or intentional
    mutation attempts fail fast without affecting harness behavior.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Dict[str, Any]) -> None:
        object.__setattr__(self, "_values", values)

    @classmethod
    def from_config(cls, config: "BenchmarkConfig") -> "ReadOnlyBenchmarkConfigView":
        values: Dict[str, Any] = {}
        for key, value in config.__dict__.items():
            if str(key).startswith("_"):
                continue
            values[str(key)] = _freeze_benchmark_config_value(value)
        return cls(values)

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name!r}")

    @property
    def timing(self) -> BenchmarkTimingView:
        return _build_timing_view_from_mapping(object.__getattribute__(self, "_values"))

    @property
    def profiling(self) -> BenchmarkProfilingView:
        return _build_profiling_view_from_mapping(object.__getattribute__(self, "_values"))

    @property
    def validity(self) -> BenchmarkValidityView:
        return _build_validity_view_from_mapping(object.__getattribute__(self, "_values"))

    @property
    def launch(self) -> BenchmarkLaunchView:
        return _build_launch_view_from_mapping(object.__getattribute__(self, "_values"))

    @property
    def output(self) -> BenchmarkOutputView:
        return _build_output_view_from_mapping(object.__getattribute__(self, "_values"))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "Benchmark config is read-only during execution. "
            "Override get_config() to return desired config values."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict(object.__getattribute__(self, '_values'))!r})"


@dataclass
class WorkloadMetadata:
    """Metadata describing workload units processed per benchmark iteration."""
    
    requests_per_iteration: float = 1.0
    tokens_per_iteration: Optional[float] = None
    samples_per_iteration: Optional[float] = None
    bytes_per_iteration: Optional[float] = None
    custom_units_per_iteration: Optional[float] = None
    custom_unit_name: Optional[str] = None
    goodput: Optional[float] = None


@dataclass
class TorchrunLaunchSpec:
    """Declarative description of a torchrun invocation."""

    script_path: Optional[Path] = None
    module_name: Optional[str] = None
    script_args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    parse_rank0_only: bool = True
    multi_gpu_required: bool = False
    name: Optional[str] = None
    config_arg_map: Dict[str, str] = field(default_factory=dict)


# BenchmarkResult is provided by benchmark_models.py
BenchmarkResult = PydanticBenchmarkResult


class BaseBenchmark:
    """Base class for benchmarks with shared functionality.
    
    Provides common patterns for device resolution, setup, teardown, validation,
    NVTX range management (_nvtx_range()), and CUDA synchronization (_synchronize()).
    Benchmarks should inherit from this class (or a helper base that derives from it).
    
    Usage:
        class MyBenchmark(BaseBenchmark):
            def __init__(self):
                super().__init__()
                self.model = None
                self.data = None
            
            def setup(self) -> None:
                torch.manual_seed(42)
                self.model = nn.Linear(256, 256).to(self.device)
                self.data = torch.randn(32, 256, device=self.device)
                torch.cuda.synchronize()
            
            def benchmark_fn(self) -> None:
                with self._nvtx_range("my_benchmark"):
                    _ = self.model(self.data)
    """
    
    allow_cpu: bool = False
    multi_gpu_required: bool = False
    required_world_size: Optional[int] = None
    story_metadata: Optional[Dict[str, Any]] = None

    def __init__(self):
        """Initialize benchmark without forcing runtime-only device checks.

        Static inspection paths should be able to instantiate benchmarks and
        read metadata or input signatures without bringing up CUDA. The first
        actual access to ``self.device`` still fails fast when CUDA is required.
        """
        self._device: Optional[torch.device] = None
        self._config = None  # Cache for get_config()
        self._workload_metadata: Optional[WorkloadMetadata] = None
        self._workload_registered: bool = False
        self._execution_marker: Optional[torch.Tensor] = None
        self._verification_payload = None

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = self._resolve_device()
        return self._device

    @device.setter
    def device(self, value: Optional[torch.device]) -> None:
        self._device = value
    
    def _resolve_device(self) -> torch.device:
        """Resolve CUDA device, failing fast if CUDA is not available.
        
        Returns:
            torch.device("cuda") if CUDA is available
            
        Raises:
            RuntimeError: If CUDA is not available (NVIDIA GPU required)
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(self, "allow_cpu", False):
            return torch.device("cpu")
        raise RuntimeError("CUDA required - NVIDIA GPU and tools must be available")
    
    def setup(self) -> None:
        """Setup phase: initialize models, data, etc.
        
        Subclasses should override this method to implement their specific setup logic.
        """
        pass
    
    def benchmark_fn(self) -> None:
        """Function to benchmark. Must be callable with no args.
        
        Subclasses must override this method to implement their benchmark logic.
        """
        raise NotImplementedError("Subclasses must implement benchmark_fn()")

    def capture_verification_payload(self) -> None:
        """Optional post-timing hook to set verification payload.

        Benchmarks that want to keep the timed hot path clean can override this
        method and call ``_set_verification_payload()`` using tensors already
        produced by the timing run. The harness calls this once after
        measurement (and VerifyRunner calls it after verify runs).
        """
        return None
    
    def teardown(self) -> None:
        """Cleanup phase.
        
        Default implementation clears CUDA cache. Subclasses can override
        to add additional cleanup logic.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Clear any cached config from a previous harness run
        self._config = None
    
    def get_config(self) -> Optional[BenchmarkConfig]:
        """Return the active harness config when available.
        
        Subclasses can override to provide custom overrides; the harness
        will still stash the merged config on ``self._config`` for runtime checks.
        """
        return getattr(self, "_config", None)

    def get_verify_inputs(self) -> Dict[str, torch.Tensor]:
        """Return input tensors used for verification/aliasing checks."""
        if isinstance(self, VerificationPayloadMixin):
            # Payload-backed benchmarks raise RuntimeError until capture_verification_payload()
            # populates the payload. Compliance tooling treats RuntimeError as "not executed yet".
            return VerificationPayloadMixin.get_verify_inputs(self)
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_verify_inputs() explicitly or call "
            "_set_verification_payload() from capture_verification_payload() (post-timing)."
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> Optional[TorchrunLaunchSpec]:
        """Optional torchrun launch description for multi-GPU benchmarks."""
        return None
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result, return error message if invalid.
        
        Default implementation returns None (no validation).
        Subclasses should override to implement validation logic.
        
        Returns:
            None if validation passes, or error message string if validation fails
        """
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return the output tensor for verification comparison.
        
        MANDATORY: Every benchmark MUST implement this method explicitly.
        There are NO fallbacks or auto-detection.
        
        This method is used by the verification system to compare outputs
        between baseline and optimized benchmarks.
        
        Returns:
            Output tensor for verification
            
        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        if isinstance(self, VerificationPayloadMixin):
            # Payload-backed benchmarks raise RuntimeError until capture_verification_payload()
            # populates the payload. Compliance tooling treats RuntimeError as "not executed yet".
            return VerificationPayloadMixin.get_verify_output(self)
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_verify_output() explicitly or call "
            "_set_verification_payload() from capture_verification_payload() (post-timing) to register outputs. "
            "Checksums or fixed scalars are not allowed—surface the real tensor (or a representative slice)."
        )

    def skip_input_verification(self) -> bool:
        """Return True to skip input equivalence verification for this benchmark.
        
        Override this in benchmarks where input verification is not applicable
        (e.g., benchmarks that intentionally use different input sizes to test
        scaling behavior, or informational benchmarks).
        
        Returns:
            False by default (verification enabled)
        """
        # Allow opt-in skips via attribute without overriding the method
        if getattr(self, "skip_input_check", False):
            return True
        return False

    def skip_output_verification(self) -> bool:
        """Return True to skip output correctness verification for this benchmark.
        
        Override this in benchmarks where output verification is not applicable
        (e.g., benchmarks that produce non-deterministic outputs, or benchmarks
        that test different algorithms with legitimately different results).
        
        Returns:
            False by default (verification enabled)
        """
        # Allow opt-in skips via attribute without overriding the method
        if getattr(self, "skip_output_check", False):
            return True
        return False

    def get_input_signature(self) -> Dict[str, Any]:
        """MANDATORY: Return a signature describing this benchmark's input workload.
        
        Used by the harness to verify that baseline and optimized benchmarks
        operate on equivalent workloads. Without this verification, performance
        comparisons are meaningless.
        
        The signature MUST capture all parameters that affect workload size:
        - Tensor shapes (batch_size, seq_len, hidden_size, etc.)
        - Model parameters (num_layers, num_heads, etc.)
        - Any other configuration that affects computational work
        
        NO AUTO-INFERENCE. NO FALLBACKS. EVERYTHING EXPLICIT.
        
        Every benchmark MUST override this method and return a non-empty dict.
        
        Returns:
            Dict describing the input workload (MUST be non-empty)
            
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        if isinstance(self, VerificationPayloadMixin):
            # Payload-backed benchmarks raise RuntimeError until capture_verification_payload()
            # populates the payload. Compliance tooling treats RuntimeError as "not executed yet".
            return VerificationPayloadMixin.get_input_signature(self)
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_input_signature() explicitly. "
            "NO AUTO-INFERENCE. NO FALLBACKS. Return a dict with workload parameters "
            "(e.g., {'batch_size': 32, 'seq_len': 512}) or call _set_verification_payload() "
            "from capture_verification_payload() (post-timing)."
        )

    def get_output_tolerance(self) -> Tuple[float, float]:
        """MANDATORY: Return (rtol, atol) tolerance for output comparison."""
        if isinstance(self, VerificationPayloadMixin):
            # Payload-backed benchmarks raise RuntimeError until capture_verification_payload()
            # populates the payload. Compliance tooling treats RuntimeError as "not executed yet".
            return VerificationPayloadMixin.get_output_tolerance(self)
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_output_tolerance() explicitly or provide a "
            "tolerance via _set_verification_payload() from capture_verification_payload() (post-timing)."
        )

    def get_custom_streams(self) -> List["torch.cuda.Stream"]:
        """Return any non-default streams used by this benchmark."""
        return []

    # When True, all streams observed by the StreamAuditor will be treated as
    # declared for timing validation. Benchmarks can set this to False to opt
    # out and rely solely on get_custom_streams().
    declare_all_streams: bool = True

    def mark_execution_complete(self) -> None:
        """Mark that this benchmark executed work on this rank/device."""
        try:
            self._execution_marker = torch.tensor([1.0], device=self.device)
        except Exception:
            self._execution_marker = torch.tensor([1.0])

    def _verify_kernel(
        self,
        kernel_fn,
        reference_fn,
        shape: tuple,
        *,
        formal: bool = False,
        kernel_source: Optional[str] = None,
    ) -> tuple:
        """Verify kernel correctness using core/verification/kernel_correctness.
        
        Helper method for benchmarks to verify their kernels match a reference.
        Can be called in validate_result() or anywhere else.
        
        Args:
            kernel_fn: The kernel function to verify
            reference_fn: Reference implementation to compare against
            shape: Input tensor shape for testing
            formal: If True, use formal verification with proofs (slower but thorough)
            kernel_source: CUDA source code (required if formal=True)
            
        Returns:
            Tuple of (passed: bool, result: VerificationResult or FormalVerificationResult)
            
        Example:
            def validate_result(self) -> Optional[str]:
                passed, result = self._verify_kernel(
                    self.my_kernel, 
                    self.reference_impl,
                    shape=(1024, 1024),
                )
                if not passed:
                    return f"Kernel verification failed: {result}"
                return None
        """
        try:
            if formal:
                from core.verification.kernel_correctness import FormalKernelVerifier
                verifier = FormalKernelVerifier(device=str(self.device))
                result = verifier.verify(kernel_fn, reference_fn, kernel_source or "")
                return (result.all_proven, result)
            else:
                from core.verification.kernel_correctness import ManualKernelVerifier
                verifier = ManualKernelVerifier(device=str(self.device))
                result = verifier.verify(kernel_fn, reference_fn, shape)
                return (result.all_passed, result)
        except ImportError:
            # Verification tools not available
            return (True, None)

    def register_workload_metadata(
        self,
        *,
        requests_per_iteration: Optional[float] = None,
        tokens_per_iteration: Optional[float] = None,
        samples_per_iteration: Optional[float] = None,
        bytes_per_iteration: Optional[float] = None,
        custom_units_per_iteration: Optional[float] = None,
        custom_unit_name: Optional[str] = None,
        goodput: Optional[float] = None,
    ) -> None:
        """Register workload metadata for throughput calculations."""
        self._workload_metadata = WorkloadMetadata(
            requests_per_iteration=requests_per_iteration if requests_per_iteration is not None else 1.0,
            tokens_per_iteration=tokens_per_iteration,
            samples_per_iteration=samples_per_iteration,
            bytes_per_iteration=bytes_per_iteration,
            custom_units_per_iteration=custom_units_per_iteration,
            custom_unit_name=custom_unit_name,
            goodput=goodput,
        )
        # Mark explicit registration for compliance/audit tooling.
        self._workload_registered = True

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        """Return workload metadata if registered."""
        if self._workload_metadata is not None:
            return self._workload_metadata
        inferred = self._infer_workload_metadata()
        if inferred is not None:
            self._workload_metadata = inferred
        return self._workload_metadata

    def _infer_workload_metadata(self) -> Optional[WorkloadMetadata]:
        """Workload inference is disabled; benchmarks must register explicitly."""
        return None

    def finalize_iteration_metrics(self) -> Optional[Dict[str, List[float]]]:
        """Optional post-sync hook for per-iteration metrics.

        The harness calls this immediately after it has synchronized the device
        for a timed iteration. Benchmarks can use it to turn deferred CUDA
        events recorded in ``benchmark_fn()`` into numeric metrics without
        introducing host-side synchronization into the timed hot path.
        """
        return None

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        """Benchmarks can override to expose custom metrics."""
        return None

    def get_story_metadata(self) -> Optional[Dict[str, Any]]:
        """Return non-numeric story/intent metadata for reports and audits."""
        candidate = getattr(self, "story_metadata", None)
        if not isinstance(candidate, dict) or not candidate:
            return None

        def _normalize(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, (str, bool, int, float)):
                return value
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                value = list(value)
            if isinstance(value, list):
                items = []
                for item in value:
                    normalized = _normalize(item)
                    if normalized is not None:
                        items.append(normalized)
                return items
            if isinstance(value, dict):
                normalized_dict: Dict[str, Any] = {}
                for key, item in value.items():
                    if not isinstance(key, str) or not key.strip():
                        continue
                    normalized = _normalize(item)
                    if normalized is not None:
                        normalized_dict[key] = normalized
                return normalized_dict
            return str(value)

        normalized_payload: Dict[str, Any] = {}
        for key, value in candidate.items():
            if not isinstance(key, str) or not key.strip():
                continue
            normalized = _normalize(value)
            if normalized is not None:
                normalized_payload[key] = normalized
        return normalized_payload or None

    def get_optimization_goal(self) -> str:
        """Return the primary optimization goal for this benchmark.
        
        Override this method to indicate what metric the optimization targets.
        The harness will use this to evaluate success appropriately.
        
        Returns:
            One of: "speed", "memory", "throughput", "latency", "power"
            Default is "speed" (lower time = better)
            
        Examples:
            - "speed": Traditional speedup (baseline_time / optimized_time)
            - "memory": Memory reduction (baseline_memory / optimized_memory)
            - "throughput": Higher is better (optimized_throughput / baseline_throughput)
            - "latency": Lower is better (baseline_latency / optimized_latency)
        """
        return "speed"

    def get_min_speedup_for_success(self) -> Optional[float]:
        """Return an explicit minimum speedup contract for speed-goal benchmarks.

        Override this on benchmarks whose local success criteria differ from the
        harness-wide fallback threshold. Return ``None`` to use the shared
        default threshold in the benchmark runner.
        """
        return None
    
    def _record_start(self) -> float:
        """Return a high-resolution timestamp for chunk-level timing."""
        return time.perf_counter()

    def _record_stop(self, start: float) -> float:
        """Return elapsed time in milliseconds since ``start``."""
        return (time.perf_counter() - start) * 1000.0
    
    def _scale_workload_by_memory(self, base_size: int) -> int:
        """Scale workload size based on available GPU memory.
        
        Args:
            base_size: Base workload size for large GPUs (>=16GB)
            
        Returns:
            Scaled workload size based on GPU memory:
            - >=16GB: base_size (100%)
            - >=8GB: base_size * 0.5 (50%)
            - >=4GB: base_size * 0.25 (25%)
            - <4GB: base_size * 0.1 (10%)
        """
        if not torch.cuda.is_available():
            return base_size
        
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_memory_gb >= 16:
            return base_size
        elif total_memory_gb >= 8:
            return int(base_size * 0.5)
        elif total_memory_gb >= 4:
            return int(base_size * 0.25)
        else:
            return int(base_size * 0.1)

    def to_device(
        self,
        tensor: torch.Tensor,
        *,
        non_blocking: Optional[bool] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Copy ``tensor`` to ``device`` while defaulting to non_blocking=True when the
        source resides in pinned host memory. Benchmarks can override ``device`` or
        ``non_blocking`` for synchronous copies.
        """

        target = device if device is not None else getattr(self, "device", None)
        if target is None:
            target = self._resolve_device()
        if target.type == "cuda" and hasattr(tensor, "is_pinned") and not tensor.is_cuda:
            blocking = tensor.is_pinned() if non_blocking is None else non_blocking
            return tensor.to(target, non_blocking=bool(blocking))
        return tensor.to(target)
    
    @contextmanager
    def _nvtx_range(self, name: str):
        """Context manager for NVTX ranges with automatic enable/disable.
        
        Automatically checks if NVTX is enabled via get_config().
        
        Args:
            name: Name for the NVTX range
            
        Usage:
            with self._nvtx_range("my_operation"):
                # code to profile
                pass
        """
        from core.profiling.nvtx_helper import nvtx_range, get_nvtx_enabled
        
        # Prefer the harness-merged config if present to honor CLI flags
        config = getattr(self, "_config", None) or self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        with nvtx_range(name, enable=enable_nvtx):
            yield
    
    def _synchronize(self) -> None:
        """Synchronize CUDA device if available.
        
        Convenience method for benchmarks to ensure operations complete.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)


def _maybe_write_subprocess_stderr(stderr: str, benchmark_name: str, config: BenchmarkConfig) -> None:
    if not stderr:
        return
    env_flag = os.environ.get("AISP_CAPTURE_SUBPROCESS_STDERR")
    if env_flag is not None:
        if str(env_flag).strip().lower() not in {"1", "true", "yes", "y", "on"}:
            return
    output_dir = getattr(config, "subprocess_stderr_dir", None) or getattr(config, "profiling_output_dir", None)
    if not output_dir:
        return
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", benchmark_name).strip("_")
        path = Path(output_dir) / f"{slug}_subprocess.stderr.log"
        path.write_text(stderr)
    except Exception as exc:
        if LOGGER_AVAILABLE:
            logger.warning("Failed to persist subprocess stderr for %s: %s", benchmark_name, exc)


class BenchmarkHarness:
    """Production-grade benchmarking harness with profiling support."""
    
    def __init__(
        self,
        mode: BenchmarkMode = BenchmarkMode.CUSTOM,
        config: Optional[BenchmarkConfig] = None,
        environment_probe: Optional[EnvironmentProbe] = None,
    ):
        self.mode = mode
        self.config = config or BenchmarkConfig()
        self.config._sync_execution_mode()
        self.config._sync_launch_via()
        self.device = self.config.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._seed_info: Optional[Dict[str, Any]] = None
        self._runtime_initialized = False
        self._thread_executor: Optional[ThreadPoolExecutor] = None
        self._environment_probe = environment_probe or EnvironmentProbe()

    def _ensure_runtime_initialized(self) -> None:
        """Perform CUDA-touching runtime setup only in the executing process."""
        if self._runtime_initialized:
            return
        self._seed_info = self._setup_reproducibility()
        self._runtime_initialized = True
    
    def _setup_reproducibility(self) -> Dict[str, Any]:
        """Setup for reproducible benchmarks.
        
        Returns:
            Dictionary with seed values that were set (for manifest capture).
        """
        seed_info: Dict[str, Any] = {
            "random_seed": None,
            "numpy_seed": None,
            "torch_seed": None,
            "cuda_seed": None,
            "deterministic_mode": self.config.deterministic,
        }
        
        if self.config.seed is not None:
            seed = self.config.seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            seed_info["random_seed"] = seed
            seed_info["numpy_seed"] = seed
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            if LOGGER_AVAILABLE:
                logger.info(f"Seeds set to {seed} (random, numpy, torch, cuda)")
        
        # Always capture current seeds for mutation detection, even if we did not explicitly seed.
        try:
            seed_info["torch_seed"] = int(torch.initial_seed())
        except Exception as exc:
            raise RuntimeError("Failed to read torch.initial_seed() for seed tracking") from exc

        if torch.cuda.is_available():
            try:
                seed_info["cuda_seed"] = int(torch.cuda.initial_seed())
            except Exception as exc:
                raise RuntimeError("Failed to read torch.cuda.initial_seed() for seed tracking") from exc

        return seed_info

    @contextmanager
    def _nvtx_range(self, name: str):
        """Context manager for NVTX ranges with automatic enable/disable.
        
        Args:
            name: Name for the NVTX range
        """
        from core.profiling.nvtx_helper import nvtx_range, get_nvtx_enabled
        
        config = self.config
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        with nvtx_range(name, enable=enable_nvtx):
            yield

    def _world_size_hint(self, config: BenchmarkConfig) -> Optional[int]:
        """Best-effort world size estimation from config."""
        nproc = getattr(config, "nproc_per_node", None)
        nnodes = getattr(config, "nnodes", None)
        world_size: Optional[int] = None
        try:
            if nproc is not None:
                world_size = int(nproc)
        except Exception:
            world_size = None
        if nnodes is not None:
            try:
                nodes_int = int(str(nnodes).split(":")[0])
                world_size = (world_size or 1) * nodes_int
            except ValueError:
                pass  # nnodes not a valid integer
        if world_size is None:
            if getattr(config, "launch_via", LaunchVia.PYTHON) == LaunchVia.PYTHON:
                if getattr(config, "multi_gpu_required", False) and torch.cuda.is_available():
                    world_size = torch.cuda.device_count()
                else:
                    world_size = 1
            elif torch.cuda.is_available():
                world_size = torch.cuda.device_count()
        return world_size

    def _visible_gpu_count(self) -> int:
        """Return number of visible GPUs based on CUDA_VISIBLE_DEVICES or CUDA runtime."""
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            tokens = [tok.strip() for tok in visible.split(",") if tok.strip()]
            return len(tokens)
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0

    def _select_single_gpu_visible(self) -> str:
        """Choose a single GPU ID from CUDA_VISIBLE_DEVICES or default to '0'."""
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            tokens = [tok.strip() for tok in visible.split(",") if tok.strip()]
            if tokens:
                return tokens[0]
        return "0"

    def _resolve_world_size_requirement(
        self,
        benchmark: BaseBenchmark,
        config: BenchmarkConfig,
    ) -> Tuple[int, bool]:
        """Resolve required world size and whether it is exact."""
        required = getattr(config, "required_world_size", None)
        if required is None:
            required = getattr(benchmark, "required_world_size", None)
        if required is not None:
            required = int(required)
            if required <= 0:
                raise ValueError(f"required_world_size must be positive, got {required}")
            return required, True
        multi_gpu_required = bool(
            getattr(config, "multi_gpu_required", False) or getattr(benchmark, "multi_gpu_required", False)
        )
        if multi_gpu_required:
            return 2, False
        return 1, False

    def _enforce_world_size_requirement(self, benchmark: BaseBenchmark, config: BenchmarkConfig) -> None:
        """Fail fast if world size/visible GPUs do not match requirements."""
        if getattr(benchmark, "allow_cpu", False) and getattr(benchmark, "device", None) is not None:
            if getattr(benchmark, "device").type != "cuda":
                return
        required, exact = self._resolve_world_size_requirement(benchmark, config)
        if getattr(config, "single_gpu", False) and required != 1:
            raise RuntimeError(
                f"single_gpu=True conflicts with required_world_size={required}. "
                "Unset --single-gpu or adjust required_world_size."
            )

        visible = self._visible_gpu_count()
        effective_visible = visible
        if getattr(config, "single_gpu", False) and getattr(config, "use_subprocess", False):
            effective_visible = 1

        world_size_hint = self._world_size_hint(config)
        if exact:
            if effective_visible and effective_visible != required:
                raise RuntimeError(
                    f"World size mismatch: requires exactly {required} visible GPU(s), "
                    f"found {effective_visible}. Set CUDA_VISIBLE_DEVICES accordingly."
                )
            if world_size_hint and world_size_hint != required:
                raise RuntimeError(
                    f"World size mismatch: requires exactly {required} rank(s), "
                    f"configured world_size={world_size_hint}."
                )
        else:
            if effective_visible and effective_visible < required:
                raise RuntimeError(
                    f"World size mismatch: requires >= {required} visible GPU(s), "
                    f"found {effective_visible}. Set CUDA_VISIBLE_DEVICES accordingly."
                )
            if world_size_hint and world_size_hint < required:
                raise RuntimeError(
                    f"World size mismatch: requires >= {required} rank(s), "
                    f"configured world_size={world_size_hint}."
                )

    def _annotate_launch_metadata(
        self,
        result: PydanticBenchmarkResult,
        config: BenchmarkConfig,
        *,
        world_size: Optional[int] = None,
        multi_gpu_required: Optional[bool] = None,
    ) -> None:
        """Attach launch metadata onto a BenchmarkResult in-place."""
        result.launch_via = getattr(config, "launch_via", None)
        resolved_world_size = world_size if world_size is not None else self._world_size_hint(config)
        result.world_size = resolved_world_size
        requires_multi = getattr(config, "multi_gpu_required", False) if multi_gpu_required is None else multi_gpu_required
        result.multi_gpu_required = bool(requires_multi)
        result.multi_gpu = bool(resolved_world_size and resolved_world_size > 1)

    def _apply_target_overrides(self, benchmark: BaseBenchmark, config: BenchmarkConfig) -> None:
        """Apply per-target CLI overrides to a benchmark instance, if supported."""
        target_label = getattr(config, "target_label", None)
        if not target_label:
            return
        overrides_map = getattr(config, "target_extra_args", {}) or {}
        overrides = _lookup_target_extra_args(overrides_map, target_label)
        if not overrides:
            return
        argv = shlex.split(overrides) if isinstance(overrides, str) else list(overrides)
        if not argv:
            return
        override_signature = (target_label, tuple(argv))
        if getattr(benchmark, "_applied_target_override_signature", None) == override_signature:
            return
        hook = getattr(benchmark, "apply_target_overrides", None)
        if callable(hook):
            try:
                hook(argv)
                setattr(benchmark, "_applied_target_override_signature", override_signature)
            except Exception:
                if LOGGER_AVAILABLE:
                    logger.warning("Ignoring target overrides for %s due to error", target_label, exc_info=True)
    
    def _create_timeout_result(
        self,
        *,
        stage: str,
        duration: float,
        limit: Optional[float],
        errors: List[str],
        benchmark_name: str,
        config: BenchmarkConfig,
        watchdog: Optional[Dict[str, Any]] = None,
    ) -> PydanticBenchmarkResult:
        """Helper to build a timeout result with consistent schema."""
        timeout_timing = TimingStats(
            mean_ms=0.0,
            median_ms=0.0,
            std_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            p50_ms=None,
            p90_ms=None,
            p95_ms=None,
            p99_ms=None,
            percentiles={},
            iterations=0,
            warmup_iterations=config.warmup,
            raw_times_ms=[],
            schemaVersion="1.0",
        )
        result = PydanticBenchmarkResult(
            timing=timeout_timing,
            errors=errors,
            timeout_stage=stage,
            timeout_duration_seconds=duration,
            timeout_limit_seconds=int(limit) if limit is not None else None,
            inference_timing=None,
            memory=None,
            artifacts=None,
            profiler_metrics=None,
            validation_status=None,
            validation_message=json.dumps(watchdog) if watchdog else None,
            benchmark_name=benchmark_name,
            device=str(self.device),
            mode=self.mode.value,
            seeds=copy.deepcopy(getattr(self, "_seed_info", None)),
            schemaVersion="1.0",
        )
        self._annotate_launch_metadata(result, config, world_size=self._world_size_hint(config))
        return result
    
    def _benchmark_with_torchrun(self, benchmark: BaseBenchmark, config: BenchmarkConfig) -> PydanticBenchmarkResult:
        """Launch benchmark via torchrun for multi-GPU targets."""
        import inspect
        print("[harness] _benchmark_with_torchrun start", flush=True)

        def _filter_logs(lines: List[str], dedupe: bool = True) -> List[str]:
            filtered: List[str] = []
            seen: set = set()
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                key = stripped.lower()
                if dedupe and key in seen:
                    continue
                seen.add(key)
                # Keep messages that either don't mention rank or explicitly mention rank 0
                if "rank 0" in key or "local_rank: 0" in key or "r0" in key:
                    filtered.append(stripped)
                elif "rank" not in key or "world_size" in key:
                    filtered.append(stripped)
            return filtered

        def _extract_tokens_per_s(lines: List[str]) -> Optional[float]:
            pattern = re.compile(r"([0-9][0-9,\\.]+)\\s*(tok(?:ens)?/s|toks/s|tok/s)", re.IGNORECASE)
            best: Optional[float] = None
            for line in lines:
                match = pattern.search(line)
                if not match:
                    continue
                try:
                    candidate = float(match.group(1).replace(",", ""))
                    best = candidate if best is None else max(best, candidate)
                except ValueError:
                    continue
            return best

        errors: List[str] = []
        spec: Optional[TorchrunLaunchSpec] = None
        try:
            getter = getattr(benchmark, "get_torchrun_spec", None)
            if callable(getter):
                spec = getter(config)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"Failed to build torchrun spec: {exc}")
        if spec is None:
            # Fallback to module path if provided
            benchmark_module = inspect.getmodule(benchmark) or inspect.getmodule(benchmark.__class__)
            module_path = Path(getattr(benchmark_module, "__file__", "")).resolve()
            spec = TorchrunLaunchSpec(script_path=module_path)
        print(
            f"[harness] torchrun spec script={spec.script_path} module={spec.module_name} args={spec.script_args}",
            flush=True,
        )

        script_path = Path(spec.script_path).resolve() if spec and spec.script_path is not None else None
        module_name = spec.module_name if spec else None
        if bool(script_path) == bool(module_name):
            raise RuntimeError(
                "SKIPPED: Torchrun launch requires exactly one of script_path or module_name."
            )
        if script_path is not None and not script_path.exists():
            raise RuntimeError("SKIPPED: Torchrun launch requested but script_path does not exist.")

        world_size_hint = self._world_size_hint(config)
        multi_gpu_required = bool(spec.multi_gpu_required or getattr(config, "multi_gpu_required", False))
        if multi_gpu_required and (world_size_hint is not None and world_size_hint < 2):
            message = "SKIPPED: Multi-GPU benchmark requires nproc_per_node>=2."
            errors.append(message)
            raise RuntimeError(message)

        nproc_per_node = getattr(config, "nproc_per_node", None) or 1
        def _sockets_permitted() -> bool:
            """Return True if the runtime can create sockets (required by torchrun rendezvous)."""
            try:
                import socket

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.close()
                return True
            except Exception:
                return False

        sockets_permitted = _sockets_permitted()

        # Torchrun rendezvous uses TCPStore even when nproc_per_node=1. Some sandboxed
        # environments disallow socket syscalls entirely; in that case, fall back to
        # launching the wrapper directly for single-process runs so we can still enforce
        # invariants like seed immutability.
        if not sockets_permitted:
            nnodes = int(getattr(config, "nnodes", None) or 1)
            if nnodes != 1:
                raise RuntimeError("Torchrun launch requires socket permissions (nnodes>1 requested).")
            if int(nproc_per_node) != 1:
                raise RuntimeError("Torchrun launch requires socket permissions (nproc_per_node>1 requested).")
            if getattr(config, "rdzv_backend", None) or getattr(config, "rdzv_endpoint", None):
                raise RuntimeError("Torchrun rendezvous flags require socket permissions in this environment.")
            print("[harness] sockets unavailable; launching torchrun wrapper directly (single process)", flush=True)
            torchrun_cmd = [sys.executable]
        else:
            torchrun_path = shutil.which("torchrun")
            if torchrun_path:
                torchrun_cmd = [
                    torchrun_path,
                    "--nproc_per_node",
                    str(nproc_per_node),
                ]
            else:
                # Avoid relying on PATH/entrypoints; works even when the venv isn't activated.
                torchrun_cmd = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--nproc_per_node",
                    str(nproc_per_node),
                ]
        if getattr(config, "nnodes", None):
            torchrun_cmd.extend(["--nnodes", str(config.nnodes)])
        if getattr(config, "rdzv_backend", None) or getattr(config, "nnodes", None):
            torchrun_cmd.extend(["--rdzv_backend", getattr(config, "rdzv_backend", None) or "c10d"])
        if getattr(config, "rdzv_endpoint", None):
            torchrun_cmd.extend(["--rdzv_endpoint", str(config.rdzv_endpoint)])

        def _config_args_from_map() -> List[str]:
            args: List[str] = []
            for key, flag in spec.config_arg_map.items():
                if not flag or not hasattr(config, key):
                    continue
                value = getattr(config, key)
                if value is None:
                    continue
                if isinstance(value, bool):
                    if value:
                        args.append(flag)
                else:
                    args.extend([flag, str(value)])
            return args

        extra_args: List[str] = []
        target_label = getattr(config, "target_label", None)
        if target_label:
            target_overrides = _lookup_target_extra_args(
                (getattr(config, "target_extra_args", {}) or {}),
                target_label,
            )
            if target_overrides:
                if isinstance(target_overrides, str):
                    extra_args.extend(shlex.split(target_overrides))
                else:
                    extra_args.extend(list(target_overrides))

        script_args = list(spec.script_args)
        script_args.extend(_config_args_from_map())
        script_args.extend(extra_args)

        if self._seed_info is None:
            self._ensure_runtime_initialized()
        expected_torch_seed = getattr(self, "_seed_info", {}).get("torch_seed")
        if expected_torch_seed is None:
            raise RuntimeError("Missing expected torch seed for torchrun enforcement")

        wrapper_args: List[str] = [
            "--aisp-expected-torch-seed",
            str(int(expected_torch_seed)),
        ]
        if script_path is not None:
            wrapper_args[0:0] = ["--aisp-target-script", str(script_path)]
        else:
            wrapper_args[0:0] = ["--aisp-target-module", str(module_name)]
        if getattr(config, "deterministic", False):
            wrapper_args.append("--aisp-deterministic")

        if torch.cuda.is_available():
            expected_cuda_seed = getattr(self, "_seed_info", {}).get("cuda_seed")
            if expected_cuda_seed is None:
                raise RuntimeError("Missing expected CUDA seed for torchrun enforcement")
            wrapper_args.extend(["--aisp-expected-cuda-seed", str(int(expected_cuda_seed))])

        full_cmd = torchrun_cmd + ["-m", "core.harness.torchrun_wrapper"] + wrapper_args + script_args
        print(f"[harness] torchrun cmd: {' '.join(full_cmd)}", flush=True)

        repo_root = Path(__file__).resolve().parents[2]
        env = build_repo_python_env(repo_root, base_env=os.environ.copy())
        env.setdefault("AISP_BENCHMARK_OWNER_RUN_ID", os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", str(os.getpid())))
        env.setdefault("AISP_BENCHMARK_OWNER_PID", os.environ.get("AISP_BENCHMARK_OWNER_PID", str(os.getpid())))
        for key in getattr(config, "env_passthrough", []) or []:
            if key in os.environ:
                env[key] = os.environ[key]
        print(f"[harness] torchrun env passthrough: {getattr(config, 'env_passthrough', [])}", flush=True)
        env.update(spec.env)
        if getattr(config, "single_gpu", False):
            env["CUDA_VISIBLE_DEVICES"] = self._select_single_gpu_visible()
        if getattr(config, "lock_gpu_clocks", False) and torch.cuda.is_available():
            env["AISP_LOCK_GPU_CLOCKS"] = "1"
            if getattr(config, "gpu_sm_clock_mhz", None) is not None:
                env["AISP_GPU_SM_CLOCK_MHZ"] = str(config.gpu_sm_clock_mhz)
            if getattr(config, "gpu_mem_clock_mhz", None) is not None:
                env["AISP_GPU_MEM_CLOCK_MHZ"] = str(config.gpu_mem_clock_mhz)
            env["AISP_RAMP_GPU_CLOCKS"] = "1"
        if not sockets_permitted:
            env.setdefault("RANK", "0")
            env.setdefault("LOCAL_RANK", "0")
            env.setdefault("WORLD_SIZE", "1")
            env.setdefault("LOCAL_WORLD_SIZE", "1")
            env.setdefault("GROUP_RANK", "0")
            env.setdefault("ROLE_RANK", "0")
            env.setdefault("ROLE_NAME", "default")
            env.setdefault("MASTER_ADDR", "127.0.0.1")
            env.setdefault("MASTER_PORT", "29500")

        stdout = ""
        stderr = ""
        elapsed = 0.0
        timeout_limit = config.get_effective_timeout("measurement")
        if timeout_limit is None:
            raise ValueError(
                "Torchrun execution requires a finite timeout; set BenchmarkConfig.measurement_timeout_seconds."
            )
        if not isinstance(timeout_limit, (int, float)) or timeout_limit <= 0:
            raise ValueError(f"Invalid torchrun measurement timeout: {timeout_limit!r}")
        timeout_limit = float(timeout_limit)

        try:
            process = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=_benchmark_child_preexec,
                env=env,
            )
            start = time.time()
            try:
                stdout, stderr = process.communicate(timeout=timeout_limit)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                elapsed = time.time() - start
                return self._create_timeout_result(
                    stage="measurement",
                    duration=elapsed,
                    limit=timeout_limit,
                    errors=["Torchrun launch exceeded timeout", *errors],
                    benchmark_name=spec.name or getattr(benchmark, "name", None) or benchmark.__class__.__name__,
                    config=config,
                )
            elapsed = time.time() - start
            if process.returncode != 0:
                stdout_lines = stdout.splitlines() if stdout else []
                stderr_lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
                skip_reason = _extract_skip_reason_from_messages([*stdout_lines, *stderr_lines])
                if skip_reason:
                    errors.append(f"SKIPPED: {skip_reason}")
                else:
                    errors.append(f"torchrun exited with code {process.returncode}")
                if stderr:
                    _maybe_write_subprocess_stderr(stderr, f"torchrun_{spec.name or benchmark.__class__.__name__}", config)
                if stderr_lines and not skip_reason:
                    # Prefer showing the root-cause message (e.g., seed mutation) instead of only the
                    # torchrun elastic summary tail.
                    interesting = [
                        line
                        for line in stderr_lines
                        if "seed mutation detected" in line.lower()
                        or "cuda seed mutation detected" in line.lower()
                    ]
                    if interesting:
                        # Add the first few matching lines (deduped) for clarity.
                        seen: set[str] = set()
                        for line in interesting:
                            key = line.lower()
                            if key in seen:
                                continue
                            seen.add(key)
                            errors.append(line)
                            if len(seen) >= 3:
                                break

                    head = stderr_lines[:25]
                    if head:
                        errors.append("stderr_head: " + " | ".join(head))

                    tail = stderr_lines[-8:]
                    if tail:
                        errors.append("stderr_tail: " + " | ".join(tail))
        except FileNotFoundError:
            raise RuntimeError("SKIPPED: torchrun not found in PATH.")

        stdout_lines = stdout.splitlines() if stdout else []
        filtered_lines = _filter_logs(stdout_lines, dedupe=spec.parse_rank0_only)
        tokens_per_s = _extract_tokens_per_s(filtered_lines)

        iterations = getattr(config, "iterations", None) or 1
        per_iter_ms = (elapsed * 1000.0) / max(iterations, 1)
        times_ms = [per_iter_ms] * max(iterations, 1)
        result = self._compute_stats(times_ms, config)

        if tokens_per_s is not None:
            result.throughput = ThroughputStats(
                tokens_per_s=tokens_per_s,
                requests_per_s=None,
                samples_per_s=None,
                bytes_per_s=None,
                custom_unit_per_s=None,
                custom_unit_name=None,
                latency_ms=result.timing.mean_ms,
                goodput=None,
                schemaVersion="1.0",
            )
            result.custom_metrics["tokens_per_s"] = tokens_per_s

        if filtered_lines:
            tail = "\n".join(filtered_lines[-5:])
            result.validation_message = tail

        if errors:
            result.errors.extend(errors)

        self._annotate_launch_metadata(
            result,
            config,
            world_size=world_size_hint,
            multi_gpu_required=multi_gpu_required,
        )
        return result
    
    def _ensure_thread_executor(self) -> None:
        """Initialize thread executor for threaded execution mode."""
        if self._thread_executor is None:
            self._thread_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="benchmark-thread",
            )
    
    def _reset_thread_executor(self) -> None:
        """Shutdown and reset thread executor."""
        if self._thread_executor is not None:
            self._thread_executor.shutdown(wait=False, cancel_futures=True)
            self._thread_executor = None
    
    @contextmanager
    def _memory_tracking(self, config: Optional[BenchmarkConfig] = None):
        """Context manager for memory tracking.
        
        Yields a MemoryStats pydantic model (or None if tracking disabled).
        
        Args:
            config: BenchmarkConfig to check for enable_memory_tracking. If None, uses self.config.
        
        Yields:
            MemoryStats pydantic model if tracking enabled, None otherwise.
        """
        # Use provided config or fall back to instance config
        check_config = config if config is not None else self.config
        
        if not check_config.enable_memory_tracking or not torch.cuda.is_available():
            yield None
            return
        
        # Create MemoryStats model
        result = MemoryStats(peak_mb=None, allocated_mb=None, reserved_mb=None, schemaVersion="1.0")
        
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        yield result
        torch.cuda.synchronize(self.device)
        
        # Set memory values
        result.peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        result.allocated_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
        
        # Try to get reserved memory if available
        try:
            result.reserved_mb = torch.cuda.memory_reserved(self.device) / (1024**2)
        except AttributeError:
            # memory_reserved may not be available in all PyTorch versions
            result.reserved_mb = None
    
    def benchmark(
        self,
        benchmark: Union[BaseBenchmark, Callable[[], Any]],
        name: Optional[str] = None
    ) -> PydanticBenchmarkResult:
        """Run benchmark and return statistical results.
        
        Uses subprocess isolation (if enabled) or threading timeout to prevent hangs.
        Default timeout is 15 seconds.
        """
        callable_wrapped = False
        # Support callable benchmarks by wrapping in a minimal BaseBenchmark
        if not isinstance(benchmark, BaseBenchmark):
            fn = benchmark
            callable_wrapped = True
            
            class _CallableBenchmark(BaseBenchmark):  # type: ignore[misc]
                def __init__(self, wrapped_fn: Callable[[], Any], bench_name: Optional[str]):
                    # Callable benchmarks should not force CUDA; allow CPU for lightweight examples/docs.
                    self.allow_cpu = True
                    super().__init__()
                    self._fn = wrapped_fn
                    self.name = bench_name or "callable_benchmark"
                
                def setup(self) -> None:
                    # No-op setup by default for raw callables
                    pass
                
                def benchmark_fn(self) -> None:
                    self._fn()
            
            benchmark = _CallableBenchmark(fn, name)
        elif name and getattr(benchmark, "name", None) is None:
            # Preserve provided name if the benchmark did not set one
            benchmark.name = name
        
        print("[harness] benchmark() start", flush=True)
        # Clone config to avoid mutating shared instance; deepcopy prevents
        # dataclasses.replace from re-running __post_init__ (which re-applies multipliers)
        config = copy.deepcopy(self.config)
        print(f"[harness] initial launch_via={config.launch_via} execution_mode={config.execution_mode}", flush=True)
        if callable_wrapped:
            # Callable benchmarks run in-process to avoid JSON parsing of custom stdout
            config.use_subprocess = False
            config.execution_mode = ExecutionMode.THREAD
        # Benchmarks sometimes derive get_config() from per-target CLI state (for example
        # forward vs fwd_bwd). Apply run-level target overrides before reading get_config().
        self._apply_target_overrides(benchmark, config)
        bench_config = benchmark.get_config()
        if bench_config is not None:
            # Benchmarks often return a persistent config instance. Work on a copy so
            # harness-only normalization cannot corrupt later profiler/verification
            # paths that call get_config() again on the same benchmark object.
            bench_config = copy.deepcopy(bench_config)
        if bench_config and _is_chapter_or_labs_benchmark(benchmark):
            bench_config.iterations = None  # type: ignore[assignment]
            bench_config.warmup = None  # type: ignore[assignment]
        if bench_config:
            # Override with benchmark-specific settings
            default_bench_config = BenchmarkConfig()
            for key, value in bench_config.__dict__.items():
                # Skip private/internal fields and None values.
                if key.startswith("_") or value is None:
                    continue

                # Never allow benchmarks to re-enable subprocess mode when we're already
                # running inside an isolated subprocess (prevents recursive launches).
                if key in ("use_subprocess", "execution_mode") and self.config.use_subprocess is False:
                    continue

                # Merge target-specific extra args.
                if key == "target_extra_args":
                    if value:
                        config.target_extra_args = {
                            **(getattr(config, "target_extra_args", {}) or {}),
                            **value,
                        }
                    continue

                # Preserve caller-specified launcher when benchmark config only supplies the default.
                if key == "launch_via":
                    default_launch = _get_default_value("launch_via", None)

                    def _norm(val):
                        if val is None:
                            return None
                        if hasattr(val, "value"):
                            return str(getattr(val, "value")).lower()
                        return str(val).lower()

                    if (
                        _norm(config.launch_via) is not None
                        and _norm(default_launch) is not None
                        and _norm(config.launch_via) != _norm(default_launch)
                        and _norm(value) == _norm(default_launch)
                    ):
                        continue

                if key == "env_passthrough" and not value:
                    continue

                # Benchmarks only override harness settings when the harness is still at defaults.
                if hasattr(default_bench_config, key):
                    default_val = getattr(default_bench_config, key)
                    current_val = getattr(config, key, default_val)
                    if current_val != default_val:
                        # Caller already chose a non-default value; keep it.
                        continue
                setattr(config, key, value)
        config._sync_execution_mode()
        config._sync_launch_via()
        print(f"[harness] execution_mode={config.execution_mode} launch_via={config.launch_via}", flush=True)
        
        previous_config = getattr(benchmark, "_config", None)

        # Fail-fast compliance gate for perf path
        if not callable_wrapped and isinstance(benchmark, BaseBenchmark):
            from core.benchmark.quarantine import detect_skip_flags

            skip_flag = detect_skip_flags(benchmark)
            if skip_flag:
                reason = skip_flag.value if hasattr(skip_flag, "value") else str(skip_flag)
                raise RuntimeError(f"SKIPPED: Benchmark compliance failed: {reason}")

            required_methods = (
                "get_input_signature",
                "get_verify_output",
                "get_output_tolerance",
                "get_verify_inputs",
                "validate_result",
            )
            for method in required_methods:
                meth = getattr(benchmark, method, None)
                if meth is None or not callable(meth):
                    raise RuntimeError(f"SKIPPED: Benchmark missing required method '{method}'")
        
        # CRITICAL: Ensure percentiles is always a list (never None)
        # This handles cases where benchmark.get_config() returns a config with percentiles=None
        # or where the config was created before __post_init__ could fix it
        if config.percentiles is None or not isinstance(config.percentiles, list):
            config.percentiles = [25, 50, 75, 99]

        if isinstance(benchmark, BaseBenchmark):
            self._enforce_world_size_requirement(benchmark, config)

        subprocess_coordinator = config.execution_mode == ExecutionMode.SUBPROCESS

        # Make merged config visible to benchmarks, but prevent runtime mutation.
        # Benchmarks read this via get_config() / self._config.
        benchmark._config = ReadOnlyBenchmarkConfigView.from_config(config)  # type: ignore[attr-defined]

        # Environment validity gate (loud warning on virtualization).
        # Note: validate_environment() also runs inside the timed harness path; this early check ensures
        # the message is visible even when using subprocess isolation.
        from core.harness.validity_checks import validate_environment

        env_result = validate_environment(
            device=self.device,
            probe=self._environment_probe,
            allow_virtualization=bool(getattr(config, "allow_virtualization", False)),
            allow_foreign_gpu_processes=bool(getattr(config, "allow_foreign_gpu_processes", False)),
        )
        if LOGGER_AVAILABLE:
            if env_result.details:
                try:
                    logger.info(
                        "ENVIRONMENT DETAILS: %s",
                        json.dumps(env_result.details, sort_keys=True),
                    )
                except Exception:
                    logger.info("ENVIRONMENT DETAILS: %s", env_result.details)
            for warning in env_result.warnings:
                logger.warning("ENVIRONMENT WARNING: %s", warning)
            for notice in env_result.notices:
                logger.info("ENVIRONMENT NOTICE: %s", notice)
        if env_result.errors:
            message = _format_environment_invalid_message(env_result.errors, config)
            enforce_env = bool(getattr(config, "enforce_environment_validation", True))
            defer_foreign_process_failure = (
                config.execution_mode == ExecutionMode.SUBPROCESS
                and _contains_only_foreign_gpu_process_errors(env_result.errors)
            )
            if enforce_env and not defer_foreign_process_failure:
                raise RuntimeError(message)
            if defer_foreign_process_failure and LOGGER_AVAILABLE:
                logger.warning(
                    "ENVIRONMENT PRECHECK DEFERRED: parent benchmark() observed only "
                    "foreign CUDA process errors before subprocess isolation; "
                    "strict validation will re-run inside the isolated subprocess. %s",
                    message,
                )
            elif defer_foreign_process_failure:
                warnings.warn(
                    "ENVIRONMENT PRECHECK DEFERRED: parent benchmark() observed only "
                    "foreign CUDA process errors before subprocess isolation; "
                    "strict validation will re-run inside the isolated subprocess. "
                    + message,
                    RuntimeWarning,
                )
            if LOGGER_AVAILABLE:
                logger.warning(message)
        
        gpu_mem_logger: Optional[GpuMemoryLogger] = None
        if not subprocess_coordinator:
            self._ensure_runtime_initialized()

        if (
            not subprocess_coordinator
            and should_enable_gpu_memory_logging(getattr(config, "enable_gpu_memory_logging", False))
        ):
            log_interval = resolve_gpu_log_interval(getattr(config, "gpu_memory_log_interval_seconds", 5.0))
            log_path = resolve_gpu_log_path(getattr(config, "gpu_memory_log_path", None))
            gpu_mem_logger = GpuMemoryLogger(self.device, interval=log_interval, log_path=log_path)
            if gpu_mem_logger.start():
                if LOGGER_AVAILABLE:
                    logger.info(
                        "GPU memory logging enabled: %s (interval=%.2fs)",
                        log_path,
                        log_interval,
                    )
            else:
                gpu_mem_logger = None

        backend_state = None
        if config.launch_via != LaunchVia.TORCHRUN and config.execution_mode != ExecutionMode.SUBPROCESS:
            policy_name = normalize_backend_policy(getattr(config, "backend_policy", None))
            backend_state = apply_backend_policy(policy_name, bool(config.deterministic))
            if self._seed_info is not None:
                self._seed_info["deterministic_mode"] = bool(config.deterministic)
            if config.deterministic and LOGGER_AVAILABLE:
                logger.info(
                    "Deterministic mode enabled (may impact performance by 5-20%). "
                    "This ensures bitwise reproducibility, but forces slower fallback kernels and ops without deterministic support may raise."
                )

        try:
            if config.launch_via == LaunchVia.TORCHRUN:
                print("[harness] dispatch torchrun", flush=True)
                return self._benchmark_with_torchrun(benchmark, config)
            if config.execution_mode == ExecutionMode.SUBPROCESS:
                print("[harness] dispatch subprocess", flush=True)
                return self._benchmark_with_subprocess(benchmark, config)
            print("[harness] dispatch threading (direct)", flush=True)
            return self._benchmark_with_threading(benchmark, config)
        finally:
            benchmark._config = previous_config  # type: ignore[attr-defined]
            if backend_state is not None:
                restore_backend_policy(backend_state)
            if gpu_mem_logger is not None:
                log_file = gpu_mem_logger.stop()
                if LOGGER_AVAILABLE:
                    logger.info("GPU memory log saved to %s", log_file)
    
    def benchmark_with_manifest(
        self, 
        benchmark: BaseBenchmark,
        run_id: Optional[str] = None
    ) -> BenchmarkRun:
        """Run benchmark with manifest and return BenchmarkRun.
        
        This method creates a RunManifest at the start, runs the benchmark,
        and returns a BenchmarkRun containing both the manifest and result.
        
        Args:
            benchmark: BaseBenchmark instance to run
            run_id: Optional run identifier (defaults to timestamp)
        
        Returns:
            BenchmarkRun with manifest and result
        """
        from datetime import datetime
        from core.benchmark.run_manifest import RunManifest
        from core.benchmark.run_manifest import SeedInfo

        subprocess_coordinator = self.config.execution_mode == ExecutionMode.SUBPROCESS
        if not subprocess_coordinator:
            self._ensure_runtime_initialized()

        # Create manifest at start
        start_time = datetime.now()

        def _serialize_manifest_value(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(k): _serialize_manifest_value(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [_serialize_manifest_value(v) for v in value]
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, torch.device):
                return str(value)
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            raise TypeError(
                f"Unsupported BenchmarkConfig value type for manifest serialization: {type(value)}"
            )

        config_dict = {k: _serialize_manifest_value(v) for k, v in self.config.__dict__.items()}
        manifest = RunManifest.create(config=config_dict, start_time=start_time)

        def _append_manifest_warning(message: str) -> None:
            if message not in manifest.collection_warnings:
                manifest.collection_warnings.append(message)
            if LOGGER_AVAILABLE:
                logger.warning(message)

        def _seed_info_to_manifest(seed_info: Optional[Dict[str, Any]]) -> Optional[SeedInfo]:
            if not seed_info:
                return None
            return SeedInfo(
                random_seed=seed_info.get("random_seed"),
                numpy_seed=seed_info.get("numpy_seed"),
                torch_seed=seed_info.get("torch_seed"),
                cuda_seed=seed_info.get("cuda_seed"),
                deterministic_mode=bool(seed_info.get("deterministic_mode", False)),
                schemaVersion="1.0",
            )

        manifest_seed_info = _seed_info_to_manifest(self._seed_info)
        if manifest_seed_info is not None:
            manifest.seeds = SeedInfo(
                random_seed=manifest_seed_info.random_seed,
                numpy_seed=manifest_seed_info.numpy_seed,
                torch_seed=manifest_seed_info.torch_seed,
                cuda_seed=manifest_seed_info.cuda_seed,
                deterministic_mode=manifest_seed_info.deterministic_mode,
                schemaVersion="1.0",
            )
        
        # Run benchmark
        result = self.benchmark(benchmark)

        result_seed_info = _seed_info_to_manifest(getattr(result, "seeds", None))
        if result_seed_info is not None:
            manifest.seeds = result_seed_info

        # The harness locks application clocks during timing. The manifest created above
        # captures pre-lock state, so patch it with the telemetry captured while clocks
        # were locked to keep the run manifest honest (app_clock must match telemetry/logs).
        try:
            locked = getattr(result, "gpu_metrics", None)
            hw = getattr(manifest, "hardware", None)
            if isinstance(locked, dict) and hw is not None:
                def _maybe_int(v: Any) -> Optional[int]:
                    if v is None:
                        return None
                    try:
                        return int(float(v))
                    except (TypeError, ValueError):
                        return None

                def _maybe_float(v: Any) -> Optional[float]:
                    if v is None:
                        return None
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return None

                hw.gpu_clock_mhz = _maybe_int(locked.get("graphics_clock_mhz")) or hw.gpu_clock_mhz
                hw.memory_clock_mhz = _maybe_int(locked.get("memory_clock_mhz")) or hw.memory_clock_mhz
                hw.gpu_app_clock_mhz = _maybe_int(locked.get("applications_clock_sm_mhz")) or hw.gpu_app_clock_mhz
                hw.memory_app_clock_mhz = (
                    _maybe_int(locked.get("applications_clock_memory_mhz")) or hw.memory_app_clock_mhz
                )
                hw.power_draw_w = _maybe_float(locked.get("power_draw_w")) or hw.power_draw_w
                hw.temperature_gpu_c = _maybe_float(locked.get("temperature_gpu_c")) or hw.temperature_gpu_c
                hw.temperature_memory_c = _maybe_float(locked.get("temperature_memory_c")) or hw.temperature_memory_c
                hw.fan_speed_pct = _maybe_float(locked.get("fan_speed_pct")) or hw.fan_speed_pct
                hw.utilization_gpu_pct = _maybe_float(locked.get("utilization_gpu_pct")) or hw.utilization_gpu_pct
                hw.utilization_memory_pct = (
                    _maybe_float(locked.get("utilization_memory_pct")) or hw.utilization_memory_pct
                )
        except Exception as exc:
            _append_manifest_warning(f"Failed to patch locked GPU telemetry into manifest: {exc}")

        try:
            runtime_env = getattr(result, "runtime_env", None) or {}
            env_info = getattr(manifest, "environment", None)
            if runtime_env and env_info is not None:
                for key, value in runtime_env.items():
                    env_info.relevant_env_vars[str(key)] = str(value)
        except Exception as exc:
            _append_manifest_warning(f"Failed to merge runtime environment overrides into manifest: {exc}")
        
        # Finalize manifest
        end_time = datetime.now()
        manifest.finalize(end_time=end_time)
        
        # Generate run_id if not provided
        if run_id is None:
            run_id = start_time.strftime("run_%Y%m%d_%H%M%S")
        
        # Create BenchmarkRun
        return BenchmarkRun(
            manifest=manifest,
            result=result,
            run_id=run_id,
            timestamp=start_time.isoformat(),
            schemaVersion="1.0",
        )
    
    def _validate_jitter_signature(self, benchmark: BaseBenchmark) -> None:
        """Fail fast if get_input_signature() cannot support jitter checks."""
        from core.benchmark.verification import coerce_input_signature, select_jitter_dimension

        try:
            sig_payload = benchmark.get_input_signature()
        except RuntimeError:
            # Payload-backed signatures may only be available post-run; defer validation.
            return

        signature = coerce_input_signature(sig_payload)

        # Allow skips only for sanctioned cases (multi-GPU or config-generation benchmarks).
        def _maybe_skip(reason: str) -> None:
            raise RuntimeError(f"SKIPPED: {reason}")

        # Multi-GPU benchmarks are skipped entirely when <2 GPUs.
        cfg = getattr(benchmark, "_config", None) or getattr(benchmark, "get_config", lambda: None)()
        multi_gpu_flag = False
        if cfg is not None and getattr(cfg, "multi_gpu_required", False):
            multi_gpu_flag = True
        if getattr(benchmark, "multi_gpu_required", False):
            multi_gpu_flag = True
        if multi_gpu_flag and torch.cuda.device_count() < 2:
            _maybe_skip("requires >=2 GPUs")

        # Config-generation / non-compute utilities may declare a verification_not_applicable_reason.
        not_applicable = getattr(benchmark, "verification_not_applicable_reason", None)
        if not_applicable:
            _maybe_skip(not_applicable)

        if select_jitter_dimension(signature) is None:
            raise RuntimeError(
                "Jitter check requires get_input_signature() to include at least one tensor shape with >=2 dims."
            )
    
    def verify(
        self,
        baseline: BaseBenchmark,
        optimized: BaseBenchmark,
        force_recache: bool = False,
    ) -> "VerifyResult":
        """Run verification on a baseline/optimized benchmark pair.
        
        This method verifies that baseline and optimized benchmarks:
        1. Have equivalent input signatures (same workload)
        2. Produce equivalent outputs (within tolerance)
        3. Pass anti-hacking checks (fresh-input, jitter)
        
        Verification runs with deterministic seeds (seed=42) and caches
        baseline golden outputs for subsequent comparisons.
        
        Args:
            baseline: The baseline benchmark instance
            optimized: The optimized benchmark instance  
            force_recache: If True, regenerate baseline golden output
            
        Returns:
            VerifyResult with verification outcome
            
        Example:
            harness = BenchmarkHarness()
            baseline = BaselineGemmBenchmark()
            optimized = OptimizedGemmBenchmark()
            
            result = harness.verify(baseline, optimized)
            if not result.passed:
                print(f"Verification failed: {result.reason}")
        """
        from core.benchmark.verify_runner import VerifyRunner, VerifyConfig
        from core.benchmark.verification import VerifyResult
        
        # Fail early on missing jitter metadata to avoid late surprises.
        self._validate_jitter_signature(baseline)
        self._validate_jitter_signature(optimized)

        runner = VerifyRunner()
        config = VerifyConfig(
            seed=42,
            force_recache=force_recache,
            verbose=LOGGER_AVAILABLE,
        )
        
        return runner.verify_pair(baseline, optimized, config)
    
    def gate_perf(
        self,
        benchmark_path: str,
    ) -> Tuple[bool, Optional[str]]:
        """Check if a benchmark is allowed to run performance measurement.
        
        Based on enforcement phase and quarantine status, determines
        whether perf measurement should proceed.
        
        Args:
            benchmark_path: Path to the benchmark file
            
        Returns:
            Tuple of (allowed, reason_if_blocked)
            
        Example:
            harness = BenchmarkHarness()
            allowed, reason = harness.gate_perf("ch01/baseline_gemm.py")
            if not allowed:
                print(f"Perf blocked: {reason}")
        """
        from core.benchmark.verify_runner import VerifyRunner
        
        runner = VerifyRunner()
        return runner.gate_perf(benchmark_path)
    
    def _benchmark_with_subprocess(self, benchmark: BaseBenchmark, config: BenchmarkConfig) -> PydanticBenchmarkResult:
        """Run benchmark in subprocess for reliable timeout cancellation."""
        import json
        import inspect
        
        errors: List[str] = []
        memory_peak_mb: Optional[float] = None
        memory_allocated_mb: Optional[float] = None
        profiling_outputs: Dict[str, str] = {}
        nsys_metrics: Dict[str, float] = {}
        ncu_metrics: Dict[str, float] = {}
        proton_metrics: Dict[str, float] = {}
        times_ms: List[float] = []
        inference_timing_data: Optional[Dict[str, List[float]]] = None
        seed_metadata = copy.deepcopy(getattr(self, "_seed_info", None))
        child_custom_metrics: Optional[Dict[str, float]] = None
        child_runtime_env: Optional[Dict[str, str]] = None
        child_gpu_metrics: Optional[Dict[str, Optional[float | str]]] = None
        stage_watchdog: Dict[str, Dict[str, Any]] = {
            "setup": {"status": "pending"},
            "warmup": {"status": "pending"},
            "profiling": {"status": "pending"},
            "measurement": {"status": "pending"},
        }
        
        # Get benchmark name and module info for error messages
        module_override = getattr(benchmark, "_module_file_override", None)
        class_override = getattr(benchmark, "_factory_name_override", None)
        benchmark_module = inspect.getmodule(benchmark)
        benchmark_class = class_override or benchmark.__class__.__name__
        benchmark_name = getattr(benchmark, '__name__', None) or getattr(benchmark, 'name', None) or benchmark_class
        
        if benchmark_module is None:
            benchmark_module = inspect.getmodule(benchmark.__class__)
        
        if benchmark_module is None:
            # Fallback to threading if we can't determine module
            return self._benchmark_with_threading(benchmark, config)
        
        module_file = module_override or getattr(benchmark_module, "__file__", None)
        if module_file is None:
            spec = getattr(benchmark_module, "__spec__", None)
            if spec is not None:
                module_file = getattr(spec, "origin", None)
        
        if module_file is None:
            # Fallback to threading if we can't determine module file
            return self._benchmark_with_threading(benchmark, config)
        
        module_path = Path(module_file).resolve()
        if not module_path.exists():
            # Fallback to threading if module file doesn't exist
            return self._benchmark_with_threading(benchmark, config)
        
        def _is_simple(value: Any) -> bool:
            """Return True for JSON-serializable scalar/collection values."""
            if isinstance(value, (str, int, float, bool)) or value is None:
                return True
            if isinstance(value, (list, tuple)):
                return all(_is_simple(v) for v in value)
            if isinstance(value, dict):
                return all(
                    isinstance(k, (str, int, float, bool)) and _is_simple(v)
                    for k, v in value.items()
                )
            return False

        # Prepare config dict (serialize all simple public fields).
        # This must mirror in-process config exactly so protections match across modes.
        config_dict: Dict[str, Any] = {}
        for key, value in config.__dict__.items():
            if str(key).startswith("_") or key == "device":
                continue
            if value is None:
                continue
            if key == "percentiles":
                if not isinstance(value, list):
                    value = [25, 50, 75, 99]
                config_dict[key] = value
                continue
            if key == "execution_mode":
                config_dict[key] = value.value if isinstance(value, ExecutionMode) else value
                continue
            if key == "launch_via":
                config_dict[key] = value.value if isinstance(value, LaunchVia) else value
                continue
            if _is_simple(value):
                config_dict[key] = value

        # CRITICAL: Ensure percentiles is always in config_dict
        if 'percentiles' not in config_dict:
            config_dict['percentiles'] = [25, 50, 75, 99]

        initial_state = {
            key: value
            for key, value in getattr(benchmark, "__dict__", {}).items()
            if not str(key).startswith("_") and _is_simple(value)
        }

        # Prepare input JSON
        input_data = {
            "benchmark_module_path": str(module_path),
            "benchmark_class_name": benchmark_class,
            "config_dict": config_dict,
            "mode": self.mode.value,
            "device": str(self.device) if self.device else None,
            "initial_state": initial_state or None,
        }
        try:
            fd, verify_output_path = tempfile.mkstemp(prefix="aisp_verify_output_", suffix=".pt")
            os.close(fd)
        except Exception as e:
            raise RuntimeError(f"Failed to allocate verify_output file for subprocess: {e}") from e
        input_data["verify_output_path"] = verify_output_path
        input_data["verify_output_max_bytes"] = _VERIFY_OUTPUT_MAX_BYTES
        
        # Spawn subprocess using isolated runner
        # Use measurement_timeout_seconds (or fallback to timeout_seconds for backward compatibility).
        # Fail fast if no timeout is configured to avoid indefinite hangs.
        measurement_timeout = getattr(config, "measurement_timeout_seconds", None)
        if measurement_timeout is None:
            measurement_timeout = getattr(config, "timeout_seconds", None)
        if measurement_timeout is None:
            raise ValueError(
                "Subprocess execution requires a finite timeout; set BenchmarkConfig.measurement_timeout_seconds "
                "(or legacy timeout_seconds)."
            )
        if not isinstance(measurement_timeout, (int, float)) or measurement_timeout <= 0:
            raise ValueError(f"Invalid measurement timeout: {measurement_timeout!r}")
        measurement_timeout = float(measurement_timeout)
        start_time = time.time()
        
        try:
            import signal
            repo_root = Path(__file__).resolve().parents[2]
            env = build_repo_python_env(repo_root, base_env=os.environ.copy())
            owner_run_id = os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", str(os.getpid()))
            owner_pid = os.environ.get("AISP_BENCHMARK_OWNER_PID", str(os.getpid()))
            env.setdefault("AISP_BENCHMARK_OWNER_RUN_ID", owner_run_id)
            env.setdefault("AISP_BENCHMARK_OWNER_PID", owner_pid)
            if getattr(config, "single_gpu", False):
                env["CUDA_VISIBLE_DEVICES"] = self._select_single_gpu_visible()
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "core.harness.isolated_runner",
                    "--aisp-owner-run-id",
                    owner_run_id,
                    "--aisp-owner-pid",
                    owner_pid,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                preexec_fn=_benchmark_child_preexec,
            )
            child_pgid = os.getpgid(process.pid)
            if LOGGER_AVAILABLE:
                logger.info(
                    "SUBPROCESS DISPATCH: coordinator_pid=%s worker_pid=%s owner_run_id=%s owner_pid=%s",
                    os.getpid(),
                    process.pid,
                    owner_run_id,
                    owner_pid,
                )
            
            # Send input JSON and wait for result
            input_json = json.dumps(input_data)
            subprocess_failed = False
            try:
                stdout, stderr = process.communicate(input=input_json, timeout=measurement_timeout)
            except BrokenPipeError as exc:
                subprocess_failed = True
                errors.append(f"Subprocess stdin closed early: {exc}")
                try:
                    if process.stdin:
                        process.stdin.close()
                except Exception as close_exc:
                    errors.append(f"Failed to close subprocess stdin after broken pipe: {close_exc}")
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except Exception as follow_exc:
                    errors.append(f"Failed to read subprocess output after broken pipe: {follow_exc}")
                    stdout = stdout or ""
                    stderr = stderr or ""
                if stderr:
                    _maybe_write_subprocess_stderr(stderr, benchmark_name, config)
                _cleanup_process_group(child_pgid)
            else:
                _cleanup_process_group(child_pgid)

            if subprocess_failed or process.returncode != 0:
                error_msg = f"Subprocess exited with code {process.returncode}"
                errors.append(error_msg)
                if stderr:
                    stderr_preview = stderr[:500] if len(stderr) > 500 else stderr
                    errors.append(f"Stderr: {stderr_preview}")
                    if LOGGER_AVAILABLE:
                        logger.error(f"Benchmark '{benchmark_name}' subprocess failed with return code {process.returncode}")
                        logger.error(f"Stderr preview: {stderr_preview}")
                    _maybe_write_subprocess_stderr(stderr, benchmark_name, config)
                times_ms = cast(List[float], [])
            else:
                # Parse JSON result from isolated runner
                try:
                    # Strip any non-JSON prefix (e.g., compilation messages from CUDA extensions)
                    # that may be printed before the JSON output. Prefer the final JSON payload
                    # emitted by isolated_runner (starts with {"success": ...}) because other
                    # libraries may print Python dicts containing '{' on stdout.
                    json_start = stdout.rfind('{"success"')
                    if json_start < 0:
                        json_start = stdout.find('{')
                    if json_start > 0:
                        # There's content before JSON - store it for debugging but parse only JSON
                        prefix = stdout[:json_start].strip()
                        if prefix:
                            # Log the prefix for debugging but don't treat as error
                            if LOGGER_AVAILABLE:
                                logger.debug(f"Stripped non-JSON prefix from subprocess output: {prefix[:200]}")
                        stdout = stdout[json_start:]
                    
                    # Parse the first JSON object and ignore any trailing noise.
                    # Prefer a real JSON parser over brace counting to avoid false
                    # truncation when strings contain braces/escapes.
                    decoder = json.JSONDecoder()
                    result_dict, json_end = decoder.raw_decode(stdout)
                    suffix = stdout[json_end:].strip()
                    if suffix and LOGGER_AVAILABLE:
                        logger.debug(f"Stripped non-JSON suffix from subprocess output: {suffix[:200]}")
                    result_success = bool(result_dict.get("success"))
                    if result_success and result_dict.get("result_json"):
                        # Deserialize Pydantic BenchmarkResult from JSON
                        result_json_str = result_dict["result_json"]
                        benchmark_result = PydanticBenchmarkResult.model_validate_json(result_json_str)
                        child_gpu_metrics = getattr(benchmark_result, "gpu_metrics", None)
                        
                        # Extract timing data
                        if benchmark_result.timing.raw_times_ms:
                            times_ms = benchmark_result.timing.raw_times_ms
                        else:
                            # Reconstruct from statistics if raw times not available
                            mean_time = benchmark_result.timing.mean_ms
                            std_time = benchmark_result.timing.std_ms
                            iterations = benchmark_result.timing.iterations
                            import numpy as np
                            synthetic_times = np.random.normal(mean_time, std_time, iterations)
                            synthetic_times = np.clip(
                                synthetic_times,
                                benchmark_result.timing.min_ms,
                                benchmark_result.timing.max_ms
                            )
                            times_ms = synthetic_times.tolist()
                        
                        # Extract memory data
                        if benchmark_result.memory:
                            memory_peak_mb = benchmark_result.memory.peak_mb
                            memory_allocated_mb = benchmark_result.memory.allocated_mb
                        
                        # Extract errors
                        errors.extend(benchmark_result.errors)
                        
                        # Extract seeds (propagate actual subprocess seed info)
                        if getattr(benchmark_result, "seeds", None) is not None:
                            seed_metadata = benchmark_result.seeds
                        
                        # Extract profiling artifacts
                        if benchmark_result.artifacts:
                            if benchmark_result.artifacts.nsys_rep:
                                profiling_outputs['nsys'] = benchmark_result.artifacts.nsys_rep
                            if benchmark_result.artifacts.ncu_rep:
                                profiling_outputs['ncu'] = benchmark_result.artifacts.ncu_rep
                            if benchmark_result.artifacts.torch_trace_json:
                                profiling_outputs['torch'] = benchmark_result.artifacts.torch_trace_json
                        
                        # Extract profiler metrics
                        if benchmark_result.profiler_metrics:
                            if benchmark_result.profiler_metrics.nsys:
                                nsys_metrics = benchmark_result.profiler_metrics.nsys.to_dict()
                            if benchmark_result.profiler_metrics.ncu:
                                ncu_metrics = benchmark_result.profiler_metrics.ncu.to_dict()
                            if benchmark_result.profiler_metrics.proton:
                                proton_metrics = benchmark_result.profiler_metrics.proton.to_dict()
                        child_custom_metrics = dict(benchmark_result.custom_metrics or {})
                        child_runtime_env = {
                            str(key): str(value)
                            for key, value in dict(benchmark_result.runtime_env or {}).items()
                        }
                        
                        # Extract verify_output/tolerance/signature from subprocess and store on benchmark
                        verify_output_data = result_dict.get("verify_output")
                        if verify_output_data is not None:
                            try:
                                import torch

                                kind = verify_output_data.get("kind") or "tensor"
                                if kind == "tensor":
                                    benchmark._subprocess_verify_output = deserialize_verify_tensor(verify_output_data)
                                elif kind == "tensor_file":
                                    path = verify_output_data.get("path")
                                    if not path:
                                        raise ValueError("verify_output kind='tensor_file' missing path")
                                    loaded = torch.load(path, map_location="cpu")
                                    if not isinstance(loaded, torch.Tensor):
                                        raise TypeError(
                                            f"verify_output tensor_file expected torch.Tensor, got {type(loaded)}"
                                        )
                                    benchmark._subprocess_verify_output = loaded
                                elif kind == "dict":
                                    tensors_obj = verify_output_data.get("tensors")
                                    if not isinstance(tensors_obj, dict) or not tensors_obj:
                                        raise ValueError("verify_output kind='dict' missing tensors")
                                    benchmark._subprocess_verify_output = {
                                        name: deserialize_verify_tensor(tensor_dict)
                                        for name, tensor_dict in tensors_obj.items()
                                    }
                                elif kind == "dict_file":
                                    path = verify_output_data.get("path")
                                    if not path:
                                        raise ValueError("verify_output kind='dict_file' missing path")
                                    loaded = torch.load(path, map_location="cpu")
                                    if not isinstance(loaded, dict):
                                        raise TypeError(
                                            f"verify_output dict_file expected dict, got {type(loaded)}"
                                        )
                                    benchmark._subprocess_verify_output = loaded
                                else:
                                    raise ValueError(f"Unknown verify_output kind '{kind}'")
                            except Exception as e:
                                errors.append(f"Failed to reconstruct verify_output from subprocess: {e}")
                                raise

                        tol_data = result_dict.get("output_tolerance")
                        if tol_data is not None:
                            try:
                                rtol = float(tol_data.get("rtol"))
                                atol = float(tol_data.get("atol"))
                                benchmark._subprocess_output_tolerance = (rtol, atol)
                            except Exception as e:
                                errors.append(f"Failed to parse output_tolerance from subprocess: {e}")
                                raise

                        sig_data = result_dict.get("input_signature")
                        if sig_data is not None:
                            benchmark._subprocess_input_signature = sig_data
                        if stderr:
                            _maybe_write_subprocess_stderr(stderr, benchmark_name, config)
                    else:
                        errors.extend(result_dict.get("errors", ["Subprocess execution failed"]))
                        times_ms = cast(List[float], [])
                        if stderr:
                            _maybe_write_subprocess_stderr(stderr, benchmark_name, config)
                except json.JSONDecodeError as e:
                    errors.append(f"Failed to parse subprocess output: {e}")
                    errors.append(f"Output: {stdout[:500]}")
                    times_ms = cast(List[float], [])
                except Exception as e:
                    errors.append(f"Error processing subprocess result: {e}")
                    errors.append(f"Traceback: {traceback.format_exc()}")
                    times_ms = cast(List[float], [])
        
        except subprocess.TimeoutExpired:
            # TIMEOUT - kill the process group
            elapsed_time = time.time() - start_time
            
            logger.error("=" * 80)
            logger.error("TIMEOUT: Benchmark execution exceeded timeout limit")
            logger.error("=" * 80)
            logger.error(f"   Benchmark: {benchmark_name}")
            logger.error(f"   Stage: measurement (benchmark iterations)")
            logger.error(f"   Timeout limit: {measurement_timeout} seconds")
            logger.error(f"   Elapsed time: {elapsed_time:.2f} seconds")
            logger.error(f"   Config: iterations={config.iterations}, warmup={config.warmup}")
            logger.error(f"   Status: Benchmark subprocess did not complete within timeout period")
            logger.error("")
            logger.error("   Possible causes:")
            logger.error("   - Benchmark is too slow for current timeout")
            logger.error("   - Benchmark is hanging or deadlocked")
            logger.error("   - GPU is under heavy load from other processes")
            logger.error("")
            logger.error("   Suggested actions:")
            logger.error(f"   - Increase timeout: config.measurement_timeout_seconds = {measurement_timeout * 2}")
            logger.error(f"   - Use timeout multiplier: config.timeout_multiplier = 2.0")
            logger.error("   - Check for GPU resource contention")
            logger.error("   - Review benchmark code for potential deadlocks")
            logger.error("")
            logger.error("   Action: Terminating subprocess to free GPU resources")
            logger.error("=" * 80)
            
            try:
                # Kill the process group (only the child, not parent)
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                # Wait a bit for graceful termination
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # Force kill if still running
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
            except (ProcessLookupError, OSError):
                # Process already terminated
                pass
            
            timeout_error_msg = (
                f"TIMEOUT: Benchmark measurement stage exceeded timeout of {measurement_timeout} seconds "
                f"(ran for {elapsed_time:.2f}s). "
                f"Consider increasing measurement_timeout_seconds or using timeout_multiplier."
            )
            errors.append(timeout_error_msg)
            times_ms = cast(List[float], [])
            
            # Create a result with timeout information even though we failed
            if not times_ms:
                return self._create_timeout_result(
                    stage="measurement",
                    duration=elapsed_time,
                    limit=measurement_timeout,
                    errors=errors,
                    benchmark_name=benchmark_name,
                    config=config,
                )
        
        except Exception as e:
            error_msg = f"Subprocess execution failed: {str(e)}"
            errors.append(error_msg)
            if LOGGER_AVAILABLE:
                logger.error(f"Benchmark '{benchmark_name}' failed during subprocess execution")
                logger.error(f"Error: {error_msg}")
            times_ms = cast(List[float], [])
        
        # Don't raise if we already returned a timeout result
        if not times_ms and 'timeout_result' not in locals():
            # Build comprehensive error message with context
            config_summary = (
                f"iterations={config.iterations}, warmup={config.warmup}, "
                f"timeout={getattr(config, 'measurement_timeout_seconds', config.timeout_seconds)}s"
            )
            error_details = [
                f"Benchmark: {benchmark_name}",
                f"Config: {config_summary}",
                f"Errors: {', '.join(errors)}",
            ]
            # Include partial results if we have any
            if memory_peak_mb is not None:
                error_details.append(f"Partial memory stats: peak={memory_peak_mb:.2f}MB")
            if profiling_outputs:
                error_details.append(f"Profiling artifacts collected: {list(profiling_outputs.keys())}")

            skip_reason = _extract_skip_reason_from_messages(errors)
            if skip_reason:
                skip_message = skip_reason if skip_reason.upper().startswith("SKIPPED") else f"SKIPPED: {skip_reason}"
                if LOGGER_AVAILABLE:
                    logger.warning("Benchmark '%s' skipped: %s", benchmark_name, skip_message)
                raise RuntimeError(skip_message)

            error_message = "\n".join(error_details)
            if LOGGER_AVAILABLE:
                logger.error("=" * 80)
                logger.error("BENCHMARK FAILED")
                logger.error("=" * 80)
                for detail in error_details:
                    logger.error(f"  {detail}")
                logger.error("=" * 80)

            # Return a structured failure result instead of raising
            stage_watchdog["measurement"] = {"status": "error", "duration": 0.0}
            failure_stage = next(
                (stage for stage, meta in stage_watchdog.items() if meta.get('status') in ('error', 'timeout')),
                "measurement",
            )
            failure_meta = stage_watchdog.get(failure_stage, {})
            duration = failure_meta.get("duration", 0.0) or 0.0
            limit_lookup = {
                "setup": config.get_effective_timeout("setup"),
                "warmup": config.get_effective_timeout("warmup"),
                "profiling": config.get_effective_timeout("profiling"),
                "measurement": getattr(config, 'measurement_timeout_seconds', config.timeout_seconds),
            }
            limit = limit_lookup.get(failure_stage, getattr(config, 'measurement_timeout_seconds', config.timeout_seconds))
            return self._create_timeout_result(
                stage=failure_stage,
                duration=duration,
                limit=limit,
                errors=errors,
                benchmark_name=benchmark_name,
                config=config,
                watchdog=stage_watchdog,
            )
        
        # Compute statistics
        result = self._compute_stats(times_ms, config)
        if child_gpu_metrics is not None:
            result.gpu_metrics = child_gpu_metrics
        if child_custom_metrics is not None:
            result.custom_metrics = child_custom_metrics
        else:
            custom_metrics = self._resolve_custom_metrics(benchmark)
            if custom_metrics:
                result.custom_metrics = custom_metrics
        story_metadata = self._resolve_story_metadata(benchmark)
        if story_metadata:
            result.story_metadata = story_metadata
        if child_runtime_env is not None:
            result.runtime_env = child_runtime_env
        self._attach_throughput_metrics(result, benchmark)
        
        # Add inference timing if available
        if inference_timing_data and inference_timing_data.get("ttft_times_ms") and inference_timing_data.get("tpot_times_ms"):
            inference_timing = self._capture_inference_timing(
                inference_timing_data["ttft_times_ms"],
                inference_timing_data["tpot_times_ms"],
                config
            )
            if inference_timing:
                result.inference_timing = inference_timing
        
        # Add memory stats
        if memory_peak_mb is not None or memory_allocated_mb is not None:
            result.memory = MemoryStats(
                peak_mb=memory_peak_mb,
                allocated_mb=memory_allocated_mb,
                reserved_mb=None,
                schemaVersion="1.0",
            )
        
        # Add errors
        result.errors = errors
        
        # Add profiling artifacts
        if profiling_outputs:
            result.artifacts = ProfilerArtifacts(
                nsys_rep=profiling_outputs.get('nsys_rep') or profiling_outputs.get('nsys'),
                ncu_rep=profiling_outputs.get('ncu_rep') or profiling_outputs.get('ncu'),
                torch_trace_json=profiling_outputs.get('torch') or profiling_outputs.get('pytorch_trace'),
                proton_report=profiling_outputs.get('proton_report') or profiling_outputs.get('proton'),
                schemaVersion="1.0",
            )
        
        # Add profiler metrics
        nsys_metrics_obj = None
        ncu_metrics_obj = None
        proton_metrics_obj = None
        
        if nsys_metrics:
            nsys_metrics_obj = NsysMetrics(
                total_gpu_time_ms=nsys_metrics.get('nsys_total_gpu_time_ms'),
                raw_metrics={k.replace('nsys_', ''): v for k, v in nsys_metrics.items() 
                            if not k.startswith('nsys_total_gpu_time_ms')},
                schemaVersion="1.0",
            )
        
        if ncu_metrics:
            ncu_metrics_obj = NcuMetrics(
                kernel_time_ms=ncu_metrics.get('ncu_kernel_time_ms'),
                sm_throughput_pct=ncu_metrics.get('ncu_sm_throughput_pct'),
                dram_throughput_pct=ncu_metrics.get('ncu_dram_throughput_pct'),
                l2_throughput_pct=ncu_metrics.get('ncu_l2_throughput_pct'),
                occupancy_pct=ncu_metrics.get('ncu_occupancy_pct'),
                raw_metrics={k.replace('ncu_', ''): v for k, v in ncu_metrics.items() 
                            if not k.startswith(('ncu_kernel_time_ms', 'ncu_sm_throughput_pct', 
                                                'ncu_dram_throughput_pct', 'ncu_l2_throughput_pct', 'ncu_occupancy_pct'))},
                schemaVersion="1.0",
            )
        
        if proton_metrics:
            occ_field = proton_metrics.get('proton_occupancy_limited_kernels_list', proton_metrics.get('proton_occupancy_limited_kernels', ""))
            if isinstance(occ_field, list):
                occupancy = [str(item) for item in occ_field if item]
            else:
                occupancy = [item for item in occ_field.split(",") if item] if isinstance(occ_field, str) else []
            summary_stats: Dict[str, Any] = {}
            summary_from_report = proton_metrics.get('proton_summary_stats')
            if isinstance(summary_from_report, dict):
                summary_stats.update(summary_from_report)
            summary_stats.update({
                k.replace('proton_', ''): v for k, v in proton_metrics.items()
                if k.startswith('proton_') and k not in {'proton_kernel_count', 'proton_occupancy_limited_kernels', 'proton_top_kernels', 'proton_summary_stats', 'proton_kernel_summaries', 'proton_occupancy_limited_kernels_list'}
            })
            kernel_summaries = proton_metrics.get('proton_kernel_summaries')
            if not isinstance(kernel_summaries, list):
                kernel_summaries = []
            proton_metrics_obj = ProtonMetrics(
                kernel_count=proton_metrics.get('proton_kernel_count'),
                occupancy_limited_kernels=occupancy,
                summary_stats=summary_stats,
                kernel_summaries=kernel_summaries,
                schemaVersion="1.0",
            )
        
        if nsys_metrics_obj or ncu_metrics_obj or proton_metrics_obj:
            result.profiler_metrics = ProfilerMetrics(
                nsys=nsys_metrics_obj,
                ncu=ncu_metrics_obj,
                proton=proton_metrics_obj,
                torch=None,
                schemaVersion="1.0",
            )
        
        self._annotate_launch_metadata(result, config)
        return result
    
    def _benchmark_with_threading(self, benchmark: BaseBenchmark, config: BenchmarkConfig) -> PydanticBenchmarkResult:
        """Run benchmark using threading (alternative to subprocess method)."""
        self._ensure_runtime_initialized()
        import inspect
        
        errors = []
        memory_peak_mb = None
        memory_allocated_mb = None
        profiling_outputs = {}
        nsys_metrics = {}
        ncu_metrics = {}
        proton_metrics = {}
        times_ms = cast(List[float], [])
        inference_timing_data = None
        seed_metadata = copy.deepcopy(getattr(self, "_seed_info", None))
        locked_gpu_metrics: Optional[Dict[str, Optional[float | str]]] = None
        captured_custom_metrics: Optional[Dict[str, float]] = None
        
        # Get benchmark name for error messages (same as subprocess path)
        benchmark_class = benchmark.__class__.__name__
        benchmark_name = getattr(benchmark, '__name__', None) or getattr(benchmark, 'name', None) or benchmark_class

        # Environment validity gate for direct thread invocations (tests call this directly).
        from core.harness.validity_checks import validate_environment

        env_result = validate_environment(
            device=self.device,
            probe=self._environment_probe,
            allow_virtualization=bool(getattr(config, "allow_virtualization", False)),
            allow_foreign_gpu_processes=bool(getattr(config, "allow_foreign_gpu_processes", False)),
        )
        if LOGGER_AVAILABLE:
            for warning in env_result.warnings:
                logger.warning("ENVIRONMENT WARNING: %s", warning)
            for notice in env_result.notices:
                logger.info("ENVIRONMENT NOTICE: %s", notice)
        if env_result.errors:
            message = _format_environment_invalid_message(env_result.errors, config)
            enforce_env = bool(getattr(config, "enforce_environment_validation", True))
            if enforce_env:
                errors.append(message)
                timing = TimingStats(
                    mean_ms=0.0,
                    median_ms=0.0,
                    std_ms=0.0,
                    min_ms=0.0,
                    max_ms=0.0,
                    p50_ms=None,
                    p90_ms=None,
                    p95_ms=None,
                    p99_ms=None,
                    percentiles={},
                    iterations=0,
                    warmup_iterations=config.warmup,
                    raw_times_ms=[],
                    schemaVersion="1.0",
                )
                result = PydanticBenchmarkResult(
                    timing=timing,
                    errors=errors,
                    inference_timing=None,
                    memory=None,
                    artifacts=None,
                    profiler_metrics=None,
                    validation_status=None,
                    validation_message=None,
                    benchmark_name=benchmark_name,
                    device=str(self.device),
                    mode=self.mode.value,
                    timeout_stage=None,
                    timeout_duration_seconds=None,
                    timeout_limit_seconds=None,
                    seeds=copy.deepcopy(getattr(self, "_seed_info", None)),
                    schemaVersion="1.0",
                )
                self._annotate_launch_metadata(result, config)
                return result
            if LOGGER_AVAILABLE:
                logger.warning(message)
        
        # Use a lock to prevent teardown from running while benchmark is executing
        execution_lock = threading.Lock()
        execution_complete = threading.Event()
        teardown_called = threading.Event()  # Track if teardown has been called
        
        # Store timeout result if one occurs (needs to be accessible from outer scope)
        timeout_result_storage: List[Optional[PydanticBenchmarkResult]] = [None]
        stage_watchdog: Dict[str, Dict[str, Any]] = {
            'setup': {'status': 'pending', 'duration': 0.0},
            'warmup': {'status': 'pending', 'duration': 0.0},
            'measurement': {'status': 'pending', 'duration': 0.0},
            'profiling': {'status': 'pending', 'duration': 0.0},
        }
        stage_start_times: Dict[str, float] = {}

        def start_stage(stage: str) -> None:
            stage_start_times[stage] = time.time()
            stage_watchdog[stage] = {'status': 'running'}

        def finish_stage(stage: str, status: str = "completed") -> None:
            start = stage_start_times.get(stage)
            duration = time.time() - start if start else 0.0
            stage_watchdog[stage] = {'status': status, 'duration': duration}

        def mark_stage(stage: str, status: str) -> None:
            entry = stage_watchdog.get(stage, {})
            entry['status'] = status
            stage_watchdog[stage] = entry
        
        def run_benchmark_internal():
            """Internal benchmark execution function."""
            nonlocal times_ms, memory_peak_mb, memory_allocated_mb, profiling_outputs, errors, nsys_metrics, ncu_metrics, timeout_result_storage, inference_timing_data, locked_gpu_metrics, captured_custom_metrics
            validity_profile = str(getattr(config, "validity_profile", "strict")).strip().lower()
            portable_mode = validity_profile == "portable"
            clock_lock_active = False
            
            with execution_lock:  # Acquire lock during execution
                lock_stack = ExitStack()
                try:
                    if getattr(config, "lock_gpu_clocks", False) and self.device.type == "cuda":
                        device_index = self.device.index if self.device.index is not None else 0
                        try:
                            lock_stack.enter_context(
                                lock_gpu_clocks(
                                    device=device_index,
                                    sm_clock_mhz=getattr(config, "gpu_sm_clock_mhz", None),
                                    mem_clock_mhz=getattr(config, "gpu_mem_clock_mhz", None),
                                )
                            )
                            clock_lock_active = True
                        except Exception:
                            if not portable_mode:
                                raise
                            if LOGGER_AVAILABLE:
                                logger.warning(
                                    "Portable validity profile: clock lock unavailable; continuing without locked clocks.",
                                    exc_info=True,
                                )

                    def _init_cuda_for_worker_thread() -> None:
                        if not torch.cuda.is_available():
                            return
                        target_device = self.device if isinstance(self.device, torch.device) else None
                        if target_device is None or target_device.type != "cuda":
                            target_device = torch.device("cuda")
                        device_index = target_device.index if target_device.index is not None else 0
                        torch.cuda.init()
                        torch.cuda.set_device(device_index)
                        torch.cuda.synchronize(device_index)

                    # Ensure the CUDA primary context exists before any kernels run.
                    if torch.cuda.is_available():
                        torch.cuda.init()
                        try:
                            device_index = None
                            if isinstance(self.device, torch.device) and self.device.type == "cuda":
                                device_index = self.device.index
                            if device_index is None:
                                device_index = torch.cuda.current_device()
                            torch.cuda.set_device(device_index)
                            target_device = torch.device("cuda", device_index)
                            # Tiny ops to bind the primary context and initialize cuBLAS handles
                            torch.empty(1, device=target_device).add_(1)
                            torch.ones((1, 1), device=target_device).matmul(
                                torch.ones((1, 1), device=target_device)
                            )
                            torch.cuda.synchronize(device_index)
                        except Exception as exc:
                            # Best-effort context warm-up; continue even if it fails, but do not swallow silently.
                            if LOGGER_AVAILABLE:
                                logger.warning("CUDA context warm-up failed (continuing): %s", exc)

                    # Apply per-target CLI overrides (e.g., backend selection) before setup.
                    self._apply_target_overrides(benchmark, config)

                    # Clear torch.compile caches BEFORE setup/warmup so compilation
                    # overhead cannot leak into measured iterations.
                    if getattr(config, "clear_compile_cache", False):
                        from core.harness.validity_checks import clear_compile_cache as _clear_compile_cache

                        _clear_compile_cache()

                    # Setup - this may include CUDA extension compilation OR torch.compile()
                    # Setup can hang. In thread-mode we cannot safely preempt a hung setup() without risking
                    # teardown races, so executor-level measurement timeouts are the only hard stop; the
                    # setup_timeout_seconds gate is enforced post-hoc once setup returns.
                    import time
                    setup_start_time = time.time()
                    setup_timeout = config.get_effective_timeout('setup')
                    from core.harness.validity_checks import check_setup_precomputation
                    def _collect_outputs_for_hash():
                        outputs = {}
                        if hasattr(benchmark, "output"):
                            outputs["output"] = getattr(benchmark, "output")
                        return outputs

                    def _run_setup_with_detection():
                        start_stage('setup')
                        try:
                            if getattr(config, "detect_setup_precomputation", True):
                                precompute_ok, precompute_err = check_setup_precomputation(
                                    _collect_outputs_for_hash,
                                    benchmark.setup,
                                )
                                if not precompute_ok:
                                    raise RuntimeError(precompute_err or "Setup pre-computation detected")
                            else:
                                benchmark.setup()
                        except Exception:
                            finish_stage('setup', status='error')
                            raise
                        finish_stage('setup')
                    
                    _run_setup_with_detection()
                    setup_time = time.time() - setup_start_time

                    if setup_timeout is not None:
                        # Thread-mode cannot safely preempt a hung setup() without risking teardown races.
                        # Instead, treat setup_timeout_seconds as a post-hoc gate and fail fast once setup returns.
                        if setup_time > setup_timeout:
                            finish_stage('setup', status='timeout')
                            timeout_error = TimeoutError(
                                f"Setup exceeded timeout of {setup_timeout}s (ran for {setup_time:.1f}s)"
                            )
                            errors.append(str(timeout_error))
                            timeout_result_storage[0] = self._create_timeout_result(
                                stage="setup",
                                duration=setup_time,
                                limit=setup_timeout,
                                errors=errors,
                                benchmark_name=benchmark_name,
                                config=config,
                                watchdog=stage_watchdog,
                            )
                            return
                        if setup_time > setup_timeout * 0.8:  # Warn if setup takes >80% of timeout
                            logger.warning(f"Setup took {setup_time:.1f}s (near timeout limit of {setup_timeout}s)")
                    else:
                        # Warn if setup is suspiciously long (even without an explicit setup timeout).
                        measurement_timeout = config.get_effective_timeout('measurement')
                        if measurement_timeout is not None and setup_time > measurement_timeout * 0.5:
                            logger.warning(f"Setup took {setup_time:.1f}s (consider setting setup_timeout_seconds)")

                    if clock_lock_active and self.device.type == "cuda":
                        device_index = self.device.index if self.device.index is not None else 0
                        ramp_gpu_clocks(device=device_index)
                        if getattr(config, "reset_memory_pool", True):
                            from core.harness.validity_checks import reset_cuda_memory_pool

                            reset_cuda_memory_pool(self.device)
                    
                    # Warmup with timeout enforcement
                    warmup_timeout = config.get_effective_timeout('warmup')
                    if config.warmup > 0:
                        warmup_start_time = time.time()
                        start_stage('warmup')
                        try:
                            self._warmup(benchmark.benchmark_fn, config.warmup, config)
                        except Exception:
                            finish_stage('warmup', status='error')
                            raise
                        warmup_time = time.time() - warmup_start_time
                        if warmup_timeout is not None and warmup_time > warmup_timeout:
                            # Thread-mode cannot safely preempt a hung warmup() without risking teardown races.
                            # Treat warmup_timeout_seconds as a post-hoc gate and fail fast once warmup returns.
                            finish_stage('warmup', status='timeout')
                            timeout_error = TimeoutError(
                                f"Warmup exceeded timeout of {warmup_timeout}s (ran for {warmup_time:.1f}s)"
                            )
                            errors.append(str(timeout_error))
                            timeout_result_storage[0] = self._create_timeout_result(
                                stage="warmup",
                                duration=warmup_time,
                                limit=warmup_timeout,
                                errors=errors,
                                benchmark_name=benchmark_name,
                                config=config,
                                watchdog=stage_watchdog,
                            )
                            return
                        finish_stage('warmup')
                    else:
                        mark_stage('warmup', 'skipped')
                    
                    if clock_lock_active and self.device.type == "cuda":
                        device_index = self.device.index if self.device.index is not None else 0
                        # Re-assert application clocks right before measurement. We've observed rare cases where
                        # application clocks drift between targets (or get reset) which invalidates results.
                        # This is a best-effort repair, but if we still cannot reach the requested clocks,
                        # fail fast to avoid recording an invalid run.
                        try:
                            target_sm = getattr(config, "gpu_sm_clock_mhz", None)
                            target_mem = getattr(config, "gpu_mem_clock_mhz", None)
                            if target_sm is not None and target_mem is not None:
                                target_sm_i = int(target_sm)
                                target_mem_i = int(target_mem)
                                physical_index = _resolve_physical_device_index(device_index)

                                def _read_app_clocks() -> tuple[int, int]:
                                    snap = query_gpu_telemetry(device_index=device_index, force_refresh=True)
                                    app_sm = int(float(snap.get("applications_clock_sm_mhz") or -1))
                                    app_mem = int(float(snap.get("applications_clock_memory_mhz") or -1))
                                    return app_sm, app_mem

                                app_sm, app_mem = _read_app_clocks()
                                if abs(app_sm - target_sm_i) > 50 or abs(app_mem - target_mem_i) > 50:
                                    last_app = (app_sm, app_mem)
                                    last_lock_retry_error: Optional[BaseException] = None
                                    for _attempt in range(3):
                                        try:
                                            _nvidia_smi(["-i", str(physical_index), "-pm", "1"])
                                            _nvidia_smi(
                                                [
                                                    "-i",
                                                    str(physical_index),
                                                    f"--applications-clocks={target_mem_i},{target_sm_i}",
                                                ]
                                            )
                                        except Exception as lock_exc:
                                            last_lock_retry_error = lock_exc
                                        time.sleep(0.05)
                                        app_sm, app_mem = _read_app_clocks()
                                        last_app = (app_sm, app_mem)
                                        if abs(app_sm - target_sm_i) <= 50 and abs(app_mem - target_mem_i) <= 50:
                                            break
                                    else:
                                        retry_detail = (
                                            f" Last lock attempt error: {last_lock_retry_error}"
                                            if last_lock_retry_error is not None
                                            else ""
                                        )
                                        raise RuntimeError(
                                            "GPU clock lock invalid at measurement start: "
                                            f"requested SM={target_sm_i}MHz Mem={target_mem_i}MHz, "
                                            f"applications SM={last_app[0]}MHz Mem={last_app[1]}MHz."
                                            f"{retry_detail}"
                                        )

                            # Short ramp to bring current clocks up to the locked application clocks.
                            ramp_gpu_clocks(device=device_index, duration_ms=10.0, max_iters=40)
                            # Capture telemetry while clocks are locked so the run manifest reflects
                            # the real measurement environment (not the post-reset state).
                            locked_gpu_metrics = query_gpu_telemetry(device_index=device_index, force_refresh=True)
                        except Exception:
                            if not portable_mode:
                                raise
                            if LOGGER_AVAILABLE:
                                logger.warning(
                                    "Portable validity profile: app-clock verification unavailable; continuing.",
                                    exc_info=True,
                                )

                    # Memory tracking: Use context manager to track peak memory during benchmark execution
                    start_stage('measurement')
                    with self._memory_tracking(config) as mem_result:
                        # Benchmark using selected mode
                        # Note: nsys/ncu profiling wraps the entire process, so it's handled separately
                        if config.enable_profiling and (config.enable_nsys or config.enable_ncu or config.enable_proton):
                            if stage_watchdog['profiling']['status'] == 'pending':
                                start_stage('profiling')
                            # Delegate to unified profiling orchestration with timeout enforcement
                            profiling_timeout = config.get_effective_timeout('profiling')
                            if PROFILING_RUNNER_AVAILABLE:
                                try:
                                    from core.profiling.profiling_runner import run_profiling_orchestration
                                    import time
                                    profiling_start_time = time.time()

                                    # Wrap _benchmark_without_profiling to match expected signature
                                    def timing_wrapper(fn: Callable, cfg: BenchmarkConfig) -> List[float]:
                                        times, _ = self._benchmark_without_profiling(fn, cfg)
                                        return times

                                    try:
                                        prof_result = run_profiling_orchestration(
                                            benchmark,
                                            config,
                                            timing_fn=timing_wrapper,
                                            output_dir=Path(config.profiling_output_dir)
                                            if config.profiling_output_dir
                                            else None,
                                        )
                                    except Exception:
                                        finish_stage('profiling', status='error')
                                        raise

                                    profiling_time = time.time() - profiling_start_time
                                    if profiling_timeout is not None and profiling_time > profiling_timeout:
                                        # Thread-mode cannot safely preempt a hung profiler run without risking teardown
                                        # races. Treat profiling_timeout_seconds as a post-hoc gate once profiling returns.
                                        timeout_error = TimeoutError(
                                            f"Profiling exceeded timeout of {profiling_timeout}s (ran for {profiling_time:.1f}s)"
                                        )
                                        errors.append(str(timeout_error))
                                        finish_stage('profiling', status='timeout')
                                        timeout_result_storage[0] = self._create_timeout_result(
                                            stage="profiling",
                                            duration=profiling_time,
                                            limit=profiling_timeout,
                                            errors=errors,
                                            benchmark_name=benchmark_name,
                                            config=config,
                                            watchdog=stage_watchdog,
                                        )
                                        return

                                    finish_stage('profiling')
                                    
                                    if prof_result:
                                        times_ms = prof_result.get("times_ms", [])
                                        if "profiling_outputs" in prof_result:
                                            profiling_outputs.update(prof_result.get("profiling_outputs", {}))
                                        nsys_metrics = prof_result.get("nsys_metrics", {})
                                        ncu_metrics = prof_result.get("ncu_metrics", {})
                                        proton_metrics = prof_result.get("proton_metrics", {})
                                except ImportError:
                                    # Fallback if orchestration function not available
                                    if stage_watchdog['profiling']['status'] == 'pending':
                                        start_stage('profiling')
                                    times_ms, profiling_outputs = self._benchmark_with_profiling(
                                        benchmark.benchmark_fn, config
                                    )
                                    finish_stage('profiling')
                                    nsys_metrics = {}
                                    ncu_metrics = {}
                                    proton_metrics = {}
                            else:
                                # Fallback to PyTorch profiler if profiling runner not available
                                if stage_watchdog['profiling']['status'] == 'pending':
                                    start_stage('profiling')
                                times_ms, profiling_outputs = self._benchmark_with_profiling(
                                    benchmark.benchmark_fn, config
                                )
                                finish_stage('profiling')
                                nsys_metrics = {}
                                ncu_metrics = {}
                                proton_metrics = {}
                        elif config.enable_profiling:
                            start_stage('profiling')
                            times_ms, profiling_outputs = self._benchmark_with_profiling(
                                benchmark.benchmark_fn, config
                            )
                            finish_stage('profiling')
                        else:
                            mark_stage('profiling', 'skipped')
                            times_ms, inference_timing_data = self._benchmark_without_profiling(benchmark.benchmark_fn, config)
                    
                    # Extract memory tracking results from context manager
                    if mem_result is not None:
                        if not isinstance(mem_result, MemoryStats):
                            raise TypeError(f"Expected MemoryStats, got {type(mem_result)}")
                        memory_peak_mb = mem_result.peak_mb
                        memory_allocated_mb = mem_result.allocated_mb

                    # Allow benchmarks to register verification payload post-timing
                    try:
                        benchmark.capture_verification_payload()
                    except Exception as exc:
                        errors.append(f"capture_verification_payload() failed: {exc}")
                        raise
                    
                    # Validate result
                    validation_error = benchmark.validate_result()
                    if validation_error:
                        errors.append(f"Validation failed: {validation_error}")
                    
                    if hasattr(benchmark, "mark_execution_complete"):
                        try:
                            benchmark.mark_execution_complete()
                        except Exception as exc:
                            _emit_harness_warning(
                                f"{benchmark.__class__.__name__}.mark_execution_complete() failed during benchmark run",
                                exc=exc,
                            )

                    # Capture benchmark-specific metrics before teardown mutates benchmark state.
                    captured_custom_metrics = self._resolve_custom_metrics(benchmark)
                    
                    if stage_watchdog['measurement']['status'] == 'running':
                        finish_stage('measurement')
                
                except Exception as e:
                    error_msg = str(e) or repr(e)
                    # Handle generator errors gracefully (common with torch.compile)
                    if "generator didn't stop after throw" in error_msg:
                        errors.append(f"Benchmark execution failed: Generator error (likely from torch.compile or async operations)")
                    else:
                        errors.append(f"Benchmark execution failed: {error_msg}")
                        errors.append(f"Traceback: {traceback.format_exc()}")
                    times_ms = cast(List[float], [])
                    if stage_watchdog['measurement']['status'] == 'running':
                        finish_stage('measurement', status='error')
                finally:
                    # Mark execution as complete before teardown
                    execution_complete.set()
                    
                    # Teardown is now safe to call - we hold the lock
                    # Only call teardown once - set flag to prevent double invocation
                    if not teardown_called.is_set():
                        try:
                            benchmark.teardown()
                            teardown_called.set()
                        except Exception as e:
                            errors.append(f"Teardown failed: {str(e)}")
                            teardown_called.set()  # Mark as called even on error
                    
                    # Force cleanup (only if enabled to avoid distorting timings)
                    if config.enable_cleanup:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                    lock_stack.close()
        
        # ALWAYS run with timeout (required, default 15 seconds). Thread-mode benchmarks execute
        # inside a managed executor so we can recycle workers if a benchmark hangs.
        measurement_timeout = getattr(config, "measurement_timeout_seconds", None)
        if measurement_timeout is None:
            measurement_timeout = getattr(config, "timeout_seconds", None)
        if measurement_timeout is None:
            raise ValueError(
                "Thread execution requires a finite timeout; set BenchmarkConfig.measurement_timeout_seconds "
                "(or legacy timeout_seconds)."
            )
        if not isinstance(measurement_timeout, (int, float)) or measurement_timeout <= 0:
            raise ValueError(f"Invalid measurement timeout: {measurement_timeout!r}")
        measurement_timeout = float(measurement_timeout)
        thread_start_time = time.time()
        self._ensure_thread_executor()
        future = self._thread_executor.submit(run_benchmark_internal)
        timeout_result: Optional[PydanticBenchmarkResult] = None
        try:
            future.result(timeout=measurement_timeout)
            elapsed_time = time.time() - thread_start_time
        except _FuturesTimeoutError:
            elapsed_time = time.time() - thread_start_time
            future.cancel()
            finish_stage('measurement', status='timeout')

            if timeout_result_storage[0] is not None:
                timeout_result = timeout_result_storage[0]
            else:
                logger.error("=" * 80)
                logger.error("TIMEOUT: Benchmark execution exceeded timeout limit")
                logger.error("=" * 80)
                logger.error(f"   Benchmark: {benchmark_name}")
                logger.error(f"   Stage: measurement (benchmark iterations)")
                logger.error(f"   Timeout limit: {measurement_timeout} seconds")
                logger.error(f"   Elapsed time: {elapsed_time:.2f} seconds")
                logger.error(f"   Config: iterations={config.iterations}, warmup={config.warmup}")
                logger.error(f"   Status: Benchmark did not complete within timeout period")
                logger.error("")
                logger.error("   Possible causes:")
                logger.error("   - Benchmark is too slow for current timeout")
                logger.error("   - Benchmark is hanging or deadlocked")
                logger.error("   - CUDA kernel is hung (cannot be interrupted from Python)")
                logger.error("   - GPU is under heavy load from other processes")
                logger.error("")
                logger.error("   Suggested actions:")
                logger.error(f"   - Increase timeout: config.measurement_timeout_seconds = {measurement_timeout * 2}")
                logger.error(f"   - Use timeout multiplier: config.timeout_multiplier = 2.0")
                logger.error("   - Check for GPU resource contention")
                logger.error("   - Review benchmark code for potential deadlocks")
                logger.error("   - Use subprocess mode (more reliable): config.use_subprocess = True")
                logger.error("")
                logger.error("   WARNING: Threading timeout cannot force-stop hung CUDA kernels.")
                logger.error("=" * 80)
                
                timeout_error_msg = (
                    f"TIMEOUT: Benchmark measurement stage exceeded timeout of {measurement_timeout} seconds "
                    f"(ran for {elapsed_time:.2f}s). "
                    f"Consider increasing measurement_timeout_seconds or using timeout_multiplier. "
                    f"Thread-mode cannot force-stop hung CUDA kernels - consider subprocess mode for stricter isolation."
                )
                errors.append(timeout_error_msg)
                times_ms = cast(List[float], [])
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                gc.collect()
                gc.collect()
                
                timeout_result = self._create_timeout_result(
                    stage="measurement",
                    duration=elapsed_time,
                    limit=measurement_timeout,
                    errors=errors,
                    benchmark_name=benchmark_name,
                    config=config,
                    watchdog=stage_watchdog,
                )
        finally:
            self._reset_thread_executor()

        if timeout_result is not None:
            return timeout_result
        
        # Check if a timeout occurred in setup/warmup/profiling stages
        if timeout_result_storage[0] is not None:
            return timeout_result_storage[0]
        
        if execution_complete.is_set():
            # Thread completed normally - results are already set
            pass
        else:
            # Thread completed but execution_complete not set - something went wrong
            if not times_ms:
                errors.append("Benchmark execution completed but no results collected")
        # Don't print success message for normal completion - only print on timeout/failure
        
        if not times_ms:
            # Build comprehensive error message with context
            config_summary = (
                f"iterations={config.iterations}, warmup={config.warmup}, "
                f"timeout={getattr(config, 'measurement_timeout_seconds', config.timeout_seconds)}s"
            )
            error_details = [
                f"Benchmark: {benchmark_name}",
                f"Config: {config_summary}",
                f"Errors: {', '.join(errors)}",
            ]
            # Include partial results if we have any
            if memory_peak_mb is not None:
                error_details.append(f"Partial memory stats: peak={memory_peak_mb:.2f}MB")
            if profiling_outputs:
                error_details.append(f"Profiling artifacts collected: {list(profiling_outputs.keys())}")

            skip_reason = _extract_skip_reason_from_messages(errors)
            if skip_reason:
                skip_message = skip_reason if skip_reason.upper().startswith("SKIPPED") else f"SKIPPED: {skip_reason}"
                if LOGGER_AVAILABLE:
                    logger.warning("Benchmark '%s' skipped: %s", benchmark_name, skip_message)
                raise RuntimeError(skip_message)

            error_message = "\n".join(error_details)
            if LOGGER_AVAILABLE:
                logger.error("=" * 80)
                logger.error("BENCHMARK FAILED")
                logger.error("=" * 80)
                for detail in error_details:
                    logger.error(f"  {detail}")
                logger.error("=" * 80)
            
            failure_stage = next(
                (stage for stage, meta in stage_watchdog.items() if meta.get('status') in ('error', 'timeout')),
                "measurement",
            )
            failure_meta = stage_watchdog.get(failure_stage, {})
            duration = failure_meta.get("duration", 0.0) or 0.0
            limit_lookup = {
                "setup": config.get_effective_timeout("setup"),
                "warmup": config.get_effective_timeout("warmup"),
                "profiling": config.get_effective_timeout("profiling"),
                "measurement": getattr(config, 'measurement_timeout_seconds', config.timeout_seconds),
            }
            limit = limit_lookup.get(failure_stage, getattr(config, 'measurement_timeout_seconds', config.timeout_seconds))
            return self._create_timeout_result(
                stage=failure_stage,
                duration=duration,
                limit=limit,
                errors=errors,
                benchmark_name=benchmark_name,
                config=config,
                watchdog=stage_watchdog,
            )
        
        # Compute statistics
        result = self._compute_stats(times_ms, config)
        if locked_gpu_metrics is not None:
            result.gpu_metrics = locked_gpu_metrics
        result.seeds = copy.deepcopy(seed_metadata)

        # Attach benchmark-specific metrics and throughput (parity with subprocess path)
        custom_metrics = captured_custom_metrics or self._resolve_custom_metrics(benchmark)
        if custom_metrics:
            result.custom_metrics = custom_metrics
        story_metadata = self._resolve_story_metadata(benchmark)
        if story_metadata:
            result.story_metadata = story_metadata
        self._attach_throughput_metrics(result, benchmark)

        # Add inference timing if available
        if inference_timing_data and inference_timing_data.get("ttft_times_ms") and inference_timing_data.get("tpot_times_ms"):
            inference_timing = self._capture_inference_timing(
                inference_timing_data["ttft_times_ms"],
                inference_timing_data["tpot_times_ms"],
                config
            )
            if inference_timing:
                result.inference_timing = inference_timing
        
        # Add memory stats
        if memory_peak_mb is not None or memory_allocated_mb is not None:
            result.memory = MemoryStats(
                peak_mb=memory_peak_mb,
                allocated_mb=memory_allocated_mb,
                reserved_mb=None,
                schemaVersion="1.0",
            )
        
        # Add errors
        result.errors = errors
        
        # Add profiling artifacts
        if profiling_outputs:
            result.artifacts = ProfilerArtifacts(
                nsys_rep=profiling_outputs.get('nsys_rep') or profiling_outputs.get('nsys'),
                ncu_rep=profiling_outputs.get('ncu_rep') or profiling_outputs.get('ncu'),
                torch_trace_json=profiling_outputs.get('torch') or profiling_outputs.get('pytorch_trace'),
                proton_report=profiling_outputs.get('proton_report') or profiling_outputs.get('proton'),
                schemaVersion="1.0",
            )
        
        # Add profiler metrics
        nsys_metrics_obj = None
        ncu_metrics_obj = None
        proton_metrics_obj = None
        
        if nsys_metrics:
            nsys_metrics_obj = NsysMetrics(
                total_gpu_time_ms=nsys_metrics.get('nsys_total_gpu_time_ms'),
                raw_metrics={k.replace('nsys_', ''): v for k, v in nsys_metrics.items() 
                            if not k.startswith('nsys_total_gpu_time_ms')},
                schemaVersion="1.0",
            )
        
        if ncu_metrics:
            ncu_metrics_obj = NcuMetrics(
                kernel_time_ms=ncu_metrics.get('ncu_kernel_time_ms'),
                sm_throughput_pct=ncu_metrics.get('ncu_sm_throughput_pct'),
                dram_throughput_pct=ncu_metrics.get('ncu_dram_throughput_pct'),
                l2_throughput_pct=ncu_metrics.get('ncu_l2_throughput_pct'),
                occupancy_pct=ncu_metrics.get('ncu_occupancy_pct'),
                raw_metrics={k.replace('ncu_', ''): v for k, v in ncu_metrics.items() 
                            if not k.startswith(('ncu_kernel_time_ms', 'ncu_sm_throughput_pct', 
                                                'ncu_dram_throughput_pct', 'ncu_l2_throughput_pct', 'ncu_occupancy_pct'))},
                schemaVersion="1.0",
            )
        
        if proton_metrics:
            occ_field = proton_metrics.get('proton_occupancy_limited_kernels_list', proton_metrics.get('proton_occupancy_limited_kernels', ""))
            if isinstance(occ_field, list):
                occupancy = [str(item) for item in occ_field if item]
            else:
                occupancy = [item for item in occ_field.split(",") if item] if isinstance(occ_field, str) else []
            summary_stats: Dict[str, Any] = {}
            summary_from_report = proton_metrics.get('proton_summary_stats')
            if isinstance(summary_from_report, dict):
                summary_stats.update(summary_from_report)
            summary_stats.update({
                k.replace('proton_', ''): v for k, v in proton_metrics.items()
                if k.startswith('proton_') and k not in {'proton_kernel_count', 'proton_occupancy_limited_kernels', 'proton_top_kernels', 'proton_summary_stats', 'proton_kernel_summaries', 'proton_occupancy_limited_kernels_list'}
            })
            kernel_summaries = proton_metrics.get('proton_kernel_summaries')
            if not isinstance(kernel_summaries, list):
                kernel_summaries = []
            proton_metrics_obj = ProtonMetrics(
                kernel_count=proton_metrics.get('proton_kernel_count'),
                occupancy_limited_kernels=occupancy,
                summary_stats=summary_stats,
                kernel_summaries=kernel_summaries,
                schemaVersion="1.0",
            )
        
        if nsys_metrics_obj or ncu_metrics_obj or proton_metrics_obj:
            result.profiler_metrics = ProfilerMetrics(
                nsys=nsys_metrics_obj,
                ncu=ncu_metrics_obj,
                proton=proton_metrics_obj,
                torch=None,
                schemaVersion="1.0",
            )
        
        self._annotate_launch_metadata(result, config)
        return result
    
    def _benchmark_with_profiling(
        self, fn: Callable, config: BenchmarkConfig
    ) -> tuple[List[float], Dict[str, str]]:
        """Benchmark with profiling enabled."""
        self._ensure_runtime_initialized()
        profiling_outputs = {}
        
        # Create profiling output directory
        if config.profiling_output_dir:
            prof_dir = Path(config.profiling_output_dir)
            prof_dir.mkdir(parents=True, exist_ok=True)
        else:
            prof_dir = Path("profiling_results")
            prof_dir.mkdir(parents=True, exist_ok=True)
        
        # Try PyTorch profiler first (best for Python benchmarks)
        try:
            import torch.profiler
            
            torch_profiler_kwargs = {}
            try:
                from core.profiling import profiler_config as profiler_config_mod
                profiler_cfg = None
                if hasattr(profiler_config_mod, "build_profiler_config_from_benchmark"):
                    profiler_cfg = profiler_config_mod.build_profiler_config_from_benchmark(config)
                elif hasattr(profiler_config_mod, "DEFAULT_PROFILER_CONFIG"):
                    profiler_cfg = profiler_config_mod.DEFAULT_PROFILER_CONFIG
                if profiler_cfg is not None:
                    torch_profiler_kwargs = profiler_cfg.get_torch_profiler_config() or {}
            except Exception:
                torch_profiler_kwargs = {}

            if not torch_profiler_kwargs:
                # Minimal fallback to avoid heavy tracing overhead
                torch_profiler_kwargs = {
                    "activities": [torch.profiler.ProfilerActivity.CUDA],
                    "record_shapes": False,
                    "profile_memory": False,
                    "with_stack": False,
                    "with_flops": False,
                    "schedule": torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
                }
            
            # Run benchmark with PyTorch profiler
            with torch.profiler.profile(**torch_profiler_kwargs) as prof:
                profile_iterations = min(config.iterations, 10)
                times_ms = cast(List[float], [])
                is_cuda = self.device.type == "cuda"
                
                if is_cuda:
                    # Create events once, reuse across iterations
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(self.device)  # Sync once before loop
                    
                    for _ in range(profile_iterations):
                        start_event.record()
                        fn()
                        end_event.record()
                        torch.cuda.synchronize(self.device)
                        times_ms.append(start_event.elapsed_time(end_event))
                        prof.step()  # Record each iteration in profiling trace
                else:
                    # CPU: use high-resolution timer
                    for _ in range(profile_iterations):
                        start_time = time.perf_counter()
                        fn()
                        end_time = time.perf_counter()
                        times_ms.append((end_time - start_time) * 1000)
                        prof.step()  # Record each iteration in profiling trace
            
            # Export profiling data
            trace_file = prof_dir / "trace.json"
            prof.export_chrome_trace(str(trace_file))
            profiling_outputs["pytorch_trace"] = str(trace_file)
            
            return times_ms, profiling_outputs
            
        except Exception as e:
            # NO SILENT FALLBACK - profiling was explicitly requested, so raise on failure
            raise RuntimeError(f"Profiling failed (no silent fallback): {e}") from e
    
    
    # Profiling methods moved to profiling.profiling_runner module
    # Use run_nsys_profiling() and run_ncu_profiling() from profiling_runner.py
    # Wrapper generation moved to profiling.profiler_wrapper module
    # Use create_benchmark_wrapper() from profiler_wrapper.py
    
    def _benchmark_without_profiling(
        self, fn: Callable, config: BenchmarkConfig
    ) -> tuple[List[float], Optional[Dict[str, List[float]]]]:
        """Benchmark without profiling.
        
        Returns:
            Tuple of (times_ms, inference_timing_data) where inference_timing_data is None
            or a dict with 'ttft_times_ms' and 'tpot_times_ms' keys
        """
        if self.mode == BenchmarkMode.TRITON:
            times_ms = self._benchmark_triton(fn, config)
            return times_ms, None
        elif self.mode == BenchmarkMode.PYTORCH:
            times_ms = self._benchmark_pytorch(fn, config)
            return times_ms, None
        else:
            return self._benchmark_custom(fn, config)
    
    def _benchmark_triton(self, fn: Callable, config: BenchmarkConfig) -> List[float]:
        """Use Triton's do_bench (returns single value per call)."""
        try:
            import triton.testing as tt
            times_ms = cast(List[float], [])
            # Triton do_bench handles warmup internally, but we do our own
            for _ in range(config.iterations):
                time_ms = tt.do_bench(fn, warmup=0, rep=1)  # We handle warmup
                times_ms.append(time_ms)
            return times_ms
        except ImportError as e:
            # Triton not installed - fail explicitly, don't silently fall back
            raise RuntimeError(f"Triton benchmarking mode requested but Triton not available: {e}") from e
    
    def _benchmark_pytorch(self, fn: Callable, config: BenchmarkConfig) -> List[float]:
        """Use PyTorch's Timer."""
        try:
            from torch.utils.benchmark import Timer
            
            timer = Timer(
                stmt="fn()",
                globals={"fn": fn},
                num_threads=1,
            )
            
            # blocked_autorange runs until min_run_time is met
            measurement = timer.blocked_autorange(
                min_run_time=config.min_run_time_ms / 1000.0  # Convert to seconds
            )
            
            # measurement.times is already in seconds; tolerate tuple returns from mocks
            if hasattr(measurement, "times"):
                raw_times = measurement.times
            elif isinstance(measurement, (tuple, list)) and measurement:
                raw_times = measurement[0]
            else:
                raise RuntimeError("PyTorch Timer returned unexpected measurement format")

            times_ms = [t * 1000 for t in raw_times]

            # If we got fewer iterations than requested, pad with repeats.
            if len(times_ms) < config.iterations:
                times_ms = (times_ms * ((config.iterations // len(times_ms)) + 1))[:config.iterations]
            elif len(times_ms) > config.iterations:
                times_ms = times_ms[:config.iterations]

            return times_ms
        except Exception as e:
            # NO SILENT FALLBACK - PyTorch Timer mode was explicitly requested
            raise RuntimeError(f"PyTorch Timer benchmarking failed: {e}") from e
    
    def _benchmark_custom(self, fn: Callable, config: BenchmarkConfig) -> tuple[List[float], Optional[Dict[str, List[float]]]]:
        """Custom benchmarking with CUDA Events for accurate GPU timing.
        
        Optimized for minimal overhead:
        - Uses CUDA Events for accurate GPU timing without blocking
        - Reuses events across iterations for efficiency
        - Synchronizes only when necessary for accurate timing
        
        Returns:
            Tuple of (times_ms, inference_timing_data) where inference_timing_data is None
            or a dict with 'ttft_times_ms' and 'tpot_times_ms' keys
        """
        self._ensure_runtime_initialized()
        times_ms: List[float] = []
        inference_timing_data: Optional[Dict[str, List[float]]] = None
        ttft_times_ms: List[float] = []
        tpot_times_ms: List[float] = []

        # Capture config snapshot for immutability verification (Option C: full config).
        # This protects against benchmarks manipulating harness protections mid-run.
        config_snapshot = config.capture_config_snapshot() if getattr(config, 'enforce_config_immutability', True) else None

        # Some benchmarks (e.g., external CUDA binaries) report their own timing.
        benchmark_obj = getattr(fn, "__self__", None)
        use_reported_time = bool(getattr(benchmark_obj, "use_reported_time", False))
        range_name = getattr(config, "name", None) or getattr(fn, "__name__", "benchmark_fn")
        
        # ===== VALIDITY PROTECTIONS (Pre-benchmark) =====
        # These checks address benchmark validity issues.
        from core.harness.validity_checks import (
            reset_cuda_memory_pool, capture_gpu_state, gc_disabled,
            clear_compile_cache, force_tensor_evaluation, validate_environment,
            capture_precision_policy_state, check_precision_policy_consistency,
            MemoryAllocationTracker, get_active_streams,
            check_benchmark_fn_antipatterns, check_benchmark_fn_sync_calls,
        )
        
        # 0. Validate environment and log device enumeration
        env_result = validate_environment(
            device=self.device,
            probe=self._environment_probe,
            allow_virtualization=bool(getattr(config, "allow_virtualization", False)),
            allow_foreign_gpu_processes=bool(getattr(config, "allow_foreign_gpu_processes", False)),
        )
        for warning in env_result.warnings:
            import warnings as warn_module
            warn_module.warn(f"ENVIRONMENT WARNING: {warning}", RuntimeWarning)
        if LOGGER_AVAILABLE:
            for notice in env_result.notices:
                logger.info("ENVIRONMENT NOTICE: %s", notice)
        if env_result.errors:
            message = _format_environment_invalid_message(env_result.errors, config)
            enforce_env = bool(getattr(config, "enforce_environment_validation", True))
            if enforce_env:
                raise RuntimeError(message)
            import warnings as warn_module
            warn_module.warn(message, RuntimeWarning)
        
        # 1. Reset memory pool to prevent memory reuse gaming
        if getattr(config, 'reset_memory_pool', True) and self.device.type == "cuda":
            reset_cuda_memory_pool(self.device)
        
        # 2. Clear torch.compile cache for consistent compilation state
        # NOTE: compile caches are cleared before setup/warmup in BenchmarkHarness.benchmark().

        # 2b. Inspect benchmark_fn source for explicit synchronization in the hot path.
        if benchmark_obj is not None and getattr(config, "detect_benchmark_fn_sync", True):
            allowed_antipatterns = getattr(
                benchmark_obj,
                "allowed_benchmark_fn_antipatterns",
                (),
            )
            sync_ok, sync_findings = check_benchmark_fn_sync_calls(
                fn,
                allowed_codes=allowed_antipatterns,
            )
            if not sync_ok:
                policy = str(getattr(config, "benchmark_fn_sync_policy", "warn")).strip().lower()
                message = (
                    "BENCHMARK_FN SYNC WARNING: explicit synchronization detected in benchmark_fn(). "
                    "This usually inflates harness timings and blocks CUDA graph capture.\n- "
                    + "\n- ".join(sync_findings)
                )
                if policy == "error":
                    raise RuntimeError(message)
                if policy == "warn":
                    import warnings as warn_module

                    warn_module.warn(message, RuntimeWarning)

        if benchmark_obj is not None and getattr(config, "detect_benchmark_fn_antipatterns", True):
            antipattern_ok, antipattern_findings = check_benchmark_fn_antipatterns(
                fn,
                allowed_codes=allowed_antipatterns,
            )
            if not antipattern_ok:
                policy = str(
                    getattr(config, "benchmark_fn_antipattern_policy", "warn")
                ).strip().lower()
                message = (
                    "BENCHMARK_FN ANTI-PATTERN WARNING: benchmark_fn() contains hot-path work "
                    "that will distort timings or hide the real kernel cost.\n- "
                    + "\n- ".join(antipattern_findings)
                )
                if policy == "error":
                    raise RuntimeError(message)
                if policy == "warn":
                    import warnings as warn_module

                    warn_module.warn(message, RuntimeWarning)

        # 3. Capture GPU state before benchmark (for consistency check)
        gpu_state_before = None
        if getattr(config, 'monitor_gpu_state', True) and self.device.type == "cuda":
            gpu_state_before = capture_gpu_state(self.device.index or 0)

        # 3b. Capture backend precision policy before benchmark
        precision_policy_before = None
        if getattr(config, 'monitor_backend_policy', True):
            precision_policy_before = capture_precision_policy_state()
        
        # 4. Start memory allocation tracking
        memory_tracker = None
        if getattr(config, 'track_memory_allocations', True) and self.device.type == "cuda":
            memory_tracker = MemoryAllocationTracker(self.device)
            memory_tracker.start()
        
        is_cuda = self.device.type == "cuda"
        force_eval_warning_emitted = False
        
        if is_cuda:
            timing_method = getattr(config, "timing_method", "cuda_event")
            if timing_method not in {"cuda_event", "wall_clock"}:
                raise RuntimeError(
                    f"Invalid timing_method '{timing_method}'. "
                    "Expected 'cuda_event' or 'wall_clock'."
                )
            if timing_method == "wall_clock" and not getattr(config, "full_device_sync", True):
                raise RuntimeError("timing_method='wall_clock' requires full_device_sync=True.")
            use_cuda_events = timing_method == "cuda_event"
            # Use CUDA Events for accurate GPU timing
            # Create events once - reuse across iterations (efficient)
            start_event = torch.cuda.Event(enable_timing=True) if use_cuda_events else None
            end_event = torch.cuda.Event(enable_timing=True) if use_cuda_events else None
            
            # L2 cache clearing buffer (Triton best practice)
            # See: https://github.com/triton-lang/triton/blob/main/python/triton/testing.py
            # Buffer size is dynamically determined based on GPU architecture:
            # - Blackwell (B100/B200): 96MB L2
            # - Hopper (H100/H200): 50MB L2
            # - Ampere (A100): 40MB L2
            # See core/harness/l2_cache_utils.py for details
            l2_cache_buffer = None
            if getattr(config, 'clear_l2_cache', False):
                from core.harness.l2_cache_utils import create_l2_flush_buffer
                l2_cache_buffer = create_l2_flush_buffer(self.device)
            
            # Gradient clearing (Triton best practice)
            grad_tensors = []
            grad_names = getattr(config, 'grad_to_none', None)
            if grad_names and benchmark_obj:
                for name in grad_names:
                    tensor = getattr(benchmark_obj, name, None)
                    if tensor is not None and hasattr(tensor, 'grad'):
                        grad_tensors.append(tensor)
            
            # Synchronize once before starting to ensure clean state
            torch.cuda.synchronize(self.device)
            
            # 4. Disable GC during timing to prevent GC interference
            # This is wrapped around the timing loop, not the full benchmark
            gc_context = gc_disabled() if getattr(config, 'disable_gc_during_timing', True) else nullcontext()
            
            # Run benchmark iterations with accurate per-iteration timing
            # CUDA Events provide accurate timing with minimal overhead
            # 
            # Cross-validation: Also track wall clock time to detect timing anomalies
            # This protects against cases where CUDA event timing is manipulated
            wall_clock_times_ms: List[float] = []
            
            # Adaptive iterations setup (Triton-style best practice)
            # If enabled, dynamically adjust iterations to achieve target duration
            use_adaptive = getattr(config, 'adaptive_iterations', False)
            min_total_duration_ms = getattr(config, 'min_total_duration_ms', 100.0)
            max_adaptive_iterations = getattr(config, 'max_adaptive_iterations', 10000)
            target_iterations = config.iterations  # Fixed iterations (or initial if adaptive)
            total_duration_ms = 0.0
            iteration_count = 0
            
            # CUDA Graph mode setup (Triton-style best practice)
            # Captures the benchmark function once, then replays for each iteration
            # This reduces CPU overhead significantly for repetitive kernel launches
            use_cuda_graph = getattr(config, 'enable_cuda_graph', False)
            cuda_graph = None
            cuda_graph_captured = False
            graph_cheat_detector = GraphCaptureCheatDetector(self.device) if use_cuda_graph else None
            internal_stream_ids: Set[int] = set()
            graph_replay_warning_emitted = False
            
            def _run_single_iteration():
                """Helper to run a single benchmark iteration and return timing."""
                nonlocal iteration_count, total_duration_ms, cuda_graph, cuda_graph_captured
                nonlocal graph_replay_warning_emitted, force_eval_warning_emitted
                
                # Clear L2 cache before each iteration (Triton best practice)
                # This ensures each iteration measures "cold cache" performance
                if l2_cache_buffer is not None:
                    l2_cache_buffer.zero_()
                    torch.cuda.synchronize(self.device)
                
                # Clear gradients (Triton best practice)
                for tensor in grad_tensors:
                    tensor.grad = None
                
                # Start wall clock timer for cross-validation
                wall_start = time.perf_counter()
                
                # Record start event (non-blocking)
                if start_event is not None:
                    start_event.record()
                
                result = None
                
                # CUDA Graph mode (Triton-style best practice)
                # If enabled and already captured, replay the graph instead of running fn()
                if use_cuda_graph and cuda_graph_captured and cuda_graph is not None:
                    # Replay captured graph (much faster than re-running fn())
                    if graph_cheat_detector is not None:
                        graph_cheat_detector.start_replay()
                    cuda_graph.replay()
                else:
                    # Execute function under test with NVTX
                    with self._nvtx_range(range_name):
                        result = fn()

                if is_cuda and getattr(config, "force_synchronize", False):
                    torch.cuda.synchronize(self.device)
                
                # Record end event (non-blocking)
                if end_event is not None:
                    end_event.record()
                
                # CRITICAL: Full device sync vs event sync (Triton/Locus best practice)
                # event.synchronize() only waits for operations on the stream where the event was recorded.
                # torch.cuda.synchronize() waits for ALL streams on ALL devices.
                # This is critical for protecting against multi-stream timing exploits.
                # See: Locus/KernelBench 2025 - 32.8% of RL kernels exploited stream timing loopholes.
                if getattr(config, 'full_device_sync', True):
                    torch.cuda.synchronize()  # Wait for ALL streams (stream-safe)
                else:
                    end_event.synchronize()  # Only wait for this event's stream (faster but less safe)
                
                # Stop wall clock timer
                wall_end = time.perf_counter()
                wall_elapsed_ms = (wall_end - wall_start) * 1000
                wall_clock_times_ms.append(wall_elapsed_ms)
                
                if start_event is not None and end_event is not None:
                    elapsed_ms = start_event.elapsed_time(end_event)
                else:
                    elapsed_ms = wall_elapsed_ms
                if use_cuda_graph and cuda_graph_captured and cuda_graph is not None and graph_cheat_detector is not None:
                    try:
                        graph_cheat_detector.end_replay(elapsed_ms)
                    except Exception as exc:
                        if not graph_replay_warning_emitted:
                            _emit_harness_warning(
                                "Graph cheat detector replay finalization failed; graph-capture audit may be incomplete",
                                exc=exc,
                            )
                            graph_replay_warning_emitted = True
                if use_reported_time:
                    reported = getattr(benchmark_obj, "last_time_ms", None)
                    if reported is not None:
                        elapsed_ms = reported
                times_ms.append(elapsed_ms)
                if benchmark_obj is not None:
                    benchmark_obj._last_elapsed_ms = elapsed_ms
                    benchmark_obj._last_wall_elapsed_ms = wall_elapsed_ms

                if benchmark_obj is not None:
                    finalize_iteration = getattr(benchmark_obj, "finalize_iteration_metrics", None)
                    if callable(finalize_iteration):
                        finalized = finalize_iteration()
                        if finalized is not None:
                            if result is None:
                                result = finalized
                            elif isinstance(result, dict):
                                merged = dict(result)
                                merged.update(finalized)
                                result = merged
                
                # Check if function returned inference timing data (only if not graph replay)
                if result is not None and isinstance(result, dict):
                    if "ttft_times_ms" in result and isinstance(result["ttft_times_ms"], list):
                        ttft_times_ms.extend(result["ttft_times_ms"])
                    if "tpot_times_ms" in result and isinstance(result["tpot_times_ms"], list):
                        tpot_times_ms.extend(result["tpot_times_ms"])
                
                iteration_count += 1
                total_duration_ms += elapsed_ms
                
                # Force evaluation of lazy tensors after timing to prevent skipped compute
                if getattr(config, 'force_tensor_evaluation', True) and result is not None:
                    try:
                        tensors_to_force: Dict[str, Any] = {}
                        if isinstance(result, torch.Tensor):
                            tensors_to_force["result"] = result
                        elif isinstance(result, dict):
                            for k, v in result.items():
                                if isinstance(v, torch.Tensor):
                                    tensors_to_force[k] = v
                        if tensors_to_force:
                            force_tensor_evaluation(tensors_to_force)
                    except Exception as exc:
                        if not force_eval_warning_emitted:
                            _emit_harness_warning(
                                "Failed to force eager tensor evaluation after timing; lazy-evaluation protection may be incomplete",
                                exc=exc,
                            )
                            force_eval_warning_emitted = True
                return elapsed_ms, result
            
            def _capture_cuda_graph():
                """Capture the benchmark function as a CUDA graph for fast replay."""
                nonlocal cuda_graph, cuda_graph_captured, internal_stream_ids
                
                try:
                    # CUDA graphs must be captured on a non-default stream. Capturing on
                    # the default stream can leave RNG in an invalid state on failure.
                    graph_warmup_iters = getattr(config, 'cuda_graph_warmup_iters', 3)
                    capture_stream = torch.cuda.Stream(device=self.device)
                    internal_stream_ids.add(int(capture_stream.cuda_stream))
                    # Ensure capture stream sees prior work.
                    capture_stream.wait_stream(torch.cuda.current_stream(self.device))
                    for _ in range(graph_warmup_iters):
                        with torch.cuda.stream(capture_stream):
                            fn()
                        torch.cuda.synchronize(self.device)
                    
                    # Capture the CUDA graph
                    # Note: The benchmark function must be deterministic and have no CPU work
                    cuda_graph = torch.cuda.CUDAGraph()
                    
                    # Warm up before capture (ensures allocations are complete)
                    with torch.cuda.stream(capture_stream):
                        fn()
                    torch.cuda.synchronize(self.device)
                    
                    # Capture the graph
                    if graph_cheat_detector is not None:
                        graph_cheat_detector.start_capture()
                    with torch.cuda.graph(cuda_graph, stream=capture_stream):
                        fn()
                    if graph_cheat_detector is not None:
                        graph_cheat_detector.end_capture()
                    torch.cuda.synchronize(self.device)
                    
                    cuda_graph_captured = True
                except Exception as e:
                    # CUDA graph capture failed - fall back to normal execution
                    # This can happen if:
                    # - The benchmark has dynamic control flow
                    # - There are CPU-GPU syncs inside benchmark_fn()
                    # - The benchmark does allocations inside the function
                    import warnings
                    warnings.warn(
                        f"CUDA graph capture failed, falling back to normal execution: {e}. "
                        "Ensure benchmark_fn() has no dynamic shapes, CPU work, or allocations.",
                        RuntimeWarning,
                    )
                    cuda_graph = None
                    cuda_graph_captured = False
            
            audit_streams_enabled = getattr(config, 'audit_stream_sync', True)
            pre_streams: List[int] = []
            declared_streams: List[Any] = []
            if audit_streams_enabled and hasattr(benchmark_obj, "get_custom_streams"):
                try:
                    streams = benchmark_obj.get_custom_streams()
                    if streams:
                        declared_streams = [streams] if isinstance(streams, torch.cuda.Stream) else list(streams)
                except Exception as exc:
                    raise RuntimeError(f"get_custom_streams() failed: {exc}")
            stream_auditor = None
            stream_context = audit_streams(self.device) if audit_streams_enabled else nullcontext()
            if audit_streams_enabled:
                pre_streams = get_active_streams(self.device, declared_streams)

            # Wrap timing loop with GC context to prevent GC interference
            with gc_context:
                with stream_context as stream_auditor:
                    if audit_streams_enabled and stream_auditor is not None:
                        for stream in declared_streams:
                            stream_auditor.record_stream_event(stream, operation="declared_stream")
                    # CUDA Graph capture (if enabled)
                    # Capture before starting timed iterations for graph replay mode
                    if use_cuda_graph:
                        _capture_cuda_graph()
                    
                    if use_adaptive:
                        # Adaptive iterations mode (Triton best practice)
                        # Run initial batch to estimate per-iteration time
                        initial_batch_size = min(5, target_iterations)  # At least 5 or requested iterations
                        
                        for _ in range(initial_batch_size):
                            _run_single_iteration()
                        
                        # If we haven't reached target duration, calculate needed iterations
                        if total_duration_ms < min_total_duration_ms:
                            avg_time = total_duration_ms / initial_batch_size if initial_batch_size > 0 else 1.0
                            remaining_duration = min_total_duration_ms - total_duration_ms
                            additional_iterations = int(remaining_duration / avg_time) + 1
                            
                            # Cap at max_adaptive_iterations
                            additional_iterations = min(additional_iterations, max_adaptive_iterations - initial_batch_size)
                            
                            # Run additional iterations
                            for _ in range(max(0, additional_iterations)):
                                _run_single_iteration()
                                # Check if we've exceeded target duration
                                if total_duration_ms >= min_total_duration_ms:
                                    break
                    else:
                        # Fixed iterations mode (original behavior)
                        for _ in range(config.iterations):
                            _run_single_iteration()
            
            # Stream audit check - verify all streams are properly synchronized
            # This detects the Locus/KernelBench stream timing vulnerability
            if audit_streams_enabled:
                post_streams = get_active_streams(self.device, declared_streams)
                sync_complete, sync_warning = check_stream_sync_completeness(pre_streams, post_streams)
                audit_ok = True
                audit_warnings: List[str] = []
                if stream_auditor is not None:
                    audit_ok, audit_warnings = stream_auditor.check_issues()
                    if audit_ok and not audit_warnings and LOGGER_AVAILABLE:
                        try:
                            info = stream_auditor.get_info()
                            logger.debug(
                                f"Stream audit passed (custom_streams={info.custom_streams_detected}, unsync_warning={info.unsync_warning})"
                            )
                        except Exception:
                            logger.debug("Stream audit passed")
                issues: List[str] = []
                if not sync_complete:
                    issues.append("Stream synchronization incomplete")
                if sync_warning:
                    issues.append(sync_warning)
                if stream_auditor is not None:
                    try:
                        info = stream_auditor.get_info()
                        declared_ids = set(post_streams)
                        if info.default_stream_id is not None:
                            declared_ids.add(info.default_stream_id)
                        if internal_stream_ids:
                            declared_ids.update(internal_stream_ids)
                        if getattr(benchmark_obj, "declare_all_streams", False):
                            declared_ids.update(info.stream_ids)
                        elif declared_streams:
                            # Only add explicitly declared streams when auto-declare is disabled
                            declared_ids.update({s.cuda_stream for s in declared_streams if hasattr(s, "cuda_stream")})
                        undeclared_streams = info.stream_ids - declared_ids
                        if undeclared_streams:
                            issues.append(
                                f"UNDECLARED STREAMS: {len(undeclared_streams)} stream(s) were created/used without being declared via get_custom_streams()."
                            )
                    except Exception as exc:
                        issues.append(f"Failed to inspect stream audit info: {exc}")
                if not audit_ok or audit_warnings:
                    issues.extend(audit_warnings)
                if issues:
                    raise RuntimeError(
                        "STREAM TIMING VIOLATION: " +
                        " | ".join(issues) +
                        " | All streams must be synchronized to avoid under-timing (see Locus/KernelBench 2025)."
                    )
            
            # Graph capture cheat detection: fail fast if capture vs replay looks suspicious
            if graph_cheat_detector is not None:
                ratio_thresh = getattr(config, 'graph_capture_cheat_ratio_threshold', 10.0)
                mem_thresh = getattr(config, 'graph_capture_memory_threshold_mb', 100.0)
                is_cheating, cheat_reason = graph_cheat_detector.check_for_cheat(
                    capture_replay_ratio_threshold=ratio_thresh,
                    memory_threshold_mb=mem_thresh,
                )
                if is_cheating:
                    raise RuntimeError(f"Graph capture cheat detected: {cheat_reason}")
                elif LOGGER_AVAILABLE:
                    try:
                        stats = graph_cheat_detector.get_stats()
                        logger.debug(f"Graph capture check passed (ratio<= {ratio_thresh}, mem<= {mem_thresh}MB): {stats}")
                    except Exception:
                        logger.debug("Graph capture check passed")
            
            # Cross-validate CUDA event timing vs wall clock timing
            # Flag anomalies where CUDA events report much less time than wall clock
            # (this could indicate timing manipulation or missing stream sync)
            timing_cross_validation_ratio = None
            if (
                timing_method == "cuda_event"
                and getattr(config, 'cross_validate_timing', True)
                and wall_clock_times_ms
                and times_ms
            ):
                cuda_median = sorted(times_ms)[len(times_ms) // 2]
                wall_median = sorted(wall_clock_times_ms)[len(wall_clock_times_ms) // 2]
                
                # Calculate the ratio for reporting
                timing_cross_validation_ratio = cuda_median / wall_median if wall_median > 0 else 1.0
                
                # Configurable threshold (default 0.5 = warn if CUDA < 50% of wall)
                threshold = getattr(config, 'timing_cross_validation_threshold', 0.5)
                
                if timing_cross_validation_ratio < threshold:
                    message = (
                        f"TIMING CROSS-VALIDATION FAILURE: CUDA event timing ({cuda_median:.3f}ms) "
                        f"is only {timing_cross_validation_ratio*100:.1f}% of wall clock timing ({wall_median:.3f}ms). "
                        f"This likely indicates timing manipulation or missing stream synchronization. "
                        f"Set full_device_sync=True and ensure all streams are synchronized."
                    )
                    if _is_chapter_or_labs_benchmark(benchmark_obj):
                        raise RuntimeError(message)
                    import warnings
                    warnings.warn(message, RuntimeWarning)
        else:
            # CPU: use high-resolution timer
            for _ in range(config.iterations):
                start_time = time.perf_counter()
                with self._nvtx_range(range_name):
                    result = fn()
                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000
                if use_reported_time:
                    reported = getattr(benchmark_obj, "last_time_ms", None)
                    if reported is not None:
                        elapsed_ms = reported
                times_ms.append(elapsed_ms)
                if benchmark_obj is not None:
                    benchmark_obj._last_elapsed_ms = elapsed_ms
                    benchmark_obj._last_wall_elapsed_ms = elapsed_ms
                
                # Check if function returned inference timing data
                if isinstance(result, dict):
                    if "ttft_times_ms" in result and isinstance(result["ttft_times_ms"], list):
                        ttft_times_ms.extend(result["ttft_times_ms"])
                    if "tpot_times_ms" in result and isinstance(result["tpot_times_ms"], list):
                        tpot_times_ms.extend(result["tpot_times_ms"])
                
                if getattr(config, 'force_tensor_evaluation', True) and result is not None:
                    try:
                        tensors_to_force: Dict[str, Any] = {}
                        if isinstance(result, torch.Tensor):
                            tensors_to_force["result"] = result
                        elif isinstance(result, dict):
                            for k, v in result.items():
                                if isinstance(v, torch.Tensor):
                                    tensors_to_force[k] = v
                        if tensors_to_force:
                            force_tensor_evaluation(tensors_to_force)
                    except Exception as exc:
                        if not force_eval_warning_emitted:
                            _emit_harness_warning(
                                "Failed to force eager tensor evaluation after timing; lazy-evaluation protection may be incomplete",
                                exc=exc,
                            )
                            force_eval_warning_emitted = True
        
        # Build inference timing data if collected
        if ttft_times_ms or tpot_times_ms:
            inference_timing_data = {
                "ttft_times_ms": ttft_times_ms,
                "tpot_times_ms": tpot_times_ms,
            }
        
        # Verify harness config wasn't modified during benchmark execution
        if config_snapshot is not None:
            unchanged, error_msg = config.verify_config_unchanged(config_snapshot)
            if not unchanged:
                raise RuntimeError(
                    f"BENCHMARK CONFIG MANIPULATION DETECTED: {error_msg}. "
                    "Benchmarks must not mutate harness configuration during execution. "
                    "Set enforce_config_immutability=False to disable this check (not recommended)."
                )

        # Detect RNG seed mutation during performance runs.
        expected_torch_seed = getattr(self, "_seed_info", {}).get("torch_seed")
        if expected_torch_seed is not None:
            current_seed = int(torch.initial_seed())
            if current_seed != expected_torch_seed:
                raise RuntimeError(
                    "Seed mutation detected during benchmark execution. "
                    f"Expected torch.initial_seed()={expected_torch_seed}, got {current_seed}. "
                    "Benchmarks MUST NOT reseed during setup/warmup/measurement; rely on harness-configured seeds."
                )
        if torch.cuda.is_available():
            expected_cuda_seed = getattr(self, "_seed_info", {}).get("cuda_seed")
            if expected_cuda_seed is not None:
                current_cuda_seed = int(torch.cuda.initial_seed())
                if current_cuda_seed != expected_cuda_seed:
                    raise RuntimeError(
                        "CUDA seed mutation detected during benchmark execution. "
                        f"Expected torch.cuda.initial_seed()={expected_cuda_seed}, got {current_cuda_seed}. "
                        "Benchmarks MUST NOT reseed during setup/warmup/measurement; rely on harness-configured seeds."
                    )
        
        # ===== VALIDITY PROTECTIONS (Post-benchmark) =====
        # Check backend precision policy for unexpected mutation
        if precision_policy_before is not None and getattr(config, 'monitor_backend_policy', True):
            precision_policy_after = capture_precision_policy_state()
            _, policy_warnings = check_precision_policy_consistency(
                precision_policy_before,
                precision_policy_after,
            )
            if policy_warnings:
                message = "BACKEND POLICY MUTATION DETECTED: " + " | ".join(policy_warnings)
                enforce_policy = bool(getattr(config, "enforce_backend_policy_immutability", True))
                if enforce_policy and (_is_chapter_or_labs_benchmark(benchmark_obj) if benchmark_obj else False):
                    raise RuntimeError(message)
                import warnings as warn_module
                warn_module.warn(message, RuntimeWarning)

        # Check GPU state consistency to detect throttling/thermal issues
        if gpu_state_before is not None and getattr(config, 'monitor_gpu_state', True):
            from core.harness.validity_checks import check_gpu_state_consistency
            gpu_state_after = capture_gpu_state(self.device.index or 0)
            is_consistent, gpu_warnings = check_gpu_state_consistency(gpu_state_before, gpu_state_after)
            
            if gpu_warnings:
                for warning in gpu_warnings:
                    import warnings as warn_module
                    warn_module.warn(f"GPU STATE WARNING: {warning}", RuntimeWarning)
        
        # Check memory allocation patterns for suspicious behavior
        if memory_tracker is not None:
            memory_tracker.stop()
            no_mem_issues, mem_warnings = memory_tracker.check_patterns()
            
            if mem_warnings:
                for warning in mem_warnings:
                    import warnings as warn_module
                    warn_module.warn(f"MEMORY TRACKING WARNING: {warning}", RuntimeWarning)
        
        return times_ms, inference_timing_data
    
    def _warmup(self, fn: Callable, warmup_iterations: int, config: Optional[BenchmarkConfig] = None) -> None:
        """Perform warmup iterations.
        
        Args:
            fn: The function to warm up
            warmup_iterations: Number of warmup iterations
            config: Optional benchmark config for isolation settings
        """
        is_cuda = self.device.type == "cuda"
        range_name = getattr(fn, "__name__", "warmup")
        for _ in range(warmup_iterations):
            with self._nvtx_range(f"{range_name}_warmup"):
                fn()
            if is_cuda and config and getattr(config, "force_synchronize", False):
                torch.cuda.synchronize(self.device)
        if is_cuda:
            torch.cuda.synchronize(self.device)
            
            # Warmup buffer isolation: Clear L2 cache after warmup (Triton best practice)
            # This prevents warmup from pre-populating caches that artificially
            # speed up measurement iterations.
            # Buffer size is dynamically determined based on GPU architecture:
            # - Blackwell (B100/B200): 96MB L2
            # - Hopper (H100/H200): 50MB L2  
            # - Ampere (A100): 40MB L2
            # See core/harness/l2_cache_utils.py for details
            isolate = getattr(config, 'isolate_warmup_cache', True) if config else True
            if isolate:
                from core.harness.l2_cache_utils import flush_l2_cache
                flush_l2_cache(self.device)
                # Force garbage collection
                import gc
                gc.collect()
                torch.cuda.empty_cache()
    
    def _compute_percentiles(self, values: List[float], percentiles: List[float]) -> Dict[float, float]:
        """Compute percentiles for a list of values.
        
        Args:
            values: List of numeric values
            percentiles: List of percentile values (e.g., [50.0, 90.0, 95.0, 99.0])
            
        Returns:
            Dictionary mapping percentile to value
        """
        if not values:
            return {}
        
        # Ensure percentiles is a list
        if percentiles is None:
            percentiles = [25, 50, 75, 99]
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        percentiles_dict = {}
        
        for p in percentiles:
            idx = int((p / 100.0) * (n - 1))
            idx = min(idx, n - 1)
            percentiles_dict[p] = sorted_values[idx]
        
        return percentiles_dict
    
    def _capture_inference_timing(
        self,
        ttft_times_ms: List[float],
        tpot_times_ms: List[float],
        config: BenchmarkConfig
    ) -> Optional[InferenceTimingStats]:
        """Capture inference timing statistics from raw TTFT and TPOT measurements.
        
        Args:
            ttft_times_ms: List of TTFT measurements (one per request) in milliseconds
            tpot_times_ms: List of TPOT measurements (one per token) in milliseconds
            config: BenchmarkConfig for percentile configuration
            
        Returns:
            InferenceTimingStats object or None if no data
        """
        if not ttft_times_ms or not tpot_times_ms:
            return None
        
        # Compute TTFT percentiles - ensure percentiles is a list
        percentiles = config.percentiles if config.percentiles is not None else [25, 50, 75, 99]
        ttft_percentiles = self._compute_percentiles(ttft_times_ms, percentiles)
        ttft_p50 = ttft_percentiles.get(50.0) or statistics.median(ttft_times_ms)
        ttft_p90 = ttft_percentiles.get(90.0)
        ttft_p95 = ttft_percentiles.get(95.0)
        ttft_p99 = ttft_percentiles.get(99.0)
        
        # Compute TPOT percentiles
        tpot_percentiles = self._compute_percentiles(tpot_times_ms, percentiles)
        tpot_p50 = tpot_percentiles.get(50.0) or statistics.median(tpot_times_ms)
        tpot_p90 = tpot_percentiles.get(90.0)
        tpot_p95 = tpot_percentiles.get(95.0)
        tpot_p99 = tpot_percentiles.get(99.0)
        
        # Count requests and tokens
        num_requests = len(ttft_times_ms)
        total_tokens_generated = len(tpot_times_ms)
        
        return InferenceTimingStats(
            ttft_mean_ms=statistics.mean(ttft_times_ms),
            ttft_p50_ms=ttft_p50,
            ttft_p90_ms=ttft_p90,
            ttft_p95_ms=ttft_p95,
            ttft_p99_ms=ttft_p99,
            ttft_percentiles=ttft_percentiles,
            tpot_mean_ms=statistics.mean(tpot_times_ms),
            tpot_p50_ms=tpot_p50,
            tpot_p90_ms=tpot_p90,
            tpot_p95_ms=tpot_p95,
            tpot_p99_ms=tpot_p99,
            tpot_percentiles=tpot_percentiles,
            num_requests=num_requests,
            total_tokens_generated=total_tokens_generated,
            raw_ttft_times_ms=ttft_times_ms,
            raw_tpot_times_ms=tpot_times_ms,
            schemaVersion="1.0",
        )
    
    def _compute_stats(
        self, times_ms: List[float], config: BenchmarkConfig
    ) -> PydanticBenchmarkResult:
        """Compute statistical measures and return Pydantic BenchmarkResult."""
        if not times_ms:
            raise ValueError("No timing data collected")
        
        # CRITICAL: Filter out None values and ensure all values are numeric
        # This prevents "'<' not supported between instances of 'NoneType' and 'list'" errors
        valid_times_ms: List[float] = []
        for t in times_ms:
            if t is not None and isinstance(t, (int, float)):
                valid_times_ms.append(float(t))
            elif isinstance(t, list):
                # If we somehow got a list, extract numeric values from it
                valid_times_ms.extend([float(x) for x in t if x is not None and isinstance(x, (int, float))])
        
        if not valid_times_ms:
            raise ValueError(f"No valid timing data collected. Original times_ms had {len(times_ms)} entries, but none were valid numbers.")
        
        sorted_times = sorted(valid_times_ms)
        n = len(sorted_times)
        
        # Compute percentiles - ensure percentiles is a list
        percentiles = config.percentiles if config.percentiles is not None else [25, 50, 75, 99]
        percentiles_dict = {}
        for p in percentiles:
            idx = int((p / 100.0) * (n - 1))
            idx = min(idx, n - 1)
            percentiles_dict[p] = sorted_times[idx]
        
        # Extract p50/p90/p95/p99 from percentiles
        p50 = percentiles_dict.get(50.0) or statistics.median(valid_times_ms)
        p90 = percentiles_dict.get(90.0)
        p95 = percentiles_dict.get(95.0)
        p99 = percentiles_dict.get(99.0)
        
        timing = TimingStats(
            mean_ms=statistics.mean(valid_times_ms),
            median_ms=statistics.median(valid_times_ms),
            std_ms=statistics.stdev(valid_times_ms) if n > 1 else 0.0,
            min_ms=min(valid_times_ms),
            max_ms=max(valid_times_ms),
            p50_ms=p50,
            p90_ms=p90,
            p95_ms=p95,
            p99_ms=p99,
            percentiles=percentiles_dict,
            iterations=n,
            warmup_iterations=config.warmup,
            raw_times_ms=valid_times_ms,  # Store cleaned data, not original
            schemaVersion="1.0",
        )
        
        device_index = None
        if torch.cuda.is_available():
            device_index = self.device.index if getattr(self.device, "index", None) is not None else torch.cuda.current_device()
        gpu_metrics = query_gpu_telemetry(device_index=device_index)

        result = PydanticBenchmarkResult(
            timing=timing,
            inference_timing=None,
            memory=None,
            artifacts=None,
            profiler_metrics=None,
            errors=[],
            validation_status=None,
            validation_message=None,
            benchmark_name=None,
            device=str(self.device),
            mode=self.mode.value,
            timeout_stage=None,
            timeout_duration_seconds=None,
            timeout_limit_seconds=None,
            gpu_metrics=gpu_metrics,
            seeds=copy.deepcopy(getattr(self, "_seed_info", None)),
            schemaVersion="1.0",
        )
        self._annotate_launch_metadata(result, config)
        return result

    def _resolve_workload_metadata(self, benchmark: BaseBenchmark) -> Optional[WorkloadMetadata]:
        """Resolve workload metadata declared by the benchmark (if any)."""
        metadata: Optional[WorkloadMetadata] = None
        getter = getattr(benchmark, "get_workload_metadata", None)
        if callable(getter):
            try:
                metadata = getter()
            except Exception as exc:  # pragma: no cover - defensive
                if LOGGER_AVAILABLE:
                    logger.debug(f"get_workload_metadata() raised: {exc}")
        if metadata is None:
            candidate = getattr(benchmark, "workload_metadata", None)
            if isinstance(candidate, WorkloadMetadata):
                metadata = candidate
        if metadata is None:
            metadata = self._infer_workload_metadata_from_attributes(benchmark)
        return metadata

    def _resolve_custom_metrics(self, benchmark: BaseBenchmark) -> Optional[Dict[str, float]]:
        """Resolve benchmark-specific custom metrics if provided."""
        getter = getattr(benchmark, "get_custom_metrics", None)
        if callable(getter):
            try:
                metrics = getter()
                if isinstance(metrics, dict) and metrics:
                    numeric_metrics: Dict[str, float] = {}
                    for key, value in metrics.items():
                        if isinstance(value, bool):
                            numeric_metrics[key] = float(value)
                        elif isinstance(value, (int, float)):
                            numeric_metrics[key] = float(value)
                    if numeric_metrics:
                        return numeric_metrics
            except Exception as exc:  # pragma: no cover - defensive
                if LOGGER_AVAILABLE:
                    logger.debug(f"get_custom_metrics() raised: {exc}")
        return None

    def _resolve_story_metadata(self, benchmark: BaseBenchmark) -> Optional[Dict[str, Any]]:
        """Resolve benchmark story metadata if provided."""
        getter = getattr(benchmark, "get_story_metadata", None)
        if callable(getter):
            try:
                payload = getter()
                if isinstance(payload, dict) and payload:
                    return copy.deepcopy(payload)
            except Exception as exc:  # pragma: no cover - defensive
                if LOGGER_AVAILABLE:
                    logger.debug(f"get_story_metadata() raised: {exc}")
        return None

    def _infer_workload_metadata_from_attributes(self, benchmark: BaseBenchmark) -> Optional[WorkloadMetadata]:
        """Workload inference is disabled; benchmarks must register explicitly."""
        return None

    def _compute_throughput_stats(
        self,
        timing: Optional[TimingStats],
        workload: Optional[WorkloadMetadata],
    ) -> Optional[ThroughputStats]:
        """Convert timing + workload metadata into throughput statistics."""
        if timing is None or timing.mean_ms <= 0:
            return None
        mean_seconds = timing.mean_ms / 1000.0
        requests_per_iteration = 1.0
        tokens_per_iteration = None
        samples_per_iteration = None
        bytes_per_iteration = None
        custom_units_per_iteration = None
        custom_unit_name = None
        goodput = None
        
        if workload is not None:
            if workload.requests_per_iteration is not None:
                requests_per_iteration = max(float(workload.requests_per_iteration), 0.0)
            if workload.tokens_per_iteration is not None:
                tokens_per_iteration = float(workload.tokens_per_iteration)
            if workload.samples_per_iteration is not None:
                samples_per_iteration = float(workload.samples_per_iteration)
            if workload.bytes_per_iteration is not None:
                bytes_per_iteration = float(workload.bytes_per_iteration)
            if workload.custom_units_per_iteration is not None:
                custom_units_per_iteration = float(workload.custom_units_per_iteration)
            custom_unit_name = workload.custom_unit_name
            goodput = workload.goodput
        
        requests_per_s = requests_per_iteration / mean_seconds if mean_seconds > 0 else None
        tokens_per_s = (
            tokens_per_iteration / mean_seconds if tokens_per_iteration is not None and mean_seconds > 0 else None
        )
        samples_per_s = (
            samples_per_iteration / mean_seconds if samples_per_iteration is not None and mean_seconds > 0 else None
        )
        bytes_per_s = (
            bytes_per_iteration / mean_seconds if bytes_per_iteration is not None and mean_seconds > 0 else None
        )
        custom_unit_per_s = (
            custom_units_per_iteration / mean_seconds
            if custom_units_per_iteration is not None and mean_seconds > 0
            else None
        )
        
        return ThroughputStats(
            requests_per_s=requests_per_s,
            tokens_per_s=tokens_per_s,
            samples_per_s=samples_per_s,
            bytes_per_s=bytes_per_s,
            custom_unit_per_s=custom_unit_per_s,
            custom_unit_name=custom_unit_name,
            latency_ms=timing.mean_ms,
            goodput=goodput,
        )

    def _attach_throughput_metrics(self, result: PydanticBenchmarkResult, benchmark: BaseBenchmark) -> None:
        """Attach throughput metrics to the benchmark result when possible."""
        workload = self._resolve_workload_metadata(benchmark)
        throughput_stats = self._compute_throughput_stats(result.timing, workload)
        if throughput_stats:
            result.throughput = throughput_stats


def compare_benchmarks(
    baseline: BaseBenchmark,
    optimized: BaseBenchmark,
    harness: Optional[BenchmarkHarness] = None,
    name: str = "Comparison",
    regression_threshold_pct: float = 5.0
) -> Dict[str, Any]:
    """Compare baseline vs optimized benchmarks and return metrics.
    
    Args:
        baseline: Baseline benchmark instance
        optimized: Optimized benchmark instance
        harness: BenchmarkHarness instance (creates new if None)
        name: Name for the comparison
        regression_threshold_pct: Percentage degradation to consider a regression (default: 5%)
        
    Returns:
        Dictionary with comparison metrics including regression detection
    """
    if harness is None:
        harness = BenchmarkHarness()
    
    baseline_result = harness.benchmark(baseline)
    optimized_result = harness.benchmark(optimized)
    
    baseline_mean = baseline_result.timing.mean_ms if baseline_result.timing else 0.0
    optimized_mean = optimized_result.timing.mean_ms if optimized_result.timing else 0.0
    speedup = baseline_mean / optimized_mean if optimized_mean > 0 else 1.0
    
    # Detect regression: optimized is slower by threshold
    regression = False
    regression_pct = None
    if speedup < 1.0:
        regression_pct = (1.0 - speedup) * 100
        regression = regression_pct >= regression_threshold_pct
    
    return {
        "name": name,
        "baseline": {
            "mean_ms": baseline_result.timing.mean_ms if baseline_result.timing else 0.0,
            "median_ms": baseline_result.timing.median_ms if baseline_result.timing else 0.0,
            "std_ms": baseline_result.timing.std_ms if baseline_result.timing else 0.0,
            "min_ms": baseline_result.timing.min_ms if baseline_result.timing else 0.0,
            "max_ms": baseline_result.timing.max_ms if baseline_result.timing else 0.0,
        },
        "optimized": {
            "mean_ms": optimized_result.timing.mean_ms if optimized_result.timing else 0.0,
            "median_ms": optimized_result.timing.median_ms if optimized_result.timing else 0.0,
            "std_ms": optimized_result.timing.std_ms if optimized_result.timing else 0.0,
            "min_ms": optimized_result.timing.min_ms if optimized_result.timing else 0.0,
            "max_ms": optimized_result.timing.max_ms if optimized_result.timing else 0.0,
        },
        "speedup": speedup,
        "regression": regression,
        "regression_pct": regression_pct,
        "baseline_result": baseline_result,
        "optimized_result": optimized_result,
    }


def benchmark_main(
    get_benchmark_fn: Callable[[], 'BaseBenchmark'],
    iterations: int = 10,
    warmup: int = 5,
    name: Optional[str] = None,
    force_sync: Optional[bool] = None,
) -> None:
    """Safe helper for standalone scripts that need to run a benchmark factory.
    
    This function avoids CUDA initialization issues by NOT calling get_benchmark_fn()
    in the parent process when using subprocess mode. Instead, it:
    1. Checks if subprocess mode would be used
    2. If so, runs the benchmark directly without harness subprocess (single process)
    3. If not, uses the harness normally
    
    Usage in standalone scripts:
        if __name__ == "__main__":
            from core.harness.benchmark_harness import benchmark_main
            benchmark_main(get_benchmark)

    Discovery, compare.py, and ``cli.aisp bench run`` only require
    ``get_benchmark()``. Harness-discoverable benchmark modules should expose the
    factory and avoid defining ``__main__`` blocks so the supported execution
    path stays consistent across the repo.
    
    This prevents the common mistake of:
        if __name__ == "__main__":
            bench = get_benchmark()  # <-- This initializes CUDA!
            harness = BenchmarkHarness(...)
            harness.benchmark(bench)  # <-- Subprocess inherits corrupted CUDA
    
    Args:
        get_benchmark_fn: Callable that returns a benchmark instance
        iterations: Number of timed iterations
        warmup: Number of warmup iterations
        name: Optional name for output
        force_sync: If True, synchronize the whole device after each iteration.
            If None, honors CLI flags/env, otherwise defaults to True for ad-hoc
            runs (unless explicitly disabled).
    """
    # Run directly without harness to avoid CUDA subprocess issues
        # The harness subprocess mode is designed for run_benchmarks.py
    # which properly manages CUDA context by spawning fresh subprocesses
    
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    sync_group = parser.add_mutually_exclusive_group()
    sync_group.add_argument("--force-sync", action="store_true", default=None)
    sync_group.add_argument("--no-force-sync", action="store_true", default=None)
    cli_args, _ = parser.parse_known_args()

    if cli_args.iterations is not None:
        iterations = cli_args.iterations
    if cli_args.warmup is not None:
        warmup = cli_args.warmup

    if force_sync is None:
        if cli_args.force_sync:
            force_sync = True
        elif cli_args.no_force_sync:
            force_sync = False

    if force_sync is None:
        env_force_sync = os.getenv("AISP_FORCE_SYNC", "").strip().lower()
        if env_force_sync:
            if env_force_sync in {"1", "true", "yes", "on"}:
                force_sync = True
            elif env_force_sync in {"0", "false", "no", "off"}:
                force_sync = False
        if force_sync is None:
            force_sync = True
            warn_msg = (
                "WARNING: ad-hoc benchmark_main run detected; enabling device-wide "
                "synchronize after each iteration. Use --no-force-sync (or "
                "AISP_FORCE_SYNC=0) to disable."
            )
            print(warn_msg, file=sys.stderr)
            warnings.warn(warn_msg, RuntimeWarning, stacklevel=1)

    benchmark = get_benchmark_fn()
    bench_name = name or getattr(benchmark, 'name', None) or benchmark.__class__.__name__
    
    # Setup
    benchmark.setup()
    
    # Check for CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.synchronize()
    
    # Warmup
    for _ in range(warmup):
        benchmark.benchmark_fn()
        if cuda_available:
            torch.cuda.synchronize()
    
    # Check if benchmark reports its own timing (e.g., CudaBinaryBenchmark parses kernel time from output)
    use_reported_time = bool(getattr(benchmark, "use_reported_time", False))
    
    # Timed runs - use CUDA events for GPU timing (accurate), perf_counter for CPU only
    times_ms: List[float] = []
    if cuda_available:
        # GPU: use CUDA Events for accurate GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()  # sync once before timing loop
        
        for _ in range(iterations):
            start_event.record()
            benchmark.benchmark_fn()
            if force_sync:
                torch.cuda.synchronize()
            end_event.record()
            end_event.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            
            # Use benchmark-reported time if available (e.g., parsed from CUDA binary output)
            if use_reported_time:
                reported = getattr(benchmark, "last_time_ms", None)
                if reported is not None:
                    elapsed_ms = reported
            
            times_ms.append(elapsed_ms)
    else:
        # CPU: use perf_counter (only valid for CPU-only benchmarks)
        for _ in range(iterations):
            start = time.perf_counter()
            benchmark.benchmark_fn()
            elapsed_ms = (time.perf_counter() - start) * 1000
            times_ms.append(elapsed_ms)
    
    # Teardown
    benchmark.teardown()
    
    # Report results
    mean = sum(times_ms) / len(times_ms) if times_ms else 0.0
    std = (sum((t - mean) ** 2 for t in times_ms) / len(times_ms)) ** 0.5 if len(times_ms) > 1 else 0.0
    min_t = min(times_ms) if times_ms else 0.0
    max_t = max(times_ms) if times_ms else 0.0
    
    print(f"\n{bench_name}: {mean:.3f} ms/iter (std={std:.3f}, min={min_t:.3f}, max={max_t:.3f})")
