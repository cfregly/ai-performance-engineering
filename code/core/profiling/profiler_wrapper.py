"""Profiler wrapper script generation.

Generates wrapper scripts for nsys/ncu profiling that import and run benchmarks.
"""

from __future__ import annotations

from contextlib import contextmanager
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
else:
    BaseBenchmark = Any  # type: ignore[assignment,misc]
    BenchmarkConfig = Any  # type: ignore[assignment,misc]


def _resolve_wrapper_loop_budget(config: BenchmarkConfig) -> tuple[int, int]:
    """Resolve warmup and profiled iteration counts for wrapper-based captures."""

    profiling_warmup = getattr(config, "profiling_warmup", None)
    if profiling_warmup is None:
        profiling_warmup = getattr(config, "warmup", 0)
    profiling_iterations = getattr(config, "profiling_iterations", None)
    if profiling_iterations is None:
        profiling_iterations = min(getattr(config, "iterations", 1), 10)

    return max(int(profiling_warmup), 0), max(int(profiling_iterations), 1)


@contextmanager
def temporary_python_profile_wrapper(wrapper_source: str) -> Iterator[Path]:
    """Materialize a temporary Python wrapper script and clean it up on exit."""

    wrapper_script = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        dir=tempfile.gettempdir(),
    )
    wrapper_path = Path(wrapper_script.name)
    try:
        wrapper_script.write(wrapper_source)
        wrapper_script.close()
        yield wrapper_path
    finally:
        wrapper_path.unlink(missing_ok=True)


def render_nsys_python_profile_wrapper(
    *,
    benchmark_path: Path,
    nvtx_includes: Optional[list[str]],
    target_label: Optional[str],
    target_override_argv: Optional[list[str]],
    validity_profile: str,
    lock_gpu_clocks_flag: bool,
    gpu_sm_clock_mhz: Optional[int],
    gpu_mem_clock_mhz: Optional[int],
) -> str:
    """Render the nsys-specific Python benchmark wrapper."""

    return f"""
from pathlib import Path
from contextlib import nullcontext

_BENCHMARK_PATH = Path(r"{benchmark_path}")
from core.utils.python_entrypoints import load_module_from_path

_BENCHMARK_MODULE = load_module_from_path("profile_benchmark_module", _BENCHMARK_PATH)
get_benchmark = _BENCHMARK_MODULE.get_benchmark

from core.harness.benchmark_harness import (
    BenchmarkConfig,
    ReadOnlyBenchmarkConfigView,
    lock_gpu_clocks,
    ramp_gpu_clocks,
)
from core.profiling.nvtx_helper import nvtx_range

def _run_profile() -> None:
    import sys
    benchmark = get_benchmark()
    _target_label = {target_label!r}
    _target_override_argv = {target_override_argv!r}
    if _target_override_argv:
        _apply_overrides = getattr(benchmark, "apply_target_overrides", None)
        if callable(_apply_overrides):
            _apply_overrides(list(_target_override_argv))
    _profiling_config = BenchmarkConfig(
        enable_profiling=True,
        enable_nvtx=True,
        nsys_nvtx_include={nvtx_includes!r},
        target_label=_target_label,
        target_extra_args={{_target_label: list(_target_override_argv)}} if _target_label and _target_override_argv else {{}},
        validity_profile={validity_profile!r},
        lock_gpu_clocks={lock_gpu_clocks_flag!r},
        gpu_sm_clock_mhz={gpu_sm_clock_mhz!r},
        gpu_mem_clock_mhz={gpu_mem_clock_mhz!r},
    )
    benchmark._config = ReadOnlyBenchmarkConfigView.from_config(_profiling_config)
    lock_ctx = (
        lock_gpu_clocks(
            device=0,
            sm_clock_mhz=getattr(_profiling_config, "gpu_sm_clock_mhz", None),
            mem_clock_mhz=getattr(_profiling_config, "gpu_mem_clock_mhz", None),
        )
        if getattr(_profiling_config, "lock_gpu_clocks", False)
        else nullcontext()
    )
    with lock_ctx:
        # Best-effort clock ramp before capture.
        try:
            ramp_gpu_clocks(device=0)
        except Exception as exc:
            print(f"[profile_warning] Failed to ramp GPU clocks before nsys capture: {{exc}}", file=sys.stderr)
        benchmark.setup()

        # Warmup (keep short; profiling is not a timing run)
        benchmark.benchmark_fn()

        # Profile exactly one execution. Nsight Systems defers CUDA activity
        # buffer flushing until cudaProfilerStop(); calling it explicitly is
        # required before the successful hard exit below.
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        _cudart = None
        _profiler_started = False
        _profile_error = None
        try:
            if torch.cuda.is_available():
                _cudart = torch.cuda.cudart()
                _start_status = int(_cudart.cudaProfilerStart())
                if _start_status != 0:
                    raise RuntimeError(
                        f"cudaProfilerStart() failed with status {{_start_status}}"
                    )
                _profiler_started = True

            with nvtx_range("compute_kernel:profile", enable=True):
                benchmark.benchmark_fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except BaseException as exc:
            _profile_error = exc
            raise
        finally:
            if _profiler_started and _cudart is not None:
                if _profile_error is not None:
                    try:
                        torch.cuda.synchronize()
                    except BaseException as exc:
                        print(
                            f"[profile_warning] CUDA synchronize failed while preserving "
                            f"the primary nsys capture error: {{exc}}",
                            file=sys.stderr,
                        )
                try:
                    _stop_status = int(_cudart.cudaProfilerStop())
                    if _stop_status != 0:
                        raise RuntimeError(
                            f"cudaProfilerStop() failed with status {{_stop_status}}"
                        )
                except BaseException as exc:
                    if _profile_error is None:
                        raise
                    print(
                        f"[profile_warning] cudaProfilerStop() failed while preserving "
                        f"the primary nsys capture error: {{exc}}",
                        file=sys.stderr,
                    )

        # Some benchmarks launch worker processes (e.g., vLLM) and need
        # graceful teardown so Nsight can exit cleanly. Others prefer hard-exit
        # to avoid teardown crashes under Nsight.
        if getattr(benchmark, "profile_require_teardown", False):
            try:
                benchmark.teardown()
            except Exception as exc:
                print(f"[profile_warning] Benchmark teardown failed after nsys capture: {{exc}}", file=sys.stderr)
            import os as _os
            _os._exit(0)
        import os as _os
        _os._exit(0)


if __name__ == "__main__":
    _run_profile()
"""


def render_ncu_python_profile_wrapper(
    *,
    benchmark_path: Path,
    configured_nvtx_includes: Optional[list[str]],
    target_label: Optional[str],
    target_override_argv: Optional[list[str]],
    profile_type: Optional[str],
    ncu_metric_set: Optional[str],
    pm_sampling_interval: Optional[int],
    ncu_replay_mode: Optional[str],
    validity_profile: str,
    lock_gpu_clocks_flag: bool,
    gpu_sm_clock_mhz: Optional[int],
    gpu_mem_clock_mhz: Optional[int],
    profile_nvtx_label: str,
) -> str:
    """Render the ncu-specific Python benchmark wrapper."""

    return f"""
from pathlib import Path
from contextlib import nullcontext

_BENCHMARK_PATH = Path(r"{benchmark_path}")
from core.utils.python_entrypoints import load_module_from_path

_BENCHMARK_MODULE = load_module_from_path("profile_benchmark_module", _BENCHMARK_PATH)
get_benchmark = _BENCHMARK_MODULE.get_benchmark

from core.harness.benchmark_harness import (
    BenchmarkConfig,
    ReadOnlyBenchmarkConfigView,
    lock_gpu_clocks,
    ramp_gpu_clocks,
)
from core.profiling.nvtx_helper import nvtx_range

def _run_profile() -> None:
    import sys
    benchmark = get_benchmark()
    _target_label = {target_label!r}
    _target_override_argv = {target_override_argv!r}
    if _target_override_argv:
        _apply_overrides = getattr(benchmark, "apply_target_overrides", None)
        if callable(_apply_overrides):
            _apply_overrides(list(_target_override_argv))
    _profiling_config = BenchmarkConfig(
        enable_profiling=True,
        enable_nsys=True,
        enable_ncu=True,
        enable_nvtx=True,
        nsys_nvtx_include={configured_nvtx_includes!r},
        target_label=_target_label,
        target_extra_args={{_target_label: list(_target_override_argv)}} if _target_label and _target_override_argv else {{}},
        profile_type={profile_type!r},
        ncu_metric_set={ncu_metric_set!r},
        pm_sampling_interval={pm_sampling_interval!r},
        ncu_replay_mode={ncu_replay_mode!r},
        validity_profile={validity_profile!r},
        lock_gpu_clocks={lock_gpu_clocks_flag!r},
        gpu_sm_clock_mhz={gpu_sm_clock_mhz!r},
        gpu_mem_clock_mhz={gpu_mem_clock_mhz!r},
    )
    benchmark._config = ReadOnlyBenchmarkConfigView.from_config(_profiling_config)
    lock_ctx = (
        lock_gpu_clocks(
            device=0,
            sm_clock_mhz=getattr(_profiling_config, "gpu_sm_clock_mhz", None),
            mem_clock_mhz=getattr(_profiling_config, "gpu_mem_clock_mhz", None),
        )
        if getattr(_profiling_config, "lock_gpu_clocks", False)
        else nullcontext()
    )
    with lock_ctx:
        try:
            ramp_gpu_clocks(device=0)
        except Exception as exc:
            print(f"[profile_warning] Failed to ramp GPU clocks before ncu capture: {{exc}}", file=sys.stderr)
        benchmark.setup()

        # Warmup (keep short; profiling is not a timing run)
        benchmark.benchmark_fn()

        # Profile execution
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with nvtx_range({profile_nvtx_label!r}, enable=True):
            benchmark.benchmark_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Some benchmarks launch worker processes (e.g., vLLM) and need
        # graceful teardown so Nsight can exit cleanly. Others prefer hard-exit
        # to avoid teardown crashes under Nsight.
        if getattr(benchmark, "profile_require_teardown", False):
            try:
                benchmark.teardown()
            except Exception as exc:
                print(f"[profile_warning] Benchmark teardown failed after ncu capture: {{exc}}", file=sys.stderr)
            import os as _os
            _os._exit(0)
        import os as _os
        _os._exit(0)


if __name__ == "__main__":
    _run_profile()
"""


def render_torch_python_profile_wrapper(
    *,
    benchmark_path: Path,
    torch_output: Path,
    target_label: Optional[str],
    target_override_argv: Optional[list[str]],
    validity_profile: str,
    lock_gpu_clocks_flag: bool,
    gpu_sm_clock_mhz: Optional[int],
    gpu_mem_clock_mhz: Optional[int],
) -> str:
    """Render the torch.profiler-specific Python benchmark wrapper."""

    return f"""
from pathlib import Path
from contextlib import nullcontext

_BENCHMARK_PATH = Path(r"{benchmark_path}")
from core.utils.python_entrypoints import load_module_from_path

_BENCHMARK_MODULE = load_module_from_path("profile_benchmark_module", _BENCHMARK_PATH)
get_benchmark = _BENCHMARK_MODULE.get_benchmark
from core.harness.benchmark_harness import (
    BenchmarkConfig,
    ReadOnlyBenchmarkConfigView,
    lock_gpu_clocks,
    ramp_gpu_clocks,
)
import torch
import torch.profiler


def _run_profile() -> None:
    benchmark = get_benchmark()
    _target_label = {target_label!r}
    _target_override_argv = {target_override_argv!r}
    if _target_override_argv:
        _apply_overrides = getattr(benchmark, "apply_target_overrides", None)
        if callable(_apply_overrides):
            _apply_overrides(list(_target_override_argv))
    profiling_config = BenchmarkConfig(
        enable_profiling=True,
        enable_nvtx=True,
        target_label=_target_label,
        target_extra_args={{_target_label: list(_target_override_argv)}} if _target_label and _target_override_argv else {{}},
        validity_profile={validity_profile!r},
        lock_gpu_clocks={lock_gpu_clocks_flag!r},
        gpu_sm_clock_mhz={gpu_sm_clock_mhz!r},
        gpu_mem_clock_mhz={gpu_mem_clock_mhz!r},
    )
    benchmark._config = ReadOnlyBenchmarkConfigView.from_config(profiling_config)

    lock_ctx = (
        lock_gpu_clocks(
            device=0,
            sm_clock_mhz=getattr(profiling_config, "gpu_sm_clock_mhz", None),
            mem_clock_mhz=getattr(profiling_config, "gpu_mem_clock_mhz", None),
        )
        if getattr(profiling_config, "lock_gpu_clocks", False) and torch.cuda.is_available()
        else nullcontext()
    )

    with lock_ctx:
        if torch.cuda.is_available():
            ramp_gpu_clocks(device=0)
        benchmark.setup()
        benchmark.benchmark_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
        ) as prof:
            benchmark.benchmark_fn()
            prof.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        benchmark.teardown()
    prof.export_chrome_trace(r"{torch_output}")
    print(r"{torch_output}")


if __name__ == "__main__":
    _run_profile()
"""


def create_benchmark_wrapper(
    benchmark: BaseBenchmark,
    benchmark_module,
    benchmark_class: str,
    config: BenchmarkConfig
) -> Optional[Path]:
    """Create a temporary Python script that runs the benchmark.
    
    The wrapper script imports the benchmark module and recreates the benchmark
    instance, then runs setup, warmup, and profiling iterations.
    
    Args:
        benchmark: BaseBenchmark instance (used to get module info)
        benchmark_module: Module object containing the benchmark
        benchmark_class: Name of the benchmark class
        config: BenchmarkConfig with iterations/warmup settings
    
    Returns:
        Path to created wrapper script, or None if creation failed
    """
    try:
        # Get module path
        if benchmark_module is None:
            return None
        
        module_name = benchmark_module.__name__
        module_file = getattr(benchmark_module, "__file__", None)
        
        # Try to get file from spec if __file__ is not available
        if module_file is None:
            spec = getattr(benchmark_module, "__spec__", None)
            if spec is not None:
                module_file = getattr(spec, "origin", None)
        
        if module_file is None:
            return None
        
        module_path = Path(module_file).resolve()
        if not module_path.exists():
            return None

        profiling_warmup, profiling_iterations = _resolve_wrapper_loop_budget(config)
        
        # Create temporary wrapper script
        wrapper_script = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, dir=tempfile.gettempdir()
        )
        
        # Determine how to instantiate the benchmark
        nvtx_includes = getattr(config, "nsys_nvtx_include", None)
        enable_profiling = getattr(config, "enable_profiling", False)
        enable_nvtx = getattr(config, "enable_nvtx", None)
        instantiation_code = f"""# Get benchmark instance (try common patterns)
benchmark = None
try:
    if hasattr(_benchmark_module, "get_benchmark"):
        benchmark = _benchmark_module.get_benchmark()
    elif hasattr(_benchmark_module, "{benchmark_class}"):
        benchmark_class = getattr(_benchmark_module, "{benchmark_class}")
        benchmark = benchmark_class()
    else:
        # Try to find any class with benchmark_fn method
        for attr_name in dir(_benchmark_module):
            attr = getattr(_benchmark_module, attr_name)
            if isinstance(attr, type) and hasattr(attr, "benchmark_fn") and callable(getattr(attr, "benchmark_fn", None)):
                benchmark = attr()
                break
except Exception as e:
    import traceback
    print("Error creating benchmark: " + str(e))
    traceback.print_exc()
    raise

if benchmark is None:
    raise RuntimeError("Could not find or instantiate benchmark instance")

# Attach profiling config so NVTX ranges are emitted in wrapper runs.
from core.harness.benchmark_harness import BenchmarkConfig, ReadOnlyBenchmarkConfigView
_profiling_config = BenchmarkConfig(
    enable_profiling={enable_profiling!r},
    enable_nvtx={enable_nvtx!r},
    nsys_nvtx_include={nvtx_includes!r},
)
benchmark._config = ReadOnlyBenchmarkConfigView.from_config(_profiling_config)
"""
        
        wrapper_content = f'''import importlib.util
from pathlib import Path
from core.utils.python_entrypoints import load_module_from_path

_MODULE_PATH = Path(r"{module_path}")
_benchmark_module = load_module_from_path("{module_name}", _MODULE_PATH)

{instantiation_code}

# Run benchmark
try:
    benchmark.setup()
    
    # Warmup
    for _ in range({profiling_warmup}):
        benchmark.benchmark_fn()
    
    # Synchronize before profiling
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Run benchmark iterations for profiling (limited for profiling overhead)
    for _ in range({profiling_iterations}):
        benchmark.benchmark_fn()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    benchmark.teardown()
except Exception as e:
    import traceback
    print("Error running benchmark: " + str(e))
    traceback.print_exc()
    raise
'''

        wrapper_script.write(wrapper_content)
        wrapper_script.close()
        
        return Path(wrapper_script.name)
    except Exception:
        return None


def create_cuda_wrapper(
    cuda_executable: str,
    args: list[str],
    config: Optional[BenchmarkConfig] = None
) -> Optional[Path]:
    """Create a wrapper script for CUDA executables.
    
    Args:
        cuda_executable: Path to CUDA executable
        args: Command-line arguments for the executable
        config: Optional BenchmarkConfig (currently unused, for future extensibility)
    
    Returns:
        Path to created wrapper script, or None if creation failed
    """
    try:
        wrapper_script = tempfile.NamedTemporaryFile(
            mode='w', suffix='.sh', delete=False, dir=tempfile.gettempdir()
        )
        
        wrapper_content = f'''#!/bin/bash
# Wrapper for CUDA executable profiling

exec "{cuda_executable}" {" ".join(args)}
'''
        
        wrapper_script.write(wrapper_content)
        wrapper_script.close()
        
        # Make executable
        wrapper_path = Path(wrapper_script.name)
        wrapper_path.chmod(0o755)
        
        return wrapper_path
    except Exception:
        return None
