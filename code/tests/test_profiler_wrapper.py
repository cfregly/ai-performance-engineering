from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from core.harness.benchmark_harness import BenchmarkConfig
from core.profiling.profiler_wrapper import (
    _resolve_wrapper_loop_budget,
    render_ncu_python_profile_wrapper,
    render_nsys_python_profile_wrapper,
    render_torch_python_profile_wrapper,
    temporary_python_profile_wrapper,
)


def test_wrapper_loop_budget_defaults_to_existing_benchmark_counts() -> None:
    config = BenchmarkConfig(iterations=20, warmup=5)
    assert _resolve_wrapper_loop_budget(config) == (5, 10)


def test_wrapper_loop_budget_honors_profiling_specific_overrides() -> None:
    config = BenchmarkConfig(
        iterations=20,
        warmup=5,
        profiling_warmup=0,
        profiling_iterations=1,
    )
    assert _resolve_wrapper_loop_budget(config) == (0, 1)


def test_temporary_python_profile_wrapper_cleans_up_file() -> None:
    wrapper_path: Path | None = None

    with temporary_python_profile_wrapper("print('ok')\n") as created_path:
        wrapper_path = created_path
        assert wrapper_path.exists()
        assert wrapper_path.read_text(encoding="utf-8") == "print('ok')\n"

    assert wrapper_path is not None
    assert not wrapper_path.exists()


def test_render_nsys_wrapper_contains_expected_config() -> None:
    wrapper = render_nsys_python_profile_wrapper(
        benchmark_path=Path("/tmp/example.py"),
        nvtx_includes=["compute_kernel:profile/"],
        target_label="labs/moe_cuda_ptx:moe_layer",
        target_override_argv=["--mode", "fwd_bwd"],
        validity_profile="strict",
        lock_gpu_clocks_flag=True,
        gpu_sm_clock_mhz=1500,
        gpu_mem_clock_mhz=2000,
    )

    assert 'Path(r"/tmp/example.py")' in wrapper
    assert "nsys_nvtx_include=['compute_kernel:profile/']" in wrapper
    assert "validity_profile='strict'" in wrapper
    assert "gpu_sm_clock_mhz=1500" in wrapper
    assert "_target_label = 'labs/moe_cuda_ptx:moe_layer'" in wrapper
    assert "_target_override_argv = ['--mode', 'fwd_bwd']" in wrapper
    assert "_apply_overrides(list(_target_override_argv))" in wrapper
    assert "target_extra_args={_target_label: list(_target_override_argv)}" in wrapper
    assert 'with nvtx_range("compute_kernel:profile", enable=True):' in wrapper
    assert 'if getattr(benchmark, "profile_require_teardown", False):' in wrapper
    assert "_os._exit(0)" in wrapper
    assert "raise SystemExit(0)" not in wrapper


def test_render_ncu_wrapper_contains_expected_config() -> None:
    wrapper = render_ncu_python_profile_wrapper(
        benchmark_path=Path("/tmp/example.py"),
        configured_nvtx_includes=["capture/"],
        target_label="labs/moe_cuda_ptx:moe_layer",
        target_override_argv=["--mode", "fwd_bwd"],
        profile_type="minimal",
        ncu_metric_set="minimal",
        pm_sampling_interval=1234,
        ncu_replay_mode="kernel",
        validity_profile="strict",
        lock_gpu_clocks_flag=True,
        gpu_sm_clock_mhz=1500,
        gpu_mem_clock_mhz=2000,
        profile_nvtx_label="capture",
    )

    assert "enable_ncu=True" in wrapper
    assert "ncu_metric_set='minimal'" in wrapper
    assert "pm_sampling_interval=1234" in wrapper
    assert "ncu_replay_mode='kernel'" in wrapper
    assert "_target_label = 'labs/moe_cuda_ptx:moe_layer'" in wrapper
    assert "_target_override_argv = ['--mode', 'fwd_bwd']" in wrapper
    assert "_apply_overrides(list(_target_override_argv))" in wrapper
    assert "with nvtx_range('capture', enable=True):" in wrapper
    assert 'if getattr(benchmark, "profile_require_teardown", False):' in wrapper
    assert "_os._exit(0)" in wrapper
    assert "raise SystemExit(0)" not in wrapper


def test_render_torch_wrapper_contains_expected_output_path() -> None:
    wrapper = render_torch_python_profile_wrapper(
        benchmark_path=Path("/tmp/example.py"),
        torch_output=Path("/tmp/trace.json"),
        target_label="labs/moe_cuda_ptx:moe_layer",
        target_override_argv=["--mode", "fwd_bwd"],
        validity_profile="portable",
        lock_gpu_clocks_flag=False,
        gpu_sm_clock_mhz=None,
        gpu_mem_clock_mhz=None,
    )

    assert 'Path(r"/tmp/example.py")' in wrapper
    assert "validity_profile='portable'" in wrapper
    assert "_target_label = 'labs/moe_cuda_ptx:moe_layer'" in wrapper
    assert "_target_override_argv = ['--mode', 'fwd_bwd']" in wrapper
    assert "_apply_overrides(list(_target_override_argv))" in wrapper
    assert 'prof.export_chrome_trace(r"/tmp/trace.json")' in wrapper


def test_profile_shell_zymtrace_sets_cuda_injection_and_manifest(tmp_path: Path) -> None:
    profile_sh = Path("core/scripts/profiling/profile.sh").resolve()
    injection_lib = tmp_path / "libzymtracecudaprofiler.so"
    injection_lib.write_text("fake", encoding="utf-8")
    marker_path = tmp_path / "env_marker.json"
    output_root = tmp_path / "profiles"
    workload = tmp_path / "workload.py"
    workload.write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "from pathlib import Path",
                "Path(os.environ['MARKER_PATH']).write_text(json.dumps({",
                "    'cuda': os.environ.get('CUDA_INJECTION64_PATH'),",
                "    'zymtrace': os.environ.get('ZYMTRACE_CUDA_INJECTION64_PATH'),",
                "}), encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "PYTHON": sys.executable,
            "PYTHONPATH": str(Path.cwd()),
            "MARKER_PATH": str(marker_path),
            "ZYMTRACE_CUDA_INJECTION64_PATH": str(injection_lib),
        }
    )
    env.pop("CUDA_INJECTION64_PATH", None)

    result = subprocess.run(
        [
            "bash",
            str(profile_sh),
            str(workload),
            "--arch",
            "sm_100",
            "--tool",
            "zymtrace",
            "--output-root",
            str(output_root),
            "--python",
            sys.executable,
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["cuda"] == str(injection_lib.resolve())
    assert marker["zymtrace"] == str(injection_lib.resolve())

    manifests = list(output_root.glob("*/zymtrace_launch_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["tool"] == "zymtrace"
    assert manifest["cuda_injection64_path"] == str(injection_lib.resolve())
    assert manifest["script"] == str(workload.resolve())


def test_profile_shell_zymtrace_fails_for_explicit_missing_injection(tmp_path: Path) -> None:
    profile_sh = Path("core/scripts/profiling/profile.sh").resolve()
    workload = tmp_path / "workload.py"
    workload.write_text("print('should not run')\n", encoding="utf-8")
    missing_lib = tmp_path / "missing-libzymtrace.so"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON": sys.executable,
            "PYTHONPATH": str(Path.cwd()),
            "CUDA_INJECTION64_PATH": str(missing_lib),
        }
    )
    env.pop("ZYMTRACE_CUDA_INJECTION64_PATH", None)

    result = subprocess.run(
        [
            "bash",
            str(profile_sh),
            str(workload),
            "--arch",
            "sm_100",
            "--tool",
            "zymtrace",
            "--output-root",
            str(tmp_path / "profiles"),
            "--python",
            sys.executable,
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "CUDA_INJECTION64_PATH is set but does not point to a file" in result.stderr
