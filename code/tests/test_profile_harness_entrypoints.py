from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
from core.scripts.harness.example_registry import EXAMPLE_BY_NAME, _example
from core.scripts.harness import profile_harness


def test_example_run_command_prefers_module_name_for_python_examples() -> None:
    example = _example(
        name="ch15_tensor_parallel_demo",
        path="ch15/tensor_parallel_demo.py",
        module_name="ch15.tensor_parallel_demo",
        description="demo",
    )

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch15.tensor_parallel_demo"]


def test_base_env_includes_repo_root_on_pythonpath() -> None:
    example = _example(
        name="ch15_tensor_parallel_demo",
        path="ch15/tensor_parallel_demo.py",
        module_name="ch15.tensor_parallel_demo",
        description="demo",
        tags=["ch15"],
    )

    env = profile_harness.base_env(example)

    assert str(REPO_ROOT) in env["PYTHONPATH"].split(os.pathsep)


def test_run_pytorch_profiler_uses_shared_profiling_runner(tmp_path: Path) -> None:
    example = EXAMPLE_BY_NAME["ch13_train_deepseek_coder"]
    context = argparse.Namespace(dry_run=True)

    results = profile_harness.run_pytorch_profiler(
        example,
        session_dir=tmp_path,
        context=context,
        modes=["full"],
        timeout=60,
    )

    assert len(results) == 1
    assert "core/scripts/profiling/pytorch_profiler_runner.py" in results[0].command[1]


def test_profile_sh_zymtrace_tool_sets_cuda_injection_env(tmp_path: Path) -> None:
    target = tmp_path / "show_zymtrace_env.py"
    target.write_text(
        "import os\n"
        "print(os.environ.get('CUDA_INJECTION64_PATH', ''))\n",
        encoding="utf-8",
    )
    injection_lib = tmp_path / "libzymtracecudaprofiler.so"
    injection_lib.write_text("", encoding="utf-8")
    output_root = tmp_path / "profiles"
    env = os.environ.copy()
    env["OUTPUT_ROOT"] = str(output_root)
    env["PYTHON"] = sys.executable
    env["ZYMTRACE_CUDA_INJECTION64_PATH"] = str(injection_lib)

    proc = subprocess.run(
        [
            str(REPO_ROOT / "core" / "scripts" / "profiling" / "profile.sh"),
            str(target),
            "--tool",
            "zymtrace",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert str(injection_lib) in proc.stdout
    manifests = list(output_root.glob("*_show_zymtrace_env/zymtrace_cuda_env.txt"))
    assert manifests
    assert f"CUDA_INJECTION64_PATH={injection_lib}" in manifests[-1].read_text(encoding="utf-8")


def test_ch20_example_registry_uses_module_launch_for_ai_kernel_generator() -> None:
    example = EXAMPLE_BY_NAME["ch20_ai_kernel_generator"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch20.ai_kernel_generator"]


def test_ch05_example_registry_uses_module_launch_for_storage_demo() -> None:
    example = EXAMPLE_BY_NAME["ch05_storage_io_optimization"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch05.storage_io_optimization"]


def test_ch06_example_registry_uses_module_launch_for_add_demo() -> None:
    example = EXAMPLE_BY_NAME["ch06_add_parallel"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch06.optimized_add"]


def test_ch07_example_registry_uses_module_launch_for_memory_access_demo() -> None:
    example = EXAMPLE_BY_NAME["ch07_memory_access"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch07.memory_access_pytorch"]


def test_ch08_example_registry_uses_module_launch_for_occupancy_demo() -> None:
    example = EXAMPLE_BY_NAME["ch08_occupancy"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch08.occupancy_pytorch"]


def test_ch09_example_registry_uses_module_launch_for_fusion_demo() -> None:
    example = EXAMPLE_BY_NAME["ch09_fusion"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch09.fusion_pytorch"]


def test_ch13_example_registry_uses_module_launch_for_custom_allocator_demo() -> None:
    example = EXAMPLE_BY_NAME["ch13_custom_allocator"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch13.custom_allocator"]


def test_ch14_example_registry_uses_module_launch_for_torch_compiler_demo() -> None:
    example = EXAMPLE_BY_NAME["ch14_torch_compiler_examples"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch14.torch_compiler_examples"]


def test_ch18_example_registry_uses_module_launch_for_flexdecoding_demo() -> None:
    example = EXAMPLE_BY_NAME["ch18_flexdecoding"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch18.baseline_flexdecoding"]


def test_ch19_example_registry_uses_module_launch_for_adaptive_parallelism_demo() -> None:
    example = EXAMPLE_BY_NAME["ch19_adaptive_parallelism"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch19.adaptive_parallelism_strategy"]


def test_ch19_example_registry_uses_module_launch_for_dynamic_precision_utilities() -> None:
    example = EXAMPLE_BY_NAME["ch19_dynamic_precision"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch19.dynamic_precision_switching"]


def test_ch19_example_registry_uses_module_launch_for_dynamic_quantized_cache_demo() -> None:
    example = EXAMPLE_BY_NAME["ch19_dynamic_quantized_cache"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch19.dynamic_quantized_cache"]


def test_ch01_example_registry_uses_module_launch_for_performance_basics() -> None:
    example = EXAMPLE_BY_NAME["ch01_performance_basics"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch01.baseline_performance"]


def test_ch02_example_registry_uses_module_launch_for_hardware_info() -> None:
    example = EXAMPLE_BY_NAME["ch02_hardware_info"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch02.hardware_info"]


def test_ch03_example_registry_uses_module_launch_for_bind_numa_affinity() -> None:
    example = EXAMPLE_BY_NAME["ch03_bind_numa_affinity"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch03.bind_numa_affinity"]


def test_tool_memory_profiler_example_registry_uses_module_launch() -> None:
    example = EXAMPLE_BY_NAME["tool_memory_profiler"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "core.profiling.memory_profiler"]
