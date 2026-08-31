"""Real build-tool orchestration checks; no CUDA compiler or GPU is simulated."""

from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
ALIASES = {"b200": "sm_100", "b300": "sm_103", "gb300": "sm_103", "gb10": "sm_121"}
CHAPTER_ALIASES = [(f"ch{chapter:02d}", alias) for chapter in range(1, 21) for alias in ALIASES]
LAB_ALIASES = [("labs/nvfp4_gemm", "b200"), ("labs/ozaki_scheme", "b200")]
GENCODE = {"sm_100": "compute_100a", "sm_103": "compute_103a", "sm_120": "compute_120", "sm_121": "compute_121"}


def make_dry_run(directory: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    # PYTHON=false prevents dependence on a visible GPU or an installed torch.
    # GNU make propagates -n into its recursive calls: no clean/build executes.
    env = os.environ.copy()
    env.pop("ARCH", None)
    return subprocess.run(
        ["make", "--no-print-directory", "-B", "-n", "PYTHON=/usr/bin/false", *arguments],
        cwd=CODE_ROOT / directory, env=env, capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize(("directory", "alias"), CHAPTER_ALIASES + LAB_ALIASES)
def test_hardware_alias_selects_architecture_without_a_visible_gpu(directory: str, alias: str) -> None:
    result = make_dry_run(directory, alias)
    assert result.returncode == 0, result.stdout + result.stderr
    architecture = ALIASES[alias]
    assert f"ARCH={architecture}" in result.stdout
    if directory not in {"ch03", "ch05", "ch13", "ch14", "ch15", "ch17"}:
        assert GENCODE[architecture] in result.stdout
        assert "_" + architecture.replace("_", "") in result.stdout
    else:
        assert "No CUDA binaries to build" in result.stdout


def test_alias_overrides_a_conflicting_parent_architecture() -> None:
    result = make_dry_run("ch01", "ARCH=sm_100", "b300")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "compute_103a" in result.stdout
    assert "compute_100a" not in result.stdout


@pytest.mark.parametrize("goals", [("b200", "b300"), ("b200", "all")])
def test_alias_rejects_goals_that_can_race_its_clean(goals: tuple[str, ...]) -> None:
    result = make_dry_run("ch01", *goals)
    assert result.returncode != 0
    assert "Run a hardware alias by itself" in result.stderr


def test_unspecified_real_build_still_requires_architecture_detection() -> None:
    result = make_dry_run("ch01", "all")
    assert result.returncode != 0
    assert "Unable to auto-detect" in result.stderr


@pytest.mark.parametrize("architecture", tuple(GENCODE))
def test_custom_cublas_make_uses_the_requested_architecture(architecture: str) -> None:
    result = make_dry_run("labs/custom_vs_cublas", f"ARCH={architecture}", "all")
    assert result.returncode == 0, result.stdout + result.stderr
    assert GENCODE[architecture] in result.stdout
    for other_arch, other_gencode in GENCODE.items():
        if other_arch != architecture:
            assert other_gencode not in result.stdout


@pytest.mark.parametrize("architecture", ("sm_120a", "sm_122", "native"))
def test_custom_cublas_rejects_unsupported_make_targets(architecture: str) -> None:
    result = make_dry_run("labs/custom_vs_cublas", f"ARCH={architecture}", "all")
    assert result.returncode != 0
    assert "Unsupported" in result.stderr


@pytest.mark.parametrize(
    ("major", "minor", "name"),
    [(10, 0, "B200/GB200"), (10, 3, "B300/GB300"), (12, 0, "RTX"), (12, 1, "GB10")],
)
def test_reported_architecture_metadata_keeps_the_actual_compute_capability(
    major: int, minor: int, name: str,
) -> None:
    detector = runpy.run_path(str(CODE_ROOT / "core/benchmark/detect_sm.py"))
    metadata = detector["get_arch_spec"](major, minor)
    assert metadata["compute_capability"] == f"{major}.{minor}"
    assert name in metadata["label"]
    assert metadata["sm"] == f"sm_{major}{minor}"
    assert metadata["sm_count"] is None  # Compute capability alone does not identify an SKU.


@pytest.mark.parametrize("architecture", ("sm_100", "sm_103", "sm_120", "sm_121"))
def test_tma_build_helper_dry_run_uses_existing_suffixed_targets(architecture: str) -> None:
    result = subprocess.run(
        ["bash", str(CODE_ROOT / "core/scripts/build_tma_demos.sh"), "--arch", architecture, "--dry-run"],
        cwd=CODE_ROOT.parent, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    suffix = architecture.replace("_", "")
    assert f"async_prefetch_2d_demo_{suffix}" in result.stdout
    assert f"tma_2d_pipeline_blackwell_{suffix}" in result.stdout
    assert "Dry run complete; no binaries were built" in result.stdout


def test_warp_ci_wrapper_reaches_real_help_from_outside_code() -> None:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    result = subprocess.run(
        ["bash", str(CODE_ROOT / "core/scripts/run_warp_specialization_ci.sh"), "--help"],
        cwd=CODE_ROOT.parent, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Compare warp specialization benchmarks" in result.stdout


def cmake_policy(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for the real configuration policy")
    script = tmp_path / "policy.cmake"
    script.write_text(
        f'include("{CODE_ROOT / "labs/custom_vs_cublas/cutlass_gemm/BlackwellBuildConfig.cmake"}")\n'
        'aisp_blackwell_architectures()\n'
        'message(STATUS "ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")\n'
        'if(DEFINED PROBE_PYTHON)\n'
        '  aisp_query_torch("${PROBE_PYTHON}")\n'
        '  message(STATUS "TORCH_ABI=${AISP_TORCH_CXX11_ABI}")\n'
        '  message(STATUS "TORCH_VERSION=${AISP_TORCH_VERSION}")\n'
        'endif()\n', encoding="utf-8",
    )
    return subprocess.run([cmake, *arguments, "-P", str(script)], capture_output=True, text=True, check=False)


@pytest.mark.parametrize("architecture", ("100a", "103a", "100a;103a"))
def test_cmake_architecture_policy_preserves_supported_override(tmp_path: Path, architecture: str) -> None:
    result = cmake_policy(tmp_path, f"-DCMAKE_CUDA_ARCHITECTURES={architecture}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"ARCHITECTURES={architecture}" in result.stdout


@pytest.mark.parametrize("architecture", ("100", "103", "120", "120a", "native", "OFF", ""))
def test_cmake_rejects_non_tcgen_architectures(tmp_path: Path, architecture: str) -> None:
    result = cmake_policy(tmp_path, f"-DCMAKE_CUDA_ARCHITECTURES={architecture}")
    assert result.returncode != 0
    assert "requires 100a and/or 103a" in result.stderr


def test_cmake_reads_abi_from_the_actual_selected_torch(tmp_path: Path) -> None:
    import torch
    result = cmake_policy(tmp_path, f"-DPROBE_PYTHON={sys.executable}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"TORCH_ABI={int(torch.compiled_with_cxx11_abi())}" in result.stdout
    assert f"TORCH_VERSION={torch.__version__}" in result.stdout


def test_cmake_rejects_failed_torch_metadata_query(tmp_path: Path) -> None:
    result = cmake_policy(tmp_path, "-DPROBE_PYTHON=/usr/bin/false")
    assert result.returncode != 0
    assert "Unable to query the selected Python's torch" in result.stderr


def test_real_cmake_project_rejects_plain_arch_before_cuda_detection(tmp_path: Path) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for the real configuration policy")
    result = subprocess.run(
        [cmake, "-S", str(CODE_ROOT / "labs/custom_vs_cublas/cutlass_gemm"),
         "-B", str(tmp_path / "build"), "-DCMAKE_CUDA_ARCHITECTURES=100"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "requires 100a and/or 103a" in result.stderr


def test_sm121_probe_resolves_existing_tma_sample_from_code_root() -> None:
    from core.verification.tma_cuda_probe import tma_sample_source
    source = tma_sample_source()
    assert source == CODE_ROOT / "ch07/async_prefetch_2d_demo.cu"
    assert source.is_file()


def test_sm121_probe_missing_compiler_is_unsupported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from core.verification.tma_cuda_probe import run_cuda_tma_sample
    assert run_cuda_tma_sample(nvcc=str(tmp_path / "missing-nvcc")) == "skip"
    assert "no CUDA compilation or execution occurred" in capsys.readouterr().out


def test_sm121_probe_real_failing_command_cannot_pass(capsys: pytest.CaptureFixture[str]) -> None:
    from core.verification.tma_cuda_probe import run_cuda_tma_sample
    # Negative command-control only: false produces no CUDA object or executable.
    assert run_cuda_tma_sample(nvcc="/usr/bin/false") == "fail"
    assert "compilation failed (exit 1)" in capsys.readouterr().out
