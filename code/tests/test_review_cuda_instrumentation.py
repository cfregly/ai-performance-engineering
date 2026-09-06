from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]


def _nvcc() -> str:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is required for CUDA compile regressions")
    return nvcc


@pytest.mark.parametrize("profiling_enabled", [False, True])
@pytest.mark.parametrize("profiling_helpers_first", [False, True])
def test_nvtx_headers_compile_in_either_order(
    tmp_path: Path,
    profiling_enabled: bool,
    profiling_helpers_first: bool,
) -> None:
    headers = [
        "core/common/headers/profiling_helpers.cuh",
        "core/common/nvtx_utils.cuh",
    ]
    if not profiling_helpers_first:
        headers.reverse()
    source = tmp_path / "nvtx_include_order.cu"
    source.write_text(
        "\n".join(
            [*(f'#include "{header}"' for header in headers), ""]
            + [
                "void instrumented_path() {",
                '  NVTX_RANGE("compute_kernel:first");',
                '  NVTX_RANGE("compute_kernel:second");',
                '  PROFILE_MEMORY_COPY("transfer_async:copy");',
                '  NVTX_MARK_COMPUTE("compute_kernel:mark");',
                "}",
            ]
        ),
        encoding="utf-8",
    )
    command = [
        _nvcc(),
        "-std=c++17",
        "-I",
        str(CODE_ROOT),
        "-c",
        str(source),
        "-o",
        str(tmp_path / "nvtx_include_order.o"),
    ]
    if profiling_enabled:
        command.insert(1, "-DENABLE_NVTX_PROFILING=1")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    diagnostics = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, diagnostics
    assert "redefined" not in diagnostics.lower()
    nm = shutil.which("nm")
    if nm is None:
        pytest.skip("nm is required to verify NVTX object semantics")
    symbol_result = subprocess.run(
        [nm, "-C", str(tmp_path / "nvtx_include_order.o")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert symbol_result.returncode == 0, symbol_result.stderr
    if profiling_enabled:
        assert "aisp_nvtx::NvtxRange" in symbol_result.stdout
        assert "nvtxRangePushEx" in symbol_result.stdout
        assert "nvtxRangePop" in symbol_result.stdout
    else:
        assert "aisp_nvtx::NvtxRange" not in symbol_result.stdout
        assert "nvtxRangePushEx" not in symbol_result.stdout


@pytest.mark.parametrize(
    ("virtual_arch", "real_arch", "expected_min_blocks"),
    [
        ("compute_100a", "sm_100a", 2),
        ("compute_103a", "sm_103a", 2),
        ("compute_120", "sm_120", 1),
        ("compute_121", "sm_121", 1),
    ],
)
def test_launch_bounds_metadata_is_valid_for_configured_architecture(
    tmp_path: Path,
    virtual_arch: str,
    real_arch: str,
    expected_min_blocks: int,
) -> None:
    source = CODE_ROOT / "ch06" / "optimized_launch_bounds_cuda.cu"
    object_result = subprocess.run(
        [
            _nvcc(),
            "-std=c++17",
            "-gencode",
            f"arch={virtual_arch},code={real_arch}",
            "-c",
            str(source),
            "-o",
            str(tmp_path / f"optimized_launch_bounds_{real_arch}.o"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    object_diagnostics = f"{object_result.stdout}\n{object_result.stderr}"
    assert object_result.returncode == 0, object_diagnostics
    assert ".minnctapersm will be ignored" not in object_diagnostics

    ptx_path = tmp_path / f"optimized_launch_bounds_{virtual_arch}.ptx"
    result = subprocess.run(
        [
            _nvcc(),
            "-std=c++17",
            f"--gpu-architecture={virtual_arch}",
            "--ptx",
            str(source),
            "-o",
            str(ptx_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    diagnostics = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, diagnostics
    assert ".minnctapersm will be ignored" not in diagnostics
    ptx = ptx_path.read_text(encoding="utf-8")
    assert ".maxntid 1024" in ptx
    assert f".minnctapersm {expected_min_blocks}" in ptx
