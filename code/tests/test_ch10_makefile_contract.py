"""Chapter 10 standalone build contracts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
CHAPTER_ROOT = CODE_ROOT / "ch10"
REQUIRED_TORCH_EXTENSION_SOURCES = {
    "tcgen05_cluster.cu",
    "tcgen05_warp_specialized.cu",
    "tcgen05_warp_specialized_cutlass.cu",
    "tcgen05_warpgroup_specialized.cu",
}
TORCH_EXTENSION_MARKERS = ("torch/extension.h", "ATen/", "PYBIND11_MODULE")
MAIN_PATTERN = re.compile(r"\bint\s+main\s*\(")
TMA_MULTICAST_TARGETS = {
    "sm_100": "100",
    "sm_103": "103",
    "sm_120": "0",
    "sm_121": "0",
}


def _standalone_build_output() -> str:
    result = subprocess.run(
        ["make", "-B", "-n", "ARCH=sm_100", "all"],
        cwd=CHAPTER_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_torch_extensions_are_not_linked_as_standalone_executables() -> None:
    build_output = _standalone_build_output()

    extension_sources = {
        source.name
        for source in CHAPTER_ROOT.glob("*.cu")
        if any(marker in source.read_text(encoding="utf-8") for marker in TORCH_EXTENSION_MARKERS)
    }
    assert extension_sources >= REQUIRED_TORCH_EXTENSION_SOURCES

    for source_name in extension_sources:
        assert source_name not in build_output


def test_every_cuda_source_with_main_is_built_as_a_standalone_executable() -> None:
    build_output = _standalone_build_output()
    standalone_sources = {
        source.name
        for source in CHAPTER_ROOT.glob("*.cu")
        if MAIN_PATTERN.search(source.read_text(encoding="utf-8"))
    }

    assert "tcgen05_blackwell.cu" in standalone_sources
    for source_name in standalone_sources:
        assert source_name in build_output


@pytest.mark.parametrize(("architecture", "feature_target"), TMA_MULTICAST_TARGETS.items())
def test_tma_multicast_build_selects_an_explicit_feature_target(
    architecture: str,
    feature_target: str,
) -> None:
    suffix = architecture.replace("_", "")
    for target in (
        f"tma_multicast_baseline_{suffix}",
        f"tma_multicast_cluster_{suffix}",
    ):
        result = subprocess.run(
            ["make", "-B", "-n", f"ARCH={architecture}", target],
            cwd=CHAPTER_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        assert f"-DTMA_MULTICAST_TARGET={feature_target}" in result.stdout


@pytest.mark.parametrize(("architecture", "feature_target"), TMA_MULTICAST_TARGETS.items())
def test_tma_multicast_verification_target_builds_both_binaries(
    architecture: str,
    feature_target: str,
) -> None:
    suffix = architecture.replace("_", "")
    result = subprocess.run(
        ["make", "-B", "-n", f"ARCH={architecture}", "verify-tma-multicast"],
        cwd=CHAPTER_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.count("-DVERIFY=1") == 2
    baseline_command = next(
        line for line in result.stdout.splitlines() if "tma_multicast_baseline.cu" in line
    )
    cluster_command = next(
        line for line in result.stdout.splitlines() if "tma_multicast_cluster.cu" in line
    )
    assert f"-DTMA_MULTICAST_TARGET={feature_target}" in baseline_command
    assert f"-o tma_multicast_baseline_verify_{suffix}" in baseline_command
    assert f"-DTMA_MULTICAST_TARGET={feature_target}" in cluster_command
    assert f"-o tma_multicast_cluster_verify_{suffix}" in cluster_command


def test_compare_dry_run_builds_tma_multicast_verification_for_every_architecture() -> None:
    result = subprocess.run(
        ["make", "-B", "-n", "ARCH=sm_100", "compare"],
        cwd=CHAPTER_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    for architecture in TMA_MULTICAST_TARGETS:
        suffix = architecture.replace("_", "")
        assert f"-o tma_multicast_baseline_verify_{suffix}" in result.stdout
        assert f"-o tma_multicast_cluster_verify_{suffix}" in result.stdout


def test_tma_multicast_has_the_cuda_13_sm103_compatibility_path() -> None:
    source = (CHAPTER_ROOT / "tma_multicast_cluster.cu").read_text(encoding="utf-8")

    assert "#if TMA_MULTICAST_TARGET == 100 || TMA_MULTICAST_TARGET == 103" in source
    assert "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)" not in source
    assert "__CUDA_ARCH_SPECIFIC__ == 1030" in source
    assert "cp.async.bulk.tensor.2d.shared::cluster.global.tile" in source
    assert ".mbarrier::complete_tx::bytes.multicast::cluster" in source


@pytest.mark.parametrize(
    "source_name",
    ["tma_multicast_baseline.cu", "tma_multicast_cluster.cu"],
)
def test_tma_multicast_unsupported_target_uses_the_cuda_skip_contract(
    source_name: str,
) -> None:
    source = (CHAPTER_ROOT / source_name).read_text(encoding="utf-8")
    unsupported_branch = source.split(
        "#if TMA_MULTICAST_TARGET == 0",
        maxsplit=1,
    )[1].split("#endif", maxsplit=1)[0]

    assert '"SKIPPED:' in unsupported_branch
    assert "return 3;" in unsupported_branch
    assert "TIME_MS" not in unsupported_branch


@pytest.mark.parametrize(
    "source_name",
    ["tma_multicast_baseline.cu", "tma_multicast_cluster.cu"],
)
def test_tma_cluster_gemm_tiles_use_opt_in_dynamic_shared_memory(
    source_name: str,
) -> None:
    source = (CHAPTER_ROOT / source_name).read_text(encoding="utf-8")

    assert "constexpr int TILE_M = 4;" in source
    assert "constexpr int TILE_N = 128;" in source
    assert "constexpr int TILE_K = 128;" in source
    assert (
        "constexpr size_t B_SMEM_BYTES =\n    static_cast<size_t>(TILE_K) * TILE_N * sizeof(float);"
    ) in source
    assert "constexpr size_t DYNAMIC_SMEM_BYTES = B_SMEM_BYTES;" in source
    assert "static_assert(DYNAMIC_SMEM_BYTES == 65536);" in source
    assert "extern __shared__ __align__(128) unsigned char smem_raw[];" in source
    assert "reinterpret_cast<float (*)[TILE_N]>(smem_raw)" in source
    assert source.count("__shared__ alignas(128) float A_smem") == 1
    assert source.count("__shared__ float A_smem") == 1
    assert "__shared__ alignas(128) float B_smem" not in source
    assert "__shared__ float B_smem" not in source
    assert "cudaDevAttrMaxSharedMemoryPerBlockOptin" in source
    assert "DYNAMIC_SMEM_BYTES + kernel_attributes.sharedSizeBytes" in source
    assert "cudaFuncAttributeMaxDynamicSharedMemorySize" in source
    assert "static_cast<int>(DYNAMIC_SMEM_BYTES)" in source
    assert "config.dynamicSmemBytes = DYNAMIC_SMEM_BYTES;" in source


def test_tma_multicast_dynamic_b_tile_preserves_transaction_bytes() -> None:
    source = (CHAPTER_ROOT / "tma_multicast_cluster.cu").read_text(encoding="utf-8")

    assert "barrier_arrive_tx(*bar, 1, B_SMEM_BYTES)" in source
    assert "barrier_arrive_tx(*bar, 1, sizeof(B_smem))" not in source


@pytest.mark.parametrize(
    ("source_name", "skip_count"),
    [("tma_multicast_baseline.cu", 3), ("tma_multicast_cluster.cu", 3)],
)
def test_tma_cluster_gemm_sources_use_the_cuda_skip_contract(
    source_name: str,
    skip_count: int,
) -> None:
    source = (CHAPTER_ROOT / source_name).read_text(encoding="utf-8")

    assert '"SKIP:' not in source
    assert "TIME_MS: 0.0" not in source
    assert source.count('"SKIPPED:') == skip_count
    assert source.count("return 3;") == skip_count


def test_flash_attention_tma_pipeline_uses_opt_in_dynamic_shared_memory() -> None:
    source = (CHAPTER_ROOT / "optimized_flash_attn_tma_micro_pipeline.cu").read_text(
        encoding="utf-8"
    )
    wrapper = (CHAPTER_ROOT / "optimized_flash_attn_tma_micro_pipeline.py").read_text(
        encoding="utf-8"
    )

    assert "constexpr int STAGES  = 3;" in source
    assert "extern __shared__ __align__(128) unsigned char smem_raw[];" in source
    assert "smem_raw + STAGES * TILE_BYTES" in source
    assert "cudaDevAttrMaxSharedMemoryPerBlockOptin" in source
    assert "DYNAMIC_SMEM_BYTES + kernel_attributes.sharedSizeBytes" in source
    assert "cudaFuncAttributeMaxDynamicSharedMemorySize" in source
    assert "static_cast<int>(DYNAMIC_SMEM_BYTES)" in source
    assert source.count("<<<grid, block, DYNAMIC_SMEM_BYTES, stream>>>") == 2
    assert "num_stages=3" in wrapper


def test_flash_attention_tma_pipeline_uses_the_cuda_skip_contract() -> None:
    source = (CHAPTER_ROOT / "optimized_flash_attn_tma_micro_pipeline.cu").read_text(
        encoding="utf-8"
    )

    assert '"SKIP:' not in source
    assert "TIME_MS: 0.0" not in source
    assert source.count('"SKIPPED:') == 7
    assert source.count("return 3;") == 7
