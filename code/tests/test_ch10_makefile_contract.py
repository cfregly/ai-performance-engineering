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
    result = subprocess.run(
        [
            "make",
            "-B",
            "-n",
            f"ARCH={architecture}",
            f"tma_multicast_cluster_{suffix}",
        ],
        cwd=CHAPTER_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"-DTMA_MULTICAST_TARGET={feature_target}" in result.stdout


def test_tma_multicast_has_the_cuda_13_sm103_compatibility_path() -> None:
    source = (CHAPTER_ROOT / "tma_multicast_cluster.cu").read_text(encoding="utf-8")

    assert "#if TMA_MULTICAST_TARGET == 100 || TMA_MULTICAST_TARGET == 103" in source
    assert "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)" not in source
    assert "__CUDA_ARCH_SPECIFIC__ == 1030" in source
    assert "cp.async.bulk.tensor.2d.shared::cluster.global.tile" in source
    assert ".mbarrier::complete_tx::bytes.multicast::cluster" in source


def test_tma_multicast_unsupported_target_uses_the_cuda_skip_contract() -> None:
    source = (CHAPTER_ROOT / "tma_multicast_cluster.cu").read_text(encoding="utf-8")
    unsupported_branch = source.split(
        "#if TMA_MULTICAST_TARGET == 0",
        maxsplit=1,
    )[1].split("#endif", maxsplit=1)[0]

    assert '"SKIPPED:' in unsupported_branch
    assert "return 3;" in unsupported_branch
    assert "TIME_MS" not in unsupported_branch
