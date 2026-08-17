"""Chapter 10 standalone build contracts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

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
