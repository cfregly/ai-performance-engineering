"""Regressions for audit residuals W1-110 and W1-102.

W1-110: the dependency-health report must read the version macros that
upstream CUTLASS defines in include/cutlass/version.h. These tests call the
real PerformanceCoreBase.get_dependency_health path on file fixtures.

W1-102: the ch02 PCIe demo cannot run on a CPU-only host, so its printed
claims are checked in the source. GPU runs of the same file are listed as
separate validation commands.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core import perf_core_base

CODE_ROOT = Path(__file__).resolve().parents[1]
PCIE_DEMO = CODE_ROOT / "ch02" / "memory_transfer_pcie_demo.cu"

# Define lines as published in NVIDIA/cutlass include/cutlass/version.h.
UPSTREAM_VERSION_H = (
    "#pragma once\n"
    "\n"
    "#include <cstdint>\n"
    "#include <string>\n"
    "\n"
    "#define CUTLASS_MAJOR 4\n"
    "#define CUTLASS_MINOR 8\n"
    "#define CUTLASS_PATCH 0\n"
    "\n"
    "#ifdef CUTLASS_VERSIONS_GENERATED\n"
    '#include "cutlass/version_extended.h"\n'
    "#else\n"
    "#define CUTLASS_BUILD 0\n"
    '#define CUTLASS_REVISION ""\n'
    "#endif\n"
    "\n"
    "#define CUTLASS_VERSION ((CUTLASS_MAJOR)*100 + (CUTLASS_MINOR)*10 + CUTLASS_PATCH)\n"
)


def _dependency_health(monkeypatch, tmp_path: Path, header: str) -> dict:
    include_dir = tmp_path / "third_party" / "cutlass" / "include" / "cutlass"
    include_dir.mkdir(parents=True)
    (include_dir / "version.h").write_text(header, encoding="utf-8")
    monkeypatch.setattr(perf_core_base, "CODE_ROOT", tmp_path)
    return perf_core_base.PerformanceCoreBase(bench_root=tmp_path).get_dependency_health()


def _cutlass_parse_warnings(result: dict) -> list[str]:
    return [w for w in result["warnings"] if "Failed to parse CUTLASS version" in w]


def test_cutlass_version_reads_upstream_macro_names(monkeypatch, tmp_path):
    result = _dependency_health(monkeypatch, tmp_path, UPSTREAM_VERSION_H)
    assert result["cutlass"]["version"] == "4.8.0"
    assert _cutlass_parse_warnings(result) == []


@pytest.mark.parametrize(
    "header",
    [
        pytest.param(
            "#define CUTLASS_VERSION ((CUTLASS_MAJOR)*100 + (CUTLASS_MINOR)*10 + CUTLASS_PATCH)\n",
            id="derived-expression-only",
        ),
        pytest.param(
            UPSTREAM_VERSION_H.replace("#define CUTLASS_PATCH 0\n", ""),
            id="missing-patch",
        ),
        pytest.param(
            UPSTREAM_VERSION_H + "#undef CUTLASS_MAJOR\n#define CUTLASS_MAJOR 5\n",
            id="conflicting-redefinition",
        ),
    ],
)
def test_cutlass_version_rejects_incomplete_or_ambiguous_headers(monkeypatch, tmp_path, header):
    result = _dependency_health(monkeypatch, tmp_path, header)
    assert result["cutlass"]["version"] is None
    assert len(_cutlass_parse_warnings(result)) == 1


def _printf_literals(source: str) -> list[str]:
    return re.findall(r'printf\("((?:[^"\\]|\\.)*)"', source)


def test_pcie_demo_printed_reference_uses_per_direction_ceiling():
    literals = _printf_literals(PCIE_DEMO.read_text(encoding="utf-8"))
    reference = [text for text in literals if "GB/s" in text and "%" not in text]
    assert any("64 GB/s per direction" in text for text in reference)
    assert not any(";" in text for text in literals)


def test_pcie_demo_comments_do_not_claim_an_unmeasured_bottleneck():
    source = PCIE_DEMO.read_text(encoding="utf-8")
    comments = re.findall(r"//([^\n]*)", source)
    assert not any("bottleneck" in comment.lower() for comment in comments)
    literals = _printf_literals(source)
    bottleneck_lines = [text for text in literals if "bottleneck" in text.lower()]
    assert bottleneck_lines == [
        "The negotiated link and actual bottleneck are not measured by this demo.\\n"
    ]
