"""Dual-architecture Make contracts."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
COMPARE_CHAPTERS = (
    "ch01",
    "ch02",
    "ch04",
    "ch06",
    "ch07",
    "ch08",
    "ch09",
    "ch10",
    "ch11",
    "ch12",
)
ARCHITECTURES = ("sm_100", "sm_103", "sm_120", "sm_121")
MAIN_PATTERN = re.compile(r"\bint\s+main\s*\(")
INTENTIONAL_NON_STANDALONE_MAIN_SOURCES = {
    "ch04": {"nvshmem_ibgda_microbench.cu"},
    "ch12": {
        "helper_baseline_dynamic_parallelism.cu",
        "helper_optimized_dynamic_parallelism.cu",
    },
}


@pytest.mark.parametrize("chapter", COMPARE_CHAPTERS)
def test_compare_stops_after_a_nonfinal_architecture_failure(
    chapter: str,
    tmp_path: Path,
) -> None:
    override = tmp_path / "no-clean.mk"
    override.write_text("clean:\n\t@:\n", encoding="utf-8")
    invocation_log = tmp_path / "architectures.txt"
    make_stub = tmp_path / "make-stub"
    make_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "architecture=\n"
        'for argument in "$@"; do\n'
        '  case "$argument" in\n'
        '    ARCH=*) architecture="${argument#ARCH=}" ;;\n'
        "  esac\n"
        "done\n"
        'printf \'%s\\n\' "$architecture" >> "${COMPARE_ARCH_LOG:?}"\n'
        '[[ "$architecture" != "sm_103" ]]\n',
        encoding="utf-8",
    )
    make_stub.chmod(0o755)
    env = os.environ.copy()
    env["COMPARE_ARCH_LOG"] = str(invocation_log)

    result = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "-f",
            "Makefile",
            "-f",
            str(override),
            f"MAKE={make_stub}",
            "ARCH=sm_100",
            f"ARCH_LIST={' '.join(ARCHITECTURES)}",
            "compare",
        ],
        cwd=CODE_ROOT / chapter,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert invocation_log.read_text(encoding="utf-8").splitlines() == [
        "sm_100",
        "sm_103",
    ]


@pytest.mark.parametrize("chapter", COMPARE_CHAPTERS)
def test_clean_covers_every_supported_architecture(chapter: str) -> None:
    result = subprocess.run(
        ["make", "-n", "ARCH=sm_100", "clean"],
        cwd=CODE_ROOT / chapter,
        check=True,
        capture_output=True,
        text=True,
    )

    for architecture in ARCHITECTURES:
        suffix = architecture.replace("_", "")
        assert f"*_{suffix}" in result.stdout


@pytest.mark.parametrize("chapter", COMPARE_CHAPTERS)
def test_all_builds_every_standalone_cuda_main(chapter: str) -> None:
    chapter_root = CODE_ROOT / chapter
    target = "nvshmem" if chapter == "ch04" else "all"
    result = subprocess.run(
        ["make", "-B", "-n", "ARCH=sm_100", target],
        cwd=chapter_root,
        check=True,
        capture_output=True,
        text=True,
    )
    ignored = INTENTIONAL_NON_STANDALONE_MAIN_SOURCES.get(chapter, set())
    standalone_sources = {
        source.name
        for source in chapter_root.glob("*.cu")
        if MAIN_PATTERN.search(source.read_text(encoding="utf-8")) and source.name not in ignored
    }

    for source_name in standalone_sources:
        assert source_name in result.stdout
