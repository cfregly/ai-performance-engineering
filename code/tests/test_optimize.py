"""Tests for retained optimizer adapters and the retired execution path."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.optimization import auto
from core.optimization.auto.input_adapters import (
    BenchmarkAdapter,
    CodeSource,
    FileAdapter,
    RepoAdapter,
    detect_input_type,
)
from examples.optimize_examples import main as campaign_example_main


def test_file_adapter_reads_one_file(tmp_path: Path) -> None:
    source_path = tmp_path / "test.py"
    source_path.write_text("print('hello')", encoding="utf-8")

    sources = list(FileAdapter(paths=[source_path]).get_sources())

    assert len(sources) == 1
    assert sources[0].name == "test"
    assert sources[0].content == "print('hello')"


def test_file_adapter_writes_to_explicit_output(tmp_path: Path) -> None:
    adapter = FileAdapter(
        paths=[tmp_path / "test.py"],
        output_dir=tmp_path / "output",
        suffix="_candidate",
    )
    source = CodeSource(path=tmp_path / "test.py", content="original", name="test")

    output_path = adapter.write_output(source, "candidate code")

    assert output_path.read_text(encoding="utf-8") == "candidate code"
    assert output_path.name == "test_candidate.py"


def test_benchmark_adapter_discovers_candidate(tmp_path: Path) -> None:
    (tmp_path / "baseline_test.py").write_text("# baseline", encoding="utf-8")
    (tmp_path / "optimized_test.py").write_text("# candidate", encoding="utf-8")

    sources = list(
        BenchmarkAdapter(
            directory=tmp_path,
            threshold=1.1,
            pattern="optimized_*.py",
        ).get_sources()
    )

    assert len(sources) == 1
    assert "test" in sources[0].name


@pytest.mark.parametrize(
    ("input_value", "expected_kind", "expected_type"),
    [
        ("https://github.com/NVIDIA/cutlass", "repo", RepoAdapter),
    ],
)
def test_detect_input_type_for_remote_repo(
    input_value: str,
    expected_kind: str,
    expected_type: type[RepoAdapter],
) -> None:
    input_kind, adapter = detect_input_type(input_value)

    assert input_kind == expected_kind
    assert isinstance(adapter, expected_type)


def test_detect_input_type_for_file(tmp_path: Path) -> None:
    source_path = tmp_path / "test.py"
    source_path.write_text("# test", encoding="utf-8")

    input_kind, adapter = detect_input_type(str(source_path))

    assert input_kind == "file"
    assert isinstance(adapter, FileAdapter)


def test_detect_input_type_for_benchmark_directory(tmp_path: Path) -> None:
    (tmp_path / "baseline_test.py").write_text("# baseline", encoding="utf-8")
    (tmp_path / "optimized_test.py").write_text("# candidate", encoding="utf-8")

    input_kind, adapter = detect_input_type(str(tmp_path))

    assert input_kind == "benchmark"
    assert isinstance(adapter, BenchmarkAdapter)


def test_auto_package_exposes_adapters_but_not_optimizer() -> None:
    assert auto.FileAdapter is FileAdapter
    assert auto.RepoAdapter is RepoAdapter
    assert not hasattr(auto, "AutoOptimizer")


@pytest.mark.parametrize(
    "module_name",
    ("core.optimization.auto", "core.optimization.auto.optimizer"),
)
def test_retired_cli_fails_closed(module_name: str) -> None:
    code_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=code_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "retired" in result.stderr
    assert "core.optimization.campaign" in result.stderr


def test_direct_legacy_optimizer_construction_fails_closed() -> None:
    from core.optimization.auto.optimizer import AutoOptimizer

    with pytest.raises(RuntimeError, match="core.optimization.campaign"):
        AutoOptimizer()


def test_campaign_example_initializes_workspace_from_temporary_specs(tmp_path: Path) -> None:
    workload_spec = tmp_path / "workload.json"
    environment_spec = tmp_path / "environment.json"
    workspace = tmp_path / "campaign"
    control_commit = "a" * 40
    workload_spec.write_text('{"target": "test"}', encoding="utf-8")
    environment_spec.write_text('{"gpu": "test"}', encoding="utf-8")

    result = campaign_example_main(
        [
            "--workspace",
            str(workspace),
            "--objective",
            "Test campaign initialization",
            "--metric",
            "latency_ms",
            "--initial-control-commit",
            control_commit,
            "--primary-case",
            "representative",
            "--frozen-case",
            "boundary",
            "--workload-spec",
            str(workload_spec),
            "--environment-spec",
            str(environment_spec),
        ]
    )

    config = json.loads((workspace / "campaign.json").read_text(encoding="utf-8"))
    assert result == 0
    assert config["initial_control_commit"] == control_commit
    assert Path(config["workload_spec"]) == workload_spec
    assert Path(config["environment_spec"]) == environment_spec
