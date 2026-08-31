"""Offline extraction round trips from a real CPU torch.profiler capture."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

CODE = Path(__file__).resolve().parents[1]
EXTRACTOR = CODE / "core/profiling/extract_pytorch_profile.py"


@pytest.fixture(scope="module")
def real_cpu_capture(tmp_path_factory):
    capture = tmp_path_factory.mktemp("real_cpu_profile") / "capture"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    command = [sys.executable, "-m", "core.scripts.profiling.pytorch_profiler_runner",
               str(CODE / "tests/fixtures/mcp_torch_profile_target.py"),
               "--output-dir", str(capture), "--profile-mode", "full"]
    result = subprocess.run(command, cwd=CODE, env=environment, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = json.loads((capture / "metadata.json").read_text())
    assert metadata["cuda_available"] is False
    assert metadata["error"] is None
    assert (capture / "trace.json").is_file()
    operators = json.loads((capture / "key_averages_full.json").read_text())
    assert any(row["name"] == "aten::mm" for row in operators)
    return capture


@pytest.mark.parametrize("selector", ["relative", "absolute", "absolute_glob"])
def test_real_capture_cli_csv_round_trip(real_cpu_capture, tmp_path, selector):
    capture = real_cpu_capture
    pattern = {"relative": capture.name, "absolute": str(capture),
               "absolute_glob": str(capture.parent / "capt*")}[selector]
    prefix = tmp_path / "roundtrip.v1"
    result = subprocess.run([sys.executable, str(EXTRACTOR), pattern, "--output-prefix", str(prefix)],
                            cwd=capture.parent, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    with (tmp_path / "roundtrip.v1_metadata.csv").open(newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    with (tmp_path / "roundtrip.v1_operators.csv").open(newline="") as handle:
        operator_rows = list(csv.DictReader(handle))
    assert len(metadata_rows) == 1
    assert metadata_rows[0]["cuda_available"] == "False"
    assert metadata_rows[0]["error"] == ""
    originals = json.loads((capture / "key_averages_full.json").read_text())
    assert len(operator_rows) == len(originals)
    source_mm = next(row for row in originals if row["name"] == "aten::mm")
    result_mm = next(row for row in operator_rows if row["name"] == "aten::mm")
    assert result_mm["mode"] == "full"
    assert int(result_mm["count"]) == source_mm["count"]
    assert float(result_mm["cpu_time_total_us"]) == source_mm["cpu_time_total_us"]
    assert float(result_mm["cuda_time_total_us"]) == source_mm["cuda_time_total_us"]


def test_unrelated_directory_does_not_create_an_empty_success_report(tmp_path):
    directory = tmp_path / "ordinary_directory"
    directory.mkdir()
    result = subprocess.run([sys.executable, str(EXTRACTOR), directory.name, "--output-prefix", str(tmp_path / "summary")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "No PyTorch profiler directories found" in result.stderr
    assert not (tmp_path / "summary_metadata.csv").exists()
