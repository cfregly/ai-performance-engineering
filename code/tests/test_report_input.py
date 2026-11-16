from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

from core.analysis.reporting.generator import generate_report


@pytest.mark.parametrize("source", ["missing.json", "http://127.0.0.1:6970"])
def test_generate_report_requires_local_json_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    def reject_network(*_args, **_kwargs):
        pytest.fail("Report generation attempted a network request")

    monkeypatch.setattr(urllib.request, "urlopen", reject_network)
    input_path = tmp_path / source if source == "missing.json" else source
    output_path = tmp_path / "report.html"

    with pytest.raises(FileNotFoundError, match="Benchmark JSON file not found"):
        generate_report(input_path, output_path, format="html")

    assert not output_path.exists()
