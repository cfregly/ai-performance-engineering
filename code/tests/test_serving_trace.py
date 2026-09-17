"""Real JSONL/CLI ingestion preserves incomplete requests and token gaps."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.analysis.serving_trace import main, summarize


def trace():
    return [
        {
            "request_id": "fast",
            "arrival_s": 0.0,
            "token_timestamps_s": [0.125, 0.25, 0.375],
            "finish_s": 0.5,
            "status": "completed",
        },
        {
            "request_id": "stall",
            "arrival_s": 0.0,
            "token_timestamps_s": [0.125, 0.25, 0.875],
            "finish_s": 1.0,
            "status": "completed",
        },
        {
            "request_id": "failed",
            "arrival_s": 0.0,
            "token_timestamps_s": [0.25],
            "finish_s": 0.75,
            "status": "failed",
        },
        {
            "request_id": "cancelled",
            "arrival_s": 0.125,
            "token_timestamps_s": [],
            "finish_s": 0.5,
            "status": "cancelled",
        },
    ]


def test_goodput_uses_per_request_slo_and_keeps_failed_records():
    report = summarize(trace(), max_ttft_s=0.25, max_tpot_s=0.5, max_itl_s=0.25)
    assert report["total_requests"] == 4
    assert report["completed_requests"] == 2
    assert report["failed_requests"] == report["cancelled_requests"] == 1
    assert report["slo_attainment_fraction"] == 0.25
    assert report["goodput_requests_per_second"] == 1.0
    assert report["goodput_tokens_per_second"] == 3.0
    assert report["emitted_tokens_per_second"] == 7.0
    assert report["completed_tokens_per_second"] == 6.0
    assert report["requests"][1]["tpot_s"] == 0.375
    assert report["requests"][1]["max_itl_s"] == 0.625
    assert report["requests"][3]["ttft_s"] is None


def test_one_token_request_has_no_invented_tpot():
    row = {
        "request_id": 1,
        "arrival_s": 2.0,
        "token_timestamps_s": [2.25],
        "finish_s": 2.5,
        "status": "completed",
    }
    report = summarize([row], max_ttft_s=0.25, max_tpot_s=0.0)
    assert report["requests"][0]["tpot_s"] is None
    assert report["requests"][0]["meets_slo"]


def test_trace_preserves_source_timestamps_failure_reason_and_extra_fields():
    records = trace()
    records[2].update(error="worker stopped after one token", model="example-model")
    report = summarize(records, max_ttft_s=0.25, max_tpot_s=0.5)
    failed = report["requests"][2]
    assert failed["token_timestamps_s"] == records[2]["token_timestamps_s"]
    assert failed["error"] == "worker stopped after one token"
    assert failed["model"] == "example-model"
    records[2]["token_timestamps_s"].append(0.5)
    assert failed["token_timestamps_s"] == [0.25]


@pytest.mark.parametrize(
    "change",
    [
        {"token_timestamps_s": [0.5, 0.25]},
        {"finish_s": 0.1},
        {"arrival_s": float("nan")},
        {"status": "unknown"},
        {"token_timestamps_s": []},
    ],
)
def test_malformed_completed_record_fails(change):
    row = trace()[0] | change
    with pytest.raises(ValueError):
        summarize([row], max_ttft_s=1.0, max_tpot_s=1.0)


def test_real_cli_roundtrip_and_registration(tmp_path):
    from core.tools.tools_commands import TOOLS

    assert TOOLS["serving-trace"].module_name == "core.analysis.serving_trace"
    source, output = tmp_path / "requests.jsonl", tmp_path / "metrics.json"
    source.write_text("\n".join(json.dumps(row) for row in trace()))
    assert (
        main(
            [
                "--trace",
                str(source),
                "--ttft-ms",
                "250",
                "--tpot-ms",
                "500",
                "--max-itl-ms",
                "250",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["slo_attainment_fraction"] == 0.25
    # Exercise the registered umbrella command, including its argument forwarding.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.aisp",
            "tools",
            "serving-trace",
            "--",
            "--trace",
            str(source),
            "--ttft-ms",
            "250",
            "--tpot-ms",
            "500",
            "--max-itl-ms",
            "250",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["slo_attainment_fraction"] == 0.25
    assert report["total_requests"] == 4
