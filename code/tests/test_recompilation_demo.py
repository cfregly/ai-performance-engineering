"""Real subprocess regressions for the Chapter 14 recompilation diagnostic."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "ch14" / "recompilation_demo.py"


def _run_demo(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )


def _successful_report(*args: str) -> dict:
    completed = _run_demo(*args)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_strict_default_warms_known_inputs_without_serving_compilation_and_rejects_late(
    tmp_path,
):
    output_path = tmp_path / "strict-report.json"
    summary = _successful_report("--iterations", "1", "--json", str(output_path))
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary["full_report_json"] == str(output_path)
    assert summary["full_request_records_emitted_to_stdout"] is False
    assert report["configuration"]["policy"] == "strict"
    assert report["process_isolation"]["established"] is True
    assert (
        report["process_isolation"]["default_compile_worker_process_id"]
        != report["process_isolation"]["guarded_policy_worker_process_id"]
    )

    default_scenario = report["scenarios"]["default_compile"]
    guarded = report["scenarios"]["guarded_policy"]
    assert default_scenario["backend_counts"]["compile_submissions"]["serving"] > 0
    assert guarded["policy"]["serving_stance"] == "fail_on_recompile"
    assert guarded["backend_counts"]["compile_submissions"]["serving"] == 0
    assert guarded["backend_counts"]["observed_eager_fallback_requests"] == 0

    known = [record for record in guarded["request_records"] if record["warmed_signature"]]
    assert known
    assert all(record["status"] == "ok" for record in known)
    assert all(record["route"] == "compiled_cache" for record in known)
    assert all(record["backend_compile_submission_delta"] == 0 for record in known)
    assert all(record["compiled_execution_delta"] == 1 for record in known)
    assert all(record["output"]["correctness"] == "pass" for record in known)

    late = [record for record in guarded["request_records"] if not record["warmed_signature"]]
    assert len(late) == 1
    assert late[0]["status"] == "rejected"
    assert late[0]["route"] == "strict_rejection"
    assert late[0]["error"]["fail_on_recompile_detected"] is True
    assert late[0]["backend_compile_submission_delta"] == 0
    assert guarded["timing"]["serving"]["count"] == len(guarded["request_records"])
    assert guarded["timing"]["serving_includes_every_attempt"] is True
    assert guarded["timing"]["slo_assertion"] == {
        "threshold_ms": None,
        "violation_count": None,
        "status": "not_configured",
    }


def test_explicit_eager_on_recompile_counts_correct_fallback_without_new_compile(tmp_path):
    output_path = tmp_path / "eager-fallback-report.json"
    summary = _successful_report(
        "--policy",
        "eager_on_recompile",
        "--iterations",
        "1",
        "--json",
        str(output_path),
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    guarded = report["scenarios"]["guarded_policy"]

    assert summary["comparison"] == report["comparison"]
    assert guarded["policy"]["fallback_opt_in"] is True
    assert guarded["policy"]["serving_stance"] == "eager_on_recompile"
    assert guarded["backend_counts"]["compile_submissions"]["serving"] == 0
    assert guarded["backend_counts"]["observed_eager_fallback_requests"] == 1
    late = [record for record in guarded["request_records"] if not record["warmed_signature"]]
    assert len(late) == 1
    assert late[0]["status"] == "ok"
    assert late[0]["route"] == "eager_fallback"
    assert late[0]["fallback_observed"] is True
    assert late[0]["backend_compile_submission_delta"] == 0
    assert late[0]["compiled_execution_delta"] == 0
    assert late[0]["output"]["correctness"] == "pass"
    assert late[0]["output"]["max_abs_error"] == 0.0


def test_cuda_inductor_contract_does_not_substitute_cpu_or_eager():
    completed = _run_demo("--device", "cpu", "--backend", "inductor")

    assert completed.returncode == 2
    assert "Supported execution contracts are cpu/eager" in completed.stderr
    assert completed.stdout == ""
