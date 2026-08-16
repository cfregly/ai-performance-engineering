from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.harness.llm_patch_worker import (
    WorkerProtocolError,
    _validate_attestation,
    _validated_source,
    build_llm_patch_worker_env,
    run_llm_patch_worker,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_source(path: Path, body: str = "VALUE = 1\n") -> None:
    path.write_text(body, encoding="utf-8")


def _base_request(original: Path, candidate: Path, *, action: str) -> dict[str, object]:
    return {
        "action": action,
        "original_file": str(original),
        "original_sha256": sha256_file(original),
        "candidate_file": str(candidate),
        "candidate_sha256": sha256_file(candidate),
    }


def test_worker_env_scrubs_common_credentials_and_keeps_runtime_values(tmp_path: Path) -> None:
    env = build_llm_patch_worker_env(
        REPO_ROOT,
        tmp_path,
        base_env={
            "PATH": "/usr/bin",
            "CUDA_VISIBLE_DEVICES": "0",
            "OPENAI_API_KEY": "secret",
            "MY_SERVICE_PASSWORD": "secret",
            "AISP_LLM_PATCH_SANDBOX_ACTIVE": "ambient-value",
        },
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "OPENAI_API_KEY" not in env
    assert "MY_SERVICE_PASSWORD" not in env
    assert "AISP_LLM_PATCH_SANDBOX_ACTIVE" not in env
    assert env["AISP_LLM_PATCH_WORKER"] == "1"
    assert str(REPO_ROOT.resolve()) in env["PYTHONPATH"].split(os.pathsep)
    assert str(tmp_path.resolve()) in env["PYTHONPATH"].split(os.pathsep)


def test_sha256_rejects_symlink_sources(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    link = tmp_path / "link.py"
    _write_source(source)
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symlinks are unavailable on this host")

    with pytest.raises(WorkerProtocolError, match="must not be a symlink"):
        sha256_file(link)


def test_file_backed_worker_rejects_changed_candidate_before_import(tmp_path: Path) -> None:
    original = tmp_path / "baseline.py"
    candidate = tmp_path / "candidate.py"
    _write_source(original)
    _write_source(candidate)
    request = _base_request(original, candidate, action="verify")
    request["candidate_sha256"] = "0" * 64

    with pytest.raises(WorkerProtocolError) as exc_info:
        _validated_source(request, "candidate_file", "candidate_sha256")

    assert exc_info.value.code == "artifact_hash_mismatch"


def test_file_backed_worker_refuses_timing_without_verification_attestation(
    tmp_path: Path,
) -> None:
    original = tmp_path / "baseline.py"
    candidate = tmp_path / "candidate.py"
    _write_source(original)
    _write_source(candidate)
    request = _base_request(original, candidate, action="benchmark")
    request.update(
        {
            "iterations": 3,
            "warmup": 1,
            "measurement_timeout_seconds": 10,
        }
    )

    with pytest.raises(WorkerProtocolError) as exc_info:
        _validate_attestation(
            request,
            original_digest=sha256_file(original),
            candidate_digest=sha256_file(candidate),
        )

    assert exc_info.value.code == "verification_required"


def test_file_backed_worker_rejects_invalid_timeout_before_sandbox_lookup(tmp_path: Path) -> None:
    original = tmp_path / "baseline.py"
    candidate = tmp_path / "candidate.py"
    _write_source(original)
    _write_source(candidate)
    request = _base_request(original, candidate, action="verify")

    response = run_llm_patch_worker(
        request,
        repo_root=REPO_ROOT,
        timeout_seconds=-1,
        python_executable=sys.executable,
    )

    assert response["success"] is False
    assert response["error_type"] == "invalid_timeout"


def test_default_verification_denial_does_not_import_generated_code(tmp_path: Path) -> None:
    marker = tmp_path / "imported.txt"
    original = tmp_path / "baseline.py"
    candidate = tmp_path / "candidate.py"
    _write_source(original)
    _write_source(
        candidate,
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n",
    )
    request = _base_request(original, candidate, action="verify")

    result = run_llm_patch_worker(
        request,
        repo_root=REPO_ROOT,
        timeout_seconds=60,
        python_executable=sys.executable,
    )

    assert result["success"] is False
    assert result["error_type"] == "sandbox_unavailable"
    assert result["timing_started"] is False
    assert result["non_promotable"] is True
    assert result["execution_policy"]["hardened_os_sandbox"] is False
    assert not marker.exists()


def test_attestation_rejects_a_backend_that_is_not_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AISP_LLM_PATCH_SANDBOX_ACTIVE", raising=False)
    original = tmp_path / "baseline.py"
    candidate = tmp_path / "candidate.py"
    _write_source(original)
    _write_source(candidate)
    request = _base_request(original, candidate, action="benchmark")
    request["verification_attestation"] = {
        "verified": True,
        "details": {
            "execution_policy": {
                "sandbox_backend": "inactive-test-backend",
                "hardened_os_sandbox": True,
                "promotable": True,
            },
            "worker_attestation": {
                "protocol_version": 1,
                "action": "verify",
                "request_id": "verification-request",
                "worker_pid": 123,
                "original_sha256": sha256_file(original),
                "candidate_sha256": sha256_file(candidate),
            },
        },
    }

    with pytest.raises(WorkerProtocolError) as exc_info:
        _validate_attestation(
            request,
            original_digest=sha256_file(original),
            candidate_digest=sha256_file(candidate),
        )

    assert exc_info.value.code == "verification_attestation_mismatch"


def test_direct_worker_subprocess_denies_execution_without_sandbox(tmp_path: Path) -> None:
    marker = tmp_path / "imported.txt"
    original = tmp_path / "baseline.py"
    candidate = tmp_path / "candidate.py"
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    _write_source(original)
    _write_source(
        candidate,
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n",
    )
    request = _base_request(original, candidate, action="verify")
    request.update({"protocol_version": 1, "request_id": "default-denial-test"})
    request_path.write_text(json.dumps(request), encoding="utf-8")
    env = build_llm_patch_worker_env(REPO_ROOT, tmp_path)
    env.pop("AISP_LLM_PATCH_SANDBOX_ACTIVE", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-P",
            "-m",
            "core.harness.llm_patch_worker",
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["success"] is False
    assert result["error_type"] == "sandbox_attestation_missing"
    assert not marker.exists()
