"""Fail-closed boundary for evaluating generated benchmark variants.

A plain subprocess does not prevent generated code from reading user files, using
the network, or changing the workspace. Evaluation therefore stays disabled until
this module can wrap the worker in a configured operating-system sandbox. Process
isolation, environment scrubbing, timeouts, and process-group cleanup remain useful
defense in depth, but they are not treated as a security boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import traceback
import uuid
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from core.utils.python_entrypoints import build_repo_python_env, load_module_from_path

PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 256 * 1024
MAX_RESULT_BYTES = 2 * 1024 * 1024
MAX_SOURCE_BYTES = 16 * 1024 * 1024
_WORKER_MODULE = "core.harness.llm_patch_worker"
_SENSITIVE_ENV_MARKERS = (
    "ACCESS_KEY",
    "API_KEY",
    "AUTH_TOKEN",
    "CLIENT_SECRET",
    "CREDENTIAL",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "SESSION_TOKEN",
)
_SENSITIVE_ENV_PREFIXES = (
    "ANTHROPIC_",
    "AWS_",
    "AZURE_",
    "GCP_",
    "GOOGLE_",
    "HF_",
    "HUGGINGFACE_",
    "OPENAI_",
    "WANDB_",
)


class WorkerProtocolError(RuntimeError):
    """Raised when a worker request or response violates the protocol."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one regular, non-symlink source file."""
    source_path = Path(path)
    if source_path.is_symlink():
        raise WorkerProtocolError("symlink_source", f"Source must not be a symlink: {source_path}")
    try:
        stat_result = source_path.stat()
    except OSError as exc:
        raise WorkerProtocolError(
            "source_unreadable", f"Cannot stat source: {source_path}"
        ) from exc
    if not source_path.is_file():
        raise WorkerProtocolError("source_not_file", f"Source is not a regular file: {source_path}")
    if stat_result.st_size > MAX_SOURCE_BYTES:
        raise WorkerProtocolError(
            "source_too_large",
            f"Source exceeds {MAX_SOURCE_BYTES} bytes: {source_path}",
        )
    digest = hashlib.sha256()
    try:
        with source_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise WorkerProtocolError(
            "source_unreadable", f"Cannot read source: {source_path}"
        ) from exc
    return digest.hexdigest()


def build_llm_patch_worker_env(
    repo_root: Path,
    candidate_dir: Path,
    *,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a benchmark-capable environment without common credential variables."""
    source_env = dict(os.environ if base_env is None else base_env)
    source_env.pop("AISP_LLM_PATCH_SANDBOX_ACTIVE", None)
    source_env.pop("AISP_LLM_PATCH_WORKER", None)
    for name in list(source_env):
        upper_name = name.upper()
        if upper_name.startswith(_SENSITIVE_ENV_PREFIXES) or any(
            marker in upper_name for marker in _SENSITIVE_ENV_MARKERS
        ):
            source_env.pop(name, None)
    env: dict[str, str] = build_repo_python_env(
        repo_root,
        base_env=source_env,
        extra_pythonpath=[candidate_dir],
    )
    env["AISP_LLM_PATCH_WORKER"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _kill_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        with suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(process.pid, signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=2)
    else:
        # A generated module may leave descendants in the worker's process group.
        with suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(process.pid, signal.SIGTERM)


def _read_log_tail(path: Path, *, limit: int = 4000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            if size > limit:
                stream.seek(-limit, os.SEEK_END)
            return stream.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _failure(error_type: str, error: str, **details: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": False,
        "error_type": error_type,
        "error": error,
    }
    payload.update(details)
    return payload


def _build_hardened_os_sandbox_command(
    command: list[str],
    *,
    repo_root: Path,
    candidate_path: Path,
    temp_dir: Path,
) -> tuple[str, list[str]] | None:
    """Return an OS-sandboxed command, or ``None`` when no backend is configured.

    A future backend must enforce filesystem write restrictions, hide user secrets,
    and disable network access while preserving the runtime paths needed by the
    benchmark. Merely finding a subprocess launcher or scrubbing environment
    variables is not sufficient. No backend currently meets that contract here.
    """
    del command, repo_root, candidate_path, temp_dir
    return None


def run_llm_patch_worker(
    request: Mapping[str, Any],
    *,
    repo_root: Path,
    timeout_seconds: float,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Run one file-backed worker request and validate its response.

    Failures are returned as structured dictionaries so the benchmark report can
    preserve the reason while refusing verification or timing promotion.
    """
    try:
        parsed_timeout = float(timeout_seconds)
    except (TypeError, ValueError):
        return _failure("invalid_timeout", f"Invalid worker timeout: {timeout_seconds!r}")
    if not math.isfinite(parsed_timeout) or parsed_timeout <= 0:
        return _failure("invalid_timeout", f"Invalid worker timeout: {timeout_seconds!r}")

    request_payload = dict(request)
    request_payload.setdefault("protocol_version", PROTOCOL_VERSION)
    request_payload.setdefault("request_id", uuid.uuid4().hex)
    request_id = request_payload.get("request_id")
    action = request_payload.get("action")
    if not isinstance(request_id, str) or not request_id:
        return _failure("invalid_request", "Worker request_id must be a non-empty string")
    if action not in {"verify", "benchmark"}:
        return _failure("invalid_request", f"Unsupported worker action: {action!r}")

    candidate_value = request_payload.get("candidate_file")
    if not isinstance(candidate_value, str) or not candidate_value:
        return _failure("invalid_request", "Worker candidate_file must be a non-empty string")
    candidate_path = Path(candidate_value).resolve()
    request_payload["candidate_file"] = str(candidate_path)

    try:
        encoded_request = json.dumps(
            request_payload,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        return _failure("invalid_request", f"Worker request is not strict JSON: {exc}")
    if len(encoded_request) > MAX_REQUEST_BYTES:
        return _failure("request_too_large", "Worker request exceeds the protocol size limit")

    with tempfile.TemporaryDirectory(prefix="aisp_llm_patch_worker_") as temp_dir_value:
        temp_dir = Path(temp_dir_value)
        request_path = temp_dir / "request.json"
        result_path = temp_dir / "result.json"
        stdout_path = temp_dir / "stdout.log"
        stderr_path = temp_dir / "stderr.log"
        request_path.write_bytes(encoded_request)

        env = build_llm_patch_worker_env(repo_root, candidate_path.parent)
        worker_command = [
            python_executable or sys.executable,
            "-P",
            "-m",
            _WORKER_MODULE,
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ]
        sandbox_plan = _build_hardened_os_sandbox_command(
            worker_command,
            repo_root=repo_root,
            candidate_path=candidate_path,
            temp_dir=temp_dir,
        )
        if sandbox_plan is None:
            return _failure(
                "sandbox_unavailable",
                "Generated-code evaluation requires a hardened OS sandbox backend",
                timing_started=False,
                non_promotable=True,
                execution_policy={
                    "sandbox_backend": None,
                    "hardened_os_sandbox": False,
                    "promotable": False,
                },
            )
        sandbox_backend, command = sandbox_plan
        env["AISP_LLM_PATCH_SANDBOX_ACTIVE"] = sandbox_backend
        try:
            with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
                process = subprocess.Popen(
                    command,
                    cwd=temp_dir,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    start_new_session=True,
                )
                timed_out = False
                try:
                    process.wait(timeout=parsed_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                finally:
                    _kill_process_group(process)
        except OSError as exc:
            return _failure("worker_launch_failed", f"Failed to launch patch worker: {exc}")

        stderr_tail = _read_log_tail(stderr_path)
        stdout_tail = _read_log_tail(stdout_path)
        if timed_out:
            return _failure(
                "worker_timeout",
                f"Patch worker exceeded {parsed_timeout:g} seconds",
                stderr_tail=stderr_tail,
                stdout_tail=stdout_tail,
            )
        if process.returncode != 0:
            return _failure(
                "worker_exit",
                f"Patch worker exited with code {process.returncode}",
                returncode=process.returncode,
                stderr_tail=stderr_tail,
                stdout_tail=stdout_tail,
            )
        if not result_path.exists() or result_path.is_symlink() or not result_path.is_file():
            return _failure(
                "worker_missing_result",
                "Patch worker did not produce a regular result file",
                stderr_tail=stderr_tail,
                stdout_tail=stdout_tail,
            )
        try:
            result_size = result_path.stat().st_size
        except OSError as exc:
            return _failure("worker_invalid_result", f"Cannot stat worker result: {exc}")
        if result_size <= 0 or result_size > MAX_RESULT_BYTES:
            return _failure(
                "worker_invalid_result",
                f"Worker result size is invalid: {result_size} bytes",
            )
        try:
            response = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return _failure("worker_invalid_result", f"Cannot decode worker result: {exc}")
        if not isinstance(response, dict):
            return _failure("worker_invalid_result", "Worker result must be a JSON object")
        if response.get("protocol_version") != PROTOCOL_VERSION:
            return _failure("worker_invalid_result", "Worker protocol version mismatch")
        if response.get("request_id") != request_id or response.get("action") != action:
            return _failure("worker_invalid_result", "Worker response identity mismatch")
        if type(response.get("success")) is not bool:
            return _failure("worker_invalid_result", "Worker success must be a JSON boolean")
        response["execution_policy"] = {
            "sandbox_backend": sandbox_backend,
            "hardened_os_sandbox": True,
            "promotable": True,
        }
        return response


def _require_request_string(request: Mapping[str, Any], name: str) -> str:
    value = request.get(name)
    if not isinstance(value, str) or not value:
        raise WorkerProtocolError("invalid_request", f"{name} must be a non-empty string")
    return value


def _validated_source(request: Mapping[str, Any], name: str, digest_name: str) -> tuple[Path, str]:
    requested_path = Path(_require_request_string(request, name))
    if requested_path.suffix != ".py":
        raise WorkerProtocolError(
            "unsupported_source", f"Only Python sources are supported: {requested_path}"
        )
    expected_digest = _require_request_string(request, digest_name)
    actual_digest = sha256_file(requested_path)
    if actual_digest != expected_digest:
        raise WorkerProtocolError(
            "artifact_hash_mismatch",
            f"{name} changed before worker execution",
        )
    return requested_path.resolve(), actual_digest


def _validate_verification_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkerProtocolError("invalid_verification", "Verification result must be an object")
    if type(value.get("verified")) is not bool:
        raise WorkerProtocolError(
            "invalid_verification",
            "Verification result must contain a boolean verified field",
        )
    errors = value.get("errors")
    if not isinstance(errors, list) or not all(isinstance(item, str) for item in errors):
        raise WorkerProtocolError(
            "invalid_verification",
            "Verification errors must be a list of strings",
        )
    if value.get("verified") is True and errors:
        raise WorkerProtocolError(
            "invalid_verification",
            "Passing verification cannot contain errors",
        )
    if not isinstance(value.get("verification_type"), str) or not value.get("verification_type"):
        raise WorkerProtocolError(
            "invalid_verification",
            "Verification type must be a non-empty string",
        )
    return value


def _run_verification(request: Mapping[str, Any]) -> dict[str, Any]:
    original_path, original_digest = _validated_source(request, "original_file", "original_sha256")
    candidate_path, candidate_digest = _validated_source(
        request, "candidate_file", "candidate_sha256"
    )
    test_shape_value = request.get("test_shape", [256, 256])
    if (
        not isinstance(test_shape_value, list)
        or not test_shape_value
        or not all(type(item) is int and item > 0 for item in test_shape_value)
    ):
        raise WorkerProtocolError("invalid_request", "test_shape must contain positive integers")

    from core.harness.run_benchmarks import _verify_patched_benchmark_in_worker

    verification = _validate_verification_result(
        _verify_patched_benchmark_in_worker(
            str(original_path),
            str(candidate_path),
            test_shape=tuple(test_shape_value),
        )
    )
    final_original_digest = sha256_file(original_path)
    final_candidate_digest = sha256_file(candidate_path)
    if final_original_digest != original_digest or final_candidate_digest != candidate_digest:
        verification["verified"] = False
        verification.setdefault("errors", []).append(
            "Source changed while the verification worker was running"
        )
        verification["verification_type"] = "source_changed_during_verification"

    details = verification.setdefault("details", {})
    if not isinstance(details, dict):
        raise WorkerProtocolError("invalid_verification", "Verification details must be an object")
    details["worker_attestation"] = {
        "protocol_version": PROTOCOL_VERSION,
        "action": "verify",
        "request_id": request["request_id"],
        "worker_pid": os.getpid(),
        "original_sha256": original_digest,
        "candidate_sha256": candidate_digest,
    }
    return verification


def _validate_attestation(
    request: Mapping[str, Any],
    *,
    original_digest: str,
    candidate_digest: str,
) -> None:
    attestation = request.get("verification_attestation")
    if not isinstance(attestation, dict):
        raise WorkerProtocolError(
            "verification_required",
            "Benchmark timing requires a verification worker attestation",
        )
    if attestation.get("verified") is not True:
        raise WorkerProtocolError(
            "verification_required",
            "Benchmark timing requires verified=true",
        )
    details = attestation.get("details")
    worker_attestation = details.get("worker_attestation") if isinstance(details, dict) else None
    execution_policy = details.get("execution_policy") if isinstance(details, dict) else None
    active_backend = os.environ.get("AISP_LLM_PATCH_SANDBOX_ACTIVE")
    if (
        not isinstance(execution_policy, dict)
        or execution_policy.get("hardened_os_sandbox") is not True
        or execution_policy.get("promotable") is not True
        or execution_policy.get("sandbox_backend") != active_backend
    ):
        raise WorkerProtocolError(
            "verification_attestation_mismatch",
            "Verification attestation lacks matching OS sandbox provenance",
        )
    if not isinstance(worker_attestation, dict):
        raise WorkerProtocolError(
            "verification_required",
            "Benchmark timing requires worker verification provenance",
        )
    required = {
        "protocol_version": PROTOCOL_VERSION,
        "action": "verify",
        "original_sha256": original_digest,
        "candidate_sha256": candidate_digest,
    }
    if any(worker_attestation.get(key) != value for key, value in required.items()):
        raise WorkerProtocolError(
            "verification_attestation_mismatch",
            "Verification attestation does not match the timed sources",
        )
    if not isinstance(worker_attestation.get("request_id"), str) or not isinstance(
        worker_attestation.get("worker_pid"), int
    ):
        raise WorkerProtocolError(
            "verification_attestation_mismatch",
            "Verification attestation identity is incomplete",
        )


def _positive_finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WorkerProtocolError("invalid_benchmark_result", f"{name} is not numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise WorkerProtocolError(
            "invalid_benchmark_result",
            f"{name} must be positive and finite",
        )
    return number


def _load_candidate_benchmark(candidate_path: Path) -> Any:
    from core.harness.benchmark_harness import BaseBenchmark

    module_name = f"_llm_patch_{candidate_path.stem}_{uuid.uuid4().hex}"
    module = load_module_from_path(module_name, candidate_path)
    candidates: list[type[Any]] = []
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseBenchmark)
            and obj is not BaseBenchmark
            and getattr(obj, "__module__", "") == module.__name__
        ):
            candidates.append(obj)
    if len(candidates) > 1:
        raise WorkerProtocolError(
            "ambiguous_benchmark_class",
            f"Generated module defines multiple benchmark classes: {candidate_path}",
        )
    if candidates:
        try:
            return candidates[0]()
        except Exception as exc:
            raise WorkerProtocolError(
                "benchmark_init_failed",
                f"Cannot instantiate generated benchmark: {type(exc).__name__}: {exc}",
            ) from exc
    factory = getattr(module, "get_benchmark", None)
    if callable(factory):
        try:
            benchmark = factory()
        except Exception as exc:
            raise WorkerProtocolError(
                "benchmark_init_failed",
                f"Generated benchmark factory failed: {type(exc).__name__}: {exc}",
            ) from exc
        if isinstance(benchmark, BaseBenchmark):
            return benchmark
    raise WorkerProtocolError(
        "class_not_found",
        f"No unambiguous benchmark class or valid factory found in: {candidate_path}",
    )


def _run_benchmark(request: Mapping[str, Any]) -> dict[str, Any]:
    original_path, original_digest = _validated_source(request, "original_file", "original_sha256")
    candidate_path, candidate_digest = _validated_source(
        request, "candidate_file", "candidate_sha256"
    )
    _validate_attestation(
        request,
        original_digest=original_digest,
        candidate_digest=candidate_digest,
    )

    iterations = request.get("iterations")
    warmup = request.get("warmup")
    measurement_timeout_seconds = request.get("measurement_timeout_seconds")
    timeout_multiplier = request.get("timeout_multiplier")
    validity_profile = request.get("validity_profile")
    enforce_environment_validation = request.get("enforce_environment_validation")
    allow_virtualization = request.get("allow_virtualization")
    allow_foreign_gpu_processes = request.get("allow_foreign_gpu_processes")
    single_gpu = request.get("single_gpu")
    if type(iterations) is not int or iterations <= 0:
        raise WorkerProtocolError("invalid_request", "iterations must be a positive integer")
    if type(warmup) is not int or warmup < 0:
        raise WorkerProtocolError("invalid_request", "warmup must be a non-negative integer")
    if (
        not isinstance(measurement_timeout_seconds, int | float)
        or isinstance(measurement_timeout_seconds, bool)
        or not math.isfinite(float(measurement_timeout_seconds))
        or float(measurement_timeout_seconds) <= 0
    ):
        raise WorkerProtocolError(
            "invalid_request",
            "measurement_timeout_seconds must be positive and finite",
        )
    if (
        not isinstance(timeout_multiplier, int | float)
        or isinstance(timeout_multiplier, bool)
        or not math.isfinite(float(timeout_multiplier))
        or float(timeout_multiplier) <= 0
    ):
        raise WorkerProtocolError(
            "invalid_request",
            "timeout_multiplier must be positive and finite",
        )
    if validity_profile not in {"strict", "portable"}:
        raise WorkerProtocolError(
            "invalid_request",
            "validity_profile must be strict or portable",
        )
    for name, value in (
        ("enforce_environment_validation", enforce_environment_validation),
        ("allow_virtualization", allow_virtualization),
        ("allow_foreign_gpu_processes", allow_foreign_gpu_processes),
        ("single_gpu", single_gpu),
    ):
        if type(value) is not bool:
            raise WorkerProtocolError("invalid_request", f"{name} must be a boolean")

    from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness

    benchmark = _load_candidate_benchmark(candidate_path)
    config = BenchmarkConfig(
        iterations=iterations,
        warmup=warmup,
        use_subprocess=True,
        measurement_timeout_seconds=int(math.ceil(float(measurement_timeout_seconds))),
        timeout_seconds=int(math.ceil(float(measurement_timeout_seconds))),
        timeout_multiplier=float(timeout_multiplier),
        validity_profile=validity_profile,
        enforce_environment_validation=enforce_environment_validation,
        allow_virtualization=allow_virtualization,
        allow_foreign_gpu_processes=allow_foreign_gpu_processes,
        single_gpu=single_gpu,
        enable_profiling=False,
        enable_nsys=False,
        enable_ncu=False,
    )
    harness = BenchmarkHarness(config=config)
    benchmark_result = harness.benchmark(benchmark)
    timing = getattr(benchmark_result, "timing", None)
    if timing is None:
        raise WorkerProtocolError("invalid_benchmark_result", "Benchmark returned no timing")
    errors = list(getattr(benchmark_result, "errors", None) or [])
    if errors:
        raise WorkerProtocolError(
            "benchmark_failed",
            "Generated benchmark failed: " + " | ".join(str(item) for item in errors[:5]),
        )
    if getattr(benchmark_result, "timeout_stage", None):
        raise WorkerProtocolError(
            "benchmark_timeout",
            f"Generated benchmark timed out in {benchmark_result.timeout_stage}",
        )
    median_ms = _positive_finite(getattr(timing, "median_ms", None), "median_ms")
    min_ms = _positive_finite(getattr(timing, "min_ms", None), "min_ms")
    completed_iterations = getattr(timing, "iterations", None)
    if type(completed_iterations) is not int or completed_iterations <= 0:
        raise WorkerProtocolError(
            "invalid_benchmark_result",
            "Benchmark iteration count must be a positive integer",
        )

    final_original_digest = sha256_file(original_path)
    final_candidate_digest = sha256_file(candidate_path)
    if final_original_digest != original_digest or final_candidate_digest != candidate_digest:
        raise WorkerProtocolError(
            "source_changed_during_benchmark",
            "Source changed while the benchmark worker was running",
        )
    return {
        "success": True,
        "time_ms": median_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "iterations": completed_iterations,
        "patched_file": str(candidate_path),
        "verification_request_id": request["verification_attestation"]["details"][
            "worker_attestation"
        ]["request_id"],
        "candidate_sha256": candidate_digest,
        "worker_pid": os.getpid(),
        "timing_isolated_subprocess": True,
    }


def _execute_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if not os.environ.get("AISP_LLM_PATCH_SANDBOX_ACTIVE"):
        raise WorkerProtocolError(
            "sandbox_attestation_missing",
            "Patch worker may run only inside an active OS sandbox backend",
        )
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise WorkerProtocolError("protocol_mismatch", "Unsupported worker protocol version")
    request_id = _require_request_string(request, "request_id")
    action = _require_request_string(request, "action")
    if action == "verify":
        result = _run_verification(request)
    elif action == "benchmark":
        result = _run_benchmark(request)
    else:
        raise WorkerProtocolError("unsupported_action", f"Unsupported worker action: {action}")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "action": action,
        "success": True,
        "result": result,
        "worker_pid": os.getpid(),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_RESULT_BYTES:
        raise WorkerProtocolError("result_too_large", "Worker result exceeds the protocol limit")
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_bytes(encoded)
    os.chmod(temporary_path, 0o600)
    os.replace(temporary_path, path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one generated benchmark variant")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    request_id = "unknown"
    action = "unknown"
    try:
        if args.request.is_symlink() or not args.request.is_file():
            raise WorkerProtocolError("invalid_request_file", "Request must be a regular file")
        request_size = args.request.stat().st_size
        if request_size <= 0 or request_size > MAX_REQUEST_BYTES:
            raise WorkerProtocolError("invalid_request_file", "Request file size is invalid")
        request = json.loads(args.request.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise WorkerProtocolError("invalid_request", "Worker request must be a JSON object")
        if isinstance(request.get("request_id"), str):
            request_id = request["request_id"]
        if isinstance(request.get("action"), str):
            action = request["action"]
        response = _execute_request(request)
    except WorkerProtocolError as exc:
        response = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "action": action,
            "success": False,
            "error_type": exc.code,
            "error": str(exc),
            "worker_pid": os.getpid(),
        }
        if action == "benchmark":
            response["timing_started"] = exc.code in {
                "benchmark_failed",
                "benchmark_timeout",
                "invalid_benchmark_result",
                "source_changed_during_benchmark",
            }
    except Exception as exc:
        response = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "action": action,
            "success": False,
            "error_type": "worker_exception",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
            "worker_pid": os.getpid(),
        }
    try:
        _atomic_write_json(args.result, response)
    except Exception as exc:
        print(f"Failed to write worker result: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
