#!/usr/bin/env python3
"""Execute bounded CUDA checks for B200 companion dependency families.

The main B200 PyTorch 2.14.1 smoke verifies imports for CUDA Python,
torchvision, torchaudio, and TorchAO. This diagnostic adds one small real CUDA
execution path for each family. It emits JSON only and does not produce
performance evidence.

Run from the ``code`` directory:

    python -m core.scripts.b200_dependency_execution_probe

Each check runs in its own process. Per-check limits total less than the
90-second hard overall deadline, and missing CUDA implementations are failures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout, suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_NOT_RUN = "not_run"

MAX_DEADLINE_SECONDS = 90.0
MAX_CAPTURED_STDERR_CHARS = 8_192
PROCESS_TERMINATION_GRACE_SECONDS = 2.0
_INTERNAL_CHECK_FLAG = "--internal-check"


@dataclass(frozen=True)
class CheckSpec:
    """One required dependency execution check."""

    name: str
    timeout_seconds: float
    coverage: str


@dataclass(frozen=True)
class CheckResult:
    """Observed result for one required dependency execution check."""

    name: str
    coverage: str
    status: str
    duration_ms: float
    timeout_seconds: float
    details: dict[str, Any] = field(default_factory=dict)
    diagnostic: str | None = None
    worker_stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CHECK_SPECS = (
    CheckSpec(
        name="cuda_python_nvrtc_kernel",
        timeout_seconds=25.0,
        coverage="ch14 OptimizedCudaPythonBenchmark NVRTC compilation and kernel execution",
    ),
    CheckSpec(
        name="torchvision_cuda_nms",
        timeout_seconds=10.0,
        coverage="torchvision.ops.nms CUDA operator dispatch and result correctness",
    ),
    CheckSpec(
        name="torchaudio_cuda_resample",
        timeout_seconds=15.0,
        coverage="torchaudio.functional.resample CUDA execution and output validity",
    ),
    CheckSpec(
        name="torchao_live_input",
        timeout_seconds=25.0,
        coverage="existing TorchAO quantized live-input correctness path on CUDA",
    ),
)


def _bounded_deadline(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0 or value > MAX_DEADLINE_SECONDS:
        raise argparse.ArgumentTypeError(
            f"value must be greater than zero and at most {MAX_DEADLINE_SECONDS:g}"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute bounded dependency CUDA checks on a B200 GPU and emit JSON."
    )
    parser.add_argument(
        "--deadline-seconds",
        type=_bounded_deadline,
        default=MAX_DEADLINE_SECONDS,
        help=f"Overall wall-clock deadline. Maximum: {MAX_DEADLINE_SECONDS:g} seconds.",
    )
    return parser


def _exception_diagnostic(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _bounded_stderr(value: str) -> str:
    value = value.strip()
    if len(value) <= MAX_CAPTURED_STDERR_CHARS:
        return value
    half = MAX_CAPTURED_STDERR_CHARS // 2
    omitted = len(value) - (half * 2)
    return value[:half] + f"\n... {omitted} characters omitted ...\n" + value[-half:]


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required distribution is not installed: {name}") from exc


def _require_b200(torch: Any) -> tuple[Any, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if int(torch.cuda.device_count()) < 1:
        raise RuntimeError("PyTorch reported no visible CUDA devices")

    device_index = int(torch.cuda.current_device())
    device = torch.device("cuda", device_index)
    properties = torch.cuda.get_device_properties(device_index)
    capability = (int(properties.major), int(properties.minor))
    if capability != (10, 0):
        raise RuntimeError(
            f"B200 requires compute capability (10, 0), got {capability} on {properties.name!r}"
        )
    if "B200" not in properties.name.upper():
        raise RuntimeError(f"expected a B200 device name, got {properties.name!r}")

    return device, {
        "device_index": device_index,
        "gpu_name": properties.name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
    }


def _check_cuda_python_nvrtc_kernel() -> dict[str, Any]:
    torch = import_module("torch")
    functional = import_module("torch.nn.functional")
    device, device_details = _require_b200(torch)
    benchmark_module = import_module("ch14.optimized_cuda_python")
    benchmark = benchmark_module.OptimizedCudaPythonBenchmark()
    benchmark.device = device
    benchmark.batch = 2
    benchmark.seq_len = 8
    benchmark.hidden = 64

    try:
        torch.manual_seed(42)
        benchmark.setup()
        if not benchmark.kernel_compiled or benchmark.kernel is None:
            raise RuntimeError("NVRTC did not compile and load the CUDA Python kernel")
        if benchmark.output is not None:
            raise RuntimeError("benchmark output existed before kernel execution")

        benchmark.benchmark_fn()
        torch.cuda.synchronize(device)
        output = benchmark.output
        if output is None:
            raise RuntimeError("CUDA Python kernel execution did not produce output")
        expected_shape = (2, 8, 64)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"CUDA Python output shape mismatch: expected {expected_shape}, got {tuple(output.shape)}"
            )
        if output.device.type != "cuda":
            raise RuntimeError(f"CUDA Python output is on {output.device}, expected CUDA")
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("CUDA Python output contains non-finite values")

        reference = functional.layer_norm(
            benchmark.input,
            normalized_shape=(benchmark.hidden,),
            weight=benchmark.weight,
            bias=benchmark.bias,
        )
        reference = functional.gelu(reference)
        reference.masked_fill_(~benchmark.mask.unsqueeze(-1), 0.0)
        reference = reference + benchmark.input
        torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
        max_abs_error = float((output.float() - reference.float()).abs().amax().item())

        return {
            **device_details,
            "distribution": {"cuda-python": _distribution_version("cuda-python")},
            "benchmark": "ch14.optimized_cuda_python.OptimizedCudaPythonBenchmark",
            "workload": {"batch": 2, "seq_len": 8, "hidden": 64},
            "nvrtc_compiled": True,
            "kernel_executed": True,
            "output": {
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "device": str(output.device),
                "finite": True,
            },
            "reference": {
                "rtol": 2e-2,
                "atol": 2e-2,
                "max_abs_error": max_abs_error,
            },
        }
    finally:
        benchmark.teardown()


def _check_torchvision_cuda_nms() -> dict[str, Any]:
    torch = import_module("torch")
    torchvision_ops = import_module("torchvision.ops")
    device, device_details = _require_b200(torch)
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 9.0, 9.0],
            [20.0, 20.0, 30.0, 30.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.8, 0.7], device=device, dtype=torch.float32)
    indices = torchvision_ops.nms(boxes, scores, 0.5)
    torch.cuda.synchronize(device)
    if indices.device.type != "cuda":
        raise RuntimeError(f"torchvision NMS output is on {indices.device}, expected CUDA")
    actual = [int(value) for value in indices.cpu().tolist()]
    if actual != [0, 2]:
        raise RuntimeError(f"torchvision NMS returned {actual}, expected [0, 2]")

    return {
        **device_details,
        "distribution": {"torchvision": _distribution_version("torchvision")},
        "operation": "torchvision.ops.nms",
        "input_boxes": 3,
        "iou_threshold": 0.5,
        "indices": actual,
        "output_device": str(indices.device),
    }


def _check_torchaudio_cuda_resample() -> dict[str, Any]:
    torch = import_module("torch")
    torchaudio_functional = import_module("torchaudio.functional")
    device, device_details = _require_b200(torch)
    waveform = torch.linspace(-1.0, 1.0, 160, device=device, dtype=torch.float32).reshape(1, 160)
    output = torchaudio_functional.resample(waveform, 16_000, 8_000)
    torch.cuda.synchronize(device)
    expected_shape = (1, 80)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"torchaudio resample shape mismatch: expected {expected_shape}, got {tuple(output.shape)}"
        )
    if output.device.type != "cuda":
        raise RuntimeError(f"torchaudio resample output is on {output.device}, expected CUDA")
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("torchaudio resample output contains non-finite values")

    return {
        **device_details,
        "distribution": {"torchaudio": _distribution_version("torchaudio")},
        "operation": "torchaudio.functional.resample",
        "input": {"shape": [1, 160], "sample_rate": 16_000},
        "output": {
            "shape": list(output.shape),
            "sample_rate": 8_000,
            "dtype": str(output.dtype),
            "device": str(output.device),
            "finite": True,
        },
    }


def _check_torchao_live_input() -> dict[str, Any]:
    torch = import_module("torch")
    benchmark_module = import_module("ch13.optimized_torchao_quantization")
    device, device_details = _require_b200(torch)
    import_error = benchmark_module.TORCHAO_IMPORT_ERROR
    if import_error is not None:
        raise RuntimeError(f"TorchAO import failed: {_exception_diagnostic(import_error)}")

    benchmark = benchmark_module.OptimizedTorchAOQuantizationBenchmark()
    benchmark.device = device
    benchmark.batch_size = 2
    benchmark.in_features = 64
    benchmark.hidden_features = 64
    benchmark.out_features = 64

    try:
        torch.manual_seed(42)
        benchmark.setup()
        declared = benchmark._verify_input
        if declared is None or declared is not benchmark.data:
            raise RuntimeError("TorchAO verification input is not the tensor consumed by the model")
        original_input = declared.clone()

        benchmark.benchmark_fn()
        torch.cuda.synchronize(device)
        if benchmark.output is None:
            raise RuntimeError("TorchAO quantized forward did not produce output")
        original_output = benchmark.output.clone()
        if tuple(original_output.shape) != (2, 64):
            raise RuntimeError(
                f"TorchAO output shape mismatch: expected (2, 64), got {tuple(original_output.shape)}"
            )
        if original_output.device.type != "cuda":
            raise RuntimeError(f"TorchAO output is on {original_output.device}, expected CUDA")
        if not bool(torch.isfinite(original_output).all().item()):
            raise RuntimeError("TorchAO output contains non-finite values")

        declared.add_(0.75)
        benchmark.benchmark_fn()
        torch.cuda.synchronize(device)
        mutated_output = benchmark.output.clone()
        if torch.equal(original_output, mutated_output):
            raise RuntimeError("TorchAO quantized forward did not respond to a live input mutation")
        if not bool(torch.isfinite(mutated_output).all().item()):
            raise RuntimeError("TorchAO output after input mutation contains non-finite values")

        benchmark.capture_verification_payload()
        payload_output = benchmark.get_verify_output()
        if not torch.equal(payload_output, mutated_output):
            raise RuntimeError(
                "TorchAO verification payload does not contain the live forward output"
            )

        declared.copy_(original_input)
        benchmark.benchmark_fn()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(benchmark.output, original_output, rtol=0.0, atol=0.0)
        max_live_input_delta = float(
            (mutated_output.float() - original_output.float()).abs().amax().item()
        )

        return {
            **device_details,
            "distribution": {"torchao": _distribution_version("torchao")},
            "benchmark": (
                "ch13.optimized_torchao_quantization.OptimizedTorchAOQuantizationBenchmark"
            ),
            "workload": {
                "batch_size": 2,
                "in_features": 64,
                "hidden_features": 64,
                "out_features": 64,
            },
            "output": {
                "shape": list(original_output.shape),
                "dtype": str(original_output.dtype),
                "device": str(original_output.device),
                "finite": True,
            },
            "live_input_changed_output": True,
            "restored_input_reproduced_output": True,
            "verification_payload_matches_live_output": True,
            "max_live_input_delta": max_live_input_delta,
        }
    finally:
        benchmark.teardown()


CHECK_FUNCTIONS: dict[str, Callable[[], dict[str, Any]]] = {
    "cuda_python_nvrtc_kernel": _check_cuda_python_nvrtc_kernel,
    "torchvision_cuda_nms": _check_torchvision_cuda_nms,
    "torchaudio_cuda_resample": _check_torchaudio_cuda_resample,
    "torchao_live_input": _check_torchao_live_input,
}


def _internal_check_main(name: str) -> int:
    check = CHECK_FUNCTIONS.get(name)
    if check is None:
        payload = {
            "status": STATUS_FAIL,
            "details": {},
            "diagnostic": f"unknown internal check: {name!r}",
        }
        print(json.dumps(payload, sort_keys=True))
        return 1

    try:
        # Dependency imports and NVRTC diagnostics must not corrupt the JSON protocol.
        with redirect_stdout(sys.stderr):
            details = check()
    except (Exception, SystemExit) as exc:
        payload = {
            "status": STATUS_FAIL,
            "details": {},
            "diagnostic": _exception_diagnostic(exc),
        }
        returncode = 1
    else:
        payload = {
            "status": STATUS_PASS,
            "details": details,
            "diagnostic": None,
        }
        returncode = 0
    print(json.dumps(payload, sort_keys=True))
    return returncode


def _terminate_process_group(
    process: subprocess.Popen[str],
    *,
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[str, str]:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    remaining = max(0.0, deadline - clock())
    try:
        return process.communicate(timeout=remaining)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        return process.communicate()


def _run_check_subprocess(
    spec: CheckSpec,
    *,
    code_root: Path,
    timeout_seconds: float,
) -> CheckResult:
    started = time.monotonic()
    deadline = started + timeout_seconds
    command = [sys.executable, "-u", str(Path(__file__).resolve()), _INTERNAL_CHECK_FLAG, spec.name]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(code_root)
        if not existing_pythonpath
        else str(code_root) + os.pathsep + existing_pythonpath
    )
    environment["PYTHONUNBUFFERED"] = "1"

    try:
        process = subprocess.Popen(
            command,
            cwd=code_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return CheckResult(
            name=spec.name,
            coverage=spec.coverage,
            status=STATUS_FAIL,
            duration_ms=(time.monotonic() - started) * 1000.0,
            timeout_seconds=timeout_seconds,
            diagnostic=_exception_diagnostic(exc),
        )

    timed_out = False
    execution_timeout = max(
        0.0,
        deadline - time.monotonic() - PROCESS_TERMINATION_GRACE_SECONDS,
    )
    try:
        stdout, stderr = process.communicate(timeout=execution_timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        stdout, stderr = _terminate_process_group(process, deadline=deadline)

    duration_ms = (time.monotonic() - started) * 1000.0
    worker_stderr = _bounded_stderr(stderr)
    if timed_out:
        return CheckResult(
            name=spec.name,
            coverage=spec.coverage,
            status=STATUS_FAIL,
            duration_ms=duration_ms,
            timeout_seconds=timeout_seconds,
            diagnostic=f"timed out after {timeout_seconds:.3f}s",
            worker_stderr=worker_stderr,
        )

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return CheckResult(
            name=spec.name,
            coverage=spec.coverage,
            status=STATUS_FAIL,
            duration_ms=duration_ms,
            timeout_seconds=timeout_seconds,
            diagnostic=f"invalid worker JSON: {_exception_diagnostic(exc)}",
            worker_stderr=worker_stderr,
        )
    if not isinstance(payload, dict):
        return CheckResult(
            name=spec.name,
            coverage=spec.coverage,
            status=STATUS_FAIL,
            duration_ms=duration_ms,
            timeout_seconds=timeout_seconds,
            diagnostic=f"worker returned a non-object JSON payload: {type(payload).__name__}",
            worker_stderr=worker_stderr,
        )

    worker_status = payload.get("status")
    details = payload.get("details")
    diagnostic = payload.get("diagnostic")
    if worker_status not in {STATUS_PASS, STATUS_FAIL} or not isinstance(details, dict):
        worker_status = STATUS_FAIL
        details = {}
        diagnostic = f"worker returned an invalid result payload: {payload!r}"
    elif worker_status == STATUS_PASS and process.returncode != 0:
        worker_status = STATUS_FAIL
        details = {}
        diagnostic = f"worker exited with status {process.returncode} after reporting pass"
    elif worker_status == STATUS_FAIL and not isinstance(diagnostic, str):
        diagnostic = f"worker exited with status {process.returncode} without a diagnostic"

    return CheckResult(
        name=spec.name,
        coverage=spec.coverage,
        status=worker_status,
        duration_ms=duration_ms,
        timeout_seconds=timeout_seconds,
        details=details,
        diagnostic=diagnostic,
        worker_stderr=worker_stderr,
    )


def execute_probe(
    code_root: Path,
    *,
    deadline_seconds: float,
    run_check: Callable[..., CheckResult] = _run_check_subprocess,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    if not math.isfinite(deadline_seconds) or not 0 < deadline_seconds <= MAX_DEADLINE_SECONDS:
        raise ValueError(
            f"deadline_seconds must be greater than zero and at most {MAX_DEADLINE_SECONDS:g}"
        )

    started_at = datetime.now(timezone.utc).isoformat()
    started = clock()
    deadline = started + deadline_seconds
    results: list[CheckResult] = []

    for index, spec in enumerate(CHECK_SPECS):
        remaining = deadline - clock()
        if remaining <= 0:
            for pending in CHECK_SPECS[index:]:
                results.append(
                    CheckResult(
                        name=pending.name,
                        coverage=pending.coverage,
                        status=STATUS_NOT_RUN,
                        duration_ms=0.0,
                        timeout_seconds=0.0,
                        diagnostic="overall deadline reached before check started",
                    )
                )
            break
        timeout_seconds = min(spec.timeout_seconds, remaining)
        try:
            result = run_check(spec, code_root=code_root, timeout_seconds=timeout_seconds)
        except (Exception, SystemExit) as exc:
            result = CheckResult(
                name=spec.name,
                coverage=spec.coverage,
                status=STATUS_FAIL,
                duration_ms=0.0,
                timeout_seconds=timeout_seconds,
                diagnostic=f"check runner failed: {_exception_diagnostic(exc)}",
            )
        results.append(result)

    passed = sum(result.status == STATUS_PASS for result in results)
    failed = sum(result.status == STATUS_FAIL for result in results)
    not_run = sum(result.status == STATUS_NOT_RUN for result in results)
    succeeded = len(results) == len(CHECK_SPECS) and failed == 0 and not_run == 0
    elapsed_ms = max(0.0, (clock() - started) * 1000.0)

    return {
        "schema_version": 1,
        "probe": "b200_dependency_execution",
        "status": STATUS_PASS if succeeded else STATUS_FAIL,
        "started_at": started_at,
        "elapsed_ms": elapsed_ms,
        "deadline_seconds": deadline_seconds,
        "environment": {
            "code_root": str(code_root.resolve()),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
        "checks": [result.to_dict() for result in results],
        "summary": {
            "required": len(CHECK_SPECS),
            "passed": passed,
            "failed": failed,
            "not_run": not_run,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == _INTERNAL_CHECK_FLAG:
        if len(raw_argv) != 2:
            raise RuntimeError("internal check mode requires exactly one check name")
        return _internal_check_main(raw_argv[1])

    args = build_parser().parse_args(raw_argv)
    code_root = Path(__file__).resolve().parents[2]
    report = execute_probe(code_root, deadline_seconds=args.deadline_seconds)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == STATUS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
