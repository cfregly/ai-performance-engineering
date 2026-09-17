"""Diagnose ``torch.compile`` recompilation policy on a bounded inference model.

This is a compiler-lifecycle diagnostic, not a baseline/optimized performance
benchmark.  It runs the ordinary ``default`` stance and a representative-warmup
policy in separate processes because Dynamo's in-process state is part of the
experiment.  The default guarded policy is ``fail_on_recompile``.  The only
fallback mode is the explicitly selected ``eager_on_recompile`` policy, and each
request that bypasses a compiled backend callable is counted.

The review was motivated by Chaim Rand's September 8, 2026 article, "Overcoming
the PyTorch Recompilation Dilemma in Real-Time Inference Workloads":
https://chaimrand.medium.com/overcoming-the-pytorch-recompilation-dilemma-in-real-time-inference-workloads-069fac5485cd
This implementation is independent and uses the public PyTorch 2.9 compiler
stance API rather than copying the article's code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional

SCHEMA_VERSION = "recompilation-diagnostic.v1"
DIAGNOSTIC_VERSION = "1.0"
IN_FEATURES = 8
OUT_FEATURES = 4
DEFAULT_ITERATIONS = 3


class DiagnosticError(RuntimeError):
    """A violated diagnostic contract with a concise user-facing message."""


@dataclass(frozen=True)
class InputCase:
    """One exact input signature used by warmup or serving."""

    name: str
    batch: int
    layout: str
    warmed: bool

    @property
    def batch_bucket(self) -> str:
        if self.batch == 0:
            return "0"
        if self.batch == 1:
            return "1"
        return "2+"


WARMUP_CASES: tuple[InputCase, ...] = (
    InputCase("batch0_contiguous", 0, "contiguous", True),
    InputCase("batch1_contiguous", 1, "contiguous", True),
    InputCase("batch2_contiguous", 2, "contiguous", True),
    InputCase("batch4_contiguous", 4, "contiguous", True),
    InputCase("batch2_strided", 2, "strided", True),
)
LATE_UNSEEN_CASE = InputCase("batch3_strided_late", 3, "strided", False)


class RowwiseInferenceModel(torch.nn.Module):
    """Small deterministic row-wise model with complete, inspectable outputs."""

    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        weight = torch.arange(
            OUT_FEATURES * IN_FEATURES, dtype=torch.float32, device=device
        ).reshape(OUT_FEATURES, IN_FEATURES)
        bias = torch.arange(OUT_FEATURES, dtype=torch.float32, device=device)
        self.register_buffer("weight", (weight - weight.mean()) / 32.0)
        self.register_buffer("bias", (bias - 1.5) / 10.0)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(functional.linear(rows, self.weight, self.bias))


class CountingBackend:
    """Wrap a real registered backend and count submissions and executions.

    PyTorch exposes backend selection by name through ``torch.compile`` but has no
    public name-to-callable resolver.  The wrapper therefore uses Dynamo's
    registry resolver only at this narrow instrumentation boundary.  The serving
    policy itself uses the public ``torch.compiler.set_stance`` API.
    """

    def __init__(self, backend_name: str) -> None:
        lookup_backend = getattr(torch._dynamo, "lookup_backend", None)
        if lookup_backend is None:
            raise DiagnosticError(
                "This diagnostic requires torch._dynamo.lookup_backend to wrap "
                "and count a registered backend."
            )
        try:
            self._backend: Callable[..., Any] = lookup_backend(backend_name)
        except Exception as error:
            raise DiagnosticError(
                f"Unable to resolve the requested backend {backend_name!r}: {error}"
            ) from error
        self.backend_name = backend_name
        self.compile_submissions = 0
        self.compiled_executions = 0
        self.resolved_backend = (
            f"{getattr(self._backend, '__module__', '<unknown>')}."
            f"{getattr(self._backend, '__name__', type(self._backend).__name__)}"
        )

    def __call__(
        self, graph_module: torch.fx.GraphModule, example_inputs: Sequence[torch.Tensor]
    ) -> Callable[..., Any]:
        self.compile_submissions += 1
        compiled = self._backend(graph_module, example_inputs)

        def counted_execution(*args: Any, **kwargs: Any) -> Any:
            self.compiled_executions += 1
            return compiled(*args, **kwargs)

        return counted_execution


def _validate_runtime(device_name: str, backend_name: str) -> torch.device:
    version_match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if version_match is None or tuple(map(int, version_match.groups())) < (2, 9):
        raise DiagnosticError(
            f"PyTorch 2.9+ is required for this diagnostic; found {torch.__version__}."
        )
    if not hasattr(torch.compiler, "set_stance"):
        raise DiagnosticError("torch.compiler.set_stance is unavailable in this runtime.")
    if not hasattr(torch.compiler, "reset"):
        raise DiagnosticError("torch.compiler.reset is unavailable in this runtime.")
    if torch._dynamo.config.suppress_errors:
        raise DiagnosticError(
            "torch._dynamo.config.suppress_errors must be False; implicit compiler "
            "fallback is forbidden by this diagnostic."
        )

    if (device_name, backend_name) == ("cpu", "eager"):
        return torch.device("cpu")
    if (device_name, backend_name) != ("cuda", "inductor"):
        raise DiagnosticError(
            "Supported execution contracts are cpu/eager (diagnostic default) and "
            "cuda/inductor (explicit code-generation run)."
        )
    if not torch.cuda.is_available():
        raise DiagnosticError(
            "CUDA/Inductor was requested, but torch.cuda.is_available() is False. "
            "No CPU or eager substitution was made."
        )
    if torch.version.cuda is None:
        raise DiagnosticError(
            "CUDA/Inductor was requested, but this PyTorch build has no CUDA runtime."
        )
    return torch.device("cuda", torch.cuda.current_device())


def _make_input(case: InputCase, device: torch.device) -> torch.Tensor:
    if case.layout == "contiguous":
        count = case.batch * IN_FEATURES
        values = torch.arange(count, dtype=torch.float32, device=device)
        return ((values.reshape(case.batch, IN_FEATURES) % 17.0) - 8.0) / 8.0
    if case.layout == "strided":
        count = case.batch * IN_FEATURES * 2
        base = torch.arange(count, dtype=torch.float32, device=device).reshape(
            case.batch, IN_FEATURES * 2
        )
        return (((base % 17.0) - 8.0) / 8.0)[:, ::2]
    raise DiagnosticError(f"Unknown input layout {case.layout!r}.")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().reshape(-1)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "preview": flat[:8].cpu().tolist(),
    }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _latency_summary(values_ms: Sequence[float]) -> dict[str, int | float | None]:
    def rounded(value: float | None) -> float | None:
        return None if value is None else round(value, 6)

    return {
        "count": len(values_ms),
        "p50_ms": rounded(_percentile(values_ms, 0.50)),
        "p95_ms": rounded(_percentile(values_ms, 0.95)),
        "p99_ms": rounded(_percentile(values_ms, 0.99)),
        "max_ms": rounded(max(values_ms) if values_ms else None),
    }


def _request_schedule(iterations: int) -> list[tuple[int, InputCase]]:
    schedule: list[tuple[int, InputCase]] = []
    for iteration in range(iterations):
        schedule.extend((iteration, case) for case in WARMUP_CASES)
        schedule.append((iteration, LATE_UNSEEN_CASE))
    return schedule


def _clean_recompile_message(error: Exception) -> str:
    message = str(error).split(" filename:", maxsplit=1)[0].strip()
    return message or type(error).__name__


def _execute_case(
    *,
    compiled_model: Callable[[torch.Tensor], torch.Tensor],
    eager_model: RowwiseInferenceModel,
    backend: CountingBackend,
    case: InputCase,
    device: torch.device,
    phase: str,
    iteration: int | None,
    route_contract: str,
) -> dict[str, Any]:
    rows = _make_input(case, device)
    submissions_before = backend.compile_submissions
    executions_before = backend.compiled_executions
    actual: torch.Tensor | None = None
    caught: Exception | None = None

    _sync(device)
    started = time.perf_counter_ns()
    try:
        actual = compiled_model(rows)
    except Exception as error:  # preserve expected strict stance rejection
        caught = error
    finally:
        _sync(device)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    submission_delta = backend.compile_submissions - submissions_before
    execution_delta = backend.compiled_executions - executions_before
    expected = eager_model(rows)
    _sync(device)

    if route_contract == "strict_rejection":
        if caught is None:
            raise DiagnosticError(
                f"Strict policy accepted unseen signature {case.name!r}; a visible "
                "fail_on_recompile rejection was required."
            )
        if "fail_on_recompile" not in str(caught):
            raise DiagnosticError(
                f"Unexpected error for strict unseen signature: {type(caught).__name__}: "
                f"{caught}"
            ) from caught
        if submission_delta != 0 or execution_delta != 0:
            raise DiagnosticError(
                "Strict rejection unexpectedly submitted or executed a compiled graph."
            )
        return {
            "case": case.name,
            "phase": phase,
            "iteration": iteration,
            "batch_bucket": case.batch_bucket,
            "warmed_signature": case.warmed,
            "input": {
                "shape": list(rows.shape),
                "stride": list(rows.stride()),
                "is_contiguous": rows.is_contiguous(),
            },
            "status": "rejected",
            "route": "strict_rejection",
            "wall_time_ms": round(elapsed_ms, 6),
            "backend_compile_submission_delta": submission_delta,
            "compiled_execution_delta": execution_delta,
            "fallback_observed": False,
            "output": {
                "actual": None,
                "eager_reference": _tensor_summary(expected),
                "correctness": "compiled_request_rejected",
            },
            "error": {
                "type": type(caught).__name__,
                "message": _clean_recompile_message(caught),
                "fail_on_recompile_detected": True,
            },
        }

    if caught is not None:
        raise DiagnosticError(
            f"Unexpected {type(caught).__name__} for {case.name!r}: {caught}"
        ) from caught
    if actual is None:
        raise DiagnosticError(f"No output was returned for {case.name!r}.")
    if submission_delta < 0 or execution_delta < 0:
        raise DiagnosticError("Compiler counters moved backwards.")

    fallback_observed = execution_delta == 0
    if route_contract == "compiled" and execution_delta != 1:
        raise DiagnosticError(
            f"Expected one compiled execution for {case.name!r}; observed {execution_delta}."
        )
    if route_contract == "eager_fallback" and (
        submission_delta != 0 or execution_delta != 0
    ):
        raise DiagnosticError(
            "eager_on_recompile did not take the expected counted eager fallback "
            f"for {case.name!r}."
        )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    max_abs_error = (
        0.0
        if actual.numel() == 0
        else float((actual - expected).abs().max().item())
    )

    if route_contract == "eager_fallback":
        route = "eager_fallback"
    elif submission_delta > 0:
        route = "compiled_new_variant"
    else:
        route = "compiled_cache"
    return {
        "case": case.name,
        "phase": phase,
        "iteration": iteration,
        "batch_bucket": case.batch_bucket,
        "warmed_signature": case.warmed,
        "input": {
            "shape": list(rows.shape),
            "stride": list(rows.stride()),
            "is_contiguous": rows.is_contiguous(),
        },
        "status": "ok",
        "route": route,
        "wall_time_ms": round(elapsed_ms, 6),
        "backend_compile_submission_delta": submission_delta,
        "compiled_execution_delta": execution_delta,
        "fallback_observed": fallback_observed,
        "output": {
            "actual": _tensor_summary(actual),
            "eager_reference": _tensor_summary(expected),
            "correctness": "pass",
            "rtol": 1e-5,
            "atol": 1e-6,
            "max_abs_error": max_abs_error,
        },
        "error": None,
    }


def _git_value(*args: str) -> str:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


def _provenance(device: torch.device, backend: CountingBackend) -> dict[str, Any]:
    script = Path(__file__).resolve()
    source_status = _git_value("status", "--short", "--", str(script))
    device_details: dict[str, Any] = {
        "requested": device.type,
        "resolved": str(device),
        "type": device.type,
    }
    if device.type == "cuda":
        device_details.update(
            {
                "name": torch.cuda.get_device_name(device),
                "compute_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    else:
        device_details["machine"] = platform.machine()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "source_status": source_status or "clean",
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "compiler_stance_api": "torch.compiler.set_stance",
        "implicit_suppress_errors": bool(torch._dynamo.config.suppress_errors),
        "backend": {
            "requested": backend.backend_name,
            "resolved_callable": backend.resolved_backend,
            "counter_boundary": "callable backend submission and returned callable execution",
        },
        "device": device_details,
        "worker_process_id": os.getpid(),
    }


def _scenario_report(
    *,
    scenario: str,
    policy: str,
    device_name: str,
    backend_name: str,
    iterations: int,
) -> dict[str, Any]:
    if iterations < 1:
        raise DiagnosticError("--iterations must be at least 1.")
    device = _validate_runtime(device_name, backend_name)
    torch.compiler.reset()
    try:
        model = RowwiseInferenceModel(device=device).eval()
        backend = CountingBackend(backend_name)
        setup_started = time.perf_counter_ns()
        compiled_model = torch.compile(
            model,
            backend=backend,
            fullgraph=True,
            dynamic=False,
        )
        setup_wall_time_ms = (time.perf_counter_ns() - setup_started) / 1_000_000.0

        warmup_records: list[dict[str, Any]] = []
        request_records: list[dict[str, Any]] = []
        warmup_submissions_before = backend.compile_submissions
        warmup_executions_before = backend.compiled_executions

        with torch.inference_mode():
            if scenario == "guarded_policy":
                with torch.compiler.set_stance("default"):
                    for case in WARMUP_CASES:
                        warmup_records.append(
                            _execute_case(
                                compiled_model=compiled_model,
                                eager_model=model,
                                backend=backend,
                                case=case,
                                device=device,
                                phase="warmup",
                                iteration=None,
                                route_contract="compiled",
                            )
                        )
            elif scenario != "default_compile":
                raise DiagnosticError(f"Unknown worker scenario {scenario!r}.")

            warmup_submissions = backend.compile_submissions - warmup_submissions_before
            warmup_executions = backend.compiled_executions - warmup_executions_before
            request_submissions_before = backend.compile_submissions
            request_executions_before = backend.compiled_executions

            if scenario == "default_compile":
                serving_stance = "default"
            elif policy == "strict":
                serving_stance = "fail_on_recompile"
            elif policy == "eager_on_recompile":
                serving_stance = "eager_on_recompile"
            else:
                raise DiagnosticError(f"Unknown policy {policy!r}.")

            with torch.compiler.set_stance(serving_stance):
                for iteration, case in _request_schedule(iterations):
                    if scenario == "guarded_policy" and not case.warmed:
                        route_contract = (
                            "strict_rejection"
                            if policy == "strict"
                            else "eager_fallback"
                        )
                    else:
                        route_contract = "compiled"
                    request_records.append(
                        _execute_case(
                            compiled_model=compiled_model,
                            eager_model=model,
                            backend=backend,
                            case=case,
                            device=device,
                            phase="serving",
                            iteration=iteration,
                            route_contract=route_contract,
                        )
                    )

        request_submissions = backend.compile_submissions - request_submissions_before
        request_executions = backend.compiled_executions - request_executions_before
        if scenario == "guarded_policy" and request_submissions != 0:
            raise DiagnosticError(
                f"Guarded serving submitted {request_submissions} new backend compile(s)."
            )

        successful = [record for record in request_records if record["status"] == "ok"]
        rejected = [record for record in request_records if record["status"] == "rejected"]
        fallbacks = [record for record in request_records if record["fallback_observed"]]
        correctness_errors = [
            record["output"]["max_abs_error"] for record in successful
        ]
        provenance = _provenance(device, backend)
        return {
            "schema_version": SCHEMA_VERSION,
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "classification": "compiler_lifecycle_diagnostic",
            "performance_claim": "none",
            "scenario": scenario,
            "policy": {
                "requested": "default" if scenario == "default_compile" else policy,
                "serving_stance": serving_stance,
                "fallback_opt_in": policy == "eager_on_recompile"
                and scenario == "guarded_policy",
                "implicit_fallback_allowed": False,
            },
            "model": {
                "name": "RowwiseInferenceModel",
                "mode": "eval_and_inference_mode",
                "in_features": IN_FEATURES,
                "out_features": OUT_FEATURES,
                "dtype": "torch.float32",
                "operation": "rowwise linear plus sigmoid",
                "parameters": "deterministic registered buffers; no RNG mutation",
            },
            "workload": {
                "iterations": iterations,
                "warmup_cases": [case.name for case in WARMUP_CASES]
                if scenario == "guarded_policy"
                else [],
                "serving_cases_per_iteration": [
                    *[case.name for case in WARMUP_CASES],
                    LATE_UNSEEN_CASE.name,
                ],
                "late_unseen_contract": {
                    "case": LATE_UNSEEN_CASE.name,
                    "batch_bucket": LATE_UNSEEN_CASE.batch_bucket,
                    "reason_unseen": "exact batch shape and strided signature omitted from warmup",
                },
            },
            "setup": {"torch_compile_wrapper_wall_time_ms": round(setup_wall_time_ms, 6)},
            "timing": {
                "clock": "time.perf_counter_ns host wall clock",
                "cuda_synchronization": device.type == "cuda",
                "warmup": _latency_summary(
                    [record["wall_time_ms"] for record in warmup_records]
                ),
                "serving": _latency_summary(
                    [record["wall_time_ms"] for record in request_records]
                ),
                "serving_includes_every_attempt": True,
                "serving_includes_rejections_and_fallbacks": True,
                "slo_assertion": {
                    "threshold_ms": None,
                    "violation_count": None,
                    "status": "not_configured",
                },
            },
            "backend_counts": {
                "compile_submissions": {
                    "warmup": warmup_submissions,
                    "serving": request_submissions,
                    "total": backend.compile_submissions,
                },
                "compiled_executions": {
                    "warmup": warmup_executions,
                    "serving": request_executions,
                    "total": backend.compiled_executions,
                },
                "observed_eager_fallback_requests": len(fallbacks),
            },
            "output_validation": {
                "successful_serving_requests": len(successful),
                "rejected_serving_requests": len(rejected),
                "successful_outputs_checked_against_eager": len(successful),
                "all_successful_outputs_correct": True,
                "max_abs_error": max(correctness_errors, default=0.0),
            },
            "warmup_records": warmup_records,
            "request_records": request_records,
            "provenance": provenance,
            "boundary": [
                "The eager backend diagnoses graph routing and is not performance evidence.",
                "The CUDA/Inductor mode remains a diagnostic and does not establish a speedup.",
                "Input creation and eager correctness checks are outside request timing.",
                "Per-request CUDA synchronization measures host-visible latency but changes overlap.",
            ],
        }
    finally:
        torch.compiler.reset()


def _worker_command(
    *,
    scenario: str,
    policy: str,
    device: str,
    backend: str,
    iterations: int,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker-scenario",
        scenario,
        "--policy",
        policy,
        "--device",
        device,
        "--backend",
        backend,
        "--iterations",
        str(iterations),
    ]


def _run_fresh_worker(**kwargs: Any) -> dict[str, Any]:
    command = _worker_command(**kwargs)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        stderr_tail = "\n".join(completed.stderr.splitlines()[-20:])
        raise DiagnosticError(
            f"Fresh-process {kwargs['scenario']} worker failed with exit code "
            f"{completed.returncode}:\n{stderr_tail}"
        )
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise DiagnosticError(
            f"Fresh-process {kwargs['scenario']} worker returned invalid JSON."
        ) from error


def run_diagnostic(
    *, policy: str, device: str, backend: str, iterations: int
) -> dict[str, Any]:
    _validate_runtime(device, backend)
    default_report = _run_fresh_worker(
        scenario="default_compile",
        policy=policy,
        device=device,
        backend=backend,
        iterations=iterations,
    )
    guarded_report = _run_fresh_worker(
        scenario="guarded_policy",
        policy=policy,
        device=device,
        backend=backend,
        iterations=iterations,
    )
    default_pid = default_report["provenance"]["worker_process_id"]
    guarded_pid = guarded_report["provenance"]["worker_process_id"]
    if default_pid == guarded_pid:
        raise DiagnosticError("Scenario process isolation was not established.")

    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "classification": "compiler_lifecycle_diagnostic",
        "performance_claim": "none",
        "configuration": {
            "policy": policy,
            "device": device,
            "backend": backend,
            "iterations": iterations,
            "strict_is_default": policy == "strict",
        },
        "process_isolation": {
            "strategy": "one fresh child process per scenario",
            "established": True,
            "default_compile_worker_process_id": default_pid,
            "guarded_policy_worker_process_id": guarded_pid,
        },
        "model": guarded_report["model"],
        "version": {
            "pytorch": guarded_report["provenance"]["pytorch_version"],
            "python": guarded_report["provenance"]["python_version"],
            "diagnostic": DIAGNOSTIC_VERSION,
        },
        "backend": guarded_report["provenance"]["backend"],
        "device": guarded_report["provenance"]["device"],
        "output_validation": {
            "default_compile": default_report["output_validation"],
            "guarded_policy": guarded_report["output_validation"],
        },
        "comparison": {
            "metric": "backend compile submissions during serving",
            "default_compile": default_report["backend_counts"]["compile_submissions"][
                "serving"
            ],
            "guarded_policy": guarded_report["backend_counts"]["compile_submissions"][
                "serving"
            ],
            "guarded_eager_fallback_requests": guarded_report["backend_counts"][
                "observed_eager_fallback_requests"
            ],
            "guarded_rejected_requests": guarded_report["output_validation"][
                "rejected_serving_requests"
            ],
            "latency_speedup_calculated": False,
        },
        "scenarios": {
            "default_compile": default_report,
            "guarded_policy": guarded_report,
        },
        "provenance": {
            "git_commit": guarded_report["provenance"]["git_commit"],
            "script_sha256": guarded_report["provenance"]["script_sha256"],
            "source_status": guarded_report["provenance"]["source_status"],
            "scenario_processes_are_fresh": True,
        },
        "boundary": [
            "No baseline/optimized speedup is claimed or calculated.",
            "CPU/eager is a graph-routing diagnostic, not a performance backend.",
            "CUDA/Inductor requires an explicit supported-device run.",
            "Production SLO validation still needs representative arrivals, queueing, and hardware controls.",
        ],
    }


def _write_json(path: Path, report: dict[str, Any]) -> None:
    if not path.parent.exists():
        raise DiagnosticError(f"JSON output parent directory does not exist: {path.parent}")
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _stdout_summary(
    report: dict[str, Any], *, json_path: Path | None
) -> dict[str, Any]:
    scenarios = report["scenarios"]
    return {
        "schema_version": report["schema_version"],
        "diagnostic_version": report["diagnostic_version"],
        "classification": report["classification"],
        "performance_claim": report["performance_claim"],
        "configuration": report["configuration"],
        "process_isolation": report["process_isolation"],
        "model": report["model"],
        "version": report["version"],
        "backend": report["backend"],
        "device": report["device"],
        "comparison": report["comparison"],
        "output_validation": report["output_validation"],
        "timing": {
            "default_compile": scenarios["default_compile"]["timing"],
            "guarded_policy": scenarios["guarded_policy"]["timing"],
        },
        "provenance": report["provenance"],
        "full_report_json": str(json_path) if json_path is not None else None,
        "full_request_records_emitted_to_stdout": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose default torch.compile behavior against a representative-warmup "
            "recompilation policy in fresh processes."
        )
    )
    parser.add_argument(
        "--policy",
        choices=("strict", "eager_on_recompile"),
        default="strict",
        help=(
            "Serving policy after warmup. strict maps to fail_on_recompile and is "
            "the default; eager_on_recompile is an explicit counted fallback."
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--backend", choices=("eager", "inductor"), default="eager")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--json",
        type=Path,
        help="Optionally write the complete JSON report to this path.",
    )
    parser.add_argument(
        "--_worker-scenario",
        choices=("default_compile", "guarded_policy"),
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args._worker_scenario is not None:
            report = _scenario_report(
                scenario=args._worker_scenario,
                policy=args.policy,
                device_name=args.device,
                backend_name=args.backend,
                iterations=args.iterations,
            )
            print(json.dumps(report, separators=(",", ":"), sort_keys=True))
            return 0

        report = run_diagnostic(
            policy=args.policy,
            device=args.device,
            backend=args.backend,
            iterations=args.iterations,
        )
        if args.json is not None:
            _write_json(args.json, report)
        print(
            json.dumps(
                _stdout_summary(report, json_path=args.json),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except DiagnosticError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
