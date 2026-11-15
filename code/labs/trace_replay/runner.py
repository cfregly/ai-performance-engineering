"""Capture complete calls and dispatch independent replays through the CLI."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from labs.nvfp4_quantization.benchmarks import NVFP4Workload, build_inputs
from labs.trace_replay.adapters import (
    CODE_ROOT,
    native_benchmark,
    output_dict,
    output_tolerances,
    reference_output,
    source_identity,
    validate_operator_inputs,
    verify_sources,
)
from labs.trace_replay.schema import (
    BUNDLE_SCHEMA,
    Bundle,
    identifier,
    load_bundle,
    load_plan,
    validate_parameters,
    validate_source,
)
from labs.trace_replay.storage import safe_file, sha256_file, write_tensor


def _now():
    return datetime.now(timezone.utc).isoformat()


class _ReplayDispositionError(RuntimeError):
    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


def _json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_failure_receipt(output_dir, event_id, phase, error, *, status=None):
    """Append a distinct receipt without replacing an earlier failure."""
    identifier(event_id)
    identifier(phase, "phase")
    status = status or ("cancelled" if isinstance(error, KeyboardInterrupt) else "failed")
    if status not in {"failed", "cancelled", "skipped", "unsupported"}:
        raise ValueError("receipt status must be failed, cancelled, skipped or unsupported")
    root = Path(output_dir) / "failures"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{event_id}-{phase}-{uuid.uuid4().hex}.json"
    _json(path, {
        "schema_version": "native-operator-failure.v1",
        "event_id": event_id,
        "phase": phase,
        "status": status,
        "error": error if isinstance(error, str) else f"{type(error).__name__}: {error}",
        "created_at": _now(),
    })
    return path


def _git_commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _generated_inputs(event):
    operation, parameters = event["operation"], event["parameters"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(event["seed"])
        if operation == "prefix_scan":
            return {"x": torch.randint(-3, 4, (parameters["numel"],), dtype=torch.int32)}
        if operation == "online_softmax":
            return {"x": torch.randn(parameters["rows"], parameters["cols"], dtype=torch.float32) * 8}
        return build_inputs(NVFP4Workload(operation.removeprefix("nvfp4."), **parameters), torch.device("cpu"))


def _store_tensors(root, event_id, group, tensors):
    result = {}
    for name, tensor in tensors.items():
        identifier(name, "tensor name")
        relative = Path("tensors") / event_id / f"{group}-{name}.bin"
        descriptor = write_tensor(root / relative, tensor)
        descriptor["path"] = relative.as_posix()
        result[name] = descriptor
    return result


def _capture(events, output_dir, device, plan_sha256=None):
    if device not in {"cpu", "cuda"}:
        raise ValueError("capture device must be cpu or cuda")
    root = Path(output_dir).absolute()
    root.mkdir(parents=True, exist_ok=False)
    operations = [event["operation"] for event in events]
    sources = source_identity(operations)
    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "created_at": _now(),
        "capture_device": device,
        "capture_backend": "cpu_reference" if device == "cpu" else "cuda_native",
        "producer": "ai-performance-engineering.native-operator-capture",
        "git_commit": _git_commit(),
        "plan_sha256": plan_sha256,
        "runtime": {"python": sys.version.split()[0], "torch": str(torch.__version__), "cuda": torch.version.cuda},
        "source_files": sources,
        "events": [],
    }
    cancelled = False
    for event in events:
        started = time.perf_counter()
        row = {key: event[key] for key in ("event_id", "operation", "parameters", "seed", "source")}
        row.update(status="failed", inputs={}, outputs={}, error=None, capture_duration_s=0.0)
        native = None
        inputs = snapshots = live = outputs = expected = reference = None
        try:
            if cancelled:
                raise KeyboardInterrupt("capture cancelled before this event started")
            inputs = event.get("input_tensors")
            if event.get("status", "completed") != "completed":
                if inputs:
                    snapshots = {name: tensor.detach().clone() for name, tensor in inputs.items()}
                    row["inputs"] = _store_tensors(root, event["event_id"], "input", snapshots)
                row["status"] = event["status"]
                row["error"] = event["error"]
                write_failure_receipt(root, event["event_id"], "source", event["error"], status=event["status"])
                continue
            if inputs is None:
                inputs = _generated_inputs(event)
            snapshots = {name: tensor.detach().clone() for name, tensor in inputs.items()}
            # Persist caller values before any operation can change live buffers.
            row["inputs"] = _store_tensors(root, event["event_id"], "input", snapshots)
            validate_operator_inputs(event["operation"], event["parameters"], snapshots)
            live = {name: tensor.to(device).clone() for name, tensor in snapshots.items()}
            if device == "cpu":
                outputs = reference_output(event["operation"], event["parameters"], live)
                row["outputs"] = _store_tensors(root, event["event_id"], "output", outputs)
            else:
                if not torch.cuda.is_available():
                    raise RuntimeError("SKIPPED: CUDA capture requires an NVIDIA GPU")
                native = native_benchmark(event["operation"], event["parameters"], live, False)
                native.setup()
                native.benchmark_fn()
                outputs = output_dict(event["operation"], native.output)
                row["outputs"] = _store_tensors(root, event["event_id"], "output", outputs)
                expected = reference_output(event["operation"], event["parameters"], live)
                for name, reference in expected.items():
                    rtol, atol = output_tolerances(event["operation"])[name]
                    torch.testing.assert_close(outputs[name], reference, rtol=rtol, atol=atol)
            if "scales" in outputs and not bool(torch.isfinite(outputs["scales"].view(torch.float8_e4m3fn).float()).all()):
                raise AssertionError("captured NVFP4 scales contain overflow or nonfinite values")
            row["status"] = "completed"
        except (Exception, KeyboardInterrupt) as exc:
            cancelled = cancelled or isinstance(exc, KeyboardInterrupt)
            row["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else "skipped" if str(exc).startswith("SKIPPED:") else "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            write_failure_receipt(root, event["event_id"], "capture", exc, status=row["status"])
        finally:
            if native is not None:
                try:
                    native.teardown()
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = f"teardown: {type(exc).__name__}: {exc}"
                    write_failure_receipt(root, event["event_id"], "teardown", exc)
            native = inputs = snapshots = live = outputs = expected = reference = None
            row["capture_duration_s"] = time.perf_counter() - started
            manifest["events"].append(row)
    if source_identity(operations) != sources:
        error = ValueError("implementation source changed during capture")
        write_failure_receipt(root, "bundle", "source", error)
        for event in manifest["events"]:
            event["status"] = "failed"
            event["error"] = str(error)
    _json(root / "manifest.json", manifest)
    with (root / "manifest.sha256").open("x", encoding="ascii") as handle:
        handle.write(sha256_file(root / "manifest.json") + "\n")
    return load_bundle(root)


def capture_plan(plan_path, output_dir, device="cpu") -> Bundle:
    """Execute an explicit generated-input recipe and retain its actual tensors."""
    plan_path = Path(plan_path)
    digest = sha256_file(plan_path)
    plan = load_plan(plan_path)
    if sha256_file(plan_path) != digest:
        raise ValueError("capture plan changed while it was being loaded")
    return _capture(plan["events"], output_dir, device, digest)


def capture_events(calls, output_dir, device="cpu") -> Bundle:
    """Retain an ordered roster of observed calls, including source failures."""
    events = []
    seen = set()
    for call in calls:
        required = {"event_id", "operation", "parameters", "inputs", "source"}
        if not isinstance(call, dict) or not required <= set(call) or set(call) - required - {"status", "error"}:
            raise ValueError("observed call fields do not match the capture API")
        event_id = identifier(call["event_id"])
        if event_id in seen:
            raise ValueError(f"duplicate event_id: {event_id}")
        seen.add(event_id)
        validate_parameters(call["operation"], call["parameters"])
        validate_source(call["source"])
        if call["source"]["kind"] != "observed_call":
            raise ValueError("caller-input capture requires source.kind=observed_call")
        status = call.get("status", "completed")
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("observed call status must be completed, failed or cancelled")
        if status != "completed" and (not isinstance(call.get("error"), str) or not call["error"]):
            raise ValueError("failed or cancelled observed calls must retain an error")
        if status == "completed" and call.get("error") is not None:
            raise ValueError("completed observed calls must not contain an error")
        if status == "completed" and not call["inputs"]:
            raise ValueError("completed observed calls require actual input tensors")
        if not isinstance(call["inputs"], dict) or any(
            not isinstance(value, torch.Tensor) for value in call["inputs"].values()
        ):
            raise ValueError("observed call inputs must be a dictionary of tensors")
        inputs = {name: value.detach().clone() for name, value in call["inputs"].items()}
        events.append({
            "event_id": event_id, "operation": call["operation"], "parameters": dict(call["parameters"]),
            "seed": None, "source": dict(call["source"]), "input_tensors": inputs,
            "status": status, "error": call.get("error"),
        })
    if not events:
        raise ValueError("observed call capture must contain at least one event")
    return _capture(events, output_dir, device)


def capture_event(operation, parameters, inputs, *, event_id, source, output_dir, device="cpu") -> Bundle:
    """Capture supplied caller values without inventing a source or RNG recipe."""
    return capture_events([{
        "event_id": event_id, "operation": operation, "parameters": parameters,
        "inputs": inputs, "source": source,
    }], output_dir, device)


def profiler_receipts(outcomes, event_root):
    """Validate explicit per-arm native reports and retain their content hashes."""
    event_root = Path(event_root).absolute()
    reports, issues = [], []
    if len(outcomes) != 1:
        return reports, ["paired replay must contain exactly one benchmark outcome"]
    baseline = outcomes[0]
    optimized = baseline.get("optimizations", [])
    if not isinstance(optimized, list) or len(optimized) != 1:
        return reports, ["paired replay must contain exactly one optimized outcome"]
    for arm, outcome in (("baseline", baseline), ("optimized", optimized[0])):
        statuses = outcome.get(f"{arm}_profiler_statuses")
        for profiler, suffix in (("nsys", ".nsys-rep"), ("ncu", ".ncu-rep")):
            if not isinstance(statuses, dict) or statuses.get(profiler) != "succeeded":
                issues.append(f"{arm}:{profiler}: native capture did not explicitly succeed")
                continue
            value = outcome.get(f"{arm}_{profiler}_rep")
            try:
                if not isinstance(value, str) or not value:
                    raise ValueError("native report path is missing")
                path = Path(value)
                if not path.is_absolute():
                    path = CODE_ROOT / path
                relative = path.absolute().relative_to(event_root / "harness")
                report = safe_file(event_root / "harness", relative.as_posix())
                if report.suffix != suffix or report.stat().st_size <= 0:
                    raise ValueError("native report must have the correct suffix and nonempty contents")
                reports.append({
                    "arm": arm, "profiler": profiler,
                    "path": report.relative_to(event_root).as_posix(),
                    "nbytes": report.stat().st_size, "sha256": sha256_file(report),
                })
            except (ValueError, OSError) as exc:
                issues.append(f"{arm}:{profiler}: {exc}")
    return reports, issues


def wait_for_owned_cli(process, receipt):
    """Forward cancellation to the owned CLI group and await its cleanup."""
    receipt["execution"] = {
        "process_id": process.pid,
        "process_group_id": process.pid,
        "start_new_session": True,
    }
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        cancellation = {
            "requested_at": _now(),
            "signal": "SIGINT",
            "process_group_id": process.pid,
            "repeated_interrupts": 0,
            "status": "waiting_for_cleanup",
        }
        receipt["cancellation"] = cancellation
        try:
            os.killpg(process.pid, signal.SIGINT)
            cancellation["signal_forwarded"] = True
        except ProcessLookupError:
            cancellation["signal_forwarded"] = False
        while True:
            try:
                returncode = process.wait()
                break
            except KeyboardInterrupt:
                # Repeated cancellation must not interrupt owned profiler cleanup.
                cancellation["repeated_interrupts"] += 1
        receipt["returncode"] = returncode
        cancellation["status"] = "cleanup_completed"
        cancellation["completed_at"] = _now()
        raise
    receipt["returncode"] = returncode
    return returncode


def replay_bundle(bundle_path, output_dir, validity_profile="strict"):
    """Dispatch one normal paired CLI run per event and retain every outcome."""
    if validity_profile not in {"strict", "portable"}:
        raise ValueError("validity_profile must be strict or portable")
    root = Path(output_dir).absolute()
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    bundle = None
    try:
        bundle = load_bundle(Path(bundle_path))
        verify_sources(bundle)
    except Exception as exc:
        receipt = write_failure_receipt(root, "bundle", "validation", exc)
        result = {"schema_version": "native-operator-replay.v1", "status": "failed", "events": [], "error": str(exc), "failure_receipt": receipt.relative_to(root).as_posix()}
        _json(root / "replay.json", result)
        return result
    cancelled = False
    for event in bundle.events:
        event_id = event["event_id"]
        event_root = root / event_id
        event_root.mkdir()
        row = {"event_id": event_id, "operation": event["operation"], "status": "failed", "source_status": event["status"]}
        try:
            if cancelled:
                raise KeyboardInterrupt("replay cancelled before this event started")
            if event["status"] != "completed":
                raise _ReplayDispositionError(f"captured event is {event['status']}: {event['error']}", event["status"])
            override = shlex.join(["--bundle", str(bundle.root), "--event", event_id])
            command = [
                sys.executable, "-m", "cli.aisp", "bench", "run",
                "--targets", "labs/trace_replay:trace_replay",
                "--target-extra-arg", f"labs/trace_replay:trace_replay={override}",
                "--profile", "minimal", "--validity-profile", validity_profile,
                "--suite-timeout", "0", "--single-gpu",
                "--artifacts-dir", str(event_root / "harness"), "--run-id", event_id,
                "--format", "json",
            ]
            _json(event_root / "command.json", {"argv": command, "bundle_sha256": bundle.sha256, "event_id": event_id})
            if not torch.cuda.is_available():
                raise _ReplayDispositionError("SKIPPED: native trace replay requires an NVIDIA CUDA GPU", "skipped")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(CODE_ROOT) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
            started = time.perf_counter()
            try:
                with (event_root / "stdout.log").open("x") as stdout, (event_root / "stderr.log").open("x") as stderr:
                    process = subprocess.Popen(
                        command, cwd=CODE_ROOT, env=environment, stdout=stdout, stderr=stderr,
                        start_new_session=True,
                    )
                    wait_for_owned_cli(process, row)
            finally:
                row["command_duration_s"] = time.perf_counter() - started
            result_paths = list((event_root / "harness").rglob("benchmark_test_results.json"))
            if len(result_paths) != 1:
                raise RuntimeError("paired benchmark did not retain exactly one result document")
            result_path = result_paths[0]
            document = json.loads(result_path.read_text())
            row["harness_result"] = result_path.relative_to(root).as_posix()
            row["harness_result_sha256"] = sha256_file(result_path)
            summaries = [item.get("summary", {}) for item in document["results"]]
            outcomes = [benchmark for chapter in document["results"] for benchmark in chapter.get("benchmarks", [])]
            row["harness_outcomes"] = [{key: outcome[key] for key in ("example", "status", "error", "skip_reason") if key in outcome} for outcome in outcomes]
            successful = sum(item.get("successful", 0) for item in summaries)
            failed = sum(item.get("failed", 0) for item in summaries)
            skipped = sum(item.get("total_skipped", 0) for item in summaries)
            if skipped and not successful and not failed:
                raise _ReplayDispositionError("paired harness skipped the required event", "skipped")
            if outcomes and all(str(outcome.get("status", "")).startswith("unsupported") for outcome in outcomes):
                raise _ReplayDispositionError("paired harness does not support the required event", "unsupported")
            reports, profiler_issues = profiler_receipts(outcomes, event_root)
            row["profiler_artifacts"] = reports
            if profiler_issues:
                row["profiler_issues"] = profiler_issues
            if process.returncode or successful != 1 or failed or skipped:
                raise RuntimeError(f"paired harness rejected the event: returncode={process.returncode}, successful={successful}, failed={failed}, skipped={skipped}")
            if profiler_issues:
                raise RuntimeError("native profiler evidence is incomplete: " + " | ".join(profiler_issues))
            row["status"] = "completed"
        except (Exception, KeyboardInterrupt) as exc:
            cancelled = cancelled or isinstance(exc, KeyboardInterrupt)
            row["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else exc.status if isinstance(exc, _ReplayDispositionError) else "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["failure_receipt"] = write_failure_receipt(root, event_id, "replay", exc, status=row["status"]).relative_to(root).as_posix()
        _json(event_root / "receipt.json", row)
        rows.append(row)
    source_error = None
    try:
        verify_sources(bundle)
    except Exception as exc:
        source_error = str(exc)
        write_failure_receipt(root, "bundle", "source", exc)
    result = {
        "schema_version": "native-operator-replay.v1", "created_at": _now(),
        "status": "completed" if source_error is None and all(row["status"] == "completed" for row in rows) else "failed",
        "bundle_sha256": bundle.sha256, "capture_backend": bundle.manifest["capture_backend"],
        "validity_profile": validity_profile, "replay_source_files": source_identity(event["operation"] for event in bundle.events),
        "events": rows,
        "semantics": "independent_operator_replay_in_record_order",
        "valid_for_serving_latency_claim": False,
    }
    if source_error is not None:
        result["error"] = source_error
    _json(root / "replay.json", result)
    return result
