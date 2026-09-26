"""Collect retained evidence for multi-GPU collective diagnosis.

The ``run`` command imports PyTorch only after argument parsing and capability
checks. The ``analyze`` command uses only the Python standard library, so an
artifact can be reviewed on a CPU host without PyTorch installed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA = "aisp.collective-diagnosis.v1"
ANALYSIS_SCHEMA = "aisp.collective-diagnosis-analysis.v1"
CLASSIFICATION_DIAGNOSTIC = "diagnostic_only_noncanonical"
COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")
SCENARIOS = (
    "healthy",
    "delayed_rank",
    "competing_gpu_workload",
    "forced_dependency",
)
DTYPES = ("float32", "float16", "bfloat16")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_message_size(value: str) -> int:
    """Parse a positive byte size such as ``256KiB`` or ``4MB``."""

    normalized = value.strip().lower().replace("_", "")
    suffixes = {
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "b": 1,
    }
    multiplier = 1
    number = normalized
    for suffix in sorted(suffixes, key=len, reverse=True):
        if normalized.endswith(suffix):
            multiplier = suffixes[suffix]
            number = normalized[: -len(suffix)]
            break
    try:
        parsed = float(number)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid message size: {value}") from exc
    result = int(parsed * multiplier)
    if parsed <= 0 or result <= 0:
        raise argparse.ArgumentTypeError("message sizes must be positive")
    return result


def _comma_values(value: str, allowed: Sequence[str], label: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError(f"{label} cannot be empty")
    invalid = sorted(set(items).difference(allowed))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unsupported {label}: {', '.join(invalid)}. Choose from {', '.join(allowed)}"
        )
    return list(dict.fromkeys(items))


def _message_sizes(value: str) -> list[int]:
    return [parse_message_size(item) for item in value.split(",") if item.strip()]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collective-diagnosis",
        description=(
            "Collect or analyze noncanonical evidence for NCCL collective stalls and lost overlap."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="run under torchrun on at least two CUDA GPUs",
    )
    run_parser.add_argument("--output", type=Path, required=True, help="rank-zero JSON artifact")
    run_parser.add_argument("--workload-id", help="Declared workload identity shared with fabric collection")
    run_parser.add_argument(
        "--collectives",
        default=",".join(COLLECTIVES),
        help="comma-separated collective names",
    )
    run_parser.add_argument(
        "--message-sizes",
        default="256KiB,4MiB,32MiB",
        help="comma-separated logical bytes per rank",
    )
    run_parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help="comma-separated scenario names",
    )
    run_parser.add_argument("--warmups", type=_nonnegative_int, default=3)
    run_parser.add_argument("--rounds", type=_positive_int, default=10)
    run_parser.add_argument("--dtype", choices=DTYPES, default="float32")
    run_parser.add_argument("--injected-rank", type=int, default=-1)
    run_parser.add_argument("--delay-ms", type=float, default=20.0)
    run_parser.add_argument("--compute-matrix-size", type=_positive_int, default=2048)
    run_parser.add_argument("--compute-iterations", type=_positive_int, default=4)
    run_parser.add_argument("--competing-iterations", type=_positive_int, default=12)
    run_parser.add_argument("--timeout-seconds", type=_positive_int, default=300)
    run_parser.add_argument("--nvtx", action="store_true", help="emit NVTX ranges")
    run_parser.add_argument(
        "--profile-dir",
        type=Path,
        help="export one intrusive PyTorch profiler trace per rank",
    )
    run_parser.add_argument(
        "--clock-sync-evidence",
        type=Path,
        help="JSON evidence used only to qualify cross-host wall-clock comparisons",
    )
    run_parser.add_argument(
        "--harness-gates",
        type=Path,
        help="retain an unverified external gate receipt as supplemental provenance",
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="analyze a retained artifact without importing PyTorch",
    )
    analyze_parser.add_argument("artifact", type=Path)
    analyze_parser.add_argument("--output", type=Path, help="write analysis JSON")
    analyze_parser.add_argument("--format", choices=("json", "text"), default="json")
    analyze_parser.add_argument(
        "--max-clock-skew-ms",
        type=float,
        help="measured cross-host wall-clock error bound from an external sync check",
    )
    return parser


def _load_optional_receipt(path: Path | None, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    receipt = dict(value)
    receipt["source_path"] = str(path)
    return receipt


def _classification(_harness_gates: dict[str, Any] | None) -> str:
    return CLASSIFICATION_DIAGNOSTIC


def _skip_payload(reason: str) -> dict[str, Any]:
    return {
        "schema": ARTIFACT_SCHEMA,
        "status": "skipped",
        "classification": CLASSIFICATION_DIAGNOSTIC,
        "created_at": _utc_now(),
        "diagnostic": f"SKIPPED: {reason}",
        "measurements": [],
    }


def _emit_skip(reason: str, output: Path | None, rank: int = 0) -> int:
    payload = _skip_payload(reason)
    if rank == 0:
        if output is not None:
            _json_write(output, payload)
        print(json.dumps(payload, sort_keys=True))
    return 2


def _host_id() -> str:
    return hashlib.sha256(platform.node().encode("utf-8")).hexdigest()[:12]


def _run_read_only_command(argv: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "reason": str(exc)}
    if completed.returncode != 0:
        reason = completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}"
        return {"status": "unavailable", "reason": reason[-1000:]}
    return {"status": "captured", "stdout": completed.stdout.strip()}


def _nvidia_smi_provenance() -> dict[str, Any]:
    fields = [
        "index",
        "name",
        "uuid",
        "pci.bus_id",
        "driver_version",
        "clocks.current.sm",
        "clocks.current.memory",
        "clocks.applications.graphics",
        "clocks.applications.memory",
        "pstate",
    ]
    query = _run_read_only_command(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]
    )
    clocks: dict[str, Any]
    if query.get("status") == "captured":
        rows = list(csv.reader(query.get("stdout", "").splitlines()))
        clocks = {
            "status": "captured",
            "gpus": [
                {field: value.strip() for field, value in zip(fields, row, strict=False)}
                for row in rows
            ],
        }
    else:
        clocks = query
    topology = _run_read_only_command(["nvidia-smi", "topo", "-m"])
    return {"clocks": clocks, "topology": topology}


def _git_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("collective_diagnosis.md").resolve(),
    ]
    relative_sources = [str(path.relative_to(root)) for path in source_paths if path.exists()]
    commit = _run_read_only_command(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = _run_read_only_command(["git", "-C", str(root), "status", "--porcelain"])
    source_status = _run_read_only_command(
        ["git", "-C", str(root), "status", "--porcelain", "--", *relative_sources]
    )
    source_diff = _run_read_only_command(
        ["git", "-C", str(root), "diff", "--binary", "--", *relative_sources]
    )
    diff_text = source_diff.get("stdout", "") if source_diff.get("status") == "captured" else ""
    return {
        "commit": commit.get("stdout") if commit.get("status") == "captured" else None,
        "dirty": bool(status.get("stdout")) if status.get("status") == "captured" else None,
        "source_files": [
            {
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in source_paths
            if path.exists()
        ],
        "source_status": (
            source_status.get("stdout", "").splitlines()
            if source_status.get("status") == "captured"
            else None
        ),
        "tracked_source_diff_sha256": (
            hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
            if source_diff.get("status") == "captured"
            else None
        ),
        "tracked_source_diff_bytes": (
            len(diff_text.encode("utf-8"))
            if source_diff.get("status") == "captured"
            else None
        ),
        "status": "captured"
        if commit.get("status") == "captured" and status.get("status") == "captured"
        else "partial",
    }


def _format_nccl_version(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        return ".".join(str(part) for part in value)
    return str(value)


def _rank_provenance(torch: Any, rank: int, local_rank: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(local_rank)
    peer_access = {
        str(other): bool(torch.cuda.can_device_access_peer(local_rank, other))
        for other in range(torch.cuda.device_count())
        if other != local_rank
    }
    selected_environment = {
        key: os.environ[key]
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "NCCL_ALGO",
            "NCCL_DEBUG",
            "NCCL_IB_DISABLE",
            "NCCL_NET_GDR_LEVEL",
            "NCCL_PROTO",
            "NCCL_SOCKET_IFNAME",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            "TORCH_NCCL_BLOCKING_WAIT",
        )
        if key in os.environ
    }
    return {
        "rank": rank,
        "local_rank": local_rank,
        "host_id": _host_id(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nccl": _format_nccl_version(torch.cuda.nccl.version()),
        "device": {
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "peer_access": peer_access,
        },
        "environment": selected_environment,
    }


def _scenario_injection(
    scenario: str,
    injected_rank: int,
    delay_ms: float,
    compute_iterations: int,
    competing_iterations: int,
) -> dict[str, Any]:
    if scenario == "healthy":
        return {"type": "none", "ranks": []}
    if scenario == "delayed_rank":
        return {
            "type": "pre_enqueue_host_sleep",
            "ranks": [injected_rank],
            "configured_delay_ms": delay_ms,
        }
    if scenario == "competing_gpu_workload":
        return {
            "type": "extra_matrix_multiply_stream",
            "ranks": [injected_rank],
            "iterations": competing_iterations,
        }
    return {
        "type": "collective_stream_waits_for_compute_event",
        "ranks": "all",
        "compute_iterations": compute_iterations,
    }


def _torch_dtype(torch: Any, name: str) -> Any:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _make_collective_buffers(
    torch: Any,
    collective: str,
    requested_bytes: int,
    world_size: int,
    rank: int,
    dtype: Any,
    device: Any,
) -> dict[str, Any]:
    element_size = torch.empty((), dtype=dtype).element_size()
    logical_numel = max(1, math.ceil(requested_bytes / element_size))
    if collective == "all_to_all":
        logical_numel = math.ceil(logical_numel / world_size) * world_size
    if collective == "all_reduce":
        input_tensor = torch.empty(logical_numel, dtype=dtype, device=device)
        output: Any = input_tensor
    elif collective == "all_gather":
        input_tensor = torch.empty(logical_numel, dtype=dtype, device=device)
        output = [torch.empty_like(input_tensor) for _ in range(world_size)]
    elif collective == "reduce_scatter":
        input_tensor = torch.empty(logical_numel * world_size, dtype=dtype, device=device)
        output = torch.empty(logical_numel, dtype=dtype, device=device)
    elif collective == "all_to_all":
        input_tensor = torch.empty(logical_numel, dtype=dtype, device=device)
        output = torch.empty_like(input_tensor)
    else:
        raise ValueError(f"unsupported collective: {collective}")
    return {
        "collective": collective,
        "input": input_tensor,
        "output": output,
        "rank": rank,
        "world_size": world_size,
        "logical_numel": logical_numel,
        "element_size": element_size,
        "requested_bytes": requested_bytes,
        "logical_payload_bytes": logical_numel * element_size,
        "input_buffer_bytes": input_tensor.numel() * element_size,
        "output_buffer_bytes": (
            sum(tensor.numel() * element_size for tensor in output)
            if isinstance(output, list)
            else output.numel() * element_size
        ),
    }


def _reset_collective_buffers(torch: Any, buffers: dict[str, Any]) -> None:
    collective = buffers["collective"]
    rank = buffers["rank"]
    world_size = buffers["world_size"]
    input_tensor = buffers["input"]
    if collective in ("all_reduce", "all_gather", "reduce_scatter"):
        input_tensor.fill_(float(rank + 1))
        output = buffers["output"]
        if isinstance(output, list):
            for tensor in output:
                tensor.zero_()
        elif output is not input_tensor:
            output.zero_()
        return
    chunk = buffers["logical_numel"] // world_size
    for destination in range(world_size):
        start = destination * chunk
        input_tensor[start : start + chunk].fill_(float(rank * world_size + destination + 1))
    buffers["output"].zero_()


def _enqueue_collective(dist: Any, buffers: dict[str, Any]) -> Any:
    collective = buffers["collective"]
    if collective == "all_reduce":
        return dist.all_reduce(buffers["input"], async_op=True)
    if collective == "all_gather":
        return dist.all_gather(buffers["output"], buffers["input"], async_op=True)
    if collective == "reduce_scatter":
        return dist.reduce_scatter_tensor(
            buffers["output"],
            buffers["input"],
            async_op=True,
        )
    return dist.all_to_all_single(buffers["output"], buffers["input"], async_op=True)


def _check_collective_output(torch: Any, buffers: dict[str, Any]) -> tuple[bool, str]:
    collective = buffers["collective"]
    output = buffers["output"]
    rank = buffers["rank"]
    world_size = buffers["world_size"]
    expected_sum = float(world_size * (world_size + 1) // 2)
    if collective in ("all_reduce", "reduce_scatter"):
        target = torch.full_like(output, expected_sum)
        passed = bool(torch.allclose(output, target, rtol=1e-3, atol=1e-3))
        return passed, f"expected every output element to equal {expected_sum}"
    if collective == "all_gather":
        passed = all(
            bool(torch.allclose(tensor, torch.full_like(tensor, float(source + 1))))
            for source, tensor in enumerate(output)
        )
        return passed, "expected each gathered tensor to contain its source rank value"
    chunk = buffers["logical_numel"] // world_size
    passed = True
    for source in range(world_size):
        actual = output[source * chunk : (source + 1) * chunk]
        expected = float(source * world_size + rank + 1)
        if not bool(torch.allclose(actual, torch.full_like(actual, expected))):
            passed = False
            break
    return passed, "expected each output chunk to match its source and destination ranks"


@contextlib.contextmanager
def _nvtx_range(torch: Any, enabled: bool, label: str) -> Iterator[None]:
    if not enabled:
        yield
        return
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _launch_matmul_work(
    torch: Any,
    stream: Any,
    operands: tuple[Any, Any, Any],
    iterations: int,
    start_event: Any,
    end_event: Any,
) -> None:
    left, right, output = operands
    with torch.cuda.stream(stream):
        start_event.record(stream)
        for _ in range(iterations):
            torch.mm(left, right, out=output)
        end_event.record(stream)


def _run_round(
    torch: Any,
    dist: Any,
    buffers: dict[str, Any],
    scenario: str,
    rank: int,
    injected_rank: int,
    delay_ms: float,
    compute_iterations: int,
    competing_iterations: int,
    streams: dict[str, Any],
    operands: tuple[Any, Any, Any],
    competing_operands: tuple[Any, Any, Any],
    nvtx: bool,
    phase: str,
    round_index: int,
) -> tuple[dict[str, Any], bool, str]:
    _reset_collective_buffers(torch, buffers)
    torch.cuda.synchronize()
    dist.barrier()

    ready_mono_ns = time.monotonic_ns()
    ready_wall_ns = time.time_ns()
    compute_start = torch.cuda.Event(enable_timing=True)
    compute_end = torch.cuda.Event(enable_timing=True)
    collective_start = torch.cuda.Event(enable_timing=True)
    collective_end = torch.cuda.Event(enable_timing=True)
    competing_start = torch.cuda.Event(enable_timing=True)
    competing_end = torch.cuda.Event(enable_timing=True)

    label = (
        f"collective_diagnosis/{phase}/{buffers['collective']}/"
        f"{buffers['logical_payload_bytes']}/{scenario}/round_{round_index}"
    )
    with _nvtx_range(torch, nvtx, label):
        _launch_matmul_work(
            torch,
            streams["compute"],
            operands,
            compute_iterations,
            compute_start,
            compute_end,
        )
        competing_launched = scenario == "competing_gpu_workload" and rank == injected_rank
        if competing_launched:
            _launch_matmul_work(
                torch,
                streams["competing"],
                competing_operands,
                competing_iterations,
                competing_start,
                competing_end,
            )
        if scenario == "delayed_rank" and rank == injected_rank:
            time.sleep(delay_ms / 1000.0)

        enqueue_mono_ns = time.monotonic_ns()
        enqueue_wall_ns = time.time_ns()
        with torch.cuda.stream(streams["collective"]):
            if scenario == "forced_dependency":
                streams["collective"].wait_event(compute_end)
            collective_start.record(streams["collective"])
            work = _enqueue_collective(dist, buffers)
            # NCCL runs on process-group streams. wait() adds the completion
            # dependency to this stream before the terminal timing event.
            work.wait()
            collective_end.record(streams["collective"])

        collective_end.synchronize()
        completion_mono_ns = time.monotonic_ns()
        completion_wall_ns = time.time_ns()
        compute_end.synchronize()
        if competing_launched:
            competing_end.synchronize()

    step_completion_mono_ns = time.monotonic_ns()
    step_completion_wall_ns = time.time_ns()
    passed, check = _check_collective_output(torch, buffers)
    record = {
        "phase": phase,
        "round": round_index,
        "ready_mono_ns": ready_mono_ns,
        "ready_wall_time_ns": ready_wall_ns,
        "enqueue_mono_ns": enqueue_mono_ns,
        "enqueue_wall_time_ns": enqueue_wall_ns,
        "completion_mono_ns": completion_mono_ns,
        "completion_wall_time_ns": completion_wall_ns,
        "step_completion_mono_ns": step_completion_mono_ns,
        "step_completion_wall_time_ns": step_completion_wall_ns,
        "readiness_to_enqueue_ms": (enqueue_mono_ns - ready_mono_ns) / 1_000_000.0,
        "enqueue_to_completion_ms": (completion_mono_ns - enqueue_mono_ns) / 1_000_000.0,
        "cuda_collective_ms": collective_start.elapsed_time(collective_end),
        "compute_cuda_ms": compute_start.elapsed_time(compute_end),
        "competing_cuda_ms": (
            competing_start.elapsed_time(competing_end) if competing_launched else None
        ),
        "step_ms": (step_completion_mono_ns - ready_mono_ns) / 1_000_000.0,
        "correct": passed,
    }
    return record, passed, check


def _new_experiment(
    collective: str,
    message_size: int,
    buffers: dict[str, Any],
    scenario: str,
    injection: dict[str, Any],
    world_size: int,
    dtype: str,
) -> dict[str, Any]:
    return {
        "collective": collective,
        "requested_message_size_bytes": message_size,
        "logical_payload_bytes_per_rank": buffers["logical_payload_bytes"],
        "input_buffer_bytes_per_rank": buffers["input_buffer_bytes"],
        "output_buffer_bytes_per_rank": buffers["output_buffer_bytes"],
        "world_size": world_size,
        "dtype": dtype,
        "scenario": scenario,
        "injection": injection,
        "comparison_key": {
            "collective": collective,
            "logical_payload_bytes_per_rank": buffers["logical_payload_bytes"],
            "world_size": world_size,
            "dtype": dtype,
        },
        "warmup_rounds": [],
        "rounds": [],
        "correctness": {"passed": True, "checks": []},
    }


def _profiler(torch: Any, enabled: bool) -> Any:
    if not enabled:
        return contextlib.nullcontext(None)
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    )


def _execute_sweep(torch: Any, dist: Any, args: argparse.Namespace) -> dict[str, Any]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    dtype = _torch_dtype(torch, args.dtype)
    injected_rank = args.injected_rank if args.injected_rank >= 0 else world_size - 1
    if injected_rank >= world_size:
        raise ValueError(f"injected rank {injected_rank} is outside world size {world_size}")
    if args.delay_ms < 0:
        raise ValueError("delay-ms cannot be negative")

    streams = {
        "compute": torch.cuda.Stream(device=device),
        "collective": torch.cuda.Stream(device=device),
        "competing": torch.cuda.Stream(device=device),
    }
    matrix_size = args.compute_matrix_size
    operands = (
        torch.randn((matrix_size, matrix_size), device=device, dtype=dtype),
        torch.randn((matrix_size, matrix_size), device=device, dtype=dtype),
        torch.empty((matrix_size, matrix_size), device=device, dtype=dtype),
    )
    competing_operands = (
        torch.randn((matrix_size, matrix_size), device=device, dtype=dtype),
        torch.randn((matrix_size, matrix_size), device=device, dtype=dtype),
        torch.empty((matrix_size, matrix_size), device=device, dtype=dtype),
    )
    torch.cuda.synchronize()

    experiments: dict[tuple[str, int, str], dict[str, Any]] = {}
    profile_enabled = args.profile_dir is not None
    with _profiler(torch, profile_enabled) as active_profiler:
        for collective in args.collective_values:
            for message_size in args.message_size_values:
                buffers = _make_collective_buffers(
                    torch,
                    collective,
                    message_size,
                    world_size,
                    rank,
                    dtype,
                    device,
                )
                for scenario in args.scenario_values:
                    injection = _scenario_injection(
                        scenario,
                        injected_rank,
                        args.delay_ms,
                        args.compute_iterations,
                        args.competing_iterations,
                    )
                    experiments[(collective, message_size, scenario)] = _new_experiment(
                        collective,
                        message_size,
                        buffers,
                        scenario,
                        injection,
                        world_size,
                        args.dtype,
                    )
                    for warmup_index in range(args.warmups):
                        record, passed, check = _run_round(
                            torch,
                            dist,
                            buffers,
                            scenario,
                            rank,
                            injected_rank,
                            args.delay_ms,
                            args.compute_iterations,
                            args.competing_iterations,
                            streams,
                            operands,
                            competing_operands,
                            args.nvtx,
                            "warmup",
                            warmup_index,
                        )
                        experiment = experiments[(collective, message_size, scenario)]
                        experiment["warmup_rounds"].append(record)
                        experiment["correctness"]["passed"] &= passed
                        if check not in experiment["correctness"]["checks"]:
                            experiment["correctness"]["checks"].append(check)
                        if active_profiler is not None:
                            active_profiler.step()

                for round_index in range(args.rounds):
                    shift = round_index % len(args.scenario_values)
                    ordered = args.scenario_values[shift:] + args.scenario_values[:shift]
                    for scenario in ordered:
                        record, passed, check = _run_round(
                            torch,
                            dist,
                            buffers,
                            scenario,
                            rank,
                            injected_rank,
                            args.delay_ms,
                            args.compute_iterations,
                            args.competing_iterations,
                            streams,
                            operands,
                            competing_operands,
                            args.nvtx,
                            "measured",
                            round_index,
                        )
                        experiment = experiments[(collective, message_size, scenario)]
                        experiment["rounds"].append(record)
                        experiment["correctness"]["passed"] &= passed
                        if check not in experiment["correctness"]["checks"]:
                            experiment["correctness"]["checks"].append(check)
                        if active_profiler is not None:
                            active_profiler.step()

    trace_path = None
    if profile_enabled and active_profiler is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        trace_path = args.profile_dir / f"collective-diagnosis-rank-{rank}.json"
        active_profiler.export_chrome_trace(str(trace_path))
    return {
        "rank": rank,
        "local_rank": local_rank,
        "host_id": _host_id(),
        "profiler_trace": str(trace_path) if trace_path is not None else None,
        "experiments": list(experiments.values()),
    }


def _run_command(args: argparse.Namespace) -> int:
    env_rank = int(os.environ.get("RANK", "0"))
    try:
        env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return _emit_skip("WORLD_SIZE is not an integer", args.output, env_rank)
    if env_world_size < 2 or "LOCAL_RANK" not in os.environ:
        return _emit_skip(
            "run requires torchrun with WORLD_SIZE >= 2 and one CUDA device per local rank",
            args.output,
            env_rank,
        )
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        return _emit_skip(f"PyTorch with distributed CUDA support is unavailable: {exc}", args.output, env_rank)
    if not torch.cuda.is_available():
        return _emit_skip("CUDA is unavailable", args.output, env_rank)
    if not dist.is_available() or not dist.is_nccl_available():
        return _emit_skip("the PyTorch NCCL backend is unavailable", args.output, env_rank)
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        return _emit_skip(
            f"LOCAL_RANK={local_rank} has no visible CUDA device",
            args.output,
            env_rank,
        )
    args.collective_values = _comma_values(args.collectives, COLLECTIVES, "collective")
    args.scenario_values = _comma_values(args.scenarios, SCENARIOS, "scenario")
    args.message_size_values = _message_sizes(args.message_sizes)
    if not args.message_size_values:
        raise ValueError("message-sizes cannot be empty")
    clock_sync_evidence = _load_optional_receipt(args.clock_sync_evidence, "clock sync evidence")
    harness_gates = _load_optional_receipt(args.harness_gates, "harness gates")

    torch.cuda.set_device(local_rank)
    initialized = False
    try:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=args.timeout_seconds),
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        initialized = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size < 2:
            return _emit_skip("initialized NCCL world has fewer than two ranks", args.output, rank)

        run_id_holder = [uuid.uuid4().hex if rank == 0 else None]
        dist.broadcast_object_list(run_id_holder, src=0)
        run_id = str(run_id_holder[0])
        local_provenance = _rank_provenance(torch, rank, local_rank)
        provenance_by_rank: list[Any] = [None] * world_size
        dist.all_gather_object(provenance_by_rank, local_provenance)

        local_result = _execute_sweep(torch, dist, args)
        rank_results: list[Any] = [None] * world_size
        dist.all_gather_object(rank_results, local_result)

        all_correct = all(
            experiment["correctness"]["passed"]
            for rank_result in rank_results
            for experiment in rank_result["experiments"]
        )
        classification = _classification(harness_gates)
        artifact = {
            "schema": ARTIFACT_SCHEMA,
            "status": "completed" if all_correct else "failed",
            "classification": classification,
            "run_id": run_id,
            "created_at": _utc_now(),
            "scope": "diagnostic",
            "claim_boundary": (
                "Durations locate symptoms and compare explicit injections. They do not identify "
                "fabric causation without independent topology, counter, and profiler evidence."
            ),
            "config": {
                "workload_id": args.workload_id,
                "collectives": args.collective_values,
                "message_sizes_bytes": args.message_size_values,
                "scenarios": args.scenario_values,
                "warmups": args.warmups,
                "rounds": args.rounds,
                "dtype": args.dtype,
                "injected_rank": (
                    args.injected_rank if args.injected_rank >= 0 else world_size - 1
                ),
                "delay_ms": args.delay_ms,
                "compute_matrix_size": args.compute_matrix_size,
                "compute_iterations": args.compute_iterations,
                "competing_iterations": args.competing_iterations,
                "scenario_order": "rotated_interleaving",
                "nvtx": args.nvtx,
                "profiler_enabled": args.profile_dir is not None,
                "profiler_intrusive": args.profile_dir is not None,
            },
            "validity": {
                "correctness_passed": all_correct,
                "harness_gates": harness_gates,
                "harness_gates_verified": False,
                "classification": classification,
            },
            "provenance": {
                "git": _git_provenance() if rank == 0 else None,
                "world_size": world_size,
                "rank_runtime": provenance_by_rank,
                "nvidia_smi": _nvidia_smi_provenance() if rank == 0 else None,
                "clock_sync_evidence": clock_sync_evidence,
                "timestamp_policy": (
                    "Monotonic markers are comparable only within one host. Cross-host wall-clock "
                    "spread requires explicit synchronization evidence."
                ),
            },
            "rank_results": rank_results,
        }
        if rank == 0:
            _json_write(args.output, artifact)
            print(
                json.dumps(
                    {
                        "status": artifact["status"],
                        "classification": classification,
                        "run_id": run_id,
                        "artifact": str(args.output),
                    },
                    sort_keys=True,
                )
            )
        status_holder = [artifact["status"] if rank == 0 else None]
        dist.broadcast_object_list(status_holder, src=0)
        return 0 if status_holder[0] == "completed" else 1
    finally:
        if initialized:
            dist.destroy_process_group()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "p50": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _artifact_groups(artifact: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    groups: dict[tuple[str, int, str], dict[str, Any]] = {}
    for rank_result in artifact.get("rank_results", []):
        rank = rank_result.get("rank")
        host_id = rank_result.get("host_id")
        for experiment in rank_result.get("experiments", []):
            key = (
                str(experiment["collective"]),
                int(experiment["logical_payload_bytes_per_rank"]),
                str(experiment["scenario"]),
            )
            group = groups.setdefault(
                key,
                {
                    "experiments": [],
                    "rounds": [],
                    "warmup_rounds": [],
                    "ranks": set(),
                    "hosts": set(),
                },
            )
            group["experiments"].append(experiment)
            group["ranks"].add(rank)
            group["hosts"].add(host_id)
            for record in experiment.get("rounds", []):
                group["rounds"].append({**record, "rank": rank, "host_id": host_id})
            for record in experiment.get("warmup_rounds", []):
                group["warmup_rounds"].append({**record, "rank": rank, "host_id": host_id})
    return groups


def _arrival_timing(
    group: dict[str, Any],
    clock_sync_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    hosts = {host for host in group["hosts"] if host is not None}
    if len(hosts) <= 1:
        field = "enqueue_mono_ns"
        method = "same_host_monotonic"
        uncertainty_ms = 0.0
    elif (
        isinstance(clock_sync_evidence, dict)
        and clock_sync_evidence.get("synchronized") is True
        and isinstance(clock_sync_evidence.get("max_error_ms"), int | float)
    ):
        field = "enqueue_wall_time_ns"
        method = "cross_host_wall_clock_with_sync_evidence"
        uncertainty_ms = float(clock_sync_evidence["max_error_ms"])
    else:
        return {
            "available": False,
            "enqueue_spread_ms": None,
            "reason": (
                "Cross-host enqueue spread is not computed without explicit clock "
                "synchronization evidence and a numeric max_error_ms."
            ),
        }
    by_round: dict[int, list[int]] = {}
    for record in group["rounds"]:
        timestamp = record.get(field)
        if isinstance(timestamp, int):
            by_round.setdefault(int(record["round"]), []).append(timestamp)
    spreads = [
        (max(values) - min(values)) / 1_000_000.0
        for values in by_round.values()
        if len(values) == len(group["ranks"])
    ]
    return {
        "available": bool(spreads),
        "method": method,
        "clock_uncertainty_ms": uncertainty_ms,
        "enqueue_spread_ms": _distribution(spreads),
        "meaning": "Observed enqueue timestamp spread. It is not proof of network arrival order.",
    }


def _group_summary(
    key: tuple[str, int, str],
    group: dict[str, Any],
    clock_sync_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    experiments = group["experiments"]
    signatures = {
        json.dumps(experiment.get("comparison_key"), sort_keys=True)
        for experiment in experiments
    }
    expected_ranks = experiments[0].get("world_size") if experiments else None
    correct = all(experiment.get("correctness", {}).get("passed") is True for experiment in experiments)
    rounds = group["rounds"]
    metrics = {}
    for field in (
        "readiness_to_enqueue_ms",
        "enqueue_to_completion_ms",
        "cuda_collective_ms",
        "compute_cuda_ms",
        "competing_cuda_ms",
        "step_ms",
    ):
        values = [
            float(record[field])
            for record in rounds
            if isinstance(record.get(field), int | float)
        ]
        metrics[field] = _distribution(values)
    return {
        "collective": key[0],
        "logical_payload_bytes_per_rank": key[1],
        "scenario": key[2],
        "injection": experiments[0].get("injection") if experiments else None,
        "ranks_observed": sorted(group["ranks"]),
        "expected_rank_count": expected_ranks,
        "warmup_samples": len(group["warmup_rounds"]),
        "measured_samples": len(rounds),
        "comparison_signature_consistent": len(signatures) == 1,
        "comparison_key": experiments[0].get("comparison_key") if len(signatures) == 1 else None,
        "rank_coverage_complete": len(group["ranks"]) == expected_ranks,
        "correctness_passed": correct,
        "metrics": metrics,
        "arrival_timing": _arrival_timing(group, clock_sync_evidence),
    }


def _wall_bounds(records: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    starts = [
        int(record["ready_wall_time_ns"])
        for record in records
        if isinstance(record.get("ready_wall_time_ns"), int)
    ]
    ends = [
        int(record.get("step_completion_wall_time_ns", record["completion_wall_time_ns"]))
        for record in records
        if isinstance(record.get("completion_wall_time_ns"), int)
    ]
    if not starts or not ends:
        return None, None
    return min(starts) / 1_000_000_000.0, max(ends) / 1_000_000_000.0


def _normalized_signals(
    groups: dict[tuple[str, int, str], dict[str, Any]],
    summaries: list[dict[str, Any]],
    workload_id: str | None = None,
) -> list[dict[str, Any]]:
    summary_by_key = {
        (
            summary["collective"],
            summary["logical_payload_bytes_per_rank"],
            summary["scenario"],
        ): summary
        for summary in summaries
    }
    signals: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        collective, payload_bytes, scenario = key
        summary = summary_by_key[key]
        diagnostic_path = f"ch04.collective_diagnosis/{collective}/{payload_bytes}"
        injection = summary.get("injection")
        if scenario != "healthy":
            healthy_group = groups.get((collective, payload_bytes, "healthy"))
            if healthy_group is not None:
                all_hosts = sorted(group["hosts"] | healthy_group["hosts"])
                for host_id in all_hosts:
                    scenario_records = [
                        record for record in group["rounds"] if record.get("host_id") == host_id
                    ]
                    healthy_records = [
                        record
                        for record in healthy_group["rounds"]
                        if record.get("host_id") == host_id
                    ]
                    scenario_steps = [
                        float(record["step_ms"])
                        for record in scenario_records
                        if isinstance(record.get("step_ms"), int | float)
                    ]
                    healthy_steps = [
                        float(record["step_ms"])
                        for record in healthy_records
                        if isinstance(record.get("step_ms"), int | float)
                    ]
                    ratio = _safe_ratio(
                        _percentile(scenario_steps, 0.95),
                        _percentile(healthy_steps, 0.95),
                    )
                    if ratio is None:
                        continue
                    start_unix_s, end_unix_s = _wall_bounds(scenario_records)
                    if (
                        start_unix_s is None
                        or end_unix_s is None
                        or end_unix_s <= start_unix_s
                    ):
                        continue
                    signals.append(
                        {
                            "metric": "collective.step_slowdown_ratio_vs_healthy",
                            "value": ratio,
                            "unit": "ratio",
                            "start_unix_s": start_unix_s,
                            "end_unix_s": end_unix_s,
                            "clock_domain": f"collector_wall_clock:{host_id}",
                            "scope": {
                                "host": host_id,
                                **({"workload_id": workload_id} if workload_id else {}),
                            },
                            "diagnostic_group": diagnostic_path,
                            "role": "symptom" if ratio > 1.0 else "measurement",
                            "scenario": scenario,
                            "reference_scenario": "healthy",
                            "injection": injection,
                            "collective": collective,
                            "logical_payload_bytes_per_rank": payload_bytes,
                        }
                    )
        timing = summary["arrival_timing"]
        spread = timing.get("enqueue_spread_ms")
        if timing.get("available") and isinstance(spread, dict) and spread.get("p95") is not None:
            start_unix_s, end_unix_s = _wall_bounds(group["rounds"])
            if start_unix_s is None or end_unix_s is None or end_unix_s <= start_unix_s:
                continue
            hosts = sorted(host for host in group["hosts"] if host is not None)
            if timing.get("method") == "same_host_monotonic" and len(hosts) == 1:
                clock_domain = f"collector_wall_clock:{hosts[0]}"
                scope: dict[str, Any] = {
                    "host": hosts[0],
                    **({"workload_id": workload_id} if workload_id else {}),
                }
            else:
                if not workload_id:
                    continue
                clock_domain = "cross_host_synchronized_wall"
                scope = {"workload_id": workload_id}
            signals.append(
                {
                    "metric": "collective.enqueue_spread_p95_ms",
                    "value": spread["p95"],
                    "unit": "ms",
                    "start_unix_s": start_unix_s,
                    "end_unix_s": end_unix_s,
                    "clock_domain": clock_domain,
                    "scope": scope,
                    "diagnostic_group": diagnostic_path,
                    "role": "measurement",
                    "scenario": scenario,
                    "injection": injection,
                    "clock_uncertainty_ms": timing.get("clock_uncertainty_ms"),
                    "measurement_clock": timing.get("method"),
                    "collective": collective,
                    "logical_payload_bytes_per_rank": payload_bytes,
                }
            )
    return signals


def _finite_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _validate_round_record(
    record: Any,
    *,
    expected_phase: str,
    expected_round: int,
    location: str,
) -> tuple[list[str], list[str]]:
    invalid: list[str] = []
    rejected: list[str] = []
    if not isinstance(record, dict):
        return [f"{location} is not an object"], []
    if record.get("phase") != expected_phase:
        invalid.append(f"{location} phase must be {expected_phase}")
    if record.get("round") != expected_round:
        invalid.append(f"{location} round index must be {expected_round}")

    timestamp_fields = (
        "ready_mono_ns",
        "enqueue_mono_ns",
        "completion_mono_ns",
        "step_completion_mono_ns",
        "ready_wall_time_ns",
        "enqueue_wall_time_ns",
        "completion_wall_time_ns",
        "step_completion_wall_time_ns",
    )
    for field in timestamp_fields:
        value = record.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            invalid.append(f"{location} {field} must be a positive integer")
    if not invalid:
        monotonic_order = [
            record["ready_mono_ns"],
            record["enqueue_mono_ns"],
            record["completion_mono_ns"],
            record["step_completion_mono_ns"],
        ]
        wall_order = [
            record["ready_wall_time_ns"],
            record["enqueue_wall_time_ns"],
            record["completion_wall_time_ns"],
            record["step_completion_wall_time_ns"],
        ]
        if monotonic_order != sorted(monotonic_order):
            invalid.append(f"{location} monotonic markers are out of order")
        if wall_order != sorted(wall_order):
            invalid.append(f"{location} wall-clock markers are out of order")

    numeric_fields = (
        "readiness_to_enqueue_ms",
        "enqueue_to_completion_ms",
        "cuda_collective_ms",
        "compute_cuda_ms",
        "step_ms",
    )
    for field in numeric_fields:
        value = record.get(field)
        if not _finite_numeric(value) or value < 0:
            invalid.append(f"{location} {field} must be a finite nonnegative number")
    competing = record.get("competing_cuda_ms")
    if competing is not None and (not _finite_numeric(competing) or competing < 0):
        invalid.append(f"{location} competing_cuda_ms must be null or finite and nonnegative")

    marker_pairs = (
        (
            "readiness_to_enqueue_ms",
            "ready_mono_ns",
            "enqueue_mono_ns",
        ),
        (
            "enqueue_to_completion_ms",
            "enqueue_mono_ns",
            "completion_mono_ns",
        ),
        (
            "step_ms",
            "ready_mono_ns",
            "step_completion_mono_ns",
        ),
    )
    for duration_field, start_field, end_field in marker_pairs:
        duration = record.get(duration_field)
        start = record.get(start_field)
        end = record.get(end_field)
        if _finite_numeric(duration) and isinstance(start, int) and isinstance(end, int):
            marker_duration = (end - start) / 1_000_000.0
            if not math.isclose(float(duration), marker_duration, rel_tol=1e-9, abs_tol=1e-9):
                invalid.append(f"{location} {duration_field} does not match its markers")
    if record.get("correct") is not True:
        rejected.append(f"{location} output correctness failed")
    return invalid, rejected


def _validate_completed_artifact(artifact: dict[str, Any]) -> tuple[list[str], list[str]]:
    invalid: list[str] = []
    rejected: list[str] = []
    if artifact.get("status") != "completed":
        rejected.append(f"source artifact status is {artifact.get('status')!r}")
    config = artifact.get("config")
    if not isinstance(config, dict):
        return ["config must be an object"], rejected
    rounds = config.get("rounds")
    warmups = config.get("warmups")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        invalid.append("config.rounds must be a positive integer")
    if not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 0:
        invalid.append("config.warmups must be a nonnegative integer")
    declared_collectives = config.get("collectives")
    declared_sizes = config.get("message_sizes_bytes")
    declared_scenarios = config.get("scenarios")
    declared_dtype = config.get("dtype")
    if (
        not isinstance(declared_collectives, list)
        or not declared_collectives
        or any(not isinstance(value, str) or not value for value in declared_collectives)
    ):
        invalid.append("config.collectives must be a nonempty string list")
        declared_collectives = []
    if (
        not isinstance(declared_sizes, list)
        or not declared_sizes
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in declared_sizes
        )
    ):
        invalid.append("config.message_sizes_bytes must be a nonempty positive integer list")
        declared_sizes = []
    if (
        not isinstance(declared_scenarios, list)
        or not declared_scenarios
        or any(not isinstance(value, str) or not value for value in declared_scenarios)
    ):
        invalid.append("config.scenarios must be a nonempty string list")
        declared_scenarios = []
    if not isinstance(declared_dtype, str) or not declared_dtype:
        invalid.append("config.dtype must be a nonempty string")
    expected_declared_keys = {
        (collective, message_size, scenario)
        for collective in declared_collectives
        for message_size in declared_sizes
        for scenario in declared_scenarios
    }

    rank_results = artifact.get("rank_results")
    if not isinstance(rank_results, list) or not rank_results:
        return [*invalid, "rank_results must contain at least two ranks"], rejected
    first_experiments = (
        rank_results[0].get("experiments") if isinstance(rank_results[0], dict) else None
    )
    candidate_world_sizes = {
        experiment.get("world_size")
        for experiment in first_experiments or []
        if isinstance(experiment, dict)
        and isinstance(experiment.get("world_size"), int)
        and not isinstance(experiment.get("world_size"), bool)
    }
    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict):
        invalid.append("provenance must be an object")
        provenance = {}
    provenance_world_size = provenance.get("world_size")
    if isinstance(provenance_world_size, int) and not isinstance(provenance_world_size, bool):
        world_size = provenance_world_size
    elif len(candidate_world_sizes) == 1:
        world_size = next(iter(candidate_world_sizes))
    else:
        world_size = None
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 2:
        invalid.append("world_size must identify at least two ranks")
        world_size = len(rank_results)
    if len(rank_results) != world_size:
        invalid.append(
            f"rank_results has {len(rank_results)} entries but world_size is {world_size}"
        )

    rank_ids: list[int] = []
    reference_keys: set[tuple[Any, ...]] | None = None
    for rank_position, rank_result in enumerate(rank_results):
        rank_location = f"rank_results[{rank_position}]"
        if not isinstance(rank_result, dict):
            invalid.append(f"{rank_location} is not an object")
            continue
        rank = rank_result.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            invalid.append(f"{rank_location}.rank must be an integer")
            continue
        rank_ids.append(rank)
        experiments = rank_result.get("experiments")
        if not isinstance(experiments, list) or not experiments:
            invalid.append(f"{rank_location}.experiments cannot be empty")
            continue
        experiment_keys: set[tuple[Any, ...]] = set()
        actual_declared_keys: set[tuple[Any, ...]] = set()
        for experiment_position, experiment in enumerate(experiments):
            location = f"{rank_location}.experiments[{experiment_position}]"
            if not isinstance(experiment, dict):
                invalid.append(f"{location} is not an object")
                continue
            collective = experiment.get("collective")
            payload_bytes = experiment.get("logical_payload_bytes_per_rank")
            scenario = experiment.get("scenario")
            dtype = experiment.get("dtype")
            requested_bytes = experiment.get("requested_message_size_bytes")
            if not isinstance(collective, str) or not collective:
                invalid.append(f"{location}.collective must be a nonempty string")
            if (
                not isinstance(payload_bytes, int)
                or isinstance(payload_bytes, bool)
                or payload_bytes <= 0
            ):
                invalid.append(
                    f"{location}.logical_payload_bytes_per_rank must be a positive integer"
                )
            if not isinstance(scenario, str) or not scenario:
                invalid.append(f"{location}.scenario must be a nonempty string")
            if not isinstance(dtype, str) or not dtype:
                invalid.append(f"{location}.dtype must be a nonempty string")
            elif dtype != declared_dtype:
                invalid.append(f"{location}.dtype does not match config.dtype")
            if (
                not isinstance(requested_bytes, int)
                or isinstance(requested_bytes, bool)
                or requested_bytes <= 0
            ):
                invalid.append(
                    f"{location}.requested_message_size_bytes must be a positive integer"
                )
            key = (repr(collective), repr(payload_bytes), repr(scenario), repr(dtype))
            if key in experiment_keys:
                invalid.append(f"{rank_location} contains duplicate experiment {key!r}")
            experiment_keys.add(key)
            if isinstance(collective, str) and isinstance(requested_bytes, int) and isinstance(
                scenario, str
            ):
                actual_declared_keys.add((collective, requested_bytes, scenario))
            if experiment.get("world_size") != world_size:
                invalid.append(f"{location}.world_size does not match artifact world_size")
            if not isinstance(experiment.get("comparison_key"), dict):
                invalid.append(f"{location}.comparison_key must be an object")
            correctness = experiment.get("correctness")
            if not isinstance(correctness, dict) or correctness.get("passed") is not True:
                rejected.append(f"{location} aggregate correctness failed")
            measured_records = experiment.get("rounds")
            warmup_records = experiment.get("warmup_rounds")
            if not isinstance(measured_records, list):
                invalid.append(f"{location}.rounds must be a list")
                measured_records = []
            if not isinstance(warmup_records, list):
                invalid.append(f"{location}.warmup_rounds must be a list")
                warmup_records = []
            if isinstance(rounds, int) and len(measured_records) != rounds:
                invalid.append(
                    f"{location}.rounds has {len(measured_records)} records, expected {rounds}"
                )
            if isinstance(warmups, int) and len(warmup_records) != warmups:
                invalid.append(
                    f"{location}.warmup_rounds has {len(warmup_records)} records, expected {warmups}"
                )
            for record_position, record in enumerate(warmup_records):
                record_invalid, record_rejected = _validate_round_record(
                    record,
                    expected_phase="warmup",
                    expected_round=record_position,
                    location=f"{location}.warmup_rounds[{record_position}]",
                )
                invalid.extend(record_invalid)
                rejected.extend(record_rejected)
            for record_position, record in enumerate(measured_records):
                record_invalid, record_rejected = _validate_round_record(
                    record,
                    expected_phase="measured",
                    expected_round=record_position,
                    location=f"{location}.rounds[{record_position}]",
                )
                invalid.extend(record_invalid)
                rejected.extend(record_rejected)
        if reference_keys is None:
            reference_keys = experiment_keys
        elif experiment_keys != reference_keys:
            invalid.append(f"{rank_location} experiment set differs from rank zero")
        if actual_declared_keys != expected_declared_keys:
            invalid.append(f"{rank_location} experiment set does not match config sweep")
    if sorted(rank_ids) != list(range(world_size)):
        invalid.append(f"rank ids must be exactly 0 through {world_size - 1}")
    return list(dict.fromkeys(invalid))[:100], list(dict.fromkeys(rejected))[:100]


def analyze_artifact(
    artifact: dict[str, Any],
    max_clock_skew_ms: float | None = None,
) -> dict[str, Any]:
    """Analyze one retained runner artifact without importing PyTorch."""

    if artifact.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError(
            f"unsupported artifact schema {artifact.get('schema')!r}, expected {ARTIFACT_SCHEMA!r}"
        )
    if artifact.get("status") == "skipped":
        return {
            "schema": ANALYSIS_SCHEMA,
            "status": "skipped",
            "classification": CLASSIFICATION_DIAGNOSTIC,
            "source_status": "skipped",
            "diagnostic": artifact.get("diagnostic"),
            "groups": [],
            "comparisons": [],
            "signals": [],
        }
    invalid_reasons, rejection_reasons = _validate_completed_artifact(artifact)
    classification = CLASSIFICATION_DIAGNOSTIC
    if invalid_reasons:
        return {
            "schema": ANALYSIS_SCHEMA,
            "status": "invalid",
            "source_status": artifact.get("status"),
            "source_schema": artifact.get("schema"),
            "source_run_id": artifact.get("run_id"),
            "classification": classification,
            "canonical": False,
            "rejection_reasons": invalid_reasons + rejection_reasons,
            "groups": [],
            "comparisons": [],
            "signals": [],
        }
    groups = _artifact_groups(artifact)
    clock_sync = artifact.get("provenance", {}).get("clock_sync_evidence")
    if max_clock_skew_ms is not None:
        if max_clock_skew_ms < 0:
            raise ValueError("max_clock_skew_ms cannot be negative")
        clock_sync = {
            "synchronized": True,
            "max_error_ms": max_clock_skew_ms,
            "method": "operator-supplied measured bound",
        }
    summaries = [
        _group_summary(key, group, clock_sync)
        for key, group in sorted(groups.items())
    ]
    summary_by_key = {
        (
            item["collective"],
            item["logical_payload_bytes_per_rank"],
            item["scenario"],
        ): item
        for item in summaries
    }
    comparisons = []
    for summary in summaries:
        if summary["scenario"] == "healthy":
            continue
        healthy = summary_by_key.get(
            (
                summary["collective"],
                summary["logical_payload_bytes_per_rank"],
                "healthy",
            )
        )
        if healthy is None:
            comparisons.append(
                {
                    "collective": summary["collective"],
                    "logical_payload_bytes_per_rank": summary["logical_payload_bytes_per_rank"],
                    "scenario": summary["scenario"],
                    "comparable": False,
                    "reason": "healthy reference is absent",
                }
            )
            continue
        equivalent = (
            summary["expected_rank_count"] == healthy["expected_rank_count"]
            and summary["ranks_observed"] == healthy["ranks_observed"]
            and summary["comparison_key"] == healthy["comparison_key"]
            and summary["rank_coverage_complete"]
            and healthy["rank_coverage_complete"]
            and summary["correctness_passed"]
            and healthy["correctness_passed"]
            and summary["comparison_signature_consistent"]
            and healthy["comparison_signature_consistent"]
        )
        scenario_step = summary["metrics"]["step_ms"]["p50"]
        healthy_step = healthy["metrics"]["step_ms"]["p50"]
        scenario_cuda = summary["metrics"]["cuda_collective_ms"]["p50"]
        healthy_cuda = healthy["metrics"]["cuda_collective_ms"]["p50"]
        comparisons.append(
            {
                "collective": summary["collective"],
                "logical_payload_bytes_per_rank": summary["logical_payload_bytes_per_rank"],
                "scenario": summary["scenario"],
                "injected_cause": summary["injection"],
                "comparable": equivalent,
                "step_p50_ratio_vs_healthy": _safe_ratio(scenario_step, healthy_step),
                "cuda_collective_p50_ratio_vs_healthy": _safe_ratio(scenario_cuda, healthy_cuda),
                "observed_pattern": (
                    f"{summary['scenario']} was measured against the equivalent healthy reference"
                    if equivalent
                    else "comparison requirements were not met"
                ),
                "causal_boundary": (
                    "The injection is known. Duration changes alone do not identify a fabric, "
                    "transport, topology, or kernel root cause."
                ),
            }
        )
    analysis_status = "rejected" if rejection_reasons else "completed"
    return {
        "schema": ANALYSIS_SCHEMA,
        "status": analysis_status,
        "source_status": artifact.get("status"),
        "source_schema": artifact.get("schema"),
        "source_run_id": artifact.get("run_id"),
        "classification": classification,
        "canonical": False,
        "rejection_reasons": rejection_reasons,
        "groups": summaries,
        "comparisons": comparisons if not rejection_reasons else [],
        "signals": _normalized_signals(groups, summaries, artifact.get("config", {}).get("workload_id")) if not rejection_reasons else [],
        "inference_limits": [
            "Collective CUDA duration, host completion time, and full-step time answer different questions.",
            "A longer duration is not evidence of fabric causation without independent counters and traces.",
            "Cross-host enqueue spread is unavailable without explicit clock synchronization evidence.",
            "Harness-gated diagnostic evidence is still not a benchmark speedup claim.",
        ],
    }


def _format_analysis_text(analysis: dict[str, Any]) -> str:
    if analysis.get("status") == "skipped":
        return str(analysis.get("diagnostic", "SKIPPED"))
    if analysis.get("status") in ("invalid", "rejected"):
        reasons = analysis.get("rejection_reasons") or ["retained artifact did not pass analysis gates"]
        return f"{str(analysis['status']).upper()}: " + " | ".join(str(reason) for reason in reasons)
    lines = [
        f"classification: {analysis['classification']}",
        "collective  bytes/rank  scenario                 step p50 ms  CUDA p50 ms",
    ]
    for group in analysis.get("groups", []):
        step = group["metrics"]["step_ms"]["p50"]
        cuda = group["metrics"]["cuda_collective_ms"]["p50"]
        lines.append(
            f"{group['collective']:<11} {group['logical_payload_bytes_per_rank']:>10}  "
            f"{group['scenario']:<24} {step if step is not None else 'n/a':>11}  "
            f"{cuda if cuda is not None else 'n/a':>11}"
        )
    lines.append("Durations do not identify fabric causation without counters and traces.")
    return "\n".join(lines)


def _analyze_command(args: argparse.Namespace) -> int:
    try:
        artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
        if not isinstance(artifact, dict):
            raise ValueError("artifact must contain a JSON object")
        analysis = analyze_artifact(artifact, max_clock_skew_ms=args.max_clock_skew_ms)
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.output is not None:
        _json_write(args.output, analysis)
    if args.format == "text":
        print(_format_analysis_text(analysis))
    else:
        print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0 if analysis.get("status") in ("completed", "skipped") else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze":
        return _analyze_command(args)
    try:
        return _run_command(args)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
