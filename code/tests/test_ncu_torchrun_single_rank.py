"""Fail-closed control-plane coverage for single-profiled-rank NCU replay.

The subprocess tests use fake workers and do not represent a GPU or profiler
capture. They exercise the real rank dispatcher, generation helper, durable
gate protocol, and capture validators.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from core.profiling.ncu_torchrun_capture import (
    SINGLE_RANK_APPLICATION_KERNEL_WINDOW_CAPTURE_RECEIPT_SCHEMA,
    _all_matching_application_rows,
    _args as _capture_args,
    _parse_rank_application_generations,
    _proc_cleanup_supported,
    _single_rank_application_evidence,
    _single_rank_counter_artifact,
    _single_rank_unit_evidence,
    build_capture_plan,
    run_capture,
)
from core.profiling.ncu_torchrun_generation import GENERATION_CONFIG_SCHEMA
from core.profiling.ncu_torchrun_single_rank import (
    EXPECTED_APPLICATION_PASSES,
    GENERATION_GATE_TIMEOUT_SECONDS,
    SINGLE_RANK_APPLICATION_CONFIG_SCHEMA,
    SINGLE_RANK_GENERATION_GATE_SCHEMA,
    _wait_for_peer_gate,
    expected_target_command,
    load_single_rank_config,
    run_gated_generation,
    validate_generation_gate_artifacts,
    validate_peer_dispatch_artifact,
)
from core.profiling.profiler_config import MINIMAL_METRICS
from tests.test_ncu_torchrun_capture import (
    _all_matching_application_csv,
    _capture_environment,
    _make_git_repo,
    _synthetic_cuda_snapshot,
    _torchrun_command,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_OWNER_ID = "a" * 32
KERNEL_SELECTOR = "regex:ncclDevKernel_AllReduce.*"


def _rank_environment(rank: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "AISP_NCU_TORCHRUN_CAPTURE_ID": CAPTURE_OWNER_ID,
            "GROUP_RANK": "0",
            "LOCAL_RANK": str(rank),
            "LOCAL_WORLD_SIZE": "2",
            "RANK": str(rank),
            "WORLD_SIZE": "2",
        }
    )
    return environment


def _rank_single_plan(
    tmp_path: Path,
    *,
    target_local_rank: int = 0,
    output_dir: Path | None = None,
    application_rdzv_port_base: int = 31_400,
    **overrides: object,
):
    values: dict[str, object] = {
        "label": "rank-single-control-plane",
        "repo_root": tmp_path,
        "source": "a" * 40,
        "output_dir": output_dir or tmp_path / "capture",
        "ncu_path": tmp_path / "ncu",
        "tcp_port": None,
        "timeout_seconds": 30,
        "nccl_nvtx_includes": (),
        "torchrun_argv": _torchrun_command(tmp_path / "target.py"),
        "all_matching_kernels": False,
        "replay_mode": "application",
        "application_topology": "rank-single",
        "target_visible_device": target_local_rank,
        "kernel_name": KERNEL_SELECTOR,
        "launch_skip": None,
        "launch_count": None,
        "expected_application_passes": EXPECTED_APPLICATION_PASSES,
        "application_rdzv_port_base": application_rdzv_port_base,
        "expected_kernel_counts": (129, 129),
        "kernel_window": (128, 1),
        "report_import_timeout_seconds": 600,
    }
    values.update(overrides)
    return build_capture_plan(**values)


def _write_generation_config(
    tmp_path: Path,
    *,
    output_dir: Path,
    worker_argv: list[str] | None = None,
    master_port_base: int = 31_400,
) -> Path:
    path = tmp_path / "generation-config.json"
    payload = {
        "schema": GENERATION_CONFIG_SCHEMA,
        "worker_argv": worker_argv or [sys.executable, "-c", "pass"],
        "output_dir": str(output_dir),
        "world_size": 2,
        "expected_passes": EXPECTED_APPLICATION_PASSES,
        "master_addr": "127.0.0.1",
        "master_port_base": master_port_base,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _stock_generation_argv(config_path: Path, worker_python: str) -> list[str]:
    return [
        worker_python,
        "-m",
        "core.profiling.ncu_torchrun_generation",
        str(config_path),
    ]


def _gate_payload(
    config_path: Path,
    *,
    worker_python: str,
    generation: int,
    rank: int,
    wrapper_pid: int,
) -> dict[str, object]:
    return {
        "schema": SINGLE_RANK_GENERATION_GATE_SCHEMA,
        "capture_owner_id": CAPTURE_OWNER_ID,
        "generation_config_path": str(config_path.resolve()),
        "generation_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "expected_passes": EXPECTED_APPLICATION_PASSES,
        "generation": generation,
        "rank": rank,
        "counter_before": generation,
        "counter_after": generation + 1,
        "stock_generation_argv": _stock_generation_argv(
            config_path, worker_python
        ),
        "stock_generation_returncode": 0,
        "wrapper_pid": wrapper_pid,
        "completed_time_ns": time.time_ns(),
    }


def _write_gate_set(
    output_dir: Path,
    config_path: Path,
    *,
    worker_python: str = sys.executable,
) -> None:
    for generation in range(EXPECTED_APPLICATION_PASSES):
        for rank in (0, 1):
            payload = _gate_payload(
                config_path,
                worker_python=worker_python,
                generation=generation,
                rank=rank,
                wrapper_pid=10_000 + generation * 2 + rank,
            )
            (output_dir / f"application-generation-gate-{generation}-rank-{rank}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )


@pytest.mark.parametrize("target_local_rank", (0, 1))
def test_rank_single_plan_maps_one_target_without_bilateral_ncu_coupling(
    tmp_path: Path, target_local_rank: int
) -> None:
    plan = _rank_single_plan(tmp_path, target_local_rank=target_local_rank)

    assert plan.is_rank_single_application is True
    assert plan.is_rank_local_application is True
    assert plan.target_visible_device == target_local_rank
    assert "--communicator" not in plan.ncu_prefix
    assert not any(item.startswith("--communicator") for item in plan.ncu_prefix)
    assert "--lockstep-kernel-launch" not in plan.ncu_prefix
    metric_value = plan.ncu_prefix[plan.ncu_prefix.index("--metrics") + 1]
    assert tuple(metric_value.split(",")) == tuple(MINIMAL_METRICS)
    assert len(MINIMAL_METRICS) == 5
    assert plan.ncu_prefix[plan.ncu_prefix.index("--replay-mode") + 1] == "application"
    assert plan.ncu_prefix[plan.ncu_prefix.index("--app-replay-mode") + 1] == "strict"
    assert plan.ncu_prefix[plan.ncu_prefix.index("--app-replay-match") + 1] == "grid"

    config = plan.rank_config()
    assert config["schema"] == SINGLE_RANK_APPLICATION_CONFIG_SCHEMA
    assert config["target_local_rank"] == target_local_rank
    assert config["expected_application_passes"] == EXPECTED_APPLICATION_PASSES
    assert config["generation_worker_argv"] == [
        plan.worker_argv[0],
        "-m",
        "core.profiling.ncu_torchrun_single_rank",
        "--generation",
        str(plan.generation_config_path),
    ]

    target_command = expected_target_command(config)
    assert target_command[target_command.index("--devices") + 1] == str(
        target_local_rank
    )
    assert target_command[target_command.index("--export") + 1] == str(
        plan.report_path(target_local_rank)
    )
    assert target_command[target_command.index("--log-file") + 1] == str(
        plan.rank_log_path(target_local_rank)
    )
    assert plan.launch_argv[-4:] == (
        plan.torchrun_argv[0],
        "-m",
        "core.profiling.ncu_torchrun_single_rank",
        str(plan.rank_config_path),
    )


def test_capture_cli_accepts_rank_single_application_topology(tmp_path: Path) -> None:
    args = _capture_args(
        [
            "--label",
            "rank-single-cli",
            "--repo-root",
            str(tmp_path),
            "--source",
            "a" * 40,
            "--output-dir",
            str(tmp_path / "capture"),
            "--ncu",
            str(tmp_path / "ncu"),
            "--timeout-seconds",
            "30",
            "--replay-mode",
            "application",
            "--application-topology",
            "rank-single",
            "--target-visible-device",
            "1",
            "--expected-application-passes",
            str(EXPECTED_APPLICATION_PASSES),
            "--application-rdzv-port-base",
            "31400",
            "--expected-kernel-counts",
            "129",
            "129",
            "--kernel-window",
            "128",
            "1",
            "--kernel-name",
            KERNEL_SELECTOR,
            "--",
            *_torchrun_command(tmp_path / "target.py"),
        ]
    )

    assert args.application_topology == "rank-single"
    assert args.target_visible_device == 1
    assert args.expected_application_passes == EXPECTED_APPLICATION_PASSES
    assert args.kernel_window == [128, 1]


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"target_local_rank": 2}, "target_local_rank must be 0 or 1"),
        ({"expected_application_passes": 8}, "must be exactly 9"),
        ({"world_size": 1}, "world_size must be exactly 2"),
    ],
)
def test_single_rank_config_rejects_role_or_replay_scope_drift(
    tmp_path: Path, update: dict[str, object], match: str
) -> None:
    output_dir = tmp_path / "capture"
    output_dir.mkdir()
    generation_config = _write_generation_config(tmp_path, output_dir=output_dir)
    ncu_prefix = [
        str(tmp_path / "ncu"),
        "--clock-control",
        "none",
        "--target-processes",
        "all",
        "--replay-mode",
        "application",
        "--app-replay-mode",
        "strict",
        "--app-replay-match",
        "grid",
        "--filter-mode",
        "per-gpu",
        "--kernel-name",
        KERNEL_SELECTOR,
        "--launch-skip",
        "128",
        "--launch-count",
        "1",
        "--kill",
        "no",
        "--check-exit-code",
        "yes",
        "--metrics",
        ",".join(MINIMAL_METRICS),
    ]
    payload: dict[str, object] = {
        "schema": SINGLE_RANK_APPLICATION_CONFIG_SCHEMA,
        "ncu_prefix": ncu_prefix,
        "generation_worker_argv": [
            sys.executable,
            "-m",
            "core.profiling.ncu_torchrun_single_rank",
            "--generation",
            str(generation_config),
        ],
        "generation_config_path": str(generation_config),
        "output_dir": str(output_dir),
        "world_size": 2,
        "target_local_rank": 0,
        "expected_application_passes": EXPECTED_APPLICATION_PASSES,
    }
    payload.update(update)
    config_path = tmp_path / "rank-config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_single_rank_config(config_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "marker set differs"),
        ("mismatch", "provenance differs"),
        ("unexpected", "marker set differs"),
        ("reused-pid", "reused wrapper process ID"),
    ],
)
def test_complete_generation_gate_set_rejects_missing_mismatched_or_stale_state(
    tmp_path: Path, mutation: str, match: str
) -> None:
    output_dir = tmp_path / "capture"
    output_dir.mkdir()
    config_path = _write_generation_config(tmp_path, output_dir=output_dir)
    _write_gate_set(output_dir, config_path)

    if mutation == "missing":
        (output_dir / "application-generation-gate-4-rank-1.json").unlink()
    elif mutation == "mismatch":
        path = output_dir / "application-generation-gate-4-rank-1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["counter_after"] = 8
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "unexpected":
        source = output_dir / "application-generation-gate-8-rank-0.json"
        (output_dir / "application-generation-gate-9-rank-0.json").write_bytes(
            source.read_bytes()
        )
    else:
        path = output_dir / "application-generation-gate-8-rank-1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["wrapper_pid"] = 10_000
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        validate_generation_gate_artifacts(
            output_dir=output_dir,
            capture_owner_id=CAPTURE_OWNER_ID,
            generation_config_path=config_path,
        )


def test_generation_gate_validates_all_eighteen_owner_bound_markers(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "capture"
    output_dir.mkdir()
    config_path = _write_generation_config(tmp_path, output_dir=output_dir)
    _write_gate_set(output_dir, config_path)

    artifacts = validate_generation_gate_artifacts(
        output_dir=output_dir,
        capture_owner_id=CAPTURE_OWNER_ID,
        generation_config_path=config_path,
    )

    assert len(artifacts) == 2 * EXPECTED_APPLICATION_PASSES
    assert {
        (item["payload"]["generation"], item["payload"]["rank"])
        for item in artifacts
    } == {
        (generation, rank)
        for generation in range(EXPECTED_APPLICATION_PASSES)
        for rank in (0, 1)
    }
    assert len({item["payload"]["wrapper_pid"] for item in artifacts}) == 18


@pytest.mark.parametrize("stale_kind", ("zero-counter", "current-gate", "prior-mismatch"))
def test_gated_generation_refuses_stale_counter_or_gate_before_worker_launch(
    tmp_path: Path, stale_kind: str
) -> None:
    output_dir = tmp_path / "capture"
    output_dir.mkdir()
    invoked = tmp_path / "worker-invoked"
    worker = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(invoked)!r}).write_text('invoked')",
    ]
    config_path = _write_generation_config(
        tmp_path, output_dir=output_dir, worker_argv=worker
    )
    environment = _rank_environment(0)

    if stale_kind == "zero-counter":
        (output_dir / "rank-0-generation.counter").write_text("0\n", encoding="ascii")
        expected = "stale zero generation counter"
    elif stale_kind == "current-gate":
        payload = _gate_payload(
            config_path,
            worker_python=sys.executable,
            generation=0,
            rank=0,
            wrapper_pid=10_000,
        )
        (output_dir / "application-generation-gate-0-rank-0.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        expected = "current rank generation gate marker"
    else:
        (output_dir / "rank-0-generation.counter").write_text("1\n", encoding="ascii")
        for rank in (0, 1):
            payload = _gate_payload(
                config_path,
                worker_python=sys.executable,
                generation=0,
                rank=rank,
                wrapper_pid=10_000 + rank,
            )
            if rank == 1:
                payload["capture_owner_id"] = "b" * 32
            (output_dir / f"application-generation-gate-0-rank-{rank}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        expected = "provenance differs"

    with pytest.raises((FileExistsError, ValueError), match=expected):
        run_gated_generation(config_path, environment)
    assert not invoked.exists()


def test_generation_gate_timeout_is_bounded_and_fails_closed(tmp_path: Path) -> None:
    output_dir = tmp_path / "capture"
    output_dir.mkdir()
    config_path = _write_generation_config(tmp_path, output_dir=output_dir)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    stock_argv = _stock_generation_argv(config_path, sys.executable)

    assert GENERATION_GATE_TIMEOUT_SECONDS == 60.0
    with pytest.raises(TimeoutError, match="waiting for rank 1 generation 0"):
        _wait_for_peer_gate(
            output_dir=output_dir,
            generation=0,
            rank=0,
            capture_owner_id=CAPTURE_OWNER_ID,
            config_path=config_path,
            config_sha256=config_hash,
            stock_argv=stock_argv,
            timeout_seconds=0.01,
        )


@pytest.mark.parametrize(
    ("state", "expected_counter", "match"),
    [
        ("missing", None, "generation counter is missing or symlinked"),
        ("mismatched", 8, "generation counter is stale or differs from 9"),
        ("stale", 9, "generation counter is stale or differs from 9"),
    ],
)
def test_rank_single_counter_artifact_rejects_missing_mismatch_or_staleness(
    tmp_path: Path,
    state: str,
    expected_counter: int | None,
    match: str,
) -> None:
    output_dir = tmp_path / "capture"
    output_dir.mkdir()
    plan = _rank_single_plan(tmp_path, output_dir=output_dir)
    counter_path = output_dir / "rank-0-generation.counter"
    start_ns = 0
    if expected_counter is not None:
        counter_path.write_text(f"{expected_counter}\n", encoding="ascii")
        if state == "stale":
            start_ns = counter_path.stat().st_mtime_ns + 1

    artifact, errors = _single_rank_counter_artifact(
        plan,
        local_rank=0,
        start_ns=start_ns,
    )

    assert artifact["path"] == str(counter_path)
    assert any(match in error for error in errors)
    if expected_counter is not None:
        assert artifact["observed_generation_counter"] == expected_counter


@pytest.mark.parametrize(
    ("duration_unit", "scale_to_ms"),
    (("ns", 1e-6), ("us", 1e-3), ("ms", 1.0), ("s", 1000.0)),
)
def test_rank_single_native_units_preserve_raw_values_and_percentage_bounds(
    tmp_path: Path, duration_unit: str, scale_to_ms: float
) -> None:
    plan = _rank_single_plan(tmp_path)
    rows = list(
        csv.reader(
            io.StringIO(
                _all_matching_application_csv(
                    process_id=30_000,
                    row_ids=(0,),
                    pass_count=EXPECTED_APPLICATION_PASSES,
                    device=0,
                )
            )
        )
    )
    duration_index = rows[0].index("gpu__time_duration.avg")
    rows[1][duration_index] = duration_unit

    artifact, errors = _all_matching_application_rows(
        plan,
        rows,
        expected_device=0,
        expected_process_id=30_000,
        expected_pass_count=EXPECTED_APPLICATION_PASSES,
        pass_count_source="the paired rank-single generations",
    )
    errors.extend(_single_rank_unit_evidence(plan, artifact))

    assert errors == []
    assert artifact["metric_units"] == {
        "gpu__time_duration.avg": duration_unit,
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": "%",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "%",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed": "%",
        "sm__warps_active.avg.pct_of_peak_sustained_active": "%",
    }
    assert artifact["duration_normalization"] == [
        {
            "row_id": 0,
            "raw_value": 1000.0,
            "raw_unit": duration_unit,
            "target_unit": "ms",
            "scale": scale_to_ms,
            "normalized_value": 1000.0 * scale_to_ms,
        }
    ]

    rows[2][
        rows[0].index("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    ] = "100.0001"
    _, over_limit_errors = _all_matching_application_rows(
        plan,
        rows,
        expected_device=0,
        expected_process_id=30_000,
        expected_pass_count=EXPECTED_APPLICATION_PASSES,
        pass_count_source="the paired rank-single generations",
    )
    assert any("percentage metric exceeds 100" in error for error in over_limit_errors)


@pytest.mark.parametrize(
    ("metric", "unit", "match"),
    [
        ("gpu__time_duration.avg", "cycles", "duration unit must be one of"),
        (
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "ratio",
            "must use '%' units",
        ),
    ],
)
def test_rank_single_native_units_reject_wrong_dimensions(
    tmp_path: Path, metric: str, unit: str, match: str
) -> None:
    plan = _rank_single_plan(tmp_path)
    rows = list(
        csv.reader(
            io.StringIO(
                _all_matching_application_csv(
                    process_id=30_000,
                    row_ids=(0,),
                    pass_count=EXPECTED_APPLICATION_PASSES,
                    device=0,
                )
            )
        )
    )
    rows[1][rows[0].index(metric)] = unit
    artifact, row_errors = _all_matching_application_rows(
        plan,
        rows,
        expected_device=0,
        expected_process_id=30_000,
        expected_pass_count=EXPECTED_APPLICATION_PASSES,
        pass_count_source="the paired rank-single generations",
    )

    assert row_errors == []
    assert any(
        match in error for error in _single_rank_unit_evidence(plan, artifact)
    )


def _wait_for_path(path: Path, *, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"Timed out waiting for {path}")
        time.sleep(0.01)


def _free_port_span(count: int, *, excluded: set[int] | None = None) -> int:
    excluded = excluded or set()
    for base in range(31_000, 65_535 - count):
        if any(port in excluded for port in range(base, base + count)):
            continue
        listeners: list[socket.socket] = []
        try:
            for port in range(base, base + count):
                listener = socket.socket()
                listener.bind(("127.0.0.1", port))
                listeners.append(listener)
        except OSError:
            continue
        finally:
            for listener in listeners:
                listener.close()
        return base
    raise AssertionError(f"Could not reserve a span of {count} local ports")


def _make_concurrent_fake_python(
    path: Path,
    *,
    output_dir: Path,
    snapshot_path: Path,
) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, subprocess, sys\n"
        "from pathlib import Path\n"
        f"output_dir = Path({str(output_dir)!r})\n"
        f"snapshot_path = Path({str(snapshot_path)!r})\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['-m', 'torch.distributed.run']:\n"
        "    child = args[args.index('--no-python') + 1:]\n"
        "    children = []\n"
        "    for rank in (0, 1):\n"
        "        env = dict(os.environ)\n"
        "        env.update(\n"
        "            LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE='2',\n"
        "            LOCAL_WORLD_SIZE='2', GROUP_RANK='0',\n"
        "        )\n"
        "        children.append(subprocess.Popen(child, env=env))\n"
        "    returncodes = [child_process.wait() for child_process in children]\n"
        "    raise SystemExit(next((code for code in returncodes if code), 0))\n"
        "if args[:2] == ['-m', 'core.harness.torchrun_wrapper']:\n"
        "    from core.benchmark.run_manifest import RuntimeProvenance\n"
        "    from core.harness.torchrun_runtime_provenance import (\n"
        "        format_torchrun_runtime_provenance_frames,\n"
        "    )\n"
        "    rank = int(os.environ['LOCAL_RANK'])\n"
        "    port = int(os.environ['MASTER_PORT'])\n"
        "    payload = json.loads(snapshot_path.read_text(encoding='utf-8'))\n"
        "    payload['process_id'] = os.getpid()\n"
        "    snapshot = RuntimeProvenance.model_validate(payload)\n"
        "    receipt = {\n"
        "        'pid': os.getpid(),\n"
        "        'rank': rank,\n"
        "        'master_port': port,\n"
        "        'use_agent_store': os.environ.get('TORCHELASTIC_USE_AGENT_STORE'),\n"
        "        'restart_count_present': 'TORCHELASTIC_RESTART_COUNT' in os.environ,\n"
        "    }\n"
        "    (output_dir / f'worker-{rank}-{port}.json').write_text(\n"
        "        json.dumps(receipt, sort_keys=True), encoding='utf-8'\n"
        "    )\n"
        "    for frame in format_torchrun_runtime_provenance_frames(\n"
        "        snapshot, local_rank=rank, rank=rank, world_size=2,\n"
        "        local_world_size=2, pipe_buf=4096,\n"
        "    ):\n"
        "        if os.write(1, frame) != len(frame):\n"
        "            raise RuntimeError('short provenance write')\n"
        "    raise SystemExit(0)\n"
        "os.execv(sys.executable, [sys.executable, *args])\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path.resolve()


def _single_row_csv_template(*, device: int, duration_unit: str) -> str:
    rows = list(
        csv.reader(
            io.StringIO(
                _all_matching_application_csv(
                    process_id=123_456_789,
                    row_ids=(0,),
                    pass_count=EXPECTED_APPLICATION_PASSES,
                    device=device,
                )
            )
        )
    )
    rows[1][rows[0].index("gpu__time_duration.avg")] = duration_unit
    stream = io.StringIO()
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue()


def _make_fake_single_rank_ncu(
    path: Path,
    *,
    output_dir: Path,
    target_rank: int,
    master_port_base: int,
) -> Path:
    raw_csv_template = _single_row_csv_template(
        device=target_rank, duration_unit="us"
    )
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, subprocess, sys\n"
        "from pathlib import Path\n"
        f"output_dir = Path({str(output_dir)!r})\n"
        f"target_rank = {target_rank}\n"
        f"master_port_base = {master_port_base}\n"
        f"raw_csv_template = {raw_csv_template!r}\n"
        "args = sys.argv[1:]\n"
        "if args == ['--version']:\n"
        "    print('Version 2026.3.1.0 (test control plane)')\n"
        "    raise SystemExit(0)\n"
        "if '--import' in args:\n"
        "    receipt = json.loads(\n"
        "        (output_dir / f'worker-{target_rank}-{master_port_base}.json').read_text(\n"
        "            encoding='utf-8'\n"
        "        )\n"
        "    )\n"
        "    sys.stdout.write(raw_csv_template.replace('123456789', str(receipt['pid'])))\n"
        "    raise SystemExit(0)\n"
        "rank = int(args[args.index('--devices') + 1])\n"
        "if rank != target_rank:\n"
        "    raise RuntimeError(f'unexpected profiled rank {rank}')\n"
        "log_path = Path(args[args.index('--log-file') + 1])\n"
        "report_path = Path(args[args.index('--export') + 1])\n"
        "worker = args[args.index('--export') + 2:]\n"
        "for pass_number in range(1, 10):\n"
        "    with log_path.open('a', encoding='utf-8') as stream:\n"
        "        stream.write(\n"
        "            f'==PROF== Profiling on device {rank} : \\\"'\n"
        "            'ncclDevKernel_AllReduce_Sum_f32_RING_LL_0\\\": '\n"
        "            f'Application replay pass {pass_number}\\n'\n"
        "        )\n"
        "    completed = subprocess.run(worker, env=dict(os.environ), check=False)\n"
        "    if completed.returncode:\n"
        "        raise SystemExit(completed.returncode)\n"
        "report_path.write_bytes(b'test single-rank report\\n')\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path.resolve()


def test_gated_entrypoint_runs_nine_fresh_paired_generations_without_overlap(
    tmp_path: Path,
) -> None:
    fixture_timeout_seconds = 30.0
    output_dir = tmp_path / "capture"
    output_dir.mkdir()
    master_port_base = 31_400
    release_path = tmp_path / "release-first-rank-one-worker"
    snapshot_path = tmp_path / "runtime-snapshot.json"
    snapshot_path.write_text(
        json.dumps(_synthetic_cuda_snapshot(1).model_dump(mode="json")),
        encoding="utf-8",
    )
    worker_path = tmp_path / "worker.py"
    worker_path.write_text(
        "import json, os, time\n"
        "from pathlib import Path\n"
        "from core.benchmark.run_manifest import RuntimeProvenance\n"
        "from core.harness.torchrun_runtime_provenance import (\n"
        "    format_torchrun_runtime_provenance_frames,\n"
        ")\n"
        f"output_dir = Path({str(output_dir)!r})\n"
        f"release = Path({str(release_path)!r})\n"
        f"snapshot_path = Path({str(snapshot_path)!r})\n"
        f"first_port = {master_port_base}\n"
        "rank = int(os.environ['LOCAL_RANK'])\n"
        "port = int(os.environ['MASTER_PORT'])\n"
        "if rank == 1 and port == first_port:\n"
        f"    deadline = time.monotonic() + {fixture_timeout_seconds!r}\n"
        "    while not release.exists():\n"
        "        if time.monotonic() >= deadline:\n"
        "            raise RuntimeError('timed out waiting for test release')\n"
        "        time.sleep(0.01)\n"
        "payload = json.loads(snapshot_path.read_text(encoding='utf-8'))\n"
        "payload['process_id'] = os.getpid()\n"
        "snapshot = RuntimeProvenance.model_validate(payload)\n"
        "receipt = {\n"
        "    'pid': os.getpid(),\n"
        "    'rank': rank,\n"
        "    'master_port': port,\n"
        "    'use_agent_store': os.environ.get('TORCHELASTIC_USE_AGENT_STORE'),\n"
        "    'restart_count_present': 'TORCHELASTIC_RESTART_COUNT' in os.environ,\n"
        "}\n"
        "(output_dir / f'worker-{rank}-{port}.json').write_text(\n"
        "    json.dumps(receipt, sort_keys=True), encoding='utf-8'\n"
        ")\n"
        "for frame in format_torchrun_runtime_provenance_frames(\n"
        "    snapshot, local_rank=rank, rank=rank, world_size=2,\n"
        "    local_world_size=2, pipe_buf=4096,\n"
        "):\n"
        "    if os.write(1, frame) != len(frame):\n"
        "        raise RuntimeError('short provenance write')\n",
        encoding="utf-8",
    )
    config_path = _write_generation_config(
        tmp_path,
        output_dir=output_dir,
        worker_argv=[sys.executable, str(worker_path)],
        master_port_base=master_port_base,
    )
    combined_stdout: list[str] = []
    active_processes: list[subprocess.Popen[str]] = []

    try:
        for generation in range(EXPECTED_APPLICATION_PASSES):
            active_processes = []
            for rank in (0, 1):
                active_processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "core.profiling.ncu_torchrun_single_rank",
                            "--generation",
                            str(config_path),
                        ],
                        cwd=CODE_ROOT,
                        env=_rank_environment(rank),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        start_new_session=True,
                    )
                )
            if generation == 0:
                _wait_for_path(
                    output_dir / "application-generation-gate-0-rank-0.json",
                    timeout_seconds=fixture_timeout_seconds,
                )
                assert active_processes[0].poll() is None
                release_path.write_text("release\n", encoding="ascii")
            completed = [
                process.communicate(timeout=fixture_timeout_seconds)
                for process in active_processes
            ]
            for rank, (stdout, stderr) in enumerate(completed):
                assert active_processes[rank].returncode == 0, stderr
                combined_stdout.append(stdout)
            active_processes = []
    finally:
        for process in active_processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in active_processes:
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate(timeout=5)

    assert (output_dir / "rank-0-generation.counter").read_text(
        encoding="ascii"
    ) == "9\n"
    assert (output_dir / "rank-1-generation.counter").read_text(
        encoding="ascii"
    ) == "9\n"
    gates = validate_generation_gate_artifacts(
        output_dir=output_dir,
        capture_owner_id=CAPTURE_OWNER_ID,
        generation_config_path=config_path,
    )
    assert len(gates) == 18

    worker_receipts = [
        json.loads(
            (output_dir / f"worker-{rank}-{master_port_base + generation}.json").read_text(
                encoding="utf-8"
            )
        )
        for generation in range(EXPECTED_APPLICATION_PASSES)
        for rank in (0, 1)
    ]
    assert len({receipt["pid"] for receipt in worker_receipts}) == 18
    assert {receipt["master_port"] for receipt in worker_receipts} == set(
        range(master_port_base, master_port_base + EXPECTED_APPLICATION_PASSES)
    )
    assert all(receipt["use_agent_store"] == "False" for receipt in worker_receipts)
    assert not any(receipt["restart_count_present"] for receipt in worker_receipts)

    plan = _rank_single_plan(
        tmp_path,
        output_dir=output_dir,
        application_rdzv_port_base=master_port_base,
    )
    plan.stdout_path.write_text("".join(combined_stdout), encoding="utf-8")
    generations, stdout_artifact = _parse_rank_application_generations(plan)
    assert [item["generation"] for item in generations] == list(
        range(EXPECTED_APPLICATION_PASSES)
    )
    assert all(item["both_post_main_receipts"] is True for item in generations)
    assert all(item["cuda_runtime_parity"] is True for item in generations)
    assert len(
        {
            process_id
            for generation in generations
            for process_id in (
                *generation["bootstrap_process_ids"].values(),
                *generation["execution_process_ids"].values(),
            )
        }
    ) == EXPECTED_APPLICATION_PASSES * 4
    assert stdout_artifact["generation_marker_count"] == 36
    assert stdout_artifact["generation_pair_count"] == 18


@pytest.mark.parametrize("target_local_rank", (0, 1))
def test_rank_single_capture_runs_one_target_and_one_unprofiled_peer_to_strict_receipt(
    tmp_path: Path, target_local_rank: int
) -> None:
    if not _proc_cleanup_supported():
        pytest.skip("rank-single native import requires supported process cleanup")

    repo, target, source = _make_git_repo(tmp_path)
    output_dir = tmp_path / "capture"
    snapshot_path = tmp_path / "runtime-snapshot.json"
    snapshot_path.write_text(
        json.dumps(_synthetic_cuda_snapshot(1).model_dump(mode="json")),
        encoding="utf-8",
    )
    fake_python = _make_concurrent_fake_python(
        tmp_path / "python",
        output_dir=output_dir,
        snapshot_path=snapshot_path,
    )
    command = _torchrun_command(target, interpreter=str(fake_python))
    torchrun_port = int(
        command[command.index("--rdzv_endpoint") + 1].rsplit(":", 1)[1]
    )
    master_port_base = _free_port_span(
        EXPECTED_APPLICATION_PASSES, excluded={torchrun_port}
    )
    fake_ncu = _make_fake_single_rank_ncu(
        tmp_path / "ncu",
        output_dir=output_dir,
        target_rank=target_local_rank,
        master_port_base=master_port_base,
    )
    plan = _rank_single_plan(
        repo,
        target_local_rank=target_local_rank,
        output_dir=output_dir,
        application_rdzv_port_base=master_port_base,
        source=source,
        ncu_path=fake_ncu,
        torchrun_argv=command,
        timeout_seconds=120.0,
        report_import_timeout_seconds=30,
    )

    result = run_capture(plan, environment=_capture_environment(repo))
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    peer_rank = 1 - target_local_rank
    failure_diagnostics = json.dumps(
        {
            "result": {
                "status": result.status,
                "returncode": result.returncode,
                "timed_out": result.timed_out,
            },
            "timeout_seconds": plan.timeout_seconds,
            "receipt": receipt,
        },
        indent=2,
        sort_keys=True,
    )

    assert result.status == "CAPTURE_ARTIFACTS_PRESENT", failure_diagnostics
    assert result.returncode == 0
    assert result.timed_out is False
    assert receipt["schema"] == (
        SINGLE_RANK_APPLICATION_KERNEL_WINDOW_CAPTURE_RECEIPT_SCHEMA
    )
    assert receipt["capture_topology"] == "single-profiled-rank-application-replay"
    assert receipt["application_topology"] == "rank-single"
    assert receipt["target_local_rank"] == target_local_rank
    assert receipt["target_visible_device"] == target_local_rank
    assert receipt["unprofiled_peer_local_rank"] == peer_rank
    assert receipt["profiled_local_ranks"] == [target_local_rank]
    assert receipt["unprofiled_local_ranks"] == [peer_rank]
    assert receipt["workload_local_ranks_executed"] == [0, 1]
    assert receipt["metrics"] == list(MINIMAL_METRICS)
    assert "tcp_communicator" not in receipt
    assert "--communicator" not in receipt["argv"]
    assert "--lockstep-kernel-launch" not in receipt["argv"]
    assert receipt["cleanup"]["natural"] is True
    assert receipt["cleanup"]["drained"] is True
    assert receipt["artifact_errors"] == []
    assert receipt["replay_generation_count"] == EXPECTED_APPLICATION_PASSES
    assert receipt["generation_gate_count"] == 18
    assert receipt["paired_generations_validated"] is True
    assert receipt["generation_process_ids_disjoint"] is True
    assert receipt["target_native_report_validated"] is True
    assert receipt["target_rank_window_rows_validated"] is True
    assert receipt["peer_ncu_artifacts_absent_validated"] is True
    assert receipt["both_devices_covered_by_this_receipt"] is False
    assert receipt["all_matching_launches_covered"] is False
    assert receipt["full_coverage_qualified"] is False
    assert receipt["numerical_output_qualified"] is False
    assert receipt["performance_qualified"] is False
    assert receipt["target_expected_kernel_count"] == 1
    assert receipt["observed_kernel_count_on_target_rank"] == 1
    assert "observed_kernel_counts" not in receipt
    assert receipt["rank_config_sha256"] == hashlib.sha256(
        plan.rank_config_path.read_bytes()
    ).hexdigest()
    assert receipt["generation_config_sha256"] == hashlib.sha256(
        plan.generation_config_path.read_bytes()
    ).hexdigest()
    assert {
        gate["payload"]["capture_owner_id"] for gate in receipt["generation_gates"]
    } == {receipt["capture_owner"]["id"]}
    assert {
        gate["payload"]["generation_config_sha256"]
        for gate in receipt["generation_gates"]
    } == {receipt["generation_config_sha256"]}
    assert all(
        gate["payload"]["stock_generation_argv"]
        == list(plan.stock_generation_worker_argv)
        for gate in receipt["generation_gates"]
    )

    artifacts = {item["local_rank"]: item for item in receipt["rank_artifacts"]}
    assert artifacts[target_local_rank]["role"] == "profiled_target"
    assert artifacts[peer_rank]["role"] == "unprofiled_peer"
    assert artifacts[peer_rank]["ncu_artifacts_absent"] is True
    assert artifacts[peer_rank]["peer_dispatch"]["payload"]["role"] == (
        "unprofiled_peer"
    )
    target_report = artifacts[target_local_rank]["application_report"]
    assert target_report["metric_units"] == {
        "gpu__time_duration.avg": "us",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": "%",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "%",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed": "%",
        "sm__warps_active.avg.pct_of_peak_sustained_active": "%",
    }
    assert target_report["duration_normalization"] == [
        {
            "row_id": 0,
            "raw_value": 1000.0,
            "raw_unit": "us",
            "target_unit": "ms",
            "scale": 0.001,
            "normalized_value": 1.0,
        }
    ]
    assert plan.report_path(target_local_rank).is_file()
    assert plan.rank_log_path(target_local_rank).is_file()
    assert plan.rank_argv_path(target_local_rank).is_file()
    assert not plan.report_path(peer_rank).exists()
    assert not plan.rank_log_path(peer_rank).exists()
    assert not plan.rank_argv_path(peer_rank).exists()

    target_argv = json.loads(
        plan.rank_argv_path(target_local_rank).read_text(encoding="utf-8")
    )
    assert target_argv == expected_target_command(plan.rank_config())
    peer_receipt = validate_peer_dispatch_artifact(
        output_dir / f"rank-{peer_rank}-peer-dispatch.json",
        rank_config_path=plan.rank_config_path,
        capture_owner_id=receipt["capture_owner"]["id"],
    )
    assert peer_receipt["role"] == "unprofiled_peer"
    assert peer_receipt["state"] == "complete"
    assert peer_receipt["invocations"] == [
        {
            "invocation": invocation,
            "returncode": 0,
            "observed_generation_counter": invocation + 1,
        }
        for invocation in range(EXPECTED_APPLICATION_PASSES)
    ]


@pytest.mark.parametrize(
    ("artifact_name", "path_method"),
    [
        ("report", "report_path"),
        ("log", "rank_log_path"),
        ("argv", "rank_argv_path"),
        ("raw_csv", "rank_csv_path"),
        ("import_stderr", "rank_import_stderr_path"),
    ],
)
def test_rank_single_evidence_rejects_any_unprofiled_peer_ncu_artifact(
    tmp_path: Path,
    artifact_name: str,
    path_method: str,
) -> None:
    output_dir = tmp_path / "capture"
    output_dir.mkdir()
    plan = _rank_single_plan(tmp_path, output_dir=output_dir, target_local_rank=0)
    forbidden_path = getattr(plan, path_method)(1)
    forbidden_path.write_text("forbidden peer artifact\n", encoding="utf-8")

    _, _, _, _, imported, errors = _single_rank_application_evidence(
        plan,
        start_ns=0,
        capture_owner_id=CAPTURE_OWNER_ID,
        allow_report_import=True,
    )

    assert imported is False
    assert any(
        f"unprofiled peer unexpectedly produced rank 1 {artifact_name}" in error
        for error in errors
    )
