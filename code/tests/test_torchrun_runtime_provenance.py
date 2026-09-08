from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
import torch

import core.harness.torchrun_runtime_provenance as runtime_transport
from core.benchmark.run_manifest import RuntimeProvenance, capture_runtime_provenance
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    LaunchVia,
    TorchrunLaunchSpec,
)
from core.harness.torchrun_runtime_provenance import (
    TORCHRUN_RUNTIME_PROVENANCE_PREFIX,
    TORCHRUN_RUNTIME_PROVENANCE_SCHEMA,
    TorchrunRuntimeProvenanceError,
    format_torchrun_runtime_provenance_frames,
    parse_torchrun_runtime_provenance_stdout,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
TEST_PIPE_BUF = 512


class _TorchrunHarnessSmokeBenchmark(BaseBenchmark):
    allow_cpu = True

    def __init__(self, target: Path, marker_dir: Path) -> None:
        super().__init__()
        self.name = "runtime-provenance-harness-smoke"
        self._target = target
        self._marker_dir = marker_dir
        self._input = torch.tensor([1.0])
        self._output = self._input + 1.0

    def benchmark_fn(self) -> None:
        self._output = self._input + 1.0

    def get_torchrun_spec(self, config: BenchmarkConfig | None = None) -> TorchrunLaunchSpec:
        return TorchrunLaunchSpec(
            script_path=self._target,
            script_args=[str(self._marker_dir)],
            parse_rank0_only=False,
            name=self.name,
        )

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        return {"input": self._input}

    def get_verify_output(self) -> torch.Tensor:
        return self._output

    def get_input_signature(self) -> dict[str, object]:
        return {"shape": tuple(self._input.shape), "dtype": str(self._input.dtype)}

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)


def _cpu_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(name, None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(CODE_ROOT)
        if not prior_pythonpath
        else os.pathsep.join((str(CODE_ROOT), prior_pythonpath))
    )
    return environment


def _wrapper_command(target: Path, *, emit: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "core.harness.torchrun_wrapper",
        "--aisp-target-script",
        str(target),
        "--aisp-expected-torch-seed",
        "42",
    ]
    if emit:
        command.append("--aisp-emit-runtime-provenance")
    return command


def _frames(
    snapshot: RuntimeProvenance,
    *,
    local_rank: int = 0,
    rank: int = 0,
    world_size: int = 1,
    local_world_size: int = 1,
    pipe_buf: int = TEST_PIPE_BUF,
) -> tuple[bytes, ...]:
    return format_torchrun_runtime_provenance_frames(
        snapshot,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        pipe_buf=pipe_buf,
    )


def _record(snapshot: RuntimeProvenance, **metadata: int) -> str:
    return b"".join(_frames(snapshot, **metadata)).decode("utf-8")


def _unchecked_single_frame(
    snapshot: RuntimeProvenance,
    *,
    envelope_process_id: int,
) -> str:
    runtime_json = json.dumps(
        snapshot.model_dump(mode="json"),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    payload = {
        "s": TORCHRUN_RUNTIME_PROVENANCE_SCHEMA,
        "r": 0,
        "lr": 0,
        "ws": 1,
        "lws": 1,
        "p": envelope_process_id,
        "i": 0,
        "n": 1,
        "h": hashlib.sha256(runtime_json).hexdigest(),
        "d": base64.b64encode(runtime_json).decode("ascii"),
    }
    return (
        TORCHRUN_RUNTIME_PROVENANCE_PREFIX
        + json.dumps(payload, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


def _unused_loopback_endpoint() -> str:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{listener.getsockname()[1]}"


def test_real_cpu_wrapper_emits_child_snapshot_and_parser_cleans_stdout(tmp_path: Path) -> None:
    target = tmp_path / "successful_target.py"
    target.write_text(
        "import os\n" "print(f'child diagnostic pid={os.getpid()}', flush=True)\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        _wrapper_command(target, emit=True),
        cwd=CODE_ROOT,
        env=_cpu_environment(),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    parsed = parse_torchrun_runtime_provenance_stdout(
        completed.stdout,
        expected_local_ranks={0},
        target="cpu",
    )
    assert parsed.raw_stdout == completed.stdout
    assert parsed.clean_stdout.startswith("child diagnostic pid=")
    assert TORCHRUN_RUNTIME_PROVENANCE_PREFIX not in parsed.clean_stdout
    assert parsed.primary_snapshot.process_id > 0
    assert parsed.primary_snapshot.process_id != os.getpid()
    assert parsed.primary_snapshot.python_executable == sys.executable
    assert parsed.execution_process_ids == {0: parsed.primary_snapshot.process_id}
    record_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(TORCHRUN_RUNTIME_PROVENANCE_PREFIX)
    ]
    assert record_lines
    pipe_buf = os.pathconf(CODE_ROOT, "PC_PIPE_BUF")
    assert all(len((line + "\n").encode("utf-8")) <= pipe_buf for line in record_lines)


def test_large_complete_inventory_round_trips_over_bounded_atomic_frames() -> None:
    snapshot = capture_runtime_provenance()
    library_versions = dict(snapshot.library_versions)
    library_versions.update(
        {
            f"synthetic-performance-library-{index:03d}": f"2026.9.{index}+{'x' * 48}"
            for index in range(160)
        }
    )
    shadowed_library_versions = dict(snapshot.shadowed_library_versions)
    shadowed_library_versions.update(
        {
            f"synthetic-performance-library-{index:03d}": [
                f"/opt/alternate-environments/{index:03d}/site-packages"
            ]
            for index in range(16)
        }
    )
    large_snapshot = snapshot.model_copy(
        update={
            "library_versions": library_versions,
            "library_versions_complete": True,
            "shadowed_library_versions": shadowed_library_versions,
        }
    )

    frames = _frames(large_snapshot)

    assert len(frames) > 2
    assert all(len(frame) <= TEST_PIPE_BUF for frame in frames)
    midpoint = len(frames) // 2
    stdout = (
        "before receipt\n"
        + b"".join(frames[:midpoint]).decode("utf-8")
        + "ordinary diagnostic\n"
        + b"".join(frames[midpoint:]).decode("utf-8")
        + "after receipt\n"
    )
    parsed = parse_torchrun_runtime_provenance_stdout(
        stdout,
        expected_local_ranks={0},
        target="cpu",
    )
    assert parsed.clean_stdout == "before receipt\nordinary diagnostic\nafter receipt\n"
    assert parsed.primary_snapshot.model_dump(mode="json") == large_snapshot.model_dump(mode="json")
    assert parsed.execution_process_ids == {0: large_snapshot.process_id}


def test_emitter_rejects_snapshot_for_a_different_process(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = capture_runtime_provenance().model_copy(update={"process_id": os.getpid() + 1})
    monkeypatch.setattr(runtime_transport, "capture_runtime_provenance", lambda: snapshot)

    with pytest.raises(TorchrunRuntimeProvenanceError, match="emitting worker"):
        runtime_transport.emit_torchrun_runtime_provenance(local_rank=0)


def test_wrapper_does_not_emit_without_explicit_flag(tmp_path: Path) -> None:
    target = tmp_path / "standalone_target.py"
    target.write_text("print('standalone output', flush=True)\n", encoding="utf-8")

    completed = subprocess.run(
        _wrapper_command(target, emit=False),
        cwd=CODE_ROOT,
        env=_cpu_environment(),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "standalone output\n"
    assert TORCHRUN_RUNTIME_PROVENANCE_PREFIX not in completed.stdout


def test_wrapper_emits_only_after_seed_validation(tmp_path: Path) -> None:
    target = tmp_path / "seed_mutating_target.py"
    target.write_text(
        "import torch\n"
        "torch.manual_seed(7)\n"
        "print('target completed before seed validation', flush=True)\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        _wrapper_command(target, emit=True),
        cwd=CODE_ROOT,
        env=_cpu_environment(),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "Seed mutation detected" in completed.stderr
    assert TORCHRUN_RUNTIME_PROVENANCE_PREFIX not in completed.stdout


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="real two-rank CPU test requires torch.distributed Gloo",
)
def test_real_two_rank_torchrun_emits_consistent_distinct_worker_snapshots(
    tmp_path: Path,
) -> None:
    target = tmp_path / "two_rank_gloo_target.py"
    target.write_text(
        "import os\n"
        "import torch\n"
        "import torch.distributed as dist\n"
        "dist.init_process_group('gloo')\n"
        "rank = int(os.environ['RANK'])\n"
        "value = torch.tensor([rank + 1], dtype=torch.int64)\n"
        "dist.all_reduce(value)\n"
        "print(f'rank={rank} all_reduce={int(value.item())}', flush=True)\n"
        "dist.destroy_process_group()\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--rdzv-backend=static",
        f"--rdzv-endpoint={_unused_loopback_endpoint()}",
        "--nproc-per-node=2",
        "--module",
        "core.harness.torchrun_wrapper",
        "--aisp-target-script",
        str(target),
        "--aisp-expected-torch-seed",
        "42",
        "--aisp-emit-runtime-provenance",
    ]

    completed = subprocess.run(
        command,
        cwd=CODE_ROOT,
        env=_cpu_environment(),
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert completed.returncode == 0, completed.stderr
    parsed = parse_torchrun_runtime_provenance_stdout(
        completed.stdout,
        expected_local_ranks={0, 1},
        target="cpu",
    )
    assert set(parsed.snapshots_by_local_rank) == {0, 1}
    assert len({snapshot.process_id for snapshot in parsed.snapshots_by_local_rank.values()}) == 2
    assert parsed.execution_process_ids == {
        local_rank: snapshot.process_id
        for local_rank, snapshot in parsed.snapshots_by_local_rank.items()
    }
    assert os.getpid() not in {
        snapshot.process_id for snapshot in parsed.snapshots_by_local_rank.values()
    }
    assert parsed.clean_stdout.count("all_reduce=3") == 2
    assert TORCHRUN_RUNTIME_PROVENANCE_PREFIX not in parsed.clean_stdout


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="real two-rank CPU test requires torch.distributed Gloo",
)
def test_benchmark_with_manifest_uses_actual_torchrun_worker_receipts(tmp_path: Path) -> None:
    marker_dir = tmp_path / "worker-markers"
    marker_dir.mkdir()
    target = tmp_path / "harness_two_rank_gloo_target.py"
    target.write_text(
        "import json\n"
        "import os\n"
        "import sys\n"
        "from datetime import datetime, timezone\n"
        "from pathlib import Path\n"
        "import torch\n"
        "import torch.distributed as dist\n"
        "dist.init_process_group('gloo')\n"
        "rank = int(os.environ['LOCAL_RANK'])\n"
        "value = torch.tensor([rank + 1], dtype=torch.int64)\n"
        "dist.all_reduce(value)\n"
        "dist.destroy_process_group()\n"
        "marker = {\n"
        "    'pid': os.getpid(),\n"
        "    'target_finished_at': datetime.now(timezone.utc).isoformat(),\n"
        "}\n"
        "Path(sys.argv[1], f'rank-{rank}.json').write_text(\n"
        "    json.dumps(marker), encoding='utf-8'\n"
        ")\n"
        "print(f'rank={rank} all_reduce={int(value.item())}', flush=True)\n",
        encoding="utf-8",
    )
    raw_capture_dir = tmp_path / "raw-subprocess"
    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=1,
        warmup=5,
        seed=42,
        launch_via=LaunchVia.TORCHRUN,
        nproc_per_node=2,
        nnodes="1",
        rdzv_backend="static",
        rdzv_endpoint=_unused_loopback_endpoint(),
        multi_gpu_required=False,
        use_subprocess=False,
        enable_profiling=False,
        lock_gpu_clocks=False,
        enforce_environment_validation=False,
        measurement_timeout_seconds=90,
        subprocess_stderr_dir=str(raw_capture_dir),
    )
    benchmark = _TorchrunHarnessSmokeBenchmark(target, marker_dir)

    run = BenchmarkHarness(config=config).benchmark_with_manifest(
        benchmark,
        run_id="torchrun-runtime-provenance-integration",
    )

    assert not run.result.errors, run.result.errors
    assert run.result.runtime_provenance is not None
    assert run.result.runtime_provenance_by_local_rank.keys() == {0, 1}
    assert run.result.runtime_provenance == run.result.runtime_provenance_by_local_rank[0]
    assert run.manifest.runtime_provenance == run.result.runtime_provenance
    assert run.result.execution_process_ids.keys() == {0, 1}
    assert len(set(run.result.execution_process_ids.values())) == 2
    assert os.getpid() not in run.result.execution_process_ids.values()

    raw_paths = list(raw_capture_dir.glob("*_subprocess.stdout.log"))
    assert len(raw_paths) == 1
    raw_stdout = raw_paths[0].read_text(encoding="utf-8")
    assert TORCHRUN_RUNTIME_PROVENANCE_PREFIX in raw_stdout
    assert raw_stdout.count("all_reduce=3") == 2
    raw_receipts = parse_torchrun_runtime_provenance_stdout(
        raw_stdout,
        expected_local_ranks={0, 1},
        target="cpu",
    )
    assert raw_receipts.execution_process_ids == run.result.execution_process_ids
    assert raw_receipts.snapshots_by_local_rank == run.result.runtime_provenance_by_local_rank

    assert run.result.validation_message is not None
    assert run.result.validation_message.count("all_reduce=3") == 2
    assert TORCHRUN_RUNTIME_PROVENANCE_PREFIX not in run.result.validation_message
    for local_rank, runtime in run.result.runtime_provenance_by_local_rank.items():
        marker = json.loads((marker_dir / f"rank-{local_rank}.json").read_text(encoding="utf-8"))
        assert marker["pid"] == run.result.execution_process_ids[local_rank]
        assert datetime.fromisoformat(runtime.captured_at) >= datetime.fromisoformat(
            marker["target_finished_at"]
        )


@pytest.mark.parametrize(
    ("raw_stdout", "message"),
    [
        ("ordinary output\n", "rank set mismatch"),
        (TORCHRUN_RUNTIME_PROVENANCE_PREFIX + "{\n", "Malformed"),
        (
            "ordinary output " + TORCHRUN_RUNTIME_PROVENANCE_PREFIX + "{}\n",
            "prefix placement",
        ),
        (TORCHRUN_RUNTIME_PROVENANCE_PREFIX + "{}", "Incomplete"),
    ],
)
def test_parser_fails_on_missing_or_malformed_frames(raw_stdout: str, message: str) -> None:
    with pytest.raises(TorchrunRuntimeProvenanceError, match=message):
        parse_torchrun_runtime_provenance_stdout(
            raw_stdout,
            expected_local_ranks={0},
            target="cpu",
        )


def test_parser_rejects_duplicate_and_partial_frame_sequences() -> None:
    snapshot = capture_runtime_provenance()
    frames = _frames(snapshot)
    record = b"".join(frames).decode("utf-8")
    with pytest.raises(TorchrunRuntimeProvenanceError, match="Duplicate"):
        parse_torchrun_runtime_provenance_stdout(
            record + record,
            expected_local_ranks={0},
            target="cpu",
        )

    assert len(frames) > 1
    with pytest.raises(TorchrunRuntimeProvenanceError, match="Incomplete"):
        parse_torchrun_runtime_provenance_stdout(
            b"".join(frames[:-1]).decode("utf-8"),
            expected_local_ranks={0},
            target="cpu",
        )


def test_parser_rejects_nonpositive_child_pid() -> None:
    snapshot = capture_runtime_provenance().model_copy(update={"process_id": 0})

    with pytest.raises(TorchrunRuntimeProvenanceError, match="process_id > 0"):
        parse_torchrun_runtime_provenance_stdout(
            _unchecked_single_frame(snapshot, envelope_process_id=1),
            expected_local_ranks={0},
            target="cpu",
        )


def test_parser_rejects_cross_rank_runtime_mismatch() -> None:
    snapshot = capture_runtime_provenance()
    other = snapshot.model_copy(
        update={
            "process_id": snapshot.process_id + 1,
            "torch_version": snapshot.torch_version + "+different-rank",
        }
    )
    stdout = _record(
        snapshot,
        local_rank=0,
        rank=0,
        world_size=2,
        local_world_size=2,
    ) + _record(
        other,
        local_rank=1,
        rank=1,
        world_size=2,
        local_world_size=2,
    )

    with pytest.raises(TorchrunRuntimeProvenanceError, match="inconsistent"):
        parse_torchrun_runtime_provenance_stdout(
            stdout,
            expected_local_ranks={0, 1},
            target="cpu",
        )
