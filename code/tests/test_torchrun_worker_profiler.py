"""Real distributed CPU controls for the worker-lifetime Torch profiler path."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import torch

from core.harness.benchmark_harness import BenchmarkConfig, LaunchVia, TorchrunLaunchSpec
from core.harness.run_benchmarks import profile_python_benchmark_torch


class _WorkerBenchmark:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._config = BenchmarkConfig(
            device=torch.device("cpu"),
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=2,
            rdzv_backend="static",
            validity_profile="portable",
            lock_gpu_clocks=False,
            measurement_timeout_seconds=60,
            profiling_timeout_seconds=60,
        )
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self._config.rdzv_endpoint = f"127.0.0.1:{listener.getsockname()[1]}"

    def get_torchrun_spec(self, config: BenchmarkConfig) -> TorchrunLaunchSpec:
        return TorchrunLaunchSpec(script_path=self.path)


def test_torch_profiler_runs_actual_two_rank_workers(tmp_path: Path) -> None:
    worker = tmp_path / "collective_worker.py"
    worker.write_text(
        """import torch
import torch.distributed as dist
dist.init_process_group('gloo')
value = torch.full((8,), float(dist.get_rank() + 1))
dist.all_reduce(value)
torch.testing.assert_close(value, torch.full((8,), 3.0))
dist.destroy_process_group()
""",
        encoding="utf-8",
    )
    trace = profile_python_benchmark_torch(
        _WorkerBenchmark(worker), worker, tmp_path, tmp_path / "traces"
    )
    assert trace is not None
    paths = [trace, trace.with_name(f"{trace.stem}.rank1{trace.suffix}")]
    for rank, path in enumerate(paths):
        payload = json.loads(path.read_text())
        assert payload["aisp_profile_scope"] == "torchrun_worker_lifetime"
        assert payload["aisp_rank"] == str(rank)
        assert any(event.get("name") == "c10d::allreduce_" for event in payload["traceEvents"])


def test_failed_torchrun_worker_does_not_produce_success_trace(tmp_path: Path) -> None:
    worker = tmp_path / "failed_worker.py"
    worker.write_text("raise RuntimeError('worker failed before capture')\n", encoding="utf-8")
    trace = profile_python_benchmark_torch(
        _WorkerBenchmark(worker), worker, tmp_path, tmp_path / "traces"
    )
    assert trace is None
    assert not list((tmp_path / "traces").glob("*_torch_trace*.json"))
    assert any(
        "worker failed before capture" in log.read_text()
        for log in (tmp_path / "traces").glob("*.stderr.log")
    )
