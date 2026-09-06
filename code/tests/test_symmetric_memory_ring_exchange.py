"""Real Gloo controls for the symmetric-memory example's ring exchange."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from ch04.symmetric_memory_example import _ring_exchange


def _worker(result_dir: Path) -> None:
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        previous_rank = (rank - 1) % world_size
        next_rank = (rank + 1) % world_size
        base = torch.arange(33, dtype=torch.float32).reshape(3, 11)
        received = torch.empty_like(base)
        iterations = 7
        for iteration in range(iterations):
            sent = base + rank * 1000 + iteration * 17
            _ring_exchange(
                sent,
                received,
                next_rank=next_rank,
                previous_rank=previous_rank,
            )
            expected = base + previous_rank * 1000 + iteration * 17
            torch.testing.assert_close(received, expected, rtol=0.0, atol=0.0)
        (result_dir / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "world_size": world_size,
                    "iterations": iterations,
                    "output": received.tolist(),
                }
            ),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


def _run_bounded(command: list[str], *, environment: dict[str, str]) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=40)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=5)
        pytest.fail(f"Gloo ring worker timed out and was terminated:\n{output[-4000:]}")
    return subprocess.CompletedProcess(command, process.returncode, output, None)


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.mark.parametrize("world_size", [2, 3])
def test_ring_exchange_posts_both_directions_and_preserves_full_outputs(
    tmp_path: Path,
    world_size: int,
) -> None:
    result_dir = tmp_path / f"results-{world_size}"
    result_dir.mkdir()
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    code_root = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(code_root), environment.get("PYTHONPATH", ""))
        if value
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc-per-node={world_size}",
        "--master-addr=127.0.0.1",
        f"--master-port={_unused_loopback_port()}",
        str(Path(__file__).resolve()),
        "--ring-worker",
        "--result-dir",
        str(result_dir),
    ]

    completed = _run_bounded(command, environment=environment)

    assert completed.returncode == 0, completed.stdout[-4000:]
    base = torch.arange(33, dtype=torch.float32).reshape(3, 11)
    for rank in range(world_size):
        payload = json.loads(
            (result_dir / f"rank-{rank}.json").read_text(encoding="utf-8")
        )
        expected = base + ((rank - 1) % world_size) * 1000 + 6 * 17
        assert payload["rank"] == rank
        assert payload["world_size"] == world_size
        assert payload["iterations"] == 7
        torch.testing.assert_close(
            torch.tensor(payload["output"]),
            expected,
            rtol=0.0,
            atol=0.0,
        )


def _parse_worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ring-worker", action="store_true", required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    worker_args = _parse_worker_args()
    _worker(worker_args.result_dir)
