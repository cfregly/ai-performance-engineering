from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ch04 import symmetric_memory_training_advanced as training

CODE_ROOT = Path(__file__).resolve().parents[1]
CHILD = CODE_ROOT / "ch04" / "symmetric_memory_training_advanced.py"


@pytest.mark.parametrize(
    ("baseline_module", "optimized_module", "steps", "target"),
    [
        (
            "labs.train_distributed.baseline_symmem_training",
            "labs.train_distributed.optimized_symmem_training",
            "120",
            "labs/train_distributed:symmem_training",
        ),
        (
            "labs.train_distributed.baseline_symmem_training_multigpu",
            "labs.train_distributed.optimized_symmem_training_multigpu",
            "80",
            "labs/train_distributed:symmem_training_multigpu",
        ),
    ],
)
def test_symmem_factories_select_only_the_transport(
    baseline_module: str,
    optimized_module: str,
    steps: str,
    target: str,
) -> None:
    baseline = importlib.import_module(baseline_module).get_benchmark()
    optimized = importlib.import_module(optimized_module).get_benchmark()

    assert baseline._multi_gpu_required is True
    assert optimized._multi_gpu_required is True
    assert baseline._default_nproc_per_node == 2
    assert optimized._default_nproc_per_node == 2
    assert baseline._target_label == optimized._target_label == target
    assert baseline._script_path.resolve() == optimized._script_path.resolve() == CHILD

    assert "--allow-single-gpu" not in baseline._base_args
    assert "--allow-single-gpu" not in optimized._base_args
    assert "--disable-symmetric" not in baseline._base_args
    assert "--disable-symmetric" not in optimized._base_args

    baseline_args = list(baseline._base_args)
    optimized_args = list(optimized._base_args)
    baseline_transport = baseline_args.index("--transport") + 1
    optimized_transport = optimized_args.index("--transport") + 1
    assert baseline_args[baseline_transport] == "nccl-broadcast"
    assert optimized_args[optimized_transport] == "symmetric-memory"
    baseline_args[baseline_transport] = "TRANSPORT"
    optimized_args[optimized_transport] = "TRANSPORT"
    assert baseline_args == optimized_args
    assert baseline_args[baseline_args.index("--steps") + 1] == steps
    assert baseline_args[baseline_args.index("--seed") + 1] == "42"


def _run_child(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(CODE_ROOT),
        }
    )
    return subprocess.run(
        [sys.executable, str(CHILD), *args],
        cwd=CODE_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
        check=False,
    )


def test_symmem_child_help_is_available_without_cuda() -> None:
    completed = _run_child("--help")

    assert completed.returncode == 0, completed.stdout
    assert "--transport {nccl-broadcast,symmetric-memory}" in completed.stdout
    assert "--allow-single-gpu" not in completed.stdout
    assert "--disable-symmetric" not in completed.stdout


@pytest.mark.parametrize("transport", ["nccl-broadcast", "symmetric-memory"])
def test_symmem_transports_report_explicit_cuda_capability_failure(transport: str) -> None:
    completed = _run_child(
        "--demo",
        "optimizer",
        "--transport",
        transport,
        "--steps",
        "1",
        "--batch-size",
        "1",
        "--hidden-dim",
        "2",
        "--output-dim",
        "1",
        "--optimizer-layers",
        "1",
    )

    assert completed.returncode == 3, completed.stdout
    assert completed.stdout.strip().startswith("SKIPPED:")
    assert "GPU" in completed.stdout or "CUDA" in completed.stdout


def test_removed_single_gpu_flag_is_rejected_by_the_parser() -> None:
    completed = _run_child("--demo", "optimizer", "--allow-single-gpu")

    assert completed.returncode == 2
    assert "unrecognized arguments: --allow-single-gpu" in completed.stdout


def test_transport_variants_share_the_exact_momentum_update() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = training._MomentumOptimizer([parameter], lr=0.1, world_size=2)

    parameter.grad = torch.tensor([0.5, -1.0])
    optimizer.step(rank=0)
    torch.testing.assert_close(optimizer.momentum_buffers[0], torch.tensor([0.5, -1.0]))
    torch.testing.assert_close(parameter, torch.tensor([0.95, -1.9]))

    parameter.grad = torch.tensor([1.0, 2.0])
    optimizer.step(rank=0)
    torch.testing.assert_close(optimizer.momentum_buffers[0], torch.tensor([1.45, 1.1]))
    torch.testing.assert_close(parameter, torch.tensor([0.805, -2.01]))


def test_nccl_baseline_broadcasts_every_updated_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    parameters = [
        torch.nn.Parameter(torch.tensor([1.0])),
        torch.nn.Parameter(torch.tensor([2.0, 3.0])),
    ]
    calls: list[tuple[torch.Tensor, int]] = []
    monkeypatch.setattr(training.dist, "broadcast", lambda tensor, src: calls.append((tensor, src)))
    optimizer = training.NCCLBroadcastOptimizer(parameters, lr=0.1, world_size=2)

    optimizer.synchronize_parameters(rank=1)

    assert [src for _, src in calls] == [0, 0]
    assert [tensor.data_ptr() for tensor, _ in calls] == [
        parameter.data_ptr() for parameter in parameters
    ]


def test_symmetric_transport_propagates_handle_creation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    def fail_strictly(_tensor: torch.Tensor) -> None:
        raise RuntimeError("strict symmetric handle creation failed")

    monkeypatch.setattr(training, "create_symmetric_memory_handle", fail_strictly)
    with pytest.raises(RuntimeError, match="strict symmetric handle creation failed"):
        training.SymmetricMemoryOptimizer([parameter], lr=0.1, world_size=2)
