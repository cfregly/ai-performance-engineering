"""Real distributed updates must match AdamW after accumulation and clipping."""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from core.optimization.manual_zero1 import OptimizerStateSharder
from labs.train_distributed.training_utils.zero1_optimizer import make_zero1_optimizer


def _run_worker(output_dir: Path, device_kind: str, implementation: str) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    if device_kind == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", device_id=device)
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.manual_seed(301 + rank)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)
        ).to(device)
        if implementation == "manual":
            ddp = model
            optimizer = OptimizerStateSharder(torch.optim.AdamW(
                model.parameters(), lr=0.01, betas=(0.9, 0.95),
                weight_decay=0.1, fused=False, foreach=False,
            ))
        else:
            ddp = DistributedDataParallel(
                model, device_ids=[local_rank] if device_kind == "cuda" else None
            )
            optimizer = make_zero1_optimizer(
                ddp.parameters(), 0.01, fused=device_kind == "cuda"
            )
        reference = copy.deepcopy(model)
        initial = [p.detach().clone() for p in model.parameters()]
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=0.01, betas=(0.9, 0.95),
            weight_decay=0.1, fused=False, foreach=False,
        )
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            assert all(p.grad is None for p in model.parameters())
            reference_optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                generator = torch.Generator(device=device).manual_seed(
                    100 + 10 * step + 2 * rank + micro
                )
                inputs = torch.randn((3, 8), generator=generator, device=device)
                targets = torch.randn((3, 2), generator=generator, device=device)
                (torch.nn.functional.mse_loss(ddp(inputs), targets) / 2).backward()
                (torch.nn.functional.mse_loss(reference(inputs), targets) / 2).backward()
            for parameter in reference.parameters():
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(world_size)
            # The manual example averages gradients inside step(), so its
            # unmodified public path has no pre-step global clipping hook.
            if implementation != "manual":
                torch.nn.utils.clip_grad_norm_(ddp.parameters(), 0.3)
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
            optimizer.step()
            reference_optimizer.step()
            for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
                copies = [torch.empty_like(actual) for _ in range(world_size)]
                dist.all_gather(copies, actual.detach())
                assert all(torch.equal(actual, peer) for peer in copies)
        assert any(
            not torch.equal(before, after)
            for before, after in zip(initial, model.parameters(), strict=True)
        ), "Training completed without any weight update"
        output_dir.mkdir(exist_ok=True)
        (output_dir / f"rank-{rank}.json").write_text(json.dumps({
            "rank": rank, "world_size": world_size, "device": device_kind,
            "implementation": implementation,
            "steps": 3, "microbatches_per_step": 2,
            "all_parameters_match_adamw": True, "weights_changed": True,
        }))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("device_kind", ["cpu", "cuda"])
@pytest.mark.parametrize("implementation", ["manual", "torch"])
def test_zero1_accumulated_clipped_updates_match_adamw(
    tmp_path: Path, world_size: int, device_kind: str, implementation: str
) -> None:
    if device_kind == "cuda" and torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} CUDA devices")
    env = {
        key: value for key, value in os.environ.items()
        if key not in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"}
    }
    env["OMP_NUM_THREADS"] = "1"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        endpoint = f"127.0.0.1:{listener.getsockname()[1]}"
    done = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
         "--rdzv-backend=static", f"--rdzv-endpoint={endpoint}",
         f"--nproc-per-node={world_size}", str(Path(__file__).resolve()),
         "--output-dir", str(tmp_path), "--device-kind", device_kind,
         "--implementation", implementation],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=90,
    )
    assert done.returncode == 0, done.stdout[-8000:]
    reports = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("rank-*.json"))]
    assert len(reports) == world_size
    assert {r["rank"] for r in reports} == set(range(world_size))
    assert all(r["weights_changed"] and r["all_parameters_match_adamw"] for r in reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-kind", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--implementation", choices=("manual", "torch"), required=True)
    arguments = parser.parse_args()
    _run_worker(arguments.output_dir, arguments.device_kind, arguments.implementation)
