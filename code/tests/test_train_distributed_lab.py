"""Tests for train_distributed benchmark-local contracts."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = REPO_ROOT / "labs" / "train_distributed"


def test_fsdp2_single_gpu_b200_contract_is_comparison() -> None:
    payload = json.loads((LAB_DIR / "expectations_b200.json").read_text(encoding="utf-8"))
    entry = payload["examples"]["fsdp2"]

    assert entry["metadata"]["optimization_goal"] == "comparison"


def test_fsdp2_multi_gpu_b200_contract_stays_speed_gated() -> None:
    payload = json.loads((LAB_DIR / "expectations_2x_b200.json").read_text(encoding="utf-8"))
    entry = payload["examples"]["fsdp2"]

    assert entry["metadata"]["optimization_goal"] == "speed"


def test_ddp_compression_single_gpu_simulation_reuses_staging_buffers() -> None:
    source = (LAB_DIR / "ddp_compression.py").read_text(encoding="utf-8")
    staging_section = source.split("comm_staging = {}", maxsplit=1)[1].split(
        "def _simulate_single_gpu_comm", maxsplit=1
    )[0]
    simulate_section = source.split("def _simulate_single_gpu_comm", maxsplit=1)[1].split(
        "if args.naive_allreduce", maxsplit=1
    )[0]

    assert 'comm_staging["cpu_float"] = _empty_cpu_staging' in staging_section
    assert 'comm_staging["quant"] = torch.empty' in staging_section
    assert 'comm_staging["cpu_sample"] = _empty_cpu_staging' in staging_section
    assert "cpu_buf = buffer.cpu()" not in simulate_section
    assert "cpu_buf.to(device)" not in simulate_section
    assert "sampled = flat[::stride].contiguous()" not in simulate_section
    assert "quant.cpu()" not in simulate_section
    assert "if max_val > 0" not in simulate_section
    assert "scale = buffer.abs().max().clamp_min(1e-12).div(127.0)" in simulate_section


def test_ddp_compression_int8_hook_masks_zero_scale_in_place() -> None:
    source = (LAB_DIR / "ddp_compression.py").read_text(encoding="utf-8")
    hook_section = source.split("def _int8_allreduce_hook", maxsplit=1)[1].split(
        "def main",
        maxsplit=1,
    )[0]

    assert "torch.where(" not in hook_section
    assert "torch.ones_like(scale)" not in hook_section
    assert "scale.masked_fill_(scale == 0, 1.0)" in hook_section


def test_zero2_gradient_sharder_reuses_reduce_buffers() -> None:
    source = (LAB_DIR / "baseline_zero2.py").read_text(encoding="utf-8")
    init_section = source.split("def __init__", maxsplit=1)[1].split(
        "def _shard_parameters",
        maxsplit=1,
    )[0]
    step_section = source.split("def step", maxsplit=1)[1].split(
        "def zero_grad",
        maxsplit=1,
    )[0]

    assert "self.local_index_set = set(self.local_indices)" in init_section
    assert "self._reduce_inputs: dict[int, torch.Tensor] = {}" in init_section
    assert "self._shard_grads: dict[int, torch.Tensor] = {}" in init_section
    assert "in_tensor.view(world_size, -1).copy_(flattened.unsqueeze(0))" in step_section
    assert "if idx in self.local_index_set:" in step_section
    assert "shard_grad.div_(world_size)" in step_section
    assert "param.grad = shard_grad.view_as(grad.data)" in step_section
    assert "(shard_grad / world_size)" not in step_section
    assert "torch.cat([flattened" not in step_section
    assert "shard_grad = torch.empty_like(flattened)\n            dist.reduce_scatter_tensor" not in step_section
