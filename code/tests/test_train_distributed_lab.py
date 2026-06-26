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
