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


def test_ddp_compression_logs_loss_through_reused_buffer() -> None:
    source = (LAB_DIR / "ddp_compression.py").read_text(encoding="utf-8")
    loop_section = source.split("for step, batch in enumerate(dataloader):", maxsplit=1)[1].split(
        "elapsed = perf_counter() - start",
        maxsplit=1,
    )[0]
    logging_section = loop_section.split("if is_main and step % 10 == 0:", maxsplit=1)[1]

    assert "loss.item()" not in loop_section
    assert "loss_value_buffer = torch.empty(1, dtype=torch.float64, device=device)" in source
    assert "loss_value_buffer[0].copy_(loss.detach())" in logging_section
    assert "loss_value = float(loss_value_buffer.detach().cpu()[0])" in logging_section
    assert "loss_value_buffer.detach().cpu().tolist()" not in logging_section


def test_train_distributed_mlp_builders_use_inplace_relu() -> None:
    for relative in (
        "baseline_zero1.py",
        "optimized_zero1.py",
        "baseline_zero1_multigpu.py",
        "optimized_zero1_multigpu.py",
        "baseline_zero2.py",
        "baseline_zero3.py",
        "baseline_zero3_multigpu.py",
        "pipeline.py",
    ):
        source = (LAB_DIR / relative).read_text(encoding="utf-8")

        assert "nn.ReLU(inplace=True)" in source
        assert "nn.ReLU()" not in source


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
    assert "world_size = get(\"ws\")" in init_section
    assert "self._reduce_inputs[idx] = torch.empty(" in init_section
    assert "param.numel() * world_size" in init_section
    assert "self._shard_grads[idx] = torch.empty(" in init_section
    assert "in_tensor.view(world_size, -1).copy_(flattened.unsqueeze(0))" in step_section
    assert "in_tensor = self._reduce_inputs[idx]" in step_section
    assert "shard_grad = self._shard_grads[idx]" in step_section
    assert "if idx in self.local_index_set:" in step_section
    assert "shard_grad.div_(world_size)" in step_section
    assert "param.grad = shard_grad.view_as(grad.data)" in step_section
    assert "(shard_grad / world_size)" not in step_section
    assert "torch.cat([flattened" not in step_section
    assert "torch.empty(" not in step_section
    assert "torch.empty_like(" not in step_section


def test_throughput_tracker_reuses_metric_payloads(monkeypatch) -> None:
    from labs.train_distributed import utils as train_utils

    perf_times = iter((100.0, 102.0, 104.0))
    monkeypatch.setattr(train_utils.time, "perf_counter", lambda: next(perf_times))

    tracker = train_utils.ThroughputTracker(warmup_steps=2)

    assert tracker.step(tokens=5) is tracker._empty_metrics
    assert tracker.step(tokens=5) is tracker._empty_metrics

    metrics = tracker.step(tokens=10, flops_per_token=2.0e12)
    assert metrics is tracker._metrics
    assert metrics["tokens_per_second"] == 5.0
    assert metrics["steps_per_second"] == 0.5
    assert metrics["total_tokens"] == 10
    assert metrics["total_time"] == 2.0
    assert metrics["tflops_per_device"] == 10.0

    metrics_without_flops = tracker.step(tokens=10)
    assert metrics_without_flops is metrics
    assert "tflops_per_device" not in metrics_without_flops


def test_zero3_param_shards_reuse_local_drop_buffers() -> None:
    for relative in ("baseline_zero3.py", "baseline_zero3_multigpu.py"):
        source = (LAB_DIR / relative).read_text(encoding="utf-8")
        init_section = source.split("class ParamShard", maxsplit=1)[1].split(
            "def all_gather",
            maxsplit=1,
        )[0]
        all_gather_section = source.split("def all_gather", maxsplit=1)[1].split(
            "def drop_full",
            maxsplit=1,
        )[0]
        drop_full_section = source.split("def drop_full", maxsplit=1)[1].split(
            "def attach_zero3_hooks",
            maxsplit=1,
        )[0]

        assert "self.full_data = torch.empty(" in init_section
        assert "self.local_shard = local_shard" in init_section
        assert "self.local_grad = torch.empty_like(local_shard)" in init_section
        assert "dist.all_gather_into_tensor(self.full_data, self.local_shard)" in all_gather_section
        assert "shards = [torch.empty_like(self.local_shard) for _ in range(self.world_size)]" not in all_gather_section
        assert "dist.all_gather(shards, self.local_shard)" not in all_gather_section
        assert "torch.cat(shards, dim=self.shard_dim)" not in all_gather_section
        assert "local = self.param.data.contiguous()" not in all_gather_section
        assert "self.param.data = self.local_shard" in drop_full_section
        assert "local = shards[self.rank].contiguous()" not in drop_full_section
        assert "self.local_grad.copy_(grad_shards[self.rank])" in drop_full_section
        assert "self.param.grad.data = self.local_grad" in drop_full_section
        assert "self.param.grad.data = grad_shards[self.rank].contiguous()" not in drop_full_section
        assert "self.full_data = None" not in drop_full_section
