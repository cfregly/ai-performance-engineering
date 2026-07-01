"""Smoke tests for shared fullstack-cluster benchmark wrapper factories."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from core.harness.validity_checks import _list_foreign_cuda_compute_processes
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, LaunchVia
from core.utils.chapter_compare_template import load_benchmark

ENTRYPOINT_MODULE = "labs.fullstack_cluster.moe_hybrid_ep_entrypoint"


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for benchmark wrappers")
@pytest.mark.parametrize(
    "relative_path",
    [
        "labs/fullstack_cluster/baseline_moe_hybrid_ep.py",
        "labs/fullstack_cluster/optimized_moe_hybrid_ep.py",
        "labs/fullstack_cluster/baseline_moe_hybrid_ep_multigpu.py",
        "labs/fullstack_cluster/optimized_moe_hybrid_ep_multigpu.py",
    ],
)
def test_fullstack_cluster_wrappers_attach_metadata_and_torchrun_script(relative_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    module = _load_module(module_path)

    bench = module.get_benchmark()
    spec = bench.get_torchrun_spec(BenchmarkConfig(launch_via="torchrun", nproc_per_node=1, iterations=1, warmup=5))

    assert isinstance(bench, BaseBenchmark)
    assert getattr(bench, "_module_file_override", None) == str(module_path)
    assert getattr(bench, "_factory_name_override", None) == "get_benchmark"
    assert Path(bench.script_path) == module_path
    assert spec.script_path is None
    assert spec.module_name == ENTRYPOINT_MODULE
    expected_args = ["--skip-preflight"]
    if "optimized_" in module_path.name:
        expected_args = ["--optimized", *expected_args]
    assert spec.script_args == expected_args
    assert "AISP_MOE_HYBRID_EP_METRICS_PATH" in spec.env
    assert spec.multi_gpu_required is ("multigpu" in module_path.name)
    assert bench._verify_output.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for benchmark wrappers")
@pytest.mark.parametrize(
    "relative_path",
    [
        "labs/fullstack_cluster/baseline_moe_hybrid_ep.py",
        "labs/fullstack_cluster/optimized_moe_hybrid_ep.py",
    ],
)
def test_fullstack_cluster_wrappers_expose_real_profile_torchrun_specs(relative_path: str, tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    module = _load_module(module_path)

    bench = module.get_benchmark()
    config = BenchmarkConfig(launch_via="torchrun", nproc_per_node=1, iterations=1, warmup=5)
    nsys_spec = bench.get_profile_torchrun_spec(profiler="nsys", config=config)
    torch_spec = bench.get_profile_torchrun_spec(
        profiler="torch",
        config=config,
        output_path=tmp_path / "trace.json",
    )

    assert nsys_spec is not None
    assert nsys_spec.script_path is None
    assert nsys_spec.module_name == ENTRYPOINT_MODULE
    expected_args = ["--skip-preflight"]
    if "optimized_" in module_path.name:
        expected_args = ["--optimized", *expected_args]
    assert nsys_spec.script_args == expected_args
    assert torch_spec is not None
    assert torch_spec.script_path is None
    assert torch_spec.module_name == ENTRYPOINT_MODULE
    expected_torch_args = [*expected_args, "--torch-profile-output", str(tmp_path / "trace.json")]
    assert torch_spec.script_args == expected_torch_args


def test_moe_hybrid_ep_entrypoint_routes_optimized_flag_and_remainder() -> None:
    from labs.fullstack_cluster import moe_hybrid_ep_entrypoint as entrypoint

    with mock.patch.object(entrypoint, "run_cli") as run_cli:
        entrypoint.main(["--optimized", "--skip-preflight", "--iters", "3"])

    run_cli.assert_called_once_with(
        optimized=True,
        argv=["--skip-preflight", "--iters", "3"],
    )


def test_moe_hybrid_ep_run_cli_discards_single_gpu_warmup_steps() -> None:
    from labs.fullstack_cluster import moe_hybrid_ep_common as common

    topology = SimpleNamespace(rank=0, world_size=1)
    observed_history = []

    class FakeTrainer:
        def __init__(self, args, topo, *, optimized):
            self.calls = 0

        def run_step(self):
            self.calls += 1
            return common.StepArtifacts(
                metrics={"moe.step.total_ms": float(self.calls)},
                loss=float(self.calls),
            )

    def fake_summary(**kwargs):
        observed_history.extend(step.metrics["moe.step.total_ms"] for step in kwargs["step_history"])
        return {"moe.step.total_ms": observed_history[-1]}

    with (
        mock.patch.object(common, "init_topology", return_value=topology),
        mock.patch.object(common, "HybridEPTrainer", FakeTrainer),
        mock.patch.object(common, "summarize_and_write_report", side_effect=fake_summary),
        mock.patch.object(common, "shutdown_topology"),
    ):
        result = common.run_cli(optimized=False, argv=["--skip-preflight", "--iters", "3"])

    assert observed_history == [6.0, 7.0, 8.0]
    assert result == {"moe.step.total_ms": 8.0}


def test_moe_hybrid_ep_run_cli_discards_multigpu_warmup_steps() -> None:
    from labs.fullstack_cluster import moe_hybrid_ep_common as common

    topology = SimpleNamespace(rank=0, world_size=2)
    observed_history = []

    class FakeTrainer:
        def __init__(self, args, topo, *, optimized):
            self.calls = 0

        def run_step(self):
            self.calls += 1
            return common.StepArtifacts(
                metrics={"moe.step.total_ms": float(self.calls)},
                loss=float(self.calls),
            )

    def fake_summary(**kwargs):
        observed_history.extend(step.metrics["moe.step.total_ms"] for step in kwargs["step_history"])
        return {"moe.step.total_ms": observed_history[-1]}

    with (
        mock.patch.object(common, "init_topology", return_value=topology),
        mock.patch.object(common, "HybridEPTrainer", FakeTrainer),
        mock.patch.object(common, "summarize_and_write_report", side_effect=fake_summary),
        mock.patch.object(common, "shutdown_topology"),
    ):
        result = common.run_cli(optimized=True, argv=["--skip-preflight", "--iters", "2"])

    assert observed_history == [3.0, 4.0]
    assert result == {"moe.step.total_ms": 4.0}


def test_moe_hybrid_ep_report_aggregates_step_history_once() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "labs"
        / "fullstack_cluster"
        / "moe_hybrid_ep_common.py"
    ).read_text(encoding="utf-8")
    summary_section = source.split("def summarize_and_write_report", maxsplit=1)[1].split(
        "def build_parser",
        maxsplit=1,
    )[0]

    assert "metric_keys = sorted(step_history[0].metrics.keys())" in summary_section
    assert "metric_totals = {key: 0.0 for key in metric_keys}" in summary_section
    assert "loss_total = 0.0" in summary_section
    assert "loss_total += step.loss" in summary_section
    assert "metric_totals[key] += step.metrics[key]" in summary_section
    assert "step_count = len(step_history)" in summary_section
    assert "sum(step.metrics" not in summary_section
    assert "sum(step.loss" not in summary_section


def test_moe_hybrid_ep_metric_reduction_batches_all_reduce_and_reuses_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from labs.fullstack_cluster import moe_hybrid_ep_common as common

    trainer = object.__new__(common.HybridEPTrainer)
    trainer.topology = SimpleNamespace(world_size=4)
    trainer.device = torch.device("cpu")
    trainer._metric_reduce_buffer = None
    trainer._metric_reduce_host_buffer = None
    observed_tensors = []

    def fake_all_reduce(tensor: torch.Tensor, op=None) -> None:
        observed_tensors.append(tensor.clone())
        tensor.mul_(float(trainer.topology.world_size))

    monkeypatch.setattr(common.dist, "all_reduce", fake_all_reduce)

    metrics = {"moe.step.total_ms": 2.0, "custom.count": 3.0}
    reduced = trainer._reduce_metrics(metrics)
    first_device_ptr = trainer._metric_reduce_buffer.data_ptr()
    first_host_ptr = trainer._metric_reduce_host_buffer.data_ptr()
    reduced_again = trainer._reduce_metrics(metrics)

    assert common._metric_should_average("moe.step.total_ms")
    assert not common._metric_should_average("custom.count")
    assert len(observed_tensors) == 2
    assert observed_tensors[0].shape == (2,)
    torch.testing.assert_close(observed_tensors[0], torch.tensor([2.0, 3.0], dtype=torch.float64))
    assert reduced["moe.step.total_ms"] == pytest.approx(2.0)
    assert reduced["custom.count"] == pytest.approx(12.0)
    assert reduced_again == reduced
    assert trainer._metric_reduce_buffer.data_ptr() == first_device_ptr
    assert trainer._metric_reduce_host_buffer.data_ptr() == first_host_ptr


def test_moe_hybrid_ep_reuses_forward_and_step_events_and_batches_count_reductions() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "labs"
        / "fullstack_cluster"
        / "moe_hybrid_ep_common.py"
    ).read_text(encoding="utf-8")
    forward_section = source.split("def forward_loss(", maxsplit=1)[1].split(
        "class HybridEPTrainer",
        maxsplit=1,
    )[0]
    run_step_section = source.split("def run_step", maxsplit=1)[1].split(
        "def _reduce_metrics",
        maxsplit=1,
    )[0]
    reduce_section = source.split("def _reduce_metrics", maxsplit=1)[1].split(
        "def summarize_and_write_report",
        maxsplit=1,
    )[0]
    router_section = source.split("class LoadBalancedRouter", maxsplit=1)[1].split(
        "class ExpertMLP",
        maxsplit=1,
    )[0]
    expert_forward = source.split("class ExpertMLP", maxsplit=1)[1].split(
        "class DeepSeekHybridEPModule",
        maxsplit=1,
    )[0]
    apply_local_section = source.split("def _apply_local_experts", maxsplit=1)[1].split(
        "def _exchange_counts",
        maxsplit=1,
    )[0]
    all_to_all_list_section = source.split("def _all_to_all_list", maxsplit=1)[1].split(
        "def _all_to_all_single",
        maxsplit=1,
    )[0]
    all_to_all_single_section = source.split("def _all_to_all_single", maxsplit=1)[1].split(
        "def _roundtrip_routes",
        maxsplit=1,
    )[0]
    roundtrip_section = source.split("def _roundtrip_routes", maxsplit=1)[1].split(
        "def _route_tokens",
        maxsplit=1,
    )[0]
    exchange_counts_section = source.split("def _exchange_counts", maxsplit=1)[1].split(
        "def _split_list",
        maxsplit=1,
    )[0]
    route_reduce_section = source.split("def _reduce_route_counts", maxsplit=1)[1].split(
        "def _token_indices",
        maxsplit=1,
    )[0]
    token_indices_section = source.split("def _token_indices", maxsplit=1)[1].split(
        "def _apply_local_experts",
        maxsplit=1,
    )[0]
    route_tokens_section = source.split("def _route_tokens", maxsplit=1)[1].split(
        "def forward_loss",
        maxsplit=1,
    )[0]
    local_count_section = source.split("def _local_expert_count_list", maxsplit=1)[1].split(
        "def _route_type_count_list",
        maxsplit=1,
    )[0]
    route_type_count_section = source.split("def _route_type_count_list", maxsplit=1)[1].split(
        "def _route_counts_list",
        maxsplit=1,
    )[0]
    route_counts_section = source.split("def _route_counts_list", maxsplit=1)[1].split(
        "def _aux_metric_values",
        maxsplit=1,
    )[0]
    aux_metric_section = source.split("def _aux_metric_values", maxsplit=1)[1].split(
        "def _apply_local_experts",
        maxsplit=1,
    )[0]

    assert "def _event_pair" in source
    assert "def _phase_events" in source
    assert "self._buffer_cache: Dict[Tuple[str, torch.device, torch.dtype], torch.Tensor]" in source
    assert "device: Optional[torch.device] = None" in source
    assert "key = (name, target_device, dtype)" in source
    assert "or cached.numel() < numel" in source
    assert "return cached[:numel].view(shape)" in source
    assert 'self.register_buffer(\n            "_gini_index",' in router_section
    assert "def _gini_index_for" in router_section
    assert "torch.arange(1, n + 1" not in router_section
    assert "log_route_probs = F.log_softmax(logits, dim=-1)" in router_section
    assert "route_probs = log_route_probs.exp()" in router_section
    assert "top_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)" in router_section
    assert "top_weights = F.softmax(top_logits, dim=-1)" in router_section
    assert '"router_entropy": -(route_probs * log_route_probs).sum(dim=-1).mean()' in router_section
    assert "route_probs = F.softmax(logits, dim=-1)" not in router_section
    assert "torch.topk(route_probs" not in router_section
    assert "top_weights = route_probs.gather(-1, top_indices)" not in router_section
    assert "torch.log(route_probs.clamp_min" not in router_section
    assert "F.silu(gate, inplace=True)" in expert_forward
    assert "gate.mul_(up)" in expert_forward
    assert "F.silu(self.gate_proj(x)) * self.up_proj(x)" not in expert_forward
    assert "repeat_interleave(self.top_k)" not in token_indices_section
    assert "torch.arange(num_tokens * self.top_k, device=device, dtype=torch.int64)" in token_indices_section
    assert 'cached.div_(self.top_k, rounding_mode="floor")' in token_indices_section
    assert "self._range_index_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}" in source
    assert "def _range_indices(self, length: int, device: torch.device) -> torch.Tensor" in source
    assert "torch.arange(length, device=device, dtype=torch.int64)" in source
    assert "expanded_tokens = hidden.index_select(0, token_indices)" in route_tokens_section
    assert "hidden.repeat_interleave(self.top_k" not in route_tokens_section
    assert "sort_idx = torch.argsort(expert_ids)" in apply_local_section
    assert '"local_outputs"' in apply_local_section
    assert '"local_sorted_outputs"' in apply_local_section
    assert "outputs.zero_()" not in apply_local_section
    assert "torch.zeros_like(tokens)" not in apply_local_section
    assert "torch.empty_like(sorted_tokens)" not in apply_local_section
    assert "expert_count_list = self._local_expert_count_list(expert_ids)" in apply_local_section
    assert "torch.bincount(expert_ids, minlength=self.local_experts).detach().cpu().tolist()" not in apply_local_section
    assert "self._local_expert_count_host_buffer: Optional[torch.Tensor] = None" in source
    assert "self._local_expert_count_list_buffer = [0] * local_experts" in source
    assert "def _local_expert_count_list(self, expert_ids: torch.Tensor)" in source
    assert "counts = torch.bincount(expert_ids, minlength=self.local_experts)" in source
    assert "host_counts.copy_(counts)" in source
    assert "count_list = self._local_expert_count_list_buffer" in source
    assert "count_list[expert_idx] = int(host_counts[expert_idx])" in source
    assert "host_counts.tolist()" not in local_count_section
    assert ".nonzero(" not in apply_local_section
    assert "bool(mask.any())" not in apply_local_section
    assert "out_slice = sorted_outputs[offset:next_offset]" in apply_local_section
    assert "out_slice.copy_(expert_out)" not in apply_local_section
    assert "out_slice.mul_(sorted_weights[offset:next_offset])" not in apply_local_section
    assert "torch.mul(expert_out, sorted_weights[offset:next_offset], out=out_slice)" in apply_local_section
    assert "sorted_outputs[offset:next_offset] = expert_out * sorted_weights[offset:next_offset]" not in apply_local_section
    assert "outputs.index_copy_(0, sort_idx, sorted_outputs)" in apply_local_section
    assert "return tensor.clone()" not in all_to_all_list_section
    assert "return tensor.clone()" not in all_to_all_single_section
    assert "return tensor" in all_to_all_list_section
    assert "return tensor" in all_to_all_single_section
    assert "dist_nn.all_to_all(recv_parts, send_parts, group=group)" in all_to_all_list_section
    assert "return torch.cat(list(result), dim=0)" not in all_to_all_list_section
    assert "return recv" in all_to_all_list_section
    assert "self._exchange_count_send_buffer = torch.empty(" in source
    assert "self._exchange_count_recv_buffer = torch.empty(" in source
    assert "dist.all_gather_into_tensor(recv_tensor, send_tensor, group=group)" in exchange_counts_section
    assert "gathered_counts = recv_tensor.view(group_size, group_size)[:, group_rank]" in exchange_counts_section
    assert "torch.stack(gathered" not in exchange_counts_section
    assert "send_tensor = torch.tensor(" not in exchange_counts_section
    assert "g[group_rank].item()" not in exchange_counts_section
    assert "self._destination_count_host_buffer: Optional[torch.Tensor] = None" in source
    assert "def _destination_count_list(self, dest_ranks: torch.Tensor, group_size: int)" in source
    assert "send_counts = self._destination_count_list(dest_ranks, group_size)" in roundtrip_section
    assert "torch.bincount(dest_ranks, minlength=group_size).tolist()" not in roundtrip_section
    assert "inverse_sort[sort_idx] = self._range_indices(sort_idx.numel(), sort_idx.device)" in roundtrip_section
    assert "torch.arange(sort_idx.numel()" not in roundtrip_section
    assert '"route_meta"' in roundtrip_section
    assert "meta[:, 0].copy_(sorted_token_indices)" in roundtrip_section
    assert "meta[:, 1].copy_(sorted_local_ids)" in roundtrip_section
    assert "meta = torch.stack(" not in roundtrip_section
    assert "torch.cuda.Event(enable_timing=True)" not in forward_section
    assert "torch.cuda.Event(enable_timing=True)" not in run_step_section
    assert roundtrip_section.count("torch.cuda.current_stream()") == 1
    assert "current_stream = torch.cuda.current_stream()" in roundtrip_section
    assert "events.start.record(current_stream)" in roundtrip_section
    assert "events.mid.record(current_stream)" in roundtrip_section
    assert "events.mid2.record(current_stream)" in roundtrip_section
    assert "events.end.record(current_stream)" in roundtrip_section
    assert ".record(torch.cuda.current_stream())" not in roundtrip_section
    assert forward_section.count("torch.cuda.current_stream()") == 1
    assert "current_stream = torch.cuda.current_stream()" in forward_section
    assert "routing_start.record(current_stream)" in forward_section
    assert "routing_end.record(current_stream)" in forward_section
    assert "self._comm_stream.wait_stream(current_stream)" in forward_section
    assert "current_stream.wait_stream(self._comm_stream)" in forward_section
    assert "torch.cuda.current_stream().wait_stream" not in forward_section
    assert ".record(torch.cuda.current_stream())" not in forward_section
    assert run_step_section.count("torch.cuda.current_stream()") == 1
    assert "current_stream = torch.cuda.current_stream()" in run_step_section
    assert "total_start.record(current_stream)" in run_step_section
    assert "total_after_forward.record(current_stream)" in run_step_section
    assert "total_after_backward.record(current_stream)" in run_step_section
    assert "total_after_sync.record(current_stream)" in run_step_section
    assert "total_end.record(current_stream)" in run_step_section
    assert ".record(torch.cuda.current_stream())" not in run_step_section
    assert "self._loss_host_buffer = torch.empty((), dtype=torch.float32, pin_memory=True)" in source
    assert "self._loss_host_buffer.copy_(loss.detach(), non_blocking=False)" in run_step_section
    assert "loss.detach().item()" not in run_step_section
    assert "loss=float(self._loss_host_buffer)" in run_step_section
    assert '"combined_outputs"' in forward_section
    assert "combined.zero_()" in forward_section
    assert "combined = torch.zeros_like(hidden)" not in forward_section
    assert '"remote_node_mask"' in forward_section
    assert "remote_node_mask.zero_()" in forward_section
    assert "torch.zeros_like(same_rank_mask)" not in forward_section
    assert "route_counts_global = route_counts" in forward_section
    assert "route_type_counts = self._route_type_count_list(" in forward_section
    assert "same_rank_count_int, same_node_count_int, remote_count_int" in forward_section
    assert "def _route_type_count_list(" in source
    assert "self._route_type_count_list_buffer = [0] * 3" in source
    assert "torch.sum(same_rank_mask, dim=None, out=count_buffer[0])" in source
    assert "torch.sum(same_node_mask, dim=None, out=count_buffer[1])" in source
    assert "torch.sum(remote_node_mask, dim=None, out=count_buffer[2])" in source
    assert "host_buffer.copy_(count_buffer)" in source
    assert "count_list = self._route_type_count_list_buffer" in source
    assert "count_list[route_idx] = int(host_buffer[route_idx])" in source
    assert "host_buffer.tolist()" not in route_type_count_section
    assert "bool(same_rank_mask.any())" not in forward_section
    assert "bool(same_node_mask.any())" not in forward_section
    assert "bool(remote_node_mask.any())" not in forward_section
    assert "same_rank_mask.sum().item()" not in forward_section
    assert "same_node_mask.sum().item()" not in forward_section
    assert "remote_node_mask.sum().item()" not in forward_section
    assert "self._reduce_route_counts(" in forward_section
    assert "return float(host_buffer[0]), float(host_buffer[1]), float(host_buffer[2])" in route_reduce_section
    assert "host_buffer.tolist()" not in route_reduce_section
    assert "route_counts_cpu = self._route_counts_list(route_counts_global)" in forward_section
    assert "tokens_per_expert=[int(x) for x in route_counts_cpu]" in forward_section
    assert "def _route_counts_list(self, route_counts: torch.Tensor)" in source
    assert "self._route_counts_list_buffer: List[int] = []" in source
    assert "host_buffer.copy_(route_counts)" in source
    assert "if len(self._route_counts_list_buffer) != count:" in source
    assert "count_list[expert_idx] = int(host_buffer[expert_idx])" in source
    assert "self._route_counts_host_buffer.tolist()" not in route_counts_section
    assert "def _aux_metric_values(self, aux: Dict[str, torch.Tensor])" in source
    assert "self._aux_metric_list_buffer = [0.0] * 4" in source
    assert "metric_buffer[0].copy_(aux[\"balance_loss\"])" in source
    assert "metric_buffer[1].copy_(aux[\"router_entropy\"])" in source
    assert "metric_buffer[2].copy_(aux[\"gini_coefficient\"])" in source
    assert "metric_buffer[3].copy_(aux[\"expert_usage_variance\"])" in source
    assert "metric_list = self._aux_metric_list_buffer" in source
    assert "metric_list[metric_idx] = float(host_buffer[metric_idx])" in source
    assert "host_buffer.tolist()" not in aux_metric_section
    assert ") = self._aux_metric_values(aux)" in forward_section
    assert "route_type_counts = torch.stack(" not in forward_section
    assert "route_counts_global.detach().cpu().tolist()" not in forward_section
    assert ").detach().cpu().tolist()" not in forward_section
    assert "route_counts_global.sum().item()" not in forward_section
    assert "route_counts_global.tolist()" not in forward_section
    assert "aux[\"balance_loss\"].detach().item()" not in forward_section
    assert "same_rank_tensor = torch.tensor" not in forward_section
    assert "dist.all_reduce(device_buffer, op=dist.ReduceOp.SUM)" in reduce_section
    assert "value = float(host_buffer[index])" in reduce_section
    assert "host_buffer.tolist()" not in reduce_section
    assert "for key, value in metrics.items()" not in reduce_section


def test_fullstack_router_uses_selected_logits_for_topk_weights() -> None:
    from labs.fullstack_cluster.moe_hybrid_ep_common import LoadBalancedRouter

    router = LoadBalancedRouter(hidden_size=4, num_experts=5, top_k=2).eval()
    x = torch.randn(7, 4)
    bias = torch.linspace(-0.2, 0.2, 5)

    with torch.inference_mode():
        logits = router.gate(x) + bias
        log_route_probs = torch.log_softmax(logits, dim=-1)
        route_probs = log_route_probs.exp()
        expected_selected, expected_indices = torch.topk(logits, router.top_k, dim=-1)
        expected_weights = torch.softmax(expected_selected, dim=-1)
        expected_entropy = -(route_probs * log_route_probs).sum(dim=-1).mean()

        actual_weights, actual_indices, aux = router(x, expert_bias=bias)

    torch.testing.assert_close(actual_indices, expected_indices)
    torch.testing.assert_close(actual_weights, expected_weights)
    torch.testing.assert_close(aux["router_entropy"], expected_entropy)


def test_moe_hybrid_ep_list_all_to_all_returns_preallocated_recv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from labs.fullstack_cluster import moe_hybrid_ep_common as common

    topology = common.TopologyInfo(
        rank=0,
        world_size=2,
        local_rank=0,
        local_world_size=2,
        node_rank=0,
        num_nodes=1,
        initialized=False,
        local_group=None,
    )
    module = common.DeepSeekHybridEPModule(
        hidden_size=2,
        num_experts=2,
        local_experts=1,
        top_k=1,
        topology=topology,
        route_mode="rank",
        optimized=False,
    )
    source = torch.arange(6, dtype=torch.float32).view(3, 2)

    def fake_all_to_all(output_parts, input_parts, group=None):
        del group
        for output, input_part in zip(output_parts, input_parts, strict=True):
            output.copy_(input_part)
        return tuple(output_parts)

    monkeypatch.setattr(common.dist_nn, "all_to_all", fake_all_to_all)

    result = module._all_to_all_list(source, [1, 2], [1, 2], group=None)

    assert result.shape == source.shape
    assert result.data_ptr() != source.data_ptr()
    torch.testing.assert_close(result, source)


def test_moe_hybrid_ep_buffer_reuses_larger_capacity_for_smaller_views() -> None:
    from labs.fullstack_cluster import moe_hybrid_ep_common as common

    topology = common.TopologyInfo(
        rank=0,
        world_size=1,
        local_rank=0,
        local_world_size=1,
        node_rank=0,
        num_nodes=1,
        initialized=False,
        local_group=None,
    )
    module = common.DeepSeekHybridEPModule(
        hidden_size=2,
        num_experts=1,
        local_experts=1,
        top_k=1,
        topology=topology,
        route_mode="rank",
        optimized=False,
    )
    key = ("scratch", torch.device("cpu"), torch.float32)

    large = module._buffer(
        "scratch",
        (4, 3),
        torch.float32,
        reuse=True,
        device=torch.device("cpu"),
    )
    backing = module._buffer_cache[key]
    small = module._buffer(
        "scratch",
        (2, 3),
        torch.float32,
        reuse=True,
        device=torch.device("cpu"),
    )
    fresh = module._buffer(
        "scratch",
        (2, 3),
        torch.float32,
        reuse=False,
        device=torch.device("cpu"),
    )
    resized = module._buffer(
        "scratch",
        (5, 3),
        torch.float32,
        reuse=True,
        device=torch.device("cpu"),
    )

    assert large.shape == (4, 3)
    assert small.shape == (2, 3)
    assert small.data_ptr() == backing.data_ptr()
    assert fresh.data_ptr() != backing.data_ptr()
    assert resized.shape == (5, 3)
    assert module._buffer_cache[key].numel() == 15


def test_moe_hybrid_ep_wrapper_reuses_latest_metrics_dict() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "labs/fullstack_cluster/moe_hybrid_ep_common.py"
    ).read_text(encoding="utf-8")
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def setup", maxsplit=1
    )[0]
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def _prepare_verification_payload", maxsplit=1
    )[0]

    assert "self._latest_metrics: Dict[str, float] = {}" in source
    assert "self._has_latest_metrics = False" in source
    assert "self._verify_probe = torch.zeros(1, dtype=torch.float32)" in source
    assert "latest_metrics = self._latest_metrics" in benchmark_section
    assert "latest_metrics.clear()" in benchmark_section
    assert "latest_metrics.update(artifacts.metrics)" in benchmark_section
    assert "dict(artifacts.metrics)" not in benchmark_section
    assert "self._latest_metrics.clear()" in setup_section
    assert 'inputs={"probe": self._verify_probe}' in capture_section
    assert "torch.zeros(" not in capture_section


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for benchmark wrappers")
def test_single_gpu_moe_hybrid_ep_uses_inprocess_step_runner() -> None:
    from labs.fullstack_cluster.baseline_moe_hybrid_ep import get_benchmark

    bench = get_benchmark()
    with mock.patch("torch.cuda.device_count", return_value=1), mock.patch("torch.cuda.is_available", return_value=True):
        config = bench.get_config()

    assert config.launch_via == LaunchVia.PYTHON
    assert config.use_subprocess is False
    assert config.single_gpu is True
    assert config.timing_method == "wall_clock"
    assert config.iterations == 1
    assert config.warmup == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for benchmark wrappers")
def test_load_benchmark_does_not_create_parent_cuda_compute_process() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    baseline_path = repo_root / "labs/fullstack_cluster/baseline_moe_hybrid_ep.py"

    bench = load_benchmark(baseline_path)
    assert bench is not None
    assert bench._verify_output.device.type == "cpu"

    loader_code = f"""
from pathlib import Path
import time
from core.utils.chapter_compare_template import load_benchmark

bench = load_benchmark(Path({str(baseline_path)!r}))
assert bench is not None
print("loaded", flush=True)
time.sleep(10)
"""

    process = subprocess.Popen(
        [sys.executable, "-u", "-c", loader_code],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = process.stdout.readline().strip() if process.stdout is not None else ""
        assert ready == "loaded", (
            process.stderr.read() if process.stderr is not None else "loader process failed before reporting ready"
        )
        assert process.poll() is None, (
            process.stderr.read() if process.stderr is not None else "loader process exited unexpectedly"
        )

        child_visible_as_compute = False
        foreign_err = None
        for _ in range(10):
            foreign, foreign_err = _list_foreign_cuda_compute_processes(
                device_index=0,
                current_pid=os.getpid(),
            )
            if foreign_err is not None:
                break
            if any(int(record.get("pid", -1)) == int(process.pid) for record in foreign):
                child_visible_as_compute = True
                break
            time.sleep(0.1)

        if foreign_err is not None:
            pytest.skip(f"NVML unavailable for foreign-process check: {foreign_err}")

        assert not child_visible_as_compute, (
            "load_benchmark() created a CUDA compute process in the parent loader subprocess"
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
