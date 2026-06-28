from __future__ import annotations

import ast
import inspect
import importlib
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ch01.optimized_performance import OptimizedPerformanceBatchBenchmark
from ch01.optimized_performance_fp16 import OptimizedPerformanceFP16Benchmark
from ch02.baseline_cublas import BaselineCublasBenchmark
from ch02.optimized_cublas import OptimizedCublasBenchmark
from core.benchmark.verification import coerce_input_signature
from core.harness.benchmark_harness import BaseBenchmark, _cleanup_process_group
from core.harness.run_benchmarks import (
    INFORMATIONAL_BENCHMARKS,
    _collect_current_run_benchmark_orphan_pids,
    _collect_stale_benchmark_orphan_pids,
    _reap_benchmark_process_leftovers,
    _reap_run_descendants,
)
from labs.flexattention.baseline_flex_attention import BaselineFlexAttentionBenchmark
from labs.flexattention.optimized_flex_attention import OptimizedFlexAttentionBenchmark
from labs.occupancy_tuning.optimized_proton_matmul_bm64_bn64_bk32_nw2 import (
    get_benchmark as get_latency_benchmark,
)
from labs.occupancy_tuning.optimized_proton_matmul_bm64_bn256_bk32 import (
    get_benchmark as get_wide_n_benchmark,
)
from labs.real_world_models.deepseek_r1_moe_optimization import (
    get_benchmark as get_deepseek_benchmark,
)
from labs.real_world_models.gpt4_architecture_optimization import (
    get_benchmark as get_gpt4_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_FUSION_SIGNATURE_MODULES = (
    "ch12.baseline_kernel_fusion",
    "ch12.optimized_kernel_fusion",
    "ch12.optimized_kernel_fusion_llm_dedicated_stream_and_prefetch_for_blackwell",
    "ch12.optimized_kernel_fusion_llm_persistent_buffer_and_stream_friendly_setup",
    "ch12.optimized_kernel_fusion_llm_reuse_static_tensor_and_simplify_setup",
)
TIMEOUT_PRONE_SIGNATURE_CASES = (
    ("ch13.baseline_bandwidth_naive", 16_777_216, (4096, 4096, 16), "float32"),
    ("ch13.optimized_bandwidth_naive", 16_777_216, (4096, 4096, 16), "float32"),
    ("ch18.baseline_vllm_v1_integration", 8, (128,), "int64"),
    ("ch18.optimized_vllm_v1_integration", 8, (128,), "int64"),
)


def test_ch01_precision_benchmarks_disable_tf32_during_setup() -> None:
    initial_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    initial_cudnn = (
        bool(torch.backends.cudnn.allow_tf32)
        if torch.backends.cudnn.is_available()
        else None
    )

    for benchmark_cls in (OptimizedPerformanceBatchBenchmark, OptimizedPerformanceFP16Benchmark):
        bench = benchmark_cls()
        if isinstance(bench, OptimizedPerformanceBatchBenchmark):
            bench.workload = SimpleNamespace(performance_microbatches=2)
        bench.batch_size = 2
        bench.num_microbatches = 2
        bench.hidden_dim = 64
        bench.setup()
        try:
            assert torch.backends.cuda.matmul.allow_tf32 is False
            if torch.backends.cudnn.is_available():
                assert torch.backends.cudnn.allow_tf32 is False
        finally:
            bench.teardown()

    assert torch.backends.cuda.matmul.allow_tf32 == initial_matmul
    if initial_cudnn is not None:
        assert torch.backends.cudnn.allow_tf32 == initial_cudnn


def test_ch01_training_mlp_uses_inplace_relu_modules() -> None:
    source = (REPO_ROOT / "ch01" / "performance_common.py").read_text(encoding="utf-8")

    assert "torch.nn.ReLU(inplace=True)" in source
    assert "torch.nn.ReLU()" not in source


def test_ch01_fp16_benchmark_precomputes_microbatch_groups() -> None:
    source = (REPO_ROOT / "ch01" / "optimized_performance_fp16.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "self._microbatch_groups = []" in setup_section
    assert "self._target_groups = []" in setup_section
    assert "self._group_sizes = []" in setup_section
    assert "data_group = tuple(self.microbatches[start : start + self.fusion])" in setup_section
    assert "target_group = tuple(self.targets[start : start + self.fusion])" in setup_section
    assert "self._group_sizes.append(len(data_group))" in setup_section
    assert "for group_data, group_targets, group_size in zip(" in benchmark_section
    assert "self.microbatches[start : start + self.fusion]" not in benchmark_section
    assert "self.targets[start : start + self.fusion]" not in benchmark_section
    assert "group_size = max(" not in benchmark_section


def test_ch02_cublas_metrics_report_gemm_workload_not_transfer_placeholders() -> None:
    baseline_metrics = BaselineCublasBenchmark().get_custom_metrics()
    optimized_metrics = OptimizedCublasBenchmark().get_custom_metrics()

    expected_flops = float(2 * 2048 * 2048 * 2048)
    assert baseline_metrics["gemm.total_flops"] == expected_flops
    assert optimized_metrics["gemm.total_flops"] == expected_flops
    assert "transfer.achieved_gbps" not in baseline_metrics
    assert "transfer.achieved_gbps" not in optimized_metrics


def test_ch04_optimized_dataparallel_reuses_gradient_staging_buffers() -> None:
    source = (REPO_ROOT / "ch04" / "optimized_dataparallel_multigpu.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "self._grad_staging: List[List[torch.Tensor]] = []" in source
    assert "self._grad_staging = [" in setup_section
    assert "torch.empty_like(param, device=master_device)" in setup_section
    assert "grad.to(master_device" not in benchmark_section
    assert "staging.copy_(grad, non_blocking=True)" in benchmark_section
    assert "reduced.add_(staging)" in benchmark_section
    assert "outputs: List[torch.Tensor] = []" not in benchmark_section
    assert "outputs.append(" not in benchmark_section
    assert "first_output: Optional[torch.Tensor] = None" in benchmark_section
    assert "grads = [param.grad for param in param_group]" not in benchmark_section
    assert "master_grad = param_group[0].grad" in benchmark_section


def test_ch04_optimizer_central_nvlink_uses_direct_copy_staging() -> None:
    source = (REPO_ROOT / "ch04" / "optimizer_central_nvlink.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "self.grad_root_buffers: List[torch.Tensor] = []" in source
    assert "self.grad_root_buffers.append(torch.empty_like(master_w, device=self.root_device))" in setup_section
    assert "model.weight.grad.to(self.root_device" not in benchmark_section
    assert "master_w.to(model.weight.device" not in benchmark_section
    assert "grad_root_buf.copy_(grad, non_blocking=True)" in benchmark_section
    assert "model.weight.data.copy_(master_w, non_blocking=True)" in benchmark_section


def test_ch04_ddp_nvlink_overlap_reuses_transfer_events_and_buffers() -> None:
    naive_source = (REPO_ROOT / "ch04" / "ddp_nvlink_naive.py").read_text(encoding="utf-8")
    source = (REPO_ROOT / "ch04" / "ddp_nvlink_overlap.py").read_text(encoding="utf-8")
    naive_setup = naive_source.split("def setup", maxsplit=1)[1].split(
        "def _simulate_allreduce", maxsplit=1
    )[0]
    naive_reduce = naive_source.split("def _simulate_allreduce", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    naive_benchmark = naive_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def _async_reduce_to_root", maxsplit=1
    )[0]
    reduce_section = source.split("def _async_reduce_to_root", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "self._root_grad_staging: List[List[torch.Tensor]] = []" in source
    assert "self._grad_ready_events: List[List[torch.cuda.Event]] = []" in source
    assert "self._update_buffers: List[torch.Tensor] = []" in source
    assert "self._grad_slots: List[torch.Tensor] = []" in naive_source
    assert "self._grad_slots: List[torch.Tensor] = []" in source
    assert "self._ordered_grad_slots: List[torch.Tensor] = []" in source
    assert "self._ordered_bucket_indices: List[int] = []" in source
    assert "self._reduction_results: List[torch.Tensor] = []" in source
    assert "self._allreduce_buffer = torch.empty_like(" in naive_setup
    assert "self._grad_slots = [" in naive_setup
    assert "self._allreduce_buffer = torch.zeros_like(" not in naive_setup
    assert "grads = []" not in naive_benchmark
    assert "grads.append(" not in naive_benchmark
    assert "buf.copy_(grads[0].to(root))" in naive_reduce
    assert "for g in grads[1:]" in naive_reduce
    assert "buf.zero_()" not in naive_reduce
    assert "self._grad_ready_events = [" in setup_section
    assert "torch.empty_like(self.models[0].weight, device=self.root_device)" in setup_section
    assert "torch.zeros_like(self.models[0].weight, device=self.root_device)" not in setup_section
    assert "[torch.cuda.Event() for _ in self.models]" in setup_section
    assert "self._ordered_grad_slots = [" in setup_section
    assert "self._ordered_bucket_indices = [bucket_idx for _, bucket_idx in bucket_order]" in setup_section
    assert "self._reduction_results = [" in setup_section
    assert "torch.cuda.Event()" not in reduce_section
    assert "g.to(self.root_device" not in reduce_section
    assert "root_buf.copy_(first)" in reduce_section
    assert "for idx, g in enumerate(grads[1:], start=1)" in reduce_section
    assert "grads = []" not in benchmark_section
    assert "reduction_results: List[torch.Tensor] = []" not in benchmark_section
    assert "sorted(zip(grads, _bucket_order())" not in benchmark_section
    assert "ordered_grads = [g for g, _ in ordered]" not in benchmark_section
    assert "reduction_results[micro] = self._async_reduce_to_root(ordered_grads, micro)" in benchmark_section
    assert "root_buf.zero_()" not in reduce_section
    assert "root_buf.to(model.weight.device" not in benchmark_section
    assert "staging.copy_(g, non_blocking=True)" in reduce_section
    assert "root_local.copy_(root_buf, non_blocking=True)" in benchmark_section


def test_ch04_gradient_compression_int8_reuses_cast_buffers() -> None:
    source = (REPO_ROOT / "ch04" / "gradient_compression_common.py").read_text(
        encoding="utf-8"
    )
    prepare_section = source.split("def _prepare_int8_buffers", maxsplit=1)[1].split(
        "def _build_bucket_slices", maxsplit=1
    )[0]
    naive_section = source.split("def _int8_all_reduce_naive", maxsplit=1)[1].split(
        "def _store_scale", maxsplit=1
    )[0]

    assert "float_buf.to(torch.int8)" not in prepare_section
    assert "float_buf[sl].to(torch.int8)" not in naive_section
    assert "self._int8_buffers[idx].copy_(float_buf)" in prepare_section
    assert "int8_buf[sl].copy_(float_buf[sl])" in naive_section


def test_ch05_optimized_storage_cpu_opens_mmap_outside_hot_loop() -> None:
    source = (REPO_ROOT / "ch05" / "optimized_storage_cpu.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    teardown_section = source.split("def teardown", maxsplit=1)[1].split(
        "def get_config", maxsplit=1
    )[0]

    assert "self._output_buffer: Optional[torch.Tensor] = None" in source
    assert 'self._mapped_array = np.load(self.filepath, mmap_mode="r")' in setup_section
    assert "self._output_buffer = torch.empty(1, device=self.device, dtype=torch.float32)" in setup_section
    assert 'np.load(self.filepath, mmap_mode="r")' not in benchmark_section
    assert "with torch.inference_mode(), self._nvtx_range(\"storage_cpu_optimized\"):" in benchmark_section
    assert "np.copyto(self._host_buffer_view, self._mapped_array)" in benchmark_section
    assert "torch.sum(self.device_buffer, dim=0, keepdim=True, out=self._output_buffer)" in benchmark_section
    assert "self.device_buffer.sum().unsqueeze(0)" not in benchmark_section
    assert "torch.empty(" not in benchmark_section
    assert "self._mapped_array = None" in teardown_section
    assert "self._output_buffer = None" in teardown_section


def test_ch19_dynamic_quantized_cache_reuses_int8_source_buffer() -> None:
    source = (REPO_ROOT / "ch19" / "baseline_dynamic_quantized_cache.py").read_text(
        encoding="utf-8"
    )
    prepare_section = source.split("def _prepare_quantized_sources", maxsplit=1)[1].split(
        "def _non_adaptive_cache_update", maxsplit=1
    )[0]

    assert "self._quant_scratch.to(torch.int8)" not in prepare_section
    assert "self._quantized_int8_src.copy_(self._quant_scratch)" in prepare_section


def test_ch07_and_ch08_sources_do_not_ship_artificial_baseline_penalties() -> None:
    hbm_copy_source = (REPO_ROOT / "ch07" / "baseline_hbm_copy.cu").read_text(encoding="utf-8")
    threshold_source = (REPO_ROOT / "ch08" / "threshold_common.cuh").read_text(encoding="utf-8")

    assert "scalar_copy_kernel<<<64, 64>>>" not in hbm_copy_source
    assert "scalar_copy_kernel<<<blocks, threads>>>" in hbm_copy_source
    assert "const volatile float* volatile_inputs" not in threshold_source
    assert "volatile float redundant_eval" not in threshold_source
    assert "expensive_transform(-value" not in threshold_source


def test_ch07_tma_copy_surfaces_scalar_vs_strict_descriptor_tma_story() -> None:
    optimized_wrapper = (REPO_ROOT / "ch07" / "optimized_tma_copy.py").read_text(encoding="utf-8")
    optimized_cuda = (REPO_ROOT / "ch07" / "optimized_tma_copy.cu").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "ch07" / "README.md").read_text(encoding="utf-8")

    assert "Pipeline + Tensor-Map Neighbor Copy" in optimized_wrapper
    assert "strict `tma_copy` path" in optimized_wrapper
    assert "dst[global_row * N + global_col] = combine_values(" in optimized_cuda
    assert "output_tile[local_row][local_col] = combine_values(" in optimized_cuda
    assert "output_tile" in optimized_cuda
    assert "usable tensor-map/TMA support" in optimized_cuda
    assert "legacy async-neighbor demo" not in readme
    assert "strict tensor-map/TMA-capable run only" in readme


def test_ch07_lookup_pytorch_reuses_table_and_timing_events() -> None:
    source = (REPO_ROOT / "ch07" / "lookup_pytorch.py").read_text(encoding="utf-8")
    run_section = source.split("def run", maxsplit=1)[1].split("def main", maxsplit=1)[0]
    main_section = source.split("def main", maxsplit=1)[1]

    assert "out = torch.empty_like" not in source
    assert "events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None" in run_section
    assert "start_event, end_event = events" in run_section
    assert "table = torch.arange(N, device=device, dtype=torch.float32)" in main_section
    assert "ms = run(random_indices, table=table, events=events)" in main_section
    assert "ms = run(coalesced_indices, table=table, events=events)" in main_section


def test_kv_locality_microbench_reuses_copy_stream_and_defers_output_tensor() -> None:
    source = (REPO_ROOT / "core" / "scripts" / "kv_locality_microbench.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def _bench_copy", maxsplit=1)[0]
    copy_section = source.split("def _bench_copy", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def get_config", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def get_custom_metrics", maxsplit=1
    )[0]

    assert "self.copy_stream = torch.cuda.Stream(device=self.device)" in setup_section
    assert "torch.cuda.Stream(" not in copy_section
    assert "with torch.cuda.stream(self.copy_stream)" in copy_section
    assert "torch.tensor(" not in benchmark_section
    assert "self._output_values = ordered" in benchmark_section
    assert "self.output = torch.tensor(self._output_values, dtype=torch.float32)" in capture_section


def test_cluster_all_reduce_tool_reuses_bandwidth_scalar_buffer() -> None:
    source = (REPO_ROOT / "cluster" / "tools" / "all_reduce_bench.py").read_text(
        encoding="utf-8"
    )
    timed_section = source.split("def timed_allreduce", maxsplit=1)[1].split(
        "def run", maxsplit=1
    )[0]
    run_section = source.split("def run", maxsplit=1)[1].split(
        "def device_id_kwargs", maxsplit=1
    )[0]

    assert "torch.tensor([size / duration])" not in source
    assert "torch.stack(" not in run_section
    assert "algbw_gather" not in run_section
    assert "def timed_allreduce(tensor, size, start_event, end_event, algbw_buffer)" in source
    assert "algbw_buffer.fill_(size / duration)" in timed_section
    assert "algbw_buffer = torch.empty(1, device=tensor.device, dtype=torch.float32)" in run_section
    assert "algbw_sum.add_(algbw_buffer)" in run_section
    assert "algbw[size] = (algbw_sum / TRIALS).item()" in run_section


def test_ch04_bandwidth_suite_reuses_comm_buffers() -> None:
    source = (REPO_ROOT / "ch04" / "bandwidth_benchmark_suite_multigpu.py").read_text(
        encoding="utf-8"
    )
    p2p_section = source.split("def benchmark_p2p_bandwidth", maxsplit=1)[1].split(
        "def measure_p2p_matrix", maxsplit=1
    )[0]
    matrix_section = source.split("def measure_p2p_matrix", maxsplit=1)[1].split(
        "def benchmark_collective", maxsplit=1
    )[0]
    collective_section = source.split("def benchmark_collective", maxsplit=1)[1].split(
        "def measure_collectives", maxsplit=1
    )[0]
    curve_section = source.split("def measure_latency_bandwidth_curve", maxsplit=1)[1].split(
        "def visualize_topology", maxsplit=1
    )[0]

    assert p2p_section.count("recv_tensor = torch.empty_like(tensor)") == 1
    assert "tensor = torch.empty(size, device=device, dtype=torch.float32)" in p2p_section
    assert "torch.randn(size, device=device, dtype=torch.float32)" not in p2p_section
    assert "torch.tensor([bw]" not in matrix_section
    assert "bw_tensor = torch.empty(1, device=torch.cuda.current_device()" in matrix_section
    assert "dist.all_reduce(tensor.clone())" not in collective_section
    assert "dist.all_reduce(tensor.clone())" not in curve_section
    assert "tensor = torch.empty(size, device=device, dtype=torch.float32)" in collective_section
    assert "tensor = torch.zeros(size, device=device, dtype=torch.float32)" not in collective_section
    assert "tensor = torch.empty(size_elements, device=device, dtype=torch.float32)" in curve_section
    assert "tensor = torch.zeros(size_elements, device=device, dtype=torch.float32)" not in curve_section
    assert "allgather_output = [torch.empty_like(tensor) for _ in range(world_size)]" in collective_section
    assert "dist.all_gather(allgather_output, tensor)" in collective_section
    assert "reducescatter_input = list(tensor.chunk(world_size))" in collective_section
    assert "dist.reduce_scatter(reducescatter_output, reducescatter_input)" in collective_section


def test_ch04_nccl_benchmark_reuses_collective_buffers() -> None:
    source = (REPO_ROOT / "ch04" / "nccl_benchmark.py").read_text(encoding="utf-8")
    collective_section = source.split("def benchmark_collective", maxsplit=1)[1].split(
        "def format_bandwidth", maxsplit=1
    )[0]
    setup_section = collective_section.split("def _run_collective", maxsplit=1)[0]
    run_section = collective_section.split("def _run_collective", maxsplit=1)[1].split(
        "for _ in range(warmup):", maxsplit=1
    )[0]

    assert "allgather_outputs = [torch.empty_like(tensor) for _ in range(world_size)]" in setup_section
    assert "torch.empty(tensor.numel() // world_size, device=device, dtype=tensor.dtype)" in setup_section
    assert "reducescatter_inputs = list(tensor.chunk(world_size))" in setup_section
    assert "[torch.empty_like(tensor) for _ in range(world_size)]" not in run_section
    assert "torch.empty(tensor.numel() // world_size" not in run_section
    assert "list(tensor.chunk(world_size))" not in run_section
    assert "dist.all_gather(allgather_outputs, tensor)" in run_section
    assert "dist.reduce_scatter(reducescatter_output, reducescatter_inputs)" in run_section


def test_ch04_symmetric_ring_allreduce_skips_dead_result_zero_fill() -> None:
    source = (REPO_ROOT / "ch04" / "symmetric_memory_multigpu.py").read_text(
        encoding="utf-8"
    )
    ring_section = source.split("def ring_allreduce_symmetric", maxsplit=1)[1].split(
        "def benchmark_traditional_allreduce", maxsplit=1
    )[0]

    assert "result = torch.zeros_like(tensor)" not in ring_section
    assert "result = torch.cat(chunks, dim=0)" in ring_section


def test_ch04_symmetric_memory_examples_reuse_recv_buffers() -> None:
    example_source = (REPO_ROOT / "ch04" / "symmetric_memory_example.py").read_text(
        encoding="utf-8"
    )
    multigpu_source = (REPO_ROOT / "ch04" / "symmetric_memory_multigpu.py").read_text(
        encoding="utf-8"
    )
    traditional_p2p = example_source.split("def benchmark_traditional_p2p", maxsplit=1)[1].split(
        "def benchmark_symmetric_memory", maxsplit=1
    )[0]
    example_multigpu = example_source.split("def benchmark_multigpu_symmetric_memory", maxsplit=1)[
        1
    ].split("def main", maxsplit=1)[0]
    traditional_allreduce = multigpu_source.split(
        "def benchmark_traditional_allreduce", maxsplit=1
    )[1].split("def benchmark_symmetric_memory_access", maxsplit=1)[0]
    symmetric_access = multigpu_source.split("def benchmark_symmetric_memory_access", maxsplit=1)[
        1
    ].split("def compare_performance", maxsplit=1)[0]
    ring_demo = multigpu_source.split("def demonstrate_ring_pattern", maxsplit=1)[1].split(
        "def demonstrate_butterfly_pattern", maxsplit=1
    )[0]
    butterfly_demo = multigpu_source.split("def demonstrate_butterfly_pattern", maxsplit=1)[1].split(
        "def main", maxsplit=1
    )[0]

    timed_loop_alloc = re.compile(
        r"for _ in range\((?:10|100|iterations)\):[\s\S]{0,320}"
        r"(?:torch\.empty_like\(tensor\)|tensor\.clone\(\))"
    )
    for section in (
        traditional_p2p,
        example_multigpu,
        traditional_allreduce,
        symmetric_access,
        ring_demo,
        butterfly_demo,
    ):
        assert timed_loop_alloc.search(section) is None

    assert "recv_tensor = torch.empty_like(tensor) if rank == peer_rank else None" in traditional_p2p
    assert "dist.send(tensor, dst=peer_rank)" in traditional_p2p
    assert "dist.all_reduce(tensor.clone())" not in traditional_allreduce
    assert "recv_tensor = torch.empty_like(tensor) if rank > 0 else None" in symmetric_access
    assert "recv_tensor = torch.empty_like(tensor)" in example_multigpu
    assert "recv_tensor = torch.empty_like(tensor)" in ring_demo
    assert "recv_tensor = torch.empty_like(tensor)" in butterfly_demo


def test_ch04_torchtitan_async_tp_zero_target_uses_square_mean_loss() -> None:
    source = (REPO_ROOT / "ch04" / "torchtitan_async_tp_multigpu_demo.py").read_text(
        encoding="utf-8"
    )
    main_section = source.split("def main", maxsplit=1)[1]

    assert "loss_fn = torch.nn.MSELoss()" not in main_section
    assert "target = torch.zeros_like(x)" not in main_section
    assert "loss = out.square().mean()" in main_section


def test_ch04_tensor_parallel_reuses_full_concat_buffers() -> None:
    from ch04.baseline_tensor_parallel import _replicate_tensor_parallel_shard

    local_out = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
    full_out = torch.empty(2, 2, 6)
    result = _replicate_tensor_parallel_shard(local_out, 2, full_out)

    assert result is full_out
    torch.testing.assert_close(full_out, torch.cat([local_out, local_out], dim=-1))

    files = [
        "baseline_tensor_parallel.py",
        "optimized_tensor_parallel_async.py",
        "baseline_tensor_parallel_allgather_multigpu.py",
        "optimized_tensor_parallel_allgather_multigpu.py",
        "baseline_tensor_parallel_multigpu.py",
        "optimized_tensor_parallel_multigpu.py",
    ]

    for filename in files:
        source = (REPO_ROOT / "ch04" / filename).read_text(encoding="utf-8")
        worker_section = source.split("def _run_worker", maxsplit=1)[1].split(
            "def main",
            maxsplit=1,
        )[0]
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]

        assert "self._full_out: Optional[torch.Tensor] = None" in source
        assert "self._full_out = torch.empty(" in source
        assert "def _replicate_tensor_parallel_shard(" in source
        assert "_replicate_tensor_parallel_shard(local_out, self._world_size, self._full_out)" in benchmark_section
        assert "proj_out = self._proj_layers[layer_idx](self._full_out)" in benchmark_section
        assert "torch.cat([local_out] * self._world_size, dim=-1, out=self._full_out)" not in benchmark_section
        assert "full_out = torch.cat([local_out] * self._world_size" not in benchmark_section
        assert "self._full_out = None" in source
        assert "full_out = torch.cat(gather_list, dim=-1)" not in worker_section
        if "torch.cat(gather_list" in worker_section:
            assert "torch.cat(gather_list, dim=-1, out=full_out)" in worker_section


def test_ch04_gradient_fusion_uses_dtype_byte_constant() -> None:
    source = (REPO_ROOT / "ch04" / "gradient_fusion_common.py").read_text(
        encoding="utf-8"
    )

    assert "FLOAT32_BYTES = torch.finfo(torch.float32).bits // 8" in source
    assert "torch.tensor([], dtype=torch.float32).element_size()" not in source
    assert "(self.tensor_kb * 1024) // FLOAT32_BYTES" in source
    assert "total_bytes = self.num_tensors * numel * FLOAT32_BYTES" in source


def test_ch04_gradient_fusion_seeds_accumulator_without_hot_loop_clear() -> None:
    source = (REPO_ROOT / "ch04" / "gradient_fusion_common.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "self._accum_buffer = torch.empty((), device=self.device, dtype=torch.float32)" in setup_section
    assert "self._accum_buffer = torch.zeros(" not in setup_section
    assert "accum.zero_()" not in benchmark_section
    assert "accum.copy_(self.fused_tensor.sum())" in benchmark_section
    assert "accum.copy_(self.tensors[0].sum())" in benchmark_section


def test_dtype_byte_sizing_avoids_empty_tensor_metadata_allocations() -> None:
    files = [
        "ch04/gradient_fusion_multigpu.py",
        "ch04/nccl_benchmark.py",
        "ch04/nvshmem_vs_nccl_benchmark.py",
        "ch11/baseline_tensor_cores_streams.py",
        "ch11/optimized_tensor_cores_streams.py",
        "ch11/stream_overlap_base.py",
        "ch15/placement_sim.py",
        "ch16/symmetric_memory_inference.py",
        "ch16/gpt_quick_test.py",
        "ch16/inference_serving_multigpu.py",
        "cluster/scripts/allreduce_latency_comp.py",
        "cluster/scripts/torch_gpu_stream_bench.py",
        "labs/blackwell_matmul/run_blackwell_matmul.py",
        "labs/flexattention/flexattention_common.py",
        "labs/train_distributed/baseline_zero2_multigpu.py",
        "labs/train_distributed/ddp_compression.py",
        "labs/train_distributed/optimized_zero2_multigpu.py",
    ]

    for filename in files:
        source = (REPO_ROOT / filename).read_text(encoding="utf-8")
        assert "torch.tensor([], dtype=" not in source
        assert "torch.empty((), dtype=" not in source
        assert "torch.tensor(0, dtype=" not in source

    assert "FLOAT16_BYTES = torch.finfo(torch.float16).bits // 8" in (
        REPO_ROOT / "ch04" / "gradient_fusion_multigpu.py"
    ).read_text(encoding="utf-8")
    assert "FLOAT16_BYTES = torch.finfo(torch.float16).bits // 8" in (
        REPO_ROOT / "ch04" / "nvshmem_vs_nccl_benchmark.py"
    ).read_text(encoding="utf-8")
    assert "return torch.finfo(dtype).bits // 8" in (
        REPO_ROOT / "ch15" / "placement_sim.py"
    ).read_text(encoding="utf-8")


def test_ch15_placement_sim_batches_session_rng_samples() -> None:
    source = (REPO_ROOT / "ch15" / "placement_sim.py").read_text(encoding="utf-8")
    simulate_section = source.split("def simulate", maxsplit=1)[1].split(
        "def _prefill_latency_ms",
        maxsplit=1,
    )[0]

    assert "prompt_token_samples = torch.randint(" in simulate_section
    assert "decode_token_samples = torch.randint(" in simulate_section
    assert "(sessions,)" in simulate_section
    assert "for sess_idx, (prompt_tokens, decode_tokens) in enumerate(" in simulate_section
    assert ".item()" not in simulate_section

    from ch15.placement_sim import PlacementConfig, PlacementSimulator

    cfg = PlacementConfig(
        prefill_tp_size=2,
        prefill_span_nodes=False,
        decode_tp_size=1,
        decode_span_nodes=False,
        decode_microbatch=4,
        remote_expert_fraction=0.25,
        router_sticky_decode=True,
        kv_transfer_policy="local_only",
        prompt_tokens=(8, 16),
        decode_tokens=(2, 6),
    )
    simulator = PlacementSimulator()

    first = simulator.simulate(cfg, sessions=8, seed=123)
    second = simulator.simulate(cfg, sessions=8, seed=123)

    assert first == second
    assert first.sessions == 8
    assert len(first.ttft_ms) == 8
    assert len(first.decode_ms) == 8


def test_ch16_symmetric_memory_checksum_reduces_on_device() -> None:
    source = (REPO_ROOT / "ch16" / "symmetric_memory_inference.py").read_text(
        encoding="utf-8"
    )
    multi_model_section = source.split("def demo_multi_model", maxsplit=1)[1].split(
        "# ============================================================================",
        maxsplit=1,
    )[0]

    assert "checksum = weights[:1024].float().sum()" in multi_model_section
    assert "dist.all_reduce(checksum)" in multi_model_section
    assert ".sum().item()" not in multi_model_section
    assert "torch.tensor(checksum" not in multi_model_section


def test_ch04_nvshmem_pipeline_defers_loss_materialization() -> None:
    source = (REPO_ROOT / "ch04" / "nvshmem_pipeline_parallel_multigpu.py").read_text(
        encoding="utf-8"
    )
    schedule_section = source.split("def run_1f1b_schedule", maxsplit=1)[1].split(
        "def close",
        maxsplit=1,
    )[0]

    assert "losses.append(loss.item())" not in schedule_section
    assert "loss_tensors.append(loss.detach())" in schedule_section
    assert "torch.stack(loss_tensors).detach().cpu().tolist()" in schedule_section


def test_ch04_training_pipeline_defers_step_loss_sync_until_logging() -> None:
    source = (REPO_ROOT / "ch04" / "training_multigpu_pipeline.py").read_text(
        encoding="utf-8"
    )
    train_step_section = source.split("def train_step", maxsplit=1)[1].split(
        "def train",
        maxsplit=1,
    )[0]
    train_loop_section = source.split("def train", maxsplit=1)[1].split(
        "# ============================================================================",
        maxsplit=1,
    )[0]

    assert "return loss.item()" not in train_step_section
    assert "return loss.detach()" in train_step_section
    assert "loss_value = float(loss)" in train_loop_section


def test_ch04_nvshmem_training_example_defers_reduced_norm_sync() -> None:
    source = (REPO_ROOT / "ch04" / "nvshmem_training_example.py").read_text(
        encoding="utf-8"
    )
    bucket_demo = source.split("def demo_gradient_bucket", maxsplit=1)[1].split(
        "# ============================================================================",
        maxsplit=1,
    )[0]
    loop_section = bucket_demo.split("for _ in range(steps):", maxsplit=1)[1].split(
        "dist.barrier()",
        maxsplit=1,
    )[0]

    assert "bucket.tensor.norm().item()" not in bucket_demo
    assert "offset = 0" in loop_section
    assert "bucket.tensor.norm()" not in loop_section
    assert "reduced_norm = float(bucket.tensor.norm())" in bucket_demo


def test_ch09_fusion_gelu_reuses_scalar_constant() -> None:
    source = (REPO_ROOT / "ch09" / "fusion_pytorch.py").read_text(encoding="utf-8")

    assert "GELU_TANH_SCALE = math.sqrt(2.0 / math.pi)" in source
    assert "torch.sqrt(torch.tensor(2.0 / torch.pi))" not in source
    assert "GELU_TANH_SCALE * (x + 0.044715 * torch.pow(x, 3))" in source
    assert "GELU_TANH_SCALE * (scaled + 0.044715 * torch.pow(scaled, 3))" in source


def test_ch14_nccl_quantization_defers_verification_clones_and_syncs() -> None:
    baseline_source = (REPO_ROOT / "ch14" / "baseline_nccl_quantization.py").read_text(
        encoding="utf-8"
    )
    optimized_source = (REPO_ROOT / "ch14" / "optimized_nccl_quantization.py").read_text(
        encoding="utf-8"
    )
    baseline_benchmark = baseline_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    baseline_capture = baseline_source.split(
        "def capture_verification_payload", maxsplit=1
    )[1].split("def teardown", maxsplit=1)[0]
    optimized_benchmark = optimized_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    optimized_setup = optimized_source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    optimized_capture = optimized_source.split(
        "def capture_verification_payload", maxsplit=1
    )[1].split("def teardown", maxsplit=1)[0]

    assert "self.tensor.detach().clone()" not in baseline_benchmark
    assert "self.output = self.tensor.detach()" in baseline_benchmark
    assert "output=self.output.detach().clone()" in baseline_capture
    assert "float(dequant.sum())" not in optimized_benchmark
    assert "self.output = dequant.clone()" not in optimized_benchmark
    assert "self._abs_buffer = torch.empty_like(self.tensor)" in optimized_setup
    assert "self.quantized = torch.empty_like(self.tensor, dtype=torch.int8)" in optimized_setup
    assert "self.dequantized = torch.empty_like(self.tensor)" in optimized_setup
    assert "self.tensor.abs()" not in optimized_benchmark
    assert ".to(torch.int8)" not in optimized_benchmark
    assert "quantized.float()" not in optimized_benchmark
    assert "torch.abs(self.tensor, out=self._abs_buffer)" in optimized_benchmark
    assert "torch.amax(self._abs_buffer, dim=1, keepdim=True, out=self._max_abs)" in optimized_benchmark
    assert "torch.mul(self.tensor, self._scales, out=self._quant_float)" in optimized_benchmark
    assert "self.quantized.copy_(self._quant_float)" in optimized_benchmark
    assert "self._quantized_float.copy_(self.quantized)" in optimized_benchmark
    assert "torch.div(self._quantized_float, self._scales, out=self.dequantized)" in optimized_benchmark
    assert "self.output = self.dequantized.detach()" in optimized_benchmark
    assert "output=self.output.detach().clone()" in optimized_capture


def test_ch14_benchmarks_do_not_force_output_sum_syncs() -> None:
    files = [
        "baseline_sliding_window.py",
        "optimized_sliding_window.py",
        "sliding_window_demo.py",
        "baseline_triton_persistent.py",
        "optimized_triton_persistent.py",
        "triton_persistent_batched_bench.py",
        "triton_persistent_demo.py",
        "flex_attention_sparse_demo.py",
    ]

    for filename in files:
        source = (REPO_ROOT / "ch14" / filename).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]
        assert "float(self.output.sum())" not in benchmark_section
        assert "float(self.output.detach().sum())" not in benchmark_section


def test_ch14_triton_persistent_demo_batches_correctness_error_reads() -> None:
    source = (REPO_ROOT / "ch14" / "triton_persistent_demo.py").read_text(
        encoding="utf-8"
    )
    verify_section = source.split("print(\"\\nVerifying correctness...\")", maxsplit=1)[1].split(
        "#============================================================================",
        maxsplit=1,
    )[0]

    assert "std_err, pers_err, atom_err = torch.stack(" in verify_section
    assert ").tolist()" in verify_section
    assert ".abs().max().item()" not in verify_section


def test_ch14_triton_examples_batches_fp8_error_reads() -> None:
    source = (REPO_ROOT / "ch14" / "triton_examples.py").read_text(encoding="utf-8")
    fp8_section = source.split("def benchmark_fp8_vs_fp16", maxsplit=1)[1].split(
        "@triton.jit",
        maxsplit=1,
    )[0]

    assert "max_diff, mean_diff = torch.stack((diff.max(), diff.mean())).tolist()" in fp8_section
    assert ".abs().max().item()" not in fp8_section
    assert ".abs().mean().item()" not in fp8_section


def test_ch14_triton_tma_batches_correctness_error_reads() -> None:
    source = (REPO_ROOT / "ch14" / "triton_tma_blackwell.py").read_text(
        encoding="utf-8"
    )
    benchmark_section = source.split("def benchmark_tma_vs_standard", maxsplit=1)[1].split(
        "def demonstrate_tma_features",
        maxsplit=1,
    )[0]

    assert "max_bias_diff, max_diff = torch.stack(" in benchmark_section
    assert "torch.abs(C_bias - C_ref).max()" in benchmark_section
    assert "torch.abs(C_tma - C_torch).max()" in benchmark_section
    assert ".max().item()" not in benchmark_section


def test_ch14_triton_nvshmem_batches_result_reads() -> None:
    source = (REPO_ROOT / "ch14" / "triton_nvshmem_example.py").read_text(encoding="utf-8")
    operation_section = source.split("def triton_multi_gpu_operation", maxsplit=1)[1].split(
        "def pytorch_symmetric_memory_approach",
        maxsplit=1,
    )[0]
    example_section = source.split("results = triton_multi_gpu_operation(tensors)", maxsplit=1)[
        1
    ].split(
        "print(\"\\n\" + \"=\" * 80)",
        maxsplit=1,
    )[0]

    assert "result_values = torch.cat(" in example_section
    assert "non_blocking=True" in example_section
    assert "[r.item() for r in results]" not in example_section
    assert "tl.store(output_ptr, local_sum)" in source
    assert "outputs = [torch.empty(1, device=t.device, dtype=t.dtype) for t in tensors]" in operation_section
    assert "outputs = [torch.zeros(" not in operation_section


def test_custom_vs_cublas_batches_correctness_scale_reads() -> None:
    source = (REPO_ROOT / "labs" / "custom_vs_cublas" / "run_lab.py").read_text(
        encoding="utf-8"
    )
    verify_section = source.split("def verify_correctness", maxsplit=1)[1].split(
        "def main",
        maxsplit=1,
    )[0]

    assert "max_diff, ref_abs_max = torch.stack(" in verify_section
    assert "(ref_fp32 - result_fp32).abs().max()" in verify_section
    assert "ref_fp32.abs().max()" in verify_section
    assert ".abs().max().item()" not in verify_section
    assert "C = torch.empty(M, N, device='cuda', dtype=torch.float32)" in source
    assert "C = torch.zeros(M, N, device='cuda', dtype=torch.float32)" not in source


def test_custom_vs_cublas_timing_helpers_use_cuda_events() -> None:
    runner_source = (REPO_ROOT / "labs" / "custom_vs_cublas" / "run_lab.py").read_text(
        encoding="utf-8"
    )
    autotune_source = (
        REPO_ROOT / "labs" / "custom_vs_cublas" / "autotune.py"
    ).read_text(encoding="utf-8")
    runner_section = runner_source.split("def benchmark_kernel", maxsplit=1)[1].split(
        "def calculate_tflops",
        maxsplit=1,
    )[0]
    autotune_section = autotune_source.split("def _benchmark_kernel", maxsplit=1)[
        1
    ].split("# Available kernels", maxsplit=1)[0]

    assert runner_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "end.synchronize()" in runner_section
    assert "return elapsed_ms" in runner_section
    assert "start.elapsed_time(end) / iters" in runner_section
    assert "time.time()" not in runner_section
    assert autotune_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "end_event.synchronize()" in autotune_section
    assert "times.append(start_event.elapsed_time(end_event))" in autotune_section
    assert "time.perf_counter()" not in autotune_section


def test_ch14_tma_config_benchmark_avoids_zero_filling_output() -> None:
    source = (REPO_ROOT / "ch14" / "benchmark_tma_configs.py").read_text(encoding="utf-8")

    assert "tl.store(c_ptrs, acc, mask=c_mask)" in source
    assert "C = torch.empty(M, N, device='cuda', dtype=torch.float32)" in source
    assert "C = torch.zeros(M, N, device='cuda', dtype=torch.float32)" not in source


def test_ch14_cublas_vs_cutlass_pair_skips_unused_setup_output_allocation() -> None:
    for filename, assignment in (
        ("baseline_cublas_vs_cutlass.py", "self.C = self._cublas_gemm(self.A, self.B)"),
        ("optimized_cublas_vs_cutlass.py", "self.C = self._cutlass_gemm(self.A, self.B)"),
    ):
        source = (REPO_ROOT / "ch14" / filename).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split(
            "def benchmark_fn",
            maxsplit=1,
        )[0]
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]

        assert "self.C = None" in setup_section
        assert "self.C = torch.zeros(" not in setup_section
        assert "self.C = torch.empty(" not in setup_section
        assert assignment in benchmark_section


def test_ch13_arithmetic_intensity_setup_avoids_redundant_zero_fill() -> None:
    source = (REPO_ROOT / "ch13" / "baseline_arithmetic_intensity.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def _chunked_matmul",
        maxsplit=1,
    )[0]
    chunked_section = source.split("def _chunked_matmul", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]

    assert "self.C = torch.empty(self.M, self.N, device=self.device, dtype=torch.float32)" in setup_section
    assert "self.C = torch.zeros(" not in setup_section
    assert "self.C.zero_()" not in chunked_section
    assert "torch.mm(self.A[:, :first_end], self.B[:first_end, :], out=self.C)" in chunked_section
    assert "for k in range(first_end, self.K, self.block_k):" in chunked_section


def test_ch03_ch05_accumulator_buffers_skip_setup_zero_fill() -> None:
    targets = (
        ("ch03/baseline_gemm.py", "self._output_buffer = torch.empty(", "result.zero_()"),
        ("ch05/baseline_vectorization.py", "self._output_buffer = torch.empty(", "result.zero_()"),
    )
    for filename, allocation, reset in targets:
        source = (REPO_ROOT / filename).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split(
            "def benchmark_fn",
            maxsplit=1,
        )[0]
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]

        assert allocation in setup_section
        assert "self._output_buffer = torch.zeros(" not in setup_section
        assert reset in benchmark_section


def test_ch05_optimized_vectorization_reuses_reduction_output_buffer() -> None:
    source = (REPO_ROOT / "ch05" / "optimized_vectorization.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert "self._output_buffer: Optional[torch.Tensor] = None" in source
    assert "self._output_buffer = torch.empty(1, device=self.device)" in setup_section
    assert "with torch.inference_mode(), self._nvtx_range(\"optimized_vectorization\"):" in benchmark_section
    assert "torch.sum(self.data, dim=0, keepdim=True, out=self._output_buffer)" in benchmark_section
    assert "self.data.sum().unsqueeze(0)" not in benchmark_section
    assert "torch.empty(" not in benchmark_section


def test_early_chapter_mlp_benchmarks_use_inplace_relu_modules() -> None:
    for relative in (
        "ch03/baseline_pinned_prefetch_mlp.py",
        "ch03/optimized_pinned_prefetch_mlp.py",
        "ch03/baseline_double_buffered_batch_provisioning.py",
        "ch03/optimized_double_buffered_batch_provisioning.py",
        "ch03/bind_numa_affinity.py",
        "ch04/baseline_nccl.py",
        "ch04/optimized_nccl.py",
        "ch04/baseline_cpu_reduction.py",
        "ch04/optimized_cpu_reduction.py",
        "ch04/baseline_disaggregated.py",
        "ch04/optimized_disaggregated.py",
        "ch04/baseline_disaggregated_multigpu.py",
        "ch04/optimized_disaggregated_multigpu.py",
        "ch04/symmetric_memory_training_advanced.py",
        "ch05/ai_common.py",
        "ch05/storage_io_optimization.py",
        "ch09/baseline_compute_bound.py",
        "ch09/optimized_compute_bound.py",
        "ch10/baseline_batch.py",
        "ch10/optimized_batch.py",
        "ch19/baseline_memory_double_buffering.py",
        "ch19/optimized_memory_double_buffering.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")

        assert "ReLU(inplace=True)" in source
        assert "nn.ReLU()" not in source
        assert "torch.nn.ReLU()" not in source


def test_ch09_compute_bound_baseline_uses_inference_mode_and_cached_nvtx() -> None:
    source = (REPO_ROOT / "ch09" / "baseline_compute_bound.py").read_text(encoding="utf-8")
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert 'with torch.inference_mode(), self._nvtx_range("baseline_compute_bound"):' in benchmark_section
    assert "get_nvtx_enabled(" not in benchmark_section
    assert "with nvtx_range(" not in benchmark_section
    assert "from core.profiling.nvtx_helper" not in source


def test_ch09_memory_and_triton_baselines_use_cached_nvtx() -> None:
    for filename, label in (
        ("baseline_memory_bound.py", "baseline_memory_bound"),
        ("baseline_triton.py", "baseline_triton"),
    ):
        source = (REPO_ROOT / "ch09" / filename).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]

        assert f'with self._nvtx_range("{label}"):' in benchmark_section
        assert "get_nvtx_enabled(" not in benchmark_section
        assert "with nvtx_range(" not in benchmark_section
        assert "from core.profiling.nvtx_helper" not in source


def test_pipeline_and_demo_activation_paths_use_inplace_relu() -> None:
    for relative in (
        "ch04/baseline_pipeline_parallel.py",
        "ch04/baseline_pipeline_parallel_multigpu.py",
        "ch04/optimized_pipeline_parallel_multigpu_1f1b.py",
        "ch04/ddp_no_overlap.py",
        "ch04/ddp_overlap.py",
        "ch13/fp8_static_demo.py",
        "ch13/memory_profiling.py",
        "ch13/baseline_warp_specialization_training.py",
        "ch14/train.py",
        "ch15/pipeline_parallel_demo.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")

        assert "torch.relu_(" in source or "nn.ReLU(inplace=True)" in source
        assert "torch.relu(" not in source
        assert "nn.ReLU()" not in source


def test_ch04_eval_reduction_and_disagg_paths_use_inference_mode() -> None:
    for relative in (
        "ch04/baseline_nccl.py",
        "ch04/baseline_cpu_reduction.py",
        "ch04/baseline_disaggregated.py",
        "ch04/optimized_disaggregated.py",
        "ch04/baseline_disaggregated_multigpu.py",
        "ch04/optimized_disaggregated_multigpu.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]

        assert "with torch.inference_mode():" in benchmark_section
        assert "with torch.no_grad():" not in benchmark_section


def test_ch06_ch12_cuda_output_buffers_skip_setup_zero_fill() -> None:
    targets = (
        "ch06/baseline_launch_bounds.py",
        "ch06/optimized_launch_bounds.py",
        "ch12/baseline_work_queue.py",
        "ch12/optimized_work_queue.py",
    )
    for filename in targets:
        source = (REPO_ROOT / filename).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split(
            "def benchmark_fn",
            maxsplit=1,
        )[0]

        assert "self.output_data = torch.empty(self.N, dtype=torch.float32, device=self.device)" in setup_section
        assert "self.output_data = torch.zeros(" not in setup_section

    launch_kernels = (REPO_ROOT / "ch06" / "cuda_extensions" / "launch_bounds_kernels.cu").read_text(
        encoding="utf-8"
    )
    work_queue_kernels = (
        REPO_ROOT / "ch12" / "cuda_extensions" / "work_queue_kernels.cu"
    ).read_text(encoding="utf-8")
    assert "output[idx] = launch_bounds_workload(input[idx]);" in launch_kernels
    assert "output[idx] = sum;" in work_queue_kernels


def test_ch04_optimized_nccl_reduction_buffers_skip_setup_zero_fill() -> None:
    source = (REPO_ROOT / "ch04" / "optimized_nccl.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert "self._output_buffer = torch.empty(" in setup_section
    assert "self._reduction_buffer = torch.empty_like(self._output_buffer)" in setup_section
    assert "self._output_buffer = torch.zeros(" not in setup_section
    assert "self._reduction_buffer = torch.zeros(" not in setup_section
    assert "with torch.inference_mode():" in benchmark_section
    assert "torch.chunk(" not in benchmark_section
    assert "for shard in" not in benchmark_section
    assert "shard_view = out.reshape(self.num_shards, reduced_rows, out.shape[1])" in benchmark_section
    assert "torch.sum(shard_view, dim=0, out=self._reduction_buffer)" in benchmark_section
    assert "self._reduction_buffer.zero_()" not in benchmark_section
    assert "self._output_buffer.copy_(self._reduction_buffer)" in benchmark_section


def test_ch04_optimized_gpu_reduction_uses_single_gpu_sum_kernel() -> None:
    source = (REPO_ROOT / "ch04" / "optimized_cpu_reduction.py").read_text(encoding="utf-8")
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert "torch.chunk(" not in benchmark_section
    assert "for shard in" not in benchmark_section
    assert "torch.sum(shard_view, dim=0, out=self._reduction_buffer)" in benchmark_section
    assert "self._reduction_buffer.zero_()" not in benchmark_section


def test_moe_cuda_naive_backend_skips_redundant_mask_any_sync() -> None:
    source = (REPO_ROOT / "labs" / "moe_cuda" / "moe_backend_common.py").read_text(
        encoding="utf-8"
    )
    naive_section = source.split("def forward_naive", maxsplit=1)[1].split(
        "def forward_vectorized",
        maxsplit=1,
    )[0]

    assert "token_ids, slot_ids = (idx == expert).nonzero(as_tuple=True)" in naive_section
    assert "if token_ids.numel() == 0:" in naive_section
    assert "torch.relu_(h)" in source
    assert "torch.relu(h)" not in source
    assert "torch.any(mask)" not in naive_section
    assert "mask.nonzero" not in naive_section
    assert "@torch.inference_mode()\ndef select_best_backend" in source
    assert "@torch.no_grad()\ndef select_best_backend" not in source
    assert 'sync_device = x.device if x.device.type == "cuda" else None' in source
    assert "torch.cuda.synchronize()" not in source


def test_custom_vs_cublas_dual_benches_batch_relative_error_reads() -> None:
    dual_cta = (REPO_ROOT / "labs" / "custom_vs_cublas" / "bench_dual_cta.py").read_text(
        encoding="utf-8"
    )
    dual_fp8 = (
        REPO_ROOT / "labs" / "custom_vs_cublas" / "bench_dual_2sm_fp8.py"
    ).read_text(encoding="utf-8")
    dual_nvfp4 = (
        REPO_ROOT / "labs" / "custom_vs_cublas" / "bench_dual_2sm_nvfp4.py"
    ).read_text(encoding="utf-8")

    dual_cta_check = dual_cta.split("def check", maxsplit=1)[1].split("def report", maxsplit=1)[0]
    dual_fp8_check = dual_fp8.split("def check", maxsplit=1)[1].split("def report", maxsplit=1)[0]
    dual_fp8_main = dual_fp8.split("if args.with_fp16:", maxsplit=1)[1].split(
        "if args.sol > 0:",
        maxsplit=1,
    )[0]
    dual_nvfp4_rel = dual_nvfp4.split("def rel_err", maxsplit=1)[1].split("def report", maxsplit=1)[0]

    for section in (dual_cta_check, dual_fp8_check, dual_fp8_main, dual_nvfp4_rel):
        assert "torch.stack(" in section
        assert ".abs().max().item()" not in section
        assert ".abs().max().item() /" not in section


def test_custom_vs_cublas_dual_benches_cache_device_constants() -> None:
    dual_fp8 = (
        REPO_ROOT / "labs" / "custom_vs_cublas" / "bench_dual_2sm_fp8.py"
    ).read_text(encoding="utf-8")
    dual_nvfp4 = (
        REPO_ROOT / "labs" / "custom_vs_cublas" / "bench_dual_2sm_nvfp4.py"
    ).read_text(encoding="utf-8")

    dual_fp8_make = dual_fp8.split("def make_fp8_data", maxsplit=1)[1].split(
        "def fp32_ref",
        maxsplit=1,
    )[0]
    dual_nvfp4_decode = dual_nvfp4.split("def decode_codes", maxsplit=1)[1].split(
        "def make_exact_data",
        maxsplit=1,
    )[0]
    dual_nvfp4_exact = dual_nvfp4.split("def make_exact_data", maxsplit=1)[1].split(
        "def quantize_nvfp4",
        maxsplit=1,
    )[0]
    dual_nvfp4_quantize = dual_nvfp4.split("def quantize_nvfp4", maxsplit=1)[1].split(
        "def make_randn_data",
        maxsplit=1,
    )[0]
    dual_nvfp4_ref = dual_nvfp4.split("def fp32_ref", maxsplit=1)[1].split(
        "def scaled_mm_nvfp4",
        maxsplit=1,
    )[0]
    dual_nvfp4_gates = dual_nvfp4.split("def run_gates", maxsplit=1)[1].split(
        "def main",
        maxsplit=1,
    )[0]

    assert "fp8_exact_values(" in dual_fp8_make
    assert "torch.tensor(" not in dual_fp8_make
    assert "e2m1_values(codes.device)" in dual_nvfp4_decode
    assert "sign =" not in dual_nvfp4_decode
    assert "return torch.where(codes >= 8, -mag, mag)" in dual_nvfp4_decode
    assert "torch.tensor(" not in dual_nvfp4_exact
    assert "e2m1_values(x.device)" in dual_nvfp4_quantize
    assert "E2M1_VALS.to(" not in dual_nvfp4_decode + dual_nvfp4_quantize
    assert "def expand_scale_blocks" in dual_nvfp4
    assert "repeat_interleave(" not in dual_nvfp4_ref
    assert "af = a_deq * expand_scale_blocks(sa)" in dual_nvfp4_ref
    assert "bf = b_deq * expand_scale_blocks(sb)" in dual_nvfp4_ref
    assert "fp8_gate_values(" in dual_nvfp4_gates
    assert "torch.tensor(" not in dual_nvfp4_gates


def test_nvfp4_gemv_dequant_expands_scales_without_repeat_interleave() -> None:
    from labs.nvfp4_gemv import optimized_submission as submission

    source = inspect.getsource(submission._nvfp4_dequant_gemv_one)
    scales_2d = torch.arange(6, dtype=torch.float32).view(2, 3)
    scales_1d = torch.arange(3, dtype=torch.float32)

    assert "repeat_interleave" not in source
    assert "torch.stack" not in source
    assert "_expand_scale_blocks(sfa)" in source
    assert "_expand_scale_blocks(sfb)" in source
    packed = torch.tensor([[0x21, 0x43]], dtype=torch.uint8)
    torch.testing.assert_close(
        submission._unpack_nvfp4_indices(packed),
        torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
    )
    torch.testing.assert_close(
        submission._expand_scale_blocks(scales_2d, 4),
        scales_2d.repeat_interleave(4, dim=-1),
    )
    torch.testing.assert_close(
        submission._expand_scale_blocks(scales_1d, 4),
        scales_1d.repeat_interleave(4, dim=-1),
    )


def test_custom_vs_cublas_nvfp4_blocked_padding_zeroes_only_tails() -> None:
    source = (
        REPO_ROOT / "labs" / "custom_vs_cublas" / "bench_dual_2sm_nvfp4.py"
    ).read_text(encoding="utf-8")
    blocked_section = source.split("def to_blocked", maxsplit=1)[1].split(
        "def pack_codes",
        maxsplit=1,
    )[0]

    assert "padded = torch.zeros(rb * 128, cb * 4" not in blocked_section
    assert "padded = torch.empty(rb * 128, cb * 4" in blocked_section
    assert "padded[rows:, :].zero_()" in blocked_section
    assert "padded[:rows, cols:].zero_()" in blocked_section


def test_nvfp4_utils_reuse_nonzero_indices_for_mismatch_counts() -> None:
    targets = (
        REPO_ROOT / "labs" / "nvfp4_gemm" / "utils.py",
        REPO_ROOT / "labs" / "nvfp4_dual_gemm" / "utils.py",
    )

    for path in targets:
        source = path.read_text(encoding="utf-8")

        assert source.count("mismatched_indices = torch.nonzero(mismatched, as_tuple=False)") == 2
        assert source.count("num_mismatched = int(mismatched_indices.shape[0])") == 2
        assert (
            source.count(
                "mismatch_index_rows = mismatched_indices[:max_print].detach().cpu().tolist()"
            )
            == 2
        )
        assert "mismatched.count_nonzero().item()" not in source
        assert "index.tolist()" not in source
        assert source.count("@torch.inference_mode()") == 2
        assert "@torch.no_grad()" not in source


def test_attention_baselines_cache_causal_masks_outside_forward() -> None:
    ch14_source = (REPO_ROOT / "ch14" / "baseline_sliding_window.py").read_text(
        encoding="utf-8"
    )
    ch14_forward = ch14_source.split("def forward", maxsplit=1)[1].split(
        "class BaselineSlidingWindowBenchmark",
        maxsplit=1,
    )[0]
    assert 'self.register_buffer(\n            "_causal_mask",' in ch14_source
    assert "def _causal_mask_for" in ch14_source
    assert "causal_mask = self._causal_mask_for(S, x.device)" in ch14_forward
    assert "torch.ones(S, S" not in ch14_forward
    ch14_mask_for = ch14_source.split("def _causal_mask_for", maxsplit=1)[1].split(
        "def forward",
        maxsplit=1,
    )[0]
    assert "torch.triu(" not in ch14_mask_for
    assert "torch.ones(" not in ch14_mask_for
    assert "self._causal_mask = pos.unsqueeze(0) > pos.unsqueeze(1)" in ch14_mask_for

    ch10_source = (REPO_ROOT / "ch10" / "baseline_flashattention3_pipeline.py").read_text(
        encoding="utf-8"
    )
    ch10_forward = ch10_source.split("def forward", maxsplit=1)[1].split(
        "class BaselineFlashAttention3Benchmark",
        maxsplit=1,
    )[0]
    assert 'self.register_buffer(\n            "_causal_mask",' in ch10_source
    assert "def _causal_mask_for" in ch10_source
    assert "mask = self._causal_mask_for(seq_len, x.device)" in ch10_forward
    assert "torch.ones(seq_len, seq_len" not in ch10_forward
    ch10_mask_for = ch10_source.split("def _causal_mask_for", maxsplit=1)[1].split(
        "def forward",
        maxsplit=1,
    )[0]
    assert "torch.triu(" not in ch10_mask_for
    assert "torch.ones(" not in ch10_mask_for
    assert "self._causal_mask = pos.unsqueeze(0) > pos.unsqueeze(1)" in ch10_mask_for

    ch13_source = (REPO_ROOT / "ch13" / "baseline_long_context_attention.py").read_text(
        encoding="utf-8"
    )
    ch13_setup = ch13_source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    assert "torch.triu(torch.ones(" not in ch13_setup
    assert "pos = torch.arange(self.seq_len, device=self.device)" in ch13_setup
    assert "mask = pos.unsqueeze(0) > pos.unsqueeze(1)" in ch13_setup

    llama_source = (
        REPO_ROOT / "labs" / "real_world_models" / "llama_3_1_8b_optimization.py"
    ).read_text(encoding="utf-8")
    llama_attention = llama_source.split("class SimplifiedAttention", maxsplit=1)[1].split(
        "def forward",
        maxsplit=1,
    )[0]
    assert "torch.triu(torch.ones(" not in llama_attention
    assert "pos = torch.arange(seq_len)" in llama_attention
    assert "causal = pos.unsqueeze(0) > pos.unsqueeze(1)" in llama_attention

    ch16_source = (REPO_ROOT / "ch16" / "baseline_dense_attention_flash.py").read_text(
        encoding="utf-8"
    )
    ch16_setup = ch16_source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    assert "torch.triu(" not in ch16_setup
    assert "torch.ones(self.max_seq_len" not in ch16_setup
    assert "self._causal_mask = pos.unsqueeze(0) > pos.unsqueeze(1)" in ch16_setup

    for filename in (
        "baseline_dense_attention_flash.py",
        "optimized_dense_attention_flash.py",
        "optimized_dense_attention_flash_blackwell_variant.py",
    ):
        dense_source = (REPO_ROOT / "ch16" / filename).read_text(encoding="utf-8")
        dense_setup = dense_source.split("def setup", maxsplit=1)[1].split(
            "def _forward",
            maxsplit=1,
        )[0]
        dense_benchmark = dense_source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]
        assert "with torch.inference_mode():" in dense_setup
        assert "with torch.inference_mode():" in dense_benchmark
        assert "with torch.no_grad():" not in dense_setup
        assert "with torch.no_grad():" not in dense_benchmark
        assert "self._enable_nvtx = get_nvtx_enabled(config) if config else False" in dense_setup
        assert "self._payload_parameter_count = sum(p.numel() for p in self.qkv_proj.parameters())" in dense_setup
        assert "get_config()" not in dense_benchmark
        assert "get_nvtx_enabled(" not in dense_benchmark
        assert "enable=self._enable_nvtx" in dense_benchmark
        assert "sum(p.numel()" not in dense_benchmark
        assert "parameter_count += " not in dense_benchmark
        assert "self._payload_parameter_count = parameter_count" not in dense_benchmark

    for filename in ("baseline_flash_sdp.py", "optimized_flash_sdp.py"):
        flash_source = (REPO_ROOT / "ch16" / filename).read_text(encoding="utf-8")
        flash_setup = flash_source.split("def setup", maxsplit=1)[1].split(
            "def benchmark_fn",
            maxsplit=1,
        )[0]
        flash_benchmark = flash_source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]
        flash_capture = flash_source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def teardown",
            maxsplit=1,
        )[0]
        assert "self._enable_nvtx = get_nvtx_enabled(config) if config else False" in flash_setup
        assert "self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())" in flash_setup
        assert "with torch.inference_mode():" in flash_benchmark
        assert "self.output = self.model(self.inputs)" in flash_benchmark
        assert "get_config()" not in flash_benchmark
        assert "get_nvtx_enabled(" not in flash_benchmark
        assert "enable=self._enable_nvtx" in flash_benchmark
        assert "sum(p.numel()" not in flash_benchmark
        assert "parameter_count=self._payload_parameter_count" in flash_capture
        assert "sum(p.numel()" not in flash_capture

    optimized_flash_source = (REPO_ROOT / "ch16" / "optimized_flash_sdp.py").read_text(
        encoding="utf-8"
    )
    flash_module_section = optimized_flash_source.split("class FlashAttentionModule", maxsplit=1)[1].split(
        "class OptimizedFlashSDPBenchmark",
        maxsplit=1,
    )[0]
    assert "self._flash_backends = [SDPBackend.FLASH_ATTENTION]" in flash_module_section
    assert "with sdpa_kernel(self._flash_backends):" in flash_module_section
    assert "with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):" not in flash_module_section

    baseline_flash_setup = (REPO_ROOT / "ch16" / "baseline_flash_sdp.py").read_text(
        encoding="utf-8"
    ).split("def setup", maxsplit=1)[1].split("def benchmark_fn", maxsplit=1)[0]
    optimized_flash_setup = (REPO_ROOT / "ch16" / "optimized_flash_sdp.py").read_text(
        encoding="utf-8"
    ).split("def setup", maxsplit=1)[1].split("def benchmark_fn", maxsplit=1)[0]
    assert "with torch.inference_mode():" in baseline_flash_setup
    assert "with torch.no_grad():" not in baseline_flash_setup
    assert "with torch.inference_mode():" in optimized_flash_setup

    cudnn_sdpa_source = (
        REPO_ROOT / "labs" / "cudnn_sdpa_bench" / "baseline_flash_sdp.py"
    ).read_text(encoding="utf-8")
    cudnn_sdpa_benchmark = cudnn_sdpa_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]
    assert "with torch.inference_mode(), _sdpa_context(backend):" in cudnn_sdpa_source
    assert "with torch.inference_mode():" in cudnn_sdpa_benchmark
    assert "self.output = self.model(self.inputs)" in cudnn_sdpa_benchmark
    assert "out.detach()" not in cudnn_sdpa_benchmark

    for relative_path in (
        "ch09/baseline_sdpa_attention.py",
        "ch09/optimized_sdpa_attention.py",
        "ch10/baseline_attention.py",
        "ch10/optimized_attention.py",
        "ch10/baseline_batch.py",
        "ch10/optimized_batch.py",
        "ch10/baseline_flash_attention.py",
        "ch10/optimized_flash_attention.py",
    ):
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]
        assert "with torch.inference_mode():" in benchmark_section
        assert "with torch.no_grad():" not in benchmark_section

    for relative_path in (
        "ch10/baseline_flash_attention.py",
        "ch10/optimized_flash_attention.py",
    ):
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split(
            "def benchmark_fn",
            maxsplit=1,
        )[0]
        assert "with torch.inference_mode():" in setup_section
        assert "with torch.no_grad():" not in setup_section

    optimized_ch10_flash = (REPO_ROOT / "ch10" / "optimized_flash_attention.py").read_text(
        encoding="utf-8"
    )
    sdpa_probe = optimized_ch10_flash.split("def _try_sdpa_backend", maxsplit=1)[1].split(
        "def _resolve_sdpa_backends",
        maxsplit=1,
    )[0]
    external_probe = optimized_ch10_flash.split("def _resolve_external_flash", maxsplit=1)[1].split(
        "def _resolve_attention_runner",
        maxsplit=1,
    )[0]
    validate_section = optimized_ch10_flash.split("def validate_result", maxsplit=1)[1].split(
        "def get_benchmark",
        maxsplit=1,
    )[0]
    assert "with torch.inference_mode(), sdpa_backend_context(candidate):" in sdpa_probe
    assert "with torch.no_grad()" not in sdpa_probe
    assert "with torch.inference_mode():" in external_probe
    assert "with torch.inference_mode():" in validate_section
    assert "with torch.no_grad():" not in external_probe
    assert "with torch.no_grad():" not in validate_section

    for relative_path in (
        "ch10/baseline_flashattention3_pipeline.py",
        "ch10/optimized_flashattention3_pipeline.py",
    ):
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split(
            "def benchmark_fn",
            maxsplit=1,
        )[0]
        assert "with torch.inference_mode():" in setup_section
        assert "with torch.no_grad():" not in setup_section

    for filename in ("baseline_flashinfer_attention.py", "optimized_flashinfer_attention.py"):
        source = (REPO_ROOT / "labs" / "flashinfer_attention" / filename).read_text(
            encoding="utf-8"
        )
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]
        assert "with torch.inference_mode():" in benchmark_section
        assert "self.output = self.out_proj(proj_in)" in benchmark_section

    ch20_source = (REPO_ROOT / "ch20" / "ai_kernel_generator.py").read_text(
        encoding="utf-8"
    )
    ch20_reference = ch20_source.split("def _reference_attention", maxsplit=1)[1].split(
        "def _compiled_flex_attention",
        maxsplit=1,
    )[0]
    ch20_benchmark = ch20_source.split("def benchmark", maxsplit=1)[1].split(
        "def main",
        maxsplit=1,
    )[0]
    assert "torch.triu(" not in ch20_reference
    assert "torch.ones(q_len, kv_len" not in ch20_reference
    assert "mask = kv_pos > q_pos" in ch20_reference
    assert ch20_benchmark.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "start.elapsed_time(end) / count" in ch20_benchmark

    ch14_demo_source = (REPO_ROOT / "ch14" / "sliding_window_demo.py").read_text(
        encoding="utf-8"
    )
    assert "torch.triu(\n                torch.ones(seq_len, seq_len" not in ch14_demo_source
    assert "causal_mask = pos.unsqueeze(0) > pos.unsqueeze(1)" in ch14_demo_source


def test_ch16_misc_benchmark_helpers_use_inference_mode() -> None:
    quick_source = (REPO_ROOT / "ch16" / "gpt_quick_test.py").read_text(encoding="utf-8")
    fp8_test_source = (REPO_ROOT / "ch16" / "test_fp8_quantization_real.py").read_text(encoding="utf-8")
    te_source = (REPO_ROOT / "ch16" / "fp8_transformer_engine.py").read_text(encoding="utf-8")
    profiling_source = (REPO_ROOT / "ch16" / "inference_profiling.py").read_text(encoding="utf-8")
    quick_benchmark = quick_source.split("def benchmark_quick", maxsplit=1)[1].split(
        "def main",
        maxsplit=1,
    )[0]
    fp8_benchmark = fp8_test_source.split("def benchmark_model", maxsplit=1)[1].split(
        "def main",
        maxsplit=1,
    )[0]
    te_convert = te_source.split("def convert_linear_layers", maxsplit=1)[1].split(
        "def transformer_engine_warning",
        maxsplit=1,
    )[0]
    quantization_manager = profiling_source.split("class QuantizationManager", maxsplit=1)[1].split(
        "class InferenceProfiler",
        maxsplit=1,
    )[0]

    assert quick_benchmark.count("with torch.inference_mode():") == 2
    assert "with torch.no_grad():" not in quick_benchmark
    assert quick_benchmark.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "time.perf_counter()" not in quick_benchmark
    assert fp8_benchmark.count("with torch.inference_mode():") == 2
    assert "with torch.no_grad():" not in fp8_benchmark
    assert fp8_benchmark.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "time.time()" not in fp8_benchmark
    assert "with torch.inference_mode():" in te_convert
    assert "with torch.no_grad():" not in te_convert
    assert quantization_manager.count("with torch.inference_mode():") == 3
    assert "with torch.no_grad():" not in quantization_manager
    assert "module.weight.copy_(quantized_weights)" in quantization_manager
    assert "module.weight.data = quantized_weights" not in quantization_manager


def test_ch16_runtime_schedulers_cache_nvtx_and_verification_dummy() -> None:
    for filename, label in (
        ("baseline_runtime_scheduler.py", "runtime_scheduler_baseline"),
        ("optimized_runtime_scheduler.py", "runtime_scheduler_optimized"),
    ):
        source = (REPO_ROOT / "ch16" / filename).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split(
            "def _run_scenario",
            maxsplit=1,
        )[0]
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def get_custom_metrics",
            maxsplit=1,
        )[0]

        assert "self._enable_nvtx = False" in source
        assert "self._verification_dummy: Optional[torch.Tensor] = None" in source
        assert 'config = getattr(self, "_config", None) or self.get_config()' in setup_section
        assert "self._enable_nvtx = get_nvtx_enabled(config) if config else False" in setup_section
        assert "self._verification_dummy = torch.zeros(1, device=self.device)" in setup_section
        assert f'with nvtx_range("{label}", enable=self._enable_nvtx):' in benchmark_section
        assert "get_config()" not in benchmark_section
        assert "get_nvtx_enabled(" not in benchmark_section
        assert 'inputs={"dummy": self._verification_dummy}' in capture_section
        assert "torch.zeros(1, device=self.device)" not in capture_section


def test_ch16_piece_and_regional_graphs_cache_nvtx_outside_hot_loop() -> None:
    direct_nvtx_files = (
        "baseline_piece_graphs.py",
        "optimized_piece_graphs.py",
        "baseline_regional_compilation.py",
    )
    for filename in direct_nvtx_files:
        source = (REPO_ROOT / "ch16" / filename).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split(
            "def benchmark_fn", maxsplit=1
        )[0]
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]

        assert "self._enable_nvtx = get_nvtx_enabled(config) if config else False" in setup_section
        assert "get_config()" not in benchmark_section
        assert "get_nvtx_enabled(" not in benchmark_section
        assert "enable=self._enable_nvtx" in benchmark_section

    optimized_source = (REPO_ROOT / "ch16" / "optimized_regional_compilation.py").read_text(
        encoding="utf-8"
    )
    optimized_setup = optimized_source.split("def setup", maxsplit=1)[1].split(
        "def get_workload_metadata", maxsplit=1
    )[0]
    optimized_benchmark = optimized_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "self._enable_nvtx = get_nvtx_enabled(config) if config else False" in optimized_setup
    assert "get_config()" not in optimized_benchmark
    assert "get_nvtx_enabled(" not in optimized_benchmark
    assert "self._run_with_cuda_graph(seq_len, self._enable_nvtx)" in optimized_benchmark


def test_ch19_token_precision_confidence_batches_scalar_transfer() -> None:
    source = (REPO_ROOT / "ch19" / "token_precision_switching.py").read_text(
        encoding="utf-8"
    )
    confidence_section = source.split("def _confidence", maxsplit=1)[1].split(
        "def _choose_precision", maxsplit=1
    )[0]

    assert "metrics = torch.stack(" in confidence_section
    assert ").detach().cpu()" in confidence_section
    assert "float(probs.max())" not in confidence_section
    assert "float(-(probs * log_probs).sum())" not in confidence_section
    assert "float(top2[0] - top2[1])" not in confidence_section


def test_ch19_dynamic_precision_batches_confidence_metric_reads() -> None:
    common_source = (REPO_ROOT / "ch19" / "dynamic_precision_benchmark_common.py").read_text(
        encoding="utf-8"
    )
    switching_source = (REPO_ROOT / "ch19" / "dynamic_precision_switching.py").read_text(
        encoding="utf-8"
    )
    host_policy_section = common_source.split("def decode_host_policy_baseline", maxsplit=1)[
        1
    ].split(
        "def decode_dynamic_precision",
        maxsplit=1,
    )[0]
    decision_section = switching_source.split("def should_use_low_precision", maxsplit=1)[
        1
    ].split(
        "def quantize_kv_cache_on_memory_pressure",
        maxsplit=1,
    )[0]
    demo_entropy_section = switching_source.split("high_conf_logits[0, 42] = 10.0", maxsplit=1)[
        1
    ].split(
        "should_use_fp8_high = should_use_low_precision",
        maxsplit=1,
    )[0]

    assert "policy_metrics = torch.stack(" in host_policy_section
    assert "compute_entropy(host_logits).mean().item()" not in host_policy_section
    assert "values.mean().item()" not in host_policy_section
    assert "log_probs = torch.log_softmax(logits, dim=-1)" in decision_section
    assert "probs = log_probs.exp()" in decision_section
    assert "entropy_values = -(probs * log_probs).sum(dim=-1)" in decision_section
    assert "entropy, max_prob = torch.stack(" in decision_section
    assert "compute_entropy(logits).mean()" not in decision_section
    assert "compute_entropy(logits).mean().item()" not in decision_section
    assert "probs.max(dim=-1).values.mean().item()" not in decision_section
    assert "high_entropy, low_entropy = torch.stack(" in demo_entropy_section
    assert "compute_entropy(high_conf_logits).item()" not in demo_entropy_section
    assert "compute_entropy(low_conf_logits).item()" not in demo_entropy_section


def test_ch19_native_fp4_batches_accuracy_metric_reads() -> None:
    source = (REPO_ROOT / "ch19" / "native_fp4_quantization.py").read_text(
        encoding="utf-8"
    )
    accuracy_section = source.split("# Accuracy", maxsplit=1)[1].split(
        "print(\"\\n\" + \"=\" * 80)",
        maxsplit=1,
    )[0]

    assert "mean_err, fp16_abs_mean = torch.stack(" in accuracy_section
    assert "with torch.inference_mode():" in accuracy_section
    assert "with torch.no_grad():" not in accuracy_section
    assert "error.mean().item()" not in accuracy_section
    assert "out_fp16.abs().mean().item()" not in accuracy_section


def test_ch19_decode_loops_preallocate_token_buffers() -> None:
    files = [
        "ch19/dynamic_precision_benchmark_common.py",
        "ch19/dynamic_precision_switching.py",
        "ch19/token_precision_switching.py",
    ]

    for filename in files:
        source = (REPO_ROOT / filename).read_text(encoding="utf-8")
        assert "torch.inference_mode()" in source
        assert "torch.no_grad()" not in source
        assert "torch.empty(\n        (batch_size, prompt_len + max_steps)" in source or (
            "torch.empty(\n            (batch_size, prompt_len + max_length)" in source
        )
        assert "torch.cat([generated, next_token]" not in source
        assert "torch.cat([tokens, next_token]" not in source
        assert "torch.cat([tokens, next_token.unsqueeze(0)]" not in source
        assert "torch.empty_like(last_step_logits[:, :1])" not in source
        assert "tuple(next_token_values.shape)" not in source
        if "top2_values" in source:
            assert "if top2_shape_tuple is None:" in source
            assert "tuple(top2_values.shape)" not in source

    common_source = (REPO_ROOT / "ch19" / "dynamic_precision_benchmark_common.py").read_text(
        encoding="utf-8"
    )
    assert common_source.count("next_token_values = torch.empty(\n                (batch_size, 1),") == 2
    assert "next_token_values.device != last_step_logits.device" not in common_source

    token_precision_source = (REPO_ROOT / "ch19" / "token_precision_switching.py").read_text(
        encoding="utf-8"
    )
    token_precision_generate = token_precision_source.split("def generate(", maxsplit=1)[1].split(
        "#",
        maxsplit=1,
    )[0]
    assert "@torch.inference_mode()\n    def generate" in token_precision_source
    assert "self._next_token_buffer = None" in token_precision_source
    assert "self._next_token_host_buffer = None" in token_precision_source
    assert "def _next_token_buffers(self, device: torch.device)" in token_precision_source
    dynamic_decode_section = token_precision_source.split("def decode_with_dynamic_precision", maxsplit=1)[1].split(
        "# ===== END dynamic_precision_inference =====",
        maxsplit=1,
    )[0]
    assert "next_token = torch.empty((batch_size, 1), device=device, dtype=prompt.dtype)" in dynamic_decode_section
    assert "top2_values = torch.empty(" in dynamic_decode_section
    assert "top2_indices = torch.empty(" in dynamic_decode_section
    assert "torch.topk(last, k=2, dim=topk_dim, out=(top2_values, top2_indices))" in dynamic_decode_section
    assert "torch.max(last_step_logits, dim=-1, keepdim=True, out=(next_token_values, next_token))" in dynamic_decode_section
    assert "next_token = torch.argmax(last_step_logits" not in dynamic_decode_section
    assert "torch.multinomial(probs, num_samples=1, out=next_token)" in token_precision_generate
    assert "tokens[:, current_len : current_len + 1].copy_(next_token.view(1, 1))" in token_precision_generate
    assert "next_token_host.copy_(next_token)" in token_precision_generate
    assert "next_token = torch.multinomial(probs, num_samples=1)" not in token_precision_generate
    assert "next_token.item()" not in token_precision_generate


def test_ch19_fp4_baseline_keeps_scale_on_device() -> None:
    source = (REPO_ROOT / "ch19" / "baseline_fp4_weight_quantization.py").read_text(
        encoding="utf-8"
    )
    quantize_section = source.split("def quantize_fp4_baseline", maxsplit=1)[1].split(
        "def dequantize_fp4_baseline", maxsplit=1
    )[0]
    dequantize_section = source.split("def dequantize_fp4_baseline", maxsplit=1)[1].split(
        "class BaselineFP4Linear", maxsplit=1
    )[0]

    assert "scale.item()" not in quantize_section
    assert "torch.tensor([scale]" not in quantize_section
    assert "scale = (absmax / FP4_MAX).clamp(min=1e-8)" in quantize_section
    assert "scale_tensor = scale.reshape(1).to(dtype=dtype, device=device)" in quantize_section
    assert "scale.item()" not in dequantize_section
    assert "dequantized = values * scale.to(values.dtype)" in dequantize_section


def test_ch19_fp4_helpers_cache_lookup_values_per_device() -> None:
    for filename in (
        "baseline_fp4_weight_quantization.py",
        "optimized_fp4_weight_quantization.py",
    ):
        source = (REPO_ROOT / "ch19" / filename).read_text(encoding="utf-8")
        helpers_section = source.split("def _fp4_values_for", maxsplit=1)[1].split(
            "def quantize_fp4_", maxsplit=1
        )[0]
        quantize_section = source.split("def quantize_fp4_", maxsplit=1)[1].split(
            "def dequantize_fp4_", maxsplit=1
        )[0]
        dequantize_section = source.split("def dequantize_fp4_", maxsplit=1)[1].split(
            "class ", maxsplit=1
        )[0]

        assert "_FP4_VALUES_CACHE: dict[torch.device, torch.Tensor] = {}" in source
        assert "_FP4_SIGNED_VALUES_CACHE: dict[torch.device, torch.Tensor] = {}" in source
        assert "cached = _FP4_VALUES_CACHE.get(device)" in helpers_section
        assert "FP4_VALUES.to(device=device)" in helpers_section
        assert "cached = _FP4_SIGNED_VALUES_CACHE.get(device)" in helpers_section
        assert "FP4_SIGNED_VALUES.to(device=device)" in helpers_section
        assert "FP4_VALUES.to(device)" not in quantize_section
        assert "FP4_VALUES.to(device)" not in dequantize_section
        assert "fp4_vals = _fp4_values_for(device)" in quantize_section
        assert "signed_fp4_vals = _fp4_signed_values_for(device)" in dequantize_section
        assert "torch.where(signs.bool()" not in dequantize_section
        assert "signs = (unpacked >> 3)" not in dequantize_section


def test_ch19_fp4_weight_quantization_uses_inference_mode() -> None:
    baseline_source = (
        REPO_ROOT / "ch19" / "baseline_fp4_weight_quantization.py"
    ).read_text(encoding="utf-8")
    source = (REPO_ROOT / "ch19" / "optimized_fp4_weight_quantization.py").read_text(
        encoding="utf-8"
    )
    baseline_setup = baseline_source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    baseline_benchmark = baseline_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]
    baseline_validate = baseline_source.split("def validate_result", maxsplit=1)[1].split(
        "def get_benchmark",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]
    validate_section = source.split("def validate_result", maxsplit=1)[1].split(
        "def get_benchmark",
        maxsplit=1,
    )[0]

    assert "with torch.inference_mode():" in baseline_setup
    assert "with torch.no_grad():" not in baseline_setup
    assert "with torch.inference_mode():" in baseline_benchmark
    assert "with torch.no_grad():" not in baseline_benchmark
    assert "with torch.inference_mode():" in baseline_validate
    assert "with torch.no_grad():" not in baseline_validate
    assert "with torch.inference_mode():" in benchmark_section
    assert "with torch.no_grad():" not in benchmark_section
    assert "with torch.inference_mode():" in validate_section
    assert "with torch.no_grad():" not in validate_section


def test_ch19_optimized_fp4_fp8_bridge_reuses_activation_and_scale_buffers() -> None:
    source = (REPO_ROOT / "ch19" / "optimized_fp4_weight_quantization.py").read_text(
        encoding="utf-8"
    )
    forward_fp8_section = source.split("def _forward_fp8", maxsplit=1)[1].split(
        "@property",
        maxsplit=1,
    )[0]

    assert "self.register_buffer('_input_fp8_buffer'" in source
    assert "self.register_buffer('_fp8_scale_a'" in source
    assert "self.register_buffer('_fp8_scale_b'" in source
    assert "def _activation_fp8_buffer(self, x_2d: torch.Tensor)" in source
    assert "def _fp8_scale_buffers(self, device: torch.device)" in source
    assert "x_fp8 = self._activation_fp8_buffer(x_2d)" in forward_fp8_section
    assert "x_fp8.copy_(x_2d)" in forward_fp8_section
    assert "scale_a, scale_b = self._fp8_scale_buffers(x.device)" in forward_fp8_section
    assert ".to(torch.float8_e4m3fn)" not in forward_fp8_section
    assert "torch.ones(1, device=x.device, dtype=torch.float32)" not in forward_fp8_section


def test_ch19_native_fp4_caches_lookup_values_per_device() -> None:
    source = (REPO_ROOT / "ch19" / "native_fp4_quantization.py").read_text(
        encoding="utf-8"
    )
    helpers_section = source.split("def _fp4_values_for", maxsplit=1)[1].split(
        "# ============================================================================",
        maxsplit=1,
    )[0]
    quantize_section = source.split("def quantize_to_fp4_packed", maxsplit=1)[1].split(
        "def dequantize_from_fp4_packed",
        maxsplit=1,
    )[0]
    dequantize_section = source.split("def dequantize_from_fp4_packed", maxsplit=1)[1].split(
        "# ============================================================================",
        maxsplit=1,
    )[0]

    assert "_FP4_VALUES_CACHE: dict[torch.device, torch.Tensor] = {}" in source
    assert "_FP4_SIGNED_VALUES_CACHE: dict[torch.device, torch.Tensor] = {}" in source
    assert "cached = _FP4_VALUES_CACHE.get(device)" in helpers_section
    assert "FP4_VALUES.to(device=device)" in helpers_section
    assert "cached = _FP4_SIGNED_VALUES_CACHE.get(device)" in helpers_section
    assert "FP4_SIGNED_VALUES.to(device=device)" in helpers_section
    assert "FP4_VALUES.to(device)" not in quantize_section
    assert "FP4_VALUES.to(device)" not in dequantize_section
    assert "fp4_vals = _fp4_values_for(device)" in quantize_section
    assert "signed_fp4_vals = _fp4_signed_values_for(device)" in dequantize_section
    assert "torch.where(signs.bool()" not in dequantize_section
    assert "signs = (unpacked >> 3)" not in dequantize_section


def test_flashattention4_timing_reuses_events_and_cpu_statistics() -> None:
    source = (REPO_ROOT / "labs" / "flashattention4" / "flashattention4_common.py").read_text(
        encoding="utf-8"
    )
    microbench_source = (
        REPO_ROOT / "labs" / "flashattention4" / "tflops_microbench.py"
    ).read_text(encoding="utf-8")
    timing_section = source.split("def measure_flashattention4_latency", maxsplit=1)[1].split(
        "def count_nonmasked_attention_elements", maxsplit=1
    )[0]
    microbench_timing_section = microbench_source.split(
        "def _benchmark_cuda_callable", maxsplit=1
    )[1].split("def _build_benchmark_callable", maxsplit=1)[0]

    assert "import statistics" in source
    assert timing_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "for _ in range(iterations):\n        start = torch.cuda.Event" not in timing_section
    assert "end.synchronize()" in timing_section
    assert "sorted_times = sorted(times_ms)" in timing_section
    assert "torch.tensor(times_ms" not in timing_section
    assert "std_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0" in timing_section
    assert microbench_timing_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert (
        "for _ in range(iterations):\n        start = torch.cuda.Event"
        not in microbench_timing_section
    )
    assert "end.synchronize()" in microbench_timing_section


def test_timed_loops_reuse_cuda_events() -> None:
    files = [
        "labs/moe_decode_blackwell_matrix/runner.py",
        "labs/cutlass_profiler_kernel_selector/run_triton_matmul.py",
        "labs/memory_bandwidth_patterns/bandwidth_patterns_common.py",
    ]

    for filename in files:
        source = (REPO_ROOT / filename).read_text(encoding="utf-8")
        assert "for _ in range(scenario.repeats):\n        start_event = torch.cuda.Event" not in source
        assert "for _ in range(iters):\n        start = torch.cuda.Event" not in source
        if filename == "labs/moe_decode_blackwell_matrix/runner.py":
            assert "with torch.inference_mode():" in source
            assert "with torch.no_grad():" not in source
        if filename == "labs/memory_bandwidth_patterns/bandwidth_patterns_common.py":
            timing_section = source.split("def measure_cuda_callable", maxsplit=1)[1].split(
                "class BandwidthPatternsBenchmarkBase",
                maxsplit=1,
            )[0]
            assert timing_section.count("torch.cuda.Event(enable_timing=True)") == 2
            assert timing_section.count("start.record()") == 1
            assert timing_section.count("end.record()") == 1
            assert timing_section.count("end.synchronize()") == 1
            assert "samples.append(start.elapsed_time(end))" not in timing_section


def test_cuda_event_timing_waits_on_terminal_event_not_whole_device() -> None:
    event_sync_files = {
        "ch02/hardware_info.py": "end.synchronize()",
        "ch02/nvlink_c2c_bandwidth_benchmark.py": "end.synchronize()",
        "ch12/bias_relu_residual_fusion_benchmark.py": "end.synchronize()",
        "ch13/fp8_static_demo.py": "end.synchronize()",
        "ch13/optimized_fp8_static.py": "end.synchronize()",
        "ch13/fp8_perchannel_demo.py": "end.synchronize()",
        "ch14/optimized_flex_attention_sparse.py": "end.synchronize()",
        "ch14/triton_persistent_demo.py": "end.synchronize()",
        "ch14/sliding_window_demo.py": "end.synchronize()",
        "ch14/flex_attention_sparse_demo.py": "end.synchronize()",
        "ch14/training_large_model_1_5x.py": "end.synchronize()",
        "ch16/inference_optimizations_blackwell.py": "end.synchronize()",
        "ch16/gpt_quick_test.py": "end.synchronize()",
        "ch16/test_fp8_quantization_real.py": "end.synchronize()",
        "ch16/moe_performance_benchmark.py": "end_event.synchronize()",
        "ch16/synthetic_moe_inference_benchmark.py": "end_event.synchronize()",
        "ch18/flex_attention_native.py": "end.synchronize()",
        "ch18/flex_attention_enhanced.py": "end.synchronize()",
        "ch18/flex_attention_large_model.py": "end.synchronize()",
        "ch19/fp8_compiled_matmul.py": "end.synchronize()",
        "ch19/native_fp4_quantization.py": "end.synchronize()",
        "ch19/native_fp6_quantization.py": "end.synchronize()",
        "ch19/native_fp8_training.py": "end.synchronize()",
        "ch20/ai_kernel_generator.py": "end.synchronize()",
        "labs/flexattention/flex_attention_cute.py": "end.synchronize()",
        "labs/cutlass_profiler_kernel_selector/run_triton_matmul.py": "end.synchronize()",
        "labs/moe_decode_blackwell_matrix/runner.py": "end_event.synchronize()",
        "labs/moe_optimization_journey/triton_fused_moe.py": "end.synchronize()",
        "labs/nvfp4_dual_gemm/env_probe_b200.py": "end.synchronize()",
        "labs/nvfp4_dual_gemm/local_eval.py": "end.synchronize()",
        "labs/nvfp4_dual_gemm/official_semantics_eval.py": "end.synchronize()",
        "labs/nvfp4_gemm/local_eval_official597.py": "end_event.synchronize()",
        "labs/nvfp4_gemm/local_eval_submission.py": "end.synchronize()",
    }

    global_wait_after_event = re.compile(
        r"end(?:_event)?\.record\(\)\n\s*torch\.cuda\.synchronize\("
    )
    for filename, expected_sync in event_sync_files.items():
        source = (REPO_ROOT / filename).read_text(encoding="utf-8")
        assert expected_sync in source
        assert global_wait_after_event.search(source) is None


def test_occupancy_tuning_variants_match_their_filenames() -> None:
    wide_n = get_wide_n_benchmark()
    latency = get_latency_benchmark()
    schedule_source = (
        REPO_ROOT / "labs" / "occupancy_tuning" / "triton_matmul_schedules.py"
    ).read_text(encoding="utf-8")

    assert wide_n.schedule.name == "bm64_bn256_bk32"
    assert wide_n.schedule.block_m == 64
    assert wide_n.schedule.block_n == 256
    assert wide_n.schedule.block_k == 32

    assert latency.schedule.name == "bm64_bn64_bk32_nw2"
    assert latency.schedule.block_m == 64
    assert latency.schedule.block_n == 64
    assert latency.schedule.block_k == 32
    assert latency.schedule.num_warps == 2
    assert "with torch.inference_mode():\n            self._reference = torch.matmul(self._a, self._b)" in schedule_source
    assert "with torch.no_grad():\n            self._reference = torch.matmul(self._a, self._b)" not in schedule_source


def test_real_world_model_entrypoints_return_harness_benchmarks() -> None:
    assert isinstance(get_deepseek_benchmark(), BaseBenchmark)
    assert isinstance(get_gpt4_benchmark(), BaseBenchmark)


def test_base_benchmark_nvtx_range_caches_config_flag() -> None:
    source = (REPO_ROOT / "core" / "harness" / "benchmark_harness.py").read_text(
        encoding="utf-8"
    )
    base_section = source.split("class BaseBenchmark:", maxsplit=1)[1].split(
        "class BenchmarkHarness",
        maxsplit=1,
    )[0]
    init_section = base_section.split("def __init__", maxsplit=1)[1].split(
        "@property\n    def device",
        maxsplit=1,
    )[0]
    teardown_section = base_section.split("def teardown", maxsplit=1)[1].split(
        "def get_config",
        maxsplit=1,
    )[0]
    nvtx_section = base_section.split("def _nvtx_range(self, name: str):", maxsplit=1)[
        1
    ].split(
        "def _synchronize",
        maxsplit=1,
    )[0]

    assert "self._nvtx_config_cache: Optional[Any] = None" in init_section
    assert "self._nvtx_enabled_cache = False" in init_section
    assert "self._nvtx_config_cache = None" in teardown_section
    assert "self._nvtx_enabled_cache = False" in teardown_section
    assert "elif config is not self._nvtx_config_cache:" in nvtx_section
    assert "self._nvtx_enabled_cache = get_nvtx_enabled(config)" in nvtx_section
    assert "with nvtx_range(name, enable=self._nvtx_enabled_cache):" in nvtx_section
    assert "enable_nvtx = get_nvtx_enabled(config)" not in nvtx_section


def test_cleanup_process_group_escalates_when_group_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    signals: list[int] = []

    def _fake_killpg(_pgid: int, sig: int) -> None:
        if sig == 0:
            return
        signals.append(sig)

    monkeypatch.setattr("core.harness.benchmark_harness.os.killpg", _fake_killpg)

    _cleanup_process_group(4242, grace_seconds=0.0)

    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_cleanup_process_group_ignores_missing_group(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing_killpg(_pgid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr("core.harness.benchmark_harness.os.killpg", _missing_killpg)

    _cleanup_process_group(4242)


def test_flexattention_metrics_use_attention_formula_and_hot_paths_skip_clone() -> None:
    baseline = BaselineFlexAttentionBenchmark()
    optimized = OptimizedFlexAttentionBenchmark()

    docs = baseline.seq_len // baseline.doc_span
    active_pairs = float(baseline.batch * baseline.heads * docs * baseline.doc_span * baseline.doc_span)
    expected_flops = float(4 * active_pairs * baseline.head_dim)
    assert baseline.get_custom_metrics()["flex_attention.total_flops"] == expected_flops
    assert optimized.get_custom_metrics()["flex_attention.total_flops"] == expected_flops

    flex_baseline_source = (REPO_ROOT / "labs" / "flexattention" / "baseline_flex_attention.py").read_text(
        encoding="utf-8"
    )
    flex_optimized_source = (REPO_ROOT / "labs" / "flexattention" / "optimized_flex_attention.py").read_text(
        encoding="utf-8"
    )
    flash4_source = (REPO_ROOT / "labs" / "flashattention4" / "flashattention4_benchmarks.py").read_text(
        encoding="utf-8"
    )
    cute_source = (REPO_ROOT / "labs" / "flexattention" / "flex_attention_cute.py").read_text(
        encoding="utf-8"
    )

    assert "self.output = output_tensor.detach().float().clone()" not in flex_baseline_source
    assert "self.output = output_tensor.detach().float().clone()" not in flex_optimized_source
    assert "self.output = result.detach().float().clone()" not in flash4_source
    assert "self._sparsity_ratio = float(self.inputs.dense_mask.float().mean())" in flash4_source
    assert "self.inputs.dense_mask.float().mean().item()" not in flash4_source
    assert cute_source.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "time.perf_counter()" not in cute_source
    assert 'signature_overrides={"sparsity_ratio": self._sparsity_ratio}' in flash4_source


def test_ch10_flash_attention_requires_real_flashattention_on_sm100() -> None:
    source = (REPO_ROOT / "ch10" / "optimized_flash_attention.py").read_text(encoding="utf-8")
    resolve_section = source.split("def _resolve_attention_runner", maxsplit=1)[1].split(
        "def _run_attention", maxsplit=1
    )[0]
    assert "candidates.append([SDPBackend.FLASH_ATTENTION])" in source
    assert "candidates.append([SDPBackend.EFFICIENT_ATTENTION])" in source
    assert "if major >= 10" in resolve_section
    assert "FAIL FAST: FlashAttention required for ch10" in resolve_section
    assert "self._selected_backend_name = candidate[0].name.lower()" in source


def test_ch10_flash_attention_prefers_external_flash_engines_before_sdpa_fallback() -> None:
    source = (REPO_ROOT / "ch10" / "optimized_flash_attention.py").read_text(encoding="utf-8")

    flash3_idx = source.index("flash_attn_3.flash_attn_interface")
    flash2_idx = source.index("flash_attn.flash_attn_interface")

    assert flash3_idx < flash2_idx
    assert "self._selected_engine_name = \"sdpa\"" in source


def test_ch18_paged_attention_uses_real_block_table_sparse_kernel() -> None:
    common_source = (REPO_ROOT / "ch18" / "paged_attn_split_common.py").read_text(encoding="utf-8")
    optimized_source = (REPO_ROOT / "ch18" / "optimized_paged_attn_layout.py").read_text(encoding="utf-8")

    assert "self.block_table" in common_source
    assert "torch.roll(block_ids" not in common_source
    assert "batch_offsets = torch.arange(self.batch_size, device=self.device, dtype=torch.int64).unsqueeze(1)" in common_source
    assert "return (block_ids.unsqueeze(0) - batch_offsets).remainder_(num_blocks)" in common_source
    assert "create_block_mask, flex_attention" in common_source
    assert "dense_mask[:, 0][allowed] = 0.0" in common_source
    assert "return create_block_mask(" in common_source
    assert 'torch.compile(flex_attention, mode="max-autotune")' in common_source
    assert "return self._flex_attention_fn(self.q, self.k_dense, self.v_dense, block_mask=self.block_mask)" in common_source
    assert "_gather_paged_kv" not in common_source
    assert "LayoutPagedAttnBase" in optimized_source


def test_ch04_nvshmem_symmetric_broadcast_overlap_defines_done_event() -> None:
    source = (REPO_ROOT / "ch04" / "nvshmem_vs_nccl_benchmark.py").read_text(encoding="utf-8")
    symmetric_section = source.split("def _measure_symmetric_broadcast", maxsplit=1)[1].split(
        "def sweep_sizes", maxsplit=1
    )[0]

    assert "done = torch.cuda.Event() if overlap_compute and comm_stream is not None else None" in symmetric_section
    assert "if overlap_compute and comm_stream is not None and done is not None:" in symmetric_section


def test_ch19_vectorization_memory_preconverts_fp16_outside_hot_loop() -> None:
    baseline_source = (REPO_ROOT / "ch19" / "baseline_vectorization_memory.py").read_text(
        encoding="utf-8"
    )
    source = (REPO_ROOT / "ch19" / "optimized_vectorization_memory.py").read_text(encoding="utf-8")
    baseline_setup = baseline_source.split("def benchmark_fn", maxsplit=1)[0]
    baseline_benchmark = baseline_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    setup_section = source.split("def benchmark_fn", maxsplit=1)[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "_cached_a_fp16" not in source
    assert "_cached_b_fp16" not in source
    assert "self._tensor_a_fp16 = self.tensor_a.to(self._compute_dtype)" in setup_section
    assert "self._tensor_b_fp16 = self.tensor_b.to(self._compute_dtype)" in setup_section
    assert "torch.add(self._tensor_a_fp16, self._tensor_b_fp16, out=self._work)" in benchmark_section
    assert ".to(self._compute_dtype)" not in benchmark_section
    for setup, benchmark in ((baseline_setup, baseline_benchmark), (setup_section, benchmark_section)):
        assert "self._enable_nvtx = get_nvtx_enabled(config) if config else False" in setup
        assert "get_config()" not in benchmark
        assert "get_nvtx_enabled(" not in benchmark
        assert "enable=self._enable_nvtx" in benchmark


def test_moe_cuda_decode_attention_preconverts_bf16_outside_hot_loop() -> None:
    baseline_source = (REPO_ROOT / "labs" / "moe_cuda" / "baseline_decode_attention.py").read_text(
        encoding="utf-8"
    )
    source = (REPO_ROOT / "labs" / "moe_cuda" / "optimized_decode_attention.py").read_text(encoding="utf-8")
    baseline_setup = baseline_source.split("def benchmark_fn", maxsplit=1)[0]
    baseline_benchmark = baseline_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def finalize_iteration_metrics", maxsplit=1
    )[0]
    setup_section = source.split("def benchmark_fn", maxsplit=1)[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def finalize_iteration_metrics", maxsplit=1
    )[0]

    assert "def _cached_bf16" not in source
    assert "self._refresh_bf16_cache(force=True)" in setup_section
    assert "self._q_bf16 = self.q.to(torch.bfloat16)" not in benchmark_section
    assert "self._refresh_bf16_cache()" in source
    assert "q = self._q_bf16" in benchmark_section
    assert "k = self._k_bf16" in benchmark_section
    assert "v = self._v_bf16" in benchmark_section
    for setup, benchmark in ((baseline_setup, baseline_benchmark), (setup_section, benchmark_section)):
        assert "self._enable_nvtx = get_nvtx_enabled(config) if config else False" in setup
        assert "get_config()" not in benchmark
        assert "get_nvtx_enabled(" not in benchmark
        assert "enable=self._enable_nvtx" in benchmark


def test_moe_cuda_decode_kernel_wrappers_cache_nvtx_outside_hot_loop() -> None:
    for name in ("baseline_decode_kernel.py", "optimized_decode_kernel.py"):
        source = (REPO_ROOT / "labs" / "moe_cuda" / name).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split("def benchmark_fn", maxsplit=1)[0]
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]

        assert "self._enable_nvtx = get_nvtx_enabled(config) if config else False" in setup_section
        assert "get_config()" not in benchmark_section
        assert "get_nvtx_enabled(" not in benchmark_section
        assert "enable=self._enable_nvtx" in benchmark_section


def test_moe_cuda_kv_transfer_defers_verification_tensors_outside_hot_loop() -> None:
    for name in (
        "baseline_kv_transfer.py",
        "optimized_kv_transfer.py",
        "optimized_kv_transfer_graphs.py",
    ):
        source = (REPO_ROOT / "labs" / "moe_cuda" / name).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split(
            "def _launch_compute" if name == "optimized_kv_transfer.py" else "def benchmark_fn",
            maxsplit=1,
        )[0]
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1]

        assert "torch.tensor(" not in benchmark_section
        assert ".clone()" not in benchmark_section
        assert ".float()" not in benchmark_section
        assert "output=self.output.detach().float().clone()" in capture_section
        assert "self.workspace = torch.empty_like(self.input_chunks)" in setup_section
        assert "self.kv_dest = torch.empty_like(self.input_chunks)" in setup_section
        assert "torch.zeros_like(self.input_chunks)" not in setup_section
        assert "self._enable_nvtx = get_nvtx_enabled(config) if config else False" in setup_section
        assert "get_config()" not in benchmark_section
        assert "get_nvtx_enabled(" not in benchmark_section
        assert "enable=self._enable_nvtx" in benchmark_section


def test_moe_cuda_grouped_router_reuses_static_dispatch_buffers() -> None:
    source = (
        REPO_ROOT / "labs" / "moe_cuda" / "optimized_router_vectorized.py"
    ).read_text(encoding="utf-8")
    forward_section = source.split("def forward(self, tokens: torch.Tensor)", maxsplit=2)[2].split(
        "class VectorizedRouterBenchmark", maxsplit=1
    )[0]
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def _check_capacity_overflow", maxsplit=1
    )[0]

    assert "configure_static_dispatch_buffers" in source
    assert "def _flat_token_indices" in source
    assert "repeat_interleave(self.top_k)" not in source
    assert 'token_indices.div_(top_k, rounding_mode="floor")' in source
    assert "model.configure_static_dispatch_buffers(self.batch_size, self.inputs.device)" in setup_section
    assert "token_indices = self._token_indices_for(tokens, batch)" in forward_section
    assert "flat_tokens = tokens.index_select(0, token_indices)" in forward_section
    assert "expert_range = self._expert_range_for(tokens)" in forward_section
    assert "self._overflow_slots_for(slots, num_slots)" in forward_section
    assert "self._dense_input_for(flat_tokens, num_slots)" in forward_section
    assert "self._output_buffer_for(tokens, batch)" in forward_section
    assert "@torch.no_grad()\n    def configure_static_dispatch_buffers" in source
    assert "@torch.inference_mode()\n    def configure_static_dispatch_buffers" not in source
    assert "with torch.inference_mode():" in setup_section
    assert "with torch.no_grad():" not in setup_section
    assert "capture_no_grad" not in source
    assert "torch.zeros_like(tokens" not in source
    assert "output.zero_()" not in forward_section
    assert "output.index_add_(0, token_indices, weighted)" not in source
    assert 'output.scatter_reduce_(0, combine_index, weighted, reduce="sum", include_self=False)' in source
    assert "model.assume_static_no_overflow = True" in setup_section
    assert ".item()) == num_slots" not in source

    from labs.moe_cuda.optimized_router_vectorized import GroupedTopKMoE, _flat_token_indices

    torch.testing.assert_close(
        _flat_token_indices(3, 1, torch.device("cpu")),
        torch.tensor([0, 1, 2], dtype=torch.int64),
    )
    torch.testing.assert_close(
        _flat_token_indices(3, 2, torch.device("cpu")),
        torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),
    )
    torch.testing.assert_close(
        _flat_token_indices(2, 3, torch.device("cpu")),
        torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64),
    )

    model = GroupedTopKMoE(hidden_size=8, num_experts=4, top_k=2, expansion=1)
    model.capacity = 64
    model.configure_static_dispatch_buffers(batch_size=3, device=torch.device("cpu"))
    token_ptr = model._static_token_indices.data_ptr()
    expert_ptr = model._static_expert_range.data_ptr()
    overflow_ptr = model._static_overflow_slots.data_ptr()
    dense_ptr = model._static_dense_input.data_ptr()
    output_ptr = model._static_output_buffer.data_ptr()

    x = torch.randn(3, 8)
    expected = model(x).clone()
    model._static_output_buffer.fill_(float("nan"))
    model.assume_static_no_overflow = True
    output = model(x)

    assert output.shape == (3, 8)
    assert not torch.isnan(output).any()
    torch.testing.assert_close(output, expected)
    assert model._static_token_indices.data_ptr() == token_ptr
    assert model._static_expert_range.data_ptr() == expert_ptr
    assert model._static_overflow_slots.data_ptr() == overflow_ptr
    assert model._static_dense_input.data_ptr() == dense_ptr
    assert model._static_output_buffer.data_ptr() == output_ptr
    torch.testing.assert_close(
        model._static_token_indices,
        torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),
    )
    assert model._static_expert_range.shape == (4, 1)
    assert model._static_overflow_slots.unique().item() == 4 * 64
    assert model._static_dense_input.shape == (4 * 64 + 1, 8)
    assert model._static_output_buffer.shape == (3, 8)


def test_moe_cuda_topk_router_configures_static_dispatch_once() -> None:
    source = (REPO_ROOT / "labs" / "moe_cuda" / "optimized_router.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split("def benchmark_fn", maxsplit=1)[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "model.calibrate_capacity(self.inputs)" in setup_section
    assert "model.configure_static_dispatch_buffers(self.batch_size, self.inputs.device)" in setup_section
    assert "model.assume_static_no_overflow = True" in setup_section
    assert "with torch.inference_mode():" in setup_section
    assert "with torch.no_grad():" not in setup_section
    assert "calibrate_capacity" not in benchmark_section
    assert "configure_static_dispatch_buffers" not in benchmark_section


def test_moe_cuda_router_wrappers_cache_nvtx_and_parameter_count() -> None:
    for name in (
        "baseline_router.py",
        "optimized_router.py",
        "optimized_router_vectorized.py",
    ):
        source = (REPO_ROOT / "labs" / "moe_cuda" / name).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split(
            "def _check_capacity_overflow" if name == "optimized_router_vectorized.py" else "def benchmark_fn",
            maxsplit=1,
        )[0]
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def teardown", maxsplit=1
        )[0]

        assert "self._enable_nvtx = get_nvtx_enabled(config) if config else False" in setup_section
        assert "self._payload_parameter_count = sum(p.numel() for p in model.parameters())" in setup_section
        assert "get_config()" not in benchmark_section
        assert "get_nvtx_enabled(" not in benchmark_section
        assert "enable=self._enable_nvtx" in benchmark_section
        assert "parameter_count=self._payload_parameter_count" in capture_section
        assert "sum(p.numel()" not in capture_section


def test_ch20_bf16_mlp_preconverts_activation_dtype_outside_hot_loop() -> None:
    source = (REPO_ROOT / "ch20" / "optimized_bf16_mlp.py").read_text(encoding="utf-8")
    setup_section = source.split("def benchmark_fn", maxsplit=1)[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "self._model_dtype =" in setup_section
    assert "self._x_model_dtype = self.x.to(dtype=self._model_dtype)" in setup_section
    assert "next(self.model.parameters()).dtype" not in benchmark_section
    assert ".to(dtype=" not in benchmark_section
    assert "self.output = self.model(self._x_model_dtype)" in benchmark_section


def test_ch20_optimized_forward_paths_use_inference_mode() -> None:
    for relative in (
        "ch20/optimized_autotuning.py",
        "ch20/optimized_bf16_mlp.py",
        "ch20/optimized_end_to_end_bandwidth.py",
        "ch20/optimized_moe.py",
        "ch20/optimized_pipeline_sequential.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]

        assert "with torch.inference_mode():" in benchmark_section
        assert "with torch.no_grad():" not in benchmark_section


def test_ch20_baseline_inference_paths_use_inference_mode() -> None:
    for relative in (
        "ch20/baseline_autotuning.py",
        "ch20/baseline_bf16_mlp.py",
        "ch20/baseline_moe.py",
        "ch20/baseline_pipeline_sequential.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]

        assert "with torch.inference_mode():" in source
        assert "with torch.inference_mode():" in benchmark_section
        assert "with torch.no_grad():" not in source


def test_ch20_bf16_mlp_uses_inplace_relu_activations() -> None:
    for relative in (
        "ch20/baseline_bf16_mlp.py",
        "ch20/optimized_bf16_mlp.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        forward_section = source.split("def forward(self, x)", maxsplit=1)[1].split(
            "class ",
            maxsplit=1,
        )[0]

        assert "torch.relu_(x)" in forward_section
        assert "torch.relu(x)" not in forward_section


def test_ch20_end_to_end_bandwidth_uses_inplace_activation() -> None:
    for relative in (
        "ch20/baseline_end_to_end_bandwidth.py",
        "ch20/optimized_end_to_end_bandwidth.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]

        assert "self.relu = nn.ReLU(inplace=True)" in source
        assert "with torch.inference_mode():" in benchmark_section
        assert "with torch.no_grad():" not in benchmark_section


def test_ch20_training_and_moe_use_inplace_relu_modules() -> None:
    for relative in (
        "ch20/baseline_training_single.py",
        "ch20/optimized_training_single.py",
        "ch20/baseline_moe.py",
        "ch20/optimized_moe.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")

        assert "nn.ReLU(inplace=True)" in source
        assert "nn.ReLU()" not in source


def test_ch20_optimized_memory_standard_uses_scalar_addcmul_constants() -> None:
    source = (REPO_ROOT / "ch20" / "optimized_memory_standard.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split("def benchmark_fn", maxsplit=1)[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "self.result = torch.empty_like(self.data).contiguous()" in setup_section
    assert "self.offset = torch.tensor(1.1" in setup_section
    assert "self.scale_tensor = torch.tensor(2.0" in setup_section
    assert "torch.full_like(self.data" not in setup_section
    assert "torch.zeros_like(self.data)" not in setup_section
    assert "torch.addcmul(self.offset, self.data, self.scale_tensor, out=self.result)" in benchmark_section

    data = torch.randn(8)
    output = torch.empty_like(data)
    torch.addcmul(torch.tensor(1.1), data, torch.tensor(2.0), out=output)
    torch.testing.assert_close(output, 1.1 + data * 2.0)


def test_ch20_integrated_kv_cache_releases_slabs_without_zero_fill() -> None:
    source = (REPO_ROOT / "ch20" / "optimized_integrated_kv_cache.py").read_text(encoding="utf-8")
    acquire_section = source.split("def _acquire_buffer", maxsplit=1)[1].split(
        "def _release_buffer", maxsplit=1
    )[0]
    release_section = source.split("def _release_buffer", maxsplit=1)[1].split(
        "def allocate", maxsplit=1
    )[0]

    assert "torch.empty(length" in acquire_section
    assert "torch.empty_like(k_buf)" in acquire_section
    assert "torch.zeros(" not in acquire_section
    assert ".zero_()" not in release_section
    assert "self._empty" in source

    from ch20.optimized_integrated_kv_cache import PagedKVCache

    cache = PagedKVCache(
        page_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    missing_k, missing_v = cache.get("missing", 0, 0, 4)
    assert missing_k.shape == (0, 1, 2)
    assert missing_v.shape == (0, 1, 2)

    old_k = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    old_v = old_k + 10.0
    cache.append_block("old", 0, old_k, old_v, 0)
    old_buffer = cache.allocations["old"][0]["buffer"]
    old_ptr = old_buffer[0].data_ptr()
    cache.free("old")

    cache.allocate("new", 2)
    new_buffer = cache.allocations["new"][0]["buffer"]
    assert new_buffer[0].data_ptr() == old_ptr
    empty_k, empty_v = cache.get("new", 0, 0, 4)
    assert empty_k.shape == (0, 1, 2)
    assert empty_v.shape == (0, 1, 2)

    new_k = torch.tensor([[[7.0, 8.0]]])
    new_v = new_k + 20.0
    cache.append_block("new", 0, new_k, new_v, 0)
    actual_k, actual_v = cache.get("new", 0, 0, 4)
    torch.testing.assert_close(actual_k, new_k)
    torch.testing.assert_close(actual_v, new_v)


def test_ch20_pipeline_sequential_reuses_setup_artifacts_outside_hot_loop() -> None:
    baseline_source = (REPO_ROOT / "ch20" / "baseline_pipeline_sequential.py").read_text(encoding="utf-8")
    optimized_source = (REPO_ROOT / "ch20" / "optimized_pipeline_sequential.py").read_text(encoding="utf-8")
    optimized_setup = optimized_source.split("def _run_pipelined_once", maxsplit=1)[0]

    for source in (baseline_source,):
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def teardown", maxsplit=1
        )[0]

        assert "self.microbatches = [chunk.contiguous() for chunk in self.inputs.chunk" in source
        assert ".chunk(" not in benchmark_section
        assert "torch.cat(" not in benchmark_section
        assert "self._last_outputs = outputs" in benchmark_section
        assert "self.output = torch.cat([out.detach() for out in self._last_outputs], dim=0)" in capture_section

    optimized_benchmark = optimized_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    optimized_capture = optimized_source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]
    optimized_run = optimized_source.split("def _run_pipelined_once", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]

    assert "self.stage_events = [" in optimized_setup
    assert "torch.cuda.Event(enable_timing=False)" in optimized_setup
    assert "self._stage_outputs = [" in optimized_setup
    assert "self._last_outputs = [" in optimized_setup
    assert "stage_outputs: list[list[Optional[torch.Tensor]]] = [" not in optimized_run
    assert "return [output for output in final_outputs if output is not None]" not in optimized_run
    assert "with torch.inference_mode():" in optimized_benchmark
    assert "with torch.no_grad():" not in optimized_benchmark
    assert "torch.cat(" not in optimized_benchmark
    assert "self._run_pipelined_once()" in optimized_benchmark
    assert "self._last_outputs = outputs" not in optimized_benchmark
    assert "torch.cat([out.detach() for out in self._last_outputs], dim=0)" not in optimized_capture
    assert "self.output = torch.cat(self._last_outputs, dim=0).detach()" in optimized_capture
    assert "torch.cuda.Event(" not in optimized_benchmark


def test_ch17_ch20_defer_verification_materialization_outside_hot_loop() -> None:
    ch17_source = (REPO_ROOT / "ch17" / "optimized_memory.py").read_text(encoding="utf-8")
    ch17_setup = ch17_source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    ch17_benchmark = ch17_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    ch17_capture = ch17_source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]

    assert "with torch.inference_mode(), torch.cuda.graph(self.graph):" in ch17_setup
    assert "with torch.inference_mode():" in ch17_benchmark
    assert "self.output = self.graph_output.clone()" not in ch17_benchmark
    assert "self.output = self.graph_output" in ch17_benchmark
    assert ".floor_()" not in ch17_benchmark
    assert "output=self.output.detach().clone()" in ch17_capture
    probe = torch.empty(128, dtype=torch.float32)
    probe.random_(0, 256)
    torch.testing.assert_close(probe, probe.floor())

    ch20_source = (REPO_ROOT / "ch20" / "baseline_end_to_end_bandwidth.py").read_text(encoding="utf-8")
    ch20_benchmark = ch20_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    ch20_capture = ch20_source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]

    assert "torch.stack(" not in ch20_benchmark
    assert "self.outputs = [None for _ in range(self.num_batches)]" in ch20_source
    assert "self.outputs[batch_idx] = out" in ch20_benchmark
    assert "self.output = torch.stack([out.detach() for out in outputs], dim=0)" in ch20_capture


def test_ch20_kernel_verifiers_defer_contiguous_payload_slice_outside_hot_loop() -> None:
    for relative in (
        "ch20/kernel_verification_tool.py",
        "ch20/proofwright_verify_tool.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def teardown", maxsplit=1
        )[0]

        assert "[:32, :32].contiguous()" not in benchmark_section
        assert "self.output = self.test_kernel(self._verify_input)[:32, :32]" in benchmark_section
        assert "output=self.output.contiguous()" in capture_section


def test_ch18_metric_wrappers_defer_output_tensors_outside_hot_loop() -> None:
    for relative in (
        "ch18/baseline_cudagraph_bucketing.py",
        "ch18/optimized_cudagraph_bucketing.py",
        "ch18/baseline_vllm_decode_graphs.py",
        "ch18/optimized_vllm_decode_graphs.py",
        "ch18/scheduling_vllm_sglang.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def teardown", maxsplit=1
        )[0]

        assert "torch.tensor(" not in benchmark_section
        assert "self._output_values =" in benchmark_section
        assert "self.output = torch.tensor(self._output_values, dtype=torch.float32)" in capture_section


def test_ch18_cudagraph_bucketing_static_inputs_avoid_zero_fill() -> None:
    for relative in (
        "ch18/optimized_cudagraph_bucketing.py",
        "ch18/cudagraph_bucketing_simulator.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        capture_section = source.split("def _capture_graph", maxsplit=1)[1].split(
            "def _find_bucket",
            maxsplit=1,
        )[0]

        assert "self.static_inputs[key] = torch.empty(" in capture_section
        assert "self.static_inputs[key] = torch.zeros(" not in capture_section


def test_ch18_dynamic_flex_attention_mask_avoids_scalar_tensor_allocation() -> None:
    source = (REPO_ROOT / "ch18" / "flex_attention_enhanced.py").read_text(encoding="utf-8")
    benchmark_section = source.split("def benchmark_attention", maxsplit=1)[1].split(
        "def detect_architecture",
        maxsplit=1,
    )[0]
    dynamic_section = source.split("class DynamicSlidingWindowAttention", maxsplit=1)[1].split(
        "def compile_module",
        maxsplit=1,
    )[0]
    large_source = (REPO_ROOT / "ch18" / "flex_attention_large_model.py").read_text(encoding="utf-8")
    large_benchmark_section = large_source.split("def benchmark_model", maxsplit=1)[1].split(
        "def test_configuration",
        maxsplit=1,
    )[0]
    large_flex_section = large_source.split("class FlexAttentionModel", maxsplit=1)[1].split(
        "def estimate_memory",
        maxsplit=1,
    )[0]
    native_source = (REPO_ROOT / "ch18" / "flex_attention_native.py").read_text(encoding="utf-8")
    native_benchmark_section = native_source.split("def benchmark_attention", maxsplit=1)[1].split(
        "def main",
        maxsplit=1,
    )[0]

    assert source.count("create_block_mask(self.mask_fn, B, H, T, T, device=Q.device)") == 3
    assert native_source.count("device=Q.device") == 2
    for timing_section in (benchmark_section, large_benchmark_section, native_benchmark_section):
        assert timing_section.count("torch.cuda.Event(enable_timing=True)") == 2
        assert "time.perf_counter()" not in timing_section
    assert "window_sizes = self.window_sizes_tensor" in dynamic_section
    assert "if window_sizes.device != q_idx.device:" in dynamic_section
    assert "window_size = window_sizes[int(h)]" in dynamic_section
    assert "torch.tensor(h" not in dynamic_section
    assert "device=x.device" in large_flex_section
    assert ".to(x.device)" not in large_flex_section


def test_ch18_flexattention_fallback_builds_sliding_window_mask_vectorized() -> None:
    source = (REPO_ROOT / "ch18" / "flexattention_sliding_window.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def run",
        maxsplit=1,
    )[0]
    fallback_section = source.split("if not FLEX_ATTENTION_AVAILABLE:", maxsplit=1)[1].split(
        "scores.masked_fill_",
        maxsplit=1,
    )[0]

    assert "torch.ones(" not in fallback_section
    assert "for i in range(self.seq_length)" not in fallback_section
    assert "self._fallback_positions = torch.arange(self.seq_length, device=self.device)" in setup_section
    assert "pos = self._fallback_positions" in fallback_section
    assert "torch.arange(self.seq_length, device=self.device)" not in fallback_section
    assert "q_pos = pos.unsqueeze(1)" in fallback_section
    assert "kv_pos = pos.unsqueeze(0)" in fallback_section
    assert "mask = (kv_pos < q_pos - half_window) | (kv_pos > q_pos + half_window)" in fallback_section


def test_ch18_flexattention_demos_use_cuda_event_timing() -> None:
    flexdecoding = (REPO_ROOT / "ch18" / "flexdecoding.py").read_text(encoding="utf-8")
    flexdecoding_benchmark = flexdecoding.split("def _benchmark", maxsplit=1)[1].split(
        "def _score_mod_causal",
        maxsplit=1,
    )[0]
    assert flexdecoding_benchmark.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "start.elapsed_time(end) / iters" in flexdecoding_benchmark

    for filename in (
        "flexattention_block_sparse.py",
        "flexattention_document_attention.py",
        "flexattention_sliding_window.py",
    ):
        source = (REPO_ROOT / "ch18" / filename).read_text(encoding="utf-8")
        helper_section = source.split("def _time_region_ms", maxsplit=1)[1].split(
            "# Check for FlexAttention" if filename != "flexattention_document_attention.py" else "try:",
            maxsplit=1,
        )[0]
        run_section = source.split("def run(self) -> float:", maxsplit=1)[1].split(
            "def cleanup",
            maxsplit=1,
        )[0]

        assert helper_section.count("torch.cuda.Event(enable_timing=True)") == 2
        assert "start.record()" in helper_section
        assert "end.record()" in helper_section
        assert "start.elapsed_time(end)" in helper_section
        assert "_time_region_ms(" in run_section
        assert "torch.cuda.synchronize()" not in run_section
        assert "time.perf_counter()" not in run_section


def test_ch18_optimized_vllm_decode_workspace_drops_unused_mask_buffer() -> None:
    source = (REPO_ROOT / "ch18" / "optimized_vllm_decode_graphs.py").read_text(encoding="utf-8")
    workspace_section = source.split("class BucketWorkspace", maxsplit=1)[1].split(
        "@dataclass\nclass KVBlock", maxsplit=1
    )[0]

    assert "def pad_to_bucket" not in source
    assert "mask:" not in workspace_section
    assert "torch.ones(self.batch, dtype=torch.bool" not in workspace_section
    assert "self.mask.numel()" not in workspace_section
    assert "self._seq_lens_profiles: Dict[Tuple[int, int], torch.Tensor] = {}" in source
    assert "def seq_lens_profile(self, batch_size: int, bucket: int) -> torch.Tensor:" in source
    assert "seq_lens[:bucket].copy_(self.seq_lens_profile(batch_size, bucket))" in source
    run_section = source.split("def run(self) -> DecodeMetrics:", maxsplit=1)[1].split(
        "def parse_args", maxsplit=1
    )[0]
    assert "seq_lens[:batch_size].fill_" not in run_section
    assert "seq_lens[batch_size:bucket].zero_()" not in run_section


def test_ch18_speculative_decoder_batches_match_control_reads() -> None:
    source = (REPO_ROOT / "ch18" / "run_vllm_decoder.py").read_text(encoding="utf-8")
    decoder_section = source.split("class SpeculativeDecoder", maxsplit=1)[1].split(
        "class VLLMMoEInferenceBenchmark",
        maxsplit=1,
    )[0]
    decode_section = source.split("def decode(", maxsplit=1)[1].split(
        "def _maybe_adjust_chunk",
        maxsplit=1,
    )[0]

    assert "self._match_summary_workspace: Optional[torch.Tensor] = None" in decoder_section
    assert "self._all_matches_workspace: Optional[torch.Tensor] = None" in decoder_section
    assert "self._draft_next_values: Optional[torch.Tensor] = None" in decoder_section
    assert "self._target_next_values: Optional[torch.Tensor] = None" in decoder_section
    assert "self._matches_workspace: Optional[torch.Tensor] = None" in decoder_section
    assert "self._selected_tokens: Optional[torch.Tensor] = None" in decoder_section
    assert "def _match_workspaces(self, device: torch.device)" in decoder_section
    assert "def prepare_workspaces(self, batch_size: int, dtype: torch.dtype, device: torch.device)" in decoder_section
    assert "torch.max(last_logits, dim=-1, keepdim=True, out=(values, token_ids))" in decoder_section
    assert "torch.eq(candidate, target_next, out=matches)" in decode_section
    assert "torch.where(matches, candidate, target_next, out=tokens)" in decode_section
    assert "with torch.inference_mode():" in decode_section
    assert "with torch.no_grad():" not in decode_section
    assert "torch.sum(matches, dim=None, out=match_summary[0])" in decode_section
    assert "torch.all(matches, out=all_matches_tensor)" in decode_section
    assert "match_summary[1].copy_(all_matches_tensor)" in decode_section
    assert "match_count, all_matches = match_summary.tolist()" in decode_section
    assert "self.accepted_tokens += int(match_count)" in decode_section
    assert "if not all_matches:" in decode_section
    assert "torch.stack(" not in decode_section
    assert "torch.argmax(draft_logits" not in decode_section
    assert "torch.argmax(target_logits" not in decode_section
    assert "tokens = torch.where(matches" not in decode_section
    assert "matches.sum().item()" not in decode_section
    assert "if not matches.all()" not in decode_section


def test_ch18_vllm_decoder_reuses_prefill_next_token_buffer() -> None:
    source = (REPO_ROOT / "ch18" / "run_vllm_decoder.py").read_text(encoding="utf-8")
    benchmark_section = source.split("class VLLMMoEInferenceBenchmark", maxsplit=1)[1]
    eager_section = benchmark_section.split("def _run_eager_path", maxsplit=1)[1].split(
        "# --------------------------------------------------------------- benchmark_fn",
        maxsplit=1,
    )[0]

    assert "self._prefill_next_values: Optional[torch.Tensor] = None" in benchmark_section
    assert "self._prefill_next_tokens: Optional[torch.Tensor] = None" in benchmark_section
    assert "def _prefill_next_token_from_logits(self, logits: torch.Tensor) -> torch.Tensor" in benchmark_section
    assert (
        "torch.max(last_logits, dim=-1, keepdim=True, out=(self._prefill_next_values, self._prefill_next_tokens))"
        in benchmark_section
    )
    assert "self.spec_decoder.prepare_workspaces(cfg.batch_size, cfg.dtype_obj, self.device)" in benchmark_section
    assert "torch.argmax(logits[:, -1, :]" not in benchmark_section
    assert 'with torch.inference_mode(), self._nvtx_range("prefill_dualpipe"):' in eager_section
    assert 'with torch.inference_mode(), self._nvtx_range("speculative_decode"):' in eager_section
    assert "torch.no_grad()" not in eager_section


def test_ch18_vllm_v1_wrappers_reuse_token_id_buffers() -> None:
    for module_name in (
        "ch18.baseline_vllm_v1_integration",
        "ch18.optimized_vllm_v1_integration",
    ):
        module = importlib.import_module(module_name)
        source = (REPO_ROOT / module_name.replace(".", "/")).with_suffix(".py").read_text(
            encoding="utf-8"
        )
        token_batches = iter(([11, 12, 13], [21, 22, 23], [31, 32, 33, 34]))
        benchmark = module.get_benchmark()
        benchmark.runner = SimpleNamespace(
            batch_size=8,
            max_tokens=128,
            run=lambda: {"token_ids": next(token_batches)},
        )
        benchmark._metrics = {}
        benchmark.output = None
        benchmark._last_token_ids = None
        benchmark._token_id_buffer = None

        benchmark.benchmark_fn()
        first_ptr = benchmark._token_id_buffer.data_ptr()
        torch.testing.assert_close(
            benchmark.output,
            torch.tensor([11, 12, 13], dtype=torch.int32),
        )

        benchmark.benchmark_fn()
        assert benchmark._token_id_buffer.data_ptr() == first_ptr
        torch.testing.assert_close(
            benchmark.output,
            torch.tensor([21, 22, 23], dtype=torch.int32),
        )

        benchmark.benchmark_fn()
        assert benchmark._token_id_buffer.numel() >= 4
        torch.testing.assert_close(
            benchmark.output,
            torch.tensor([31, 32, 33, 34], dtype=torch.int32),
        )

        assert "self._token_id_buffer: Optional[torch.Tensor] = None" in source
        assert "def _materialize_token_ids" in source
        assert "torch.as_tensor(token_ids" not in source


def test_ch18_paged_vllm_cache_reset_is_metadata_only() -> None:
    source = (REPO_ROOT / "ch18" / "run_vllm_decoder.py").read_text(encoding="utf-8")
    cache_section = source.split("class PagedKVCache", maxsplit=1)[1].split(
        "class SpeculativeDecoder",
        maxsplit=1,
    )[0]

    assert "self.buffer = torch.empty(" in cache_section
    assert "self.buffer = torch.zeros(" not in cache_section
    assert "self.buffer.zero_()" not in cache_section
    assert "self.tokens_written = 0" in cache_section
    assert "self.page_faults = 0" in cache_section


def test_ch18_optimized_rope_q_cache_uses_inplace_rope_scratch() -> None:
    from ch18.rope_q_cache_common import apply_rope, apply_rope_inplace

    baseline_source = (REPO_ROOT / "ch18" / "baseline_rope_q_cache.py").read_text(encoding="utf-8")
    baseline_setup = baseline_source.split("def benchmark_fn", maxsplit=1)[0]
    source = (REPO_ROOT / "ch18" / "optimized_rope_q_cache.py").read_text(encoding="utf-8")
    setup_section = source.split("def benchmark_fn", maxsplit=1)[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "self.cache = torch.empty(" in baseline_setup
    assert "self.cache = torch.zeros(" not in baseline_setup
    assert "self.cache = torch.empty(" in setup_section
    assert "self.cache = torch.zeros(" not in setup_section
    assert "self.rope_scratch = torch.empty(" in setup_section
    assert "apply_rope_inplace(q, cos_t, sin_t, self.rope_scratch)" in benchmark_section
    assert "apply_rope(q, cos_t, sin_t)" not in benchmark_section

    q = torch.randn(2, 3, 8, dtype=torch.float32)
    cos = torch.randn(1, 1, 8, dtype=torch.float32)
    sin = torch.randn(1, 1, 8, dtype=torch.float32)
    expected = apply_rope(q.clone(), cos, sin)
    actual_input = q.clone()
    scratch = torch.empty_like(actual_input[..., :4])

    actual = apply_rope_inplace(actual_input, cos, sin, scratch)

    assert actual is actual_input
    torch.testing.assert_close(actual, expected)


def test_dynamic_router_wrappers_defer_metric_tensors_outside_hot_loop() -> None:
    for relative in (
        "labs/dynamic_router/baseline_dynamic_router_vllm.py",
        "labs/dynamic_router/optimized_dynamic_router_vllm.py",
        "labs/dynamic_router/baseline_dual_pool_vllm.py",
        "labs/dynamic_router/optimized_dual_pool_vllm.py",
        "labs/dynamic_router/topology_probe.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def teardown", maxsplit=1
        )[0]

        assert "torch.tensor(" not in benchmark_section
        assert "from labs.dynamic_router import vllm_runner" not in benchmark_section
        assert "self._metric_values = metric_values" in benchmark_section
        assert "self.output = torch.tensor(self._metric_values, dtype=torch.float32).unsqueeze(0)" in capture_section


def test_ch17_pipeline_parallelism_defers_multigpu_concat_outside_hot_loop() -> None:
    source = (REPO_ROOT / "ch17" / "optimized_pipeline_parallelism.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def get_custom_streams", maxsplit=1
    )[0]

    assert "with torch.inference_mode(), torch.autocast(\"cuda\", dtype=torch.bfloat16):" in setup_section
    assert "with torch.inference_mode(), torch.autocast(\"cuda\", dtype=torch.bfloat16):" in benchmark_section
    assert "self._stage_buffers: List[List[Optional[torch.Tensor]]] = []" in source
    assert "self._stage_transfer_buffers: List[List[Optional[torch.Tensor]]] = []" in source
    assert "self._stage_devices: List[torch.device] = []" in source
    assert "self._final_output_slots: List[torch.Tensor] = []" in source
    assert "self._last_final_output_count: int = 0" in source
    assert "self._stage_buffers = [" in setup_section
    assert "stage_output_features = [" in setup_section
    assert "self._stage_transfer_buffers = []" in setup_section
    assert "transfer_buffer.copy_(out, non_blocking=True)" in benchmark_section
    assert "self._stage_devices = [next(stage.parameters()).device for stage in self.pipeline_stages]" in setup_section
    assert "self._final_output_slots = [" in setup_section
    assert "stage_buffers: List[List[Optional[torch.Tensor]]] = [" not in benchmark_section
    assert "stage_buffers[0] = list(self.microbatch_inputs)" not in benchmark_section
    assert "stage_devices = [next(stage.parameters()).device for stage in self.pipeline_stages]" not in benchmark_section
    assert ".to(stage_devices[stage_idx])" not in benchmark_section
    assert ".to(next_device)" not in benchmark_section
    assert "final_outputs = [o for o in stage_buffers[num_stages] if o is not None]" not in benchmark_section
    assert "any(len(row) != self.micro_batches for row in self._stage_buffers)" not in benchmark_section
    assert "p.numel() for stage in self.pipeline_stages for p in stage.parameters()" not in benchmark_section
    assert "torch.cat(final_outputs" not in benchmark_section
    assert "self._last_final_outputs = final_outputs" in benchmark_section
    assert "self.output = torch.cat(self._last_final_outputs[: self._last_final_output_count], dim=0)" in capture_section


def test_ch17_inference_wrappers_use_inference_mode() -> None:
    for relative in (
        "ch17/baseline_inference_full.py",
        "ch17/optimized_inference_full.py",
        "ch17/baseline_memory.py",
        "ch17/baseline_routing_static.py",
        "ch17/optimized_routing_static.py",
        "ch17/baseline_pipeline_parallelism.py",
        "ch17/baseline_prefill_decode_disagg.py",
        "ch17/optimized_prefill_decode_disagg.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]
        assert "with torch.inference_mode():" in benchmark_section
        assert "with torch.no_grad():" not in benchmark_section
        if relative in {"ch17/baseline_inference_full.py", "ch17/optimized_inference_full.py"}:
            setup_section = source.split("def setup", maxsplit=1)[1].split(
                "def benchmark_fn",
                maxsplit=1,
            )[0]
            assert "with torch.inference_mode():" in setup_section
            assert "with torch.no_grad():" not in setup_section


def test_ch17_inference_models_use_inplace_relu_on_layer_outputs() -> None:
    for relative in (
        "ch17/baseline_inference_full.py",
        "ch17/optimized_inference_full.py",
        "ch17/baseline_routing_static.py",
        "ch17/optimized_routing_static.py",
        "ch17/prefill_decode_disagg_monolithic_common.py",
        "ch17/prefill_decode_disagg_multigpu_common.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")

        assert "torch.relu_(layer(x))" in source
        assert "torch.relu(layer(x))" not in source


def test_ch05_distributed_reduction_defers_verification_scalars_outside_hot_loop() -> None:
    baseline_source = (REPO_ROOT / "ch05" / "baseline_distributed_multigpu.py").read_text(encoding="utf-8")
    optimized_source = (REPO_ROOT / "ch05" / "optimized_distributed_multigpu.py").read_text(encoding="utf-8")
    baseline_benchmark = baseline_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    baseline_capture = baseline_source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]
    optimized_benchmark = optimized_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    optimized_setup = optimized_source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]

    assert "torch.tensor(" not in baseline_benchmark
    assert "self._cpu_total = cpu_total" in baseline_benchmark
    assert "self.output = torch.tensor(" in baseline_capture
    assert "[self._cpu_total]," in baseline_capture
    assert "self.local_sums = [torch.empty(1, device=t.device, dtype=torch.float32) for t in self.data]" in optimized_setup
    assert "self.reduced_sums = [torch.empty_like(t) for t in self.local_sums]" in optimized_setup
    assert "torch.zeros(1" not in optimized_setup
    assert "torch.zeros_like" not in optimized_setup
    assert "with torch.inference_mode(), self._nvtx_range(\"optimized_distributed_multigpu\"):" in optimized_benchmark
    assert "torch.sum(tensor, dim=0, keepdim=True, out=self.local_sums[idx])" in optimized_benchmark
    assert "tensor.sum().view(1)" not in optimized_benchmark
    assert ".copy_(tensor.sum()" not in optimized_benchmark
    assert "self.output = self.reduced_sums[0].detach().clone()" not in optimized_benchmark
    assert "torch.no_grad()" not in optimized_benchmark
    assert "torch.cuda.synchronize()" not in optimized_benchmark
    assert "self.output = self.reduced_sums[0]" in optimized_benchmark


def test_persistent_decode_graphs_reuses_timing_events_outside_hot_loop() -> None:
    source = (REPO_ROOT / "labs" / "persistent_decode" / "optimized_persistent_decode_graphs.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def benchmark_fn", maxsplit=1)[0]
    piecewise_capture_section = source.split("def _capture_piecewise_graphs", maxsplit=1)[1].split(
        "def _capture_full_graph", maxsplit=1
    )[0]
    full_capture_section = source.split("def _capture_full_graph", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def finalize_iteration_metrics", maxsplit=1
    )[0]

    assert "torch.cuda.Event(enable_timing=True)" in setup_section
    assert "self._full_events = {" in setup_section
    assert "self._piecewise_events = {" in setup_section
    assert "torch.cuda.Event(" not in piecewise_capture_section
    assert "torch.cuda.Event(" not in full_capture_section
    assert "torch.cuda.Event(" not in benchmark_section
    assert "self.inputs.out.zero_()" not in full_capture_section
    assert 'start = self._full_events["start"]' in benchmark_section
    assert 'start_prefill = self._piecewise_events["start_prefill"]' in benchmark_section


def test_persistent_decode_tma_reuses_timing_events_outside_hot_loop() -> None:
    source = (REPO_ROOT / "labs" / "persistent_decode" / "optimized_tma_prefill_decode.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def benchmark_fn", maxsplit=1)[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def finalize_iteration_metrics", maxsplit=1
    )[0]

    assert "torch.cuda.Event(enable_timing=True)" in setup_section
    assert "torch.cuda.Event(" not in benchmark_section
    assert 'start = self._full_events["start"]' in benchmark_section
    assert 'start_prefill = self._piecewise_events["start_prefill"]' in benchmark_section
    assert 'start_decode = self._piecewise_events["start_decode"]' in benchmark_section


def test_persistent_decode_baseline_reuses_decode_step_buffers() -> None:
    source = (REPO_ROOT / "labs" / "persistent_decode" / "baseline_persistent_decode.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def _decode_step",
        maxsplit=1,
    )[0]
    decode_step_section = source.split("def _decode_step", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]

    assert "self._product_buffer: Optional[torch.Tensor] = None" in source
    assert "self._dot_buffer: Optional[torch.Tensor] = None" in source
    assert "self._product_buffer = torch.empty(" in setup_section
    assert "self._dot_buffer = torch.empty(" in setup_section
    assert "torch.mul(q_t, k_t, out=product)" in decode_step_section
    assert "torch.sum(product, dim=-1, keepdim=True, out=dot)" in decode_step_section
    assert "torch.mul(v_t, dot, out=self.inputs.out[:, t, :])" in decode_step_section
    assert "(q_t * k_t).sum" not in decode_step_section
    assert "v_t * dot" not in decode_step_section


def test_persistent_decode_tma_buffers_avoid_zero_fill_before_overwrite() -> None:
    prefill_targets = [
        REPO_ROOT / "labs" / "persistent_decode" / "baseline_tma_prefill_decode.py",
        REPO_ROOT / "labs" / "persistent_decode" / "baseline_native_tma_prefill_decode.py",
        REPO_ROOT / "labs" / "persistent_decode" / "optimized_tma_prefill_decode.py",
        REPO_ROOT / "labs" / "persistent_decode" / "optimized_native_tma_prefill_decode.py",
    ]
    for path in prefill_targets:
        source = path.read_text(encoding="utf-8")
        assert "self.prefill_dst = torch.empty_like(self.prefill_src)" in source
        assert "self.prefill_dst = torch.zeros_like(self.prefill_src)" not in source

    for path in prefill_targets[2:]:
        source = path.read_text(encoding="utf-8")
        decode_graph_section = source.split("def _decode_graph", maxsplit=1)[1].split(
            "def benchmark_fn", maxsplit=1
        )[0]
        assert "self.graph_out = torch.empty_like(self.inputs.out)" in source
        assert "self.graph_out.zero_()" not in decode_graph_section


def test_optimized_flexdecode_graph_reuses_static_input_without_zero_fill() -> None:
    source = (REPO_ROOT / "ch18" / "optimized_flexdecoding_graphs.py").read_text(encoding="utf-8")
    setup_section = source.split("def _initialize_and_capture", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split("def teardown", maxsplit=1)[0]

    assert "self.static_decode_in = torch.empty_like(self.decode_token)" in setup_section
    assert "self.static_decode_in = torch.zeros_like(self.decode_token)" not in setup_section
    assert "self.static_decode_in.copy_(self.decode_token)" in benchmark_section


def test_ch18_flexdecoding_benchmarks_use_inference_mode() -> None:
    for filename in (
        "baseline_flexdecoding.py",
        "optimized_flexdecoding.py",
        "optimized_flexdecoding_graphs.py",
    ):
        source = (REPO_ROOT / "ch18" / filename).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "\n    def ", maxsplit=1
        )[0]

        assert "with torch.inference_mode():" in benchmark_section
        assert "with torch.no_grad():" not in benchmark_section


def test_ch18_optimized_flexdecoding_reuses_sdpa_backend_list() -> None:
    source = (REPO_ROOT / "ch18" / "optimized_flexdecoding.py").read_text(encoding="utf-8")
    init_section = source.split("def __init__", maxsplit=1)[1].split("def setup", maxsplit=1)[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def get_custom_metrics", maxsplit=1
    )[0]

    assert "self._flash_attention_backends = [SDPBackend.FLASH_ATTENTION]" in init_section
    assert "with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):" not in benchmark_section
    assert "with sdpa_kernel(self._flash_attention_backends):" in benchmark_section


def test_paged_kv_offload_prefetch_event_is_preallocated_outside_hot_loop() -> None:
    source = (REPO_ROOT / "labs" / "persistent_decode" / "paged_kv_offload_common.py").read_text(encoding="utf-8")
    setup_section = source.split("def benchmark_fn", maxsplit=1)[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "self.prefetch_event = torch.cuda.Event() if buffer_count == 2 else None" in setup_section
    assert "torch.cuda.Event(" not in benchmark_section
    assert "Prefetch event not initialized for async two-buffer prefetch" in benchmark_section


def test_paged_kv_offload_hot_page_buffers_avoid_zero_fill() -> None:
    source = (REPO_ROOT / "labs" / "persistent_decode" / "paged_kv_offload_common.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "# -------------------- Benchmark --------------------", maxsplit=1
    )[0]

    assert "self.hot_k_bufs = [torch.empty(" in setup_section
    assert "self.hot_v_bufs = [torch.empty_like(self.hot_k_bufs[0])" in setup_section
    assert "self.hot_k_bufs = [torch.zeros(" not in setup_section
    assert "self.hot_v_bufs = [torch.zeros_like(" not in setup_section


def test_nvlink_offload_copies_directly_between_preallocated_buffers() -> None:
    source = (REPO_ROOT / "labs" / "persistent_decode" / "nvlink_offload_common.py").read_text(
        encoding="utf-8"
    )
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert ".to(self.device" not in benchmark_section
    assert '.to("cpu"' not in benchmark_section
    assert "copy_(cpu_slice, non_blocking=self.cfg.non_blocking)" in benchmark_section
    assert "target.copy_(self.gpu_cache[..., :slice_len, :], non_blocking=self.cfg.non_blocking)" in benchmark_section


def test_cache_aware_disagg_reuses_request_events_and_defers_output_stack() -> None:
    source = (REPO_ROOT / "labs" / "cache_aware_disagg_inference" / "cache_aware_disagg_common.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def benchmark_fn", maxsplit=1)[0]
    helper_section = source.split("def _extend_cache_buffer", maxsplit=1)[1].split(
        "class CacheAwareDisaggBenchmark",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]

    assert "torch.cuda.Event(enable_timing=True)" in setup_section
    assert "torch.cuda.Event(" not in benchmark_section
    assert "with torch.inference_mode():" in setup_section
    assert "with torch.inference_mode():" in benchmark_section
    assert "torch.stack(" not in benchmark_section
    assert "prefix_parts" not in setup_section
    assert "prefix_buffer = torch.empty(" in setup_section
    assert "prefix_buffer[:, offset:next_offset].copy_(chunk_kv)" in setup_section
    assert "torch.cat(prefix_parts" not in setup_section
    assert "self._kv_buffers = {" in setup_section
    assert "kv_buffer = torch.empty(" in helper_section
    assert "kv_buffer = torch.empty(" not in benchmark_section
    assert "kv_buffers = self._kv_buffers" in benchmark_section
    assert "self._worker_caches = [{} for _ in range(self.cfg.logical_decode_workers)]" in setup_section
    assert "self._owners = {}" in setup_section
    assert "worker_caches = self._worker_caches" in benchmark_section
    assert "owners = self._owners" in benchmark_section
    assert "worker_caches = [{} for _ in range(self.cfg.logical_decode_workers)]" not in benchmark_section
    assert "owners: Dict[int, int] = {}" not in benchmark_section
    assert "for cache in worker_caches:" in benchmark_section
    assert "cache.clear()" in benchmark_section
    assert "_extend_cache_buffer(" in benchmark_section
    assert "kv_buffer[:, current_kv_len:next_kv_len].copy_(chunk_kv)" in helper_section
    assert "torch.cat((accumulated_kv, chunk_kv), dim=1)" not in benchmark_section
    assert "request_start, prefill_end, decode_end = request_events[event_idx]" in benchmark_section
    assert "self._last_outputs = [torch.empty(0) for _ in self.request_plans]" in setup_section
    assert "outputs: List[torch.Tensor] = []" not in benchmark_section
    assert "outputs = self._last_outputs" in benchmark_section
    assert "output_idx = 0" in benchmark_section
    assert "outputs.append(" not in benchmark_section
    assert "outputs[output_idx] = output" in benchmark_section
    assert "output_idx += 1" in benchmark_section
    assert "self._last_outputs = outputs" in benchmark_section
    assert "self._outputs_ready = True" in benchmark_section
    assert "if self.prompts is None or not self._outputs_ready:" in capture_section
    assert "self.output = torch.stack(self._last_outputs, dim=0)" in capture_section


def test_cache_aware_disagg_multigpu_reuses_kv_buffers_in_hot_path() -> None:
    source = (
        REPO_ROOT
        / "labs"
        / "cache_aware_disagg_inference"
        / "cache_aware_disagg_multigpu_common.py"
    ).read_text(encoding="utf-8")
    helper_section = source.split("def _extend_cache_buffer", maxsplit=1)[1].split(
        "def _world_size_hint", maxsplit=1
    )[0]
    run_iteration_section = source.split("def run_iteration", maxsplit=1)[1].split(
        "reduced = torch.tensor", maxsplit=1
    )[0]
    reduced_metrics_section = source.split("if rank == 0:", maxsplit=1)[1].split(
        "dist.destroy_process_group()", maxsplit=1
    )[0]
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def _prepare_verification_payload", maxsplit=1
    )[0]

    assert "kv_buffer = torch.empty(" in helper_section
    assert "target_device = cache.device" in helper_section
    assert "device=target_device" in helper_section
    assert "kv_buffer[:, current_kv_len:next_kv_len].copy_(chunk_kv, non_blocking=True)" in helper_section
    assert "prefix_parts" not in setup_section
    assert "prefix_buffer = torch.empty(" in setup_section
    assert "prefix_buffer[:, offset:next_offset].copy_(chunk_kv)" in setup_section
    assert "torch.cat(\n                    prefix_parts," not in setup_section
    assert "torch.cat((base, recv_chunk), dim=1)" not in run_iteration_section
    assert "torch.cat((cache, chunk_kv), dim=1)" not in benchmark_section
    assert "chunk_kv = chunk_kv.to(" not in benchmark_section
    assert "seed.to(decode_device)" not in benchmark_section
    assert "self._decode_seed_buffers[rank] = torch.empty(" in source
    assert "seed_buffer.copy_(seed, non_blocking=True)" in benchmark_section
    assert "seed_buffer," in benchmark_section
    assert "self._active_caches = {rank: {} for rank in self._decode_models}" in setup_section
    assert "self._kv_buffer_pools = {rank: {} for rank in self._decode_models}" in setup_section
    assert "active_caches = self._active_caches" in benchmark_section
    assert "kv_buffers = self._kv_buffer_pools" in benchmark_section
    assert "active_caches = {rank: {} for rank in self._decode_models}" not in benchmark_section
    assert "kv_buffers = {rank: {} for rank in self._decode_models}" not in benchmark_section
    assert "set(active_caches)" not in benchmark_section
    assert "set(kv_buffers)" not in benchmark_section
    assert "for rank in self._decode_models:" in benchmark_section
    assert "for cache in active_caches.values():" in benchmark_section
    assert "cache.clear()" in benchmark_section
    assert "_extend_cache_buffer(" in run_iteration_section
    assert "_extend_cache_buffer(" in benchmark_section
    assert "self._output_parts = [torch.empty(0) for _ in self._request_plans]" in setup_section
    assert "with torch.inference_mode():" in setup_section
    assert "with torch.inference_mode():" in benchmark_section
    assert "with torch.no_grad():" not in setup_section
    assert "with torch.no_grad():" not in benchmark_section
    assert "outputs: List[torch.Tensor] = []" not in benchmark_section
    assert "outputs = self._output_parts" in benchmark_section
    assert "output_idx = 0" in benchmark_section
    assert "outputs.append(" not in benchmark_section
    assert "outputs[output_idx] = output.detach()" in benchmark_section
    assert "output_idx += 1" in benchmark_section
    assert "self._outputs_ready = True" in benchmark_section
    assert "if not self._outputs_ready or self._verify_prompt is None:" in capture_section
    assert "reduced_values = reduced.detach().cpu().tolist()" in reduced_metrics_section
    assert ".item()" not in reduced_metrics_section


def test_nanochat_kv_cache_growth_avoids_cat_with_uninitialized_tail() -> None:
    source = (REPO_ROOT / "labs" / "nanochat_fullstack" / "nanochat" / "engine.py").read_text(
        encoding="utf-8"
    )
    grow_section = source.split("def _maybe_grow_cache", maxsplit=1)[1].split(
        "def get_block_info", maxsplit=1
    )[0]

    assert "torch.cat([self.kv_cache, additional_cache]" not in grow_section
    assert "grown_cache = torch.empty(grown_shape, dtype=dtype, device=device)" in grow_section
    assert "grown_cache[:, :, :, :, :old_seq_len, :].copy_(self.kv_cache)" in grow_section


def test_nanochat_incremental_benchmark_uses_cuda_event_timing() -> None:
    source = (
        REPO_ROOT / "labs" / "nanochat_fullstack" / "benchmark_incremental_optimizations.py"
    ).read_text(encoding="utf-8")
    helper_section = source.split("def _time_region_seconds", maxsplit=1)[1].split(
        "def benchmark_inference",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_inference", maxsplit=1)[1].split(
        "def run_incremental_benchmark",
        maxsplit=1,
    )[0]
    timed_section = benchmark_section.split("# Benchmark prefill", maxsplit=1)[1].split(
        "# Cleanup",
        maxsplit=1,
    )[0]

    assert helper_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "start.record()" in helper_section
    assert "end.record()" in helper_section
    assert "start.elapsed_time(end) / 1000.0" in helper_section
    assert "decode_token_steps = tuple(" in benchmark_section
    assert "decode_tokens[:, t:t + 1]" in benchmark_section
    assert "for step_ids in decode_token_steps[: min(8, self.decode_len)]:" in benchmark_section
    assert "for step_ids in decode_token_steps:" in timed_section
    assert "decode_tokens[:, t:t+1]" not in timed_section
    assert "decode_tokens[:, t:t + 1]" not in timed_section
    assert timed_section.count("self._time_region_seconds(") == 2
    assert "torch.cuda.synchronize()" not in timed_section
    assert "time.time()" not in timed_section


def test_nanochat_b200_flag_benchmark_uses_cuda_event_timing() -> None:
    source = (
        REPO_ROOT / "labs" / "nanochat_fullstack" / "scripts" / "bench_b200_flags.py"
    ).read_text(encoding="utf-8")
    helper_section = source.split("def _time_cuda_region_seconds", maxsplit=1)[1].split(
        "def configure_mode",
        maxsplit=1,
    )[0]
    run_once_section = source.split("def bench_once", maxsplit=1)[1].split(
        "def run_benchmark",
        maxsplit=1,
    )[0]

    assert helper_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "start.record()" in helper_section
    assert "end.record()" in helper_section
    assert "start.elapsed_time(end) / 1000.0" in helper_section
    assert "decode_token_steps = tuple(" in run_once_section
    assert "decode_tokens[:, t:t + 1]" in run_once_section
    assert "for step_ids in decode_token_steps:" in run_once_section
    assert "decode_tokens[:, t:t+1]" not in run_once_section
    assert run_once_section.count("_time_cuda_region_seconds(") == 2
    assert "time.time()" not in run_once_section
    assert "torch.cuda.synchronize()" not in run_once_section


def test_nanochat_gpt_generate_preallocates_token_buffer() -> None:
    source = (REPO_ROOT / "labs" / "nanochat_fullstack" / "nanochat" / "gpt.py").read_text(
        encoding="utf-8"
    )
    generate_section = source.split("def generate(self, tokens, max_tokens", maxsplit=1)[1]

    assert "self._generate_next_ids = None" in source
    assert "self._generate_choice_ids = None" in source
    assert "self._generate_probs = None" in source
    assert "def _generate_long_buffer" in source
    assert "def _generate_like_buffer" in source
    assert "def _generate_token_host_buffer" in source
    assert "ids = torch.empty((1, total_len), dtype=torch.long, device=device)" in generate_section
    assert "next_ids = self._generate_long_buffer(\"_generate_next_ids\", (1, 1), device)" in generate_section
    assert "choice = self._generate_long_buffer(\"_generate_choice_ids\", (1, 1), device)" in generate_section
    assert "logits = self.forward(ids[:, :cur_len])" in generate_section
    assert "torch.topk(logits, k, dim=-1, out=(top_vals, top_idx))" in generate_section
    assert "torch.softmax(top_vals, dim=-1, out=probs)" in generate_section
    assert "torch.multinomial(probs, num_samples=1, generator=rng, out=choice)" in generate_section
    assert "torch.gather(top_idx, 1, choice, out=next_ids)" in generate_section
    assert "torch.multinomial(probs, num_samples=1, generator=rng, out=next_ids)" in generate_section
    assert "torch.max(logits, dim=-1, keepdim=True, out=(max_values, next_ids))" in generate_section
    assert "ids[:, cur_len:cur_len + 1].copy_(next_ids)" in generate_section
    assert "token_host.copy_(next_ids.view(-1)[:1])" in generate_section
    assert "token = int(token_host[0])" in generate_section
    assert "ids = torch.cat((ids, next_ids), dim=1)" not in generate_section
    assert "next_ids = torch.multinomial" not in generate_section
    assert "choice = torch.multinomial" not in generate_section
    assert "next_ids = torch.argmax" not in generate_section
    assert "next_ids.item()" not in generate_section
    assert "logits[logits <" not in generate_section


def test_nanochat_optimized_inference_reuses_decode_step_views() -> None:
    source = (
        REPO_ROOT / "labs" / "nanochat_fullstack" / "optimized_nanochat_inference.py"
    ).read_text(encoding="utf-8")
    init_section = source.split("def __init__", maxsplit=1)[1].split(
        "def setup",
        maxsplit=1,
    )[0]
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert "self.decode_token_steps: tuple[torch.Tensor, ...] = ()" in init_section
    assert "self.decode_token_steps = tuple(" in setup_section
    assert "self.decode_tokens[:, t : t + 1]" in setup_section
    assert "for step_ids in self.decode_token_steps[: min(4, self.decode_len)]:" in setup_section
    assert "or not self.decode_token_steps" in benchmark_section
    assert "for step_ids in self.decode_token_steps:" in benchmark_section
    assert "self.decode_tokens[:, t : t + 1]" not in benchmark_section


def test_nanochat_prefix_causal_mask_avoids_zero_fill_and_tril_allocations() -> None:
    source = (REPO_ROOT / "labs" / "nanochat_fullstack" / "nanochat" / "gpt.py").read_text(
        encoding="utf-8"
    )
    causal_section = source.split("def _causal_mask_for", maxsplit=1)[1].split(
        "def _prefix_causal_mask_for",
        maxsplit=1,
    )[0]
    prefix_section = source.split("def _prefix_causal_mask_for", maxsplit=1)[1].split(
        "def _flash3_attention",
        maxsplit=1,
    )[0]

    assert "torch.ones((t_q, t_k)" not in causal_section
    assert "torch.tril(" not in causal_section
    assert "self._causal_mask_cache = k_pos <= q_pos" in causal_section
    assert "mask = torch.zeros((t_q, t_k)" not in prefix_section
    assert "torch.tril(" not in prefix_section
    assert "q_pos = torch.arange(t_q, device=device).unsqueeze(1)" in prefix_section
    assert "k_pos = torch.arange(t_k, device=device).unsqueeze(0)" in prefix_section
    assert "mask = k_pos <= (prefix_len + q_pos)" in prefix_section


def test_nanochat_padded_kv_rotary_bounds_avoid_device_scalar_read() -> None:
    source = (REPO_ROOT / "labs" / "nanochat_fullstack" / "nanochat" / "gpt.py").read_text(
        encoding="utf-8"
    )
    forward_prefix = source.split(
        "def forward(self, idx, targets=None, kv_cache=None",
        maxsplit=1,
    )[1].split(
        "# Forward the trunk of the Transformer",
        maxsplit=1,
    )[0]

    assert "max_pos = kv_cache.get_pos() + T" in forward_prefix
    assert "positions.max().item()" not in forward_prefix


def test_nanochat_loss_eval_batches_reduced_totals() -> None:
    source = (REPO_ROOT / "labs" / "nanochat_fullstack" / "nanochat" / "loss_eval.py").read_text(
        encoding="utf-8"
    )

    assert "totals = torch.stack((total_nats.to(torch.float64), total_bytes.to(torch.float64)))" in source
    assert "dist.all_reduce(totals, op=dist.ReduceOp.SUM)" in source
    assert "total_nats, total_bytes = totals.detach().cpu().tolist()" in source
    assert "@torch.inference_mode()\ndef evaluate_bpb" in source
    assert "@torch.no_grad()" not in source
    assert "dist.all_reduce(total_nats" not in source
    assert "dist.all_reduce(total_bytes" not in source
    assert "total_nats = total_nats.item()" not in source
    assert "total_bytes = total_bytes.item()" not in source
    assert "y_safe = y.clamp_min(0)" in source
    assert "num_bytes2d = token_bytes[y_safe] * valid.to(dtype=token_bytes.dtype)" in source
    assert "torch.zeros_like(y)" not in source


def test_nanochat_dist_muon_reuses_padding_buffers() -> None:
    source = (REPO_ROOT / "labs" / "nanochat_fullstack" / "nanochat" / "muon.py").read_text(
        encoding="utf-8"
    )
    dist_muon_source = source.split("class DistMuon", maxsplit=1)[1]
    init_section = dist_muon_source.split("def __init__", maxsplit=1)[1].split(
        "def step",
        maxsplit=1,
    )[0]
    step_section = dist_muon_source.split("def step", maxsplit=1)[1]

    assert "scatter_pad_buffer=torch.empty_like(group_params[0])" in init_section
    assert "gather_pad_buffers=[" in init_section
    assert "rs_output = params[owner_idx].grad if owner_idx < len(params) else scatter_pad_buffer" in step_section
    assert "ag_output.extend(gather_pad_buffers[:missing])" in step_section
    assert "torch.empty_like(zero_buffer)" not in step_section


def test_nanochat_dist_adamw_reuses_update_buffers() -> None:
    source = (REPO_ROOT / "labs" / "nanochat_fullstack" / "nanochat" / "adamw.py").read_text(
        encoding="utf-8"
    )
    step_section = source.split("def step", maxsplit=1)[1]

    assert 'if "_grad_slice" not in state:' in step_section
    assert 'grad_slice = state["_grad_slice"]' in step_section
    assert 'if "step" not in state:' in step_section
    assert "state['denom'] = torch.empty_like(p_slice)" in step_section
    assert "torch.sqrt(exp_avg_sq, out=denom)" in step_section
    assert "torch.div(exp_avg, denom, out=g_slice)" in step_section
    assert "denom = exp_avg_sq.sqrt()" not in step_section
    assert "update = exp_avg.div" not in step_section


def test_nanochat_clustered_attention_fallback_uses_native_sdpa_gqa(monkeypatch: pytest.MonkeyPatch) -> None:
    from labs.nanochat_fullstack.nanochat.kernels import clustered_attention as clustered_attention_module

    source = inspect.getsource(clustered_attention_module.clustered_attention)
    flash3_source = inspect.getsource(clustered_attention_module._flash3_clustered)
    assert "repeat_interleave" not in source
    assert "repeat_interleave" not in flash3_source
    assert "inspect.signature" not in flash3_source
    assert "_flash3_accepts_clusters" in flash3_source
    assert "enable_gqa=enable_gqa" in source

    calls: dict[str, object] = {}

    def _fake_sdpa(q, k, v, **kwargs):
        calls["k_heads"] = k.size(1)
        calls["v_heads"] = v.size(1)
        calls["enable_gqa"] = kwargs.get("enable_gqa")
        calls["is_causal"] = kwargs.get("is_causal")
        return torch.zeros_like(q)

    monkeypatch.setattr(clustered_attention_module.F, "scaled_dot_product_attention", _fake_sdpa)

    q = torch.randn(1, 4, 3, 2)
    k = torch.randn(1, 2, 3, 2)
    v = torch.randn(1, 2, 3, 2)
    expanded = clustered_attention_module._expand_gqa_heads(k, 2)

    torch.testing.assert_close(expanded, k.repeat_interleave(2, dim=1))

    output = clustered_attention_module.clustered_attention(
        q,
        k,
        v,
        attn_mask=None,
        causal=True,
        enable_gqa=True,
    )

    assert output.shape == q.shape
    assert calls == {"k_heads": 2, "v_heads": 2, "enable_gqa": True, "is_causal": True}


def test_nanochat_core_eval_batches_option_loss_reads() -> None:
    source = (REPO_ROOT / "labs" / "nanochat_fullstack" / "nanochat" / "core_eval.py").read_text(
        encoding="utf-8"
    )
    option_section = source.split("elif task_type in ['multiple_choice', 'schema']:", maxsplit=1)[1].split(
        "else:",
        maxsplit=1,
    )[0]

    assert "mean_losses = torch.stack(" in option_section
    assert ").detach().cpu().tolist()" in option_section
    assert ".mean().item()" not in option_section
    assert source.count("@torch.inference_mode()") >= 2
    assert "@torch.no_grad()" not in source


def test_ch15_disaggregated_multigpu_defers_output_cpu_concat() -> None:
    source = (
        REPO_ROOT / "ch15" / "baseline_disaggregated_inference_multigpu.py"
    ).read_text(encoding="utf-8")
    decode_helper = source.split("def _run_decode", maxsplit=1)[1].split(
        "def _run_torchrun_worker", maxsplit=1
    )[0]
    torchrun_worker = source.split("def _run_torchrun_worker", maxsplit=1)[1].split(
        "class _DisaggregatedInferenceMultiGPUBenchmark", maxsplit=1
    )[0]
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def _prepare_verification_payload", maxsplit=1
    )[0]

    assert "outputs = [torch.empty(0) for _ in range(len(kv_chunks))]" in decode_helper
    assert "with torch.inference_mode():" in decode_helper
    assert "with torch.inference_mode():" in torchrun_worker
    assert "with torch.inference_mode():" in benchmark_section
    assert "request_kv_cache = kv_cache" in decode_helper
    assert "request_kv_cache = allocate_kv_cache(" in decode_helper
    assert "request_kv_cache[:, : cfg.context_window].copy_(kv_prompt)" in decode_helper
    assert "outputs.append(" not in decode_helper
    assert "outputs[output_idx] = tokens" in decode_helper
    assert "decode_kv_cache = allocate_kv_cache(" in torchrun_worker
    assert "decode_outputs = [torch.empty(0) for _ in range(cfg.requests_per_rank)]" in torchrun_worker
    assert "outputs[req_idx] = tokens" in torchrun_worker
    assert "outputs.append(" not in torchrun_worker
    assert "decode_kv_cache = allocate_kv_cache(" in setup_section
    assert "decode_outputs=[torch.empty(0) for _ in range(self.cfg.requests_per_rank)]" in setup_section
    assert "transfer_kv_chunks=[torch.empty(0) for _ in range(self.cfg.requests_per_rank)]" in setup_section
    assert "transfer_seed_chunks=[torch.empty(0) for _ in range(self.cfg.requests_per_rank)]" in setup_section
    assert "self._pending_outputs = [" in setup_section
    assert "out.detach().cpu()" not in benchmark_section
    assert "torch.cat([out.detach().cpu()" not in benchmark_section
    assert "outputs: List[torch.Tensor] = []" not in benchmark_section
    assert "[kv.to(pair.decode_device" not in benchmark_section
    assert "[seed.to(pair.decode_device" not in benchmark_section
    assert "pair.transfer_kv_chunks[req_idx] = kv_chunks[req_idx].to(" in benchmark_section
    assert "pair.transfer_seed_chunks[req_idx] = seed_chunks[req_idx].to(" in benchmark_section
    assert "outputs = self._pending_outputs" in benchmark_section
    assert "output_idx = 0" in benchmark_section
    assert "kv_cache=pair.decode_kv_cache" in benchmark_section
    assert "outputs=pair.decode_outputs" in benchmark_section
    assert "outputs.extend(" not in benchmark_section
    assert "outputs[output_idx] = decoded_tokens" in benchmark_section
    assert "output_idx += 1" in benchmark_section
    assert "self._pending_outputs = outputs" in benchmark_section
    assert "torch.cat([out.detach().cpu() for out in self._pending_outputs], dim=0)" in capture_section


def test_ch15_sdpa_attention_reuses_kv_concat_buffers() -> None:
    from ch15.disaggregated_inference_multigpu import ScaledDotProductAttentionLayer

    layer = ScaledDotProductAttentionLayer(
        embed_dim=4,
        num_heads=2,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
    )
    past_k = torch.arange(12, dtype=torch.float32).view(1, 2, 3, 2)
    past_v = past_k + 100
    k = torch.arange(4, dtype=torch.float32).view(1, 2, 1, 2)
    v = k + 200
    out_k, out_v = layer._concat_kv_cache(past_k, past_v, k, v)

    torch.testing.assert_close(out_k, torch.cat([past_k, k], dim=2))
    torch.testing.assert_close(out_v, torch.cat([past_v, v], dim=2))

    source = (REPO_ROOT / "ch15" / "disaggregated_inference_multigpu.py").read_text(
        encoding="utf-8"
    )
    attention_section = source.split("class ScaledDotProductAttentionLayer", maxsplit=1)[1].split(
        "class PrefillKernel",
        maxsplit=1,
    )[0]

    assert "self._k_cat_buffer: Optional[torch.Tensor] = None" in attention_section
    assert "def _concat_kv_cache(" in attention_section
    assert "k, v = self._concat_kv_cache(past_k, past_v, k, v)" in attention_section
    assert "torch.cat([past_k, k]" not in attention_section
    assert "torch.cat([past_v, v]" not in attention_section


def test_ch15_decode_worker_reuses_sampling_buffers() -> None:
    source = (REPO_ROOT / "ch15" / "disaggregated_inference_multigpu.py").read_text(
        encoding="utf-8"
    )
    decode_worker_section = source.split("class DecodeWorker", maxsplit=1)[1].split(
        "class MoERouter",
        maxsplit=1,
    )[0]
    generate_section = decode_worker_section.split("def generate_next_token", maxsplit=1)[1]

    assert "self._sample_logits = torch.empty(" in decode_worker_section
    assert "self._sample_probs = torch.empty_like(self._sample_logits)" in decode_worker_section
    assert "self._sample_token = torch.empty(1, dtype=torch.long" in decode_worker_section
    assert "self._sample_token_host = torch.empty(1, dtype=torch.long" in decode_worker_section
    assert "self._sample_logits.copy_(logits[0])" in generate_section
    assert "torch.softmax(self._sample_logits, dim=-1, out=self._sample_probs)" in generate_section
    assert "torch.multinomial(self._sample_probs, num_samples=1, out=self._sample_token)" in generate_section
    assert "self._last_token_id.copy_(self._sample_token)" in generate_section
    assert "self._sample_token_host.copy_(self._sample_token)" in generate_section
    assert "token_index = int(self._sample_token_host[0])" in generate_section
    assert "logits[0].float()" not in generate_section
    assert "next_token = torch.multinomial" not in generate_section
    assert "next_token.to(self.device)" not in generate_section
    assert "next_token.item()" not in generate_section


def test_ch15_moe_inference_reuses_next_token_buffer() -> None:
    source = (REPO_ROOT / "ch15" / "moe_inference_common.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def _prepare_iteration_metrics",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def finalize_iteration_metrics",
        maxsplit=1,
    )[0]

    assert "self._next_token_buffer: Optional[torch.Tensor] = None" in source
    assert "self._next_token_buffer = torch.empty((cfg.batch_size, 1)" in setup_section
    assert "def _next_token_from_logits" in source
    assert "with torch.inference_mode():" in benchmark_section
    assert "torch.max(logits_last, dim=-1, keepdim=True, out=(self._next_token_values, self._next_token_buffer))" in source
    assert "seed_tokens = self._next_token_from_logits(logits[:, -1, :])" in benchmark_section
    assert "seed_tokens = self._next_token_from_logits(decode_logits[:, -1, :])" in benchmark_section
    assert "torch.argmax(" not in benchmark_section


def test_ch15_single_disaggregated_optimized_reuses_next_token_buffer() -> None:
    source = (REPO_ROOT / "ch15" / "disaggregated_inference_single_common.py").read_text(
        encoding="utf-8"
    )
    optimized_section = source.split(
        "class OptimizedDisaggregatedInferenceSingleGPUBenchmark",
        maxsplit=1,
    )[1]

    assert "self._next_token_buffer: Optional[torch.Tensor] = None" in source
    assert "def _next_token_from_logits" in source
    assert "with torch.inference_mode():" in optimized_section
    assert "torch.max(logits_last, dim=-1, keepdim=True, out=(self._next_token_values, self._next_token_buffer))" in source
    assert "seed_tokens = self._next_token_from_logits(logits[:, -1, :])" in optimized_section
    assert "tokens = self._next_token_from_logits(decode_logits[:, -1, :])" in optimized_section
    assert "torch.argmax(" not in optimized_section


def test_ch15_baseline_kv_cache_nvlink_pool_reuses_gather_buffers() -> None:
    for filename in (
        "baseline_kv_cache_nvlink_pool.py",
        "baseline_kv_cache_nvlink_pool_multigpu.py",
    ):
        source = (REPO_ROOT / "ch15" / filename).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split(
            "def benchmark_fn", maxsplit=1
        )[0]
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]

        assert "self._k_gather_buffer = torch.empty(" in setup_section
        assert "self._v_gather_buffer = torch.empty_like(self._k_gather_buffer)" in setup_section
        assert "_slots" in setup_section
        assert "gathered_k" not in benchmark_section
        assert "gathered_v" not in benchmark_section
        assert "torch.cat(" not in benchmark_section
        assert ".to(self.device" not in benchmark_section
        assert ".append(" not in benchmark_section
        assert ".pop(0)" not in benchmark_section
        assert "with torch.inference_mode(), self._nvtx_range(" in benchmark_section
        assert "self._k_gather_buffer[:, gather_idx : gather_idx + 1, :].copy_(" in benchmark_section
        assert "k_all = self._k_gather_buffer[:, :gather_idx, :]" in benchmark_section
        assert "v_all = self._v_gather_buffer[:, :gather_idx, :]" in benchmark_section


def test_ch15_optimized_kv_cache_nvlink_pool_reuses_gather_buffers() -> None:
    for filename in (
        "optimized_kv_cache_nvlink_pool.py",
        "optimized_kv_cache_nvlink_pool_multigpu.py",
    ):
        source = (REPO_ROOT / "ch15" / filename).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split(
            "def _place_kv", maxsplit=1
        )[0]
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]

        assert "self._k_gather_buffer = torch.empty(" in setup_section
        assert "self._v_gather_buffer = torch.empty_like(self._k_gather_buffer)" in setup_section
        assert "self._cache_key_slots = [" in setup_section
        assert "self._cache_value_slots = [" in setup_section
        assert "self._tier_slots = [\"\"] * self.seq_len" in setup_section
        assert "torch.cat(" not in benchmark_section
        assert ".to(self.device" not in benchmark_section
        assert ".append(" not in benchmark_section
        assert "with torch.inference_mode(), self._nvtx_range(" in benchmark_section
        assert "self._gather_kv_into_buffers(cache_k, cache_v, tiers, step + 1)" in benchmark_section

    from ch15.optimized_kv_cache_nvlink_pool import (
        OptimizedKVCacheNvlinkPoolBenchmark as SinglePool,
    )
    from ch15.optimized_kv_cache_nvlink_pool_multigpu import (
        OptimizedKVCacheNvlinkPoolBenchmark as MultiPool,
    )

    for benchmark_cls in (SinglePool, MultiPool):
        bench = benchmark_cls()
        bench._k_gather_buffer = torch.empty(2, 3, 4)
        bench._v_gather_buffer = torch.empty(2, 3, 4)
        cache_k = [torch.full((2, 1, 4), float(idx)) for idx in range(3)]
        cache_v = [torch.full((2, 1, 4), float(idx + 10)) for idx in range(3)]

        gathered_k, gathered_v = bench._gather_kv_into_buffers(
            cache_k,
            cache_v,
            ["local", "peer", "host"],
        )

        assert gathered_k.data_ptr() == bench._k_gather_buffer.data_ptr()
        assert gathered_v.data_ptr() == bench._v_gather_buffer.data_ptr()
        torch.testing.assert_close(gathered_k, torch.cat(cache_k, dim=1))
        torch.testing.assert_close(gathered_v, torch.cat(cache_v, dim=1))


def test_ch02_grace_coherent_memory_defers_verification_slice_clone() -> None:
    for filename in (
        "baseline_grace_coherent_memory.py",
        "optimized_grace_coherent_memory.py",
    ):
        source = (REPO_ROOT / "ch02" / filename).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def teardown", maxsplit=1
        )[0]

        assert "cpu_data[:1000].detach().cpu().clone()" not in benchmark_section
        assert "self.output = None" in benchmark_section
        assert "self.output = self._impl.cpu_data[:1000].detach().clone()" in capture_section


def test_ch15_guided_decoding_reuses_mask_and_slice_buffers() -> None:
    source = (REPO_ROOT / "ch15" / "guided_decoding_common.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "self.masked_logits_buffer = torch.empty_like(self.logits)" in source
    assert "self.output_buffer = torch.empty(" in source
    assert "self.allowed_mask" not in source
    assert "self.disallowed_mask_buffer = torch.logical_not(mask)" not in setup_section
    assert "disallowed = torch.ones(self.vocab_size, dtype=torch.bool, device=self.device)" in setup_section
    assert "disallowed[self.allowed_token_ids.to(self.device)] = False" in setup_section
    assert "masked = logits.masked_fill" not in benchmark_section
    assert ".index_select(1, self.slice_ids)" not in benchmark_section
    assert "masked_logits.masked_fill_(self.disallowed_mask_buffer, float(\"-inf\"))" in benchmark_section
    assert "torch.index_select(masked_logits, 1, self.slice_ids, out=output)" in benchmark_section
    assert "torch.index_select(masked_logits, 1, self.slice_ids_buffer, out=output)" in benchmark_section


def test_ch17_multigpu_prefill_decode_reuses_overlap_events_and_defers_output_stack() -> None:
    source = (REPO_ROOT / "ch17" / "prefill_decode_disagg_multigpu_common.py").read_text(
        encoding="utf-8"
    )
    prefill_helper = source.split("def _run_prefill", maxsplit=1)[1].split(
        "def _run_decode", maxsplit=1
    )[0]
    decode_helper = source.split("def _run_decode", maxsplit=1)[1].split(
        "def _run_torchrun_worker", maxsplit=1
    )[0]
    worker_section = source.split("def _run_torchrun_worker", maxsplit=1)[1].split(
        "class _PrefillDecodeMultiGPUBenchmark", maxsplit=1
    )[0]
    run_iteration_section = worker_section.split("def run_iteration", maxsplit=1)[1].split(
        "_barrier()", maxsplit=1
    )[0]
    class_section = source.split("class _PrefillDecodeMultiGPUBenchmark", maxsplit=1)[1]
    setup_section = class_section.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    benchmark_section = class_section.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    capture_section = class_section.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def _prepare_verification_payload", maxsplit=1
    )[0]

    assert "kv_chunks = [torch.empty(0) for _ in range(cfg.requests_per_rank)]" in prefill_helper
    assert "seed_chunks = [torch.empty(0) for _ in range(cfg.requests_per_rank)]" in prefill_helper
    assert "kv_chunks[req_idx] = kv_cache" in prefill_helper
    assert "seed_chunks[req_idx] = seed" in prefill_helper
    assert ".append(" not in prefill_helper
    assert "outputs = [torch.empty(0) for _ in range(len(kv_chunks))]" in decode_helper
    assert "for output_idx, (kv_cache, seed) in enumerate(zip(kv_chunks, seed_chunks)):" in decode_helper
    assert "outputs[output_idx] = model.decode(seed, kv_cache, cfg.decode_tokens)" in decode_helper
    assert "outputs.append(" not in decode_helper
    assert "torch.cuda.Event(blocking=False)" in worker_section
    assert "ready = ready_events[group_idx]" in run_iteration_section
    assert "torch.cuda.Event(" not in run_iteration_section
    assert "with torch.inference_mode():" in run_iteration_section
    assert "expected_outputs = len(self._pairs) * self.cfg.requests_per_rank" in setup_section
    assert "self._pending_outputs = [torch.empty(0) for _ in range(expected_outputs)]" in setup_section
    assert "transfer_kv_chunks=[" in setup_section
    assert "self.cfg.context_window" in setup_section
    assert "transfer_seed_chunks=[" in setup_section
    assert "device=decode_device" in setup_section
    assert "with torch.inference_mode():" in benchmark_section
    assert "torch.stack(" not in benchmark_section
    assert ".detach().cpu()" not in benchmark_section
    assert "outputs = self._pending_outputs" in benchmark_section
    assert "output_idx = 0" in benchmark_section
    assert "outputs[output_idx] = pair.decode_model.decode(" in benchmark_section
    assert "transfer_seed," in benchmark_section
    assert "transfer_kv," in benchmark_section
    assert "for decoded_output in decoded:" in benchmark_section
    assert "outputs[output_idx] = decoded_output" in benchmark_section
    assert "output_idx += 1" in benchmark_section
    assert "self._pending_outputs = outputs" in benchmark_section
    assert "outputs: List[torch.Tensor] = []" not in benchmark_section
    assert "kv_chunks = [kv.to(pair.decode_device) for kv in kv_chunks]" not in benchmark_section
    assert "seed_chunks = [seed.to(pair.decode_device) for seed in seed_chunks]" not in benchmark_section
    assert "kv_cache = kv_cache.to(pair.decode_device" not in benchmark_section
    assert "seed = seed.to(pair.decode_device" not in benchmark_section
    assert "transfer_kv.copy_(kv_cache, non_blocking=True)" in benchmark_section
    assert "transfer_seed.copy_(seed, non_blocking=True)" in benchmark_section
    assert "pair.transfer_kv_chunks[req_idx].copy_(" in benchmark_section
    assert "pair.transfer_seed_chunks[req_idx].copy_(" in benchmark_section
    assert ".to(pair.decode_device)" not in benchmark_section
    assert "outputs.append(" not in benchmark_section
    assert "outputs.extend(" not in benchmark_section
    assert "self._output = torch.stack(" in capture_section
    assert "[out.detach().cpu() for out in self._pending_outputs]" in capture_section


def test_deepseek_moe_reuses_timing_events_and_defers_verification_casts() -> None:
    from labs.real_world_models.deepseek_r1_moe_optimization import MoELayer

    torch.manual_seed(123)
    layer = MoELayer(hidden_size=4, num_experts=3, top_k=2, intermediate_size=8)
    x = torch.randn(2, 3, 4)
    output, _ = layer(x)
    route_count_host = layer._route_count_host_buffer
    routing_weights, selected_experts, _ = layer.router(x)
    x_flat = x.view(-1, x.shape[-1])
    expected = torch.zeros_like(x_flat)
    for token_idx in range(x_flat.shape[0]):
        for route_idx in range(layer.top_k):
            expert_idx = int(selected_experts.view(-1, layer.top_k)[token_idx, route_idx])
            expert_out = layer.experts[expert_idx](x_flat[token_idx : token_idx + 1]).squeeze(0)
            expected[token_idx] += expert_out * routing_weights.view(-1, layer.top_k)[token_idx, route_idx]
    torch.testing.assert_close(output.view_as(expected), expected, rtol=1e-5, atol=1e-5)
    layer(x)
    assert layer._route_count_host_buffer is route_count_host

    source = (REPO_ROOT / "labs" / "real_world_models" / "deepseek_r1_moe_optimization.py").read_text(
        encoding="utf-8"
    )
    router_forward = source.split("class LoadBalancedRouter", maxsplit=1)[1].split(
        "class ExpertMLP",
        maxsplit=1,
    )[0]
    moe_forward = source.split("class MoELayer", maxsplit=1)[1].split(
        "class DeepSeekR1MoEOptimization",
        maxsplit=1,
    )[0]
    setup_section = source.split("def benchmark_fn", maxsplit=1)[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def finalize_iteration_metrics", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def get_custom_metrics", maxsplit=1
    )[0]

    assert "torch.cuda.Event(enable_timing=True)" in setup_section
    assert 'self.register_buffer(\n            "_gini_index",' in router_forward
    assert "def _gini_index_for" in router_forward
    assert "torch.arange(1, n + 1" not in router_forward
    assert "def _route_token_ids" in moe_forward
    assert "repeat_interleave(self.top_k)" not in moe_forward
    assert "output_flat = torch.empty_like(x_flat)" in moe_forward
    assert "output_flat = torch.zeros_like(x_flat)" not in moe_forward
    assert "output_flat.index_copy_(0, token_indices, expert_output * weights)" in moe_forward
    assert 'token_ids.div_(routes, rounding_mode="floor")' in moe_forward
    assert "self._route_count_host_buffer: Optional[torch.Tensor] = None" in moe_forward
    assert "def _route_count_list(self, expert_ids: torch.Tensor)" in moe_forward
    assert "counts = torch.bincount(expert_ids, minlength=self.num_experts)" in moe_forward
    assert "self._route_count_host_buffer.copy_(counts)" in moe_forward
    assert "torch.argsort(remaining_experts)" in moe_forward
    assert "first_count_list = self._route_count_list(first_experts)" in moe_forward
    assert "route_count_list = self._route_count_list(remaining_experts)" in moe_forward
    assert "torch.bincount(remaining_experts, minlength=self.num_experts).detach().cpu().tolist()" not in moe_forward
    assert "torch.bincount(first_experts, minlength=self.num_experts).detach().cpu().tolist()" not in moe_forward
    assert ".nonzero(" not in moe_forward
    assert "output_flat.index_add_(0, token_indices, expert_output * weights)" in moe_forward
    assert "torch.cuda.Event(" not in benchmark_section
    assert "start_event, end_event = self._timing_events" in benchmark_section
    assert "with torch.inference_mode():" in benchmark_section
    assert ".detach().float().clone()" not in benchmark_section
    assert "self._last_aux_metrics.clear()" in benchmark_section
    assert "self._last_aux_metrics[key] = value.detach()" in benchmark_section
    assert "self._last_aux_metrics = {" not in benchmark_section
    assert "self.output = output[:1, : min(4, output.shape[1]), : min(8, output.shape[2])]" in benchmark_section
    assert "output=self.output.detach().float().clone()" in capture_section


def test_gpt4_architecture_runner_reuses_cuda_timing_events() -> None:
    source = (
        REPO_ROOT / "labs" / "real_world_models" / "gpt4_architecture_optimization.py"
    ).read_text(encoding="utf-8")
    class_section = source.split("class GPT4ArchitectureOptimization:", maxsplit=1)[1].split(
        "class GPT4ArchitectureOptimizationBenchmark",
        maxsplit=1,
    )[0]
    init_section = class_section.split("def __init__", maxsplit=1)[1].split(
        "def _estimate_memory",
        maxsplit=1,
    )[0]
    setup_section = class_section.split("def setup", maxsplit=1)[1].split(
        "def run",
        maxsplit=1,
    )[0]
    run_section = class_section.split("def run", maxsplit=1)[1].split(
        "def cleanup",
        maxsplit=1,
    )[0]

    assert "self._timing_events: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None" in init_section
    assert setup_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "]).to(self.device).to(torch.bfloat16).eval()" in setup_section
    assert "@torch.inference_mode()\n    def run" in class_section
    assert "start_event, end_event = self._timing_events" in run_section
    assert "start_event.record()" in run_section
    assert "end_event.record()" in run_section
    assert "end_event.synchronize()" in run_section
    assert "elapsed_ms = start_event.elapsed_time(end_event)" in run_section
    assert "torch.no_grad()" not in run_section
    assert "torch.cuda.synchronize()" not in run_section


def test_ch14_attention_eager_sdpa_avoids_hot_path_host_sync_and_stack() -> None:
    baseline_source = (REPO_ROOT / "ch14" / "baseline_attention_eager_sdpa.py").read_text(encoding="utf-8")
    optimized_source = (REPO_ROOT / "ch14" / "optimized_attention_eager_sdpa.py").read_text(encoding="utf-8")

    baseline_benchmark = baseline_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    baseline_capture = baseline_source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]
    optimized_benchmark = optimized_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "float(stacked.sum())" not in baseline_benchmark
    assert "torch.stack(" not in baseline_benchmark
    assert "outputs = []" not in baseline_benchmark
    assert "outputs.append(" not in baseline_benchmark
    assert "self._last_outputs[output_idx] = torch.matmul(attn, vh)" in baseline_benchmark
    assert "output_idx += 1" in baseline_benchmark
    assert "stacked = torch.stack(self._last_outputs, dim=1)" in baseline_capture
    assert "float(out.sum())" not in optimized_benchmark


def test_ch14_wrappers_cache_nvtx_enabled_outside_hot_loop() -> None:
    for filename in (
        "baseline_attention_eager_sdpa.py",
        "optimized_attention_eager_sdpa.py",
        "baseline_cublas_vs_cutlass.py",
        "optimized_cublas_vs_cutlass.py",
        "baseline_model_compile_reduced_precision.py",
        "optimized_model_compile_reduced_precision.py",
        "baseline_nccl_quantization.py",
        "optimized_nccl_quantization.py",
    ):
        source = (REPO_ROOT / "ch14" / filename).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split(
            "def benchmark_fn", maxsplit=1
        )[0]
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]

        assert "self._enable_nvtx = get_nvtx_enabled(config) if config else False" in setup_section
        assert "get_config()" not in benchmark_section
        assert "get_nvtx_enabled(" not in benchmark_section
        assert "enable=self._enable_nvtx" in benchmark_section


def test_ch14_forward_benchmarks_use_inference_mode() -> None:
    benchmark_files = (
        "baseline_attention_eager_sdpa.py",
        "optimized_attention_eager_sdpa.py",
        "baseline_flex_attention_sparse.py",
        "optimized_flex_attention_sparse.py",
        "baseline_graph_break_control_flow.py",
        "optimized_graph_break_control_flow.py",
        "baseline_model_compile_reduced_precision.py",
        "optimized_model_compile_reduced_precision.py",
        "baseline_regional_triton.py",
        "optimized_regional_triton.py",
        "baseline_sliding_window.py",
        "optimized_sliding_window.py",
        "flash_attention_sdpa_bench.py",
        "flex_attention_sparse_demo.py",
        "sliding_window_demo.py",
    )
    setup_files = benchmark_files[1:]
    verification_files = (
        "baseline_graph_break_control_flow.py",
        "optimized_graph_break_control_flow.py",
    )

    for filename in benchmark_files:
        source = (REPO_ROOT / "ch14" / filename).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]
        assert "torch.inference_mode()" in benchmark_section
        assert "torch.no_grad()" not in benchmark_section

    for filename in setup_files:
        source = (REPO_ROOT / "ch14" / filename).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split(
            "def benchmark_fn",
            maxsplit=1,
        )[0]
        assert "torch.inference_mode()" in setup_section
        assert "torch.no_grad()" not in setup_section

    for filename in verification_files:
        source = (REPO_ROOT / "ch14" / filename).read_text(encoding="utf-8")
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def teardown",
            maxsplit=1,
        )[0]
        assert "torch.inference_mode()" in capture_section
        assert "torch.no_grad()" not in capture_section


def test_ch14_compile_tools_use_inference_mode() -> None:
    paths = (
        "torch_compile_large_model.py",
        "torch_compiler_examples.py",
        "inspect_compiled_code.py",
    )

    for filename in paths:
        source = (REPO_ROOT / "ch14" / filename).read_text(encoding="utf-8")
        assert "torch.inference_mode()" in source
        assert "torch.no_grad()" not in source


def test_ch14_flash_attention_sdpa_bench_defers_output_clone_and_host_sync() -> None:
    source = (REPO_ROOT / "ch14" / "flash_attention_sdpa_bench.py").read_text(encoding="utf-8")
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]

    assert "float(output" not in benchmark_section
    assert ".detach().clone()" not in benchmark_section
    assert "self.output = output.detach()" in benchmark_section
    assert "output=self.output.detach().clone()" in capture_section


def test_ch14_training_large_model_defers_step_loss_sync() -> None:
    source = (REPO_ROOT / "ch14" / "training_large_model_1_5x.py").read_text(
        encoding="utf-8"
    )
    train_step_section = source.split("def training_step", maxsplit=1)[1].split(
        "def benchmark_training",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_training", maxsplit=1)[1].split(
        "def estimate_memory",
        maxsplit=1,
    )[0]

    assert "return loss.item()" not in train_step_section
    assert "return loss.detach()" in train_step_section
    assert benchmark_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "start.record()" in benchmark_section
    assert "end.record()" in benchmark_section
    assert "elapsed = start.elapsed_time(end) / 1000.0" in benchmark_section
    assert "time.perf_counter()" not in benchmark_section


def test_nanochat_base_train_defers_grad_norm_sync_until_logging() -> None:
    source = (
        REPO_ROOT / "labs" / "nanochat_fullstack" / "scripts" / "base_train.py"
    ).read_text(encoding="utf-8")
    loop_section = source.split("# single training step", maxsplit=1)[1].split(
        "# state update",
        maxsplit=1,
    )[0]
    step_section = loop_section.split("# logging", maxsplit=1)[0]
    logging_section = loop_section.split("# logging", maxsplit=1)[1]

    assert "grad_norm_tensor.item()" not in loop_section
    assert "grad_norm_tensor = None" in step_section
    assert "grad_norm_tensor = torch.nn.utils.clip_grad_norm_" in step_section
    assert "log_tensors = [train_loss.to(torch.float64)]" in logging_section
    assert "log_values = torch.stack(log_tensors).detach().cpu().tolist()" in logging_section
    assert "grad_norm = log_values[1] if grad_clip_enabled else 0.0" in logging_section
    assert "train_loss.item()" not in logging_section


def test_nanochat_chat_sft_batches_training_log_syncs() -> None:
    source = (
        REPO_ROOT / "labs" / "nanochat_fullstack" / "scripts" / "chat_sft.py"
    ).read_text(encoding="utf-8")
    eval_section = source.split("# evaluate the validation loss", maxsplit=1)[1].split(
        "if last_step:",
        maxsplit=1,
    )[0]
    logging_section = source.split("# logging", maxsplit=1)[1].split(
        "step += 1",
        maxsplit=1,
    )[0]

    assert eval_section.count("with torch.inference_mode(), autocast_ctx:") == 2
    assert "torch.no_grad()" not in eval_section
    assert "torch.stack((" in logging_section
    assert "train_loss.to(torch.float64)" in logging_section
    assert "num_tokens.to(torch.float64)" in logging_section
    assert ")).detach().cpu().tolist()" in logging_section
    assert "train_loss.item()" not in logging_section
    assert "num_tokens.item()" not in logging_section


def test_nanochat_tok_train_batches_token_byte_stat_syncs() -> None:
    source = (
        REPO_ROOT / "labs" / "nanochat_fullstack" / "scripts" / "tok_train.py"
    ).read_text(encoding="utf-8")
    report_section = source.split("# Log to report", maxsplit=1)[1]

    assert "token_byte_stats = torch.stack((" in report_section
    assert ")).tolist()" in report_section
    assert "token_bytes_min, token_bytes_max, token_bytes_mean, token_bytes_std = token_byte_stats" in report_section
    assert "token_bytes_nonzero.min().item()" not in report_section
    assert "token_bytes_nonzero.max().item()" not in report_section
    assert "token_bytes_nonzero.mean().item()" not in report_section
    assert "token_bytes_nonzero.std().item()" not in report_section


def test_ch19_memory_allocator_sizes_allocations_without_tensor_materialization() -> None:
    source = (
        REPO_ROOT / "ch19" / "memory_allocator_with_monitoring.py"
    ).read_text(encoding="utf-8")
    allocate_section = source.split("def allocate", maxsplit=1)[1].split(
        "for attempt in range",
        maxsplit=1,
    )[0]

    assert "from math import prod" in source
    assert "size_bytes = prod(shape) * torch.finfo(dtype).bits // 8" in allocate_section
    assert "torch.tensor(shape).prod().item()" not in allocate_section


def test_train_distributed_pipeline_defers_microbatch_loss_syncs() -> None:
    source = (REPO_ROOT / "labs" / "train_distributed" / "pipeline.py").read_text(
        encoding="utf-8"
    )
    schedule_section = source.split("class PipelineExperiment", maxsplit=1)[1]

    assert "def _finish_pipeline_loss" in source
    assert "loss_values.append(loss.detach())" in schedule_section
    assert "_finish_pipeline_loss(loss_values, n_micro)" in schedule_section
    assert "loss_total += loss.item()" not in schedule_section


def test_nanochat_chat_eval_batches_count_reductions() -> None:
    source = (
        REPO_ROOT / "labs" / "nanochat_fullstack" / "scripts" / "chat_eval.py"
    ).read_text(encoding="utf-8")

    assert "def _reduce_counts(num_passed, total, device):" in source
    assert "counts = torch.tensor([num_passed, total], dtype=torch.long, device=device)" in source
    assert "dist.all_reduce(counts, op=dist.ReduceOp.SUM)" in source
    assert "num_passed, total = counts.detach().cpu().tolist()" in source
    assert source.count("num_passed, total = _reduce_counts(num_passed, total, device)") == 2
    assert "with torch.inference_mode():" in source
    assert "with torch.no_grad():" not in source
    assert "num_passed_tensor = torch.tensor" not in source
    assert "total_tensor = torch.tensor" not in source
    assert "num_passed_tensor.item()" not in source
    assert "total_tensor.item()" not in source


def test_nanochat_chat_eval_batches_categorical_predictions() -> None:
    source = (
        REPO_ROOT / "labs" / "nanochat_fullstack" / "scripts" / "chat_eval.py"
    ).read_text(encoding="utf-8")
    categorical_section = source.split("def run_categorical_eval", maxsplit=1)[1].split(
        "# Aggregate results across all ranks",
        maxsplit=1,
    )[0]

    assert "predicted_choice_indices = torch.empty(len(conversations), dtype=torch.long, device=device)" in categorical_section
    assert "predicted_choice_indices[idx] = focus_logits.argmax(dim=-1)" in categorical_section
    assert "predicted_choice_indices = predicted_choice_indices.detach().cpu().tolist()" in categorical_section
    assert "argmax_letter_id = focus_logits.argmax(dim=-1).item()" not in categorical_section


def test_nanochat_chat_rl_batches_eval_and_rollout_syncs() -> None:
    source = (
        REPO_ROOT / "labs" / "nanochat_fullstack" / "scripts" / "chat_rl.py"
    ).read_text(encoding="utf-8")

    assert "eval_totals = torch.empty(device_batch_size + 1" in source
    assert "dist.all_reduce(eval_totals, op=dist.ReduceOp.SUM)" in source
    assert "eval_values = eval_totals.detach().cpu().tolist()" in source
    assert "passk_values = [value / num_records for value in eval_values[1:]]" in source
    assert "loss_item, reward_item = torch.stack((" in source
    assert "rewards_list.append(rewards_all.mean())" in source
    assert "mean_reward_tensor = torch.stack(rewards_list).mean()" in source
    assert "dist.all_reduce(summary, op=dist.ReduceOp.AVG)" in source
    assert "mean_reward, mean_sequence_length = summary.detach().cpu().tolist()" in source
    assert "num_records.item()" not in source
    assert "passk[k - 1].item()" not in source
    assert "loss.item()" not in source
    assert "rewards.mean().item()" not in source
    assert "rewards_all.mean().item()" not in source
    assert "mean_reward_tensor.item()" not in source
    assert "mean_sequence_length_tensor.item()" not in source


def test_ch16_tensor_parallel_attention_avoids_mask_completeness_sync() -> None:
    source = (REPO_ROOT / "ch16" / "inference_serving_multigpu.py").read_text(
        encoding="utf-8"
    )
    cached_attention_section = source.split("if kv_cache is None:", maxsplit=1)[
        1
    ].split("# Reshape and project", maxsplit=1)[0]

    assert "valid_mask.all().item()" not in cached_attention_section
    assert "has_padding = False" in cached_attention_section
    assert "if write_pos + delta_len < required_seq_len:" in cached_attention_section
    assert "if has_padding" in cached_attention_section
    assert "attn_k.zero_()" not in cached_attention_section
    assert "attn_v.zero_()" not in cached_attention_section
    assert "valid_mask.fill_(False)" in cached_attention_section


def test_ch16_inference_serving_tracks_packed_max_tokens_on_host() -> None:
    source = (REPO_ROOT / "ch16" / "inference_serving_multigpu.py").read_text(
        encoding="utf-8"
    )
    generate_batch_section = source.split("def generate_batch", maxsplit=1)[1].split(
        "needs_cache_fetch = any(",
        maxsplit=1,
    )[0]

    assert "max_tokens = 1" in generate_batch_section
    assert "seq_len = len(token_source)" in generate_batch_section
    assert "seq_len = 1" in generate_batch_section
    assert "self._token_workspace[pack_idx, 0] = int(state.generated_tokens[-1])" in generate_batch_section
    assert "max_tokens = max(max_tokens, seq_len)" in generate_batch_section
    assert "token_source = [state.generated_tokens[-1]]" not in generate_batch_section
    assert "lengths[:batch_size].max().item()" not in generate_batch_section
    assert ".max().item()" not in generate_batch_section


def test_ch16_inference_serving_reuses_sampled_token_buffers() -> None:
    source = (REPO_ROOT / "ch16" / "inference_serving_multigpu.py").read_text(
        encoding="utf-8"
    )
    init_section = source.split("self._temperature_workspace = torch.ones", maxsplit=1)[
        1
    ].split("self._last_token_lengths", maxsplit=1)[0]
    generate_batch_section = source.split("def generate_batch", maxsplit=1)[1].split(
        "def serve_loop",
        maxsplit=1,
    )[0]

    assert "self._sampled_token_workspace = torch.empty(" in init_section
    assert "self._sampled_token_host_workspace = torch.empty(" in init_section
    assert 'pin_memory=self.device.type == "cuda"' in init_section
    assert "sampled_tokens_2d = self._sampled_token_workspace[:batch_size, :]" in generate_batch_section
    assert "next_tokens_device = sampled_tokens_2d[:, 0]" in generate_batch_section
    assert "torch.multinomial(probs, num_samples=1, out=sampled_tokens_2d)" in generate_batch_section
    assert "generated_host = self._sampled_token_host_workspace[:batch_size]" in generate_batch_section
    assert "generated_host.copy_(next_tokens_device)" in generate_batch_section
    assert "generated = generated_host.tolist()" in generate_batch_section
    assert "torch.empty(batch_size, dtype=torch.long, device=probs.device)" not in generate_batch_section
    assert "torch.multinomial(probs, num_samples=1).squeeze(-1)" not in generate_batch_section
    assert "next_tokens_device.cpu()" not in generate_batch_section


def test_ch16_inference_serving_flushes_kv_views_without_stack() -> None:
    source = (REPO_ROOT / "ch16" / "inference_serving_multigpu.py").read_text(
        encoding="utf-8"
    )
    flush_section = source.split("def _flush_to_cache", maxsplit=1)[1].split(
        "if self.device.type == \"cuda\":",
        maxsplit=1,
    )[0]

    assert "key_tensor = attn_keys[:, pack_idx, head_slice, :num_tokens, :]" in flush_section
    assert "value_tensor = attn_values[:, pack_idx, head_slice, :num_tokens, :]" in flush_section
    assert "torch.stack(" not in flush_section
    assert "key_layers = []" not in flush_section


def test_ch16_blackwell_tensor_parallel_reuses_gather_buffers() -> None:
    source = (REPO_ROOT / "ch16" / "inference_optimizations_blackwell.py").read_text(
        encoding="utf-8"
    )
    init_section = source.split("class TensorParallelMultiGPU", maxsplit=1)[1].split(
        "def shard_kv_cache",
        maxsplit=1,
    )[0]
    forward_section = source.split("def forward(self, input_ids, kv_cache=None):", maxsplit=1)[1].split(
        "def benchmark_multigpu_tensor_parallel",
        maxsplit=1,
    )[0]

    assert "self._gathered_outputs = None" in init_section
    assert "self._final_output = None" in init_section
    assert "torch.cat(self._gathered_outputs, dim=-1, out=self._final_output)" in forward_section
    assert "final_output = torch.cat(gathered_outputs, dim=-1)" not in forward_section


def test_ch16_blackwell_inference_demo_uses_cuda_event_timing() -> None:
    source = (REPO_ROOT / "ch16" / "inference_optimizations_blackwell.py").read_text(
        encoding="utf-8"
    )
    helper_section = source.split("def _benchmark_cuda_latency_ms", maxsplit=1)[1].split(
        "# ============================================================================",
        maxsplit=1,
    )[0]
    demo_section = source.split("def compare_inference_methods", maxsplit=1)[1].split(
        "def benchmark_multigpu_tensor_parallel",
        maxsplit=1,
    )[0]

    assert helper_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "start.record()" in helper_section
    assert "end.record()" in helper_section
    assert "start.elapsed_time(end) / iterations" in helper_section
    assert demo_section.count("_benchmark_cuda_latency_ms(") == 3
    assert "time.time()" not in demo_section


def test_ch16_moe_feedforward_seeds_output_from_first_route() -> None:
    source = (REPO_ROOT / "ch16" / "moe_performance_benchmark.py").read_text(
        encoding="utf-8"
    )
    forward_section = source.split("class MoEFeedForward", maxsplit=1)[1].split(
        "class DenseFeedForward",
        maxsplit=1,
    )[0]

    assert "output = torch.empty_like(flat)" in forward_section
    assert "torch.zeros_like(flat)" not in forward_section
    assert "token_ids = (expert_ids == expert_id).nonzero(as_tuple=True)[0]" in forward_section
    assert "if token_ids.numel() == 0:" in forward_section
    assert "if k == 0:" in forward_section
    assert "output[token_ids] = weighted_out" in forward_section
    assert "output[token_ids] += weighted_out" in forward_section
    assert "mask.any()" not in forward_section


def test_ch16_moe_performance_benchmark_uses_cuda_event_timing() -> None:
    source = (REPO_ROOT / "ch16" / "moe_performance_benchmark.py").read_text(
        encoding="utf-8"
    )
    benchmark_section = source.split("def benchmark_model", maxsplit=1)[1].split(
        "def parse_args",
        maxsplit=1,
    )[0]
    cuda_section = benchmark_section.split('if input_ids.device.type == "cuda":', maxsplit=2)[
        2
    ].split("else:", maxsplit=1)[0]
    cpu_section = benchmark_section.split("else:", maxsplit=1)[1]

    assert benchmark_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "start_event.record()" in cuda_section
    assert "end_event.record()" in cuda_section
    assert "elapsed = start_event.elapsed_time(end_event) / 1000.0" in cuda_section
    assert "time.time()" not in cuda_section
    assert "time.time()" in cpu_section


def test_ch16_perplexity_eval_accumulates_loss_on_device() -> None:
    source = (REPO_ROOT / "ch16" / "perplexity_eval.py").read_text(encoding="utf-8")
    loop_section = source.split("with torch.inference_mode():", maxsplit=1)[1].split(
        "avg_loss =",
        maxsplit=1,
    )[0]

    assert "total_loss = torch.zeros((), device=device, dtype=torch.float32)" in source
    assert "token_ids_i64 = torch.tensor(tokens, device=device, dtype=torch.int64)" in source
    assert "token_ids_i32 = token_ids_i64.to(torch.int32)" in source
    assert "token_ids_i32.narrow(0, start, args.seq_len).unsqueeze(0)" in loop_section
    assert "token_ids_i64.narrow(0, start + 1, args.seq_len).unsqueeze(0)" in loop_section
    assert "torch.tensor(context" not in loop_section
    assert "torch.tensor(target" not in loop_section
    assert "loss.item()" not in loop_section
    assert "total_loss += loss.detach()" in loop_section
    assert "with torch.no_grad():" not in source
    assert "perplexity = math.exp(avg_loss)" in source


def test_ch16_block_sparse_bsr_build_uses_vectorized_metadata() -> None:
    from ch16.block_sparse_attention_utils import (
        build_block_sparse_pattern,
        build_bsr_from_block_mask,
        build_dense_attention_mask,
    )

    source = (REPO_ROOT / "ch16" / "block_sparse_attention_utils.py").read_text(
        encoding="utf-8"
    )
    pattern_section = source.split("def build_block_sparse_pattern", maxsplit=1)[1].split(
        "def build_dense_attention_mask", maxsplit=1
    )[0]
    dense_mask_section = source.split("def build_dense_attention_mask", maxsplit=1)[1].split(
        "def build_bsr_from_block_mask", maxsplit=1
    )[0]
    bsr_section = source.split("def build_bsr_from_block_mask", maxsplit=1)[1]

    for filename in ("baseline_flashinfer_block_sparse.py", "optimized_flashinfer_block_sparse.py"):
        benchmark_source = (REPO_ROOT / "ch16" / filename).read_text(encoding="utf-8")
        benchmark_section = benchmark_source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]
        assert "with torch.inference_mode():" in benchmark_section

    pattern = build_block_sparse_pattern(seq_len=16, block_size=4, window_blocks=1)
    expected_pattern = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [False, True, True, True],
            [False, False, True, True],
        ],
        dtype=torch.bool,
    )
    block_mask = torch.tensor(
        [
            [True, False, True],
            [False, True, True],
            [True, False, False],
        ]
    )
    indptr, indices, sparsity_ratio = build_bsr_from_block_mask(
        block_mask,
        device=torch.device("cpu"),
    )
    dense_mask = build_dense_attention_mask(
        block_mask,
        block_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    torch.testing.assert_close(pattern, expected_pattern)
    assert dense_mask.shape == (6, 6)
    assert dense_mask[0, 0] == 0.0
    assert torch.isneginf(dense_mask[0, 2])
    torch.testing.assert_close(indptr, torch.tensor([0, 2, 4, 5], dtype=torch.int32))
    torch.testing.assert_close(indices, torch.tensor([0, 2, 1, 2, 0], dtype=torch.int32))
    assert sparsity_ratio == 1.0 - (5.0 / 9.0)
    assert "row_ids = torch.arange(blocks).unsqueeze(1)" in pattern_section
    assert "col_ids = torch.arange(blocks).unsqueeze(0)" in pattern_section
    assert "for row in range(blocks)" not in pattern_section
    assert 'torch.full(block_mask.shape, float("-inf"), device=device, dtype=dtype)' in dense_mask_section
    assert "values.masked_fill_(block_mask.to(device=device, dtype=torch.bool), 0.0)" in dense_mask_section
    assert "values[:, None, :, None].expand(blocks, block_size, blocks, block_size).reshape(" in dense_mask_section
    assert "repeat_interleave(block_size" not in dense_mask_section
    assert "torch.tensor(float(\"-inf\")" not in dense_mask_section
    assert "torch.where(" not in dense_mask_section
    assert "torch.nonzero(mask, as_tuple=False)[:, 1]" in bsr_section
    assert "torch.cumsum(row_counts, dim=0, out=indptr_src[1:])" in bsr_section
    assert ".tolist()" not in bsr_section
    assert "block_mask.sum().item()" not in bsr_section


def test_ch16_synthetic_moe_benchmark_hoists_inference_mode() -> None:
    source = (REPO_ROOT / "ch16" / "synthetic_moe_inference_benchmark.py").read_text(
        encoding="utf-8"
    )
    benchmark_function = source.split("def benchmark_inference", maxsplit=1)[1].split(
        "def main", maxsplit=1
    )[0]

    assert benchmark_function.count("with torch.inference_mode():") == 3
    assert "with torch.no_grad():" not in benchmark_function
    assert "for _ in range(num_warmup):\n            if use_autocast:" in benchmark_function
    assert "for _ in range(count):\n                if use_autocast:" in benchmark_function
    assert benchmark_function.count("torch.cuda.Event(enable_timing=True)") == 2


def test_ch16_gpt_large_benchmark_uses_inference_mode() -> None:
    source = (REPO_ROOT / "ch16" / "gpt_large_benchmark.py").read_text(encoding="utf-8")
    benchmark_function = source.split("def benchmark_model", maxsplit=1)[1].split(
        "def validate_multi_gpu_equivalence",
        maxsplit=1,
    )[0]
    validation_function = source.split("def validate_multi_gpu_equivalence", maxsplit=1)[1].split(
        "def format_result",
        maxsplit=1,
    )[0]

    assert "@torch.inference_mode()" in source
    assert "@torch.no_grad()" not in source
    assert "with torch.inference_mode():" in validation_function
    assert "torch.no_grad()" not in benchmark_function
    assert "torch.no_grad()" not in validation_function


def test_ch15_moe_validation_batches_report_loss_reads() -> None:
    source = (REPO_ROOT / "ch15" / "moe_validation" / "moe_validation.py").read_text(
        encoding="utf-8"
    )
    stats_logger_section = source.split("class MoEStatsLogger", maxsplit=1)[1].split(
        "def _set_router_config",
        maxsplit=1,
    )[0]
    report_section = source.split("summary = moe_logger.summarize()", maxsplit=1)[1].split(
        "record = {",
        maxsplit=1,
    )[0]
    sweep_section = source.split("class MoeValidationSweep", maxsplit=1)[1].split(
        "def main",
        maxsplit=1,
    )[0]
    main_section = source.split("def main", maxsplit=1)[1]
    run_once_section = source.split("def _run_once", maxsplit=1)[1].split(
        "def run",
        maxsplit=1,
    )[0]

    assert "self._overflow_tensors.append(overflow_mask.detach().sum())" in stats_logger_section
    assert (
        "self._entropy_tensors.append(entropy_val.detach().to(dtype=torch.float32).reshape(()))"
        in stats_logger_section
    )
    assert "if torch.is_tensor(entropy_val):" in stats_logger_section
    assert "self.entropy.append(float(entropy_val))" in stats_logger_section
    assert "torch.stack(self._overflow_tensors).sum().detach().cpu().tolist()" in stats_logger_section
    assert "if valid.any()" not in stats_logger_section
    assert "overflow_mask.sum().item()" not in stats_logger_section
    assert "loss_values = loss_readback.detach().cpu().tolist()" in report_section
    assert "loss_values[1] / max(decode_loss_count, 1)" in report_section
    assert "decode_losses" not in report_section
    assert "sum(loss.item() for loss in decode_losses)" not in report_section
    assert "token_loss.item()" not in report_section
    assert "self._next_token_values: Optional[torch.Tensor] = None" in sweep_section
    assert "self._next_token_buffer: Optional[torch.Tensor] = None" in sweep_section
    assert "self._loss_readback: Optional[torch.Tensor] = None" in sweep_section
    assert "self._loss_readback = torch.empty(2, device=self.device, dtype=torch.float32)" in sweep_section
    assert "def _next_token_from_logits(self, logits: torch.Tensor) -> torch.Tensor" in sweep_section
    assert "torch.max(logits_last, dim=-1, keepdim=True, out=(self._next_token_values, self._next_token_buffer))" in sweep_section
    assert "with torch.inference_mode():" in run_once_section
    assert "with torch.no_grad():" not in run_once_section
    assert "loss_readback.zero_()" in run_once_section
    assert "loss_readback[0].copy_(token_loss.detach())" in run_once_section
    assert "loss_readback[1].add_(step_loss.detach())" in run_once_section
    assert "decode_loss_count += 1" in run_once_section
    assert "decode_losses" not in run_once_section
    assert "seed_tokens = self._next_token_from_logits(logits[:, -1, :])" in run_once_section
    assert "seed_tokens = self._next_token_from_logits(decode_logits[:, -1, :])" in run_once_section
    assert "torch.argmax(" not in run_once_section
    assert "config_payload = asdict(cfg)" in main_section
    assert "if isinstance(value, torch.dtype):" in main_section
    assert '"config": config_payload' in main_section
    assert '"config": asdict(cfg)' not in main_section


def test_ch15_expert_parallelism_batches_expert_metadata_reads() -> None:
    source = (REPO_ROOT / "ch15" / "expert_parallelism.py").read_text(encoding="utf-8")
    local_section = source.split("def forward_local", maxsplit=1)[1].split(
        "def forward_distributed",
        maxsplit=1,
    )[0]
    distributed_section = source.split("def forward_distributed", maxsplit=1)[1].split(
        "def _run_local",
        maxsplit=1,
    )[0]

    for section in (local_section, distributed_section):
        assert "overflow_flags = [bool(flag) for flag in mask_overflow.detach().cpu().tolist()]" in section
        assert "eid.item()" not in section
        assert "mask.any()" not in section
    assert "unique_expert_ids = [int(eid) for eid in torch.unique(expert_ids).detach().cpu().tolist()]" in local_section
    assert "for eid_int in [int(eid) for eid in torch.unique(recv_ids).detach().cpu().tolist()]" in distributed_section


def test_ch15_parallel_demos_use_inference_mode() -> None:
    for relative in (
        "ch15/tensor_parallel_demo.py",
        "ch15/pipeline_parallel_demo.py",
        "ch15/expert_parallelism.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "with torch.inference_mode():" in source
        assert "with torch.no_grad():" not in source


def test_ch17_dynamic_routing_defers_output_tensor_outside_hot_loop() -> None:
    source = (REPO_ROOT / "ch17" / "baseline_dynamic_routing.py").read_text(encoding="utf-8")
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def get_config", maxsplit=1
    )[0]

    assert "torch.tensor(" not in benchmark_section
    assert "self._output_values = [float(served), float(rejects), float(offloaded)]" in benchmark_section
    assert "self.output = torch.tensor(self._output_values, dtype=torch.float32)" in capture_section


def test_ch17_moe_router_remote_buffers_avoid_zero_fill() -> None:
    for relative in (
        "ch17/baseline_moe_router_uniform.py",
        "ch17/optimized_moe_router_uniform_topology.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split("def benchmark_fn", maxsplit=1)[0]
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]

        assert "self._remote_buf_a = torch.empty(" in setup_section
        assert "self._remote_buf_b = torch.empty(" in setup_section
        assert "self._remote_buf_a = torch.zeros(" not in setup_section
        assert "self._remote_buf_b = torch.zeros(" not in setup_section
        if relative.endswith("optimized_moe_router_uniform_topology.py"):
            assert "spill.any()" not in setup_section
            assert "expert_ids = torch.where(spill, spill_ids, expert_ids)" in setup_section
        assert "with torch.inference_mode():" in setup_section
        assert "with torch.inference_mode():" in benchmark_section
        assert "torch.index_select(flat, 0, self._remote_idx, out=self._remote_buf_a[:, : self.hidden_size])" in benchmark_section
        assert "self._remote_buf_b.copy_(self._remote_buf_a)" in benchmark_section
        assert "self._remote_buf_a.copy_(self._remote_buf_b)" in benchmark_section


def test_ch17_dynamic_routing_vectorized_path_reuses_masks() -> None:
    source = (REPO_ROOT / "ch17" / "baseline_dynamic_routing.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def _make_metrics",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert "self._queue_lengths = torch.empty(self.batch_size, dtype=torch.int32)" in setup_section
    assert "self._queue_lengths = torch.zeros(self.batch_size, dtype=torch.int32)" not in setup_section
    assert "self._remaining_lengths = torch.empty_like(self._prompt_lengths)" in setup_section
    assert "self._long_prefill = torch.empty_like(self._priorities, dtype=torch.bool)" in setup_section
    assert "self._served_offload_mask = torch.empty_like(self._long_prefill)" in setup_section
    assert "self._count_values = torch.empty(2, dtype=torch.int64, device=self._priorities.device)" in setup_section
    assert "self._queue_length_rows = self._queue_length_table.tolist()" in setup_section
    assert "long_prefill = (" not in benchmark_section
    assert "capacity = self._queue_lengths" not in benchmark_section
    assert "offload_mask = long_prefill & capacity" not in benchmark_section
    assert "admit_mask = torch.ones_like" not in benchmark_section
    assert "(~admit_mask)" not in benchmark_section
    assert "torch.sub(" in benchmark_section and "out=self._remaining_lengths" in benchmark_section
    assert "out=self._long_prefill" in benchmark_section
    assert "out=self._capacity_mask" in benchmark_section
    assert "out=self._offload_mask" in benchmark_section
    assert "torch.ne(self._priorities, 0, out=self._admit_mask)" in benchmark_section
    assert "self._admit_mask.fill_(True)" in benchmark_section
    assert "out=self._served_offload_mask" in benchmark_section
    assert "queue_depth = queue_lengths_host[idx % self.batch_size]" in benchmark_section
    assert "queue_lengths[idx % queue_lengths.numel()].item()" not in benchmark_section
    timed_section = benchmark_section.split("elapsed_ms = self._record_stop(start)", maxsplit=1)[0]
    vectorized_timed_section = timed_section.split(
        "        else:\n            # Python loop-based routing",
        maxsplit=1,
    )[0]
    post_timing_section = benchmark_section.split("elapsed_ms = self._record_stop(start)", maxsplit=1)[1]
    assert "rejects_tensor =" not in benchmark_section
    assert "offloaded_tensor =" not in benchmark_section
    assert "torch.sum(self._admit_mask, dim=(), dtype=torch.int64, out=self._count_values[0])" in vectorized_timed_section
    assert "self._count_values[0].neg_().add_(self.batch_size)" in vectorized_timed_section
    assert "torch.sum(self._served_offload_mask, dim=(), dtype=torch.int64, out=self._count_values[1])" in vectorized_timed_section
    assert "count_values_ready = True" in vectorized_timed_section
    assert ".item()" not in vectorized_timed_section
    assert "self._count_values[0].copy_(" not in post_timing_section
    assert "self._count_values[1].copy_(" not in post_timing_section
    assert "rejects_value, offloaded_value = self._count_values.tolist()" in post_timing_section
    assert "torch.stack(" not in benchmark_section
    assert "rejects = int(rejects_value)" in post_timing_section
    assert "offloaded = int(offloaded_value)" in post_timing_section
    assert "rejects_tensor.item()" not in post_timing_section
    assert "offloaded_tensor.item()" not in post_timing_section


def test_hf_decoder_cache_defers_verification_copy_outside_hot_loop() -> None:
    source = (
        REPO_ROOT / "core" / "benchmark" / "hf_decoder_cache_benchmark.py"
    ).read_text(encoding="utf-8")
    baseline_source = (
        REPO_ROOT / "labs" / "decode_optimization" / "baseline_decode_hf_cache.py"
    ).read_text(encoding="utf-8")
    optimized_source = (
        REPO_ROOT / "labs" / "decode_optimization" / "optimized_decode_hf_cache.py"
    ).read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def _prepare_iteration",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown",
        maxsplit=1,
    )[0]

    assert "self._verification_token = torch.empty(" in setup_section
    assert "self._prefill_token_buffer = torch.empty(" in setup_section
    assert "self._decode_next_token_buffer = torch.empty(" in setup_section
    assert "self._generated_tokens_buffer = torch.empty(" in setup_section
    assert "self._done_mask_buffer = torch.empty(" in setup_section
    assert "self._prompt_pos = torch.arange(" in setup_section
    assert "def _next_token_from_logits" in source
    assert "torch.max(logits_last, dim=-1, out=(values, self._decode_next_token_buffer))" in source
    assert "def _update_done_mask" in source
    assert "verification_token = next_token.detach().to(torch.int32).clone()" not in benchmark_section
    assert "prompt_pos = torch.arange(" not in benchmark_section
    assert "self._prefill_token_buffer.copy_(next_token)" in benchmark_section
    assert "verification_token = self._prefill_token_buffer.detach()" in benchmark_section
    assert "torch.argmax(" not in benchmark_section
    assert "torch.where(" not in source
    assert "torch.stack(generated" not in source
    assert "generated.append" not in source
    assert "cache_position=self._prompt_pos" in benchmark_section
    assert "self._verification_token.copy_(self.output)" in capture_section
    assert "self.output = self._verification_token" in capture_section
    assert 'eos_sync_mode="blocking"' in baseline_source
    assert 'eos_sync_mode="async_streamed"' in optimized_source


def test_continuous_batching_reuses_state_buffers() -> None:
    source = (REPO_ROOT / "core" / "utils" / "continuous_batching.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert "self.state_buffers: List[torch.Tensor] = []" in source
    assert "self.state_buffers.append(torch.empty_like(samples))" in setup_section
    assert "state = samples.clone()" not in benchmark_section
    assert "state = self.state_buffers[idx]" in benchmark_section
    assert "state.copy_(samples)" in benchmark_section


def test_ch06_roofline_ilp_defers_verification_tensors_outside_hot_loop() -> None:
    source = (REPO_ROOT / "ch06" / "roofline_analysis_ilp.py").read_text(encoding="utf-8")
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]

    assert "torch.tensor(" not in benchmark_section
    assert "self._output_values = [" in benchmark_section
    assert "self._ridge_point_value = self.analyzer.ridge_point" in benchmark_section
    assert "self.output = torch.tensor(self._output_values, dtype=torch.float32)" in capture_section
    assert "self._verify_input = torch.tensor([self._ridge_point_value], dtype=torch.float32)" in capture_section


def test_ch06_warp_divergence_baseline_reuses_result_buffer() -> None:
    source = (REPO_ROOT / "ch06" / "baseline_warp_divergence_ilp.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert "self._output_buffer = torch.empty_like(self.input)" in setup_section
    assert "result = self.input.clone()" not in benchmark_section
    assert "result = self._output_buffer" in benchmark_section
    assert "result.copy_(self.input)" in benchmark_section
    assert "positive = result[mask]" in benchmark_section
    assert "negative = result[~mask]" in benchmark_section


def test_ch13_precisionfp8_defers_verification_forwards_and_casts_outside_hot_loop() -> None:
    training_pair = (
        "baseline_precisionfp8.py",
        "optimized_precisionfp8.py",
        "optimized_precisionfp8_rowwise.py",
        "optimized_precisionfp8_rowwise_gw_hp.py",
    )
    forward_pair = (
        ("baseline_precisionfp8_pad_inner.py", "self.output = benchmark_out"),
        ("optimized_precisionfp8_pad_inner.py", "self.output = benchmark_out"),
        ("baseline_precisionfp8_pad_inner_matmul.py", "self.output = out"),
        ("optimized_precisionfp8_pad_inner_matmul.py", "self.output = out"),
    )

    for name in training_pair:
        source = (REPO_ROOT / "ch13" / name).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def teardown", maxsplit=1
        )[0]

        assert "verify_out = self.model(self._verify_input_fp16)" not in benchmark_section
        assert ".detach().float().clone()" not in benchmark_section
        assert "verify_out = self.model(self._verify_input_fp16)" in capture_section
        assert "with torch.inference_mode():" in capture_section
        assert "with torch.no_grad():" not in capture_section
        assert "output=self.output.detach().float().clone()" in capture_section

    for name, expected_assignment in forward_pair:
        source = (REPO_ROOT / "ch13" / name).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def teardown", maxsplit=1
        )[0]

        assert ".detach().float().clone()" not in benchmark_section
        assert expected_assignment in benchmark_section
        assert "output=self.output.detach().float().clone()" in capture_section


def test_ch13_mlp_benchmarks_use_inplace_relu_modules() -> None:
    for name in (
        "baseline_precisionfp8.py",
        "optimized_precisionfp8.py",
        "optimized_precisionfp8_rowwise.py",
        "optimized_precisionfp8_rowwise_gw_hp.py",
        "baseline_precisionfp8_pad_inner.py",
        "optimized_precisionfp8_pad_inner.py",
        "baseline_dataloader_default.py",
        "optimized_dataloader_default.py",
        "baseline_autograd_standard.py",
        "optimized_autograd_standard.py",
        "baseline_memory_profiling.py",
        "optimized_memory_profiling.py",
        "baseline_quantization.py",
        "optimized_quantization.py",
        "baseline_torchao_quantization.py",
        "optimized_torchao_quantization.py",
        "fsdp_example.py",
        "train.py",
        "memory_profiling.py",
    ):
        source = (REPO_ROOT / "ch13" / name).read_text(encoding="utf-8")

        assert "nn.ReLU(inplace=True)" in source
        assert "nn.ReLU()" not in source

    optimized_quantization = (REPO_ROOT / "ch13" / "optimized_quantization.py").read_text(encoding="utf-8")
    int8_forward = optimized_quantization.split("def forward(self, x: torch.Tensor)", maxsplit=1)[1].split(
        "class OptimizedQuantizationBenchmark",
        maxsplit=1,
    )[0]
    assert "torch.relu_(x)" in int8_forward
    assert "torch.relu(x)" not in int8_forward


def test_ch13_training_benchmarks_defer_verification_materialization_outside_hot_loop() -> None:
    for name in (
        "baseline_training_standard.py",
        "optimized_training_standard.py",
        "baseline_training_speed.py",
    ):
        source = (REPO_ROOT / "ch13" / name).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def teardown", maxsplit=1
        )[0]

        assert "logits[:1, :1, :8].detach().float().clone()" not in benchmark_section
        assert "self.output = None" in benchmark_section
        assert "verify_logits = self.model(self.input_ids)" in capture_section
        assert "self.output = verify_logits[:1, :1, :8].detach().float().clone()" in capture_section

    optimized_speed = (REPO_ROOT / "ch13" / "optimized_training_speed.py").read_text(encoding="utf-8")
    optimized_benchmark = optimized_speed.split("def benchmark_fn", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]

    assert "output_buffer" not in optimized_speed
    assert "logits[:1, :1, :8]" not in optimized_speed
    assert "self.output = None" in optimized_benchmark


def test_ch13_training_models_reuse_position_id_buffers() -> None:
    sources = [
        REPO_ROOT / "ch13" / "training_speed_common.py",
        REPO_ROOT / "ch13" / "baseline_training_standard.py",
        REPO_ROOT / "ch13" / "optimized_training_standard.py",
    ]

    for path in sources:
        source = path.read_text(encoding="utf-8")
        forward_section = source.split("def forward", maxsplit=1)[1].split(
            "return",
            maxsplit=1,
        )[0]

        assert 'self.register_buffer(\n            "_position_ids",' in source
        assert "torch.arange(seq_len, device=input_ids.device)" not in forward_section
        assert "pos_ids = self._position_ids[:, :seq_len].expand(batch_size, -1)" in forward_section


def test_ch04_multi_node_transformer_reuses_position_id_buffer() -> None:
    source = (REPO_ROOT / "ch04" / "multi_node_blackwell.py").read_text(
        encoding="utf-8"
    )
    forward_section = source.split("class MultiNodeTransformer", maxsplit=1)[1].split(
        "def create_multigpu_device_mesh",
        maxsplit=1,
    )[0]

    assert 'self.register_buffer(\n            "_position_ids",' in source
    assert "torch.arange(T, device=input_ids.device)" not in forward_section
    assert "pos = self._position_ids[:, :T]" in forward_section


def test_ch04_multi_node_training_defers_repeated_loss_syncs() -> None:
    source = (REPO_ROOT / "ch04" / "multi_node_blackwell.py").read_text(
        encoding="utf-8"
    )
    train_section = source.split("def train_multi_node", maxsplit=1)[1].split(
        "# ============================================================================",
        maxsplit=1,
    )[0]

    assert "loss.item()" not in train_section
    assert "epoch_loss_tensors = []" in train_section
    assert "stats['losses'].append(loss_value)" in train_section
    assert "float(torch.stack(epoch_loss_tensors).sum())" in train_section


def test_ch13_expert_parallel_batches_recv_split_materialization() -> None:
    source = (REPO_ROOT / "ch13" / "expert_parallel_common.py").read_text(
        encoding="utf-8"
    )
    split_section = source.split("def gather_recv_splits", maxsplit=1)[1].split(
        "def pack_tokens",
        maxsplit=1,
    )[0]

    assert "recv_counts = torch.stack(gathered, dim=0)[:, rank]" in split_section
    assert "recv_counts.detach().cpu().tolist()" in split_section
    assert "gathered[src][rank].item()" not in split_section


def test_ch13_sequence_parallel_surrogate_reuses_full_sequence_buffer() -> None:
    from ch13.baseline_sequence_parallel_multigpu import _replicate_sequence_shard

    source = (REPO_ROOT / "ch13" / "baseline_sequence_parallel_multigpu.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert "self._full_sequence = torch.empty(" in setup_section
    assert "_replicate_sequence_shard(out_partial, self._world_size, self._full_sequence)" in benchmark_section
    assert "torch.cat([out_partial] * self._world_size" not in benchmark_section
    assert "full_sequence = torch.cat([out_partial]" not in benchmark_section
    assert "full_sequence = self._norms[layer_idx](self._full_sequence)" in benchmark_section

    out_partial = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
    full_sequence = torch.empty(2, 6, 4)
    result = _replicate_sequence_shard(out_partial, 2, full_sequence)

    assert result.data_ptr() == full_sequence.data_ptr()
    torch.testing.assert_close(
        result,
        torch.cat([out_partial, out_partial], dim=1),
    )


def test_ch13_sequence_parallel_worker_reuses_full_sequence_buffer() -> None:
    source = (REPO_ROOT / "ch13" / "sequence_parallel_benchmark_common.py").read_text(
        encoding="utf-8"
    )
    run_section = source.split("def run_sequence_parallel", maxsplit=1)[1]
    step_section = run_section.split("def _step", maxsplit=1)[1].split(
        "for _ in range(max(warmup, 0)):",
        maxsplit=1,
    )[0]

    assert "full_sequence_buf = torch.empty(" in run_section
    assert "torch.cat(gather_buf, dim=1, out=full_sequence_buf)" in step_section
    assert "full_sequence = torch.cat(gather_buf, dim=1)" not in step_section


def test_fp8_demo_and_moe_lab_defer_verification_clones_outside_hot_loop() -> None:
    perchannel_source = (REPO_ROOT / "ch13" / "fp8_perchannel_demo.py").read_text(encoding="utf-8")
    perchannel_stats = perchannel_source.split("def get_quantization_stats", maxsplit=1)[1].split(
        "#============================================================================",
        maxsplit=1,
    )[0]
    perchannel_accuracy = perchannel_source.split("def measure_accuracy", maxsplit=1)[1].split(
        "def measure_throughput",
        maxsplit=1,
    )[0]
    perchannel_benchmark = perchannel_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    perchannel_capture = perchannel_source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]

    assert "float(output.detach().sum())" not in perchannel_benchmark
    assert ".detach().float().clone()" not in perchannel_benchmark
    assert "self.output = output" in perchannel_benchmark
    assert "output=self.output.detach().float().clone()" in perchannel_capture
    assert "torch.stack(" in perchannel_stats
    assert "self.input_amax_history.mean().item()" not in perchannel_stats
    assert "self.amax_counter.item()" not in perchannel_stats
    assert "pt_error_value, pc_error_value = torch.stack((pt_error, pc_error)).tolist()" in perchannel_accuracy
    assert "pt_error.item()" not in perchannel_accuracy
    assert "pc_error.item()" not in perchannel_accuracy

    moe_source = (
        REPO_ROOT / "labs" / "moe_optimization_journey" / "level6_native_fp8.py"
    ).read_text(encoding="utf-8")
    moe_setup = moe_source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    moe_benchmark = moe_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    moe_capture = moe_source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def get_extra_metrics", maxsplit=1
    )[0]

    assert ".detach().float().clone()" not in moe_benchmark
    assert "expert_indices_cpu = torch.randint(0, E, (batch_seq, K))" in moe_setup
    assert "torch.bincount(expert_indices_cpu.view(-1), minlength=E).tolist()" in moe_setup
    assert "torch.bincount(sorted_expert_ids, minlength=E).tolist()" not in moe_setup
    assert "sorted_order = torch.argsort(flat_idx, stable=True)" in moe_setup
    assert "self.sorted_order" not in moe_setup
    assert "self.expert_indices" not in moe_setup
    assert "self.expert_weights" not in moe_setup
    assert "torch.arange(batch_seq, device=self.device).repeat_interleave(K)" not in moe_setup
    assert "torch.arange(batch_seq * K, device=self.device, dtype=torch.int64)" in moe_setup
    assert 'expanded_token_indices.div_(K, rounding_mode="floor")' in moe_setup
    assert "self._sorted_token_indices = expanded_token_indices.index_select(0, sorted_order)" in moe_setup
    assert "self._sorted_weights = expert_weights.view(-1).index_select(0, sorted_order)" in moe_setup
    assert "self._sorted_tokens = torch.empty(" in moe_setup
    assert "self._tokens_fp8_buffer = torch.empty(" in moe_setup
    assert "self._hidden_fp8_buffer = torch.empty(" in moe_setup
    assert "x.repeat_interleave(self.TOP_K" not in moe_benchmark
    assert "self.expert_weights.view(-1)[self.sorted_order]" not in moe_benchmark
    assert "torch.index_select(x, 0, self._sorted_token_indices, out=self._sorted_tokens)" in moe_benchmark
    assert ".to(torch.float8_e4m3fn)" not in moe_benchmark
    assert "tokens_fp8_slice.copy_(tokens_e)" in moe_benchmark
    assert "F.silu(gate, inplace=True)" in moe_benchmark
    assert "gate.mul_(up)" in moe_benchmark
    assert "hidden_fp8_slice.copy_(gate)" in moe_benchmark
    assert "expert_out.mul_(weights_e)" in moe_benchmark
    assert "output[token_slice].copy_(expert_out)" in moe_benchmark
    assert "expert_out * weights_e" not in moe_benchmark
    assert "self.output = output[:1, : min(8, output.shape[1])]" in moe_benchmark
    assert "self._payload_param_count = int(" in moe_source
    assert "output=self.output.detach().float().clone()" in moe_capture


def test_moe_pad_quant_vectorized_router_reuses_topk_token_ids() -> None:
    source = (
        REPO_ROOT / "labs" / "moe_optimization_journey" / "optimized_moe_pad_quant.py"
    ).read_text(encoding="utf-8")
    vectorized_router = source.split("def _vectorized_forward_grouped", maxsplit=1)[1].split(
        "class OptimizedMoEPadQuantBenchmark", maxsplit=1
    )[0]
    install_section = source.split("def _install_vectorized_router", maxsplit=1)[1].split(
        "def setup", maxsplit=1
    )[0]

    assert "def _flat_topk_token_ids" in source
    assert "def _cached_topk_token_ids" in source
    assert 'token_ids.div_(top_k, rounding_mode="floor")' in source
    assert "def _dispatch_slot_buffer" in source
    assert "x.repeat_interleave(top_k" not in vectorized_router
    assert "token_ids = _cached_topk_token_ids(self, batch_seq, top_k, x.device)" in vectorized_router
    assert "rep_x = x.index_select(0, token_ids)" in vectorized_router
    assert "torch.zeros(" not in vectorized_router
    assert "padded = _dispatch_slot_buffer(" in vectorized_router
    assert "padded.zero_()" not in vectorized_router
    assert "Every row selected by `slots` is overwritten" in vectorized_router
    assert "module._dispatch_token_ids = None" in install_section
    assert "module._dispatch_padded = None" in install_section

    from labs.moe_optimization_journey.optimized_moe_pad_quant import (
        _dispatch_slot_buffer,
        _flat_topk_token_ids,
    )

    torch.testing.assert_close(
        _flat_topk_token_ids(3, 1, torch.device("cpu")),
        torch.tensor([0, 1, 2], dtype=torch.int64),
    )
    torch.testing.assert_close(
        _flat_topk_token_ids(3, 2, torch.device("cpu")),
        torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),
    )
    holder = SimpleNamespace()
    first = _dispatch_slot_buffer(
        holder,
        rows=4,
        hidden=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second = _dispatch_slot_buffer(
        holder,
        rows=4,
        hidden=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    resized = _dispatch_slot_buffer(
        holder,
        rows=5,
        hidden=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert first.data_ptr() == second.data_ptr()
    assert resized.data_ptr() != first.data_ptr()


def test_ch13_fp8_benchmarks_defer_unused_syncs_and_output_clones() -> None:
    targets = {
        "fp8_perchannel_bench.py": "self.output = output.detach()",
        "fp8_static_demo.py": "self.output = output.detach()",
        "optimized_precisionfp8_te.py": "self.output = self.output_buffer.detach()",
    }

    for name, output_assignment in targets.items():
        source = (REPO_ROOT / "ch13" / name).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def teardown", maxsplit=1
        )[0]

        assert ".detach().clone()" not in benchmark_section
        assert "float(output" not in benchmark_section
        assert "float(self.output" not in benchmark_section
        assert output_assignment in benchmark_section
        assert "output=self.output.detach().clone()" in capture_section


def test_ch13_static_fp8_calibration_defers_amax_scalar_reads() -> None:
    targets = ("fp8_static_demo.py", "optimized_fp8_static.py")

    for name in targets:
        source = (REPO_ROOT / "ch13" / name).read_text(encoding="utf-8")
        stats_section = source.split("class CalibrationStats", maxsplit=1)[1].split(
            "class StaticFP8Linear",
            maxsplit=1,
        )[0]

        assert "self._amax_tensors.append(tensor.detach().abs().amax())" in stats_section
        assert "_amax_materialize_buffer: Optional[torch.Tensor]" in stats_section
        assert "_amax_materialize_host_buffer: Optional[torch.Tensor]" in stats_section
        assert "def _materialize_buffers(" in stats_section
        assert "value_slice = value_buffer[:count]" in stats_section
        assert "value_slice[idx].copy_(value)" in stats_section
        assert "host_slice.copy_(value_slice)" in stats_section
        assert "values = host_slice.tolist()" in stats_section
        assert "torch.stack(self._amax_tensors).detach().cpu().tolist()" not in stats_section
        assert "tensor.abs().max().item()" not in stats_section
        assert "self.running_amax = max(self.running_amax, current_amax)" not in stats_section

    demo_source = (REPO_ROOT / "ch13" / "fp8_static_demo.py").read_text(encoding="utf-8")
    scale_section = demo_source.split("def get_all_scales", maxsplit=1)[1].split(
        "#============================================================================",
        maxsplit=1,
    )[0]
    info_section = demo_source.split("def _calibration_info_list", maxsplit=1)[1].split(
        "class StaticFP8Model",
        maxsplit=1,
    )[0]

    assert "self._scale_values: Optional[torch.Tensor] = None" in demo_source
    assert "self._scale_values_host: Optional[torch.Tensor] = None" in demo_source
    assert "scale_slice = self._scale_values[:count]" in scale_section
    assert "scale_slice[2 * idx].copy_(layer.input_scale)" in scale_section
    assert "scale_slice[2 * idx + 1].copy_(layer.weight_scale)" in scale_section
    assert "scale_host.copy_(scale_slice)" in scale_section
    assert "scale_values = scale_host.tolist()" in scale_section
    assert "scale_values = torch.stack(" not in scale_section
    assert "layer.input_scale.item()" not in scale_section
    assert "self.is_calibrated.item()" not in info_section
    assert "self._calibration_info_host: Optional[torch.Tensor] = None" in demo_source
    assert "def _calibration_info_list(self)" in demo_source
    assert "values[0].copy_(self.is_calibrated)" in info_section
    assert "values[1].copy_(self.input_scale)" in info_section
    assert "values[2].copy_(self.weight_scale)" in info_section
    assert "self._calibration_info_host.copy_(values)" in info_section
    assert "is_calibrated, input_scale, weight_scale = self._calibration_info_list()" in info_section
    assert "is_calibrated, input_scale, weight_scale = torch.stack(" not in info_section


def test_ch13_optimized_static_fp8_reuses_activation_quant_buffers() -> None:
    source = (REPO_ROOT / "ch13" / "optimized_fp8_static.py").read_text(encoding="utf-8")
    forward_section = source.split("def forward(self, x: torch.Tensor)", maxsplit=1)[1].split(
        "#============================================================================",
        maxsplit=1,
    )[0]

    assert '"_input_scaled_buffer"' in source
    assert '"_input_fp8_buffer"' in source
    assert "def _activation_buffers(self, x_2d: torch.Tensor)" in source
    assert "torch.div(x_2d, self.input_scale, out=input_scaled)" in forward_section
    assert "x_fp8.copy_(input_scaled)" in forward_section
    assert "(x_2d / self.input_scale).to(torch.float8_e4m3fn)" not in forward_section


def test_ch13_precisionmixed_and_kv_cache_defer_verification_clones_outside_hot_loop() -> None:
    precision_targets = {
        "baseline_precisionmixed.py": "output=self.output.detach().clone()",
        "optimized_precisionmixed.py": "output=self.output.detach().float().clone()",
    }
    kv_targets = {
        "baseline_kv_cache_naive.py": "self.output = token.detach()",
        "optimized_kv_cache_naive.py": "self.output = hidden.detach()",
        "optimized_kv_cache_naive_flash_blockwise.py": "self.output = hidden[:, -1:, :].detach()",
        "optimized_kv_cache_naive_pool.py": "self.output = hidden.detach()",
    }

    for name, capture_materialization in precision_targets.items():
        source = (REPO_ROOT / "ch13" / name).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def teardown", maxsplit=1
        )[0]

        assert ".detach().clone()" not in benchmark_section
        assert "self.output = outputs.detach()" in benchmark_section
        assert capture_materialization in capture_section

    for name, output_assignment in kv_targets.items():
        source = (REPO_ROOT / "ch13" / name).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def teardown", maxsplit=1
        )[0]

        assert ".detach().clone()" not in benchmark_section
        assert output_assignment in benchmark_section
        assert "output=self.output.float()" in capture_section

    flash_source = (
        REPO_ROOT / "ch13" / "optimized_kv_cache_naive_flash_blockwise.py"
    ).read_text(encoding="utf-8")
    assert "batch_size=self.batch_size" in flash_source
    assert "for batch_idx in range(batch_size)" not in flash_source
    assert "k_block = k.permute(2, 0, 1, 3).contiguous()" in flash_source
    assert "kv_cache.append_block(request_id, layer_idx, k_block, v_block, cache_pos)" in flash_source
    assert "torch.cat([cached_k, k]" not in flash_source
    assert "torch.cat([cached_v, v]" not in flash_source
    assert "layer.configure_kv_workspace(" in flash_source


def test_flash_blockwise_attention_reuses_workspace_for_cached_kv() -> None:
    from ch13.optimized_kv_cache_naive import PagedKVCache
    from ch13.optimized_kv_cache_naive_flash_blockwise import FlashBlockwiseAttentionLayer

    layer = FlashBlockwiseAttentionLayer(hidden_dim=6, num_heads=2, head_dim=3, dtype=torch.float32)
    layer.configure_kv_workspace(
        max_seq_len=4,
        batch_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    kv_cache = PagedKVCache(
        page_size=4,
        batch_size=1,
        num_layers=1,
        num_heads=2,
        head_dim=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    kv_cache.allocate("req", 4)
    cached_k_block = torch.arange(12, dtype=torch.float32).view(2, 1, 2, 3)
    cached_v_block = torch.arange(12, 24, dtype=torch.float32).view(2, 1, 2, 3)
    kv_cache.append_block("req", 0, cached_k_block, cached_v_block, 0)
    k = torch.arange(24, 30, dtype=torch.float32).view(1, 2, 1, 3)
    v = torch.arange(30, 36, dtype=torch.float32).view(1, 2, 1, 3)

    actual_k, actual_v = layer._cached_attention_inputs(k, v, kv_cache, "req", 0, cache_pos=2)

    expected_k = torch.cat([cached_k_block.permute(1, 2, 0, 3), k], dim=2)
    expected_v = torch.cat([cached_v_block.permute(1, 2, 0, 3), v], dim=2)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)
    assert actual_k.is_contiguous()
    assert actual_v.is_contiguous()
    assert actual_k.data_ptr() == layer._workspace_k.data_ptr()
    assert actual_v.data_ptr() == layer._workspace_v.data_ptr()


def test_ch13_paged_kv_cache_releases_slabs_without_zero_fill() -> None:
    source = (REPO_ROOT / "ch13" / "optimized_kv_cache_naive.py").read_text(encoding="utf-8")
    acquire_section = source.split("def _acquire_buffer", maxsplit=1)[1].split(
        "def _release_buffer", maxsplit=1
    )[0]
    release_section = source.split("def _release_buffer", maxsplit=1)[1].split(
        "def allocate", maxsplit=1
    )[0]

    assert "torch.empty(" in acquire_section
    assert "torch.empty_like(k_buf)" in acquire_section
    assert "torch.zeros(" not in acquire_section
    assert ".zero_()" not in release_section
    assert "self._empty" in source

    from ch13.optimized_kv_cache_naive import PagedKVCache

    cache = PagedKVCache(
        page_size=4,
        batch_size=1,
        num_layers=1,
        num_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    missing_k, missing_v = cache.get("missing", 0, 0, 4)
    assert missing_k.shape == (0, 1, 1, 2)
    assert missing_v.shape == (0, 1, 1, 2)

    old_k = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]])
    old_v = old_k + 10.0
    cache.append_block("old", 0, old_k, old_v, 0)
    old_buffer = cache.allocations["old"][0]["buffer"]
    old_ptr = old_buffer[0].data_ptr()
    cache.free("old")

    cache.allocate("new", 2)
    new_buffer = cache.allocations["new"][0]["buffer"]
    assert new_buffer[0].data_ptr() == old_ptr
    empty_k, empty_v = cache.get("new", 0, 0, 4)
    assert empty_k.shape == (0, 1, 1, 2)
    assert empty_v.shape == (0, 1, 1, 2)

    new_k = torch.tensor([[[[7.0, 8.0]]]])
    new_v = new_k + 20.0
    cache.append_block("new", 0, new_k, new_v, 0)
    actual_k, actual_v = cache.get("new", 0, 0, 4)
    torch.testing.assert_close(actual_k, new_k)
    torch.testing.assert_close(actual_v, new_v)


def test_ch16_radix_attention_reuses_token_and_kv_buffers() -> None:
    source = (REPO_ROOT / "ch16" / "radix_attention_example.py").read_text(encoding="utf-8")
    forward_section = source.split("def forward(self, token: int", maxsplit=1)[1].split(
        "def generate_next", maxsplit=1
    )[0]
    generate_next_section = source.split("def generate_next", maxsplit=1)[1].split(
        "def generate_with_radix",
        maxsplit=1,
    )[0]

    assert source.count("@torch.inference_mode()") >= 2
    assert "with torch.no_grad():" not in forward_section
    assert "with torch.no_grad():" not in generate_next_section
    assert "torch.tensor([token]" not in forward_section
    assert "torch.cat([state.kv_cache.keys, k]" not in forward_section
    assert "state.kv_cache.append(k, v)" in forward_section
    assert "self._sampling_noise = None" in source
    assert "def _sampling_like_buffer" in source
    assert "torch.randn(logits.shape, dtype=logits.dtype, device=logits.device, out=noise)" in generate_next_section
    assert "torch.topk(logits, k, dim=-1, out=(top_k_logits, top_k_indices))" in generate_next_section
    assert "torch.softmax(top_k_logits, dim=-1, out=probs)" in generate_next_section
    assert "torch.multinomial(probs, num_samples=1, out=selected_idx)" in generate_next_section
    assert "torch.gather(top_k_indices, 1, selected_idx, out=next_token_device)" in generate_next_section
    assert "next_token = int(next_token_host[0])" in generate_next_section
    assert "torch.randn_like(logits)" not in generate_next_section
    assert "selected_idx = torch.multinomial" not in generate_next_section
    assert ".item()" not in generate_next_section

    from ch16.radix_attention_example import ModelState, SimpleTransformerModel

    model = SimpleTransformerModel(
        vocab_size=32,
        hidden_dim=16,
        num_heads=4,
        device=torch.device("cpu"),
    )
    model._cache_block_tokens = 4
    state = ModelState(hidden_dim=model.hidden_dim, num_heads=model.num_heads, device=model.device)

    first_state = model.forward(1, state)
    second_state = model.forward(2, first_state)

    first_cache = first_state.kv_cache
    second_cache = second_state.kv_cache
    assert first_cache.seq_len == 1
    assert second_cache.seq_len == 2
    assert first_cache.keys.data_ptr() == second_cache.keys.data_ptr()
    assert first_cache.key_view.shape[0] == 1
    assert second_cache.key_view.shape[0] == 2

    token, generated_state = model.generate_next(second_state)
    noise_ptr = model._sampling_noise.data_ptr()
    topk_ptr = model._sampling_topk_logits.data_ptr()
    probs_ptr = model._sampling_probs.data_ptr()
    assert isinstance(token, int)
    assert generated_state.context is not None

    token, generated_state = model.generate_next(generated_state)
    assert isinstance(token, int)
    assert model._sampling_noise.data_ptr() == noise_ptr
    assert model._sampling_topk_logits.data_ptr() == topk_ptr
    assert model._sampling_probs.data_ptr() == probs_ptr


def test_medusa_eagle_avoids_inner_loop_wall_clock_timing() -> None:
    source = (REPO_ROOT / "ch15" / "medusa_eagle_speculative_benchmarks.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def _run_family_speculative_decode", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    seed_section = source.split("def _draft_seed_tokens", maxsplit=1)[1].split(
        "def _run_family_speculative_decode",
        maxsplit=1,
    )[0]

    assert "self._accept_prefix = torch.empty(wl.speculative_k, device=self.device, dtype=torch.int32)" in setup_section
    assert "self._accept_count = torch.empty((), device=self.device, dtype=torch.int32)" in setup_section
    assert "self._draft_head_offsets = torch.arange(wl.speculative_k, device=self.device, dtype=torch.int64).view(1, -1)" in setup_section
    assert "self._draft_seed_buffer = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.int64)" in setup_section
    assert "self._draft_block_values = torch.empty((1, wl.speculative_k), device=self.device, dtype=wl.dtype)" in setup_section
    assert "self._target_next_values = torch.empty((1, wl.speculative_k), device=self.device, dtype=wl.dtype)" in setup_section
    assert "self._matches = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.bool)" in setup_section
    assert "with torch.inference_mode():" in benchmark_section
    assert "time.perf_counter" not in benchmark_section
    assert "draft_time_ms=None" in benchmark_section
    assert "verify_time_ms=None" in benchmark_section
    assert ".nonzero(" not in benchmark_section
    assert ".argmax(" not in benchmark_section
    assert "torch.arange(k" not in seed_section
    assert "torch.add(prev.expand(-1, k), head_offsets, out=seed_tokens)" in seed_section
    assert "seed_tokens.remainder_(self.workload.vocab_size)" in seed_section
    assert "torch.max(logits_d, dim=-1, out=(draft_values, draft_block))" in benchmark_section
    assert "torch.max(logits_t, dim=-1, out=(target_values, target_next))" in benchmark_section
    assert "torch.eq(target_next, self._draft_ids[:, :k], out=matches)" in benchmark_section
    assert "torch.cumprod(matches[0], dim=0, dtype=torch.int32, out=accept_prefix)" in benchmark_section
    assert "torch.sum(accept_prefix, dim=0, out=self._accept_count)" in benchmark_section


def test_medusa_eagle_validation_batches_output_bounds_check() -> None:
    source = (REPO_ROOT / "ch15" / "medusa_eagle_speculative_benchmarks.py").read_text(
        encoding="utf-8"
    )
    validate_section = source.split("def validate_result", maxsplit=1)[1].split(
        "def get_benchmark",
        maxsplit=1,
    )[0]

    assert "torch.any((self.output < 0) | (self.output >= self.workload.vocab_size))" in validate_section
    assert "torch.any(self.output < 0) or torch.any(self.output >= self.workload.vocab_size)" not in validate_section


def test_medusa_eagle_verification_batches_summary_scalar_reads() -> None:
    source = (REPO_ROOT / "ch15" / "medusa_eagle_speculative_benchmarks.py").read_text(
        encoding="utf-8"
    )
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def get_workload_metadata",
        maxsplit=1,
    )[0]

    assert "verify_summary = torch.stack(" in capture_section
    assert ").detach().cpu()" in capture_section
    assert "self.input_ids[0, 0].item()" not in capture_section
    assert "in_vocab.item()" not in capture_section


def test_ch15_speculative_decode_reuses_acceptance_buffers() -> None:
    source = (REPO_ROOT / "ch15" / "speculative_decoding_benchmarks.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def _run_speculative_decode", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert "self._accept_prefix = torch.empty(wl.speculative_k, device=self.device, dtype=torch.int32)" in setup_section
    assert "self._accept_count = torch.empty((), device=self.device, dtype=torch.int32)" in setup_section
    assert "self._draft_next_values = torch.empty((1,), device=self.device, dtype=wl.dtype)" in setup_section
    assert "self._target_next_values = torch.empty((1, wl.speculative_k), device=self.device, dtype=wl.dtype)" in setup_section
    assert "self._matches = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.bool)" in setup_section
    assert "with torch.inference_mode():" in benchmark_section
    assert ".nonzero(" not in benchmark_section
    assert "mismatch =" not in benchmark_section
    assert "torch.max(logits_d[:, 0, :], dim=-1, out=(self._draft_next_values, self._draft_next_tokens))" in benchmark_section
    assert "torch.max(logits_t, dim=-1, out=(target_values, target_next))" in benchmark_section
    assert "torch.eq(target_next, self._draft_ids[:, :k], out=matches)" in benchmark_section
    assert ".argmax(" not in benchmark_section
    assert "torch.cumprod(matches[0], dim=0, dtype=torch.int32, out=accept_prefix)" in benchmark_section
    assert "torch.sum(accept_prefix, dim=0, out=self._accept_count)" in benchmark_section
    assert "accept_k = int(self._accept_count.item())" in benchmark_section


def test_labs_speculative_decode_reuses_acceptance_buffers() -> None:
    source = (
        REPO_ROOT / "labs" / "speculative_decode" / "optimized_speculative_decode.py"
    ).read_text(encoding="utf-8")
    common_source = (
        REPO_ROOT / "labs" / "speculative_decode" / "speculative_decode_common.py"
    ).read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert "self._accept_prefix = torch.empty(" in setup_section
    assert "self._accept_count = torch.empty((), device=self.device, dtype=torch.int32)" in setup_section
    assert "self._draft_next_values = torch.empty((1,), device=self.device, dtype=wl.dtype)" in setup_section
    assert "self._target_next_values = torch.empty((1, wl.speculative_k), device=self.device, dtype=wl.dtype)" in setup_section
    assert "self._matches = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.bool)" in setup_section
    assert ".nonzero(" not in benchmark_section
    assert "mismatch =" not in benchmark_section
    assert "torch.max(logits_d[:, 0, :], dim=-1, out=(self._draft_next_values, self._draft_next_tokens))" in benchmark_section
    assert "torch.max(logits_t, dim=-1, out=(target_values, target_next))" in benchmark_section
    assert "torch.eq(target_next, self._draft_ids[:, :k], out=matches)" in benchmark_section
    assert ".argmax(" not in benchmark_section
    assert "torch.cumprod(matches[0], dim=0, dtype=torch.int32, out=accept_prefix)" in benchmark_section
    assert "torch.sum(accept_prefix, dim=0, out=self._accept_count)" in benchmark_section
    assert "accept_k = int(self._accept_count.item())" in benchmark_section
    assert common_source.count("with torch.inference_mode():") >= 2
    assert "with torch.no_grad():" not in common_source


def test_labs_baseline_speculative_decode_reuses_next_token_buffer() -> None:
    source = (
        REPO_ROOT / "labs" / "speculative_decode" / "baseline_speculative_decode.py"
    ).read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert "self._next_token_values = torch.empty((1,), device=self.device, dtype=wl.dtype)" in setup_section
    assert "self._next_token_ids = torch.empty((1,), device=self.device, dtype=torch.long)" in setup_section
    assert "with torch.inference_mode():" in benchmark_section
    assert "torch.max(logits[:, 0, :], dim=-1, out=(self._next_token_values, self._next_token_ids))" in benchmark_section
    assert ".argmax(" not in benchmark_section


def test_ch15_speculative_decode_common_uses_inference_mode_for_setup_mutations() -> None:
    source = (REPO_ROOT / "ch15" / "speculative_decoding_common.py").read_text(
        encoding="utf-8"
    )

    assert source.count("with torch.inference_mode():") >= 2
    assert "with torch.no_grad():" not in source


def test_ch19_double_buffering_reuses_copy_events_outside_hot_loop() -> None:
    baseline_source = (REPO_ROOT / "ch19" / "baseline_memory_double_buffering.py").read_text(encoding="utf-8")
    source = (REPO_ROOT / "ch19" / "optimized_memory_double_buffering.py").read_text(encoding="utf-8")
    baseline_setup = baseline_source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    baseline_benchmark = baseline_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    baseline_capture = baseline_source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]
    setup_section = source.split("def benchmark_fn", maxsplit=1)[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]

    assert "self._enable_nvtx = get_nvtx_enabled(config) if config else False" in baseline_setup
    assert "self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())" in baseline_setup
    assert "with torch.inference_mode():" in baseline_benchmark
    assert "with torch.no_grad():" not in baseline_benchmark
    assert "get_config()" not in baseline_benchmark
    assert "get_nvtx_enabled(" not in baseline_benchmark
    assert "enable=self._enable_nvtx" in baseline_benchmark
    assert "parameter_count=self._payload_parameter_count" in baseline_capture
    assert "sum(p.numel()" not in baseline_capture
    assert "self._enable_nvtx = get_nvtx_enabled(config) if config else False" in setup_section
    assert "self._payload_parameter_count = sum(p.numel() for p in params)" in setup_section
    assert "self.copy_events = [torch.cuda.Event(blocking=False) for _ in range(2)]" in setup_section
    assert "self.buffers = [self.buffer_a, self.buffer_b]" in setup_section
    assert "torch.cuda.Event(" not in benchmark_section
    assert "get_config()" not in benchmark_section
    assert "get_nvtx_enabled(" not in benchmark_section
    assert "enable=self._enable_nvtx" in benchmark_section
    assert "buffers = [self.buffer_a, self.buffer_b]" not in benchmark_section
    assert "buffers = self.buffers" in benchmark_section
    assert "with torch.inference_mode():" in benchmark_section
    assert "with torch.no_grad():" not in benchmark_section
    assert "copy_events = self.copy_events" in benchmark_section
    assert "Double buffers or copy events not initialized" in benchmark_section
    assert "parameter_count=self._payload_parameter_count" in capture_section
    assert "sum(p.numel()" not in capture_section


def test_ch04_multigpu_symmetric_memory_reuses_timing_events_outside_hot_loop() -> None:
    targets = (
        ("baseline_symmetric_memory_perf_multigpu.py", "self._timing_pair"),
        ("optimized_symmetric_memory_perf_multigpu.py", "self._timing_pairs"),
    )
    for filename, expected_field in targets:
        source = (REPO_ROOT / "ch04" / filename).read_text(encoding="utf-8")
        setup_section = source.split("def benchmark_fn", maxsplit=1)[0]
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def finalize_iteration_metrics", maxsplit=1
        )[0]

        assert "torch.cuda.Event(enable_timing=True)" in setup_section
        assert "torch.cuda.Event(" not in benchmark_section
        assert expected_field in benchmark_section
        assert "Timing events not initialized" in benchmark_section


def test_ch04_symmetric_queue_batches_head_tail_reads() -> None:
    source = (REPO_ROOT / "ch04" / "symmetric_memory_data_structures.py").read_text(
        encoding="utf-8"
    )
    queue_section = source.split("class LockFreeRingBuffer", maxsplit=1)[1].split(
        "# ============================================================================",
        maxsplit=1,
    )[0]

    assert "torch.stack((self.tail[0], self.head[0])).tolist()" in queue_section
    assert "torch.stack((self.head[0], self.tail[0])).tolist()" in queue_section
    assert "self.tail.item()" not in queue_section
    assert "self.head.item()" not in queue_section


def test_ch04_optimized_bandwidth_suite_reuses_timing_events_outside_hot_loop() -> None:
    source = (REPO_ROOT / "ch04" / "optimized_bandwidth_benchmark_suite_multigpu.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def benchmark_fn", maxsplit=1)[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def finalize_iteration_metrics", maxsplit=1
    )[0]

    assert "torch.cuda.Event(enable_timing=True)" in setup_section
    assert "torch.cuda.Event(" not in benchmark_section
    assert "self._pending_timing_pairs = self._timing_pairs[: len(self.streams)]" in benchmark_section
    assert "Timing events not initialized" in benchmark_section


def test_ch04_nvshmem_microbench_defers_output_tensor_outside_hot_loop() -> None:
    source = (REPO_ROOT / "ch04" / "nvshmem_ibgda_microbench_multigpu.py").read_text(
        encoding="utf-8"
    )
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def get_config", maxsplit=1
    )[0]

    assert "torch.tensor(" not in benchmark_section
    assert "self._last_output_values = [self._parsed_metrics.get(\"bandwidth_gbps\", 0.0)]" in benchmark_section
    assert "self._last_output = torch.tensor(" in capture_section


def test_ch15_single_disaggregated_defers_output_cat_outside_hot_loop() -> None:
    source = (REPO_ROOT / "ch15" / "disaggregated_inference_single_common.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def _allocate_kv_cache", maxsplit=1
    )[0]
    output_helper = source.split("def _set_output_from_tokens", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    baseline_benchmark = source.split("class BaselineDisaggregatedInferenceSingleGPUBenchmark", maxsplit=1)[1].split(
        "def _variant_metrics", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]

    assert "torch.cat(" not in output_helper
    assert "self._pending_outputs = [torch.empty(0) for _ in range(self.cfg.requests_per_rank)]" in setup_section
    assert "self._pending_outputs = outputs" in output_helper
    assert "torch.cat(" not in baseline_benchmark
    assert "outputs: List[torch.Tensor] = []" not in baseline_benchmark
    assert "outputs = self._pending_outputs" in baseline_benchmark
    assert "output_idx = 0" in baseline_benchmark
    assert "with torch.inference_mode():" in baseline_benchmark
    assert "outputs.append(" not in baseline_benchmark
    assert "outputs[output_idx] = self._run_decode_loop(self._baseline_kv_cache, seed_tokens)" in baseline_benchmark
    assert "output_idx += 1" in baseline_benchmark
    assert "kv_cpu.to(self.device)" not in baseline_benchmark
    assert "hidden.cpu()" not in baseline_benchmark
    assert "self._baseline_kv_cache = self._allocate_kv_cache()" in baseline_benchmark
    assert "self._kv_host_staging.copy_(hidden, non_blocking=False)" in baseline_benchmark
    assert "self._baseline_kv_cache[:, : self.cfg.context_window].copy_(" in baseline_benchmark
    assert "self._output = torch.cat(self._pending_outputs, dim=0)" in capture_section


def test_ch17_single_prefill_decode_host_handoff_copies_into_existing_kv_cache() -> None:
    source = (REPO_ROOT / "ch17" / "prefill_decode_disagg_single_common.py").read_text(
        encoding="utf-8"
    )
    base_section = source.split("class _PrefillDecodeSingleGPUBase", maxsplit=1)[1].split(
        "class BaselinePrefillDecodeSingleGPUBenchmark",
        maxsplit=1,
    )[0]
    baseline_benchmark = source.split("class BaselinePrefillDecodeSingleGPUBenchmark", maxsplit=1)[1].split(
        "class OptimizedPrefillDecodeSingleGPUBenchmark",
        maxsplit=1,
    )[0]
    optimized_benchmark = source.split("class OptimizedPrefillDecodeSingleGPUBenchmark", maxsplit=1)[1]

    assert "self._pending_outputs: List[torch.Tensor] = []" in base_section
    assert "self._pending_outputs = [torch.empty(0) for _ in range(self.cfg.requests_per_rank)]" in base_section
    assert "self._output = torch.stack(self._pending_outputs, dim=0)" in base_section
    assert "kv_cache = kv_cpu.to(self.device)" not in baseline_benchmark
    assert "kv_cache.cpu()" not in baseline_benchmark
    assert "self._kv_host_staging.copy_(kv_cache, non_blocking=False)" in baseline_benchmark
    assert "kv_cache.copy_(self._kv_host_staging, non_blocking=False)" in baseline_benchmark
    assert "outputs = self._pending_outputs" in baseline_benchmark
    assert "output_idx = 0" in baseline_benchmark
    assert "with torch.inference_mode():" in baseline_benchmark
    assert "outputs[output_idx] = self.decode_model.decode(seed, kv_cache, self.cfg.decode_tokens)" in baseline_benchmark
    assert "output_idx += 1" in baseline_benchmark
    assert "outputs: List[torch.Tensor] = []" not in baseline_benchmark
    assert "outputs.append(" not in baseline_benchmark
    assert "torch.stack(" not in baseline_benchmark
    assert "self._output = decoded.view(" in optimized_benchmark
    assert "self._pending_outputs.clear()" in optimized_benchmark
    assert "self._pending_outputs = []" not in optimized_benchmark
    assert "with torch.inference_mode():" in optimized_benchmark
    assert "list(" not in optimized_benchmark


def test_ch15_inference_placement_defers_output_tensor_outside_hot_loop() -> None:
    source = (REPO_ROOT / "ch15" / "baseline_inference_placement.py").read_text(encoding="utf-8")
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]

    assert "torch.tensor(" not in benchmark_section
    assert "self._output_values = [" in benchmark_section
    assert "self.output = torch.tensor(self._output_values, dtype=torch.float32)" in capture_section


def test_ch15_kv_cache_math_preconcats_static_inputs() -> None:
    source = (REPO_ROOT / "ch15" / "kv_cache_management_math.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert "self._sequence_inputs: Optional[torch.Tensor] = None" in source
    assert "self.cache_buffer = torch.empty(" in setup_section
    assert "self.cache_buffer = torch.zeros(" not in setup_section
    assert "self._sequence_inputs = torch.empty_like(self.cache_buffer)" in setup_section
    assert "torch.cat(self.inputs, dim=1, out=self._sequence_inputs)" in setup_section
    assert "torch.cat(self.inputs" not in benchmark_section
    assert "with torch.inference_mode():" in benchmark_section
    assert "queries = self._sequence_inputs" in benchmark_section
    assert "k_cache = self._sequence_inputs" in benchmark_section


def test_ch15_kv_cache_management_wrappers_use_inference_mode() -> None:
    for relative in (
        "ch15/baseline_kv_cache_management.py",
        "ch15/optimized_kv_cache_management.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]
        assert "with torch.inference_mode():" in benchmark_section
        assert "with torch.no_grad():" not in benchmark_section


def test_ch15_wide_ep_packs_directly_into_reusable_buffers() -> None:
    baseline_source = (REPO_ROOT / "ch15" / "baseline_wide_ep.py").read_text(encoding="utf-8")
    optimized_source = (REPO_ROOT / "ch15" / "optimized_wide_ep.py").read_text(encoding="utf-8")
    baseline_benchmark = baseline_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]
    baseline_setup = baseline_source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    optimized_setup = optimized_source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    optimized_benchmark = optimized_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert "self._dest_ranks = torch.div(" in baseline_setup
    assert "self._rank_indices.append(indices)" in baseline_setup
    assert "self._rank_offsets.append((offset, next_offset))" in baseline_setup
    assert "self._perm = torch.cat(self._rank_indices, dim=0)" in baseline_setup
    assert "with torch.inference_mode():" in baseline_setup
    assert "with torch.inference_mode():" in baseline_benchmark
    assert "self._dest_ranks = torch.div(" in optimized_setup
    assert "self._perm = torch.argsort(self._dest_ranks)" in optimized_setup
    assert "with torch.inference_mode():" in optimized_setup
    assert "with torch.inference_mode():" in optimized_benchmark
    assert "dest_ranks = torch.div(" not in baseline_benchmark
    assert "dest_ranks = torch.div(" not in optimized_benchmark
    assert "mask = dest_ranks == r" not in baseline_benchmark
    assert ".nonzero(" not in baseline_benchmark
    assert "torch.argsort(" not in optimized_benchmark
    assert "send_buf = torch.cat(send_tokens" not in baseline_benchmark
    assert "recv_buf.copy_(send_buf)" not in baseline_benchmark
    assert "torch.cat(send_tokens" not in baseline_benchmark
    assert "torch.index_select(flat, 0, indices, out=recv_buf[start:end])" in baseline_benchmark
    assert "send_buf = flat.index_select(0, perm)" not in optimized_benchmark
    assert "recv_buf.copy_(send_buf)" not in optimized_benchmark
    assert "recv_back.copy_(recv_out)" not in baseline_benchmark
    assert "recv_back.copy_(recv_out)" not in optimized_benchmark
    assert "out_flat.index_copy_(0, perm, recv_out)" in baseline_benchmark
    assert "out_flat.index_copy_(0, perm, recv_out)" in optimized_benchmark
    assert "perm = self._perm" in optimized_benchmark
    assert "torch.index_select(flat, 0, perm, out=recv_buf)" in optimized_benchmark


def test_ch15_moe_overlap_and_routing_use_inference_mode() -> None:
    for relative in (
        "ch15/baseline_moe_overlap.py",
        "ch15/optimized_moe_overlap_shared_expert.py",
        "ch15/moe_routing_benchmark_common.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split(
            "def benchmark_fn",
            maxsplit=1,
        )[0]
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]
        assert "with torch.inference_mode():" in setup_section
        assert "with torch.inference_mode():" in benchmark_section
        assert "with torch.no_grad():" not in setup_section
        assert "with torch.no_grad():" not in benchmark_section


def test_ch15_moe_comm_exchange_reuses_static_pack_buffers() -> None:
    source = (REPO_ROOT / "ch15" / "moe_comm_exchange_benchmarks.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def get_custom_streams",
        maxsplit=1,
    )[0]
    baseline_section = source.split("def _run_baseline", maxsplit=1)[1].split(
        "def _run_overlap",
        maxsplit=1,
    )[0]
    overlap_section = source.split("def _run_overlap", maxsplit=1)[1].split(
        "def _run_hierarchical",
        maxsplit=1,
    )[0]
    hierarchical_section = source.split("def _run_hierarchical", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert "self._baseline_perm = torch.cat(baseline_perm_parts, dim=0)" in setup_section
    assert "self._baseline_packed = torch.empty_like(flat)" in setup_section
    assert "self._local_packed = torch.empty(" in setup_section
    assert "self._group_offsets = torch.empty(" in setup_section
    assert "self._group_offsets = torch.zeros(" not in setup_section
    assert "torch.cumsum(group_counts, dim=0, out=self._group_offsets[1:])" in setup_section
    assert "with torch.inference_mode():" in setup_section
    assert "with torch.no_grad():" not in setup_section
    for run_section in (baseline_section, overlap_section, hierarchical_section):
        assert "with torch.inference_mode():" in run_section
        assert "with torch.no_grad():" not in run_section
    assert "self._baseline_out" not in source
    assert "self._local_out" not in source
    assert "self._remote_out" not in source
    assert "self._hierarchical_out" not in source
    assert "send_tokens" not in baseline_section
    assert "send_pos" not in baseline_section
    assert ".nonzero(" not in baseline_section
    assert "torch.cat(" not in baseline_section
    assert "torch.index_select(flat, 0, self._baseline_perm, out=self._baseline_packed)" in baseline_section
    assert "baseline_out = self.expert(self._baseline_packed)" in baseline_section
    assert "self._out_flat.index_copy_(0, self._baseline_perm, baseline_out)" in baseline_section
    assert "local_tokens = flat.index_select" not in overlap_section
    assert "torch.index_select(flat, 0, self._local_perm, out=self._local_packed)" in overlap_section
    assert "local_out = self.expert(self._local_packed)" in overlap_section
    assert "self._out_flat.index_copy_(0, self._local_perm, local_out)" in overlap_section
    assert "remote_out = self.expert(self._remote_packed)" in overlap_section
    assert "self._out_flat.index_copy_(0, self._remote_perm, remote_out)" in overlap_section
    assert "group_out = self.expert(self._hierarchical_packed[start:end])" in hierarchical_section
    assert (
        "self._out_flat.index_copy_(0, self._hierarchical_perm[start:end], group_out)"
        in hierarchical_section
    )
    assert ".copy_(self.expert(" not in source


def test_moe_parallelism_plan_benchmark_reuses_summary_buffer() -> None:
    source = (REPO_ROOT / "labs" / "moe_parallelism" / "benchmarking.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    finalize_section = source.split("def _finalize_output", maxsplit=1)[1].split(
        "def run_benchmark",
        maxsplit=1,
    )[0]

    assert "self._summary_buffer: Optional[torch.Tensor] = None" in source
    assert "self._summary_buffer = torch.empty((1, 3), dtype=torch.float32)" in setup_section
    assert "torch.tensor([metric_values]" not in finalize_section
    assert "for index, value in enumerate(metric_values):" in finalize_section
    assert "self._summary_buffer[0, index] = float(value)" in finalize_section
    assert "self.output = self._summary_buffer.detach()" in finalize_section


def test_ch19_fp8_calibration_free_defers_output_materialization_outside_hot_loop() -> None:
    source = (REPO_ROOT / "ch19" / "fp8_calibration_free_tool.py").read_text(encoding="utf-8")
    scale_section = source.split("def _compute_scale", maxsplit=1)[1].split(
        "def _quantize_fp8",
        maxsplit=1,
    )[0]
    run_section = source.split("def run(self) -> torch.Tensor", maxsplit=1)[1].split(
        "def cleanup", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]

    assert ".detach().clone()" not in run_section
    assert ".item()" not in run_section
    assert "self.output_slice = x[:1, :1, : min(16, x.shape[-1])]" in run_section
    assert "torch.tensor(" not in benchmark_section
    assert "self._output = output" in benchmark_section
    assert "output=self._output.detach().float().clone()" in capture_section
    assert "with torch.inference_mode():" in scale_section
    assert "with torch.no_grad():" not in scale_section


def test_ch19_fp8_compiled_matmul_uses_cuda_event_timing() -> None:
    source = (REPO_ROOT / "ch19" / "fp8_compiled_matmul.py").read_text(encoding="utf-8")
    benchmark_section = source.split("def benchmark_matmul", maxsplit=1)[1].split(
        "def main",
        maxsplit=1,
    )[0]

    assert benchmark_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "start.record()" in benchmark_section
    assert "end.record()" in benchmark_section
    assert "start.elapsed_time(end) / max(iters, 1)" in benchmark_section
    assert "time.perf_counter()" not in benchmark_section


def test_ch19_quantization_validator_reuses_timing_events() -> None:
    source = (REPO_ROOT / "ch19" / "validate_quantization_performance.py").read_text(
        encoding="utf-8"
    )
    benchmark_section = source.split("def benchmark_function", maxsplit=1)[1].split(
        "# Calculate statistics",
        maxsplit=1,
    )[0]
    warmup_section = benchmark_section.split("# Warmup", maxsplit=1)[1].split(
        "# Clear memory stats",
        maxsplit=1,
    )[0]
    timing_section = benchmark_section.split(
        "with nvtx.range(compute_label):",
        maxsplit=1,
    )[1]
    cuda_timing_section = timing_section.split("if cuda_available:", maxsplit=1)[
        1
    ].split("else:", maxsplit=1)[0]
    before_sample_loop = cuda_timing_section.split(
        "for i in range(benchmark_iters):",
        maxsplit=1,
    )[0]
    sample_loop = cuda_timing_section.split(
        "for i in range(benchmark_iters):",
        maxsplit=1,
    )[1]

    assert 'warmup_label = standardize_nvtx_label(f"warmup:{self.name}_{precision}")' in benchmark_section
    assert 'compute_label = standardize_nvtx_label(f"compute_math:{self.name}_{precision}")' in benchmark_section
    assert "iteration_labels = [" in benchmark_section
    assert 'standardize_nvtx_label(f"iteration:{self.name}_{precision}_{i}")' in benchmark_section
    assert "with nvtx.range(warmup_label):" in warmup_section
    assert "standardize_nvtx_label(" not in warmup_section
    assert "torch.cuda.synchronize()" not in warmup_section.split(
        "for _ in range(warmup_iters):", maxsplit=1
    )[1].split("if cuda_available:", maxsplit=1)[0]
    assert "if cuda_available:\n            torch.cuda.synchronize()" in warmup_section
    assert before_sample_loop.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "torch.cuda.Event(enable_timing=True)" not in sample_loop
    assert "start_event.record()" in sample_loop
    assert "end_event.record()" in sample_loop
    assert "end_event.synchronize()" in sample_loop
    assert "times.append(start_event.elapsed_time(end_event))" in sample_loop
    assert "with nvtx.range(iteration_labels[i]):" in sample_loop
    assert "standardize_nvtx_label(" not in sample_loop


def test_ch19_nvfp4_training_defers_verification_forward_outside_hot_loop() -> None:
    for filename in ("baseline_nvfp4_training.py", "optimized_nvfp4_training.py"):
        source = (REPO_ROOT / "ch19" / filename).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def get_input_signature",
            maxsplit=1,
        )[0]

        assert ".float().clone()" not in benchmark_section
        assert "self.output = None" in benchmark_section
        assert "with torch.inference_mode():" in capture_section
        assert "with torch.no_grad():" not in capture_section
        assert "self.model(self._verify_input)" in capture_section
        assert ".float().clone()" in capture_section


def test_ch13_regional_compile_moves_verification_materialization_out_of_hot_loop() -> None:
    targets = {
        "baseline_regional_compile.py": "self.output = self.compiled_model(x).detach()",
        "optimized_regional_compile.py": "self.output = self.model(x).detach()",
    }

    for name, output_assignment in targets.items():
        source = (REPO_ROOT / "ch13" / name).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def teardown", maxsplit=1
        )[0]

        assert ".detach().float().clone()" not in benchmark_section
        assert ".detach().clone()" not in benchmark_section
        assert output_assignment in benchmark_section
        assert "self._verify_output = self.output" in benchmark_section
        assert "output=self._verify_output.float().clone()" in capture_section


def test_ch13_inference_precision_benchmarks_use_inference_mode() -> None:
    filenames = (
        "baseline_attention_standard.py",
        "optimized_attention_standard.py",
        "baseline_long_context_attention.py",
        "optimized_long_context_attention.py",
        "baseline_quantization.py",
        "optimized_quantization.py",
        "baseline_torchao_quantization.py",
        "optimized_torchao_quantization.py",
        "optimized_torchao_quantization_compiled.py",
        "baseline_fp4_perchannel.py",
        "optimized_fp4_perchannel.py",
        "baseline_fp8_perchannel.py",
        "optimized_fp8_perchannel.py",
        "fp8_perchannel_bench.py",
        "baseline_fp8_static.py",
        "optimized_fp8_static.py",
        "fp8_static_demo.py",
        "baseline_precisionfp8_pad_inner.py",
        "optimized_precisionfp8_pad_inner.py",
        "baseline_precisionfp8_pad_inner_matmul.py",
        "optimized_precisionfp8_pad_inner_matmul.py",
        "fp8_perchannel_demo.py",
        "baseline_warp_specialization_training.py",
        "optimized_warp_specialization_training.py",
        "baseline_regional_compile.py",
        "optimized_regional_compile.py",
    )

    for filename in filenames:
        source = (REPO_ROOT / "ch13" / filename).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]
        assert "torch.inference_mode()" in benchmark_section
        assert "torch.no_grad()" not in benchmark_section

    setup_files = (
        "baseline_fp4_perchannel.py",
        "optimized_fp4_perchannel.py",
        "baseline_fp8_perchannel.py",
        "optimized_fp8_perchannel.py",
        "fp8_perchannel_bench.py",
    )
    for filename in setup_files:
        source = (REPO_ROOT / "ch13" / filename).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split(
            "def benchmark_fn",
            maxsplit=1,
        )[0]
        assert "torch.inference_mode()" in setup_section
        assert "torch.no_grad()" not in setup_section


def test_ch13_static_and_training_helpers_use_inference_mode() -> None:
    filenames = (
        "baseline_training_standard.py",
        "optimized_training_standard.py",
        "baseline_training_speed.py",
        "context_parallelism.py",
        "fp8_static_demo.py",
        "optimized_fp8_static.py",
        "fp8_perchannel_demo.py",
    )

    for filename in filenames:
        source = (REPO_ROOT / "ch13" / filename).read_text(encoding="utf-8")
        assert "torch.inference_mode()" in source
        assert "torch.no_grad()" not in source


def test_ch13_optimized_fp8_perchannel_reuses_input_scale_buffer() -> None:
    source = (REPO_ROOT / "ch13" / "optimized_fp8_perchannel.py").read_text(
        encoding="utf-8"
    )
    forward_section = source.split("def forward(self, x: torch.Tensor)", maxsplit=1)[
        1
    ].split(
        "class OptimizedFP8PerChannelBenchmark",
        maxsplit=1,
    )[0]

    assert 'self.register_buffer("_scale_a_buffer", torch.empty(0), persistent=False)' in source
    assert 'self.register_buffer("_input_scaled_buffer", torch.empty(0), persistent=False)' in source
    assert '"_input_fp8_buffer"' in source
    assert "def _activation_buffers(self, x_2d: torch.Tensor)" in source
    assert "scale_a = self._scale_a_buffer" in forward_section
    assert "scale_a.copy_(input_scale)" in forward_section
    assert "torch.div(x_2d, input_scale, out=input_scaled)" in forward_section
    assert "x_fp8.copy_(input_scaled)" in forward_section
    assert "(x_2d / input_scale).to(torch.float8_e4m3fn)" not in forward_section
    assert ".expand(x_fp8.size(0), 1).contiguous()" not in forward_section


def test_ch13_fp8_perchannel_bench_caches_weight_quantization() -> None:
    source = (REPO_ROOT / "ch13" / "fp8_perchannel_bench.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    forward_section = source.split("def forward(self, x: torch.Tensor)", maxsplit=1)[
        1
    ].split(
        "class OptimizedFP8PerChannelBenchmark",
        maxsplit=1,
    )[0]

    assert 'self.register_buffer("_weight_q", torch.empty(0), persistent=False)' in source
    assert "def prepare_fp8_weights" in source
    assert "self.model.prepare_fp8_weights()" in setup_section
    assert "torch.inference_mode()" in setup_section
    assert "torch.no_grad()" not in setup_section
    assert "weight_q = self._weight_q" in forward_section
    assert "weight_scale = self._weight_scale" in forward_section
    assert "output_q.mul_(input_scale)" in forward_section
    assert "output_q.mul_(weight_scale)" in forward_section
    assert "combined_scale = input_scale * weight_scale" not in forward_section

    from ch13.fp8_perchannel_bench import FP8PerChannelLinear

    torch.manual_seed(123)
    layer = FP8PerChannelLinear(8, 6)
    x = torch.randn(2, 3, 8)
    expected = layer(x)
    layer.prepare_fp8_weights()
    actual = layer(x)
    torch.testing.assert_close(actual, expected)


def test_ch16_and_lab_forward_benchmarks_use_inference_mode() -> None:
    paths = (
        "ch16/awq_gptq_smoothquant_benchmarks.py",
        "labs/moe_optimization_journey/baseline_moe_pad_quant.py",
        "labs/moe_optimization_journey/optimized_moe_pad_quant.py",
        "labs/moe_optimization_journey/level4_triton.py",
        "labs/moe_optimization_journey/level6_full_stack.py",
        "labs/moe_optimization_journey/moe_benchmark.py",
        "labs/train_distributed/training_utils/torchrun_harness.py",
    )

    for path in paths:
        source = (REPO_ROOT / path).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]
        assert "torch.inference_mode()" in benchmark_section
        assert "torch.no_grad()" not in benchmark_section


def test_ch13_memory_profiling_pair_keeps_compute_dtype_fixed_and_direct_output_capture() -> None:
    baseline_source = (REPO_ROOT / "ch13" / "baseline_memory_profiling.py").read_text(encoding="utf-8")
    optimized_source = (REPO_ROOT / "ch13" / "optimized_memory_profiling.py").read_text(encoding="utf-8")
    baseline_benchmark = baseline_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    optimized_benchmark = optimized_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert 'signature_equivalence_group = "ch13_memory_profiling_checkpointing"' in baseline_source
    assert 'signature_equivalence_group = "ch13_memory_profiling_checkpointing"' in optimized_source
    assert "dtype=torch.bfloat16" not in optimized_source
    assert "self.inputs_fp32" not in optimized_source
    assert "self.targets_fp32" not in optimized_source
    for source in (baseline_source, optimized_source):
        assert "dtype=torch.float32" in source
    assert ".detach().clone()" not in baseline_benchmark
    assert ".detach().clone()" not in optimized_benchmark
    assert "self.output = outputs.detach()" in baseline_benchmark
    assert "self.output = outputs.detach()" in optimized_benchmark
    assert "output=self.output.detach().float().clone()" in baseline_source
    assert "output=self.output.detach().float().clone()" in optimized_source
    assert "self.output_buffer" not in optimized_source
    assert 'return "memory"' in baseline_source
    assert 'return "memory"' in optimized_source


def test_ch12_kernel_launches_pair_keeps_hot_path_work_fixed() -> None:
    baseline_source = (REPO_ROOT / "ch12" / "baseline_kernel_launches.py").read_text(encoding="utf-8")
    optimized_source = (REPO_ROOT / "ch12" / "optimized_kernel_launches.py").read_text(encoding="utf-8")
    baseline_benchmark = baseline_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert ".clone()" not in baseline_benchmark
    assert "self.x_input =" in optimized_source
    assert "self.x_capture" not in optimized_source
    assert "self.graph_output = self.work_a" in optimized_source
    assert "with torch.inference_mode(), torch.cuda.graph(self.graph):" in optimized_source


def test_optimized_benchmarks_hoist_nvtx_helpers() -> None:
    for relative in (
        "ch03/optimized_pinned_prefetch_mlp.py",
        "ch04/optimized_cpu_reduction.py",
        "ch04/optimized_nccl.py",
        "ch04/optimized_reinit_comm.py",
        "ch04/optimized_reinit_comm_multigpu.py",
        "ch12/optimized_cuda_graphs.py",
        "ch12/optimized_graph_bandwidth.py",
        "ch12/optimized_kernel_fusion.py",
        "ch12/optimized_kernel_fusion_llm_dedicated_stream_and_prefetch_for_blackwell.py",
        "ch12/optimized_kernel_fusion_llm_persistent_buffer_and_stream_friendly_setup.py",
        "ch12/optimized_kernel_fusion_llm_reuse_static_tensor_and_simplify_setup.py",
        "ch12/optimized_kernel_launches.py",
        "ch12/optimized_work_queue.py",
        "ch14/optimized_attention_eager_sdpa.py",
        "ch14/optimized_cublas_vs_cutlass.py",
        "ch14/optimized_model_compile_reduced_precision.py",
        "ch14/optimized_nccl_quantization.py",
        "ch16/optimized_dense_attention_flash.py",
        "ch16/optimized_regional_compilation.py",
        "ch19/optimized_memory_double_buffering.py",
        "ch19/optimized_nvfp4_training.py",
        "ch20/optimized_integrated_kv_cache.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        pre_benchmark = source.split("def benchmark_fn", maxsplit=1)[0]
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]

        assert "from core.profiling.nvtx_helper import" in pre_benchmark
        assert "from core.profiling.nvtx_helper import" not in benchmark_section


def test_benchmark_functions_do_not_import_nvtx_helpers_in_hot_path() -> None:
    paths = list(REPO_ROOT.glob("ch*/*.py")) + list((REPO_ROOT / "labs").rglob("*.py"))
    ignored_parts = {
        "vendor",
        "third_party",
        "top_submission_candidates",
        "modal697_candidates",
        "candidate_submission",
    }
    violations: list[str] = []

    for path in paths:
        if any(part in ignored_parts for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != "benchmark_fn":
                continue
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.ImportFrom)
                    and child.module == "core.profiling.nvtx_helper"
                ):
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{child.lineno}")

    assert violations == []


def test_ch08_to_ch12_kernel_wrappers_use_inference_mode() -> None:
    paths = (
        "ch08/optimized_tcgen05_custom_vs_cublas.py",
        "ch09/baseline_tcgen05_tma_pipeline.py",
        "ch09/optimized_tcgen05_tma_pipeline.py",
        "ch10/baseline_flashattention3_pipeline.py",
        "ch10/optimized_flashattention3_pipeline.py",
        "ch10/baseline_matmul_tcgen05_epilogue.py",
        "ch10/optimized_matmul_tcgen05_epilogue.py",
        "ch10/baseline_matmul_tcgen05_pipelined.py",
        "ch10/optimized_matmul_tcgen05_pipelined.py",
        "ch10/baseline_matmul_tcgen05_vs_cublas.py",
        "ch10/optimized_matmul_tcgen05_vs_cublas.py",
        "ch10/baseline_tcgen05_warp_specialization.py",
        "ch10/optimized_tcgen05_warp_specialization.py",
        "ch10/baseline_tcgen05_warp_specialization_cutlass.py",
        "ch10/optimized_tcgen05_warp_specialization_cutlass.py",
        "ch10/baseline_tcgen05_warpgroup_specialization.py",
        "ch10/optimized_tcgen05_warpgroup_specialization.py",
        "ch10/warpgroup_specialization_demo.py",
        "ch11/baseline_tensor_cores_streams.py",
        "ch11/optimized_tensor_cores_streams.py",
        "ch11/stream_overlap_base.py",
        "ch12/baseline_kernel_launches.py",
        "ch12/optimized_kernel_launches.py",
    )

    for path in paths:
        source = (REPO_ROOT / path).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]
        assert "torch.inference_mode()" in benchmark_section
        assert "torch.no_grad()" not in benchmark_section


def test_ch11_stream_benchmarks_use_cached_nvtx_range() -> None:
    stream_base = (REPO_ROOT / "ch11" / "stream_overlap_base.py").read_text(encoding="utf-8")
    baseline_tensor_cores = (
        REPO_ROOT / "ch11" / "baseline_tensor_cores_streams.py"
    ).read_text(encoding="utf-8")
    optimized_tensor_cores = (
        REPO_ROOT / "ch11" / "optimized_tensor_cores_streams.py"
    ).read_text(encoding="utf-8")

    assert stream_base.count("with self._nvtx_range(self.label):") == 2
    assert "with self._nvtx_range(self.label):" in baseline_tensor_cores
    assert "with self._nvtx_range(self.nvtx_label):" in optimized_tensor_cores

    for source in (stream_base, baseline_tensor_cores, optimized_tensor_cores):
        assert "get_nvtx_enabled(" not in source
        assert "with nvtx_range(" not in source


def test_ch12_core_benchmarks_use_cached_nvtx_range() -> None:
    expected_labels = {
        "baseline_kernel_launches.py": "kernel_launches",
        "optimized_kernel_launches.py": "kernel_launches",
        "baseline_kernel_fusion.py": "kernel_fusion",
        "optimized_kernel_fusion.py": "kernel_fusion",
        "baseline_cuda_graphs.py": "cuda_graphs",
        "optimized_cuda_graphs.py": "cuda_graphs",
        "baseline_work_queue.py": "work_queue",
        "optimized_work_queue.py": "work_queue",
        "baseline_graph_bandwidth.py": "graph_bandwidth",
        "optimized_graph_bandwidth.py": "optimized_graph_bandwidth_graph",
        "optimized_cuda_graphs_router.py": "cuda_graphs_router",
    }

    for filename, label in expected_labels.items():
        source = (REPO_ROOT / "ch12" / filename).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]
        assert f'with self._nvtx_range("{label}"):' in benchmark_section
        assert "get_nvtx_enabled(" not in benchmark_section
        assert "with nvtx_range(" not in benchmark_section
        assert "from core.profiling.nvtx_helper" not in source


def test_ch11_ch12_standalone_timing_tools_use_inference_mode() -> None:
    paths = (
        "ch11/memory_async_demo.py",
        "ch11/event_timing_demo.py",
        "ch11/stream_priority_demo.py",
        "ch12/graph_capture_demo.py",
        "ch12/instantiation_overhead_demo.py",
        "ch12/graph_replay_benchmark.py",
    )

    for path in paths:
        source = (REPO_ROOT / path).read_text(encoding="utf-8")
        assert "torch.inference_mode()" in source
        assert "torch.no_grad()" not in source

    instantiation_source = (
        REPO_ROOT / "ch12" / "instantiation_overhead_demo.py"
    ).read_text(encoding="utf-8")
    rebuild_section = instantiation_source.split("# Rebuild loop", maxsplit=1)[1]
    assert "with torch.inference_mode(), torch.cuda.graph(g2):" in rebuild_section


def test_ch12_bias_relu_residual_batches_verification_metric_reads() -> None:
    source = (REPO_ROOT / "ch12" / "bias_relu_residual_fusion_benchmark.py").read_text(
        encoding="utf-8"
    )
    correctness_section = source.split("# Correctness check", maxsplit=1)[1].split(
        "baseline_ms = time_kernel",
        maxsplit=1,
    )[0]

    assert "max_abs_baseline, max_abs_fused, l2_baseline, l2_fused = torch.stack(" in source
    assert "torch.linalg.vector_norm(baseline_error)" in source
    assert ".item()" not in correctness_section


def test_ch13_precisionfp8_pad_inner_runs_single_forward_per_timed_iteration() -> None:
    baseline_source = (REPO_ROOT / "ch13" / "baseline_precisionfp8_pad_inner.py").read_text(encoding="utf-8")
    optimized_source = (REPO_ROOT / "ch13" / "optimized_precisionfp8_pad_inner.py").read_text(encoding="utf-8")
    baseline_benchmark = baseline_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    optimized_benchmark = optimized_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert baseline_benchmark.count("self.model(") == 1
    assert optimized_benchmark.count("self.model(") == 1


def test_ch13_matmul_and_regional_compile_keep_precision_fixed_across_pairs() -> None:
    baseline_matmul = (REPO_ROOT / "ch13" / "baseline_matmul_pytorch.py").read_text(encoding="utf-8")
    optimized_matmul = (REPO_ROOT / "ch13" / "optimized_matmul_pytorch.py").read_text(encoding="utf-8")
    baseline_regional = (REPO_ROOT / "ch13" / "baseline_regional_compile.py").read_text(encoding="utf-8")
    optimized_regional = (REPO_ROOT / "ch13" / "optimized_regional_compile.py").read_text(encoding="utf-8")

    assert "dtype=torch.float32" not in baseline_matmul
    assert "dtype=torch.float16" in baseline_matmul
    assert "dtype=torch.float16" in optimized_matmul
    assert "self.compiled_model = torch.compile(" in baseline_regional
    assert "dtype=torch.bfloat16" in baseline_regional
    assert "dtype=torch.bfloat16" in optimized_regional


def test_ch10_tcgen05_warp_specialized_kernel_uses_direct_epilogue_copy() -> None:
    optimized_wrapper = (REPO_ROOT / "ch10" / "optimized_tcgen05_warp_specialization.py").read_text(
        encoding="utf-8"
    )
    kernel_source = (REPO_ROOT / "ch10" / "tcgen05_warp_specialized.cu").read_text(encoding="utf-8")

    assert "matmul_tcgen05_warp_specialized(self.matrix_a, self.matrix_b)" in optimized_wrapper
    assert "torch::zeros({m, n}, options)" not in kernel_source
    assert "axpby(" not in kernel_source
    assert "copy(tDrAcc, tDgD);" in kernel_source


def test_moe_cuda_graphs_journey_uses_real_graph_capture_and_correct_leveling() -> None:
    benchmark_source = (REPO_ROOT / "labs" / "moe_optimization_journey" / "moe_benchmark.py").read_text(
        encoding="utf-8"
    )
    model_source = (REPO_ROOT / "labs" / "moe_optimization_journey" / "moe_model.py").read_text(
        encoding="utf-8"
    )
    cuda_graph_source = (REPO_ROOT / "labs" / "moe_optimization_journey" / "level5_cudagraphs.py").read_text(
        encoding="utf-8"
    )
    optimized_entrypoint = (
        REPO_ROOT / "labs" / "moe_optimization_journey" / "optimized_moe_cuda_graphs.py"
    ).read_text(
        encoding="utf-8"
    )
    optimized_main_entrypoint = (
        REPO_ROOT / "labs" / "moe_optimization_journey" / "optimized_moe.py"
    ).read_text(
        encoding="utf-8"
    )

    assert "self._cuda_graph = torch.cuda.CUDAGraph()" in model_source
    assert "self._cuda_graph.replay()" in model_source
    assert "with torch.inference_mode():" in model_source
    assert "with torch.no_grad():" not in model_source
    assert "self.output = logits[:, :1, : min(8, logits.shape[-1])]" in benchmark_source
    assert ".float().clone()" not in benchmark_source.split("def capture_verification_payload", maxsplit=1)[0]
    assert "Level6CUDAGraphs" in cuda_graph_source
    assert "LEVEL = 6" in cuda_graph_source
    assert "Level6CUDAGraphs" in optimized_entrypoint
    assert "Level7Compiled" in optimized_main_entrypoint


def test_persistent_decode_verification_clone_stays_out_of_hot_path() -> None:
    targets = [
        REPO_ROOT / "labs" / "persistent_decode" / "baseline_persistent_decode.py",
        REPO_ROOT / "labs" / "persistent_decode" / "optimized_persistent_decode_cuda.py",
        REPO_ROOT / "labs" / "persistent_decode" / "optimized_persistent_decode_graphs.py",
        REPO_ROOT / "labs" / "persistent_decode" / "optimized_persistent_decode_triton.py",
        REPO_ROOT / "labs" / "persistent_decode" / "baseline_tma_prefill_decode.py",
        REPO_ROOT / "labs" / "persistent_decode" / "optimized_tma_prefill_decode.py",
    ]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        benchmark_section = text.split("def capture_verification_payload", maxsplit=1)[0]
        assert ".float().clone()" not in benchmark_section
        assert ".detach().clone()" not in benchmark_section


def test_iteration_seed_and_clone_fixes_for_reviewed_pairs_remain_applied() -> None:
    baseline_pipeline = (REPO_ROOT / "ch10" / "baseline_pipeline_3stage.py").read_text(encoding="utf-8")
    optimized_pipeline = (REPO_ROOT / "ch10" / "optimized_pipeline_3stage.py").read_text(encoding="utf-8")
    baseline_gluon = (
        REPO_ROOT / "labs" / "flashattention_gluon" / "baseline_flashattention_gluon.py"
    ).read_text(encoding="utf-8")
    optimized_gluon = (
        REPO_ROOT / "labs" / "flashattention_gluon" / "optimized_flashattention_gluon.py"
    ).read_text(encoding="utf-8")
    blackwell = (REPO_ROOT / "labs" / "blackwell_matmul" / "blackwell_benchmarks.py").read_text(
        encoding="utf-8"
    )
    blackwell_tcgen05 = (
        REPO_ROOT / "labs" / "blackwell_matmul" / "optimized_blackwell_matmul_tcgen05.py"
    ).read_text(encoding="utf-8")
    baseline_double_buffer = (REPO_ROOT / "ch19" / "baseline_memory_double_buffering.py").read_text(
        encoding="utf-8"
    )
    optimized_double_buffer = (REPO_ROOT / "ch19" / "optimized_memory_double_buffering.py").read_text(
        encoding="utf-8"
    )
    baseline_rack_prep = (REPO_ROOT / "ch03" / "baseline_rack_prep.py").read_text(encoding="utf-8")
    optimized_rack_prep = (REPO_ROOT / "ch03" / "optimized_rack_prep.py").read_text(encoding="utf-8")
    baseline_pinned_prefetch_mlp = (
        REPO_ROOT / "ch03" / "baseline_pinned_prefetch_mlp.py"
    ).read_text(encoding="utf-8")
    optimized_pinned_prefetch_mlp = (
        REPO_ROOT / "ch03" / "optimized_pinned_prefetch_mlp.py"
    ).read_text(encoding="utf-8")
    baseline_gemm = (REPO_ROOT / "ch03" / "baseline_gemm.py").read_text(encoding="utf-8")

    for source in (baseline_pipeline, optimized_pipeline, baseline_gluon, optimized_gluon):
        assert "iterations=10" in source
        assert "warmup=5" in source

    assert "torch.manual_seed(42)" in blackwell
    assert "torch.cuda.manual_seed_all(42)" in blackwell
    assert "with torch.inference_mode():" in blackwell
    assert "with torch.no_grad():" not in blackwell
    assert "with torch.inference_mode():" in blackwell_tcgen05
    assert "with torch.no_grad():" not in blackwell_tcgen05
    for source in (baseline_double_buffer, optimized_double_buffer):
        assert "torch.manual_seed(42)" in source
        assert "torch.cuda.manual_seed_all(42)" in source

    assert "host_template.pin_memory()" in optimized_rack_prep
    assert "host_template.clone().pin_memory()" in optimized_rack_prep
    for source, label in (
        (baseline_rack_prep, "baseline_rack_prep"),
        (optimized_rack_prep, "optimized_rack_prep"),
        (baseline_pinned_prefetch_mlp, "baseline_pinned_prefetch_mlp"),
        (optimized_pinned_prefetch_mlp, "optimized_pinned_prefetch_mlp"),
        (baseline_gemm, "baseline_gemm"),
    ):
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]
        assert f'with self._nvtx_range("{label}"):' in benchmark_section
        assert "get_nvtx_enabled(" not in benchmark_section
        assert "with nvtx_range(" not in benchmark_section
        assert "from core.profiling.nvtx_helper" not in source


def test_ch15_optimized_monolithic_uses_token_equivalent_decode_steps() -> None:
    common_source = (REPO_ROOT / "ch15" / "inference_monolithic_common.py").read_text(encoding="utf-8")
    optimized_source = (REPO_ROOT / "ch15" / "optimized_inference_monolithic.py").read_text(encoding="utf-8")

    assert "def decode_step(" in common_source
    assert "torch.relu_(layer(x))" in common_source
    assert "torch.relu(layer(x))" not in common_source
    assert "with torch.inference_mode():" in optimized_source
    assert "for token_idx in range(num_tokens):" in optimized_source
    assert "buffer[:, token_idx : token_idx + 1, :] = current" in optimized_source
    assert "self._compiled_decode = torch.compile(_full_decode, mode=\"reduce-overhead\")" in optimized_source
    assert "self.output = self._compiled_decode(kv_cache)" in optimized_source
    assert "self.output = self.model.decode(kv_cache, num_tokens=self.num_tokens)" not in optimized_source


def test_decode_handoff_benchmarks_do_not_allocate_placeholder_outputs_in_hot_path() -> None:
    paths = (
        "ch15/baseline_inference_monolithic.py",
        "ch15/disaggregated_inference_single_common.py",
        "ch15/prefill_decode_disagg_common.py",
        "ch15/baseline_disaggregated_inference_multigpu.py",
        "ch17/prefill_decode_disagg_single_common.py",
        "ch17/prefill_decode_disagg_multigpu_common.py",
        "labs/cache_aware_disagg_inference/cache_aware_disagg_common.py",
        "labs/cache_aware_disagg_inference/cache_aware_disagg_multigpu_common.py",
    )

    for path in paths:
        source = (REPO_ROOT / path).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload",
            maxsplit=1,
        )[0]
        assert "torch.empty(0)" not in benchmark_section
        assert "not initialized" in benchmark_section


def test_ch15_baseline_monolithic_uses_harness_timing_not_per_token_cuda_events() -> None:
    source = (REPO_ROOT / "ch15" / "baseline_inference_monolithic.py").read_text(encoding="utf-8")
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def finalize_iteration_metrics", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]

    assert "torch.cuda.Event" not in source
    assert "with torch.inference_mode():" in benchmark_section
    assert "torch.cat(" not in benchmark_section
    assert "self._last_decoded_tokens = [torch.empty(0) for _ in range(self.num_tokens)]" in source
    assert "decoded_tokens = self._last_decoded_tokens" in benchmark_section
    assert "token_idx = 0" in benchmark_section
    assert "decoded_tokens[token_idx] = decode_state" in benchmark_section
    assert "token_idx += 1" in benchmark_section
    assert "self._last_decoded_tokens = decoded_tokens" in benchmark_section
    assert "decoded_tokens = []" not in benchmark_section
    assert "decoded_tokens.append(" not in benchmark_section
    assert "self.output = torch.cat(self._last_decoded_tokens, dim=1)" in capture_section
    assert "self._last_elapsed_ms" in source
    assert "finalize_iteration_metrics" in source
    assert "self.model.decode(decode_state, num_tokens=1)" in source


def test_ch17_monolithic_decode_fast_paths_single_token() -> None:
    source = (REPO_ROOT / "ch17" / "prefill_decode_disagg_monolithic_common.py").read_text(
        encoding="utf-8"
    )
    decode_section = source.split("def decode", maxsplit=1)[1]

    assert "if token_count == 1:" in decode_section
    assert "return x" in decode_section
    assert "output = kv_cache.new_empty(" in decode_section
    assert "output[:, token_idx : token_idx + 1, :].copy_(x)" in decode_section
    assert "torch.cat(outputs" not in decode_section


def test_ch03_pageable_copy_is_not_marked_informational() -> None:
    assert "pageable_copy" not in INFORMATIONAL_BENCHMARKS.get("ch03", set())


def test_clean_all_benchmark_pairs_tracker_is_rebaselined() -> None:
    tracker = REPO_ROOT / ".cursor" / "plans" / "clean_all_benchmark_pairs_6db4c258.plan.md"
    text = tracker.read_text(encoding="utf-8")

    assert "status: pending" not in text
    assert "Rebaselined on 2026-03-16 against current repo truth" in text
    assert "tests/test_benchmark_hygiene_regressions.py" in text


def test_run_benchmarks_reaps_orphaned_benchmark_processes_from_older_runs() -> None:
    with tempfile.TemporaryDirectory() as proc_dir:
        proc_root = Path(proc_dir)

        orphan_dir = proc_root / "1234"
        orphan_dir.mkdir()
        (orphan_dir / "stat").write_text("1234 (python3) S 1 0 0 0 0\n", encoding="utf-8")
        (orphan_dir / "environ").write_bytes(
            b"AISP_BENCHMARK_OWNER_RUN_ID=old-run\0PWD=" + str(REPO_ROOT).encode("utf-8") + b"\0"
        )
        (orphan_dir / "cmdline").write_bytes(b"/usr/bin/python3\0-m\0core.harness.isolated_runner\0")

        live_parent_dir = proc_root / "4321"
        live_parent_dir.mkdir()
        (live_parent_dir / "stat").write_text("4321 (python3) S 2222 0 0 0 0\n", encoding="utf-8")
        (live_parent_dir / "environ").write_bytes(
            b"AISP_BENCHMARK_OWNER_RUN_ID=other-run\0PWD=" + str(REPO_ROOT).encode("utf-8") + b"\0"
        )
        (live_parent_dir / "cmdline").write_bytes(b"/usr/bin/python3\0-m\0core.harness.isolated_runner\0")

        parent_dir = proc_root / "2222"
        parent_dir.mkdir()
        (parent_dir / "stat").write_text("2222 (python3) S 1 0 0 0 0\n", encoding="utf-8")

        current_run_dir = proc_root / "9999"
        current_run_dir.mkdir()
        (current_run_dir / "stat").write_text("9999 (python3) S 1 0 0 0 0\n", encoding="utf-8")
        (current_run_dir / "environ").write_bytes(
            b"AISP_BENCHMARK_OWNER_RUN_ID=current-run\0PWD=" + str(REPO_ROOT).encode("utf-8") + b"\0"
        )
        (current_run_dir / "cmdline").write_bytes(b"/usr/bin/python3\0-m\0core.harness.isolated_runner\0")

        stale = _collect_stale_benchmark_orphan_pids(
            current_run_id="current-run",
            repo_root=REPO_ROOT,
            proc_root=proc_root,
        )

        assert stale == [1234]


def test_run_benchmarks_identifies_detached_current_run_processes_by_owner_pid() -> None:
    with tempfile.TemporaryDirectory() as proc_dir:
        proc_root = Path(proc_dir)

        owner_dir = proc_root / "555"
        owner_dir.mkdir()
        (owner_dir / "stat").write_text("555 (python3) S 1 0 0 0 0\n", encoding="utf-8")

        orphan_dir = proc_root / "1234"
        orphan_dir.mkdir()
        (orphan_dir / "stat").write_text("1234 (python3) S 1 0 0 0 0\n", encoding="utf-8")
        (orphan_dir / "environ").write_bytes(
            b"AISP_BENCHMARK_OWNER_RUN_ID=current-run\0"
            b"AISP_BENCHMARK_OWNER_PID=555\0"
            b"PWD=" + str(REPO_ROOT).encode("utf-8") + b"\0"
        )
        (orphan_dir / "cmdline").write_bytes(b"/usr/bin/python3\0-m\0core.harness.isolated_runner\0")

        descendant_dir = proc_root / "1235"
        descendant_dir.mkdir()
        (descendant_dir / "stat").write_text("1235 (python3) S 555 0 0 0 0\n", encoding="utf-8")
        (descendant_dir / "environ").write_bytes(
            b"AISP_BENCHMARK_OWNER_RUN_ID=current-run\0"
            b"AISP_BENCHMARK_OWNER_PID=555\0"
            b"PWD=" + str(REPO_ROOT).encode("utf-8") + b"\0"
        )
        (descendant_dir / "cmdline").write_bytes(b"/usr/bin/python3\0-m\0core.harness.isolated_runner\0")

        other_owner_dir = proc_root / "1236"
        other_owner_dir.mkdir()
        (other_owner_dir / "stat").write_text("1236 (python3) S 1 0 0 0 0\n", encoding="utf-8")
        (other_owner_dir / "environ").write_bytes(
            b"AISP_BENCHMARK_OWNER_RUN_ID=current-run\0"
            b"AISP_BENCHMARK_OWNER_PID=999\0"
            b"PWD=" + str(REPO_ROOT).encode("utf-8") + b"\0"
        )
        (other_owner_dir / "cmdline").write_bytes(b"/usr/bin/python3\0-m\0core.harness.isolated_runner\0")

        current_orphans = _collect_current_run_benchmark_orphan_pids(
            current_run_id="current-run",
            current_owner_pid=555,
            repo_root=REPO_ROOT,
            proc_root=proc_root,
        )

        assert current_orphans == [1234]


def test_run_benchmarks_identifies_detached_current_run_processes_by_cmdline_owner_markers() -> None:
    with tempfile.TemporaryDirectory() as proc_dir:
        proc_root = Path(proc_dir)

        owner_dir = proc_root / "555"
        owner_dir.mkdir()
        (owner_dir / "stat").write_text("555 (python3) S 1 0 0 0 0\n", encoding="utf-8")

        orphan_dir = proc_root / "1234"
        orphan_dir.mkdir()
        (orphan_dir / "stat").write_text("1234 (python3) S 1 0 0 0 0\n", encoding="utf-8")
        (orphan_dir / "environ").write_bytes(b"PWD=" + str(REPO_ROOT).encode("utf-8") + b"\0")
        (orphan_dir / "cmdline").write_bytes(
            b"/usr/bin/python3\0-m\0core.profiling.nsys_capture_helper\0"
            b"--aisp-owner-run-id\0current-run\0--aisp-owner-pid\0"
            b"555\0"
        )

        current_orphans = _collect_current_run_benchmark_orphan_pids(
            current_run_id="current-run",
            current_owner_pid=555,
            repo_root=REPO_ROOT,
            proc_root=proc_root,
        )

        assert current_orphans == [1234]


def test_run_benchmarks_reaps_orphaned_benchmark_processes_from_older_runs_via_cmdline_marker() -> None:
    with tempfile.TemporaryDirectory() as proc_dir:
        proc_root = Path(proc_dir)

        orphan_dir = proc_root / "1234"
        orphan_dir.mkdir()
        (orphan_dir / "stat").write_text("1234 (python3) S 1 0 0 0 0\n", encoding="utf-8")
        (orphan_dir / "environ").write_bytes(b"PWD=" + str(REPO_ROOT).encode("utf-8") + b"\0")
        (orphan_dir / "cmdline").write_bytes(
            b"/usr/bin/python3\0-m\0core.profiling.nsys_capture_helper\0"
            b"--aisp-owner-run-id\0old-run\0"
        )

        stale = _collect_stale_benchmark_orphan_pids(
            current_run_id="current-run",
            repo_root=REPO_ROOT,
            proc_root=proc_root,
        )

        assert stale == [1234]


def test_run_benchmarks_reaps_current_run_descendants() -> None:
    process = subprocess.Popen(
        ["python", "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        time.sleep(0.2)
        assert process.poll() is None

        _reap_run_descendants("unit_test_descendant_cleanup", grace_seconds=0.2)

        deadline = time.time() + 5.0
        while process.poll() is None and time.time() < deadline:
            time.sleep(0.05)

        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


def test_benchmark_leftover_cleanup_does_not_kill_unmarked_current_descendants() -> None:
    process = subprocess.Popen(
        ["python", "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        time.sleep(0.2)
        assert process.poll() is None

        _reap_benchmark_process_leftovers(
            "unit_test_preserve_unmarked_descendant",
            current_run_id="unit-test-run",
            current_owner_pid=999999,
            repo_root=REPO_ROOT,
        )

        time.sleep(0.5)
        assert process.poll() is None
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


def test_ch12_kernel_fusion_variants_publish_static_input_signatures_without_execution() -> None:
    for module_name in KERNEL_FUSION_SIGNATURE_MODULES:
        module = importlib.import_module(module_name)
        benchmark = module.get_benchmark()
        signature = coerce_input_signature(benchmark.get_input_signature())

        assert signature.batch_size == 16_000_000
        assert signature.dtypes["workload"] == "float32"
        assert signature.shapes["workload"] == (16_000_000, 10)


def test_timeout_prone_pairs_publish_static_input_signatures_without_execution() -> None:
    for module_name, expected_batch_size, expected_shape, expected_dtype in TIMEOUT_PRONE_SIGNATURE_CASES:
        module = importlib.import_module(module_name)
        benchmark = module.get_benchmark()
        signature = coerce_input_signature(benchmark.get_input_signature())

        assert signature.batch_size == expected_batch_size
        assert signature.dtypes["workload"] == expected_dtype
        assert signature.shapes["workload"] == expected_shape
