from __future__ import annotations

import inspect

import torch

import ch19.baseline_fp4_weight_quantization as baseline_fp4
import ch19.baseline_mxfp8_moe as baseline_mxfp8_moe
import ch13.compiled_autograd as compiled_autograd_demo
import ch19.mxfp8_moe_common as mxfp8_moe_common
import ch19.native_fp4_quantization as native_fp4
import ch19.native_fp6_quantization as native_fp6
import ch19.optimized_mxfp8_moe as optimized_mxfp8_moe
import ch19.optimized_fp4_weight_quantization as optimized_fp4
from ch13.optimized_autograd_standard import OptimizedAutogradCompiledBenchmark
from ch19.mxfp8_moe_common import bucket_by_expert, restore_bucketed_reduce
from ch19.native_fp6_quantization import FP6Tensor


def test_optimized_autograd_standard_uses_wall_clock_with_full_sync() -> None:
    bench = OptimizedAutogradCompiledBenchmark()
    config = bench.get_config()
    assert config.timing_method == "wall_clock"
    assert config.full_device_sync is True


def test_optimized_autograd_standard_declares_capture_stream_when_present() -> None:
    bench = OptimizedAutogradCompiledBenchmark()
    sentinel = object()
    bench.capture_stream = sentinel  # type: ignore[assignment]
    assert bench.get_custom_streams() == [sentinel]


def test_optimized_autograd_standard_skips_post_capture_output_zero_fill() -> None:
    setup_source = inspect.getsource(OptimizedAutogradCompiledBenchmark.setup)
    train_step_source = inspect.getsource(OptimizedAutogradCompiledBenchmark._train_step)

    assert "self.output_buffer.zero_()" not in setup_source
    assert "self.output_buffer.copy_(outputs)" in train_step_source


def test_compiled_autograd_demo_uses_current_stream_and_monotonic_cpu_timing() -> None:
    for fn in (
        compiled_autograd_demo.benchmark_standard_autograd,
        compiled_autograd_demo.benchmark_compiled_autograd,
        compiled_autograd_demo.benchmark_forward_and_backward_compiled,
    ):
        source = inspect.getsource(fn)
        assert source.count("torch.cuda.Event(enable_timing=True)") == 2
        assert "current_stream = torch.cuda.current_stream()" in source
        assert "start.record(current_stream)" in source
        assert "end.record(current_stream)" in source
        assert "start.record()" not in source
        assert "end.record()" not in source
        assert "start_time = time.perf_counter()" in source
        assert "elapsed_ms = (time.perf_counter() - start_time) / num_iters * 1000" in source
        assert "time.time()" not in source


def test_restore_bucketed_reduce_casts_weighted_output_and_reuses_buffer() -> None:
    source = inspect.getsource(mxfp8_moe_common.restore_bucketed_reduce)

    assert "weighted_out: Optional[torch.Tensor] = None" in source
    assert "bucket_token_ids_expanded: Optional[torch.Tensor] = None" in source
    assert "weights_expanded: Optional[torch.Tensor] = None" in source
    assert "weight_out_expanded: Optional[torch.Tensor] = None" in source
    assert "bucket_token_ids_expanded = bucket_token_ids.unsqueeze(-1).expand_as(output)" in source
    assert "weights_expanded = weights.unsqueeze(-1)" in source
    assert "weight_out_expanded = weight_out.unsqueeze(-1)" in source
    assert "weighted_output = output.to(dtype=out.dtype, copy=True)" in source
    assert "weighted_output.mul_(weights_expanded)" in source
    assert "torch.mul(output, weights_expanded, out=weighted_output)" in source
    assert "out.scatter_add_(0, bucket_token_ids_expanded, weighted_output)" in source
    assert "out.div_(weight_out_expanded)" in source

    output = torch.tensor(
        [
            [2.0, 4.0, 6.0],
            [6.0, 8.0, 10.0],
            [1.0, 3.0, 5.0],
        ],
        dtype=torch.bfloat16,
    )
    bucket_token_ids = torch.tensor([0, 0, 1], dtype=torch.int64)
    weights = torch.tensor([0.25, 0.75, 1.0], dtype=torch.float16)
    out = torch.empty((2, 3), dtype=torch.float16)
    weight_out = torch.empty((2,), dtype=torch.float16)
    weighted_out = torch.empty_like(output, dtype=torch.float16)
    scatter_index = bucket_token_ids.unsqueeze(-1).expand_as(output)
    weight_factors = weights.unsqueeze(-1)
    weight_out_factors = weight_out.unsqueeze(-1)

    restored = restore_bucketed_reduce(
        output,
        bucket_token_ids,
        num_tokens=2,
        weights=weights,
        out=out,
        weight_out=weight_out,
        weighted_out=weighted_out,
        bucket_token_ids_expanded=scatter_index,
        weights_expanded=weight_factors,
        weight_out_expanded=weight_out_factors,
    )

    expected = torch.tensor(
        [
            [5.0, 7.0, 9.0],
            [1.0, 3.0, 5.0],
        ],
        dtype=torch.float16,
    )
    assert restored.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        weighted_out,
        torch.tensor(
            [
                [0.5, 1.0, 1.5],
                [4.5, 6.0, 7.5],
                [1.0, 3.0, 5.0],
            ],
            dtype=torch.float16,
        ),
    )
    assert torch.equal(weight_out, torch.tensor([1.0, 1.0], dtype=torch.float16))
    assert torch.allclose(restored, expected, atol=1e-3, rtol=0.0)


def test_bucket_by_expert_sorts_once_and_preserves_metadata() -> None:
    source = inspect.getsource(mxfp8_moe_common.bucket_by_expert)

    assert "torch.argsort(flat_assignments)" in source
    assert "counts_tensor = torch.bincount(flat_assignments, minlength=num_experts)" in source
    assert "counts_host = counts_tensor.detach().cpu()" in source
    assert "for expert in range(num_experts):" in source
    assert "count = int(counts_host[expert])" in source
    assert "m_splits.append(count)" in source
    assert "counts_tensor.detach().cpu().tolist()" not in source
    assert "expert_range = torch.arange(num_experts, device=tokens.device, dtype=torch.int64)" in source
    assert "expert_order_tensor = expert_range[counts_tensor[:num_experts] > 0]" in source
    assert "torch.tensor(expert_order_list" not in source
    assert ".nonzero(" not in source
    assert "torch.cat(" not in source

    tokens = torch.arange(20, dtype=torch.float32).view(5, 4)
    assignments = torch.tensor([2, 0, 2, 1, 0], dtype=torch.int64)
    token_ids = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64)

    bucketed, m_splits, gather_index, expert_order, bucket_token_ids, expert_order_list = bucket_by_expert(
        tokens,
        assignments,
        num_experts=4,
        token_ids=token_ids,
        return_expert_order_list=True,
    )

    torch.testing.assert_close(assignments.index_select(0, gather_index), torch.tensor([0, 0, 1, 2, 2]))
    torch.testing.assert_close(bucketed, tokens.index_select(0, gather_index))
    torch.testing.assert_close(bucket_token_ids, token_ids.index_select(0, gather_index))
    torch.testing.assert_close(expert_order, torch.tensor([0, 1, 2], dtype=torch.int64))
    assert m_splits == [2, 1, 2]
    assert expert_order_list == [0, 1, 2]

    route_assignments = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.int64)
    route_token_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
    route_bucketed, _, route_gather, _, route_bucket_token_ids = bucket_by_expert(
        tokens[:3],
        route_assignments,
        num_experts=2,
        token_ids=route_token_ids,
    )

    torch.testing.assert_close(
        route_bucketed,
        tokens[:3].index_select(0, route_token_ids.index_select(0, route_gather)),
    )
    torch.testing.assert_close(route_bucket_token_ids, route_token_ids.index_select(0, route_gather))


def test_optimized_mxfp8_moe_reuses_token_ids_and_keeps_reorder_on_device() -> None:
    module_source = inspect.getsource(optimized_mxfp8_moe)
    class_source = inspect.getsource(optimized_mxfp8_moe.OptimizedMXFP8MoEBenchmark)
    setup_source = inspect.getsource(optimized_mxfp8_moe.OptimizedMXFP8MoEBenchmark.setup)
    supergroup_source = inspect.getsource(optimized_mxfp8_moe.OptimizedMXFP8MoEBenchmark._supergroup_tokens)
    forward_source = inspect.getsource(optimized_mxfp8_moe.OptimizedMXFP8MoEBenchmark._forward_grouped)
    graph_source = inspect.getsource(optimized_mxfp8_moe.OptimizedMXFP8MoEBenchmark._capture_graph)

    assert "def _flat_topk_token_ids" in module_source
    assert 'token_ids.div_(top_k, rounding_mode="floor")' in module_source
    assert "self._bucket_token_scatter_index: Optional[torch.Tensor] = None" in class_source
    assert "self._gating_weight_factors: Optional[torch.Tensor] = None" in class_source
    assert "self._restored_weight_factors: Optional[torch.Tensor] = None" in class_source
    assert "self._graph_weight_factors: Optional[torch.Tensor] = None" in class_source
    assert "repeat_interleave(" not in setup_source
    assert "with torch.inference_mode():" in setup_source
    assert "with torch.no_grad():" not in setup_source
    assert "expanded_inputs = self.inputs.index_select(0, token_ids)" in setup_source
    assert "self._bucket_token_scatter_index = bucket_token_ids.unsqueeze(-1).expand(" in setup_source
    assert "self._gating_weight_factors = gating_weights.unsqueeze(-1)" in setup_source
    assert "self._restored_weight_factors = self._restored_weight.unsqueeze(-1)" in setup_source
    assert "expert_order[idx].item()" not in supergroup_source
    assert "expert_order.index_select(0, order_tensor)" in supergroup_source
    assert "row_order = torch.empty_like(base_rows)" in supergroup_source
    assert "bucketed.index_select(0, row_order)" in supergroup_source
    assert "bucket_indices.index_select(0, row_order)" in supergroup_source
    assert "bucket_token_ids.index_select(0, row_order)" in supergroup_source
    assert "gating_weights.index_select(0, row_order)" in supergroup_source
    assert "self._weighted_out = torch.empty((bucketed.shape[0], self.ffn_dim)" in setup_source
    assert "weighted_out=self._weighted_out" in forward_source
    assert "bucket_token_ids_expanded=self._bucket_token_scatter_index" in forward_source
    assert "weights_expanded=self._gating_weight_factors" in forward_source
    assert "weight_out_expanded=self._restored_weight_factors" in forward_source
    assert "weighted_out=self._graph_weighted_out" in graph_source
    assert "self._graph_weight_factors = self._graph_weight.unsqueeze(-1)" in graph_source
    assert "bucket_token_ids_expanded=self._bucket_token_scatter_index" in graph_source
    assert "weights_expanded=self._gating_weight_factors" in graph_source
    assert "weight_out_expanded=self._graph_weight_factors" in graph_source
    assert "reordered_inputs" not in supergroup_source
    assert "torch.cat(reordered" not in supergroup_source

    torch.testing.assert_close(
        optimized_mxfp8_moe._flat_topk_token_ids(3, 1, torch.device("cpu")),
        torch.tensor([0, 1, 2], dtype=torch.int64),
    )
    torch.testing.assert_close(
        optimized_mxfp8_moe._flat_topk_token_ids(3, 2, torch.device("cpu")),
        torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),
    )

    bench = optimized_mxfp8_moe.OptimizedMXFP8MoEBenchmark()
    bucketed = torch.arange(6, dtype=torch.float32).view(6, 1)
    bucket_indices = torch.arange(10, 16, dtype=torch.int64)
    expert_order = torch.tensor([7, 8, 9], dtype=torch.int64)
    bucket_token_ids = torch.arange(20, 26, dtype=torch.int64)
    gating_weights = torch.arange(30, 36, dtype=torch.float32)

    (
        new_bucketed,
        new_splits,
        new_indices,
        new_order,
        new_token_ids,
        new_weights,
    ) = bench._supergroup_tokens(
        bucketed,
        [2, 3, 1],
        bucket_indices,
        expert_order,
        bucket_token_ids,
        gating_weights,
    )

    expected_rows = torch.tensor([2, 3, 4, 0, 1, 5], dtype=torch.int64)
    assert new_splits == [3, 2, 1]
    torch.testing.assert_close(new_bucketed, bucketed.index_select(0, expected_rows))
    torch.testing.assert_close(new_indices, bucket_indices.index_select(0, expected_rows))
    torch.testing.assert_close(new_order, torch.tensor([8, 7, 9], dtype=torch.int64))
    torch.testing.assert_close(new_token_ids, bucket_token_ids.index_select(0, expected_rows))
    torch.testing.assert_close(new_weights, gating_weights.index_select(0, expected_rows))


def test_baseline_mxfp8_moe_reuses_bucketed_output_buffer() -> None:
    source = inspect.getsource(baseline_mxfp8_moe.BaselineMXFP8MoEBenchmark)
    setup_source = inspect.getsource(baseline_mxfp8_moe.BaselineMXFP8MoEBenchmark.setup)
    run_source = inspect.getsource(baseline_mxfp8_moe.BaselineMXFP8MoEBenchmark._run_naive)
    teardown_source = inspect.getsource(baseline_mxfp8_moe.BaselineMXFP8MoEBenchmark.teardown)

    assert "self._bucketed_out: Optional[torch.Tensor] = None" in source
    assert "self._bucketed_out = torch.empty_like(self._restored_out)" in setup_source
    assert "self._bucketed_out.narrow(0, offset, m).copy_(" in run_source
    assert "restore_bucketed(\n            self._bucketed_out," in run_source
    assert "outputs: List[torch.Tensor]" not in run_source
    assert "outputs.append(" not in run_source
    assert "torch.cat(outputs" not in run_source
    assert "self._bucketed_out = None" in teardown_source


def test_mxfp8_moe_benchmark_wrappers_use_inference_mode() -> None:
    baseline_setup = inspect.getsource(baseline_mxfp8_moe.BaselineMXFP8MoEBenchmark.setup)
    baseline_benchmark = inspect.getsource(baseline_mxfp8_moe.BaselineMXFP8MoEBenchmark.benchmark_fn)
    baseline_capture = inspect.getsource(
        baseline_mxfp8_moe.BaselineMXFP8MoEBenchmark.capture_verification_payload
    )
    optimized_setup = inspect.getsource(optimized_mxfp8_moe.OptimizedMXFP8MoEBenchmark.setup)
    optimized_benchmark = inspect.getsource(optimized_mxfp8_moe.OptimizedMXFP8MoEBenchmark.benchmark_fn)
    optimized_capture = inspect.getsource(
        optimized_mxfp8_moe.OptimizedMXFP8MoEBenchmark.capture_verification_payload
    )

    for setup_source, benchmark_source, capture_source, label in (
        (baseline_setup, baseline_benchmark, baseline_capture, "mxfp8_moe_baseline"),
        (optimized_setup, optimized_benchmark, optimized_capture, "mxfp8_moe_optimized"),
    ):
        assert f'with torch.inference_mode(), nvtx_range("{label}", enable=self._enable_nvtx):' in benchmark_source
        assert "torch.no_grad()" not in benchmark_source
        assert "self._enable_nvtx = get_nvtx_enabled(config) if config else False" in setup_source
        assert "self._payload_parameter_count = self.weights.numel()" in setup_source
        assert "get_config()" not in benchmark_source
        assert "get_nvtx_enabled(" not in benchmark_source
        assert "parameter_count=self._payload_parameter_count" in capture_source
        assert "self.weights.numel()" not in capture_source


def test_native_fp6_quantization_avoids_tensor_bool_scale_branch() -> None:
    source = inspect.getsource(FP6Tensor._quantize_fp6)
    forward_source = inspect.getsource(native_fp6.FP6Linear.forward)

    assert "if abs_max > 0" not in source
    assert "torch.where(" not in source
    assert "torch.ones_like(abs_max)" not in source
    assert "scale.masked_fill_(abs_max == 0, 1.0)" in source
    assert "is_blackwell()" not in forward_source
    assert "hasattr(torch, '_scaled_mm')" not in forward_source
    assert "return F.linear(x, weight, self.bias)" in forward_source

    data = torch.zeros(8, dtype=torch.float16)
    fp6 = FP6Tensor(data)

    assert fp6.scales.device == data.device
    torch.testing.assert_close(fp6.scales, torch.ones_like(fp6.scales))


def test_native_fp4_fp6_demo_timing_uses_cuda_events() -> None:
    for module in (native_fp4, native_fp6):
        source = inspect.getsource(module._benchmark_forward)
        assert source.count("torch.cuda.Event(enable_timing=True)") == 2
        assert "current_stream = torch.cuda.current_stream()" in source
        assert "start.record(current_stream)" in source
        assert "end.record(current_stream)" in source
        assert "start.record()" not in source
        assert "end.record()" not in source
        assert "start.elapsed_time(end) / (count * 1000.0)" in source
        assert "time.time()" not in source

    fp4_demo = inspect.getsource(native_fp4.benchmark_fp4)
    fp6_demo = inspect.getsource(native_fp6.benchmark_fp6_vs_fp16)

    assert fp4_demo.count("_benchmark_forward(") == 4
    assert "time.perf_counter()" not in fp4_demo
    assert fp6_demo.count("_benchmark_forward(") == 2
    assert "time.time()" not in fp6_demo


def test_native_fp4_fp8_bridge_reuses_weight_activation_and_scale_buffers() -> None:
    source = inspect.getsource(native_fp4.FP4Linear)
    forward_fp8_source = inspect.getsource(native_fp4.FP4Linear._forward_fp8)

    assert "self.register_buffer('_weight_fp8_cache', None)" in source
    assert "self.register_buffer('_input_fp8_buffer'" in source
    assert "self.register_buffer('_fp8_scale_a'" in source
    assert "self.register_buffer('_fp8_scale_b'" in source
    assert "def _get_weight_fp8(self) -> torch.Tensor:" in source
    assert "def _activation_fp8_buffer(self, x_2d: torch.Tensor)" in source
    assert "or input_fp8.size(0) < x_2d.size(0)" in source
    assert "return input_fp8[: x_2d.size(0)]" in source
    assert "def _fp8_scale_buffers(self, device: torch.device)" in source
    assert "self._weight_fp8_cache = None" in inspect.getsource(native_fp4.FP4Linear.quantize)
    assert "self._weight_fp8_cache = None" in inspect.getsource(native_fp4.FP4Linear.clear_cache)
    assert "weight_fp8 = self._get_weight_fp8()" in forward_fp8_source
    assert "x_fp8 = self._activation_fp8_buffer(x_2d)" in forward_fp8_source
    assert "x_fp8.copy_(x_2d)" in forward_fp8_source
    assert "scale_a, scale_b = self._fp8_scale_buffers(x.device)" in forward_fp8_source
    assert ".to(torch.float8_e4m3fn)" not in forward_fp8_source
    assert "torch.ones(1, device=x.device, dtype=torch.float32)" not in forward_fp8_source

    for linear_cls in (native_fp4.FP4Linear, optimized_fp4.OptimizedFP4Linear):
        layer = linear_cls(8, 4, dtype=torch.float32, mode="fp8")
        first = layer._activation_fp8_buffer(torch.empty((6, 8), dtype=torch.float32))
        first_ptr = first.data_ptr()
        smaller = layer._activation_fp8_buffer(torch.empty((3, 8), dtype=torch.float32))

        assert smaller.shape == (3, 8)
        assert smaller.data_ptr() == first_ptr
        assert layer._input_fp8_buffer.shape == (6, 8)


def test_fp4_dequantization_decodes_signed_lookup_without_where() -> None:
    for module, function_name in (
        (baseline_fp4, "dequantize_fp4_baseline"),
        (optimized_fp4, "dequantize_fp4_optimized"),
        (native_fp4, "dequantize_from_fp4_packed"),
    ):
        source = inspect.getsource(getattr(module, function_name))
        assert "torch.where(signs.bool()" not in source
        assert "signs = (unpacked >> 3)" not in source
        assert "signed_fp4_vals = _fp4_signed_values_for(device)" in source
        assert "torch.stack([high, low]" not in source
        assert "unpacked = _unpack_fp4_codes(packed_data)" in source

    for module in (optimized_fp4, native_fp4):
        source = inspect.getsource(module._unpack_fp4_codes)
        assert "dtype=torch.long" in source
        assert "return unpacked.long()" not in source

    for module, function_name in (
        (optimized_fp4, "dequantize_fp4_optimized"),
        (native_fp4, "dequantize_from_fp4_packed"),
    ):
        source = inspect.getsource(getattr(module, function_name))
        assert "blocks.mul_(scales.unsqueeze(-1))" in source
        assert "flat = blocks.flatten()[:n_orig]" in source
        assert "dequantized = blocks * scales.unsqueeze(-1)" not in source

    packed = torch.tensor([(0 << 4) | 9, (2 << 4) | 15], dtype=torch.uint8)
    expected = torch.tensor([0.0, -0.5, 1.0, -6.0], dtype=torch.float32)
    unpacked_expected = torch.tensor([0, 9, 2, 15], dtype=torch.long)

    torch.testing.assert_close(baseline_fp4._unpack_fp4_codes(packed), unpacked_expected)
    torch.testing.assert_close(optimized_fp4._unpack_fp4_codes(packed), unpacked_expected)
    torch.testing.assert_close(native_fp4._unpack_fp4_codes(packed), unpacked_expected)

    baseline = baseline_fp4.dequantize_fp4_baseline(
        packed,
        torch.ones(1, dtype=torch.float32),
        torch.Size([4]),
        dtype=torch.float32,
    )
    optimized = optimized_fp4.dequantize_fp4_optimized(
        packed,
        torch.ones(1, dtype=torch.float32),
        torch.Size([4]),
        block_size=4,
        dtype=torch.float32,
    )
    native = native_fp4.dequantize_from_fp4_packed(
        packed,
        torch.ones(1, dtype=torch.float32),
        torch.Size([4]),
        block_size=4,
        dtype=torch.float32,
    )

    torch.testing.assert_close(baseline, expected)
    torch.testing.assert_close(optimized, expected)
    torch.testing.assert_close(native, expected)
