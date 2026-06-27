from __future__ import annotations

import inspect

import torch

import ch19.mxfp8_moe_common as mxfp8_moe_common
import ch19.optimized_mxfp8_moe as optimized_mxfp8_moe
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


def test_restore_bucketed_reduce_casts_weighted_output_and_reuses_buffer() -> None:
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

    restored = restore_bucketed_reduce(
        output,
        bucket_token_ids,
        num_tokens=2,
        weights=weights,
        out=out,
        weight_out=weight_out,
    )

    expected = torch.tensor(
        [
            [5.0, 7.0, 9.0],
            [1.0, 3.0, 5.0],
        ],
        dtype=torch.float16,
    )
    assert restored.data_ptr() == out.data_ptr()
    assert torch.equal(weight_out, torch.tensor([1.0, 1.0], dtype=torch.float16))
    assert torch.allclose(restored, expected, atol=1e-3, rtol=0.0)


def test_bucket_by_expert_sorts_once_and_preserves_metadata() -> None:
    source = inspect.getsource(mxfp8_moe_common.bucket_by_expert)

    assert "torch.argsort(flat_assignments)" in source
    assert "torch.bincount(flat_assignments, minlength=num_experts).detach().cpu().tolist()" in source
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


def test_optimized_mxfp8_moe_reuses_token_ids_and_keeps_reorder_on_device() -> None:
    module_source = inspect.getsource(optimized_mxfp8_moe)
    setup_source = inspect.getsource(optimized_mxfp8_moe.OptimizedMXFP8MoEBenchmark.setup)
    supergroup_source = inspect.getsource(optimized_mxfp8_moe.OptimizedMXFP8MoEBenchmark._supergroup_tokens)

    assert "def _flat_topk_token_ids" in module_source
    assert 'token_ids.div_(top_k, rounding_mode="floor")' in module_source
    assert "repeat_interleave(" not in setup_source
    assert "expanded_inputs = self.inputs.index_select(0, token_ids)" in setup_source
    assert "expert_order[idx].item()" not in supergroup_source
    assert "expert_order.index_select(0, order_tensor)" in supergroup_source

    torch.testing.assert_close(
        optimized_mxfp8_moe._flat_topk_token_ids(3, 1, torch.device("cpu")),
        torch.tensor([0, 1, 2], dtype=torch.int64),
    )
    torch.testing.assert_close(
        optimized_mxfp8_moe._flat_topk_token_ids(3, 2, torch.device("cpu")),
        torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),
    )


def test_native_fp6_quantization_avoids_tensor_bool_scale_branch() -> None:
    source = inspect.getsource(FP6Tensor._quantize_fp6)

    assert "if abs_max > 0" not in source
    assert "torch.where(abs_max > 0, abs_max / 16.0, torch.ones_like(abs_max))" in source

    data = torch.zeros(8, dtype=torch.float16)
    fp6 = FP6Tensor(data)

    assert fp6.scales.device == data.device
    torch.testing.assert_close(fp6.scales, torch.ones_like(fp6.scales))
