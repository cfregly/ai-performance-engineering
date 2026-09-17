"""Bit-format, swizzle-address and actual fused-backend correctness tests."""

import math

import pytest
import torch

from labs.nvfp4_quantization.reference import (
    dequantize,
    encode_e2m1,
    evaluate_reference,
    pack_e2m1,
    quantize_blocks,
    swizzle_scales,
    unpack_e2m1,
    unswizzle_scales,
)


def scalar_code(value):
    levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    code = min(range(8), key=lambda i: (abs(abs(value) - levels[i]), i % 2))
    return code | (8 if math.copysign(1.0, value) < 0 else 0)


def test_every_fp4_code_and_rounding_midpoint():
    values = [0.0, -0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 10.0]
    values += [-v for v in values]
    x = torch.tensor(values)
    assert encode_e2m1(x).tolist() == [scalar_code(v) for v in values]
    packed = pack_e2m1(x)
    assert packed.tolist() == [
        scalar_code(a) | scalar_code(b) << 4 for a, b in zip(values[::2], values[1::2], strict=True)
    ]
    all_codes = torch.arange(16, dtype=torch.uint8)
    encoded = all_codes[::2] | (all_codes[1::2] << 4)
    assert encode_e2m1(unpack_e2m1(encoded)).tolist() == list(range(16))


@pytest.mark.parametrize("rows,groups", [(1, 1), (3, 7), (129, 5), (256, 448)])
def test_scale_swizzle_matches_independent_byte_addresses(rows, groups):
    scales = (
        torch.arange(2 * rows * groups, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .reshape(2, rows, groups)
    )
    swizzled = swizzle_scales(scales)
    padded_rows, padded_groups = swizzled.shape[1:]
    expected = torch.zeros_like(swizzled).flatten()
    for batch in range(2):
        for row in range(rows):
            for group in range(groups):
                offset = (
                    batch * padded_rows * padded_groups
                    + (row // 128) * padded_groups * 128
                    + (group // 4) * 512
                    + (row % 32) * 16
                    + ((row % 128) // 32) * 4
                    + group % 4
                )
                expected[offset] = scales[batch, row, group]
    assert torch.equal(swizzled.flatten(), expected)
    assert torch.equal(unswizzle_scales(swizzled, rows, groups), scales)


def test_known_quantization_bytes_global_scale_and_zero_blocks():
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    ).reshape(1, 1, 16)
    for multiplier in [1.0, 2.0, 32.0]:
        global_scale = torch.tensor([multiplier])
        result = quantize_blocks(values, global_scale)
        assert result.packed.flatten().tolist() == [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]
        assert result.scales.view(torch.float8_e4m3fn).float().item() == multiplier
        assert torch.equal(dequantize(result, global_scale), values)
    zeros = quantize_blocks(torch.zeros(2, 3, 16), torch.ones(2), swizzled=True)
    assert not zeros.packed.any() and not zeros.scales.any()
    assert torch.equal(dequantize(zeros, torch.ones(2)), torch.zeros(2, 3, 16))


def test_add_rmsnorm_preserves_input_and_returns_updated_residual():
    x = torch.arange(32, dtype=torch.bfloat16).reshape(1, 2, 16) / 16
    residual = torch.ones_like(x)
    before = residual.clone()
    weight = torch.ones(16, dtype=torch.bfloat16)
    result = evaluate_reference("add_rmsnorm", x, torch.ones(1), residual=residual, weight=weight)
    assert torch.equal(residual, before)
    assert torch.equal(result.residual, x + residual)
    h = (x + residual).double()
    normalized = (h / (h.square().mean(-1, keepdim=True) + 1e-6).sqrt()).to(torch.bfloat16)
    expected = quantize_blocks(normalized, torch.ones(1))
    assert torch.equal(result.packed, expected.packed)
    assert torch.equal(result.scales, expected.scales)


def test_expert_mask_and_silu_fusion():
    generator = torch.Generator().manual_seed(602)
    x = torch.randn(2, 3, 64, generator=generator).bfloat16()
    mask = torch.tensor([0, 2], dtype=torch.int32)
    result = evaluate_reference("silu_mul", x, torch.ones(2), valid_rows=mask)
    assert result.swizzled
    assert not result.packed[0].any() and not result.packed[1, 2].any()
    gate, up = x.double().chunk(2, -1)
    expected = quantize_blocks(
        (gate / (1 + (-gate).exp()) * up).bfloat16(), torch.ones(2), swizzled=True, valid_rows=mask
    )
    assert torch.equal(result.packed, expected.packed)
    assert torch.equal(result.scales, expected.scales)


def test_six_workload_configs_and_three_real_operation_pairs():
    import importlib

    from labs.nvfp4_quantization.benchmarks import WORKLOADS

    assert len(WORKLOADS) == 6
    assert sorted(w.cols for w in WORKLOADS.values()) == [2048, 4096, 7168, 8192, 14336, 14336]
    for name, workload in WORKLOADS.items():
        for prefix in ["baseline", "optimized"]:
            bench = importlib.import_module(
                f"labs.nvfp4_quantization.{prefix}_{workload.kind}"
            ).get_benchmark()
            bench.apply_target_overrides(["--workload", name])
            assert bench.workload == workload
            assert bench.get_config().warmup > 0
            if not torch.cuda.is_available():
                with pytest.raises(RuntimeError, match="SKIPPED:.*SM100"):
                    bench.setup()


def test_fused_plan_input_validation_rejects_unsafe_shapes_and_metadata():
    from labs.nvfp4_quantization.benchmarks import NVFP4Workload, build_inputs, validate_inputs

    workload = NVFP4Workload("silu_mul", 2, 3, 32)
    tensors = build_inputs(workload, torch.device("cpu"))
    validate_inputs(workload, tensors)
    for change in [
        {"x": tensors["x"][:, :, :32]},
        {"global_scale": torch.tensor([1.0, float("nan")])},
        {"global_scale": torch.tensor([1.0, 0.0])},
        {"valid_rows": torch.tensor([-1, 3], dtype=torch.int32)},
        {"valid_rows": torch.tensor([4, 3], dtype=torch.int32)},
    ]:
        with pytest.raises(ValueError):
            validate_inputs(workload, tensors | change)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real Blackwell SM100 GPU required")
@pytest.mark.parametrize("kind", ["quantize", "add_rmsnorm", "silu_mul"])
def test_fused_cuda_full_buffers_and_replayed_inputs(kind):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires SM100")
    pytest.importorskip("triton")
    from labs.nvfp4_quantization.benchmarks import WORKLOADS, FusedPlan, NVFP4Workload, build_inputs

    # Include both padding/ragged cases and all six published B200 shapes.
    workloads = [NVFP4Workload(kind, 2, 3, 48)]
    workloads.extend(workload for workload in WORKLOADS.values() if workload.kind == kind)
    for workload in workloads:
        inputs = build_inputs(workload, torch.device("cuda"))
        inputs["global_scale"].copy_(torch.linspace(0.5, 1.5, workload.batch, device="cuda"))
        if kind == "silu_mul":
            inputs["valid_rows"][0] = 0
            inputs["valid_rows"][-1] = workload.rows
        plan = FusedPlan(workload, inputs)
        for delta in [0.0, 0.125]:
            inputs["x"].add_(delta)
            result = plan.run()
            reference = evaluate_reference(kind, **inputs)
            assert torch.equal(result.packed, reference.packed)
            assert torch.equal(result.scales, reference.scales)
            if result.residual is not None:
                assert torch.equal(result.residual, reference.residual)
