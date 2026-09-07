"""Seed ownership controls for reusable benchmark setup implementations."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from ch16.awq_gptq_smoothquant_benchmarks import PTQQuantizationBenchmark, PTQWorkload
from core.benchmark.flexattention_sliding_window import (
    SlidingWindowAttentionBenchmark,
    SlidingWindowConfig,
)
from core.benchmark.nvfp4_mlp import NVFP4MLPConfig, create_mlp_weights
from core.benchmark.tcgen05_matmul_base import Tcgen05MatmulBenchmarkBase
from core.harness.benchmark_harness import BaseBenchmark
from tests.protection_test_utils import preserve_rng_state


CODE_ROOT = Path(__file__).resolve().parents[1]
SETUP_CLASSES = {
    "ch18/paged_attn_split_common.py": ("DensePagedAttnBase", "LayoutPagedAttnBase"),
    "ch16/awq_gptq_smoothquant_benchmarks.py": ("PTQQuantizationBenchmark",),
    "ch11/stream_overlap_base.py": ("StridedStreamBaseline", "ConcurrentStreamOptimized"),
    "ch17/prefill_decode_disagg_single_common.py": ("_PrefillDecodeSingleGPUBase",),
    "ch17/prefill_decode_disagg_multigpu_common.py": ("_PrefillDecodeMultiGPUBenchmark",),
    "core/benchmark/nvfp4_mlp.py": ("NVFP4MLPBenchmark",),
    "core/benchmark/flexattention_sliding_window.py": ("SlidingWindowAttentionBenchmark",),
    "core/benchmark/tcgen05_matmul_base.py": ("Tcgen05MatmulBenchmarkBase",),
    "core/utils/continuous_batching.py": ("ContinuousBatchingBase",),
}
GLOBAL_SEED_CALLS = {
    "torch.manual_seed",
    "torch.cuda.manual_seed",
    "torch.cuda.manual_seed_all",
}


class _CpuTcgen05(Tcgen05MatmulBenchmarkBase):
    matrix_rows = 5
    matrix_cols = 3
    shared_dim = 4
    tensor_dtype = torch.float32

    def __init__(self) -> None:
        BaseBenchmark.__init__(self)
        self.device = torch.device("cpu")
        self.matrix_a = None
        self.matrix_b = None
        self.output = None
        self.parameter_count = 0


def _ptq_snapshot(seed: int) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, int]:
    torch.manual_seed(seed)
    benchmark = PTQQuantizationBenchmark(scheme="baseline", label="cpu_seed_control")
    benchmark.device = torch.device("cpu")
    benchmark.workload = PTQWorkload(
        batch_size=4,
        in_features=6,
        hidden_features=8,
        out_features=5,
        calibration_samples=3,
        dtype=torch.float32,
    )
    benchmark.setup()
    assert benchmark.reference_model is not None
    assert benchmark.inputs is not None
    with torch.inference_mode():
        output = benchmark.reference_model(benchmark.inputs)
    return (
        tuple(parameter.detach().clone() for parameter in benchmark.reference_model.parameters()),
        benchmark.inputs.detach().clone(),
        output.detach().clone(),
        int(torch.initial_seed()),
    )


def _flex_snapshot(seed: int) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, int]:
    torch.manual_seed(seed)
    benchmark = SlidingWindowAttentionBenchmark(
        use_flex=False,
        cfg=SlidingWindowConfig(
            batch_size=1,
            num_heads=2,
            seq_len=6,
            head_dim=4,
            window_size=3,
            dtype=torch.float32,
        ),
    )
    benchmark.device = torch.device("cpu")
    benchmark.setup()
    benchmark.benchmark_fn()
    assert benchmark.q is not None and benchmark.k is not None and benchmark.v is not None
    assert benchmark.output is not None
    return (
        tuple(tensor.detach().clone() for tensor in (benchmark.q, benchmark.k, benchmark.v)),
        benchmark.output.detach().clone(),
        int(torch.initial_seed()),
    )


def _tcgen_snapshot(seed: int) -> tuple[tuple[torch.Tensor, ...], int]:
    torch.manual_seed(seed)
    benchmark = _CpuTcgen05()
    benchmark.setup()
    assert benchmark.matrix_a is not None and benchmark.matrix_b is not None
    return (
        (benchmark.matrix_a.detach().clone(), benchmark.matrix_b.detach().clone()),
        int(torch.initial_seed()),
    )


def _assert_tensor_tuples_equal(left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]) -> None:
    assert len(left) == len(right)
    assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))


def test_common_setups_do_not_reset_the_harness_seed() -> None:
    checked = 0
    for relative_path, class_names in SETUP_CLASSES.items():
        tree = ast.parse((CODE_ROOT / relative_path).read_text(encoding="utf-8"))
        classes = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in class_names
        }
        assert classes.keys() == set(class_names)
        for class_name in class_names:
            setup = next(
                node
                for node in classes[class_name].body
                if isinstance(node, ast.FunctionDef) and node.name == "setup"
            )
            calls = {ast.unparse(node.func) for node in ast.walk(setup) if isinstance(node, ast.Call)}
            assert calls.isdisjoint(GLOBAL_SEED_CALLS), f"{relative_path}:{class_name}"
            checked += 1
    assert checked == 11


def test_real_cpu_ptq_setup_preserves_default_and_fresh_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with preserve_rng_state():
        params_42, inputs_42, output_42, observed_42 = _ptq_snapshot(42)
        repeated_params, repeated_inputs, repeated_output, repeated_seed = _ptq_snapshot(42)
        params_1042, inputs_1042, output_1042, observed_1042 = _ptq_snapshot(1042)

    _assert_tensor_tuples_equal(params_42, repeated_params)
    assert torch.equal(inputs_42, repeated_inputs)
    assert torch.equal(output_42, repeated_output)
    assert observed_42 == repeated_seed == 42
    assert observed_1042 == 1042
    assert not torch.equal(params_42[0], params_1042[0])
    assert not torch.equal(inputs_42, inputs_1042)
    assert not torch.equal(output_42, output_1042)


def test_real_cpu_flex_and_tcgen_inputs_follow_the_caller_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with preserve_rng_state():
        flex_42, output_42, flex_seed_42 = _flex_snapshot(42)
        repeated_flex, repeated_output, repeated_flex_seed = _flex_snapshot(42)
        flex_1042, output_1042, flex_seed_1042 = _flex_snapshot(1042)
        tcgen_42, tcgen_seed_42 = _tcgen_snapshot(42)
        repeated_tcgen, repeated_tcgen_seed = _tcgen_snapshot(42)
        tcgen_1042, tcgen_seed_1042 = _tcgen_snapshot(1042)

    _assert_tensor_tuples_equal(flex_42, repeated_flex)
    _assert_tensor_tuples_equal(tcgen_42, repeated_tcgen)
    assert torch.equal(output_42, repeated_output)
    assert flex_seed_42 == repeated_flex_seed == tcgen_seed_42 == repeated_tcgen_seed == 42
    assert flex_seed_1042 == tcgen_seed_1042 == 1042
    assert not torch.equal(flex_42[0], flex_1042[0])
    assert not torch.equal(output_42, output_1042)
    assert not torch.equal(tcgen_42[0], tcgen_1042[0])


def test_nvfp4_private_weight_generator_remains_an_alignment_control() -> None:
    config = NVFP4MLPConfig(batch_size=2, d_model=4, d_ff=6, num_layers=1)
    with preserve_rng_state():
        torch.manual_seed(42)
        weights_42 = create_mlp_weights(config, device=torch.device("cpu"), dtype=torch.float32)
        observed_42 = int(torch.initial_seed())
        torch.manual_seed(1042)
        weights_1042 = create_mlp_weights(config, device=torch.device("cpu"), dtype=torch.float32)
        observed_1042 = int(torch.initial_seed())

    flat_42 = tuple(tensor for layer in weights_42 for tensor in layer if tensor is not None)
    flat_1042 = tuple(tensor for layer in weights_1042 for tensor in layer if tensor is not None)
    _assert_tensor_tuples_equal(flat_42, flat_1042)
    assert observed_42 == 42
    assert observed_1042 == 1042
