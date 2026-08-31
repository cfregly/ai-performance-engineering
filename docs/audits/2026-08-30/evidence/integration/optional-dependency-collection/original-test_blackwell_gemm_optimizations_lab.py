"""Tests for the Blackwell grouped GEMM optimization lab."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from core.discovery import discover_benchmarks
from core.harness.benchmark_harness import BaseBenchmark
from labs.blackwell_gemm_optimizations.blackwell_grouped_gemm_autotune import (
    experimental_variant_names,
    public_variant_names,
)
from labs.blackwell_gemm_optimizations.blackwell_grouped_gemm_common import (
    BlackwellGroupedGemmWorkload,
    _assignment_counts_cpu,
    _gather_packed_tokens,
    build_state,
    run_variant,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = REPO_ROOT / "labs" / "blackwell_gemm_optimizations"


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blackwell_grouped_gemm_discovery_finds_primary_and_alias_targets() -> None:
    pairs = discover_benchmarks(LAB_DIR)
    discovered = {example_name for _, _, example_name in pairs}
    assert discovered == {
        "blackwell_grouped_gemm",
        "blackwell_grouped_gemm_large_tiles",
        "blackwell_grouped_gemm_full_stack",
        "blackwell_grouped_gemm_persistent",
    }

    primary = next(pair for pair in pairs if pair[2] == "blackwell_grouped_gemm")
    _, optimized_paths, _ = primary
    assert {path.name for path in optimized_paths} == {
        "optimized_blackwell_grouped_gemm_large_tiles.py",
        "optimized_blackwell_grouped_gemm_full_stack.py",
        "optimized_blackwell_grouped_gemm_persistent.py",
    }


@pytest.mark.parametrize(
    "relative_path",
    [
        "labs/blackwell_gemm_optimizations/baseline_blackwell_grouped_gemm.py",
        "labs/blackwell_gemm_optimizations/optimized_blackwell_grouped_gemm_large_tiles.py",
        "labs/blackwell_gemm_optimizations/optimized_blackwell_grouped_gemm_full_stack.py",
        "labs/blackwell_gemm_optimizations/optimized_blackwell_grouped_gemm_persistent.py",
    ],
)
def test_blackwell_grouped_gemm_wrappers_expose_get_benchmark(relative_path: str) -> None:
    module_path = REPO_ROOT / relative_path
    module = _load_module(module_path)
    bench = module.get_benchmark()
    assert isinstance(bench, BaseBenchmark)
    assert getattr(bench, "_module_file_override", None) == str(module_path)
    assert getattr(bench, "_factory_name_override", None) == "get_benchmark"


def test_blackwell_grouped_gemm_schedule_registry_is_complete() -> None:
    assert set(public_variant_names()) == {
        "baseline",
        "large_tiles",
        "full_stack",
        "persistent",
    }
    assert set(experimental_variant_names()) == {
        "fast_math_control",
        "latency10",
        "two_cta",
        "tile_n256",
    }


def test_blackwell_grouped_gemm_skewed_counts_avoid_tensor_roundtrip() -> None:
    source = (LAB_DIR / "blackwell_grouped_gemm_common.py").read_text(encoding="utf-8")
    counts_section = source.split("def _assignment_counts_cpu", maxsplit=1)[1].split(
        "def _build_route_weights",
        maxsplit=1,
    )[0]

    assert "torch.linspace(" not in counts_section
    assert ".sum().item()" not in counts_section
    assert ".tolist()" not in counts_section

    workload = BlackwellGroupedGemmWorkload(
        num_tokens=17,
        num_experts=4,
        hidden_dim=32,
        expert_ffn_dim=64,
        dtype=torch.float16,
        histogram="skewed",
    )

    assert _assignment_counts_cpu(workload) == (8, 5, 3, 1)
    assert sum(_assignment_counts_cpu(workload)) == workload.num_tokens


def test_blackwell_grouped_gemm_padding_row_is_preallocated() -> None:
    source = (LAB_DIR / "blackwell_grouped_gemm_common.py").read_text(encoding="utf-8")
    setup_section = source.split("def build_state", maxsplit=1)[1].split(
        "def require_blackwell_grouped_gemm_support",
        maxsplit=1,
    )[0]

    assert "torch.cat([x" not in setup_section
    assert "zero_row" not in setup_section
    assert "x_with_padding[workload.num_tokens].zero_()" in setup_section


def test_blackwell_grouped_gemm_reuses_packed_token_view() -> None:
    source = (LAB_DIR / "blackwell_grouped_gemm_common.py").read_text(encoding="utf-8")
    kernel_source = (LAB_DIR / "blackwell_grouped_gemm_kernel.py").read_text(encoding="utf-8")
    gather_section = source.split("def _gather_packed_tokens", maxsplit=1)[1].split(
        "def run_variant",
        maxsplit=1,
    )[0]
    run_section = source.split("def run_variant", maxsplit=1)[1].split(
        "class BlackwellGroupedGemmBenchmark",
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
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown",
        maxsplit=1,
    )[0]

    assert "out_view: Optional[torch.Tensor] = None" in gather_section
    assert "if out_view is not None:" in gather_section
    assert "return out_view" in gather_section
    assert "packed_tokens_view: torch.Tensor | None = None" in run_section
    assert "_gather_packed_tokens(state, packed_tokens_flat, packed_tokens_view)" in run_section
    assert "padded_route_weight_factors: torch.Tensor" in source
    assert "padded_route_weight_factors = padded_route_weights.unsqueeze(-1).to(workload.dtype)" in source
    assert "state.padded_route_weight_factors" in run_section
    assert "route_weight_factors: torch.Tensor | None = None" in kernel_source
    assert "route_weight_factors = route_weights.unsqueeze(-1).to(out.dtype)" in kernel_source
    assert "out.mul_(route_weight_factors)" in kernel_source
    assert "self._packed_tokens_view = self._flat_packed_tokens.view(" in setup_section
    assert "with torch.inference_mode(), self._nvtx_range(self.label):" in benchmark_section
    assert "with self._nvtx_range(self.label):" not in benchmark_section
    assert "torch.no_grad()" not in benchmark_section
    assert "packed_tokens_view=self._packed_tokens_view" in benchmark_section
    assert "self._verify_output_buffer: Optional[torch.Tensor] = None" in source
    assert "self._verification_shape_tensor: Optional[torch.Tensor] = None" in source
    assert "self._verify_output_buffer = torch.empty(" in setup_section
    assert "self._verification_shape_tensor = torch.tensor(" in setup_section
    assert "verification_slice = self.output[" in capture_section
    assert "self._verify_output_buffer.copy_(verification_slice)" in capture_section
    assert 'inputs={"shape": self._verification_shape_tensor}' in capture_section
    assert "output=self._verify_output_buffer" in capture_section
    assert "torch.tensor(" not in capture_section
    assert "self._verify_output_buffer = None" in source
    assert "self._verification_shape_tensor = None" in source

    workload = BlackwellGroupedGemmWorkload(
        num_tokens=17,
        num_experts=4,
        hidden_dim=32,
        expert_ffn_dim=64,
        dtype=torch.float16,
        histogram="balanced",
    )
    state = build_state(workload, torch.device("cpu"))
    flat = torch.empty(
        workload.num_experts * state.max_count,
        workload.hidden_dim,
        dtype=workload.dtype,
    )
    view = flat.view(workload.num_experts, state.max_count, workload.hidden_dim)

    gathered = _gather_packed_tokens(state, flat, view)

    assert gathered is view
    assert gathered.data_ptr() == flat.data_ptr()


def test_blackwell_grouped_gemm_matrix_runner_records_timing_on_current_stream() -> None:
    source = (LAB_DIR / "compare_blackwell_grouped_gemm_matrix.py").read_text(
        encoding="utf-8"
    )
    timing_section = source.split("def _measure_variant", maxsplit=1)[1].split(
        "def _iter_variant_rows",
        maxsplit=1,
    )[0]

    assert timing_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "current_stream = torch.cuda.current_stream(device)" in timing_section
    assert "start.record(current_stream)" in timing_section
    assert "end.record(current_stream)" in timing_section
    assert "start.record()" not in timing_section
    assert "end.record()" not in timing_section


@pytest.mark.parametrize("histogram", ["balanced", "skewed"])
def test_blackwell_grouped_gemm_build_state_packs_routes_on_cpu(
    histogram: str,
) -> None:
    workload = BlackwellGroupedGemmWorkload(
        num_tokens=17,
        num_experts=4,
        hidden_dim=32,
        expert_ffn_dim=64,
        dtype=torch.float16,
        histogram=histogram,
    )
    state = build_state(workload, torch.device("cpu"))

    counts = tuple(int(count) for count in state.counts.tolist())
    padded_indices = state.flat_padded_indices.view(
        workload.num_experts,
        state.max_count,
    )
    assert counts == state.counts_cpu

    for expert_id, count in enumerate(counts):
        valid_indices = padded_indices[expert_id, :count]
        if histogram == "balanced":
            expected_indices = torch.arange(
                expert_id,
                workload.num_tokens,
                workload.num_experts,
                dtype=torch.long,
            )
        else:
            start = sum(counts[:expert_id])
            expected_indices = torch.arange(start, start + count, dtype=torch.long)
        assert torch.equal(valid_indices, expected_indices)

        padded_tail = padded_indices[expert_id, count:]
        route_tail = state.padded_route_weights[expert_id, count:]
        factor_tail = state.padded_route_weight_factors[expert_id, count:]
        assert torch.all(padded_tail == workload.num_tokens)
        assert torch.count_nonzero(route_tail).item() == 0
        assert torch.count_nonzero(factor_tail).item() == 0

    gathered = torch.index_select(
        state.x_with_padding,
        0,
        state.flat_padded_indices,
    ).view(
        workload.num_experts,
        state.max_count,
        workload.hidden_dim,
    )
    reference = torch.bmm(gathered.float(), state.expert_weights.float())
    reference *= state.padded_route_weights.unsqueeze(-1)
    reference = reference.to(workload.dtype)
    torch.testing.assert_close(reference.float(), state.reference_output.float())
    assert state.padded_route_weight_factors.shape == (*state.padded_route_weights.shape, 1)
    torch.testing.assert_close(
        state.padded_route_weight_factors[..., 0],
        state.padded_route_weights.to(workload.dtype),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Blackwell grouped GEMM validation",
)
def test_blackwell_grouped_gemm_public_variants_match_reference() -> None:
    workload = BlackwellGroupedGemmWorkload(
        num_tokens=128,
        num_experts=4,
        hidden_dim=256,
        expert_ffn_dim=512,
        dtype=torch.float16,
    )
    state = build_state(workload, torch.device("cuda"))

    for variant in public_variant_names():
        packed = torch.empty(
            workload.num_experts * state.max_count,
            workload.hidden_dim,
            device="cuda",
            dtype=workload.dtype,
        )
        out = torch.empty(
            workload.num_experts,
            state.max_count,
            workload.expert_ffn_dim,
            device="cuda",
            dtype=workload.dtype,
        )
        result = run_variant(
            state,
            variant=variant,
            packed_tokens_flat=packed,
            output_buffer=out,
        )
        torch.testing.assert_close(
            result.output.float(),
            state.reference_output.float(),
            atol=3.5e-1,
            rtol=5e-2,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Blackwell grouped GEMM validation",
)
def test_blackwell_grouped_gemm_experimental_variants_execute() -> None:
    workload = BlackwellGroupedGemmWorkload(
        num_tokens=128,
        num_experts=4,
        hidden_dim=256,
        expert_ffn_dim=512,
        dtype=torch.float16,
    )
    state = build_state(workload, torch.device("cuda"))

    for experimental in experimental_variant_names():
        packed = torch.empty(
            workload.num_experts * state.max_count,
            workload.hidden_dim,
            device="cuda",
            dtype=workload.dtype,
        )
        out = torch.empty(
            workload.num_experts,
            state.max_count,
            workload.expert_ffn_dim,
            device="cuda",
            dtype=workload.dtype,
        )
        result = run_variant(
            state,
            variant="full_stack",
            experimental=experimental,
            packed_tokens_flat=packed,
            output_buffer=out,
        )
        torch.testing.assert_close(
            result.output.float(),
            state.reference_output.float(),
            atol=3.5e-1,
            rtol=5e-2,
        )
