from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent


def test_tma_copy_accounts_for_three_source_reads_and_one_destination_write() -> None:
    from ch07.baseline_tma_copy import BaselineTMACopyBenchmark
    from ch07.optimized_tma_copy import OptimizedTMACopyBenchmark

    for benchmark_type in (BaselineTMACopyBenchmark, OptimizedTMACopyBenchmark):
        benchmark = benchmark_type()
        n_elems = benchmark._workload_params["N"]
        metadata = benchmark.get_workload_metadata()

        assert benchmark._workload_params["source_reads_per_output"] == 3
        assert "redundant_reads" not in benchmark._workload_params
        assert metadata.bytes_per_iteration == n_elems * (3 + 1) * 4


def test_invalid_ch07_expectations_are_retired_until_fresh_gpu_measurement() -> None:
    for filename in ("expectations_b200.json", "expectations_4x_gb200.json"):
        payload = json.loads((CODE_ROOT / "ch07" / filename).read_text(encoding="utf-8"))
        examples = payload["examples"]

        assert "lookup_cuda" not in examples
        assert "tma_copy_cuda" not in examples


def test_vector_mlp_roofline_counts_gemv_flops_and_parameter_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ch09.baseline_compute_bound import BaselineComputeBoundBenchmark
    from ch09.optimized_compute_bound import OptimizedComputeBoundBenchmark
    from core.benchmark import metrics as benchmark_metrics

    def capture_metrics(**kwargs: float | str | None) -> dict[str, float | str | None]:
        return kwargs

    monkeypatch.setattr(benchmark_metrics, "compute_roofline_metrics", capture_metrics)

    for benchmark_type in (BaselineComputeBoundBenchmark, OptimizedComputeBoundBenchmark):
        benchmark = benchmark_type()
        result = benchmark.get_custom_metrics()
        assert result is not None

        parameter_elements = 4 * benchmark.N * benchmark.N + 3 * benchmark.N
        expected_flops = 8 * benchmark.N * benchmark.N * benchmark.repeats
        expected_bytes = (
            (parameter_elements + 2 * benchmark.N)
            * torch.tensor([], dtype=torch.float16).element_size()
            * benchmark.repeats
        )
        assert result["total_flops"] == expected_flops
        assert result["total_bytes"] == expected_bytes


def test_b200_compute_bound_expectation_drops_invalid_roofline_metrics() -> None:
    payload = json.loads((CODE_ROOT / "ch09" / "expectations_b200.json").read_text(encoding="utf-8"))
    assert "custom_metrics" not in payload["examples"]["compute_bound"]


def test_nvfp4_vec16_scale_layout_is_a_bijection(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A host C++ compiler is required for the scale-layout contract test")

    source = tmp_path / "scale_layout_test.cc"
    binary = tmp_path / "scale_layout_test"
    source.write_text(
        r'''#include <cstddef>
#include <utility>
#include <vector>

#include "code/core/common/headers/nvfp4_scale_layout.cuh"

int main() {
  for (const auto [rows, reduction] : {std::pair{128, 64}, std::pair{256, 128}}) {
    const int scale_columns = reduction / 16;
    std::vector<bool> seen(static_cast<std::size_t>(rows) * scale_columns, false);
    for (int row = 0; row < rows; ++row) {
      for (int scale_column = 0; scale_column < scale_columns; ++scale_column) {
        const std::size_t offset = aisp::nvfp4_vec16_scale_offset(
            row, scale_column, reduction);
        if (offset >= seen.size() || seen[offset]) return 1;
        seen[offset] = true;
      }
    }
    for (bool value : seen) if (!value) return 2;
  }
  if (aisp::nvfp4_vec16_scale_offset(0, 0, 128) != 0) return 3;
  if (aisp::nvfp4_vec16_scale_offset(0, 4, 128) != 512) return 4;
  if (aisp::nvfp4_vec16_scale_offset(1, 0, 128) != 16) return 5;
  if (aisp::nvfp4_vec16_scale_offset(32, 0, 128) != 4) return 6;
  if (aisp::nvfp4_vec16_scale_offset(127, 7, 128) != 1023) return 7;
  if (aisp::nvfp4_vec16_scale_offset(128, 0, 128) != 1024) return 8;
  return 0;
}
''',
        encoding="utf-8",
    )
    compile_result = subprocess.run(
        [compiler, "-std=c++17", "-I", str(REPO_ROOT), str(source), "-o", str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run([str(binary)], check=False, capture_output=True, text=True)
    assert run_result.returncode == 0, run_result.stderr


def test_nvfp4_perchannel_sources_use_swizzled_scale_layout() -> None:
    for filename in (
        "baseline_cublas_gemm_fp4_perchannel.cu",
        "optimized_cublas_gemm_fp4_perchannel.cu",
    ):
        source = (CODE_ROOT / "ch09" / filename).read_text(encoding="utf-8")
        assert '#include "../core/common/headers/nvfp4_scale_layout.cuh"' in source
        assert "aisp::nvfp4_vec16_scale_offset" in source
        assert "scales[r * num_scale_cols + block]" not in source
        assert "kN, kK, 1, kN" in source


def test_compiled_autograd_context_uses_an_available_pytorch_entrypoint() -> None:
    from torch._dynamo import compiled_autograd

    from ch13.compiled_autograd import compiled_autograd_context

    x = torch.tensor([2.0], requires_grad=True)
    with compiled_autograd_context(backend="eager"):
        assert compiled_autograd.compiled_autograd_enabled is True
        x.square().sum().backward()

    assert compiled_autograd.compiled_autograd_enabled is False
    torch.testing.assert_close(x.grad, torch.tensor([4.0]))


def test_te_delayed_scaling_recipe_uses_transformer_engine_2x_signature() -> None:
    from ch13.optimized_precisionfp8_te import _create_delayed_scaling_recipe

    class StrictDelayedScaling:
        def __init__(
            self,
            *,
            margin: int,
            amax_history_len: int,
            amax_compute_algo: str,
            scaling_factor_compute_algo: object,
        ) -> None:
            self.kwargs = {
                "margin": margin,
                "amax_history_len": amax_history_len,
                "amax_compute_algo": amax_compute_algo,
                "scaling_factor_compute_algo": scaling_factor_compute_algo,
            }

    class FakeRecipeModule:
        DelayedScaling = StrictDelayedScaling

    recipe = _create_delayed_scaling_recipe(FakeRecipeModule)
    assert recipe.kwargs == {
        "margin": 0,
        "amax_history_len": 1024,
        "amax_compute_algo": "max",
        "scaling_factor_compute_algo": None,
    }


@pytest.mark.parametrize(
    "benchmark_path,class_name",
    [
        ("ch13.baseline_regional_compile", "BaselineFullGraphCompileBenchmark"),
        ("ch13.optimized_regional_compile", "OptimizedRegionalCompileBenchmark"),
    ],
)
def test_regional_compile_tolerance_rejects_all_zero_output(
    benchmark_path: str,
    class_name: str,
) -> None:
    module = __import__(benchmark_path, fromlist=[class_name])
    benchmark_type = getattr(module, class_name)
    benchmark = benchmark_type()
    benchmark.batch_size = 1
    benchmark._verify_x = torch.ones((1, 1, 1), dtype=torch.bfloat16)
    benchmark._verify_output = torch.ones((1, 1, 1), dtype=torch.bfloat16)
    benchmark._verify_output_buffer = torch.empty((1, 1, 1), dtype=torch.float32)

    benchmark.capture_verification_payload()
    rtol, atol = benchmark.get_output_tolerance()
    assert (rtol, atol) == (1e-2, 5e-2)
    assert not torch.allclose(
        torch.zeros_like(benchmark._verify_output_buffer),
        benchmark._verify_output_buffer,
        rtol=rtol,
        atol=atol,
    )


def test_regional_compile_tolerance_accepts_the_matching_bf16_pair() -> None:
    from ch13.baseline_regional_compile import TinyTransformerBlock as BaselineBlock
    from ch13.optimized_regional_compile import TinyTransformerBlock as OptimizedBlock

    torch.manual_seed(42)
    baseline = BaselineBlock(hidden=64, num_heads=4, mlp_hidden=256).to(
        dtype=torch.bfloat16
    ).eval()
    optimized = OptimizedBlock(hidden=64, num_heads=4, mlp_hidden=256).to(
        dtype=torch.bfloat16
    ).eval()
    optimized.load_state_dict(baseline.state_dict())
    inputs = torch.randn((2, 16, 64), dtype=torch.bfloat16)

    with torch.inference_mode():
        expected = baseline(inputs.clone())
        actual = optimized(inputs.clone())

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-2)
