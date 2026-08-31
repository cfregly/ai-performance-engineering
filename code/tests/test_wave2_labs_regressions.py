from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from labs.cutlass_profiler_kernel_selector import (
    compare_against_baselines,
    run_cutlass_profiler_sweep,
)
from labs.cutlass_profiler_kernel_selector.run_cutlass_profiler_sweep import (
    parse_best_result,
)
from labs.dynamic_router.eval_stack import CheapEvalStack, EvalConfig
from labs.dynamic_router.scorecard import _load_run
from labs.occupancy_tuning.sweep_schedules import (
    benchmark_schedule,
)
from labs.occupancy_tuning.sweep_schedules import (
    parse_args as parse_occupancy_args,
)
from labs.occupancy_tuning.triton_matmul_schedules import (
    BASELINE_SCHEDULE,
    TRITON_MATMUL_OUTPUT_TOLERANCE,
    TritonMatmulProtonBenchmark,
)
from labs.real_world_models.gpt4_architecture_optimization import (
    GPT4ArchitectureOptimization,
)
from labs.real_world_models.llama_3_1_8b_optimization import (
    LLAMA_BF16_OUTPUT_TOLERANCE,
)


def test_cutlass_profiler_records_the_winning_operation_name(tmp_path: Path) -> None:
    csv_path = tmp_path / "profile.gemm.csv"
    csv_path.write_text(
        "OperationKind,Operation,Runtime,GFLOPs\n"
        "Gemm,slow_kernel,2.0,1000\n"
        "Gemm,winning_kernel,1.0,2000\n",
        encoding="utf-8",
    )

    best = parse_best_result(csv_path)

    assert best["kernel"] == "winning_kernel"
    assert best["runtime_ms"] == 1.0
    assert best["gflops"] == 2000.0


def test_cutlass_profiler_rejects_stale_csv_from_an_earlier_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = run_cutlass_profiler_sweep.GemmShape(
        name="shape_a",
        m=16,
        n=32,
        k=64,
    )
    (tmp_path / "shape_a.gemm.csv").write_text(
        "Operation,Runtime,GFLOPs\nstale_kernel,1.0,1000\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_cutlass_profiler_sweep.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Completed", (), {"returncode": 0})(),
    )

    with pytest.raises(FileNotFoundError, match="did not emit a fresh CSV"):
        run_cutlass_profiler_sweep.run_profiler_for_shape(
            tmp_path / "cutlass_profiler",
            shape,
            tmp_path,
        )


@pytest.mark.parametrize(
    "script_name",
    (
        "run_cutlass_profiler_sweep.py",
        "run_triton_matmul.py",
        "compare_against_baselines.py",
    ),
)
def test_cutlass_profiler_documented_scripts_support_direct_execution(
    script_name: str,
) -> None:
    repo_code = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_code / "labs/cutlass_profiler_kernel_selector" / script_name),
            "--help",
        ],
        cwd=repo_code,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_triton_matmul_masks_the_tail_of_the_k_dimension() -> None:
    repo_code = Path(__file__).resolve().parents[1]
    source = (
        repo_code / "labs/cutlass_profiler_kernel_selector/run_triton_matmul.py"
    ).read_text(encoding="utf-8")

    assert "k_mask = offs_k < k_remaining" in source
    assert "(offs_m[:, None] < M) & k_mask[None, :]" in source
    assert "k_mask[:, None] & (offs_n[None, :] < N)" in source
    assert source.count("other=0.0") >= 2
    assert "else:\n    _matmul_kernel = None" in source


def test_cutlass_comparison_fails_when_provider_omits_a_baseline_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline.json"
    provider = tmp_path / "provider.json"
    baseline.write_text(
        json.dumps(
            {
                "provider": "cutlass",
                "results": [
                    {"name": "shape_a", "tflops": 1.0},
                    {"name": "shape_b", "tflops": 2.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    provider.write_text(
        json.dumps(
            {"provider": "triton", "results": [{"name": "shape_a", "tflops": 1.1}]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(compare_against_baselines, "ARTIFACT_DIR", tmp_path)

    assert (
        compare_against_baselines.main(
            ["--baseline", str(baseline), "--providers", str(provider)]
        )
        == 1
    )
    assert not (tmp_path / "comparison.json").exists()


def test_dynamic_router_scorecard_divides_drops_by_tokens(tmp_path: Path) -> None:
    (tmp_path / "moe_router.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"drops": 1, "total_tokens": 100},
                {"drops": 2, "total_tokens": 50},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = _load_run(tmp_path)

    assert result["drop_rate"] == pytest.approx(3 / 150)


def test_dynamic_router_scorecard_rejects_legacy_window_counts(tmp_path: Path) -> None:
    (tmp_path / "moe_router.jsonl").write_text(
        json.dumps({"drops": 1}) + "\n", encoding="utf-8"
    )

    assert math.isnan(_load_run(tmp_path)["drop_rate"])


def test_dynamic_router_missing_metrics_error_names_the_supported_flag(
    tmp_path: Path,
) -> None:
    stack = CheapEvalStack(EvalConfig(metrics_dir=tmp_path, allow_missing_metrics=False))

    with pytest.raises(FileNotFoundError, match="--allow-missing-metrics") as error:
        stack._load_real_metrics()

    assert "EVAL_STACK_ALLOW_MISSING" not in str(error.value)


def test_synthetic_router_records_complete_window_token_counts() -> None:
    stack = object.__new__(CheapEvalStack)
    stack.cfg = EvalConfig(request_count=5, use_vllm=False)
    stack._rng = random.Random(stack.cfg.seed)

    router_rows, _, summary = stack._simulate_moe(optimized=False)

    assert [row["total_tokens"] for row in router_rows] == [64, 36]
    assert sum(row["total_tokens"] for row in router_rows) == 100
    assert sum(row["drops"] for row in router_rows) / 100 == pytest.approx(
        summary["token_drop_rate"]
    )


def test_llama_bf16_verification_rejects_zero_and_grossly_wrong_outputs() -> None:
    rtol, atol = LLAMA_BF16_OUTPUT_TOLERANCE
    reference = torch.tensor([[-0.75, -0.25, 0.25, 0.75]], dtype=torch.float32)

    assert not torch.allclose(torch.zeros_like(reference), reference, rtol=rtol, atol=atol)
    assert not torch.allclose(reference * 0.5, reference, rtol=rtol, atol=atol)
    repo_code = Path(__file__).resolve().parents[1]
    for name in ("baseline_llama_3_1_8b.py", "optimized_llama_3_1_8b.py"):
        source = (repo_code / "labs/real_world_models" / name).read_text(encoding="utf-8")
        assert "output_tolerance=LLAMA_BF16_OUTPUT_TOLERANCE" in source
        assert "output_tolerance=(0.1, 1.0)" not in source


def test_llama_baseline_preserves_bf16_math_and_builds_the_declared_depth() -> None:
    repo_code = Path(__file__).resolve().parents[1]
    source = (
        repo_code / "labs/real_world_models/llama_3_1_8b_optimization.py"
    ).read_text(encoding="utf-8")

    assert "scores = torch.matmul(q, k.transpose(-2, -1))" in source
    assert "attn = torch.matmul(probs, v)" in source
    assert "q_fp32 = q.float()" not in source
    assert "v_fp32 = v.float()" not in source
    assert "for _ in range(self.NUM_LAYERS)" in source
    assert "for _ in range(4)" not in source


def test_gpt4_proxy_reports_only_optimizations_it_executes() -> None:
    proxy = GPT4ArchitectureOptimization(
        batch_size=1,
        seq_length=1,
        use_moe=True,
        use_fp8=True,
        use_context_parallel=True,
    )

    assert proxy.requested_use_moe is True
    assert proxy.requested_use_fp8 is True
    assert proxy.requested_use_context_parallel is True
    assert proxy.use_moe is False
    assert proxy.use_fp8 is False
    assert proxy.use_context_parallel is False


def test_occupancy_sweep_documented_cli_supports_direct_execution() -> None:
    repo_code = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_code / "labs/occupancy_tuning/sweep_schedules.py"),
            "--list",
        ],
        cwd=repo_code,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "bm64_bn64_bk32" in result.stdout


def test_occupancy_sweep_cli_uses_the_documented_csv_option() -> None:
    args = parse_occupancy_args(["--csv", "custom.csv", "--list"])

    assert args.csv == Path("custom.csv")
    with pytest.raises(SystemExit):
        parse_occupancy_args(["--output", "custom.csv"])


def test_occupancy_sweep_rejects_torch_compile_before_gpu_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "labs.occupancy_tuning.sweep_schedules._ensure_cuda",
        lambda: None,
    )

    with pytest.raises(ValueError, match="torch.compile is unsupported"):
        benchmark_schedule(
            BASELINE_SCHEDULE,
            size_m=16,
            size_n=16,
            size_k=16,
            dtype=torch.float16,
            iterations=1,
            warmup=0,
            use_compile=True,
        )


def test_occupancy_validation_rejects_zero_and_grossly_wrong_outputs() -> None:
    assert TRITON_MATMUL_OUTPUT_TOLERANCE == (0.02, 0.02)
    benchmark = TritonMatmulProtonBenchmark(BASELINE_SCHEDULE, size=2, size_k=2)
    benchmark._reference = torch.tensor([[-0.75, -0.25], [0.25, 0.75]])
    benchmark._validation_scalars = torch.empty(2, dtype=torch.float32)

    benchmark._output = benchmark._reference.clone()
    assert benchmark.validate_result() is None

    benchmark._output = torch.zeros_like(benchmark._reference)
    assert "Pairwise tolerance exceeded" in (benchmark.validate_result() or "")

    benchmark._output = benchmark._reference * 0.5
    assert "Pairwise tolerance exceeded" in (benchmark.validate_result() or "")


def test_occupancy_readme_uses_only_supported_commands() -> None:
    repo_code = Path(__file__).resolve().parents[1]
    readme = (repo_code / "labs/occupancy_tuning/README.md").read_text(encoding="utf-8")

    assert "sweep_schedules.py --csv" in readme
    assert "sweep_schedules.py --output" not in readme
    assert " --validate" not in readme
    assert "--target-extra-arg" not in readme
