"""Real CLI regressions for registered demo entrypoints."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("demo_name", "cli_args", "expected_diagnostic"),
    [
        (
            "ch10-warpgroup-specialization",
            [],
            "CUDA required for ch10 warpgroup specialization demo",
        ),
        (
            "ch13-fp8-perchannel",
            [],
            "CUDA required for ch13 FP8 per-channel demo",
        ),
        (
            "ch09-online-softmax",
            [],
            "CUDA required for ch09 online softmax demo",
        ),
        (
            "ch13-transformer-engine-fp8",
            [],
            "CUDA required for ch13 Transformer Engine FP8 demo",
        ),
        (
            "labs-nvfp4-add-rmsnorm",
            [],
            "CUDA required for labs NVFP4 add/RMSNorm demo",
        ),
        (
            "labs-decode-multigpu",
            ["--nproc-per-node", "2"],
            "CUDA required for multi-GPU decode demo",
        ),
    ],
)
def test_registered_demo_reaches_explicit_cuda_gate(
    demo_name: str,
    cli_args: list[str],
    expected_diagnostic: str,
) -> None:
    """A CUDA-hidden real CLI launch must execute the script, not exit zero after import."""
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(CODE_ROOT),
        }
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cli.aisp",
            "demos",
            demo_name,
            *cli_args,
        ],
        cwd=CODE_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail(f"Demo process timed out:\n{stdout}{stderr}")

    combined_output = stdout + stderr
    assert process.returncode != 0, combined_output
    assert expected_diagnostic in combined_output


def test_b200_compatibility_demo_defaults_match_the_bounded_workloads() -> None:
    from ch09 import online_softmax_demo
    from ch13 import transformer_engine_fp8_demo
    from labs.nvfp4_quantization import add_rmsnorm_demo

    softmax_args = online_softmax_demo.build_parser().parse_args([])
    assert (softmax_args.rows, softmax_args.cols) == (4, 257)
    softmax = online_softmax_demo.build_benchmark(
        rows=softmax_args.rows,
        cols=softmax_args.cols,
    )
    assert softmax.optimized
    assert (softmax.rows, softmax.cols) == (4, 257)
    assert (online_softmax_demo.OUTPUT_RTOL, online_softmax_demo.OUTPUT_ATOL) == (
        2e-5,
        2e-7,
    )

    fp8_args = transformer_engine_fp8_demo.build_parser().parse_args([])
    assert (fp8_args.batch, fp8_args.hidden) == (16, 256)
    assert transformer_engine_fp8_demo.get_te_precision_output_tolerances()["prediction"] == (
        0.4,
        1.0,
    )

    nvfp4_args = add_rmsnorm_demo.build_parser().parse_args([])
    assert nvfp4_args.workload == "rmsnorm_2048"
    nvfp4 = add_rmsnorm_demo.build_benchmark(workload_name=nvfp4_args.workload)
    assert nvfp4.optimized
    assert (nvfp4.workload.kind, nvfp4.workload.batch) == ("add_rmsnorm", 1)
    assert (nvfp4.workload.rows, nvfp4.workload.cols) == (128, 2048)


def test_transformer_engine_demo_rejects_nonfinite_training_evidence() -> None:
    from ch13.transformer_engine_fp8_demo import (
        TrainingStepResult,
        assert_finite_training_step,
    )

    finite = TrainingStepResult(
        output=torch.ones(2, 2),
        loss=torch.ones(()),
        input_gradient=torch.ones(2, 2),
        parameter_gradients={"weight": torch.ones(2, 2)},
    )
    assert_finite_training_step("finite", finite)

    nonfinite = TrainingStepResult(
        output=torch.ones(2, 2),
        loss=torch.tensor(float("nan")),
        input_gradient=torch.ones(2, 2),
        parameter_gradients={"weight": torch.ones(2, 2)},
    )
    with pytest.raises(AssertionError, match="nonfinite loss contains nonfinite values"):
        assert_finite_training_step("nonfinite", nonfinite)
