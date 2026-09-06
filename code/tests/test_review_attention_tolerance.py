"""Verification regressions for the FlashAttention benchmark pair."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from core.benchmark.verify_runner import VerifyRunner

LAB = Path(__file__).parents[1] / "labs" / "flashattention_gluon"
CODE_ROOT = LAB.parents[1]


def _load_cpu_baseline_class():
    """Load the real baseline module without requiring Triton for its shared input type."""

    common_name = "labs.flashattention_gluon.flashattention_gluon_common"
    previous_common = sys.modules.get(common_name)
    common = types.ModuleType(common_name)

    class FlashAttentionInputs:
        def __init__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
            self.q = q
            self.k = k
            self.v = v

    def build_flashattention_inputs(*, batch, seq_len, heads, head_dim, dtype, device):
        shape = (batch, heads, seq_len, head_dim)
        generator = torch.Generator(device=device).manual_seed(torch.initial_seed())
        q = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        k = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        v = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        return FlashAttentionInputs(q=q, k=k, v=v)

    common.FlashAttentionInputs = FlashAttentionInputs
    common.build_flashattention_inputs = build_flashattention_inputs
    sys.modules[common_name] = common
    try:
        source = LAB / "baseline_flashattention_gluon.py"
        spec = importlib.util.spec_from_file_location("_review_cpu_flashattention_baseline", source)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.BaselineFlashAttentionGluonBenchmark
    finally:
        if previous_common is None:
            sys.modules.pop(common_name, None)
        else:
            sys.modules[common_name] = previous_common


def _declared_output_tolerance(path: Path) -> tuple[float, float]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tolerances: list[tuple[float, float]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            not isinstance(node.func, ast.Attribute)
            or node.func.attr != "_set_verification_payload"
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == "output_tolerance":
                tolerances.append(ast.literal_eval(keyword.value))
    assert len(tolerances) == 1
    return tolerances[0]


@pytest.mark.parametrize("precision", ["medium", "high", "highest"])
def test_strict_fp32_matmul_restores_every_precision_surface(precision: str) -> None:
    """The strict scope must restore each supported caller precision policy."""

    probe = r'''
import importlib.util
import json
import sys
import types

import torch

def snapshot():
    return {
        "global": torch.get_float32_matmul_precision(),
        "legacy_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "legacy_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "new_matmul": str(torch.backends.cuda.matmul.fp32_precision),
        "new_cudnn": str(torch.backends.cudnn.conv.fp32_precision),
    }

common_name = "labs.flashattention_gluon.flashattention_gluon_common"
common = types.ModuleType(common_name)
common.FlashAttentionInputs = object
common.build_flashattention_inputs = lambda **_kwargs: None
sys.modules[common_name] = common
source = sys.argv[1]
# Let the harness install its CUDA precision policy and compatibility accessors
# on CUDA-enabled PyTorch builds without launching a kernel. The caller policy
# is established after import because benchmark_harness intentionally applies
# its process policy while the baseline module is imported.
torch.cuda.is_available = lambda: True
spec = importlib.util.spec_from_file_location("_fresh_flashattention_baseline", source)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

precision = sys.argv[2]
module.configure_tf32(matmul_precision=precision, cudnn_precision=precision)
torch.set_float32_matmul_precision(precision)
tf32_enabled = precision != "highest"
torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
torch.backends.cudnn.allow_tf32 = tf32_enabled

before = snapshot()
with module._strict_fp32_matmul():
    inside = snapshot()
after = snapshot()
print(json.dumps({"before": before, "inside": inside, "after": after}, sort_keys=True))
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(CODE_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            probe,
            str(LAB / "baseline_flashattention_gluon.py"),
            precision,
        ],
        cwd=CODE_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    state = json.loads(result.stdout.strip().splitlines()[-1])

    assert state["inside"]["global"] == "highest"
    assert state["inside"]["new_matmul"] == "ieee"
    assert state["inside"]["legacy_matmul"] is False
    assert state["before"]["global"] == precision
    expected_backend = "ieee" if precision == "highest" else "tf32"
    assert state["before"]["new_matmul"] == expected_backend
    assert state["before"]["new_cudnn"] == expected_backend
    assert state["before"]["legacy_matmul"] is (precision != "highest")
    assert state["before"]["legacy_cudnn"] is (precision != "highest")
    assert state["after"] == state["before"]


@pytest.mark.parametrize(
    "benchmark_source",
    [
        LAB / "baseline_flashattention_gluon.py",
        LAB / "optimized_flashattention_gluon.py",
    ],
)
def test_fp16_payload_tolerance_rejects_zero_output(benchmark_source: Path) -> None:
    tolerance = _declared_output_tolerance(benchmark_source)
    # These FP32 tensors model the benchmark transport buffers, whose values
    # have already been rounded to FP16 by the measured implementations.
    captured = torch.tensor([0.5, -0.25, 0.125, -0.0625], dtype=torch.float32)
    malformed = torch.zeros_like(captured)
    runner = VerifyRunner()

    assert tolerance == (1e-3, 1e-5)
    assert runner.compare_perf_outputs(captured, captured.clone(), tolerance).passed
    assert runner.compare_perf_outputs(captured, malformed, (0.1, 1.0)).passed
    comparison = runner.compare_perf_outputs(captured, malformed, tolerance)
    assert not comparison.passed
    assert comparison.max_diff == 0.5


@pytest.mark.parametrize(
    ("shape", "seed"),
    [
        ((1, 1, 1, 1), 0),
        ((1, 1, 65, 1), 1),
        ((2, 3, 65, 40), 0),
        ((2, 3, 65, 40), 1),
        ((2, 3, 65, 40), 2),
    ],
)
def test_actual_cpu_baseline_matches_fp64_oracle_rounded_once(
    shape: tuple[int, int, int, int], seed: int
) -> None:
    """The baseline keeps FP16 I/O but rounds strict reference math only once."""

    previous_precision = torch.get_float32_matmul_precision()
    benchmark = _load_cpu_baseline_class()()
    benchmark._device = torch.device("cpu")
    batch, heads, seq_len, head_dim = shape
    benchmark.batch = batch
    benchmark.batch_size = batch
    benchmark.heads = heads
    benchmark.seq_len = seq_len
    benchmark.head_dim = head_dim
    benchmark.hidden_dim = heads * head_dim
    benchmark.dtype = torch.float16

    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    ambient_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    try:
        benchmark.setup()
        benchmark.benchmark_fn()
        assert torch.get_float32_matmul_precision() == "high"
        assert bool(torch.backends.cuda.matmul.allow_tf32) == ambient_tf32
        assert benchmark.inputs is not None
        assert benchmark.output is not None
        assert benchmark.output.dtype == torch.float16
        assert all(item.dtype == torch.float16 for item in (
            benchmark.inputs.q,
            benchmark.inputs.k,
            benchmark.inputs.v,
        ))

        q = benchmark.inputs.q.double()
        k = benchmark.inputs.k.double()
        v = benchmark.inputs.v.double()
        oracle = (
            torch.softmax(
                torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim),
                dim=-1,
            )
            @ v
        ).to(torch.float16)

        benchmark.capture_verification_payload()
        comparison = VerifyRunner().compare_perf_outputs(
            oracle.float(),
            benchmark.get_verify_output(),
            benchmark.get_output_tolerance(),
        )
        assert comparison.passed, comparison

        verify_inputs = benchmark.get_verify_inputs()
        assert set(verify_inputs) == {"q", "k", "v"}
        assert all(tuple(item.shape) == shape for item in verify_inputs.values())
        assert all(item.dtype == torch.float16 for item in verify_inputs.values())
        signature = benchmark.get_input_signature()
        assert signature.precision_flags.fp16
        assert not signature.precision_flags.bf16
        assert signature.precision_flags.tf32 == ambient_tf32
        assert torch.get_float32_matmul_precision() == "high"
        assert bool(torch.backends.cuda.matmul.allow_tf32) == ambient_tf32
    finally:
        benchmark.teardown()
        torch.set_float32_matmul_precision(previous_precision)


def test_cpu_benchmark_does_not_synchronize_cuda_on_cuda_visible_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit CPU benchmark stays on CPU when the host also exposes CUDA."""

    benchmark = _load_cpu_baseline_class()()
    benchmark._device = torch.device("cpu")
    synchronize_calls: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize_calls.append)

    benchmark._synchronize()

    assert synchronize_calls == []
