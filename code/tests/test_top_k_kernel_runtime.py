from __future__ import annotations

from pathlib import Path

import pytest
import torch

import labs.top_k_kernel.top_k_kernel_common as topk_common
from labs.top_k_kernel.top_k_kernel_common import TopKKernelBenchmark, TopKKernelWorkload

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_softmax_topk_backward_reuses_grad_probs_cast() -> None:
    source = (REPO_ROOT / "labs" / "top_k_kernel" / "top_k_kernel_common.py").read_text(
        encoding="utf-8"
    )
    backward_section = source.split("def _softmax_topk_backward", maxsplit=1)[1].split(
        "def _reshape_group_rows", maxsplit=1
    )[0]

    assert "grad_probs_float = grad_probs.float()" in backward_section
    assert backward_section.count("grad_probs.float()") == 1
    assert "q[..., 0] += 1.0" in source
    assert "torch.tensor(1.0" not in source


def test_top_k_input_block_bias_uses_position_arithmetic() -> None:
    source = (REPO_ROOT / "labs" / "top_k_kernel" / "top_k_kernel_common.py").read_text(
        encoding="utf-8"
    )
    build_inputs_section = source.split("def build_inputs", maxsplit=1)[1].split(
        "def _group_q",
        maxsplit=1,
    )[0]

    assert ".repeat_interleave(workload.positions_per_block)" not in build_inputs_section
    assert "torch.arange(workload.compressed_k_len, dtype=torch.int64)" in build_inputs_section
    assert '.div_(workload.positions_per_block, rounding_mode="floor")' in build_inputs_section


def test_top_k_repeats_k_heads_with_expand_reshape() -> None:
    source = (REPO_ROOT / "labs" / "top_k_kernel" / "top_k_kernel_common.py").read_text(
        encoding="utf-8"
    )
    repeat_section = source.split("def _repeat_k_over_query_heads", maxsplit=1)[1].split(
        "def _build_block_k",
        maxsplit=1,
    )[0]

    assert "repeat_interleave(" not in repeat_section
    assert ".expand(batch_size, kv_heads, workload.gqa_size, k_len, head_dim)" in repeat_section
    assert ".reshape(batch_size, kv_heads * workload.gqa_size, k_len, head_dim)" in repeat_section

    workload = TopKKernelWorkload(heads=4, kv_heads=2, compressed_k_len=3, head_dim=1)
    k = torch.arange(6, dtype=torch.float32).view(1, 2, 3, 1)

    repeated = topk_common._repeat_k_over_query_heads(k, workload)

    expected = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5], dtype=torch.float32).view(1, 4, 3, 1)
    torch.testing.assert_close(repeated, expected)


def test_compare_top_k_matrix_uses_cuda_event_timing() -> None:
    source = (REPO_ROOT / "labs" / "top_k_kernel" / "compare_top_k_matrix.py").read_text(
        encoding="utf-8"
    )
    measure_section = source.split("def _measure_case", maxsplit=1)[1].split(
        "def _print_table",
        maxsplit=1,
    )[0]

    assert measure_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "start.record()" in measure_section
    assert "end.record()" in measure_section
    assert "end.synchronize()" in measure_section
    assert "return start.elapsed_time(end) / iters" in measure_section
    assert "time.perf_counter()" not in measure_section
    assert "time.time()" not in measure_section


@CUDA_REQUIRED
def test_forward_benchmark_uses_inference_mode_without_output_clones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = TopKKernelBenchmark(backend="baseline", label="baseline_topk_runtime_test")
    bench.apply_target_overrides(
        [
            "--batch-size",
            "1",
            "--heads",
            "2",
            "--kv-heads",
            "1",
            "--q-len",
            "64",
            "--compressed-k-len",
            "64",
            "--head-dim",
            "32",
            "--top-k",
            "4",
            "--selection-block-size",
            "16",
            "--mode",
            "forward",
            "--dtype",
            "fp16",
        ]
    )

    bench.setup()
    try:
        probs = torch.randn(1, 2, 64, 4, device="cuda", dtype=torch.float32)
        indices = torch.zeros(1, 2, 64, 4, device="cuda", dtype=torch.long)
        calls = 0

        def fake_select(
            q: torch.Tensor,
            k: torch.Tensor,
            workload: topk_common.TopKKernelWorkload,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal calls
            calls += 1
            assert q is bench.inputs.q
            assert k is bench.inputs.k
            assert workload is bench.workload
            assert torch.is_inference_mode_enabled()
            return probs, indices

        monkeypatch.setattr(topk_common, "baseline_top_k_select", fake_select)

        bench.benchmark_fn()

        assert calls == 1
        assert bench.outputs is not None
        assert bench.outputs.probs.data_ptr() == probs.data_ptr()
        assert bench.outputs.indices.data_ptr() == indices.data_ptr()
    finally:
        bench.teardown()
