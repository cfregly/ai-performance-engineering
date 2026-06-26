from __future__ import annotations

from pathlib import Path

import pytest
import torch

import labs.top_k_kernel.top_k_kernel_common as topk_common
from labs.top_k_kernel.top_k_kernel_common import TopKKernelBenchmark

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
