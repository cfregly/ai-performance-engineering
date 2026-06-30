from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_ch15_optimized_dep2_parallel_fails_fast_when_compile_breaks() -> None:
    source = _read("ch15/optimized_dep2_parallel.py")
    assert "FAIL FAST: torch.compile failed for optimized_dep2_parallel" in source
    assert "self._compiled_fn = _run" not in source


def test_ch15_disaggregated_inference_no_longer_swallows_sdpa_backend_failures() -> None:
    source = _read("ch15/disaggregated_inference_multigpu.py")
    assert "falling back gracefully" not in source
    assert "return nullcontext()" not in source
    assert "return sdpa_kernel(list(PREFERRED_SDP_BACKENDS))" in source


def test_ch18_paged_attention_sparse_path_fails_fast_on_compile_errors() -> None:
    source = _read("ch18/paged_attn_split_common.py")
    assert "FAIL FAST: torch.compile(flex_attention) failed for paged attention sparse path" in source
    assert "return flex_attention" not in source


def test_ch18_nvfp4_trtllm_tool_no_longer_uses_placeholder_outputs_or_eager_fallback() -> None:
    source = _read("ch18/nvfp4_trtllm_tool.py")
    setup_section = source.split("def setup", maxsplit=1)[1].split("def benchmark_fn", maxsplit=1)[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert "torch.tensor([float(len(outputs))]" not in source
    assert "FAIL FAST: TRT-LLM generate returned an unsupported output payload" in source
    assert "FAIL FAST: Transformer Engine FP8 path failed in nvfp4_trtllm_tool" in source
    assert "self._enable_nvtx = get_nvtx_enabled(config) if config else False" in setup_section
    assert "self._empty_iteration_result = {}" in source
    assert "self._fp8_autocast = None" in source
    assert "self._fp8_autocast = fp8_autocast" in setup_section
    assert "with self._fp8_autocast():" in benchmark_section
    assert "from transformer_engine.pytorch import fp8_autocast" not in benchmark_section
    assert 'if self._trt_runner is not None and self.inputs is not None:' in benchmark_section
    assert 'hasattr(self, "_trt_runner")' not in benchmark_section
    assert "get_nvtx_enabled(" not in benchmark_section
    assert "self.get_config()" not in benchmark_section
    assert "torch.as_tensor(" not in benchmark_section
    assert 'first = outputs.get("output_ids")' in benchmark_section
    assert 'raise TypeError(f"expected Tensor output, got {type(first).__name__}")' in benchmark_section
    assert benchmark_section.count("return self._empty_iteration_result") == 2
    assert "return {}" not in benchmark_section
