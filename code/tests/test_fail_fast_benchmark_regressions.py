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
    assert "from transformer_engine" not in source
    assert "nn.Linear" not in source
    assert "fp8_autocast" not in source
    assert "from_engine" not in source
    assert "torch.cuda.graph" not in source
    assert "torch.cuda.CUDAGraph" not in source
    assert "ModelRunner" in setup_section
    assert "self._trt_runner = ModelRunner.from_dir(" in setup_section
    assert "_inspect_nvfp4_engine_assets(self._configured_engine_dir())" in setup_section
    assert "FAIL FAST: TensorRT-LLM failed to load the validated NVFP4 engine" in setup_section
    assert "self._enable_nvtx = get_nvtx_enabled(config) if config else False" in setup_section
    assert "self._empty_iteration_result: dict[str, float] = {}" in source
    assert 'outputs = self._trt_runner.generate(' in benchmark_section
    assert "max_new_tokens=self.max_new_tokens" in benchmark_section
    assert "end_id=0" in benchmark_section
    assert "pad_id=0" in benchmark_section
    assert "return_dict=True" in benchmark_section
    assert "self.output = self._require_output_ids(outputs)" in benchmark_section
    assert "FAIL FAST: TensorRT-LLM NVFP4 generate failed" in benchmark_section
    assert "get_nvtx_enabled(" not in benchmark_section
    assert "self.get_config()" not in benchmark_section
    assert benchmark_section.count("return self._empty_iteration_result") == 1
    assert "return {}" not in benchmark_section
