from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ch05.baseline_decompression import CPUDecompressionBenchmark
from ch05.optimized_decompression import GPUDecompressionBenchmark


REPO_ROOT = Path(__file__).resolve().parents[1]


def _assert_decompression_clone_deferred(bench) -> None:
    bench.setup()
    try:
        result = bench.benchmark_fn()
        assert result is not None
        assert bench.output is not None
        assert bench.output.numel() == result["decompressed_len"]
        output_ptr = bench.output.data_ptr()

        bench.capture_verification_payload()
        payload = bench._verification_payload
        assert payload.output.numel() == 4096
        assert bench._verify_output_buffer is not None
        assert payload.output.data_ptr() == bench._verify_output_buffer.data_ptr()
        assert payload.output.data_ptr() != output_ptr

        payload_ptr = payload.output.data_ptr()
        bench.capture_verification_payload()
        assert bench._verification_payload.output.data_ptr() == payload_ptr
    finally:
        bench.teardown()


def test_cpu_decompression_defers_full_output_clone_to_capture() -> None:
    source = (REPO_ROOT / "ch05" / "baseline_decompression.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]

    assert "counts_i64" not in source
    assert "torch.repeat_interleave(self.values, self.counts)" in source
    assert "self._verify_output_buffer: Optional[torch.Tensor] = None" in source
    assert "self._verify_output_buffer = torch.empty(4096, dtype=torch.float32)" in setup_section
    assert "self._verify_output_buffer.copy_(self.output[: self._verify_output_buffer.numel()])" in capture_section
    assert "output=self._verify_output_buffer" in capture_section
    assert "self.output[:4096].detach().clone()" not in capture_section
    assert '"counts": counts' in capture_section
    assert '"values": values' in capture_section
    assert "counts.detach().clone()" not in capture_section
    assert "values.detach().clone()" not in capture_section
    assert 'with torch.inference_mode(), self._nvtx_range("cpu_decompress"):' in benchmark_section
    assert "torch.no_grad()" not in benchmark_section
    assert "with self._nvtx_range(" not in benchmark_section
    assert "get_nvtx_enabled(" not in benchmark_section
    assert "with nvtx_range(" not in benchmark_section
    assert "from core.profiling.nvtx_helper" not in source
    assert "self._run_len = int(run_len)" in setup_section
    assert "self._decompressed_len = int(total_len)" in setup_section
    assert 'self._result_metrics = {"latency_ms": 0.0, "decompressed_len": 0}' in source
    assert 'self._result_metrics["latency_ms"] = latency_ms' in benchmark_section
    assert 'self._result_metrics["decompressed_len"] = self._decompressed_len' in benchmark_section
    assert "return self._result_metrics" in benchmark_section
    assert 'return {"latency_ms": latency_ms' not in benchmark_section
    assert "self.output = decompressed" in benchmark_section
    assert "decompressed.detach()" not in benchmark_section
    assert "self.counts[0].item()" not in source
    assert "run_length = self._run_len if run_count > 0 else 0" in source

    _assert_decompression_clone_deferred(CPUDecompressionBenchmark())


def test_gpu_decompression_reuses_preallocated_broadcast_output() -> None:
    source = (REPO_ROOT / "ch05" / "optimized_decompression.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]

    assert "self._output_matrix = torch.empty((num_runs, run_len)" in setup_section
    assert "self._output_flat = self._output_matrix.reshape(-1)" in setup_section
    assert "self._values_column = self.values.unsqueeze(1)" in setup_section
    assert "self._verify_output_buffer = torch.empty(4096, device=self.device, dtype=self.values.dtype)" in setup_section
    assert "self._verify_output_buffer.copy_(self.output[: self._verify_output_buffer.numel()])" in capture_section
    assert "output=self._verify_output_buffer" in capture_section
    assert "self.output[:4096].detach().clone()" not in capture_section
    assert 'inputs={"counts": counts, "values": values}' in capture_section
    assert "counts.detach().clone()" not in capture_section
    assert "values.detach().clone()" not in capture_section
    assert "counts_i64" not in source
    assert "torch.repeat_interleave" not in benchmark_section
    assert "self._output_matrix.copy_(self._values_column)" in benchmark_section
    assert "self.values.unsqueeze(1)" not in benchmark_section
    assert "out = self._output_flat" in benchmark_section
    assert "self.output = out" in benchmark_section
    assert "out.detach()" not in benchmark_section
    assert 'with torch.inference_mode(), self._nvtx_range("gpu_decompress_rle"):' in benchmark_section
    assert "torch.no_grad()" not in benchmark_section
    assert "with self._nvtx_range(" not in benchmark_section
    assert "get_nvtx_enabled(" not in benchmark_section
    assert "with nvtx_range(" not in benchmark_section
    assert "from core.profiling.nvtx_helper" not in source
    assert "self._decompressed_len = int(total_len)" in setup_section
    assert 'self._result_metrics = {"latency_ms": 0.0, "decompressed_len": 0}' in source
    assert 'self._result_metrics["latency_ms"] = latency_ms' in benchmark_section
    assert 'self._result_metrics["decompressed_len"] = self._decompressed_len' in benchmark_section
    assert "return self._result_metrics" in benchmark_section
    assert 'return {"latency_ms": latency_ms' not in benchmark_section
    assert "self.counts[0].item()" not in source
    assert "run_length = self._run_len if run_count > 0 else 0" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU decompression")
def test_gpu_decompression_defers_full_output_clone_to_capture() -> None:
    _assert_decompression_clone_deferred(GPUDecompressionBenchmark())
