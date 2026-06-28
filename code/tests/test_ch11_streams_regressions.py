from pathlib import Path

import torch

from ch11.baseline_streams import BaselineStreamsBenchmark
from ch11.optimized_streams import OptimizedStreamsBenchmark


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_optimized_streams_compute_reuses_preallocated_result_buffers() -> None:
    source = (REPO_ROOT / "ch11" / "optimized_streams.py").read_text(encoding="utf-8")
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def teardown",
        maxsplit=1,
    )[0]
    compute_section = source.split("def _compute", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]

    assert "self._scratch0: Optional[torch.Tensor] = None" in source
    assert "self._scratch1: Optional[torch.Tensor] = None" in source
    assert "self._scratch_pair: Optional[tuple[torch.Tensor, torch.Tensor]] = None" in source
    assert "self._chunk_triplets: List[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []" in source
    assert "self._scratch0 = torch.empty(self.N, dtype=torch.float32, device=self.device)" in source
    assert "self._scratch1 = torch.empty(self.N, dtype=torch.float32, device=self.device)" in source
    assert "self._scratch_pair = (self._scratch0, self._scratch1)" in source
    assert "self._chunk_triplets = list(zip(self.host_data, self.device_data, self.results, strict=True))" in source
    assert "with torch.inference_mode(), self._nvtx_range(\"streams_pipelined\"):" in benchmark_section
    assert "chunks = self._chunk_triplets" in benchmark_section
    assert "first_device.copy_(first_host, non_blocking=True)" in benchmark_section
    assert "for i, (_, device_chunk, result_chunk) in enumerate(chunks):" in benchmark_section
    assert "next_device.copy_(next_host, non_blocking=True)" in benchmark_section
    assert "self._compute(device_chunk, result_chunk)" in benchmark_section
    assert "self.device_data[i]" not in benchmark_section
    assert "self.host_data[i]" not in benchmark_section
    assert "self.results[i]" not in benchmark_section
    assert "self.results[i] = self._compute" not in benchmark_section
    assert "scratch0, scratch1 = self._scratch_pair" in compute_section
    assert ".view_as(data)" not in compute_section
    assert "torch.sin(result, out=scratch0)" in compute_section
    assert "torch.cos(result, out=scratch1)" in compute_section
    assert "torch.tanh(scratch0, out=out)" in compute_section
    assert "torch.sigmoid(scratch0, out=scratch1)" in compute_section

    bench = OptimizedStreamsBenchmark()
    data = torch.linspace(-0.25, 0.25, 32, dtype=torch.float32)
    out = torch.empty_like(data)
    bench._scratch0 = torch.empty_like(data)
    bench._scratch1 = torch.empty_like(data)
    bench._scratch_pair = (bench._scratch0, bench._scratch1)

    expected = data
    for _ in range(3):
        expected = torch.sin(expected) * torch.cos(expected) + expected * 0.1
        expected = torch.tanh(expected) + torch.sigmoid(expected) * 0.5

    actual = bench._compute(data, out)

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected)


def test_baseline_streams_compute_reuses_preallocated_result_buffers() -> None:
    source = (REPO_ROOT / "ch11" / "baseline_streams.py").read_text(encoding="utf-8")
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def teardown",
        maxsplit=1,
    )[0]
    compute_section = source.split("def _compute", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]

    assert "self._scratch0: Optional[torch.Tensor] = None" in source
    assert "self._scratch1: Optional[torch.Tensor] = None" in source
    assert "self._scratch_pair: Optional[tuple[torch.Tensor, torch.Tensor]] = None" in source
    assert "self._chunk_triplets: List[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []" in source
    assert "self._scratch0 = torch.empty(self.N, dtype=torch.float32, device=self.device)" in source
    assert "self._scratch1 = torch.empty(self.N, dtype=torch.float32, device=self.device)" in source
    assert "self._scratch_pair = (self._scratch0, self._scratch1)" in source
    assert "self._chunk_triplets = list(zip(self.host_data, self.device_data, self.results, strict=True))" in source
    assert "with torch.inference_mode(), self._nvtx_range(\"baseline_streams_sequential\"):" in benchmark_section
    assert "for host_chunk, device_chunk, result_chunk in self._chunk_triplets:" in benchmark_section
    assert "device_chunk.copy_(host_chunk)" in benchmark_section
    assert "self._compute(device_chunk, result_chunk)" in benchmark_section
    assert "self.device_data[i]" not in benchmark_section
    assert "self.host_data[i]" not in benchmark_section
    assert "self.results[i]" not in benchmark_section
    assert "self.results[i] = self._compute" not in benchmark_section
    assert "scratch0, scratch1 = self._scratch_pair" in compute_section
    assert ".view_as(data)" not in compute_section
    assert "torch.sin(result, out=scratch0)" in compute_section
    assert "torch.cos(result, out=scratch1)" in compute_section
    assert "torch.tanh(scratch0, out=out)" in compute_section
    assert "torch.sigmoid(scratch0, out=scratch1)" in compute_section

    bench = BaselineStreamsBenchmark()
    data = torch.linspace(-0.25, 0.25, 32, dtype=torch.float32)
    out = torch.empty_like(data)
    bench._scratch0 = torch.empty_like(data)
    bench._scratch1 = torch.empty_like(data)
    bench._scratch_pair = (bench._scratch0, bench._scratch1)

    expected = data
    for _ in range(3):
        expected = torch.sin(expected) * torch.cos(expected) + expected * 0.1
        expected = torch.tanh(expected) + torch.sigmoid(expected) * 0.5

    actual = bench._compute(data, out)

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected)


def test_streams_verification_reuses_sample_output_buffer() -> None:
    for filename in ("baseline_streams.py", "optimized_streams.py"):
        source = (REPO_ROOT / "ch11" / filename).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split(
            "def _compute",
            maxsplit=1,
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def get_benchmark",
            maxsplit=1,
        )[0]
        teardown_section = source.split("def teardown", maxsplit=1)[1].split(
            "def get_config",
            maxsplit=1,
        )[0]

        assert "self._verify_output: Optional[torch.Tensor] = None" in source
        assert "self._verify_indices: tuple[int, int, int] = ()" in source
        assert "self._verify_slice_len = 0" in source
        assert "self._verify_slice_len = min(256, self.N)" in setup_section
        assert "self._verify_indices = (0, self.num_chunks // 2, self.num_chunks - 1)" in setup_section
        assert "self._verify_output = torch.empty(" in setup_section
        assert "torch.cat(" not in capture_section
        assert "[self.results[i][:256].detach() for i in indices]" not in capture_section
        assert "for slot, result_idx in enumerate(self._verify_indices):" in capture_section
        assert "self._verify_output[start : start + slice_len].copy_(self.results[result_idx][:slice_len])" in capture_section
        assert "output=self._verify_output" in capture_section
        assert "self._verify_output = None" in teardown_section
        assert "self._verify_indices = ()" in teardown_section
