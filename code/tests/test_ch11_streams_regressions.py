from pathlib import Path

import torch

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
    assert "self._scratch0 = torch.empty(self.N, dtype=torch.float32, device=self.device)" in source
    assert "self._scratch1 = torch.empty(self.N, dtype=torch.float32, device=self.device)" in source
    assert "self._compute(self.device_data[i], self.results[i])" in benchmark_section
    assert "self.results[i] = self._compute" not in benchmark_section
    assert "torch.sin(result, out=scratch0)" in compute_section
    assert "torch.cos(result, out=scratch1)" in compute_section
    assert "torch.tanh(scratch0, out=out)" in compute_section
    assert "torch.sigmoid(scratch0, out=scratch1)" in compute_section

    bench = OptimizedStreamsBenchmark()
    data = torch.linspace(-0.25, 0.25, 32, dtype=torch.float32)
    out = torch.empty_like(data)
    bench._scratch0 = torch.empty_like(data)
    bench._scratch1 = torch.empty_like(data)

    expected = data
    for _ in range(3):
        expected = torch.sin(expected) * torch.cos(expected) + expected * 0.1
        expected = torch.tanh(expected) + torch.sigmoid(expected) * 0.5

    actual = bench._compute(data, out)

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected)
