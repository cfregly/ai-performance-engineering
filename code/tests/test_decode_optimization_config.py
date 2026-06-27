from pathlib import Path

import pytest
import torch

from core.harness.benchmark_harness import ExecutionMode
from labs.decode_optimization.baseline_decode import get_benchmark as get_baseline_decode
from labs.decode_optimization.baseline_decode_pinned import (
    get_benchmark as get_baseline_decode_pinned,
)
from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig
from labs.decode_optimization.optimized_decode_graph import (
    get_benchmark as get_optimized_decode_graph,
)
from labs.decode_optimization.optimized_decode_pinned import (
    get_benchmark as get_optimized_decode_pinned,
)
from labs.decode_optimization.optimized_decode_ultimate import (
    get_benchmark as get_optimized_decode_ultimate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_decode_benchmark_uses_subprocess_execution() -> None:
    bench = DecodeBenchmark(DecodeConfig())

    config = bench.get_config()

    assert config.use_subprocess is True
    assert config.execution_mode == ExecutionMode.SUBPROCESS


def test_decode_variants_inherit_subprocess_execution() -> None:
    for factory in (
        get_baseline_decode,
        get_baseline_decode_pinned,
        get_optimized_decode_pinned,
        get_optimized_decode_graph,
        get_optimized_decode_ultimate,
    ):
        config = factory().get_config()
        assert config.use_subprocess is True
        assert config.execution_mode == ExecutionMode.SUBPROCESS


def test_decode_graph_capture_avoids_unused_full_vocab_output_copy() -> None:
    source = (REPO_ROOT / "labs" / "decode_optimization" / "decode_common.py").read_text(
        encoding="utf-8"
    )
    graph_section = source.split("def _capture_decode_graph", maxsplit=1)[1].split(
        "def _prefill", maxsplit=1
    )[0]

    assert "self.graph_logits = torch.empty" not in graph_section
    assert "self.graph_next_token = torch.empty" not in graph_section
    assert "self.graph_logits.copy_" not in graph_section
    assert "self.graph_next_token.copy_" not in graph_section
    assert "self.next_token_out = torch.empty_like" not in source
    assert graph_section.count("self.current_tokens.copy_(next_token)") == 2
    assert "if self.cfg.graph_full_iteration:\n                    self.current_tokens.copy_(next_token)" not in graph_section


def test_decode_state_buffers_are_overwritten_without_zero_fill() -> None:
    source = (REPO_ROOT / "labs" / "decode_optimization" / "decode_common.py").read_text(
        encoding="utf-8"
    )
    init_section = source.split("def _init_buffers", maxsplit=1)[1].split(
        "def _maybe_compile", maxsplit=1
    )[0]
    graph_section = source.split("def _capture_decode_graph", maxsplit=1)[1].split(
        "def _prefill", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def finalize_iteration_metrics", maxsplit=1
    )[0]

    assert "self.state_buffer = torch.empty(" in init_section
    assert "self.state_buffer = torch.zeros(" not in init_section
    assert "self.state_buffer.zero_()" not in graph_section
    assert "self.current_tokens.zero_()" not in graph_section
    assert "self.state_buffer.zero_()" not in benchmark_section
    assert "self.current_tokens.zero_()" not in benchmark_section
    assert "self.state_buffer.copy_(prefill_state)" in graph_section
    assert "self.current_tokens.copy_(self.gpu_prompt[:, -1])" in graph_section


def test_decode_step_reuses_next_token_buffer() -> None:
    source = (REPO_ROOT / "labs" / "decode_optimization" / "decode_common.py").read_text(
        encoding="utf-8"
    )
    init_section = source.split("def _init_buffers", maxsplit=1)[1].split(
        "def _maybe_compile",
        maxsplit=1,
    )[0]
    decode_step_section = source.split("def _decode_step", maxsplit=1)[1].split(
        "def _get_fp8_context",
        maxsplit=1,
    )[0]

    assert "self._decode_next_token_values: Optional[torch.Tensor] = None" in source
    assert "self._decode_next_token: Optional[torch.Tensor] = None" in source
    assert "self._decode_next_token_values = torch.empty((bsz,), device=self.device, dtype=self.dtype)" in init_section
    assert "self._decode_next_token = torch.empty((bsz,), device=self.device, dtype=torch.long)" in init_section
    assert "torch.max(logits, dim=-1, out=(self._decode_next_token_values, self._decode_next_token))" in decode_step_section
    assert "torch.argmax(logits, dim=-1)" not in decode_step_section


def test_decode_nvtx_import_is_cached_outside_iteration_hot_paths() -> None:
    source = (REPO_ROOT / "labs" / "decode_optimization" / "decode_common.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "# Model + buffer init",
        maxsplit=1,
    )[0]
    prefetch_section = source.split("def _benchmark_prefetch_batches", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def finalize_iteration_metrics",
        maxsplit=1,
    )[0]

    assert "self._nvtx = _cuda_nvtx()" in setup_section
    assert "import torch.cuda.nvtx" not in prefetch_section
    assert "import torch.cuda.nvtx" not in benchmark_section
    assert "nvtx = self._nvtx" in prefetch_section
    assert "nvtx = self._nvtx" in benchmark_section


def test_decode_pinned_pair_uses_transfer_heavy_workload_with_only_pin_state_changed() -> None:
    baseline = get_baseline_decode_pinned()
    optimized = get_optimized_decode_pinned()

    assert baseline.cfg.host_payload_mb == 512
    assert optimized.cfg.host_payload_mb == 512
    assert baseline.cfg.batch_size == optimized.cfg.batch_size == 64
    assert baseline.cfg.prompt_tokens == optimized.cfg.prompt_tokens == 2048
    assert baseline.cfg.prefetch_batches == optimized.cfg.prefetch_batches == 2
    assert baseline.cfg.hidden_size == optimized.cfg.hidden_size == 256
    assert baseline.cfg.use_pinned_host is False
    assert optimized.cfg.use_pinned_host is True
    assert baseline.cfg.use_copy_stream is False
    assert optimized.cfg.use_copy_stream is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for decode event setup")
def test_decode_benchmark_reuses_timing_events() -> None:
    bench = DecodeBenchmark(
        DecodeConfig(
            batch_size=1,
            prompt_tokens=4,
            decode_tokens=1,
            hidden_size=16,
            vocab_size=64,
            iterations=1,
            warmup=0,
        )
    )
    bench.setup()
    try:
        event_ids = {name: id(event) for name, event in bench._timing_events.items()}

        bench.benchmark_fn()
        bench.finalize_iteration_metrics()
        bench.benchmark_fn()
        bench.finalize_iteration_metrics()

        assert set(event_ids) == {"prefill_start", "prefill_end", "decode_start", "decode_end"}
        assert {name: id(event) for name, event in bench._timing_events.items()} == event_ids
    finally:
        bench.teardown()
