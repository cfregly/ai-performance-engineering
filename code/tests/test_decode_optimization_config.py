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

    assert "self.graph_logits" not in source
    assert "self.graph_next_token" not in source
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
    assert "return hidden, self._decode_next_token" in decode_step_section
    assert "return logits, hidden, self._decode_next_token" not in decode_step_section


def test_decode_hot_loops_do_not_bind_unused_logits() -> None:
    common_source = (REPO_ROOT / "labs" / "decode_optimization" / "decode_common.py").read_text(
        encoding="utf-8"
    )
    baseline_warp_source = (
        REPO_ROOT / "labs" / "decode_optimization" / "baseline_decode_warp_specialized.py"
    ).read_text(encoding="utf-8")
    optimized_warp_source = (
        REPO_ROOT / "labs" / "decode_optimization" / "optimized_decode_warp_specialized.py"
    ).read_text(encoding="utf-8")

    for source in (common_source, baseline_warp_source, optimized_warp_source):
        assert "_, next_state, next_token = self.decode_fn" not in source
        assert "logits, next_state, next_token = self.decode_fn" not in source
        assert "next_state, next_token = self.decode_fn" in source


def test_decode_common_inference_paths_skip_autograd_bookkeeping() -> None:
    source = (REPO_ROOT / "labs" / "decode_optimization" / "decode_common.py").read_text(
        encoding="utf-8"
    )
    init_model_section = source.split("def _init_model", maxsplit=1)[1].split(
        "def _cache_te_weight_workspaces",
        maxsplit=1,
    )[0]
    te_cache_section = source.split("def _cache_te_weight_workspaces", maxsplit=1)[1].split(
        "def _init_buffers",
        maxsplit=1,
    )[0]
    prefill_section = source.split("def _prefill", maxsplit=1)[1].split(
        "def _decode_step",
        maxsplit=1,
    )[0]
    decode_step_section = source.split("def _decode_step", maxsplit=1)[1].split(
        "def _get_fp8_context",
        maxsplit=1,
    )[0]

    assert "with torch.inference_mode(), te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):" in te_cache_section
    assert "with torch.inference_mode(), self.sdpa_ctx_factory():" in prefill_section
    assert "with torch.inference_mode(), self.sdpa_ctx_factory():" in decode_step_section
    assert "with torch.inference_mode():" in init_model_section
    assert "with torch.no_grad():" not in init_model_section
    assert "with torch.no_grad(), te.fp8_autocast" not in te_cache_section
    assert "with torch.no_grad(), self.sdpa_ctx_factory()" not in prefill_section
    assert "with torch.no_grad(), self.sdpa_ctx_factory()" not in decode_step_section


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
    assert "self._nvtx_labels = {" in setup_section
    assert 'standardize_nvtx_label("compute_math:prefill")' in setup_section
    assert 'standardize_nvtx_label("compute_math:decode")' in setup_section
    assert "import torch.cuda.nvtx" not in prefetch_section
    assert "import torch.cuda.nvtx" not in benchmark_section
    assert "nvtx = self._nvtx" in prefetch_section
    assert "nvtx = self._nvtx" in benchmark_section
    assert "standardize_nvtx_label(" not in prefetch_section
    assert "standardize_nvtx_label(" not in benchmark_section
    assert 'nvtx.range_push(self._nvtx_labels["prefill_decode_0"])' in prefetch_section
    assert 'nvtx.range_push(self._nvtx_labels["prefill_decode_1"])' in prefetch_section
    assert 'nvtx.range_push(self._nvtx_labels["prefill"])' in benchmark_section
    assert 'nvtx.range_push(self._nvtx_labels["decode"])' in benchmark_section


def test_decode_iteration_metrics_reuse_tuple_event_state() -> None:
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
    finalize_section = source.split("def finalize_iteration_metrics", maxsplit=1)[1].split(
        "def _finalize_output",
        maxsplit=1,
    )[0]

    assert "self._timing_event_tuple = (" in setup_section
    assert "iter_start, batch0_end, _, iter_end = self._timing_event_tuple" in prefetch_section
    assert "prefill_start, prefill_end, decode_start, decode_end = self._timing_event_tuple" in benchmark_section
    assert "self._pending_iteration_events = (iter_start, batch0_end, batch0_end, iter_end)" in prefetch_section
    assert "self._pending_iteration_events = (prefill_start, prefill_end, decode_start, decode_end)" in benchmark_section
    assert "prefill_start, prefill_end, decode_start, decode_end = self._pending_iteration_events" in finalize_section
    assert "self._pending_iteration_events = {" not in source
    assert 'self._pending_iteration_events["' not in source


def test_decode_prompt_copy_waits_on_consumer_stream() -> None:
    source = (REPO_ROOT / "labs" / "decode_optimization" / "decode_common.py").read_text(
        encoding="utf-8"
    )
    copy_section = source.split("def _copy_prompts_to_device", maxsplit=1)[1].split(
        "def _copy_prompt_to_device_idx",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def finalize_iteration_metrics",
        maxsplit=1,
    )[0]

    assert "wait_stream: Optional[torch.cuda.Stream] = None" in copy_section
    assert "active_stream = self.copy_stream or wait_stream or current_stream" in copy_section
    assert "wait_stream.wait_stream(active_stream)" in copy_section
    assert "current_stream.wait_stream(active_stream)" in copy_section
    assert "torch.cuda.current_stream().wait_stream(self.copy_stream)" not in copy_section
    assert "copy_wait_stream = (" in benchmark_section
    assert "self._copy_prompts_to_device(wait_stream=copy_wait_stream)" in benchmark_section


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
        assert bench._timing_event_tuple is not None
        tuple_ids = tuple(id(event) for event in bench._timing_event_tuple)

        bench.benchmark_fn()
        bench.finalize_iteration_metrics()
        bench.benchmark_fn()
        bench.finalize_iteration_metrics()

        assert set(event_ids) == {"prefill_start", "prefill_end", "decode_start", "decode_end"}
        assert {name: id(event) for name, event in bench._timing_events.items()} == event_ids
        assert bench._timing_event_tuple is not None
        assert tuple(id(event) for event in bench._timing_event_tuple) == tuple_ids
    finally:
        bench.teardown()
