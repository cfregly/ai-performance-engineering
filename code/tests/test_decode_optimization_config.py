from pathlib import Path

import pytest
import torch

from core.harness.benchmark_harness import ExecutionMode
from labs.decode_optimization.baseline_decode import get_benchmark as get_baseline_decode
from labs.decode_optimization.baseline_decode_candidate_logits import (
    get_benchmark as get_baseline_decode_candidate_logits,
)
from labs.decode_optimization.baseline_decode_pinned import (
    get_benchmark as get_baseline_decode_pinned,
)
from labs.decode_optimization.baseline_decode_device_resident import (
    get_benchmark as get_baseline_decode_device_resident,
)
from labs.decode_optimization.baseline_decode_prefix_state_cache import (
    get_benchmark as get_baseline_decode_prefix_state_cache,
)
from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig
from labs.decode_optimization.optimized_decode_candidate_logits import (
    get_benchmark as get_optimized_decode_candidate_logits,
)
from labs.decode_optimization.optimized_decode_device_resident import (
    get_benchmark as get_optimized_decode_device_resident,
)
from labs.decode_optimization.optimized_decode_prefix_state_cache import (
    get_benchmark as get_optimized_decode_prefix_state_cache,
)
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
        get_baseline_decode_candidate_logits,
        get_baseline_decode_pinned,
        get_baseline_decode_device_resident,
        get_baseline_decode_prefix_state_cache,
        get_optimized_decode_candidate_logits,
        get_optimized_decode_device_resident,
        get_optimized_decode_prefix_state_cache,
        get_optimized_decode_pinned,
        get_optimized_decode_graph,
        get_optimized_decode_ultimate,
    ):
        config = factory().get_config()
        assert config.use_subprocess is True
        assert config.execution_mode == ExecutionMode.SUBPROCESS


def test_decode_common_caches_runtime_feature_handles() -> None:
    source = (REPO_ROOT / "labs" / "decode_optimization" / "decode_common.py").read_text(
        encoding="utf-8"
    )
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "if self.cfg.use_copy_stream", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def _signature_parameter_count", maxsplit=1
    )[0]
    teardown_section = source.split("def teardown", maxsplit=1)[1]

    assert '_GRAPH_POOL_TRIM = getattr(torch.cuda, "graph_pool_trim", None)' in source
    assert '_CUDA_MATMUL_BACKEND = getattr(torch.backends.cuda, "matmul", None)' in source
    assert '_CUDNN_BACKEND = getattr(torch.backends, "cudnn", None)' in source
    assert "if _GRAPH_POOL_TRIM is not None:" in setup_section
    assert "_GRAPH_POOL_TRIM()" in setup_section
    assert 'hasattr(torch.cuda, "graph_pool_trim")' not in setup_section
    assert "if _HAS_CUDA_MATMUL_ALLOW_TF32:" in setup_section
    assert "_CUDA_MATMUL_BACKEND.allow_tf32 = True" in setup_section
    assert "if _HAS_CUDNN_ALLOW_TF32:" in setup_section
    assert "_CUDNN_BACKEND.allow_tf32 = True" in setup_section
    assert "getattr(torch.backends.cuda.matmul" not in setup_section
    assert "self.gpu_prompt: Optional[torch.Tensor] = None" in source
    assert "self.state_buffer: Optional[torch.Tensor] = None" in source
    assert "if self.gpu_prompt is None or self.state_buffer is None:" in capture_section
    assert "not hasattr(self, name) or getattr(self, name) is None" not in capture_section
    assert "if _GRAPH_POOL_TRIM is not None:" in teardown_section
    assert "_GRAPH_POOL_TRIM()" in teardown_section
    assert 'hasattr(torch.cuda, "graph_pool_trim")' not in teardown_section


def test_decode_graph_capture_avoids_unused_full_vocab_output_and_token_copy() -> None:
    source = (REPO_ROOT / "labs" / "decode_optimization" / "decode_common.py").read_text(
        encoding="utf-8"
    )
    graph_section = source.split("def _capture_decode_graph", maxsplit=1)[1].split(
        "def _prefill", maxsplit=1
    )[0]
    loop_section = source.split("def _run_decode_loop", maxsplit=1)[1].split(
        "# Core math",
        maxsplit=1,
    )[0]

    assert "self.graph_logits" not in source
    assert "self.graph_next_token" not in source
    assert "self.graph_logits = torch.empty" not in graph_section
    assert "self.graph_next_token = torch.empty" not in graph_section
    assert "self.graph_logits.copy_" not in graph_section
    assert "self.graph_next_token.copy_" not in graph_section
    assert "self.next_token_out = torch.empty_like" not in source
    assert graph_section.count("self._run_decode_loop()") == 2
    assert "tokens = self.current_tokens" in loop_section
    assert "next_state, next_token = self.decode_fn(tokens, self.state_buffer)" in loop_section
    assert "self.state_buffer.copy_(next_state)" in loop_section
    assert "tokens = next_token" in loop_section
    assert "self.current_tokens.copy_(next_token)" not in loop_section
    assert graph_section.count("self.current_tokens.copy_(next_token)") == 0
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
    assert "self.current_tokens.copy_(self.gpu_prompt_last_token)" in graph_section


def test_decode_prompt_last_token_view_is_cached_outside_hot_path() -> None:
    source = (REPO_ROOT / "labs" / "decode_optimization" / "decode_common.py").read_text(
        encoding="utf-8"
    )
    init_section = source.split("def _init_buffers", maxsplit=1)[1].split(
        "# Compiled / graphed helpers",
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

    assert "self.gpu_prompt_last_tokens: list[torch.Tensor] = []" in source
    assert "self.gpu_prompt_last_token: Optional[torch.Tensor] = None" in source
    assert "self.gpu_prompt_last_tokens.append(gpu_prompt.select(1, prompt - 1))" in init_section
    assert "self.gpu_prompt_last_token = self.gpu_prompt_last_tokens[0]" in init_section
    assert "self.gpu_prompt_last_token = self.gpu_prompt_last_tokens[1]" in prefetch_section
    assert "self.current_tokens.copy_(self.gpu_prompt_last_token)" in benchmark_section
    assert "self.gpu_prompt[:, -1]" not in benchmark_section
    assert "prompt[:, -1]" not in prefetch_section


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
    assert "self._decode_token_hidden: Optional[torch.Tensor] = None" in source
    assert "self._decode_combined: Optional[torch.Tensor] = None" in source
    assert "self._logits_buffer: Optional[torch.Tensor] = None" in source
    assert "self._lm_head_weight_t: Optional[torch.Tensor] = None" in source
    assert "self._decode_token_hidden = torch.empty_like(self.state_buffer)" in init_section
    assert "self._decode_combined = torch.empty_like(self.state_buffer)" in init_section
    assert "needs_full_vocab_logits = (" in init_section
    assert "self._lm_head_weight_t = self.lm_head.weight.t()" in init_section
    assert "self._logits_buffer = torch.empty(" in init_section
    assert "self._decode_next_token_values = torch.empty((bsz,), device=self.device, dtype=self.dtype)" in init_section
    assert "self._decode_next_token = torch.empty((bsz,), device=self.device, dtype=torch.long)" in init_section
    assert 'raise RuntimeError("Decode buffers must be initialized before _decode_step()")' in decode_step_section
    assert "or self._decode_token_hidden is None" in decode_step_section
    assert "token_hidden = torch.index_select(" in decode_step_section
    assert "self.embedding.weight," in decode_step_section
    assert "out=self._decode_token_hidden" in decode_step_section
    assert "token_hidden = self.embedding(tokens)" not in decode_step_section
    assert "torch.add(token_hidden, state, out=self._decode_combined)" in decode_step_section
    assert "hidden = self.decode_mlp(self._decode_combined)" in decode_step_section
    assert "torch.mm(hidden, self._lm_head_weight_t, out=self._logits_buffer)" in decode_step_section
    assert "logits = self._logits_buffer" in decode_step_section
    assert "logits = self.lm_head(hidden)" in decode_step_section
    assert "combined = token_hidden + state" not in decode_step_section
    assert "torch.max(logits, dim=-1, out=(self._decode_next_token_values, self._decode_next_token))" in decode_step_section
    assert "torch.argmax(logits, dim=-1)" not in decode_step_section
    assert "self._decode_combined = torch.empty_like(token_hidden)" not in decode_step_section
    assert "logits = torch.empty(" not in decode_step_section
    assert "self._decode_next_token_values = torch.empty(" not in decode_step_section
    assert "self._decode_next_token = torch.empty(" not in decode_step_section
    assert "tuple(self._decode_combined.shape)" not in decode_step_section
    assert "tuple(self._decode_next_token_values.shape)" not in decode_step_section
    assert "tuple(self._decode_next_token.shape)" not in decode_step_section
    assert "return hidden, self._decode_next_token" in decode_step_section
    assert "return logits, hidden, self._decode_next_token" not in decode_step_section


def test_decode_step_reuses_full_vocab_logits_buffer_on_cpu() -> None:
    cfg = DecodeConfig(
        batch_size=2,
        prompt_tokens=4,
        decode_tokens=1,
        hidden_size=8,
        vocab_size=16,
    )
    bench = DecodeBenchmark(cfg)
    bench.device = torch.device("cpu")
    torch.manual_seed(1234)
    bench._init_model()
    bench._init_buffers()

    assert bench._logits_buffer is not None
    assert bench._lm_head_weight_t is not None
    assert bench._decode_token_hidden is not None
    logits_ptr = bench._logits_buffer.data_ptr()
    token_hidden_ptr = bench._decode_token_hidden.data_ptr()

    with torch.inference_mode():
        tokens = torch.tensor([1, 2], dtype=torch.long)
        state = torch.randn(cfg.batch_size, cfg.hidden_size, dtype=bench.dtype)
        next_state, next_token = bench._run_decode_step_math(tokens, state)
        expected_hidden = bench.embedding(tokens)
        expected_token = torch.argmax(bench.lm_head(next_state), dim=-1)

        assert bench._logits_buffer.data_ptr() == logits_ptr
        assert bench._decode_token_hidden.data_ptr() == token_hidden_ptr
        torch.testing.assert_close(bench._decode_token_hidden, expected_hidden)
        torch.testing.assert_close(next_token, expected_token)

        next_state, next_token = bench._run_decode_step_math(tokens, next_state)
        expected_hidden = bench.embedding(tokens)
        expected_token = torch.argmax(bench.lm_head(next_state), dim=-1)

    assert bench._logits_buffer.data_ptr() == logits_ptr
    assert bench._decode_token_hidden.data_ptr() == token_hidden_ptr
    torch.testing.assert_close(bench._decode_token_hidden, expected_hidden)
    torch.testing.assert_close(next_token, expected_token)


def test_decode_loop_hands_next_token_buffer_forward_on_cpu() -> None:
    cfg = DecodeConfig(
        batch_size=2,
        prompt_tokens=4,
        decode_tokens=3,
        hidden_size=8,
        vocab_size=16,
    )
    bench = DecodeBenchmark(cfg)
    bench.device = torch.device("cpu")
    torch.manual_seed(1234)
    bench._init_model()
    bench._init_buffers()
    bench.decode_fn = bench._run_decode_step_math

    with torch.inference_mode():
        prompt = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
        initial_state = bench._run_prefill_math(prompt)
        initial_tokens = prompt[:, -1].clone()

        copy_forward_state = initial_state.clone()
        copy_forward_tokens = initial_tokens.clone()
        for _ in range(cfg.decode_tokens):
            next_state, next_token = bench._run_decode_step_math(
                copy_forward_tokens,
                copy_forward_state,
            )
            copy_forward_state.copy_(next_state)
            copy_forward_tokens.copy_(next_token)

        bench.state_buffer.copy_(initial_state)
        bench.current_tokens.copy_(initial_tokens)
        bench._run_decode_loop()

    torch.testing.assert_close(bench.state_buffer, copy_forward_state)


def test_candidate_logits_only_skips_full_vocab_logits_buffer_on_cpu() -> None:
    cfg = DecodeConfig(
        batch_size=2,
        prompt_tokens=4,
        decode_tokens=1,
        hidden_size=8,
        vocab_size=16,
        candidate_vocab_size=1,
        candidate_logits_only=True,
    )
    bench = DecodeBenchmark(cfg)
    bench.device = torch.device("cpu")
    torch.manual_seed(1234)
    bench._init_model()
    bench._init_buffers()

    assert bench._logits_buffer is None
    assert bench._lm_head_weight_t is None


def test_candidate_logits_only_reuses_cached_candidate_weight_transpose_on_cpu() -> None:
    cfg = DecodeConfig(
        batch_size=2,
        prompt_tokens=4,
        decode_tokens=1,
        hidden_size=8,
        vocab_size=16,
        candidate_vocab_size=4,
        candidate_logits_only=True,
    )
    bench = DecodeBenchmark(cfg)
    bench.device = torch.device("cpu")
    torch.manual_seed(1234)
    bench._init_model()
    bench._init_buffers()

    assert bench._candidate_lm_weight is not None
    assert bench._candidate_lm_weight_t is not None
    assert bench._candidate_token_ids is not None
    candidate_weight_t_ptr = bench._candidate_lm_weight_t.data_ptr()
    assert bench._logits_buffer is None

    with torch.inference_mode():
        tokens = torch.tensor([1, 2], dtype=torch.long)
        state = torch.randn(cfg.batch_size, cfg.hidden_size, dtype=bench.dtype)
        next_state, next_token = bench._run_decode_step_math(tokens, state)
        candidate_scores = bench.lm_head(next_state).index_select(
            1,
            bench._candidate_token_ids,
        )
        candidate_positions = torch.argmax(candidate_scores, dim=-1)
        expected_token = bench._candidate_token_ids.index_select(0, candidate_positions)

        assert bench._candidate_lm_weight_t.data_ptr() == candidate_weight_t_ptr
        torch.testing.assert_close(next_token, expected_token)

        next_state, next_token = bench._run_decode_step_math(tokens, next_state)
        candidate_scores = bench.lm_head(next_state).index_select(
            1,
            bench._candidate_token_ids,
        )
        candidate_positions = torch.argmax(candidate_scores, dim=-1)
        expected_token = bench._candidate_token_ids.index_select(0, candidate_positions)

    assert bench._candidate_lm_weight_t.data_ptr() == candidate_weight_t_ptr
    torch.testing.assert_close(next_token, expected_token)


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
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def _refresh_static_custom_metrics",
        maxsplit=1,
    )[0]
    compile_section = source.split("def _maybe_compile", maxsplit=1)[1].split(
        "def _capture_decode_graph",
        maxsplit=1,
    )[0]
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
    decode_math_section = source.split("def _run_decode_step_math", maxsplit=1)[1].split(
        "def _get_fp8_context",
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

    assert "self.prefill_fn = self._run_prefill_math" in setup_section
    assert "self.decode_fn = self._run_decode_step_math" in setup_section
    assert "torch.compile(self._run_prefill_math" in compile_section
    assert "torch.compile(self._run_decode_step_math" in compile_section
    assert "with self._get_fp8_context(), torch.inference_mode(), self.sdpa_ctx_factory():" in prefetch_section
    assert "with self._get_fp8_context(), torch.inference_mode(), self.sdpa_ctx_factory():" in benchmark_section
    assert "with torch.inference_mode(), te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):" in te_cache_section
    assert "with torch.inference_mode(), self.sdpa_ctx_factory():" in prefill_section
    assert "with torch.inference_mode(), self.sdpa_ctx_factory():" in decode_step_section
    assert "with torch.inference_mode()" not in decode_math_section
    assert "self._decode_combined is None" not in decode_math_section
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
    init_section = source.split("def __init__", maxsplit=1)[1].split(
        "def setup",
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
    assert "self._iteration_ttft_times = [0.0]" in init_section
    assert "self._iteration_tpot_times = [0.0] * self.cfg.decode_tokens" in init_section
    assert '"ttft_times_ms": self._iteration_ttft_times' in init_section
    assert '"tpot_times_ms": self._iteration_tpot_times' in init_section
    assert "def _refresh_static_custom_metrics" in source
    assert "self._refresh_static_custom_metrics()" in setup_section
    assert "iter_start, batch0_end, _, iter_end = self._timing_event_tuple" in prefetch_section
    assert "prefill_start, prefill_end, decode_start, decode_end = self._timing_event_tuple" in benchmark_section
    assert "self._pending_iteration_events = (iter_start, batch0_end, batch0_end, iter_end)" in prefetch_section
    assert "self._pending_iteration_events = (prefill_start, prefill_end, decode_start, decode_end)" in benchmark_section
    assert "prefill_start, prefill_end, decode_start, decode_end = self._pending_iteration_events" in finalize_section
    assert "metrics = self._custom_metrics" in finalize_section
    assert 'metrics["ttft_ms"] = float(ttft_ms)' in finalize_section
    assert 'metrics["decode_time_ms"] = float(decode_ms)' in finalize_section
    assert 'metrics["tpot_mean_ms"] = float(tpot_ms)' in finalize_section
    assert "self._iteration_ttft_times[0] = ttft_ms" in finalize_section
    assert "for idx in range(len(tpot_times)):" in finalize_section
    assert "tpot_times[idx] = tpot_ms" in finalize_section
    assert "return self._iteration_metric_payload" in finalize_section
    assert "self._custom_metrics = {" not in finalize_section
    assert '"ttft_times_ms": [ttft_ms]' not in finalize_section
    assert "[tpot_ms] * self.cfg.decode_tokens" not in finalize_section
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
    assert "current_stream = torch.cuda.current_stream()" in benchmark_section
    assert "prefill_stream = self.compute_stream or current_stream" in benchmark_section
    assert "current_stream.wait_stream(self.graph_stream)" in benchmark_section
    assert "current_stream.wait_stream(self.compute_stream)" in benchmark_section
    assert "torch.cuda.current_stream().wait_stream(self.graph_stream)" not in benchmark_section
    assert "torch.cuda.current_stream().wait_stream(self.compute_stream)" not in benchmark_section
    assert "if not self.cfg.reuse_device_prompt:" in benchmark_section
    assert "copy_wait_stream = (" in benchmark_section
    assert "self._copy_prompts_to_device(wait_stream=copy_wait_stream)" in benchmark_section
    assert "with torch.cuda.stream(decode_stream):" in benchmark_section
    assert "with torch.cuda.stream(self.compute_stream or torch.cuda.current_stream()):" not in benchmark_section


def test_decode_prefetch_overlaps_second_copy_only_when_async_safe() -> None:
    source = (REPO_ROOT / "labs" / "decode_optimization" / "decode_common.py").read_text(
        encoding="utf-8"
    )
    prefetch_section = source.split("def _benchmark_prefetch_batches", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]
    first_copy_idx = prefetch_section.index(
        "event0 = self._copy_prompt_to_device_idx(0, stream=copy_stream, record_event=True)"
    )
    early_second_copy_idx = prefetch_section.index(
        "event1 = (\n"
        "            self._copy_prompt_to_device_idx(1, stream=copy_stream, record_event=True)"
    )
    batch0_compute_idx = prefetch_section.index(
        "self._run_prefill_decode(\n"
        "                self.gpu_prompts[0],"
    )
    fallback_second_copy_idx = prefetch_section.rindex(
        "event1 = self._copy_prompt_to_device_idx(1, stream=copy_stream, record_event=True)"
    )

    assert "can_overlap_second_copy = bool(" in prefetch_section
    assert "self.cfg.use_pinned_host and copy_stream is not prefill_stream" in prefetch_section
    assert "current_stream = torch.cuda.current_stream()" in prefetch_section
    assert "prefill_stream = self.compute_stream or current_stream" in prefetch_section
    assert "current_stream.wait_stream(self.compute_stream)" in prefetch_section
    assert "torch.cuda.current_stream().wait_stream(self.compute_stream)" not in prefetch_section
    assert "if can_overlap_second_copy" in prefetch_section
    assert first_copy_idx < early_second_copy_idx < batch0_compute_idx
    assert batch0_compute_idx < fallback_second_copy_idx
    assert "if event1 is None:" in prefetch_section


def test_decode_device_resident_path_skips_hot_loop_staging() -> None:
    source = (REPO_ROOT / "labs" / "decode_optimization" / "decode_common.py").read_text(
        encoding="utf-8"
    )
    config_section = source.split("@dataclass", maxsplit=1)[1].split(
        "class DecodeBenchmark",
        maxsplit=1,
    )[0]
    init_section = source.split("def _init_buffers", maxsplit=1)[1].split(
        "# Compiled / graphed helpers",
        maxsplit=1,
    )[0]
    populate_section = source.split("def _populate_device_resident_inputs", maxsplit=1)[1].split(
        "# Compiled / graphed helpers",
        maxsplit=1,
    )[0]
    copy_section = source.split("def _copy_prompts_to_device", maxsplit=1)[1].split(
        "def _copy_prompt_to_device_idx",
        maxsplit=1,
    )[0]
    copy_idx_section = source.split("def _copy_prompt_to_device_idx", maxsplit=1)[1].split(
        "def _timing_event",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def finalize_iteration_metrics",
        maxsplit=1,
    )[0]

    assert "reuse_device_prompt: bool = False" in config_section
    assert "if self.cfg.reuse_device_prompt:" in init_section
    assert "self._populate_device_resident_inputs()" in init_section
    assert "gpu_prompt.copy_(self.host_prompts[idx], non_blocking=False)" in populate_section
    assert "gpu_payload.copy_(self.host_payloads[idx], non_blocking=False)" in populate_section
    assert "torch.cuda.synchronize()" in populate_section
    assert 'metrics["reuse_device_prompt"] = float(self.cfg.reuse_device_prompt)' in source
    assert 'metrics["prompt_copies_per_iteration"] = prompt_copy_count' in source
    assert "if self.cfg.reuse_device_prompt:" in copy_section
    assert "return" in copy_section.split("if self.cfg.reuse_device_prompt:", maxsplit=1)[1]
    assert "if self.cfg.reuse_device_prompt:" in copy_idx_section
    assert "event.record(active_stream)" in copy_idx_section
    assert "if not self.cfg.reuse_device_prompt:" in benchmark_section
    guarded_copy_section = benchmark_section.split(
        "if not self.cfg.reuse_device_prompt:",
        maxsplit=1,
    )[1].split("if nvtx:", maxsplit=1)[0]
    assert "self._copy_prompts_to_device(wait_stream=copy_wait_stream)" in guarded_copy_section


def test_decode_device_resident_pair_changes_only_residency_policy() -> None:
    baseline = get_baseline_decode_device_resident()
    optimized = get_optimized_decode_device_resident()

    assert baseline.cfg.batch_size == optimized.cfg.batch_size == 64
    assert baseline.cfg.prompt_tokens == optimized.cfg.prompt_tokens == 2048
    assert baseline.cfg.decode_tokens == optimized.cfg.decode_tokens == 16
    assert baseline.cfg.prefetch_batches == optimized.cfg.prefetch_batches == 1
    assert baseline.cfg.host_payload_mb == optimized.cfg.host_payload_mb == 512
    assert baseline.cfg.hidden_size == optimized.cfg.hidden_size == 256
    assert baseline.cfg.use_pinned_host is optimized.cfg.use_pinned_host is True
    assert baseline.cfg.use_copy_stream is optimized.cfg.use_copy_stream is False
    assert baseline.cfg.use_compute_stream is optimized.cfg.use_compute_stream is False
    assert baseline.cfg.reuse_device_prompt is False
    assert optimized.cfg.reuse_device_prompt is True


def test_decode_prefix_state_cache_skips_static_prefill_compute() -> None:
    source = (REPO_ROOT / "labs" / "decode_optimization" / "decode_common.py").read_text(
        encoding="utf-8"
    )
    config_section = source.split("@dataclass", maxsplit=1)[1].split(
        "class DecodeBenchmark",
        maxsplit=1,
    )[0]
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def _refresh_static_custom_metrics",
        maxsplit=1,
    )[0]
    cache_section = source.split("def _populate_resident_prefill_states", maxsplit=1)[1].split(
        "# Compiled / graphed helpers",
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

    assert "reuse_prefill_state: bool = False" in config_section
    assert "if self.cfg.reuse_prefill_state and not self.cfg.reuse_device_prompt:" in source
    assert "self._resident_prefill_states: list[torch.Tensor] = []" in source
    assert "self._resident_prefill_state: Optional[torch.Tensor] = None" in source
    assert "self._populate_resident_prefill_states()" in setup_section
    assert "cached_state = torch.empty_like(self.state_buffer)" in cache_section
    assert "cached_state.copy_(self.prefill_fn(gpu_prompt))" in cache_section
    assert "self._resident_prefill_state = self._resident_prefill_states[0]" in cache_section
    assert 'metrics["reuse_prefill_state"] = float(self.cfg.reuse_prefill_state)' in source
    assert 'metrics["prefill_computes_per_iteration"] = (' in source
    assert "prefill_state = self._resident_prefill_state" in benchmark_section
    assert "if prefill_state is None:\n                    prefill_state = self.prefill_fn(self.gpu_prompt)" in benchmark_section
    assert "self._resident_prefill_states[0] if self.cfg.reuse_prefill_state else None" in prefetch_section
    assert "self._resident_prefill_states[1] if self.cfg.reuse_prefill_state else None" in prefetch_section


def test_decode_prefix_state_cache_pair_changes_only_prefill_reuse_policy() -> None:
    baseline = get_baseline_decode_prefix_state_cache()
    optimized = get_optimized_decode_prefix_state_cache()

    assert baseline.cfg.batch_size == optimized.cfg.batch_size == 64
    assert baseline.cfg.prompt_tokens == optimized.cfg.prompt_tokens == 2048
    assert baseline.cfg.decode_tokens == optimized.cfg.decode_tokens == 1
    assert baseline.cfg.prefetch_batches == optimized.cfg.prefetch_batches == 1
    assert baseline.cfg.host_payload_mb == optimized.cfg.host_payload_mb == 0
    assert baseline.cfg.hidden_size == optimized.cfg.hidden_size == 256
    assert baseline.cfg.use_pinned_host is optimized.cfg.use_pinned_host is True
    assert baseline.cfg.use_copy_stream is optimized.cfg.use_copy_stream is False
    assert baseline.cfg.use_compute_stream is optimized.cfg.use_compute_stream is False
    assert baseline.cfg.reuse_device_prompt is optimized.cfg.reuse_device_prompt is True
    assert baseline.cfg.reuse_prefill_state is False
    assert optimized.cfg.reuse_prefill_state is True


def test_decode_candidate_logits_path_avoids_full_vocab_projection_when_enabled() -> None:
    source = (REPO_ROOT / "labs" / "decode_optimization" / "decode_common.py").read_text(
        encoding="utf-8"
    )
    config_section = source.split("@dataclass", maxsplit=1)[1].split(
        "class DecodeBenchmark",
        maxsplit=1,
    )[0]
    init_section = source.split("def _init_buffers", maxsplit=1)[1].split(
        "def _populate_device_resident_inputs",
        maxsplit=1,
    )[0]
    decode_math_section = source.split("def _run_decode_step_math", maxsplit=1)[1].split(
        "def _get_fp8_context",
        maxsplit=1,
    )[0]

    assert "candidate_vocab_size: int = 0" in config_section
    assert "candidate_logits_only: bool = False" in config_section
    assert "self._candidate_token_ids: Optional[torch.Tensor] = None" in source
    assert "self._candidate_lm_weight: Optional[torch.Tensor] = None" in source
    assert "self._candidate_lm_weight_t: Optional[torch.Tensor] = None" in source
    assert "self._candidate_scores: Optional[torch.Tensor] = None" in source
    assert "self._forced_candidate_tokens: Optional[torch.Tensor] = None" in source
    assert "torch.arange(" in init_section
    assert "if self.cfg.candidate_logits_only and candidate_count == 1:" in init_section
    assert "self._forced_candidate_tokens = torch.zeros(" in init_section
    assert "self.lm_head.weight.index_select(0, self._candidate_token_ids).contiguous()" in init_section
    assert "self._candidate_lm_weight_t = self._candidate_lm_weight.t()" in init_section
    assert "needs_full_vocab_logits = (" in init_section
    assert "self._candidate_token_ids is None or not self.cfg.candidate_logits_only" in init_section
    assert 'metrics["candidate_vocab_size"] = float(self.cfg.candidate_vocab_size)' in source
    assert 'metrics["candidate_logits_only"] = float(self.cfg.candidate_logits_only)' in source
    assert 'metrics["effective_logits_vocab_size"] = float(' in source
    assert "if self._candidate_token_ids is not None:" in decode_math_section
    assert "if self._forced_candidate_tokens is not None:" in decode_math_section
    assert "return hidden, self._forced_candidate_tokens" in decode_math_section
    assert "torch.mm(hidden, self._candidate_lm_weight_t, out=self._candidate_scores)" in decode_math_section
    assert "self._candidate_lm_weight.t()" not in decode_math_section
    assert "logits = self.lm_head(hidden)" in decode_math_section
    assert "torch.index_select(\n                    logits," in decode_math_section
    assert "torch.index_select(\n                self._candidate_token_ids," in decode_math_section


def test_decode_candidate_logits_pair_changes_only_projection_policy() -> None:
    baseline = get_baseline_decode_candidate_logits()
    optimized = get_optimized_decode_candidate_logits()

    assert baseline.cfg.batch_size == optimized.cfg.batch_size == 32
    assert baseline.cfg.prompt_tokens == optimized.cfg.prompt_tokens == 128
    assert baseline.cfg.decode_tokens == optimized.cfg.decode_tokens == 64
    assert baseline.cfg.prefetch_batches == optimized.cfg.prefetch_batches == 1
    assert baseline.cfg.host_payload_mb == optimized.cfg.host_payload_mb == 0
    assert baseline.cfg.hidden_size == optimized.cfg.hidden_size == 512
    assert baseline.cfg.vocab_size == optimized.cfg.vocab_size == 131072
    assert baseline.cfg.candidate_vocab_size == optimized.cfg.candidate_vocab_size == 1
    assert baseline.cfg.reuse_device_prompt is optimized.cfg.reuse_device_prompt is False
    assert baseline.cfg.candidate_logits_only is False
    assert optimized.cfg.candidate_logits_only is True


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
        metrics_payload = bench.finalize_iteration_metrics()
        custom_metrics = bench.get_custom_metrics()
        assert metrics_payload is not None
        ttft_times = metrics_payload["ttft_times_ms"]
        tpot_times = metrics_payload["tpot_times_ms"]

        bench.benchmark_fn()
        next_metrics_payload = bench.finalize_iteration_metrics()

        assert set(event_ids) == {"prefill_start", "prefill_end", "decode_start", "decode_end"}
        assert {name: id(event) for name, event in bench._timing_events.items()} == event_ids
        assert bench._timing_event_tuple is not None
        assert tuple(id(event) for event in bench._timing_event_tuple) == tuple_ids
        assert next_metrics_payload is metrics_payload
        assert next_metrics_payload is not None
        assert next_metrics_payload["ttft_times_ms"] is ttft_times
        assert next_metrics_payload["tpot_times_ms"] is tpot_times
        assert bench.get_custom_metrics() is custom_metrics
    finally:
        bench.teardown()
