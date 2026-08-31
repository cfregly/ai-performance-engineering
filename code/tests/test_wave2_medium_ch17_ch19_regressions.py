"""Focused CPU/static regressions for Wave 2 findings W2-066 through W2-073."""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ch17.optimized_pipeline_parallelism import OptimizedPipelineParallelismBenchmark
from ch17.prefill_decode_disagg_monolithic_common import SimpleLLM
from ch18.eos_early_exit_common import EosEarlyExitBenchmark, EosEarlyExitConfig
from ch18.run_vllm_decoder import GraphMode, VLLMMoEInferenceBenchmark
from ch18.v1_bucketed_decode_loop import build_vllm_steps

CODE_ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (CODE_ROOT / relative).read_text(encoding="utf-8")


class _FakeCudaEvent:
    def __init__(self, elapsed: float = 0.0) -> None:
        self.elapsed = elapsed
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def elapsed_time(self, end: _FakeCudaEvent) -> float:
        return end.elapsed


def test_w2_066_pipeline_streams_are_bound_to_each_stage_device() -> None:
    source = _source("ch17/optimized_pipeline_parallelism.py")
    setup = source.split("def setup", maxsplit=1)[1].split("def benchmark_fn", maxsplit=1)[0]

    assert "self._stage_devices = [next(stage.parameters()).device" in setup
    assert "torch.cuda.Stream(device=stage_device, priority=-1)" in setup
    assert "for stage_device in self._stage_devices:" in setup
    assert "with torch.cuda.device(stage_device):" in setup
    assert "torch.cuda.Stream(priority=-1)" not in setup


def test_w2_067_stage_metrics_finalize_cuda_event_durations() -> None:
    first_start = _FakeCudaEvent()
    first_end = _FakeCudaEvent(1.25)
    second_start = _FakeCudaEvent()
    second_end = _FakeCudaEvent(2.75)
    third_start = _FakeCudaEvent()
    third_end = _FakeCudaEvent(4.5)
    bench = object.__new__(OptimizedPipelineParallelismBenchmark)
    bench._pending_stage_timings = True
    bench._stage_timing_events = [
        [(first_start, first_end), (second_start, second_end)],
        [(third_start, third_end)],
    ]
    bench._last_stage_durations_ms = [0.0, 0.0]

    assert bench.finalize_iteration_metrics() is None

    assert bench._last_stage_durations_ms == [4.0, 4.5]
    assert first_end.synchronize_calls == 1
    assert second_end.synchronize_calls == 1
    assert third_end.synchronize_calls == 1
    assert bench._pending_stage_timings is False


def test_pipeline_verification_rejects_partial_or_misshaped_microbatches() -> None:
    source = _source("ch17/optimized_pipeline_parallelism.py")
    benchmark = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def finalize_iteration_metrics",
        maxsplit=1,
    )[0]
    capture = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def get_custom_streams",
        maxsplit=1,
    )[0]

    assert "if final_count != self.micro_batches:" in benchmark
    assert "self._last_final_output_count != self.micro_batches" in capture
    assert "tuple(out.shape) != expected_shape" in capture
    assert "if offset != output_buffer.shape[0]:" in capture
    assert "output_buffer[:offset]" not in capture


def test_w2_068_disaggregated_streams_overlap_independent_requests() -> None:
    source = _source("ch17/optimized_prefill_decode_disagg.py")
    benchmark = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def finalize_iteration_metrics",
        maxsplit=1,
    )[0]

    assert "next_decode_state = self.model.prefill(self.prompt)" in benchmark
    assert "token_output = self._decode_state" in benchmark
    assert "self.decode_stream.wait_event(self._decode_ready)" in benchmark
    assert "self.decode_stream.wait_event(self._prefill_done)" not in benchmark
    assert "self._decode_state = next_decode_state" in benchmark
    assert "self._decode_ready, self._prefill_done = self._prefill_done, self._decode_ready" in benchmark


def test_w2_069_prefill_is_repeatable_and_depends_on_the_full_prompt() -> None:
    model = SimpleLLM(hidden_dim=4, num_layers=1).to(dtype=torch.bfloat16).eval()
    with torch.no_grad():
        model.layers[0].weight.copy_(torch.eye(4, dtype=torch.bfloat16))
        model.layers[0].bias.zero_()

    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    changed_prefix = torch.tensor([[9, 2]], dtype=torch.long)
    with torch.inference_mode():
        first = model.prefill(prompt).clone()
        first_buffer_ptr = model._prefill_input_buffer.data_ptr()
        second = model.prefill(prompt).clone()
        changed = model.prefill(changed_prefix).clone()

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert not torch.equal(first, changed)
    assert model._prefill_input_buffer.data_ptr() == first_buffer_ptr


def test_w2_070_eos_exit_uses_observed_completion_before_forced_limit() -> None:
    cfg = EosEarlyExitConfig(
        batch_size=2,
        prompt_tokens=1,
        decode_tokens=5,
        force_eos_after_tokens=5,
        hidden_size=2,
        vocab_size=8,
        stop_on_all_done=True,
    )
    bench = EosEarlyExitBenchmark(cfg)
    bench.generated_tokens = torch.empty(2, cfg.decode_tokens, dtype=torch.long)
    bench.done_mask_buffer = torch.empty(2, dtype=torch.bool)
    bench.eos_compare_buffer = torch.empty(2, dtype=torch.bool)
    bench._prefill = lambda: None  # type: ignore[method-assign]
    decode_tokens = iter(
        (
            torch.tensor([3, 4], dtype=torch.long),
            torch.tensor([cfg.eos_token_id, cfg.eos_token_id], dtype=torch.long),
            torch.tensor([5, 5], dtype=torch.long),
        )
    )
    bench._decode_step = lambda: next(decode_tokens)  # type: ignore[method-assign]

    bench.benchmark_fn()

    assert bench._decoded_steps == 2
    assert bench._completion_checks == 2
    assert torch.equal(
        bench.output[:, 2:],
        torch.full((2, 3), cfg.eos_token_id, dtype=torch.long),
    )


def test_w2_071_requested_graph_mode_fails_explicitly_before_capture(monkeypatch) -> None:
    monkeypatch.delenv("OPT_MOE_ENABLE_GRAPHS", raising=False)
    monkeypatch.setenv("OPT_MOE_GRAPH_MODE", "piecewise")
    bench = VLLMMoEInferenceBenchmark()
    source = inspect.getsource(VLLMMoEInferenceBenchmark.setup)

    assert bench.graph_mode is GraphMode.PIECEWISE
    assert bench.enable_graphs is True
    assert "self.graph_mode = GraphMode.EAGER" not in source
    capture_calls: list[str] = []
    bench._cuda_available = True
    bench._capture_full_graph = lambda: capture_calls.append("full")  # type: ignore[method-assign]
    bench._capture_piecewise_graphs = lambda: capture_calls.append("piecewise")  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="data-dependent host reads"):
        bench._prepare_graphs()
    assert capture_calls == []

    with pytest.raises(RuntimeError, match="data-dependent host reads"):
        bench._require_graph_capture_support()

    eager = object.__new__(VLLMMoEInferenceBenchmark)
    eager.graph_mode = GraphMode.EAGER
    assert eager._require_graph_capture_support() is None

    prepare = inspect.getsource(VLLMMoEInferenceBenchmark._prepare_graphs)
    rejection = prepare.index("self._require_graph_capture_support()")
    full_capture = prepare.index("self._capture_full_graph()")
    piecewise_capture = prepare.index("self._capture_piecewise_graphs()")
    assert rejection < full_capture
    assert rejection < piecewise_capture


def test_w2_072_bucketed_loop_uses_vllm_016_params_keyword(monkeypatch) -> None:
    class FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeEngine:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self._unfinished = True

        def add_request(self, *, request_id: str, prompt: str, params: object) -> None:
            self.calls.append(
                {"request_id": request_id, "prompt": prompt, "params": params}
            )

        def has_unfinished_requests(self) -> bool:
            return self._unfinished

        def step(self) -> list[object]:
            self._unfinished = False
            return [SimpleNamespace(finished=True)]

    class FakeLLMEngine:
        engine = FakeEngine()

        @classmethod
        def from_engine_args(cls, _args: object) -> FakeEngine:
            return cls.engine

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.EngineArgs = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_vllm.LLMEngine = FakeLLMEngine
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    args = SimpleNamespace(model="local-model", max_tokens=4)

    steps = list(build_vllm_steps(args))

    assert len(steps) == 1
    assert [call["request_id"] for call in FakeLLMEngine.engine.calls] == [
        "req-0",
        "req-1",
    ]
    main_source = inspect.getsource(sys.modules[build_vllm_steps.__module__].main)
    assert "falling back to mock engine" not in main_source


def test_w2_073_d2h_copies_complete_before_cpu_dequantization() -> None:
    source = _source("ch19/baseline_dynamic_quantized_cache.py")
    finalize = source.split("def _finalize_quantized_output", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]

    scale_copy = finalize.index("scale_cpu.copy_(self._last_scale")
    packed_copy = finalize.index("packed_view.copy_(self._last_packed_dst")
    synchronize = finalize.index("torch.cuda.current_stream(self.device).synchronize()")
    cpu_dequant = finalize.index("dequantized.copy_(packed_cpu.view(torch.int8))")
    assert scale_copy < synchronize
    assert packed_copy < synchronize < cpu_dequant
