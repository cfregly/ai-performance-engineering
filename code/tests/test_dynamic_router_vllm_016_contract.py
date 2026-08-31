"""Focused vLLM 0.16 API contracts for all dynamic-router benchmark modes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from labs.dynamic_router import vllm_runner
from labs.dynamic_router.topology import TopologySnapshot


class _FakeSamplingParams:
    def __init__(
        self,
        *,
        temperature: float,
        max_tokens: int,
        ignore_eos: bool,
        output_kind: object,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.output_kind = output_kind


class _FakeEngineArgs:
    """The vLLM 0.16 EngineArgs surface used by this lab (no ``device`` kwarg)."""

    created: list[_FakeEngineArgs] = []

    def __init__(
        self,
        *,
        model: str,
        tensor_parallel_size: int,
        trust_remote_code: bool,
        gpu_memory_utilization: float,
        enable_prefix_caching: bool,
        enforce_eager: bool,
    ) -> None:
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_prefix_caching = enable_prefix_caching
        self.enforce_eager = enforce_eager
        self.created.append(self)

    def create_engine_config(self) -> SimpleNamespace:
        return SimpleNamespace(device_config=SimpleNamespace(device=None, device_type="cuda"))


class _FakeEngine:
    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config
        self.pending: dict[str, tuple[list[int], _FakeSamplingParams]] = {}
        self.add_calls: list[dict[str, object]] = []

    def add_request(
        self,
        *,
        request_id: str,
        prompt: list[int],
        params: _FakeSamplingParams,
        arrival_time: float,
    ) -> None:
        self.add_calls.append(
            {
                "request_id": request_id,
                "prompt": prompt,
                "params": params,
                "arrival_time": arrival_time,
            }
        )
        self.pending[request_id] = (prompt, params)

    def step(self) -> list[SimpleNamespace]:
        outputs = [
            SimpleNamespace(
                request_id=request_id,
                outputs=[SimpleNamespace(token_ids=list(range(params.max_tokens)))],
                finished=True,
            )
            for request_id, (_, params) in self.pending.items()
        ]
        self.pending.clear()
        return outputs

    def get_num_unfinished_requests(self) -> int:
        return len(self.pending)


class _FakeLLMEngine:
    created: list[_FakeEngine] = []

    @classmethod
    def from_vllm_config(cls, config: SimpleNamespace) -> _FakeEngine:
        engine = _FakeEngine(config)
        cls.created.append(engine)
        return engine


@pytest.fixture
def vllm_016_api(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeEngineArgs.created.clear()
    _FakeLLMEngine.created.clear()
    monkeypatch.setattr(vllm_runner, "EngineArgs", _FakeEngineArgs)
    monkeypatch.setattr(vllm_runner, "LLMEngine", _FakeLLMEngine)
    monkeypatch.setattr(vllm_runner, "SamplingParams", _FakeSamplingParams)
    monkeypatch.setattr(
        vllm_runner,
        "RequestOutputKind",
        SimpleNamespace(CUMULATIVE="cumulative"),
    )
    monkeypatch.setattr(vllm_runner, "_assert_vllm_runtime_ready", lambda: None)
    monkeypatch.setattr(vllm_runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vllm_runner.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(vllm_runner.torch.cuda, "synchronize", lambda _index: None)
    monkeypatch.setattr(
        vllm_runner.torch.cuda,
        "mem_get_info",
        lambda _index: (8 * 1024**3, 16 * 1024**3),
    )
    monkeypatch.setattr(vllm_runner.time, "sleep", lambda _seconds: None)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        model="/models/local-test-model",
        prefill_gpus="0",
        decode_gpus="0,1",
        req_count=2,
        max_tokens=1,
        long_prompt_tokens=4,
        short_prompt_tokens=2,
        prefill_burst=1,
        decode_requests=1,
        continue_requests=1,
        prefill_ctx_thresh=3,
        use_v1_core_loop=False,
    )


def _topology() -> TopologySnapshot:
    return TopologySnapshot(
        gpu_numa={0: 0, 1: 1},
        distance={0: [10, 20], 1: [20, 10]},
        timestamp=1.0,
        gpu_numa_status="complete",
    )


def test_wrapper_uses_vllm_016_engine_and_request_signatures(vllm_016_api: None) -> None:
    wrapper = vllm_runner._VllmWrapper("gpu1", 1, "/models/local-test-model")
    request = vllm_runner.Request(req_id="req-0", prompt_tokens=4, expected_new_tokens=2)
    runtime = vllm_runner._RequestRuntime(request, "gpu1", admitted_at=10.0)

    wrapper.add_request(runtime)

    assert _FakeEngineArgs.created[0].model == "/models/local-test-model"
    assert _FakeEngineArgs.created[0].enable_prefix_caching is False
    assert str(_FakeLLMEngine.created[0].config.device_config.device) == "cuda:1"
    assert _FakeLLMEngine.created[0].add_calls == [
        {
            "request_id": "req-0",
            "prompt": [1, 1, 1, 1],
            "params": _FakeLLMEngine.created[0].pending["req-0"][1],
            "arrival_time": 10.0,
        }
    ]
    assert _FakeLLMEngine.created[0].pending["req-0"][1].ignore_eos is True


def test_pinned_api_mismatch_fails_explicitly_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vllm_runner, "EngineArgs", lambda **_kwargs: SimpleNamespace())

    with pytest.raises(RuntimeError, match="SKIPPED: Pinned vLLM API mismatch"):
        vllm_runner._build_vllm_engine(_FakeLLMEngine, "/models/local-test-model", 0)


@pytest.mark.parametrize(
    ("call_path", "mode"),
    [
        ("dynamic", "baseline"),
        ("dynamic", "optimized"),
        ("dual_pool", "shared"),
        ("dual_pool", "dual"),
    ],
)
def test_all_four_benchmark_call_paths_use_current_vllm_api(
    vllm_016_api: None,
    call_path: str,
    mode: str,
) -> None:
    args = _args()
    if call_path == "dynamic":
        summary = vllm_runner.run_vllm_routing_with_topology(
            mode,
            topology_snapshot=_topology(),
            req_count=2,
            max_tokens=1,
            cli_args=args,
        )
        expected_requests = 2
    else:
        if mode == "dual":
            args.decode_gpus = "1"
        summary = vllm_runner.run_dual_pool_vllm_with_topology(
            mode,
            topology_snapshot=_topology(),
            long_prompt_tokens=4,
            short_prompt_tokens=2,
            prefill_burst=1,
            decode_requests=1,
            continue_requests=1,
            max_tokens=1,
            prefill_ctx_thresh=3,
            cli_args=args,
        )
        expected_requests = 3

    assert summary["mode"] == mode
    assert summary["requests"] == expected_requests
    assert summary["completed"] == expected_requests
    assert {str(engine.config.device_config.device) for engine in _FakeLLMEngine.created} == {
        "cuda:0",
        "cuda:1",
    }
    assert sum(len(engine.add_calls) for engine in _FakeLLMEngine.created) == expected_requests


def test_v1_direct_loop_completes_required_engine_post_step() -> None:
    post_steps: list[bool] = []
    wrapper = vllm_runner._VllmV1Wrapper.__new__(vllm_runner._VllmV1Wrapper)
    wrapper._core = SimpleNamespace(
        step_fn=lambda: ({}, False),
        post_step=lambda *, model_executed: post_steps.append(model_executed),
    )
    wrapper._inflight = {}

    assert wrapper.step() == ([], [], 0)
    assert post_steps == [False]


def test_finished_request_cannot_verify_without_declared_model_output() -> None:
    wrapper = vllm_runner._VllmWrapper.__new__(vllm_runner._VllmWrapper)
    request = vllm_runner.Request(
        req_id="req-empty",
        prompt_tokens=4,
        expected_new_tokens=2,
    )
    wrapper._inflight = {
        request.req_id: vllm_runner._RequestRuntime(
            request,
            "gpu0",
            admitted_at=10.0,
        )
    }
    wrapper._completed_output_token_ids = {}
    output = SimpleNamespace(
        request_id=request.req_id,
        outputs=[SimpleNamespace(token_ids=[])],
        finished=True,
    )

    with pytest.raises(RuntimeError, match="completed with 0 output tokens; expected 2"):
        wrapper._consume_request_outputs([output], observed_at=11.0)

    assert wrapper._completed_output_token_ids == {}
