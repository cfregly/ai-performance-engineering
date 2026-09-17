"""Control-plane contracts for the opt-in two-wave routing workload."""

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
    def __init__(
        self,
        *,
        model: str,
        tensor_parallel_size: int,
        trust_remote_code: bool,
        gpu_memory_utilization: float,
        enable_prefix_caching: bool,
        enforce_eager: bool,
        attention_backend: str | None = None,
    ) -> None:
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_prefix_caching = enable_prefix_caching
        self.enforce_eager = enforce_eager
        self.attention_backend = attention_backend

    def create_engine_config(self) -> SimpleNamespace:
        return SimpleNamespace(
            device_config=SimpleNamespace(device=None, device_type="cuda")
        )


class _WaveEngine:
    def __init__(self, config: SimpleNamespace, *, long_steps: int) -> None:
        self.config = config
        self.long_steps = long_steps
        self.pending: dict[
            str, tuple[list[int], _FakeSamplingParams, int]
        ] = {}
        self.add_calls: list[dict[str, object]] = []
        self.step_calls = 0

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
        steps = self.long_steps if len(prompt) == 4 else 1
        self.pending[request_id] = (prompt, params, steps)

    def step(self) -> list[SimpleNamespace]:
        self.step_calls += 1
        outputs: list[SimpleNamespace] = []
        for request_id, (prompt, params, steps) in list(self.pending.items()):
            steps -= 1
            if steps > 0:
                self.pending[request_id] = (prompt, params, steps)
                continue
            outputs.append(
                SimpleNamespace(
                    request_id=request_id,
                    outputs=[
                        SimpleNamespace(
                            token_ids=[
                                sum(prompt) + index
                                for index in range(params.max_tokens)
                            ]
                        )
                    ],
                    finished=True,
                )
            )
            del self.pending[request_id]
        return outputs

    def get_num_unfinished_requests(self) -> int:
        return len(self.pending)

    def abort_request(self, request_ids: list[str]) -> None:
        for request_id in request_ids:
            self.pending.pop(request_id, None)


class _WaveLLMEngine:
    created: list[_WaveEngine] = []
    long_steps = 3

    @classmethod
    def from_vllm_config(cls, config: SimpleNamespace) -> _WaveEngine:
        engine = _WaveEngine(config, long_steps=cls.long_steps)
        cls.created.append(engine)
        return engine


@pytest.fixture
def wave_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    _WaveLLMEngine.created.clear()
    _WaveLLMEngine.long_steps = 3
    monkeypatch.setattr(vllm_runner, "EngineArgs", _FakeEngineArgs)
    monkeypatch.setattr(vllm_runner, "LLMEngine", _WaveLLMEngine)
    monkeypatch.setattr(vllm_runner, "SamplingParams", _FakeSamplingParams)
    monkeypatch.setattr(
        vllm_runner,
        "RequestOutputKind",
        SimpleNamespace(CUMULATIVE="cumulative"),
    )
    monkeypatch.setattr(vllm_runner, "_assert_vllm_runtime_ready", lambda: None)
    monkeypatch.setattr(vllm_runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vllm_runner.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        vllm_runner.time,
        "sleep",
        lambda _seconds: pytest.fail("two-wave control plane must not sleep"),
    )


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        model="/models/local-test-model",
        attention_backend=None,
        prefill_gpus="0",
        decode_gpus="0,1",
        req_count=6,
        max_tokens=2,
        long_prompt_tokens=4,
        short_prompt_tokens=2,
        prefill_burst=2,
        decode_requests=1,
        continue_requests=3,
        prefill_ctx_thresh=3,
        long_spillover_limit=0,
        routing_arrival_profile=(
            vllm_runner.ROUTING_ARRIVAL_TWO_WAVE_IMBALANCE
        ),
        use_v1_core_loop=False,
    )


def _topology() -> TopologySnapshot:
    return TopologySnapshot(
        gpu_numa={0: 0, 1: 1},
        distance={0: [10, 20], 1: [20, 10]},
        timestamp=1.0,
        gpu_numa_status="complete",
    )


def _run(mode: str, args: SimpleNamespace) -> dict[str, object]:
    lengths = vllm_runner.routing_prompt_lengths(args)
    return vllm_runner.run_vllm_routing_with_topology(
        mode,
        topology_snapshot=_topology(),
        cli_args=args,
        prompt_token_ids=vllm_runner.build_prompt_token_ids(lengths),
    )


def test_two_wave_cold_pair_keeps_background_fixed_and_full_outputs(
    wave_vllm: None,
) -> None:
    args = _args()
    baseline = _run("baseline", args)
    optimized = _run("optimized", args)

    for summary in (baseline, optimized):
        assert summary["routing_arrival_profile"] == "two-wave-imbalance"
        assert summary["requests"] == summary["completed"] == 6
        assert summary["arrival_gate_steps"] == 1
        assert summary["arrival_gate_loaded_queue_depth"] == 2
        assert summary["arrival_gate_idle_queue_depth"] == 0
        assert summary["fixed_background_placement"] == 1.0
        assert summary["requests_admitted_background_long_gpu0"] == 2
        assert summary["requests_admitted_background_long_gpu1"] == 0
        assert summary["requests_admitted_background_short_gpu0"] == 0
        assert summary["requests_admitted_background_short_gpu1"] == 1
        assert "ttft_ms_p95_background_long" in summary
        assert "ttft_ms_p95_background_short" in summary
        assert "ttft_ms_p95_foreground_short" in summary
        assert len(summary[vllm_runner.VERIFICATION_OUTPUT_KEY]) == 18
        assert summary["lifecycle.engine_reuse_count"] == 0.0

    assert baseline["requests_admitted_foreground_short_gpu0"] == 2
    assert baseline["requests_admitted_foreground_short_gpu1"] == 1
    assert baseline["foreground_first_to_idle_gpu"] == 0.0
    assert optimized["foreground_first_to_idle_gpu"] == 1.0
    assert (
        baseline[vllm_runner.VERIFICATION_OUTPUT_KEY]
        == optimized[vllm_runner.VERIFICATION_OUTPUT_KEY]
    )

    for run_engines in (_WaveLLMEngine.created[:2], _WaveLLMEngine.created[2:]):
        calls_by_device = {
            str(engine.config.device_config.device): engine.add_calls
            for engine in run_engines
        }
        assert [
            len(call["prompt"]) for call in calls_by_device["cuda:0"][:2]
        ] == [4, 4]
        assert len(calls_by_device["cuda:1"][0]["prompt"]) == 2


def test_two_wave_reused_session_resets_request_state(
    wave_vllm: None,
) -> None:
    args = _args()
    session = vllm_runner.create_vllm_routing_session(
        "optimized",
        topology_snapshot=_topology(),
        cli_args=args,
        warmup_runs=1,
    )
    try:
        prompt = vllm_runner.build_prompt_token_ids(
            vllm_runner.routing_prompt_lengths(args)
        )
        first = vllm_runner.run_vllm_routing_with_topology(
            "optimized",
            topology_snapshot=_topology(),
            cli_args=args,
            req_count=args.req_count,
            max_tokens=args.max_tokens,
            prompt_token_ids=prompt,
            engine_session=session,
        )
        second = vllm_runner.run_vllm_routing_with_topology(
            "optimized",
            topology_snapshot=_topology(),
            cli_args=args,
            prompt_token_ids=prompt,
            engine_session=session,
        )

        assert first["lifecycle.warmup_runs"] == 1.0
        assert first["lifecycle.steady_state_runs"] == 0.0
        assert second["lifecycle.warmup_runs"] == 1.0
        assert second["lifecycle.steady_state_runs"] == 1.0
        assert second["lifecycle.engine_reuse_count"] == 1.0
        assert first[vllm_runner.VERIFICATION_OUTPUT_KEY] == second[
            vllm_runner.VERIFICATION_OUTPUT_KEY
        ]
        completed_after_second: set[str] = set()
        for engine in session.engines.values():
            assert engine.queue_depth() == 0
            assert not engine._inflight
            completed_after_second.update(engine._completed_output_token_ids)
        assert completed_after_second == {
            f"session-0001-req-{index}" for index in range(args.req_count)
        }
    finally:
        session.close()


def test_two_wave_validates_cli_mix_and_runtime_aliases(
    wave_vllm: None,
) -> None:
    args = _args()
    assert vllm_runner.routing_prompt_lengths(args) == [4, 4, 2, 2, 2, 2]
    with pytest.raises(ValueError, match="req_count override must match"):
        vllm_runner.routing_prompt_lengths(args, req_count=7)

    valid_argv = [
        "--routing-arrival-profile",
        "two-wave-imbalance",
        "--req-count",
        "6",
        "--prefill-burst",
        "2",
        "--decode-requests",
        "1",
        "--continue-requests",
        "3",
    ]
    parsed = vllm_runner.parse_vllm_target_overrides(valid_argv)
    assert vllm_runner.routing_prompt_lengths(parsed) == [
        4096,
        4096,
        128,
        128,
        128,
        128,
    ]
    invalid_argv = list(valid_argv)
    invalid_argv[3] = "7"
    with pytest.raises(ValueError, match="--req-count must equal"):
        vllm_runner.parse_vllm_target_overrides(invalid_argv)

    session = vllm_runner.create_vllm_routing_session(
        "baseline", topology_snapshot=_topology(), cli_args=args
    )
    try:
        prompt = vllm_runner.build_prompt_token_ids(
            vllm_runner.routing_prompt_lengths(args)
        )
        with pytest.raises(ValueError, match="max_tokens override must match"):
            vllm_runner.run_vllm_routing_with_topology(
                "baseline",
                topology_snapshot=_topology(),
                cli_args=args,
                max_tokens=args.max_tokens + 1,
                prompt_token_ids=prompt,
                engine_session=session,
            )
        assert session.lifecycle_metrics()["lifecycle.failed_runs"] == 0.0
    finally:
        session.close()


def test_two_wave_fails_when_background_never_exposes_required_imbalance(
    wave_vllm: None,
) -> None:
    _WaveLLMEngine.long_steps = 1
    with pytest.raises(
        RuntimeError,
        match="prerequisite was not observed: both background queues drained",
    ):
        _run("optimized", _args())

    assert len(_WaveLLMEngine.created) == 2
    assert all(engine.get_num_unfinished_requests() == 0 for engine in _WaveLLMEngine.created)
