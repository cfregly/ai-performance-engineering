"""Focused vLLM 0.16 API contracts for all dynamic-router benchmark modes."""

from __future__ import annotations

import importlib
import os
from functools import partial
from types import SimpleNamespace

import pytest
import torch

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
        attention_backend: str | None = None,
    ) -> None:
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_prefix_caching = enable_prefix_caching
        self.enforce_eager = enforce_eager
        self.attention_backend = attention_backend
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
                outputs=[
                    SimpleNamespace(
                        token_ids=[sum(prompt) + index for index in range(params.max_tokens)]
                    )
                ],
                finished=True,
            )
            for request_id, (prompt, params) in self.pending.items()
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
    # This CPU fake models two devices. Match its visibility contract even when
    # the surrounding real-GPU test shard was launched with a single device.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
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
        long_spillover_limit=0,
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
    runtime = vllm_runner._RequestRuntime(
        request, "gpu1", admitted_at=10.0, admitted_monotonic=100.0,
    )

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


def test_explicit_attention_backend_reaches_engine_config(vllm_016_api: None) -> None:
    vllm_runner._VllmWrapper("gpu0", 0, "/models/local-test-model", attention_backend="TRITON_ATTN")
    assert _FakeEngineArgs.created[-1].attention_backend == "TRITON_ATTN"


def test_attention_backend_cli_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vllm_runner.sys, "argv", ["router", "--attention-backend", "TRITON_ATTN"])
    assert vllm_runner._parse_cli_args().attention_backend == "TRITON_ATTN"
    monkeypatch.setattr(vllm_runner.sys, "argv", ["router"])
    assert vllm_runner._parse_cli_args().attention_backend is None


def test_long_spillover_cli_is_bounded_opt_in() -> None:
    assert vllm_runner.parse_vllm_target_overrides([]).long_spillover_limit == 0
    assert (
        vllm_runner.parse_vllm_target_overrides(
            ["--long-spillover-limit", "1"]
        ).long_spillover_limit
        == 1
    )
    with pytest.raises(ValueError, match="long-spillover-limit must be non-negative"):
        vllm_runner.parse_vllm_target_overrides(
            ["--long-spillover-limit", "-1"]
        )


def test_uuid_cuda_visibility_fails_before_cuda_or_engine_construction(
    vllm_016_api: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visible = (
        "GPU-7f4a68da-6dbf-a5df-0493-5f3f6e7786fd,"
        "GPU-b4de3d1a-a4fd-27e2-688b-dd8bfb31bfbb"
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(
        vllm_runner.torch.cuda,
        "is_available",
        lambda: pytest.fail("UUID visibility must fail before CUDA inspection"),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Pinned vLLM .* does not accept GPU UUID tokens.*"
            r"same UUIDs to numeric physical indices.*same order"
        ),
    ):
        vllm_runner.create_dual_pool_vllm_session(
            "dual",
            topology_snapshot=_topology(),
            cli_args=_args(),
        )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == visible
    assert _FakeEngineArgs.created == []
    assert _FakeLLMEngine.created == []


def test_mig_cuda_visibility_rejects_parent_gpu_substitution(
    vllm_016_api: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visible = "MIG-GPU-7f4a68da-6dbf-a5df-0493-5f3f6e7786fd/1/0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(
        vllm_runner.torch.cuda,
        "is_available",
        lambda: pytest.fail("MIG visibility must fail before CUDA inspection"),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Pinned vLLM .* does not support MIG UUID tokens.*"
            r"Keep the MIG allocation unchanged.*do not replace it with the parent GPU"
        ),
    ):
        vllm_runner.create_dual_pool_vllm_session(
            "dual",
            topology_snapshot=_topology(),
            cli_args=_args(),
        )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == visible
    assert _FakeEngineArgs.created == []
    assert _FakeLLMEngine.created == []


@pytest.mark.parametrize("visible", [None, "0", "0,1", "7,3"])
def test_numeric_or_unset_cuda_visibility_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    visible: str | None,
) -> None:
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(vllm_runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vllm_runner.torch.cuda, "device_count", lambda: 2)

    assert (
        vllm_runner._require_vllm_host(
            workload_label="vLLM visibility contract", minimum_gpus=2
        )
        == 2
    )
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible


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
            prompt_token_ids=vllm_runner.build_prompt_token_ids([64, 64]),
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
            prompt_token_ids=vllm_runner.build_prompt_token_ids([4, 2, 2]),
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


def test_optimized_routing_uses_run_local_admission_depth_without_cuda_polling(
    vllm_016_api: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synchronize_calls: list[int] = []
    monkeypatch.setattr(
        vllm_runner.torch.cuda,
        "synchronize",
        lambda index: synchronize_calls.append(index),
    )
    args = _args()
    args.req_count = 4

    for _ in range(2):
        first_new_engine = len(_FakeLLMEngine.created)
        summary = vllm_runner.run_vllm_routing_with_topology(
            "optimized",
            topology_snapshot=_topology(),
            cli_args=args,
            prompt_token_ids=vllm_runner.build_prompt_token_ids([64] * args.req_count),
        )
        new_engines = _FakeLLMEngine.created[first_new_engine:]
        requests_by_device = {
            str(engine.config.device_config.device): [
                call["request_id"] for call in engine.add_calls
            ]
            for engine in new_engines
        }

        assert requests_by_device == {
            "cuda:0": ["session-0000-req-0", "session-0000-req-2"],
            "cuda:1": ["session-0000-req-1", "session-0000-req-3"],
        }
        assert summary["requests"] == summary["completed"] == 4
        assert summary["requests_admitted_gpu0"] == 2
        assert summary["requests_admitted_gpu1"] == 2
        assert len(summary[vllm_runner.VERIFICATION_OUTPUT_KEY]) == 8

    assert synchronize_calls == []


@pytest.mark.parametrize(
    (
        "mode",
        "decode_gpus",
        "spillover_limit",
        "expected_counts",
        "expected_prefill_counts",
        "expected_decode_counts",
        "expected_pool_counts",
        "expected_actual_spillover",
    ),
    [
        (
            "shared",
            "0,1",
            0,
            {"cuda:0": 51, "cuda:1": 51},
            {"cuda:0": 3, "cuda:1": 3},
            {"cuda:0": 48, "cuda:1": 48},
            (2, 2),
            0,
        ),
        (
            "shared",
            "0,1",
            1,
            {"cuda:0": 51, "cuda:1": 51},
            {"cuda:0": 3, "cuda:1": 3},
            {"cuda:0": 48, "cuda:1": 48},
            (2, 2),
            0,
        ),
        (
            "dual",
            "1",
            0,
            {"cuda:0": 6, "cuda:1": 96},
            {"cuda:0": 6, "cuda:1": 0},
            {"cuda:0": 0, "cuda:1": 96},
            (1, 1),
            0,
        ),
        (
            "dual",
            "1",
            1,
            {"cuda:0": 5, "cuda:1": 97},
            {"cuda:0": 5, "cuda:1": 1},
            {"cuda:0": 0, "cuda:1": 96},
            (1, 1),
            1,
        ),
    ],
)
def test_dual_pool_uses_run_local_admission_depth_across_reused_runs(
    vllm_016_api: None,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    decode_gpus: str,
    spillover_limit: int,
    expected_counts: dict[str, int],
    expected_prefill_counts: dict[str, int],
    expected_decode_counts: dict[str, int],
    expected_pool_counts: tuple[int, int],
    expected_actual_spillover: int,
) -> None:
    args = _args()
    args.decode_gpus = decode_gpus
    args.long_spillover_limit = spillover_limit
    args.long_prompt_tokens = 4096
    args.short_prompt_tokens = 128
    args.prefill_burst = 6
    args.decode_requests = 48
    args.continue_requests = 48
    args.prefill_ctx_thresh = 2048
    prompt_lengths = vllm_runner.dual_pool_prompt_lengths(args)
    prompt_token_ids = vllm_runner.build_prompt_token_ids(prompt_lengths)
    session = vllm_runner.create_dual_pool_vllm_session(
        mode,
        topology_snapshot=_topology(),
        cli_args=args,
        warmup_runs=1,
    )
    for wrapper in session.engines.values():
        monkeypatch.setattr(
            wrapper,
            "snapshot_metrics",
            lambda **_kwargs: pytest.fail("post-admission snapshot must not run"),
        )

    try:
        previous_add_counts = {
            gpu_id: len(wrapper.engine.add_calls)
            for gpu_id, wrapper in session.engines.items()
        }
        for run_index in range(2):
            summary = vllm_runner.run_dual_pool_vllm_with_topology(
                mode,
                topology_snapshot=_topology(),
                cli_args=args,
                prompt_token_ids=prompt_token_ids,
                engine_session=session,
            )
            observed_counts = {}
            observed_prefill_counts = {}
            observed_decode_counts = {}
            for gpu_id, wrapper in session.engines.items():
                current = len(wrapper.engine.add_calls)
                device = f"cuda:{wrapper.device_index}"
                new_calls = wrapper.engine.add_calls[
                    previous_add_counts[gpu_id] : current
                ]
                observed_counts[device] = len(new_calls)
                observed_prefill_counts[device] = sum(
                    len(call["prompt"]) == args.long_prompt_tokens
                    for call in new_calls
                )
                observed_decode_counts[device] = sum(
                    len(call["prompt"]) == args.short_prompt_tokens
                    for call in new_calls
                )
                new_request_ids = [
                    call["request_id"]
                    for call in new_calls
                ]
                assert all(
                    request_id.startswith(f"session-{run_index:04d}-")
                    for request_id in new_request_ids
                )
                previous_add_counts[gpu_id] = current

            assert observed_counts == expected_counts
            assert observed_prefill_counts == expected_prefill_counts
            assert observed_decode_counts == expected_decode_counts
            assert summary["requests"] == summary["completed"] == 102
            for gpu_index in (0, 1):
                device = f"cuda:{gpu_index}"
                gpu_id = f"gpu{gpu_index}"
                assert summary[f"requests_admitted_{gpu_id}"] == expected_counts[device]
                assert (
                    summary[f"prefill_requests_admitted_{gpu_id}"]
                    == expected_prefill_counts[device]
                )
                assert (
                    summary[f"decode_requests_admitted_{gpu_id}"]
                    == expected_decode_counts[device]
                )
            assert (
                summary["prefill_gpu_count"],
                summary["decode_gpu_count"],
            ) == expected_pool_counts
            assert summary["long_spillover_limit"] == spillover_limit
            assert summary["long_spillover_requests"] == expected_actual_spillover
            expected_outputs = []
            for prompt_length in prompt_lengths:
                expected_outputs.extend((1, prompt_length))
            assert summary[vllm_runner.VERIFICATION_OUTPUT_KEY] == expected_outputs
    finally:
        session.close()


@pytest.mark.parametrize(
    ("call_path", "mode"),
    [
        ("dynamic", "baseline"),
        ("dynamic", "optimized"),
        ("dual_pool", "shared"),
        ("dual_pool", "dual"),
    ],
)
def test_blocking_vllm_loops_do_not_add_poll_delay(
    vllm_016_api: None,
    monkeypatch: pytest.MonkeyPatch,
    call_path: str,
    mode: str,
) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        vllm_runner.time,
        "sleep",
        lambda seconds: sleep_calls.append(seconds),
    )
    args = _args()
    if call_path == "dynamic":
        vllm_runner.run_vllm_routing_with_topology(
            mode,
            topology_snapshot=_topology(),
            cli_args=args,
            prompt_token_ids=vllm_runner.build_prompt_token_ids([64, 64]),
        )
    else:
        if mode == "dual":
            args.decode_gpus = "1"
        vllm_runner.run_dual_pool_vllm_with_topology(
            mode,
            topology_snapshot=_topology(),
            cli_args=args,
            prompt_token_ids=vllm_runner.build_prompt_token_ids([4, 2, 2]),
        )

    assert sleep_calls == []


def test_explicit_zero_request_groups_do_not_restore_default_work() -> None:
    args = _args()
    assert vllm_runner.dual_pool_prompt_lengths(
        args, prefill_burst=0, decode_requests=1, continue_requests=0,
        short_prompt_tokens=2,
    ) == [2]
    with pytest.raises(ValueError, match="at least one request"):
        vllm_runner.dual_pool_prompt_lengths(args, prefill_burst=0, decode_requests=0, continue_requests=0)
    with pytest.raises(ValueError, match="req_count must be positive"):
        vllm_runner.routing_prompt_lengths(args, req_count=0)
    with pytest.raises(ValueError, match="prompt token counts must be positive"):
        vllm_runner.dual_pool_prompt_lengths(args, long_prompt_tokens=0)


def test_zero_request_groups_reach_actual_admission_loop(vllm_016_api: None) -> None:
    summary = vllm_runner.run_dual_pool_vllm(
        "shared", cli_args=_args(), topology_snapshot=_topology(),
        prefill_burst=0, decode_requests=1, continue_requests=0,
        short_prompt_tokens=2, max_tokens=1,
    )
    assert summary["requests"] == summary["completed"] == 1
    assert sum(len(engine.add_calls) for engine in _FakeLLMEngine.created) == 1


def test_prompt_token_ids_preserve_default_prompts_and_are_live() -> None:
    prompt_lengths = [4, 2]
    prompt_token_ids = vllm_runner.build_prompt_token_ids(prompt_lengths)

    assert prompt_token_ids.shape == (1, 6)
    assert prompt_token_ids.dtype == torch.int64
    assert vllm_runner._split_prompt_token_ids(
        prompt_token_ids, prompt_lengths
    ) == [[1, 1, 1, 1], [1, 1]]

    prompt_token_ids[0, (0, 4)] = 0
    assert vllm_runner._split_prompt_token_ids(
        prompt_token_ids, prompt_lengths
    ) == [[0, 1, 1, 1], [0, 1]]


def test_request_admission_rejects_non_cpu_prompt_ids_before_conversion() -> None:
    prompt_token_ids = torch.empty((1, 6), dtype=torch.int64, device="meta")
    with pytest.raises(ValueError, match="requires CPU prompt_token_ids"):
        vllm_runner._split_prompt_token_ids(prompt_token_ids, [4, 2])


@pytest.mark.parametrize("call_path", ["dynamic", "dual_pool"])
def test_changed_prompt_ids_change_complete_generated_outputs(
    vllm_016_api: None,
    call_path: str,
) -> None:
    args = _args()
    if call_path == "dynamic":
        prompt_lengths = vllm_runner.routing_prompt_lengths(args)
        run = partial(
            vllm_runner.run_vllm_routing_with_topology,
            "optimized",
            topology_snapshot=_topology(),
            cli_args=args,
        )
    else:
        args.decode_gpus = "1"
        prompt_lengths = vllm_runner.dual_pool_prompt_lengths(args)
        run = partial(
            vllm_runner.run_dual_pool_vllm_with_topology,
            "dual",
            topology_snapshot=_topology(),
            cli_args=args,
        )

    prompt_token_ids = vllm_runner.build_prompt_token_ids(prompt_lengths)
    first_new_engine = len(_FakeLLMEngine.created)
    original = run(prompt_token_ids=prompt_token_ids)
    prompt_token_ids.zero_()
    changed = run(prompt_token_ids=prompt_token_ids)

    verification_key = vllm_runner.VERIFICATION_OUTPUT_KEY
    assert original[verification_key] != changed[verification_key]
    add_calls = [
        call
        for engine in _FakeLLMEngine.created[first_new_engine:]
        for call in engine.add_calls
    ]
    assert any(0 in call["prompt"] for call in add_calls)


@pytest.mark.parametrize(
    ("module_name", "benchmark_name", "expected_requests", "expected_tokens"),
    [
        (
            "labs.dynamic_router.baseline_dynamic_router_vllm",
            "BaselineDynamicRouterVllmBenchmark",
            2,
            130,
        ),
        (
            "labs.dynamic_router.optimized_dynamic_router_vllm",
            "OptimizedDynamicRouterVllmBenchmark",
            2,
            130,
        ),
        (
            "labs.dynamic_router.baseline_dual_pool_vllm",
            "BaselineDualPoolVllmBenchmark",
            3,
            11,
        ),
        (
            "labs.dynamic_router.optimized_dual_pool_vllm",
            "OptimizedDualPoolVllmBenchmark",
            3,
            11,
        ),
    ],
)
def test_vllm_wrappers_declare_multigpu_and_bind_live_prompt_input(
    vllm_016_api: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    module_name: str,
    benchmark_name: str,
    expected_requests: int,
    expected_tokens: int,
) -> None:
    from core.benchmark.verification import select_jitter_dimension
    from core.benchmark.verify_runner import VerifyConfig, VerifyRunner

    module = importlib.import_module(module_name)
    args = _args()
    if "dual_pool" in module_name:
        args.decode_gpus = "1"
    monkeypatch.setattr(vllm_runner, "_CLI_ARGS", args)
    monkeypatch.setattr(module, "detect_topology", lambda **_kwargs: _topology())

    benchmark_type = getattr(module, benchmark_name)
    benchmark = benchmark_type()
    assert benchmark.multi_gpu_required is True
    assert benchmark.get_config().multi_gpu_required is True
    assert benchmark._is_deterministic is True
    metadata = benchmark.get_workload_metadata()
    assert metadata.requests_per_iteration == expected_requests
    assert metadata.tokens_per_iteration == expected_tokens

    try:
        benchmark.setup()
        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        signature = benchmark.get_input_signature()
        assert select_jitter_dimension(signature) == ("prompt_token_ids", 1)
        prompt_ids = benchmark.get_verify_inputs()["prompt_token_ids"]
        assert prompt_ids.shape[0] == 1
        assert prompt_ids.shape[1] == sum(benchmark._prompt_lengths)

        torch.manual_seed(123)
        jitter_ok, jitter_error = VerifyRunner(
            cache_dir=tmp_path / "cache"
        )._run_jitter_check(benchmark, signature, VerifyConfig())
        assert jitter_ok, jitter_error
    finally:
        benchmark.teardown()


def test_vllm_verification_rejects_metric_only_summary() -> None:
    from labs.dynamic_router.verification import require_verification_output

    with pytest.raises(RuntimeError, match="timing and routing metrics are not model outputs"):
        require_verification_output({"ttft_ms_p95": 1.0, "completed": 2})


@pytest.mark.parametrize(
    ("baseline_module_name", "optimized_module_name"),
    [
        (
            "labs.dynamic_router.baseline_dynamic_router_vllm",
            "labs.dynamic_router.optimized_dynamic_router_vllm",
        ),
        (
            "labs.dynamic_router.baseline_dual_pool_vllm",
            "labs.dynamic_router.optimized_dual_pool_vllm",
        ),
    ],
)
def test_full_vllm_pair_verification_uses_live_generated_outputs(
    vllm_016_api: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    baseline_module_name: str,
    optimized_module_name: str,
) -> None:
    from core.benchmark.verify_runner import VerifyRunner

    baseline_module = importlib.import_module(baseline_module_name)
    optimized_module = importlib.import_module(optimized_module_name)
    args = _args()
    if "dual_pool" in baseline_module_name:
        args.decode_gpus = "1"
    monkeypatch.setattr(vllm_runner, "_CLI_ARGS", args)
    for module in (baseline_module, optimized_module):
        monkeypatch.setattr(module, "detect_topology", lambda **_kwargs: _topology())

    baseline = baseline_module.get_benchmark()
    optimized = optimized_module.get_benchmark()
    baseline.device = torch.device("cpu")
    optimized.device = torch.device("cpu")
    result = VerifyRunner(cache_dir=tmp_path / "cache").verify_pair(
        baseline,
        optimized,
    )

    assert result.passed, result.reason


def test_v1_direct_loop_completes_required_engine_post_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post_steps: list[bool] = []
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        vllm_runner.time,
        "sleep",
        lambda seconds: sleep_calls.append(seconds),
    )
    wrapper = vllm_runner._VllmV1Wrapper.__new__(vllm_runner._VllmV1Wrapper)
    wrapper._core = SimpleNamespace(
        step_fn=lambda: ({}, False),
        post_step=lambda *, model_executed: post_steps.append(model_executed),
    )
    wrapper._inflight = {}

    assert wrapper.step() == ([], [], 0)
    assert post_steps == [False]
    assert sleep_calls == [0.0]


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
            admitted_monotonic=10.0,
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
