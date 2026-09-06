"""Real telemetry/parser math with request-output payload fixtures, not vLLM execution."""

import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from labs.dynamic_router.router_policy import Router
from labs.dynamic_router.router_round_robin import Request
from labs.dynamic_router.vllm_runner import _RequestRuntime, _RoutingTelemetry, _VllmWrapper


def output(request, count, finished=False):
    return SimpleNamespace(request_id=request, outputs=[SimpleNamespace(token_ids=list(range(count)))], finished=finished)


def runtime(name, *, expected_new_tokens=8):
    return _RequestRuntime(
        Request(
            req_id=name,
            prompt_tokens=4,
            expected_new_tokens=expected_new_tokens,
        ),
        "gpu0",
        100.0,
    )


def test_idle_engine_is_not_asked_to_block_for_output():
    class IdleEngine:
        def get_num_unfinished_requests(self):
            return 0

        def step(self):
            raise AssertionError("Idle vLLM step blocks waiting for output")

    wrapper = _VllmWrapper.__new__(_VllmWrapper)
    wrapper.engine = IdleEngine()
    assert wrapper.step() == ([], [], 0)


def test_cumulative_output_payloads_count_each_token_once_and_ttft_once():
    wrapper = _VllmWrapper.__new__(_VllmWrapper)
    wrapper._inflight = {
        "a": runtime("a", expected_new_tokens=3),
        "b": runtime("b", expected_new_tokens=5),
    }
    completed, first, tokens = wrapper._consume_request_outputs([output("a", 1), output("b", 2)], 100.25)
    assert completed == [] and tokens == 3
    assert first == [("a", 250), ("b", 250)]
    completed, first, tokens = wrapper._consume_request_outputs([output("a", 1), output("b", 4)], 100.5)
    assert completed == [] and first == [] and tokens == 2
    completed, first, tokens = wrapper._consume_request_outputs([output("a", 3, True), output("b", 5, True)], 101)
    assert completed == ["a", "b"] and first == [] and tokens == 3
    assert wrapper._inflight == {}
    assert wrapper._completed_output_token_ids == {
        "a": (0, 1, 2),
        "b": (0, 1, 2, 3, 4),
    }
    assert wrapper._consume_request_outputs([output("a", 3, True)], 102) == ([], [], 0)


def test_zero_output_does_not_create_a_first_token_sample():
    request = runtime("a")
    assert request.observe_cumulative_tokens(0, 100.5) == (0, None)
    assert request.observe_cumulative_tokens(2, 101.25) == (2, 1250)
    assert request.observe_cumulative_tokens(2, 102) == (0, None)
    with pytest.raises(RuntimeError, match="Cumulative.*decreased"):
        request.observe_cumulative_tokens(1, 103)


def test_latency_and_throughput_have_independent_units_and_idle_steps_decay():
    metrics = _RoutingTelemetry()
    assert metrics.snapshot_args() == {"ttft_ema": None, "tpot_ema": 0}
    metrics.observe([("first", 400)], 8)
    assert metrics.snapshot_args() == {"ttft_ema": 400, "tpot_ema": 8}
    metrics.observe([], 0)
    assert metrics.snapshot_args()["ttft_ema"] == 400
    assert metrics.snapshot_args()["tpot_ema"] == pytest.approx(5.6)
    metrics.observe([("next", 200)], 3)
    assert metrics.snapshot_args()["ttft_ema"] == pytest.approx(340)
    assert metrics.snapshot_args()["tpot_ema"] == pytest.approx(4.82)


def test_real_router_prefers_lower_ttft_when_other_telemetry_is_equal():
    router = Router()
    for gid, latency in (("gpu0", 1000), ("gpu1", 50)):
        router.register_gpu(gid, is_prefill=True, is_decode=True)
        telemetry = _RoutingTelemetry()
        telemetry.observe([("request", latency)], 8)
        values = telemetry.snapshot_args()
        router.update_metrics(gid, {"ttft_ms": values["ttft_ema"], "tpot": values["tpot_ema"],
                                    "queue_depth": 1, "mem_free_gb": 10})
    assert router.choose_prefill_gpu() == "gpu1"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Actual two-GPU vLLM stack required")
def test_real_vllm_routing_with_explicit_local_model():
    from labs.dynamic_router.vllm_runner import _CLI_ARGS, run_vllm_routing

    model = os.environ.get("AISP_AUDIT_VLLM_MODEL")
    if not model:
        pytest.skip("Set AISP_AUDIT_VLLM_MODEL to an existing local model for real runtime validation")
    assert Path(model).is_dir(), "Acceptance must use the explicit local model, not download an inferred model"
    args = copy.copy(_CLI_ARGS)
    args.model, args.decode_gpus = model, "0,1"
    result = run_vllm_routing("optimized", req_count=4, max_tokens=2, cli_args=args)
    assert result["completed"] == result["requests"] == 4
    assert result["ttft_ms_mean"] > 0
    assert result["ttft_ms_p95"] >= result["ttft_ms_p50"] > 0
