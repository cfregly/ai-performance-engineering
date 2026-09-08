"""
vLLM-backed dynamic router runner.

This replaces the virtual simulator with real LLMEngine instances. It is a thin,
opt-in harness hook: if vLLM or the model is unavailable, it raises SKIPPED.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import io
import json
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import wraps
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from core.harness.serving_stack import get_serving_stack_pins

try:
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.sampling_params import RequestOutputKind
except Exception as exc:  # pragma: no cover - optional dep
    EngineArgs = None  # type: ignore
    LLMEngine = None  # type: ignore
    SamplingParams = None  # type: ignore
    RequestOutputKind = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from labs.dynamic_router.router_policy import EWMA, Router, SequenceInfo
from labs.dynamic_router.router_round_robin import Request
from labs.dynamic_router.topology import TopologySnapshot, detect_topology
from labs.dynamic_router.verification import VERIFICATION_OUTPUT_KEY


def _skip(reason: str) -> None:
    raise RuntimeError(f"SKIPPED: {reason}")


_SERVING_STACK_PINS = get_serving_stack_pins()
_PINNED_SERVING_STACK = _SERVING_STACK_PINS.pinned_stack_str
_EXPECTED_TORCH_VERSION = _SERVING_STACK_PINS.torch_version
_EXPECTED_VLLM_DIST_VERSION = _SERVING_STACK_PINS.vllm_version
_EXPECTED_FLASHINFER_DIST_VERSION = _SERVING_STACK_PINS.flashinfer_version
WARMUP_ITERATIONS = 5
STEADY_STATE_ITERATIONS = 3


def _is_vllm_abi_mismatch_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        ("undefined symbol" in text and ("vllm/_c.abi3.so" in text or "vllm._c" in text))
        or "c10_cuda_check_implementation" in text
    )


def _format_vllm_import_error(exc: BaseException) -> str:
    if _is_vllm_abi_mismatch_error(exc):
        return (
            "vLLM ABI mismatch detected while importing compiled extensions. "
            f"Pin and reinstall the benchmark-host stack ({_PINNED_SERVING_STACK}). "
            "Then verify with: "
            "`python -c \"import importlib, importlib.metadata as md, torch, vllm; "
            "importlib.import_module('vllm._C'); "
            "print(torch.__version__, md.version('vllm'), vllm.__version__)\"`.\n"
            f"Original error: {exc}"
        )
    return f"vLLM import failed: {exc}"


def _distribution_version(dist_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _assert_serving_stack_versions() -> None:
    torch_version = torch.__version__
    if torch_version != _EXPECTED_TORCH_VERSION:
        _skip(
            "Serving stack mismatch: expected "
            f"torch=={_EXPECTED_TORCH_VERSION}, got {torch_version}. "
            f"Pin and reinstall {_PINNED_SERVING_STACK}."
        )

    vllm_version = _distribution_version("vllm")
    if vllm_version is None:
        _skip("vLLM is not installed. Pin and reinstall " + _PINNED_SERVING_STACK + ".")
    if vllm_version != _EXPECTED_VLLM_DIST_VERSION:
        _skip(
            "Serving stack mismatch: expected "
            f"vllm=={_EXPECTED_VLLM_DIST_VERSION}, got {vllm_version}. "
            f"Pin and reinstall {_PINNED_SERVING_STACK}."
        )

    flashinfer_version = _distribution_version("flashinfer-python")
    if flashinfer_version is None:
        _skip(
            "flashinfer-python is not installed. "
            f"Pin and reinstall {_PINNED_SERVING_STACK}."
        )
    if flashinfer_version != _EXPECTED_FLASHINFER_DIST_VERSION:
        _skip(
            "Serving stack mismatch: expected "
            f"flashinfer-python=={_EXPECTED_FLASHINFER_DIST_VERSION}, got {flashinfer_version}. "
            f"Pin and reinstall {_PINNED_SERVING_STACK}."
        )


def _assert_vllm_runtime_ready() -> None:
    """Fail fast with actionable remediation before launching any lab workload."""
    _assert_serving_stack_versions()
    if _IMPORT_ERROR is not None:
        _skip(_format_vllm_import_error(_IMPORT_ERROR))
    try:
        importlib.import_module("vllm._C")
    except Exception as exc:  # pragma: no cover - optional dep/runtime ABI
        _skip(_format_vllm_import_error(exc))


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", type=str, help="Local HF model path/id for vLLM.")
    parser.add_argument(
        "--attention-backend", type=str, default=None,
        help="Explicit vLLM attention backend; use TRITON_ATTN with VLLM_BATCH_INVARIANT=1 for exact dual-pool comparisons.",
    )
    parser.add_argument("--prefill-gpus", type=str, default=None, help="Comma list of GPU ids for prefill pool.")
    parser.add_argument("--decode-gpus", type=str, default=None, help="Comma list of GPU ids for decode pool.")
    parser.add_argument("--req-count", type=int, default=16, help="Number of requests for routing demo.")
    parser.add_argument("--max-tokens", type=int, default=16, help="Max tokens per request.")
    parser.add_argument("--long-prompt-tokens", type=int, default=4096, help="Long prompt size for dual-pool demo.")
    parser.add_argument("--short-prompt-tokens", type=int, default=128, help="Short prompt size for decode-heavy demo.")
    parser.add_argument("--prefill-burst", type=int, default=6, help="Number of long prompts to inject for prefill load.")
    parser.add_argument("--decode-requests", type=int, default=48, help="Decode-style requests for dual-pool demo.")
    parser.add_argument("--continue-requests", type=int, default=48, help="Continuation requests for dual-pool demo.")
    parser.add_argument("--prefill-ctx-thresh", type=int, default=2048, help="Threshold to route to prefill pool.")
    parser.add_argument(
        "--use-v1-core-loop",
        action="store_true",
        help="Drive vLLM V1 EngineCore directly with the optimized polling loop (Inproc only).",
    )
    return parser.parse_known_args()[0]


_CLI_ARGS = _parse_cli_args()


def routing_prompt_lengths(
    cli_args: argparse.Namespace,
    *,
    req_count: Optional[int] = None,
) -> List[int]:
    """Return the exact per-request prompt lengths for the routing workload."""
    count = cli_args.req_count if req_count is None else req_count
    if count <= 0:
        raise ValueError("req_count must be positive")
    return [64] * count


def dual_pool_prompt_lengths(
    cli_args: argparse.Namespace,
    *,
    long_prompt_tokens: Optional[int] = None,
    short_prompt_tokens: Optional[int] = None,
    prefill_burst: Optional[int] = None,
    decode_requests: Optional[int] = None,
    continue_requests: Optional[int] = None,
) -> List[int]:
    """Return the exact request order and prompt lengths for the dual-pool workload."""
    long_tokens = cli_args.long_prompt_tokens if long_prompt_tokens is None else long_prompt_tokens
    short_tokens = cli_args.short_prompt_tokens if short_prompt_tokens is None else short_prompt_tokens
    prefill_count = cli_args.prefill_burst if prefill_burst is None else prefill_burst
    decode_count = cli_args.decode_requests if decode_requests is None else decode_requests
    continue_count = cli_args.continue_requests if continue_requests is None else continue_requests
    if long_tokens <= 0 or short_tokens <= 0:
        raise ValueError("prompt token counts must be positive")
    if prefill_count < 0 or decode_count < 0 or continue_count < 0:
        raise ValueError("request counts must be non-negative")
    lengths = [long_tokens] * prefill_count
    lengths.extend([short_tokens] * (decode_count + continue_count))
    if not lengths:
        raise ValueError("dual-pool workload must contain at least one request")
    return lengths


def build_prompt_token_ids(prompt_lengths: Sequence[int]) -> torch.Tensor:
    """Build the flat live input containing every request's actual token IDs."""
    lengths = [int(length) for length in prompt_lengths]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("prompt_lengths must contain only positive values")
    return torch.ones((1, sum(lengths)), dtype=torch.int64, device="cpu")


def _split_prompt_token_ids(
    prompt_token_ids: torch.Tensor,
    prompt_lengths: Sequence[int],
) -> List[List[int]]:
    """Split the flat verification input into the exact vLLM request prompts."""
    lengths = [int(length) for length in prompt_lengths]
    expected_shape = (1, sum(lengths))
    if tuple(prompt_token_ids.shape) != expected_shape:
        raise ValueError(
            "prompt_token_ids shape mismatch: "
            f"expected {expected_shape}, got {tuple(prompt_token_ids.shape)}"
        )
    if prompt_token_ids.dtype != torch.int64:
        raise TypeError("prompt_token_ids must use torch.int64")
    if prompt_token_ids.device.type != "cpu":
        raise ValueError("vLLM request admission requires CPU prompt_token_ids")
    if bool((prompt_token_ids < 0).any()):
        raise ValueError("prompt_token_ids must be non-negative")

    flat_token_ids = prompt_token_ids.reshape(-1).tolist()
    requests: List[List[int]] = []
    offset = 0
    for length in lengths:
        requests.append(flat_token_ids[offset : offset + length])
        offset += length
    return requests


@dataclass
class _RequestRuntime:
    req: Request
    gpu_id: str
    admitted_at: float
    ttft_ms: Optional[float] = None
    finished: bool = False
    role: str = "shared"
    observed_output_tokens: int = 0

    def observe_cumulative_tokens(self, total: int, observed_at: float) -> Tuple[int, Optional[float]]:
        """Consume one cumulative request output, returning delta and new TTFT."""
        if total < self.observed_output_tokens:
            raise RuntimeError("Cumulative vLLM output token count decreased")
        delta = total - self.observed_output_tokens
        self.observed_output_tokens = total
        first_ttft = None
        if self.ttft_ms is None and delta > 0:
            self.ttft_ms = (observed_at - self.admitted_at) * 1000.0
            first_ttft = self.ttft_ms
        return delta, first_ttft


class _RoutingTelemetry:
    """Keep first-token milliseconds separate from output tokens per poll step.

    The existing router's `tpot` field represents a higher-is-better throughput
    proxy here, not time per output token. No tokens/second claim is made.
    """

    def __init__(self) -> None:
        self.ttft_ms = EWMA(0.3)
        self.tokens_per_step = EWMA(0.3)

    def observe(self, ttft_samples: List[Tuple[str, float]], tokens: int) -> None:
        for _, sample in ttft_samples:
            self.ttft_ms.update(sample)
        self.tokens_per_step.update(float(tokens))

    def snapshot_args(self) -> Dict[str, Optional[float]]:
        return {
            "ttft_ema": self.ttft_ms.get(default=None),
            "tpot_ema": self.tokens_per_step.get(),
        }


def _build_vllm_engine(engine_cls, model_id: str, device_index: int, *, attention_backend: str | None = None):
    """Build a pinned-vLLM engine on one logical CUDA device.

    vLLM 0.16 removed ``device`` from ``EngineArgs``. Its single-process
    executor still selects a logical rank from ``VllmConfig.device_config``,
    so set that current config field before constructing the engine.
    """
    if EngineArgs is None:
        _skip(_format_vllm_import_error(_IMPORT_ERROR or RuntimeError("EngineArgs is unavailable")))

    engine_args = EngineArgs(
        model=model_id,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.5,
        # Every request supplies the declared number of token IDs directly.
        # Disable prefix caching so repeated synthetic inputs still exercise
        # the full prefill length on every request.
        enable_prefix_caching=False,
        enforce_eager=True,
        attention_backend=attention_backend,
    )
    create_engine_config = getattr(engine_args, "create_engine_config", None)
    from_vllm_config = getattr(engine_cls, "from_vllm_config", None)
    if not callable(create_engine_config) or not callable(from_vllm_config):
        _skip(
            "Pinned vLLM API mismatch: expected EngineArgs.create_engine_config(), "
            f"VllmConfig.device_config, and LLMEngine.from_vllm_config() in vLLM "
            f"{_EXPECTED_VLLM_DIST_VERSION}."
        )
    vllm_config = create_engine_config()
    device_config = getattr(vllm_config, "device_config", None)
    if device_config is None:
        _skip(
            "Pinned vLLM API mismatch: expected VllmConfig.device_config in "
            f"vLLM {_EXPECTED_VLLM_DIST_VERSION}."
        )
    device_config.device = torch.device("cuda", device_index)
    return from_vllm_config(vllm_config)


class _VllmWrapper:
    """Minimal wrapper around LLMEngine for metrics and request tracking."""

    def __init__(self, gpu_id: str, device_index: int, model_id: str, *, attention_backend: str | None = None) -> None:
        _assert_vllm_runtime_ready()
        if EngineArgs is None or LLMEngine is None or SamplingParams is None:
            _skip(_format_vllm_import_error(_IMPORT_ERROR or RuntimeError("unknown vLLM import failure")))

        self.gpu_id = gpu_id
        self.device_index = device_index
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.engine = _build_vllm_engine(LLMEngine, model_id, device_index, attention_backend=attention_backend)
        captured = buf.getvalue().strip()
        if captured:
            try:
                lines = [ln for ln in captured.splitlines() if ln]
                print(json.dumps({"event": "vllm_engine_init_stdout", "gpu": gpu_id, "lines": lines}), file=sys.stderr)
            except Exception:
                print(captured, file=sys.stderr)
        self._inflight: Dict[str, _RequestRuntime] = {}
        self._completed_output_token_ids: Dict[str, Tuple[int, ...]] = {}
        self._closed = False

    def add_request(
        self,
        rt: _RequestRuntime,
        prompt_token_ids: Optional[Sequence[int]] = None,
    ) -> None:
        if rt.req.expected_new_tokens <= 0:
            raise ValueError("expected_new_tokens must be positive")
        params = SamplingParams(
            temperature=0.0,
            max_tokens=rt.req.expected_new_tokens,
            # The verification payload must contain actual model output for
            # every declared decode step.  Otherwise an immediate EOS can
            # produce an empty completion in both arms and make an exact
            # baseline/optimized comparison pass without checking any token.
            ignore_eos=True,
            output_kind=RequestOutputKind.CUMULATIVE,
        )
        if rt.req.prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        # vLLM accepts pretokenized ``list[int]`` prompts. Supplying the token
        # IDs directly makes the requested prefill length exact; a text string
        # such as ``"x" * prompt_tokens`` can collapse to far fewer BPE tokens.
        if prompt_token_ids is None:
            prompt_ids = [1] * rt.req.prompt_tokens
        else:
            prompt_ids = [int(token_id) for token_id in prompt_token_ids]
            if len(prompt_ids) != rt.req.prompt_tokens:
                raise ValueError(
                    f"Request {rt.req.req_id} received {len(prompt_ids)} prompt tokens; "
                    f"expected {rt.req.prompt_tokens}"
                )
            if any(token_id < 0 for token_id in prompt_ids):
                raise ValueError("prompt token ids must be non-negative")
        self.engine.add_request(
            request_id=rt.req.req_id,
            prompt=prompt_ids,
            params=params,
            arrival_time=rt.admitted_at,
        )
        self._inflight[rt.req.req_id] = rt

    def step(self, now: Optional[float] = None) -> Tuple[List[str], List[Tuple[str, float]], int]:
        """
        Advance engine.

        Returns:
          - finished_ids: request ids that completed in this step
          - ttft_samples_ms: list of (req_id, ttft_ms) for first tokens observed
          - tokens_emitted: total tokens emitted this step
        """
        # vLLM's multiprocessing engine waits for an output in step(). An idle
        # replica has nothing to emit and would stall the entire routing loop.
        if self.engine.get_num_unfinished_requests() == 0:
            return [], [], 0
        outputs = self.engine.step()
        # A pre-step caller timestamp omits the engine work that emits the first
        # token. Retain the argument for compatibility, but observe after step.
        return self._consume_request_outputs(outputs, time.time())

    def _consume_request_outputs(self, outputs, observed_at: float) -> Tuple[List[str], List[Tuple[str, float]], int]:
        """Parse cumulative vLLM output payloads without timing or engine mocks."""
        ttft_samples: List[Tuple[str, float]] = []
        finished_ids: List[str] = []
        tokens_emitted = 0
        for ro in outputs:
            rid = ro.request_id
            rt = self._inflight.get(rid)
            if rt is None:
                continue
            # Detect first token
            if ro.outputs:
                output_token_count = sum(len(o.token_ids) for o in ro.outputs)
                delta, first_ttft = rt.observe_cumulative_tokens(output_token_count, observed_at)
                if first_ttft is not None:
                    ttft_samples.append((rid, first_ttft))
                tokens_emitted += delta
            if ro.finished:
                finished_ids.append(rid)
                rt.finished = True
                completed_outputs = getattr(self, "_completed_output_token_ids", None)
                if completed_outputs is None:
                    completed_outputs = {}
                    self._completed_output_token_ids = completed_outputs
                token_ids = tuple(
                    int(token_id) for output in ro.outputs for token_id in output.token_ids
                )
                if len(token_ids) != rt.req.expected_new_tokens:
                    raise RuntimeError(
                        f"Request {rid} completed with {len(token_ids)} output tokens; "
                        f"expected {rt.req.expected_new_tokens}"
                    )
                completed_outputs[rid] = token_ids
                self._inflight.pop(rid, None)
        return finished_ids, ttft_samples, tokens_emitted

    def queue_depth(self) -> int:
        return self.engine.get_num_unfinished_requests()

    def reset_request_state(self) -> None:
        """Prepare an idle engine for another exact-output workload."""
        if self._closed:
            raise RuntimeError(f"vLLM engine {self.gpu_id} is already closed")
        unfinished = self.engine.get_num_unfinished_requests()
        if unfinished or self._inflight:
            raise RuntimeError(
                f"Cannot reuse vLLM engine {self.gpu_id} with unfinished requests: "
                f"engine={unfinished}, tracked={len(self._inflight)}"
            )
        self._completed_output_token_ids.clear()

    def close(self, *, force: bool = False) -> None:
        """Shut down the pinned vLLM EngineCore after all requests drain."""
        if self._closed:
            return
        unfinished = self.engine.get_num_unfinished_requests()
        if unfinished or self._inflight:
            if not force:
                raise RuntimeError(
                    f"Cannot close vLLM engine {self.gpu_id} with unfinished requests: "
                    f"engine={unfinished}, tracked={len(self._inflight)}"
                )
        errors: List[str] = []
        if force and self._inflight:
            abort_request = getattr(self.engine, "abort_request", None)
            if callable(abort_request):
                request_ids = list(self._inflight)
                try:
                    abort_request(request_ids)
                except Exception as exc:
                    errors.append(f"abort {request_ids}: {exc}")
            self._inflight.clear()
        core_client = getattr(self.engine, "engine_core", None)
        shutdown = getattr(core_client, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:
                errors.append(f"EngineCore shutdown: {exc}")
        self._closed = True
        if errors:
            raise RuntimeError("; ".join(errors))

    def snapshot_metrics(self, ttft_ema: Optional[float], tpot_ema: float) -> Dict[str, float]:
        mem_free_gb = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device_index)
            free_bytes, _ = torch.cuda.mem_get_info(self.device_index)
            mem_free_gb = free_bytes / (1024**3)
        host_local = max(mem_free_gb * 0.25, 0.0)
        metrics = {
            "tpot": tpot_ema,
            "queue_depth": float(self.queue_depth()),
            "mem_free_gb": mem_free_gb,
            "kv_hit_rate": 0.0,
            "host_kv_local_gb": host_local,
            "host_kv_remote_gb": 0.0,
        }
        if ttft_ema is not None:
            metrics["ttft_ms"] = ttft_ema
        return metrics


class _VllmV1Wrapper(_VllmWrapper):
    """
    V1 EngineCore path that uses the optimized polling loop semantics.

    This is intentionally limited to the in-process EngineCore (multiprocess off)
    so we can access ``engine_core.step_fn()`` and surface the executed flag.
    """

    def __init__(self, gpu_id: str, device_index: int, model_id: str, *, attention_backend: str | None = None) -> None:
        _assert_vllm_runtime_ready()
        if EngineArgs is None or SamplingParams is None:
            _skip(f"vLLM import failed: {_IMPORT_ERROR}")
        try:
            from vllm.v1.engine import EngineCoreOutputs
            from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
        except Exception as exc:  # pragma: no cover - optional dep
            _skip(f"vLLM V1 import failed: {exc}")

        self._EngineCoreOutputs = EngineCoreOutputs
        self.gpu_id = gpu_id
        self.device_index = device_index
        # Keep EngineCore in-process so we can drive step_fn() directly.
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.engine = _build_vllm_engine(V1LLMEngine, model_id, device_index, attention_backend=attention_backend)
        captured = buf.getvalue().strip()
        if captured:
            try:
                lines = [ln for ln in captured.splitlines() if ln]
                print(json.dumps({"event": "vllm_engine_v1_init_stdout", "gpu": gpu_id, "lines": lines}), file=sys.stderr)
            except Exception:
                print(captured, file=sys.stderr)
        core_client = getattr(self.engine, "engine_core", None)
        if core_client is None or not hasattr(core_client, "engine_core"):
            _skip("V1 EngineCore client is not available in-process; disable VLLM_ENABLE_V1_MULTIPROCESSING.")
        self._core = core_client.engine_core
        if not hasattr(self._core, "step_fn"):
            _skip("EngineCore.step_fn is unavailable; update vLLM to V1 or disable --use-v1-core-loop.")
        self._inflight: Dict[str, _RequestRuntime] = {}
        self._completed_output_token_ids: Dict[str, Tuple[int, ...]] = {}
        self._closed = False

    def step(self, now: Optional[float] = None) -> Tuple[List[str], List[Tuple[str, float]], int]:
        outputs_dict, executed = self._core.step_fn()
        self._core.post_step(model_executed=executed)
        ttft_samples: List[Tuple[str, float]] = []
        finished_ids: List[str] = []
        tokens_emitted = 0

        if outputs_dict:
            # Inproc returns {client_idx: EngineCoreOutputs}
            engine_core_outputs = outputs_dict.get(0)
            if engine_core_outputs is None:
                # fall back to treating dict as single output (rare)
                if isinstance(outputs_dict, self._EngineCoreOutputs):
                    engine_core_outputs = outputs_dict
            if engine_core_outputs is not None and engine_core_outputs.outputs:
                processed = self.engine.output_processor.process_outputs(
                    engine_core_outputs.outputs,
                    engine_core_timestamp=engine_core_outputs.timestamp,
                    iteration_stats=None,
                )
                # Maintain abort parity with LLMEngine.step()
                if processed.reqs_to_abort:
                    self.engine.engine_core.abort_requests(processed.reqs_to_abort)
                # Scheduler stats and MM cache logging are best-effort here.
                self.engine.output_processor.update_scheduler_stats(engine_core_outputs.scheduler_stats)

                finished_ids, ttft_samples, tokens_emitted = self._consume_request_outputs(
                    processed.request_outputs, time.time(),
                )

        # Keep polling if scheduler deferred execution this step.
        if executed is False and not finished_ids:
            time.sleep(0.0)

        return finished_ids, ttft_samples, tokens_emitted


@dataclass
class _GPUHandle:
    gpu_id: str
    device_index: int
    is_prefill: bool
    is_decode: bool
    numa_node: Optional[int] = None


class VllmEngineSession:
    """Own reusable vLLM engines and retain their full lifecycle timings."""

    def __init__(
        self,
        *,
        workload_kind: str,
        mode: str,
        handles: Sequence[_GPUHandle],
        model_id: str,
        attention_backend: Optional[str],
        wrapper_cls: type[_VllmWrapper],
        warmup_runs: int,
    ) -> None:
        if warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        self.workload_kind = workload_kind
        self.mode = mode
        self.handles = tuple(handles)
        self.model_id = model_id
        self.attention_backend = attention_backend
        self.warmup_runs = int(warmup_runs)
        self._created_at = time.perf_counter()
        self._closed = False
        self._run_count = 0
        self._active_run: Optional[Tuple[str, float]] = None
        self._phase_durations_ms: Dict[str, List[float]] = {
            "warmup": [],
            "steady_state": [],
        }
        self._failed_runs: List[Dict[str, object]] = []
        self._primary_failure: Optional[BaseException] = None
        self._teardown_ms: Optional[float] = None
        self._end_to_end_ms: Optional[float] = None
        self.engine_startup_ms = 0.0
        self.engines: Dict[str, _VllmWrapper] = {}

        startup_start = time.perf_counter()
        try:
            for handle in self.handles:
                self.engines[handle.gpu_id] = wrapper_cls(
                    handle.gpu_id,
                    handle.device_index,
                    model_id,
                    attention_backend=attention_backend,
                )
        except BaseException as exc:
            self.engine_startup_ms = (time.perf_counter() - startup_start) * 1000.0
            self._primary_failure = exc
            cleanup_start = time.perf_counter()
            cleanup_errors: List[str] = []
            for engine in self.engines.values():
                try:
                    engine.close(force=True)
                except Exception as cleanup_exc:
                    cleanup_errors.append(f"{engine.gpu_id}: {cleanup_exc}")
            self._teardown_ms = (time.perf_counter() - cleanup_start) * 1000.0
            self._end_to_end_ms = (time.perf_counter() - self._created_at) * 1000.0
            self._closed = True
            self._emit_lifecycle("startup_failed", cleanup_errors)
            if cleanup_errors and hasattr(exc, "add_note"):
                exc.add_note(
                    "vLLM partial-startup cleanup errors: " + "; ".join(cleanup_errors)
                )
            raise
        self.engine_startup_ms = (time.perf_counter() - startup_start) * 1000.0

    def validate_layout(
        self,
        *,
        workload_kind: str,
        mode: str,
        handles: Sequence[_GPUHandle],
        model_id: str,
        attention_backend: Optional[str],
    ) -> None:
        if self._closed:
            raise RuntimeError("vLLM engine session is closed")
        expected = (
            workload_kind,
            mode,
            tuple(handles),
            model_id,
            attention_backend,
        )
        actual = (
            self.workload_kind,
            self.mode,
            self.handles,
            self.model_id,
            self.attention_backend,
        )
        if actual != expected:
            raise RuntimeError(
                "Reusable vLLM engine session does not match the requested workload layout"
            )

    def begin_run(self) -> Tuple[str, str]:
        if self._active_run is not None:
            raise RuntimeError("vLLM engine session already has an active run")
        for engine in self.engines.values():
            engine.reset_request_state()
        phase = "warmup" if self._run_count < self.warmup_runs else "steady_state"
        request_prefix = f"session-{self._run_count:04d}-"
        self._active_run = (phase, time.perf_counter())
        return phase, request_prefix

    def finish_run(self, phase: str) -> None:
        active = self._active_run
        if active is None or active[0] != phase:
            raise RuntimeError("vLLM engine session run phase is inconsistent")
        elapsed_ms = (time.perf_counter() - active[1]) * 1000.0
        self._phase_durations_ms[phase].append(elapsed_ms)
        self._run_count += 1
        self._active_run = None

    def abort_run(
        self,
        exc: BaseException,
        *,
        retain_inactive_failure: bool = False,
    ) -> None:
        """Release an active run lease while retaining its primary failure."""
        active = self._active_run
        if active is None:
            if retain_inactive_failure and self._primary_failure is None:
                self._primary_failure = exc
            return
        if self._primary_failure is None:
            self._primary_failure = exc
        self._failed_runs.append(
            {
                "phase": active[0],
                "elapsed_ms": (time.perf_counter() - active[1]) * 1000.0,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        self._active_run = None

    def lifecycle_metrics(self) -> Dict[str, float]:
        warmup_samples = self._phase_durations_ms["warmup"]
        steady_samples = self._phase_durations_ms["steady_state"]
        metrics = {
            "lifecycle.setup_engine_startup_ms": self.engine_startup_ms,
            "lifecycle.engine_startup_ms": self.engine_startup_ms,
            "lifecycle.engine_count": float(len(self.engines)),
            "lifecycle.warmup_runs": float(len(warmup_samples)),
            "lifecycle.warmup_request_processing_ms_total": float(sum(warmup_samples)),
            "lifecycle.warmup_request_processing_ms_mean": (
                float(sum(warmup_samples) / len(warmup_samples)) if warmup_samples else 0.0
            ),
            "lifecycle.steady_state_runs": float(len(steady_samples)),
            "lifecycle.steady_state_request_processing_ms_total": float(sum(steady_samples)),
            "lifecycle.steady_state_request_processing_ms_mean": (
                float(sum(steady_samples) / len(steady_samples)) if steady_samples else 0.0
            ),
            "lifecycle.steady_state_request_processing_ms_last": (
                float(steady_samples[-1]) if steady_samples else 0.0
            ),
            "lifecycle.engine_reuse_count": float(max(self._run_count - 1, 0)),
            "lifecycle.failed_runs": float(len(self._failed_runs)),
            "lifecycle.elapsed_before_teardown_ms": (
                time.perf_counter() - self._created_at
            )
            * 1000.0,
        }
        if self._teardown_ms is not None:
            metrics["lifecycle.engine_teardown_ms"] = self._teardown_ms
        if self._end_to_end_ms is not None:
            metrics["lifecycle.end_to_end_ms"] = self._end_to_end_ms
        return metrics

    def _emit_lifecycle(self, disposition: str, errors: Sequence[str]) -> None:
        failure = self._primary_failure
        print(
            json.dumps(
                {
                    "event": "vllm_engine_lifecycle",
                    "disposition": disposition,
                    "workload_kind": self.workload_kind,
                    "mode": self.mode,
                    "setup_engine_startup_ms": self.engine_startup_ms,
                    "engine_startup_ms": self.engine_startup_ms,
                    "warmup_request_processing_ms": self._phase_durations_ms["warmup"],
                    "steady_state_request_processing_ms": self._phase_durations_ms[
                        "steady_state"
                    ],
                    "failed_runs": self._failed_runs,
                    "engine_teardown_ms": self._teardown_ms,
                    "end_to_end_ms": self._end_to_end_ms,
                    "request_state_reset_per_run": True,
                    "primary_failure": (
                        {"type": type(failure).__name__, "message": str(failure)}
                        if failure is not None
                        else None
                    ),
                    "shutdown_errors": list(errors),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    def close(self, *, preserve_primary_error: bool = False) -> List[str]:
        if self._closed:
            return []
        if self._active_run is not None:
            if self._primary_failure is None:
                raise RuntimeError("Cannot close vLLM engine session during an active run")
            self.abort_run(self._primary_failure)
        teardown_start = time.perf_counter()
        errors: List[str] = []
        for engine in self.engines.values():
            try:
                engine.close(force=self._primary_failure is not None)
            except Exception as exc:
                errors.append(f"{engine.gpu_id}: {exc}")
        self._teardown_ms = (time.perf_counter() - teardown_start) * 1000.0
        self._end_to_end_ms = (time.perf_counter() - self._created_at) * 1000.0
        self._closed = True
        if self._primary_failure is not None:
            disposition = "failed_run"
        else:
            disposition = "completed"
        if errors:
            disposition += "_with_teardown_errors"
        self._emit_lifecycle(disposition, errors)
        if errors and self._primary_failure is not None and hasattr(
            self._primary_failure, "add_note"
        ):
            self._primary_failure.add_note(
                "vLLM engine teardown errors: " + "; ".join(errors)
            )
        if errors and not preserve_primary_error and self._primary_failure is None:
            raise RuntimeError("vLLM engine teardown failed: " + "; ".join(errors))
        return errors


def _parse_device_list(raw: Optional[str], default: str, max_device: int) -> List[int]:
    raw = raw or default
    ids: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            continue
        idx = int(part)
        if 0 <= idx < max_device:
            ids.append(idx)
    return sorted(set(ids))


def _percentile_from_ordered(data_sorted: List[float], pct: float) -> float:
    assert 0.0 <= pct <= 100.0
    k = (len(data_sorted) - 1) * (pct / 100.0)
    f = int(k // 1)
    c = int(k // 1 + 1)
    if f == c or c >= len(data_sorted):
        return data_sorted[f]
    d0 = data_sorted[f] * (c - k)
    d1 = data_sorted[c] * (k - f)
    return d0 + d1


def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    data.sort()
    return _percentile_from_ordered(data, pct)


def _percentiles(data: List[float], pcts: Tuple[float, ...]) -> Tuple[float, ...]:
    if not data:
        return tuple(0.0 for _ in pcts)
    data.sort()
    return tuple(_percentile_from_ordered(data, pct) for pct in pcts)


def _build_handles(
    mode: str, prefill_ids: List[int], decode_ids: List[int], gpu_numa: Optional[Dict[int, Optional[int]]] = None
) -> List[_GPUHandle]:
    handles: List[_GPUHandle] = []
    all_ids = sorted(set(prefill_ids + decode_ids))
    for idx in all_ids:
        gpu_id = f"gpu{idx}"
        numa_node = gpu_numa.get(idx) if gpu_numa else None
        if mode == "shared":
            handles.append(
                _GPUHandle(
                    gpu_id=gpu_id,
                    device_index=idx,
                    is_prefill=True,
                    is_decode=True,
                    numa_node=numa_node,
                )
            )
        else:
            handles.append(
                _GPUHandle(
                    gpu_id=gpu_id,
                    device_index=idx,
                    is_prefill=idx in prefill_ids,
                    is_decode=idx in decode_ids,
                    numa_node=numa_node,
                )
            )
    return handles


def _require_vllm_host(*, workload_label: str, minimum_gpus: int) -> int:
    if not torch.cuda.is_available():
        _skip(f"CUDA is required for {workload_label}.")
    total_gpus = torch.cuda.device_count()
    if total_gpus < minimum_gpus:
        _skip(f"{workload_label} requires at least {minimum_gpus} GPUs.")
    return total_gpus


def _routing_session_layout(
    *,
    mode: str,
    topology_snapshot: TopologySnapshot,
    cli_args: argparse.Namespace,
) -> Tuple[str, List[_GPUHandle], str, Optional[str], type[_VllmWrapper]]:
    total_gpus = _require_vllm_host(
        workload_label="vLLM routing demo",
        minimum_gpus=2,
    )
    model_id = cli_args.model
    if not model_id:
        _skip("Pass --model <local HF path/id> to run vLLM demo.")
    _assert_vllm_runtime_ready()
    decode_ids = _parse_device_list(cli_args.decode_gpus, "0,1", total_gpus)
    if not decode_ids:
        decode_ids = list(range(min(2, total_gpus)))
    handles = _build_handles(
        "shared",
        decode_ids,
        decode_ids,
        gpu_numa=topology_snapshot.gpu_numa,
    )
    return (
        mode,
        handles,
        model_id,
        getattr(cli_args, "attention_backend", None),
        _VllmWrapper,
    )


def _dual_pool_session_layout(
    *,
    mode: str,
    topology_snapshot: TopologySnapshot,
    cli_args: argparse.Namespace,
) -> Tuple[str, List[_GPUHandle], str, Optional[str], type[_VllmWrapper]]:
    total_gpus = _require_vllm_host(
        workload_label="Dual-pool demo",
        minimum_gpus=2,
    )
    model_id = cli_args.model
    if not model_id:
        _skip("Pass --model <local HF path/id> to run vLLM dual-pool demo.")
    _assert_vllm_runtime_ready()

    normalized_mode = mode.lower()
    if normalized_mode in {"dual", "dual_pool", "optimized"}:
        normalized_mode = "dual"
    else:
        normalized_mode = "shared"

    prefill_ids = _parse_device_list(cli_args.prefill_gpus, "0", total_gpus)
    decode_default = "1" if total_gpus > 1 else "0"
    decode_ids = _parse_device_list(cli_args.decode_gpus, decode_default, total_gpus)
    if not prefill_ids:
        prefill_ids = [0]
    if not decode_ids:
        decode_ids = [1] if total_gpus > 1 else [0]
    if normalized_mode == "dual":
        if not prefill_ids:
            _skip("Dual mode needs at least one prefill GPU.")
        if not decode_ids:
            _skip("Dual mode needs at least one decode GPU.")
        if not (set(prefill_ids) - set(decode_ids)) or not (
            set(decode_ids) - set(prefill_ids)
        ):
            _skip(
                "Dual mode needs at least one GPU dedicated to prefill and one to decode. "
                "Adjust VLLM_PREFILL_GPUS/VLLM_DECODE_GPUS."
            )

    handles = _build_handles(
        normalized_mode,
        prefill_ids,
        decode_ids,
        gpu_numa=topology_snapshot.gpu_numa,
    )
    wrapper_cls = (
        _VllmV1Wrapper
        if getattr(cli_args, "use_v1_core_loop", False)
        else _VllmWrapper
    )
    return (
        normalized_mode,
        handles,
        model_id,
        getattr(cli_args, "attention_backend", None),
        wrapper_cls,
    )


def create_vllm_routing_session(
    mode: str,
    *,
    topology_snapshot: TopologySnapshot,
    cli_args: Optional[argparse.Namespace] = None,
    warmup_runs: int = 0,
) -> VllmEngineSession:
    """Construct routing engines once so request processing can be timed separately."""
    args = cli_args or _CLI_ARGS
    normalized_mode, handles, model_id, attention_backend, wrapper_cls = (
        _routing_session_layout(
            mode=mode,
            topology_snapshot=topology_snapshot,
            cli_args=args,
        )
    )
    return VllmEngineSession(
        workload_kind="dynamic_router",
        mode=normalized_mode,
        handles=handles,
        model_id=model_id,
        attention_backend=attention_backend,
        wrapper_cls=wrapper_cls,
        warmup_runs=warmup_runs,
    )


def create_dual_pool_vllm_session(
    mode: str,
    *,
    topology_snapshot: TopologySnapshot,
    cli_args: Optional[argparse.Namespace] = None,
    warmup_runs: int = 0,
) -> VllmEngineSession:
    """Construct shared or dual-pool engines once for steady-state replay."""
    args = cli_args or _CLI_ARGS
    normalized_mode, handles, model_id, attention_backend, wrapper_cls = (
        _dual_pool_session_layout(
            mode=mode,
            topology_snapshot=topology_snapshot,
            cli_args=args,
        )
    )
    return VllmEngineSession(
        workload_kind="dual_pool",
        mode=normalized_mode,
        handles=handles,
        model_id=model_id,
        attention_backend=attention_backend,
        wrapper_cls=wrapper_cls,
        warmup_runs=warmup_runs,
    )


def _manage_engine_session(session_factory):
    """Close call-owned sessions on success or failure without masking failures."""

    def decorate(run_fn):
        @wraps(run_fn)
        def managed(mode, *args, **kwargs):
            session = kwargs.get("engine_session")
            owns_session = session is None
            if owns_session:
                topology_snapshot = kwargs.get("topology_snapshot")
                if topology_snapshot is None:
                    raise TypeError("topology_snapshot must be passed by keyword")
                session = session_factory(
                    mode,
                    topology_snapshot=topology_snapshot,
                    cli_args=kwargs.get("cli_args"),
                    warmup_runs=0,
                )
                kwargs["engine_session"] = session
            try:
                summary = run_fn(mode, *args, **kwargs)
            except BaseException as exc:
                session.abort_run(exc, retain_inactive_failure=owns_session)
                if owns_session:
                    session.close(preserve_primary_error=True)
                raise
            if owns_session:
                session.close()
                if isinstance(summary, dict):
                    summary.update(session.lifecycle_metrics())
            return summary

        return managed

    return decorate


def _collect_verification_output_token_ids(
    engines: Dict[str, _VllmWrapper], request_ids: List[str]
) -> List[int]:
    """Frame completed model token ids in workload order for exact pair verification."""
    completed: Dict[str, Tuple[int, ...]] = {}
    for engine in engines.values():
        for request_id, token_ids in engine._completed_output_token_ids.items():
            if request_id in completed:
                raise RuntimeError(f"Duplicate completed output for request {request_id}")
            completed[request_id] = token_ids

    expected = set(request_ids)
    missing = expected - set(completed)
    unexpected = set(completed) - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if unexpected:
            details.append(f"unexpected={sorted(unexpected)}")
        raise RuntimeError("Incomplete verification output capture: " + ", ".join(details))

    framed: List[int] = []
    for request_id in request_ids:
        token_ids = completed[request_id]
        framed.append(len(token_ids))
        framed.extend(token_ids)
    return framed


@_manage_engine_session(create_vllm_routing_session)
def run_vllm_routing_with_topology(
    mode: str,
    *,
    topology_snapshot: TopologySnapshot,
    req_count: Optional[int] = None,
    max_tokens: Optional[int] = None,
    cli_args: Optional[argparse.Namespace] = None,
    prompt_token_ids: torch.Tensor,
    engine_session: Optional[VllmEngineSession] = None,
) -> Dict[str, float]:
    """Run a small vLLM-backed routing demo with a precomputed topology snapshot."""
    args = cli_args or _CLI_ARGS
    normalized_mode, handles, model_id, attention_backend, wrapper_cls = (
        _routing_session_layout(
            mode=mode,
            topology_snapshot=topology_snapshot,
            cli_args=args,
        )
    )
    if engine_session is None:
        raise RuntimeError("managed vLLM routing call did not receive an engine session")
    session = engine_session
    session.validate_layout(
        workload_kind="dynamic_router",
        mode=normalized_mode,
        handles=handles,
        model_id=model_id,
        attention_backend=attention_backend,
    )
    run_phase, request_prefix = session.begin_run()
    engines = session.engines

    prompt_lengths = routing_prompt_lengths(args, req_count=req_count)
    req_count_val = len(prompt_lengths)
    max_tokens_val = args.max_tokens if max_tokens is None else max_tokens
    if max_tokens_val <= 0:
        raise ValueError("max_tokens must be positive")
    request_prompt_token_ids = _split_prompt_token_ids(prompt_token_ids, prompt_lengths)

    topo = topology_snapshot
    gpu_numa = topo.gpu_numa

    # Router selection
    router = Router() if mode == "optimized" else None
    if router:
        for gid in engines:
            router.register_gpu(
                gid,
                is_prefill=True,
                is_decode=True,
                numa_node=gpu_numa.get(int(gid.replace("gpu", ""))),
            )

    ttft_samples: List[float] = []
    ttft_total_ms = 0.0
    completed = 0
    telemetry = {gid: _RoutingTelemetry() for gid in engines}
    engine_ids = tuple(engines)
    request_ids: List[str] = []

    # Submit all requests up front
    for i in range(req_count_val):
        rid = f"{request_prefix}req-{i}"
        request_ids.append(rid)
        req = Request(
            req_id=rid,
            prompt_tokens=prompt_lengths[i],
            expected_new_tokens=max_tokens_val,
        )
        admitted = time.time()
        if router:
            # Round-trip through Router for placement
            gid = router.choose_prefill_gpu() or "gpu0"
        else:
            gid = engine_ids[i % len(engine_ids)]
        rt = _RequestRuntime(req=req, gpu_id=gid, admitted_at=admitted)
        engines[gid].add_request(rt, request_prompt_token_ids[i])

    active = True
    while active:
        active = False
        for gid, eng in engines.items():
            finished_ids, ttft_new, tokens = eng.step()
            if finished_ids or eng.queue_depth() > 0:
                active = True
            completed += len(finished_ids)
            if ttft_new:
                for _, sample in ttft_new:
                    ttft_samples.append(sample)
                    ttft_total_ms += sample
            telemetry[gid].observe(ttft_new, tokens)
            # Push metrics into router
            if router:
                router.update_metrics(gid, eng.snapshot_metrics(**telemetry[gid].snapshot_args()))
        time.sleep(0.01)
        if completed >= req_count_val:
            break

    summary: Dict[str, float] = {
        "mode": mode,
        "requests": req_count_val,
        "completed": completed,
        "ttft_ms_mean": float(ttft_total_ms / len(ttft_samples)) if ttft_samples else 0.0,
    }
    summary["ttft_ms_p50"], summary["ttft_ms_p95"] = _percentiles(ttft_samples, (50.0, 95.0))
    for gid in engines:
        summary[f"tpot_tok_per_step_{gid}"] = telemetry[gid].tokens_per_step.get()
    summary[VERIFICATION_OUTPUT_KEY] = _collect_verification_output_token_ids(
        engines, request_ids
    )
    session.finish_run(run_phase)
    summary.update(session.lifecycle_metrics())
    return summary


def run_vllm_routing(
    mode: str,
    req_count: Optional[int] = None,
    max_tokens: Optional[int] = None,
    cli_args: Optional[argparse.Namespace] = None,
    topology_snapshot: Optional[TopologySnapshot] = None,
    prompt_token_ids: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    topo = topology_snapshot or detect_topology(max_gpus=torch.cuda.device_count())
    if prompt_token_ids is None:
        prompt_token_ids = build_prompt_token_ids(routing_prompt_lengths(cli_args or _CLI_ARGS, req_count=req_count))
    return run_vllm_routing_with_topology(
        mode,
        topology_snapshot=topo,
        req_count=req_count,
        max_tokens=max_tokens,
        cli_args=cli_args,
        prompt_token_ids=prompt_token_ids,
    )


@_manage_engine_session(create_dual_pool_vllm_session)
def run_dual_pool_vllm_with_topology(
    mode: str,
    *,
    topology_snapshot: TopologySnapshot,
    long_prompt_tokens: Optional[int] = None,
    short_prompt_tokens: Optional[int] = None,
    prefill_burst: Optional[int] = None,
    decode_requests: Optional[int] = None,
    continue_requests: Optional[int] = None,
    max_tokens: Optional[int] = None,
    prefill_ctx_thresh: Optional[int] = None,
    cli_args: Optional[argparse.Namespace] = None,
    prompt_token_ids: torch.Tensor,
    engine_session: Optional[VllmEngineSession] = None,
) -> Dict[str, float]:
    """
    Dual-pool vLLM experiment: compare shared-pool vs disaggregated prefill/decode.
    """
    args = cli_args or _CLI_ARGS
    normalized_mode, handles, model_id, attention_backend, wrapper_cls = (
        _dual_pool_session_layout(
            mode=mode,
            topology_snapshot=topology_snapshot,
            cli_args=args,
        )
    )
    total_gpus = torch.cuda.device_count()

    long_prompt_tokens = args.long_prompt_tokens if long_prompt_tokens is None else long_prompt_tokens
    short_prompt_tokens = args.short_prompt_tokens if short_prompt_tokens is None else short_prompt_tokens
    prefill_burst = args.prefill_burst if prefill_burst is None else prefill_burst
    decode_requests = args.decode_requests if decode_requests is None else decode_requests
    continue_requests = args.continue_requests if continue_requests is None else continue_requests
    max_tokens = args.max_tokens if max_tokens is None else max_tokens
    prefill_ctx_thresh = args.prefill_ctx_thresh if prefill_ctx_thresh is None else prefill_ctx_thresh
    max_tokens_val = max_tokens
    if max_tokens_val <= 0:
        raise ValueError("max_tokens must be positive")
    prompt_lengths = dual_pool_prompt_lengths(
        args,
        long_prompt_tokens=long_prompt_tokens,
        short_prompt_tokens=short_prompt_tokens,
        prefill_burst=prefill_burst,
        decode_requests=decode_requests,
        continue_requests=continue_requests,
    )
    request_prompt_token_ids = _split_prompt_token_ids(prompt_token_ids, prompt_lengths)

    prefill_ids = _parse_device_list(args.prefill_gpus, "0", total_gpus)
    decode_default = "1" if total_gpus > 1 else "0"
    decode_ids = _parse_device_list(args.decode_gpus, decode_default, total_gpus)

    if not prefill_ids:
        prefill_ids = [0]
    if not decode_ids:
        decode_ids = [1] if total_gpus > 1 else [0]

    prefill_handles = [h for h in handles if h.is_prefill]
    decode_handles = [h for h in handles if h.is_decode]
    if not prefill_handles or not decode_handles:
        _skip("No usable GPUs after parsing pool assignments.")
    if engine_session is None:
        raise RuntimeError("managed dual-pool call did not receive an engine session")
    session = engine_session
    session.validate_layout(
        workload_kind="dual_pool",
        mode=normalized_mode,
        handles=handles,
        model_id=model_id,
        attention_backend=attention_backend,
    )
    run_phase, request_prefix = session.begin_run()
    engines = session.engines

    router = Router()
    for h in handles:
        router.register_gpu(
            h.gpu_id,
            is_prefill=h.is_prefill,
            is_decode=h.is_decode,
            numa_node=h.numa_node,
        )

    workload: List[Tuple[Request, str]] = []
    next_id = 0

    def _enqueue(n: int, prompt_tokens: int, hint: str) -> None:
        nonlocal next_id
        for _ in range(n):
            rid = f"{request_prefix}req-{next_id}"
            next_id += 1
            workload.append(
                (
                    Request(
                        req_id=rid,
                        prompt_tokens=prompt_tokens,
                        expected_new_tokens=max_tokens_val,
                        priority=0,
                    ),
                    hint,
                )
            )

    _enqueue(prefill_burst, long_prompt_tokens, "prefill")
    _enqueue(decode_requests, short_prompt_tokens, "decode")
    _enqueue(continue_requests, short_prompt_tokens, "decode")

    prefill_pool_ids = [h.gpu_id for h in prefill_handles]
    decode_pool_ids = [h.gpu_id for h in decode_handles]
    decode_numa_hint = decode_handles[0].numa_node if decode_handles else None

    requests: Dict[str, _RequestRuntime] = {}
    req_roles: Dict[str, str] = {}
    if len(workload) != len(request_prompt_token_ids):
        raise RuntimeError("prompt input request count does not match the routed workload")
    for request_index, (req, hint) in enumerate(workload):
        route = "prefill" if hint == "prefill" or req.prompt_tokens >= prefill_ctx_thresh else "decode"
        if route == "prefill":
            target = router.choose_prefill_gpu() or (prefill_pool_ids[0] if prefill_pool_ids else None)
        else:
            seq = SequenceInfo(
                seq_id=req.req_id,
                current_gpu="",
                kv_gpus=set(),
                expected_tokens_remaining=req.expected_new_tokens,
                priority=req.priority,
                numa_node=decode_numa_hint,
            )
            target = router.choose_decode_gpu(seq)
            if target is None:
                if decode_pool_ids:
                    target = decode_pool_ids[0]
                elif prefill_pool_ids:
                    target = prefill_pool_ids[0]
        if target is None:
            _skip("No GPU available for routed request.")
        rt = _RequestRuntime(req=req, gpu_id=target, admitted_at=time.time(), role=route)
        engines[target].add_request(rt, request_prompt_token_ids[request_index])
        requests[req.req_id] = rt
        req_roles[req.req_id] = route

    ttft_samples: List[float] = []
    pool_ttft: Dict[str, List[float]] = {"prefill": [], "decode": []}
    queue_depth_totals: Dict[str, float] = {"prefill": 0.0, "decode": 0.0}
    queue_depth_counts: Dict[str, int] = {"prefill": 0, "decode": 0}
    completed: Set[str] = set()
    telemetry = {h.gpu_id: _RoutingTelemetry() for h in handles}

    active = True
    while active:
        active = False
        for handle in handles:
            eng = engines[handle.gpu_id]
            finished_ids, ttft_new, tokens = eng.step()
            if finished_ids or eng.queue_depth() > 0:
                active = True
            for rid, ttft_ms in ttft_new:
                ttft_samples.append(ttft_ms)
                role = req_roles.get(rid, "shared")
                if role in pool_ttft:
                    pool_ttft[role].append(ttft_ms)
            telemetry[handle.gpu_id].observe(ttft_new, tokens)
            router.update_metrics(
                handle.gpu_id,
                eng.snapshot_metrics(**telemetry[handle.gpu_id].snapshot_args()),
            )
            qd = eng.queue_depth()
            if handle.is_prefill:
                queue_depth_totals["prefill"] += float(qd)
                queue_depth_counts["prefill"] += 1
            if handle.is_decode:
                queue_depth_totals["decode"] += float(qd)
                queue_depth_counts["decode"] += 1
            for rid in finished_ids:
                completed.add(rid)
        time.sleep(0.01)
        if len(completed) >= len(req_roles):
            break

    ttft_p50, ttft_p95 = _percentiles(ttft_samples, (50.0, 95.0))
    prefill_ttft_p50, prefill_ttft_p95 = _percentiles(pool_ttft["prefill"], (50.0, 95.0))
    decode_ttft_p50, decode_ttft_p95 = _percentiles(pool_ttft["decode"], (50.0, 95.0))

    summary: Dict[str, float] = {
        "mode": normalized_mode,
        "requests": len(req_roles),
        "completed": len(completed),
        "prefill_gpu_count": len(prefill_ids),
        "decode_gpu_count": len(decode_ids),
        "ttft_ms_p50": ttft_p50,
        "ttft_ms_p95": ttft_p95,
        "prefill_ttft_ms_p50": prefill_ttft_p50,
        "prefill_ttft_ms_p95": prefill_ttft_p95,
        "decode_ttft_ms_p50": decode_ttft_p50,
        "decode_ttft_ms_p95": decode_ttft_p95,
        "queue_depth_prefill_mean": (
            queue_depth_totals["prefill"] / queue_depth_counts["prefill"]
            if queue_depth_counts["prefill"]
            else 0.0
        ),
        "queue_depth_decode_mean": (
            queue_depth_totals["decode"] / queue_depth_counts["decode"]
            if queue_depth_counts["decode"]
            else 0.0
        ),
        "long_prompt_tokens": float(long_prompt_tokens),
        "short_prompt_tokens": float(short_prompt_tokens),
        "prefill_burst": float(prefill_burst),
        "decode_requests": float(decode_requests),
        "continue_requests": float(continue_requests),
        "prefill_ctx_thresh": float(prefill_ctx_thresh),
        "max_tokens": float(max_tokens_val),
    }
    for gid in engines:
        summary[f"tpot_tok_per_step_{gid}"] = telemetry[gid].tokens_per_step.get()
    summary[VERIFICATION_OUTPUT_KEY] = _collect_verification_output_token_ids(
        engines, list(req_roles)
    )
    session.finish_run(run_phase)
    summary.update(session.lifecycle_metrics())
    return summary


def run_dual_pool_vllm(
    mode: str,
    long_prompt_tokens: Optional[int] = None,
    short_prompt_tokens: Optional[int] = None,
    prefill_burst: Optional[int] = None,
    decode_requests: Optional[int] = None,
    continue_requests: Optional[int] = None,
    max_tokens: Optional[int] = None,
    prefill_ctx_thresh: Optional[int] = None,
    cli_args: Optional[argparse.Namespace] = None,
    topology_snapshot: Optional[TopologySnapshot] = None,
    prompt_token_ids: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    topo = topology_snapshot or detect_topology(max_gpus=torch.cuda.device_count())
    if prompt_token_ids is None:
        prompt_token_ids = build_prompt_token_ids(dual_pool_prompt_lengths(
            cli_args or _CLI_ARGS, long_prompt_tokens=long_prompt_tokens,
            short_prompt_tokens=short_prompt_tokens, prefill_burst=prefill_burst,
            decode_requests=decode_requests, continue_requests=continue_requests,
        ))
    return run_dual_pool_vllm_with_topology(
        mode,
        topology_snapshot=topo,
        long_prompt_tokens=long_prompt_tokens,
        short_prompt_tokens=short_prompt_tokens,
        prefill_burst=prefill_burst,
        decode_requests=decode_requests,
        continue_requests=continue_requests,
        max_tokens=max_tokens,
        prefill_ctx_thresh=prefill_ctx_thresh,
        cli_args=cli_args,
        prompt_token_ids=prompt_token_ids,
    )
