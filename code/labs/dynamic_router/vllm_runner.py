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
from typing import Dict, List, Optional, Set, Tuple

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


def _build_vllm_engine(engine_cls, model_id: str, device_index: int):
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

    def __init__(self, gpu_id: str, device_index: int, model_id: str) -> None:
        _assert_vllm_runtime_ready()
        if EngineArgs is None or LLMEngine is None or SamplingParams is None:
            _skip(_format_vllm_import_error(_IMPORT_ERROR or RuntimeError("unknown vLLM import failure")))

        self.gpu_id = gpu_id
        self.device_index = device_index
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.engine = _build_vllm_engine(LLMEngine, model_id, device_index)
        captured = buf.getvalue().strip()
        if captured:
            try:
                lines = [ln for ln in captured.splitlines() if ln]
                print(json.dumps({"event": "vllm_engine_init_stdout", "gpu": gpu_id, "lines": lines}), file=sys.stderr)
            except Exception:
                print(captured, file=sys.stderr)
        self._inflight: Dict[str, _RequestRuntime] = {}
        self._completed_output_token_ids: Dict[str, Tuple[int, ...]] = {}

    def add_request(self, rt: _RequestRuntime) -> None:
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
        prompt_token_ids = [1] * rt.req.prompt_tokens
        self.engine.add_request(
            request_id=rt.req.req_id,
            prompt=prompt_token_ids,
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

    def __init__(self, gpu_id: str, device_index: int, model_id: str) -> None:
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
            self.engine = _build_vllm_engine(V1LLMEngine, model_id, device_index)
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


def run_vllm_routing_with_topology(
    mode: str,
    *,
    topology_snapshot: TopologySnapshot,
    req_count: Optional[int] = None,
    max_tokens: Optional[int] = None,
    cli_args: Optional[argparse.Namespace] = None,
) -> Dict[str, float]:
    """Run a small vLLM-backed routing demo with a precomputed topology snapshot."""
    if not torch.cuda.is_available():
        _skip("CUDA is required for vLLM routing demo.")
    if torch.cuda.device_count() < 2:
        _skip("vLLM routing demo requires at least 2 GPUs.")

    args = cli_args or _CLI_ARGS
    model_id = args.model
    if not model_id:
        _skip("Pass --model <local HF path/id> to run vLLM demo.")
    _assert_vllm_runtime_ready()

    req_count_val = req_count or args.req_count
    max_tokens_val = max_tokens or args.max_tokens

    topo = topology_snapshot
    gpu_numa = topo.gpu_numa

    decode_ids = _parse_device_list(args.decode_gpus, "0,1", torch.cuda.device_count())
    if not decode_ids:
        decode_ids = list(range(min(2, torch.cuda.device_count())))
    engines = {f"gpu{idx}": _VllmWrapper(f"gpu{idx}", idx, model_id) for idx in decode_ids}

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
        rid = f"req-{i}"
        request_ids.append(rid)
        req = Request(req_id=rid, prompt_tokens=64, expected_new_tokens=max_tokens_val)
        admitted = time.time()
        if router:
            # Round-trip through Router for placement
            gid = router.choose_prefill_gpu() or "gpu0"
        else:
            gid = engine_ids[i % len(engine_ids)]
        rt = _RequestRuntime(req=req, gpu_id=gid, admitted_at=admitted)
        engines[gid].add_request(rt)

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
    return summary


def run_vllm_routing(
    mode: str,
    req_count: Optional[int] = None,
    max_tokens: Optional[int] = None,
    cli_args: Optional[argparse.Namespace] = None,
    topology_snapshot: Optional[TopologySnapshot] = None,
) -> Dict[str, float]:
    topo = topology_snapshot or detect_topology(max_gpus=torch.cuda.device_count())
    return run_vllm_routing_with_topology(
        mode,
        topology_snapshot=topo,
        req_count=req_count,
        max_tokens=max_tokens,
        cli_args=cli_args,
    )


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
) -> Dict[str, float]:
    """
    Dual-pool vLLM experiment: compare shared-pool vs disaggregated prefill/decode.
    """
    if not torch.cuda.is_available():
        _skip("CUDA is required for vLLM dual-pool demo.")

    total_gpus = torch.cuda.device_count()
    if total_gpus < 2:
        _skip("Dual-pool demo requires at least 2 GPUs.")

    args = cli_args or _CLI_ARGS
    model_id = args.model
    if not model_id:
        _skip("Pass --model <local HF path/id> to run vLLM dual-pool demo.")
    _assert_vllm_runtime_ready()

    normalized_mode = mode.lower()
    if normalized_mode in {"dual", "dual_pool", "optimized"}:
        normalized_mode = "dual"
    else:
        normalized_mode = "shared"

    long_prompt_tokens = long_prompt_tokens or args.long_prompt_tokens
    short_prompt_tokens = short_prompt_tokens or args.short_prompt_tokens
    prefill_burst = prefill_burst or args.prefill_burst
    decode_requests = decode_requests or args.decode_requests
    continue_requests = continue_requests or args.continue_requests
    max_tokens = max_tokens or args.max_tokens
    prefill_ctx_thresh = prefill_ctx_thresh or args.prefill_ctx_thresh
    max_tokens_val = max(1, max_tokens)

    prefill_ids = _parse_device_list(args.prefill_gpus, "0", total_gpus)
    decode_default = "1" if total_gpus > 1 else "0"
    decode_ids = _parse_device_list(args.decode_gpus, decode_default, total_gpus)

    if not prefill_ids:
        prefill_ids = [0]
    if not decode_ids:
        decode_ids = [1] if total_gpus > 1 else [0]

    if normalized_mode == "dual":
        if not set(prefill_ids):
            _skip("Dual mode needs at least one prefill GPU.")
        if not set(decode_ids):
            _skip("Dual mode needs at least one decode GPU.")
        if not (set(prefill_ids) - set(decode_ids)) or not (set(decode_ids) - set(prefill_ids)):
            _skip("Dual mode needs at least one GPU dedicated to prefill and one to decode. Adjust VLLM_PREFILL_GPUS/VLLM_DECODE_GPUS.")

    topo = topology_snapshot
    handles = _build_handles(normalized_mode, prefill_ids, decode_ids, gpu_numa=topo.gpu_numa)
    prefill_handles = [h for h in handles if h.is_prefill]
    decode_handles = [h for h in handles if h.is_decode]
    if not prefill_handles or not decode_handles:
        _skip("No usable GPUs after parsing pool assignments.")

    wrapper_cls = _VllmV1Wrapper if getattr(args, "use_v1_core_loop", False) else _VllmWrapper
    engines = {h.gpu_id: wrapper_cls(h.gpu_id, h.device_index, model_id) for h in handles}

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
            rid = f"req-{next_id}"
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
    for req, hint in workload:
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
        engines[target].add_request(rt)
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
) -> Dict[str, float]:
    topo = topology_snapshot or detect_topology(max_gpus=torch.cuda.device_count())
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
    )
