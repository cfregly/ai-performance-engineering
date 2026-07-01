"""Optimized MoE inference benchmark inspired by vLLM + NVIDIA Dynamo + SGLang."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

_RESET_PEAK_MEMORY_STATS = getattr(torch.cuda, "reset_peak_memory_stats", None)

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.profiling.gpu_memory_logger import (  # noqa: E402
    GpuMemoryLogger,
    resolve_gpu_log_interval,
    resolve_gpu_log_path,
)
from core.profiling.gpu_telemetry import query_gpu_telemetry  # noqa: E402
from core.optimization.moe_inference import (  # noqa: E402
    MoEFeedForward,
    MoEFeedForwardSortedDispatch,
    MoeInferenceConfig,
    SimpleMoEGPT,
    dtype_bytes,
    env_override_float,
    env_override_int,
)
from ch17.dynamic_routing import (  # noqa: E402
    DisaggregatedRouter,
    Priority,
    Request,
    WorkerMetrics,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin


DEFAULT_SPEC_CONFIG = Path(__file__).parent / "spec_configs" / "draft_and_verify.json"


class GraphMode(Enum):
    EAGER = "eager"
    FULL = "full"
    PIECEWISE = "piecewise"
    FULL_AND_PIECEWISE = "full_and_piecewise"

    @classmethod
    def from_str(cls, raw: Optional[str]) -> "GraphMode":
        normalized = (raw or cls.FULL_AND_PIECEWISE.value).strip().lower().replace("-", "_")
        for mode in cls:
            if normalized == mode.value:
                return mode
        return cls.FULL_AND_PIECEWISE


class SpeculatorConfig:
    """Speculators-style configuration for speculative decoding."""

    def __init__(
        self,
        method: str = "draft_model",
        draft_model: str = "SimpleMoEGPT-draft",
        verifier_model: str = "SimpleMoEGPT-target",
        chunk_size: int = 4,
        max_spec_tokens: int = 4,
        acceptance_target: float = 0.6,
        fallback_chunk_size: Optional[int] = None,
    ) -> None:
        self.method = method
        self.draft_model = draft_model
        self.verifier_model = verifier_model
        self.chunk_size = chunk_size
        self.max_spec_tokens = max_spec_tokens
        self.acceptance_target = acceptance_target
        self.fallback_chunk_size = fallback_chunk_size

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeculatorConfig":
        return cls(
            method=str(data.get("method", "draft_model")),
            draft_model=str(data.get("draft_model", "SimpleMoEGPT-draft")),
            verifier_model=str(data.get("verifier_model", "SimpleMoEGPT-target")),
            chunk_size=max(1, int(data.get("chunk_size", 4))),
            max_spec_tokens=max(1, int(data.get("max_spec_tokens", 4))),
            acceptance_target=float(data.get("acceptance_target", 0.6)),
            fallback_chunk_size=(
                max(1, int(data["fallback_chunk_size"])) if data.get("fallback_chunk_size") is not None else None
            ),
        )

    @classmethod
    def load(cls, path: Optional[Path]) -> "SpeculatorConfig":
        if path is None or not path.exists():
            return cls()
        payload: Any
        if path.suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise ImportError("pyyaml is required to load YAML speculator configs")
            payload = yaml.safe_load(path.read_text())
        else:
            payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Speculator config must be a mapping, got {type(payload)}")
        return cls.from_dict(payload)

    def summary(self) -> str:
        return (
            f"method={self.method}, chunk={self.chunk_size}, "
            f"fallback={self.fallback_chunk_size}, target_accept={self.acceptance_target}"
        )


class PagedKVCache:
    """Lightweight paged KV cache for benchmarking."""

    def __init__(
        self,
        *,
        batch_size: int,
        max_tokens: int,
        hidden: int,
        dtype: torch.dtype,
        device: torch.device,
        page_size: int,
    ) -> None:
        self.buffer = torch.empty(batch_size, max_tokens, hidden, dtype=dtype, device=device)
        self.page_size = max(1, page_size)
        self.max_tokens = max_tokens
        self.tokens_written = 0
        self.page_faults = 0

    def reset(self) -> None:
        self.tokens_written = 0
        self.page_faults = 0

    def mark_prefill(self, tokens: int) -> None:
        self._update_usage(position=0, length=tokens)

    def write(self, position: int, values: torch.Tensor) -> None:
        length = values.size(1)
        self.buffer[:, position:position + length].copy_(values)
        self._update_usage(position=position, length=length)

    def _update_usage(self, position: int, length: int) -> None:
        prev_tokens = self.tokens_written
        self.tokens_written = max(self.tokens_written, position + length)
        prev_pages = math.ceil(prev_tokens / self.page_size)
        new_pages = math.ceil(self.tokens_written / self.page_size)
        if new_pages > prev_pages:
            self.page_faults += (new_pages - prev_pages)

    @property
    def occupancy_ratio(self) -> float:
        if self.max_tokens <= 0:
            return 0.0
        return min(1.0, self.tokens_written / self.max_tokens)

    @property
    def memory_gb(self) -> float:
        return self.buffer.element_size() * self.buffer.nelement() / (1024 ** 3)


class SpeculativeDecoder:
    """SGLang-style speculative decode helper using draft and target models."""

    def __init__(self, target_model: SimpleMoEGPT, draft_model: SimpleMoEGPT, config: SpeculatorConfig):
        self.target_model = target_model
        self.draft_model = draft_model
        self.config = config
        self._base_chunk = max(1, config.chunk_size)
        self.chunk_size = self._base_chunk
        self._fallback_chunk = max(1, config.fallback_chunk_size) if config.fallback_chunk_size else None
        self.accepted_tokens = 0
        self.total_tokens = 0
        self._match_summary_workspace: Optional[torch.Tensor] = None
        self._draft_next_values: Optional[torch.Tensor] = None
        self._draft_next_tokens: Optional[torch.Tensor] = None
        self._target_next_values: Optional[torch.Tensor] = None
        self._target_next_tokens: Optional[torch.Tensor] = None
        self._matches_workspace: Optional[torch.Tensor] = None
        self._selected_tokens: Optional[torch.Tensor] = None
        self._per_token_times: List[float] = []

    def reset(self) -> None:
        self.accepted_tokens = 0
        self.total_tokens = 0
        self.chunk_size = self._base_chunk

    def _match_count_workspace(self, device: torch.device) -> torch.Tensor:
        if (
            self._match_summary_workspace is None
            or self._match_summary_workspace.device != device
        ):
            self._match_summary_workspace = torch.empty((), dtype=torch.long, device=device)
        return self._match_summary_workspace

    @staticmethod
    def _next_token_from_logits(
        logits: torch.Tensor,
        values: Optional[torch.Tensor],
        token_ids: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        last_logits = logits if logits.dim() == 2 else logits[:, -1, :]
        shape = (last_logits.shape[0], 1)
        if (
            values is None
            or token_ids is None
            or values.device != last_logits.device
            or token_ids.device != last_logits.device
            or values.dtype != last_logits.dtype
            or tuple(values.shape) != shape
            or tuple(token_ids.shape) != shape
        ):
            values = torch.empty(shape, dtype=last_logits.dtype, device=last_logits.device)
            token_ids = torch.empty(shape, dtype=torch.long, device=last_logits.device)
        torch.max(last_logits, dim=-1, keepdim=True, out=(values, token_ids))
        return values, token_ids

    def _selection_workspace(self, reference: torch.Tensor) -> torch.Tensor:
        if (
            self._selected_tokens is None
            or self._selected_tokens.device != reference.device
            or self._selected_tokens.dtype != reference.dtype
            or self._selected_tokens.dim() != reference.dim()
            or self._selected_tokens.size(0) < reference.size(0)
            or tuple(self._selected_tokens.shape[1:]) != tuple(reference.shape[1:])
        ):
            self._selected_tokens = torch.empty_like(reference)
        return self._selected_tokens[: reference.size(0)]

    def _matches_buffer(self, reference: torch.Tensor) -> torch.Tensor:
        if (
            self._matches_workspace is None
            or self._matches_workspace.device != reference.device
            or self._matches_workspace.dim() != reference.dim()
            or self._matches_workspace.size(0) < reference.size(0)
            or tuple(self._matches_workspace.shape[1:]) != tuple(reference.shape[1:])
        ):
            self._matches_workspace = torch.empty(reference.shape, dtype=torch.bool, device=reference.device)
        return self._matches_workspace[: reference.size(0)]

    def prepare_workspaces(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> None:
        shape = (batch_size, 1)
        self._draft_next_values = torch.empty(shape, dtype=dtype, device=device)
        self._draft_next_tokens = torch.empty(shape, dtype=torch.long, device=device)
        self._target_next_values = torch.empty(shape, dtype=dtype, device=device)
        self._target_next_tokens = torch.empty(shape, dtype=torch.long, device=device)
        self._matches_workspace = torch.empty(shape, dtype=torch.bool, device=device)
        self._selected_tokens = torch.empty(shape, dtype=torch.long, device=device)
        self._match_summary_workspace = torch.empty((), dtype=torch.long, device=device)

    def decode(
        self,
        seed_tokens: torch.Tensor,
        total_tokens: int,
        paged_cache: PagedKVCache,
        base_position: int,
    ) -> Tuple[torch.Tensor, List[float], int, float]:
        tokens = seed_tokens
        emitted = 0
        decode_total_ms = 0.0
        if len(self._per_token_times) < total_tokens:
            self._per_token_times = [0.0] * total_tokens
        per_token_times = self._per_token_times
        match_elements = int(seed_tokens.numel())

        with torch.inference_mode():
            while emitted < total_tokens:
                chunk = min(self.chunk_size, self.config.max_spec_tokens, total_tokens - emitted)
                for _ in range(chunk):
                    start = time.perf_counter()
                    draft_hidden, draft_logits = self.draft_model.decode(tokens)
                    self._draft_next_values, candidate = self._next_token_from_logits(
                        draft_logits,
                        self._draft_next_values,
                        self._draft_next_tokens,
                    )
                    self._draft_next_tokens = candidate

                    target_hidden, target_logits = self.target_model.decode(
                        tokens,
                        kv_cache=paged_cache.buffer,
                        position=base_position + emitted,
                    )
                    paged_cache.write(base_position + emitted, target_hidden)

                    self._target_next_values, target_next = self._next_token_from_logits(
                        target_logits,
                        self._target_next_values,
                        self._target_next_tokens,
                    )
                    self._target_next_tokens = target_next
                    matches = self._matches_buffer(candidate)
                    torch.eq(candidate, target_next, out=matches)
                    match_summary = self._match_count_workspace(matches.device)
                    torch.sum(matches, dim=None, out=match_summary)
                    self.total_tokens += match_elements
                    tokens = self._selection_workspace(candidate)
                    torch.where(matches, candidate, target_next, out=tokens)

                    # This host read is required for control flow; keep it after token selection
                    # so it also accounts for the queued decode work used by the timing sample.
                    match_count = int(match_summary.item())
                    all_matches = match_count == match_elements
                    self.accepted_tokens += int(match_count)
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    per_token_times[emitted] = elapsed_ms
                    decode_total_ms += elapsed_ms
                    emitted += 1

                    if not all_matches:
                        break
        self._maybe_adjust_chunk()
        return tokens, per_token_times, emitted, decode_total_ms

    def _maybe_adjust_chunk(self) -> None:
        if self._fallback_chunk is None:
            return
        if self.acceptance_rate() < self.config.acceptance_target:
            self.chunk_size = self._fallback_chunk

    def acceptance_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_tokens

    def current_chunk_size(self) -> int:
        return self.chunk_size


class VLLMMoEInferenceBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized MoE inference benchmark with paged KV cache + speculative decode."""

    allowed_benchmark_fn_antipatterns = ("host_transfer", "sync")

    def __init__(self) -> None:
        super().__init__()
        self.config = self._build_config()
        self._cuda_available = torch.cuda.is_available()
        self.model: Optional[SimpleMoEGPT] = None
        self.draft_model: Optional[SimpleMoEGPT] = None
        self.prompts: Optional[torch.Tensor] = None
        self.router = DisaggregatedRouter(config_path=os.getenv("DYNAMO_ROUTER_CONFIG"))
        self.paged_cache: Optional[PagedKVCache] = None
        self.spec_decoder: Optional[SpeculativeDecoder] = None
        self.spec_config_path: Optional[Path] = None
        self.spec_config: SpeculatorConfig = SpeculatorConfig()
        self.enable_graphs = os.getenv("OPT_MOE_ENABLE_GRAPHS") == "1"
        self.prefill_workers = env_override_int("OPT_MOE_PREFILL_WORKERS", 2)
        self.decode_workers = env_override_int("OPT_MOE_DECODE_WORKERS", 2)
        self._dtype_bytes = dtype_bytes(self.config.dtype_obj)
        total_tokens = self.config.context_window + self.config.decode_tokens
        self.engine_variant = "v1"
        self.graph_mode = GraphMode.from_str(os.getenv("OPT_MOE_GRAPH_MODE"))
        self.max_capture_tokens = env_override_int("OPT_MOE_MAX_CAPTURE_TOKENS", total_tokens)
        self._full_graph: Optional[torch.cuda.CUDAGraph] = None
        self._full_prefill_done: Optional[torch.cuda.Event] = None
        self._full_decode_done: Optional[torch.cuda.Event] = None
        self._full_replay_start: Optional[torch.cuda.Event] = None
        self._full_replay_end: Optional[torch.cuda.Event] = None
        self._piecewise_prefill_graph: Optional[torch.cuda.CUDAGraph] = None
        self._piecewise_decode_graph: Optional[torch.cuda.CUDAGraph] = None
        self._piecewise_prefill_done: Optional[torch.cuda.Event] = None
        self._piecewise_decode_done: Optional[torch.cuda.Event] = None
        self._piecewise_replay_prefill_start: Optional[torch.cuda.Event] = None
        self._piecewise_replay_decode_start: Optional[torch.cuda.Event] = None
        self._piecewise_replay_end: Optional[torch.cuda.Event] = None
        self._prefill_next_values: Optional[torch.Tensor] = None
        self._prefill_next_tokens: Optional[torch.Tensor] = None
        self._captured_full_spec_accept: Optional[float] = None
        self._captured_full_spec_chunk: Optional[float] = None
        self._captured_piecewise_spec_accept: Optional[float] = None
        self._captured_piecewise_spec_chunk: Optional[float] = None
        self._workload_metadata = WorkloadMetadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=float(self.config.tokens_per_iteration),
        )
        self.output = None
        self._ttft_total_ms: float = 0.0
        self._ttft_count: int = 0
        self._tpot_total_ms: float = 0.0
        self._tpot_count: int = 0
        self._throughput_total: float = 0.0
        self._throughput_count: int = 0
        self._spec_accept_total: float = 0.0
        self._spec_accept_count: int = 0
        self._spec_chunk_total: float = 0.0
        self._spec_chunk_count: int = 0
        self._nvlink_total_gbps: float = 0.0
        self._nvlink_count: int = 0
        self._nvlink_measured_total_gbps: float = 0.0
        self._nvlink_measured_count: int = 0
        self._prefill_share_total: float = 0.0
        self._prefill_share_count: int = 0
        self._paged_hit_total: float = 0.0
        self._paged_hit_count: int = 0
        self._page_faults_total: float = 0.0
        self._page_faults_count: int = 0
        self._memory_total_gb: float = 0.0
        self._memory_count: int = 0
        self._memory_bytes_to_gb = 1.0 / (1024 ** 3)
        self._memory_poll_pending = False
        self._iteration = 0
        self._router_prefix_cache_lengths: List[int] = []
        self._router_prefix_count: int = 0
        self._router_prompt_stub: List[int] = []
        self._router_prompt_len: int = 0
        self._router_requests: List[Request] = []
        self._router_request_count: int = 0
        self._router_batch_range = range(0)
        self._router_devnull = None
        self._mem_logger: Optional[GpuMemoryLogger] = None
        self._mem_log_path: Optional[Path] = None
        self._nvlink_warned: bool = False
        self._nvlink_status: str = "unknown"
        self._iteration_ttft_times: List[float] = [0.0]
        self._iteration_tpot_times: List[float] = [0.0] * self.config.decode_tokens
        self._iteration_metric_payload: Dict[str, object] = {
            "ttft_times_ms": self._iteration_ttft_times,
            "tpot_times_ms": self._iteration_tpot_times,
            "graph_path": "eager",
        }
        self._payload_parameter_count = 0
        self.register_workload_metadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=float(self.config.tokens_per_iteration),
        )

    def _build_config(self) -> MoeInferenceConfig:
        return MoeInferenceConfig(
            vocab_size=env_override_int("OPT_MOE_VOCAB", 16384),
            hidden_size=env_override_int("OPT_MOE_HIDDEN", 1024),
            ffn_size=env_override_int("OPT_MOE_FFN", 4096),
            num_layers=env_override_int("OPT_MOE_LAYERS", 8),
            num_moe_layers=env_override_int("OPT_MOE_MOE_LAYERS", 4),
            num_experts=env_override_int("OPT_MOE_EXPERTS", 32),
            top_k=2,
            moe_layer_frequency=max(1, env_override_int("OPT_MOE_MOE_FREQ", 2)),
            batch_size=env_override_int("OPT_MOE_BATCH", 1),
            context_window=env_override_int("OPT_MOE_CONTEXT", 512),
            decode_tokens=env_override_int("OPT_MOE_DECODE", 32),
            router_noise=env_override_float("OPT_MOE_ROUTER_NOISE", 0.0),
            dtype=torch.bfloat16,
        )

    def _replace_moe_dispatch(self, model: SimpleMoEGPT, cfg: MoeInferenceConfig) -> int:
        converted = 0
        for block in getattr(model, "layers", []):
            ff = getattr(block, "ff", None)
            if not isinstance(ff, MoEFeedForward) or isinstance(ff, MoEFeedForwardSortedDispatch):
                continue
            replacement = MoEFeedForwardSortedDispatch(
                cfg.hidden_size,
                cfg.ffn_size,
                num_experts=cfg.num_experts,
                top_k=cfg.top_k,
                router_noise=cfg.router_noise,
                capacity_factor=cfg.capacity_factor,
                device=self.device,
                dtype=cfg.dtype_obj,
            )
            replacement.load_state_dict(ff.state_dict(), strict=True)
            block.ff = replacement
            converted += 1
        return converted

    # --------------------------------------------------------------------- setup
    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        cfg = self.config
        self.model = SimpleMoEGPT(cfg, device=self.device).eval()
        self._replace_moe_dispatch(self.model, cfg)
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())

        self.prompts = torch.randint(
            0,
            cfg.vocab_size,
            (cfg.batch_size, cfg.context_window),
            device=self.device,
        )

        draft_cfg = MoeInferenceConfig(
            vocab_size=cfg.vocab_size,
            hidden_size=max(512, cfg.hidden_size // 2),
            ffn_size=max(1024, cfg.ffn_size // 2),
            num_layers=max(2, cfg.num_layers // 2),
            num_moe_layers=max(1, cfg.num_moe_layers // 2),
            num_experts=max(4, cfg.num_experts // 2),
            top_k=1,
            moe_layer_frequency=cfg.moe_layer_frequency,
            batch_size=cfg.batch_size,
            context_window=cfg.context_window,
            decode_tokens=cfg.decode_tokens,
            router_noise=cfg.router_noise,
            dtype=cfg.dtype_obj,
        )
        self.draft_model = SimpleMoEGPT(draft_cfg, device=self.device).eval()
        self._replace_moe_dispatch(self.draft_model, draft_cfg)

        config_path: Optional[Path] = None
        if self.spec_config_path and self.spec_config_path.exists():
            config_path = self.spec_config_path
        elif DEFAULT_SPEC_CONFIG.exists():
            config_path = DEFAULT_SPEC_CONFIG
        self.spec_config = SpeculatorConfig.load(config_path)
        self.paged_cache = PagedKVCache(
            batch_size=cfg.batch_size,
            max_tokens=cfg.context_window + cfg.decode_tokens,
            hidden=cfg.hidden_size,
            dtype=cfg.dtype_obj,
            device=self.device,
            page_size=env_override_int("OPT_MOE_PAGE_SIZE", 512),
        )
        self.spec_decoder = SpeculativeDecoder(
            target_model=self.model,
            draft_model=self.draft_model,
            config=self.spec_config,
        )
        prefix_period = max(1, cfg.context_window // 4)
        self._router_prefix_cache_lengths = [idx % prefix_period for idx in range(cfg.batch_size)]
        self._router_prefix_count = len(self._router_prefix_cache_lengths)
        self._router_prompt_stub = [0] * cfg.context_window
        self._router_prompt_len = cfg.context_window
        self._router_requests = [
            Request(
                id=f"req-0-{idx}",
                prompt_tokens=self._router_prompt_stub,
                priority=Priority.STANDARD,
                timestamp=0.0,
                prefix_cached_length=self._router_prefix_cache_lengths[idx],
                expected_output_length=cfg.decode_tokens,
            )
            for idx in range(cfg.batch_size)
        ]
        self._router_request_count = cfg.batch_size
        self._router_batch_range = range(cfg.batch_size)
        self._router_devnull = open(os.devnull, "w")
        # Force eager path so verification can capture decode tokens deterministically.
        self.graph_mode = GraphMode.EAGER
        self._refresh_router_metrics()
        torch.cuda.synchronize(self.device)
        self._memory_poll_pending = False
        if self._cuda_available and _RESET_PEAK_MEMORY_STATS is not None:
            _RESET_PEAK_MEMORY_STATS(self.device)
            log_path = resolve_gpu_log_path(None)
            logger = GpuMemoryLogger(
                device=self.device,
                interval=resolve_gpu_log_interval(1.0),
                log_path=log_path,
            )
            if logger.start():
                self._mem_logger = logger
                self._mem_log_path = log_path
        if self.enable_graphs and self._cuda_available:
            self._prepare_graphs()

    def _refresh_router_metrics(self) -> None:
        timestamp = time.time()
        for idx in range(self.prefill_workers):
            metrics = WorkerMetrics(
                queue_length=max(1, self.config.batch_size // max(1, self.prefill_workers)),
                gpu_utilization=45.0 + idx * 3.0,
                memory_usage=48.0 + idx * 2.5,
                kv_cache_usage=50.0 + idx * 4.0,
                active_requests=max(1, self.config.batch_size // max(1, self.prefill_workers)),
                last_updated=timestamp,
            )
            self.router.update_worker_metrics("prefill", f"prefill-{idx}", metrics)

        for idx in range(self.decode_workers):
            metrics = WorkerMetrics(
                queue_length=max(1, self.config.batch_size // max(1, self.decode_workers)),
                gpu_utilization=55.0 + idx * 4.0,
                memory_usage=58.0 + idx * 3.0,
                kv_cache_usage=62.0 + idx * 3.5,
                active_requests=max(1, self.config.batch_size // max(1, self.decode_workers)),
                last_updated=timestamp,
            )
            self.router.update_worker_metrics("decode", f"decode-{idx}", metrics)

    def _prefill_next_token_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        last_logits = logits if logits.dim() == 2 else logits[:, -1, :]
        shape = (last_logits.shape[0], 1)
        if (
            self._prefill_next_values is None
            or self._prefill_next_values.device != last_logits.device
            or self._prefill_next_values.dtype != last_logits.dtype
            or self._prefill_next_values.size(0) < shape[0]
            or self._prefill_next_values.size(1) != shape[1]
        ):
            self._prefill_next_values = torch.empty(shape, dtype=last_logits.dtype, device=last_logits.device)
        if (
            self._prefill_next_tokens is None
            or self._prefill_next_tokens.device != last_logits.device
            or self._prefill_next_tokens.dtype != torch.long
            or self._prefill_next_tokens.size(0) < shape[0]
            or self._prefill_next_tokens.size(1) != shape[1]
        ):
            self._prefill_next_tokens = torch.empty(shape, device=last_logits.device, dtype=torch.long)
        values = self._prefill_next_values[: shape[0]]
        tokens = self._prefill_next_tokens[: shape[0]]
        torch.max(last_logits, dim=-1, keepdim=True, out=(values, tokens))
        return tokens

    # ----------------------------------------------------------- graph preparation
    def _prepare_graphs(self) -> None:
        cfg = self.config
        if not self.enable_graphs:
            return
        if not self._cuda_available:
            return
        if self.graph_mode == GraphMode.EAGER:
            return
        if self.spec_decoder is not None:
            self.spec_decoder.prepare_workspaces(cfg.batch_size, cfg.dtype_obj, self.device)
        if self._prefill_next_values is None:
            self._prefill_next_values = torch.empty(
                (cfg.batch_size, 1),
                device=self.device,
                dtype=cfg.dtype_obj,
            )
        if self._prefill_next_tokens is None:
            self._prefill_next_tokens = torch.empty(
                (cfg.batch_size, 1),
                device=self.device,
                dtype=torch.long,
            )
        if self.graph_mode in (GraphMode.FULL, GraphMode.FULL_AND_PIECEWISE):
            self._capture_full_graph()
        if self.graph_mode in (GraphMode.PIECEWISE, GraphMode.FULL_AND_PIECEWISE):
            self._capture_piecewise_graphs()

    def _capture_full_graph(self) -> None:
        if self._full_graph is not None or not self._cuda_available:
            return
        cfg = self.config
        self._full_graph = torch.cuda.CUDAGraph()
        self._full_prefill_done = torch.cuda.Event(enable_timing=True)
        self._full_decode_done = torch.cuda.Event(enable_timing=True)
        self._full_replay_start = torch.cuda.Event(enable_timing=True)
        self._full_replay_end = torch.cuda.Event(enable_timing=True)
        # Reset state so the captured path mirrors a clean request.
        if self.paged_cache is not None:
            self.paged_cache.reset()
        if self.spec_decoder is not None:
            self.spec_decoder.reset()
            self.spec_decoder._fallback_chunk = None  # stabilize capture path
        with torch.cuda.graph(self._full_graph):
            hidden, logits = self.model.prefill(self.prompts, kv_cache=self.paged_cache.buffer, cache_start=0)  # type: ignore[arg-type]
            self._prefill_next_token_from_logits(logits)
            if self._full_prefill_done is not None:
                self._full_prefill_done.record()
            _tokens, _, _, _ = self.spec_decoder.decode(  # type: ignore[call-arg]
                self._prefill_next_tokens,
                cfg.decode_tokens,
                self.paged_cache,  # type: ignore[arg-type]
                base_position=cfg.context_window,
            )
            if self._full_decode_done is not None:
                self._full_decode_done.record()
        # Measure capture-time acceptance to reuse for metrics since replay skips Python.
        if self.spec_decoder is not None:
            self._captured_full_spec_accept = self.spec_decoder.acceptance_rate()
            self._captured_full_spec_chunk = float(self.spec_decoder.current_chunk_size())
            self.spec_decoder.reset()
        torch.cuda.synchronize(self.device)

    def _capture_piecewise_graphs(self) -> None:
        if self._piecewise_prefill_graph is not None and self._piecewise_decode_graph is not None:
            return
        if not self._cuda_available:
            return
        cfg = self.config
        self._piecewise_prefill_graph = torch.cuda.CUDAGraph()
        self._piecewise_decode_graph = torch.cuda.CUDAGraph()
        self._piecewise_prefill_done = torch.cuda.Event(enable_timing=True)
        self._piecewise_decode_done = torch.cuda.Event(enable_timing=True)
        self._piecewise_replay_prefill_start = torch.cuda.Event(enable_timing=True)
        self._piecewise_replay_decode_start = torch.cuda.Event(enable_timing=True)
        self._piecewise_replay_end = torch.cuda.Event(enable_timing=True)

        # Prefill graph: compute tokens for decode and populate KV cache.
        if self.paged_cache is not None:
            self.paged_cache.reset()
        if self.spec_decoder is not None:
            self.spec_decoder.reset()
        with torch.cuda.graph(self._piecewise_prefill_graph):
            hidden, logits = self.model.prefill(self.prompts, kv_cache=self.paged_cache.buffer, cache_start=0)  # type: ignore[arg-type]
            self._prefill_next_token_from_logits(logits)
            if self._piecewise_prefill_done is not None:
                self._piecewise_prefill_done.record()

        # Decode graph: reuse prefetched tokens and capture decode path.
        with torch.cuda.graph(self._piecewise_decode_graph):
            _tokens, _, _, _ = self.spec_decoder.decode(  # type: ignore[call-arg]
                self._prefill_next_tokens,
                cfg.decode_tokens,
                self.paged_cache,  # type: ignore[arg-type]
                base_position=cfg.context_window,
            )
            if self._piecewise_decode_done is not None:
                self._piecewise_decode_done.record()

        if self.spec_decoder is not None:
            self._captured_piecewise_spec_accept = self.spec_decoder.acceptance_rate()
            self._captured_piecewise_spec_chunk = float(self.spec_decoder.current_chunk_size())
            self.spec_decoder.reset()
        torch.cuda.synchronize(self.device)

    def _can_use_full_graph(self) -> bool:
        total_tokens = self.config.context_window + self.config.decode_tokens
        return self._full_graph is not None and total_tokens <= self.max_capture_tokens

    def _can_use_piecewise_graph(self) -> bool:
        return self._piecewise_prefill_graph is not None and self._piecewise_decode_graph is not None

    def _replay_full_graph(self) -> Tuple[List[float], List[float], str, float, float, int, int]:
        if not self._can_use_full_graph():
            raise RuntimeError("Full graph replay requested but not captured")
        assert self._full_prefill_done is not None and self._full_decode_done is not None
        assert self._full_replay_start is not None and self._full_replay_end is not None
        ttft_times = self._iteration_ttft_times
        tpot_times = self._iteration_tpot_times
        start = self._full_replay_start
        end = self._full_replay_end
        current_stream = torch.cuda.current_stream(self.device)
        start.record(current_stream)
        self._full_graph.replay()  # type: ignore[union-attr]
        end.record(current_stream)
        torch.cuda.synchronize(self.device)
        ttft_ms = start.elapsed_time(self._full_prefill_done)
        decode_total_ms = self._full_prefill_done.elapsed_time(self._full_decode_done)
        decode_count = self.config.decode_tokens
        per_token_ms = decode_total_ms / max(1, decode_count)
        ttft_times[0] = ttft_ms
        for idx in range(decode_count):
            tpot_times[idx] = per_token_ms
        return ttft_times, tpot_times, "full_graph", ttft_ms, decode_total_ms, 1, decode_count

    def _replay_piecewise_graph(self) -> Tuple[List[float], List[float], str, float, float, int, int]:
        if not self._can_use_piecewise_graph():
            raise RuntimeError("Piecewise graph replay requested but not captured")
        assert self._piecewise_prefill_done is not None and self._piecewise_decode_done is not None
        assert self._piecewise_replay_prefill_start is not None
        assert self._piecewise_replay_decode_start is not None
        assert self._piecewise_replay_end is not None
        if self.paged_cache is not None:
            self.paged_cache.reset()
        ttft_times = self._iteration_ttft_times
        tpot_times = self._iteration_tpot_times
        start_prefill = self._piecewise_replay_prefill_start
        start_decode = self._piecewise_replay_decode_start
        end = self._piecewise_replay_end
        current_stream = torch.cuda.current_stream(self.device)
        start_prefill.record(current_stream)
        self._piecewise_prefill_graph.replay()  # type: ignore[union-attr]
        start_decode.record(current_stream)
        self._piecewise_decode_graph.replay()  # type: ignore[union-attr]
        end.record(current_stream)
        torch.cuda.synchronize(self.device)
        ttft_ms = start_prefill.elapsed_time(self._piecewise_prefill_done)
        decode_total_ms = self._piecewise_prefill_done.elapsed_time(self._piecewise_decode_done)
        decode_count = self.config.decode_tokens
        per_token_ms = decode_total_ms / max(1, decode_count)
        ttft_times[0] = ttft_ms
        for idx in range(decode_count):
            tpot_times[idx] = per_token_ms
        return ttft_times, tpot_times, "piecewise_graph", ttft_ms, decode_total_ms, 1, decode_count

    def _run_eager_path(self) -> Tuple[List[float], List[float], str, float, float, torch.Tensor, float, float, int, int]:
        if self.model is None or self.prompts is None or self.paged_cache is None or self.spec_decoder is None:
            raise RuntimeError("Benchmark not initialized")

        cfg = self.config
        paged_cache = self.paged_cache  # type: ignore[assignment]
        spec = self.spec_decoder  # type: ignore[assignment]
        paged_cache.reset()
        spec.reset()

        ttft_times = self._iteration_ttft_times
        tpot_times = self._iteration_tpot_times

        with torch.inference_mode(), self._nvtx_range("prefill_dualpipe"):
            prefill_start = self._record_start()
            hidden, logits = self.model.prefill(self.prompts, kv_cache=paged_cache.buffer, cache_start=0)
            torch.cuda.synchronize(self.device)
            ttft_ms = self._record_stop(prefill_start)
            ttft_times[0] = ttft_ms
            paged_cache.mark_prefill(cfg.context_window)

        next_tokens = self._prefill_next_token_from_logits(logits)
        with torch.inference_mode(), self._nvtx_range("speculative_decode"):
            chunk_used = spec.current_chunk_size()
            tokens, decode_times, decode_count, decode_total_ms = spec.decode(
                next_tokens,
                cfg.decode_tokens,
                paged_cache,
                base_position=cfg.context_window,
            )
            for idx in range(decode_count):
                tpot_times[idx] = decode_times[idx]
        return (
            ttft_times,
            tpot_times,
            "eager",
            spec.acceptance_rate(),
            float(chunk_used),
            tokens,
            ttft_ms,
            decode_total_ms,
            1,
            decode_count,
        )

    # --------------------------------------------------------------- benchmark_fn
    def benchmark_fn(self) -> Dict[str, object]:
        if self.model is None or self.prompts is None or self.paged_cache is None or self.spec_decoder is None:
            raise RuntimeError("Benchmark not initialized")

        cfg = self.config
        paged_cache = self.paged_cache  # type: ignore[assignment]
        spec = self.spec_decoder  # type: ignore[assignment]

        logical_index = self.device.index if self.device.index is not None else None
        telemetry_before = query_gpu_telemetry(logical_index)

        prefill_assignments = 0
        decode_assignments = 0
        prefix_cache_lengths = self._router_prefix_cache_lengths
        prefix_count = self._router_prefix_count
        if not prefix_cache_lengths or not prefix_count:
            raise RuntimeError("setup() must initialize router prefix-cache lengths")
        prompt_stub = self._router_prompt_stub
        if self._router_prompt_len != cfg.context_window:
            raise RuntimeError("setup() must initialize router prompt stub")
        router_requests = self._router_requests
        if self._router_request_count != cfg.batch_size:
            raise RuntimeError("setup() must initialize router requests")
        if self._router_devnull is None:
            raise RuntimeError("setup() must initialize router stdout sink")
        iteration = self._iteration
        route_timestamp = time.time()
        with contextlib.redirect_stdout(self._router_devnull):
            for idx in self._router_batch_range:
                req = router_requests[idx]
                req.id = f"req-{iteration}-{idx}"
                req.timestamp = route_timestamp
                req.prefix_cached_length = prefix_cache_lengths[(idx + iteration) % prefix_count]
                stage, _ = self.router.route_request(req)
                if stage == "prefill":
                    prefill_assignments += 1
                else:
                    decode_assignments += 1

        ttft_times = self._iteration_ttft_times
        tpot_times = self._iteration_tpot_times
        ttft_total_ms = 0.0
        tpot_total_ms = 0.0
        ttft_count = 0
        tpot_count = 0
        graph_path = "eager"
        spec_accept_used: Optional[float] = None
        spec_chunk_used: Optional[float] = None

        tokens: Optional[torch.Tensor] = None
        # Select execution mode based on requested graph mode and capture viability.
        with torch.inference_mode(), self._nvtx_range("graph_mode_select"):
            if self.graph_mode == GraphMode.EAGER:
                (
                    ttft_times,
                    tpot_times,
                    graph_path,
                    spec_accept_used,
                    spec_chunk_used,
                    tokens,
                    ttft_total_ms,
                    tpot_total_ms,
                    ttft_count,
                    tpot_count,
                ) = self._run_eager_path()
            elif self.graph_mode == GraphMode.FULL and self._can_use_full_graph():
                (
                    ttft_times,
                    tpot_times,
                    graph_path,
                    ttft_total_ms,
                    tpot_total_ms,
                    ttft_count,
                    tpot_count,
                ) = self._replay_full_graph()
                spec_accept_used = self._captured_full_spec_accept
                spec_chunk_used = self._captured_full_spec_chunk
            elif self.graph_mode == GraphMode.PIECEWISE and self._can_use_piecewise_graph():
                (
                    ttft_times,
                    tpot_times,
                    graph_path,
                    ttft_total_ms,
                    tpot_total_ms,
                    ttft_count,
                    tpot_count,
                ) = self._replay_piecewise_graph()
                spec_accept_used = self._captured_piecewise_spec_accept
                spec_chunk_used = self._captured_piecewise_spec_chunk
            elif self.graph_mode == GraphMode.FULL_AND_PIECEWISE and self._can_use_full_graph():
                (
                    ttft_times,
                    tpot_times,
                    graph_path,
                    ttft_total_ms,
                    tpot_total_ms,
                    ttft_count,
                    tpot_count,
                ) = self._replay_full_graph()
                spec_accept_used = self._captured_full_spec_accept
                spec_chunk_used = self._captured_full_spec_chunk
            elif self.graph_mode == GraphMode.FULL_AND_PIECEWISE and self._can_use_piecewise_graph():
                # Fallback for prompts that exceed the capture boundary
                with self._nvtx_range("graph_fallback_piecewise"):
                    (
                        ttft_times,
                        tpot_times,
                        graph_path,
                        ttft_total_ms,
                        tpot_total_ms,
                        ttft_count,
                        tpot_count,
                    ) = self._replay_piecewise_graph()
                    spec_accept_used = self._captured_piecewise_spec_accept
                    spec_chunk_used = self._captured_piecewise_spec_chunk
            else:
                (
                    ttft_times,
                    tpot_times,
                    graph_path,
                    spec_accept_used,
                    spec_chunk_used,
                    tokens,
                    ttft_total_ms,
                    tpot_total_ms,
                    ttft_count,
                    tpot_count,
                ) = self._run_eager_path()

        telemetry_after = query_gpu_telemetry(logical_index)

        total_time_s = (ttft_total_ms + tpot_total_ms) / 1000.0
        throughput = cfg.tokens_per_iteration / max(total_time_s, 1e-6)
        prefill_bytes = cfg.batch_size * cfg.context_window * cfg.hidden_size * self._dtype_bytes
        nvlink_gbps = 0.0
        if ttft_count and ttft_total_ms > 0:
            nvlink_gbps = (prefill_bytes * 8.0 / 1e9) / (ttft_total_ms / 1000.0)
        measured_nvlink = self._compute_nvlink_delta(telemetry_before, telemetry_after, total_time_s)
        self._nvlink_status = telemetry_after.get("nvlink_status", "unknown")

        if ttft_count:
            self._ttft_total_ms += ttft_total_ms
            self._ttft_count += ttft_count
        if tpot_count:
            self._tpot_total_ms += tpot_total_ms
            self._tpot_count += tpot_count
        self._throughput_total += throughput
        self._throughput_count += 1
        if spec_accept_used is None:
            spec_accept_used = spec.acceptance_rate()
        if spec_chunk_used is None:
            spec_chunk_used = spec.current_chunk_size()
        self._spec_accept_total += spec_accept_used
        self._spec_accept_count += 1
        self._spec_chunk_total += spec_chunk_used
        self._spec_chunk_count += 1
        self._nvlink_total_gbps += nvlink_gbps
        self._nvlink_count += 1
        if measured_nvlink is not None:
            self._nvlink_measured_total_gbps += measured_nvlink
            self._nvlink_measured_count += 1
        else:
            if not self._nvlink_warned:
                self._nvlink_warned = True
        routed_requests = max(prefill_assignments + decode_assignments, 1)
        self._prefill_share_total += prefill_assignments / routed_requests
        self._prefill_share_count += 1
        self._paged_hit_total += paged_cache.occupancy_ratio
        self._paged_hit_count += 1
        self._page_faults_total += float(paged_cache.page_faults)
        self._page_faults_count += 1
        self._memory_poll_pending = self._cuda_available

        if tokens is not None:
            self.output = tokens
        self._iteration += 1
        self._refresh_router_metrics()
        iteration_payload = self._iteration_metric_payload
        iteration_payload["graph_path"] = graph_path
        return iteration_payload

    def capture_verification_payload(self) -> None:
        if self.model is None or self.prompts is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"prompt": self.prompts},
            output=self.output.to(dtype=torch.float32),
            batch_size=int(self.prompts.shape[0]),
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": self.config.dtype_obj == torch.float16,
                "bf16": self.config.dtype_obj == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if self._cuda_available else False,
            },
            output_tolerance=(1e-2, 1e-2),
        )

    def finalize_iteration_metrics(self) -> Optional[Dict[str, float]]:
        if not self._memory_poll_pending:
            return None
        self._memory_poll_pending = False
        if self._cuda_available:
            peak_bytes = torch.cuda.max_memory_allocated(self.device)  # type: ignore[arg-type]
            if peak_bytes:
                self._memory_total_gb += peak_bytes * self._memory_bytes_to_gb
                self._memory_count += 1
            if _RESET_PEAK_MEMORY_STATS is not None:
                _RESET_PEAK_MEMORY_STATS(self.device)
        return None

    def _compute_nvlink_delta(
        self,
        telemetry_before: Dict[str, Optional[float]],
        telemetry_after: Dict[str, Optional[float]],
        elapsed_s: float,
    ) -> Optional[float]:
        if elapsed_s <= 0:
            return None
        tx_before = telemetry_before.get("nvlink_tx_bytes_total") if telemetry_before else None
        tx_after = telemetry_after.get("nvlink_tx_bytes_total") if telemetry_after else None
        rx_before = telemetry_before.get("nvlink_rx_bytes_total") if telemetry_before else None
        rx_after = telemetry_after.get("nvlink_rx_bytes_total") if telemetry_after else None
        if None in (tx_before, tx_after, rx_before, rx_after):
            return None
        delta_tx = max(0.0, tx_after - tx_before)
        delta_rx = max(0.0, rx_after - rx_before)
        total_delta = delta_tx + delta_rx
        if total_delta <= 0.0:
            return None
        return (total_delta * 8.0) / (elapsed_s * 1e9)

    # ------------------------------------------------------------------ lifecycle
    def teardown(self) -> None:
        self.finalize_iteration_metrics()
        self.model = None
        self.draft_model = None
        self.prompts = None
        self.paged_cache = None
        self.spec_decoder = None
        self._full_replay_start = None
        self._full_replay_end = None
        self._piecewise_replay_prefill_start = None
        self._piecewise_replay_decode_start = None
        self._piecewise_replay_end = None
        self._router_prefix_cache_lengths = []
        self._router_prefix_count = 0
        self._router_prompt_stub = []
        self._router_prompt_len = 0
        self._router_requests = []
        self._router_request_count = 0
        self._router_batch_range = range(0)
        if self._router_devnull is not None:
            self._router_devnull.close()
            self._router_devnull = None
        if self._cuda_available:
            torch.cuda.empty_cache()
        if self._mem_logger is not None:
            self._mem_logger.stop()
            self._mem_logger = None

    # ------------------------------------------------------------------- configs
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=6, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload_metadata

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        self.finalize_iteration_metrics()
        if not self._throughput_count:
            return None
        def mean(total: float, count: int) -> float:
            return float(total / count) if count else 0.0

        metrics = {
            "optimized_moe.throughput_tok_s": mean(self._throughput_total, self._throughput_count),
            "optimized_moe.ttft_mean_ms": mean(self._ttft_total_ms, self._ttft_count),
            "optimized_moe.tpot_mean_ms": mean(self._tpot_total_ms, self._tpot_count),
            "optimized_moe.spec_accept_rate": mean(self._spec_accept_total, self._spec_accept_count),
            "optimized_moe.spec_chunk_size": mean(self._spec_chunk_total, self._spec_chunk_count),
            "optimized_moe.nvlink_reported_gbps": mean(self._nvlink_total_gbps, self._nvlink_count),
            "optimized_moe.prefill_share": mean(self._prefill_share_total, self._prefill_share_count),
            "optimized_moe.paged_cache_occupancy": mean(self._paged_hit_total, self._paged_hit_count),
            "optimized_moe.page_faults": mean(self._page_faults_total, self._page_faults_count),
        }
        if self._nvlink_measured_count:
            metrics["optimized_moe.nvlink_measured_gbps"] = float(
                self._nvlink_measured_total_gbps / self._nvlink_measured_count
            )
        else:
            code = {
                "ok": 0.0,
                "nvlink_counters_missing": 1.0,
                "nvlink_disabled": 2.0,
                "nvml_unavailable": 3.0,
            }.get(self._nvlink_status, 4.0)
            metrics["optimized_moe.nvlink_status_code"] = code
        if self._memory_count:
            metrics["optimized_moe.peak_memory_gb"] = float(self._memory_total_gb / self._memory_count)
        if self.paged_cache is not None:
            metrics["optimized_moe.kv_cache_gb"] = self.paged_cache.memory_gb
        return metrics

    def validate_result(self) -> Optional[str]:
        if not self._ttft_count:
            return "No TTFT samples captured"
        if not self._tpot_count:
            return "No decode tokens captured"
        return None

def get_benchmark() -> BaseBenchmark:
    return VLLMMoEInferenceBenchmark()


def _run_harness(
    iterations: Optional[int],
    warmup: Optional[int],
    spec_config: Optional[str],
    graph_mode: Optional[str],
    max_capture_tokens: Optional[int],
) -> None:
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    if spec_config:
        benchmark.spec_config_path = Path(spec_config)
    if graph_mode:
        benchmark.graph_mode = GraphMode.from_str(graph_mode)
    if max_capture_tokens is not None:
        benchmark.max_capture_tokens = max_capture_tokens
    config = benchmark.get_config()
    if iterations is not None:
        config.iterations = iterations
    if warmup is not None:
        config.warmup = warmup
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"Optimized MoE inference mean latency: {mean_ms:.3f} ms")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="vLLM-style MoE inference benchmark harness.")
    parser.add_argument("--iterations", type=int, help="Override benchmark iterations")
    parser.add_argument("--warmup", type=int, help="Override warmup iterations")
    parser.add_argument(
        "--graph-mode",
        type=str,
        choices=[m.value for m in GraphMode],
        help="Graph execution mode (default: full_and_piecewise or OPT_MOE_GRAPH_MODE env)",
    )
    parser.add_argument(
        "--max-capture-tokens",
        type=int,
        help="Maximum total tokens allowed for graph capture before falling back to piecewise/eager",
    )
    parser.add_argument(
        "--spec-config",
        type=str,
        help=f"Optional path to Speculators-style config; defaults to {DEFAULT_SPEC_CONFIG}",
    )
    args = parser.parse_args(argv)
    _run_harness(
        args.iterations,
        args.warmup,
        args.spec_config,
        args.graph_mode,
        args.max_capture_tokens,
    )
    return 0
