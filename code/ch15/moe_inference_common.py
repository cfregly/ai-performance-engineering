"""Shared single-GPU MoE inference benchmark logic."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.metrics import compute_inference_metrics
from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.optimization.moe_inference import (
    MoEFeedForward,
    MoEFeedForwardSortedDispatch,
    MoeInferenceConfig,
    SimpleMoEGPT,
    allocate_kv_cache,
    env_override_float,
    env_override_int,
)
from core.profiling.gpu_memory_logger import (
    GpuMemoryLogger,
    resolve_gpu_log_interval,
    resolve_gpu_log_path,
)
from core.profiling.gpu_telemetry import query_gpu_telemetry


class _MoeInferenceBenchmarkBase(VerificationPayloadMixin, BaseBenchmark):
    """Shared setup, timing, and verification logic for chapter 15 MoE inference."""

    def __init__(self, *, label: str) -> None:
        super().__init__()
        self.label = label
        self.config = self._build_config()
        self.batch_size = int(self.config.batch_size)
        self.max_batch_size = int(self.config.batch_size)
        self._total_tokens = int(self.config.tokens_per_iteration)
        self._total_requests = int(self.config.batch_size)
        self._ttft_ms = 0.0
        self._tpot_ms = 0.0

        self.model: Optional[SimpleMoEGPT] = None
        self.prompts: Optional[torch.Tensor] = None
        self.kv_cache: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._next_token_buffer: Optional[torch.Tensor] = None
        self._next_token_values: Optional[torch.Tensor] = None
        self._history: Dict[str, List[float]] = {
            "ttft": [],
            "tpot": [],
            "throughput": [],
            "nvlink": [],
            "nvlink_measured": [],
            "memory_gb": [],
        }
        self._workload_metadata = WorkloadMetadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=float(self.config.tokens_per_iteration),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=float(self.config.tokens_per_iteration),
        )
        self._mem_logger: Optional[GpuMemoryLogger] = None
        self._mem_log_path: Optional[Path] = None
        self._nvlink_warned = False
        self._nvlink_status = "unknown"
        self._telemetry_before: Dict[str, Optional[float]] = {}
        self._prefill_start_event: Optional[torch.cuda.Event] = None
        self._prefill_end_event: Optional[torch.cuda.Event] = None
        self._decode_start_event: Optional[torch.cuda.Event] = None
        self._decode_end_event: Optional[torch.cuda.Event] = None

    def _build_config(self) -> MoeInferenceConfig:
        return MoeInferenceConfig(
            vocab_size=env_override_int("BASELINE_MOE_VOCAB", 16384),
            hidden_size=env_override_int("BASELINE_MOE_HIDDEN", 1024),
            ffn_size=env_override_int("BASELINE_MOE_FFN", 4096),
            num_layers=env_override_int("BASELINE_MOE_LAYERS", 8),
            num_moe_layers=env_override_int("BASELINE_MOE_MOE_LAYERS", 4),
            num_experts=env_override_int("BASELINE_MOE_EXPERTS", 32),
            top_k=2,
            moe_layer_frequency=max(1, env_override_int("BASELINE_MOE_MOE_FREQ", 2)),
            batch_size=env_override_int("BASELINE_MOE_BATCH", 1),
            context_window=env_override_int("BASELINE_MOE_CONTEXT", 512),
            decode_tokens=env_override_int("BASELINE_MOE_DECODE", 64),
            router_noise=env_override_float("BASELINE_MOE_ROUTER_NOISE", 0.0),
            dtype=torch.bfloat16,
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: MoE inference benchmark requires CUDA")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        cfg = self.config
        self.model = SimpleMoEGPT(cfg, device=self.device).eval()
        self.prompts = torch.randint(
            0,
            cfg.vocab_size,
            (cfg.batch_size, cfg.context_window),
            device=self.device,
        )
        total_tokens = cfg.context_window + cfg.decode_tokens
        self.kv_cache = allocate_kv_cache(
            cfg.batch_size,
            total_tokens,
            cfg.hidden_size,
            cfg.dtype_obj,
            self.device,
        )
        self._next_token_buffer = torch.empty((cfg.batch_size, 1), dtype=torch.long, device=self.device)
        self._next_token_values = None
        torch.cuda.synchronize(self.device)
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats(self.device)
            log_path = resolve_gpu_log_path(None)
            logger = GpuMemoryLogger(
                device=self.device,
                interval=resolve_gpu_log_interval(1.0),
                log_path=log_path,
            )
            if logger.start():
                self._mem_logger = logger
                self._mem_log_path = log_path

    def _prepare_iteration_metrics(self) -> None:
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats(self.device)
        logical_index = self.device.index if self.device.index is not None else None
        self._telemetry_before = query_gpu_telemetry(logical_index)
        self._prefill_start_event = torch.cuda.Event(enable_timing=True)
        self._prefill_end_event = torch.cuda.Event(enable_timing=True)
        self._decode_start_event = torch.cuda.Event(enable_timing=True)
        self._decode_end_event = torch.cuda.Event(enable_timing=True)

    def _next_token_from_logits(self, logits_last: torch.Tensor) -> torch.Tensor:
        if self._next_token_buffer is None:
            raise RuntimeError("Next-token buffer is not initialized")
        batch_size = logits_last.size(0)
        if tuple(self._next_token_buffer.shape) != (batch_size, 1):
            self._next_token_buffer = torch.empty((batch_size, 1), dtype=torch.long, device=logits_last.device)
        if (
            self._next_token_values is None
            or self._next_token_values.device != logits_last.device
            or self._next_token_values.dtype != logits_last.dtype
            or tuple(self._next_token_values.shape) != (batch_size, 1)
        ):
            self._next_token_values = torch.empty_like(logits_last[:, :1])
        torch.max(logits_last, dim=-1, keepdim=True, out=(self._next_token_values, self._next_token_buffer))
        return self._next_token_buffer

    def benchmark_fn(self) -> None:
        if self.model is None or self.prompts is None or self.kv_cache is None:
            raise RuntimeError("Model, prompts, or KV cache not initialized")
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: MoE inference benchmark requires CUDA")

        self._prepare_iteration_metrics()
        stream = torch.cuda.current_stream(device=self.device)
        cfg = self.config

        with torch.inference_mode():
            with self._nvtx_range(self.label):
                if any(
                    event is None
                    for event in (
                        self._prefill_start_event,
                        self._prefill_end_event,
                        self._decode_start_event,
                        self._decode_end_event,
                    )
                ):
                    raise RuntimeError("Iteration timing events not initialized")
                self._prefill_start_event.record(stream)
                _hidden, logits = self.model.prefill(self.prompts, kv_cache=self.kv_cache, cache_start=0)
                seed_tokens = self._next_token_from_logits(logits[:, -1, :])
                self._prefill_end_event.record(stream)

                self._decode_start_event.record(stream)
                for step in range(cfg.decode_tokens):
                    _hidden, decode_logits = self.model.decode(
                        seed_tokens,
                        kv_cache=self.kv_cache,
                        position=cfg.context_window + step,
                    )
                    seed_tokens = self._next_token_from_logits(decode_logits[:, -1, :])
                self._decode_end_event.record(stream)
                self.output = seed_tokens.detach()

    def finalize_iteration_metrics(self) -> Optional[Dict[str, List[float]]]:
        if any(
            event is None
            for event in (
                self._prefill_start_event,
                self._prefill_end_event,
                self._decode_start_event,
                self._decode_end_event,
            )
        ):
            return None

        prefill_ms = float(self._prefill_start_event.elapsed_time(self._prefill_end_event))
        total_decode_ms = float(self._decode_start_event.elapsed_time(self._decode_end_event))
        avg_tpot_ms = total_decode_ms / max(float(self.config.decode_tokens), 1.0)
        self._ttft_ms = prefill_ms
        self._tpot_ms = avg_tpot_ms

        total_time_s = (prefill_ms + total_decode_ms) / 1000.0
        throughput = self.config.tokens_per_iteration / max(total_time_s, 1e-6)
        logical_index = self.device.index if self.device.index is not None else None
        telemetry_after = query_gpu_telemetry(logical_index)
        nvlink_gbps = telemetry_after.get("nvlink_tx_gbps") or 0.0
        measured_nvlink = self._compute_nvlink_delta(self._telemetry_before, telemetry_after, total_time_s)
        self._nvlink_status = telemetry_after.get("nvlink_status", "unknown")

        self._history["ttft"].append(prefill_ms)
        self._history["tpot"].extend([avg_tpot_ms] * self.config.decode_tokens)
        self._history["throughput"].append(throughput)
        self._history["nvlink"].append(nvlink_gbps)
        if measured_nvlink is not None:
            self._history["nvlink_measured"].append(measured_nvlink)
        elif not self._nvlink_warned:
            self._nvlink_warned = True

        peak_bytes = torch.cuda.max_memory_allocated(self.device)
        if peak_bytes:
            self._history["memory_gb"].append(peak_bytes / (1024 ** 3))

        self._prefill_start_event = None
        self._prefill_end_event = None
        self._decode_start_event = None
        self._decode_end_event = None

        return {
            "ttft_times_ms": [prefill_ms],
            "tpot_times_ms": [avg_tpot_ms] * self.config.decode_tokens,
        }

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

    def teardown(self) -> None:
        self.model = None
        self.prompts = None
        self.kv_cache = None
        self.output = None
        self._prefill_start_event = None
        self._prefill_end_event = None
        self._decode_start_event = None
        self._decode_end_event = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self._mem_logger is not None:
            self._mem_logger.stop()
            self._mem_logger = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            timing_method="wall_clock",
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload_metadata

    def get_custom_metrics(self) -> Optional[dict]:
        if not self._history["ttft"] or not self._history["tpot"]:
            return None

        metrics = compute_inference_metrics(
            ttft_ms=sum(self._history["ttft"]) / len(self._history["ttft"]),
            tpot_ms=sum(self._history["tpot"]) / len(self._history["tpot"]),
            total_tokens=self._total_tokens,
            total_requests=self._total_requests,
            batch_size=self.batch_size,
            max_batch_size=self.max_batch_size,
        )
        if self._history["throughput"]:
            metrics["inference.measured_tokens_per_second"] = (
                sum(self._history["throughput"]) / len(self._history["throughput"])
            )
        if self._history["nvlink"]:
            metrics["inference.nvlink_tx_gbps"] = sum(self._history["nvlink"]) / len(self._history["nvlink"])
        if self._history["nvlink_measured"]:
            metrics["inference.nvlink_measured_gbps"] = (
                sum(self._history["nvlink_measured"]) / len(self._history["nvlink_measured"])
            )
        if self._history["memory_gb"]:
            metrics["inference.peak_memory_gb"] = max(self._history["memory_gb"])
        return metrics

    def validate_result(self) -> Optional[str]:
        if not self._history["ttft"]:
            return "No TTFT samples recorded"
        if not self._history["tpot"]:
            return "No TPOT samples recorded"
        return None

    def capture_verification_payload(self) -> None:
        if self.output is None or self.prompts is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        param_count = sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0
        self._set_verification_payload(
            inputs={"prompt": self.prompts},
            output=self.output.to(dtype=torch.float32),
            batch_size=int(self.prompts.shape[0]),
            parameter_count=param_count,
            precision_flags={
                "fp16": self.config.dtype_obj == torch.float16,
                "bf16": self.config.dtype_obj == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-2, 1e-2),
        )


class BaselineMoeInferenceBenchmark(_MoeInferenceBenchmarkBase):
    """Baseline MoE inference benchmark (single-GPU sequential prefill + decode)."""

    def __init__(self) -> None:
        super().__init__(label="baseline_moe_inference")


class OptimizedMoeInferenceBenchmark(_MoeInferenceBenchmarkBase):
    """Optimized MoE inference benchmark with sorted expert dispatch."""

    def __init__(self) -> None:
        super().__init__(label="optimized_moe_inference")

    def setup(self) -> None:
        super().setup()
        if self.model is None:
            raise RuntimeError("Model not initialized")

        for block in getattr(self.model, "layers", []):
            ff = getattr(block, "ff", None)
            if ff is None or not isinstance(ff, MoEFeedForward):
                continue
            if isinstance(ff, MoEFeedForwardSortedDispatch):
                continue
            replacement = MoEFeedForwardSortedDispatch(
                self.config.hidden_size,
                self.config.ffn_size,
                num_experts=self.config.num_experts,
                top_k=self.config.top_k,
                router_noise=self.config.router_noise,
                capacity_factor=self.config.capacity_factor,
                device=self.device,
                dtype=self.config.dtype_obj,
            )
            replacement.load_state_dict(ff.state_dict(), strict=True)
            block.ff = replacement


__all__ = [
    "BaselineMoeInferenceBenchmark",
    "OptimizedMoeInferenceBenchmark",
    "attach_benchmark_metadata",
]
