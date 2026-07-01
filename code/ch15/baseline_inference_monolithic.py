"""baseline_inference_monolithic.py - Monolithic inference (baseline).

Single service handles both prefill and decode - blocks each other.
Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ch15.inference_monolithic_common import SimpleLLM
from core.benchmark.verification_mixin import VerificationPayloadMixin  # noqa: E402
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class BaselineInferenceMonolithicBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Monolithic inference baseline using the shared harness conventions."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[SimpleLLM] = None
        self.prompt: Optional[torch.Tensor] = None
        self.kv_cache: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._empty_iteration_result: Dict[str, List[float]] = {}
        # Workload dimensions for signature matching
        self.batch_size = 1
        self.prefill_seq = 64
        self.num_tokens = 128
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=self.prefill_seq + self.num_tokens,
        )
        self._verify_prompt: Optional[torch.Tensor] = None
        self._last_elapsed_ms: Optional[float] = None
        self._metrics_pending = False
        self._last_decoded_tokens: List[torch.Tensor] = []
        self._decoded_token_count = 0
        self._decode_token_indices = range(self.num_tokens)
        self._decode_output_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._ttft_metric_values = [0.0]
        self._tpot_metric_values = [0.0] * self.num_tokens
        self._iteration_metric_payload: Dict[str, List[float]] = {
            "ttft_times_ms": self._ttft_metric_values,
            "tpot_times_ms": self._tpot_metric_values,
        }
        self._enable_nvtx = False
        self._payload_parameter_count = 0
        self._ttft_total_ms = 0.0
        self._tpot_total_ms = 0.0
        self._ttft_count = 0
        self._tpot_count = 0
    
    def setup(self) -> None:
        """Setup: initialize model and data."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.model = SimpleLLM(vocab_size=10000, hidden_dim=512, num_layers=8).to(self.device).to(torch.bfloat16).eval()
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        self.prompt = (torch.arange(self.prefill_seq, device=self.device, dtype=torch.int64) % 10000).unsqueeze(0)
        self.kv_cache = None
        self.output = None
        self._ttft_total_ms = 0.0
        self._tpot_total_ms = 0.0
        self._ttft_count = 0
        self._tpot_count = 0
        self._last_decoded_tokens = [torch.empty(0) for _ in range(self.num_tokens)]
        self._decoded_token_count = len(self._last_decoded_tokens)
        self._decode_token_indices = range(self.num_tokens)
        self._decode_output_buffer = torch.empty(
            (self.batch_size, self.num_tokens, self.model.hidden_dim),
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._verify_output_buffer = torch.empty_like(self._decode_output_buffer, dtype=torch.float32)
        if len(self._tpot_metric_values) != self.num_tokens:
            self._tpot_metric_values = [0.0] * self.num_tokens
            self._iteration_metric_payload["tpot_times_ms"] = self._tpot_metric_values
        self._verify_prompt = self.prompt.detach().clone()

    def benchmark_fn(self) -> Optional[dict]:
        if self.model is None or self.prompt is None:
            raise RuntimeError("Model or prompt not initialized")

        with nvtx_range("inference_monolithic", enable=self._enable_nvtx):
            with torch.inference_mode():
                kv_cache = self.model.prefill(self.prompt)
                decoded_tokens = self._last_decoded_tokens
                if self._decoded_token_count != self.num_tokens:
                    raise RuntimeError("Decode output slots not initialized")
                decode_token_indices = self._decode_token_indices
                decode_state = kv_cache

                for token_idx in decode_token_indices:
                    decode_state = self.model.decode_step(decode_state)
                    decoded_tokens[token_idx] = decode_state

                if not decoded_tokens:
                    raise RuntimeError("Decode loop produced no tokens")

                self._last_decoded_tokens = decoded_tokens
                self.output = None
                self._metrics_pending = True
                return self._empty_iteration_result

    def finalize_iteration_metrics(self) -> Optional[Dict[str, List[float]]]:
        if self._last_elapsed_ms is None or not self._metrics_pending:
            return None

        # Use the harness-timed iteration latency and split it by token-equivalent work:
        # prefill processes the full prompt, while decode advances one token at a time.
        total_token_work = float(self.prefill_seq + self.num_tokens)
        if total_token_work <= 0:
            return None

        ttft_ms = float(self._last_elapsed_ms) * (float(self.prefill_seq) / total_token_work)
        tpot_mean_ms = float(self._last_elapsed_ms) / total_token_work
        tpot_times_ms = self._tpot_metric_values
        if len(tpot_times_ms) != self.num_tokens:
            tpot_times_ms = [0.0] * self.num_tokens
            self._tpot_metric_values = tpot_times_ms
            self._iteration_metric_payload["tpot_times_ms"] = tpot_times_ms
        for idx in range(len(tpot_times_ms)):
            tpot_times_ms[idx] = tpot_mean_ms

        self._ttft_total_ms += ttft_ms
        self._tpot_total_ms += tpot_mean_ms * self.num_tokens
        self._ttft_count += 1
        self._tpot_count += self.num_tokens
        self._metrics_pending = False
        self._ttft_metric_values[0] = ttft_ms
        return self._iteration_metric_payload

    def capture_verification_payload(self) -> None:
        self.finalize_iteration_metrics()
        if self.model is None or self.prompt is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        if self.output is None:
            if not self._last_decoded_tokens:
                raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
            if self._decode_output_buffer is None:
                raise RuntimeError("setup() must initialize decode output buffer")
            token_offset = 0
            for decoded in self._last_decoded_tokens:
                token_width = decoded.shape[1]
                next_offset = token_offset + token_width
                self._decode_output_buffer[:, token_offset:next_offset, :].copy_(decoded)
                token_offset = next_offset
            if token_offset != self._decode_output_buffer.shape[1]:
                raise RuntimeError("unexpected decode output shape")
            self.output = self._decode_output_buffer
        if self._verify_output_buffer is None:
            raise RuntimeError("setup() must initialize verification output buffer")
        self._verify_output_buffer.copy_(self.output, non_blocking=False)
        self._set_verification_payload(
            inputs={"prompt": self.prompt},
            output=self._verify_output_buffer,
            batch_size=int(self.prompt.shape[0]),
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.model = None
        self.prompt = None
        self.kv_cache = None
        self.output = None
        self._last_decoded_tokens = []
        self._decoded_token_count = 0
        self._decode_output_buffer = None
        self._verify_output_buffer = None
        self._last_elapsed_ms = None
        self._metrics_pending = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        self.finalize_iteration_metrics()
        if self._ttft_count <= 0:
            return None
        return {
            "monolithic.ttft_ms": float(self._ttft_total_ms / self._ttft_count),
            "monolithic.tpot_mean_ms": float(
                self._tpot_total_ms / self._tpot_count if self._tpot_count else 0.0
            ),
        }

    def validate_result(self) -> Optional[str]:
        self.finalize_iteration_metrics()
        if self._ttft_count <= 0:
            return "No TTFT samples recorded"
        if self._tpot_count <= 0:
            return "No TPOT samples recorded"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineInferenceMonolithicBenchmark()
