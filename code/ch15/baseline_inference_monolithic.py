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
        self._history: Dict[str, List[float]] = {"ttft": [], "tpot": []}
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
    
    def setup(self) -> None:
        """Setup: initialize model and data."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model = SimpleLLM(vocab_size=10000, hidden_dim=512, num_layers=8).to(self.device).to(torch.bfloat16).eval()
        self.prompt = (torch.arange(self.prefill_seq, device=self.device, dtype=torch.int64) % 10000).unsqueeze(0)
        self.kv_cache = None
        self.output = None
        self._last_decoded_tokens = [torch.empty(0) for _ in range(self.num_tokens)]
        self._verify_prompt = self.prompt.detach().clone()

    def benchmark_fn(self) -> Optional[dict]:
        if self.model is None or self.prompt is None:
            raise RuntimeError("Model or prompt not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())

        with nvtx_range("inference_monolithic", enable=enable_nvtx):
            with torch.no_grad():
                kv_cache = self.model.prefill(self.prompt)
                decoded_tokens = self._last_decoded_tokens
                if len(decoded_tokens) != self.num_tokens:
                    decoded_tokens = [torch.empty(0) for _ in range(self.num_tokens)]
                decode_state = kv_cache
                token_idx = 0

                for _ in range(self.num_tokens):
                    decoded = self.model.decode(decode_state, num_tokens=1)
                    decode_state = decoded[:, -1:, :]
                    decoded_tokens[token_idx] = decode_state
                    token_idx += 1

                if token_idx == 0:
                    raise RuntimeError("Decode loop produced no tokens")

                self._last_decoded_tokens = decoded_tokens
                self.output = None
                self._metrics_pending = True
                return {}

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
        tpot_times_ms = [tpot_mean_ms] * self.num_tokens

        self._history["ttft"].append(ttft_ms)
        self._history["tpot"].extend(tpot_times_ms)
        self._metrics_pending = False
        return {
            "ttft_times_ms": [ttft_ms],
            "tpot_times_ms": tpot_times_ms,
        }

    def capture_verification_payload(self) -> None:
        self.finalize_iteration_metrics()
        if self.model is None or self.prompt is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        if self.output is None:
            if not self._last_decoded_tokens:
                raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
            self.output = torch.cat(self._last_decoded_tokens, dim=1)
        self._set_verification_payload(
            inputs={"prompt": self.prompt},
            output=self.output.float(),
            batch_size=int(self.prompt.shape[0]),
            parameter_count=sum(p.numel() for p in self.model.parameters()),
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
        if not self._history["ttft"]:
            return None
        return {
            "monolithic.ttft_ms": float(sum(self._history["ttft"]) / len(self._history["ttft"])),
            "monolithic.tpot_mean_ms": float(sum(self._history["tpot"]) / len(self._history["tpot"])),
        }

    def validate_result(self) -> Optional[str]:
        self.finalize_iteration_metrics()
        if not self._history["ttft"]:
            return "No TTFT samples recorded"
        if not self._history["tpot"]:
            return "No TPOT samples recorded"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineInferenceMonolithicBenchmark()
