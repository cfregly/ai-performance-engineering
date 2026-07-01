"""optimized_inference_monolithic.py - Monolithic inference (optimized).

Pairs with: baseline_inference_monolithic.py

This variant keeps the same prefill+autoregressive decode workload as the
baseline, but routes the full request through a compiled graph and avoids the
baseline's repeated list growth inside the hot path.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata  # noqa: E402
from ch15.inference_monolithic_common import SimpleLLM
from core.benchmark.verification_mixin import VerificationPayloadMixin  # noqa: E402


class OptimizedInferenceMonolithicBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Monolithic inference optimized benchmark using shared harness conventions."""

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[SimpleLLM] = None
        self.prompt: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._compiled_inference = None
        self._decode_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0

        self.batch_size = 1
        self.prefill_seq = 64
        self.num_tokens = 128
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=self.prefill_seq + self.num_tokens,
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model = SimpleLLM(vocab_size=10000, hidden_dim=512, num_layers=8).to(self.device).to(torch.bfloat16).eval()
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        self.prompt = (torch.arange(self.prefill_seq, device=self.device, dtype=torch.int64) % 10000).unsqueeze(0)
        self._decode_buffer = torch.empty(
            (self.batch_size, self.num_tokens, self.model.hidden_dim),
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._verify_output_buffer = torch.empty_like(self._decode_buffer, dtype=torch.float32)
        self.output = None
        # The batch=1 request is launch-overhead bound: prefill is one narrow prompt pass,
        # then decode runs num_tokens x num_layers tiny Linears. Compile the whole request
        # with reduce-overhead so inductor cudagraphs the launch train into one replay.
        num_tokens = self.num_tokens
        model = self.model

        def _full_inference(prompt: torch.Tensor, buffer: torch.Tensor) -> torch.Tensor:
            kv_cache = model.prefill(prompt)
            current = kv_cache
            for token_idx in range(num_tokens):
                current = model.decode_step(current)
                buffer[:, token_idx : token_idx + 1, :] = current
            return buffer

        self._compiled_inference = torch.compile(_full_inference, mode="reduce-overhead")
        with torch.inference_mode():
            for _ in range(5):
                _ = self._compiled_inference(self.prompt, self._decode_buffer)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if (
            self.model is None
            or self.prompt is None
            or self._compiled_inference is None
            or self._decode_buffer is None
        ):
            raise RuntimeError("Model or prompt not initialized")

        with self._nvtx_range("inference_monolithic_optimized"):
            with torch.inference_mode():
                self.output = self._compiled_inference(self.prompt, self._decode_buffer)

    def capture_verification_payload(self) -> None:
        if (
            self.model is None
            or self.prompt is None
            or self.output is None
            or self._verify_output_buffer is None
        ):
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
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
        self.output = None
        self._compiled_inference = None
        self._decode_buffer = None
        self._verify_output_buffer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        # warmup>=10 so the reduce-overhead cudagraph of the decode loop is captured before timing.
        return BenchmarkConfig(iterations=20, warmup=10)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "No output produced"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedInferenceMonolithicBenchmark()
