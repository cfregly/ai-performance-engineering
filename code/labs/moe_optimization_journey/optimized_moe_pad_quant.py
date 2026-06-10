"""Optimized MoE pad+quant + finalize+slice benchmark (torch.compile fusion)."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range
from core.utils.compile_utils import get_optimal_compile_mode
from labs.moe_optimization_journey.moe_pad_quant_common import build_moe_pad_quant_model


class OptimizedMoEPadQuantBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: torch.compile fuses pad+quant+finalize+slice chain."""

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.compiled = None
        self.inputs: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

        self.vocab_size = 32000
        self.hidden = 512
        self.intermediate = 2048
        self.num_experts = 32
        self.num_experts_per_tok = 2
        self.batch = 8
        self.seq_len = 128
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        model, _ = build_moe_pad_quant_model(
            hidden_size=self.hidden,
            intermediate_size=self.intermediate,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            vocab_size=self.vocab_size,
            num_layers=1,
            num_heads=8,
            level=4,
        )
        self.model = model.to(self.device, dtype=torch.bfloat16)
        self.model.eval()
        # get_optimal_compile_mode keeps max-autotune on the pinned toolchain but
        # falls back to "default" on sm_103 + Triton >= 3.6 (where max-autotune
        # emits an unloadable tcgen05.wait.st kernel). On that sm_103 fallback,
        # "default" leaves this small launch-bound MoE (1024 tokens) at ~1.27x vs
        # eager; cudagraph capture ("reduce-overhead") cuts the per-kernel launch
        # overhead to ~2.2x and still avoids the tcgen05 bug. Static shapes here
        # (batch=8, seq=128) make cudagraphs safe.
        compile_mode = get_optimal_compile_mode("max-autotune")
        if compile_mode == "default":
            compile_mode = "reduce-overhead"
        self.compiled = torch.compile(self.model, mode=compile_mode)
        self.inputs = torch.randint(
            0, self.vocab_size, (self.batch, self.seq_len), device=self.device
        )
        # Warm compile
        with torch.no_grad():
            _ = self.compiled(self.inputs)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.compiled is None or self.inputs is None:
            raise RuntimeError("Benchmark not initialized")
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("moe_pad_quant_optimized", enable=enable_nvtx):
            with torch.no_grad():
                self.output = self.compiled(self.inputs)
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.output is None or self.inputs is None:
            raise RuntimeError("benchmark_fn() did not produce output")
        self._set_verification_payload(
            inputs={"input_ids": self.inputs.detach()},
            output=self.output.detach().clone(),
            batch_size=self.batch,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def teardown(self) -> None:
        if self.compiled is not None:
            del self.compiled
            self.compiled = None
        if self.model is not None:
            del self.model
            self.model = None
        self.inputs = None
        self.output = None
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=6,
            warmup=6,
            # Match the existing MoE journey convention: compile-heavy MoE
            # surfaces can crash during subprocess teardown on some hosts.
            use_subprocess=False,
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedMoEPadQuantBenchmark()
