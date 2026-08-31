#!/usr/bin/env python3
"""Native E4M3 MoE experiment using three scaled GEMMs per routed expert.

Weights use per-expert tensor scales; activation scaling is included in timing.
Every routed output is unsorted and combined. Accuracy and performance remain
unqualified until the actual target passes full-output verification and an
externally reviewed error policy. Historical utilization/speedup claims were
withdrawn after the unscaled-cast defect; no new speedup is asserted here.
"""

import torch
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark
from labs.moe_optimization_journey.native_fp8_math import (
    combine_sorted_routes,
    full_output_errors,
    load_accuracy_limits,
    quantize_e4m3,
    reference_moe,
)


class NativeFP8MoE(VerificationPayloadMixin, BaseBenchmark):
    """MoE benchmark with native FP8 via _scaled_mm."""

    WARMUP = 5
    ITERATIONS = 10

    # Model config
    HIDDEN_SIZE = 4096
    INTERMEDIATE_SIZE = 11008
    NUM_EXPERTS = 8
    TOP_K = 2
    BATCH_SIZE = 16
    SEQ_LEN = 4096  # 64K tokens

    def setup(self) -> None:
        self.accuracy_limits = load_accuracy_limits()
        self._setup_workload()

    def _setup_workload(self) -> None:
        """Actual workload setup, also used by the non-accepting calibration CLI."""
        if not torch.cuda.is_available():
            raise RuntimeError("Native FP8 MoE requires a real CUDA GPU")
        if self.HIDDEN_SIZE % 16 or self.INTERMEDIATE_SIZE % 16:
            raise ValueError("Native scaled GEMM hidden/intermediate dimensions must be multiples of 16")
        if min(self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE, self.NUM_EXPERTS,
               self.TOP_K, self.BATCH_SIZE, self.SEQ_LEN) <= 0:
            raise ValueError("Native FP8 MoE dimensions must be positive")
        import gc

        self.device = 'cuda'

        # Clean up CUDA state to prevent RNG corruption from previous benchmarks
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        try:
            if hasattr(torch.cuda, 'graph_pool_trim'):
                torch.cuda.graph_pool_trim()
        except Exception:
            pass

        rng = torch.Generator(device="cpu").manual_seed(torch.initial_seed())

        H = self.HIDDEN_SIZE
        I = self.INTERMEDIATE_SIZE
        E = self.NUM_EXPERTS
        K = self.TOP_K
        self._expert_range = range(E)
        batch_seq = self.BATCH_SIZE * self.SEQ_LEN

        print("=" * 60)
        print("LEVEL 6: NATIVE FP8 MoE")
        print("=" * 60)
        print(f"Config: H={H}, I={I}, E={E}, K={K}, tokens={batch_seq:,}")
        print()

        # Input and weights - use CPU randn + to(device) to avoid CUDA RNG graph issues
        self.x = torch.randn(batch_seq, H, dtype=torch.bfloat16, generator=rng).to(self.device)
        self._verify_output_buffer = torch.empty(
            (batch_seq, H),
            device=self.device,
            dtype=torch.float32,
        )

        # BF16 reference weights
        w1 = torch.randn(E, H, I, dtype=torch.bfloat16, generator=rng).to(self.device)
        w3 = torch.randn(E, H, I, dtype=torch.bfloat16, generator=rng).to(self.device)
        w2 = torch.randn(E, I, H, dtype=torch.bfloat16, generator=rng).to(self.device)

        # Preserve independent original BF16 weights on CPU for full verification.
        self._reference_weights_cpu = (w1.cpu(), w3.cpu(), w2.cpu())
        for name, weight in (("w1", w1), ("w3", w3), ("w2", w2)):
            encoded = torch.empty_like(weight.transpose(-1, -2).contiguous(),
                                       dtype=torch.float8_e4m3fn)
            scales = [quantize_e4m3(weight[e].T, encoded[e]) for e in range(E)]
            setattr(self, name + "_fp8", encoded)
            setattr(self, name + "_scales", scales)

        # Routing - use CPU tensors + to(device)
        expert_indices_cpu = torch.randint(0, E, (batch_seq, K), generator=rng)
        expert_weights_cpu = F.softmax(
            torch.randn(batch_seq, K, generator=rng), dim=-1
        ).to(torch.bfloat16)
        self._reference_ids_cpu = expert_indices_cpu
        self._reference_routing_cpu = expert_weights_cpu
        expert_indices = expert_indices_cpu.to(self.device)
        expert_weights = expert_weights_cpu.to(self.device)

        # Pre-compute routing
        flat_idx = expert_indices.view(-1)
        sorted_order = torch.argsort(flat_idx, stable=True)
        self._inverse_order = torch.argsort(sorted_order)
        self._combined_output_buffer = torch.empty((batch_seq, H), device=self.device, dtype=torch.float32)
        self._unsorted_output_buffer = torch.empty((batch_seq * K, H), device=self.device, dtype=self.x.dtype)
        self.counts = torch.bincount(expert_indices_cpu.view(-1), minlength=E).tolist()
        expanded_token_indices = torch.arange(batch_seq * K, device=self.device, dtype=torch.int64)
        if K != 1:
            expanded_token_indices.div_(K, rounding_mode="floor")
        self._sorted_token_indices = expanded_token_indices.index_select(0, sorted_order)
        self._sorted_weights = expert_weights.view(-1).index_select(0, sorted_order)
        self._sorted_tokens = torch.empty(
            batch_seq * K,
            H,
            device=self.device,
            dtype=self.x.dtype,
        )
        self._output_buffer = torch.empty(
            batch_seq * K,
            H,
            device=self.device,
            dtype=self.x.dtype,
        )
        max_expert_tokens = max(self.counts)
        self._tokens_fp8_buffer = torch.empty(
            max_expert_tokens,
            H,
            device=self.device,
            dtype=torch.float8_e4m3fn,
        )
        self._hidden_fp8_buffer = torch.empty(
            max_expert_tokens,
            I,
            device=self.device,
            dtype=torch.float8_e4m3fn,
        )
        self._expert_token_views = []
        self._expert_weight_views = []
        self._expert_output_views = []
        self._expert_tokens_fp8_views = []
        self._expert_hidden_fp8_views = []
        offset = 0
        for count in self.counts:
            next_offset = offset + count
            self._expert_token_views.append(self._sorted_tokens[offset:next_offset])
            self._expert_weight_views.append(self._sorted_weights[offset:next_offset].unsqueeze(-1))
            self._expert_output_views.append(self._output_buffer[offset:next_offset])
            self._expert_tokens_fp8_views.append(self._tokens_fp8_buffer[:count])
            self._expert_hidden_fp8_views.append(self._hidden_fp8_buffer[:count])
            offset = next_offset
        self._payload_param_count = int(self.w1_fp8.numel() + self.w2_fp8.numel() + self.w3_fp8.numel())

        print(f"FP8 weight memory: {(self.w1_fp8.numel() + self.w3_fp8.numel() + self.w2_fp8.numel()) / 1e9:.2f} GB")
        print("Independent BF16 weights retained on CPU for verification; this is not total-memory compression.")
        print()

    def benchmark_fn(self) -> None:
        """Run FP8 MoE forward pass."""
        x = self.x
        output = self._combined_output_buffer

        torch.index_select(x, 0, self._sorted_token_indices, out=self._sorted_tokens)

        for e in self._expert_range:
            count = self.counts[e]
            if count == 0:
                continue

            tokens_e = self._expert_token_views[e]
            tokens_fp8_slice = self._expert_tokens_fp8_views[e]
            tokens_scale = quantize_e4m3(tokens_e, tokens_fp8_slice)
            weights_e = self._expert_weight_views[e]

            # Native FP8 matmul via _scaled_mm
            gate = torch._scaled_mm(
                tokens_fp8_slice, self.w1_fp8[e].T,
                scale_a=tokens_scale, scale_b=self.w1_scales[e],
                out_dtype=torch.bfloat16
            )
            gate = F.silu(gate, inplace=True)

            up = torch._scaled_mm(
                tokens_fp8_slice, self.w3_fp8[e].T,
                scale_a=tokens_scale, scale_b=self.w3_scales[e],
                out_dtype=torch.bfloat16
            )

            gate.mul_(up)
            hidden_fp8_slice = self._expert_hidden_fp8_views[e]
            hidden_scale = quantize_e4m3(gate, hidden_fp8_slice)

            expert_out = torch._scaled_mm(
                hidden_fp8_slice, self.w2_fp8[e].T,
                scale_a=hidden_scale, scale_b=self.w2_scales[e],
                out_dtype=torch.bfloat16
            )

            torch.mul(expert_out, weights_e, out=self._expert_output_views[e])

        combine_sorted_routes(self._output_buffer, self._inverse_order, self.TOP_K,
                              self._unsorted_output_buffer, output)
        self.output = output
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        verify_output = getattr(self, "_verify_output_buffer", None)
        if self.output is None or verify_output is None:
            raise RuntimeError("benchmark_fn() must run before verification capture")
        self._check_full_output()
        param_count = self._payload_param_count
        verify_output.copy_(self.output)
        self._set_verification_payload(
            inputs={"x": self.x.detach()},
            output=verify_output,
            batch_size=self.BATCH_SIZE,
            parameter_count=param_count,
            precision_flags={"bf16": True, "fp8": True, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(self.accuracy_limits.pairwise_rtol,
                              self.accuracy_limits.pairwise_atol),
        )

    def _check_full_output(self):
        reference = reference_moe(self.x, self._reference_weights_cpu,
                                  self._reference_ids_cpu, self._reference_routing_cpu)
        self.accuracy_errors = full_output_errors(self.output, reference)
        self.accuracy_limits.check(self.accuracy_errors)
        return self.accuracy_errors

    def validate_result(self) -> str | None:
        self._check_full_output()
        return None

    def get_extra_metrics(self) -> dict:
        batch_seq = self.BATCH_SIZE * self.SEQ_LEN
        total_flops = batch_seq * self.TOP_K * 3 * 2 * self.HIDDEN_SIZE * self.INTERMEDIATE_SIZE
        return {
            "total_flops": total_flops,
            "reference_b200_dense_fp8_peak_tflops": 4500,
            # A named published ceiling, not detection of this device or measured utilization.
            "accuracy_policy_configured": getattr(self, "accuracy_limits", None) is not None,
        }

    def teardown(self) -> None:
        for name in (
            "x",
            "w1_fp8",
            "w2_fp8",
            "w3_fp8",
            "w1_scales", "w2_scales", "w3_scales",
            "_reference_weights_cpu", "_reference_ids_cpu", "_reference_routing_cpu",
            "_inverse_order", "_combined_output_buffer", "_unsorted_output_buffer",
            "_expert_token_views", "_expert_weight_views", "_expert_output_views",
            "_expert_tokens_fp8_views", "_expert_hidden_fp8_views",
            "_sorted_token_indices",
            "_sorted_weights",
            "_sorted_tokens",
            "_output_buffer",
            "_tokens_fp8_buffer",
            "_hidden_fp8_buffer",
            "_expert_output_buffer",
            "_expert_range",
            "output",
            "_verify_output_buffer",
        ):
            if hasattr(self, name):
                setattr(self, name, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().teardown()

def get_benchmark() -> NativeFP8MoE:
    return NativeFP8MoE()
