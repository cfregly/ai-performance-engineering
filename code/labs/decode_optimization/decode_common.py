"""Decode loop microbenchmark with baseline and optimized paths.

Shared helpers and configs for the decode_optimization lab variants.
Tests serving optimizations (pinned memory, streams, CUDA graphs, FP8/FP4, torch.compile)
on a simplified MLP-based decode loop.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from core.profiling.nvtx_helper import standardize_nvtx_label

try:
    from core.harness.arch_config import prefer_sdpa_backends
    from core.utils.compile_utils import enable_tf32
except Exception:  # pragma: no cover - defensive import
    prefer_sdpa_backends = None  # type: ignore
    enable_tf32 = None  # type: ignore

from core.benchmark.verification import InputSignature, PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.harness.hardware_capabilities import detect_capabilities

try:  # Optional but strongly recommended for fast variants
    import transformer_engine.pytorch as te
    import transformer_engine.pytorch.constants as te_constants
    from transformer_engine.common.recipe import DelayedScaling
    from transformer_engine.pytorch import LayerNormMLP as TELayerNormMLP
    from transformer_engine.pytorch import Linear as TELinear
    from transformer_engine.pytorch import quantized_model_init

    TE_AVAILABLE = True
except Exception:  # pragma: no cover - safe fallback
    te = None  # type: ignore
    TELinear = None  # type: ignore
    TELayerNormMLP = None  # type: ignore
    quantized_model_init = None  # type: ignore
    DelayedScaling = None  # type: ignore
    te_constants = None  # type: ignore
    TE_AVAILABLE = False


def _te_version_at_least(major: int, minor: int = 0) -> bool:
    if not TE_AVAILABLE or not hasattr(te, "__version__"):
        return False
    try:
        parts = te.__version__.split(".")
        return int(parts[0]) > major or (int(parts[0]) == major and int(parts[1]) >= minor)
    except Exception:
        return False


def _is_blackwell_family() -> bool:
    cap = detect_capabilities()
    if cap is not None:
        # Treat Blackwell (SM100/SM103) and Grace-Blackwell (SM12x) as Blackwell-class for defaults.
        return cap.architecture in {"blackwell", "blackwell_ultra", "grace_blackwell"}
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Cannot determine architecture: capability probe unavailable and CUDA not available."
        )
    cc_major, _ = torch.cuda.get_device_capability()
    return cc_major >= 10


@dataclass
class DecodeConfig:
    batch_size: int = 4
    prompt_tokens: int = 256
    decode_tokens: int = 64
    prefetch_batches: int = 1
    host_payload_mb: int = 0
    hidden_size: int = 1024
    vocab_size: int = 8192
    use_fp8: bool = False
    use_fp4: bool = False
    use_te_mlp: bool = False
    use_pinned_host: bool = False
    use_copy_stream: bool = False
    use_compute_stream: bool = False
    use_cuda_graphs: bool = False
    graph_full_iteration: bool = False
    use_torch_compile: bool = False
    iterations: int = 8
    warmup: int = 10
    label: str = "decode_optimization"


class DecodeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Lightweight decode loop benchmark for testing serving optimizations."""

    def __init__(self, cfg: DecodeConfig):
        super().__init__()
        self.cfg = cfg
        self.dtype = torch.bfloat16
        self.copy_stream: Optional[torch.cuda.Stream] = None
        self.compute_stream: Optional[torch.cuda.Stream] = None
        self.graph_stream: Optional[torch.cuda.Stream] = None
        self.decode_graph: Optional[torch.cuda.CUDAGraph] = None
        self.graph_includes_prefill: bool = False
        self.graph_logits: Optional[torch.Tensor] = None
        self.graph_next_token: Optional[torch.Tensor] = None
        self._custom_metrics: Dict[str, float] = {}
        self._pending_iteration_events: Optional[Dict[str, torch.cuda.Event]] = None
        self._fp8_enabled: bool = False
        self._fp4_enabled: bool = False
        self._graph_error: Optional[str] = None
        self._compile_error: Optional[str] = None
        self._tf32_enabled: bool = False
        self.sdpa_ctx_factory = (
            prefer_sdpa_backends if prefer_sdpa_backends is not None else nullcontext
        )
        self.fp8_recipe = None
        self.output: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self.host_prompts: list[torch.Tensor] = []
        self.gpu_prompts: list[torch.Tensor] = []
        self.host_payloads: list[torch.Tensor] = []
        self.gpu_payloads: list[torch.Tensor] = []
        self.host_payload: Optional[torch.Tensor] = None
        self.gpu_payload: Optional[torch.Tensor] = None
        self._copy_done_events: list[torch.cuda.Event] = []
        self._timing_events: dict[str, torch.cuda.Event] = {}
        self._payload_bytes = 0

        if self.cfg.prefetch_batches < 1:
            raise ValueError("prefetch_batches must be >= 1")
        if self.cfg.prefetch_batches > 2:
            raise NotImplementedError("prefetch_batches > 2 is not supported")
        if self.cfg.host_payload_mb < 0:
            raise ValueError("host_payload_mb must be >= 0")
        if self.cfg.use_fp4 and self.cfg.use_fp8:
            raise ValueError("use_fp4 and use_fp8 are mutually exclusive")
        if self.cfg.use_te_mlp and not TE_AVAILABLE:
            raise RuntimeError(
                "SKIPPED: use_te_mlp requested but Transformer Engine is unavailable"
            )

        if self.cfg.use_fp4:
            if not TE_AVAILABLE:
                raise RuntimeError("SKIPPED: FP4 requested but Transformer Engine is unavailable")
            if not _is_blackwell_family():
                raise RuntimeError("SKIPPED: FP4 decode requires Blackwell-class hardware")
            if getattr(te_constants, "NVFP4_BLOCK_SCALING_SIZE", None) is None:
                raise RuntimeError(
                    "SKIPPED: FP4 decode requires NVFP4 support in Transformer Engine"
                )
            try:
                from transformer_engine.common.recipe import NVFP4BlockScaling
            except Exception as exc:
                raise RuntimeError(
                    "SKIPPED: FP4 decode requires NVFP4BlockScaling support"
                ) from exc
            self.fp8_recipe = DelayedScaling(float8_block_scaling=NVFP4BlockScaling())
            self._fp4_enabled = True
        elif self.cfg.use_fp8:
            if not TE_AVAILABLE:
                raise RuntimeError("SKIPPED: FP8 requested but Transformer Engine is unavailable")
            try:
                # Prefer an inference-friendly FP8 recipe for perf stability.
                # Float8CurrentScaling avoids delayed amax reductions that can introduce
                # iteration-to-iteration jitter in short microbench loops.
                from transformer_engine.common.recipe import Float8CurrentScaling
            except Exception as exc:
                raise RuntimeError(
                    "SKIPPED: FP8 decode requires Float8CurrentScaling support"
                ) from exc
            self.fp8_recipe = Float8CurrentScaling()
            self._fp8_enabled = True
        self.register_workload_metadata(
            requests_per_iteration=float(self.cfg.batch_size * self.cfg.prefetch_batches),
            tokens_per_iteration=float(
                self.cfg.prefetch_batches
                * self.cfg.batch_size
                * (self.cfg.prompt_tokens + self.cfg.decode_tokens)
            ),
        )

    def setup(self) -> None:
        import gc

        # CRITICAL: Clean up CUDA state from previous benchmarks
        # This prevents "Offset increment outside graph capture" errors
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        try:
            if hasattr(torch.cuda, "graph_pool_trim"):
                torch.cuda.graph_pool_trim()
        except Exception:
            pass

        # Reset CUDA RNG state
        try:
            device_idx = torch.cuda.current_device()
            gen = torch.cuda.default_generators[device_idx]
            gen.set_offset(0)
            gen.manual_seed(42)
        except Exception:
            pass

        try:
            torch._dynamo.reset()
        except Exception:
            pass

        try:
            torch._inductor.cudagraph_trees.reset_cudagraph_trees()
        except Exception:
            pass

        # Ensure deterministic RNG state for verification (harness seed is 42).
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        if enable_tf32 is not None:
            enable_tf32(set_global_precision=True)
        else:
            torch.set_float32_matmul_precision("high")
        # Pin TF32 backend flags explicitly for deterministic verification payloads.
        try:
            if hasattr(torch.backends.cuda, "matmul") and hasattr(
                torch.backends.cuda.matmul, "allow_tf32"
            ):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
            self._tf32_enabled = bool(getattr(torch.backends.cuda.matmul, "allow_tf32", True))
        except Exception:
            self._tf32_enabled = True
        if self.cfg.use_copy_stream:
            self.copy_stream = torch.cuda.Stream()
        if self.cfg.use_compute_stream:
            self.compute_stream = torch.cuda.Stream()
        if self.cfg.prefetch_batches > 1 and self.cfg.use_cuda_graphs:
            raise RuntimeError("prefetch_batches > 1 is incompatible with CUDA graphs")
        self._init_model()
        self._init_buffers()
        if torch.cuda.is_available():
            self._copy_done_events = [torch.cuda.Event() for _ in range(self.cfg.prefetch_batches)]
            self._timing_events = {
                name: torch.cuda.Event(enable_timing=True)
                for name in ("prefill_start", "prefill_end", "decode_start", "decode_end")
            }
        self._cache_te_weight_workspaces()
        # Default to eager helpers; swap in compiled variants below when enabled.
        self.prefill_fn = self._prefill
        self.decode_fn = self._decode_step
        if self.cfg.use_torch_compile:
            self._maybe_compile()
        if self.cfg.use_cuda_graphs:
            self._capture_decode_graph()
        total_tokens = (
            self.cfg.prefetch_batches
            * self.cfg.batch_size
            * (self.cfg.prompt_tokens + self.cfg.decode_tokens)
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.cfg.batch_size * self.cfg.prefetch_batches),
            tokens_per_iteration=float(total_tokens),
        )

    # Model + buffer init
    def _init_model(self) -> None:
        hs = self.cfg.hidden_size
        vs = self.cfg.vocab_size
        # Create embedding on CPU first to avoid CUDA RNG graph capture issues
        # then move to device. This ensures parameter init uses CPU RNG.
        self.embedding = nn.Embedding(vs, hs, dtype=self.dtype).to(self.device)

        use_te_modules = bool(
            TE_AVAILABLE and (self.cfg.use_fp8 or self.cfg.use_fp4 or self.cfg.use_te_mlp)
        )
        te_init_context = nullcontext()
        if use_te_modules and (self._fp8_enabled or self._fp4_enabled):
            if quantized_model_init is None or self.fp8_recipe is None:
                raise RuntimeError("FP8/FP4 requires Transformer Engine quantized_model_init")
            te_init_context = quantized_model_init(enabled=True, recipe=self.fp8_recipe)

        def _linear(in_features: int, out_features: int, *, bias: bool = True) -> nn.Module:
            """Create a Linear layer with deterministic CPU initialization.

            For Transformer Engine FP8/FP4 variants we still initialize weights on CPU
            (torch.nn.Linear) and then copy them into the TE module. This keeps
            baseline/optimized weights identical and ensures output verification is
            meaningful (differences reflect precision, not random init drift).
            """
            # Always create a CPU reference for deterministic initialization
            ref = nn.Linear(in_features, out_features, bias=bias, dtype=self.dtype)
            if not use_te_modules:
                return ref.to(self.device)

            # TE Linear must be created on device; copy weights/bias from ref.
            te_linear = TELinear(
                in_features,
                out_features,
                bias=bias,
                params_dtype=self.dtype,
                device=self.device,
            )
            with torch.no_grad():
                te_linear.weight.copy_(ref.weight.to(self.device))
                if bias and te_linear.bias is not None and ref.bias is not None:
                    te_linear.bias.copy_(ref.bias.to(self.device))
            return te_linear

        def _te_mlp(hidden_dim: int, ffn_dim: int) -> nn.Module:
            if TELayerNormMLP is None:
                raise RuntimeError("Transformer Engine LayerNormMLP unavailable")
            ref_ln = nn.LayerNorm(hidden_dim, dtype=self.dtype)
            ref_fc1 = nn.Linear(hidden_dim, ffn_dim, bias=True, dtype=self.dtype)
            ref_fc2 = nn.Linear(ffn_dim, hidden_dim, bias=True, dtype=self.dtype)
            te_mlp = TELayerNormMLP(
                hidden_dim,
                ffn_dim,
                params_dtype=self.dtype,
                device=self.device,
            )
            with torch.no_grad():
                te_mlp.layer_norm_weight.copy_(ref_ln.weight.to(self.device))
                te_mlp.layer_norm_bias.copy_(ref_ln.bias.to(self.device))
                te_mlp.fc1_weight.copy_(ref_fc1.weight.to(self.device))
                te_mlp.fc1_bias.copy_(ref_fc1.bias.to(self.device))
                te_mlp.fc2_weight.copy_(ref_fc2.weight.to(self.device))
                te_mlp.fc2_bias.copy_(ref_fc2.bias.to(self.device))
            return te_mlp

        with te_init_context:
            if use_te_modules and self.cfg.use_te_mlp:
                self.prefill_mlp = _te_mlp(hs, hs * 2)
                self.decode_mlp = _te_mlp(hs, hs)
            else:
                self.prefill_mlp = nn.Sequential(
                    nn.LayerNorm(hs, dtype=self.dtype).to(self.device),
                    _linear(hs, hs * 2),
                    nn.GELU(),
                    _linear(hs * 2, hs),
                )
                self.decode_mlp = nn.Sequential(
                    nn.LayerNorm(hs, dtype=self.dtype).to(self.device),
                    _linear(hs, hs),
                    nn.GELU(),
                    _linear(hs, hs),
                )
            self.lm_head = _linear(hs, vs, bias=False)
        if self.cfg.use_fp8 and TE_AVAILABLE and not self._fp4_enabled:
            self._fp8_enabled = True
        # Parameter count used for verification metadata
        modules = (self.embedding, self.prefill_mlp, self.decode_mlp, self.lm_head)
        self.parameter_count = sum(p.numel() for m in modules for p in m.parameters())

    def _cache_te_weight_workspaces(self) -> None:
        """Pre-quantize TE weights by running a warmup forward pass.

        The correct way to initialize FP8 workspaces is via forward passes under
        fp8_autocast, not by calling get_weight_workspace() manually.
        """
        if (
            not TE_AVAILABLE
            or not (self._fp8_enabled or self._fp4_enabled)
            or os.getenv("DECODE_SKIP_TE_CACHE") == "1"
        ):
            return

        # Warmup FP8 caches by running forward passes - this is the proper API
        # Use CPU randn + to(device) to avoid CUDA RNG graph capture issues
        bsz = self.cfg.batch_size
        hs = self.cfg.hidden_size
        dummy_hidden = torch.randn(bsz, hs, dtype=self.dtype).to(self.device)
        dummy_seq = torch.randn(bsz, self.cfg.prompt_tokens, hs, dtype=self.dtype).to(self.device)

        passes = 4 if self._fp4_enabled else 2
        with torch.no_grad(), te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
            for _ in range(passes):
                # Warmup prefill MLP
                _ = self.prefill_mlp(dummy_seq)
                # Warmup decode MLP
                _ = self.decode_mlp(dummy_hidden)
                # Warmup lm_head
                _ = self.lm_head(dummy_hidden)

        torch.cuda.synchronize()

    def _init_buffers(self) -> None:
        bsz, prompt = self.cfg.batch_size, self.cfg.prompt_tokens
        self.host_prompts = []
        self.gpu_prompts = []
        for _ in range(self.cfg.prefetch_batches):
            host_prompt = torch.randint(
                0, self.cfg.vocab_size, (bsz, prompt), dtype=torch.long, device="cpu"
            )
            if self.cfg.use_pinned_host:
                host_prompt = host_prompt.pin_memory()
            self.host_prompts.append(host_prompt)
            self.gpu_prompts.append(torch.empty_like(host_prompt, device=self.device))
        self.host_prompt = self.host_prompts[0]
        self.gpu_prompt = self.gpu_prompts[0]
        if self.cfg.host_payload_mb:
            self._payload_bytes = int(self.cfg.host_payload_mb * 1024 * 1024)
            for _ in range(self.cfg.prefetch_batches):
                host_payload = torch.empty((self._payload_bytes,), dtype=torch.uint8, device="cpu")
                if self.cfg.use_pinned_host:
                    host_payload = host_payload.pin_memory()
                self.host_payloads.append(host_payload)
                self.gpu_payloads.append(torch.empty_like(host_payload, device=self.device))
            self.host_payload = self.host_payloads[0]
            self.gpu_payload = self.gpu_payloads[0]
        self.state_buffer = torch.zeros(
            (bsz, self.cfg.hidden_size), device=self.device, dtype=self.dtype
        )
        self.current_tokens = torch.empty((bsz,), device=self.device, dtype=torch.long)
        self.next_token_out = torch.empty_like(self.current_tokens)

    # Compiled / graphed helpers
    def _maybe_compile(self) -> None:
        # NO FALLBACK - torch.compile must work
        # When using explicit CUDA graphs, don't use reduce-overhead mode (which uses internal graphs)
        # as this causes "Cannot prepare for replay during capturing stage" errors
        compile_mode = "default" if self.cfg.use_cuda_graphs else "reduce-overhead"
        self.prefill_fn = torch.compile(self._prefill, mode=compile_mode, fullgraph=False)
        self.decode_fn = torch.compile(self._decode_step, mode=compile_mode, fullgraph=True)
        self._compile_error = None

    def _capture_decode_graph(self) -> None:
        # Allocate static outputs once
        bsz = self.cfg.batch_size
        self.graph_stream = torch.cuda.Stream()
        self.graph_logits = torch.empty(
            (bsz, self.cfg.vocab_size), device=self.device, dtype=self.dtype
        )
        self.graph_next_token = torch.empty((bsz,), device=self.device, dtype=torch.long)
        # Ensure prompt buffer is initialized with valid tokens before capture
        self.gpu_prompt.copy_(self.host_prompt, non_blocking=False)

        def _prime_decode_state() -> None:
            prefill_state = self.prefill_fn(self.gpu_prompt)
            self.state_buffer.copy_(prefill_state)
            self.current_tokens.copy_(self.gpu_prompt[:, -1])

        # NO FALLBACK - CUDA graph capture must succeed
        # Warm up to populate kernels/caches prior to capture
        with torch.cuda.stream(self.graph_stream):
            if not self.cfg.graph_full_iteration:
                _prime_decode_state()
            for _ in range(2):
                if self.cfg.graph_full_iteration:
                    _prime_decode_state()
                logits, next_state, next_token = self.decode_fn(
                    self.current_tokens, self.state_buffer
                )
                self.state_buffer.copy_(next_state)
                self.graph_next_token.copy_(next_token)
                self.current_tokens.copy_(next_token)
        torch.cuda.synchronize()
        self.decode_graph = torch.cuda.CUDAGraph()
        # Reset state before capture for determinism
        self.state_buffer.zero_()
        self.current_tokens.zero_()
        if not self.cfg.graph_full_iteration:
            with torch.cuda.stream(self.graph_stream):
                _prime_decode_state()
        torch.cuda.synchronize()
        with torch.cuda.graph(self.decode_graph, stream=self.graph_stream):
            if self.cfg.graph_full_iteration:
                _prime_decode_state()
            for _ in range(self.cfg.decode_tokens):
                logits, next_state, next_token = self.decode_fn(
                    self.current_tokens, self.state_buffer
                )
                self.state_buffer.copy_(next_state)
                self.graph_logits.copy_(logits)
                self.graph_next_token.copy_(next_token)
                if self.cfg.graph_full_iteration:
                    self.current_tokens.copy_(next_token)
        torch.cuda.synchronize()
        self.graph_includes_prefill = bool(self.cfg.graph_full_iteration)
        self._graph_error = None

    # Core math - NOTE: fp8_autocast should be managed at benchmark_fn level
    # to avoid per-call overhead and memory leaks.
    # All operations use torch.no_grad() since this is inference (no backward pass).
    def _prefill(self, tokens: torch.Tensor) -> torch.Tensor:
        """Prefill phase - fp8_autocast managed externally."""
        with torch.no_grad(), self.sdpa_ctx_factory():
            embeds = self.embedding(tokens)
            hidden = self.prefill_mlp(embeds)
        return hidden[:, -1, :]

    def _decode_step(
        self, tokens: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single decode step - fp8_autocast managed externally."""
        with torch.no_grad(), self.sdpa_ctx_factory():
            token_hidden = self.embedding(tokens)
            combined = token_hidden + state
            hidden = self.decode_mlp(combined)
            logits = self.lm_head(hidden)
        next_token = torch.argmax(logits, dim=-1)
        return logits, hidden, next_token

    def _get_fp8_context(self):
        """Return fp8_autocast context if FP8 is enabled, else nullcontext."""
        if (
            (self._fp8_enabled or self._fp4_enabled)
            and te is not None
            and self.fp8_recipe is not None
        ):
            return te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe)
        return nullcontext()

    # Execution helpers
    def _copy_prompts_to_device(self) -> None:
        non_blocking = bool(self.cfg.use_pinned_host)
        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                self.gpu_prompt.copy_(self.host_prompt, non_blocking=non_blocking)
                if self.host_payload is not None and self.gpu_payload is not None:
                    self.gpu_payload.copy_(self.host_payload, non_blocking=non_blocking)
            torch.cuda.current_stream().wait_stream(self.copy_stream)
        else:
            self.gpu_prompt.copy_(self.host_prompt, non_blocking=non_blocking)
            if self.host_payload is not None and self.gpu_payload is not None:
                self.gpu_payload.copy_(self.host_payload, non_blocking=non_blocking)

    def _copy_prompt_to_device_idx(
        self,
        idx: int,
        *,
        stream: Optional[torch.cuda.Stream],
        record_event: bool,
    ) -> Optional[torch.cuda.Event]:
        if idx < 0 or idx >= len(self.host_prompts):
            raise ValueError(f"prompt index {idx} out of range")
        non_blocking = bool(self.cfg.use_pinned_host)
        active_stream = stream or torch.cuda.current_stream()
        with torch.cuda.stream(active_stream):
            self.gpu_prompts[idx].copy_(self.host_prompts[idx], non_blocking=non_blocking)
            if self.host_payloads and self.gpu_payloads:
                self.gpu_payloads[idx].copy_(self.host_payloads[idx], non_blocking=non_blocking)
            if record_event:
                if idx >= len(self._copy_done_events):
                    raise RuntimeError("copy events not initialized")
                event = self._copy_done_events[idx]
                event.record(active_stream)
                return event
        return None

    def _timing_event(self, name: str) -> torch.cuda.Event:
        try:
            return self._timing_events[name]
        except KeyError as exc:
            raise RuntimeError("Decode timing events were not initialized") from exc

    def _run_prefill_decode(self, prompt: torch.Tensor, stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(stream):
            prefill_state = self.prefill_fn(prompt)
            self.state_buffer.copy_(prefill_state)
            self.current_tokens.copy_(prompt[:, -1])
            for _ in range(self.cfg.decode_tokens):
                _, next_state, next_token = self.decode_fn(self.current_tokens, self.state_buffer)
                self.state_buffer.copy_(next_state)
                self.current_tokens.copy_(next_token)

    def _benchmark_prefetch_batches(self) -> None:
        if self.cfg.prefetch_batches != 2:
            raise RuntimeError("prefetch_batches must be 2 for pipelined decode")
        if self.decode_graph is not None:
            raise RuntimeError("prefetch_batches > 1 cannot run with CUDA graphs")

        # Timers via CUDA events
        iter_start = self._timing_event("prefill_start")
        batch0_end = self._timing_event("prefill_end")
        iter_end = self._timing_event("decode_end")

        # Streams for copy/compute
        prefill_stream = self.compute_stream or torch.cuda.current_stream()
        copy_stream = self.copy_stream or prefill_stream
        timing_stream = prefill_stream

        # NVTX ranges for profiling clarity
        try:
            import torch.cuda.nvtx as nvtx  # type: ignore
        except Exception:
            nvtx = None  # type: ignore

        iter_start.record(timing_stream)
        event0 = self._copy_prompt_to_device_idx(0, stream=copy_stream, record_event=True)
        if event0 is not None:
            prefill_stream.wait_event(event0)

        with self._get_fp8_context():
            if nvtx:
                nvtx.range_push(standardize_nvtx_label("compute_math:prefill_decode_0"))
            self._run_prefill_decode(self.gpu_prompts[0], prefill_stream)
            if nvtx:
                nvtx.range_pop()
            batch0_end.record(timing_stream)

            event1 = self._copy_prompt_to_device_idx(1, stream=copy_stream, record_event=True)
            if event1 is not None:
                prefill_stream.wait_event(event1)

            if nvtx:
                nvtx.range_push(standardize_nvtx_label("compute_math:prefill_decode_1"))
            self._run_prefill_decode(self.gpu_prompts[1], prefill_stream)
            if nvtx:
                nvtx.range_pop()
            iter_end.record(timing_stream)

        if self.compute_stream is not None:
            torch.cuda.current_stream().wait_stream(self.compute_stream)

        self.gpu_prompt = self.gpu_prompts[1]
        self.host_prompt = self.host_prompts[1]
        if self.gpu_payloads:
            self.gpu_payload = self.gpu_payloads[1]
            self.host_payload = self.host_payloads[1]

        self._pending_iteration_events = {
            "prefill_start": iter_start,
            "prefill_end": batch0_end,
            "decode_start": batch0_end,
            "decode_end": iter_end,
        }

    def benchmark_fn(self) -> None:
        if self.cfg.prefetch_batches > 1:
            self._benchmark_prefetch_batches()
            return
        # Timers via CUDA events
        prefill_start = self._timing_event("prefill_start")
        prefill_end = self._timing_event("prefill_end")
        decode_start = self._timing_event("decode_start")
        decode_end = self._timing_event("decode_end")

        # Choose streams for work/timing
        prefill_stream = self.compute_stream or torch.cuda.current_stream()
        decode_stream = self.graph_stream if self.decode_graph is not None else prefill_stream
        timing_stream = decode_stream

        # NVTX ranges for profiling clarity
        try:
            import torch.cuda.nvtx as nvtx  # type: ignore
        except Exception:
            nvtx = None  # type: ignore

        prefill_start.record(prefill_stream)
        self._copy_prompts_to_device()
        if nvtx:
            nvtx.range_push(standardize_nvtx_label("compute_math:prefill"))

        # Single FP8 context for entire forward pass to avoid workspace churn
        with self._get_fp8_context():
            # Prefill (or reset when full graph already contains it)
            if self.decode_graph is not None and self.graph_includes_prefill:
                self.state_buffer.zero_()
                self.current_tokens.zero_()
            else:
                prefill_state = self.prefill_fn(self.gpu_prompt)
                self.state_buffer.copy_(prefill_state)
                self.current_tokens.copy_(self.gpu_prompt[:, -1])
            prefill_end.record(prefill_stream)

            # Ensure decode stream waits for prefill when streams differ
            if decode_stream is not prefill_stream:
                decode_stream.wait_event(prefill_end)

            if nvtx:
                nvtx.range_pop()  # prefill
                nvtx.range_push(standardize_nvtx_label("compute_math:decode"))

            # Decode
            decode_start.record(timing_stream)
            if self.decode_graph is not None:
                # Replay once; graph already captures the decode loop
                if self.graph_stream is not None:
                    with torch.cuda.stream(self.graph_stream):
                        self.decode_graph.replay()
                    torch.cuda.current_stream().wait_stream(self.graph_stream)
                else:
                    self.decode_graph.replay()
            else:
                with torch.cuda.stream(self.compute_stream or torch.cuda.current_stream()):
                    for _ in range(self.cfg.decode_tokens):
                        logits, next_state, next_token = self.decode_fn(
                            self.current_tokens, self.state_buffer
                        )
                        self.state_buffer.copy_(next_state)
                        self.current_tokens.copy_(next_token)
                if self.compute_stream is not None:
                    torch.cuda.current_stream().wait_stream(self.compute_stream)
            decode_end.record(timing_stream)

        if nvtx:
            nvtx.range_pop()
        self._pending_iteration_events = {
            "prefill_start": prefill_start,
            "prefill_end": prefill_end,
            "decode_start": decode_start,
            "decode_end": decode_end,
        }

    def finalize_iteration_metrics(self) -> Optional[Dict[str, list[float]]]:
        if not self._pending_iteration_events:
            return None

        prefill_start = self._pending_iteration_events["prefill_start"]
        prefill_end = self._pending_iteration_events["prefill_end"]
        decode_start = self._pending_iteration_events["decode_start"]
        decode_end = self._pending_iteration_events["decode_end"]
        self._pending_iteration_events = None

        ttft_ms = prefill_start.elapsed_time(prefill_end) if prefill_end.query() else 0.0
        decode_ms = decode_start.elapsed_time(decode_end) if decode_end.query() else 0.0
        total_ms = (
            prefill_start.elapsed_time(decode_end) if decode_end.query() else ttft_ms + decode_ms
        )

        eps_ms = 1e-6
        ttft_ms = max(ttft_ms, eps_ms)
        decode_ms = max(decode_ms, eps_ms)
        total_ms = max(total_ms, ttft_ms + decode_ms)

        tpot_ms = decode_ms / max(self.cfg.decode_tokens, 1)
        tokens_per_iter = float(
            self.cfg.prefetch_batches
            * self.cfg.batch_size
            * (self.cfg.prompt_tokens + self.cfg.decode_tokens)
        )
        tokens_per_s = tokens_per_iter / max(total_ms / 1000.0, 1e-6)

        self._custom_metrics = {
            "tokens_per_iteration": tokens_per_iter,
            "prompt_tokens": float(self.cfg.prompt_tokens),
            "decode_tokens": float(self.cfg.decode_tokens),
            "hidden_size": float(self.cfg.hidden_size),
            "prefetch_batches": float(self.cfg.prefetch_batches),
            "host_payload_mb": float(self.cfg.host_payload_mb),
            "use_pinned_host": float(self.cfg.use_pinned_host),
            "use_copy_stream": float(self.cfg.use_copy_stream),
            "use_compute_stream": float(self.cfg.use_compute_stream),
            "use_cuda_graphs": float(self.decode_graph is not None),
            "graph_full_iteration": float(self.graph_includes_prefill),
            "use_torch_compile": float(self.cfg.use_torch_compile and not self._compile_error),
            "use_fp8": float(self._fp8_enabled),
            "fp8_fallback": float(1.0 if (self.cfg.use_fp8 and not self._fp8_enabled) else 0.0),
            "use_fp4": float(self._fp4_enabled),
            "use_te_mlp": float(self.cfg.use_te_mlp),
            "ttft_ms": float(ttft_ms),
            "decode_time_ms": float(decode_ms),
            "tpot_mean_ms": float(tpot_ms),
            "tokens_per_s": float(tokens_per_s),
            "total_time_ms": float(total_ms),
        }
        if self._compile_error:
            self._custom_metrics["compile_error"] = 1.0
        if self._graph_error:
            self._custom_metrics["graph_capture_failed"] = 1.0

        return {
            "ttft_times_ms": [ttft_ms],
            "tpot_times_ms": [tpot_ms] * self.cfg.decode_tokens,
        }

    def _finalize_output(self) -> None:
        """Capture a slice of model state for verification."""
        hidden_slice = self.state_buffer[:1, : min(8, self.state_buffer.shape[1])].float()
        summary_tensor = hidden_slice.reshape(1, -1)
        self.output = summary_tensor.detach().clone()

    def capture_verification_payload(self) -> None:
        if any(
            not hasattr(self, name) or getattr(self, name) is None
            for name in ("gpu_prompt", "state_buffer")
        ):
            raise RuntimeError(
                "setup() and benchmark_fn() must be called before capture_verification_payload()"
            )
        self._finalize_output()
        if self.output is None:
            raise RuntimeError(
                "benchmark_fn() must populate output before capture_verification_payload()"
            )
        config_tensor = torch.tensor(
            [
                self.cfg.batch_size,
                self.cfg.prompt_tokens,
                self.cfg.decode_tokens,
                self.cfg.prefetch_batches,
                self.cfg.host_payload_mb,
            ],
            device="cpu",
            dtype=torch.int64,
        )
        inputs = {
            "gpu_prompt": self.gpu_prompt,
            "state_buffer": self.state_buffer,
            "config": config_tensor,
        }
        if self.gpu_payload is not None:
            inputs["host_payload"] = self.gpu_payload
        self._set_verification_payload(
            inputs=inputs,
            output=self.output,
            batch_size=int(self.cfg.batch_size),
            parameter_count=int(self.parameter_count),
            precision_flags={
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "fp8": bool(self._fp8_enabled),
                "tf32": bool(self._tf32_enabled),
            },
            output_tolerance=(0.1, 1.0),
        )

    def _signature_parameter_count(self) -> int:
        hs = int(self.cfg.hidden_size)
        vs = int(self.cfg.vocab_size)
        # embedding + lm_head + prefill/decode MLPs (+ layernorm/bias terms)
        return int((2 * vs * hs) + (6 * hs * hs) + (9 * hs))

    def get_input_signature(self) -> InputSignature:
        """Return the exact verification signature for strict input equivalence."""
        shapes = {
            "gpu_prompt": (int(self.cfg.batch_size), int(self.cfg.prompt_tokens)),
            "state_buffer": (int(self.cfg.batch_size), int(self.cfg.hidden_size)),
            "config": (5,),
            "output": (1, min(8, int(self.cfg.hidden_size))),
        }
        dtypes = {
            "gpu_prompt": str(torch.int64),
            "state_buffer": str(self.dtype),
            "config": str(torch.int64),
            "output": str(torch.float32),
        }
        if self.cfg.host_payload_mb:
            payload_bytes = int(self.cfg.host_payload_mb * 1024 * 1024)
            shapes["host_payload"] = (payload_bytes,)
            dtypes["host_payload"] = str(torch.uint8)
        return InputSignature(
            shapes=shapes,
            dtypes=dtypes,
            batch_size=int(self.cfg.batch_size * self.cfg.prefetch_batches),
            parameter_count=int(self.parameter_count)
            if self.parameter_count
            else self._signature_parameter_count(),
            precision_flags=PrecisionFlags(
                fp16=bool(self.dtype == torch.float16),
                bf16=bool(self.dtype == torch.bfloat16),
                fp8=bool(self._fp8_enabled),
                tf32=True,
            ),
        )

    def validate_result(self) -> Optional[str]:
        if torch.isnan(self.state_buffer).any():
            return "NaN detected in decode state"
        return None

    def teardown(self) -> None:
        # Release model buffers between variants to keep allocator usage low.
        # Explicitly clear CUDA graphs/streams to avoid teardown-time crashes in some
        # PyTorch/CUDA combinations (e.g., when the subprocess exits soon after replay).
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        self.decode_graph = None
        self.graph_stream = None
        self.copy_stream = None
        self.compute_stream = None

        for attr in (
            "embedding",
            "prefill_mlp",
            "decode_mlp",
            "lm_head",
            "host_prompt",
            "host_prompts",
            "gpu_prompt",
            "gpu_prompts",
            "host_payload",
            "gpu_payload",
            "host_payloads",
            "gpu_payloads",
            "state_buffer",
            "current_tokens",
            "next_token_out",
            "graph_logits",
            "graph_next_token",
            "_copy_done_events",
            "_timing_events",
        ):
            if hasattr(self, attr):
                setattr(self, attr, None)
        if torch.cuda.is_available():
            try:
                if hasattr(torch.cuda, "graph_pool_trim"):
                    torch.cuda.graph_pool_trim()
            except Exception:
                pass
            torch.cuda.empty_cache()
        self.output = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        # Decode variants mix CUDA graphs, torch.compile, custom streams, and TE state.
        # Running them in one long-lived interpreter is not robust: the labs minimal
        # sweep hit a teardown-time segfault after a graph variant completed timing.
        # Per-variant subprocess isolation is the correct benchmark-local fix.
        return BenchmarkConfig(
            iterations=self.cfg.iterations,
            warmup=self.cfg.warmup,
            percentiles=[50, 90, 99],
            use_subprocess=True,
        )

    def get_custom_metrics(self) -> Dict[str, float]:
        return self._custom_metrics
