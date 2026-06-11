"""labs.moe_cuda/optimized_router_vectorized.py - Grouped expert dispatch step.

This step groups token assignments by expert to run larger GEMMs, then wraps the
forward pass in a CUDA graph to remove Python dispatch overhead. Baseline uses a
per-token weight gather path; this variant is the standard MoE grouped dispatch.
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

# Kill-switch: AISP_MOE_CUDA_COMPILE=0 restores the pre-compile (B45) behavior
# of capturing the eager forward. Default ON.
_COMPILE_ENV = "AISP_MOE_CUDA_COMPILE"

# Opt-in (Front M2): AISP_MOE_CUDA_GELU_EPILOGUE=1 forces Inductor to fuse the
# GELU into the GEMM1 Triton-template epilogue (benchmark_epilogue_fusion=False).
# Default OFF: measured ~1.03x end-to-end (below the 1.05x gate) because the
# erf cost is conserved — it moves into the template epilogue (+58us) instead
# of disappearing with the standalone kernel (52-69us). At B51 the fusion was
# refused by ACCIDENT: the single benchmarked epilogue candidate
# (max_epilogue_benchmarked_choices=1, BLOCK 128/128/128 stages=4) fails Triton
# smem precompile on SM103 (262160B > 232448B) -> ms_fused=inf -> refuse.
_GELU_EPILOGUE_ENV = "AISP_MOE_CUDA_GELU_EPILOGUE"

# Front M3 (default ON; kill switch AISP_MOE_CUDA_GEMM1_ATEN=0 restores the
# B51 GEMM routing): rewrites both expert GEMMs from baddbmm(bias, x, w) to
# bmm(x, w) + bias so the honest autotuner routes GEMM1 to the extern nvjet
# kernel (59.2us, beta-zero) instead of the ~104us Triton template, with the
# bias adds fused into adjacent pointwise kernels (GELU for GEMM1, the
# gather-back for GEMM2) for free. GEMM2 was already extern at B51 but paid a
# hidden 40us broadcast bias-copy; same fix applies. Measured 1.31x median
# end-to-end vs B51 (0.4065 -> 0.3114 ms, 6x6 interleaved reps, bit-identical
# output on the lab input).
# Why not just pin max_autotune_gemm_backends="ATEN": the B54 "autotune
# mis-benchmark" hypothesis is FALSIFIED — extern baddbmm with the broadcast
# bias stride [2048, 0, 1] GENUINELY costs ~0.14ms because at::baddbmm_out
# materializes the bias into the output buffer first (74.6us broadcast
# direct_copy kernel) and then runs nvjet with beta=1 (66.9us); Inductor's
# 0.1402ms benchmark of that choice is honest, and the backend pin measured
# WORSE end-to-end (369.8 vs 313.4 GPU us/call, Front M3 probes r02/r03).
# The bias copy is a dataflow cost, so the fix is to remove the broadcast
# bias from the GEMM op, not to override the tuner.
# Requires the compile path (ignored under AISP_MOE_CUDA_COMPILE=0: eager
# bmm + broadcast-add costs ~150us extra, only Inductor fuses it for free).
# Mutually exclusive with AISP_MOE_CUDA_GELU_EPILOGUE (that flag's loud-assert
# semantics assume GEMM1 is a Triton template that GELU can fuse into): an
# explicit GELU_EPILOGUE=1 turns the default off; setting BOTH explicitly is a
# loud error in setup().
_GEMM1_ATEN_ENV = "AISP_MOE_CUDA_GEMM1_ATEN"

# Front M4 (default ON; kill switch AISP_MOE_CUDA_CAPACITY_RIGHTSIZE=0
# restores the B56 2x-padded capacity): calibrate the static
# per-expert slot capacity to the OBSERVED max tokens/expert (rounded up to a
# multiple of 64) instead of 2x that value. At the lab input (seed-42,
# batch 4096 x top-2 = 8192 assignments over 32 experts, max load 309) the
# incumbent capacity is 640 -> 20480 dense GEMM rows of which only 8192 are
# live (60% padding); right-sized capacity is 320 -> 10240 rows (20% padding),
# halving the M dim of both expert GEMMs and the GELU/dispatch pointwise work.
# CORRECTNESS: capacity is a correctness parameter under static shapes (the
# manual CUDA graph requires them). Right-sizing stays static — calibrated
# once in setup() on the lab input — and arms a LOUD overflow guard: the
# forward accumulates an in-graph overflow counter (sticky across replays)
# that is checked host-side in setup() and again after the timed reps; any
# routed assignment beyond capacity FAILS the run instead of being silently
# dropped to the trash row. Semantics on a non-overflowing input are
# unchanged: padded rows are inert in bmm/GELU, and live rows are gathered
# back by the same slot indices.
_CAPACITY_RIGHTSIZE_ENV = "AISP_MOE_CUDA_CAPACITY_RIGHTSIZE"

# Test/tuning override: AISP_MOE_CUDA_CAPACITY=<int> forces an explicit static
# capacity. The overflow guard is armed for any override, so a too-small
# capacity fails loudly (used by the Front M4 evidence to demonstrate the
# guard firing) — tokens are never silently dropped.
_CAPACITY_OVERRIDE_ENV = "AISP_MOE_CUDA_CAPACITY"


def _compile_enabled() -> bool:
    return os.environ.get(_COMPILE_ENV, "1").strip().lower() not in ("0", "false", "off")


def _gelu_epilogue_enabled() -> bool:
    return os.environ.get(_GELU_EPILOGUE_ENV, "0").strip().lower() in ("1", "true", "on")


def _gemm1_aten_enabled() -> bool:
    raw = os.environ.get(_GEMM1_ATEN_ENV)
    if raw is None:
        # Default ON (Front M3 win); yields to an explicit
        # AISP_MOE_CUDA_GELU_EPILOGUE=1, whose loud-assert semantics need the
        # GEMM1 Triton template that this rewrite removes.
        return not _gelu_epilogue_enabled()
    return raw.strip().lower() in ("1", "true", "on")


def _capacity_rightsize_enabled() -> bool:
    # Default ON (Front M4 win, 1.32x); =0 restores B56's 2x-padded capacity.
    return os.environ.get(_CAPACITY_RIGHTSIZE_ENV, "1").strip().lower() in ("1", "true", "on")


def _capacity_override() -> Optional[int]:
    raw = os.environ.get(_CAPACITY_OVERRIDE_ENV)
    if raw is None or not raw.strip():
        return None
    value = int(raw.strip())
    if value < 1:
        raise ValueError(f"{_CAPACITY_OVERRIDE_ENV} must be a positive integer, got {value}")
    return value


class VectorizedTopKMoE(nn.Module):
    """Top-k router with batched expert MLPs and scatter accumulation."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, expansion: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expanded = hidden_size * expansion
        self.router = nn.Linear(hidden_size, num_experts)

        # Pack expert weights for vectorized matmuls: [E, H, H*exp] and [E, H*exp, H].
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_size, self.expanded))
        self.b1 = nn.Parameter(torch.zeros(num_experts, self.expanded))
        self.w2 = nn.Parameter(torch.empty(num_experts, self.expanded, hidden_size))
        self.b2 = nn.Parameter(torch.zeros(num_experts, hidden_size))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # pragma: no cover - benchmarked
        logits = self.router(tokens)
        top_scores, expert_ids = torch.topk(logits, self.top_k, dim=-1)
        probs = torch.softmax(top_scores, dim=-1, dtype=tokens.dtype)

        flat_tokens = tokens.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, self.hidden_size)
        flat_expert_ids = expert_ids.reshape(-1)
        flat_probs = probs.reshape(-1, 1).to(tokens.dtype)

        w1 = self.w1[flat_expert_ids]
        b1 = self.b1[flat_expert_ids]
        # Avoid baddbmm meta-shape expand issues by separating matmul + bias add
        hidden = torch.bmm(flat_tokens.unsqueeze(1), w1).squeeze(1) + b1
        hidden = F.gelu(hidden)

        w2 = self.w2[flat_expert_ids]
        b2 = self.b2[flat_expert_ids]
        expert_out = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2
        weighted = expert_out * flat_probs

        output = torch.zeros_like(tokens, dtype=tokens.dtype)
        token_indices = torch.arange(tokens.shape[0], device=tokens.device).repeat_interleave(self.top_k)
        output.index_add_(0, token_indices, weighted)
        return output


class GroupedTopKMoE(VectorizedTopKMoE):
    """Top-k router that groups assignments by expert (larger matmuls).

    Dispatch is a sortless fixed-capacity dense layout so every op in the
    forward has a static shape: rank-within-expert via one-hot cumsum,
    index_copy_ into ``[num_experts * capacity, hidden]`` slots, batched GEMMs
    over all experts at once, then gather back by the same slot indices (which
    restores original token order for free). No ``nonzero``/boolean masking
    and no per-expert Python loop, so the whole forward is CUDA-graph
    capturable as a single graph.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, expansion: int = 2) -> None:
        super().__init__(hidden_size, num_experts, top_k, expansion)
        self.capacity: Optional[int] = None
        # Front M3 (AISP_MOE_CUDA_GEMM1_ATEN): bmm + deferred bias instead of
        # baddbmm. Only set this under torch.compile — eagerly the standalone
        # broadcast bias adds cost ~150us each; Inductor fuses them for free.
        self.use_bmm_bias_fuse: bool = False
        # Front M4 (AISP_MOE_CUDA_CAPACITY_RIGHTSIZE): calibrate capacity to
        # the observed max load (round-64) instead of 2x it; capacity_override
        # forces an explicit value. Either one arms capacity_guard: the
        # forward accumulates routed-beyond-capacity assignments into the
        # in-graph overflow_total buffer (sticky across graph replays), which
        # the benchmark checks host-side — overflow FAILS loudly, tokens are
        # never silently dropped.
        self.capacity_rightsize: bool = False
        self.capacity_override: Optional[int] = None
        self.capacity_guard: bool = False
        self.register_buffer("overflow_total", torch.zeros((), dtype=torch.int64), persistent=False)

    @torch.no_grad()
    def calibrate_capacity(self, tokens: torch.Tensor) -> int:
        """Pick the static per-expert slot capacity from observed routing.

        Must run eagerly (it is the ONLY host sync of the optimized path);
        call it once in ``setup()`` BEFORE graph capture. Capacity is 2x the
        observed max tokens/expert, rounded up to a multiple of 64, clamped to
        the routed-token count. Front M4: with ``capacity_rightsize`` the 2x
        overprovision is dropped (capacity = max load rounded up to 64) and
        the loud overflow guard covers the (static) residual risk; a
        ``capacity_override`` wins over both.
        """
        if self.capacity_override is not None:
            self.capacity = self.capacity_override
            return self.capacity
        logits = self.router(tokens)
        _, expert_ids = torch.topk(logits, self.top_k, dim=-1)
        counts = torch.bincount(expert_ids.reshape(-1), minlength=self.num_experts)
        max_load = int(counts.max().item())
        total_assignments = tokens.shape[0] * self.top_k
        factor = 1 if self.capacity_rightsize else 2
        capacity = min(((max_load * factor + 63) // 64) * 64, total_assignments)
        self.capacity = max(capacity, 64)
        return self.capacity

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # pragma: no cover - benchmarked
        if self.capacity is None:
            self.calibrate_capacity(tokens)
        capacity = self.capacity
        assert capacity is not None

        logits = self.router(tokens)
        top_scores, expert_ids = torch.topk(logits, self.top_k, dim=-1)
        probs = torch.softmax(top_scores, dim=-1, dtype=tokens.dtype)

        batch = tokens.shape[0]
        flat_tokens = tokens.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, self.hidden_size)
        flat_expert_ids = expert_ids.reshape(-1)
        flat_probs = probs.reshape(-1, 1).to(tokens.dtype)
        token_indices = torch.arange(batch, device=tokens.device).repeat_interleave(self.top_k)

        # Rank of each assignment within its expert: one-hot cumsum (sortless,
        # static shapes, no nonzero). Layout is [E, N] so the scan runs along
        # the contiguous inner dim (the [N, E] outer-dim scan has only E
        # parallel lanes and is ~2 ms; this form is ~10 us).
        expert_range = torch.arange(self.num_experts, device=tokens.device).unsqueeze(1)
        one_hot_t = (flat_expert_ids.unsqueeze(0) == expert_range).long()
        ranks = one_hot_t.cumsum(1).gather(0, flat_expert_ids.unsqueeze(0)).squeeze(0) - 1

        # Unique slot per assignment; overflow beyond capacity (never happens
        # with a calibrated capacity) is routed to a trash row so shapes stay
        # static and writes stay in-bounds.
        num_slots = self.num_experts * capacity
        valid = ranks < capacity
        slots = flat_expert_ids * capacity + ranks.clamp_max(capacity - 1)
        slots = torch.where(valid, slots, torch.full_like(slots, num_slots))

        if self.capacity_guard:
            # Front M4 AUDIT RULE: overflow must be LOUD, never a silent drop.
            # Sticky in-graph counter (accumulates across graph replays);
            # checked host-side in setup() and again after the timed reps —
            # any assignment routed beyond capacity fails the run.
            self.overflow_total.add_((~valid).sum())

        x_dense = flat_tokens.new_zeros(num_slots + 1, self.hidden_size)
        x_dense.index_copy_(0, slots, flat_tokens)
        x_grouped = x_dense[:num_slots].view(self.num_experts, capacity, self.hidden_size)

        # Batched expert MLPs over all experts at once; zero rows are inert.
        if self.use_bmm_bias_fuse:
            # Front M3: pure bmm lets the honest autotuner pick the extern
            # beta-zero nvjet kernel (59.2us vs the 104us Triton template that
            # wins for baddbmm); Inductor fuses the bias adds into the GELU /
            # gather-back pointwise kernels for free. baddbmm's broadcast bias
            # is what makes the extern choice lose autotune HONESTLY:
            # at::baddbmm_out materializes the bias into the output buffer
            # (74.6us broadcast copy) before an nvjet beta=1 GEMM (0.1402ms
            # benchmarked total vs 0.1080ms for the template).
            hidden = F.gelu(torch.bmm(x_grouped, self.w1) + self.b1.unsqueeze(1))
            expert_out = torch.bmm(hidden, self.w2) + self.b2.unsqueeze(1)
        else:
            # baddbmm folds the bias add into the GEMM epilogue (the
            # standalone broadcast adds cost ~150 us each at these shapes —
            # under torch.compile see use_bmm_bias_fuse above).
            hidden = torch.baddbmm(self.b1.unsqueeze(1), x_grouped, self.w1)
            hidden = F.gelu(hidden)
            expert_out = torch.baddbmm(self.b2.unsqueeze(1), hidden, self.w2)

        # Gather back by the same slot indices (restores token order) and
        # zero out any overflow rows before the weighted scatter-accumulate.
        out_flat = expert_out.reshape(num_slots, self.hidden_size)
        gathered = out_flat[slots.clamp_max(num_slots - 1)]
        weighted = gathered * (flat_probs * valid.unsqueeze(1).to(tokens.dtype))

        output = torch.zeros_like(tokens, dtype=tokens.dtype)
        output.index_add_(0, token_indices, weighted)
        return output


class VectorizedRouterBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark for grouped top-k router with CUDA graphs."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 1024
        self.num_experts = 32
        self.top_k = 2
        # Match baseline batch_size for fair comparison
        self.batch_size = 4096
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_output: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._compile_enabled: bool = False
        self._gelu_epilogue_enabled: bool = False
        self._gemm1_aten_enabled: bool = False
        self._capacity_rightsize_enabled: bool = False
        self._capacity: Optional[int] = None
        self._compile_seconds: Optional[float] = None
        self._dynamo_graph_breaks: Optional[int] = None
        self._dynamo_unique_graphs: Optional[int] = None
        self._kernels_per_replay: Optional[float] = None
        tokens = self.batch_size * self.top_k
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        import gc

        # CRITICAL: Clean up CUDA state from previous benchmarks
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        try:
            if hasattr(torch.cuda, 'graph_pool_trim'):
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

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = GroupedTopKMoE(self.hidden_size, self.num_experts, self.top_k, expansion=2)
        model = model.to(self.device, dtype=torch.bfloat16)
        model.eval()
        self.model = model

        # Use CPU randn + to(device) to avoid CUDA RNG graph capture issues
        self.inputs = torch.randn(
            self.batch_size,
            self.hidden_size,
            dtype=torch.bfloat16,
        ).to(self.device)

        self.output = None

        # Calibrate the static per-expert slot capacity and warm up kernels
        # (cuBLAS workspaces, one-hot/cumsum) eagerly BEFORE graph capture:
        # the only host sync of the optimized path lives in calibrate_capacity().
        # Front M4: right-sizing / explicit override arm the loud overflow
        # guard (see GroupedTopKMoE); the incumbent default path is untouched.
        self._capacity_rightsize_enabled = _capacity_rightsize_enabled()
        capacity_override = _capacity_override()
        model.capacity_rightsize = self._capacity_rightsize_enabled
        model.capacity_override = capacity_override
        model.capacity_guard = self._capacity_rightsize_enabled or capacity_override is not None
        self._capacity = model.calibrate_capacity(self.inputs)
        with torch.no_grad():
            for _ in range(3):
                model(self.inputs)
        torch.cuda.synchronize(self.device)
        self._check_capacity_overflow("eager warmup")

        # torch.compile the static forward so Inductor fuses the GELU/dispatch
        # epilogues, then manually capture the COMPILED callable below.
        # Composition: Inductor's own cudagraph machinery stays OFF
        # ("max-autotune-no-cudagraphs") because the manual torch.cuda.CUDAGraph
        # capture owns the graph; replay() keeps the zero-Python hot path.
        # AISP_MOE_CUDA_COMPILE=0 restores the B45 eager-capture behavior.
        self._compile_enabled = _compile_enabled()
        self._gelu_epilogue_enabled = self._compile_enabled and _gelu_epilogue_enabled()
        self._gemm1_aten_enabled = self._compile_enabled and _gemm1_aten_enabled()
        if self._gelu_epilogue_enabled and self._gemm1_aten_enabled:
            raise RuntimeError(
                f"{_GELU_EPILOGUE_ENV}=1 and {_GEMM1_ATEN_ENV}=1 are mutually exclusive: "
                "epilogue fusion needs a Triton GEMM template, the ATEN pin removes it"
            )
        self._compile_seconds = None
        self._dynamo_graph_breaks = None
        self._dynamo_unique_graphs = None
        self._kernels_per_replay = None
        forward_fn = self.model
        capture_no_grad = False
        if self._compile_enabled:
            from torch._dynamo.utils import counters as dynamo_counters

            if self._gelu_epilogue_enabled:
                # Front M2: Inductor's benchmark-driven template-epilogue fusion
                # refuses the GEMM1+GELU fusion (its sole benchmarked candidate
                # smem-OOMs on SM103 -> inf slowdown). Disabling the benchmark
                # makes the scheduler fuse heuristically: the standalone
                # triton_poi_fused_gelu kernel disappears into
                # triton_tem_fused_baddbmm_gelu_*. Changing this config also
                # changes the Inductor cache key, so arms never share artifacts.
                import torch._inductor.config as inductor_config

                inductor_config.benchmark_epilogue_fusion = False

            if self._gemm1_aten_enabled:
                # Front M3: switch the forward to the bmm + fused-bias dataflow
                # (see use_bmm_bias_fuse in GroupedTopKMoE). Dynamo guards on
                # the attribute, and the changed graph changes the Inductor
                # cache key, so the arms never share compiled artifacts.
                assert isinstance(self.model, GroupedTopKMoE)
                self.model.use_bmm_bias_fuse = True

            dynamo_counters.clear()
            compile_start = time.perf_counter()
            compiled = torch.compile(
                self.model,
                fullgraph=True,  # AUDIT RULE: any graph break must fail loudly
                dynamic=False,
                mode="max-autotune-no-cudagraphs",
            )
            # Compile + autotune happen here (setup, untimed). Warmup and the
            # capture below MUST share grad mode, otherwise capture triggers a
            # Dynamo recompile whose autotuning would be captured into the graph.
            with torch.no_grad():
                for _ in range(3):
                    compiled(self.inputs)
            torch.cuda.synchronize(self.device)
            self._compile_seconds = time.perf_counter() - compile_start

            graph_breaks = sum(dynamo_counters["graph_break"].values())
            unique_graphs = int(dynamo_counters["stats"].get("unique_graphs", 0))
            self._dynamo_graph_breaks = graph_breaks
            self._dynamo_unique_graphs = unique_graphs
            if graph_breaks != 0:
                raise RuntimeError(
                    f"torch.compile path has {graph_breaks} graph break(s); refusing silent partial compile"
                )
            if unique_graphs < 1:
                raise RuntimeError(
                    "torch.compile silently fell back to eager (0 unique graphs compiled); "
                    "refusing to run the un-compiled path as the optimized arm"
                )
            forward_fn = compiled
            capture_no_grad = True

        # Capture the forward pass into a CUDA graph to hide Python dispatch overhead.
        self.graph = None
        self.static_output = None
        try:
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                assert forward_fn is not None and self.inputs is not None
                if capture_no_grad:
                    with torch.no_grad():
                        self.static_output = forward_fn(self.inputs)
                else:
                    self.static_output = forward_fn(self.inputs)
            torch.cuda.synchronize(self.device)
        except Exception:
            self.graph = None
            self.static_output = None
            if self._compile_enabled:
                # AUDIT RULE (B45): no silent eager fallback on the compiled path.
                raise
        if self._compile_enabled and self.graph is None:
            raise RuntimeError("CUDA-graph capture of the compiled forward did not produce a graph")

        # AUDIT (Front M2): count kernels per replay and prove the fusion state.
        # Profiling lives in setup() (untimed); exported via get_custom_metrics.
        if self.graph is not None:
            from torch.profiler import ProfilerActivity, profile as torch_profile

            torch.cuda.synchronize(self.device)
            with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    self.graph.replay()
                torch.cuda.synchronize(self.device)
            kernel_count = 0.0
            standalone_gelu = []
            template_bmm = []
            for evt in prof.key_averages():
                if (getattr(evt, "self_device_time_total", 0) or 0) <= 0:
                    continue
                kernel_count += evt.count
                if "fused_gelu" in evt.key and evt.key.startswith("triton_poi"):
                    standalone_gelu.append(evt.key)
                if "triton_tem" in evt.key and "bmm" in evt.key:
                    template_bmm.append(evt.key)
            self._kernels_per_replay = kernel_count / 5.0
            if self._gelu_epilogue_enabled and standalone_gelu:
                # AUDIT RULE (B45/B51): the opt-in fusion path must fail loudly
                # if the standalone GELU kernel survived (fusion silently refused).
                raise RuntimeError(
                    "AISP_MOE_CUDA_GELU_EPILOGUE=1 but a standalone GELU kernel "
                    f"is still launched per replay: {standalone_gelu}"
                )
            if self._gemm1_aten_enabled and template_bmm:
                # AUDIT RULE (B45/B51): the opt-in extern-GEMM path must fail
                # loudly if a Triton bmm/baddbmm template still runs (the
                # extern routing was silently lost, e.g. an autotune flip).
                raise RuntimeError(
                    "AISP_MOE_CUDA_GEMM1_ATEN=1 but a Triton GEMM template "
                    f"kernel still runs per replay: {template_bmm}"
                )
            # Front M4: the warmups/capture/profile replays above all ran the
            # in-graph overflow counter; a calibrated-too-small capacity fails
            # HERE, before any timed rep.
            self._check_capacity_overflow("compile warmup / graph capture / profile replays")

    def _check_capacity_overflow(self, phase: str) -> None:
        """Front M4 loud overflow guard (host side, untimed phases only).

        Reads the sticky in-graph counter the forward accumulates whenever
        capacity right-sizing / an explicit capacity override is active. Any
        routed assignment beyond the static capacity is a correctness failure
        (the trash row would silently drop it), so the run aborts.
        """
        model = self.model
        if not isinstance(model, GroupedTopKMoE) or not model.capacity_guard:
            return
        torch.cuda.synchronize(self.device)
        overflow = int(model.overflow_total.item())
        if overflow != 0:
            raise RuntimeError(
                f"capacity overflow guard tripped during {phase}: {overflow} "
                f"assignment-pass(es) routed beyond static capacity={model.capacity}; "
                "refusing to silently drop tokens — raise "
                f"{_CAPACITY_OVERRIDE_ENV} or unset {_CAPACITY_RIGHTSIZE_ENV}"
            )

    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None:
            raise RuntimeError("Model not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("moe_cuda_router_vectorized", enable=enable_nvtx):
            if self.graph is not None:
                self.graph.replay()
                if self.static_output is None:
                    raise RuntimeError("CUDA graph replay missing static output buffer")
                self.output = self.static_output
            else:
                with torch.inference_mode():
                    self.output = self.model(self.inputs)
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.inputs is None or self.output is None or self.model is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        # Front M4: every timed replay accumulated the in-graph overflow
        # counter; check it before the output is allowed to verify.
        self._check_capacity_overflow("timed replays")
        self._set_verification_payload(
            inputs={"input": self.inputs.detach()},
            output=self.output.detach().float().clone(),
            batch_size=self.batch_size,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            precision_flags={"bf16": True, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.graph = None
        self.static_output = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        # setup_timeout covers the one-time torch.compile/autotune cost, which
        # lives in setup() (untimed) and is reported via compile_seconds.
        return BenchmarkConfig(iterations=10, warmup=10, setup_timeout_seconds=600)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline analysis metrics."""
        # Estimate problem size for roofline analysis
        n = getattr(self, 'N', 0) or getattr(self, 'hidden_dim', 0) or 4096
        batch = getattr(self, 'batch_size', 1) or getattr(self, 'batch', 1)
        # Simple FLOP estimate for linear layers
        flops = 2.0 * batch * n * n  # Rough estimate
        bytes_moved = batch * n * 4.0  # Input/output bytes
        arithmetic_intensity = flops / max(bytes_moved, 1.0)
        metrics = {
            "router_vectorized.estimated_flops": flops,
            "router_vectorized.estimated_bytes": bytes_moved,
            "router_vectorized.arithmetic_intensity": arithmetic_intensity,
            # AUDIT RULE (B45): prove the captured/compiled path is live.
            "router_vectorized.manual_graph_active": 1.0 if self.graph is not None else 0.0,
            "router_vectorized.compile_enabled": 1.0 if self._compile_enabled else 0.0,
            "router_vectorized.gelu_epilogue_enabled": 1.0 if self._gelu_epilogue_enabled else 0.0,
            "router_vectorized.gemm1_aten_enabled": 1.0 if self._gemm1_aten_enabled else 0.0,
            # Front M4: prove which static capacity the dense dispatch used.
            "router_vectorized.capacity_rightsize_enabled": 1.0 if self._capacity_rightsize_enabled else 0.0,
        }
        if self._capacity is not None:
            metrics["router_vectorized.capacity"] = float(self._capacity)
        if self._kernels_per_replay is not None:
            metrics["router_vectorized.kernels_per_replay"] = float(self._kernels_per_replay)
        if self._compile_seconds is not None:
            metrics["router_vectorized.compile_seconds"] = float(self._compile_seconds)
        if self._dynamo_graph_breaks is not None:
            metrics["router_vectorized.dynamo_graph_breaks"] = float(self._dynamo_graph_breaks)
        if self._dynamo_unique_graphs is not None:
            metrics["router_vectorized.dynamo_unique_graphs"] = float(self._dynamo_unique_graphs)
        return metrics

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Vectorized router missing"
        if self.inputs is None:
            return "Inputs missing"
        return None

def get_benchmark() -> BaseBenchmark:
    return VectorizedRouterBenchmark()
