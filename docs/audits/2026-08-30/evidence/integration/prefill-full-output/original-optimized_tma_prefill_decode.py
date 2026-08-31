"""Optimized prefill vs. decode microbench with simple TMA burst shaping.

What it demonstrates for Nsight Systems:
- Prefill: double-buffered-ish pipeline using multiple streams + max_in_flight
  guard to show how shaping reduces contention.
- Decode: graph-captured token loop to trim host launch gaps.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import os
from enum import Enum
from typing import Optional

import torch

from core.benchmark.blackwell_requirements import ensure_blackwell_tma_supported
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.harness.cuda_capabilities import tma_support_status
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    get_stream_priorities,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)


def _enable_blackwell_compiler_defaults() -> None:
    """Turn on Blackwell-friendly defaults for torch.compile/Inductor."""
    try:
        from torch._inductor import config as triton_cfg  # type: ignore

        if hasattr(triton_cfg, "tma_support"):
            triton_cfg.tma_support = True
        if hasattr(triton_cfg.cuda, "enable_tma"):
            triton_cfg.cuda.enable_tma = True
    except Exception:
        # Inductor may be unavailable in stripped-down builds; fail soft.
        pass


class GraphMode(Enum):
    FULL = "full"
    PIECEWISE = "piecewise"
    FULL_AND_PIECEWISE = "full_and_piecewise"

    @classmethod
    def from_str(cls, raw: str | None) -> "GraphMode":
        normalized = (raw or cls.FULL_AND_PIECEWISE.value).strip().lower().replace("-", "_")
        for mode in cls:
            if normalized == mode.value:
                return mode
        return cls.FULL_AND_PIECEWISE


class TmaBurstConfig:
    """Simple config holder for burst shaping knobs."""

    def __init__(self, chunk_k: int = 128, max_in_flight: int = 2) -> None:
        self.chunk_k = chunk_k  # tile size (elements) for each cp.async bulk transfer
        self.max_in_flight = max_in_flight


_TMA_CP_ASYNC_EXT: object | None = None


def _load_cp_async_tma_ext() -> object:
    """Compile and return a tiny extension that issues cp.async.bulk.tensor 1D copies."""
    global _TMA_CP_ASYNC_EXT
    if _TMA_CP_ASYNC_EXT is not None:
        return _TMA_CP_ASYNC_EXT

    repo_root = Path(__file__).resolve().parent.parent.parent
    include_dir = repo_root / "core" / "common" / "headers"

    cpp_src = r"""
#include <torch/extension.h>
void tma_copy_tile(torch::Tensor src, torch::Tensor dst, int64_t tile_elems);
"""

    cuda_src = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda/barrier>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <utility>

#include "tma_helpers.cuh"

#if CUDART_VERSION < 13000
void tma_copy_tile(torch::Tensor /*src*/, torch::Tensor /*dst*/, int64_t /*tile_elems*/) {
    TORCH_CHECK(false, "tma_copy_tile: CUDA 13.0+ required for cp.async.bulk.tensor");
}
#else

namespace {
namespace cde = cuda::device::experimental;
constexpr int THREADS = 256;
#define PIPELINE_STAGES 2

template <int TILE>
__global__ void tma_copy_kernel(const __grid_constant__ CUtensorMap in_desc,
                                const __grid_constant__ CUtensorMap out_desc,
                                int total_tiles) {
    constexpr std::size_t BYTES_PER_TILE = static_cast<std::size_t>(TILE) * sizeof(float);
    __shared__ alignas(128) float stage_buffers[PIPELINE_STAGES][TILE];
    using block_barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(block_barrier) unsigned char barrier_storage[PIPELINE_STAGES][sizeof(block_barrier)];

    if (threadIdx.x == 0) {
        for (int stage = 0; stage < PIPELINE_STAGES; ++stage) {
            init(reinterpret_cast<block_barrier*>(barrier_storage[stage]), blockDim.x);
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    cuda::barrier<cuda::thread_scope_block>::arrival_token tokens[PIPELINE_STAGES];

    auto issue = [&](int tile_idx) {
        if (tile_idx >= total_tiles) {
            return;
        }
        const int stage = tile_idx % PIPELINE_STAGES;
        auto* bar_ptr = reinterpret_cast<block_barrier*>(barrier_storage[stage]);
        auto& bar = *bar_ptr;

        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_1d_global_to_shared(
                &stage_buffers[stage],
                &in_desc,
                tile_idx * TILE,
                bar);
            tokens[stage] = cuda::device::barrier_arrive_tx(bar, 1, BYTES_PER_TILE);
        } else {
            tokens[stage] = bar.arrive();
        }
    };

    const int preload = std::min(total_tiles, PIPELINE_STAGES);
    for (int t = 0; t < preload; ++t) {
        issue(t);
    }

    for (int tile = 0; tile < total_tiles; ++tile) {
        const int stage = tile % PIPELINE_STAGES;
        auto* bar_ptr = reinterpret_cast<block_barrier*>(barrier_storage[stage]);
        auto& bar = *bar_ptr;

        bar.wait(std::move(tokens[stage]));
        __syncthreads();

        // Optional math hook (prefill is a straight copy so this is a no-op).
        float* tile_ptr = stage_buffers[stage];
        for (int i = threadIdx.x; i < TILE; i += blockDim.x) {
            tile_ptr[i] = tile_ptr[i];
        }
        cde::fence_proxy_async_shared_cta();
        __syncthreads();

        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_1d_shared_to_global(
                &out_desc,
                tile * TILE,
                &stage_buffers[stage]);
            cde::cp_async_bulk_commit_group();
            cde::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();

        const int next = tile + PIPELINE_STAGES;
        if (next < total_tiles) {
            issue(next);
        }
    }
}

template <int TILE>
void launch_kernel(const CUtensorMap& in_desc, const CUtensorMap& out_desc, int total_tiles, cudaStream_t stream) {
    tma_copy_kernel<TILE><<<1, THREADS, 0, stream>>>(in_desc, out_desc, total_tiles);
}

}  // namespace

void tma_copy_tile(torch::Tensor src, torch::Tensor dst, int64_t tile_elems) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "tma_copy_tile: src/dst must be CUDA");
    TORCH_CHECK(src.scalar_type() == torch::kFloat && dst.scalar_type() == torch::kFloat,
                "tma_copy_tile: only float32 supported");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(),
                "tma_copy_tile: tensors must be contiguous");
    TORCH_CHECK(src.numel() == dst.numel(), "tma_copy_tile: size mismatch");
    TORCH_CHECK(tile_elems > 0, "tma_copy_tile: tile_elems must be positive");

    if (!cuda_tma::device_supports_tma()) {
        TORCH_CHECK(false, "tma_copy_tile: device lacks TMA support");
    }

    const int elems = static_cast<int>(src.numel());
    int tile = static_cast<int>(tile_elems);
    const auto limits = cuda_arch::get_tma_limits();
    tile = std::min(tile, static_cast<int>(limits.max_1d_box_size));
    TORCH_CHECK(tile > 0, "tma_copy_tile: invalid tile size after clamp");

    c10::cuda::CUDAGuard guard(src.get_device());
    auto encode = cuda_tma::load_cuTensorMapEncodeTiled();
    TORCH_CHECK(encode, "tma_copy_tile: cuTensorMapEncodeTiled unavailable");

    CUtensorMap in_desc{};
    CUtensorMap out_desc{};
    bool ok = cuda_tma::make_1d_tensor_map(in_desc, encode, src.data_ptr(), elems, tile) &&
              cuda_tma::make_1d_tensor_map(out_desc, encode, dst.data_ptr(), elems, tile);
    TORCH_CHECK(ok, "tma_copy_tile: descriptor creation failed");

    const int total_tiles = (elems + tile - 1) / tile;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (tile) {
        case 64: launch_kernel<64>(in_desc, out_desc, total_tiles, stream); break;
        case 128: launch_kernel<128>(in_desc, out_desc, total_tiles, stream); break;
        case 256: launch_kernel<256>(in_desc, out_desc, total_tiles, stream); break;
        default: launch_kernel<128>(in_desc, out_desc, total_tiles, stream); break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tma_copy_tile launch failed: ", cudaGetErrorString(err));
}
#endif  // CUDART_VERSION < 13000

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tma_copy_tile", &tma_copy_tile, "cp.async.bulk.tensor 1D copy");
}
"""

    try:
        from torch.utils.cpp_extension import load_inline
    except ImportError as exc:  # pragma: no cover - torch always available in CI
        raise RuntimeError(f"SKIPPED: torch extension loader unavailable ({exc})") from exc

    try:
        _TMA_CP_ASYNC_EXT = load_inline(
            name="persistent_decode_tma_cp_async_ext",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=None,
            extra_cuda_cflags=[
                "--std=c++17",
                "--use_fast_math",
                "-lineinfo",
                "-gencode=arch=compute_100,code=sm_100",
                "-gencode=arch=compute_103,code=sm_103",
                "-gencode=arch=compute_120,code=sm_120",
                "-gencode=arch=compute_121,code=sm_121",
            ],
            extra_ldflags=["-lcuda"],
            extra_include_paths=[str(include_dir)],
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(f"SKIPPED: failed to build TMA cp.async extension ({exc})") from exc

    return _TMA_CP_ASYNC_EXT


class OptimizedTmaPrefillDecodeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Prefill with shaped cp.async.bulk.tensor bursts + graph-captured decode."""

    def __init__(self, *, graph_mode: "GraphMode | None" = None, max_capture_seq: int | None = None) -> None:
        super().__init__()
        _enable_blackwell_compiler_defaults()
        self.device = resolve_device()
        self.inputs = None
        self.batch, self.seq_len, self.head_dim = resolve_shapes()
        self.batch_size = self.batch
        self.hidden_dim = self.head_dim
        self.prefill_chunks = 8
        self.prefill_chunk_elems = 128 * 128
        self.cfg = TmaBurstConfig()
        self._prio_low, self._prio_high = get_stream_priorities()
        # Keep custom stream count <= 2 to satisfy the harness StreamAuditor.
        # (multi-stream >2 triggers a timing violation to prevent under-timing exploits).
        self.prefill_streams = [torch.cuda.Stream(priority=self._prio_low)]
        self.decode_stream = torch.cuda.Stream(priority=self._prio_high)
        self.decode_graph = torch.cuda.CUDAGraph()
        self.full_graph: torch.cuda.CUDAGraph | None = None
        self.graph_q = None
        self.graph_k = None
        self.graph_v = None
        self.graph_out = None
        self.graph_mode = graph_mode or GraphMode.from_str(os.getenv("PD_GRAPH_MODE"))
        self.max_capture_seq = max_capture_seq or int(os.getenv("PD_MAX_CAPTURE_SEQ", self.seq_len))
        self._ttft_metric_values: list[float] = [0.0]
        self._tpot_metric_values: list[float] = [0.0] * self.seq_len
        self._iteration_metric_payload: dict[str, list[float]] = {
            "ttft_times_ms": self._ttft_metric_values,
            "tpot_times_ms": self._tpot_metric_values,
        }
        self._pending_graph_path: str | None = None
        self._pending_start: torch.cuda.Event | None = None
        self._pending_prefill_end: torch.cuda.Event | None = None
        self._pending_decode_start: torch.cuda.Event | None = None
        self._pending_decode_end: torch.cuda.Event | None = None
        self._tma_ext: object | None = None
        self._full_events: dict[str, torch.cuda.Event] = {}
        self._piecewise_events: dict[str, torch.cuda.Event] = {}
        self._prefill_events: list[torch.cuda.Event] = []
        self._prefill_work: list[
            tuple[torch.cuda.Stream, torch.Tensor, torch.Tensor, torch.cuda.Event]
        ] = []
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())
        self.output: torch.Tensor | None = None
        self._output_view: torch.Tensor | None = None
        self._verify_output_buffer: torch.Tensor | None = None

    def setup(self) -> None:
        ensure_blackwell_tma_supported("optimized_tma_prefill_decode")
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.inputs = build_inputs(self.device)
        self._output_view = self.inputs.out[:1, : min(8, self.inputs.out.shape[1])]
        self._verify_output_buffer = torch.empty_like(self._output_view, dtype=torch.float32)
        # Skip on GPUs without TMA support to avoid false regressions.
        supported, reason = tma_support_status()
        if not supported:
            raise RuntimeError(f"SKIP: TMA not supported on this GPU ({reason})")
        self._tma_ext = _load_cp_async_tma_ext()
        self.prefill_src = torch.randn(
            self.prefill_chunks, self.prefill_chunk_elems, device=self.device
        )
        self.prefill_dst = torch.empty_like(self.prefill_src)

        # Graph-captured decode loop to cut host gaps during profiling.
        self.graph_q = self.inputs.q.clone()
        self.graph_k = self.inputs.k.clone()
        self.graph_v = self.inputs.v.clone()
        self.graph_out = torch.empty_like(self.inputs.out)
        self._full_events = {
            "start": torch.cuda.Event(enable_timing=True),
            "end": torch.cuda.Event(enable_timing=True),
        }
        self._piecewise_events = {
            "start_prefill": torch.cuda.Event(enable_timing=True),
            "end_prefill": torch.cuda.Event(enable_timing=True),
            "start_decode": torch.cuda.Event(enable_timing=True),
            "end_decode": torch.cuda.Event(enable_timing=True),
        }
        self._prefill_events = [
            torch.cuda.Event(enable_timing=False, blocking=False)
            for _ in range(self.prefill_chunks)
        ]
        self._prefill_work = [
            (
                self.prefill_streams[idx % len(self.prefill_streams)],
                src,
                dst,
                event,
            )
            for idx, (src, dst, event) in enumerate(
                zip(
                    self.prefill_src.unbind(0),
                    self.prefill_dst.unbind(0),
                    self._prefill_events,
                    strict=True,
                )
            )
        ]

        torch.cuda.synchronize()
        with torch.cuda.graph(self.decode_graph, stream=self.decode_stream):
            self._decode_body(self.graph_q, self.graph_k, self.graph_v, self.graph_out)
        torch.cuda.synchronize()
        self._capture_full_graph()

    def _capture_full_graph(self) -> None:
        if self.graph_mode == GraphMode.PIECEWISE:
            return
        if self._tma_ext is None:
            raise RuntimeError("TMA extension not initialized")
        self.full_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.full_graph, stream=self.decode_stream):
            # Simplified full-iteration capture: single-stream cp.async.bulk.tensor prefill + captured decode.
            flat_src = self.prefill_src.reshape(-1)
            flat_dst = self.prefill_dst.reshape(-1)
            self._tma_ext.tma_copy_tile(flat_src, flat_dst, self.cfg.chunk_k)
            self._decode_body(self.graph_q, self.graph_k, self.graph_v, self.graph_out)
            self.inputs.out.copy_(self.graph_out)
        torch.cuda.synchronize()

    def _decode_body(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor
    ) -> None:
        # Simple per-token dot product.
        for t in range(self.seq_len):
            q_t = q[:, t, :]
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            dot = (q_t * k_t).sum(dim=-1, keepdim=True)
            out[:, t, :] = v_t * dot

    def _prefill_shaped(self, *, async_only: bool = False) -> deque[torch.cuda.Event] | None:
        """Launch cp.async.bulk.tensor copies on multiple streams with a max_in_flight cap."""
        if self._tma_ext is None:
            raise RuntimeError("TMA extension not initialized")
        if len(self._prefill_work) != self.prefill_chunks:
            raise RuntimeError("Prefill work not initialized")
        events: deque[torch.cuda.Event] = deque()
        for stream, src, dst, evt in self._prefill_work:
            with torch.cuda.stream(stream):
                self._tma_ext.tma_copy_tile(src, dst, self.cfg.chunk_k)
            evt.record(stream)
            events.append(evt)
            if len(events) > self.cfg.max_in_flight:
                stream.wait_event(events.popleft())

        if async_only:
            return events

        # Drain remaining work.
        current_stream = torch.cuda.current_stream()
        for evt in events:
            current_stream.wait_event(evt)
        return None

    def _decode_graph(self) -> None:
        assert self.inputs is not None
        # Refresh graph inputs to show a realistic copy-before-replay pattern.
        with torch.cuda.stream(self.decode_stream):
            self.graph_q.copy_(self.inputs.q)
            self.graph_k.copy_(self.inputs.k)
            self.graph_v.copy_(self.inputs.v)
            self.decode_graph.replay()
            # Mirror back to inputs.out so validation stays consistent.
            self.inputs.out.copy_(self.graph_out)

    def benchmark_fn(self) -> None:
        if self.inputs is None or self._output_view is None:
            raise RuntimeError("Inputs not initialized")

        current_stream = torch.cuda.current_stream()
        use_full = (
            self.graph_mode == GraphMode.FULL
            or (self.graph_mode == GraphMode.FULL_AND_PIECEWISE and self.seq_len <= self.max_capture_seq)
        )
        if use_full and self.full_graph is not None:
            start = self._full_events["start"]
            end = self._full_events["end"]
            with torch.inference_mode(), self._nvtx_range("full_graph_high_pri"):
                with torch.cuda.stream(self.decode_stream):
                    start.record(self.decode_stream)
                    self.full_graph.replay()
                    end.record(self.decode_stream)
            current_stream.wait_stream(self.decode_stream)
            self._pending_graph_path = "full_graph"
            self._pending_start = start
            self._pending_prefill_end = end
            self._pending_decode_start = start
            self._pending_decode_end = end
            if self.inputs is not None:
                self.output = self._output_view
            else:
                raise RuntimeError("Inputs not initialized for verification")
            return

        with torch.inference_mode(), self._nvtx_range("prefill_shaped_low_pri"):
            start_prefill = self._piecewise_events["start_prefill"]
            end_prefill = self._piecewise_events["end_prefill"]
            start_prefill.record(current_stream)
            pref_events = self._prefill_shaped(async_only=True)
        with torch.inference_mode(), self._nvtx_range(
            "decode_graph_high_pri" if self.graph_mode != GraphMode.FULL_AND_PIECEWISE else "graph_fallback_piecewise"
        ):
            start_decode = self._piecewise_events["start_decode"]
            end_decode = self._piecewise_events["end_decode"]
            with torch.cuda.stream(self.decode_stream):
                start_decode.record(self.decode_stream)
                self._decode_graph()
                end_decode.record(self.decode_stream)
        if pref_events:
            for evt in pref_events:
                current_stream.wait_event(evt)
        end_prefill.record(current_stream)
        current_stream.wait_stream(self.decode_stream)
        self._pending_graph_path = "piecewise_graph"
        self._pending_start = start_prefill
        self._pending_prefill_end = end_prefill
        self._pending_decode_start = start_decode
        self._pending_decode_end = end_decode
        if self.inputs is not None:
            self.output = self._output_view
        else:
            raise RuntimeError("Inputs not initialized for verification")

    def _ensure_tpot_metric_values(self) -> list[float]:
        if len(self._tpot_metric_values) != self.seq_len:
            self._tpot_metric_values = [0.0] * self.seq_len
            self._iteration_metric_payload["tpot_times_ms"] = self._tpot_metric_values
        return self._tpot_metric_values

    def finalize_iteration_metrics(self) -> dict[str, list[float]] | None:
        start = self._pending_start
        prefill_end = self._pending_prefill_end
        decode_start = self._pending_decode_start
        decode_end = self._pending_decode_end
        graph_path = self._pending_graph_path
        if start is None or prefill_end is None or decode_start is None or decode_end is None or graph_path is None:
            return None

        self._pending_graph_path = None
        self._pending_start = None
        self._pending_prefill_end = None
        self._pending_decode_start = None
        self._pending_decode_end = None

        ttft_ms = float(start.elapsed_time(prefill_end))
        decode_ms = float(decode_start.elapsed_time(decode_end))
        per_token_ms = decode_ms / max(1, self.seq_len)
        self._ttft_metric_values[0] = ttft_ms
        tpot_times_ms = self._ensure_tpot_metric_values()
        for idx in range(len(tpot_times_ms)):
            tpot_times_ms[idx] = per_token_ms
        return self._iteration_metric_payload

    def capture_verification_payload(self) -> None:
        if self.inputs is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={
                "q": self.inputs.q,
                "k": self.inputs.k,
                "v": self.inputs.v,
            },
            output=self._verify_output_buffer,
            batch_size=self.batch,
            parameter_count=0,
            precision_flags={
                "fp16": self.inputs.q.dtype == torch.float16,
                "bf16": self.inputs.q.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def get_custom_streams(self):
        streams = list(self.prefill_streams) if self.prefill_streams else []
        streams.append(self.decode_stream)
        return streams

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None
        self.full_graph = None
        self.output = None
        self._output_view = None
        self._verify_output_buffer = None
        self._prefill_events = []
        self._prefill_work = []
        self._pending_graph_path = None
        self._pending_start = None
        self._pending_prefill_end = None
        self._pending_decode_start = None
        self._pending_decode_end = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=10,
            measurement_timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        return {
            "tma_prefill_decode.batch_size": float(getattr(self, 'batch_size', 0)),
            "tma_prefill_decode.seq_len": float(getattr(self, 'seq_len', 0)),
            "tma_prefill_decode.hidden_dim": float(getattr(self, 'hidden_dim', 0)),
        }

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None

def get_benchmark() -> BaseBenchmark:
    return OptimizedTmaPrefillDecodeBenchmark()
