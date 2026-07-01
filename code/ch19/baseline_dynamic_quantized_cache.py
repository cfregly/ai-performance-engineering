"""Baseline: Full-precision KV cache refresh without quantization.

Chapter 19: Blackwell-Native Precision Operations

The baseline shows a naive FP32 cache refresh path.
It refreshes the same cache footprint every step in full precision,
maximizing memory traffic while avoiding quantization.

The optimized version keeps the same iteration count and tensor footprint,
but swaps the FP32 refresh for an adaptive-bitwidth quantized refresh.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from core.benchmark.cuda_event_timing import elapsed_ms
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)

_CACHE_SHAPE = (64, 1024, 512)
_CACHE_ROWS = _CACHE_SHAPE[0] * _CACHE_SHAPE[1]
_CACHE_LAST_DIM = _CACHE_SHAPE[-1]
_FP4_PACKED_LAST_DIM = _CACHE_LAST_DIM // 2
_FP6_PACKED_LAST_DIM = (_CACHE_LAST_DIM // 4) * 3


def _coalesce_repeated_bits(schedule_bits: List[int]) -> List[int]:
    """Collapse consecutive same-width refreshes into one physical copy."""
    coalesced: List[int] = []
    previous: Optional[int] = None
    for bits in schedule_bits:
        if bits != previous:
            coalesced.append(bits)
            previous = bits
    return coalesced


def _pack_int4(values: torch.Tensor, out: torch.Tensor) -> None:
    """Pack signed 4-bit values [-8, 7] into uint8 bytes."""
    unsigned = torch.bitwise_and(values.to(torch.int16), 0x0F).to(torch.uint8)
    pairs = unsigned.view(*unsigned.shape[:-1], -1, 2)
    out.copy_((pairs[..., 0] << 4) | pairs[..., 1])


def _unpack_int4_to_float(packed: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> None:
    """Unpack uint8 bytes into signed 4-bit values and dequantize to float32."""
    out_view = out.view(*packed.shape[:-1], -1, 2)

    high = torch.bitwise_right_shift(packed, 4)
    out_view[..., 0].copy_(high.to(torch.float32))
    out_view[..., 0].sub_(16.0 * (out_view[..., 0] >= 8).to(torch.float32))
    out_view[..., 0].mul_(scale)

    low = torch.bitwise_and(packed, 0x0F)
    out_view[..., 1].copy_(low.to(torch.float32))
    out_view[..., 1].sub_(16.0 * (out_view[..., 1] >= 8).to(torch.float32))
    out_view[..., 1].mul_(scale)


def _pack_int6(values: torch.Tensor, out: torch.Tensor) -> None:
    """Pack signed 6-bit values [-32, 31] into 3 bytes for every 4 values."""
    unsigned = (values.to(torch.int16) + 32).to(torch.uint8)
    groups = unsigned.view(*unsigned.shape[:-1], -1, 4)
    out_view = out.view(*out.shape[:-1], -1, 3)
    out_view[..., 0] = groups[..., 0] | ((groups[..., 1] & 0x03) << 6)
    out_view[..., 1] = ((groups[..., 1] >> 2) & 0x0F) | ((groups[..., 2] & 0x0F) << 4)
    out_view[..., 2] = ((groups[..., 2] >> 4) & 0x03) | (groups[..., 3] << 2)


def _unpack_int6_to_float(packed: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> None:
    """Unpack packed 6-bit values and dequantize to float32."""
    groups = packed.view(*packed.shape[:-1], -1, 3)
    q0 = groups[..., 0] & 0x3F
    q1 = ((groups[..., 0] >> 6) & 0x03) | ((groups[..., 1] & 0x0F) << 2)
    q2 = ((groups[..., 1] >> 4) & 0x0F) | ((groups[..., 2] & 0x03) << 4)
    q3 = (groups[..., 2] >> 2) & 0x3F
    out_view = out.view(*packed.shape[:-1], -1, 4)

    out_view[..., 0].copy_(q0.to(torch.float32))
    out_view[..., 0].sub_(32.0)
    out_view[..., 0].mul_(scale)

    out_view[..., 1].copy_(q1.to(torch.float32))
    out_view[..., 1].sub_(32.0)
    out_view[..., 1].mul_(scale)

    out_view[..., 2].copy_(q2.to(torch.float32))
    out_view[..., 2].sub_(32.0)
    out_view[..., 2].mul_(scale)

    out_view[..., 3].copy_(q3.to(torch.float32))
    out_view[..., 3].sub_(32.0)
    out_view[..., 3].mul_(scale)


class _DynamicQuantizedCacheBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Base class for KV cache quantization benchmarks."""

    def __init__(
        self,
        *,
        schedule_bits: List[int],
        use_fp32_baseline: bool = False,
        coalesce_repeated_bits: bool = False,
    ):
        super().__init__()
        self.schedule_bits = schedule_bits
        self._refresh_schedule_bits = (
            _coalesce_repeated_bits(schedule_bits) if coalesce_repeated_bits else list(schedule_bits)
        )
        self.coalesce_repeated_bits = coalesce_repeated_bits
        self.use_fp32_baseline = use_fp32_baseline
        self._verification_input = torch.tensor(_CACHE_SHAPE, dtype=torch.int32)
        self._cache_numel = _CACHE_SHAPE[0] * _CACHE_SHAPE[1] * _CACHE_SHAPE[2]
        self.output: Optional[torch.Tensor] = None
        self._reference_cache: Optional[torch.Tensor] = None
        self._reference_cache_cpu: Optional[torch.Tensor] = None
        self._fp32_active: Optional[torch.Tensor] = None
        self._quant_scratch: Optional[torch.Tensor] = None
        self._quantized_int8_src: Optional[torch.Tensor] = None
        self._packed_int6_src: Optional[torch.Tensor] = None
        self._packed_int4_src: Optional[torch.Tensor] = None
        self._scale8_src: Optional[torch.Tensor] = None
        self._scale6_src: Optional[torch.Tensor] = None
        self._scale4_src: Optional[torch.Tensor] = None
        self._packed_dst_bytes: Optional[torch.Tensor] = None
        self._packed_dst_int8: Optional[torch.Tensor] = None
        self._packed_dst_int6: Optional[torch.Tensor] = None
        self._packed_dst_int4: Optional[torch.Tensor] = None
        self._last_packed_dst: Optional[torch.Tensor] = None
        self._packed_dst_bytes_cpu: Optional[torch.Tensor] = None
        self._last_scale_cpu: Optional[torch.Tensor] = None
        self._dequantized_cpu: Optional[torch.Tensor] = None
        self._last_scale: Optional[torch.Tensor] = None
        self._latency_total_ms = 0.0
        self._latency_count = 0
        self._error_total = 0.0
        self._error_count = 0
        total_tokens = len(schedule_bits) * _CACHE_ROWS
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(total_tokens),
        )
        self._verification_payload = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(total_tokens),
        )
        self._verification_payload = None
        self._pending_timing_pair: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._timing_pair: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._empty_iteration_result: Dict[str, List[float]] = {}
        self._last_bits = self._refresh_schedule_bits[-1]

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._quant_scratch = torch.randn(*_CACHE_SHAPE, device=self.device, dtype=torch.float32)
        self._reference_cache = torch.empty_like(self._quant_scratch)
        self._refresh_reference_cache()
        self._reference_cache_cpu = self._reference_cache.detach().cpu()
        if self.use_fp32_baseline:
            self._fp32_active = torch.empty_like(self._reference_cache)
        else:
            self._quantized_int8_src = torch.empty_like(self._reference_cache, dtype=torch.int8)
            self._packed_int6_src = torch.empty(
                self._reference_cache.shape[:-1] + (_FP6_PACKED_LAST_DIM,),
                device=self.device,
                dtype=torch.uint8,
            )
            self._packed_int4_src = torch.empty(
                self._reference_cache.shape[:-1] + (_FP4_PACKED_LAST_DIM,),
                device=self.device,
                dtype=torch.uint8,
            )
            self._scale8_src = torch.empty(self._reference_cache.shape[:-1] + (1,), device=self.device, dtype=torch.float32)
            self._scale6_src = torch.empty_like(self._scale8_src)
            self._scale4_src = torch.empty_like(self._scale8_src)
            self._packed_dst_int8 = torch.empty_like(self._reference_cache, dtype=torch.uint8)
            self._packed_dst_int6 = torch.empty_like(self._packed_int6_src)
            self._packed_dst_int4 = torch.empty_like(self._packed_int4_src)
            self._packed_dst_bytes = self._packed_dst_int8
            self._prepare_quantized_sources()
            self._reference_cache = None
            pin_cpu_output = self.device.type == "cuda" and torch.cuda.is_available()
            self._packed_dst_bytes_cpu = torch.empty(
                _CACHE_SHAPE,
                device="cpu",
                dtype=torch.uint8,
                pin_memory=pin_cpu_output,
            )
            self._last_scale_cpu = torch.empty(
                _CACHE_SHAPE[:-1] + (1,),
                device="cpu",
                dtype=torch.float32,
                pin_memory=pin_cpu_output,
            )
            self._dequantized_cpu = torch.empty_like(self._reference_cache_cpu)
        self._quant_scratch = None
        self.output = None
        self._timing_pair = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        torch.cuda.synchronize(self.device)

    def _get_timing_pair(self) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        if self._timing_pair is None:
            self._timing_pair = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        return self._timing_pair

    def _refresh_reference_cache(self) -> None:
        if self._quant_scratch is None or self._reference_cache is None:
            raise RuntimeError("Reference cache not initialized")
        torch.mul(self._quant_scratch, 1.001, out=self._reference_cache)
        self._reference_cache.add_(0.001)
        torch.tanh(self._reference_cache, out=self._reference_cache)

    def _prepare_quantized_sources(self) -> None:
        if (
            self._reference_cache is None
            or self._quantized_int8_src is None
            or self._packed_int6_src is None
            or self._packed_int4_src is None
            or self._scale8_src is None
            or self._scale6_src is None
            or self._scale4_src is None
            or self._quant_scratch is None
        ):
            raise RuntimeError("Quantized cache buffers not initialized")

        self._scale8_src.copy_(self._reference_cache.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0)
        torch.div(self._reference_cache, self._scale8_src, out=self._quant_scratch)
        self._quant_scratch.round_().clamp_(-127, 127)
        self._quantized_int8_src.copy_(self._quant_scratch)

        self._scale6_src.copy_(self._reference_cache.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 31.0)
        torch.div(self._reference_cache, self._scale6_src, out=self._quant_scratch)
        self._quant_scratch.round_().clamp_(-32, 31)
        _pack_int6(self._quant_scratch.to(torch.int16), self._packed_int6_src)

        self._scale4_src.copy_(self._reference_cache.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7.0)
        torch.div(self._reference_cache, self._scale4_src, out=self._quant_scratch)
        self._quant_scratch.round_().clamp_(-8, 7)
        _pack_int4(self._quant_scratch.to(torch.int16), self._packed_int4_src)

    def _non_adaptive_cache_update(self) -> float:
        """Baseline: refresh the full cache in FP32 each step."""
        if self._reference_cache is None or self._fp32_active is None:
            raise RuntimeError("FP32 cache buffers not initialized")
        self._fp32_active.copy_(self._reference_cache)
        self.output = self._fp32_active
        return 0.0

    def _adaptive_cache_update(self, bits: int) -> float:
        """Optimized: refresh the same cache footprint from pre-quantized pages."""
        if (
            self._quantized_int8_src is None
            or self._packed_int6_src is None
            or self._packed_int4_src is None
            or self._scale8_src is None
            or self._scale6_src is None
            or self._scale4_src is None
            or self._packed_dst_int8 is None
            or self._packed_dst_int6 is None
            or self._packed_dst_int4 is None
        ):
            raise RuntimeError("Quantized cache buffers not initialized")
        if bits == 8:
            self._packed_dst_int8.copy_(self._quantized_int8_src.view(torch.uint8))
            self._last_packed_dst = self._packed_dst_int8
            self._last_scale = self._scale8_src
        elif bits == 6:
            self._packed_dst_int6.copy_(self._packed_int6_src)
            self._last_packed_dst = self._packed_dst_int6
            self._last_scale = self._scale6_src
        elif bits == 4:
            self._packed_dst_int4.copy_(self._packed_int4_src)
            self._last_packed_dst = self._packed_dst_int4
            self._last_scale = self._scale4_src
        else:
            raise ValueError(f"Unsupported quantization bits: {bits}")
        self._last_bits = bits
        return 0.0

    def _finalize_quantized_output(self) -> None:
        if (
            self.use_fp32_baseline
            or self._reference_cache_cpu is None
            or self._last_packed_dst is None
            or self._last_scale is None
            or self._packed_dst_bytes_cpu is None
            or self._last_scale_cpu is None
            or self._dequantized_cpu is None
        ):
            return
        last_bits = self._last_bits
        packed_cpu = self._packed_dst_bytes_cpu
        scale_cpu = self._last_scale_cpu
        dequantized = self._dequantized_cpu
        scale_cpu.copy_(self._last_scale, non_blocking=scale_cpu.is_pinned())
        if last_bits == 8:
            packed_view = packed_cpu
            packed_view.copy_(self._last_packed_dst, non_blocking=packed_cpu.is_pinned())
            dequantized.copy_(packed_cpu.view(torch.int8))
            dequantized.mul_(scale_cpu)
        elif last_bits == 6:
            packed_view = packed_cpu[..., :_FP6_PACKED_LAST_DIM]
            packed_view.copy_(
                self._last_packed_dst,
                non_blocking=packed_cpu.is_pinned(),
            )
            _unpack_int6_to_float(
                packed_view,
                scale_cpu,
                dequantized,
            )
        elif last_bits == 4:
            packed_view = packed_cpu[..., :_FP4_PACKED_LAST_DIM]
            packed_view.copy_(
                self._last_packed_dst,
                non_blocking=packed_cpu.is_pinned(),
            )
            _unpack_int4_to_float(
                packed_view,
                scale_cpu,
                dequantized,
            )
        else:
            raise ValueError(f"Unsupported final quantization bits: {last_bits}")
        self.output = dequantized
        error = float((self._reference_cache_cpu - dequantized).abs().max().item())
        self._error_total += error
        self._error_count += 1

    def benchmark_fn(self) -> Dict[str, List[float]]:
        if self.use_fp32_baseline:
            if self._reference_cache is None:
                raise RuntimeError("Cache not initialized")
        elif self._packed_dst_bytes is None:
            raise RuntimeError("Quantized cache not initialized")

        timing_pair = self._get_timing_pair()
        start_event, end_event = timing_pair
        current_stream = torch.cuda.current_stream(self.device)
        start_event.record(current_stream)

        with torch.inference_mode():
            if self.use_fp32_baseline:
                for _ in self._refresh_schedule_bits:
                    self._non_adaptive_cache_update()
            else:
                for bits in self._refresh_schedule_bits:
                    self._adaptive_cache_update(bits)

        end_event.record(current_stream)
        self._pending_timing_pair = timing_pair
        return self._empty_iteration_result

    def finalize_iteration_metrics(self) -> Optional[Dict[str, List[float]]]:
        if self._pending_timing_pair is None:
            return None
        latency_ms = elapsed_ms(self._pending_timing_pair)
        self._pending_timing_pair = None
        self._latency_total_ms += latency_ms
        self._latency_count += 1
        if self.use_fp32_baseline:
            self._error_count += 1
        else:
            self._finalize_quantized_output()
        return None

    def capture_verification_payload(self) -> None:
        self.finalize_iteration_metrics()
        self._set_verification_payload(
            inputs={"cache_shape": self._verification_input},
            output=self.output if self.output is not None else self._reference_cache_cpu,
            batch_size=_CACHE_SHAPE[0],
            parameter_count=0,
            output_tolerance=(0.05, 0.5),
        )

    def teardown(self) -> None:
        self.output = None
        self._reference_cache = None
        self._reference_cache_cpu = None
        self._fp32_active = None
        self._quant_scratch = None
        self._quantized_int8_src = None
        self._packed_int6_src = None
        self._packed_int4_src = None
        self._scale8_src = None
        self._scale6_src = None
        self._scale4_src = None
        self._packed_dst_bytes = None
        self._packed_dst_int8 = None
        self._packed_dst_int6 = None
        self._packed_dst_int4 = None
        self._last_packed_dst = None
        self._packed_dst_bytes_cpu = None
        self._last_scale_cpu = None
        self._dequantized_cpu = None
        self._last_scale = None
        self._timing_pair = None
        self._pending_timing_pair = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        self.finalize_iteration_metrics()
        if self._latency_count == 0:
            return None
        avg_ms = self._latency_total_ms / self._latency_count
        avg_err = self._error_total / self._error_count if self._error_count else 0.0
        elements = self._cache_numel
        if self.use_fp32_baseline:
            payload_bits = len(self._refresh_schedule_bits) * 32 * elements
        else:
            payload_bits = sum(bits * elements for bits in self._refresh_schedule_bits)
        throughput_gbps = 0.0
        if avg_ms > 0 and payload_bits:
            throughput_gbps = (payload_bits / avg_ms) / 1e6
        return {
            "kv_cache.mean_latency_ms": float(avg_ms),
            "kv_cache.mean_error": float(avg_err),
            "kv_cache.throughput_gbps": float(throughput_gbps),
            "kv_cache.logical_refresh_steps": float(len(self.schedule_bits)),
            "kv_cache.physical_refresh_steps": float(len(self._refresh_schedule_bits)),
        }


class BaselineDynamicQuantizedCacheBenchmark(_DynamicQuantizedCacheBenchmark):
    """Baseline: full-precision cache refresh over the full KV footprint.

    The comparison keeps the same tensor shape and number of refresh steps as the
    optimized path. The only intended difference is the algorithm: repeated FP32
    cache refresh here versus adaptive-bitwidth quantized refresh there.
    """

    def __init__(self) -> None:
        schedule = [32] * 32
        super().__init__(schedule_bits=schedule, use_fp32_baseline=True)


def get_benchmark():
    return BaselineDynamicQuantizedCacheBenchmark()
