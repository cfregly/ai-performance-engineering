"""Shared workload helpers and benchmark classes for the training-hotpath lab."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.training_hotpath.training_hotpath_extension import load_training_hotpath_extension


@dataclass
class MetricReductionWorkload:
    batch_size: int = 32
    max_num_tokens: int = 384
    responders: int = 256


@dataclass
class GradientReductionWorkload:
    num_segments: int = 128
    min_segment_length: int = 4096
    max_segment_length: int = 16384


@dataclass
class PaddingAwareWorkload:
    batch_size: int = 16
    max_num_tokens: int = 512
    min_num_tokens: int = 64
    input_size: int = 128
    hidden_size: int = 256
    projection_size: int = 512
    num_heads: int = 8
    num_blocks: int = 4
    output_size: int = 128


def _metric_workload_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-num-tokens", type=int, default=None)
    parser.add_argument("--responders", type=int, default=None)
    return parser


def _gradient_workload_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-segments", type=int, default=None)
    parser.add_argument("--min-segment-length", type=int, default=None)
    parser.add_argument("--max-segment-length", type=int, default=None)
    return parser


def _padding_workload_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-num-tokens", type=int, default=None)
    parser.add_argument("--min-num-tokens", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--projection-size", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--output-size", type=int, default=None)
    return parser


def apply_metric_reduction_overrides(
    workload: MetricReductionWorkload,
    argv: list[str],
) -> MetricReductionWorkload:
    args, _ = _metric_workload_parser().parse_known_args(argv)
    return MetricReductionWorkload(
        batch_size=args.batch_size if args.batch_size is not None else workload.batch_size,
        max_num_tokens=args.max_num_tokens if args.max_num_tokens is not None else workload.max_num_tokens,
        responders=args.responders if args.responders is not None else workload.responders,
    )


def apply_gradient_reduction_overrides(
    workload: GradientReductionWorkload,
    argv: list[str],
) -> GradientReductionWorkload:
    args, _ = _gradient_workload_parser().parse_known_args(argv)
    return GradientReductionWorkload(
        num_segments=args.num_segments if args.num_segments is not None else workload.num_segments,
        min_segment_length=(
            args.min_segment_length if args.min_segment_length is not None else workload.min_segment_length
        ),
        max_segment_length=(
            args.max_segment_length if args.max_segment_length is not None else workload.max_segment_length
        ),
    )


def apply_padding_aware_overrides(
    workload: PaddingAwareWorkload,
    argv: list[str],
) -> PaddingAwareWorkload:
    args, _ = _padding_workload_parser().parse_known_args(argv)
    return PaddingAwareWorkload(
        batch_size=args.batch_size if args.batch_size is not None else workload.batch_size,
        max_num_tokens=args.max_num_tokens if args.max_num_tokens is not None else workload.max_num_tokens,
        min_num_tokens=args.min_num_tokens if args.min_num_tokens is not None else workload.min_num_tokens,
        input_size=args.input_size if args.input_size is not None else workload.input_size,
        hidden_size=args.hidden_size if args.hidden_size is not None else workload.hidden_size,
        projection_size=args.projection_size if args.projection_size is not None else workload.projection_size,
        num_heads=args.num_heads if args.num_heads is not None else workload.num_heads,
        num_blocks=args.num_blocks if args.num_blocks is not None else workload.num_blocks,
        output_size=args.output_size if args.output_size is not None else workload.output_size,
    )


def build_metric_inputs(workload: MetricReductionWorkload, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(4242)
    preds = torch.randn(
        workload.batch_size,
        workload.max_num_tokens,
        workload.responders,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device)
    targets = torch.randn(
        workload.batch_size,
        workload.max_num_tokens,
        workload.responders,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device)
    return preds, targets


def scalar_metric_reduction(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    responders = preds.shape[-1]
    pred_sq = []
    target_sq = []
    covar = []
    for index in range(responders):
        pred_col = preds[..., index]
        target_col = targets[..., index]
        pred_sq.append((pred_col * pred_col).sum())
        target_sq.append((target_col * target_col).sum())
        covar.append((pred_col * target_col).sum())
    return torch.cat(
        (
            torch.stack(pred_sq, dim=0),
            torch.stack(target_sq, dim=0),
            torch.stack(covar, dim=0),
        ),
        dim=0,
    )


def vectorized_metric_reduction(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred_flat = preds.reshape(-1, preds.shape[-1])
    target_flat = targets.reshape(-1, targets.shape[-1])
    pred_sq = (pred_flat * pred_flat).sum(dim=0)
    target_sq = (target_flat * target_flat).sum(dim=0)
    covar = (pred_flat * target_flat).sum(dim=0)
    return torch.cat((pred_sq, target_sq, covar), dim=0)


def build_gradient_inputs(
    workload: GradientReductionWorkload,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cpu_generator = torch.Generator()
    cpu_generator.manual_seed(4242)
    lengths = torch.randint(
        workload.min_segment_length,
        workload.max_segment_length + 1,
        (workload.num_segments,),
        generator=cpu_generator,
        dtype=torch.int64,
    )
    offsets = torch.zeros(workload.num_segments + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    total = int(offsets[-1].item())
    values = torch.randn(total, generator=cpu_generator, dtype=torch.float32)
    return values.to(device=device), offsets.to(device=device)


def build_segment_metadata(offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute segment ids and lengths in setup() so benchmark_fn() stays on-device."""

    lengths = offsets[1:] - offsets[:-1]
    segment_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=offsets.device, dtype=torch.int64),
        lengths,
    )
    return segment_ids, lengths.to(dtype=torch.float32)


def baseline_segment_abs_mean(
    flat: torch.Tensor,
    segment_ids: torch.Tensor,
    segment_lengths: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Baseline torch segmented reduction without host round-trips."""

    out.zero_()
    out.scatter_add_(0, segment_ids, flat.abs())
    out.div_(segment_lengths.clamp_min(1.0))
    return out


def active_mask_and_rows(
    seq_lens: torch.Tensor,
    max_num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(max_num_tokens, device=seq_lens.device)
    active_mask = positions[None, :] < seq_lens[:, None]
    active_rows = active_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1).to(torch.int64)
    return active_mask, active_rows


class DenseLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, generator: torch.Generator) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, generator=generator) * 0.02)
        self.bias = nn.Parameter(torch.randn(out_features, generator=generator) * 0.01)

    def forward(self, x: torch.Tensor, *, active_rows: torch.Tensor, extension) -> torch.Tensor:
        del active_rows, extension
        return F.linear(x, self.weight, self.bias)


class PackedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, generator: torch.Generator) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, generator=generator) * 0.02)
        self.bias = nn.Parameter(torch.randn(out_features, generator=generator) * 0.01)

    def forward(self, x: torch.Tensor, *, active_rows: torch.Tensor, extension) -> torch.Tensor:
        if extension is None:
            raise RuntimeError("PackedLinear requires the training_hotpath CUDA extension")
        total_rows = x.shape[0] * x.shape[1]
        flat = x.reshape(total_rows, x.shape[-1]).contiguous()
        packed = extension.pack_rows(flat, active_rows)
        packed_out = F.linear(packed, self.weight, self.bias)
        restored = extension.scatter_rows(packed_out.contiguous(), active_rows, total_rows)
        return restored.reshape(x.shape[0], x.shape[1], packed_out.shape[-1])


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        projection_size: int,
        num_heads: int,
        *,
        linear_cls: type[nn.Module],
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.ln1 = nn.LayerNorm(hidden_size)
        self.qkv = linear_cls(hidden_size, hidden_size * 3, generator=generator)
        self.out_proj = linear_cls(hidden_size, hidden_size, generator=generator)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.up_gate = linear_cls(hidden_size, projection_size * 2, generator=generator)
        self.down = linear_cls(projection_size, hidden_size, generator=generator)

    def forward(
        self,
        x: torch.Tensor,
        *,
        active_mask: torch.Tensor,
        active_rows: torch.Tensor,
        extension,
    ) -> torch.Tensor:
        y = self.ln1(x)
        qkv = self.qkv(y, active_rows=active_rows, extension=extension)
        query, key, value = qkv.chunk(3, dim=-1)
        batch_size, num_tokens, _ = x.shape
        query = query.view(batch_size, num_tokens, self.num_heads, self.head_size).transpose(1, 2)
        key = key.view(batch_size, num_tokens, self.num_heads, self.head_size).transpose(1, 2)
        value = value.view(batch_size, num_tokens, self.num_heads, self.head_size).transpose(1, 2)
        attn_mask = active_mask[:, None, None, :]
        attn = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(batch_size, num_tokens, self.hidden_size)
        x = x + self.out_proj(attn, active_rows=active_rows, extension=extension)
        x = x * active_mask.unsqueeze(-1)

        y = self.ln2(x)
        up, gate = self.up_gate(y, active_rows=active_rows, extension=extension).chunk(2, dim=-1)
        y = F.silu(up) * gate
        x = x + self.down(y, active_rows=active_rows, extension=extension)
        return x * active_mask.unsqueeze(-1)


class ToyTransformer(nn.Module):
    def __init__(self, workload: PaddingAwareWorkload, *, optimized: bool, device: torch.device) -> None:
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(4242)
        linear_cls = PackedLinear if optimized else DenseLinear
        self.input_proj = linear_cls(workload.input_size, workload.hidden_size, generator=generator)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    workload.hidden_size,
                    workload.projection_size,
                    workload.num_heads,
                    linear_cls=linear_cls,
                    generator=generator,
                )
                for _ in range(workload.num_blocks)
            ]
        )
        self.output_proj = linear_cls(workload.hidden_size, workload.output_size, generator=generator)

    def forward(
        self,
        x: torch.Tensor,
        *,
        active_mask: torch.Tensor,
        active_rows: torch.Tensor,
        extension,
    ) -> torch.Tensor:
        x = self.input_proj(x, active_rows=active_rows, extension=extension)
        x = x * active_mask.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, active_mask=active_mask, active_rows=active_rows, extension=extension)
        x = self.output_proj(x, active_rows=active_rows, extension=extension)
        return x * active_mask.unsqueeze(-1)


def build_padding_inputs(
    workload: PaddingAwareWorkload,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cpu_generator = torch.Generator()
    cpu_generator.manual_seed(4242)
    seq_lens = torch.randint(
        workload.min_num_tokens,
        workload.max_num_tokens + 1,
        (workload.batch_size,),
        generator=cpu_generator,
        dtype=torch.int64,
    ).to(device=device)
    inputs = torch.randn(
        workload.batch_size,
        workload.max_num_tokens,
        workload.input_size,
        generator=cpu_generator,
        dtype=torch.float32,
    ).to(device=device)
    active_mask, active_rows = active_mask_and_rows(seq_lens, workload.max_num_tokens)
    return inputs, seq_lens, active_rows


class MetricReductionVectorizedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark scalar per-output metric aggregation vs a fused single-pass reduction.

    The optimized arm calls the lab extension's metric_reduction_fused kernel:
    one read of preds and targets produces all three per-responder sums with
    fp32 accumulation, instead of three separate mul+sum passes (plus
    temporaries) over the same 25 MB of inputs.
    """

    preferred_ncu_replay_mode = "application"

    def __init__(self, *, optimized: bool, label: str) -> None:
        super().__init__()
        self.optimized = optimized
        self.label = label
        self.workload = MetricReductionWorkload()
        self.preds: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._extension = None
        self._custom_metrics: dict[str, float] = {}
        self._refresh_workload_metadata()

    def _refresh_workload_metadata(self) -> None:
        total = self.workload.batch_size * self.workload.max_num_tokens * self.workload.responders
        self._workload = WorkloadMetadata(requests_per_iteration=1.0, tokens_per_iteration=float(total))

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("labs.training_hotpath metric-reduction benchmarks require CUDA")
        self.preds, self.targets = build_metric_inputs(self.workload, self.device)
        self.output = None
        if self.optimized:
            self._extension = load_training_hotpath_extension()
            self._extension.metric_reduction_fused(self.preds, self.targets)
        else:
            self._extension = None
        total = self.workload.batch_size * self.workload.max_num_tokens * self.workload.responders
        self._custom_metrics = {
            "metric_reduction.is_vectorized": 1.0 if self.optimized else 0.0,
            "metric_reduction.uses_cuda_extension": 1.0 if self.optimized else 0.0,
            "metric_reduction.responders": float(self.workload.responders),
            "metric_reduction.total_elements": float(total),
        }
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        if self.preds is None or self.targets is None:
            raise RuntimeError("Metric inputs not initialized")
        if self.optimized:
            if self._extension is None:
                raise RuntimeError("CUDA extension not loaded")
            self.output = self._extension.metric_reduction_fused(self.preds, self.targets)
        else:
            self.output = scalar_metric_reduction(self.preds, self.targets)

    def capture_verification_payload(self) -> None:
        if self.output is None or self.preds is None or self.targets is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"preds": self.preds, "targets": self.targets},
            output=self.output,
            batch_size=self.workload.batch_size,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(1e-4, 1e-4),
        )

    def teardown(self) -> None:
        self.preds = None
        self.targets = None
        self.output = None
        self._extension = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=200,
            warmup=10,
            setup_timeout_seconds=90,
            measurement_timeout_seconds=90,
            ncu_replay_mode="application",
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return dict(self._custom_metrics)

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        expected = self.workload.responders * 3
        if self.output.shape != (expected,):
            return f"Unexpected output shape: {tuple(self.output.shape)}"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None

    def apply_target_overrides(self, argv: list[str]) -> None:
        self.workload = apply_metric_reduction_overrides(self.workload, argv)
        self._refresh_workload_metadata()


class MetricReductionCudaBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark vectorized torch reduction vs fused CUDA segmented reduction."""

    preferred_ncu_replay_mode = "application"

    def __init__(self, *, optimized: bool, label: str) -> None:
        super().__init__()
        self.optimized = optimized
        self.label = label
        self.workload = GradientReductionWorkload()
        self.flat: Optional[torch.Tensor] = None
        self.offsets: Optional[torch.Tensor] = None
        self.segment_ids: Optional[torch.Tensor] = None
        self.segment_lengths: Optional[torch.Tensor] = None
        self._baseline_out: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._extension = None
        self._custom_metrics: dict[str, float] = {}
        self._refresh_workload_metadata()

    def _refresh_workload_metadata(self) -> None:
        avg = (self.workload.min_segment_length + self.workload.max_segment_length) / 2.0
        total = avg * self.workload.num_segments
        self._workload = WorkloadMetadata(requests_per_iteration=1.0, tokens_per_iteration=float(total))

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("labs.training_hotpath metric-reduction benchmarks require CUDA")
        self.flat, self.offsets = build_gradient_inputs(self.workload, self.device)
        self.segment_ids, self.segment_lengths = build_segment_metadata(self.offsets)
        self._baseline_out = torch.empty(self.workload.num_segments, device=self.device, dtype=torch.float32)
        self.output = None
        if self.optimized:
            self._extension = load_training_hotpath_extension()
            self._extension.segment_abs_mean(self.flat, self.offsets)
        else:
            self._extension = None
        total = int(self.offsets[-1].item()) if self.offsets is not None else 0
        self._custom_metrics = {
            "metric_reduction.is_fused_cuda": 1.0 if self.optimized else 0.0,
            "metric_reduction.uses_cuda_extension": 1.0 if self.optimized else 0.0,
            "metric_reduction.segments": float(self.workload.num_segments),
            "metric_reduction.total_elements": float(total),
        }
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        if self.flat is None or self.offsets is None:
            raise RuntimeError("Gradient inputs not initialized")
        if self.optimized:
            if self._extension is None:
                raise RuntimeError("CUDA extension not loaded")
            self.output = self._extension.segment_abs_mean(self.flat, self.offsets)
        else:
            if self.segment_ids is None or self.segment_lengths is None or self._baseline_out is None:
                raise RuntimeError("Baseline segment metadata not initialized")
            self.output = baseline_segment_abs_mean(
                self.flat,
                self.segment_ids,
                self.segment_lengths,
                self._baseline_out,
            )

    def capture_verification_payload(self) -> None:
        if self.output is None or self.flat is None or self.offsets is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"flat": self.flat, "offsets": self.offsets},
            output=self.output,
            batch_size=self.workload.num_segments,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(1e-5, 1e-5),
        )

    def teardown(self) -> None:
        self.flat = None
        self.offsets = None
        self.segment_ids = None
        self.segment_lengths = None
        self._baseline_out = None
        self.output = None
        self._extension = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            setup_timeout_seconds=180,
            measurement_timeout_seconds=120,
            ncu_replay_mode="application",
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return dict(self._custom_metrics)

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        if self.output.shape != (self.workload.num_segments,):
            return f"Unexpected output shape: {tuple(self.output.shape)}"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None

    def apply_target_overrides(self, argv: list[str]) -> None:
        self.workload = apply_gradient_reduction_overrides(self.workload, argv)
        self._refresh_workload_metadata()


class PaddingAwareTransformerBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark dense padded transformer math vs packed-row padding-aware projections."""

    preferred_ncu_replay_mode = "application"

    def __init__(self, *, optimized: bool, label: str) -> None:
        super().__init__()
        self.optimized = optimized
        self.label = label
        self.workload = PaddingAwareWorkload()
        self.inputs: Optional[torch.Tensor] = None
        self.seq_lens: Optional[torch.Tensor] = None
        self.active_rows: Optional[torch.Tensor] = None
        self.model: Optional[ToyTransformer] = None
        self.output: Optional[torch.Tensor] = None
        self._active_mask: Optional[torch.Tensor] = None
        self._extension = None
        self._custom_metrics: dict[str, float] = {}
        self._refresh_workload_metadata()

    def _refresh_workload_metadata(self) -> None:
        total = self.workload.batch_size * self.workload.max_num_tokens * self.workload.hidden_size
        self._workload = WorkloadMetadata(requests_per_iteration=1.0, tokens_per_iteration=float(total))

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("labs.training_hotpath padding-aware transformer benchmark requires CUDA")
        self.inputs, self.seq_lens, self.active_rows = build_padding_inputs(self.workload, self.device)
        active_mask, _ = active_mask_and_rows(self.seq_lens, self.workload.max_num_tokens)
        self._active_mask = active_mask
        self._extension = load_training_hotpath_extension() if self.optimized else None
        if self._extension is not None:
            flat = self.inputs.reshape(-1, self.inputs.shape[-1]).contiguous()
            packed = self._extension.pack_rows(flat, self.active_rows)
            self._extension.scatter_rows(packed, self.active_rows, flat.shape[0])
        self.model = ToyTransformer(self.workload, optimized=self.optimized, device=self.device).to(self.device)
        self.output = None
        active_tokens = int(self.seq_lens.sum().item())
        total_tokens = self.workload.batch_size * self.workload.max_num_tokens
        active_fraction = active_tokens / float(total_tokens)
        self._custom_metrics = {
            "padding_aware.enabled": 1.0 if self.optimized else 0.0,
            "padding_aware.uses_cuda_extension": 1.0 if self.optimized else 0.0,
            "padding_aware.active_token_fraction": float(active_fraction),
            "padding_aware.padded_token_fraction": float(1.0 - active_fraction),
            "padding_aware.num_blocks": float(self.workload.num_blocks),
        }
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None or self.seq_lens is None or self.active_rows is None or self.model is None:
            raise RuntimeError("Padding-aware benchmark state not initialized")
        self.output = self.model(
            self.inputs,
            active_mask=self._active_mask,
            active_rows=self.active_rows,
            extension=self._extension,
        )

    def capture_verification_payload(self) -> None:
        if self.output is None or self.inputs is None or self.seq_lens is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"inputs": self.inputs, "seq_lens": self.seq_lens},
            output=self.output,
            batch_size=self.workload.batch_size,
            parameter_count=sum(parameter.numel() for parameter in self.model.parameters()) if self.model is not None else 0,
            precision_flags={"fp16": False, "bf16": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(1e-5, 1e-5),
        )

    def teardown(self) -> None:
        self.inputs = None
        self.seq_lens = None
        self.active_rows = None
        self.model = None
        self.output = None
        self._extension = None
        self._active_mask = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=15,
            warmup=5,
            enable_memory_tracking=True,
            setup_timeout_seconds=240,
            measurement_timeout_seconds=120,
            ncu_replay_mode="application",
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return dict(self._custom_metrics)

    def get_optimization_goal(self) -> str:
        """Packed-row padding avoidance is judged here as a memory-reduction tradeoff."""
        return "memory"

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        expected = (self.workload.batch_size, self.workload.max_num_tokens, self.workload.output_size)
        if self.output.shape != expected:
            return f"Unexpected output shape: {tuple(self.output.shape)}"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None

    def apply_target_overrides(self, argv: list[str]) -> None:
        self.workload = apply_padding_aware_overrides(self.workload, argv)
        self._refresh_workload_metadata()
