"""Microbatch pipeline with stream/event overlap across stages."""

from __future__ import annotations

from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.device_utils import require_cuda_device
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

resolve_device = partial(require_cuda_device, "CUDA required for ch20")


class SimpleStage(nn.Module):
    """Pipeline stage with FFN and residual connection."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ffn(x)
        out.add_(x)
        return self.norm(out)


class OptimizedPipelineOverlapBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: same microbatch work with stage-stream overlap."""
    
    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.stages: Optional[nn.ModuleList] = None
        self.inputs: Optional[torch.Tensor] = None
        self.microbatches: Optional[list[torch.Tensor]] = None
        self._microbatch_groups: list[tuple[int, torch.Tensor]] = []
        self.stage_streams: list[torch.cuda.Stream] = []
        self.stage_events: list[list[torch.cuda.Event]] = []
        self.output = None
        self._last_outputs: Optional[list[torch.Tensor]] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._stage_outputs: list[list[Optional[torch.Tensor]]] = []
        self._stage_schedule: list[
            tuple[
                int,
                nn.Module,
                torch.cuda.Stream,
                list[torch.cuda.Event],
                list[Optional[torch.Tensor]],
            ]
        ] = []
        self._last_output_count: int = 0
        self.batch_size = 512
        self.hidden_dim = 1536
        self.num_stages = 4
        self.repeats = 6
        self._repeat_range = range(self.repeats)
        self.num_microbatches = 8
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size),
            samples_per_iteration=float(self.batch_size),
        )
        self._payload_parameter_count = 0
    
    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        self.stages = nn.ModuleList(
            [SimpleStage(self.hidden_dim).to(self.device).half() for _ in range(self.num_stages)]
        ).eval()
        self._payload_parameter_count = sum(p.numel() for p in self.stages.parameters())

        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        self.microbatches = [chunk.contiguous() for chunk in self.inputs.chunk(self.num_microbatches, dim=0)]
        self._microbatch_groups = list(enumerate(self.microbatches))
        self._output_buffer = torch.empty_like(self.inputs)
        self._verify_output_buffer = torch.empty_like(self._output_buffer, dtype=torch.float32)
        self.stage_streams = [torch.cuda.Stream(device=self.device) for _ in range(self.num_stages)]
        self.stage_events = [
            [torch.cuda.Event(enable_timing=False) for _ in range(self.num_microbatches)]
            for _ in range(self.num_stages)
        ]
        self._stage_outputs = [
            [None for _ in range(self.num_microbatches)]
            for _ in range(self.num_stages)
        ]
        self._last_outputs = [
            torch.empty(0, device=self.device, dtype=torch.float16)
            for _ in range(self.num_microbatches)
        ]
        self._stage_schedule = [
            (stage_idx, stage, stream, event_row, output_row)
            for stage_idx, (stage, stream, event_row, output_row) in enumerate(
                zip(
                    self.stages,
                    self.stage_streams,
                    self.stage_events,
                    self._stage_outputs,
                    strict=True,
                )
            )
        ]
        self._repeat_range = range(self.repeats)

    def _run_pipelined_once(self) -> list[torch.Tensor]:
        assert self.stages is not None and self.microbatches is not None
        if not self.stage_events:
            raise RuntimeError("Pipeline events not initialized")
        if (
            len(self._stage_outputs) != self.num_stages
            or self._last_outputs is None
            or len(self._last_outputs) != self.num_microbatches
            or len(self._microbatch_groups) != self.num_microbatches
            or len(self._stage_schedule) != self.num_stages
        ):
            raise RuntimeError("Pipeline output slots not initialized")
        for _, _, _, event_row, output_row in self._stage_schedule:
            if len(event_row) != self.num_microbatches or len(output_row) != self.num_microbatches:
                raise RuntimeError("Pipeline output slots not initialized")

        last_stage_idx = self.num_stages - 1
        for microbatch_idx, microbatch in self._microbatch_groups:
            stage_input = microbatch
            previous_event: Optional[torch.cuda.Event] = None
            for stage_idx, stage, stream, event_row, output_row in self._stage_schedule:
                with torch.cuda.stream(stream):
                    if previous_event is not None:
                        stream.wait_event(previous_event)
                    stage_output = stage(stage_input)
                    output_row[microbatch_idx] = stage_output
                    event = event_row[microbatch_idx]
                    event.record(stream)
                if stage_idx == last_stage_idx:
                    self._last_outputs[microbatch_idx] = stage_output
                stage_input = stage_output
                previous_event = event

        current_stream = torch.cuda.current_stream(self.device)
        for stream in self.stage_streams:
            current_stream.wait_stream(stream)

        self._last_output_count = len(self._microbatch_groups)
        return self._last_outputs
    
    def benchmark_fn(self) -> None:
        assert self.inputs is not None and self.stages is not None and self.microbatches is not None
        with self._nvtx_range("pipeline_sequential_optimized"):
            with torch.inference_mode():
                for _ in self._repeat_range:
                    self._run_pipelined_once()

    def capture_verification_payload(self) -> None:
        if (
            self.inputs is None
            or self._last_outputs is None
            or self._output_buffer is None
            or self._verify_output_buffer is None
            or self.stages is None
        ):
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        if self._last_output_count != len(self._last_outputs):
            raise RuntimeError("Incomplete pipeline outputs before verification capture")
        torch.cat(self._last_outputs, dim=0, out=self._output_buffer)
        self.output = self._output_buffer.detach()
        self._verify_output_buffer.copy_(self.output, non_blocking=False)
        self._set_verification_payload(
            inputs={"inputs": self.inputs},
            output=self._verify_output_buffer,
            batch_size=self.batch_size,
            parameter_count=self._payload_parameter_count,
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.stages = None
        self.inputs = None
        self.microbatches = None
        self._microbatch_groups = []
        self.stage_streams = []
        self.stage_events = []
        self.output = None
        self._last_outputs = None
        self._output_buffer = None
        self._verify_output_buffer = None
        self._stage_outputs = []
        self._stage_schedule = []
        self._last_output_count = 0
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_ai_optimization_metrics
        return compute_ai_optimization_metrics(
            original_time_ms=None,
            ai_optimized_time_ms=getattr(self, '_last_elapsed_ms', None),
            suggestions_applied=None,
            suggestions_total=None,
        )

    def validate_result(self) -> Optional[str]:
        if self.stages is None:
            return "Stages not initialized"
        return None

    def get_custom_streams(self) -> list["torch.cuda.Stream"]:
        return list(self.stage_streams)

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        return super().get_input_signature()


def get_benchmark() -> BaseBenchmark:
    return OptimizedPipelineOverlapBenchmark()
