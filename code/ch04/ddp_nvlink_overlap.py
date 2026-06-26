"""ddp_nvlink_overlap.py

Topology- and overlap-aware DDP-style loop. Enables peer access, reorders
gradient buckets by device distance (lowest ID first as a proxy), and overlaps
gradient transfer with computation using a dedicated communication stream.
Falls back to single-GPU if only one device is present.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from core.benchmark.gpu_requirements import skip_if_insufficient_gpus, require_peer_access
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin


def _enable_peer_access() -> None:
    num = torch.cuda.device_count()
    skip_if_insufficient_gpus(2)
    for src in range(num):
        for dst in range(num):
            if src == dst:
                continue
            if torch.cuda.can_device_access_peer(src, dst):
                try:
                    torch.cuda.device(src).enable_peer_access(dst)
                except RuntimeError:
                    pass


def _bucket_order() -> List[Tuple[int, int]]:
    """Return (device_id, bucket_index) pairs ordered by device id (proxy for distance)."""
    return [(idx, idx) for idx in range(torch.cuda.device_count())]


class OptimizedDdpNvlinkOverlapBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Overlapped gradient transfers over NVLink with peer access enabled."""

    def __init__(self):
        super().__init__()
        self.models: List[nn.Linear] = []
        self._inputs: List[List[torch.Tensor]] = []
        self.output: Optional[torch.Tensor] = None
        self.microbatches = 2
        self.batch_size = 8
        self.hidden = 512
        self.root_device = torch.device("cuda:0")
        self._reduce_buffers: List[torch.Tensor] = []
        self._root_grad_staging: List[List[torch.Tensor]] = []
        self._grad_ready_events: List[List[torch.cuda.Event]] = []
        self._update_buffers: List[torch.Tensor] = []
        tokens = self.batch_size * self.hidden * self.microbatches
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size * self.microbatches),
            tokens_per_iteration=float(tokens),
        )
        self.comm_stream = torch.cuda.Stream(device=self.root_device)

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        _enable_peer_access()
        num = torch.cuda.device_count()
        skip_if_insufficient_gpus(2)
        require_peer_access(0, 1)
        for rank in range(num):
            device = f"cuda:{rank}"
            self.models.append(nn.Linear(self.hidden, self.hidden).to(device))
        self._inputs = []
        for _micro in range(self.microbatches):
            micro_inputs: List[torch.Tensor] = []
            for model in self.models:
                micro_inputs.append(torch.randn(self.batch_size, self.hidden, device=model.weight.device))
            self._inputs.append(micro_inputs)
        self._reduce_buffers = [
            torch.zeros_like(self.models[0].weight, device=self.root_device)
            for _ in range(self.microbatches)
        ]
        self._root_grad_staging = [
            [
                torch.empty_like(self.models[0].weight, device=self.root_device)
                for _ in self.models
            ]
            for _ in range(self.microbatches)
        ]
        self._grad_ready_events = [
            [torch.cuda.Event() for _ in self.models]
            for _ in range(self.microbatches)
        ]
        self._update_buffers = [
            torch.empty_like(model.weight, device=model.weight.device)
            for model in self.models
        ]
        self._synchronize()

    def _async_reduce_to_root(self, grads: List[torch.Tensor], buffer_index: int) -> torch.Tensor:
        """Asynchronously accumulate gradients on the root device."""
        root_buf = self._reduce_buffers[buffer_index]
        root_buf.zero_()
        event_row = self._grad_ready_events[buffer_index]
        staging_row = self._root_grad_staging[buffer_index]
        for idx, _ in enumerate(grads):
            event_row[idx].record()

        with torch.cuda.stream(self.comm_stream):
            for idx, g in enumerate(grads):
                evt = event_row[idx]
                self.comm_stream.wait_event(evt)
                if g.device == self.root_device:
                    root_buf.add_(g)
                else:
                    staging = staging_row[idx]
                    staging.copy_(g, non_blocking=True)
                    root_buf.add_(staging)
        return root_buf

    def benchmark_fn(self) -> None:
        assert self.models
        with self._nvtx_range("optimized_ddp_multigpu_nvlink_overlap"):
            reduction_results: List[torch.Tensor] = []
            # Process microbatches; overlap reduction of previous with compute of next
            for micro in range(self.microbatches):
                grads = []
                for model_idx, model in enumerate(self.models):
                    x = self._inputs[micro][model_idx]
                    y = model(x)
                    loss = y.pow(2).mean()
                    loss.backward()
                    grads.append(model.weight.grad)

                # Reorder buckets (simple proxy: ascending device id)
                ordered = sorted(zip(grads, _bucket_order()), key=lambda kv: kv[1][0])
                ordered_grads = [g for g, _ in ordered]

                reduction_results.append(self._async_reduce_to_root(ordered_grads, micro))

            # Finalize reductions and apply updates
            torch.cuda.current_stream(self.root_device).wait_stream(self.comm_stream)
            for model_idx, (model, root_buf) in enumerate(zip(self.models, reduction_results)):
                if root_buf.device != model.weight.device:
                    root_local = self._update_buffers[model_idx]
                    root_local.copy_(root_buf, non_blocking=True)
                else:
                    root_local = root_buf
                with torch.no_grad():
                    root_local.mul_(1.0 / len(self.models))
                    model.weight.add_(-1e-3, root_local)
                    model.weight.grad.zero_()
                    model.bias.grad.zero_()
            self.output = self.models[0].weight.detach()

    def capture_verification_payload(self) -> None:
        if self.output is None or not self._inputs:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        x_probe = self._inputs[0][0]
        param_count = sum(p.numel() for m in self.models for p in m.parameters())
        weight_slice = self.output[:8, :8].to(dtype=torch.float32).clone()
        self._set_verification_payload(
            inputs={"x": x_probe},
            output=weight_slice,
            batch_size=int(x_probe.shape[0]),
            parameter_count=param_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.models.clear()
        self._inputs = []
        self._reduce_buffers = []
        self._root_grad_staging = []
        self._grad_ready_events = []
        self._update_buffers = []
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if not self.models:
            return "Models not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    return OptimizedDdpNvlinkOverlapBenchmark()
