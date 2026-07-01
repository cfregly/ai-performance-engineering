"""Shared helpers for the parameterized CUDA graph launch lab.

The lab keeps the compute path PyTorch-first:
- pinned host request slots hold per-request inputs, scalar scale values, and outputs
- stable device buffers back the captured CUDA graph
- the optimized path mutates executable memcpy-node params between replays
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig

try:
    from cuda.bindings import runtime as cudart
except Exception as exc:  # pragma: no cover - exercised on hosts without cuda-bindings
    cudart = None  # type: ignore[assignment]
    _CUDA_BINDINGS_IMPORT_ERROR = exc
else:
    _CUDA_BINDINGS_IMPORT_ERROR = None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


@dataclass(frozen=True)
class ParameterizedGraphConfig:
    batch_size: int = _env_int("AISP_PARAMETERIZED_CUDA_GRAPHS_BATCH_SIZE", 32)
    hidden_size: int = _env_int("AISP_PARAMETERIZED_CUDA_GRAPHS_HIDDEN_SIZE", 1024)
    expansion_factor: int = _env_int("AISP_PARAMETERIZED_CUDA_GRAPHS_EXPANSION", 2)
    request_slots: int = _env_int("AISP_PARAMETERIZED_CUDA_GRAPHS_REQUEST_SLOTS", 4)
    iterations: int = _env_int("AISP_PARAMETERIZED_CUDA_GRAPHS_ITERATIONS", 16)
    warmup: int = _env_int("AISP_PARAMETERIZED_CUDA_GRAPHS_WARMUP", 10)


SlotMemcpyBinding = Tuple[int, int, int, int, int, int, int, int, int]


class _ResidualScaleBlock(nn.Module):
    """Two-layer residual MLP with a request-dependent scale tensor."""

    def __init__(self, hidden_size: int, expansion_factor: int) -> None:
        super().__init__()
        expanded = hidden_size * expansion_factor
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, expanded)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(expanded, hidden_size)

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.act(self.fc1(self.norm(x))))
        return x + hidden * scale

    def forward_into(self, x: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.act(self.fc1(self.norm(x))))
        torch.mul(hidden, scale, out=out)
        out.add_(x)
        return out


class ParameterizedGraphBenchmarkBase(VerificationPayloadMixin, BaseBenchmark):
    """Shared benchmark state for recapture and parameterized replay variants."""

    def __init__(self, cfg: Optional[ParameterizedGraphConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ParameterizedGraphConfig()
        self.model: Optional[_ResidualScaleBlock] = None
        self.capture_stream: Optional[torch.cuda.Stream] = None
        self.device_input: Optional[torch.Tensor] = None
        self.device_scale: Optional[torch.Tensor] = None
        self.device_output: Optional[torch.Tensor] = None
        self.host_inputs: List[torch.Tensor] = []
        self.host_scales: List[torch.Tensor] = []
        self.host_outputs: List[torch.Tensor] = []
        self._slot_memcpy_bindings: List[SlotMemcpyBinding] = []
        self.parameter_count: int = 0
        self._slot_cursor = 0
        self._last_slot = 0
        self._last_graph: Optional[torch.cuda.CUDAGraph] = None
        self._verify_input_buffer: Optional[torch.Tensor] = None
        self._verify_scale_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            samples_per_iteration=float(self.cfg.batch_size),
            bytes_per_iteration=float(
                (self.cfg.batch_size * self.cfg.hidden_size * 4 * 2) + 4
            ),
        )

    @property
    def graph_capture_enabled(self) -> bool:
        return True

    @property
    def optimized_parameter_updates(self) -> bool:
        return False

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        self.capture_stream = torch.cuda.Stream()
        self.model = _ResidualScaleBlock(
            hidden_size=self.cfg.hidden_size,
            expansion_factor=self.cfg.expansion_factor,
        ).to(device=self.device, dtype=torch.float32)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

        self.device_input = torch.empty(
            (self.cfg.batch_size, self.cfg.hidden_size),
            device=self.device,
            dtype=torch.float32,
        )
        self.device_scale = torch.empty((1,), device=self.device, dtype=torch.float32)
        self.device_output = torch.empty_like(self.device_input)

        self._build_request_slots()
        self._verify_input_buffer = torch.empty_like(self.host_inputs[0])
        self._verify_scale_buffer = torch.empty_like(self.host_scales[0])
        self._verify_output_buffer = torch.empty((2, 16), dtype=torch.float32, device="cpu")
        self._warmup_eager_path()

    def _build_request_slots(self) -> None:
        self.host_inputs = [torch.empty(0) for _ in range(self.cfg.request_slots)]
        self.host_scales = [torch.empty(0) for _ in range(self.cfg.request_slots)]
        self.host_outputs = [torch.empty(0) for _ in range(self.cfg.request_slots)]
        for slot_idx in range(self.cfg.request_slots):
            host_input = torch.randn(
                (self.cfg.batch_size, self.cfg.hidden_size),
                dtype=torch.float32,
                device="cpu",
            ).pin_memory()
            scale_value = 0.5 + (0.25 * slot_idx)
            host_scale = torch.tensor(
                [scale_value],
                dtype=torch.float32,
                device="cpu",
            ).pin_memory()
            host_output = torch.empty_like(host_input, device="cpu").pin_memory()
            self.host_inputs[slot_idx] = host_input
            self.host_scales[slot_idx] = host_scale
            self.host_outputs[slot_idx] = host_output
        self._refresh_slot_memcpy_bindings()

    def _warmup_eager_path(self) -> None:
        if self.capture_stream is None:
            raise RuntimeError("capture stream not initialized")
        with torch.cuda.stream(self.capture_stream):
            for _ in range(3):
                self._schedule_request_program(0)
        torch.cuda.synchronize()

    def _require_runtime_state(self) -> Tuple[_ResidualScaleBlock, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("model is not initialized")
        if self.device_input is None or self.device_scale is None or self.device_output is None:
            raise RuntimeError("device buffers are not initialized")
        return self.model, self.device_input, self.device_scale, self.device_output

    def _schedule_request_program(
        self,
        slot_idx: int,
        *,
        host_output: Optional[torch.Tensor] = None,
    ) -> None:
        model, device_input, device_scale, device_output = self._require_runtime_state()
        slot_input = self.host_inputs[slot_idx]
        slot_scale = self.host_scales[slot_idx]
        slot_output = host_output if host_output is not None else self.host_outputs[slot_idx]

        with torch.inference_mode():
            device_input.copy_(slot_input, non_blocking=True)
            device_scale.copy_(slot_scale, non_blocking=True)
            model.forward_into(device_input, device_scale, device_output)
            slot_output.copy_(device_output, non_blocking=True)

    def _refresh_slot_memcpy_bindings(self) -> None:
        """Precompute stable graph memcpy arguments for each request slot."""

        _, device_input, device_scale, device_output = self._require_runtime_state()
        device_input_ptr = device_input.data_ptr()
        device_scale_ptr = device_scale.data_ptr()
        device_output_ptr = device_output.data_ptr()
        self._slot_memcpy_bindings = [
            (
                device_input_ptr,
                host_input.data_ptr(),
                host_input.numel() * host_input.element_size(),
                device_scale_ptr,
                host_scale.data_ptr(),
                host_scale.numel() * host_scale.element_size(),
                host_output.data_ptr(),
                device_output_ptr,
                host_output.numel() * host_output.element_size(),
            )
            for host_input, host_scale, host_output in zip(
                self.host_inputs,
                self.host_scales,
                self.host_outputs,
                strict=True,
            )
        ]

    def _next_slot(self) -> int:
        slot_idx = self._slot_cursor % self.cfg.request_slots
        self._slot_cursor += 1
        self._last_slot = slot_idx
        return slot_idx

    def _current_output_slice(self) -> torch.Tensor:
        if self._verify_output_buffer is None:
            raise RuntimeError("verification output buffer is not initialized")
        host_output = self.host_outputs[self._last_slot]
        self._verify_output_buffer.copy_(host_output[:2, :16])
        return self._verify_output_buffer

    def _run_verification_slot(self, slot_idx: int) -> None:
        self._last_slot = slot_idx
        self._schedule_request_program(slot_idx)

    def capture_verification_payload(self) -> None:
        slot_idx = 0
        self._run_verification_slot(slot_idx)
        self._synchronize()
        if self._verify_input_buffer is None or self._verify_scale_buffer is None:
            raise RuntimeError("verification input buffers are not initialized")
        self._verify_input_buffer.copy_(self.host_inputs[slot_idx])
        self._verify_scale_buffer.copy_(self.host_scales[slot_idx])
        self._set_verification_payload(
            inputs={
                "x": self._verify_input_buffer,
                "scale": self._verify_scale_buffer,
            },
            output=self._current_output_slice(),
            batch_size=self.cfg.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", True)),
            },
            output_tolerance=(1e-5, 1e-5),
            signature_overrides={
                "graph_capture_enabled": self.graph_capture_enabled,
                "num_streams": 1,
            },
        )

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=self.cfg.iterations,
            warmup=self.cfg.warmup,
            measurement_timeout_seconds=180,
        )

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return {
            "parameterized_graph_launch.batch_size": float(self.cfg.batch_size),
            "parameterized_graph_launch.hidden_size": float(self.cfg.hidden_size),
            "parameterized_graph_launch.request_slots": float(self.cfg.request_slots),
            "parameterized_graph_launch.recapture_per_iteration": float(
                not self.optimized_parameter_updates
            ),
            "parameterized_graph_launch.exec_memcpy_param_updates": float(
                3 if self.optimized_parameter_updates else 0
            ),
        }

    def validate_result(self) -> Optional[str]:
        if not torch.isfinite(self.host_outputs[self._last_slot]).all():
            return "Non-finite host output detected"
        return None

    def teardown(self) -> None:
        self._last_graph = None
        self.capture_stream = None
        self.model = None
        self.device_input = None
        self.device_scale = None
        self.device_output = None
        self.host_inputs = []
        self.host_scales = []
        self.host_outputs = []
        self._slot_memcpy_bindings = []
        self._verify_input_buffer = None
        self._verify_scale_buffer = None
        self._verify_output_buffer = None
        super().teardown()


class ParameterizedGraphRecaptureBenchmark(ParameterizedGraphBenchmarkBase):
    """Baseline: recapture a new graph for each request slot."""

    def benchmark_fn(self) -> None:
        if self.capture_stream is None:
            raise RuntimeError("capture stream not initialized")
        slot_idx = self._next_slot()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.capture_stream):
            self._schedule_request_program(slot_idx)
        self._last_graph = graph
        graph.replay()

    def _run_verification_slot(self, slot_idx: int) -> None:
        if self.capture_stream is None:
            raise RuntimeError("capture stream not initialized")
        self._last_slot = slot_idx
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.capture_stream):
            self._schedule_request_program(slot_idx)
        graph.replay()


class ParameterizedGraphReplayBenchmark(ParameterizedGraphBenchmarkBase):
    """Optimized: patch memcpy-node params on an already-instantiated graph."""

    def __init__(self, cfg: Optional[ParameterizedGraphConfig] = None) -> None:
        super().__init__(cfg)
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_exec: Optional[int] = None
        self._input_node = None
        self._scale_node = None
        self._output_node = None

    @property
    def optimized_parameter_updates(self) -> bool:
        return True

    def setup(self) -> None:
        if cudart is None:
            raise RuntimeError(
                "cuda-bindings runtime APIs are required for the optimized parameterized replay path"
            ) from _CUDA_BINDINGS_IMPORT_ERROR
        super().setup()
        if self.capture_stream is None:
            raise RuntimeError("capture stream not initialized")

        self.graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self._schedule_request_program(0)
        self.graph.instantiate()
        self._graph_exec = self.graph.raw_cuda_graph_exec()
        self._bind_memcpy_nodes(slot_idx=0)

    def _bind_memcpy_nodes(self, *, slot_idx: int) -> None:
        if cudart is None or self.graph is None:
            raise RuntimeError("cuda graph runtime bindings are unavailable")

        raw_graph = self.graph.raw_cuda_graph()
        _, _, node_count = cudart.cudaGraphGetNodes(raw_graph)
        _, nodes, _ = cudart.cudaGraphGetNodes(raw_graph, node_count)

        node_by_endpoints = {}
        for node in nodes:
            _, node_type = cudart.cudaGraphNodeGetType(node)
            if int(node_type) != int(cudart.cudaGraphNodeType.cudaGraphNodeTypeMemcpy):
                continue
            _, params = cudart.cudaGraphMemcpyNodeGetParams(node)
            node_by_endpoints[(int(params.srcPtr.ptr), int(params.dstPtr.ptr))] = node

        try:
            (
                device_input_ptr,
                host_input_ptr,
                _input_nbytes,
                device_scale_ptr,
                host_scale_ptr,
                _scale_nbytes,
                host_output_ptr,
                device_output_ptr,
                _output_nbytes,
            ) = self._slot_memcpy_bindings[slot_idx]
        except IndexError as exc:
            raise RuntimeError("slot memcpy bindings are not initialized") from exc
        input_key = (host_input_ptr, device_input_ptr)
        scale_key = (host_scale_ptr, device_scale_ptr)
        output_key = (device_output_ptr, host_output_ptr)

        try:
            self._input_node = node_by_endpoints[input_key]
            self._scale_node = node_by_endpoints[scale_key]
            self._output_node = node_by_endpoints[output_key]
        except KeyError as exc:
            known = ", ".join(
                f"(src=0x{src:x}, dst=0x{dst:x})" for src, dst in sorted(node_by_endpoints.keys())
            )
            raise RuntimeError(
                "Failed to locate captured memcpy nodes for input/scale/output bindings. "
                f"Known memcpy endpoints: {known}"
            ) from exc

    def _check_cudart(self, result: Sequence[object], context: str) -> None:
        err = int(result[0]) if result else 0
        if err != 0:
            raise RuntimeError(f"{context} failed with cudaError={err}")

    def _update_exec_params_for_slot(self, slot_idx: int) -> None:
        if cudart is None or self._graph_exec is None:
            raise RuntimeError("graph exec is not initialized")
        if self._input_node is None or self._scale_node is None or self._output_node is None:
            raise RuntimeError("memcpy nodes were not bound")
        try:
            (
                device_input_ptr,
                host_input_ptr,
                input_nbytes,
                device_scale_ptr,
                host_scale_ptr,
                scale_nbytes,
                host_output_ptr,
                device_output_ptr,
                output_nbytes,
            ) = self._slot_memcpy_bindings[slot_idx]
        except IndexError as exc:
            raise RuntimeError("slot memcpy bindings are not initialized") from exc

        self._check_cudart(
            cudart.cudaGraphExecMemcpyNodeSetParams1D(
                self._graph_exec,
                self._input_node,
                device_input_ptr,
                host_input_ptr,
                input_nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            ),
            "updating input memcpy node",
        )
        self._check_cudart(
            cudart.cudaGraphExecMemcpyNodeSetParams1D(
                self._graph_exec,
                self._scale_node,
                device_scale_ptr,
                host_scale_ptr,
                scale_nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            ),
            "updating scale memcpy node",
        )
        self._check_cudart(
            cudart.cudaGraphExecMemcpyNodeSetParams1D(
                self._graph_exec,
                self._output_node,
                host_output_ptr,
                device_output_ptr,
                output_nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            ),
            "updating output memcpy node",
        )

    def benchmark_fn(self) -> None:
        if self.graph is None:
            raise RuntimeError("parameterized graph was not captured")
        slot_idx = self._next_slot()
        self._update_exec_params_for_slot(slot_idx)
        self.graph.replay()

    def _run_verification_slot(self, slot_idx: int) -> None:
        if self.graph is None:
            raise RuntimeError("parameterized graph was not captured")
        self._last_slot = slot_idx
        self._update_exec_params_for_slot(slot_idx)
        self.graph.replay()

    def teardown(self) -> None:
        self.graph = None
        self._graph_exec = None
        self._input_node = None
        self._scale_node = None
        self._output_node = None
        super().teardown()


__all__ = [
    "ParameterizedGraphConfig",
    "ParameterizedGraphRecaptureBenchmark",
    "ParameterizedGraphReplayBenchmark",
]
