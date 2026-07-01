"""Shared harness logic for prefill/decode disaggregation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.gpu_requirements import require_peer_access
from core.benchmark.wrapper_utils import attach_benchmark_metadata as attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


@dataclass(frozen=True)
class PrefillDecodeDisaggConfig:
    """Shape configuration for the disaggregated prefill/decode benchmark family."""

    batch_size: int = 8
    prefill_length: int = 1024
    decode_length: int = 64
    hidden_size: int = 2048

    @property
    def tokens_per_request(self) -> int:
        return self.prefill_length + self.decode_length


class PrefillDecodeDisaggBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Parameterized benchmark for host-staged or device-local KV handoff."""

    multi_gpu_required = False
    allowed_benchmark_fn_antipatterns: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        use_host_staging: bool,
        multi_gpu: bool,
        label: str,
        cfg: Optional[PrefillDecodeDisaggConfig] = None,
    ) -> None:
        super().__init__()
        self.use_host_staging = bool(use_host_staging)
        self.multi_gpu = bool(multi_gpu)
        self.multi_gpu_required = self.multi_gpu
        self.label = label
        self.cfg = cfg or PrefillDecodeDisaggConfig()
        self.batch_size = int(self.cfg.batch_size)
        self.prefill_length = int(self.cfg.prefill_length)
        self.decode_length = int(self.cfg.decode_length)
        self.hidden_size = int(self.cfg.hidden_size)
        self._decode_step_range = range(self.decode_length)

        self._workload: Optional[WorkloadMetadata] = None
        self._refresh_workload_metadata()

        self.pairs: list[tuple[torch.device, torch.device]] = []
        self.prefill_models: list[nn.Module] = []
        self.decode_models: list[nn.Module] = []
        self.prefill_inputs: list[torch.Tensor] = []
        self._prefill_weight_t: dict[int, torch.Tensor] = {}
        self._decode_weight_t: dict[int, torch.Tensor] = {}
        self._decode_token_staging: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._host_staging: dict[str, torch.Tensor] = {}
        self._handoff_staging: dict[str, torch.Tensor] = {}
        self._request_groups: list[tuple[torch.device, nn.Module, nn.Module, torch.Tensor]] = []
        self._request_output_groups: list[
            tuple[int, torch.device, nn.Module, nn.Module, torch.Tensor]
        ] = []
        self._verify_probe: Optional[torch.Tensor] = None
        self._output_shards: Optional[list[torch.Tensor]] = None
        self._output_shard_count = 0
        self._verify_output_stack: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._parameter_count = 0

    def _refresh_workload_metadata(self) -> None:
        tokens = self.batch_size * (self.prefill_length + self.decode_length)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def _resolve_pairs(self) -> list[tuple[torch.device, torch.device]]:
        if not self.multi_gpu:
            return [(self.device, self.device)]

        device_count = torch.cuda.device_count()
        if device_count < 2:
            raise RuntimeError("SKIPPED: prefill/decode disaggregation requires >=2 GPUs")
        if device_count % 2 != 0:
            raise RuntimeError(
                "SKIPPED: requires even GPU count for prefill/decode pairing; set CUDA_VISIBLE_DEVICES accordingly"
            )
        return [
            (torch.device(f"cuda:{idx}"), torch.device(f"cuda:{idx + 1}"))
            for idx in range(0, device_count, 2)
        ]

    def _require_peer_paths(self, pairs: Sequence[tuple[torch.device, torch.device]]) -> None:
        if self.use_host_staging:
            return
        for prefill_device, decode_device in pairs:
            prefill_idx = prefill_device.index
            decode_idx = decode_device.index
            if prefill_idx is None or decode_idx is None or prefill_idx == decode_idx:
                continue
            require_peer_access(prefill_idx, decode_idx)

    def _split_sizes(self, num_pairs: int) -> list[int]:
        if self.batch_size < num_pairs:
            self.batch_size = num_pairs
            self._refresh_workload_metadata()

        base = self.batch_size // num_pairs
        remainder = self.batch_size % num_pairs
        if base == 0 and remainder == 0:
            raise RuntimeError("batch_size must be >= number of GPU pairs")
        return [base + (1 if idx < remainder else 0) for idx in range(num_pairs)]

    def _empty_cpu_staging(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        try:
            return torch.empty(shape, device="cpu", dtype=dtype, pin_memory=True)
        except RuntimeError:
            return torch.empty(shape, device="cpu", dtype=dtype)

    @staticmethod
    def _staging_numel(shape: torch.Size) -> int:
        numel = 1
        for dim in shape:
            numel *= int(dim)
        return numel

    @staticmethod
    def _device_matches(actual: torch.device, expected: torch.device) -> bool:
        if actual == expected:
            return True
        if actual.type != expected.type:
            return False
        if (
            actual.type == "cuda"
            and expected.index is None
            and torch.cuda.is_available()
        ):
            return actual.index == torch.cuda.current_device()
        return False

    def _decode_staging_view(
        self,
        staging_key: str,
        shape: torch.Size,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        numel = self._staging_numel(shape)
        buffer = self._handoff_staging.get(staging_key)
        if (
            buffer is None
            or not self._device_matches(buffer.device, device)
            or buffer.dtype != dtype
            or buffer.numel() < numel
        ):
            buffer = torch.empty(numel, device=device, dtype=dtype)
            self._handoff_staging[staging_key] = buffer
        return buffer[:numel].view(shape)

    def _host_staging_view(
        self,
        staging_key: str,
        shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        numel = self._staging_numel(shape)
        buffer = self._host_staging.get(staging_key)
        if (
            buffer is None
            or buffer.device.type != "cpu"
            or buffer.dtype != dtype
            or buffer.numel() < numel
        ):
            buffer = self._empty_cpu_staging(torch.Size((numel,)), dtype)
            self._host_staging[staging_key] = buffer
        return buffer[:numel].view(shape)

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for prefill/decode disaggregation")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.pairs = self._resolve_pairs()
        self._require_peer_paths(self.pairs)
        split_sizes = self._split_sizes(len(self.pairs))

        data_gen = torch.Generator().manual_seed(1234)
        cpu_inputs = torch.randn(
            self.batch_size,
            self.prefill_length,
            self.hidden_size,
            generator=data_gen,
            dtype=torch.bfloat16,
        )

        self.prefill_models = []
        self.decode_models = []
        self.prefill_inputs = []
        self._prefill_weight_t = {}
        self._decode_weight_t = {}
        self._decode_token_staging = {}
        self._host_staging = {}
        self._handoff_staging = {}
        self._request_groups = []
        self._request_output_groups = []
        self._decode_step_range = range(self.decode_length)
        self._output_shards = []
        self._output_shard_count = 0
        probe_width = min(256, self.hidden_size)
        probe_shape = torch.Size((1, 1, probe_width))
        self._verify_probe = self._empty_cpu_staging(probe_shape, torch.bfloat16)
        verify_shape = torch.Size((min(2, self.batch_size), min(256, self.hidden_size)))
        self._verify_output_stack = self._empty_cpu_staging(verify_shape, torch.bfloat16)
        self._verify_output_buffer = self._empty_cpu_staging(verify_shape, torch.float32)
        self._parameter_count = 0

        offset = 0
        for (prefill_device, decode_device), split_size in zip(self.pairs, split_sizes):
            prefill_model = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(
                prefill_device,
                dtype=torch.bfloat16,
            ).eval()
            decode_model = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(
                decode_device,
                dtype=torch.bfloat16,
            ).eval()
            self.prefill_models.append(prefill_model)
            self.decode_models.append(decode_model)
            self._prefill_weight_t[id(prefill_model)] = prefill_model.weight.detach().t()
            self._decode_weight_t[id(decode_model)] = decode_model.weight.detach().t()
            self._parameter_count += sum(p.numel() for p in prefill_model.parameters())
            self._parameter_count += sum(p.numel() for p in decode_model.parameters())

            slice_end = offset + split_size
            batch_slice = cpu_inputs[offset:slice_end].to(prefill_device)
            self.prefill_inputs.append(batch_slice)
            self._request_groups.extend(
                (decode_device, prefill_model, decode_model, batch_slice[idx : idx + 1])
                for idx in range(batch_slice.shape[0])
            )
            staging_key = str(decode_device)
            staging_shape = torch.Size((1, self.prefill_length, self.hidden_size))
            staging_numel = self._staging_numel(staging_shape)
            self._handoff_staging[staging_key] = torch.empty(
                staging_numel,
                device=decode_device,
                dtype=torch.bfloat16,
            )
            first_decode_token = torch.empty(
                1,
                1,
                self.hidden_size,
                device=decode_device,
                dtype=torch.bfloat16,
            )
            self._decode_token_staging[staging_key] = (
                first_decode_token,
                torch.empty_like(first_decode_token),
            )
            if self.use_host_staging:
                self._host_staging[staging_key] = self._empty_cpu_staging(
                    torch.Size((staging_numel,)),
                    torch.bfloat16,
                )
            offset = slice_end
        self._request_output_groups = [
            (output_idx, decode_device, prefill_model, decode_model, request)
            for output_idx, (decode_device, prefill_model, decode_model, request) in enumerate(
                self._request_groups
            )
        ]
        self._output_shards = [
            torch.empty(self.hidden_size, device=decode_device, dtype=torch.bfloat16)
            for _, decode_device, _, _, _ in self._request_output_groups
        ]
        self._output_shard_count = len(self._output_shards)

        self._verify_probe.copy_(
            self.prefill_inputs[0][:1, :1, :probe_width],
            non_blocking=False,
        )
        for prefill_device, decode_device in self.pairs:
            torch.cuda.synchronize(prefill_device)
            torch.cuda.synchronize(decode_device)

    def _decode_token_buffer_pair(
        self,
        staging_key: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        buffers = self._decode_token_staging.get(staging_key)
        shape = torch.Size((1, 1, self.hidden_size))
        if (
            buffers is None
            or not self._device_matches(buffers[0].device, device)
            or buffers[0].dtype != dtype
            or buffers[0].shape != shape
        ):
            first = torch.empty(shape, device=device, dtype=dtype)
            buffers = (first, torch.empty_like(first))
            self._decode_token_staging[staging_key] = buffers
        return buffers

    def _prefill_into_decode_kv(
        self,
        prefill_model: nn.Module,
        request: torch.Tensor,
        decode_device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.use_host_staging or not self._device_matches(
            request.device,
            decode_device,
        ):
            return None
        weight_t = self._prefill_weight_t.get(id(prefill_model))
        if weight_t is None:
            return None
        staging_key = str(decode_device)
        output_shape = request.shape[:-1] + torch.Size((int(weight_t.shape[1]),))
        decode_buf = self._decode_staging_view(
            staging_key,
            output_shape,
            device=decode_device,
            dtype=weight_t.dtype,
        )
        torch.matmul(request, weight_t, out=decode_buf)
        bias = getattr(prefill_model, "bias", None)
        if isinstance(bias, torch.Tensor):
            decode_buf.add_(bias.detach())
        return decode_buf

    def _decode_into_output_shard(
        self,
        decode_model: nn.Module,
        kv_decode: torch.Tensor,
        decode_device: torch.device,
        output_shard: torch.Tensor,
    ) -> None:
        token_state = kv_decode[:, -1:, :]
        decode_weight_t = self._decode_weight_t.get(id(decode_model))
        if decode_weight_t is None:
            for _ in self._decode_step_range:
                token_state = decode_model(token_state)
            output_shard.copy_(token_state.reshape(-1), non_blocking=True)
            return

        if self.decode_length <= 0:
            output_shard.copy_(token_state.reshape(-1), non_blocking=True)
            return

        token_buffers = self._decode_token_buffer_pair(
            str(decode_device),
            device=decode_device,
            dtype=decode_weight_t.dtype,
        )
        bias = getattr(decode_model, "bias", None)
        last_step_idx = self.decode_length - 1
        output_state = output_shard.view(1, 1, self.hidden_size)
        for step_idx in self._decode_step_range:
            next_state = (
                output_state
                if step_idx == last_step_idx
                else token_buffers[step_idx & 1]
            )
            torch.matmul(token_state, decode_weight_t, out=next_state)
            if isinstance(bias, torch.Tensor):
                next_state.add_(bias.detach())
            token_state = next_state

    def _handoff_kv(self, prefill_out: torch.Tensor, decode_device: torch.device) -> torch.Tensor:
        staging_key = str(decode_device)
        if not self.use_host_staging and self._device_matches(
            prefill_out.device,
            decode_device,
        ):
            return prefill_out

        decode_buf = self._decode_staging_view(
            staging_key,
            prefill_out.shape,
            device=decode_device,
            dtype=prefill_out.dtype,
        )
        if self.use_host_staging:
            host_buf = self._host_staging_view(
                staging_key,
                prefill_out.shape,
                prefill_out.dtype,
            )
            host_buf.copy_(prefill_out, non_blocking=False)
            decode_buf.copy_(host_buf, non_blocking=False)
            return decode_buf
        decode_buf.copy_(prefill_out, non_blocking=True)
        return decode_buf

    def benchmark_fn(self) -> None:
        if (
            not self.prefill_models
            or not self.decode_models
            or not self.prefill_inputs
            or not self._request_groups
            or not self._request_output_groups
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")

        outputs = self._output_shards
        if outputs is None or self._output_shard_count != self.batch_size:
            raise RuntimeError("Decode output shards not initialized")
        with self._nvtx_range(self.label):
            with torch.inference_mode():
                for (
                    output_idx,
                    decode_device,
                    prefill_model,
                    decode_model,
                    request,
                ) in self._request_output_groups:
                    kv_decode = self._prefill_into_decode_kv(
                        prefill_model,
                        request,
                        decode_device,
                    )
                    if kv_decode is None:
                        prefill_out = prefill_model(request)
                        kv_decode = self._handoff_kv(prefill_out, decode_device)
                    self._decode_into_output_shard(
                        decode_model,
                        kv_decode,
                        decode_device,
                        outputs[output_idx],
                    )

        self._output_shards = outputs

    def capture_verification_payload(self) -> None:
        if self._output_shards is None or self._verify_probe is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        if self._verify_output_stack is None:
            raise RuntimeError("Verification output stack not initialized")
        if self._verify_output_buffer is None:
            raise RuntimeError("Verification output buffer not initialized")

        selected_count = min(2, len(self._output_shards))
        verify_width = self._verify_output_stack.shape[1]
        for output_idx in range(selected_count):
            self._verify_output_stack[output_idx].copy_(
                self._output_shards[output_idx][:verify_width],
                non_blocking=False,
            )
        verify_output = self._verify_output_buffer[:selected_count]
        verify_output.copy_(self._verify_output_stack[:selected_count], non_blocking=False)
        self._set_verification_payload(
            inputs={"probe": self._verify_probe},
            output=verify_output,
            batch_size=int(self.batch_size),
            parameter_count=int(self._parameter_count),
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.prefill_models = []
        self.decode_models = []
        self.prefill_inputs = []
        self.pairs = []
        self._prefill_weight_t = {}
        self._decode_weight_t = {}
        self._decode_token_staging = {}
        self._host_staging = {}
        self._handoff_staging = {}
        self._request_groups = []
        self._request_output_groups = []
        self._verify_probe = None
        self._output_shards = None
        self._output_shard_count = 0
        self._verify_output_stack = None
        self._verify_output_buffer = None
        self._parameter_count = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=5,
            warmup=5,
            multi_gpu_required=self.multi_gpu,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


class HostStagedPrefillDecodeDisaggBenchmark(PrefillDecodeDisaggBenchmark):
    """Host-staged KV handoff benchmark."""

    allowed_benchmark_fn_antipatterns = ("host_transfer",)

    def __init__(
        self,
        *,
        multi_gpu: bool,
        label: str,
        cfg: Optional[PrefillDecodeDisaggConfig] = None,
    ) -> None:
        super().__init__(
            use_host_staging=True,
            multi_gpu=multi_gpu,
            label=label,
            cfg=cfg,
        )


class PeerPrefillDecodeDisaggBenchmark(PrefillDecodeDisaggBenchmark):
    """Peer/direct KV handoff benchmark."""

    allowed_benchmark_fn_antipatterns: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        multi_gpu: bool,
        label: str,
        cfg: Optional[PrefillDecodeDisaggConfig] = None,
    ) -> None:
        super().__init__(
            use_host_staging=False,
            multi_gpu=multi_gpu,
            label=label,
            cfg=cfg,
        )
