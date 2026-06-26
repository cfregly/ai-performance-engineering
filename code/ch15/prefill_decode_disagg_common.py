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

        self._workload: Optional[WorkloadMetadata] = None
        self._refresh_workload_metadata()

        self.pairs: list[tuple[torch.device, torch.device]] = []
        self.prefill_models: list[nn.Module] = []
        self.decode_models: list[nn.Module] = []
        self.prefill_inputs: list[torch.Tensor] = []
        self._host_staging: dict[str, torch.Tensor] = {}
        self._handoff_staging: dict[str, torch.Tensor] = {}
        self._verify_probe: Optional[torch.Tensor] = None
        self._output_shards: Optional[list[torch.Tensor]] = None
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
        self._host_staging = {}
        self._handoff_staging = {}
        self._output_shards = None
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
            self._parameter_count += sum(p.numel() for p in prefill_model.parameters())
            self._parameter_count += sum(p.numel() for p in decode_model.parameters())

            slice_end = offset + split_size
            batch_slice = cpu_inputs[offset:slice_end].to(prefill_device)
            self.prefill_inputs.append(batch_slice)
            staging_key = str(decode_device)
            staging_shape = torch.Size((1, self.prefill_length, self.hidden_size))
            self._handoff_staging[staging_key] = torch.empty(
                staging_shape,
                device=decode_device,
                dtype=torch.bfloat16,
            )
            if self.use_host_staging:
                self._host_staging[staging_key] = self._empty_cpu_staging(
                    staging_shape,
                    torch.bfloat16,
                )
            offset = slice_end

        self._verify_probe = self.prefill_inputs[0][:1, :1, :256].detach()
        for prefill_device, decode_device in self.pairs:
            torch.cuda.synchronize(prefill_device)
            torch.cuda.synchronize(decode_device)

    def _handoff_kv(self, prefill_out: torch.Tensor, decode_device: torch.device) -> torch.Tensor:
        staging_key = str(decode_device)
        decode_buf = self._handoff_staging.get(staging_key)
        if decode_buf is None or decode_buf.shape != prefill_out.shape:
            decode_buf = torch.empty_like(prefill_out, device=decode_device)
            self._handoff_staging[staging_key] = decode_buf
        if self.use_host_staging:
            host_buf = self._host_staging.get(staging_key)
            if host_buf is None or host_buf.shape != prefill_out.shape:
                host_buf = self._empty_cpu_staging(prefill_out.shape, prefill_out.dtype)
                self._host_staging[staging_key] = host_buf
            host_buf.copy_(prefill_out, non_blocking=False)
            decode_buf.copy_(host_buf, non_blocking=False)
            return decode_buf
        if prefill_out.device == decode_device:
            return prefill_out
        decode_buf.copy_(prefill_out, non_blocking=True)
        return decode_buf

    def benchmark_fn(self) -> None:
        if not self.prefill_models or not self.decode_models or not self.prefill_inputs:
            raise RuntimeError("setup() must run before benchmark_fn()")

        outputs: list[torch.Tensor] = []
        with self._nvtx_range(self.label):
            with torch.no_grad():
                for (_, decode_device), prefill_model, decode_model, batch in zip(
                    self.pairs,
                    self.prefill_models,
                    self.decode_models,
                    self.prefill_inputs,
                ):
                    for idx in range(batch.shape[0]):
                        prefill_out = prefill_model(batch[idx : idx + 1])
                        kv_decode = self._handoff_kv(prefill_out, decode_device)
                        token_state = kv_decode[:, -1:, :]
                        for _ in range(self.decode_length):
                            token_state = decode_model(token_state)
                        outputs.append(token_state.squeeze(0).squeeze(0))

        self._output_shards = outputs

    def capture_verification_payload(self) -> None:
        if self._output_shards is None or self._verify_probe is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")

        selected = self._output_shards[:2]
        output_cpu = torch.stack([tensor.detach().cpu() for tensor in selected], dim=0)
        output_slice = output_cpu[:, :256].float().clone()
        self._set_verification_payload(
            inputs={"probe": self._verify_probe.detach().cpu()},
            output=output_slice,
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
        self._host_staging = {}
        self._handoff_staging = {}
        self._verify_probe = None
        self._output_shards = None
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
