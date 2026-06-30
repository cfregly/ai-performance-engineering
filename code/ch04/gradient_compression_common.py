"""Shared gradient compression benchmark logic (single- and multi-GPU)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin


class GradientCompressionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Gradient all-reduce benchmark with optional compression."""

    def __init__(
        self,
        *,
        compression: str,
        equivalence_group: str,
        output_tolerance: tuple[float, float],
        tensor_size_mb: int = 128,
        multi_gpu: bool = True,
        simulate_single_gpu_transfer: bool = False,
        comm_only: bool = False,
        use_prealloc_buffers: bool = True,
        bucket_mb: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.multi_gpu_required = bool(multi_gpu)
        self.signature_equivalence_group = equivalence_group
        self.signature_equivalence_ignore_fields = ("precision_flags",)
        self.compression = compression  # "none", "fp16", "int8"
        self.output_tolerance = output_tolerance
        self.tensor_size_mb = tensor_size_mb
        self.multi_gpu = bool(multi_gpu)
        self.simulate_single_gpu_transfer = bool(simulate_single_gpu_transfer)
        self.comm_only = bool(comm_only)
        self.use_prealloc_buffers = bool(use_prealloc_buffers)
        self.bucket_mb = int(bucket_mb) if bucket_mb else 0
        self.world_size = 0
        self.devices: List[torch.device] = []
        self.inputs: List[torch.Tensor] = []
        self.output: Optional[torch.Tensor] = None
        self._fp32_output: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._fp32_buffers: List[torch.Tensor] = []
        self._fp32_outputs: List[torch.Tensor] = []
        self._fp16_buffers: List[torch.Tensor] = []
        self._fp16_outputs: List[torch.Tensor] = []
        self._fp16_copy_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._fp16_output_fp32: Optional[torch.Tensor] = None
        self._int8_buffers: List[torch.Tensor] = []
        self._int8_outputs: List[torch.Tensor] = []
        self._int8_float_buffers: List[torch.Tensor] = []
        self._int8_max_vals: List[torch.Tensor] = []
        self._int8_scales: List[torch.Tensor] = []
        self._int8_output_fp32: Optional[torch.Tensor] = None
        self._bucket_slices: List[slice] = []
        tokens = float(tensor_size_mb * 1024 * 1024)
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens,
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens,
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        if self.multi_gpu:
            self.world_size = torch.cuda.device_count()
            if self.world_size < 2:
                raise RuntimeError("SKIPPED: requires >=2 GPUs")
            self.devices = [torch.device(f"cuda:{idx}") for idx in range(self.world_size)]
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("SKIPPED: requires CUDA")
            self.world_size = 1
            self.devices = [self.device]
        numel = (self.tensor_size_mb * 1024 * 1024) // 4  # FP32 bytes
        self.inputs = [
            torch.randn(numel, device=device, dtype=torch.float32) for device in self.devices
        ]
        self._verify_input = self.inputs[0]
        self.output = None
        self._fp32_output = torch.empty_like(self.inputs[0])
        self._verify_output_buffer = torch.empty_like(self.inputs[0])
        self._bucket_slices = self._build_bucket_slices()
        self._fp32_outputs = [torch.empty_like(t) for t in self.inputs]
        if not self.multi_gpu and self.simulate_single_gpu_transfer:
            self._fp32_buffers = [torch.empty_like(t) for t in self.inputs]
        if self.compression == "fp16":
            self._fp16_buffers = [
                torch.empty_like(t, dtype=torch.float16) for t in self.inputs
            ]
            self._fp16_outputs = [torch.empty_like(t) for t in self._fp16_buffers]
            self._fp16_copy_pairs = list(zip(self.inputs, self._fp16_buffers, strict=True))
            self._fp16_output_fp32 = torch.empty_like(self.inputs[0])
        elif self.compression == "int8":
            self._int8_buffers = [
                torch.empty_like(t, dtype=torch.int8) for t in self.inputs
            ]
            self._int8_outputs = [torch.empty_like(t) for t in self._int8_buffers]
            self._int8_float_buffers = [torch.empty_like(t) for t in self.inputs]
            self._int8_max_vals = [
                torch.empty((), device=t.device, dtype=torch.float32) for t in self.inputs
            ]
            self._int8_scales = [
                torch.empty((), device=t.device, dtype=torch.float32) for t in self.inputs
            ]
            self._int8_output_fp32 = torch.empty_like(self.inputs[0])
        if self.comm_only and self.compression == "fp16":
            if not self._fp16_copy_pairs:
                raise RuntimeError("FP16 buffers not initialized for comm-only mode")
            for src, buf in self._fp16_copy_pairs:
                buf.copy_(src)
        if self.comm_only and self.compression == "int8":
            self._prepare_int8_buffers()
        self._synchronize_all()

    def benchmark_fn(self) -> None:
        if not self.inputs:
            raise RuntimeError("Inputs not initialized")
        with torch.inference_mode(), self._nvtx_range(f"gradient_compression_{self.compression}"):
            if self.compression == "none":
                if self.multi_gpu:
                    torch.cuda.nccl.all_reduce(self.inputs, outputs=self._fp32_outputs)
                    self.output = self._fp32_outputs[0]
                else:
                    if self.simulate_single_gpu_transfer:
                        if not self._fp32_buffers or not self._fp32_outputs:
                            raise RuntimeError("FP32 transfer buffers not initialized")
                        self._fp32_buffers[0].copy_(self.inputs[0])
                        self._fp32_outputs[0].copy_(self._fp32_buffers[0])
                        self.output = self._fp32_outputs[0]
                    else:
                        if self._fp32_output is None:
                            raise RuntimeError("FP32 output buffer not initialized")
                        self._fp32_output.copy_(self.inputs[0])
                        self.output = self._fp32_output
            elif self.compression == "fp16":
                if self.comm_only or self.use_prealloc_buffers:
                    if not self._fp16_buffers:
                        raise RuntimeError("FP16 buffers not initialized")
                    if not self.comm_only:
                        for src, buf in self._fp16_copy_pairs:
                            buf.copy_(src)
                    if self.multi_gpu:
                        torch.cuda.nccl.all_reduce(self._fp16_buffers, outputs=self._fp16_outputs)
                        reduced = self._fp16_outputs[0]
                    else:
                        if self.simulate_single_gpu_transfer:
                            if not self._fp16_outputs:
                                raise RuntimeError("FP16 output buffers not initialized")
                            self._fp16_outputs[0].copy_(self._fp16_buffers[0])
                            reduced = self._fp16_outputs[0]
                        else:
                            reduced = self._fp16_buffers[0]
                    if self.comm_only:
                        self.output = reduced
                    else:
                        if self._fp16_output_fp32 is None:
                            raise RuntimeError("FP16 output buffer not initialized")
                        self._fp16_output_fp32.copy_(reduced)
                        self.output = self._fp16_output_fp32
                else:
                    self.output = self._fp16_all_reduce_naive()
            elif self.compression == "int8":
                if self.comm_only:
                    self.output = self._int8_all_reduce(prequantized=True, dequantize=False)
                elif self.use_prealloc_buffers:
                    self.output = self._int8_all_reduce(prequantized=False, dequantize=True)
                else:
                    self.output = self._int8_all_reduce_naive()
            else:
                raise ValueError(f"Unknown compression mode: {self.compression}")
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def _prepare_int8_buffers(self) -> None:
        if not self._int8_buffers or not self._int8_float_buffers:
            raise RuntimeError("INT8 buffers not initialized")
        for src, max_buf in zip(self.inputs, self._int8_max_vals):
            max_buf.copy_(src.abs().max())
        if self.multi_gpu:
            # NCCL op value 2 maps to MAX for torch.cuda.nccl.all_reduce.
            torch.cuda.nccl.all_reduce(self._int8_max_vals, op=2)
            limit = max(1, 127 // self.world_size)
        else:
            limit = 127
        for idx, src in enumerate(self.inputs):
            scale = self._int8_max_vals[idx] / float(limit)
            self._store_scale(self._int8_scales[idx], scale)
            float_buf = self._int8_float_buffers[idx]
            float_buf.copy_(src)
            float_buf.div_(self._int8_scales[idx])
            float_buf.round_()
            float_buf.clamp_(-limit, limit)
            self._int8_buffers[idx].copy_(float_buf)

    def _build_bucket_slices(self) -> List[slice]:
        if not self.inputs:
            return [slice(None)]
        if self.bucket_mb <= 0:
            return [slice(None)]
        numel = self.inputs[0].numel()
        bucket_elems = max(1, (self.bucket_mb * 1024 * 1024) // 4)
        if bucket_elems >= numel:
            return [slice(None)]
        slices = []
        for start in range(0, numel, bucket_elems):
            end = min(start + bucket_elems, numel)
            slices.append(slice(start, end))
        return slices

    def _fp16_all_reduce_naive(self) -> torch.Tensor:
        if not self._fp16_buffers or not self._fp16_copy_pairs or self._fp16_output_fp32 is None:
            raise RuntimeError("FP16 buffers not initialized")
        for src, buf in self._fp16_copy_pairs:
            buf.copy_(src)
        if self.multi_gpu:
            if len(self._bucket_slices) > 1:
                for sl in self._bucket_slices:
                    bucket_inputs = [buf[sl] for buf in self._fp16_buffers]
                    bucket_outputs = [out[sl] for out in self._fp16_outputs]
                    torch.cuda.nccl.all_reduce(bucket_inputs, outputs=bucket_outputs)
            else:
                torch.cuda.nccl.all_reduce(self._fp16_buffers, outputs=self._fp16_outputs)
            reduced = self._fp16_outputs[0]
        else:
            if self.simulate_single_gpu_transfer:
                for sl in self._bucket_slices:
                    self._fp16_outputs[0][sl].copy_(self._fp16_buffers[0][sl])
                reduced = self._fp16_outputs[0]
            else:
                reduced = self._fp16_buffers[0]
        self._fp16_output_fp32.copy_(reduced)
        return self._fp16_output_fp32

    def _int8_all_reduce(
        self,
        *,
        prequantized: bool = False,
        dequantize: bool = True,
    ) -> torch.Tensor:
        if not self._int8_buffers or not self._int8_float_buffers:
            raise RuntimeError("INT8 buffers not initialized")
        if not prequantized:
            self._prepare_int8_buffers()
        if self.multi_gpu:
            torch.cuda.nccl.all_reduce(self._int8_buffers, outputs=self._int8_outputs)
            reduced = self._int8_outputs[0]
        else:
            if self.simulate_single_gpu_transfer:
                if not self._int8_outputs:
                    raise RuntimeError("INT8 output buffers not initialized")
                self._int8_outputs[0].copy_(self._int8_buffers[0])
                reduced = self._int8_outputs[0]
            else:
                reduced = self._int8_buffers[0]
        if not dequantize:
            return reduced
        if self._int8_output_fp32 is None:
            raise RuntimeError("INT8 output buffer not initialized")
        self._int8_output_fp32.copy_(reduced)
        self._int8_output_fp32.mul_(self._int8_scales[0])
        return self._int8_output_fp32

    def _int8_all_reduce_naive(self) -> torch.Tensor:
        if (
            not self._int8_buffers
            or not self._int8_outputs
            or not self._int8_float_buffers
            or not self._int8_max_vals
            or not self._int8_scales
            or self._int8_output_fp32 is None
        ):
            raise RuntimeError("INT8 buffers not initialized")
        if self.multi_gpu:
            limit = max(1, 127 // self.world_size)
        else:
            limit = 127
        for src, max_buf in zip(self.inputs, self._int8_max_vals):
            max_buf.copy_(src.abs().max())
        if self.multi_gpu:
            torch.cuda.nccl.all_reduce(self._int8_max_vals, op=2)
        for idx, max_val in enumerate(self._int8_max_vals):
            scale = max_val / float(limit)
            self._store_scale(self._int8_scales[idx], scale)
        for idx, src in enumerate(self.inputs):
            scale = self._int8_scales[idx]
            int8_buf = self._int8_buffers[idx]
            float_buf = self._int8_float_buffers[idx]
            for sl in self._bucket_slices:
                float_buf[sl].copy_(src[sl])
                float_buf[sl].div_(scale)
                float_buf[sl].round_()
                float_buf[sl].clamp_(-limit, limit)
                int8_buf[sl].copy_(float_buf[sl])
        if self.multi_gpu:
            if len(self._bucket_slices) > 1:
                for sl in self._bucket_slices:
                    bucket_inputs = [buf[sl] for buf in self._int8_buffers]
                    bucket_outputs = [out[sl] for out in self._int8_outputs]
                    torch.cuda.nccl.all_reduce(bucket_inputs, outputs=bucket_outputs)
            else:
                torch.cuda.nccl.all_reduce(self._int8_buffers, outputs=self._int8_outputs)
            reduced = self._int8_outputs[0]
        else:
            if self.simulate_single_gpu_transfer:
                for sl in self._bucket_slices:
                    self._int8_outputs[0][sl].copy_(self._int8_buffers[0][sl])
                reduced = self._int8_outputs[0]
            else:
                reduced = self._int8_buffers[0]
        self._int8_output_fp32.copy_(reduced)
        self._int8_output_fp32.mul_(self._int8_scales[0])
        return self._int8_output_fp32

    def _store_scale(self, target: torch.Tensor, raw_scale: torch.Tensor) -> None:
        target.copy_(raw_scale)
        target.masked_fill_(target == 0, 1.0)

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        precision_flags = {
            "fp16": self.compression == "fp16",
            "bf16": False,
            "fp8": False,
            "tf32": torch.backends.cuda.matmul.allow_tf32,
        }
        output = self.output
        if self.comm_only and output is not None:
            if self.compression == "fp16":
                if self._fp16_output_fp32 is None:
                    raise RuntimeError("FP16 output buffer not initialized")
                self._fp16_output_fp32.copy_(output)
                output = self._fp16_output_fp32
            elif self.compression == "int8":
                if self._int8_output_fp32 is None:
                    raise RuntimeError("INT8 output buffer not initialized")
                if not self._int8_scales:
                    raise RuntimeError("INT8 scales not initialized")
                self._int8_output_fp32.copy_(output)
                self._int8_output_fp32.mul_(self._int8_scales[0])
                output = self._int8_output_fp32
        if self._verify_output_buffer is None:
            raise RuntimeError("Verification output buffer not initialized")
        self._verify_output_buffer.copy_(output)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=int(self._verify_input.shape[0]),
            parameter_count=0,
            precision_flags=precision_flags,
            output_tolerance=self.output_tolerance,
            signature_overrides={
                "world_size": self.world_size,
                "ranks": list(range(self.world_size)),
                "collective_type": "all_reduce",
            },
        )

    def teardown(self) -> None:
        self.inputs = []
        self.output = None
        self._fp32_output = None
        self._verify_input = None
        self._verify_output_buffer = None
        self._fp32_buffers = []
        self._fp32_outputs = []
        self._fp16_buffers = []
        self._fp16_outputs = []
        self._fp16_copy_pairs = []
        self._fp16_output_fp32 = None
        self._int8_buffers = []
        self._int8_outputs = []
        self._int8_float_buffers = []
        self._int8_max_vals = []
        self._int8_scales = []
        self._int8_output_fp32 = None
        self._bucket_slices = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            multi_gpu_required=self.multi_gpu,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if not self.inputs:
            return "Inputs not initialized"
        return None

    def _synchronize_all(self) -> None:
        for device in self.devices or [self.device]:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
